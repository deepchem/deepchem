"""SE(3) rigid-frame utilities and Invariant Point Attention.

This module provides the geometric primitives used by the SE(3)-aware
RFDiffusion denoiser. Two groups of objects are exposed:

* Rigid-frame utilities — :func:`build_backbone_frames`,
  :func:`apply_rigid`, :func:`apply_inverse_rigid`, :func:`invert_rigid`,
  :func:`compose_rigids`, and :func:`make_identity_rigid`. They use the
  row-vector convention: a rigid transform ``T = (R, t)`` maps a point
  ``x ∈ ℝ³`` to ``x @ R.T + t``. Rotation matrices are right-handed
  (``det(R) = +1``), and frames are constructed from N / Cα / C backbone
  atoms following AlphaFold2 supplementary Algorithm 21 (Jumper et al.,
  Nature 596, 583–589, 2021).

* :class:`InvariantPointAttention` — the three-term Invariant Point
  Attention layer from AlphaFold2 supplementary Algorithm 22 (Jumper et
  al., 2021). The scalar output is SE(3)-invariant: every term that
  depends on residue frames goes through point distances or a
  frame-local round-trip, so a global rigid transform of the input
  frames leaves the output unchanged.

Developer-facing summary
------------------------
If another module needs residue geometry, call
``build_backbone_frames(backbone)`` and use the returned
``(rotations, translations)`` pair as the per-residue frame state.
If another module needs the SE(3)-aware attention block, call
``InvariantPointAttention(single_repr, rotations, translations,
pair_repr=None, mask=None)`` and it will return updated single-residue
features of the same shape as ``single_repr``.

Only PyTorch is required — no external geometry library.

References
----------
.. [Jumper2021] Jumper, J. et al. *Highly accurate protein structure
   prediction with AlphaFold.* Nature 596, 583–589 (2021).
   doi:10.1038/s41586-021-03819-2.
.. [Watson2023] Watson, J. L. et al. *De novo design of protein structure
   and function with RFdiffusion.* Nature 620, 1089–1100 (2023).
   doi:10.1038/s41586-023-06415-8.
.. [Yim2023] Yim, J. et al. *SE(3) diffusion model with application to
   protein backbone generation.* Proceedings of the 40th International
   Conference on Machine Learning, PMLR 202 (2023).
"""

import math
from typing import Optional, Sequence, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    raise ImportError('These classes require PyTorch to be installed.')


# ---------------------------------------------------------------------------
# Rigid-frame utilities
# ---------------------------------------------------------------------------


def _normalize(vector: torch.Tensor, eps: float) -> torch.Tensor:
    """ε-stabilized vector normalization along the last dimension.

    Parameters
    ----------
    vector : torch.Tensor
        Tensor of shape ``(..., 3)``.
    eps : float
        Lower bound on the norm; vectors with norm below ``ε`` are scaled
        as if their norm were ``ε``.

    Returns
    -------
    torch.Tensor
        Vector divided by ``max(‖v‖, ε)``.
    """
    norm = torch.linalg.norm(vector, dim=-1, keepdim=True)
    return vector / (norm + eps)


def build_backbone_frames(
        backbone: torch.Tensor,
        eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct residue-local frames from N, Cα, C backbone coordinates.

    Implements AlphaFold2 supplementary Algorithm 21
    (``rigidFrom3Points``). Given backbone atoms ``(N, Cα, C)`` per
    residue, the local frame has its origin at ``Cα`` and orthonormal
    axes ``(ê₁, ê₂, ê₃)`` obtained by Gram–Schmidt orthogonalization with
    an ε-stabilized normalization to guard against numerically degenerate
    configurations:

    .. math::

        \\hat e_1 &= (C - C_\\alpha) / \\| C - C_\\alpha \\|, \\\\
        u_2      &= (N - C_\\alpha)
                    - \\hat e_1 \\big( \\hat e_1 \\cdot (N - C_\\alpha) \\big),
                    \\\\
        \\hat e_2 &= u_2 / \\| u_2 \\|, \\\\
        \\hat e_3 &= \\hat e_1 \\times \\hat e_2.

    The returned rotation has the orthonormal axes as its **columns**,
    so a point ``x_{local}`` in the residue frame maps to
    ``x_{global} = x_{local} @ R.T + t`` (row-vector convention).

    Parameters
    ----------
    backbone : torch.Tensor
        Backbone coordinates of shape ``(..., 3, 3)`` with atoms ordered
        as ``(N, Cα, C)`` along the second-to-last dimension.
    eps : float, default 1e-8
        ε-stabilization threshold for normalization. Must be positive.

    Returns
    -------
    rotations : torch.Tensor
        Rotation matrices of shape ``(..., 3, 3)`` with
        ``det(R) = +1`` and ``Rᵀ R = I`` to within ``O(ε)``.
    translations : torch.Tensor
        Frame origins (Cα coordinates) of shape ``(..., 3)``.

    Raises
    ------
    ValueError
        If ``backbone`` does not have shape ``(..., 3, 3)`` or if
        ``eps`` is non-positive.

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.torch_models.rfdiffusion_se3 import (
    ...     build_backbone_frames)
    >>> backbone = torch.tensor([[[-1.2, 1.1, 0.0],
    ...                           [0.0, 0.0, 0.0],
    ...                           [1.5, 0.0, 0.0]]])
    >>> R, t = build_backbone_frames(backbone)
    >>> tuple(R.shape), tuple(t.shape)
    ((1, 3, 3), (1, 3))
    >>> bool(torch.allclose(R.transpose(-1, -2) @ R, torch.eye(3),
    ...                     atol=1e-6))
    True

    References
    ----------
    .. [1] Jumper, J. et al. *Highly accurate protein structure
       prediction with AlphaFold.* Nature 596, 583–589 (2021).
    """
    if backbone.shape[-2:] != (3, 3):
        raise ValueError('backbone must have shape (..., 3, 3).')
    if eps <= 0:
        raise ValueError('eps must be positive.')

    n_atom = backbone[..., 0, :]
    ca_atom = backbone[..., 1, :]
    c_atom = backbone[..., 2, :]

    x_axis = _normalize(c_atom - ca_atom, eps)
    n_direction = n_atom - ca_atom
    y_axis = n_direction - (n_direction * x_axis).sum(dim=-1,
                                                      keepdim=True) * x_axis
    y_axis = _normalize(y_axis, eps)
    z_axis = torch.linalg.cross(x_axis, y_axis, dim=-1)

    rotations = torch.stack((x_axis, y_axis, z_axis), dim=-1)
    return rotations, ca_atom


def make_identity_rigid(
        shape: Sequence[int],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create identity rigid transforms ``(I₃, 0)`` with a given batch shape.

    Parameters
    ----------
    shape : sequence of int
        Batch shape ``(..., )`` for the transforms. May be empty for a
        single frame.
    device : torch.device, optional
        Device for the returned tensors. Defaults to the current default
        device.
    dtype : torch.dtype, optional
        Data type for the returned tensors. Defaults to
        ``torch.get_default_dtype()``.

    Returns
    -------
    rotations : torch.Tensor
        Identity rotations of shape ``(*shape, 3, 3)``. Always a freshly
        allocated tensor, safe to mutate.
    translations : torch.Tensor
        Zero translations of shape ``(*shape, 3)``.
    """
    shape = tuple(shape)
    rotation = torch.eye(3, device=device, dtype=dtype)
    rotation = rotation.expand(*shape, 3, 3).contiguous().clone()
    translation = torch.zeros(*shape, 3, device=device, dtype=dtype)
    return rotation, translation


def _expand_rigid_to_points(
        rotations: torch.Tensor, translations: torch.Tensor,
        points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Broadcast rigid transforms over leading point dimensions.

    ``rotations`` has shape ``(..., 3, 3)`` and ``translations`` has shape
    ``(..., 3)``; the function inserts singleton axes so they align with
    ``points`` having shape ``(..., extra, 3)`` for arbitrarily many
    ``extra`` axes.
    """
    extra_dims = points.dim() - translations.dim()
    if extra_dims < 0:
        raise ValueError(
            'points must have at least the transform batch dimensions.')
    for _ in range(extra_dims):
        rotations = rotations.unsqueeze(-3)
        translations = translations.unsqueeze(-2)
    return rotations, translations


def apply_rigid(rotations: torch.Tensor, translations: torch.Tensor,
                points: torch.Tensor) -> torch.Tensor:
    """Apply rigid transforms ``T = (R, t)`` to points: ``x ↦ x Rᵀ + t``.

    Parameters
    ----------
    rotations : torch.Tensor
        Rotation matrices of shape ``(..., 3, 3)``. The axes of the
        local frame are the **columns** of ``R``.
    translations : torch.Tensor
        Translation vectors of shape ``(..., 3)``.
    points : torch.Tensor
        Points of shape ``(..., 3)`` or ``(..., n_extra_dims, 3)``.
        Broadcasting against the transform batch shape is automatic.

    Returns
    -------
    torch.Tensor
        Transformed points, same shape as ``points``.
    """
    rotations, translations = _expand_rigid_to_points(rotations, translations,
                                                      points)
    rotated = torch.matmul(points.unsqueeze(-2),
                           rotations.transpose(-1, -2)).squeeze(-2)
    return rotated + translations


def apply_inverse_rigid(rotations: torch.Tensor, translations: torch.Tensor,
                        points: torch.Tensor) -> torch.Tensor:
    """Apply the inverse rigid transform ``T⁻¹``: ``y ↦ (y − t) R``.

    Equivalent to ``apply_rigid(*invert_rigid(R, t), y)`` but avoids
    constructing the inverse explicitly. The composition with
    :func:`apply_rigid` reproduces the identity to machine precision.

    Parameters
    ----------
    rotations : torch.Tensor
        Rotation matrices of shape ``(..., 3, 3)``.
    translations : torch.Tensor
        Translation vectors of shape ``(..., 3)``.
    points : torch.Tensor
        Points of shape ``(..., 3)`` or ``(..., n_extra_dims, 3)``.

    Returns
    -------
    torch.Tensor
        Points expressed in the local frame.
    """
    rotations, translations = _expand_rigid_to_points(rotations, translations,
                                                      points)
    centered = points - translations
    return torch.matmul(centered.unsqueeze(-2), rotations).squeeze(-2)


def invert_rigid(
        rotations: torch.Tensor,
        translations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Invert rigid transforms.

    Given ``T = (R, t)`` returns ``T⁻¹ = (Rᵀ, −t R)`` so that
    ``compose_rigids(T, T⁻¹)`` is the identity transform.

    Parameters
    ----------
    rotations : torch.Tensor
        Rotation matrices of shape ``(..., 3, 3)``.
    translations : torch.Tensor
        Translation vectors of shape ``(..., 3)``.

    Returns
    -------
    inv_rotations : torch.Tensor
        ``Rᵀ`` of shape ``(..., 3, 3)``.
    inv_translations : torch.Tensor
        ``−t R`` of shape ``(..., 3)``.
    """
    inv_rotations = rotations.transpose(-1, -2)
    inv_translations = -torch.matmul(translations.unsqueeze(-2),
                                     rotations).squeeze(-2)
    return inv_rotations, inv_translations


def compose_rigids(
        rotations_a: torch.Tensor, translations_a: torch.Tensor,
        rotations_b: torch.Tensor,
        translations_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compose two rigid transforms: ``T_a ∘ T_b``.

    The returned transform first applies ``T_b`` and then ``T_a``:

    .. math::

        (T_a \\circ T_b)(x) = T_a(T_b(x))
                            = (x R_b^\\top + t_b) R_a^\\top + t_a.

    Parameters
    ----------
    rotations_a, rotations_b : torch.Tensor
        Rotation matrices of shape ``(..., 3, 3)``.
    translations_a, translations_b : torch.Tensor
        Translation vectors of shape ``(..., 3)``.

    Returns
    -------
    rotations : torch.Tensor
        Composite rotation ``R_a R_b`` of shape ``(..., 3, 3)``.
    translations : torch.Tensor
        Composite translation ``t_b R_a^\\top + t_a`` of shape
        ``(..., 3)``.
    """
    rotations = torch.matmul(rotations_a, rotations_b)
    translations = apply_rigid(rotations_a, translations_a, translations_b)
    return rotations, translations


# ---------------------------------------------------------------------------
# Invariant Point Attention
# ---------------------------------------------------------------------------


class InvariantPointAttention(nn.Module):
    """Invariant Point Attention (AlphaFold2 supplementary Algorithm 22).

    The layer mixes residue-level scalar features with point features
    that live in residue-local frames. Its scalar output is invariant
    under any global rigid transform of all input frames because every
    frame-dependent quantity enters through (a) Euclidean distances
    between points or (b) a round-trip ``T⁻¹ ∘ T`` that cancels the
    global motion exactly.

    The attention logit decomposes into three additive terms — scalar
    QK, point-distance, and pair-bias — combined with the AlphaFold2
    weighting:

    .. math::

        a_{ij}^{h} \\propto \\exp \\Big( w_L \\big[ s_{ij}^{h}
        - \\tfrac{1}{2} w_C\\, \\gamma^{h}
              \\sum_p \\| T_i \\hat q_{ip}^{h}
                       - T_j \\hat k_{jp}^{h} \\|^2
        + b_{ij}^{h} \\big] \\Big),

    where

    * :math:`s_{ij}^{h} = \\langle q_i^{h}, k_j^{h} \\rangle /
      \\sqrt{c}` is the scaled dot-product of the scalar Q/K heads of
      width :math:`c = embed\\_dim / num\\_heads`,
    * :math:`\\hat q^{h}_{ip}, \\hat k^{h}_{jp} \\in \\mathbb{R}^{3}`
      are :math:`p = 1, \\dots, N_{qk}` query/key points per head,
      expressed in their residue's local frame and lifted to the
      global frame by the rigid transform :math:`T_i`,
    * :math:`\\gamma^{h} = \\mathrm{softplus}(\\hat\\gamma^{h})` is a
      learnable, strictly positive per-head gating scalar,
    * :math:`w_C = \\sqrt{2/(9 N_{qk})}` is the canonical point-term
      normalization constant from Algorithm 22,
    * :math:`b_{ij}^{h}` is a per-head bias linearly projected from the
      pair representation :math:`z_{ij} \\in \\mathbb{R}^{c_z}` if a
      ``pair_dim`` is supplied; the term is omitted otherwise,
    * :math:`w_L = \\sqrt{1/L}` with :math:`L = 3` when the pair-bias
      term is active and :math:`L = 2` otherwise — a standard variance
      stabilization over the number of additive logit sources.

    Each token receives the concatenation of (i) the scalar attention
    output ``Σⱼ aᵢⱼ vⱼ``, (ii) optionally the pair output
    ``Σⱼ aᵢⱼ zᵢⱼ`` (only when ``pair_dim`` is given), and (iii) the
    value-point output mapped back into the residue's own local frame
    together with its per-point Euclidean norm. The concatenation is
    passed through a final linear layer back to ``embed_dim``.

    Parameters
    ----------
    embed_dim : int
        Scalar input / output channel count. Must be a positive multiple
        of ``num_heads``.
    num_heads : int, default 8
        Number of attention heads.
    num_qk_points : int, default 4
        Query/key points per head.
    num_v_points : int, default 8
        Value points per head.
    pair_dim : int, optional
        Width :math:`c_z` of an optional pair representation
        ``z_{ij} ∈ ℝ^{c_z}``. When ``None`` (default) both the pair-bias
        logit and the pair attention output are disabled.
    dropout : float, default 0.0
        Dropout applied to the attention weights.
    eps : float, default 1e-8
        ε-stabilization for the point-norm features so their gradient
        stays finite at zero.

    Raises
    ------
    ValueError
        If ``embed_dim``, ``num_heads``, ``num_qk_points``, or
        ``num_v_points`` is non-positive, if ``embed_dim`` is not a
        multiple of ``num_heads``, or if ``dropout`` is outside
        ``[0, 1)``.

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.torch_models.rfdiffusion_se3 import (
    ...     InvariantPointAttention, build_backbone_frames)
    >>> _ = torch.manual_seed(0)
    >>> backbone = torch.randn(2, 5, 3, 3)
    >>> R, t = build_backbone_frames(backbone)
    >>> layer = InvariantPointAttention(embed_dim=16, num_heads=4)
    >>> out = layer(torch.randn(2, 5, 16), R, t)
    >>> tuple(out.shape)
    (2, 5, 16)

    References
    ----------
    .. [1] Jumper, J. et al. *Highly accurate protein structure
       prediction with AlphaFold.* Nature 596, 583–589 (2021).
       Supplementary Algorithm 22.
    .. [2] Watson, J. L. et al. *De novo design of protein structure
       and function with RFdiffusion.* Nature 620, 1089–1100 (2023).
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 num_qk_points: int = 4,
                 num_v_points: int = 8,
                 pair_dim: Optional[int] = None,
                 dropout: float = 0.0,
                 eps: float = 1e-8) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError('embed_dim must be positive.')
        if num_heads <= 0:
            raise ValueError('num_heads must be positive.')
        if embed_dim % num_heads != 0:
            raise ValueError('embed_dim must be divisible by num_heads.')
        if num_qk_points <= 0:
            raise ValueError('num_qk_points must be positive.')
        if num_v_points <= 0:
            raise ValueError('num_v_points must be positive.')
        if pair_dim is not None and pair_dim <= 0:
            raise ValueError('pair_dim must be positive when provided.')
        if not 0.0 <= dropout < 1.0:
            raise ValueError('dropout must lie in [0, 1).')
        if eps <= 0:
            raise ValueError('eps must be positive.')

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_qk_points = num_qk_points
        self.num_v_points = num_v_points
        self.pair_dim = pair_dim
        self.eps = eps

        # Scalar projections (per AF2 Algorithm 22 line 1).
        self.query_scalar = nn.Linear(embed_dim, embed_dim)
        self.key_scalar = nn.Linear(embed_dim, embed_dim)
        self.value_scalar = nn.Linear(embed_dim, embed_dim)

        # Point projections (per Algorithm 22 line 2).
        self.query_points = nn.Linear(embed_dim, num_heads * num_qk_points * 3)
        self.key_points = nn.Linear(embed_dim, num_heads * num_qk_points * 3)
        self.value_points = nn.Linear(embed_dim, num_heads * num_v_points * 3)

        # Pair-bias projection b_{ij}^h = z_{ij} W_b (per Algorithm 22 line 3).
        if pair_dim is not None:
            self.pair_bias = nn.Linear(pair_dim, num_heads, bias=False)
        else:
            self.pair_bias = None

        # Per-head softplus-gated scalar γ^h. Initialised so that
        # softplus(γ̂) ≈ ln 2, which mirrors the AF2 reference init.
        self.point_weights = nn.Parameter(
            torch.full((num_heads,), math.log(math.exp(1.0) - 1.0)))

        # Output projection: scalar concat + (pair concat if any) + point
        # vectors + point norms.
        point_features = num_heads * num_v_points * (3 + 1)
        pair_features = num_heads * pair_dim if pair_dim is not None else 0
        self.output = nn.Linear(embed_dim + pair_features + point_features,
                                embed_dim)
        self.dropout = nn.Dropout(dropout)

        # AF2 weighting constants (registered as buffers so they move with
        # the module under .to(device)).
        w_c = math.sqrt(2.0 / (9.0 * num_qk_points))
        num_sources = 3.0 if pair_dim is not None else 2.0
        w_l = math.sqrt(1.0 / num_sources)
        self.register_buffer('w_c',
                             torch.tensor(w_c, dtype=torch.get_default_dtype()))
        self.register_buffer('w_l',
                             torch.tensor(w_l, dtype=torch.get_default_dtype()))

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _reshape_scalar(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape ``(B, L, H·c)`` → ``(B, H, L, c)``."""
        batch_size, seq_len, _ = tensor.shape
        tensor = tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return tensor.permute(0, 2, 1, 3)

    def _reshape_points(self, tensor: torch.Tensor,
                        num_points: int) -> torch.Tensor:
        """Reshape ``(B, L, H·P·3)`` → ``(B, L, H, P, 3)``."""
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(batch_size, seq_len, self.num_heads, num_points, 3)

    # ------------------------------------------------------------------ #
    # Forward pass                                                       #
    # ------------------------------------------------------------------ #

    def forward(self,
                single_repr: torch.Tensor,
                rotations: torch.Tensor,
                translations: torch.Tensor,
                pair_repr: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply Invariant Point Attention.

        Parameters
        ----------
        single_repr : torch.Tensor
            Residue scalar features of shape
            ``(batch, num_residues, embed_dim)``.
        rotations : torch.Tensor
            Residue frame rotations of shape
            ``(batch, num_residues, 3, 3)``.
        translations : torch.Tensor
            Residue frame translations of shape
            ``(batch, num_residues, 3)``.
        pair_repr : torch.Tensor, optional
            Pair representation ``z`` of shape
            ``(batch, num_residues, num_residues, pair_dim)``. Required
            iff the layer was constructed with ``pair_dim`` set.
        mask : torch.Tensor, optional
            Boolean (or ``{0,1}``) residue mask of shape
            ``(batch, num_residues)``. Masked positions are excluded
            from attention (as keys) and their output is set to zero
            (as queries).

        Returns
        -------
        torch.Tensor
            Updated scalar features of shape
            ``(batch, num_residues, embed_dim)``.

        Raises
        ------
        ValueError
            If any input tensor has an inconsistent shape, if
            ``pair_repr`` is missing while ``pair_dim`` is set, or if
            ``pair_repr`` is supplied to a layer constructed without
            ``pair_dim``.
        """
        if single_repr.dim() != 3:
            raise ValueError(
                'single_repr must have shape (batch, num_residues, embed_dim).')
        if rotations.shape != single_repr.shape[:2] + (3, 3):
            raise ValueError(
                'rotations must have shape (batch, num_residues, 3, 3).')
        if translations.shape != single_repr.shape[:2] + (3,):
            raise ValueError(
                'translations must have shape (batch, num_residues, 3).')

        batch_size, seq_len, _ = single_repr.shape
        device = single_repr.device
        dtype = single_repr.dtype

        if (pair_repr is None) != (self.pair_dim is None):
            raise ValueError(
                'pair_repr must be provided iff pair_dim was set at '
                'construction.')
        if pair_repr is not None:
            expected_pair = (batch_size, seq_len, seq_len, self.pair_dim)
            if pair_repr.shape != expected_pair:
                raise ValueError(
                    'pair_repr must have shape '
                    '(batch, num_residues, num_residues, pair_dim).')

        if mask is not None:
            if mask.shape != (batch_size, seq_len):
                raise ValueError('mask must have shape (batch, num_residues).')
            mask = mask.to(dtype=torch.bool, device=device)

        # ----------- Scalar projections (B, H, L, c) -----------------
        query_scalar = self._reshape_scalar(self.query_scalar(single_repr))
        key_scalar = self._reshape_scalar(self.key_scalar(single_repr))
        value_scalar = self._reshape_scalar(self.value_scalar(single_repr))

        # ----------- Point projections in the local frame -----------
        query_points_local = self._reshape_points(
            self.query_points(single_repr), self.num_qk_points)
        key_points_local = self._reshape_points(self.key_points(single_repr),
                                                self.num_qk_points)
        value_points_local = self._reshape_points(
            self.value_points(single_repr), self.num_v_points)

        # Lift points from local to global frames T_i.
        query_points = apply_rigid(rotations, translations, query_points_local)
        key_points = apply_rigid(rotations, translations, key_points_local)
        value_points = apply_rigid(rotations, translations, value_points_local)

        # Reorder to (B, H, L, P, 3) so the head dimension precedes seq.
        query_points = query_points.permute(0, 2, 1, 3, 4)
        key_points = key_points.permute(0, 2, 1, 3, 4)
        value_points = value_points.permute(0, 2, 1, 3, 4)

        # ----------- Three logit terms ------------------------------
        scalar_logits = torch.matmul(query_scalar, key_scalar.transpose(-1, -2))
        scalar_logits = scalar_logits / math.sqrt(self.head_dim)

        # Squared point distances summed over points p.
        # query_points: (B, H, L_q, P, 3) -> (B, H, L_q, 1, P, 3)
        # key_points:   (B, H, L_k, P, 3) -> (B, H, 1, L_k, P, 3)
        point_diff = query_points.unsqueeze(3) - key_points.unsqueeze(2)
        point_dist2 = point_diff.pow(2).sum(dim=(-1, -2))

        gamma = F.softplus(self.point_weights).view(1, self.num_heads, 1, 1)
        w_c = self.w_c.to(dtype=dtype)
        point_logits = -0.5 * w_c * gamma * point_dist2

        if self.pair_bias is not None:
            assert pair_repr is not None  # for type checkers
            bias = self.pair_bias(pair_repr)  # (B, L_q, L_k, H)
            bias = bias.permute(0, 3, 1, 2)  # (B, H, L_q, L_k)
        else:
            bias = torch.zeros_like(scalar_logits)

        w_l = self.w_l.to(dtype=dtype)
        logits = w_l * (scalar_logits + point_logits + bias)

        # ----------- Masking ---------------------------------------
        if mask is not None:
            key_mask = mask.view(batch_size, 1, 1, seq_len)
            logits = logits.masked_fill(~key_mask,
                                        torch.finfo(logits.dtype).min)

        attention = torch.softmax(logits, dim=-1)
        attention = self.dropout(attention)

        # ----------- Outputs ---------------------------------------
        # Scalar output (B, H, L, c) -> (B, L, H·c).
        scalar_out = torch.matmul(attention, value_scalar)
        scalar_out = scalar_out.permute(0, 2, 1,
                                        3).reshape(batch_size, seq_len,
                                                   self.embed_dim)

        # Point output: weighted sum in the *global* frame then mapped
        # back into the residue-local frame via T_i^{-1}.
        # attention: (B, H, L_q, L_k); value_points: (B, H, L_k, P, 3).
        point_out_global = torch.einsum('bhij,bhjpd->bhipd', attention,
                                        value_points)
        # Reorder to (B, L_q, H, P, 3) to match the rigid-transform
        # broadcast convention used in apply_inverse_rigid.
        point_out_global = point_out_global.permute(0, 2, 1, 3, 4)
        point_out_local = apply_inverse_rigid(rotations, translations,
                                              point_out_global)
        point_norms = torch.sqrt(
            point_out_local.pow(2).sum(dim=-1) + self.eps)
        point_out_local = point_out_local.reshape(batch_size, seq_len, -1)
        point_norms = point_norms.reshape(batch_size, seq_len, -1)

        feature_list = [scalar_out]
        if self.pair_bias is not None:
            # Pair output: Σ_j a_{ij}^h z_{ij}^{c_z}  → (B, L_q, H, c_z).
            assert pair_repr is not None
            pair_out = torch.einsum('bhij,bijc->bihc', attention, pair_repr)
            pair_out = pair_out.reshape(batch_size, seq_len, -1)
            feature_list.append(pair_out)
        feature_list.extend([point_out_local, point_norms])

        output = self.output(torch.cat(feature_list, dim=-1))

        if mask is not None:
            output = output * mask.unsqueeze(-1).to(dtype=output.dtype)
        return output

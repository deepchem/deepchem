"""Invariant Point Attention for SE(3)-equivariant protein structure models.

This module adds :class:`InvariantPointAttention`, the core geometric
attention block from AlphaFold2 (Algorithm 22).  The layer computes
attention over residues using scalar features plus 3-D point features
that live in each residue's local frame.  The key property is that the
scalar output is exactly invariant to any global rigid transform of the
input frames, so the network does not need to track absolute
orientation.

The three logit contributions are:

* Scaled dot-product of the scalar Q/K projections.
* Negative half-sum of squared point distances between query and key
  points lifted to the global frame.
* Optional per-head bias projected from a pairwise representation.

This is PR 6 in the RFDiffusion integration.  It depends on the
rigid-frame utilities in ``rfdiffusion_frames.py`` (PR 5 / #4982).

References
----------
.. [Jumper2021] Jumper, J. et al. *Highly accurate protein structure
   prediction with AlphaFold.* Nature 596, 583-589 (2021).
.. [Watson2023] Watson, J. L. et al. *De novo design of protein
   structure and function with RFdiffusion.* Nature 620, 1089-1100
   (2023).
"""

import math
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    raise ImportError('rfdiffusion_ipa requires PyTorch to be installed.')

from deepchem.models.torch_models.rfdiffusion_frames import (
    apply_inverse_rigid,
    apply_rigid,
)


class InvariantPointAttention(nn.Module):
    """SE(3)-invariant attention over residue frames.

    Computes per-residue attention logits using three additive terms
    (scalar QK, point-distance, and optional pair bias) following
    AlphaFold2 supplementary Algorithm 22.  The scalar output is
    invariant to any global rigid motion of the input frames.

    Parameters
    ----------
    embed_dim : int
        Scalar input / output channel width.  Must be a positive
        multiple of ``num_heads``.
    num_heads : int, default 8
        Number of attention heads.
    num_qk_points : int, default 4
        Number of 3-D query / key points per head.
    num_v_points : int, default 8
        Number of 3-D value points per head.
    pair_dim : int, optional
        Width of an optional pairwise feature tensor ``z_{ij}``.  When
        set, the layer accepts ``pair_repr`` in :meth:`forward` and uses
        it both as a logit bias and as an additional output feature.
    dropout : float, default 0.0
        Dropout probability applied to attention weights.
    eps : float, default 1e-8
        Small constant added inside the point-norm sqrt to keep
        gradients finite.

    Raises
    ------
    ValueError
        If any parameter is out of range or inconsistent.

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.torch_models.rfdiffusion_ipa import (
    ...     InvariantPointAttention)
    >>> from deepchem.models.torch_models.rfdiffusion_frames import (
    ...     build_backbone_frames)
    >>> _ = torch.manual_seed(0)
    >>> backbone = torch.randn(2, 5, 3, 3)
    >>> R, t = build_backbone_frames(backbone)
    >>> layer = InvariantPointAttention(embed_dim=16, num_heads=4)
    >>> out = layer(torch.randn(2, 5, 16), R, t)
    >>> tuple(out.shape)
    (2, 5, 16)
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

        # Scalar Q / K / V projections.
        self.query_scalar = nn.Linear(embed_dim, embed_dim)
        self.key_scalar = nn.Linear(embed_dim, embed_dim)
        self.value_scalar = nn.Linear(embed_dim, embed_dim)

        # 3-D point projections.  Output is interpreted as
        # (num_heads * num_points, 3) and reshaped accordingly.
        self.query_points = nn.Linear(embed_dim, num_heads * num_qk_points * 3)
        self.key_points = nn.Linear(embed_dim, num_heads * num_qk_points * 3)
        self.value_points = nn.Linear(embed_dim, num_heads * num_v_points * 3)

        # Optional pair-bias projection.
        self.pair_bias: Optional[nn.Linear]
        if pair_dim is not None:
            self.pair_bias = nn.Linear(pair_dim, num_heads, bias=False)
        else:
            self.pair_bias = None

        # Learnable per-head point-distance gate (scalar γ^h).
        # Initialised so that softplus(γ̂) ≈ ln 2 at the start of training.
        self.point_weights = nn.Parameter(
            torch.full((num_heads,), math.log(math.exp(1.0) - 1.0)))

        # Output linear: concatenation of scalar + (pair) + point vectors +
        # point norms.
        point_features = num_heads * num_v_points * (3 + 1)
        pair_features = num_heads * pair_dim if pair_dim is not None else 0
        self.output = nn.Linear(embed_dim + pair_features + point_features,
                                embed_dim)
        self.dropout = nn.Dropout(dropout)

        # AF2 weighting constants stored as buffers so they move with the
        # module when .to(device) is called.
        w_c = math.sqrt(2.0 / (9.0 * num_qk_points))
        num_sources = 3.0 if pair_dim is not None else 2.0
        w_l = math.sqrt(1.0 / num_sources)
        self.register_buffer('w_c',
                             torch.tensor(w_c, dtype=torch.get_default_dtype()))
        self.register_buffer('w_l',
                             torch.tensor(w_l, dtype=torch.get_default_dtype()))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reshape_scalar(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape ``(B, L, H*c)`` to ``(B, H, L, c)``."""
        b, l, _ = tensor.shape
        return tensor.view(b, l, self.num_heads,
                           self.head_dim).permute(0, 2, 1, 3)

    def _reshape_points(self, tensor: torch.Tensor,
                        num_points: int) -> torch.Tensor:
        """Reshape ``(B, L, H*P*3)`` to ``(B, L, H, P, 3)``."""
        b, l, _ = tensor.shape
        return tensor.view(b, l, self.num_heads, num_points, 3)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self,
                single_repr: torch.Tensor,
                rotations: torch.Tensor,
                translations: torch.Tensor,
                pair_repr: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply IPA to update per-residue scalar features.

        Parameters
        ----------
        single_repr : torch.Tensor
            Residue features, shape ``(batch, num_residues, embed_dim)``.
        rotations : torch.Tensor
            Residue frame rotations, shape
            ``(batch, num_residues, 3, 3)``.
        translations : torch.Tensor
            Residue frame origins, shape ``(batch, num_residues, 3)``.
        pair_repr : torch.Tensor, optional
            Pairwise features, shape
            ``(batch, num_residues, num_residues, pair_dim)``.  Required
            when the layer was built with ``pair_dim`` set; forbidden
            otherwise.
        mask : torch.Tensor, optional
            Boolean residue mask, shape ``(batch, num_residues)``.
            Masked residues are excluded as attention keys and their
            output rows are zeroed.

        Returns
        -------
        torch.Tensor
            Updated residue features, shape
            ``(batch, num_residues, embed_dim)``.

        Raises
        ------
        ValueError
            On shape or consistency mismatches.
        """
        if single_repr.dim() != 3:
            raise ValueError(
                'single_repr must have shape (batch, num_residues, embed_dim).')
        b, l, _ = single_repr.shape
        device = single_repr.device
        dtype = single_repr.dtype

        if rotations.shape != (b, l, 3, 3):
            raise ValueError(
                'rotations must have shape (batch, num_residues, 3, 3).')
        if translations.shape != (b, l, 3):
            raise ValueError(
                'translations must have shape (batch, num_residues, 3).')

        if (pair_repr is None) != (self.pair_dim is None):
            raise ValueError(
                'pair_repr must be provided iff pair_dim was set at '
                'construction.')
        if pair_repr is not None and pair_repr.shape != (b, l, l,
                                                         self.pair_dim):
            raise ValueError('pair_repr must have shape '
                             '(batch, num_residues, num_residues, pair_dim).')

        if mask is not None and mask.shape != (b, l):
            raise ValueError('mask must have shape (batch, num_residues).')
        if mask is not None:
            mask = mask.to(dtype=torch.bool, device=device)

        # ---------- Scalar Q / K / V projections (B, H, L, c) --------
        qs = self._reshape_scalar(self.query_scalar(single_repr))
        ks = self._reshape_scalar(self.key_scalar(single_repr))
        vs = self._reshape_scalar(self.value_scalar(single_repr))

        # ---------- 3-D point projections in local frames (B, L, H, P, 3)
        qp_local = self._reshape_points(self.query_points(single_repr),
                                        self.num_qk_points)
        kp_local = self._reshape_points(self.key_points(single_repr),
                                        self.num_qk_points)
        vp_local = self._reshape_points(self.value_points(single_repr),
                                        self.num_v_points)

        # Lift points from local residue frame to global frame using T_i.
        qp = apply_rigid(rotations, translations,
                         qp_local).permute(0, 2, 1, 3, 4)
        kp = apply_rigid(rotations, translations,
                         kp_local).permute(0, 2, 1, 3, 4)
        vp = apply_rigid(rotations, translations,
                         vp_local).permute(0, 2, 1, 3, 4)

        # ---------- Logits --------------------------------------------
        scalar_logits = torch.matmul(qs, ks.transpose(-1, -2))
        scalar_logits = scalar_logits / math.sqrt(self.head_dim)

        # Point-distance term: (B, H, L_q, L_k).
        # qp: (B, H, L_q, P, 3), kp: (B, H, L_k, P, 3)
        point_diff = qp.unsqueeze(3) - kp.unsqueeze(2)
        point_dist2 = point_diff.pow(2).sum(dim=(-1, -2))

        gamma = F.softplus(self.point_weights).view(1, self.num_heads, 1, 1)
        point_logits = -0.5 * self.w_c.to(dtype=dtype) * gamma * point_dist2

        if self.pair_bias is not None:
            assert pair_repr is not None
            bias = self.pair_bias(pair_repr).permute(0, 3, 1, 2)  # (B, H, L, L)
        else:
            bias = torch.zeros_like(scalar_logits)

        logits = self.w_l.to(dtype=dtype) * (scalar_logits + point_logits +
                                             bias)

        # ---------- Mask and attention --------------------------------
        if mask is not None:
            logits = logits.masked_fill(~mask.view(b, 1, 1, l),
                                        torch.finfo(logits.dtype).min)

        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        # ---------- Outputs -------------------------------------------
        # Scalar output (B, L, H*c).
        scalar_out = torch.matmul(attn, vs)
        scalar_out = scalar_out.permute(0, 2, 1,
                                        3).reshape(b, l, self.embed_dim)

        # Point output: weighted sum in global frame, then back to local.
        # attn: (B, H, L, L), vp: (B, H, L, P, 3)
        vp_out_global = torch.einsum('bhij,bhjpd->bhipd', attn, vp)
        vp_out_global = vp_out_global.permute(0, 2, 1, 3, 4)
        vp_out_local = apply_inverse_rigid(rotations, translations,
                                           vp_out_global)
        vp_norms = torch.sqrt(vp_out_local.pow(2).sum(-1) + self.eps)
        vp_out_flat = vp_out_local.reshape(b, l, -1)
        vp_norms_flat = vp_norms.reshape(b, l, -1)

        parts = [scalar_out]
        if self.pair_bias is not None:
            assert pair_repr is not None
            pair_out = torch.einsum('bhij,bijc->bihc', attn, pair_repr)
            parts.append(pair_out.reshape(b, l, -1))
        parts.extend([vp_out_flat, vp_norms_flat])

        out = self.output(torch.cat(parts, dim=-1))
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(dtype=out.dtype)
        return out

"""Sequence (1D) and structure (3D) track updates for RFDiffusion.

This is the second half of RFDiffusion's RoseTTAFold-style multi-track
network — the blocks that turn the pair representation back into
single-residue features and rigid backbone frames, plus the block and
stack that tie all three tracks together.

* :class:`PairBiasedSingleAttention` — 2D -> 1D update: single-track
  self-attention with a pair-derived logit bias.
* :class:`SingleTransition` — position-wise feed-forward on the single
  track with adaLN-style timestep conditioning.
* :func:`quaternion_to_rotation_matrix` — quaternion -> rotation helper.
* :class:`BackboneUpdate` — 3D head that predicts a per-residue rigid
  frame update (quaternion rotation + local translation), composed so
  the frames move equivariantly under any global rigid motion.
* :class:`RFDiffusionTrackBlock` — one full single -> pair -> single ->
  frames iteration.
* :class:`RFDiffusionMultiTrackStack` — a stack of those blocks with a
  ``forward(single, t_emb, attention_mask=None) -> single`` signature,
  so it drops straight in for the baseline transformer stack in
  :mod:`deepchem.models.torch_models.rfdiffusion`.

It builds on the earlier RFDiffusion PRs: the pair-track blocks (PR 7,
``rfdiffusion_pair_track``), the Invariant Point Attention block (PR 6,
``rfdiffusion_ipa``), and the rigid-frame helpers (PR 5,
``rfdiffusion_frames``). Every pair update stays symmetric
(:math:`z_{ij} = z_{ji}`) and every frame update is equivariant — the
test suite checks both.

References
----------
.. [Jumper2021] Jumper, J. *et al.* "Highly accurate protein structure
   prediction with AlphaFold." *Nature* **596** (2021): 583–589.
.. [Baek2021] Baek, M. *et al.* "Accurate prediction of protein
   structures and interactions using a three-track neural network."
   *Science* **373** (2021): 871–876.
.. [Watson2023] Watson, J. L. *et al.* "De novo design of protein
   structure and function with RFdiffusion." *Nature* **620** (2023):
   1089–1100.
"""

import math
from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    raise ImportError(
        'rfdiffusion_sequence_track requires PyTorch to be installed.')

from deepchem.models.torch_models.rfdiffusion_frames import make_identity_rigid
from deepchem.models.torch_models.rfdiffusion_ipa import (
    InvariantPointAttention)
from deepchem.models.torch_models.rfdiffusion_pair_track import (
    OuterProductMean,
    PairTransition,
    RelativePositionEmbedding,
    TriangleAttention,
    TriangleMultiplicativeUpdate,
)

# ---------------------------------------------------------------------------
# 2D → 1D update: pair-biased self-attention
# ---------------------------------------------------------------------------


class PairBiasedSingleAttention(nn.Module):
    """Multi-head self-attention on the single track biased by the pair.

    The attention logits are augmented by a per-head linear projection
    of the pair representation, giving the pair channel a direct
    pathway to modulate the single track. This is the AlphaFold2
    "MSA row-wise attention with pair bias" reduced to a single MSA
    row.

    Parameters
    ----------
    embed_dim : int
        Single-track channel size :math:`C_s`.
    pair_dim : int
        Pair-track channel size :math:`C_z`.
    num_heads : int, default 8
        Number of attention heads.
    dropout : float, default 0.0
        Dropout probability applied to attention weights.
    """

    def __init__(self,
                 embed_dim: int,
                 pair_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.0) -> None:
        super().__init__()
        if embed_dim <= 0 or pair_dim <= 0 or num_heads <= 0:
            raise ValueError(
                'embed_dim, pair_dim and num_heads must be positive.')
        if embed_dim % num_heads != 0:
            raise ValueError('embed_dim must be divisible by num_heads.')
        if not 0.0 <= dropout < 1.0:
            raise ValueError('dropout must lie in [0, 1).')
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.norm_single = nn.LayerNorm(embed_dim)
        self.norm_pair = nn.LayerNorm(pair_dim)
        self.linear_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.linear_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.linear_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.linear_bias = nn.Linear(pair_dim, num_heads, bias=False)
        self.linear_out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self,
                single: torch.Tensor,
                pair: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        single : torch.Tensor
            Single representation of shape ``(B, L, C_s)``.
        pair : torch.Tensor
            Pair representation of shape ``(B, L, L, C_z)``.
        mask : torch.Tensor, optional
            Boolean mask of shape ``(B, L)`` where True marks valid
            residues. Masked keys are excluded from attention; masked
            queries receive zero output.

        Returns
        -------
        torch.Tensor
            Updated single representation, same shape as input.
        """
        if single.dim() != 3:
            raise ValueError('single must be 3D (B, L, C_s).')
        if pair.dim() != 4:
            raise ValueError('pair must be 4D (B, L, L, C_z).')

        s_norm = self.norm_single(single)
        z_norm = self.norm_pair(pair)
        batch, seq_len, _ = s_norm.shape
        q = self.linear_q(s_norm).view(batch, seq_len, self.num_heads,
                                       self.head_dim)
        k = self.linear_k(s_norm).view(batch, seq_len, self.num_heads,
                                       self.head_dim)
        v = self.linear_v(s_norm).view(batch, seq_len, self.num_heads,
                                       self.head_dim)
        # logits: (B, H, L_q, L_k) = (1/√d) qᵢ · kⱼ + bᵢⱼ
        logits = torch.einsum('bihd,bjhd->bhij', q, k) * self.scale
        bias = self.linear_bias(z_norm).permute(0, 3, 1, 2)
        logits = logits + bias
        if mask is not None:
            mask_bool = mask.to(dtype=torch.bool)
            key_mask = mask_bool.view(batch, 1, 1, seq_len)
            logits = logits.masked_fill(~key_mask,
                                        torch.finfo(logits.dtype).min)
        attn = F.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhij,bjhd->bihd', attn, v)
        out = out.reshape(batch, seq_len, self.embed_dim)
        out = self.linear_out(out)
        if mask is not None:
            out = out * mask.to(dtype=out.dtype).unsqueeze(-1)
        return out


# ---------------------------------------------------------------------------
# Single transition (feed-forward) with adaptive time conditioning
# ---------------------------------------------------------------------------


class SingleTransition(nn.Module):
    """Position-wise feed-forward block on the single track with
    adaLN-style timestep conditioning.

    The block applies layer-norm, computes a time-dependent
    :math:`(scale, shift)` pair from the timestep embedding, modulates
    the normalised representation, then runs a two-layer MLP with GELU
    activation.

    Parameters
    ----------
    embed_dim : int
        Single-track channel size :math:`C_s`.
    expansion : int, default 4
        Hidden expansion of the MLP.
    dropout : float, default 0.0
        Dropout probability applied to the MLP hidden activations.
    """

    def __init__(self,
                 embed_dim: int,
                 expansion: int = 4,
                 dropout: float = 0.0) -> None:
        super().__init__()
        if embed_dim <= 0 or expansion <= 0:
            raise ValueError('embed_dim and expansion must be positive.')
        if not 0.0 <= dropout < 1.0:
            raise ValueError('dropout must lie in [0, 1).')
        self.norm = nn.LayerNorm(embed_dim)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim * 2),
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * expansion, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, single: torch.Tensor,
                t_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        single : torch.Tensor
            Single representation of shape ``(B, L, C_s)``.
        t_emb : torch.Tensor
            Timestep embedding of shape ``(B, C_s)``.

        Returns
        -------
        torch.Tensor
            Updated single representation, same shape as input.
        """
        time_cond = self.time_mlp(t_emb)
        scale, shift = time_cond.chunk(2, dim=-1)
        h = self.norm(single)
        h = h * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.mlp(h)


# ---------------------------------------------------------------------------
# Backbone update head
# ---------------------------------------------------------------------------


def quaternion_to_rotation_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Convert a (possibly unnormalised) quaternion to a rotation matrix.

    Uses the standard formula for a unit quaternion :math:`q = (w, x,
    y, z)`:

    .. math::

        R = \\begin{pmatrix}
            1 - 2y^2 - 2z^2 & 2xy - 2wz & 2xz + 2wy \\\\
            2xy + 2wz & 1 - 2x^2 - 2z^2 & 2yz - 2wx \\\\
            2xz - 2wy & 2yz + 2wx & 1 - 2x^2 - 2y^2
        \\end{pmatrix}.

    The quaternion is normalised internally so callers do not need to
    pre-normalise.

    Parameters
    ----------
    quat : torch.Tensor
        Quaternion of shape ``(..., 4)`` with components in the order
        :math:`(w, x, y, z)`.

    Returns
    -------
    torch.Tensor
        Rotation matrix of shape ``(..., 3, 3)``.
    """
    norm = torch.linalg.norm(quat, dim=-1, keepdim=True).clamp(min=1e-12)
    w, x, y, z = (quat / norm).unbind(dim=-1)
    two = 2.0
    r00 = 1.0 - two * (y * y + z * z)
    r01 = two * (x * y - w * z)
    r02 = two * (x * z + w * y)
    r10 = two * (x * y + w * z)
    r11 = 1.0 - two * (x * x + z * z)
    r12 = two * (y * z - w * x)
    r20 = two * (x * z - w * y)
    r21 = two * (y * z + w * x)
    r22 = 1.0 - two * (x * x + y * y)
    row0 = torch.stack([r00, r01, r02], dim=-1)
    row1 = torch.stack([r10, r11, r12], dim=-1)
    row2 = torch.stack([r20, r21, r22], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


class BackboneUpdate(nn.Module):
    """Predicts a per-residue rigid-frame update from the single track.

    The head linearly projects the (layer-normed) single representation
    to six channels: three real components :math:`(b, c, d)` of a
    quaternion with implicit :math:`w = 1`, and three local-frame
    translation components :math:`\\Delta t_{\\text{local}}`.

    The frame update is then composed as

    .. math::

        R_i^{\\text{new}} = R_i \\, R_{\\Delta},\\qquad
        t_i^{\\text{new}} = R_i \\, \\Delta t_{\\text{local}} + t_i,

    so that under any global rigid motion :math:`(R_g, t_g)` acting on
    the input frames the output frames transform equivariantly.
    Zero-initialisation of the head's final linear layer makes the
    initial update an identity, a standard diffusion practice.

    Parameters
    ----------
    embed_dim : int
        Single-track channel size :math:`C_s`.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError('embed_dim must be positive.')
        self.norm = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, 6)
        # Zero-init so the initial update is the identity transform.
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(
        self,
        single: torch.Tensor,
        rotations: torch.Tensor,
        translations: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the backbone update.

        Parameters
        ----------
        single : torch.Tensor
            Single representation of shape ``(B, L, C_s)``.
        rotations : torch.Tensor
            Current frame rotations of shape ``(B, L, 3, 3)``.
        translations : torch.Tensor
            Current frame translations of shape ``(B, L, 3)``.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Updated ``(rotations, translations)`` of shapes ``(B, L, 3,
            3)`` and ``(B, L, 3)``.
        """
        update = self.linear(self.norm(single))
        quat_imag = update[..., :3]
        delta_t_local = update[..., 3:]
        # Construct the quaternion (1, b, c, d). Using an implicit
        # w=1 component biases the initial update toward identity.
        ones = torch.ones_like(quat_imag[..., :1])
        quat = torch.cat([ones, quat_imag], dim=-1)
        r_update = quaternion_to_rotation_matrix(quat)
        new_rotations = rotations @ r_update
        # t_new = R · Δt_local + t. Use einsum to broadcast cleanly.
        new_translations = (
            torch.einsum('blij,blj->bli', rotations, delta_t_local) +
            translations)
        return new_rotations, new_translations


# ---------------------------------------------------------------------------
# Full multi-track block and stack
# ---------------------------------------------------------------------------


class RFDiffusionTrackBlock(nn.Module):
    """One iteration of the three-track RFDiffusion architecture.

    Each block executes, in order:

    1. **Single → pair** via :class:`OuterProductMean`.
    2. **Pair self-update** via outgoing and incoming
       :class:`TriangleMultiplicativeUpdate`, starting- and ending-node
       :class:`TriangleAttention`, and a :class:`PairTransition`. After
       each update the pair is symmetrised, preserving
       :math:`z_{ij} = z_{ji}`.
    3. **Pair → single** via :class:`PairBiasedSingleAttention`,
       followed by an adaLN-style :class:`SingleTransition` modulated
       by the timestep embedding.
    4. **Frame update** via :class:`InvariantPointAttention` (using the
       updated single and pair) and :class:`BackboneUpdate`.

    Every operation that touches the pair is symmetric in :math:`(i,
    j)`, and every operation that touches the frames is built so that
    composing a global rigid motion on the input frames yields the same
    motion on the output frames.

    Parameters
    ----------
    embed_dim : int
        Single-track channel size :math:`C_s`.
    pair_dim : int
        Pair-track channel size :math:`C_z`.
    num_heads : int, default 8
        Heads for the single-track and IPA attentions.
    pair_num_heads : int, default 4
        Heads for the triangular self-attention.
    triangle_hidden_dim : int, default 64
        Hidden dimension of the triangular multiplicative update.
    triangle_head_dim : int, default 32
        Per-head channel size for triangular attention.
    opm_hidden_dim : int, default 16
        Projection size of the outer-product mean.
    num_qk_points : int, default 4
        Number of query/key 3-D points for IPA.
    num_v_points : int, default 8
        Number of value 3-D points for IPA.
    dropout : float, default 0.0
        Dropout probability for attention paths and MLPs.
    chunk_size : int, optional
        Default chunk size applied to triangular operations. ``None``
        means no chunking (dense path). Tests verify chunked and dense
        outputs match to within ``atol=1e-5``.
    """

    def __init__(self,
                 embed_dim: int,
                 pair_dim: int,
                 num_heads: int = 8,
                 pair_num_heads: int = 4,
                 triangle_hidden_dim: int = 64,
                 triangle_head_dim: int = 32,
                 opm_hidden_dim: int = 16,
                 num_qk_points: int = 4,
                 num_v_points: int = 8,
                 dropout: float = 0.0,
                 chunk_size: Optional[int] = None) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.outer_product_mean = OuterProductMean(embed_dim, pair_dim,
                                                   opm_hidden_dim)
        self.tri_mul_out = TriangleMultiplicativeUpdate(pair_dim,
                                                        triangle_hidden_dim,
                                                        outgoing=True)
        self.tri_mul_in = TriangleMultiplicativeUpdate(pair_dim,
                                                       triangle_hidden_dim,
                                                       outgoing=False)
        self.tri_attn_start = TriangleAttention(pair_dim,
                                                pair_num_heads,
                                                triangle_head_dim,
                                                starting_node=True)
        self.tri_attn_end = TriangleAttention(pair_dim,
                                              pair_num_heads,
                                              triangle_head_dim,
                                              starting_node=False)
        self.pair_transition = PairTransition(pair_dim)
        self.pair_to_single = PairBiasedSingleAttention(embed_dim,
                                                        pair_dim,
                                                        num_heads=num_heads,
                                                        dropout=dropout)
        self.single_transition = SingleTransition(embed_dim, dropout=dropout)
        self.ipa = InvariantPointAttention(embed_dim,
                                           num_heads=num_heads,
                                           num_qk_points=num_qk_points,
                                           num_v_points=num_v_points,
                                           pair_dim=pair_dim,
                                           dropout=dropout)
        self.ipa_norm = nn.LayerNorm(embed_dim)
        self.backbone_update = BackboneUpdate(embed_dim)

    def forward(
        self,
        single: torch.Tensor,
        pair: torch.Tensor,
        rotations: torch.Tensor,
        translations: torch.Tensor,
        t_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one multi-track update.

        Parameters
        ----------
        single : torch.Tensor
            Shape ``(B, L, C_s)``.
        pair : torch.Tensor
            Shape ``(B, L, L, C_z)``.
        rotations : torch.Tensor
            Shape ``(B, L, 3, 3)``.
        translations : torch.Tensor
            Shape ``(B, L, 3)``.
        t_emb : torch.Tensor
            Timestep embedding of shape ``(B, C_s)``.
        mask : torch.Tensor, optional
            Residue mask of shape ``(B, L)``.
        chunk_size : int, optional
            Override the block-level default chunk size for this call.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Updated ``(single, pair, rotations, translations)``.
        """
        active_chunk = (chunk_size
                        if chunk_size is not None else self.chunk_size)
        # Single → pair (residual + symmetrise).
        pair = pair + self.outer_product_mean(single)
        pair = 0.5 * (pair + pair.transpose(-2, -3))
        # Pair self-updates.
        pair = pair + self.tri_mul_out(pair, chunk_size=active_chunk)
        pair = 0.5 * (pair + pair.transpose(-2, -3))
        pair = pair + self.tri_mul_in(pair, chunk_size=active_chunk)
        pair = 0.5 * (pair + pair.transpose(-2, -3))
        pair = pair + self.tri_attn_start(pair, chunk_size=active_chunk)
        pair = 0.5 * (pair + pair.transpose(-2, -3))
        pair = pair + self.tri_attn_end(pair, chunk_size=active_chunk)
        pair = 0.5 * (pair + pair.transpose(-2, -3))
        pair = pair + self.pair_transition(pair)
        pair = 0.5 * (pair + pair.transpose(-2, -3))
        # Pair → single + transition.
        single = single + self.pair_to_single(single, pair, mask=mask)
        single = single + self.single_transition(single, t_emb)
        # Frame update: IPA produces an invariant single increment.
        ipa_out = self.ipa(self.ipa_norm(single),
                           rotations,
                           translations,
                           pair_repr=pair,
                           mask=mask)
        single = single + ipa_out
        rotations, translations = self.backbone_update(single, rotations,
                                                       translations)
        return single, pair, rotations, translations


class RFDiffusionMultiTrackStack(nn.Module):
    """Stacked multi-track architecture matching ``DiffusionTransformerBlock``.

    Wraps a sequence of :class:`RFDiffusionTrackBlock` updates so that
    the public :meth:`forward` signature ``forward(single, t_emb,
    attention_mask=None) -> single`` is identical to the baseline
    transformer block stack in
    :mod:`deepchem.models.torch_models.rfdiffusion`. The pair
    representation and rigid frames live entirely within the stack and
    are initialised fresh on every forward pass.

    Use :meth:`forward_tracks` from tests to retrieve the final
    ``(single, pair, rotations, translations)`` tuple — this enables
    direct equivariance checks on the frame outputs.

    Parameters
    ----------
    embed_dim : int
        Single-track channel size :math:`C_s`.
    pair_dim : int
        Pair-track channel size :math:`C_z`.
    num_blocks : int, default 2
        Number of stacked multi-track blocks.
    num_heads : int, default 8
        Heads for single-track and IPA attention.
    pair_num_heads : int, default 4
        Heads for triangular self-attention.
    chunk_size : int, optional
        Default chunk size for triangular operations.
    dropout : float, default 0.0
        Shared dropout probability.
    max_relative_position : int, default 32
        Clip radius for the relative-position embedding used to
        initialise the pair representation.
    opm_hidden_dim : int, default 16
        Hidden size of the outer-product-mean projection used in each
        block.
    triangle_hidden_dim : int, default 64
        Hidden size for the triangular multiplicative update.
    triangle_head_dim : int, default 32
        Per-head channel size for triangular attention.
    num_qk_points : int, default 4
        IPA query/key 3-D point count.
    num_v_points : int, default 8
        IPA value 3-D point count.

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.torch_models.rfdiffusion_sequence_track import (
    ...     RFDiffusionMultiTrackStack)
    >>> stack = RFDiffusionMultiTrackStack(
    ...     embed_dim=32, pair_dim=16, num_blocks=2, num_heads=4,
    ...     pair_num_heads=2, triangle_hidden_dim=16,
    ...     triangle_head_dim=8, opm_hidden_dim=8)
    >>> single = torch.randn(2, 6, 32)
    >>> t_emb = torch.randn(2, 32)
    >>> out = stack(single, t_emb)
    >>> out.shape
    torch.Size([2, 6, 32])
    """

    def __init__(self,
                 embed_dim: int,
                 pair_dim: int,
                 num_blocks: int = 2,
                 num_heads: int = 8,
                 pair_num_heads: int = 4,
                 chunk_size: Optional[int] = None,
                 dropout: float = 0.0,
                 max_relative_position: int = 32,
                 opm_hidden_dim: int = 16,
                 triangle_hidden_dim: int = 64,
                 triangle_head_dim: int = 32,
                 num_qk_points: int = 4,
                 num_v_points: int = 8) -> None:
        super().__init__()
        if num_blocks <= 0:
            raise ValueError('num_blocks must be positive.')
        self.embed_dim = embed_dim
        self.pair_dim = pair_dim
        self.num_blocks = num_blocks
        self.chunk_size = chunk_size
        # Pair initialiser: relpos + initial OPM-style outer product
        # mean of the input single representation.
        self.rel_pos = RelativePositionEmbedding(pair_dim,
                                                 max_relative_position)
        self.initial_opm = OuterProductMean(embed_dim, pair_dim, opm_hidden_dim)
        self.blocks = nn.ModuleList([
            RFDiffusionTrackBlock(embed_dim,
                                  pair_dim,
                                  num_heads=num_heads,
                                  pair_num_heads=pair_num_heads,
                                  triangle_hidden_dim=triangle_hidden_dim,
                                  triangle_head_dim=triangle_head_dim,
                                  opm_hidden_dim=opm_hidden_dim,
                                  num_qk_points=num_qk_points,
                                  num_v_points=num_v_points,
                                  dropout=dropout,
                                  chunk_size=chunk_size)
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)

    # -- DropIn-compatible forward ------------------------------------

    def forward(self,
                single: torch.Tensor,
                t_emb: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Drop-in replacement for the baseline transformer block stack.

        Parameters
        ----------
        single : torch.Tensor
            Single representation of shape ``(B, L, C_s)``.
        t_emb : torch.Tensor
            Timestep embedding of shape ``(B, C_s)``.
        attention_mask : torch.Tensor, optional
            Boolean mask of shape ``(B, L)``.

        Returns
        -------
        torch.Tensor
            Updated single representation, shape ``(B, L, C_s)``.
        """
        single_out, _, _, _ = self.forward_tracks(single, t_emb, attention_mask)
        return single_out

    # -- Full multi-track API (for tests and downstream consumers) ----

    def forward_tracks(
        self,
        single: torch.Tensor,
        t_emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotations: Optional[torch.Tensor] = None,
        translations: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run all tracks and return
        ``(single, pair, rotations, translations)``.

        Parameters
        ----------
        single : torch.Tensor
            Shape ``(B, L, C_s)``.
        t_emb : torch.Tensor
            Shape ``(B, C_s)``.
        attention_mask : torch.Tensor, optional
            Shape ``(B, L)``.
        rotations : torch.Tensor, optional
            Initial frame rotations of shape ``(B, L, 3, 3)``. If
            ``None``, identity rotations are used.
        translations : torch.Tensor, optional
            Initial frame translations of shape ``(B, L, 3)``. If
            ``None``, zero translations are used.
        chunk_size : int, optional
            Override the stack-level default chunk size for this call.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Final ``(single, pair, rotations, translations)``.
        """
        if single.dim() != 3:
            raise ValueError('single must be 3D (B, L, C_s).')
        if single.size(-1) != self.embed_dim:
            raise ValueError(
                f'Expected embed_dim {self.embed_dim}, got {single.size(-1)}.')
        if t_emb.dim() != 2 or t_emb.size(-1) != self.embed_dim:
            raise ValueError('t_emb must have shape (B, embed_dim).')

        batch, seq_len, _ = single.shape
        device = single.device
        dtype = single.dtype

        # Mask coercion.
        mask = None
        if attention_mask is not None:
            mask = attention_mask.to(device=device, dtype=torch.bool)
            if mask.shape != (batch, seq_len):
                raise ValueError('attention_mask must have shape (B, L).')

        # Initial frames.  Use concrete Tensor-typed locals (R, T) so mypy
        # can verify the Tuple[Tensor, Tensor, Tensor, Tensor] return type.
        if rotations is None:
            R, t_id = make_identity_rigid((batch, seq_len),
                                          device=device,
                                          dtype=dtype)
            T: torch.Tensor = (translations
                               if translations is not None else t_id)
        else:
            R = rotations
            T = (translations if translations is not None else torch.zeros(
                batch, seq_len, 3, device=device, dtype=dtype))

        # Initial pair: symmetric relpos + symmetric OPM.
        rel = self.rel_pos(seq_len, device=device).to(dtype)
        pair = rel.unsqueeze(0).expand(batch, -1, -1, -1).contiguous()
        pair = pair + self.initial_opm(single)
        pair = 0.5 * (pair + pair.transpose(-2, -3))

        active_chunk = (chunk_size
                        if chunk_size is not None else self.chunk_size)

        for block in self.blocks:
            single, pair, R, T = block(single,
                                       pair,
                                       R,
                                       T,
                                       t_emb,
                                       mask=mask,
                                       chunk_size=active_chunk)
            # Re-mask single for masked residues so that downstream
            # consumers see exactly zeros for padded entries.
            if mask is not None:
                single = single * mask.unsqueeze(-1).to(single.dtype)

        single = self.final_norm(single)
        return single, pair, R, T

"""RoseTTAFold-style multi-track architecture for RFDiffusion.

This module implements the three-track network used by RFDiffusion to
denoise protein backbones:

* **1D track** — a single representation :math:`s_i \\in \\mathbb{R}^{L
  \\times C_s}` updated by pair-biased self-attention and a residual
  feed-forward block.
* **2D track** — a pair representation :math:`z_{ij} \\in
  \\mathbb{R}^{L \\times L \\times C_z}` updated by an outer-product
  mean from the single track, triangular multiplicative updates
  (outgoing and incoming), triangular self-attention (starting and
  ending node), and a pair transition. Because every pair operation is
  :math:`\\mathcal{O}(L^2)` (or :math:`\\mathcal{O}(L^3)` for the
  triangular updates), each pair operation accepts a ``chunk_size``
  argument that bounds the working-set size along the chunked axis.
* **3D track** — rigid frames :math:`T_i = (R_i, t_i) \\in SE(3)`
  updated by the verified Invariant Point Attention layer
  (Algorithm 22 of [Jumper2021]_) followed by a backbone-update head
  that predicts a per-residue translation :math:`\\Delta t_i \\in
  \\mathbb{R}^3` (in the local frame) and a quaternion residual
  :math:`(1, b, c, d)` from which a rotation update
  :math:`R_{\\Delta}` is built. Frames are composed as
  :math:`R_i^{\\text{new}} = R_i R_{\\Delta}` and
  :math:`t_i^{\\text{new}} = R_i \\Delta t_i + t_i`.

All pair updates are symmetrised so that the invariant
:math:`z_{ij} = z_{ji}` is preserved across the full stack — a
property exercised by the test suite.

The public class :class:`RFDiffusionMultiTrackStack` exposes a forward
signature identical to a stack of
:class:`~deepchem.models.torch_models.rfdiffusion.DiffusionTransformerBlock`
layers, ``forward(single, t_emb, attention_mask=None) -> single``, so
it is a drop-in replacement for the baseline denoiser's transformer
stack. A second method :meth:`forward_tracks` returns the full
``(single, pair, R, t)`` tuple for tests that exercise frame-level
equivariance.

Developer-facing summary
------------------------
For most integrations, :class:`RFDiffusionMultiTrackStack` is the main
entry point. Use ``forward(single, t_emb, attention_mask=None)`` when
you only need updated single-residue features, and use
``forward_tracks(...)`` when a downstream module needs direct access to
the pair representation or the updated rigid frames.

References
----------
.. [Jumper2021] Jumper, J. *et al.* "Highly accurate protein structure
   prediction with AlphaFold." *Nature* **596** (2021): 583–589.
.. [Baek2021] Baek, M. *et al.* "Accurate prediction of protein structures
   and interactions using a three-track neural network." *Science*
   **373** (2021): 871–876.
.. [Watson2023] Watson, J. L. *et al.* "De novo design of protein structure
   and function with RFdiffusion." *Nature* **620** (2023): 1089–1100.
"""

import math
from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    raise ImportError('rfdiffusion_tracks requires PyTorch to be installed.')

from deepchem.models.torch_models.rfdiffusion_se3 import (
    InvariantPointAttention,
    make_identity_rigid,
)


# ---------------------------------------------------------------------------
# Embedding utilities
# ---------------------------------------------------------------------------


class RelativePositionEmbedding(nn.Module):
    """Symmetric clipped relative-position embedding for the pair track.

    Given a sequence of length :math:`L` and a clip radius
    :math:`r_{\\max}`, this module returns a tensor
    :math:`p_{ij} \\in \\mathbb{R}^{L \\times L \\times C_z}` where each
    entry is the embedding of :math:`\\min(|i - j|, r_{\\max})`. Because
    the index depends on :math:`|i-j|` rather than :math:`i-j`, the
    resulting embedding is exactly symmetric:
    :math:`p_{ij} = p_{ji}`.

    Parameters
    ----------
    pair_dim : int
        Output channel size :math:`C_z`.
    max_relative_position : int, default 32
        Clip radius :math:`r_{\\max}`. Indices beyond this distance map
        to the same embedding bucket.

    Examples
    --------
    >>> import torch
    >>> rel = RelativePositionEmbedding(pair_dim=16, max_relative_position=4)
    >>> emb = rel(8)
    >>> emb.shape
    torch.Size([8, 8, 16])
    """

    def __init__(self,
                 pair_dim: int,
                 max_relative_position: int = 32) -> None:
        super().__init__()
        if pair_dim <= 0:
            raise ValueError('pair_dim must be positive.')
        if max_relative_position <= 0:
            raise ValueError('max_relative_position must be positive.')
        self.pair_dim = pair_dim
        self.max_relative_position = max_relative_position
        self.embedding = nn.Embedding(max_relative_position + 1, pair_dim)

    def forward(self, seq_len: int, device=None) -> torch.Tensor:
        """Build the :math:`(L, L, C_z)` symmetric relative-position tensor.

        Parameters
        ----------
        seq_len : int
            Sequence length :math:`L`.
        device : torch.device, optional
            Device on which to allocate the position index tensor.

        Returns
        -------
        torch.Tensor
            Relative-position embeddings of shape ``(L, L, pair_dim)``.
        """
        device = device or self.embedding.weight.device
        idx = torch.arange(seq_len, device=device)
        diff = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        diff = diff.clamp(max=self.max_relative_position)
        return self.embedding(diff)


# ---------------------------------------------------------------------------
# 1D → 2D update: outer-product mean
# ---------------------------------------------------------------------------


class OuterProductMean(nn.Module):
    """Outer-product update from the single track to the pair track.

    For each pair :math:`(i, j)`, computes the symmetrised flattened
    outer product

    .. math::

        o_{ij} = W_{\\text{out}}\\,\\big(L_a(s_i) \\otimes L_b(s_j)\\big)
        + W_{\\text{out}}\\,\\big(L_a(s_j) \\otimes L_b(s_i)\\big),

    where :math:`L_a, L_b: \\mathbb{R}^{C_s} \\to \\mathbb{R}^{c_h}` are
    linear projections and the symmetrisation enforces
    :math:`o_{ij} = o_{ji}` exactly.

    Parameters
    ----------
    embed_dim : int
        Single-track channel size :math:`C_s`.
    pair_dim : int
        Pair-track channel size :math:`C_z`.
    hidden_dim : int, default 32
        Projection size :math:`c_h` for the outer product.

    Notes
    -----
    The pre-projection layer norm follows AlphaFold2's outer-product-mean
    placement; only one residue context (the single representation
    itself) participates because RFDiffusion does not use an MSA stack.
    """

    def __init__(self,
                 embed_dim: int,
                 pair_dim: int,
                 hidden_dim: int = 32) -> None:
        super().__init__()
        if embed_dim <= 0 or pair_dim <= 0 or hidden_dim <= 0:
            raise ValueError(
                'embed_dim, pair_dim and hidden_dim must all be positive.')
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.linear_a = nn.Linear(embed_dim, hidden_dim)
        self.linear_b = nn.Linear(embed_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim * hidden_dim, pair_dim)

    def forward(self, single: torch.Tensor) -> torch.Tensor:
        """Compute the outer-product mean.

        Parameters
        ----------
        single : torch.Tensor
            Single representation of shape ``(B, L, C_s)``.

        Returns
        -------
        torch.Tensor
            Pair update of shape ``(B, L, L, C_z)`` (symmetric in i, j).
        """
        single = self.layer_norm(single)
        a = self.linear_a(single)  # (B, L, c_h)
        b = self.linear_b(single)  # (B, L, c_h)
        # Outer product per pair: (B, L, L, c_h, c_h) → flatten last two.
        outer_ab = torch.einsum('bic,bjd->bijcd', a, b)
        outer_ba = torch.einsum('bjc,bid->bijcd', a, b)
        outer = 0.5 * (outer_ab + outer_ba)
        flat = outer.flatten(start_dim=-2)
        return self.linear_out(flat)


# ---------------------------------------------------------------------------
# Triangular multiplicative update
# ---------------------------------------------------------------------------


class TriangleMultiplicativeUpdate(nn.Module):
    """Triangle multiplicative update for the pair track.

    Implements AlphaFold2 Algorithms 11 / 12 [Jumper2021]_. The
    "outgoing" variant computes

    .. math::

        \\tilde z_{ij} = g_{ij} \\odot W_{\\text{out}}\\,
        \\mathrm{LN}\\Big( \\sum_k a_{ik} \\odot b_{jk} \\Big),

    and the "incoming" variant replaces the summation by
    :math:`\\sum_k a_{ki} \\odot b_{kj}`. The values :math:`a, b, g`
    are gated linear projections of the layer-normalised pair
    representation.

    A ``chunk_size`` argument controls the working-set size along the
    :math:`i` axis when forming the cubic-cost contraction, bounding
    peak memory to :math:`\\mathcal{O}(\\text{chunk\\_size} \\cdot L^2
    \\cdot c_h)` independently of :math:`L`.

    Parameters
    ----------
    pair_dim : int
        Input/output channel size :math:`C_z`.
    hidden_dim : int, default 128
        Hidden channel size :math:`c_h` for the gated projections.
    outgoing : bool, default True
        If True, use the outgoing variant (sum over :math:`a_{ik}
        b_{jk}`); otherwise use the incoming variant (sum over
        :math:`a_{ki} b_{kj}`).
    """

    def __init__(self,
                 pair_dim: int,
                 hidden_dim: int = 128,
                 outgoing: bool = True) -> None:
        super().__init__()
        if pair_dim <= 0 or hidden_dim <= 0:
            raise ValueError('pair_dim and hidden_dim must be positive.')
        self.outgoing = outgoing
        self.layer_norm_in = nn.LayerNorm(pair_dim)
        self.linear_a = nn.Linear(pair_dim, hidden_dim)
        self.linear_a_gate = nn.Linear(pair_dim, hidden_dim)
        self.linear_b = nn.Linear(pair_dim, hidden_dim)
        self.linear_b_gate = nn.Linear(pair_dim, hidden_dim)
        self.linear_g = nn.Linear(pair_dim, pair_dim)
        self.layer_norm_out = nn.LayerNorm(hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, pair_dim)

    def forward(self,
                pair: torch.Tensor,
                chunk_size: Optional[int] = None) -> torch.Tensor:
        """Apply the triangular multiplicative update.

        Parameters
        ----------
        pair : torch.Tensor
            Pair representation of shape ``(B, L, L, C_z)``.
        chunk_size : int, optional
            If given, chunk the :math:`i` axis of the cubic contraction
            into pieces of at most ``chunk_size`` rows. Must produce
            the same output as the unchunked path (verified by tests).

        Returns
        -------
        torch.Tensor
            Updated pair representation, same shape as input.
        """
        if pair.dim() != 4:
            raise ValueError(
                'pair must have shape (batch, L, L, pair_dim), got '
                f'{tuple(pair.shape)}.')
        z = self.layer_norm_in(pair)
        a = torch.sigmoid(self.linear_a_gate(z)) * self.linear_a(z)
        b = torch.sigmoid(self.linear_b_gate(z)) * self.linear_b(z)
        g = torch.sigmoid(self.linear_g(z))
        contracted = self._chunked_contract(a, b, chunk_size)
        contracted = self.layer_norm_out(contracted)
        return g * self.linear_out(contracted)

    def _chunked_contract(self,
                          a: torch.Tensor,
                          b: torch.Tensor,
                          chunk_size: Optional[int]) -> torch.Tensor:
        """Compute :math:`\\sum_k a_{*k} \\odot b_{*k}` chunked over ``i``."""
        seq_len = a.size(1)
        if chunk_size is None or chunk_size >= seq_len:
            if self.outgoing:
                return torch.einsum('bikc,bjkc->bijc', a, b)
            return torch.einsum('bkic,bkjc->bijc', a, b)
        results = []
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            if self.outgoing:
                # a_chunk: (B, chunk, L, c) — k dim is L.
                a_chunk = a[:, start:end]
                results.append(torch.einsum('bikc,bjkc->bijc', a_chunk, b))
            else:
                # Need a[k, i_chunk]: slice on i (axis 2) of bkic.
                a_chunk = a[:, :, start:end]
                results.append(torch.einsum('bkic,bkjc->bijc', a_chunk, b))
        return torch.cat(results, dim=1)


# ---------------------------------------------------------------------------
# Triangular self-attention
# ---------------------------------------------------------------------------


class TriangleAttention(nn.Module):
    """Triangular self-attention for the pair track.

    Implements AlphaFold2 Algorithms 13 / 14 [Jumper2021]_. In the
    *starting-node* variant the attention runs along columns of each
    row of the pair representation: for each :math:`i`, queries
    :math:`q_{ij}` attend to keys :math:`k_{ik}` with the bias term
    :math:`b_{jk}` derived linearly from the pair representation at
    rows shared with the row being attended over. The *ending-node*
    variant operates symmetrically along columns.

    Memory is bounded by chunking over the :math:`i` axis (rows for the
    starting-node variant, columns for the ending-node variant).

    Parameters
    ----------
    pair_dim : int
        Input/output channel size :math:`C_z`.
    num_heads : int, default 4
        Number of attention heads :math:`H`.
    head_dim : int, default 32
        Per-head channel size :math:`d_h`.
    starting_node : bool, default True
        Selects the starting-node (True) or ending-node (False) variant.
    """

    def __init__(self,
                 pair_dim: int,
                 num_heads: int = 4,
                 head_dim: int = 32,
                 starting_node: bool = True) -> None:
        super().__init__()
        if pair_dim <= 0 or num_heads <= 0 or head_dim <= 0:
            raise ValueError(
                'pair_dim, num_heads and head_dim must be positive.')
        self.pair_dim = pair_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.starting_node = starting_node
        self.layer_norm = nn.LayerNorm(pair_dim)
        inner = num_heads * head_dim
        self.linear_q = nn.Linear(pair_dim, inner, bias=False)
        self.linear_k = nn.Linear(pair_dim, inner, bias=False)
        self.linear_v = nn.Linear(pair_dim, inner, bias=False)
        self.linear_bias = nn.Linear(pair_dim, num_heads, bias=False)
        self.linear_gate = nn.Linear(pair_dim, inner)
        self.linear_out = nn.Linear(inner, pair_dim)
        self.register_buffer('scale',
                             torch.tensor(1.0 / math.sqrt(head_dim)))

    def forward(self,
                pair: torch.Tensor,
                chunk_size: Optional[int] = None) -> torch.Tensor:
        """Apply triangular self-attention.

        Parameters
        ----------
        pair : torch.Tensor
            Pair representation of shape ``(B, L, L, C_z)``.
        chunk_size : int, optional
            Chunk size for the row-index axis. Must produce identical
            results to the unchunked path.

        Returns
        -------
        torch.Tensor
            Updated pair representation, same shape as input.
        """
        if pair.dim() != 4:
            raise ValueError('pair must be 4D (B, L, L, C_z).')
        # For ending-node, swap i and j axes, run the starting-node
        # kernel, then swap back. This guarantees identical numerics
        # to a hand-written ending-node implementation.
        z = pair if self.starting_node else pair.transpose(-2, -3)
        z = self.layer_norm(z)
        q = self._split_heads(self.linear_q(z))
        k = self._split_heads(self.linear_k(z))
        v = self._split_heads(self.linear_v(z))
        bias = self.linear_bias(z)  # (B, L, L, H) — uses row-i context.
        gate = torch.sigmoid(self.linear_gate(z))

        out = self._chunked_attend(q, k, v, bias, chunk_size)
        out = out * gate
        out = self.linear_out(out)
        if not self.starting_node:
            out = out.transpose(-2, -3)
        return out

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape ``(B, L, L, H*d) -> (B, L, L, H, d)``."""
        batch, n1, n2, _ = tensor.shape
        return tensor.view(batch, n1, n2, self.num_heads, self.head_dim)

    def _chunked_attend(self,
                        q: torch.Tensor,
                        k: torch.Tensor,
                        v: torch.Tensor,
                        bias: torch.Tensor,
                        chunk_size: Optional[int]) -> torch.Tensor:
        """Row-wise scaled-dot-product attention with an additive bias.

        For each row :math:`i`, the attention is computed across columns
        :math:`(j, k)`. The bias :math:`b_{jk} = W_b z_{jk}` depends
        only on the *column* pair :math:`(j, k)` and is broadcast across
        all rows :math:`i` — this is precisely the AlphaFold2 starting-
        node triangle-attention formulation [Jumper2021]_. Because
        logits and attention weights are formed independently for each
        row :math:`i`, chunking along :math:`i` is exactly numerically
        equivalent to the dense path.
        """
        seq_len = q.size(1)
        if chunk_size is None or chunk_size >= seq_len:
            return self._attend_block(q, k, v, bias)
        outputs = []
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            outputs.append(
                self._attend_block(q[:, start:end], k[:, start:end],
                                   v[:, start:end], bias))
        return torch.cat(outputs, dim=1)

    @staticmethod
    def _attend_block(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      bias: torch.Tensor) -> torch.Tensor:
        """Compute attention for a (possibly chunked) block of rows.

        Parameters
        ----------
        q, k, v : torch.Tensor
            Per-row queries/keys/values of shape ``(B, L_i, L, H, d)``.
        bias : torch.Tensor
            Pair bias of shape ``(B, L, L, H)`` indexed by ``(j, k)``;
            broadcast across the row axis :math:`i`.
        """
        scale = 1.0 / math.sqrt(q.size(-1))
        logits = torch.einsum('bijhd,bikhd->bhijk', q, k) * scale
        # bias: (B, L_j, L_k, H) → (B, H, 1, L_j, L_k) so it broadcasts
        # across the row index i.
        logits = logits + bias.permute(0, 3, 1, 2).unsqueeze(2)
        attn = F.softmax(logits, dim=-1)
        out = torch.einsum('bhijk,bikhd->bijhd', attn, v)
        return out.flatten(start_dim=-2)


# ---------------------------------------------------------------------------
# Pair transition (feed-forward)
# ---------------------------------------------------------------------------


class PairTransition(nn.Module):
    """Two-layer feed-forward block applied position-wise to the pair."""

    def __init__(self, pair_dim: int, expansion: int = 4) -> None:
        super().__init__()
        if pair_dim <= 0 or expansion <= 0:
            raise ValueError('pair_dim and expansion must be positive.')
        self.layer_norm = nn.LayerNorm(pair_dim)
        self.linear_1 = nn.Linear(pair_dim, pair_dim * expansion)
        self.linear_2 = nn.Linear(pair_dim * expansion, pair_dim)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        """Apply the pair transition.

        Parameters
        ----------
        pair : torch.Tensor
            Pair representation of shape ``(B, L, L, C_z)``.

        Returns
        -------
        torch.Tensor
            Transformed pair representation, same shape as input.
        """
        z = self.layer_norm(pair)
        return self.linear_2(F.relu(self.linear_1(z)))


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
        self.register_buffer('scale',
                             torch.tensor(1.0 / math.sqrt(self.head_dim)))

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
        logits = torch.einsum('bihd,bjhd->bhij', q, k) * float(self.scale)
        bias = self.linear_bias(z_norm).permute(0, 3, 1, 2)
        logits = logits + bias
        if mask is not None:
            mask_bool = mask.to(dtype=torch.bool)
            key_mask = mask_bool.view(batch, 1, 1, seq_len)
            logits = logits.masked_fill(
                ~key_mask, torch.finfo(logits.dtype).min)
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
            torch.einsum('blij,blj->bli', rotations, delta_t_local)
            + translations)
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
        self.tri_mul_out = TriangleMultiplicativeUpdate(
            pair_dim, triangle_hidden_dim, outgoing=True)
        self.tri_mul_in = TriangleMultiplicativeUpdate(
            pair_dim, triangle_hidden_dim, outgoing=False)
        self.tri_attn_start = TriangleAttention(pair_dim,
                                                pair_num_heads,
                                                triangle_head_dim,
                                                starting_node=True)
        self.tri_attn_end = TriangleAttention(pair_dim,
                                              pair_num_heads,
                                              triangle_head_dim,
                                              starting_node=False)
        self.pair_transition = PairTransition(pair_dim)
        self.pair_to_single = PairBiasedSingleAttention(
            embed_dim,
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
        active_chunk = chunk_size if chunk_size is not None else self.chunk_size
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
        ipa_out = self.ipa(self.ipa_norm(single), rotations, translations,
                           pair_repr=pair, mask=mask)
        single = single + ipa_out
        rotations, translations = self.backbone_update(
            single, rotations, translations)
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
        self.initial_opm = OuterProductMean(embed_dim, pair_dim,
                                            opm_hidden_dim)
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
                attention_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
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
        single_out, _, _, _ = self.forward_tracks(single, t_emb,
                                                  attention_mask)
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
        """Run all tracks and return ``(single, pair, rotations, translations)``.

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
            raise ValueError(
                't_emb must have shape (B, embed_dim).')

        batch, seq_len, _ = single.shape
        device = single.device
        dtype = single.dtype

        # Mask coercion.
        mask = None
        if attention_mask is not None:
            mask = attention_mask.to(device=device, dtype=torch.bool)
            if mask.shape != (batch, seq_len):
                raise ValueError(
                    'attention_mask must have shape (B, L).')

        # Initial frames.
        if rotations is None:
            rotations, translations_id = make_identity_rigid(
                (batch, seq_len), device=device, dtype=dtype)
            if translations is None:
                translations = translations_id
        elif translations is None:
            translations = torch.zeros(batch, seq_len, 3,
                                       device=device, dtype=dtype)

        # Initial pair: symmetric relpos + symmetric OPM.
        rel = self.rel_pos(seq_len, device=device).to(dtype)
        pair = rel.unsqueeze(0).expand(batch, -1, -1, -1).contiguous()
        pair = pair + self.initial_opm(single)
        pair = 0.5 * (pair + pair.transpose(-2, -3))

        active_chunk = (chunk_size
                        if chunk_size is not None else self.chunk_size)

        for block in self.blocks:
            single, pair, rotations, translations = block(
                single, pair, rotations, translations, t_emb,
                mask=mask, chunk_size=active_chunk)
            # Re-mask single for masked residues so that downstream
            # consumers see exactly zeros for padded entries.
            if mask is not None:
                single = single * mask.unsqueeze(-1).to(single.dtype)

        single = self.final_norm(single)
        return single, pair, rotations, translations

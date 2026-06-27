"""Pair-track (2D) update blocks for the RFDiffusion multi-track network.

RFDiffusion denoises protein backbones with a RoseTTAFold-style network
that keeps three coupled representations in sync: a 1D single-residue
track, a 2D pairwise track :math:`z_{ij}`, and a 3D rigid-frame track.
This module holds the **2D pair-track** half of that stack — the blocks
that build and refine the pairwise representation before it feeds back
into the single and structure tracks.

The blocks follow the AlphaFold2 / RoseTTAFold pair-stack recipe:

* :class:`RelativePositionEmbedding` seeds the pair tensor from clipped
  relative sequence offsets :math:`|i - j|`.
* :class:`OuterProductMean` injects the 1D single track into the 2D
  pair track through a symmetrised outer product.
* :class:`TriangleMultiplicativeUpdate` (outgoing / incoming) and
  :class:`TriangleAttention` (starting / ending node) are the two
  triangle operations that mix pair edges while respecting the
  triangle-inequality structure of a distance-like tensor.
* :class:`PairTransition` is the position-wise feed-forward that closes
  each pair-stack block.

The seed embedding and the outer-product mean are symmetric by
construction (:math:`z_{ij} = z_{ji}`); the triangle operations are the
standard AlphaFold2 building blocks that the full track block later
combines into a symmetry-preserving update. Each triangle operation
costs :math:`\\mathcal{O}(L^3)` / :math:`\\mathcal{O}(L^2)`, so it takes
an optional ``chunk_size`` that bounds peak memory without changing the
numerics — the test suite checks the chunked path matches the dense one
exactly.

This is the 2D-track half of the multi-track stack. It sits alongside
the SE(3) rigid-frame utilities (PR #4982) and the Invariant Point
Attention block (PR #5028); the later track-integration PR wires the
1D, 2D, and 3D tracks together into a full block. Nothing here imports
those modules, so the pair stack stands on its own.

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
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    raise ImportError(
        'rfdiffusion_pair_track requires PyTorch to be installed.')

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

    def __init__(self, pair_dim: int, max_relative_position: int = 32) -> None:
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

    def _chunked_contract(self, a: torch.Tensor, b: torch.Tensor,
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
        self.register_buffer('scale', torch.tensor(1.0 / math.sqrt(head_dim)))

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

    def _chunked_attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
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

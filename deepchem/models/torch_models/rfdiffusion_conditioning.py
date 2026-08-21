"""Conditioning modules for RFDiffusion All-Atom.

Two conditioning mechanisms used on top of the core RFdiffusion denoiser
when generating a chain that needs to know about something outside
itself [Watson2023]_, [Krishna2024]_:

- :class:`LengthConditioning` -- lets the denoiser know the target chain
  length up front (useful for symmetric-assembly and fixed-length binder
  design, where the timestep embedding alone doesn't carry that
  information).
- :class:`BinderCrossAttention` -- standard multi-head cross-attention
  from the generated chain onto a frozen partner representation (e.g. a
  target protein a binder is being designed against), so the generated
  chain's features can depend on the partner without the partner itself
  being denoised.

References
----------
.. [Watson2023] Watson, J. L., et al. "De novo design of protein
   structure and function with RFdiffusion." Nature 620 (2023)
   1089-1100.
.. [Krishna2024] Krishna, R., et al. "Generalized biomolecular modeling
   and design with RoseTTAFold All-Atom." Science 384 (2024) eadl2528.

Notes
-----
This module requires PyTorch to be installed.
"""

import math
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    raise ImportError(
        'rfdiffusion_conditioning requires PyTorch to be installed.')

__all__ = ['LengthConditioning', 'BinderCrossAttention']


class LengthConditioning(nn.Module):
    """Embeds a target chain length into a conditioning vector.

    Normalizes the length by ``max_length`` and passes it through a
    small MLP, the same shape of transform used for the timestep
    embedding elsewhere in this stack. The output is meant to be added
    to (or concatenated with) the timestep embedding before it reaches
    the denoiser blocks.

    Parameters
    ----------
    embed_dim : int
        Output embedding dimension.
    max_length : int, default 1000
        Chain length used to normalize the input to roughly ``[0, 1]``.
    hidden_dim : int, optional
        Hidden dimension of the MLP. Defaults to ``embed_dim``.

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.torch_models.rfdiffusion_conditioning import (
    ...     LengthConditioning)
    >>> cond = LengthConditioning(embed_dim=32, max_length=500)
    >>> lengths = torch.tensor([50, 120, 500])
    >>> out = cond(lengths)
    >>> out.shape
    torch.Size([3, 32])
    """

    def __init__(self,
                 embed_dim: int,
                 max_length: int = 1000,
                 hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError('embed_dim must be positive.')
        if max_length <= 0:
            raise ValueError('max_length must be positive.')
        hidden_dim = hidden_dim or embed_dim
        self.max_length = max_length
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, lengths: torch.Tensor) -> torch.Tensor:
        """Embed a batch of target lengths.

        Parameters
        ----------
        lengths : torch.Tensor
            Integer or float chain lengths, shape ``(batch,)``.

        Returns
        -------
        torch.Tensor
            Length embeddings, shape ``(batch, embed_dim)``.
        """
        normalized = (lengths.float() / self.max_length).unsqueeze(-1)
        return self.net(normalized)


class BinderCrossAttention(nn.Module):
    """Multi-head cross-attention onto a frozen partner representation.

    The generated chain's single representation is used as the query;
    a partner chain's (frozen, not denoised) single representation is
    used as the key and value. This lets a binder design condition on
    its target without the target's own features being part of the
    diffusion process.

    Parameters
    ----------
    embed_dim : int
        Channel size of both the query and the partner representation.
    num_heads : int, default 8
        Number of attention heads.
    dropout : float, default 0.0
        Dropout probability applied to the attention weights.

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.torch_models.rfdiffusion_conditioning import (
    ...     BinderCrossAttention)
    >>> attn = BinderCrossAttention(embed_dim=32, num_heads=4)
    >>> query = torch.randn(2, 10, 32)   # generated chain
    >>> partner = torch.randn(2, 15, 32)  # frozen target
    >>> out = attn(query, partner)
    >>> out.shape
    torch.Size([2, 10, 32])
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.0) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError('embed_dim must be positive.')
        if num_heads <= 0:
            raise ValueError('num_heads must be positive.')
        if embed_dim % num_heads != 0:
            raise ValueError('embed_dim must be divisible by num_heads.')
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                query: torch.Tensor,
                partner: torch.Tensor,
                partner_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Attend from the generated chain onto the frozen partner.

        Parameters
        ----------
        query : torch.Tensor
            Generated-chain single representation, shape ``(batch,
            L_query, embed_dim)``.
        partner : torch.Tensor
            Frozen partner single representation, shape ``(batch,
            L_partner, embed_dim)``.
        partner_mask : torch.Tensor, optional
            Boolean validity mask over the partner residues, shape
            ``(batch, L_partner)``. ``True``/1 marks a valid (attendable)
            partner residue.

        Returns
        -------
        torch.Tensor
            Updated query representation, same shape as ``query``.
        """
        batch, l_query, _ = query.shape
        l_partner = partner.shape[1]

        q = self.q_proj(query).view(batch, l_query, self.num_heads,
                                    self.head_dim).transpose(1, 2)
        k = self.k_proj(partner).view(batch, l_partner, self.num_heads,
                                      self.head_dim).transpose(1, 2)
        v = self.v_proj(partner).view(batch, l_partner, self.num_heads,
                                      self.head_dim).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if partner_mask is not None:
            bias = torch.zeros_like(partner_mask, dtype=logits.dtype)
            bias = bias.masked_fill(~partner_mask.bool(), float('-inf'))
            logits = logits + bias[:, None, None, :]

        weights = self.dropout(F.softmax(logits, dim=-1))
        out = torch.matmul(weights, v)  # (batch, heads, L_query, head_dim)
        out = out.transpose(1, 2).reshape(batch, l_query, self.embed_dim)
        return self.out_proj(out)

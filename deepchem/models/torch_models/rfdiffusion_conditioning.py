"""Advanced conditioning blocks for RFDiffusion.

Two conditioning mechanisms are exposed here:

* :class:`BinderCrossAttention` — a single multi-head cross-attention
  block over a *frozen* target representation. Used to condition the
  generated chain on a binder partner without backpropagating into
  the target weights [Watson2023]_.
* :class:`LengthConditioning` — sinusoidal embedding of a desired
  output chain length followed by an MLP, added to the time embedding
  so the model can be steered to specific lengths.

Developer-facing summary
------------------------
Use :class:`LengthConditioning` when a caller wants to bias generation
to a target chain length, and use :class:`BinderCrossAttention` when a
generated chain should attend to a frozen partner representation. Both
blocks return tensors that can be added into an existing denoiser
without changing its outer training loop.

References
----------
.. [Watson2023] Watson, J. L., et al. "De novo design of protein
   structure and function with RFdiffusion." Nature 620 (2023)
   1089-1100.
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

__all__ = [
    'BinderCrossAttention',
    'LengthConditioning',
    'sinusoidal_length_embedding',
]


def sinusoidal_length_embedding(length: torch.Tensor,
                                embed_dim: int,
                                max_period: float = 10000.0
                                ) -> torch.Tensor:
    """Sinusoidal embedding of integer lengths.

    Equivalent in form to the timestep embedding of :class:`Transformer`
    architectures: half of the embedding is ``sin``, half is ``cos``,
    with frequencies geometrically spaced between 1 and
    ``1/max_period``.

    Parameters
    ----------
    length : torch.Tensor
        Integer or float tensor of shape ``(B,)`` containing the
        requested chain lengths.
    embed_dim : int
        Embedding dimensionality (must be even).
    max_period : float, default 10000.0
        Largest period used in the geometric frequency progression.

    Returns
    -------
    torch.Tensor
        Embedding of shape ``(B, embed_dim)``.
    """
    if embed_dim % 2 != 0:
        raise ValueError('embed_dim must be even.')
    half = embed_dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32,
                       device=length.device) / half)
    args = length.to(torch.float32).unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class LengthConditioning(nn.Module):
    """Project a desired chain length into the time-embedding space.

    The module computes a sinusoidal embedding of the requested length
    and feeds it through a two-layer MLP whose output dimensionality
    matches ``time_dim``. The result is *added* to the time embedding
    by the caller — this preserves backward compatibility for callers
    that do not pass a length.

    Parameters
    ----------
    time_dim : int
        Output dimensionality (matches the time embedding of the
        backbone diffusion model).
    embed_dim : int, optional
        Internal embedding dimensionality used for the sinusoidal
        encoding. Defaults to ``time_dim``.
    max_period : float, default 10000.0
        Sinusoidal frequency parameter.
    """

    def __init__(self,
                 time_dim: int,
                 embed_dim: Optional[int] = None,
                 max_period: float = 10000.0) -> None:
        super().__init__()
        if embed_dim is None:
            embed_dim = time_dim
        if embed_dim % 2 != 0:
            raise ValueError('embed_dim must be even.')
        self.embed_dim = int(embed_dim)
        self.max_period = float(max_period)
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, length: torch.Tensor) -> torch.Tensor:
        """Return the length-conditioned addend of shape ``(B, time_dim)``."""
        embed = sinusoidal_length_embedding(
            length, self.embed_dim, max_period=self.max_period)
        return self.mlp(embed)


class BinderCrossAttention(nn.Module):
    """Multi-head cross-attention from generated chain to a frozen binder.

    The target (binder) representation is processed through fixed key /
    value projections whose parameters are flagged ``requires_grad=False``
    and therefore *do not receive gradient updates*. The generated
    chain serves as the query. A residual + LayerNorm is applied around
    the attention block.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of both query and target representations.
    num_heads : int, default 4
        Number of attention heads. ``embed_dim`` must be divisible by
        ``num_heads``.
    dropout : float, default 0.0
        Attention dropout probability.

    Notes
    -----
    "Frozen target" here means the projection layers applied to the
    target are non-trainable. Callers may additionally detach the target
    tensor itself prior to passing it in to fully cut off the gradient
    flow.
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 4,
                 dropout: float = 0.0) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError('embed_dim must be divisible by num_heads.')
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.dropout = float(dropout)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        # Freeze the key and value projections — they act on the binder
        # which is treated as a fixed conditioning signal.
        for param in self.k_proj.parameters():
            param.requires_grad = False
        for param in self.v_proj.parameters():
            param.requires_grad = False
        # Zero-initialise the output projection so the module is the
        # identity at initialisation (drop-in compatible with the
        # existing diffusion transformer stack).
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self,
                query: torch.Tensor,
                target: torch.Tensor,
                target_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply cross-attention from ``query`` to ``target``.

        Parameters
        ----------
        query : torch.Tensor
            Generated-chain features of shape ``(B, L_q, embed_dim)``.
        target : torch.Tensor
            Binder features of shape ``(B, L_k, embed_dim)``. Pass
            ``target.detach()`` to additionally block gradient flow
            into the upstream binder encoder.
        target_mask : torch.Tensor, optional
            Boolean mask of shape ``(B, L_k)``; ``True`` for valid
            positions.

        Returns
        -------
        torch.Tensor
            Attended features of shape ``(B, L_q, embed_dim)``.
        """
        if query.ndim != 3 or target.ndim != 3:
            raise ValueError('query and target must be 3-D tensors.')
        if query.shape[-1] != self.embed_dim:
            raise ValueError('query last dim must match embed_dim.')
        batch_size, len_q, _ = query.shape
        len_k = target.shape[1]
        q = self.q_proj(self.norm_q(query))
        k = self.k_proj(self.norm_kv(target))
        v = self.v_proj(self.norm_kv(target))
        # Split heads.
        q = q.view(batch_size, len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, len_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, len_k, self.num_heads, self.head_dim).transpose(1, 2)
        scale = 1.0 / math.sqrt(self.head_dim)
        logits = torch.matmul(q, k.transpose(-1, -2)) * scale
        if target_mask is not None:
            if target_mask.shape != (batch_size, len_k):
                raise ValueError('target_mask must be (B, L_k).')
            additive = torch.where(
                target_mask.unsqueeze(1).unsqueeze(1),
                torch.zeros_like(logits),
                torch.full_like(logits, float('-inf')))
            logits = logits + additive
        attn = torch.softmax(logits, dim=-1)
        if self.training and self.dropout > 0.0:
            attn = F.dropout(attn, p=self.dropout)
        out = torch.matmul(attn, v)  # (B, H, L_q, D)
        out = out.transpose(1, 2).reshape(batch_size, len_q, self.embed_dim)
        return query + self.out_proj(out)

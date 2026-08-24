
#New Changes below

import math
from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    raise ImportError("These classes require PyTorch to be installed")


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention from `Attention Is All You Need`_.

    .. _Attention Is All You Need: https://arxiv.org/abs/1706.03762

    Examples
    --------
    >>> import torch
    >>> import torch.nn as nn
    >>> from deepchem.models.attention import ScaledDotProductAttention
    >>> attn = ScaledDotProductAttention()
    >>> x = torch.ones(1, 5)
    >>> Q = nn.Parameter(torch.ones(5))
    >>> K = nn.Parameter(torch.ones(5))
    >>> V = nn.Parameter(torch.ones(5))
    >>> query, key, value = Q * x, K * x, V * x
    >>> x_out, attn_score = attn(query, key, value)
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _safe_fill_value(tensor: torch.Tensor) -> float:
        """Returns a dtype-safe large negative value for attention masking.

        Parameters
        ----------
        tensor: torch.Tensor
            The scores tensor whose dtype determines the safe minimum.

        Returns
        -------
        float
            Half of the minimum finite value for the tensor's dtype,
            safe for fp16, bf16, and fp32.
        """
        return torch.finfo(tensor.dtype).min / 2

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        dropout: Optional[nn.Dropout] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot-product attention.

        Parameters
        ----------
        query: torch.Tensor
            Query tensor of shape (..., seq_len, d_k).
        key: torch.Tensor
            Key tensor of shape (..., seq_len, d_k).
        value: torch.Tensor
            Value tensor of shape (..., seq_len, d_v).
        mask: torch.Tensor, optional
            Boolean mask of shape (..., seq_len, seq_len).
            Positions where mask == 0 are blocked from attending.
        dropout: nn.Dropout, optional
            Dropout applied to attention weights after softmax.

        Returns
        -------
        output: torch.Tensor
            Context vector of shape (..., seq_len, d_v).
        p_attn: torch.Tensor
            Attention weight matrix of shape (..., seq_len, seq_len).
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, self._safe_fill_value(scores))

        p_attn = scores.softmax(dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class SelfAttention(nn.Module):
    """Self-Attention layer.

    Given :math:`X \\in \\mathbb{R}^{n \\times \\text{in\\_features}}`,
    computes :math:`a = \\text{softmax}(W_2 \\tanh(W_1 X))` and returns
    :math:`y = aX \\in \\mathbb{R}^{\\text{out\\_features} \\times \\text{in\\_features}}`.

    Parameters
    ----------
    in_features: int
        Dimensionality of input token features.
    out_features: int
        Number of attention heads / output feature dimension.
    hidden_size: int, optional (default 128)
        Dimensionality of the intermediate tanh projection.

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.attention import SelfAttention
    >>> layer = SelfAttention(in_features=16, out_features=8, hidden_size=32)
    >>> X = torch.randn(10, 16)
    >>> embedding, attn = layer(X)
    >>> embedding.shape
    torch.Size([8, 16])
    >>> attn.shape
    torch.Size([8, 10])
    """

    def __init__(self, in_features: int, out_features: int, hidden_size: int = 128):
        super().__init__()
        self.w1 = nn.Parameter(torch.FloatTensor(hidden_size, in_features))
        self.w2 = nn.Parameter(torch.FloatTensor(out_features, hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.w1)
        nn.init.xavier_normal_(self.w2)

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute self-attention weighted embedding.

        Parameters
        ----------
        X: torch.Tensor
            Input features of shape :math:`(n, \\text{in\\_features})`.

        Returns
        -------
        embedding: torch.Tensor
            Attended output of shape :math:`(\\text{out\\_features}, \\text{in\\_features})`.
        attn: torch.Tensor
            Attention matrix of shape :math:`(\\text{out\\_features}, n)`.
        """
        x = torch.tanh(F.linear(X, self.w1))
        x = F.linear(x, self.w2)
        attn = F.softmax(x.transpose(0, 1), dim=-1)
        embedding = torch.matmul(attn, X)
        return embedding, attn
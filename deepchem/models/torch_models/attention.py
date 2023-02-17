import math
from typing import Optional
try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    raise ImportError("These classes require PyTorch to be installed")


class ScaledDotProductAttention(nn.Module):
    """The Scaled Dot Production Attention operation from `Attention Is All You Need <https://arxiv.org/abs/1706.03762>_` paper.

    Example
    -------
    >>> from deepchem.models import ScaledDotProductAttention as SDPA
    >>> attn = SDPA()
    >>> x = torch.ones(1, 5)
    >>> # Linear layers for making query, key, value
    >>> Q, K, V = nn.Parameter(torch.ones(5)), nn.Parameter(torch.ones(5)), nn.Parameter(torch.ones(5))
    >>> query, key, value = Q * x, K * x, V * x
    >>> x_out, attn_score = attn(query, key, value)
    """

    def __init__(self):
        self.epsilon = -1e9
        super(ScaledDotProductAttention, self).__init__()

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                dropout: Optional[nn.Dropout] = None):
        """
        Parameters
        ----------
        query: torch.Tensor
            Query tensor for attention
        key: torch.Tensor
            Key tensor for attention
        value: torch.Tensor
            Value tensor for attention
        mask: torch.Tensor (optional)
            Mask to apply during attention computation
        dropout: nn.Dropout (optional)
            Dropout layer for attention output
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, self.epsilon)

        p_attn = scores.softmax(dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class SelfAttention(nn.Module):
    """SelfAttention Layer

    Given $X\in \mathbb{R}^{n \times in_feature}$, the attention is calculated by: $a=softmax(W_2tanh(W_1X))$, where
    $W_1 \in \mathbb{R}^{hidden \times in_feature}$, $W_2 \in \mathbb{R}^{out_feature \times hidden}$.
    The final output is $y=aX$ where $y \in \mathbb{R}^{n \times out_feature}$.

    Parameters
    ----------
    in_features: int
        Dimension of input features
    out_features: int
        Dimension of output features
    hidden_size: int
        Dimension of hidden layer
    """

    def __init__(self, in_features, out_features, hidden_size=128):
        super(SelfAttention, self).__init__()
        self.w1 = torch.nn.Parameter(torch.FloatTensor(hidden_size,
                                                       in_features))
        self.w2 = torch.nn.Parameter(
            torch.FloatTensor(out_features, hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.w1)
        nn.init.xavier_normal_(self.w2)

    def forward(self, X):
        """The forward function.

        Parameters
        ----------
        X: torch.Tensor
            input feature of shape $\mathbb{R}^{n \times in_feature}$.

        Returns
        -------
        embedding: torch.Tensor
            The final embedding of shape $\mathbb{R}^{out_features \times in_feature}$
        attention-matrix: torch.Tensor
            The attention matrix
        """
        x = torch.tanh(torch.matmul(self.w1, X.transpose(1, 0)))
        x = torch.matmul(self.w2, x)
        attn = torch.nn.functional.softmax(x, dim=-1)
        x = torch.matmul(attn, X)
        return x, attn

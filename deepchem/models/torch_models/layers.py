import math
import copy
try:
  import torch
  import torch.nn as nn
except:
  raise ImportError('These classes require Torch to be installed.')


def clones(module, N):
  """Produce N identical layers.

  This function returns a stack of Modules replicated N times. 
  
  Parameters
  ----------
  N: int
    Number of identical layers to be returned.
  module: nn.Module
    The module where the N identical layers are to be added.
  
  Returns
  -------
  Torch module with N identical layers.
  """

  return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class ScaleNorm(nn.Module):
  """Apply Scale Normalization to input.

  The ScaleNorm layer first computes the square root of the scale, then computes the matrix/vector norm of the input tensor.
  The norm value is calculated as `sqrt(scale) / matrix norm`.
  Finally, the result is returned as `input_tensor * norm value`.

  This layer can be used instead of LayerNorm when a scaled version of the norm is required.
  Instead of performing the scaling operation (`scale / norm`) in a lambda-like layer, we are defining it within this layer to make prototyping more efficient.

  References
  ----------
  .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264
  
  Examples
  --------
  >>> import deepchem as dc
  >>> scale = 0.35
  >>> layer = dc.models.torch_models.ScaleNorm(scale)
  >>> input_tensor = torch.tensor([[1.269, 39.36], [0.00918, -9.12]])
  >>> output_tensor = layer.forward(input_tensor)
  """

  def __init__(self, scale, eps=1e-5):
    """Initialize a ScaleNorm layer.

    Parameters
    ----------
    scale: Real number or single element tensor
      Scale magnitude.
    eps: float
      Epsilon value.
    """

    super(ScaleNorm, self).__init__()
    self.scale = nn.Parameter(torch.tensor(math.sqrt(scale)))
    self.eps = eps

  def forward(self, x):
    norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
    return x * norm


class Encoder(nn.Module):
  """Encoder block for the Molecule Attention Transformer in [1]_.
  
  A stack of N layers which form the encoder block.
  
  References
  ----------
  .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264
  """

  def __init__(self, self_attn_layer, feed_fwd_layer, d_model, dropout, N):
    """Initialize an Encoder block.

    Parameters
    ----------
    self_attn_layer: dc.torch_models.layers or nn.Module
      Self-Attention layer to be used in the encoder block.
    feed_fwd_layer: dc.torch_models.layers or nn.Module
      Feed-Forward layer to be used in the encoder block.
    d_model: int
      Size of model
    dropout: float
      Probability of Dropout
    N: int
      Number of identical layers to be stacked.
    """

    super(Encoder, self).__init__()
    layer = EncoderLayer(
        self_attn_layer=self_attn_layer,
        feed_fwd_layer=feed_fwd_layer,
        d_model=d_model,
        dropout=dropout)
    self.layers = clones(layer, N)
    self.norm = nn.LayerNorm(layer.size)

  def forward(self, x, mask, **kwargs):
    for layer in self.layers:
      x = layer(x, mask, **kwargs)
    return self.norm(x)


class SublayerConnection(nn.Module):
  """SublayerConnection layer as used in [1]_.
  
  The SublayerConnection layer is a residual layer which is then passed through Layer Normalization.
  
  References
  ----------
  .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264
  """

  def __init__(self, size, dropout):
    """Initialize a SublayerConnection Layer.
    Parameters
    ----------
    size: int
      Size of layer.
    dropout: float
      Dropout probability.
    """

    super(SublayerConnection, self).__init__()
    self.norm = nn.LayerNorm(size)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, sublayer):
    return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
  """Encoder layer for use in the Molecular Attention Transformer.
  
  The Encoder layer is formed by adding self-attention and feed-forward to the encoder block.
  
  References
  ----------
  .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264
  """

  def __init__(self, self_attn_layer, feed_fwd_layer, d_model, dropout):
    """Initialize an Encoder layer.

    Parameters
    ----------
    self_attn_layer: dc.torch_models.layers or nn.Module
      Self-Attention layer to be used in the encoder layer.
    feed_fwd_layer: dc.torch_models.layers or nn.Module
      Feed-Forward layer to be used in the encoder layer.
    d_model: int
      Size of model
    dropout: float
      Probability of Dropout
    """

    super(EncoderLayer, self).__init__()
    self.self_attn = self_attn_layer
    self.feed_forward = feed_fwd_layer
    self.sublayer = clones(SublayerConnection(size=d_model, dropout=dropout), 2)
    self.size = d_model

  def forward(self, x, mask, **kwargs):
    "Follow Figure 1 (left) for connections."
    x = self.sublayer[0](x,
                         lambda x: self.self_attn(x, x, x, mask=mask, **kwargs))
    return self.sublayer[1](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
  """Multi-Headed Attention layer for the Molecular Attention Transformer [1]_

  References
  ----------
  .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264
  """
  def __init__(self,
               attention,
               h,
               d_model,
               dropout,
               output_bias = True):
    super().__init__()
    assert d_model % h == 0

    self.d_k = d_model // h
    self.h = h

    self.linear_layers = clones(nn.Linear(d_model, d_model), 3)
    self.dropout = nn.Dropout(dropout)
    self.output_linear = nn.Linear(d_model, d_model, output_bias)
    self.attention = attention

  def forward(self,
              query,
              key,
              value,
              mask = None,
              **kwargs):
    if mask is not None:
      mask = mask.unsqueeze(1)

    batch_size = query.size(0)

    query, key, value = [
        layer(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        for layer, x in zip(self.linear_layers, (query, key, value))
    ]

    x, _ = self.attention(
        query, key, value, mask=mask, dropout=self.dropout, **kwargs)
    x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

    return self.output_linear(x)


class PositionwiseFeedForward(nn.Module):
  """Implements FFN equation."""

  def __init__(self,
               *,
               d_input,
               d_hidden = None,
               d_output = None,
               activation,
               n_layers,
               dropout):
    super(PositionwiseFeedForward, self).__init__()
    self.n_layers = n_layers
    d_output = d_output if d_output is not None else d_input
    d_hidden = d_hidden if d_hidden is not None else d_input
    if n_layers == 1:
      self.linears = [nn.Linear(d_input, d_output)]
    else:
      self.linears = [nn.Linear(d_input, d_hidden)] + \
                      [nn.Linear(d_hidden, d_hidden) for _ in range(n_layers - 2)] + \
                      [nn.Linear(d_hidden, d_output)]

    self.linears = nn.ModuleList(self.linears)
    self.dropout = clones(nn.Dropout(dropout), n_layers)
    self.act_func = activation

  def forward(self, x):

    if self.n_layers == 0:
      return x

    elif self.n_layers == 1:
      return self.dropout[0](self.act_func(self.linears[0](x)))

    else:
      for i in range(self.n_layers - 1):
        x = self.dropout[i](self.act_func(self.linears[i](x)))
      return self.linears[-1](x)

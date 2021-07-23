import math
import copy
import numpy as np
try:
  import torch
  import torch.nn as nn
  import torch.nn.functional as F
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

  Examples
  --------
  >>> import deepchem as dc
  >>> d_model = 1024
  >>> cloned_layer = dc.models.torch_models.layers.clones(nn.Linear(1024, 1024), 3)
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
  >>> layer = dc.models.torch_models.layers.ScaleNorm(scale)
  >>> input_tensor = torch.Tensor([[1.269, 39.36], [0.00918, -9.12]])
  >>> output_tensor = layer(input_tensor)
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
  
  A stack of N layers which form the encoder block. The block primarily consists of a self-attention layer and a feed-forward layer.
  This block is constructed from its basic layer: EncoderLayer. See dc.models.torch_models.layers.EncoderLayer for more details regarding the working of the block.

  References
  ----------
  .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264
  
  Examples
  --------
  >>> import deepchem as dc
  >>> attention = dc.models.torch_models.layers.MATAttention('softmax', 0.33, 0.33)
  >>> self_attn_layer = dc.models.torch_models.layers.MultiHeadedAttention(8, 1024, 0.1, attention)
  >>> feed_fwd_layer = dc.models.torch_models.layers.PositionwiseFeedForward(d_input = 1024, activation = torch.nn.ReLU(), n_layers = 1, dropout = 0.1)
  >>> block = dc.models.torch_models.layers.Encoder(self_attn_layer = self_attn_layer, feed_fwd_layer = feed_fwd_layer, d_model = 1024, dropout = 0.0, N = 3)
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
      Size of dense layer.
    dropout: float
      Dropout probability.
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
    """Output computation for the Encoder block.

    Parameters
    ----------
    x: torch.Tensor
      Input tensor.
    mask: torch.Tensor
      Mask for padding.
    """

    for layer in self.layers:
      x = layer(x, mask, **kwargs)
    return self.norm(x)


class SublayerConnection(nn.Module):
  """SublayerConnection layer which establishes a residual connection, as used in the Molecular Attention Transformer [1]_.
  
  The SublayerConnection layer is a residual layer which is then passed through Layer Normalization.
  The residual connection is established by computing the dropout-adjusted layer output of a normalized input tensor and adding this to the originial input tensor. 
  
  References
  ----------
  .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264
  
  Examples
  --------
  >>> import deepchem as dc
  >>> scale = 0.35
  >>> layer = dc.models.torch_models.layers.SublayerConnection(2, 0.)
  >>> output = layer(torch.Tensor([1.,2.]), nn.Linear(2,1))
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
    """Output computation for the SublayerConnection layer.
    
    Takes an input tensor x, then adds the dropout-adjusted sublayer output for normalized x to it.

    Parameters
    ----------
    x: torch.Tensor
      Input tensor.
    sublayer: nn.Module
      Layer whose output for normalized x will be added to x.
    """
    return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
  """Encoder layer for use in the Molecular Attention Transformer [1]_.
  
  The Encoder layer is formed by adding self-attention and feed-forward to the encoder block. 
  It is the basis of the Encoder block.
  
  References
  ----------
  .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264
  
  Examples
  --------
  >>> import deepchem as dc
  >>> attention = dc.models.torch_models.layers.MATAttention('softmax', 0.33, 0.33)
  >>> self_attn_layer = dc.models.torch_models.layers.MultiHeadedAttention(h = 8, d_model = 1024, dropout = 0.1, attention = attention)
  >>> feed_fwd_layer = dc.models.torch_models.layers.PositionwiseFeedForward(d_input = 1024, activation = torch.nn.ReLU(), n_layers = 1, dropout = 0.1)
  >>> layer = dc.models.torch_models.layers.EncoderLayer(self_attn_layer = self_attn_layer, feed_fwd_layer = feed_fwd_layer, d_model = 1024, dropout = 0.1, N = 3)
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
      Size of dense layer.
    dropout: float
      Dropout probability.
    """

    super(EncoderLayer, self).__init__()
    self.self_attn = self_attn_layer
    self.feed_forward = feed_fwd_layer
    self.sublayer = clones(SublayerConnection(size=d_model, dropout=dropout), 2)
    self.size = d_model

  def forward(self, x, mask, **kwargs):
    """Output computation for the Encoder layer.

    Parameters
    ----------
    query: torch.Tensor
      Standard query parameter for attention.
    key: torch.Tensor
      Standard key parameter for attention.
    value: torch.Tensor
      Standard value parameter for attention.
    mask: torch.Tensor
      Masks out padding values so that they are not taken into account when computing the attention score.
    """
    x = self.sublayer[0](x,
                         lambda x: self.self_attn(x, x, x, mask=mask, **kwargs))
    return self.sublayer[1](x, self.feed_forward)


class MATAttention(nn.Module):
  """Primary molecular self-attention layer for the Molecular Attention Transformer [1]_.
  
  This layer also computes attention score with the given query, mask and value along with the mask so that the score is not influenced by padding.
  The score is first calculated as the result of multiplying the query and the transpose of the key, then divided by the root of the size of the transpose of the key. 
  The result is then passed through softmax activation. This forms our p_attn.

  The weights tensor is calculated by computing the summation of the products of lambda attention with p_attn, lambda_distance with the distance matrix, and lambda adjacency with the adjacency matrix, and finally adjusting it with dropout.

  
  References
  ----------
  .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264
  
  Examples
  --------
  >>> import deepchem as dc
  >>> attention = dc.models.torch_models.layers.MATAttention(dist_kernel = 'softmax', lambda_attention = 0.33, lambda_distance = 0.33)
  """

  def __init__(self, dist_kernel, lambda_attention, lambda_distance):
    """Initialize a MATAttention layer.

    Parameters
    ----------
    dist_kernel: str
      Kernel activation to be used. Can be either 'softmax' for softmax or 'exp' for exponential.
    lambda_attention: float
      Constant to be multiplied with the attention matrix.
    lambda_distance: float
      Constant to be multiplied with the distance matrix.
    """

    super().__init__()
    if dist_kernel == "softmax":
      self.dist_kernel = lambda x: torch.softmax(-x, dim=-1)
    elif dist_kernel == "exp":
      self.dist_kernel = lambda x: torch.exp(-x)

    self.lambda_attention = lambda_attention
    self.lambda_distance = lambda_distance
    self.lambda_adjacency = 1.0 - self.lambda_attention - self.lambda_distance

  def forward(self,
              query,
              key,
              value,
              mask,
              dropout,
              adj_matrix,
              distance_matrix,
              eps=1e-6,
              inf=1e12):
    """Output computation for the MATAttention layer.

    Parameters
    ----------
    query: torch.Tensor
      Standard query parameter for attention.
    key: torch.Tensor
      Standard key parameter for attention.
    value: torch.Tensor
      Standard value parameter for attention.
    mask: torch.Tensor
      Masks out padding values so that they are not taken into account when computing the attention score.
    dropout: float
      Dropout probability.
    adj_matrix: np.ndarray
      Adjacency matrix of the input molecule, returned from dc.feat.MATFeaturizer()
    dist_matrix: np.ndarray
      Distance matrix of the input molecule, returned from dc.feat.MATFeaturizer()
    eps: float
      Epsilon value
    inf: float
      Value of infinity to be used.
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
      scores = scores.masked_fill(
          mask.unsqueeze(1).repeat(1, query.shape[1], query.shape[2], 1) == 0,
          -inf)
    p_attn = F.softmax(scores, dim=-1)

    adj_matrix = adj_matrix / (adj_matrix.sum(dim=-1).unsqueeze(2) + eps)
    p_adj = adj_matrix.unsqueeze(1).repeat(1, query.shape[1], 1, 1)

    distance_matrix = distance_matrix.masked_fill(
        mask.repeat(1, mask.shape[-1], 1) == 0, np.inf)
    distance_matrix = self.dist_kernel(distance_matrix)
    p_dist = distance_matrix.unsqueeze(1).repeat(1, query.shape[1], 1, 1)

    p_weighted = self.lambda_attention * p_attn + self.lambda_distance * p_dist + self.lambda_adjacency * p_adj
    p_weighted = dropout(p_weighted)

    return torch.matmul(p_weighted, value), p_attn


class MultiHeadedAttention(nn.Module):
  """Converts an existing attention layer to a multi-headed attention module.

  Multi-Headed attention the attention mechanism multiple times parallely through the multiple attention heads.
  Thus, different subsequences of a given sequences can be processed differently.
  The query, key and value parameters are split multiple ways and each split is passed separately through a different attention head.

  References
  ----------
  .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264
  
  Examples
  --------
  >>> import deepchem as dc
  >>> attention = dc.models.torch_models.layers.MATAttention('softmax', 0.33, 0.33')
  >>> self_attn_layer = dc.models.torch_models.layers.MultiHeadedAttention(h = 8, d_model = 1024, dropout = 0.1, attention = attention)
  """

  def __init__(self, attention, h, d_model, dropout, output_bias=True):
    """Initialize a multi-headed attention layer.

    Parameters
    ----------
    attention: nn.Module
      Module to be used as the attention layer.
    h: int
      Number of attention heads.
    d_model: int
      Size of dense layer.
    dropout: float
      Dropout probability.
    output_bias: bool
      If True, dense layers will use bias vectors.
    """

    super().__init__()

    self.d_k = d_model // h
    self.h = h

    self.linear_layers = clones(nn.Linear(d_model, d_model), 3)
    self.dropout = nn.Dropout(dropout)
    self.output_linear = nn.Linear(d_model, d_model, output_bias)
    self.attention = attention

  def forward(self, query, key, value, mask=None, **kwargs):
    """Output computation for MultiHeadedAttention layer.

    Parameters
    ----------
    query: torch.Tensor
      Standard query parameter for attention.
    key: torch.Tensor
      Standard key parameter for attention.
    value: torch.Tensor
      Standard value parameter for attention.
    mask: torch.Tensor
      Masks out padding values so that they are not taken into account when computing the attention score.
    """

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
  """PositionwiseFeedForward is a layer used to define the position-wise feed-forward (FFN) algorithm for the Molecular Attention Transformer [1]_
  
  Each layer in the encoder contains a fully connected feed-forward network which applies two linear transformations and the given activation function.
  This is done in addition to the SublayerConnection module.

  References
  ----------
  .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264
  
  Examples
  --------
  >>> import deepchem as dc
  >>> feed_fwd_layer = dc.models.torch_models.layers.PositionwiseFeedForward(d_input = 1024, activation = torch.nn.ReLU(), n_layers = 1, dropout = 0.1)
  """

  def __init__(self,
               *,
               d_input,
               d_hidden=None,
               d_output=None,
               activation,
               n_layers,
               dropout):
    """Initialize a PositionwiseFeedForward layer.

    Parameters
    ----------
    d_input: int
      Size of input layer.
    d_hidden: int
      Size of hidden layer.
    d_output: int
      Size of output layer.
    activation: torch.nn.modules.activation
      Activation function to be used.
    n_layers: int
      Number of layers.
    dropout: float
      Dropout probability.
    """

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
    """Output Computation for the PositionwiseFeedForward layer.

    Parameters
    ----------
    x: torch.Tensor
      Input tensor.
    """

    if self.n_layers == 0:
      return x

    elif self.n_layers == 1:
      return self.dropout[0](self.act_func(self.linears[0](x)))

    else:
      for i in range(self.n_layers - 1):
        x = self.dropout[i](self.act_func(self.linears[i](x)))
      return self.linears[-1](x)

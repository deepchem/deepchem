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
    
    def __init__(self, layer, N, scale_norm):
        """Initialize an Encoder block.
        Parameters
        ----------
        layer: dc.layers
          Layer to be stacked in the encoder block.
        N: int
          Number of identical layers to be stacked.
        scale_norm: Bool
          If True, uses ScaleNorm, else uses LayerNorm.
        """

        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        if scale_norm:
            self.norm = ScaleNorm(layer.size)
        else:
            self.norm = torch.LayerNorm(layer.size)
    
    def forward(self, x, mask, adj_matrix, distances_matrix, edges_att):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask, adj_matrix, distances_matrix, edges_att)
        return self.norm(x)

class SublayerConnection(nn.Module):
  """SublayerConnection layer as used in [1]_.
  
  The SublayerConnection layer is a residual layer which is then passed through Layer Normalization.
  
  References
  ----------
  .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264
  """
    
  def __init__(self, size, dropout, scale_norm):
      """Initialize a SublayerConnection Layer.
      Parameters
      ----------
      size: int
        Normalized shape of LayerNorm/ScaleNorm layer.
      dropout: float
        Dropout probability.
      scale_norm: Bool
        If True, uses ScaleNorm, else uses LayerNorm.
      """

      super(SublayerConnection, self).__init__()
      self.norm = ScaleNorm(size) if scale_norm else torch.LayerNorm(size)
      self.dropout = nn.Dropout(dropout)

  def forward(self, x, sublayer):
      "Apply residual connection to any sublayer with the same size."
      return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
  """Encoder layer for use in the Molecular Attention Transformer.
  
  The Encoder layer is formed by adding self-attention and feed-forward to the encoder block.
  
  References
  ----------
  .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264
  """

  def __init__(self, size, self_attn, feed_forward, dropout, scale_norm):
      """Initialize an Encoder layer.

      Parameters
      ----------
      size: int
        Normalized shape/length of ScaleNorm/LayerNorm layer.
      self_attn: Tensor
        The p_attn attribute from the attention function for the Molecular Attention Transformer.
      feed_forward: Bool
        If True, uses ScaleNorm, else uses LayerNorm.
      dropout:
        asd
      scale_norm: Bool
        If True, uses ScaleNorm, else uses LayerNorm.
      """

      super(EncoderLayer, self).__init__()
      self.self_attn = self_attn
      self.feed_forward = feed_forward
      self.sublayer = clones(SublayerConnection(size, dropout, scale_norm), 2)
      self.size = size

  def forward(self, x, mask, adj_matrix, distances_matrix, edges_att):
      "Follow Figure 1 (left) for connections."
      x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, adj_matrix, distances_matrix, edges_att, mask))
      return self.sublayer[1](x, self.feed_forward)

class EdgeFeaturesLayer(nn.Module):
  def __init__(self, d_model, d_edge, h, dropout):
      super(EdgeFeaturesLayer, self).__init__()
      assert d_model % h == 0
      d_k = d_model // h
      self.linear = nn.Linear(d_edge, 1, bias=False)
      with torch.no_grad():
          self.linear.weight.fill_(0.25)

  def forward(self, x):
      p_edge = x.permute(0, 2, 3, 1)
      p_edge = self.linear(p_edge).permute(0, 3, 1, 2)
      return torch.relu(p_edge)
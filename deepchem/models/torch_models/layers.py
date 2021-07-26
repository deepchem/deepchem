import math
import numpy as np
from typing import Any, Tuple
try:
  import torch
  import torch.nn as nn
  import torch.nn.functional as F
except:
  raise ImportError('These classes require Torch to be installed.')


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
  >>> from deepchem.models.torch_models.layers import ScaleNorm
  >>> scale = 0.35
  >>> layer = dc.models.torch_models.layers.ScaleNorm(scale)
  >>> input_tensor = torch.Tensor([[1.269, 39.36], [0.00918, -9.12]])
  >>> output_tensor = layer(input_tensor)
  """

  def __init__(self, scale: float, eps: float = 1e-5):
    """Initialize a ScaleNorm layer.

    Parameters
    ----------
    scale: float
      Scale magnitude.
    eps: float
      Epsilon value. Default = 1e-5.
    """
    super(ScaleNorm, self).__init__()
    self.scale = nn.Parameter(torch.tensor(math.sqrt(scale)))
    self.eps = eps

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
    return x * norm


class MATEmbedding(nn.Module):
  """Embedding layer to create embedding for inputs.

  In an embedding layer, input is taken and converted to a vector representation for each input.
  In the MATEmbedding layer, an input tensor is processed through a dropout-adjusted linear layer and the resultant vector is returned.

  References
  ----------
  .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264

  Examples
  --------
  >>> import deepchem as dc
  >>> layer = MATEmbedding(d_input = 1024, d_output = 1024, dropout_p = 0.2)
  """

  def __init__(self, *, d_input: int, d_output: int, dropout_p: float):
    """Initialize a MATEmbedding layer.

    Parameters
    ----------
    d_input: int
      Size of input layer.
    d_output: int
      Size of output layer.
    dropout_p: float
      Dropout probability for layer.
    """

    super(MATEmbedding, self).__init__()
    self.lut = nn.Linear(d_input, d_output)
    self.dropout = nn.Dropout(dropout_p)

  def forward(self, x):
    """Computation for the MATEmbedding layer.

    Parameters
    ----------
    x: torch.Tensor
      Input tensor to be converted into a vector.
    """
    return self.dropout(self.lut(x))


class MATGenerator(nn.Module):
  """MATGenerator defines the linear and softmax generator step for the Molecular Attention Transformer [1]_.

  In the MATGenerator, a Generator is defined which performs the Linear + Softmax generation step.
  Depending on the type of aggregation selected, the attention output layer performs different operations.

  References
  ----------
  .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264

  Examples
  --------
  >>> import deepchem as dc
  >>> layer = MATGenerator(hsize = 1024, aggregation_type = 'mean', d_output = 1, n_layers = 1, attn_hidden = 128, attn_out = 4)
  """

  def __init__(self,
               *,
               hsize,
               aggregation_type,
               d_output,
               n_layers,
               dropout_p,
               attn_hidden=128,
               attn_out=4):
    """Initialize a MATGenerator.

    Parameters
    ----------
    hsize: int
      Size of input layer.
    aggregation_type: str
      Type of aggregation to be used. Can be 'grover', 'mean' or 'contextual'.
    d_output: int
      Size of output layer.
    dropout_p: float
      Dropout probability for layer.
    attn_hidden: int
      Size of hidden attention layer.
    attn_out: int
      Size of output attention layer.
    """

    super(MATGenerator, self).__init__()

    if aggregation_type == 'grover':
      self.att_net = nn.Sequential(
          nn.Linear(hsize, attn_hidden, bias=False),
          nn.Tanh(),
          nn.Linear(attn_hidden, attn_out, bias=False),
      )
      hsize *= attn_out

    if n_layers == 1:
      self.proj = nn.Linear(hsize, d_output)

    else:
      self.proj = []
      for i in range(n_layers - 1):
        self.proj.append(nn.Linear(hsize, attn_hidden))
        self.proj.append(nn.LeakyReLU(negative_slope=0.1))
        self.proj.append(nn.LayerNorm(attn_hidden))
        self.proj.append(nn.Dropout(dropout_p))
      self.proj.append(nn.Linear(attn_hidden, d_output))
      self.proj = torch.nn.Sequential(*self.proj)
    self.aggregation_type = aggregation_type

  def forward(self, x, mask):
    """Computation for the MATGenerator layer.

    Parameters
    ----------
    x: torch.Tensor
      Input tensor.
    mask: torch.Tensor
      Mask for padding so that padded values do not get included in attention score calculation.
    """

    mask = mask.unsqueeze(-1).float()
    out_masked = x * mask
    if self.aggregation_type == 'mean':
      out_sum = out_masked.sum(dim=1)
      mask_sum = mask.sum(dim=(1))
      out_avg_pooling = out_sum / mask_sum
    elif self.aggregation_type == 'grover':
      out_attn = self.att_net(out_masked)
      out_attn = out_attn.masked_fill(mask == 0, -1e9)
      out_attn = F.softmax(out_attn, dim=1)
      out_avg_pooling = torch.matmul(
          torch.transpose(out_attn, -1, -2), out_masked)
      out_avg_pooling = out_avg_pooling.view(out_avg_pooling.size(0), -1)
    elif self.aggregation_type == 'contextual':
      out_avg_pooling = x
    projected = self.proj(out_avg_pooling)
    return projected

import math
import numpy as np
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
  >>> import deepchem as dc
  >>> scale = 0.35
  >>> layer = dc.models.torch_models.layers.ScaleNorm(scale)
  >>> input_tensor = torch.tensor([[1.269, 39.36], [0.00918, -9.12]])
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

  def forward(self, x: torch.Tensor):
    norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
    return x * norm


class MultiHeadedMATAttention(nn.Module):
  """First constructs an attention layer tailored to the Molecular Attention Transformer [1]_ and then converts it into Multi-Headed Attention.

  In Multi-Headed attention the attention mechanism multiple times parallely through the multiple attention heads.
  Thus, different subsequences of a given sequences can be processed differently.
  The query, key and value parameters are split multiple ways and each split is passed separately through a different attention head.
  References
  ----------
  .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264
  Examples
  --------
  >>> import deepchem as dc
  >>> from rdkit import Chem
  >>> mol = Chem.MolFromSmiles("CC")
  >>> adj_matrix = Chem.GetAdjacencyMatrix(mol)
  >>> distance_matrix = Chem.GetDistanceMatrix(mol)
  >>> import deepchem.models.torch_models.layers
  >>> layer = dc.models.torch_models.layers.MultiHeadedMATAttention(dist_kernel='softmax', lambda_attention=0.33, lambda_distance=0.33, h=2, hsize=2, dropout_p=0.0)
  >>> input_tensor = torch.tensor([[1., 2.], [5., 6.]])
  >>> mask = torch.tensor([[1., 1.], [1., 1.]])
  >>> result = layer(input_tensor, input_tensor, input_tensor, mask, 0.0, adj_matrix, distance_matrix)
  """

  def __init__(self,
               dist_kernel: str,
               lambda_attention: float,
               lambda_distance: float,
               h: int,
               hsize: int,
               dropout_p: float,
               output_bias: bool = True):
    """Initialize a multi-headed attention layer.
    Parameters
    ----------
    dist_kernel: str
      Kernel activation to be used. Can be either 'softmax' for softmax or 'exp' for exponential.
    lambda_attention: float
      Constant to be multiplied with the attention matrix.
    lambda_distance: float
      Constant to be multiplied with the distance matrix.
    h: int
      Number of attention heads.
    hsize: int
      Size of dense layer.
    dropout_p: float
      Dropout probability.
    output_bias: bool
      If True, dense layers will use bias vectors.
    """
    super().__init__()
    if dist_kernel == "softmax":
      self.dist_kernel = lambda x: torch.softmax(-x, dim=-1)
    elif dist_kernel == "exp":
      self.dist_kernel = lambda x: torch.exp(-x)
    self.lambda_attention = lambda_attention
    self.lambda_distance = lambda_distance
    self.lambda_adjacency = 1.0 - self.lambda_attention - self.lambda_distance
    self.d_k = hsize // h
    self.h = h
    linear_layer = nn.Linear(hsize, hsize)
    self.linear_layers = nn.ModuleList([linear_layer for _ in range(3)])
    self.dropout_p = nn.Dropout(dropout_p)
    self.output_linear = nn.Linear(hsize, hsize, output_bias)

  def _single_attention(self,
                        query: torch.Tensor,
                        key: torch.Tensor,
                        value: torch.Tensor,
                        mask: torch.Tensor,
                        dropout_p: float,
                        adj_matrix: np.ndarray,
                        distance_matrix: np.ndarray,
                        eps: float = 1e-6,
                        inf: float = 1e12):
    """Defining and computing output for a single MAT attention layer.
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
    dropout_p: float
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

    adj_matrix = adj_matrix / (
        torch.sum(torch.tensor(adj_matrix), dim=-1).unsqueeze(1) + eps)
    p_adj = adj_matrix.repeat(1, query.shape[1], 1, 1)

    distance_matrix = torch.tensor(distance_matrix).masked_fill(
        mask.repeat(1, mask.shape[-1], 1) == 0, np.inf)
    distance_matrix = self.dist_kernel(distance_matrix)
    p_dist = distance_matrix.unsqueeze(1).repeat(1, query.shape[1], 1, 1)
    p_weighted = self.lambda_attention * p_attn + self.lambda_distance * p_dist + self.lambda_adjacency * p_adj
    p_weighted = self.dropout_p(p_weighted)

    bd = value.broadcast_to(p_weighted.shape)
    return torch.matmul(p_weighted.float(), bd.float()), p_attn

  def forward(self,
              query: torch.Tensor,
              key: torch.Tensor,
              value: torch.Tensor,
              mask: torch.Tensor,
              dropout_p: float,
              adj_matrix: np.ndarray,
              distance_matrix: np.ndarray,
              eps: float = 1e-6,
              inf: float = 1e12):
    """Output computation for the MultiHeadedAttention layer.
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

    x, _ = self._single_attention(query, key, value, mask, dropout_p,
                                  adj_matrix, distance_matrix, eps, inf)
    x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

    return self.output_linear(x)

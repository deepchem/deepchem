import math
import numpy as np
from typing import Any, Tuple, Optional, Sequence, List, Union

try:
  import torch
  from torch import Tensor
  import torch.nn as nn
  import torch.nn.functional as F
except ModuleNotFoundError:
  raise ImportError('These classes require PyTorch to be installed.')

try:
  from torch_scatter import scatter_mean
except ModuleNotFoundError:
  pass

from deepchem.utils.typing import OneOrMany, ActivationFn
from deepchem.utils.pytorch_utils import get_activation

try:
  from collections.abc import Sequence as SequenceCollection
except:
  from collections import Sequence as SequenceCollection


class CNNModule(nn.Module):
  """A 1, 2, or 3 dimensional convolutional network for either regression or classification.
  The network consists of the following sequence of layers:
  - A configurable number of convolutional layers
  - A global pooling layer (either max pool or average pool)
  - A final fully connected layer to compute the output
  It optionally can compose the model from pre-activation residual blocks, as
  described in https://arxiv.org/abs/1603.05027, rather than a simple stack of
  convolution layers.  This often leads to easier training, especially when using a
  large number of layers.  Note that residual blocks can only be used when
  successive layers have the same output shape.  Wherever the output shape changes, a
  simple convolution layer will be used even if residual=True.
  Examples
  --------
  >>> model = CNNModule(n_tasks=5, n_features=8, dims=2, layer_filters=[3,8,8,16], kernel_size=3, n_classes = 7, mode='classification', uncertainty=False, padding='same')
  >>> x = torch.ones(2, 224, 224, 8)
  >>> x = model(x)
  >>> for tensor in x:
  ...    print(tensor.shape)
  torch.Size([2, 5, 7])
  torch.Size([2, 5, 7])
  """

  def __init__(self,
               n_tasks: int,
               n_features: int,
               dims: int,
               layer_filters: List[int] = [100],
               kernel_size: OneOrMany[int] = 5,
               strides: OneOrMany[int] = 1,
               weight_init_stddevs: OneOrMany[float] = 0.02,
               bias_init_consts: OneOrMany[float] = 1.0,
               dropouts: OneOrMany[float] = 0.5,
               activation_fns: OneOrMany[ActivationFn] = 'relu',
               pool_type: str = 'max',
               mode: str = 'classification',
               n_classes: int = 2,
               uncertainty: bool = False,
               residual: bool = False,
               padding: Union[int, str] = 'valid') -> None:
    """Create a CNN.
    Parameters
    ----------
    n_tasks: int
      number of tasks
    n_features: int
      number of features
    dims: int
      the number of dimensions to apply convolutions over (1, 2, or 3)
    layer_filters: list
      the number of output filters for each convolutional layer in the network.
      The length of this list determines the number of layers.
    kernel_size: int, tuple, or list
      a list giving the shape of the convolutional kernel for each layer.  Each
      element may be either an int (use the same kernel width for every dimension)
      or a tuple (the kernel width along each dimension).  Alternatively this may
      be a single int or tuple instead of a list, in which case the same kernel
      shape is used for every layer.
    strides: int, tuple, or list
      a list giving the stride between applications of the  kernel for each layer.
      Each element may be either an int (use the same stride for every dimension)
      or a tuple (the stride along each dimension).  Alternatively this may be a
      single int or tuple instead of a list, in which case the same stride is
      used for every layer.
    weight_init_stddevs: list or float
      the standard deviation of the distribution to use for weight initialization
      of each layer.  The length of this list should equal len(layer_filters)+1,
      where the final element corresponds to the dense layer.  Alternatively this
      may be a single value instead of a list, in which case the same value is used
      for every layer.
    bias_init_consts: list or float
      the value to initialize the biases in each layer to.  The length of this
      list should equal len(layer_filters)+1, where the final element corresponds
      to the dense layer.  Alternatively this may be a single value instead of a
      list, in which case the same value is used for every layer.
    dropouts: list or float
      the dropout probability to use for each layer.  The length of this list should equal len(layer_filters).
      Alternatively this may be a single value instead of a list, in which case the same value is used for every layer
    activation_fns: str or list
      the torch activation function to apply to each layer. The length of this list should equal
      len(layer_filters).  Alternatively this may be a single value instead of a list, in which case the
      same value is used for every layer, 'relu' by default
    pool_type: str
      the type of pooling layer to use, either 'max' or 'average'
    mode: str
      Either 'classification' or 'regression'
    n_classes: int
      the number of classes to predict (only used in classification mode)
    uncertainty: bool
      if True, include extra outputs and loss terms to enable the uncertainty
      in outputs to be predicted
    residual: bool
      if True, the model will be composed of pre-activation residual blocks instead
      of a simple stack of convolutional layers.
    padding: str, int or tuple
      the padding to use for convolutional layers, either 'valid' or 'same'
    """

    super(CNNModule, self).__init__()

    if dims not in (1, 2, 3):
      raise ValueError('Number of dimensions must be 1, 2 or 3')

    if mode not in ['classification', 'regression']:
      raise ValueError("mode must be either 'classification' or 'regression'")

    self.n_tasks = n_tasks
    self.n_features = n_features
    self.dims = dims
    self.mode = mode
    self.n_classes = n_classes
    self.uncertainty = uncertainty
    self.mode = mode
    self.layer_filters = layer_filters
    self.residual = residual

    n_layers = len(layer_filters)

    # PyTorch layers require input and output channels as parameter
    # if only one layer to make the model creating loop below work, multiply layer_filters wutg 2
    if len(layer_filters) == 1:
      layer_filters = layer_filters * 2

    if not isinstance(kernel_size, SequenceCollection):
      kernel_size = [kernel_size] * n_layers
    if not isinstance(strides, SequenceCollection):
      strides = [strides] * n_layers
    if not isinstance(dropouts, SequenceCollection):
      dropouts = [dropouts] * n_layers
    if isinstance(activation_fns,
                  str) or not isinstance(activation_fns, SequenceCollection):
      activation_fns = [activation_fns] * n_layers
    if not isinstance(weight_init_stddevs, SequenceCollection):
      weight_init_stddevs = [weight_init_stddevs] * n_layers
    if not isinstance(bias_init_consts, SequenceCollection):
      bias_init_consts = [bias_init_consts] * n_layers

    self.activation_fns = [get_activation(f) for f in activation_fns]
    self.dropouts = dropouts

    if uncertainty:

      if mode != 'regression':
        raise ValueError("Uncertainty is only supported in regression mode")

      if any(d == 0.0 for d in dropouts):
        raise ValueError(
            'Dropout must be included in every layer to predict uncertainty')

    # Python tuples use 0 based indexing, dims defines number of dimension for convolutional operation
    ConvLayer = (nn.Conv1d, nn.Conv2d, nn.Conv3d)[self.dims - 1]

    if pool_type == 'average':
      PoolLayer = (F.avg_pool1d, F.avg_pool2d, F.avg_pool3d)[self.dims - 1]
    elif pool_type == 'max':
      PoolLayer = (F.max_pool1d, F.max_pool2d, F.max_pool3d)[self.dims - 1]
    else:
      raise ValueError("pool_type must be either 'average' or 'max'")

    self.PoolLayer = PoolLayer
    self.layers = nn.ModuleList()

    in_shape = n_features

    for out_shape, size, stride, weight_stddev, bias_const in zip(
        layer_filters, kernel_size, strides, weight_init_stddevs,
        bias_init_consts):

      layer = ConvLayer(in_channels=in_shape,
                        out_channels=out_shape,
                        kernel_size=size,
                        stride=stride,
                        padding=padding,
                        dilation=1,
                        groups=1,
                        bias=True)

      nn.init.normal_(layer.weight, 0, weight_stddev)

      # initializing layer bias with nn.init gives mypy typecheck error
      # using the following workaround
      if layer.bias is not None:
        layer.bias = nn.Parameter(torch.full(layer.bias.shape, bias_const))

      self.layers.append(layer)

      in_shape = out_shape

    self.classifier_ffn = nn.LazyLinear(self.n_tasks * self.n_classes)
    self.output_layer = nn.LazyLinear(self.n_tasks)
    self.uncertainty_layer = nn.LazyLinear(self.n_tasks)

  def forward(self, inputs: OneOrMany[torch.Tensor]) -> List[Any]:
    """
    Parameters
    ----------
    x: torch.Tensor
      Input Tensor
    Returns
    -------
    torch.Tensor
      Output as per use case : regression/classification
    """
    if isinstance(inputs, torch.Tensor):
      x, dropout_switch = inputs, None
    else:
      x, dropout_switch = inputs

    x = torch.transpose(x, 1, -1)  # n h w c -> n c h w

    prev_layer = x

    for layer, activation_fn, dropout in zip(self.layers, self.activation_fns,
                                             self.dropouts):
      x = layer(x)

      if dropout > 0. and dropout_switch:
        x = F.dropout(x, dropout)

      # residual blocks can only be used when successive layers have the same output shape
      if self.residual and x.shape[1] == prev_layer.shape[1]:
        x = x + prev_layer

      if activation_fn is not None:
        x = activation_fn(x)

      prev_layer = x

    x = self.PoolLayer(x, kernel_size=x.size()[2:])

    outputs = []
    batch_size = x.shape[0]

    x = torch.reshape(x, (batch_size, -1))

    if self.mode == "classification":

      logits = self.classifier_ffn(x)
      logits = logits.view(batch_size, self.n_tasks, self.n_classes)
      output = F.softmax(logits, dim=2)
      outputs = [output, logits]

    else:
      output = self.output_layer(x)
      output = output.view(batch_size, self.n_tasks)

      if self.uncertainty:
        log_var = self.uncertainty_layer(x)
        log_var = log_var.view(batch_size, self.n_tasks, 1)
        var = torch.exp(log_var)
        outputs = [output, var, output, log_var]
      else:
        outputs = [output]

    return outputs


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
  >>> layer = ScaleNorm(scale)
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

  def forward(self, x: torch.Tensor) -> torch.Tensor:
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
  >>> from deepchem.models.torch_models.layers import MultiHeadedMATAttention, MATEmbedding
  >>> import deepchem as dc
  >>> import torch
  >>> input_smile = "CC"
  >>> feat = dc.feat.MATFeaturizer()
  >>> input_smile = "CC"
  >>> out = feat.featurize(input_smile)
  >>> node = torch.tensor(out[0].node_features).float().unsqueeze(0)
  >>> adj = torch.tensor(out[0].adjacency_matrix).float().unsqueeze(0)
  >>> dist = torch.tensor(out[0].distance_matrix).float().unsqueeze(0)
  >>> mask = torch.sum(torch.abs(node), dim=-1) != 0
  >>> layer = MultiHeadedMATAttention(
  ...    dist_kernel='softmax',
  ...    lambda_attention=0.33,
  ...    lambda_distance=0.33,
  ...    h=16,
  ...    hsize=1024,
  ...    dropout_p=0.0)
  >>> op = MATEmbedding()(node)
  >>> output = layer(op, op, op, mask, adj, dist)
  """

  def __init__(self,
               dist_kernel: str = 'softmax',
               lambda_attention: float = 0.33,
               lambda_distance: float = 0.33,
               h: int = 16,
               hsize: int = 1024,
               dropout_p: float = 0.0,
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
                        adj_matrix: torch.Tensor,
                        distance_matrix: torch.Tensor,
                        dropout_p: float = 0.0,
                        eps: float = 1e-6,
                        inf: float = 1e12) -> Tuple[torch.Tensor, torch.Tensor]:
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
    adj_matrix: torch.Tensor
      Adjacency matrix of the input molecule, returned from dc.feat.MATFeaturizer()
    dist_matrix: torch.Tensor
      Distance matrix of the input molecule, returned from dc.feat.MATFeaturizer()
    dropout_p: float
      Dropout probability.
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
        torch.sum(torch.tensor(adj_matrix), dim=-1).unsqueeze(2) + eps)

    if len(adj_matrix.shape) <= 3:
      p_adj = adj_matrix.unsqueeze(1).repeat(1, query.shape[1], 1, 1)
    else:
      p_adj = adj_matrix.repeat(1, query.shape[1], 1, 1)

    distance_matrix = torch.tensor(distance_matrix).squeeze().masked_fill(
        mask.repeat(1, mask.shape[-1], 1) == 0, np.inf)

    distance_matrix = self.dist_kernel(distance_matrix)

    p_dist = distance_matrix.unsqueeze(1).repeat(1, query.shape[1], 1, 1)

    p_weighted = self.lambda_attention * p_attn + self.lambda_distance * p_dist + self.lambda_adjacency * p_adj
    p_weighted = self.dropout_p(p_weighted)

    return torch.matmul(p_weighted.float(), value.float()), p_attn

  def forward(self,
              query: torch.Tensor,
              key: torch.Tensor,
              value: torch.Tensor,
              mask: torch.Tensor,
              adj_matrix: torch.Tensor,
              distance_matrix: torch.Tensor,
              dropout_p: float = 0.0,
              eps: float = 1e-6,
              inf: float = 1e12) -> torch.Tensor:
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
    adj_matrix: torch.Tensor
      Adjacency matrix of the input molecule, returned from dc.feat.MATFeaturizer()
    dist_matrix: torch.Tensor
      Distance matrix of the input molecule, returned from dc.feat.MATFeaturizer()
    dropout_p: float
      Dropout probability.
    eps: float
      Epsilon value
    inf: float
      Value of infinity to be used.
    """
    if mask is not None and len(mask.shape) <= 2:
      mask = mask.unsqueeze(1)

    batch_size = query.size(0)
    query, key, value = [
        layer(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        for layer, x in zip(self.linear_layers, (query, key, value))
    ]

    x, _ = self._single_attention(query, key, value, mask, adj_matrix,
                                  distance_matrix, dropout_p, eps, inf)
    x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

    return self.output_linear(x)


class MATEncoderLayer(nn.Module):
  """Encoder layer for use in the Molecular Attention Transformer [1]_.

  The MATEncoder layer primarily consists of a self-attention layer (MultiHeadedMATAttention) and a feed-forward layer (PositionwiseFeedForward).
  This layer can be stacked multiple times to form an encoder.

  References
  ----------
  .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264

  Examples
  --------
  >>> from rdkit import Chem
  >>> import torch
  >>> import deepchem
  >>> from deepchem.models.torch_models.layers import MATEmbedding, MATEncoderLayer
  >>> input_smile = "CC"
  >>> feat = deepchem.feat.MATFeaturizer()
  >>> out = feat.featurize(input_smile)
  >>> node = torch.tensor(out[0].node_features).float().unsqueeze(0)
  >>> adj = torch.tensor(out[0].adjacency_matrix).float().unsqueeze(0)
  >>> dist = torch.tensor(out[0].distance_matrix).float().unsqueeze(0)
  >>> mask = torch.sum(torch.abs(node), dim=-1) != 0
  >>> layer = MATEncoderLayer()
  >>> op = MATEmbedding()(node)
  >>> output = layer(op, mask, adj, dist)
  """

  def __init__(self,
               dist_kernel: str = 'softmax',
               lambda_attention: float = 0.33,
               lambda_distance: float = 0.33,
               h: int = 16,
               sa_hsize: int = 1024,
               sa_dropout_p: float = 0.0,
               output_bias: bool = True,
               d_input: int = 1024,
               d_hidden: int = 1024,
               d_output: int = 1024,
               activation: str = 'leakyrelu',
               n_layers: int = 1,
               ff_dropout_p: float = 0.0,
               encoder_hsize: int = 1024,
               encoder_dropout_p: float = 0.0):
    """Initialize a MATEncoder layer.

    Parameters
    ----------
    dist_kernel: str
      Kernel activation to be used. Can be either 'softmax' for softmax or 'exp' for exponential, for the self-attention layer.
    lambda_attention: float
      Constant to be multiplied with the attention matrix in the self-attention layer.
    lambda_distance: float
      Constant to be multiplied with the distance matrix in the self-attention layer.
    h: int
      Number of attention heads for the self-attention layer.
    sa_hsize: int
      Size of dense layer in the self-attention layer.
    sa_dropout_p: float
      Dropout probability for the self-attention layer.
    output_bias: bool
      If True, dense layers will use bias vectors in the self-attention layer.
    d_input: int
      Size of input layer in the feed-forward layer.
    d_hidden: int
      Size of hidden layer in the feed-forward layer.
    d_output: int
      Size of output layer in the feed-forward layer.
    activation: str
      Activation function to be used in the feed-forward layer.
      Can choose between 'relu' for ReLU, 'leakyrelu' for LeakyReLU, 'prelu' for PReLU,
      'tanh' for TanH, 'selu' for SELU, 'elu' for ELU and 'linear' for linear activation.
    n_layers: int
      Number of layers in the feed-forward layer.
    dropout_p: float
      Dropout probability in the feeed-forward layer.
    encoder_hsize: int
      Size of Dense layer for the encoder itself.
    encoder_dropout_p: float
      Dropout probability for connections in the encoder layer.
    """
    super(MATEncoderLayer, self).__init__()
    self.self_attn = MultiHeadedMATAttention(dist_kernel, lambda_attention,
                                             lambda_distance, h, sa_hsize,
                                             sa_dropout_p, output_bias)
    self.feed_forward = PositionwiseFeedForward(d_input, d_hidden, d_output,
                                                activation, n_layers,
                                                ff_dropout_p)
    layer = SublayerConnection(size=encoder_hsize, dropout_p=encoder_dropout_p)
    self.sublayer = nn.ModuleList([layer for _ in range(2)])
    self.size = encoder_hsize

  def forward(self,
              x: torch.Tensor,
              mask: torch.Tensor,
              adj_matrix: torch.Tensor,
              distance_matrix: torch.Tensor,
              sa_dropout_p: float = 0.0) -> torch.Tensor:
    """Output computation for the MATEncoder layer.

    In the MATEncoderLayer intialization, self.sublayer is defined as an nn.ModuleList of 2 layers. We will be passing our computation through these layers sequentially.
    nn.ModuleList is subscriptable and thus we can access it as self.sublayer[0], for example.

    Parameters
    ----------
    x: torch.Tensor
      Input tensor.
    mask: torch.Tensor
      Masks out padding values so that they are not taken into account when computing the attention score.
    adj_matrix: torch.Tensor
      Adjacency matrix of a molecule.
    distance_matrix: torch.Tensor
      Distance matrix of a molecule.
    sa_dropout_p: float
      Dropout probability for the self-attention layer (MultiHeadedMATAttention).
    """
    x = self.sublayer[0](x,
                         self.self_attn(x,
                                        x,
                                        x,
                                        mask=mask,
                                        dropout_p=sa_dropout_p,
                                        adj_matrix=adj_matrix,
                                        distance_matrix=distance_matrix))
    return self.sublayer[1](x, self.feed_forward(x))


class SublayerConnection(nn.Module):
  """SublayerConnection layer which establishes a residual connection, as used in the Molecular Attention Transformer [1]_.

  The SublayerConnection layer is a residual layer which is then passed through Layer Normalization.
  The residual connection is established by computing the dropout-adjusted layer output of a normalized tensor and adding this to the original input tensor.

  References
  ----------
  .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264

  Examples
  --------
  >>> from deepchem.models.torch_models.layers import SublayerConnection
  >>> scale = 0.35
  >>> layer = SublayerConnection(2, 0.)
  >>> input_ar = torch.tensor([[1., 2.], [5., 6.]])
  >>> output = layer(input_ar, input_ar)
  """

  def __init__(self, size: int, dropout_p: float = 0.0):
    """Initialize a SublayerConnection Layer.

    Parameters
    ----------
    size: int
      Size of layer.
    dropout_p: float
      Dropout probability.
    """
    super(SublayerConnection, self).__init__()
    self.norm = nn.LayerNorm(size)
    self.dropout_p = nn.Dropout(dropout_p)

  def forward(self, x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    """Output computation for the SublayerConnection layer.

    Takes an input tensor x, then adds the dropout-adjusted sublayer output for normalized x to it.
    This is done to add a residual connection followed by LayerNorm.

    Parameters
    ----------
    x: torch.Tensor
      Input tensor.
    output: torch.Tensor
      Layer whose normalized output will be added to x.
    """
    if x is None:
      return self.dropout(self.norm(output))
    return x + self.dropout_p(self.norm(output))


class PositionwiseFeedForward(nn.Module):
  """PositionwiseFeedForward is a layer used to define the position-wise feed-forward (FFN) algorithm for the Molecular Attention Transformer [1]_

  Each layer in the MAT encoder contains a fully connected feed-forward network which applies two linear transformations and the given activation function.
  This is done in addition to the SublayerConnection module.

  Note: This modified version of `PositionwiseFeedForward` class contains `dropout_at_input_no_act` condition to facilitate its use in defining
        the feed-forward (FFN) algorithm for the Directed Message Passing Neural Network (D-MPNN) [2]_

  References
  ----------
  .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264
  .. [2] Analyzing Learned Molecular Representations for Property Prediction https://arxiv.org/pdf/1904.01561.pdf

  Examples
  --------
  >>> from deepchem.models.torch_models.layers import PositionwiseFeedForward
  >>> feed_fwd_layer = PositionwiseFeedForward(d_input = 2, d_hidden = 2, d_output = 2, activation = 'relu', n_layers = 1, dropout_p = 0.1)
  >>> input_tensor = torch.tensor([[1., 2.], [5., 6.]])
  >>> output_tensor = feed_fwd_layer(input_tensor)
  """

  def __init__(self,
               d_input: int = 1024,
               d_hidden: int = 1024,
               d_output: int = 1024,
               activation: str = 'leakyrelu',
               n_layers: int = 1,
               dropout_p: float = 0.0,
               dropout_at_input_no_act: bool = False):
    """Initialize a PositionwiseFeedForward layer.

    Parameters
    ----------
    d_input: int
      Size of input layer.
    d_hidden: int (same as d_input if d_output = 0)
      Size of hidden layer.
    d_output: int (same as d_input if d_output = 0)
      Size of output layer.
    activation: str
      Activation function to be used. Can choose between 'relu' for ReLU, 'leakyrelu' for LeakyReLU, 'prelu' for PReLU,
      'tanh' for TanH, 'selu' for SELU, 'elu' for ELU and 'linear' for linear activation.
    n_layers: int
      Number of layers.
    dropout_p: float
      Dropout probability.
    dropout_at_input_no_act: bool
      If true, dropout is applied on the input tensor. For single layer, it is not passed to an activation function.
    """
    super(PositionwiseFeedForward, self).__init__()

    self.dropout_at_input_no_act: bool = dropout_at_input_no_act

    if activation == 'relu':
      self.activation: Any = nn.ReLU()

    elif activation == 'leakyrelu':
      self.activation = nn.LeakyReLU(0.1)

    elif activation == 'prelu':
      self.activation = nn.PReLU()

    elif activation == 'tanh':
      self.activation = nn.Tanh()

    elif activation == 'selu':
      self.activation = nn.SELU()

    elif activation == 'elu':
      self.activation = nn.ELU()

    elif activation == "linear":
      self.activation = lambda x: x

    self.n_layers: int = n_layers
    d_output = d_output if d_output != 0 else d_input
    d_hidden = d_hidden if d_hidden != 0 else d_input

    if n_layers == 1:
      self.linears: Any = [nn.Linear(d_input, d_output)]

    else:
      self.linears = [nn.Linear(d_input, d_hidden)] + \
                      [nn.Linear(d_hidden, d_hidden) for _ in range(n_layers - 2)] + \
                      [nn.Linear(d_hidden, d_output)]

    self.linears = nn.ModuleList(self.linears)
    dropout_layer = nn.Dropout(dropout_p)
    self.dropout_p = nn.ModuleList([dropout_layer for _ in range(n_layers)])

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Output Computation for the PositionwiseFeedForward layer.

    Parameters
    ----------
    x: torch.Tensor
      Input tensor.
    """
    if not self.n_layers:
      return x

    if self.n_layers == 1:
      if self.dropout_at_input_no_act:
        return self.linears[0](self.dropout_p[0](x))
      else:
        return self.dropout_p[0](self.activation(self.linears[0](x)))

    else:
      if self.dropout_at_input_no_act:
        x = self.dropout_p[0](x)
      for i in range(self.n_layers - 1):
        x = self.dropout_p[i](self.activation(self.linears[i](x)))
      return self.linears[-1](x)


class MATEmbedding(nn.Module):
  """Embedding layer to create embedding for inputs.

  In an embedding layer, input is taken and converted to a vector representation for each input.
  In the MATEmbedding layer, an input tensor is processed through a dropout-adjusted linear layer and the resultant vector is returned.

  References
  ----------
  .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264

  Examples
  --------
  >>> from deepchem.models.torch_models.layers import MATEmbedding
  >>> layer = MATEmbedding(d_input = 3, d_output = 3, dropout_p = 0.2)
  >>> input_tensor = torch.tensor([1., 2., 3.])
  >>> output = layer(input_tensor)
  """

  def __init__(self,
               d_input: int = 36,
               d_output: int = 1024,
               dropout_p: float = 0.0):
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
    self.linear_unit = nn.Linear(d_input, d_output)
    self.dropout = nn.Dropout(dropout_p)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Computation for the MATEmbedding layer.

    Parameters
    ----------
    x: torch.Tensor
      Input tensor to be converted into a vector.
    """
    return self.dropout(self.linear_unit(x))


class MATGenerator(nn.Module):
  """MATGenerator defines the linear and softmax generator step for the Molecular Attention Transformer [1]_.

  In the MATGenerator, a Generator is defined which performs the Linear + Softmax generation step.
  Depending on the type of aggregation selected, the attention output layer performs different operations.

  References
  ----------
  .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264

  Examples
  --------
  >>> from deepchem.models.torch_models.layers import MATGenerator
  >>> layer = MATGenerator(hsize = 3, aggregation_type = 'mean', d_output = 1, n_layers = 1, dropout_p = 0.3, attn_hidden = 128, attn_out = 4)
  >>> input_tensor = torch.tensor([1., 2., 3.])
  >>> mask = torch.tensor([1., 1., 1.])
  >>> output = layer(input_tensor, mask)
  """

  def __init__(self,
               hsize: int = 1024,
               aggregation_type: str = 'mean',
               d_output: int = 1,
               n_layers: int = 1,
               dropout_p: float = 0.0,
               attn_hidden: int = 128,
               attn_out: int = 4):
    """Initialize a MATGenerator.

    Parameters
    ----------
    hsize: int
      Size of input layer.
    aggregation_type: str
      Type of aggregation to be used. Can be 'grover', 'mean' or 'contextual'.
    d_output: int
      Size of output layer.
    n_layers: int
      Number of layers in MATGenerator.
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
      self.proj: Any = nn.Linear(hsize, d_output)

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

  def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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
      out_avg_pooling = torch.matmul(torch.transpose(out_attn, -1, -2),
                                     out_masked)
      out_avg_pooling = out_avg_pooling.view(out_avg_pooling.size(0), -1)

    elif self.aggregation_type == 'contextual':
      out_avg_pooling = x

    return self.proj(out_avg_pooling)


class GraphNetwork(torch.nn.Module):
  """Graph Networks

  A Graph Network [1]_ takes a graph as input and returns an updated graph
  as output. The output graph has same structure as input graph but it
  has updated node features, edge features and global state features.

  Parameters
  ----------
  n_node_features: int
    Number of features in a node
  n_edge_features: int
    Number of features in a edge
  n_global_features: int
    Number of global features
  is_undirected: bool, optional (default True)
    Directed or undirected graph
  residual_connection: bool, optional (default True)
    If True, the layer uses a residual connection during training

  Example
  -------
  >>> import torch
  >>> from deepchem.models.torch_models.layers import GraphNetwork as GN
  >>> n_nodes, n_node_features = 5, 10
  >>> n_edges, n_edge_features = 5, 2
  >>> n_global_features = 4
  >>> node_features = torch.randn(n_nodes, n_node_features)
  >>> edge_features = torch.randn(n_edges, n_edge_features)
  >>> edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]).long()
  >>> global_features = torch.randn(1, n_global_features)
  >>> gn = GN(n_node_features=n_node_features, n_edge_features=n_edge_features, n_global_features=n_global_features)
  >>> node_features, edge_features, global_features = gn(node_features, edge_index, edge_features, global_features)

  References
  ----------
  .. [1] Battaglia et al, Relational inductive biases, deep learning, and graph networks. https://arxiv.org/abs/1806.01261 (2018)
  """

  def __init__(self,
               n_node_features: int = 32,
               n_edge_features: int = 32,
               n_global_features: int = 32,
               is_undirected: bool = True,
               residual_connection: bool = True):
    super().__init__()
    self.n_node_features = n_node_features
    self.n_edge_features = n_edge_features
    self.n_global_features = n_global_features
    self.is_undirected = is_undirected
    self.residual_connection = residual_connection

    self.edge_models, self.node_models, self.global_models = torch.nn.ModuleList(
    ), torch.nn.ModuleList(), torch.nn.ModuleList()
    self.edge_models.append(
        nn.Linear(in_features=n_node_features * 2 + n_edge_features +
                  n_global_features,
                  out_features=32))
    self.node_models.append(
        nn.Linear(in_features=n_node_features + n_edge_features +
                  n_global_features,
                  out_features=32))
    self.global_models.append(
        nn.Linear(in_features=n_node_features + n_edge_features +
                  n_global_features,
                  out_features=32))

    # Used for converting edges back to their original shape
    self.edge_dense = nn.Linear(in_features=32, out_features=n_edge_features)
    self.node_dense = nn.Linear(in_features=32, out_features=n_node_features)
    self.global_dense = nn.Linear(in_features=32,
                                  out_features=n_global_features)

  def reset_parameters(self) -> None:
    self.edge_dense.reset_parameters()
    self.node_dense.reset_parameters()
    self.global_dense.reset_parameters()
    for i in range(0, len(self.edge_models)):
      self.edge_models[i].reset_parameters()
    for i in range(0, len(self.node_models)):
      self.node_models[i].reset_parameters()
    for i in range(0, len(self.global_models)):
      self.global_models[i].reset_parameters()

  def _update_edge_features(self, node_features, edge_index, edge_features,
                            global_features, batch):
    src_index, dst_index = edge_index
    out = torch.cat((node_features[src_index], node_features[dst_index],
                     edge_features, global_features[batch]),
                    dim=1)
    assert out.shape[
        1] == self.n_node_features * 2 + self.n_edge_features + self.n_global_features
    for model in self.edge_models:
      out = model(out)
    return self.edge_dense(out)

  def _update_node_features(self, node_features, edge_index, edge_features,
                            global_features, batch):

    src_index, dst_index = edge_index
    # Compute mean edge features for each node by dst_index (each node
    # receives information from edges which have that node as its destination,
    # hence the computation uses dst_index to aggregate information)
    edge_features_mean_by_node = scatter_mean(edge_features, dst_index, dim=0)
    out = torch.cat(
        (node_features, edge_features_mean_by_node, global_features[batch]),
        dim=1)
    for model in self.node_models:
      out = model(out)
    return self.node_dense(out)

  def _update_global_features(self, node_features, edge_features,
                              global_features, node_batch_map, edge_batch_map):
    edge_features_mean = scatter_mean(edge_features, edge_batch_map, dim=0)
    node_features_mean = scatter_mean(node_features, node_batch_map, dim=0)
    out = torch.cat((edge_features_mean, node_features_mean, global_features),
                    dim=1)
    for model in self.global_models:
      out = model(out)
    return self.global_dense(out)

  def forward(self,
              node_features: Tensor,
              edge_index: Tensor,
              edge_features: Tensor,
              global_features: Tensor,
              batch: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
    """Output computation for a GraphNetwork

    Parameters
    ----------
    node_features: torch.Tensor
      Input node features of shape :math:`(|\mathcal{V}|, F_n)`
    edge_index: torch.Tensor
      Edge indexes of shape :math:`(2, |\mathcal{E}|)`
    edge_features: torch.Tensor
      Edge features of the graph, shape: :math:`(|\mathcal{E}|, F_e)`
    global_features: torch.Tensor
      Global features of the graph, shape: :math:`(F_g, 1)` where, :math:`|\mathcal{V}|` and :math:`|\mathcal{E}|` denotes the number of nodes and edges in the graph, :math:`F_n`, :math:`F_e`, :math:`F_g` denotes the number of node features, edge features and global state features respectively.
    batch: torch.LongTensor (optional, default: None)
      A vector that maps each node to its respective graph identifier. The attribute is used only when more than one graph are batched together during a single forward pass.
    """
    if batch is None:
      batch = node_features.new_zeros(node_features.size(0), dtype=torch.int64)

    node_features_copy, edge_features_copy, global_features_copy = node_features, edge_features, global_features
    if self.is_undirected is True:
      # holding bi-directional edges in case of undirected graphs
      edge_index = torch.cat((edge_index, edge_index.flip([0])), dim=1)
      edge_features_len = edge_features.shape[0]
      edge_features = torch.cat((edge_features, edge_features), dim=0)
    edge_batch_map = batch[edge_index[0]]
    edge_features = self._update_edge_features(node_features, edge_index,
                                               edge_features, global_features,
                                               edge_batch_map)
    node_features = self._update_node_features(node_features, edge_index,
                                               edge_features, global_features,
                                               batch)
    global_features = self._update_global_features(node_features, edge_features,
                                                   global_features, batch,
                                                   edge_batch_map)

    if self.is_undirected is True:
      # coonverting edge features to its original shape
      split = torch.split(edge_features, [edge_features_len, edge_features_len])
      edge_features = (split[0] + split[1]) / 2

    if self.residual_connection:
      edge_features += edge_features_copy
      node_features += node_features_copy
      global_features += global_features_copy

    return node_features, edge_features, global_features

  def __repr__(self) -> str:
    return (
        f'{self.__class__.__name__}(n_node_features={self.n_node_features}, n_edge_features={self.n_edge_features}, n_global_features={self.n_global_features}, is_undirected={self.is_undirected}, residual_connection={self.residual_connection})'
    )


class Affine(nn.Module):
  """Class which performs the Affine transformation.

  This transformation is based on the affinity of the base distribution with
  the target distribution. A geometric transformation is applied where
  the parameters performs changes on the scale and shift of a function
  (inputs).

  Normalizing Flow transformations must be bijective in order to compute
  the logarithm of jacobian's determinant. For this reason, transformations
  must perform a forward and inverse pass.

  Example
  --------
  >>> import deepchem as dc
  >>> from deepchem.models.torch_models.layers import Affine
  >>> import torch
  >>> from torch.distributions import MultivariateNormal
  >>> # initialize the transformation layer's parameters
  >>> dim = 2
  >>> samples = 96
  >>> transforms = Affine(dim)
  >>> # forward pass based on a given distribution
  >>> distribution = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
  >>> input = distribution.sample(torch.Size((samples, dim)))
  >>> len(transforms.forward(input))
  2
  >>> # inverse pass based on a distribution
  >>> len(transforms.inverse(input))
  2

  """

  def __init__(self, dim: int) -> None:
    """Create a Affine transform layer.

    Parameters
    ----------
    dim: int
      Value of the Nth dimension of the dataset.

    """

    super().__init__()
    self.dim = dim
    self.scale = nn.Parameter(torch.zeros(self.dim))
    self.shift = nn.Parameter(torch.zeros(self.dim))

  def forward(self, x: Sequence) -> Tuple[torch.Tensor, torch.Tensor]:
    """Performs a transformation between two different distributions. This
    particular transformation represents the following function:
    y = x * exp(a) + b, where a is scale parameter and b performs a shift.
    This class also returns the logarithm of the jacobians determinant
    which is useful when invert a transformation and compute the
    probability of the transformation.

    input shape: (samples, dim)
    output shape: (samples, dim)
    """

    y = torch.exp(self.scale) * x + self.shift
    det_jacobian = torch.exp(self.scale.sum())
    log_det_jacobian = torch.ones(y.shape[0]) * torch.log(det_jacobian)

    return y, log_det_jacobian

  def inverse(self, y: Sequence) -> Tuple[torch.Tensor, torch.Tensor]:
    """Performs a transformation between two different distributions.
    This transformation represents the bacward pass of the function
    mention before. Its mathematical representation is x = (y - b) / exp(a)
    , where "a" is scale parameter and "b" performs a shift. This class
    also returns the logarithm of the jacobians determinant which is
    useful when invert a transformation and compute the probability of
    the transformation.

    input shape: (samples, dim)
    output shape: (samples, dim)
    """

    x = (y - self.shift) / torch.exp(self.scale)
    det_jacobian = 1 / torch.exp(self.scale.sum())
    inverse_log_det_jacobian = torch.ones(x.shape[0]) * torch.log(det_jacobian)

    return x, inverse_log_det_jacobian


class InteratomicL2Distances(nn.Module):
  """Compute (squared) L2 Distances between atoms given neighbors.

  This class is the pytorch implementation of the original InteratomicL2Distances layer implemented in Keras.
  Pairwise distance (L2) is calculated between input atoms, given the number of neighbors to consider, along with the number of descriptors for every atom.

  Examples
  --------
  >>> atoms = 5
  >>> neighbors = 2
  >>> coords = np.random.rand(atoms, 3)
  >>> neighbor_list = np.random.randint(0, atoms, size=(atoms, neighbors))
  >>> layer = InteratomicL2Distances(atoms, neighbors, 3)
  >>> result = np.array(layer([coords, neighbor_list]))
  >>> result.shape
  (5, 2)

  """

  def __init__(self, N_atoms: int, M_nbrs: int, ndim: int, **kwargs):
    """Constructor for this layer.

    Parameters
    ----------
    N_atoms: int
      Number of atoms in the system total.
    M_nbrs: int
      Number of neighbors to consider when computing distances.
    n_dim:  int
      Number of descriptors for each atom.
    """
    super(InteratomicL2Distances, self).__init__(**kwargs)
    self.N_atoms = N_atoms
    self.M_nbrs = M_nbrs
    self.ndim = ndim

  def __repr__(self) -> str:
    return (
        f'{self.__class__.__name__}(N_atoms={self.N_atoms}, M_nbrs={self.M_nbrs}, ndim={self.ndim})'
    )

  def forward(
      self, inputs: List[Union[torch.Tensor,
                               List[Union[int, float]]]]) -> torch.Tensor:
    """Invokes this layer.

    Parameters
    ----------
    inputs: list
      Should be of form `inputs=[coords, nbr_list]` where `coords` is a
      tensor of shape `(None, N, 3)` and `nbr_list` is a list.

    Returns
    -------
    Tensor of shape `(N_atoms, M_nbrs)` with interatomic distances.
    """
    if len(inputs) != 2:
      raise ValueError("InteratomicDistances requires coords,nbr_list")
    coords, nbr_list = (torch.tensor(inputs[0]), torch.tensor(inputs[1]))
    N_atoms, M_nbrs, ndim = self.N_atoms, self.M_nbrs, self.ndim
    # Shape (N_atoms, M_nbrs, ndim)
    nbr_coords = coords[nbr_list.long()]
    # Shape (N_atoms, M_nbrs, ndim)
    tiled_coords = torch.tile(torch.reshape(coords, (N_atoms, 1, ndim)),
                              (1, M_nbrs, 1))
    # Shape (N_atoms, M_nbrs)
    return torch.sum((tiled_coords - nbr_coords)**2, dim=2)

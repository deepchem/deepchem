import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Union, Optional
try:
  from collections.abc import Sequence as SequenceCollection
except:
  from collections import Sequence as SequenceCollection

class Reshape(nn.Module):
    def __init__(self,*args) -> None:
      super(Reshape, self).__init__()
      self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class CNN(nn.Module):
  """A 1, 2, or 3 dimensional Convolutional Neural Network for Convolutional Neural ODE Model
  Encodes input into high dimensional space which serves as input to ODEBlock

  This Encoder Layer consists of the following sequence of layers:

  - A configurable number of convolutional layers
  - A global pooling layer (either max pool or average pool)

  Examples
  --------
  >>> encoder = CNN(layer_filters=[3,16,8,32,16], conv_dim=2, kernel_size=3)
  >>> x = torch.ones(5,3,224,224)
  >>> y = encoder(x)
  >>> y.shape
  torch.Size([5,16])

  """

  def __init__(self,
               n_tasks: int,
               n_features: int,
               conv_dim: int,
               layer_filters: List[int],
               kernel_size: Optional[Union[int, List[int]]] = 5,
               strides: Optional[Union[int, List[int]]] = 1,
               dropouts: Optional[Union[int, List[int]]] = 0.5,
               activation_fns: Optional[Union[nn.Module, List[nn.Module]]] = nn.ReLU,
               pool_type: Optional[str] = 'max',
               mode: Optional[str] = 'classification',
               n_classes: Optional[int] = 2,
               uncertainty : Optional[bool] = False,
               residual : Optional[bool] = False,
               padding: Optional[str] = 'valid') -> None:
    """Create a CNN.
    In addition to the following arguments, this class also accepts
    all the keyword arguments from TensorGraph.
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
    dropouts: list or float
      the dropout probablity to use for each layer.  The length of this list should equal len(layer_filters).
      Alternatively this may be a single value instead of a list, in which case the same value is used for every layer.
    activation_fns: list or object
      the Tensorflow activation function to apply to each layer.  The length of this list should equal
      len(layer_filters).  Alternatively this may be a single value instead of a list, in which case the
      same value is used for every layer.
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
    padding: str
      the type of padding to use for convolutional layers, either 'valid' or 'same'
    """
 

    super(CNN, self).__init__()

    if conv_dim not in (1, 2, 3):
      raise ValueError('Number of dimensions must be 1, 2 or 3')

    if mode not in ['classification', 'regression']:
      raise ValueError("mode must be either 'classification' or 'regression'")

    self.n_tasks = n_tasks
    self.n_features = n_features
    self.conv_dim = conv_dim
    self.mode = mode
    self.n_classes = n_classes
    self.uncertainty = uncertainty
    self.mode = mode
    self.layer_filters = layer_filters
    n_layers = len(layer_filters) - 1
    if not isinstance(kernel_size, list):
      kernel_size = [kernel_size] * n_layers
    if not isinstance(strides, SequenceCollection):
      strides = [strides] * n_layers
    if not isinstance(dropouts, SequenceCollection):
      dropouts = [dropouts] * n_layers
    if not isinstance(activation_fns, SequenceCollection):
      activation_fns = [activation_fns] * n_layers

    #No weight decay penalty,
    #No explicit regularization

    if uncertainty:
      if mode!= 'regression':
        raise ValueError("Uncertainty is only supported in regression mode")
      if any(d == 0.0 for d in dropouts):
        raise ValueError('Dropout must be included in every layer to predict uncertainty')

    ConvLayer = (nn.Conv1d, nn.Conv2d, nn.Conv3d)[self.conv_dim - 1]

    if pool_type == 'average':
      PoolLayer = (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)[self.conv_dim - 1]
    elif pool_type == 'max':
      PoolLayer = (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)[self.conv_dim - 1]
    else:
      raise ValueError("pool_type must be either 'average' or 'max'")

    self.layers = nn.ModuleList()

    #prev filters = in_shape
    for in_shape, out_shape, size, stride, dropout, activation_fn in zip(
        layer_filters, layer_filters[1:], kernel_size, strides, dropouts,
        activation_fns):
      
      self.layers.append(
          ConvLayer(in_channels=in_shape,
                    out_channels=out_shape,
                    kernel_size=size,
                    stride=stride,
                    padding=padding,
                    dilation=1,
                    groups=1,
                    bias=True))
      
      if dropout>0.0:
        self.layers.append(nn.Dropout(dropout))

      # residual cut how ?
      if activation_fn is not None:
        self.layers.append(activation_fn())
              
      self.layers.append(PoolLayer(size))


  def forward(self, x: torch.Tensor):
    """
    Parameters
    ----------
    x: torch.Tensor
      Input Tensor

    Returns
    -------
    torch.Tensor
      The tensor to be fed into ODEBlock,
      the length of this tensor is equal to ODEBlock Input Dimension
    """

    for layer in self.layers:
      x = layer(x)
    
    outputs, output_types = x, None
    x = Reshape(-1)(x)
    if self.mode is "classification":
      logits = nn.Linear(x.shape[0], self.n_tasks * self.n_classes)(x)
      logits = Reshape(self.n_tasks, self.n_classes)(logits)
      output = F.softmax(logits)
      outputs = [output, logits]
      output_types = ['prediction','loss']
    else:
      output = nn.Linear(x.shape[0], self.n_tasks)(x)
      output = Reshape(self.n_tasks, 1)(output)

      if self.uncertainty:
        
        log_var = Reshape(self.n_tasks, 1)(nn.Linear(x.shape[0], self.n_tasks)(x))
        print(f"z = {log_var.shape}")
        var = torch.exp(log_var)
        outputs = [output, var, output, log_var]
        output_types = ['prediction', 'variance', 'loss', 'loss']
      else:
        outputs = [output]
        output_types = ["prediction"]

    return outputs, output_types

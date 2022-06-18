from turtle import forward
import torch
import torch.nn as nn
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
    """Create a CNN

      Parameters
      ----------
      conv_dim: int
        The number of dimensions to apply convolutions over (1, 2 or 3)
      layer_filters: List[int]
        The number of output filters for each convolutional layer in the network.
        the length of this list determines number of layers
      kernel_size: Optional[Union[int, List[int]]]
        The shape of the convolutional kernel for each layer. Each
        element may be either an int (use the same kernel width for every dimension)
        or a tuple (the kernel width along each dimension). Alternatively this may
        be a single int or tuple instead of a list, in which case the same kernel
        shape is used for every layer.
      strides: Optional[Union[int, List[int]]]
        The stride applications of the kernel for each layer. Each
        element may be either an int (use the same stride width for every dimension)
        or a tuple (the kernel width along each dimension). Alternatively this may be
        a single int or tuple instead of a list, in which case the same stride used
        for every layer.
      dropouts: Optional[Union[int, List[int]]]
        Dropout values to be applied on each layer
      activation_fns: Optional[Union[nn.Module, List[nn.Module]]]
        Activation function to be applied on each layer
      pool_type: Optional[str]
        The type of pooling layer to use, either 'max' or 'average'
      padding: Optional[str]
        The type of padding to use for convolutional layers, either 'valid' or 'same'
      """

    super(CNN, self).__init__()

    if conv_dim not in (1, 2, 3):
      raise ValueError('Number of dimensions must be 1, 2 or 3')

    if mode not in ['classification', 'regression']:
      raise ValueError("mode must be either 'classification' or 'regression'")

    if residual and padding.lower() != 'same':
      raise ValueError("Residual blocks can only be used when padding is 'same'")

    self.n_tasks = n_tasks
    self.n_features = n_features
    self.conv_dim = conv_dim
    self.mode = mode
    self.n_classes = n_classes
    self.uncertainty = uncertainty

    n_layers = len(layer_filters)

    if not isinstance(kernel_size, list):
      kernel_size = [kernel_size] * n_layers

    if not isinstance(strides, SequenceCollection):
      strides = [strides] * n_layers

    if not isinstance(dropouts, SequenceCollection):
      dropouts = [dropouts] * n_layers

    if not isinstance(activation_fns, SequenceCollection):
      activation_fns = [activation_fns] * n_layers

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

    for in_shape, out_shape, size, stride, dropout, activation_fn in zip(
        layer_filters, layer_filters[1:], kernel_size, strides, dropouts,
        activation_fns):
      convblock = nn.Sequential()

      convblock.append(
          ConvLayer(in_channels=in_shape,
                    out_channels=out_shape,
                    kernel_size=size,
                    stride=stride,
                    padding=padding,
                    dilation=1,
                    groups=1,
                    bias=True))
      if dropout>0.0:
        convblock.append(nn.Dropout(dropout))

      if mode == 'classification':
        logits = Reshape((n_tasks,1))

      convblock.append(activation_fn())
      convblock.append(PoolLayer(size))

      self.layers.append(convblock)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    out = x

    for i in range(len(self.layers)):
      out = self.layers[i](out)

    batch_size, out_len = out.shape[:2]
    out = out.view(batch_size, out_len)

    return out
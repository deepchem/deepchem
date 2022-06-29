import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from typing import List, Union, Any, Type

try:
  from collections.abc import Sequence as SequenceCollection
except:
  from collections import Sequence as SequenceCollection


class CNN(nn.Module):
  """A 1, 2, or 3 dimensional convolutional network for either regression or classification.
    The network consists of the following sequence of layers:
    - A configurable number of convolutional layers
    - A global pooling layer (either max pool or average pool)
    - A final dense layer to compute the output
    It optionally can compose the model from pre-activation residual blocks, as
    described in https://arxiv.org/abs/1603.05027, rather than a simple stack of
    convolution layers.  This often leads to easier training, especially when using a
    large number of layers.  Note that residual blocks can only be used when
    successive layers have the same output shape.  Wherever the output shape changes, a
    simple convolution layer will be used even if residual=True.
    Examples
    --------
    >>> model = CNN(n_tasks=5, n_features=8, dims=2, layer_filters=[3,8,8,16], kernel_size=3, n_classes = 7, mode='classification', uncertainty=False)
    >>> x = torch.ones(2, 8, 224, 224)
    >>> y = model(x)
    >>> type(y)
    <class 'list'>
    >>> len(y)
    2
    >>> for tensor in y:
    ...    print(tensor.shape)
    torch.Size([2, 5, 7])
    torch.Size([2, 5, 7])
    """

  def __init__(self,
               n_tasks: int,
               n_features: int,
               dims: int,
               layer_filters: List[int] = [100],
               kernel_size: Union[int, List[int]] = 5,
               strides: Union[int, List[int]] = 1,
               dropouts: Union[float, List[float]] = 0.5,
               activation_fns=nn.ReLU,
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
        dropouts: list or float
          the dropout probability to use for each layer.  The length of this list should equal len(layer_filters).
          Alternatively this may be a single value instead of a list, in which case the same value is used for every layer
        activation_fns: list or object
          the torch activation function to apply to each layer. The length of this list should equal
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
        padding: str, int or tuple
          the padding to use for convolutional layers, either 'valid' or 'same'
        """

    super(CNN, self).__init__()

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

    if not isinstance(kernel_size, list):
      kernel_size = [kernel_size] * n_layers
    if not isinstance(strides, SequenceCollection):
      strides = [strides] * n_layers
    if not isinstance(dropouts, SequenceCollection):
      dropouts = [dropouts] * n_layers
    if not isinstance(activation_fns, SequenceCollection):
      activation_fns = [activation_fns] * n_layers

    if uncertainty:

      if mode != 'regression':
        raise ValueError("Uncertainty is only supported in regression mode")

      if any(d == 0.0 for d in dropouts):
        raise ValueError(
            'Dropout must be included in every layer to predict uncertainty')

    # Python tuples use 0 based indexing, dims defines number of dimension for convolutional operation
    ConvLayer = (nn.Conv1d, nn.Conv2d, nn.Conv3d)[self.dims - 1]

    PoolLayer: Type
    if pool_type == 'average':
      PoolLayer = (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)[self.dims - 1]
    elif pool_type == 'max':
      PoolLayer = (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)[self.dims - 1]
    else:
      raise ValueError("pool_type must be either 'average' or 'max'")

    self.layers = nn.ModuleList()

    in_shape = n_features

    for out_shape, size, stride, dropout, activation_fn in zip(
        layer_filters, kernel_size, strides, dropouts, activation_fns):

      self.layers.append(
          ConvLayer(in_channels=in_shape,
                    out_channels=out_shape,
                    kernel_size=size,
                    stride=stride,
                    padding=padding,
                    dilation=1,
                    groups=1,
                    bias=True))

      if dropout > 0.0:
        self.layers.append(nn.Dropout(dropout))

      if activation_fn is not None:
        self.layers.append(activation_fn())

      self.layers.append(PoolLayer(size))
      in_shape = out_shape

  def forward(self, x: torch.Tensor) -> List[Any]:
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
    prev_layer = x

    for layer in self.layers:
      x = layer(x)
      # residual blocks can only be used when successive layers have the same output shape
      if self.residual and layer.in_channels == layer.out_channels:
        x = x + prev_layer
      prev_layer = x

    outputs = []
    batch_size = x.shape[0]

    pattern = ("b c h -> b (c h)", "b c h w -> b (c h w)",
               "b c h w a -> b (c h w a)")[self.dims - 1]
    x = rearrange(x, pattern)

    if self.mode == "classification":

      logits = nn.Linear(x.shape[1], self.n_tasks * self.n_classes)(x)
      logits = logits.view(batch_size, self.n_tasks, self.n_classes)
      output = F.softmax(logits, dim=1)
      outputs = [output, logits]

    else:
      output = nn.Linear(x.shape[1], self.n_tasks)(x)
      output = output.view(batch_size, self.n_tasks)

      if self.uncertainty:
        log_var = (nn.Linear(x.shape[1], self.n_tasks)(x))
        log_var = log_var.view(batch_size, self.n_tasks, 1)
        var = torch.exp(log_var)
        outputs = [output, var, output, log_var]

      else:
        outputs = [output]

    return outputs

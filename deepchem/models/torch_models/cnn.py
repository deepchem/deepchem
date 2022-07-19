import torch
import torch.nn as nn
import torch.nn.functional as F
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.losses import L2Loss
from deepchem.metrics import to_one_hot

from typing import List, Union, Any, Callable, Optional
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
  >>> model = CNNModule(n_tasks=5, n_features=8, dims=2, layer_filters=[3,8,8,16], kernel_size=3, n_classes = 7, mode='classification', uncertainty=False)
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

  bias: torch.Tensor

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
    bias_init_consts: list or loat
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

    for out_shape, size, stride, weight_stddev, bias_const, dropout in zip(
        layer_filters, kernel_size, strides, weight_init_stddevs,
        bias_init_consts, dropouts):

      block = nn.Sequential()

      layer = ConvLayer(in_channels=in_shape,
                        out_channels=out_shape,
                        kernel_size=size,
                        stride=stride,
                        padding=padding,
                        dilation=1,
                        groups=1,
                        bias=True)

      nn.init.normal_(layer.weight, 0, weight_stddev)

      nn.init.constant_(layer.bias, bias_const)

      block.append(layer)

      if dropout > 0.0:
        block.append(nn.Dropout(dropout))

      self.layers.append(block)

      in_shape = out_shape

    self.classifier_ffn = nn.LazyLinear(self.n_tasks * self.n_classes)
    self.regressor_ffn1 = nn.LazyLinear(self.n_tasks)
    self.regressor_ffn2 = nn.LazyLinear(self.n_tasks)

  def forward(self, x: torch.Tensor) -> List[Any]:
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
    prev_layer = x

    for block, activation_fn in zip(self.layers, self.activation_fns):
      x = block(x)
      # residual blocks can only be used when successive layers have the same output shape
      if self.residual and x.shape[1] == prev_layer.shape[1]:
        x = x + prev_layer

      x = activation_fn(x)

      prev_layer = x

    x = self.PoolLayer(x, kernel_size=x.size()[2:])

    outputs = []
    batch_size = x.shape[0]

    x = torch.reshape(x, (batch_size, -1))

    if self.mode == "classification":

      logits = self.classifier_ffn(x)
      logits = logits.view(batch_size, self.n_tasks, self.n_classes)
      output = F.softmax(logits, dim=1)
      outputs = [output, logits]

    else:
      output = self.regressor_ffn1(x)
      output = output.view(batch_size, self.n_tasks)

      if self.uncertainty:
        log_var = self.regressor_ffn2(x)
        log_var = log_var.view(batch_size, self.n_tasks, 1)
        var = torch.exp(log_var)
        outputs = [output, var, output, log_var]

      else:
        outputs = [output]

    return outputs


class CNN(TorchModel):

  def __init__(self,
               n_tasks: int,
               n_features: int,
               dims: int,
               layer_filters: List[int] = [100],
               kernel_size: OneOrMany[int] = 5,
               strides: OneOrMany[int] = 1,
               weight_init_stddevs: OneOrMany[float] = 0.02,
               bias_init_consts: OneOrMany[float] = 1.0,
               weight_decay_penalty: float = 0.0,
               weight_decay_penalty_type: str = 'l2',
               dropouts: OneOrMany[float] = 0.5,
               activation_fns: OneOrMany[ActivationFn] = 'relu',
               pool_type: str = 'max',
               mode: str = 'classification',
               n_classes: int = 2,
               uncertainty: bool = False,
               residual: bool = False,
               padding: Union[int, str] = 'valid',
               **kwargs) -> None:
    """TorchModel wrapper for CNN

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
    bias_init_consts: list or loat
      the value to initialize the biases in each layer to.  The length of this
      list should equal len(layer_filters)+1, where the final element corresponds
      to the dense layer.  Alternatively this may be a single value instead of a
      list, in which case the same value is used for every layer.
    weight_decay_penalty: float
      the magnitude of the weight decay penalty to use
    weight_decay_penalty_type: str
      the type of penalty to use for weight decay, either 'l1' or 'l2'
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
    self.mode = mode
    self.n_classes = n_classes
    self.n_tasks = n_tasks

    self.model = CNNModule(n_tasks=n_tasks,
                           n_features=n_features,
                           dims=dims,
                           layer_filters=layer_filters,
                           kernel_size=kernel_size,
                           strides=strides,
                           weight_init_stddevs=weight_init_stddevs,
                           bias_init_consts=bias_init_consts,
                           dropouts=dropouts,
                           activation_fns=activation_fns,
                           pool_type=pool_type,
                           mode=mode,
                           n_classes=n_classes,
                           uncertainty=uncertainty,
                           residual=residual,
                           padding=padding)

    regularization_loss: Optional[Callable]

    if weight_decay_penalty != 0:
      weights = [layer.weight for layer in self.model.layers]
      if weight_decay_penalty_type == 'l1':
        regularization_loss = lambda: weight_decay_penalty * torch.sum(
            torch.stack([torch.abs(w).sum() for w in weights]))
      else:
        regularization_loss = lambda: weight_decay_penalty * torch.sum(
            torch.stack([torch.square(w).sum() for w in weights]))
    else:
      regularization_loss = None

    loss: Union[L2Loss, Callable[[Any, Any, Any], Any]]

    if uncertainty:

      def loss(outputs, labels, weights):

        diff = labels[0] - outputs[0]

        return torch.mean(diff**2 / torch.exp(outputs[1]) + outputs[1])

    else:
      loss = L2Loss()

    if self.mode == 'classification':
      output_types = ['prediction', 'loss']
    else:
      if uncertainty:
        output_types = ['prediction', 'variance', 'loss', 'loss']
      else:
        output_types = ["prediction"]

    super(CNN, self).__init__(self.model,
                              loss=loss,
                              output_types=output_types,
                              regularization_loss=regularization_loss,
                              **kwargs)

  def default_generator(self,
                        dataset,
                        epochs=1,
                        mode='fit',
                        deterministic=True,
                        pad_batches=True):

    for epoch in range(epochs):

      for (X_b, y_b, w_b,
           ids_b) in dataset.iterbatches(batch_size=self.batch_size,
                                         deterministic=deterministic,
                                         pad_batches=pad_batches):

        if self.mode == 'classification':
          if y_b is not None:
            y_b = to_one_hot(y_b.flatten(), self.n_classes)\
                .reshape(-1, self.n_tasks, self.n_classes)

        yield ([X_b], [y_b], [w_b])

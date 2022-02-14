"""PyTorch implementation of fully connected networks.
"""
import logging
import numpy as np
import torch
import torch.nn.functional as F
try:
  from collections.abc import Sequence as SequenceCollection
except:
  from collections import Sequence as SequenceCollection

import deepchem as dc
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.losses import _make_pytorch_shapes_consistent
from deepchem.metrics import to_one_hot

from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union
from deepchem.utils.typing import ActivationFn, LossFn, OneOrMany
from deepchem.utils.pytorch_utils import get_activation

logger = logging.getLogger(__name__)


class MultitaskClassifier(TorchModel):
  """A fully connected network for multitask classification.

  This class provides lots of options for customizing aspects of the model: the
  number and widths of layers, the activation functions, regularization methods,
  etc.

  It optionally can compose the model from pre-activation residual blocks, as
  described in https://arxiv.org/abs/1603.05027, rather than a simple stack of
  dense layers.  This often leads to easier training, especially when using a
  large number of layers.  Note that residual blocks can only be used when
  successive layers have the same width.  Wherever the layer width changes, a
  simple dense layer will be used even if residual=True.
  """

  def __init__(self,
               n_tasks: int,
               n_features: int,
               layer_sizes: Sequence[int] = [1000],
               weight_init_stddevs: OneOrMany[float] = 0.02,
               bias_init_consts: OneOrMany[float] = 1.0,
               weight_decay_penalty: float = 0.0,
               weight_decay_penalty_type: str = 'l2',
               dropouts: OneOrMany[float] = 0.5,
               activation_fns: OneOrMany[ActivationFn] = 'relu',
               n_classes: int = 2,
               residual: bool = False,
               **kwargs) -> None:
    """Create a MultitaskClassifier.

    In addition to the following arguments, this class also accepts
    all the keyword arguments from TensorGraph.

    Parameters
    ----------
    n_tasks: int
      number of tasks
    n_features: int
      number of features
    layer_sizes: list
      the size of each dense layer in the network.  The length of
      this list determines the number of layers.
    weight_init_stddevs: list or float
      the standard deviation of the distribution to use for weight
      initialization of each layer.  The length of this list should
      equal len(layer_sizes).  Alternatively this may be a single
      value instead of a list, in which case the same value is used
      for every layer.
    bias_init_consts: list or float
      the value to initialize the biases in each layer to.  The
      length of this list should equal len(layer_sizes).
      Alternatively this may be a single value instead of a list, in
      which case the same value is used for every layer.
    weight_decay_penalty: float
      the magnitude of the weight decay penalty to use
    weight_decay_penalty_type: str
      the type of penalty to use for weight decay, either 'l1' or 'l2'
    dropouts: list or float
      the dropout probablity to use for each layer.  The length of this list should equal len(layer_sizes).
      Alternatively this may be a single value instead of a list, in which case the same value is used for every layer.
    activation_fns: list or object
      the PyTorch activation function to apply to each layer.  The length of this list should equal
      len(layer_sizes).  Alternatively this may be a single value instead of a list, in which case the
      same value is used for every layer.  Standard activation functions from torch.nn.functional can be specified by name.
    n_classes: int
      the number of classes
    residual: bool
      if True, the model will be composed of pre-activation residual blocks instead
      of a simple stack of dense layers.
    """
    self.n_tasks = n_tasks
    self.n_features = n_features
    self.n_classes = n_classes
    n_layers = len(layer_sizes)
    if not isinstance(weight_init_stddevs, SequenceCollection):
      weight_init_stddevs = [weight_init_stddevs] * n_layers
    if not isinstance(bias_init_consts, SequenceCollection):
      bias_init_consts = [bias_init_consts] * n_layers
    if not isinstance(dropouts, SequenceCollection):
      dropouts = [dropouts] * n_layers
    if isinstance(activation_fns,
                  str) or not isinstance(activation_fns, SequenceCollection):
      activation_fns = [activation_fns] * n_layers
    activation_fns = [get_activation(f) for f in activation_fns]

    # Define the PyTorch Module that implements the model.

    class PytorchImpl(torch.nn.Module):

      def __init__(self):
        super(PytorchImpl, self).__init__()
        self.layers = torch.nn.ModuleList()
        prev_size = n_features
        for size, weight_stddev, bias_const in zip(
            layer_sizes, weight_init_stddevs, bias_init_consts):
          layer = torch.nn.Linear(prev_size, size)
          torch.nn.init.normal_(layer.weight, 0, weight_stddev)
          torch.nn.init.constant_(layer.bias, bias_const)
          self.layers.append(layer)
          prev_size = size
        self.output_layer = torch.nn.Linear(prev_size, n_tasks * n_classes)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)
        torch.nn.init.constant_(self.output_layer.bias, 0)

      def forward(self, x):
        prev_size = n_features
        next_activation = None
        for size, layer, dropout, activation_fn, in zip(
            layer_sizes, self.layers, dropouts, activation_fns):
          y = x
          if next_activation is not None:
            y = next_activation(x)
          y = layer(y)
          if dropout > 0.0 and self.training:
            y = F.dropout(y, dropout)
          if residual and prev_size == size:
            y = x + y
          x = y
          prev_size = size
          next_activation = activation_fn
        if next_activation is not None:
          y = next_activation(y)
        neural_fingerprint = y
        y = self.output_layer(y)
        logits = torch.reshape(y, (-1, n_tasks, n_classes))
        output = F.softmax(logits, dim=2)
        return (output, logits, neural_fingerprint)

    model = PytorchImpl()
    regularization_loss: Optional[Callable]
    if weight_decay_penalty != 0:
      weights = [layer.weight for layer in model.layers]
      if weight_decay_penalty_type == 'l1':
        regularization_loss = lambda: weight_decay_penalty * torch.sum(torch.stack([torch.abs(w).sum() for w in weights]))
      else:
        regularization_loss = lambda: weight_decay_penalty * torch.sum(torch.stack([torch.square(w).sum() for w in weights]))
    else:
      regularization_loss = None
    super(MultitaskClassifier, self).__init__(
        model,
        dc.models.losses.SoftmaxCrossEntropy(),
        output_types=['prediction', 'loss', 'embedding'],
        regularization_loss=regularization_loss,
        **kwargs)

  def default_generator(
      self,
      dataset: dc.data.Dataset,
      epochs: int = 1,
      mode: str = 'fit',
      deterministic: bool = True,
      pad_batches: bool = True) -> Iterable[Tuple[List, List, List]]:
    for epoch in range(epochs):
      for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=deterministic,
          pad_batches=pad_batches):
        if y_b is not None:
          y_b = to_one_hot(y_b.flatten(), self.n_classes).reshape(
              -1, self.n_tasks, self.n_classes)
        yield ([X_b], [y_b], [w_b])


class MultitaskRegressor(TorchModel):
  """A fully connected network for multitask regression.

  This class provides lots of options for customizing aspects of the model: the
  number and widths of layers, the activation functions, regularization methods,
  etc.

  It optionally can compose the model from pre-activation residual blocks, as
  described in https://arxiv.org/abs/1603.05027, rather than a simple stack of
  dense layers.  This often leads to easier training, especially when using a
  large number of layers.  Note that residual blocks can only be used when
  successive layers have the same width.  Wherever the layer width changes, a
  simple dense layer will be used even if residual=True.
  """

  def __init__(self,
               n_tasks: int,
               n_features: int,
               layer_sizes: Sequence[int] = [1000],
               weight_init_stddevs: OneOrMany[float] = 0.02,
               bias_init_consts: OneOrMany[float] = 1.0,
               weight_decay_penalty: float = 0.0,
               weight_decay_penalty_type: str = 'l2',
               dropouts: OneOrMany[float] = 0.5,
               activation_fns: OneOrMany[ActivationFn] = 'relu',
               uncertainty: bool = False,
               residual: bool = False,
               **kwargs) -> None:
    """Create a MultitaskRegressor.

    In addition to the following arguments, this class also accepts all the keywork arguments
    from TensorGraph.

    Parameters
    ----------
    n_tasks: int
      number of tasks
    n_features: int
      number of features
    layer_sizes: list
      the size of each dense layer in the network.  The length of this list determines the number of layers.
    weight_init_stddevs: list or float
      the standard deviation of the distribution to use for weight initialization of each layer.  The length
      of this list should equal len(layer_sizes)+1.  The final element corresponds to the output layer.
      Alternatively this may be a single value instead of a list, in which case the same value is used for every layer.
    bias_init_consts: list or float
      the value to initialize the biases in each layer to.  The length of this list should equal len(layer_sizes)+1.
      The final element corresponds to the output layer.  Alternatively this may be a single value instead of a list,
      in which case the same value is used for every layer.
    weight_decay_penalty: float
      the magnitude of the weight decay penalty to use
    weight_decay_penalty_type: str
      the type of penalty to use for weight decay, either 'l1' or 'l2'
    dropouts: list or float
      the dropout probablity to use for each layer.  The length of this list should equal len(layer_sizes).
      Alternatively this may be a single value instead of a list, in which case the same value is used for every layer.
    activation_fns: list or object
      the PyTorch activation function to apply to each layer.  The length of this list should equal
      len(layer_sizes).  Alternatively this may be a single value instead of a list, in which case the
      same value is used for every layer.  Standard activation functions from torch.nn.functional can be specified by name.
    uncertainty: bool
      if True, include extra outputs and loss terms to enable the uncertainty
      in outputs to be predicted
    residual: bool
      if True, the model will be composed of pre-activation residual blocks instead
      of a simple stack of dense layers.
    """
    self.n_tasks = n_tasks
    self.n_features = n_features
    n_layers = len(layer_sizes)
    if not isinstance(weight_init_stddevs, SequenceCollection):
      weight_init_stddevs = [weight_init_stddevs] * (n_layers + 1)
    if not isinstance(bias_init_consts, SequenceCollection):
      bias_init_consts = [bias_init_consts] * (n_layers + 1)
    if not isinstance(dropouts, SequenceCollection):
      dropouts = [dropouts] * n_layers
    if isinstance(activation_fns,
                  str) or not isinstance(activation_fns, SequenceCollection):
      activation_fns = [activation_fns] * n_layers
    activation_fns = [get_activation(f) for f in activation_fns]
    if uncertainty:
      if any(d == 0.0 for d in dropouts):
        raise ValueError(
            'Dropout must be included in every layer to predict uncertainty')

    # Define the PyTorch Module that implements the model.

    class PytorchImpl(torch.nn.Module):

      def __init__(self):
        super(PytorchImpl, self).__init__()
        self.layers = torch.nn.ModuleList()
        prev_size = n_features
        for size, weight_stddev, bias_const in zip(
            layer_sizes, weight_init_stddevs, bias_init_consts):
          layer = torch.nn.Linear(prev_size, size)
          torch.nn.init.normal_(layer.weight, 0, weight_stddev)
          torch.nn.init.constant_(layer.bias, bias_const)
          self.layers.append(layer)
          prev_size = size
        self.output_layer = torch.nn.Linear(prev_size, n_tasks)
        torch.nn.init.normal_(self.output_layer.weight, 0,
                              weight_init_stddevs[-1])
        torch.nn.init.constant_(self.output_layer.bias, bias_init_consts[-1])
        self.uncertainty_layer = torch.nn.Linear(prev_size, n_tasks)
        torch.nn.init.normal_(self.output_layer.weight, 0,
                              weight_init_stddevs[-1])
        torch.nn.init.constant_(self.output_layer.bias, 0)

      def forward(self, inputs):
        x, dropout_switch = inputs
        prev_size = n_features
        next_activation = None
        for size, layer, dropout, activation_fn, in zip(
            layer_sizes, self.layers, dropouts, activation_fns):
          y = x
          if next_activation is not None:
            y = next_activation(x)
          y = layer(y)
          if dropout > 0.0 and dropout_switch:
            y = F.dropout(y, dropout)
          if residual and prev_size == size:
            y = x + y
          x = y
          prev_size = size
          next_activation = activation_fn
        if next_activation is not None:
          y = next_activation(y)
        neural_fingerprint = y
        output = torch.reshape(self.output_layer(y), (-1, n_tasks, 1))
        if uncertainty:
          log_var = torch.reshape(self.uncertainty_layer(y), (-1, n_tasks, 1))
          var = torch.exp(log_var)
          return (output, var, output, log_var, neural_fingerprint)
        else:
          return (output, neural_fingerprint)

    model = PytorchImpl()
    regularization_loss: Optional[Callable]
    if weight_decay_penalty != 0:
      weights = [layer.weight for layer in model.layers]
      if weight_decay_penalty_type == 'l1':
        regularization_loss = lambda: weight_decay_penalty * torch.sum(torch.stack([torch.abs(w).sum() for w in weights]))
      else:
        regularization_loss = lambda: weight_decay_penalty * torch.sum(torch.stack([torch.square(w).sum() for w in weights]))
    else:
      regularization_loss = None
    loss: Union[dc.models.losses.Loss, LossFn]
    if uncertainty:
      output_types = ['prediction', 'variance', 'loss', 'loss', 'embedding']

      def loss(outputs, labels, weights):
        output, labels = _make_pytorch_shapes_consistent(outputs[0], labels[0])
        diff = labels - output
        losses = diff * diff / torch.exp(outputs[1]) + outputs[1]
        w = weights[0]
        if len(w.shape) < len(losses.shape):
          if isinstance(w, torch.Tensor):
            shape = tuple(w.shape)
          else:
            shape = w.shape
          shape = tuple(-1 if x is None else x for x in shape)
          w = w.reshape(shape + (1,) * (len(losses.shape) - len(w.shape)))

        loss = losses * w
        loss = loss.mean()
        if regularization_loss is not None:
          loss += regularization_loss()
        return loss
    else:
      output_types = ['prediction', 'embedding']
      loss = dc.models.losses.L2Loss()
    super(MultitaskRegressor, self).__init__(
        model,
        loss,
        output_types=output_types,
        regularization_loss=regularization_loss,
        **kwargs)

  def default_generator(
      self,
      dataset: dc.data.Dataset,
      epochs: int = 1,
      mode: str = 'fit',
      deterministic: bool = True,
      pad_batches: bool = True) -> Iterable[Tuple[List, List, List]]:
    for epoch in range(epochs):
      for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=deterministic,
          pad_batches=pad_batches):
        if mode == 'predict':
          dropout = np.array(0.0)
        else:
          dropout = np.array(1.0)
        yield ([X_b, dropout], [y_b], [w_b])


class MultitaskFitTransformRegressor(MultitaskRegressor):
  """Implements a MultitaskRegressor that performs on-the-fly transformation during fit/predict.

  Examples
  --------
  >>> n_samples = 10
  >>> n_features = 3
  >>> n_tasks = 1
  >>> ids = np.arange(n_samples)
  >>> X = np.random.rand(n_samples, n_features, n_features)
  >>> y = np.zeros((n_samples, n_tasks))
  >>> w = np.ones((n_samples, n_tasks))
  >>> dataset = dc.data.NumpyDataset(X, y, w, ids)
  >>> fit_transformers = [dc.trans.CoulombFitTransformer(dataset)]
  >>> model = dc.models.MultitaskFitTransformRegressor(n_tasks, [n_features, n_features],
  ...     dropouts=[0.], learning_rate=0.003, weight_init_stddevs=[np.sqrt(6)/np.sqrt(1000)],
  ...     batch_size=n_samples, fit_transformers=fit_transformers)
  >>> model.n_features
  12
  """

  def __init__(self,
               n_tasks: int,
               n_features: int,
               fit_transformers: Sequence[dc.trans.Transformer] = [],
               batch_size: int = 50,
               **kwargs):
    """Create a MultitaskFitTransformRegressor.

    In addition to the following arguments, this class also accepts all the keywork arguments
    from MultitaskRegressor.

    Parameters
    ----------
    n_tasks: int
      number of tasks
    n_features: list or int
      number of features
    fit_transformers: list
      List of dc.trans.FitTransformer objects
    """
    self.fit_transformers = fit_transformers

    # Run fit transformers on dummy dataset to determine n_features after transformation

    if isinstance(n_features, list):
      X_b = np.ones([batch_size] + n_features)
    elif isinstance(n_features, int):
      X_b = np.ones([batch_size, n_features])
    else:
      raise ValueError("n_features should be list or int")
    empty: np.ndarray = np.array([])
    for transformer in fit_transformers:
      assert transformer.transform_X and not (transformer.transform_y or
                                              transformer.transform_w)
      X_b, _, _, _ = transformer.transform_array(X_b, empty, empty, empty)
    n_features = X_b.shape[1]
    logger.info("n_features after fit_transform: %d", int(n_features))
    super(MultitaskFitTransformRegressor, self).__init__(
        n_tasks, n_features, batch_size=batch_size, **kwargs)

  def default_generator(
      self,
      dataset: dc.data.Dataset,
      epochs: int = 1,
      mode: str = 'fit',
      deterministic: bool = True,
      pad_batches: bool = True) -> Iterable[Tuple[List, List, List]]:
    empty: np.ndarray = np.array([])
    for epoch in range(epochs):
      for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=deterministic,
          pad_batches=pad_batches):
        if y_b is not None:
          y_b = y_b.reshape(-1, self.n_tasks, 1)
        if X_b is not None:
          if mode == 'fit':
            for transformer in self.fit_transformers:
              X_b, _, _, _ = transformer.transform_array(
                  X_b, empty, empty, empty)
        if mode == 'predict':
          dropout = np.array(0.0)
        else:
          dropout = np.array(1.0)
        yield ([X_b, dropout], [y_b], [w_b])

  def predict_on_generator(
      self,
      generator: Iterable[Tuple[Any, Any, Any]],
      transformers: List[dc.trans.Transformer] = [],
      output_types: Optional[OneOrMany[str]] = None) -> OneOrMany[np.ndarray]:

    def transform_generator():
      for inputs, labels, weights in generator:
        X_t = inputs[0]
        for transformer in self.fit_transformers:
          X_t = transformer.X_transform(X_t)
        yield ([X_t] + inputs[1:], labels, weights)

    return super(MultitaskFitTransformRegressor, self).predict_on_generator(
        transform_generator(), transformers, output_types)

"""TensorFlow implementation of fully connected networks.
"""
import logging
import warnings
import time
import numpy as np
import tensorflow as tf
import threading
import collections

import deepchem as dc
from deepchem.models import KerasModel
from deepchem.models.layers import SwitchedDropout
from deepchem.metrics import to_one_hot
from tensorflow.keras.layers import Input, Dense, Reshape, Softmax, Dropout, Activation, Lambda

from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union
from deepchem.utils.typing import KerasActivationFn, LossFn, OneOrMany

logger = logging.getLogger(__name__)


class MultitaskClassifier(KerasModel):
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
               weight_decay_penalty_type: str = "l2",
               dropouts: OneOrMany[float] = 0.5,
               activation_fns: OneOrMany[KerasActivationFn] = tf.nn.relu,
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
      the Tensorflow activation function to apply to each layer.  The length of this list should equal
      len(layer_sizes).  Alternatively this may be a single value instead of a list, in which case the
      same value is used for every layer.
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
    if not isinstance(weight_init_stddevs, collections.Sequence):
      weight_init_stddevs = [weight_init_stddevs] * n_layers
    if not isinstance(bias_init_consts, collections.Sequence):
      bias_init_consts = [bias_init_consts] * n_layers
    if not isinstance(dropouts, collections.Sequence):
      dropouts = [dropouts] * n_layers
    if not isinstance(activation_fns, collections.Sequence):
      activation_fns = [activation_fns] * n_layers
    if weight_decay_penalty != 0.0:
      if weight_decay_penalty_type == 'l1':
        regularizer = tf.keras.regularizers.l1(weight_decay_penalty)
      else:
        regularizer = tf.keras.regularizers.l2(weight_decay_penalty)
    else:
      regularizer = None

    # Add the input features.

    mol_features = Input(shape=(n_features,))
    prev_layer = mol_features
    prev_size = n_features
    next_activation = None

    # Add the dense layers

    for size, weight_stddev, bias_const, dropout, activation_fn in zip(
        layer_sizes, weight_init_stddevs, bias_init_consts, dropouts,
        activation_fns):
      layer = prev_layer
      if next_activation is not None:
        layer = Activation(next_activation)(layer)
      layer = Dense(
          size,
          kernel_initializer=tf.keras.initializers.TruncatedNormal(
              stddev=weight_stddev),
          bias_initializer=tf.constant_initializer(value=bias_const),
          kernel_regularizer=regularizer)(layer)
      if dropout > 0.0:
        layer = Dropout(rate=dropout)(layer)
      if residual and prev_size == size:
        prev_layer = Lambda(lambda x: x[0] + x[1])([prev_layer, layer])
      else:
        prev_layer = layer
      prev_size = size
      next_activation = activation_fn
    if next_activation is not None:
      prev_layer = Activation(activation_fn)(prev_layer)
    self.neural_fingerprint = prev_layer
    logits = Reshape((n_tasks,
                      n_classes))(Dense(n_tasks * n_classes)(prev_layer))
    output = Softmax()(logits)
    model = tf.keras.Model(inputs=mol_features, outputs=[output, logits])
    super(MultitaskClassifier, self).__init__(
        model,
        dc.models.losses.SoftmaxCrossEntropy(),
        output_types=['prediction', 'loss'],
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


class MultitaskRegressor(KerasModel):
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
               weight_decay_penalty_type: str = "l2",
               dropouts: OneOrMany[float] = 0.5,
               activation_fns: OneOrMany[KerasActivationFn] = tf.nn.relu,
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
      the Tensorflow activation function to apply to each layer.  The length of this list should equal
      len(layer_sizes).  Alternatively this may be a single value instead of a list, in which case the
      same value is used for every layer.
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
    if not isinstance(weight_init_stddevs, collections.Sequence):
      weight_init_stddevs = [weight_init_stddevs] * (n_layers + 1)
    if not isinstance(bias_init_consts, collections.Sequence):
      bias_init_consts = [bias_init_consts] * (n_layers + 1)
    if not isinstance(dropouts, collections.Sequence):
      dropouts = [dropouts] * n_layers
    if not isinstance(activation_fns, collections.Sequence):
      activation_fns = [activation_fns] * n_layers
    if weight_decay_penalty != 0.0:
      if weight_decay_penalty_type == 'l1':
        regularizer = tf.keras.regularizers.l1(weight_decay_penalty)
      else:
        regularizer = tf.keras.regularizers.l2(weight_decay_penalty)
    else:
      regularizer = None
    if uncertainty:
      if any(d == 0.0 for d in dropouts):
        raise ValueError(
            'Dropout must be included in every layer to predict uncertainty')

    # Add the input features.

    mol_features = Input(shape=(n_features,))
    dropout_switch = Input(shape=tuple())
    prev_layer = mol_features
    prev_size = n_features
    next_activation = None

    # Add the dense layers

    for size, weight_stddev, bias_const, dropout, activation_fn in zip(
        layer_sizes, weight_init_stddevs, bias_init_consts, dropouts,
        activation_fns):
      layer = prev_layer
      if next_activation is not None:
        layer = Activation(next_activation)(layer)
      layer = Dense(
          size,
          kernel_initializer=tf.keras.initializers.TruncatedNormal(
              stddev=weight_stddev),
          bias_initializer=tf.constant_initializer(value=bias_const),
          kernel_regularizer=regularizer)(layer)
      if dropout > 0.0:
        layer = SwitchedDropout(rate=dropout)([layer, dropout_switch])
      if residual and prev_size == size:
        prev_layer = Lambda(lambda x: x[0] + x[1])([prev_layer, layer])
      else:
        prev_layer = layer
      prev_size = size
      next_activation = activation_fn
    if next_activation is not None:
      prev_layer = Activation(activation_fn)(prev_layer)
    self.neural_fingerprint = prev_layer
    output = Reshape((n_tasks, 1))(Dense(
        n_tasks,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(
            stddev=weight_init_stddevs[-1]),
        bias_initializer=tf.constant_initializer(
            value=bias_init_consts[-1]))(prev_layer))
    loss: Union[dc.models.losses.Loss, LossFn]
    if uncertainty:
      log_var = Reshape((n_tasks, 1))(Dense(
          n_tasks,
          kernel_initializer=tf.keras.initializers.TruncatedNormal(
              stddev=weight_init_stddevs[-1]),
          bias_initializer=tf.constant_initializer(value=0.0))(prev_layer))
      var = Activation(tf.exp)(log_var)
      outputs = [output, var, output, log_var]
      output_types = ['prediction', 'variance', 'loss', 'loss']

      def loss(outputs, labels, weights):
        diff = labels[0] - outputs[0]
        return tf.reduce_mean(diff * diff / tf.exp(outputs[1]) + outputs[1])
    else:
      outputs = [output]
      output_types = ['prediction']
      loss = dc.models.losses.L2Loss()
    model = tf.keras.Model(
        inputs=[mol_features, dropout_switch], outputs=outputs)
    super(MultitaskRegressor, self).__init__(
        model, loss, output_types=output_types, **kwargs)

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

  Example:

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
    for transformer in fit_transformers:
      assert transformer.transform_X and not (transformer.transform_y or
                                              transformer.transform_w)
      X_b, _, _, _ = transformer.transform_array(X_b, None, None, None)
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
              X_b, _, _, _ = transformer.transform_array(X_b, None, None, None)
        if mode == 'predict':
          dropout = np.array(0.0)
        else:
          dropout = np.array(1.0)
        yield ([X_b, dropout], [y_b], [w_b])

  def predict_on_generator(
      self,
      generator: Iterable[Tuple[Any, Any, Any]],
      transformers: List[dc.trans.Transformer] = [],
      outputs: Optional[OneOrMany[tf.Tensor]] = None,
      output_types: Optional[OneOrMany[str]] = None) -> OneOrMany[np.ndarray]:

    def transform_generator():
      for inputs, labels, weights in generator:
        X_t = inputs[0]
        for transformer in self.fit_transformers:
          X_t = transformer.X_transform(X_t)
        yield ([X_t] + inputs[1:], labels, weights)

    return super(MultitaskFitTransformRegressor, self).predict_on_generator(
        transform_generator(), transformers, outputs, output_types)

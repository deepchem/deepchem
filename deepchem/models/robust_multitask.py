import numpy as np
import tensorflow as tf
import collections

import logging
from deepchem.metrics import to_one_hot
from deepchem.models import KerasModel
from deepchem.models.layers import Stack
from deepchem.models.losses import SoftmaxCrossEntropy, L2Loss

logger = logging.getLogger(__name__)


class RobustMultitaskClassifier(KerasModel):
  """Implements a neural network for robust multitasking.

  The key idea of this model is to have bypass layers that feed
  directly from features to task output. This might provide some
  flexibility toroute around challenges in multitasking with
  destructive interference. 

  References
  ----------
  This technique was introduced in [1]_

  .. [1] Ramsundar, Bharath, et al. "Is multitask deep learning practical for pharma?." Journal of chemical information and modeling 57.8 (2017): 2068-2076.

  """

  def __init__(self,
               n_tasks,
               n_features,
               layer_sizes=[1000],
               weight_init_stddevs=0.02,
               bias_init_consts=1.0,
               weight_decay_penalty=0.0,
               weight_decay_penalty_type="l2",
               dropouts=0.5,
               activation_fns=tf.nn.relu,
               n_classes=2,
               bypass_layer_sizes=[100],
               bypass_weight_init_stddevs=[.02],
               bypass_bias_init_consts=[1.],
               bypass_dropouts=[.5],
               **kwargs):
    """  Create a RobustMultitaskClassifier.

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
      of this list should equal len(layer_sizes).  Alternatively this may be a single value instead of a list,
      in which case the same value is used for every layer.
    bias_init_consts: list or loat
      the value to initialize the biases in each layer to.  The length of this list should equal len(layer_sizes).
      Alternatively this may be a single value instead of a list, in which case the same value is used for every layer.
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
    bypass_layer_sizes: list
      the size of each dense layer in the bypass network. The length of this list determines the number of bypass layers.
    bypass_weight_init_stddevs: list or float
      the standard deviation of the distribution to use for weight initialization of bypass layers.
      same requirements as weight_init_stddevs
    bypass_bias_init_consts: list or float
      the value to initialize the biases in bypass layers
      same requirements as bias_init_consts
    bypass_dropouts: list or float
      the dropout probablity to use for bypass layers.
      same requirements as dropouts
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

    n_bypass_layers = len(bypass_layer_sizes)
    if not isinstance(bypass_weight_init_stddevs, collections.Sequence):
      bypass_weight_init_stddevs = [bypass_weight_init_stddevs
                                   ] * n_bypass_layers
    if not isinstance(bypass_bias_init_consts, collections.Sequence):
      bypass_bias_init_consts = [bypass_bias_init_consts] * n_bypass_layers
    if not isinstance(bypass_dropouts, collections.Sequence):
      bypass_dropouts = [bypass_dropouts] * n_bypass_layers
    bypass_activation_fns = [activation_fns[0]] * n_bypass_layers

    # Add the input features.
    mol_features = tf.keras.Input(shape=(n_features,))
    prev_layer = mol_features

    # Add the shared dense layers
    for size, weight_stddev, bias_const, dropout, activation_fn in zip(
        layer_sizes, weight_init_stddevs, bias_init_consts, dropouts,
        activation_fns):
      layer = tf.keras.layers.Dense(
          size,
          activation=activation_fn,
          kernel_initializer=tf.keras.initializers.TruncatedNormal(
              stddev=weight_stddev),
          bias_initializer=tf.constant_initializer(value=bias_const),
          kernel_regularizer=regularizer)(prev_layer)
      if dropout > 0.0:
        layer = tf.keras.layers.Dropout(rate=dropout)(layer)
      prev_layer = layer
    top_multitask_layer = prev_layer

    task_outputs = []
    for i in range(self.n_tasks):
      prev_layer = mol_features
      # Add task-specific bypass layers
      for size, weight_stddev, bias_const, dropout, activation_fn in zip(
          bypass_layer_sizes, bypass_weight_init_stddevs,
          bypass_bias_init_consts, bypass_dropouts, bypass_activation_fns):
        layer = tf.keras.layers.Dense(
            size,
            activation=activation_fn,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=weight_stddev),
            bias_initializer=tf.constant_initializer(value=bias_const),
            kernel_regularizer=regularizer)(prev_layer)
        if dropout > 0.0:
          layer = tf.keras.layers.Dropout(rate=dropout)(layer)
        prev_layer = layer
      top_bypass_layer = prev_layer

      if n_bypass_layers > 0:
        task_layer = tf.keras.layers.Concatenate(axis=1)(
            [top_multitask_layer, top_bypass_layer])
      else:
        task_layer = top_multitask_layer

      task_out = tf.keras.layers.Dense(n_classes)(task_layer)
      task_outputs.append(task_out)

    logits = Stack(axis=1)(task_outputs)
    output = tf.keras.layers.Softmax()(logits)
    model = tf.keras.Model(inputs=mol_features, outputs=[output, logits])
    super(RobustMultitaskClassifier, self).__init__(
        model,
        SoftmaxCrossEntropy(),
        output_types=['prediction', 'loss'],
        **kwargs)

  def default_generator(self,
                        dataset,
                        epochs=1,
                        mode='fit',
                        deterministic=True,
                        pad_batches=True):
    for epoch in range(epochs):
      for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=deterministic,
          pad_batches=pad_batches):
        if y_b is not None:
          y_b = to_one_hot(y_b.flatten(), self.n_classes).reshape(
              -1, self.n_tasks, self.n_classes)
        yield ([X_b], [y_b], [w_b])

  def create_estimator_inputs(self, feature_columns, weight_column, features,
                              labels, mode):
    tensors = {}
    for layer, column in zip(self.features, feature_columns):
      tensors[layer] = tf.feature_column.input_layer(features, [column])
    if weight_column is not None:
      tensors[self.task_weights[0]] = tf.feature_column.input_layer(
          features, [weight_column])
    if labels is not None:
      tensors[self.labels[0]] = tf.one_hot(
          tf.cast(labels, tf.int32), self.n_classes)
    return tensors


class RobustMultitaskRegressor(KerasModel):
  """Implements a neural network for robust multitasking.

  The key idea of this model is to have bypass layers that feed
  directly from features to task output. This might provide some
  flexibility toroute around challenges in multitasking with
  destructive interference.

  References
  ----------
  .. [1]   Ramsundar, Bharath, et al. "Is multitask deep learning practical for pharma?." Journal of chemical information and modeling 57.8 (2017): 2068-2076.

  """

  def __init__(self,
               n_tasks,
               n_features,
               layer_sizes=[1000],
               weight_init_stddevs=0.02,
               bias_init_consts=1.0,
               weight_decay_penalty=0.0,
               weight_decay_penalty_type="l2",
               dropouts=0.5,
               activation_fns=tf.nn.relu,
               bypass_layer_sizes=[100],
               bypass_weight_init_stddevs=[.02],
               bypass_bias_init_consts=[1.],
               bypass_dropouts=[.5],
               **kwargs):
    """ Create a RobustMultitaskRegressor.

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
      of this list should equal len(layer_sizes).  Alternatively this may be a single value instead of a list,
      in which case the same value is used for every layer.
    bias_init_consts: list or loat
      the value to initialize the biases in each layer to.  The length of this list should equal len(layer_sizes).
      Alternatively this may be a single value instead of a list, in which case the same value is used for every layer.
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
    bypass_layer_sizes: list
      the size of each dense layer in the bypass network. The length of this list determines the number of bypass layers.
    bypass_weight_init_stddevs: list or float
      the standard deviation of the distribution to use for weight initialization of bypass layers.
      same requirements as weight_init_stddevs
    bypass_bias_init_consts: list or float
      the value to initialize the biases in bypass layers
      same requirements as bias_init_consts
    bypass_dropouts: list or float
      the dropout probablity to use for bypass layers.
      same requirements as dropouts
    """
    self.n_tasks = n_tasks
    self.n_features = n_features
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

    n_bypass_layers = len(bypass_layer_sizes)
    if not isinstance(bypass_weight_init_stddevs, collections.Sequence):
      bypass_weight_init_stddevs = [bypass_weight_init_stddevs
                                   ] * n_bypass_layers
    if not isinstance(bypass_bias_init_consts, collections.Sequence):
      bypass_bias_init_consts = [bypass_bias_init_consts] * n_bypass_layers
    if not isinstance(bypass_dropouts, collections.Sequence):
      bypass_dropouts = [bypass_dropouts] * n_bypass_layers
    bypass_activation_fns = [activation_fns[0]] * n_bypass_layers

    # Add the input features.
    mol_features = tf.keras.Input(shape=(n_features,))
    prev_layer = mol_features

    # Add the shared dense layers
    for size, weight_stddev, bias_const, dropout, activation_fn in zip(
        layer_sizes, weight_init_stddevs, bias_init_consts, dropouts,
        activation_fns):
      layer = tf.keras.layers.Dense(
          size,
          activation=activation_fn,
          kernel_initializer=tf.keras.initializers.TruncatedNormal(
              stddev=weight_stddev),
          bias_initializer=tf.constant_initializer(value=bias_const),
          kernel_regularizer=regularizer)(prev_layer)
      if dropout > 0.0:
        layer = tf.keras.layers.Dropout(rate=dropout)(layer)
      prev_layer = layer
    top_multitask_layer = prev_layer

    task_outputs = []
    for i in range(self.n_tasks):
      prev_layer = mol_features
      # Add task-specific bypass layers
      for size, weight_stddev, bias_const, dropout, activation_fn in zip(
          bypass_layer_sizes, bypass_weight_init_stddevs,
          bypass_bias_init_consts, bypass_dropouts, bypass_activation_fns):
        layer = tf.keras.layers.Dense(
            size,
            activation=activation_fn,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=weight_stddev),
            bias_initializer=tf.constant_initializer(value=bias_const),
            kernel_regularizer=regularizer)(prev_layer)
        if dropout > 0.0:
          layer = tf.keras.layers.Dropout(rate=dropout)(layer)
        prev_layer = layer
      top_bypass_layer = prev_layer

      if n_bypass_layers > 0:
        task_layer = tf.keras.layers.Concatenate(axis=1)(
            [top_multitask_layer, top_bypass_layer])
      else:
        task_layer = top_multitask_layer

      task_out = tf.keras.layers.Dense(1)(task_layer)
      task_outputs.append(task_out)

    outputs = tf.keras.layers.Concatenate(axis=1)(task_outputs)
    model = tf.keras.Model(inputs=mol_features, outputs=outputs)
    super(RobustMultitaskRegressor, self).__init__(model, L2Loss(), **kwargs)

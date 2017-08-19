"""TensorFlow implementation of fully connected networks.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import warnings
import time
import numpy as np
import tensorflow as tf
import threading
import collections

import deepchem as dc
from deepchem.nn import model_ops
from deepchem.utils.save import log
from deepchem.metrics import to_one_hot, from_one_hot
from deepchem.models.tensorflow_models import TensorflowGraph
from deepchem.models.tensorflow_models import TensorflowGraphModel
from deepchem.models.tensorflow_models import TensorflowClassifier
from deepchem.models.tensorflow_models import TensorflowRegressor
from deepchem.metrics import to_one_hot

from deepchem.models.tensorgraph.tensor_graph import TensorGraph, TFWrapper
from deepchem.models.tensorgraph.layers import Feature, Label, Weights, WeightedError, Dense, Dropout, WeightDecay, Reshape, SoftMaxCrossEntropy, L2Loss


class TensorGraphMultiTaskClassifier(TensorGraph):

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
               **kwargs):
    """Create a TensorGraphMultiTaskClassifier.

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
    """
    super(TensorGraphMultiTaskClassifier, self).__init__(**kwargs)
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

    # Add the input features.

    mol_features = Feature(shape=(None, n_features))
    prev_layer = mol_features

    # Add the dense layers

    for size, weight_stddev, bias_const, dropout, activation_fn in zip(
        layer_sizes, weight_init_stddevs, bias_init_consts, dropouts,
        activation_fns):
      layer = Dense(
          in_layers=[prev_layer],
          out_channels=size,
          activation_fn=activation_fn,
          weights_initializer=TFWrapper(
              tf.truncated_normal_initializer, stddev=weight_stddev),
          biases_initializer=TFWrapper(
              tf.constant_initializer, value=bias_const))
      if dropout > 0.0:
        layer = Dropout(dropout, in_layers=[layer])
      prev_layer = layer

    # Compute the loss function for each label.

    output = Reshape(
        shape=(-1, n_tasks, n_classes),
        in_layers=[
            Dense(in_layers=[prev_layer], out_channels=n_tasks * n_classes)
        ])
    self.add_output(output)
    labels = Label(shape=(None, n_tasks, n_classes))
    weights = Weights(shape=(None, n_tasks))
    loss = SoftMaxCrossEntropy(in_layers=[labels, output])
    weighted_loss = WeightedError(in_layers=[loss, weights])
    if weight_decay_penalty != 0.0:
      weighted_loss = WeightDecay(
          weight_decay_penalty,
          weight_decay_penalty_type,
          in_layers=[weighted_loss])
    self.set_loss(weighted_loss)

  def default_generator(self,
                        dataset,
                        epochs=1,
                        predict=False,
                        deterministic=True,
                        pad_batches=True):
    for epoch in range(epochs):
      for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=deterministic,
          pad_batches=pad_batches):
        feed_dict = dict()
        if y_b is not None and not predict:
          feed_dict[self.labels[0]] = to_one_hot(y_b.flatten(),
                                                 self.n_classes).reshape(
                                                     -1, self.n_tasks,
                                                     self.n_classes)
        if X_b is not None:
          feed_dict[self.features[0]] = X_b
        if w_b is not None and not predict:
          feed_dict[self.task_weights[0]] = w_b
        yield feed_dict

  def predict_proba(self, dataset, transformers=[], outputs=None):
    return super(TensorGraphMultiTaskClassifier, self).predict(
        dataset, transformers, outputs)

  def predict(self, dataset, transformers=[], outputs=None):
    """
    Uses self to make predictions on provided Dataset object.

    Parameters
    ----------
    dataset: dc.data.Dataset
      Dataset to make prediction on
    transformers: list
      List of dc.trans.Transformers.
    outputs: object 
      If outputs is None, then will assume outputs = self.outputs[0] (single
      output). If outputs is a Layer/Tensor, then will evaluate and return as a
      single ndarray. If outputs is a list of Layers/Tensors, will return a list
      of ndarrays.

    Returns
    -------
    y_pred: numpy ndarray or list of numpy ndarrays
    """
    # Results is of shape (n_samples, n_tasks, n_classes)
    retval = super(TensorGraphMultiTaskClassifier, self).predict(
        dataset, transformers, outputs)
    # retval is of shape (n_samples, n_tasks)
    return np.argmax(retval, axis=2)


class TensorGraphMultiTaskRegressor(TensorGraph):

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
               **kwargs):
    """Create a TensorGraphMultiTaskRegressor.

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
    """
    super(TensorGraphMultiTaskRegressor, self).__init__(**kwargs)
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

    # Add the input features.

    mol_features = Feature(shape=(None, n_features))
    prev_layer = mol_features

    # Add the dense layers

    for size, weight_stddev, bias_const, dropout, activation_fn in zip(
        layer_sizes, weight_init_stddevs, bias_init_consts, dropouts,
        activation_fns):
      layer = Dense(
          in_layers=[prev_layer],
          out_channels=size,
          activation_fn=activation_fn,
          weights_initializer=TFWrapper(
              tf.truncated_normal_initializer, stddev=weight_stddev),
          biases_initializer=TFWrapper(
              tf.constant_initializer, value=bias_const))
      if dropout > 0.0:
        layer = Dropout(dropout, in_layers=[layer])
      prev_layer = layer

    # Compute the loss function for each label.

    output = Reshape(
        shape=(-1, n_tasks, 1),
        in_layers=[
            Dense(
                in_layers=[prev_layer],
                out_channels=n_tasks,
                weights_initializer=TFWrapper(
                    tf.truncated_normal_initializer,
                    stddev=weight_init_stddevs[-1]),
                biases_initializer=TFWrapper(
                    tf.constant_initializer, value=bias_init_consts[-1]))
        ])
    self.add_output(output)
    labels = Label(shape=(None, n_tasks, 1))
    weights = Weights(shape=(None, n_tasks))
    loss = L2Loss(in_layers=[labels, output])
    weighted_loss = WeightedError(in_layers=[loss, weights])
    if weight_decay_penalty != 0.0:
      weighted_loss = WeightDecay(
          weight_decay_penalty,
          weight_decay_penalty_type,
          in_layers=[weighted_loss])
    self.set_loss(weighted_loss)

  def default_generator(self,
                        dataset,
                        epochs=1,
                        predict=False,
                        deterministic=True,
                        pad_batches=True):
    for epoch in range(epochs):
      for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=deterministic,
          pad_batches=pad_batches):
        feed_dict = dict()
        if y_b is not None and not predict:
          feed_dict[self.labels[0]] = y_b.reshape(-1, self.n_tasks, 1)
        if X_b is not None:
          feed_dict[self.features[0]] = X_b
        if w_b is not None and not predict:
          feed_dict[self.task_weights[0]] = w_b
        yield feed_dict


class TensorGraphMultiTaskFitTransformRegressor(TensorGraphMultiTaskRegressor):
  """Implements a TensorGraphMultiTaskRegressor that performs on-the-fly transformation during fit/predict.

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
  >>> model = dc.models.TensorflowMultiTaskFitTransformRegressor(n_tasks, [n_features, n_features],
  ...     dropouts=[0.], learning_rate=0.003, weight_init_stddevs=[np.sqrt(6)/np.sqrt(1000)],
  ...     batch_size=n_samples, fit_transformers=fit_transformers, n_evals=1)
  n_features after fit_transform: 12
  """

  def __init__(self,
               n_tasks,
               n_features,
               fit_transformers=[],
               n_evals=1,
               batch_size=50,
               **kwargs):
    """Create a TensorGraphMultiTaskFitTransformRegressor.

    In addition to the following arguments, this class also accepts all the keywork arguments
    from TensorGraphMultiTaskRegressor.

    Parameters
    ----------
    n_tasks: int
      number of tasks
    n_features: list or int
      number of features
    fit_transformers: list
      List of dc.trans.FitTransformer objects
    n_evals: int
      Number of evalations per example at predict time
    """
    self.fit_transformers = fit_transformers
    self.n_evals = n_evals

    # Run fit transformers on dummy dataset to determine n_features after transformation

    if isinstance(n_features, list):
      X_b = np.ones([batch_size] + n_features)
    elif isinstance(n_features, int):
      X_b = np.ones([batch_size, n_features])
    else:
      raise ValueError("n_features should be list or int")
    for transformer in fit_transformers:
      X_b = transformer.X_transform(X_b)
    n_features = X_b.shape[1]
    print("n_features after fit_transform: %d" % int(n_features))
    super(TensorGraphMultiTaskFitTransformRegressor, self).__init__(
        n_tasks, n_features, batch_size=batch_size, **kwargs)

  def default_generator(self,
                        dataset,
                        epochs=1,
                        predict=False,
                        deterministic=True,
                        pad_batches=True):
    for epoch in range(epochs):
      for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=deterministic,
          pad_batches=pad_batches):
        feed_dict = dict()
        if y_b is not None and not predict:
          feed_dict[self.labels[0]] = y_b.reshape(-1, self.n_tasks, 1)
        if X_b is not None:
          if not predict:
            for transformer in self.fit_transformers:
              X_b = transformer.X_transform(X_b)
          feed_dict[self.features[0]] = X_b
        if w_b is not None and not predict:
          feed_dict[self.task_weights[0]] = w_b
        yield feed_dict

  def predict_on_generator(self, generator, transformers=[], outputs=None):

    def transform_generator():
      for feed_dict in generator:
        X = feed_dict[self.features[0]]
        for i in range(self.n_evals):
          X_t = X
        for transformer in self.fit_transformers:
          X_t = transformer.X_transform(X_t)
        feed_dict[self.features[0]] = X_t
        yield feed_dict

    return super(TensorGraphMultiTaskFitTransformRegressor,
                 self).predict_on_generator(transform_generator(), transformers,
                                            outputs)


class TensorflowMultiTaskClassifier(TensorflowClassifier):
  """Implements an icml model as configured in a model_config.proto."""

  def build(self, graph, name_scopes, training):
    """Constructs the graph architecture as specified in its config.

    This method creates the following Placeholders:
      mol_features: Molecule descriptor (e.g. fingerprint) tensor with shape
        batch_size x n_features.
    """
    warnings.warn("TensorflowMultiTaskClassifier is deprecated. "
                  "Will be removed in DeepChem 1.4.", DeprecationWarning)
    placeholder_scope = TensorflowGraph.get_placeholder_scope(
        graph, name_scopes)
    n_features = self.n_features
    with graph.as_default():
      with placeholder_scope:
        mol_features = tf.placeholder(
            tf.float32, shape=[None, n_features], name='mol_features')

      layer_sizes = self.layer_sizes
      weight_init_stddevs = self.weight_init_stddevs
      bias_init_consts = self.bias_init_consts
      dropouts = self.dropouts
      lengths_set = {
          len(layer_sizes),
          len(weight_init_stddevs),
          len(bias_init_consts),
          len(dropouts),
      }
      assert len(lengths_set) == 1, 'All layer params must have same length.'
      n_layers = lengths_set.pop()
      assert n_layers > 0, 'Must have some layers defined.'

      label_placeholders = self.add_label_placeholders(graph, name_scopes)
      weight_placeholders = self.add_example_weight_placeholders(
          graph, name_scopes)
      if training:
        graph.queue = tf.FIFOQueue(
            capacity=5,
            dtypes=[tf.float32] *
            (len(label_placeholders) + len(weight_placeholders) + 1))
        graph.enqueue = graph.queue.enqueue([mol_features] + label_placeholders
                                            + weight_placeholders)
        queue_outputs = graph.queue.dequeue()
        labels = queue_outputs[1:len(label_placeholders) + 1]
        weights = queue_outputs[len(label_placeholders) + 1:]
        prev_layer = queue_outputs[0]
      else:
        labels = label_placeholders
        weights = weight_placeholders
        prev_layer = mol_features

      prev_layer_size = n_features
      for i in range(n_layers):
        layer = tf.nn.relu(
            model_ops.fully_connected_layer(
                tensor=prev_layer,
                size=layer_sizes[i],
                weight_init=tf.truncated_normal(
                    shape=[prev_layer_size, layer_sizes[i]],
                    stddev=weight_init_stddevs[i]),
                bias_init=tf.constant(
                    value=bias_init_consts[i], shape=[layer_sizes[i]])))
        layer = model_ops.dropout(layer, dropouts[i], training)
        prev_layer = layer
        prev_layer_size = layer_sizes[i]

      output = model_ops.multitask_logits(layer, self.n_tasks)
    return (output, labels, weights)

  def construct_feed_dict(self, X_b, y_b=None, w_b=None, ids_b=None):
    """Construct a feed dictionary from minibatch data.

    TODO(rbharath): ids_b is not used here. Can we remove it?

    Args:
      X_b: np.ndarray of shape (batch_size, n_features)
      y_b: np.ndarray of shape (batch_size, n_tasks)
      w_b: np.ndarray of shape (batch_size, n_tasks)
      ids_b: List of length (batch_size) with datapoint identifiers.
    """
    orig_dict = {}
    orig_dict["mol_features"] = X_b
    for task in range(self.n_tasks):
      if y_b is not None:
        orig_dict["labels_%d" % task] = to_one_hot(y_b[:, task])
      else:
        # Dummy placeholders
        orig_dict["labels_%d" %
                  task] = np.squeeze(to_one_hot(np.zeros((self.batch_size,))))
      if w_b is not None:
        orig_dict["weights_%d" % task] = w_b[:, task]
      else:
        # Dummy placeholders
        orig_dict["weights_%d" % task] = np.ones((self.batch_size,))
    return TensorflowGraph.get_feed_dict(orig_dict)


class TensorflowMultiTaskRegressor(TensorflowRegressor):
  """Implements an icml model as configured in a model_config.proto."""

  def build(self, graph, name_scopes, training):
    """Constructs the graph architecture as specified in its config.

    This method creates the following Placeholders:
      mol_features: Molecule descriptor (e.g. fingerprint) tensor with shape
        batch_size x n_features.
    """
    warnings.warn("TensorflowMultiTaskRegressor is deprecated. "
                  "Will be removed in DeepChem 1.4.", DeprecationWarning)
    n_features = self.n_features
    placeholder_scope = TensorflowGraph.get_placeholder_scope(
        graph, name_scopes)
    with graph.as_default():
      with placeholder_scope:
        mol_features = tf.placeholder(
            tf.float32, shape=[None, n_features], name='mol_features')

      layer_sizes = self.layer_sizes
      weight_init_stddevs = self.weight_init_stddevs
      bias_init_consts = self.bias_init_consts
      dropouts = self.dropouts
      lengths_set = {
          len(layer_sizes),
          len(weight_init_stddevs),
          len(bias_init_consts),
          len(dropouts),
      }
      assert len(lengths_set) == 1, 'All layer params must have same length.'
      n_layers = lengths_set.pop()
      assert n_layers > 0, 'Must have some layers defined.'

      label_placeholders = self.add_label_placeholders(graph, name_scopes)
      weight_placeholders = self.add_example_weight_placeholders(
          graph, name_scopes)
      if training:
        graph.queue = tf.FIFOQueue(
            capacity=5,
            dtypes=[tf.float32] *
            (len(label_placeholders) + len(weight_placeholders) + 1))
        graph.enqueue = graph.queue.enqueue([mol_features] + label_placeholders
                                            + weight_placeholders)
        queue_outputs = graph.queue.dequeue()
        labels = queue_outputs[1:len(label_placeholders) + 1]
        weights = queue_outputs[len(label_placeholders) + 1:]
        prev_layer = queue_outputs[0]
      else:
        labels = label_placeholders
        weights = weight_placeholders
        prev_layer = mol_features

      prev_layer_size = n_features
      for i in range(n_layers):
        layer = tf.nn.relu(
            model_ops.fully_connected_layer(
                tensor=prev_layer,
                size=layer_sizes[i],
                weight_init=tf.truncated_normal(
                    shape=[prev_layer_size, layer_sizes[i]],
                    stddev=weight_init_stddevs[i]),
                bias_init=tf.constant(
                    value=bias_init_consts[i], shape=[layer_sizes[i]])))
        layer = model_ops.dropout(layer, dropouts[i], training)
        prev_layer = layer
        prev_layer_size = layer_sizes[i]

      output = []
      for task in range(self.n_tasks):
        output.append(
            tf.squeeze(
                model_ops.fully_connected_layer(
                    tensor=prev_layer,
                    size=layer_sizes[i],
                    weight_init=tf.truncated_normal(
                        shape=[prev_layer_size, 1],
                        stddev=weight_init_stddevs[i]),
                    bias_init=tf.constant(value=bias_init_consts[i], shape=[1
                                                                           ]))))
    return (output, labels, weights)

  def construct_feed_dict(self, X_b, y_b=None, w_b=None, ids_b=None):
    """Construct a feed dictionary from minibatch data.

    TODO(rbharath): ids_b is not used here. Can we remove it?

    Args:
      X_b: np.ndarray of shape (batch_size, n_features)
      y_b: np.ndarray of shape (batch_size, n_tasks)
      w_b: np.ndarray of shape (batch_size, n_tasks)
      ids_b: List of length (batch_size) with datapoint identifiers.
    """
    orig_dict = {}
    orig_dict["mol_features"] = X_b
    for task in range(self.n_tasks):
      if y_b is not None:
        orig_dict["labels_%d" % task] = y_b[:, task]
      else:
        # Dummy placeholders
        orig_dict["labels_%d" % task] = np.squeeze(np.zeros((self.batch_size,)))
      if w_b is not None:
        orig_dict["weights_%d" % task] = w_b[:, task]
      else:
        # Dummy placeholders
        orig_dict["weights_%d" % task] = np.ones((self.batch_size,))
    return TensorflowGraph.get_feed_dict(orig_dict)


class TensorflowMultiTaskFitTransformRegressor(TensorflowMultiTaskRegressor):
  """Implements a TensorflowMultiTaskRegressor that performs on-the-fly transformation during fit/predict

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
  >>> model = dc.models.TensorflowMultiTaskFitTransformRegressor(n_tasks, [n_features, n_features],
  ...     dropouts=[0.], learning_rate=0.003, weight_init_stddevs=[np.sqrt(6)/np.sqrt(1000)],
  ...     batch_size=n_samples, fit_transformers=fit_transformers, n_evals=1)
  n_features after fit_transform: 12
  """

  def __init__(self,
               n_tasks,
               n_features,
               logdir=None,
               layer_sizes=[1000],
               weight_init_stddevs=[.02],
               bias_init_consts=[1.],
               penalty=0.0,
               penalty_type="l2",
               dropouts=[0.5],
               learning_rate=0.002,
               momentum=.8,
               optimizer="adam",
               batch_size=50,
               fit_transformers=[],
               n_evals=1,
               verbose=True,
               seed=None,
               **kwargs):
    """Initialize TensorflowMultiTaskFitTransformRegressor

    Parameters
    ----------
    n_tasks: int
      Number of tasks
    n_features: list or int
      Number of features.
    logdir: str
      Location to save data
    layer_sizes: list
      List of layer sizes.
    weight_init_stddevs: list
      List of standard deviations for weights (sampled from zero-mean
      gaussians). One for each layer.
    bias_init_consts: list
      List of bias initializations. One for each layer.
    penalty: float
      Amount of penalty (l2 or l1 applied)
    penalty_type: str
      Either "l2" or "l1"
    dropouts: list
      List of dropout amounts. One for each layer.
    learning_rate: float
      Learning rate for model.
    momentum: float
      Momentum. Only applied if optimizer=="momentum"
    optimizer: str
      Type of optimizer applied.
    batch_size: int
      Size of minibatches for training.
    fit_transformers: list
      List of dc.trans.FitTransformer objects
    n_evals: int
      Number of evalations per example at predict time
    verbose: True
      Perform logging.
    seed: int
      If not none, is used as random seed for tensorflow.

    """
    warnings.warn("TensorflowMultiTaskFitTransformRegressor "
                  "is deprecated. "
                  "Will be removed in DeepChem 1.4.", DeprecationWarning)

    self.fit_transformers = fit_transformers
    self.n_evals = n_evals

    # Run fit transformers on dummy dataset to determine n_features after transformation
    if isinstance(n_features, list):
      X_b = np.ones([batch_size] + n_features)
    elif isinstance(n_features, int):
      X_b = np.ones([batch_size, n_features])
    else:
      raise ValueError("n_features should be list or int")

    for transformer in self.fit_transformers:
      X_b = transformer.X_transform(X_b)
    n_features = X_b.shape[1]
    print("n_features after fit_transform: %d" % int(n_features))

    TensorflowGraphModel.__init__(
        self,
        n_tasks,
        n_features,
        logdir=logdir,
        layer_sizes=layer_sizes,
        weight_init_stddevs=weight_init_stddevs,
        bias_init_consts=bias_init_consts,
        penalty=penalty,
        penalty_type=penalty_type,
        dropouts=dropouts,
        learning_rate=learning_rate,
        momentum=momentum,
        optimizer=optimizer,
        batch_size=batch_size,
        pad_batches=False,
        verbose=verbose,
        seed=seed,
        **kwargs)

  def fit(self,
          dataset,
          nb_epoch=10,
          max_checkpoints_to_keep=5,
          log_every_N_batches=50,
          checkpoint_interval=10,
          **kwargs):
    """Perform fit transformations on each minibatch. Fit the model.

    Parameters
    ----------
    dataset: dc.data.Dataset
      Dataset object holding training data
    nb_epoch: 10
      Number of training epochs.
    max_checkpoints_to_keep: int
      Maximum number of checkpoints to keep; older checkpoints will be deleted.
    log_every_N_batches: int
      Report every N batches. Useful for training on very large datasets,
      where epochs can take long time to finish.
    checkpoint_interval: int
      Frequency at which to write checkpoints, measured in epochs

    Raises
    ------
    AssertionError
      If model is not in training mode.
    """
    ############################################################## TIMING
    time1 = time.time()
    ############################################################## TIMING
    log("Training for %d epochs" % nb_epoch, self.verbose)
    with self.train_graph.graph.as_default():
      train_op = self.get_training_op(self.train_graph.graph,
                                      self.train_graph.loss)
      with self._get_shared_session(train=True) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=max_checkpoints_to_keep)
        # Save an initial checkpoint.
        saver.save(sess, self._save_path, global_step=0)

        # Define the code that runs on a separate thread to feed data into the queue.
        def enqueue(sess, dataset, nb_epoch, epoch_end_indices):
          index = 0
          for epoch in range(nb_epoch):
            for X_b, y_b, w_b, ids_b in dataset.iterbatches(
                self.batch_size, pad_batches=self.pad_batches):
              for transformer in self.fit_transformers:
                X_b = transformer.X_transform(X_b)
              feed_dict = self.construct_feed_dict(X_b, y_b, w_b, ids_b)
              sess.run(self.train_graph.graph.enqueue, feed_dict=feed_dict)
              index += 1
            epoch_end_indices.append(index)
          sess.run(self.train_graph.graph.queue.close())

        epoch_end_indices = []
        enqueue_thread = threading.Thread(
            target=enqueue, args=[sess, dataset, nb_epoch, epoch_end_indices])
        enqueue_thread.daemon = True
        enqueue_thread.start()

        # Main training loop.
        try:
          epoch = 0
          index = 0
          index_in_epoch = 0
          avg_loss = 0.0
          while True:
            if index_in_epoch % log_every_N_batches == 0:
              log("On batch %d" % index_in_epoch, self.verbose)
            # Run training op.
            fetches = self.train_graph.output + [
                train_op, self.train_graph.loss
            ]
            fetched_values = sess.run(fetches)
            loss = fetched_values[-1]
            avg_loss += loss
            index += 1
            index_in_epoch += 1
            if len(epoch_end_indices) > 0 and index >= epoch_end_indices[0]:
              # We have reached the end of an epoch.
              if epoch % checkpoint_interval == checkpoint_interval - 1:
                saver.save(sess, self._save_path, global_step=epoch)
              avg_loss = float(avg_loss) / index_in_epoch
              log('Ending epoch %d: Average loss %g' % (epoch, avg_loss),
                  self.verbose)
              epoch += 1
              index_in_epoch = 0
              avg_loss = 0.0
              del epoch_end_indices[0]
        except tf.errors.OutOfRangeError:
          # We have reached the end of the data.
          pass
        # Always save a final checkpoint when complete.
        saver.save(sess, self._save_path, global_step=epoch + 1)
    ############################################################## TIMING
    time2 = time.time()
    print("TIMING: model fitting took %0.3f s" % (time2 - time1), self.verbose)
    ############################################################## TIMING

  def predict_on_batch(self, X):
    """Return model output for the provided input. Each example is evaluated
        self.n_evals times.

    Restore(checkpoint) must have previously been called on this object.

    Args:
      dataset: dc.data.Dataset object.

    Returns:
      Tuple of three numpy arrays with shape n_examples x n_tasks (x ...):
        output: Model outputs.
        labels: True labels.
        weights: Example weights.
      Note that the output and labels arrays may be more than 2D, e.g. for
      classifier models that return class probabilities.

    Raises:
      AssertionError: If model is not in evaluation mode.
      ValueError: If output and labels are not both 3D or both 2D.
    """
    X_evals = []
    for i in range(self.n_evals):
      X_t = X
      for transformer in self.fit_transformers:
        X_t = transformer.X_transform(X_t)
      X_evals.append(X_t)
    len_unpadded = len(X_t)
    if self.pad_batches:
      for i in range(self.n_evals):
        X_evals[i] = pad_features(self.batch_size, X_evals[i])
    if not self._restored_model:
      self.restore()
    with self.eval_graph.graph.as_default():

      # run eval data through the model
      n_tasks = self.n_tasks
      outputs = []
      with self._get_shared_session(train=False).as_default():

        n_samples = len(X_evals[0])
        for i in range(self.n_evals):

          output = []
          feed_dict = self.construct_feed_dict(X_evals[i])
          data = self._get_shared_session(train=False).run(
              self.eval_graph.output, feed_dict=feed_dict)
          batch_outputs = np.asarray(data[:n_tasks], dtype=float)
          # reshape to batch_size x n_tasks x ...
          if batch_outputs.ndim == 3:
            batch_outputs = batch_outputs.transpose((1, 0, 2))
          elif batch_outputs.ndim == 2:
            batch_outputs = batch_outputs.transpose((1, 0))
          # Handle edge case when batch-size is 1.
          elif batch_outputs.ndim == 1:
            n_samples = len(X)
            batch_outputs = batch_outputs.reshape((n_samples, n_tasks))
          else:
            raise ValueError('Unrecognized rank combination for output: %s' %
                             (batch_outputs.shape))
          # Prune away any padding that was added
          batch_outputs = batch_outputs[:n_samples]
          output.append(batch_outputs)

          outputs.append(np.squeeze(np.concatenate(output)))

    outputs = np.mean(np.array(outputs), axis=0)
    outputs = np.copy(outputs)

    # Handle case of 0-dimensional scalar output
    if len(outputs.shape) > 0:
      return outputs[:len_unpadded]
    else:
      outputs = np.reshape(outputs, (1,))
      return outputs

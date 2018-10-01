"""TensorFlow implementation of fully connected networks.
"""
from __future__ import division
from __future__ import unicode_literals

import logging
import warnings
import time
import numpy as np
import tensorflow as tf
import threading
import collections

import deepchem as dc
from deepchem.models.tensorgraph import model_ops
from deepchem.utils.save import log
from deepchem.metrics import to_one_hot, from_one_hot
from deepchem.metrics import to_one_hot

from deepchem.models.tensorgraph.tensor_graph import TensorGraph, TFWrapper
from deepchem.models.tensorgraph.layers import Feature, Label, Weights, WeightedError, Dense, Dropout, WeightDecay, Reshape, SoftMax, SoftMaxCrossEntropy, L2Loss, ReduceSum, ReduceMean, Exp

logger = logging.getLogger(__name__)


class MultitaskClassifier(TensorGraph):

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
    bias_init_consts: list or loat
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
    """
    super(MultitaskClassifier, self).__init__(**kwargs)
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
    self.neural_fingerprint = prev_layer

    logits = Reshape(
        shape=(-1, n_tasks, n_classes),
        in_layers=[
            Dense(in_layers=[prev_layer], out_channels=n_tasks * n_classes)
        ])
    output = SoftMax(logits)
    self.add_output(output)
    labels = Label(shape=(None, n_tasks, n_classes))
    weights = Weights(shape=(None, n_tasks))
    loss = SoftMaxCrossEntropy(in_layers=[labels, logits])
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


class MultitaskRegressor(TensorGraph):

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
               uncertainty=False,
               **kwargs):
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
    """
    super(MultitaskRegressor, self).__init__(**kwargs)
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
    if uncertainty:
      if any(d == 0.0 for d in dropouts):
        raise ValueError(
            'Dropout must be included in every layer to predict uncertainty')

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
    self.neural_fingerprint = prev_layer

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
    weights = Weights(shape=(None, n_tasks, 1))
    if uncertainty:
      log_var = Reshape(
          shape=(-1, n_tasks, 1),
          in_layers=[
              Dense(
                  in_layers=[prev_layer],
                  out_channels=n_tasks,
                  weights_initializer=TFWrapper(
                      tf.truncated_normal_initializer,
                      stddev=weight_init_stddevs[-1]),
                  biases_initializer=TFWrapper(
                      tf.constant_initializer, value=0.0))
          ])
      var = Exp(log_var)
      self.add_variance(var)
      diff = labels - output
      weighted_loss = weights * (diff * diff / var + log_var)
      weighted_loss = ReduceSum(ReduceMean(weighted_loss, axis=[1, 2]))
    else:
      weighted_loss = ReduceSum(L2Loss(in_layers=[labels, output, weights]))
    if weight_decay_penalty != 0.0:
      weighted_loss = WeightDecay(
          weight_decay_penalty,
          weight_decay_penalty_type,
          in_layers=[weighted_loss])
    self.set_loss(weighted_loss)


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
  ...     batch_size=n_samples, fit_transformers=fit_transformers, n_evals=1)
  >>> model.n_features
  12
  """

  def __init__(self,
               n_tasks,
               n_features,
               fit_transformers=[],
               n_evals=1,
               batch_size=50,
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
    logger.info("n_features after fit_transform: %d", int(n_features))
    super(MultitaskFitTransformRegressor, self).__init__(
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

    return super(MultitaskFitTransformRegressor, self).predict_on_generator(
        transform_generator(), transformers, outputs)

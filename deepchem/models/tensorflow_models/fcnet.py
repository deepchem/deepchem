"""TensorFlow implementation of the models from the ICML-2015 paper.


hyperparam_dict = {
    "single": Hyperparams(num_layers=1,
                          num_hidden=1200,
                          node_depth=1,
                          nonlinearity=ACTIVATION_RECTIFIED_LINEAR,
                          weight_init=GaussianWeightInit(0.01),
                          bias_init=ConstantBiasInit(0.5),
                          dropout=1.),
    "deep": Hyperparams(num_layers=4,
                        num_hidden=1000,
                        node_depth=1,
                        nonlinearity=ACTIVATION_RECTIFIED_LINEAR,
                        weight_init=GaussianWeightInit(0.01),
                        bias_init=ConstantBiasInit(0.5),
                        dropout=1.),
    "deepaux": Hyperparams(num_layers=4,
                        num_hidden=1000,
                        auxiliary_softmax_layers=[0, 1, 2],
                        auxiliary_softmax_weight=0.3,
                        node_depth=1,
                        nonlinearity=ACTIVATION_RECTIFIED_LINEAR,
                        weight_init=GaussianWeightInit(0.01),
                        bias_init=ConstantBiasInit(0.5),
                        dropout=1.),
    "py": Hyperparams(num_layers=2,
                      num_hidden=[2000, 100],
                      node_depth=1,
                      nonlinearity=ACTIVATION_RECTIFIED_LINEAR,
                      weight_init=[GaussianWeightInit(0.01),
                                   GaussianWeightInit(0.04)],
                      bias_init=[ConstantBiasInit(0.5),
                                 ConstantBiasInit(3.0)],
                      dropout=1.),
    "pydrop1": Hyperparams(num_layers=2,
                           num_hidden=[2000, 100],
                           node_depth=1,
                           nonlinearity=ACTIVATION_RECTIFIED_LINEAR,
                           weight_init=[GaussianWeightInit(0.01),
                                        GaussianWeightInit(0.04)],
                           bias_init=[ConstantBiasInit(0.5),
                                      ConstantBiasInit(3.0)],
                           dropout=[0.75, 1.]),
    "pydrop2": Hyperparams(num_layers=2,
                           num_hidden=[2000, 100],
                           node_depth=1,
                           nonlinearity=ACTIVATION_RECTIFIED_LINEAR,
                           weight_init=[GaussianWeightInit(0.01),
                                        GaussianWeightInit(0.04)],
                           bias_init=[ConstantBiasInit(0.5),
                                      ConstantBiasInit(3.0)],
                           dropout=[0.75, 0.75])}
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import time
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import logging

from deepchem.metrics import from_one_hot
from deepchem.models.tensorflow_models import TensorflowClassifier
from deepchem.models.tensorflow_models import TensorflowRegressor
from deepchem.models.tensorflow_models import model_ops
from deepchem.metrics import to_one_hot

def softmax(x):
    """Simple numpy softmax implementation
    """
    # (n_samples, n_classes)
    if len(x.shape) == 2:
      row_max = np.max(x, axis = 1)
      x -= row_max.reshape((x.shape[0], 1))
      x = np.exp(x)
      row_sum = np.sum(x, axis = 1)
      x /= row_sum.reshape((x.shape[0], 1))
    # (n_samples, n_tasks, n_classes)
    elif len(x.shape) == 3:
      row_max = np.max(x, axis = 2)
      x -= row_max.reshape(x.shape[:2] + (1,))
      x = np.exp(x)
      row_sum = np.sum(x, axis = 2)
      x /= row_sum.reshape(x.shape[:2] + (1,))
    return x

class TensorflowMultiTaskClassifier(TensorflowClassifier):
  """Implements an icml model as configured in a model_config.proto."""

  def build(self):
    """Constructs the graph architecture as specified in its config.

    This method creates the following Placeholders:
      mol_features: Molecule descriptor (e.g. fingerprint) tensor with shape
        batch_size x num_features.
    """
    assert len(self.model_params["data_shape"]) == 1
    num_features = self.model_params["data_shape"][0]
    with self.graph.as_default():
      with tf.name_scope(self.placeholder_scope):
        self.mol_features = tf.placeholder(
            tf.float32,
            shape=[self.model_params["batch_size"],
                   num_features],
            name='mol_features')

      layer_sizes = self.model_params["layer_sizes"]
      weight_init_stddevs = self.model_params["weight_init_stddevs"]
      bias_init_consts = self.model_params["bias_init_consts"]
      dropouts = self.model_params["dropouts"]
      lengths_set = {
          len(layer_sizes),
          len(weight_init_stddevs),
          len(bias_init_consts),
          len(dropouts),
          }
      assert len(lengths_set) == 1, 'All layer params must have same length.'
      num_layers = lengths_set.pop()
      assert num_layers > 0, 'Must have some layers defined.'

      prev_layer = self.mol_features
      prev_layer_size = num_features 
      for i in xrange(num_layers):
        layer = tf.nn.relu(model_ops.FullyConnectedLayer(
            tensor=prev_layer,
            size=layer_sizes[i],
            weight_init=tf.truncated_normal(
                shape=[prev_layer_size, layer_sizes[i]],
                stddev=weight_init_stddevs[i]),
            bias_init=tf.constant(value=bias_init_consts[i],
                                  shape=[layer_sizes[i]])))
        layer = model_ops.Dropout(layer, dropouts[i])
        #layer = tf.nn.dropout(layer, dropouts[i])
        prev_layer = layer
        prev_layer_size = layer_sizes[i]

      self.output = model_ops.MultitaskLogits(
          layer, self.model_params["num_classification_tasks"])

  def construct_feed_dict(self, X_b, y_b=None, w_b=None, ids_b=None):
    """Construct a feed dictionary from minibatch data.

    TODO(rbharath): ids_b is not used here. Can we remove it?

    Args:
      X_b: np.ndarray of shape (batch_size, num_features)
      y_b: np.ndarray of shape (batch_size, num_tasks)
      w_b: np.ndarray of shape (batch_size, num_tasks)
      ids_b: List of length (batch_size) with datapoint identifiers.
    """ 
    orig_dict = {}
    orig_dict["mol_features"] = X_b
    for task in xrange(self.num_tasks):
      if y_b is not None:
        orig_dict["labels_%d" % task] = to_one_hot(y_b[:, task])
      else:
        # Dummy placeholders
        orig_dict["labels_%d" % task] = np.squeeze(to_one_hot(
            np.zeros((self.model_params["batch_size"],))))
      if w_b is not None:
        orig_dict["weights_%d" % task] = w_b[:, task]
      else:
        # Dummy placeholders
        orig_dict["weights_%d" % task] = np.ones(
            (self.model_params["batch_size"],)) 
    return self._get_feed_dict(orig_dict)

  def predict_proba_on_batch(self, X):
    """Return model output for the provided input.

    Restore(checkpoint) must have previously been called on this object.

    Args:
      dataset: deepchem.datasets.dataset object.

    Returns:
      Tuple of three numpy arrays with shape num_examples x num_tasks (x ...):
        output: Model outputs.
      Note that the output arrays may be more than 2D, e.g. for
      classifier models that return class probabilities.

    Raises:
      AssertionError: If model is not in evaluation mode.
      ValueError: If output and labels are not both 3D or both 2D.
    """
    ######### DEBUG
    if not self._restored_model:
      self.restore()
    ######### DEBUG
    with self.graph.as_default():
      ########## DEBUG
      assert not model_ops.is_training()
      ########## DEBUG
      self.require_attributes(['output'])

      # run eval data through the model
      num_tasks = self.num_tasks
      outputs = []
      with self._get_shared_session().as_default():
        feed_dict = self.construct_feed_dict(X)
        data = self._get_shared_session().run(
            self.output, feed_dict=feed_dict)
        batch_outputs = np.asarray(data[:num_tasks], dtype=float)
        # reshape to batch_size x num_tasks x ...
        if batch_outputs.ndim == 3:
          batch_outputs = batch_outputs.transpose((1, 0, 2))
        elif batch_outputs.ndim == 2:
          batch_outputs = batch_outputs.transpose((1, 0))
        else:
          raise ValueError(
              'Unrecognized rank combination for output: %s ' %
              (batch_outputs.shape,))
        outputs.append(batch_outputs)

        # We apply softmax to predictions to get class probabilities.
        outputs = softmax(np.squeeze(np.hstack(outputs)))

    return np.copy(outputs)

class TensorflowMultiTaskRegressor(TensorflowRegressor):
  """Implements an icml model as configured in a model_config.proto."""

  def build(self):
    """Constructs the graph architecture as specified in its config.

    This method creates the following Placeholders:
      mol_features: Molecule descriptor (e.g. fingerprint) tensor with shape
        batch_size x num_features.
    """
    assert len(self.model_params["data_shape"]) == 1
    num_features = self.model_params["data_shape"][0]
    with self.graph.as_default():
      with tf.name_scope(self.placeholder_scope):
        self.mol_features = tf.placeholder(
            tf.float32,
            shape=[self.model_params["batch_size"],
                   num_features],
            name='mol_features')

      layer_sizes = self.model_params["layer_sizes"]
      weight_init_stddevs = self.model_params["weight_init_stddevs"]
      bias_init_consts = self.model_params["bias_init_consts"]
      dropouts = self.model_params["dropouts"]
      lengths_set = {
          len(layer_sizes),
          len(weight_init_stddevs),
          len(bias_init_consts),
          len(dropouts),
          }
      assert len(lengths_set) == 1, 'All layer params must have same length.'
      num_layers = lengths_set.pop()
      assert num_layers > 0, 'Must have some layers defined.'

      prev_layer = self.mol_features
      prev_layer_size = num_features 
      for i in xrange(num_layers):
        layer = tf.nn.relu(model_ops.FullyConnectedLayer(
            tensor=prev_layer,
            size=layer_sizes[i],
            weight_init=tf.truncated_normal(
                shape=[prev_layer_size, layer_sizes[i]],
                stddev=weight_init_stddevs[i]),
            bias_init=tf.constant(value=bias_init_consts[i],
                                  shape=[layer_sizes[i]])))
        layer = model_ops.Dropout(layer, dropouts[i])
        #layer = tf.nn.dropout(layer, keep_prob)
        prev_layer = layer
        prev_layer_size = layer_sizes[i]

      self.output = []
      for task in range(self.num_tasks):
        self.output.append(tf.squeeze(
            model_ops.FullyConnectedLayer(
                tensor=prev_layer,
                size=layer_sizes[i],
                weight_init=tf.truncated_normal(
                    shape=[prev_layer_size, 1],
                    stddev=weight_init_stddevs[i]),
                bias_init=tf.constant(value=bias_init_consts[i],
                                      shape=[1]))))

  def construct_feed_dict(self, X_b, y_b=None, w_b=None, ids_b=None):
    """Construct a feed dictionary from minibatch data.

    TODO(rbharath): ids_b is not used here. Can we remove it?

    Args:
      X_b: np.ndarray of shape (batch_size, num_features)
      y_b: np.ndarray of shape (batch_size, num_tasks)
      w_b: np.ndarray of shape (batch_size, num_tasks)
      ids_b: List of length (batch_size) with datapoint identifiers.
    """ 
    orig_dict = {}
    orig_dict["mol_features"] = X_b
    for task in xrange(self.num_tasks):
      if y_b is not None:
        orig_dict["labels_%d" % task] = y_b[:, task]
      else:
        # Dummy placeholders
        orig_dict["labels_%d" % task] = np.squeeze(
            np.zeros((self.model_params["batch_size"],)))
      if w_b is not None:
        orig_dict["weights_%d" % task] = w_b[:, task]
      else:
        # Dummy placeholders
        orig_dict["weights_%d" % task] = np.ones(
            (self.model_params["batch_size"],)) 
    return self._get_feed_dict(orig_dict)

  def predict_on_batch(self, X):
    """Return model output for the provided input.

    Restore(checkpoint) must have previously been called on this object.

    Args:
      dataset: deepchem.datasets.dataset object.

    Returns:
      Tuple of three numpy arrays with shape num_examples x num_tasks (x ...):
        output: Model outputs.
        labels: True labels.
        weights: Example weights.
      Note that the output and labels arrays may be more than 2D, e.g. for
      classifier models that return class probabilities.

    Raises:
      AssertionError: If model is not in evaluation mode.
      ValueError: If output and labels are not both 3D or both 2D.
    """
    ########## DEBUG
    if not self._restored_model:
      self.restore()
    ########## DEBUG
    with self.graph.as_default():
      ########### DEBUG
      #assert not model_ops.is_training()
      ########### DEBUG
      self.require_attributes(['output'])

      # run eval data through the model
      num_tasks = self.num_tasks
      outputs = []
      with self._get_shared_session().as_default():
        feed_dict = self.construct_feed_dict(X)
        data = self._get_shared_session().run(
            self.output, feed_dict=feed_dict)
        batch_outputs = np.asarray(data[:num_tasks], dtype=float)
        # reshape to batch_size x num_tasks x ...
        if batch_outputs.ndim == 3:
          batch_outputs = batch_outputs.transpose((1, 0, 2))
        elif batch_outputs.ndim == 2:
          batch_outputs = batch_outputs.transpose((1, 0))
        else:
          raise ValueError(
              'Unrecognized rank combination for output: %s' %
              (batch_outputs.shape))
        outputs.append(batch_outputs)

        outputs = np.squeeze(np.concatenate(outputs)) 

    return np.copy(outputs)


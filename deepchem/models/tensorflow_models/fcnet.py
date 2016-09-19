"""TensorFlow implementation of fully connected networks. 
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import time
import numpy as np
import tensorflow as tf

from deepchem.metrics import from_one_hot
from deepchem.models.tensorflow_models import TensorflowClassifier
from deepchem.models.tensorflow_models import TensorflowRegressor
from deepchem.models.tensorflow_models import model_ops
from deepchem.metrics import to_one_hot
from deepchem.datasets import pad_features

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

  def build(self, graph, name_scopes):
    """Constructs the graph architecture as specified in its config.

    This method creates the following Placeholders:
      mol_features: Molecule descriptor (e.g. fingerprint) tensor with shape
        batch_size x n_features.
    """
    placeholder_scope = self._get_placeholder_scope(graph, name_scopes)
    n_features = self.n_features
    with graph.as_default():
      with tf.name_scope(placeholder_scope):
        self.mol_features = tf.placeholder(
            tf.float32,
            shape=[None, n_features],
            name='mol_features')

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

      prev_layer = self.mol_features
      prev_layer_size = n_features 
      for i in xrange(n_layers):
        layer = tf.nn.relu(model_ops.FullyConnectedLayer(
            tensor=prev_layer,
            size=layer_sizes[i],
            weight_init=tf.truncated_normal(
                shape=[prev_layer_size, layer_sizes[i]],
                stddev=weight_init_stddevs[i]),
            bias_init=tf.constant(value=bias_init_consts[i],
                                  shape=[layer_sizes[i]])))
        layer = model_ops.Dropout(layer, dropouts[i])
        prev_layer = layer
        prev_layer_size = layer_sizes[i]

      self.output = model_ops.MultitaskLogits(
          layer, self.n_tasks)

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
    for task in xrange(self.n_tasks):
      if y_b is not None:
        orig_dict["labels_%d" % task] = to_one_hot(y_b[:, task])
      else:
        # Dummy placeholders
        orig_dict["labels_%d" % task] = np.squeeze(to_one_hot(
            np.zeros((self.batch_size,))))
      if w_b is not None:
        orig_dict["weights_%d" % task] = w_b[:, task]
      else:
        # Dummy placeholders
        orig_dict["weights_%d" % task] = np.ones(
            (self.batch_size,)) 
    return self._get_feed_dict(orig_dict)

  def predict_proba_on_batch(self, X):
    """Return model output for the provided input.

    Restore(checkpoint) must have previously been called on this object.

    Args:
      dataset: deepchem.datasets.dataset object.

    Returns:
      Tuple of three numpy arrays with shape n_examples x n_tasks (x ...):
        output: Model outputs.
      Note that the output arrays may be more than 2D, e.g. for
      classifier models that return class probabilities.

    Raises:
      AssertionError: If model is not in evaluation mode.
      ValueError: If output and labels are not both 3D or both 2D.
    """
    if not self._restored_model:
      self.restore()
    with self.eval_graph.graph.as_default():
      assert not model_ops.is_training()
      self.require_attributes(['output'])

      # run eval data through the model
      n_tasks = self.n_tasks
      outputs = []
      with self._get_shared_session().as_default():
        feed_dict = self.construct_feed_dict(X)
        data = self._get_shared_session().run(
            self.output, feed_dict=feed_dict)
        batch_outputs = np.asarray(data[:n_tasks], dtype=float)
        # reshape to batch_size x n_tasks x ...
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

  def build(self, graph):
    """Constructs the graph architecture as specified in its config.

    This method creates the following Placeholders:
      mol_features: Molecule descriptor (e.g. fingerprint) tensor with shape
        batch_size x n_features.
    """
    n_features = self.n_inputs
    with graph.as_default():
      with tf.name_scope(self.placeholder_scope):
        self.mol_features = tf.placeholder(
            tf.float32,
            shape=[None, n_features],
            name='mol_features')

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

      prev_layer = self.mol_features
      prev_layer_size = n_features 
      for i in xrange(n_layers):
        layer = tf.nn.relu(model_ops.FullyConnectedLayer(
            tensor=prev_layer,
            size=layer_sizes[i],
            weight_init=tf.truncated_normal(
                shape=[prev_layer_size, layer_sizes[i]],
                stddev=weight_init_stddevs[i]),
            bias_init=tf.constant(value=bias_init_consts[i],
                                  shape=[layer_sizes[i]])))
        layer = model_ops.Dropout(layer, dropouts[i])
        prev_layer = layer
        prev_layer_size = layer_sizes[i]

      self.output = []
      for task in range(self.n_tasks):
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
      X_b: np.ndarray of shape (batch_size, n_features)
      y_b: np.ndarray of shape (batch_size, n_tasks)
      w_b: np.ndarray of shape (batch_size, n_tasks)
      ids_b: List of length (batch_size) with datapoint identifiers.
    """ 
    orig_dict = {}
    orig_dict["mol_features"] = X_b
    for task in xrange(self.n_tasks):
      if y_b is not None:
        orig_dict["labels_%d" % task] = y_b[:, task]
      else:
        # Dummy placeholders
        orig_dict["labels_%d" % task] = np.squeeze(
            np.zeros((self.batch_size,)))
      if w_b is not None:
        orig_dict["weights_%d" % task] = w_b[:, task]
      else:
        # Dummy placeholders
        orig_dict["weights_%d" % task] = np.ones(
            (self.batch_size,)) 
    return self._get_feed_dict(orig_dict)

  def predict_on_batch(self, X):
    """Return model output for the provided input.

    Restore(checkpoint) must have previously been called on this object.

    Args:
      dataset: deepchem.datasets.dataset object.

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
    if not self._restored_model:
      self.restore()
    with self.graph.as_default():
      assert not model_ops.is_training()
      self.require_attributes(['output'])

      # run eval data through the model
      n_tasks = self.n_tasks
      outputs = []
      with self._get_shared_session().as_default():
        n_samples = len(X)
        # Some tensorflow models can't handle variadic batches,
        # especially models using tf.pack, tf.split. Pad batch-size
        # to handle these cases.
        X = pad_features(self.batch_size, X)
        feed_dict = self.construct_feed_dict(X)
        data = self._get_shared_session().run(
            self.output, feed_dict=feed_dict)
        batch_outputs = np.asarray(data[:n_tasks], dtype=float)
        # reshape to batch_size x n_tasks x ...
        if batch_outputs.ndim == 3:
          batch_outputs = batch_outputs.transpose((1, 0, 2))
        elif batch_outputs.ndim == 2:
          batch_outputs = batch_outputs.transpose((1, 0))
        # Handle edge case when batch-size is 1.
        elif batch_outputs.ndim == 1:
          #print("X.shape, batch_outputs.shape")
          #print(X.shape, batch_outputs.shape)
          n_samples = len(X)
          batch_outputs = batch_outputs.reshape((n_samples, n_tasks))
        else:
          raise ValueError(
              'Unrecognized rank combination for output: %s' %
              (batch_outputs.shape))
        # Prune away any padding that was added
        batch_outputs = batch_outputs[:n_samples]
        outputs.append(batch_outputs)

        outputs = np.squeeze(np.concatenate(outputs)) 

    return np.copy(outputs)


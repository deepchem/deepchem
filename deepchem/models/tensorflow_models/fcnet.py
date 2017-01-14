"""TensorFlow implementation of fully connected networks. 
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import time
import numpy as np
import tensorflow as tf

from deepchem.nn import model_ops
from deepchem.metrics import from_one_hot
from deepchem.models.tensorflow_models import TensorflowGraph
from deepchem.models.tensorflow_models import TensorflowClassifier
from deepchem.models.tensorflow_models import TensorflowRegressor
from deepchem.metrics import to_one_hot

class TensorflowMultiTaskClassifier(TensorflowClassifier):
  """Implements an icml model as configured in a model_config.proto."""

  def build(self, graph, name_scopes, training):
    """Constructs the graph architecture as specified in its config.

    This method creates the following Placeholders:
      mol_features: Molecule descriptor (e.g. fingerprint) tensor with shape
        batch_size x n_features.
    """
    placeholder_scope = TensorflowGraph.get_placeholder_scope(
        graph, name_scopes)
    n_features = self.n_features
    with graph.as_default():
      with placeholder_scope:
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
      for i in range(n_layers):
        layer = tf.nn.relu(model_ops.fully_connected_layer(
            tensor=prev_layer,
            size=layer_sizes[i],
            weight_init=tf.truncated_normal(
                shape=[prev_layer_size, layer_sizes[i]],
                stddev=weight_init_stddevs[i]),
            bias_init=tf.constant(value=bias_init_consts[i],
                                  shape=[layer_sizes[i]])))
        layer = model_ops.dropout(layer, dropouts[i], training)
        prev_layer = layer
        prev_layer_size = layer_sizes[i]

      output = model_ops.multitask_logits(
          layer, self.n_tasks)
    return output

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
        orig_dict["labels_%d" % task] = np.squeeze(to_one_hot(
            np.zeros((self.batch_size,))))
      if w_b is not None:
        orig_dict["weights_%d" % task] = w_b[:, task]
      else:
        # Dummy placeholders
        orig_dict["weights_%d" % task] = np.ones(
            (self.batch_size,)) 
    return TensorflowGraph.get_feed_dict(orig_dict)


class TensorflowMultiTaskRegressor(TensorflowRegressor):
  """Implements an icml model as configured in a model_config.proto."""

  def build(self, graph, name_scopes, training):
    """Constructs the graph architecture as specified in its config.

    This method creates the following Placeholders:
      mol_features: Molecule descriptor (e.g. fingerprint) tensor with shape
        batch_size x n_features.
    """
    n_features = self.n_features
    placeholder_scope = TensorflowGraph.get_placeholder_scope(
        graph, name_scopes)
    with graph.as_default():
      with placeholder_scope:
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
      for i in range(n_layers):
        layer = tf.nn.relu(model_ops.fully_connected_layer(
            tensor=prev_layer,
            size=layer_sizes[i],
            weight_init=tf.truncated_normal(
                shape=[prev_layer_size, layer_sizes[i]],
                stddev=weight_init_stddevs[i]),
            bias_init=tf.constant(value=bias_init_consts[i],
                                  shape=[layer_sizes[i]])))
        layer = model_ops.dropout(layer, dropouts[i], training)
        prev_layer = layer
        prev_layer_size = layer_sizes[i]

      output = []
      for task in range(self.n_tasks):
        output.append(tf.squeeze(
            model_ops.fully_connected_layer(
                tensor=prev_layer,
                size=layer_sizes[i],
                weight_init=tf.truncated_normal(
                    shape=[prev_layer_size, 1],
                    stddev=weight_init_stddevs[i]),
                bias_init=tf.constant(value=bias_init_consts[i],
                                      shape=[1]))))
      return output

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
        orig_dict["labels_%d" % task] = np.squeeze(
            np.zeros((self.batch_size,)))
      if w_b is not None:
        orig_dict["weights_%d" % task] = w_b[:, task]
      else:
        # Dummy placeholders
        orig_dict["weights_%d" % task] = np.ones(
            (self.batch_size,)) 
    return TensorflowGraph.get_feed_dict(orig_dict)

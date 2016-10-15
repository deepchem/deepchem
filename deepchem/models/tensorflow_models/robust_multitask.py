from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf

from deepchem.models.tensorflow_models.fcnet import TensorflowMultiTaskRegressor
from deepchem.models.tensorflow_models import model_ops


class RobustMultitaskRegressor(TensorflowMultiTaskRegressor):
  """Implements a neural network for robust multitasking.
  
  Key idea is to have bypass layers that feed directly from features to task
  output. Hopefully will allow tasks to route around bad multitasking.
  """

  def build(self):
    """Constructs the graph architecture as specified in its config.

    This method creates the following Placeholders:
      mol_features: Molecule descriptor (e.g. fingerprint) tensor with shape
        batch_size x num_features.
    """
    ############################################################### DEBUG
    print("ENTERING BUILD!")
    ############################################################### DEBUG
    assert len(self.model_params["data_shape"]) == 1
    num_features = self.model_params["data_shape"][0]
    with self.graph.as_default():
      with tf.name_scope(self.placeholder_scope):
        self.mol_features = tf.placeholder(
            tf.float32,
            shape=[None, num_features],
            name='mol_features')

      layer_sizes = self.model_params["layer_sizes"]
      weight_init_stddevs = self.model_params["weight_init_stddevs"]
      bias_init_consts = self.model_params["bias_init_consts"]
      dropouts = self.model_params["dropouts"]

      bypass_layer_sizes = self.model_params["bypass_layer_sizes"]
      bypass_weight_init_stddevs = self.model_params["bypass_weight_init_stddevs"]
      bypass_bias_init_consts = self.model_params["bypass_bias_init_consts"]
      bypass_dropouts = self.model_params["bypass_dropouts"]

      lengths_set = {
          len(layer_sizes),
          len(weight_init_stddevs),
          len(bias_init_consts),
          len(dropouts),
          }
      assert len(lengths_set) == 1, 'All layer params must have same length.'
      num_layers = lengths_set.pop()
      assert num_layers > 0, 'Must have some layers defined.'

      bypass_lengths_set = {
          len(bypass_layer_sizes),
          len(bypass_weight_init_stddevs),
          len(bypass_bias_init_consts),
          len(bypass_dropouts),
          }
      assert len(bypass_lengths_set) == 1, 'All bypass_layer params must have same length.'
      num_bypass_layers = bypass_lengths_set.pop()

      prev_layer = self.mol_features
      prev_layer_size = num_features 
      for i in xrange(num_layers):
        # layer has shape [None, layer_sizes[i]]
        ########################################################## DEBUG
        print("Adding weights of shape %s" % str([prev_layer_size, layer_sizes[i]]))
        ########################################################## DEBUG
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
      # top_multitask_layer has shape [None, layer_sizes[-1]]
      top_multitask_layer = prev_layer
      for task in range(self.num_tasks):
        # TODO(rbharath): Might want to make it feasible to have multiple
        # bypass layers.
        # Construct task bypass layer
        prev_bypass_layer = self.mol_features
        prev_bypass_layer_size = num_features
        for i in xrange(num_bypass_layers):
          # bypass_layer has shape [None, bypass_layer_sizes[i]]
          ########################################################## DEBUG
          print("Adding bypass weights of shape %s"
                % str([prev_bypass_layer_size, bypass_layer_sizes[i]]))
          ########################################################## DEBUG
          bypass_layer = tf.nn.relu(model_ops.FullyConnectedLayer(
            tensor = prev_bypass_layer,
            size = bypass_layer_sizes[i],
            weight_init=tf.truncated_normal(
                shape=[prev_bypass_layer_size, bypass_layer_sizes[i]],
                stddev=bypass_weight_init_stddevs[i]),
            bias_init=tf.constant(value=bypass_bias_init_consts[i],
                                  shape=[bypass_layer_sizes[i]])))
    
          bypass_layer = model_ops.Dropout(bypass_layer, bypass_dropouts[i])
          prev_bypass_layer = bypass_layer
          prev_bypass_layer_size = bypass_layer_sizes[i]
        top_bypass_layer = prev_bypass_layer

        if num_bypass_layers > 0:
          # task_layer has shape [None, layer_sizes[-1] + bypass_layer_sizes[-1]]
          task_layer = tf.concat(1, [top_multitask_layer, top_bypass_layer])
          task_layer_size = layer_sizes[-1] + bypass_layer_sizes[-1]
        else:
          task_layer = top_multitask_layer
          task_layer_size = layer_sizes[-1]
        ########################################################## DEBUG
        print("Adding output weights of shape %s"
              % str([task_layer_size, 1]))
        ########################################################## DEBUG
        #################################################### DEBUG
        print("task_layer_size")
        print(task_layer_size)
        #################################################### DEBUG
        self.output.append(tf.squeeze(
            model_ops.FullyConnectedLayer(
                tensor=task_layer,
                size=task_layer_size,
                weight_init=tf.truncated_normal(
                    shape=[task_layer_size, 1],
                    stddev=weight_init_stddevs[-1]),
                bias_init=tf.constant(value=bias_init_consts[-1],
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


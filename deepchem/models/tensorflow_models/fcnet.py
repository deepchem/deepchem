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

#!/usr/bin/python
#
# Copyright 2015 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import logging

from deepchem.models.tensorflow_models import TensorflowClassifier
from deepchem.models.tensorflow_models import model_ops
from deepchem.utils.evaluate import to_one_hot

class TensorflowMultiTaskClassifier(TensorflowClassifier):
  """Implements an icml model as configured in a model_config.proto."""

  def build(self):
    """Constructs the graph architecture as specified in its config.

    This method creates the following Placeholders:
      mol_features: Molecule descriptor (e.g. fingerprint) tensor with shape
        batch_size x num_features.
    """
    with self.graph.as_default():
      with tf.name_scope(self.placeholder_scope):
        self.mol_features = tf.placeholder(
            tf.float32,
            shape=[self.model_params["batch_size"],
                   self.model_params["num_features"]],
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
      prev_layer_size = self.model_params["num_features"]
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
        prev_layer = layer
        prev_layer_size = layer_sizes[i]

      self.output = model_ops.MultitaskLogits(
          layer, self.model_params["num_classification_tasks"])

  # TODO(rbharath): Copying this out for now. Ensure this isn't harmful
  #def add_labels_and_weights(self):
  #  """Parse Label protos and create tensors for labels and weights.

  #  This method creates the following Placeholders in the graph:
  #    labels: Tensor with shape batch_size x num_tasks containing serialized
  #      Label protos.
  #  """
  #  config = self.config
  #  with tf.name_scope(self.placeholder_scope):
  #    labels = tf.placeholder(
  #        tf.string,
  #        shape=[config.batch_size, config.num_classification_tasks],
  #        name='labels')
  #  self.labels = label_ops.MultitaskLabelClasses(labels, config.num_classes)
  #  self.weights = label_ops.MultitaskLabelWeights(labels)

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
    orig_dict["valid"] = np.ones((self.model_params["batch_size"],), dtype=bool)
    return self._get_feed_dict(orig_dict)

  # TODO(rbharath): This explicit manipulation of scopes is ugly. Is there a
  # better design here?
  def _get_feed_dict(self, named_values):
    feed_dict = {}
    for name, value in named_values.iteritems():
      feed_dict['{}/{}:0'.format(self.placeholder_root, name)] = value
    return feed_dict

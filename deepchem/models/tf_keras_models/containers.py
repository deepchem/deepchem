"""Borrows Keras Container Abstraction for Models
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Han Altae-Tran and Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

from keras.engine.topology import Container

'''
class GraphContainer(Container):
  def __init__(self, sess, input, output, graph_topology, **kwargs):
    """
    Parameters
    ----------
    sess: tf.Session
      The tensorflow session for this Keras model.
    input: tf.Tensor
      Input tensor for container.
    output: tf.Tensor
      Output tensor for container.
    graph_topology: deepchem.models.tf_keras_models.graph_topology.GraphTopology
      Manager for topology placeholders.
    """
    self.graph_topology = graph_topology 
    kwargs["input"] = input
    kwargs["output"] = output

    super(GraphContainer, self).__init__(**kwargs)
    self.sess = sess  # Save Tensorflow session

    self.op_mode_objects = {}

  def get_graph_topology(self):
    return self.graph_topology

  def get_batch_size(self):
    return self.graph_topology.get_batch_size()

  def get_num_output_features(self):
    """Gets the output shape of the featurization layers of the network"""
    return self.layers[-1].output_shape[1]

  def get_output(self):
    return self.output
'''

class SupportGraphContainer(Container):
  def __init__(self, sess, **kwargs):
    # Remove the keywords for the GraphModel
    self.graph_topology_test = kwargs.pop("graph_topology_test")
    self.graph_topology_support = kwargs.pop("graph_topology_support")

    super(SupportGraphContainer, self).__init__(**kwargs)
    self.sess = sess  # Save Tensorflow session

    self.op_mode_objects = {}

  def get_test(self):
    return self.graph_topology_test.get_inputs()

  def get_support(self):
    return self.graph_topology_support.get_inputs()

  def get_test_batch_size(self):
    return self.graph_topology_test.get_batch_size()

  def get_support_batch_size(self):
    return self.graph_topology_support.get_batch_size()

  def get_featurization_dim(self):
    # Gets the output shape of the featurization layers of the network
    return self.layers[-1].output_shape[1]

  def get_test_output(self):
    return self.output[0]
  
  def get_support_output(self):
    return self.output[1]

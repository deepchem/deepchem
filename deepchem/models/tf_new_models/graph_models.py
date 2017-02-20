"""
Convenience classes for assembling graph models.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Han Altae-Tran and Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

import tensorflow as tf
from deepchem.nn.layers import GraphGather
from deepchem.models.tf_new_models.graph_topology import GraphTopology

class SequentialGraph(object):
  """An analog of Keras Sequential class for Graph data.

  Like the Sequential class from Keras, but automatically passes topology
  placeholders from GraphTopology to each graph layer (from layers) added
  to the network. Non graph layers don't get the extra placeholders. 
  """
  def __init__(self, n_feat):
    """
    Parameters
    ----------
    n_feat: int
      Number of features per atom.
    """
    self.graph = tf.Graph()
    config = tf.ConfigProto(allow_soft_placement=True)
    self.session = tf.Session(graph=self.graph, config=config)
    with self.graph.as_default():
      self.graph_topology = GraphTopology(n_feat)
      self.output = self.graph_topology.get_atom_features_placeholder()
    # Keep track of the layers
    self.layers = []  

  def add(self, layer):
    """Adds a new layer to model."""
    with self.graph.as_default():
      # For graphical layers, add connectivity placeholders 
      if type(layer).__name__ in ['GraphConv', 'GraphGather', 'GraphPool']:
        if (len(self.layers) > 0 and hasattr(self.layers[-1], "__name__")):
          assert self.layers[-1].__name__ != "GraphGather", \
                  'Cannot use GraphConv or GraphGather layers after a GraphGather'
            
        self.output = layer(
            [self.output] + self.graph_topology.get_topology_placeholders())
      else:
        self.output = layer(self.output)

      # Add layer to the layer list
      self.layers.append(layer)

  def get_graph_topology(self):
    return self.graph_topology

  def get_num_output_features(self):
    """Gets the output shape of the featurization layers of the network"""
    return self.layers[-1].output_shape[1]
  
  def return_outputs(self):
    return self.output

  def return_inputs(self):
    return self.graph_topology.get_input_placeholders()

  def get_layer(self, layer_id):
    return self.layers[layer_id]

class SequentialSupportGraph(object):
  """An analog of Keras Sequential model for test/support models."""
  def __init__(self, n_feat):
    """
    Parameters
    ----------
    n_feat: int
      Number of atomic features.
    """
    # Create graph topology and x
    self.test_graph_topology = GraphTopology(n_feat, name='test')
    self.support_graph_topology = GraphTopology(n_feat, name='support')
    self.test = self.test_graph_topology.get_atom_features_placeholder()
    self.support = self.support_graph_topology.get_atom_features_placeholder()

    # Keep track of the layers
    self.layers = []  
    # Whether or not we have used the GraphGather layer yet
    self.bool_pre_gather = True  

  def add(self, layer):
    """Adds a layer to both test/support stacks.

    Note that the layer transformation is performed independently on the
    test/support tensors.
    """
    self.layers.append(layer)

    # Update new value of x
    if type(layer).__name__ in ['GraphConv', 'GraphGather', 'GraphPool']:
      assert self.bool_pre_gather, "Cannot apply graphical layers after gather."
          
      self.test = layer([self.test] + self.test_graph_topology.topology)
      self.support = layer([self.support] + self.support_graph_topology.topology)
    else:
      self.test = layer(self.test)
      self.support = layer(self.support)

    if type(layer).__name__ == 'GraphGather':
      self.bool_pre_gather = False  # Set flag to stop adding topology

  def add_test(self, layer):
    """Adds a layer to test."""
    self.layers.append(layer)

    # Update new value of x
    if type(layer).__name__ in ['GraphConv', 'GraphPool', 'GraphGather']:
      self.test = layer([self.test] + self.test_graph_topology.topology)
    else:
      self.test = layer(self.test)

  def add_support(self, layer):
    """Adds a layer to support."""
    self.layers.append(layer)

    # Update new value of x
    if type(layer).__name__ in ['GraphConv', 'GraphPool', 'GraphGather']:
      self.support = layer([self.support] + self.support_graph_topology.topology)
    else:
      self.support = layer(self.support)

  def join(self, layer):
    """Joins test and support to a two input two output layer"""
    self.layers.append(layer)
    self.test, self.support = layer([self.test, self.support])

  def get_test_output(self):
    return self.test

  def get_support_output(self):
    return self.support
  
  def return_outputs(self):
    return [self.test] + [self.support]

  def return_inputs(self):
    return (self.test_graph_topology.get_inputs()
            + self.support_graph_topology.get_inputs())

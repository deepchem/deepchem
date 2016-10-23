"""
Convenience classes for assembling graph models.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Han Altae-Tran and Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"


from deepchem.models.tf_keras_models.keras_layers import GraphGather
'''
from deepchem.models.tf_keras_models.containers import GraphContainer
from deepchem.models.tf_keras_models.containers import SupportGraphContainer
'''
from deepchem.models.tf_keras_models.graph_topology import GraphTopology

class SequentialGraphModel(object):
  """An analog of Keras Sequential class for Graph data.

  Like the Sequential class from Keras, but automatically passes topology
  placeholders from GraphTopology to each graph layer (from keras_layers) added
  to the network. Non graph layers don't get the extra placeholders. 
  """
  def __init__(self, n_atoms, n_feat):
    """
    Parameters
    ----------
    n_atoms: int
      (Max?) Number of atoms in system.
    n_feat: int
      Number of features per atom.
    """
    
    #super(SequentialGraphModel, self).__init__()
    # Create graph topology and x
    self.graph_topology = GraphTopology(n_atoms, n_feat)
    self.output = self.graph_topology.get_atom_features_placeholder()
    # Keep track of the layers
    self.layers = []  

  def add(self, layer):
    """Adds a new layer to model."""
    # For graphical layers, add connectivity placeholders 
    if type(layer).__name__ in ['GraphConv', 'GraphGather', 'GraphPool']:
      if (len(self.layers) > 0 and hasattr(self.layers[-1], "__name__")):
        assert (self.layers[-1].__name__ != "GraphGather",
                'Cannot use GraphConv or GraphGather layers after a GraphGather')
          
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

class SequentialSupportGraphModel(object):
  """An analog of Keras Sequential model for test/support models."""
  def __init__(self, n_test, n_support, n_feat):
    """
    Parameters
    ----------
    n_test: int
      Number of test atoms.
    n_support: int
      Number of support atoms.
    n_feat: int
      Number of atomic features.
    """
    self.n_test = n_test
    self.n_support = n_support

    # Create graph topology and x
    self.test_graph_topology = GraphTopology(
        n_test, n_feat, name='test')
    self.support_graph_topology = GraphTopology(
        n_support, n_feat, name='support')
    self.test = self.test_graph_topology.get_atom_features_placeholder()
    self.support = self.support_graph_topology.get_atom_features_placeholder()

    # Keep track of the layers
    self.layers = []  
    # Whether or not we have used the GraphGather layer yet
    self.bool_pre_gather = True  

  def add(self, layer):
    # Add layer to the layer list
    self.layers.append(layer)

    # Update new value of x
    if type(layer).__name__ in ['GraphConv', 'GraphGather', 'GraphPool']:
      assert (self.bool_pre_gather,
              'Cannot use GraphConv or GraphGather layers after a GraphGather')
          
      self.test = layer([self.test] + self.test_graph_topology.topology)
      self.support = layer([self.support] + self.support_graph_topology.topology)
    else:
      self.test = layer(self.test)
      self.support = layer(self.support)

    if type(layer).__name__ == 'GraphGather':
      self.bool_pre_gather = False  # Set flag to stop adding topology

  def join(self, layer, swap=False):
    """Joins test and support to a two input two output layer"""
    self.layers.append(layer)
    if not swap:
        self.test, self.support = layer([self.test, self.support])
    else:
        self.support, self.test = layer([self.support, self.test])
  
  def graph_gather(self, activation='linear'):
    gather1 = GraphGather(self.test_batch_size, activation=activation)
    gather2 = GraphGather(self.support_batch_size, activation=activation)

    self.layers.append(gather1)
    self.layers.append(gather2)
    
    self.test = gather1([self.test] + self.test_graph_topology.topology)
    self.support = gather2([self.support] + self.support_graph_topology.topology)
    
    self.bool_pre_gather = False

  '''
  def return_container(self, sess):
    return SupportGraphContainer(
        sess, input=self.return_inputs(),
        output=self.return_outputs(), 
        graph_topology_test=self.test_graph_topology,
        graph_topology_support=self.support_graph_topology)
  '''
  
  def return_outputs(self):
    return [self.test] + [self.support]

  def return_inputs(self):
    return (self.test_graph_topology.get_inputs()
            + self.support_graph_topology.get_inputs())

  def get_layer(self, layer_id):
    return self.layers[layer_id]

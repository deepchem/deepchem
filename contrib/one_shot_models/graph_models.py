"""
Convenience classes for assembling graph models.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Han Altae-Tran and Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import warnings
import tensorflow as tf
from deepchem.nn.layers import GraphGather
from deepchem.models.tf_new_models.graph_topology import GraphTopology, DTNNGraphTopology, DAGGraphTopology, WeaveGraphTopology, AlternateWeaveGraphTopology


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
    warnings.warn("SequentialGraph is deprecated. "
                  "Will be removed in DeepChem 1.4.", DeprecationWarning)
    self.graph = tf.Graph()
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

        self.output = layer([self.output] +
                            self.graph_topology.get_topology_placeholders())
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


class SequentialDTNNGraph(SequentialGraph):
  """An analog of Keras Sequential class for Coulomb Matrix data.

  automatically generates and passes topology placeholders to each layer. 
  """

  def __init__(self, n_distance=100, distance_min=-1., distance_max=18.):
    """
    Parameters
    ----------
    n_distance: int, optional
      granularity of distance matrix
      step size will be (distance_max-distance_min)/n_distance
    distance_min: float, optional
      minimum distance of atom pairs, default = -1 Angstorm
    distance_max: float, optional
      maximum distance of atom pairs, default = 18 Angstorm
    """
    warnings.warn("SequentialDTNNGraph is deprecated. "
                  "Will be removed in DeepChem 1.4.", DeprecationWarning)
    self.graph = tf.Graph()
    with self.graph.as_default():
      self.graph_topology = DTNNGraphTopology(
          n_distance, distance_min=distance_min, distance_max=distance_max)
      self.output = self.graph_topology.get_atom_number_placeholder()
    # Keep track of the layers
    self.layers = []

  def add(self, layer):
    """Adds a new layer to model."""
    with self.graph.as_default():
      if type(layer).__name__ in ['DTNNStep']:
        self.output = layer([self.output] +
                            self.graph_topology.get_topology_placeholders())
      elif type(layer).__name__ in ['DTNNGather']:
        self.output = layer(
            [self.output, self.graph_topology.atom_membership_placeholder])
      else:
        self.output = layer(self.output)
      self.layers.append(layer)


class SequentialDAGGraph(SequentialGraph):
  """SequentialGraph for DAG models
  """

  def __init__(self, n_atom_feat=75, max_atoms=50):
    """
    Parameters
    ----------
    n_atom_feat: int, optional
      Number of features per atom.
    max_atoms: int, optional
      Maximum number of atoms in a molecule, should be defined based on dataset
    """
    warnings.warn("SequentialDAGGraph is deprecated. "
                  "Will be removed in DeepChem 1.4.", DeprecationWarning)
    self.graph = tf.Graph()
    with self.graph.as_default():
      self.graph_topology = DAGGraphTopology(
          n_atom_feat=n_atom_feat, max_atoms=max_atoms)
      self.output = self.graph_topology.get_atom_features_placeholder()
    self.layers = []

  def add(self, layer):
    """Adds a new layer to model."""
    with self.graph.as_default():
      if type(layer).__name__ in ['DAGLayer']:
        self.output = layer([self.output] +
                            self.graph_topology.get_topology_placeholders())
      elif type(layer).__name__ in ['DAGGather']:
        self.output = layer(
            [self.output, self.graph_topology.membership_placeholder])
      else:
        self.output = layer(self.output)
      self.layers.append(layer)


class SequentialWeaveGraph(SequentialGraph):
  """SequentialGraph for Weave models
  """

  def __init__(self, max_atoms=50, n_atom_feat=75, n_pair_feat=14):
    """
    Parameters
    ----------
    max_atoms: int, optional
      Maximum number of atoms in a molecule, should be defined based on dataset
    n_atom_feat: int, optional
      Number of features per atom.
    n_pair_feat: int, optional
      Number of features per pair of atoms.
    """
    warnings.warn("SequentialWeaveGraph is deprecated. "
                  "Will be removed in DeepChem 1.4.", DeprecationWarning)
    self.graph = tf.Graph()
    self.max_atoms = max_atoms
    self.n_atom_feat = n_atom_feat
    self.n_pair_feat = n_pair_feat
    with self.graph.as_default():
      self.graph_topology = WeaveGraphTopology(self.max_atoms, self.n_atom_feat,
                                               self.n_pair_feat)
      self.output = self.graph_topology.get_atom_features_placeholder()
      self.output_P = self.graph_topology.get_pair_features_placeholder()
    self.layers = []

  def add(self, layer):
    """Adds a new layer to model."""
    with self.graph.as_default():
      if type(layer).__name__ in ['WeaveLayer']:
        self.output, self.output_P = layer([
            self.output, self.output_P
        ] + self.graph_topology.get_topology_placeholders())
      elif type(layer).__name__ in ['WeaveConcat']:
        self.output = layer(
            [self.output, self.graph_topology.atom_mask_placeholder])
      elif type(layer).__name__ in ['WeaveGather']:
        self.output = layer(
            [self.output, self.graph_topology.membership_placeholder])
      else:
        self.output = layer(self.output)
      self.layers.append(layer)


class AlternateSequentialWeaveGraph(SequentialGraph):
  """Alternate implementation of SequentialGraph for Weave models
  """

  def __init__(self, batch_size, max_atoms=50, n_atom_feat=75, n_pair_feat=14):
    """
    Parameters
    ----------
    batch_size: int
      number of molecules in a batch
    max_atoms: int, optional
      Maximum number of atoms in a molecule, should be defined based on dataset
    n_atom_feat: int, optional
      Number of features per atom.
    n_pair_feat: int, optional
      Number of features per pair of atoms.
    """
    warnings.warn("AlternateSequentialWeaveGraph is deprecated. "
                  "Will be removed in DeepChem 1.4.", DeprecationWarning)
    self.graph = tf.Graph()
    self.batch_size = batch_size
    self.max_atoms = max_atoms
    self.n_atom_feat = n_atom_feat
    self.n_pair_feat = n_pair_feat
    with self.graph.as_default():
      self.graph_topology = AlternateWeaveGraphTopology(
          self.batch_size, self.max_atoms, self.n_atom_feat, self.n_pair_feat)
      self.output = self.graph_topology.get_atom_features_placeholder()
      self.output_P = self.graph_topology.get_pair_features_placeholder()
    self.layers = []

  def add(self, layer):
    """Adds a new layer to model."""
    with self.graph.as_default():
      if type(layer).__name__ in ['AlternateWeaveLayer']:
        self.output, self.output_P = layer([
            self.output, self.output_P
        ] + self.graph_topology.get_topology_placeholders())
      elif type(layer).__name__ in ['AlternateWeaveGather']:
        self.output = layer(
            [self.output, self.graph_topology.atom_split_placeholder])
      else:
        self.output = layer(self.output)
      self.layers.append(layer)


class SequentialSupportGraph(object):
  """An analog of Keras Sequential model for test/support models."""

  def __init__(self, n_feat):
    """
    Parameters
    ----------
    n_feat: int
      Number of atomic features.
    """
    warnings.warn("SequentialSupportWeaveGraph is deprecated. "
                  "Will be removed in DeepChem 1.4.", DeprecationWarning)
    self.graph = tf.Graph()
    with self.graph.as_default():
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
    with self.graph.as_default():
      self.layers.append(layer)

      # Update new value of x
      if type(layer).__name__ in ['GraphConv', 'GraphGather', 'GraphPool']:
        assert self.bool_pre_gather, "Cannot apply graphical layers after gather."

        self.test = layer([self.test] + self.test_graph_topology.topology)
        self.support = layer([self.support] +
                             self.support_graph_topology.topology)
      else:
        self.test = layer(self.test)
        self.support = layer(self.support)

      if type(layer).__name__ == 'GraphGather':
        self.bool_pre_gather = False  # Set flag to stop adding topology

  def add_test(self, layer):
    """Adds a layer to test."""
    with self.graph.as_default():
      self.layers.append(layer)

      # Update new value of x
      if type(layer).__name__ in ['GraphConv', 'GraphPool', 'GraphGather']:
        self.test = layer([self.test] + self.test_graph_topology.topology)
      else:
        self.test = layer(self.test)

  def add_support(self, layer):
    """Adds a layer to support."""
    with self.graph.as_default():
      self.layers.append(layer)

      # Update new value of x
      if type(layer).__name__ in ['GraphConv', 'GraphPool', 'GraphGather']:
        self.support = layer([self.support] +
                             self.support_graph_topology.topology)
      else:
        self.support = layer(self.support)

  def join(self, layer):
    """Joins test and support to a two input two output layer"""
    with self.graph.as_default():
      self.layers.append(layer)
      self.test, self.support = layer([self.test, self.support])

  def get_test_output(self):
    return self.test

  def get_support_output(self):
    return self.support

  def return_outputs(self):
    return [self.test] + [self.support]

  def return_inputs(self):
    return (self.test_graph_topology.get_inputs() +
            self.support_graph_topology.get_inputs())

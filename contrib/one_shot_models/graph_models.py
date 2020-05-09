"""
Convenience classes for assembling graph models.
"""
import warnings
import tensorflow as tf
from deepchem.nn.layers import GraphGather
from deepchem.models.tf_new_models.graph_topology import GraphTopology, DTNNGraphTopology, DAGGraphTopology, WeaveGraphTopology, AlternateWeaveGraphTopology

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

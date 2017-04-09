"""Manages Placeholders for Graph convolution networks.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Han Altae-Tran and Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import numpy as np
import tensorflow as tf
from deepchem.nn.copy import Input
from deepchem.feat.mol_graphs import ConvMol


def merge_two_dicts(x, y):
  z = x.copy()
  z.update(y)
  return z


def merge_dicts(l):
  """Convenience function to merge list of dictionaries."""
  merged = {}
  for dict in l:
    merged = merge_two_dicts(merged, dict)
  return merged


class GraphTopology(object):
  """Manages placeholders associated with batch of graphs and their topology"""

  def __init__(self, n_feat, name='topology', max_deg=10, min_deg=0):
    """
    Note that batch size is not specified in a GraphTopology object. A batch
    of molecules must be combined into a disconnected graph and fed to topology
    directly to handle batches.

    Parameters
    ----------
    n_feat: int
      Number of features per atom.
    name: str, optional
      Name of this manager.
    max_deg: int, optional
      Maximum #bonds for atoms in molecules.
    min_deg: int, optional
      Minimum #bonds for atoms in molecules.
    """

    #self.n_atoms = n_atoms
    self.n_feat = n_feat

    self.name = name
    self.max_deg = max_deg
    self.min_deg = min_deg

    self.atom_features_placeholder = tensor = tf.placeholder(
        dtype='float32',
        shape=(None, self.n_feat),
        name=self.name + '_atom_features')
    self.deg_adj_lists_placeholders = [
        tf.placeholder(
            dtype='int32',
            shape=(None, deg),
            name=self.name + '_deg_adj' + str(deg))
        for deg in range(1, self.max_deg + 1)
    ]
    self.deg_slice_placeholder = tf.placeholder(
        dtype='int32',
        shape=(self.max_deg - self.min_deg + 1, 2),
        name=self.name + '_deg_slice')
    self.membership_placeholder = tf.placeholder(
        dtype='int32', shape=(None,), name=self.name + '_membership')

    # Define the list of tensors to be used as topology
    self.topology = [self.deg_slice_placeholder, self.membership_placeholder]
    self.topology += self.deg_adj_lists_placeholders

    self.inputs = [self.atom_features_placeholder]
    self.inputs += self.topology

  def get_input_placeholders(self):
    """All placeholders.

    Contains atom_features placeholder and topology placeholders.
    """
    return self.inputs

  def get_topology_placeholders(self):
    """Returns topology placeholders

    Consists of deg_slice_placeholder, membership_placeholder, and the
    deg_adj_list_placeholders.
    """
    return self.topology

  def get_atom_features_placeholder(self):
    return self.atom_features_placeholder

  def get_deg_adjacency_lists_placeholders(self):
    return self.deg_adj_lists_placeholders

  def get_deg_slice_placeholder(self):
    return self.deg_slice_placeholder

  def get_membership_placeholder(self):
    return self.membership_placeholder

  def batch_to_feed_dict(self, batch):
    """Converts the current batch of mol_graphs into tensorflow feed_dict.

    Assigns the graph information in array of ConvMol objects to the
    placeholders tensors

    params
    ------
    batch : np.ndarray
      Array of ConvMol objects

    returns
    -------
    feed_dict : dict
      Can be merged with other feed_dicts for input into tensorflow
    """
    # Merge mol conv objects
    batch = ConvMol.agglomerate_mols(batch)
    atoms = batch.get_atom_features()
    deg_adj_lists = [
        batch.deg_adj_lists[deg] for deg in range(1, self.max_deg + 1)
    ]

    # Generate dicts
    deg_adj_dict = dict(
        list(zip(self.deg_adj_lists_placeholders, deg_adj_lists)))
    atoms_dict = {
        self.atom_features_placeholder: atoms,
        self.deg_slice_placeholder: batch.deg_slice,
        self.membership_placeholder: batch.membership
    }
    return merge_dicts([atoms_dict, deg_adj_dict])


class DTNNGraphTopology(GraphTopology):
  """Manages placeholders associated with batch of graphs and their topology"""

  def __init__(self,
               max_n_atoms,
               n_distance=100,
               distance_min=-1.,
               distance_max=18.,
               name='DTNN_topology'):
    """
    Parameters
    ----------
    max_n_atoms: int
      maximum number of atoms in a molecule
    n_distance: int, optional
      granularity of distance matrix
      step size will be (distance_max-distance_min)/n_distance
    distance_min: float, optional
      minimum distance of atom pairs, default = -1 Angstorm
    distance_max: float, optional
      maximum distance of atom pairs, default = 18 Angstorm
    """

    #self.n_atoms = n_atoms
    self.name = name
    self.max_n_atoms = max_n_atoms
    self.n_distance = n_distance
    self.distance_min = distance_min
    self.distance_max = distance_max

    self.atom_number_placeholder = tf.placeholder(
        dtype='int32',
        shape=(None, self.max_n_atoms),
        name=self.name + '_atom_number')
    self.atom_mask_placeholder = tf.placeholder(
        dtype='float32',
        shape=(None, self.max_n_atoms),
        name=self.name + '_atom_mask')
    self.distance_matrix_placeholder = tf.placeholder(
        dtype='float32',
        shape=(None, self.max_n_atoms, self.max_n_atoms, self.n_distance),
        name=self.name + '_distance_matrix')
    self.distance_matrix_mask_placeholder = tf.placeholder(
        dtype='float32',
        shape=(None, self.max_n_atoms, self.max_n_atoms),
        name=self.name + '_distance_matrix_mask')

    # Define the list of tensors to be used as topology
    self.topology = [
        self.distance_matrix_placeholder, self.distance_matrix_mask_placeholder
    ]
    self.inputs = [self.atom_number_placeholder]
    self.inputs += self.topology

  def get_atom_number_placeholder(self):
    return self.atom_number_placeholder

  def get_distance_matrix_placeholder(self):
    return self.distance_matrix_placeholder

  def batch_to_feed_dict(self, batch):
    """Converts the current batch of Coulomb Matrix into tensorflow feed_dict.

    Assigns the atom number and distance info to the
    placeholders tensors

    params
    ------
    batch : np.ndarray
      Array of Coulomb Matrix

    returns
    -------
    feed_dict : dict
      Can be merged with other feed_dicts for input into tensorflow
    """
    # Extract atom numbers
    atom_number = np.asarray(list(map(np.diag, batch)))
    atom_mask = np.sign(atom_number)
    atom_number = np.asarray(
        np.round(np.power(2 * atom_number, 1 / 2.4)), dtype=int)
    ZiZj = []
    for molecule in atom_number:
      ZiZj.append(np.outer(molecule, molecule))
    ZiZj = np.asarray(ZiZj)
    distance_matrix = np.expand_dims(batch[:], axis=3)
    distance_matrix = np.concatenate(
        [distance_matrix] * self.n_distance, axis=3)
    distance_matrix_mask = batch[:]
    for im, molecule in enumerate(batch):
      for ir, row in enumerate(molecule):
        for ie, element in enumerate(row):
          if element > 0 and ir != ie:
            # expand a float value distance to a distance vector
            distance_matrix[im, ir, ie, :] = self.gauss_expand(
                ZiZj[im, ir, ie] / element, self.n_distance, self.distance_min,
                self.distance_max)
            distance_matrix_mask[im, ir, ie] = 1
          else:
            distance_matrix[im, ir, ie, :] = 0
            distance_matrix_mask[im, ir, ie] = 0
    # Generate dicts
    dict_DTNN = {
        self.atom_number_placeholder: atom_number,
        self.atom_mask_placeholder: atom_mask,
        self.distance_matrix_placeholder: distance_matrix,
        self.distance_matrix_mask_placeholder: distance_matrix_mask
    }
    return dict_DTNN

  @staticmethod
  def gauss_expand(distance, n_distance, distance_min, distance_max):
    step_size = (distance_max - distance_min) / n_distance
    steps = np.array([distance_min + i * step_size for i in range(n_distance)])
    distance_vector = np.exp(-np.square(distance - steps) / (2 * step_size**2))
    return distance_vector


class DAGGraphTopology(GraphTopology):
  """GraphTopology for DAG models
  """

  def __init__(self, n_feat, batch_size, name='topology', max_atoms=50):

    self.n_feat = n_feat
    self.name = name
    self.max_atoms = max_atoms
    self.batch_size = batch_size
    self.atom_features_placeholder = tf.placeholder(
        dtype='float32',
        shape=(self.batch_size * self.max_atoms, self.n_feat),
        name=self.name + '_atom_features')

    self.parents_placeholder = tf.placeholder(
        dtype='int32',
        shape=(self.batch_size * self.max_atoms, self.max_atoms,
               self.max_atoms),
        # molecule * atom(graph) => step => features
        name=self.name + '_parents')

    self.calculation_orders_placeholder = tf.placeholder(
        dtype='int32',
        shape=(self.batch_size * self.max_atoms, self.max_atoms),
        # molecule * atom(graph) => step
        name=self.name + '_orders')

    self.membership_placeholder = tf.placeholder(
        dtype='int32',
        shape=(self.batch_size * self.max_atoms),
        name=self.name + '_membership')

    # Define the list of tensors to be used as topology
    self.topology = [
        self.parents_placeholder, self.calculation_orders_placeholder,
        self.membership_placeholder
    ]

    self.inputs = [self.atom_features_placeholder]
    self.inputs += self.topology

  def get_parents_placeholder(self):
    return self.parents_placeholder

  def get_calculation_orders_placeholder(self):
    return self.calculation_orders_placeholder

  def batch_to_feed_dict(self, batch):
    """Converts the current batch of mol_graphs into tensorflow feed_dict.

    Assigns the graph information in array of ConvMol objects to the
    placeholders tensors for DAG models

    params
    ------
    batch : np.ndarray
      Array of ConvMol objects

    returns
    -------
    feed_dict : dict
      Can be merged with other feed_dicts for input into tensorflow
    """

    atoms_per_mol = [mol.get_num_atoms() for mol in batch]
    n_atom_features = batch[0].get_atom_features().shape[1]
    membership = np.concatenate(
        [
            np.array([1] * n_atoms + [0] * (self.max_atoms - n_atoms))
            for i, n_atoms in enumerate(atoms_per_mol)
        ],
        axis=0)

    atoms_all = []
    # calculation orders for a batch of molecules
    parents_all = []
    calculation_orders = []
    for idm, mol in enumerate(batch):
      # padding atom features vector of each molecule with 0
      atom_features_padded = np.concatenate(
          [
              mol.get_atom_features(), np.zeros(
                  (self.max_atoms - atoms_per_mol[idm], n_atom_features))
          ],
          axis=0)
      atoms_all.append(atom_features_padded)

      # calculation orders for DAGs
      parents = mol.parents
      # number of DAGs should equal number of atoms
      assert len(parents) == atoms_per_mol[idm]
      parents_all.extend(parents[:])
      # padding with `max_atoms`
      parents_all.extend([
          self.max_atoms * np.ones((self.max_atoms, self.max_atoms), dtype=int)
          for i in range(self.max_atoms - atoms_per_mol[idm])
      ])
      for parent in parents:
        # index for an atom in `parents_all` and `atoms_all` is different, 
        # this function changes the index from the position in current molecule(DAGs, `parents_all`) 
        # to position in batch of molecules(`atoms_all`)
        # only used in tf.gather on `atom_features_placeholder`
        calculation_orders.append(self.index_changing(parent[:, 0], idm))

      # padding with `batch_size*max_atoms`
      calculation_orders.extend([
          self.batch_size * self.max_atoms * np.ones(
              (self.max_atoms,), dtype=int)
          for i in range(self.max_atoms - atoms_per_mol[idm])
      ])

    atoms_all = np.concatenate(atoms_all, axis=0)
    parents_all = np.stack(parents_all, axis=0)
    calculation_orders = np.stack(calculation_orders, axis=0)
    atoms_dict = {
        self.atom_features_placeholder: atoms_all,
        self.membership_placeholder: membership,
        self.parents_placeholder: parents_all,
        self.calculation_orders_placeholder: calculation_orders
    }

    return atoms_dict

  def index_changing(self, index, n_mol):
    output = np.zeros_like(index)
    for ide, element in enumerate(index):
      if element < self.max_atoms:
        output[ide] = element + n_mol * self.max_atoms
      else:
        output[ide] = self.batch_size * self.max_atoms
    return output


class WeaveGraphTopology(GraphTopology):
  """Manages placeholders associated with batch of graphs and their topology"""

  def __init__(self, batch_size, n_atom_feat, n_pair_feat, 
               max_atoms=100, name='Weave_topology'):
    """
    Parameters
    ----------
    max_atoms: int
      maximum number of atoms in a molecule
    n_atom_feat: int
      number of basic features of each atom
    n_pair_feat: int
      number of basic features of each pair
    """

    #self.n_atoms = n_atoms
    self.name = name
    self.batch_size = batch_size
    self.n_atom_feat = n_atom_feat
    self.n_pair_feat = n_pair_feat
    self.max_atoms = max_atoms * batch_size

    self.atom_features_placeholder = tf.placeholder(
        dtype='float32',
        shape=(None, self.n_atom_feat),
        name=self.name + '_atom_features')
    self.pair_features_placeholder = tf.placeholder(
        dtype='float32',
        shape=(None, self.n_pair_feat),
        name=self.name + '_pair_features')
    self.pair_split_placeholder = tf.placeholder(
        dtype='int32', shape=(self.max_atoms,), 
        name=self.name + '_pair_split')
    self.pair_membership_placeholder = tf.placeholder(
        dtype='bool', shape=(self.max_atoms,), 
        name=self.name + '_pair_membership')
    self.atom_split_placeholder = tf.placeholder(
        dtype='int32', shape=(self.batch_size,), 
        name=self.name + '_atom_split')
    self.atom_to_pair_placeholder = tf.placeholder(
        dtype='int32', shape=(None,2), 
        name=self.name + '_atom_to_pair')
    
    
    # Define the list of tensors to be used as topology
    self.topology = [self.pair_split_placeholder, self.pair_membership_placeholder,
                     self.atom_split_placeholder, self.atom_to_pair_placeholder]
    self.inputs = [self.atom_features_placeholder]
    self.inputs += self.topology

  def get_pair_features_placeholder(self):
    return self.pair_features_placeholder
    
  def batch_to_feed_dict(self, batch):
    """Converts the current batch of WeaveMol into tensorflow feed_dict.

    Assigns the atom features and pair features to the
    placeholders tensors

    params
    ------
    batch : np.ndarray
      Array of WeaveMol

    returns
    -------
    feed_dict : dict
      Can be merged with other feed_dicts for input into tensorflow
    """
    # Extract atom numbers
    atom_feat = []
    pair_feat = []
    atom_split = []
    atom_to_pair = []
    pair_split = []
    max_atoms = self.max_atoms
    start = 0
    for im, mol in enumerate(batch):
      n_atoms = mol.get_num_atoms()
      # number of atoms in each molecule
      atom_split.append(n_atoms)
      # index of pair features
      C0, C1 = np.meshgrid(np.arange(n_atoms), np.arange(n_atoms))
      atom_to_pair.append(np.transpose(np.array([C1.flatten()+start, C0.flatten()+start])))
      start = start + n_atoms
      # number of pairs for each atom
      pair_split.extend([n_atoms]*n_atoms)
      # atom features
      atom_feat.append(mol.get_atom_features())
      # pair features
      pair_feat.append(np.reshape(mol.get_pair_features(), 
                                  (n_atoms*n_atoms, self.n_pair_feat)))
      
    atom_feat = np.concatenate(atom_feat, axis=0)
    pair_feat = np.concatenate(pair_feat, axis=0)
    atom_to_pair = np.concatenate(atom_to_pair, axis=0)
    atom_split = np.array(atom_split)
    n_pair = len(pair_split)
    pair_split = np.pad(pair_split, ((0, max_atoms-n_pair)), 'constant')
    pair_membership = np.array([True]*n_pair + [False]*(max_atoms-n_pair))
    
    # Generate dicts
    dict_DTNN = {
        self.atom_features_placeholder: atom_feat,
        self.pair_features_placeholder: pair_feat,
        self.pair_split_placeholder: pair_split,
        self.pair_membership_placeholder: pair_membership,
        self.atom_split_placeholder: atom_split,
        self.atom_to_pair_placeholder: atom_to_pair
    }
    return dict_DTNN

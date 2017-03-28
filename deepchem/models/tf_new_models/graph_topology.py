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
    # Merge mol conv objects

    atoms_per_mol = [mol.get_num_atoms() for mol in batch]
    n_atom_features = batch[0].get_atom_features().shape[1]
    membership = np.concatenate(
        [
            np.array([1] * n_atoms + [0] * (self.max_atoms - n_atoms))
            for i, n_atoms in enumerate(atoms_per_mol)
        ],
        axis=0)

    atoms_all = []
    parents_all = []
    # calculation orders for a batch of molecules
    calculation_orders = []
    for idm, mol in enumerate(batch):
      atom_features_padded = np.concatenate(
          [
              mol.get_atom_features(), np.zeros(
                  (self.max_atoms - atoms_per_mol[idm], n_atom_features))
          ],
          axis=0)
      # padding atom features vector of each molecule with 0
      atoms_all.append(atom_features_padded)

      parents = self.UG_to_DAG(mol)
      # calculation orders for DAGs
      assert len(parents) == atoms_per_mol[idm]
      # number of DAGs should equal number of atoms
      parents_all.extend(parents[:])
      parents_all.extend([
          self.max_atoms * np.ones((self.max_atoms, self.max_atoms), dtype=int)
          for i in range(self.max_atoms - atoms_per_mol[idm])
      ])
      # padding with max_atoms
      for parent in parents:
        calculation_orders.append(self.indice_changing(parent[:, 0], idm))
        # the indice for a specific atom in the molecule's DAGs and atom_features_placeholder
        # is different, this function changes the indice from the position in current molecule(DAGs) 
        # to position in batch of molecules(atom_features_placeholder)
        # and this is only going to be used in tf.gather on atom_features_placeholder
      calculation_orders.extend([
          self.batch_size * self.max_atoms * np.ones(
              (self.max_atoms,), dtype=int)
          for i in range(self.max_atoms - atoms_per_mol[idm])
      ])
      # padding with (batch_size*max_atoms)

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

  def indice_changing(self, indice, n_mol):
    output = np.zeros_like(indice)
    for ide, element in enumerate(indice):
      if element < self.max_atoms:
        output[ide] = element + n_mol * self.max_atoms
      else:
        output[ide] = self.batch_size * self.max_atoms
    return output

  def UG_to_DAG(self, sample):
    """This function generates the DAGs for a molecule
    """
    parents = []
    # list of DAGs, one DAG represents the calculation orders
    # stemming from one specific atom in the molecule,
    # hence this list include k elements for a molecule with k atoms
    UG = sample.get_adjacency_list()
    # starting from the adjacency list derived by graphconv featurizer
    n_atoms = sample.get_num_atoms()
    # number of graphs need to be generated
    max_atoms = self.max_atoms
    # for a graph on a molecule with k atoms, there will be k steps, 
    # each step calculate graph features for one atom,
    # maximum number of steps is the same as max_atoms
    for count in range(n_atoms):
      # each iteration generates one DAG
      # stemming from atom with indice `count`
      DAG = []
      parent = [[] for i in range(n_atoms)]
      # list of lists, each element(also a list) represents the calculation order
      # for every atom in the molecule in the current graph
      current_atoms = [count]
      # starting from the atom with indice `count`
      atoms_indicator = np.ones((n_atoms,))
      # flags, whether the atom is already included in the DAG
      atoms_indicator[count] = 0
      # atom `count` is in the DAG
      radial = 0
      # recording number of radial propagation steps
      while np.sum(atoms_indicator) > 0:
        # in this while loop, atoms directly connected to `count` will be first added into
        # the DAG(radial=0), then atoms two-bond away from `count` will be added in the
        # second loop(radial=1). Atoms i-bond away will be added in loop i
        if radial > n_atoms:
          break
          # when molecules have separate parts, starting from one part, it is not possible
          # to include all atoms.
        next_atoms = []
        # reinitialize targets for next iteration
        for current_atom in current_atoms:
          for atom_adj in UG[current_atom]:
            # atoms connected to current_atom
            if atoms_indicator[atom_adj] > 0:
              DAG.append((current_atom, atom_adj))
              # this for loop generates the dependency map of this DAG
              # atoms that connected to current_atoms(and not included in the DAG yet)
              # are added into DAG, and will be the current_atoms for next iteration.
              atoms_indicator[atom_adj] = 0
              next_atoms.append(atom_adj)
              # including into targets for next iteration
        current_atoms = next_atoms
        # into next iteration, finding atoms connected one more bond away
        radial = radial + 1
      for edge in reversed(DAG):
        # DAG starts from the target atom, hence the calculation should go in reverse
        parent[edge[0]].append(edge[1])
        # edge[1] is the parent of edge[0]
        parent[edge[0]].extend(parent[edge[1]])
        # all the parents of edge[1] is also the parents of edge[0]
      # after this for loop, parents[i] is the list that includes all parents of atom i
      for ids, atom in enumerate(parent):
        parent[ids].insert(0, ids)
        # manually adding the atom indice into its parents list
      # after this for loop, parents[i][0] = i, parents[i][1:] are all parents of atom i
      parent = sorted(parent, key=len)
      # key part of this function, atoms with less parents come first,
      # so when we do a for loop on the list , atoms without parents will be first calculated
      # then atoms with more parents can be calculated based on calculated graph features.
      # the starting atom of this DAG will be calculated in the last step, 
      # since every other atom is its parent.
      for ids, atom in enumerate(parent):
        n_par = len(atom)
        parent[ids].extend([max_atoms for i in range(max_atoms - n_par)])
        # padding with max_atoms
      while len(parent) < max_atoms:
        parent.insert(0, [max_atoms] * max_atoms)
        # padding
      parents.append(np.array(parent))
      # parents[i] is the calculation order for the DAG stemming from atom i,
      # which is a max_atoms * max_atoms numpy array(after padding)
    return parents

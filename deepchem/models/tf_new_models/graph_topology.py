"""Manages Placeholders for Graph convolution networks.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Han Altae-Tran and Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

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

class DTNNGraphTopology(object):
  """Manages placeholders associated with batch of graphs and their topology"""

  def __init__(self, max_n_atoms, n_distance, name='DTNN_topology'):
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
    self.name = name
    self.max_n_atoms = max_n_atoms
    self.n_distance = n_distance
    
    self.atom_number_placeholder = tf.placeholder(
        dtype='int32', 
        shape=(None,self.max_n_atoms), 
        name=self.name + '_atom_number')
    self.distance_matrix_placeholder = tf.placeholder(
        dtype='float32',
        shape=(None, self.max_n_atoms, self.max_n_atoms, self.n_distance),
        name=self.name + '_distance_matrix')
    self.distance_matrix_mask_placeholder = tf.placeholder(
        dtype=tf.float32,
        shape=(None, self.max_n_atoms, self.max_n_atoms),
        name=self.name + '_distance_matrix_mask')

    # Define the list of tensors to be used as topology
    self.topology = [self.distance_matrix_placeholder, 
                     self.distance_matrix_mask_placeholder]
    self.inputs = [self.atom_number_placeholder]
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

  def get_atom_number_placeholder(self):
    return self.atom_number_placeholder

  def get_distance_matrix_placeholder(self):
    return self.distance_matrix_placeholder

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
    atom_number = np.asarray(map(np.diag, batch))
    atom_number = np.asarray(np.round(np.power(2*atom_number, 1/2.4)), dtype=int)
    ZiZj = []
    for molecule in atom_number:
        ZiZj.append(np.outer(molecule, molecule))
    ZiZj = np.asarray(ZiZj)
    distance_matrix = batch[:]
    for im, molecule in enumerate(batch):
      for ir, row in enumerate(molecule):
        for ie, element in enumerate(row):
          if element>0 and ir != ie:
            distance_matrix[im, ir, ie] = ZiZj[im, ir,ie]/element
          else:
            distance_matrix[im, ir, ie] = 0
    distance_matrix_mask = np.asarray(np.asarray(distance_matrix, dtype=bool), dtype=float)
    distance_matrix = self.gauss_expand(distance_matrix)
    # Generate dicts
    dict_DTNN = {
        self.atom_number_placeholder: atom_number,
        self.distance_matrix_placeholder: distance_matrix,
        self.distance_matrix_mask_placeholder: distance_matrix_mask
    }
    return dict_DTNN
    
  @staticmethod
  def gauss_expand(distance_matrix):
    for im, molecule in enumerate(distance_matrix):
      for ir, row in enumerate(molecule):
        for ie, element in enumerate(row):
          if element>0
"""Manages Placeholders for Graph convolution networks.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Han Altae-Tran and Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

from keras.layers import Input
from keras import backend as K
from deepchem.featurizers.mol_graphs import ConvMol

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
  def __init__(self, n_atoms, n_feat, batch_size, name='topology', max_deg=6,
               min_deg=0):
    """
    Parameters
    ----------
    n_atoms: int
      Number of atoms (max) in graphs.
    n_feat: int
      Number of features per atom.
    batch_size: int
      Number of molecules per batch.
    name: str, optional
      Name of this manager.
    max_deg: int, optional
      Maximum #bonds for atoms in molecules.
    min_deg: int, optional
      Minimum #bonds for atoms in molecules.
    """
    
    self.n_atoms = n_atoms
    self.n_feat = n_feat
    self.batch_size = batch_size

    self.name = name
    self.max_deg = max_deg
    self.min_deg = min_deg
    self.init_keras_placeholders()

  def init_keras_placeholders(self):
    self.nodes_placeholder = Input(
        tensor=K.placeholder(
            shape=(None, self.n_feat), dtype='float32',
            name=self.name+'_nodes'))
    self.deg_adj_lists_placeholders = [
        Input(tensor=K.placeholder(
          shape=(None, deg), dtype='int32', name=self.name+'_deg_adj'+str(deg)))
        for deg in range(1, self.max_deg+1)]
    self.deg_slice_placeholder = Input(
        tensor=K.placeholder(
            shape=(self.max_deg-self.min_deg+1,2),
            name="deg_slice", dtype='int32'),
        name=self.name+'_deg_slice')
    self.membership_placeholder = Input(
          tensor=K.placeholder(shape=(None,), dtype='int32', name="membership"),
          name=self.name+'_membership')

    # Define the list of tensors to be used as topology
    self.topology = [self.deg_slice_placeholder, self.membership_placeholder]
    self.topology.extend(self.deg_adj_lists_placeholders)

    self.inputs = [self.nodes_placeholder]
    self.inputs.extend(self.topology)

  def get_inputs(self):
    return self.inputs

  def get_topology(self):
    return self.topology

  def get_batch_size(self):
    return self.batch_size

  def get_nodes(self):
    return self.nodes_placeholder

  def get_deg_adj_lists(self):
    return self.deg_adj_lists_placeholders

  def get_deg_slice(self):
    return self.deg_slice_placeholder

  def get_membership(self):
    return self.membership_placeholder

  # TODO(rbharath): It's still not clear to me that this should live alone like
  # this... It's awkward to separate out part of the batch construction this
  # way.
  def batch_to_feed_dict(self, batch):
    """Converts the current batch into a feed_dict used by tensorflow.

    Assigns the graph information in batch to the placeholders tensors

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

    atoms = batch.nodes
    deg_adj_lists = [batch.deg_adj_lists[deg]
                     for deg in range(1, self.max_deg+1)]

    # Generate dicts
    deg_adj_dict = dict(zip(self.deg_adj_lists_placeholders, deg_adj_lists))
    atoms_dict = {self.nodes_placeholder : atoms,
                  self.deg_slice_placeholder : batch.deg_slice,
                  self.membership_placeholder : batch.membership}
    return merge_dicts([atoms_dict, deg_adj_dict])

def extract_topology(x):
  # Extracts the topology tensors from x
  topology = x[1::]

  # Extract parsed topology information
  deg_slice = topology[0]
  membership = topology[1]
  deg_adj_lists = topology[2::]

  return deg_slice, membership, deg_adj_lists

def extract_nodes(x):
  # Extracts the nodes from x (just the first tensor in the list of tensors)
  return x[0]

def extract_membership(x):
  return x[2]

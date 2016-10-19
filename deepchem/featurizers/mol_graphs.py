"""
Data Structures used to represented molecules for convolutions.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Han Altae-Tran and Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

import csv
import random
import numpy as np

max_deg = 6
min_deg = 0

def index_sum(l, offset=0):
  """Returns cumulative sums for set of counts.

  Returns the cumulative sums for a set of counts with the first returned value
  starting at 0. I.e [3,2,4] -> [0, 3, 5]. Useful for reindexing
  """
  return np.delete(np.insert(np.cumsum(l), 0, 0), -1) + offset

def index_sum_with_final(l, offset=0):
  """Returns cumulative sums for set of counts.

  TODO(rbharath): How is this different from index_sum?

  Returns the cumulative sums for a set of counts with the first returned value
  starting at 0. I.e [3,2,4] -> [0, 3, 5, 9]. Keeps final sum for searching. 
  Useful for reindexing.
  """
  return np.insert(np.cumsum(l), 0, 0) + offset

class ConvMol(object):
  """Holds information about a molecules.
  """
  def __init__(self, nodes, canon_adj_list, max_deg=6, min_deg=0):
    """
    Parameters
    ----------
    nodes: np.ndarray
      Has shape (n_atoms, n_feat)
    canon_ad_list: list
      List of length n_atoms, with neighor indices of each atom.
    max_deg: int, optional
      Maximum degree of any atom.
    min_deg: int, optional
      Minimum degree of any atom.
    """

    self.nodes = nodes
    self.n_atoms, self.n_feat = nodes.shape
    self.deg_list = np.array([len(nbr) for nbr in canon_adj_list], dtype=np.int32)
    self.canon_adj_list = canon_adj_list
    self.deg_adj_lists = []
    self.deg_slice = []
    self.max_deg = max_deg
    self.min_deg = min_deg
    
    self.membership = self.get_num_nodes() * [0]
    
    self.deg_sort()            

    # Get the degree id list (which corrects for min_deg)
    self.deg_id_list = np.array(self.deg_list)-min_deg

    # Get the size of each degree block
    deg_size = [self.get_deg_size(deg)
                for deg in range(self.min_deg, self.max_deg+1)]

    # Get the the start indices for items in each block
    self.deg_start = index_sum_with_final(deg_size)

    # Get the node indices when they are reset when the degree changes
    deg_block_indices = [i - self.deg_start[self.deg_list[i]] 
                         for i in range(self.nodes.shape[0])]
    
    # Convert to numpy array
    self.deg_block_indices = np.array(deg_block_indices)

  def get_nodes_with_deg(self, deg):
    # Retrieves nodes with the specific degree
    start_ind = self.deg_slice[deg-self.min_deg,0]
    sz = self.deg_slice[deg-self.min_deg,1]
    return self.nodes[start_ind:(start_ind+sz),:]

  def get_deg_size(self, deg):
    """Returns the number of nodes with the given degree"""
    return self.deg_slice[deg-self.min_deg,1]

  def get_num_nodes(self):
    return self.nodes.shape[0]
        
  def deg_sort(self):
    """ Sort the order of the nodes by degree, maintaining original order
        whenever two nodes have the same degree. 
    """
    old_ind = range(self.get_num_nodes())
    deg_list = self.deg_list
    new_ind = list(np.lexsort((old_ind, deg_list)))

    N_nodes = self.get_num_nodes()
    
    # Reorder old nodes
    self.nodes = self.nodes[new_ind,:]

    # Reorder old deg lists
    self.deg_list = [self.deg_list[i] for i in new_ind]
    
    # Sort membership
    self.membership = [self.membership[i] for i in new_ind]
    
    # Create old to new dictionary. not exactly intuitive
    old_to_new = dict(zip(new_ind, old_ind))
    
    # Reorder adjacency lists
    self.canon_adj_list = [self.canon_adj_list[i] for i in new_ind]
    self.canon_adj_list = [[old_to_new[k] for k in self.canon_adj_list[i]]
                           for i in range(len(new_ind))]

    # Get numpy version of degree list for indexing
    deg_array = np.array(self.deg_list)

    # Initialize adj_lists, which supports min_deg = 1 only
    self.deg_adj_lists = (self.max_deg + 1 - self.min_deg)* [0]

    # Parse as deg separated
    for deg in range(self.min_deg, self.max_deg+1):
      # Get indices corresponding to the current degree
      rng = np.array(range(N_nodes))
      indices = rng[deg_array==deg]

      # Extract and save adjacency list for the current degree
      to_cat = [self.canon_adj_list[i] for i in indices]
      if len(to_cat) > 0:
        adj_list = np.vstack([self.canon_adj_list[i] for i in indices])
        self.deg_adj_lists[deg-self.min_deg] = adj_list
          
      else:
        self.deg_adj_lists[deg-self.min_deg] = np.zeros([0, deg], dtype=np.int32)

    # Construct the slice information
    deg_slice = np.zeros([self.max_deg+1-self.min_deg, 2], dtype=np.int32)

    for deg in range(self.min_deg, self.max_deg+1):
      if deg == 0:
        deg_size = np.sum(deg_array==deg)
      else:
        deg_size = self.deg_adj_lists[deg-self.min_deg].shape[0]
          
      deg_slice[deg-self.min_deg, 1] = deg_size
      # Get the cumulative indices after the first index
      if deg > self.min_deg:
        deg_slice[deg-self.min_deg, 0] = (
            deg_slice[deg-self.min_deg-1,0] + deg_slice[deg-self.min_deg-1,1])

    # Set indices with zero sized slices to zero to avoid indexing errors
    deg_slice[:,0] *= (deg_slice[:,1]!=0)
    self.deg_slice = deg_slice                         

  def get_deg_slice(self):
    """Returns degree-slice tensor.
  
    The deg_slice tensor allows indexing into a flattened version of the
    molecule's atoms.  In general, deg_slice has shape (max_deg+1-min_deg, 2). For
    degree deg, deg_slice[deg][0] is the starting index in the flattened adjacency
    list of atoms with degree deg. Then deg_slice[deg][1] is the number of atoms
    with degree deg.

    Returns
    -------
    deg_slice: np.ndarray 
      Shape (max_deg+1-min_deg, 2)
    """
    return self.deg_slice

  @staticmethod
  def get_null_mol(N_feat):
    """ Get one molecule with one node in each deg block, with all the nodes
    connected to each other, and containing N_feat features
    
    args
    ----
    N_feat : (int) number of features for the nodes in the null molecule
    """
    # Use random insted of zeros to prevent weird issues with summing to zero
    nodes = np.random.uniform(0,1,[self.max_deg+1-self.min_deg, N_feat])
    canon_adj_list = [i*[i-self.min_deg] for i in range(self.min_deg, self.max_deg+1)]

    return ConvMol(nodes, canon_adj_list)

  @staticmethod
  def agglomerate_mols(mol_list, max_deg=6, min_deg=0):
    """Concatenates list of ConvMol's into one mol object that can be used to feed 
    into tensorflow placeholders. The indexing of the molecules are preseved during the
    combination, but the indexing of the atoms are greatly changed.
    
    args
    ----
    mol_list : list of ConvMol objects to be combined into one molecule."""

    N_mols = len(mol_list)

    N_nodes_mol = [mol_list[k].get_num_nodes() for k in range(N_mols)]
    N_nodes_cum = index_sum(N_nodes_mol)

    # Get nodes by degree
    nodes_by_deg = [mol_list[k].get_nodes_with_deg(deg)
                    for deg in range(min_deg, max_deg+1)
                    for k in range(N_mols)]

    # stack the nodes
    nodes = np.vstack(nodes_by_deg)

    # Get the size of each atom list separated by molecule id, then by degree
    mol_deg_sz = [[mol_list[k].get_deg_size(deg) for k in range(N_mols)]
                 for deg in range(min_deg, max_deg+1)]
    
    deg_sz = map(np.sum, mol_deg_sz)  # Get the final size of each degree block
    # Get the index at which each degree starts, not resetting after each degree
    # And not stopping at any speciic molecule

    deg_start = index_sum(deg_sz)
    # Get the tensorflow object required for slicing (deg x 2) matrix, with the
    # first column telling the start indices of each degree block and the
    # second colum telling the size of each degree block

    # Input for tensorflow 
    deg_slice = np.array(zip(deg_start,deg_sz))
    
    # Determines the membership (atom i belongs to membership[i] molecule)
    membership = [k
                  for deg in range(min_deg, max_deg+1)
                  for k in range(N_mols)
                  for i in range(mol_deg_sz[deg][k])]

    # Get the index at which each deg starts, resetting after each degree
    # (deg x N_mols) matrix describing the start indices when you count up the atoms
    # in the final representation, stopping at each molecule, 
    # resetting every time the degree changes
    start_by_deg = np.vstack([index_sum(l) for l in mol_deg_sz])  
        
    # Gets the degree resetting block indices for the atoms in each molecule
    # Here, the indices reset when the molecules change, and reset when the
    # degree changes
    deg_block_indices = [mol.deg_block_indices for mol in mol_list]

    # Get the degree id lookup list. It allows us to search for the degree of a
    # molecule mol_id with corresponding atom mol_atom_id using
    # deg_id_lists[mol_id,mol_atom_id]
    deg_id_lists = [mol.deg_id_list for mol in mol_list]

    # This is used for convience in the following function (explained below)
    start_per_mol = deg_start[:,np.newaxis] + start_by_deg

    def to_final_id(mol_atom_id, mol_id):
      # Get the degree id (corrected for min_deg) of the considered atom
      deg_id = deg_id_lists[mol_id][mol_atom_id]

      # Return the final index of atom mol_atom_id in molecule mol_id.  Using
      # the degree of this atom, must find the index in the molecule's original
      # degree block corresponding to degree id deg_id (second term), and then
      # calculate which index this degree block ends up in the final
      # representation (first term). The sum of the two is the final indexn
      return start_per_mol[deg_id,mol_id] + deg_block_indices[mol_id][mol_atom_id]
  
    # Initialize the new degree separated adjacency lists
    deg_adj_lists = [np.zeros([deg_sz[deg],deg], dtype=np.int32) 
                     for deg in range(min_deg, max_deg+1)]

    # Update the old adjcency lists with the new atom indices and then combine
    # all together
    for deg in range(min_deg, max_deg+1):
      row = 0  # Initialize counter
      deg_id = deg-min_deg  # Get corresponding degree id

      # Iterate through all the molecules
      for mol_id in range(N_mols):
        # Get the adjacency lists for this molecule and current degree id
        nbr_list = mol_list[mol_id].deg_adj_lists[deg_id]

        # Correct all atom indices to the final indices, and then save the
        # results into the new adjacency lists 
        for i in range(nbr_list.shape[0]):
          for j in range(nbr_list.shape[1]): 
            deg_adj_lists[deg_id][row,j] = to_final_id(nbr_list[i,j], mol_id)
            
          # Increment once row is done
          row += 1

    # Get the final aggregated molecule
    concat_mol = MultiConvMol(
      nodes, deg_adj_lists, deg_slice, membership, N_mols)
    return concat_mol

class MultiConvMol(object):
  """Holds information about multiple molecules, for use in feeding information into
  tensorflow or keras. Generated using the agglomerate_mols function
  """
  def __init__(self, nodes, deg_adj_lists, deg_slice, membership, N_mols):

    self.nodes = nodes
    self.deg_adj_lists = deg_adj_lists
    self.deg_slice = deg_slice
    self.membership = membership
    self.N_mols = N_mols
    self.N_nodes = nodes.shape[0]

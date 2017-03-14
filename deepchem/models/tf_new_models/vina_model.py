"""
Implements Autodock Vina's pose-generation in tensorflow.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import numpy as np
import tensorflow as tf
from deepchem.models import Model
from deepchem.nn import model_ops
import deepchem.utils.rdkit_util as rdkit_util

def compute_neighbor_list(coords, nbr_cutoff, N, M, ndim=3, k=5):
  """Computes a neighbor list from atom coordinates.

  Parameters
  ----------
  N: int
    Max number atoms
  M: int
    Max number neighbors
  ndim: int
    Dimensionality of space.
  k: int
    Number of nearest neighbors to pull down.
  """
  start = tf.reduce_min(coords)
  stop = tf.reduce_max(coords)
  cells = get_cells(start, stop, nbr_cutoff, ndim)
  atoms_in_cells = put_atoms_in_cells(coords, cells, N, ndim, k)
  
  # Associate each atom with cell it belongs to. O(N)
  cell_to_atoms, atom_to_cell = put_atoms_in_cells(
      coords, x_bins, y_bins, z_bins)
  # Associate each cell with its neighbor cells. Assumes periodic boundary   
  # conditions, so does wrapround. O(constant)    
  N_x, N_y, N_z = len(x_bins), len(y_bins), len(z_bins)   
  neighbor_cell_map = compute_neighbor_cell_map(N_x, N_y, N_z)    
    
  # For each atom, loop through all atoms in its cell and neighboring cells.    
  # Accept as neighbors only those within threshold. This computation should be   
  # O(Nm), where m is the number of atoms within a set of neighboring-cells.
  neighbor_list = {}
  for atom in range(N):
    cell = atom_to_cell[atom]
    neighbor_cells = neighbor_cell_map[cell]
    # For smaller systems especially, the periodic boundary conditions can
    # result in neighboring cells being seen multiple times. Use a set() to
    # make sure duplicate neighbors are ignored. Convert back to list before
    # returning. 
    neighbor_list[atom] = set()
    for neighbor_cell in neighbor_cells:
      atoms_in_cell = cell_to_atoms[neighbor_cell]
      for neighbor_atom in atoms_in_cell:
        if neighbor_atom == atom:    
           continue    
        # TODO(rbharath): How does distance need to be modified here to   
        # account for periodic boundary conditions?   
        dist = np.linalg.norm(coords[atom] - coords[neighbor_atom])   
        if dist < neighbor_cutoff:    
          neighbor_list[atom].add((neighbor_atom, dist))    
             
    # Sort neighbors by distance    
    closest_neighbors = sorted(   
        list(neighbor_list[atom]), key=lambda elt: elt[1])    
    closest_neighbors = [nbr for (nbr, dist) in closest_neighbors]    
    # Pick up to max_num_neighbors    
    closest_neighbors = closest_neighbors[:max_num_neighbors]   
    neighbor_list[atom] = closest_neighbors
  return neighbor_list   

def get_cells(start, stop, nbr_cutoff, ndim=3):
  """Returns the locations of all grid points in box.

  Suppose start is -10 Angstrom, stop is 10 Angstrom, nbr_cutoff is 1.
  Then would return a list of length 20^3 whose entries would be
  [(-10, -10, -10), (-10, -10, -9), ..., (9, 9, 9)]

  TODO(rbharath): Make this work in more than 3 dimensions.
  """
  return tf.reshape(tf.transpose(tf.pack(tf.meshgrid(
      *[tf.range(start, stop, nbr_cutoff) for _ in range(ndim)]))), (-1, ndim))
     
def put_atoms_in_cells(coords, cells, N, ndim, k=5):
  """Place each atom into cells. O(N) runtime.    
  
  Let N be the number of atoms.
      
  Parameters    
  ----------    
  coords: tf.Tensor 
    (N, 3) shape.
  cells: tf.Tensor
    (box_size**ndim, ndim) shape.
  N: int
    Number atoms
  ndim: int
    Dimensionality of input space
  k: int
    Number of nearest neighbors.
  """   
  n_cells = int(cells.get_shape()[0])

  # Tile both cells and coords to form arrays of size (n_cells*N, ndim)
  tiled_cells = tf.reshape(tf.tile(cells, (1, N)), (n_cells*N, ndim))
  # TODO(rbharath): Change this for tf 1.0
  # n_cells tensors of shape (N, 1)
  tiled_cells = tf.split_v(tiled_cells, n_cells)

  # Shape (N*n_cells, 1) after tile
  tiled_coords = tf.tile(coords, (n_cells, 1))
  # List of n_cells tensors of shape (N, 1)
  tiled_coords = tf.split_v(tiled_coords, n_cells)

  # Lists of length n_cells
  coords_rel = [tf.to_float(coords) - tf.to_float(cells)
                for (coords, cells) in zip(tiled_coords, tiled_cells)]
  coords_norm = [tf.reduce_sum(rel**2, axis=1) for rel in coords_rel]

  # Lists of length n_cells
  # Get indices of k atoms closest to each cell point
  closest_inds = [tf.nn.top_k(norm, k=k)[1] for norm in coords_norm]
  # n_cells tensors of shape (k, ndim)
  closest_atoms = [tf.gather(coords, inds) for inds in closest_inds]

  return closest_atoms

  # TODO(rbharath):
  #   - Need to find neighbors of the cells (+/- 1 in every dimension).
  #   - Need to group closest atoms amongst cell neighbors
  #   - Need to do another top_k to find indices of closest neighbors.
  #   - Return N lists corresponding to neighbors for every atom.
  
        
def compute_neighbor_cell_map(cells, ndim):
  """Compute neighbors of cells in grid.    

  # TODO(rbharath): Do we need to handle periodic boundary conditions
  properly here?
      
  Parameters    
  ----------    
  cells: tf.Tensor
    (box_size**ndim, ndim) shape.
  """   
  if ndim != 3:
    raise ValueError("Not defined for dimensions besides 3")
  # Number of neighbors of central cube in 3-space is
  # 3^2 (top-face) + 3^2 (bottom-face) + (3^2-1) (middle-band)
  # TODO(rbharath)
  k = 9 + 9 + 8 # (26 faces on Rubik's cube for example)
  n_cells = int(cells.get_shape()[0])
  # Tile cells to form arrays of size (n_cells*n_cells, ndim)
  # Two tilings (a, b, c, a, b, c, ...) vs. (a, a, a, b, b, b, etc.)
  # Tile (a, a, a, b, b, b, etc.)
  tiled_centers = tf.reshape(tf.tile(cells, (1, N)), (n_cells*N, ndim))
  # Tile (a, b, c, a, b, c, ...)
  tiled_cells = tf.tile(cells, (n_cells, 1))

  # Lists of length n_cells
  coords_rel = [tf.to_float(cells) - tf.to_float(centers)
                for (cells, centers) in zip(tiled_centers, tiled_cells)]
  coords_norm = [tf.reduce_sum(rel**2, axis=1) for rel in coords_rel]

  # Lists of length n_cells
  # Get indices of k atoms closest to each cell point
  # n_cells tensors of shape (26, ndim)
  closest_inds = [tf.nn.top_k(norm, k=k)[1] for norm in coords_norm]

  return closest_inds


def cutoff(d):
  """Truncates interactions that are too far away."""
  return tf.cond(d < 8, d, 0)

def gauss_1(d):
  """Computes first Gaussian interaction term.

  Note that d must be in Angstrom
  """
  return tf.exp(-(d/0.5)**2)

def gauss_2(d):
  """Computes second Gaussian interaction term.

  Note that d must be in Angstrom.
  """
  return tf.exp(-((d-3)/2)^2)


def repulsion(d):
  """Computes repulsion interaction term."""
  return tf.cond(d < 0, d**2, 0)

def hydrophobic(d):
  """Compute hydrophobic interaction term."""
  return tf.cond(d < 0.5, 1,
                 tf.cond(d < 1.5, 1.5 - d,  0))

def hbond(d):
  """Computes hydrogen bond term."""
  return tf.cond(d < -0.7, 1,
                 tf.cond(d < 0, (1.0/0.7)(0-d), 0))

def g(c, w, Nrot):
  """Nonlinear function mapping interactions to free energy."""
  return c/(1 + w*Nrot)
  

class VinaModel(Model):

  def __init__(self,
               logdir=None,
               batch_size=50):
    """Vina models.

    .. math:: c = \sum_{i < j} f_{t_i,t_j}(r_{ij})

    Over all pairs of atoms that can move relative to one-another. :math:`t_i` is the
    atomtype of atom :math:`i`.

    Can view as

    .. math:: c = c_\textrm{inter} + c_\textrm{intra}

    depending on whether atoms can move relative to one another. Free energey is
    predicted only from :math:`c_\textrm{inter}`. Let :math:`R_t` be the Van der Waal's radius of
    atom of type t. Then define surface distance

    .. math:: d_{ij} = r_{ij} - R_{t_i} - R_{t_j}

    Then the energy term is

    .. math:: f_{t_i,t_j}(r_{ij}) = \textrm{cutoff}(d_{ij}, h_{t_i,t_j}(d_{ij}))

    where
  
    .. math:: \textrm{cutoff}(d, x) = \begin{cases} x & d < 8 \textrm{ Angstrom} \\ 0 & \textrm{otherwise} \end{cases}

    The inner function can be further broken down into a sum of terms

    .. math:: h_{t_i,t_j}(d) = w_1\textrm{gauss}_1(d) + w_2\textrm{gauss}_2(d) + w_3\textrm{repulsion}(d) + w_4\textrm{hydrophobic}(d) + w_5\textrm{hbond}(d)

    these terms are defined as follows (all constants are in Angstroms):

    .. math:: 
         \textrm{gauss}_1(d) = \exp(-(d/(0.5))^2)
         \textrm{gauss}_2(d) = \exp(-((d-3)/(2))^2)
         \textrm{repulsion}(d) = \begin{cases} d^2 & d < 0 \\ 0 & d \geq 0 \end{cases}
         \textrm{hydrophobic}(d) = \begin{cases} 1 & d < 0.5 \\ 1.5 - d & \textrm{otherwise} \\ 0 & d > 1.5 \end{cases}
         \textrm{hbond}(d) = \begin{cases} 1 & d < -0.7 \\ (1.0/.7)(0 - d) & \textrm{otherwise} \\ 0 & d > 0 \end{cases}

    The free energy of binding is computed as a function of the intermolecular interactions

    ..math:: s = g(c_\textrm{inter})

    This function is defined as

    ..math:: g(c) = \frac{c}{1 + wN_\textrm{rot}}

    Where :math:`w` is a weight parameter and :math:`N_\textrm{rot}` is the number of
    rotatable bonds between heavy atoms in the ligand.

    Gradients are taken backwards through the binding-free energy function with
    respect to the position of the ligand and with respect to the torsions of
    rotatable bonds and flexible ligands.

    TODO(rbharath): It's not clear to me how the effect of the torsions on the :math:`d_{ij}` is
    computed. Is there a way to get distances from torsions?

    The idea is that mutations are applied to the ligand, and then gradient descent is
    used to optimize starting from the initial structure. The code to compute the mutations
    is specified

    https://github.com/mwojcikowski/smina/blob/master/src/lib/mutate.cpp

    Seems to do random quaternion rotations of the ligand. It's not clear to me yet
    how the flexible and rotatable bonds are handled for the system.

    Need to know an initial search space for the compound. Typically a cubic
    binding box.

    References
    ----------
    Autodock Vina Paper:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3041641/
    Smina Paper:
    http://pubs.acs.org/doi/pdf/10.1021/ci300604z
    Omega Paper (ligand conformation generation):
    http://www.sciencedirect.com/science/article/pii/S1093326302002048
    QuickVina:
    http://www.cil.ntu.edu.sg/Courses/papers/journal/QuickVina.pdf
    """
    pass

  def __init__(self, max_local_steps=10, max_mutations=10):
    self.max_local_steps = max_local_steps
    self.max_mutations = max_mutations
    self.graph = self.construct_graph()
    self.sess = tf.Session(graph=self.graph)


  def construct_graph(self):
    """Builds the computational graph for Vina."""
    # TODO(rbharath): Fill in for real
    return tf.Graph()

  def fit(self, dataset):
    """Fit to actual data."""
    # TODO(rbharath): Add an actual fit method.
    return

  def mutate_conformer(protein, ligand):
    """Performs a mutation on the ligand position."""
    return

  def generate_conformation(self, protein, ligand, max_steps=10):
    """Performs the global search for conformations."""
    best_conf = None
    best_score = np.inf
    conf = self.sample_random_conformation()
    for i in range(max_steps):
      mut_conf = self.mutate_conformer(conf)
      loc_conf = self.gradient_minimize(mut_conf)
      if best_conf is None:
        best_conf = loc_conf
      else:
        loc_score = self.score(loc_conf)
        if loc_score < best_score:
          best_conf = loc_conf
    return best_conf
        
      


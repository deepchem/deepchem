"""
Implements Autodock Vina's pose-generation in tensorflow.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import warnings
import numpy as np
import tensorflow as tf
from deepchem.models import Model
from deepchem.nn import model_ops
import deepchem.utils.rdkit_util as rdkit_util


def compute_neighbor_list(coords, nbr_cutoff, N, M, n_cells, ndim=3, k=5):
  """Computes a neighbor list from atom coordinates.

  Parameters
  ----------
  coords: tf.Tensor
    Shape (N, ndim)
  N: int
    Max number atoms
  M: int
    Max number neighbors
  ndim: int
    Dimensionality of space.
  k: int
    Number of nearest neighbors to pull down.

  Returns
  -------
  nbr_list: tf.Tensor
    Shape (N, M) of atom indices
  """
  start = tf.to_int32(tf.reduce_min(coords))
  stop = tf.to_int32(tf.reduce_max(coords))
  cells = get_cells(start, stop, nbr_cutoff, ndim=ndim)
  # Associate each atom with cell it belongs to. O(N*n_cells)
  # Shape (n_cells, k)
  atoms_in_cells, _ = put_atoms_in_cells(coords, cells, N, n_cells, ndim, k)
  # Shape (N, 1)
  cells_for_atoms = get_cells_for_atoms(coords, cells, N, n_cells, ndim)

  # Associate each cell with its neighbor cells. Assumes periodic boundary
  # conditions, so does wrapround. O(constant)
  # Shape (n_cells, 26)
  neighbor_cells = compute_neighbor_cells(cells, ndim, n_cells)

  # Shape (N, 26)
  neighbor_cells = tf.squeeze(tf.gather(neighbor_cells, cells_for_atoms))

  # coords of shape (N, ndim)
  # Shape (N, 26, k, ndim)
  tiled_coords = tf.tile(tf.reshape(coords, (N, 1, 1, ndim)), (1, 26, k, 1))

  # Shape (N, 26, k)
  nbr_inds = tf.gather(atoms_in_cells, neighbor_cells)

  # Shape (N, 26, k)
  atoms_in_nbr_cells = tf.gather(atoms_in_cells, neighbor_cells)

  # Shape (N, 26, k, ndim)
  nbr_coords = tf.gather(coords, atoms_in_nbr_cells)

  # For smaller systems especially, the periodic boundary conditions can
  # result in neighboring cells being seen multiple times. Maybe use tf.unique to
  # make sure duplicate neighbors are ignored?

  # TODO(rbharath): How does distance need to be modified here to
  # account for periodic boundary conditions?
  # Shape (N, 26, k)
  dists = tf.reduce_sum((tiled_coords - nbr_coords)**2, axis=3)

  # Shape (N, 26*k)
  dists = tf.reshape(dists, [N, -1])

  # TODO(rbharath): This will cause an issue with duplicates!
  # Shape (N, M)
  closest_nbr_locs = tf.nn.top_k(dists, k=M)[1]

  # N elts of size (M,) each
  split_closest_nbr_locs = [
      tf.squeeze(locs) for locs in tf.split(closest_nbr_locs, N)
  ]

  # Shape (N, 26*k)
  nbr_inds = tf.reshape(nbr_inds, [N, -1])

  # N elts of size (26*k,) each
  split_nbr_inds = [tf.squeeze(split) for split in tf.split(nbr_inds, N)]

  # N elts of size (M,) each
  neighbor_list = [
      tf.gather(nbr_inds, closest_nbr_locs)
      for (nbr_inds,
           closest_nbr_locs) in zip(split_nbr_inds, split_closest_nbr_locs)
  ]

  # Shape (N, M)
  neighbor_list = tf.stack(neighbor_list)

  return neighbor_list


def get_cells_for_atoms(coords, cells, N, n_cells, ndim=3):
  """Compute the cells each atom belongs to.

  Parameters
  ----------
  coords: tf.Tensor
    Shape (N, ndim)
  cells: tf.Tensor
    (box_size**ndim, ndim) shape.
  Returns
  -------
  cells_for_atoms: tf.Tensor
    Shape (N, 1)
  """
  n_cells = int(n_cells)
  # Tile both cells and coords to form arrays of size (n_cells*N, ndim)
  tiled_cells = tf.tile(cells, (N, 1))
  # N tensors of shape (n_cells, 1)
  tiled_cells = tf.split(tiled_cells, N)

  # Shape (N*n_cells, 1) after tile
  tiled_coords = tf.reshape(tf.tile(coords, (1, n_cells)), (n_cells * N, ndim))
  # List of N tensors of shape (n_cells, 1)
  tiled_coords = tf.split(tiled_coords, N)

  # Lists of length N
  coords_rel = [
      tf.to_float(coords) - tf.to_float(cells)
      for (coords, cells) in zip(tiled_coords, tiled_cells)
  ]
  coords_norm = [tf.reduce_sum(rel**2, axis=1) for rel in coords_rel]

  # Lists of length n_cells
  # Get indices of k atoms closest to each cell point
  closest_inds = [tf.nn.top_k(-norm, k=1)[1] for norm in coords_norm]

  # TODO(rbharath): tf.stack for tf 1.0
  return tf.stack(closest_inds)


def compute_closest_neighbors(coords,
                              cells,
                              atoms_in_cells,
                              neighbor_cells,
                              N,
                              n_cells,
                              ndim=3,
                              k=5):
  """Computes nearest neighbors from neighboring cells.

  TODO(rbharath): Make this pass test

  Parameters
  ---------
  atoms_in_cells: list
    Of length n_cells. Each entry tensor of shape (k, ndim)
  neighbor_cells: tf.Tensor 
    Of shape (n_cells, 26).
  N: int
    Number atoms
  """
  n_cells = int(n_cells)
  # Tensor of shape (n_cells, k, ndim)
  #atoms_in_cells = tf.stack(atoms_in_cells)

  cells_for_atoms = get_cells_for_atoms(coords, cells, N, n_cells, ndim)
  all_closest = []
  for atom in range(N):
    atom_vec = coords[atom]
    cell = cells_for_atoms[atom]
    nbr_inds = tf.gather(neighbor_cells, tf.to_int32(cell))
    # Tensor of shape (26, k, ndim)
    nbr_atoms = tf.gather(atoms_in_cells, nbr_inds)
    # Reshape to (26*k, ndim)
    nbr_atoms = tf.reshape(nbr_atoms, (-1, 3))
    # Subtract out atom vector. Still of shape (26*k, ndim) due to broadcast.
    nbr_atoms = nbr_atoms - atom_vec
    # Dists of shape (26*k, 1)
    nbr_dists = tf.reduce_sum(nbr_atoms**2, axis=1)
    # Of shape (k, ndim)
    closest_inds = tf.nn.top_k(nbr_dists, k=k)[1]
    all_closest.append(closest_inds)
  return all_closest


def get_cells(start, stop, nbr_cutoff, ndim=3):
  """Returns the locations of all grid points in box.

  Suppose start is -10 Angstrom, stop is 10 Angstrom, nbr_cutoff is 1.
  Then would return a list of length 20^3 whose entries would be
  [(-10, -10, -10), (-10, -10, -9), ..., (9, 9, 9)]

  Returns
  -------
  cells: tf.Tensor
    (box_size**ndim, ndim) shape.
  """
  return tf.reshape(
      tf.transpose(
          tf.stack(
              tf.meshgrid(
                  * [tf.range(start, stop, nbr_cutoff) for _ in range(ndim)]))),
      (-1, ndim))


def put_atoms_in_cells(coords, cells, N, n_cells, ndim, k=5):
  """Place each atom into cells. O(N) runtime.    
  
  Let N be the number of atoms.
      
  Parameters    
  ----------    
  coords: tf.Tensor 
    (N, 3) shape.
  cells: tf.Tensor
    (n_cells, ndim) shape.
  N: int
    Number atoms
  ndim: int
    Dimensionality of input space
  k: int
    Number of nearest neighbors.

  Returns
  -------
  closest_atoms: tf.Tensor 
    Of shape (n_cells, k, ndim)
  """
  n_cells = int(n_cells)
  # Tile both cells and coords to form arrays of size (n_cells*N, ndim)
  tiled_cells = tf.reshape(tf.tile(cells, (1, N)), (n_cells * N, ndim))
  # TODO(rbharath): Change this for tf 1.0
  # n_cells tensors of shape (N, 1)
  tiled_cells = tf.split(tiled_cells, n_cells)

  # Shape (N*n_cells, 1) after tile
  tiled_coords = tf.tile(coords, (n_cells, 1))
  # List of n_cells tensors of shape (N, 1)
  tiled_coords = tf.split(tiled_coords, n_cells)

  # Lists of length n_cells
  coords_rel = [
      tf.to_float(coords) - tf.to_float(cells)
      for (coords, cells) in zip(tiled_coords, tiled_cells)
  ]
  coords_norm = [tf.reduce_sum(rel**2, axis=1) for rel in coords_rel]

  # Lists of length n_cells
  # Get indices of k atoms closest to each cell point
  closest_inds = [tf.nn.top_k(norm, k=k)[1] for norm in coords_norm]
  # n_cells tensors of shape (k, ndim)
  closest_atoms = tf.stack([tf.gather(coords, inds) for inds in closest_inds])
  # Tensor of shape (n_cells, k)
  closest_inds = tf.stack(closest_inds)

  return closest_inds, closest_atoms

  # TODO(rbharath):
  #   - Need to find neighbors of the cells (+/- 1 in every dimension).
  #   - Need to group closest atoms amongst cell neighbors
  #   - Need to do another top_k to find indices of closest neighbors.
  #   - Return N lists corresponding to neighbors for every atom.


def compute_neighbor_cells(cells, ndim, n_cells):
  """Compute neighbors of cells in grid.    

  # TODO(rbharath): Do we need to handle periodic boundary conditions
  properly here?
  # TODO(rbharath): This doesn't handle boundaries well. We hard-code
  # looking for 26 neighbors, which isn't right for boundary cells in
  # the cube.
      
  Note n_cells is box_size**ndim. 26 is the number of neighbors of a cube in
  a grid (including diagonals).

  Parameters    
  ----------    
  cells: tf.Tensor
    (n_cells, 26) shape.
  """
  n_cells = int(n_cells)
  if ndim != 3:
    raise ValueError("Not defined for dimensions besides 3")
  # Number of neighbors of central cube in 3-space is
  # 3^2 (top-face) + 3^2 (bottom-face) + (3^2-1) (middle-band)
  # TODO(rbharath)
  k = 9 + 9 + 8  # (26 faces on Rubik's cube for example)
  #n_cells = int(cells.get_shape()[0])
  # Tile cells to form arrays of size (n_cells*n_cells, ndim)
  # Two tilings (a, b, c, a, b, c, ...) vs. (a, a, a, b, b, b, etc.)
  # Tile (a, a, a, b, b, b, etc.)
  tiled_centers = tf.reshape(
      tf.tile(cells, (1, n_cells)), (n_cells * n_cells, ndim))
  # Tile (a, b, c, a, b, c, ...)
  tiled_cells = tf.tile(cells, (n_cells, 1))

  # Lists of n_cells tensors of shape (N, 1)
  tiled_centers = tf.split(tiled_centers, n_cells)
  tiled_cells = tf.split(tiled_cells, n_cells)

  # Lists of length n_cells
  coords_rel = [
      tf.to_float(cells) - tf.to_float(centers)
      for (cells, centers) in zip(tiled_centers, tiled_cells)
  ]
  coords_norm = [tf.reduce_sum(rel**2, axis=1) for rel in coords_rel]

  # Lists of length n_cells
  # Get indices of k atoms closest to each cell point
  # n_cells tensors of shape (26,)
  closest_inds = tf.stack([tf.nn.top_k(norm, k=k)[1] for norm in coords_norm])

  return closest_inds


def cutoff(d, x):
  """Truncates interactions that are too far away."""
  return tf.where(d < 8, x, tf.zeros_like(x))


def gauss_1(d):
  """Computes first Gaussian interaction term.

  Note that d must be in Angstrom
  """
  return tf.exp(-(d / 0.5)**2)


def gauss_2(d):
  """Computes second Gaussian interaction term.

  Note that d must be in Angstrom.
  """
  return tf.exp(-((d - 3) / 2)**2)


def repulsion(d):
  """Computes repulsion interaction term."""
  return tf.where(d < 0, d**2, tf.zeros_like(d))


def hydrophobic(d):
  """Compute hydrophobic interaction term."""
  return tf.where(d < 0.5,
                  tf.ones_like(d), tf.where(d < 1.5, 1.5 - d, tf.zeros_like(d)))


def hbond(d):
  """Computes hydrogen bond term."""
  return tf.where(d < -0.7,
                  tf.ones_like(d),
                  tf.where(d < 0, (1.0 / 0.7) * (0 - d), tf.zeros_like(d)))


def g(c, Nrot):
  """Nonlinear function mapping interactions to free energy."""
  w = tf.Variable(tf.random_normal([
      1,
  ], stddev=.3))
  return c / (1 + w * Nrot)


def h(d):
  """Sum of energy terms used in Autodock Vina.

  .. math:: h_{t_i,t_j}(d) = w_1\textrm{gauss}_1(d) + w_2\textrm{gauss}_2(d) + w_3\textrm{repulsion}(d) + w_4\textrm{hydrophobic}(d) + w_5\textrm{hbond}(d)

  """
  w_1 = tf.Variable(tf.random_normal([
      1,
  ], stddev=.3))
  w_2 = tf.Variable(tf.random_normal([
      1,
  ], stddev=.3))
  w_3 = tf.Variable(tf.random_normal([
      1,
  ], stddev=.3))
  w_4 = tf.Variable(tf.random_normal([
      1,
  ], stddev=.3))
  w_5 = tf.Variable(tf.random_normal([
      1,
  ], stddev=.3))
  return w_1 * gauss_1(d) + w_2 * gauss_2(d) + w_3 * repulsion(
      d) + w_4 * hydrophobic(d) + w_5 * hbond(d)


class VinaModel(Model):

  def __init__(self, logdir=None, batch_size=50):
    """Vina models.

    .. math:: c = \sum_{i < j} f_{t_i,t_j}(r_{ij})

    Over all pairs of atoms that can move relative to one-another. :math:`t_i` is the
    atomtype of atom :math:`i`.

    Can view as

    .. math:: c = c_\textrm{inter} + c_\textrm{intra}

    depending on whether atoms can move relative to one another. Free energy is
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
    warnings.warn("VinaModel is deprecated. "
                  "Will be removed in DeepChem 1.4.", DeprecationWarning)
    self.max_local_steps = max_local_steps
    self.max_mutations = max_mutations
    self.graph, self.input_placeholders, self.output_placeholder = self.construct_graph(
    )
    self.sess = tf.Session(graph=self.graph)

  def construct_graph(self,
                      N_protein=1000,
                      N_ligand=100,
                      M=50,
                      ndim=3,
                      k=5,
                      nbr_cutoff=6):
    """Builds the computational graph for Vina."""
    graph = tf.Graph()
    with graph.as_default():
      n_cells = 64
      # TODO(rbharath): Make this handle minibatches
      protein_coords_placeholder = tf.placeholder(
          tf.float32, shape=(N_protein, 3))
      ligand_coords_placeholder = tf.placeholder(
          tf.float32, shape=(N_ligand, 3))
      protein_Z_placeholder = tf.placeholder(tf.int32, shape=(N_protein,))
      ligand_Z_placeholder = tf.placeholder(tf.int32, shape=(N_ligand,))

      label_placeholder = tf.placeholder(tf.float32, shape=(1,))

      # Shape (N_protein+N_ligand, 3)
      coords = tf.concat(
          [protein_coords_placeholder, ligand_coords_placeholder], axis=0)
      # Shape (N_protein+N_ligand,)
      Z = tf.concat([protein_Z_placeholder, ligand_Z_placeholder], axis=0)

      # Shape (N_protein+N_ligand, M)
      nbr_list = compute_neighbor_list(
          coords, nbr_cutoff, N_protein + N_ligand, M, n_cells, ndim=ndim, k=k)
      all_interactions = []

      # Shape (N_protein+N_ligand,)
      all_atoms = tf.range(N_protein + N_ligand)
      # Shape (N_protein+N_ligand, 3)
      atom_coords = tf.gather(coords, all_atoms)
      # Shape (N_protein+N_ligand,)
      atom_Z = tf.gather(Z, all_atoms)
      # Shape (N_protein+N_ligand, M)
      nbrs = tf.squeeze(tf.gather(nbr_list, all_atoms))
      # Shape (N_protein+N_ligand, M, 3)
      nbr_coords = tf.gather(coords, nbrs)

      # Shape (N_protein+N_ligand, M)
      nbr_Z = tf.gather(Z, nbrs)
      # Shape (N_protein+N_ligand, M, 3)
      tiled_atom_coords = tf.tile(
          tf.reshape(atom_coords, (N_protein + N_ligand, 1, 3)), (1, M, 1))

      # Shape (N_protein+N_ligand, M)
      dists = tf.reduce_sum((tiled_atom_coords - nbr_coords)**2, axis=2)

      # TODO(rbharath): Need to subtract out Van-der-Waals radii from dists

      # Shape (N_protein+N_ligand, M)
      atom_interactions = h(dists)
      # Shape (N_protein+N_ligand, M)
      cutoff_interactions = cutoff(dists, atom_interactions)

      # TODO(rbharath): Use RDKit to compute number of rotatable bonds in ligand.
      Nrot = 1

      # TODO(rbharath): Autodock Vina only uses protein-ligand interactions in
      # computing free-energy. This implementation currently uses all interaction
      # terms. Not sure if this makes a difference.

      # Shape (N_protein+N_ligand, M)
      free_energy = g(cutoff_interactions, Nrot)
      # Shape () -- scalar
      energy = tf.reduce_sum(atom_interactions)

      loss = 0.5 * (energy - label_placeholder)**2

    return (graph, (protein_coords_placeholder, protein_Z_placeholder,
                    ligand_coords_placeholder, ligand_Z_placeholder),
            label_placeholder)

  def fit(self, X_protein, Z_protein, X_ligand, Z_ligand, y):
    """Fit to actual data."""
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

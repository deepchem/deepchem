"""
Atomic coordinate featurizer.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Joseph Gomes"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "LGPL v2.1+"

import numpy as np
from deepchem.featurizers import Featurizer

class AtomicCoordinates(Featurizer):
  """
  Nx3 matrix of Cartesian coordinates [Angstrom]
  """
  name = ['atomic_coordinates']

  def _featurize(self, mol):
    """
    Calculate atomic coodinates.

    Parameters
    ----------
    mol : RDKit Mol
          Molecule.
    """

    N = mol.GetNumAtoms()
    coords = np.zeros((N,3))

    # RDKit stores atomic coordinates in Angstrom. Atomic unit of length is the
    # bohr (1 bohr = 0.529177 Angstrom). Converting units makes gradient calculation
    # consistent with most QM software packages.
    coords_in_bohr = [mol.GetConformer(0).GetAtomPosition(i).__div__(0.52917721092)
                      for i in xrange(N)]

    for atom in xrange(N):
      coords[atom,0] = coords_in_bohr[atom].x
      coords[atom,1] = coords_in_bohr[atom].y
      coords[atom,2] = coords_in_bohr[atom].z

    coords = [coords]
    return coords

def get_cells(coords, neighbor_cutoff):
  """Computes cells given molecular coordinates."""
  x_max, x_min = np.amax(coords[:, 0]), np.amin(coords[:, 0])
  y_max, y_min = np.amax(coords[:, 1]), np.amin(coords[:, 1])
  z_max, z_min = np.amax(coords[:, 2]), np.amin(coords[:, 2])

  # Compute cells for this molecule. O(constant)
  x_bins, y_bins, z_bins = [], [], []
  x_current, y_current, z_current = x_min, y_min, z_min
  while x_current < x_max:
    x_bins.append((x_current, x_current+neighbor_cutoff))
    x_current += neighbor_cutoff
  while y_current < y_max:
    y_bins.append((y_current, y_current+neighbor_cutoff))
    y_current += neighbor_cutoff
  while z_current < z_max:
    z_bins.append((z_current, z_current+neighbor_cutoff))
    z_current += neighbor_cutoff
  return x_bins, y_bins, z_bins

def put_atoms_in_cells(coords, x_bins, y_bins, z_bins):
  """Place each atom into cells. O(N) runtime.
  
  Parameters
  ----------
  coords: np.ndarray
    (N, 3) array where N is number of atoms
  x_bins: list
    List of (cell_start, cell_end) for x-coordinate
  y_bins: list
    List of (cell_start, cell_end) for y-coordinate
  z_bins: list
    List of (cell_start, cell_end) for z-coordinate
  """
  N = coords.shape[0]
  cell_to_atoms = {}
  atom_to_cell = {}
  for x_ind in range(len(x_bins)):
    for y_ind in range(len(y_bins)):
      for z_ind in range(len(z_bins)):
        cell_to_atoms[(x_ind, y_ind, z_ind)] = []
    
  for atom in range(N):
    x_coord, y_coord, z_coord = coords[atom]
    x_ind, y_ind, z_ind = None, None, None
    for ind, (x_cell_min, x_cell_max) in enumerate(x_bins):
      if x_coord >= x_cell_min and x_coord <= x_cell_max:
        x_ind = ind
        break
    if x_ind is None:
      raise ValueError("No x-cell found!")
    for ind, (y_cell_min, y_cell_max) in enumerate(y_bins):
      if y_coord >= y_cell_min and y_coord <= y_cell_max:
        y_ind = ind
        break
    if y_ind is None:
      raise ValueError("No y-cell found!")
    for ind, (z_cell_min, z_cell_max) in enumerate(z_bins):
      if z_coord >= z_cell_min and z_coord <= z_cell_max:
        z_ind = ind
        break
    if z_ind is None:
      raise ValueError("No z-cell found!")
    cell_to_atoms[(x_ind, y_ind, z_ind)].append(atom)
    atom_to_cell[atom] = (x_ind, y_ind, z_ind)
  return cell_to_atoms, atom_to_cell

def compute_neighbor_cell_map(N_x, N_y, N_z):
  """Compute neighbors of cells in grid.
  
  Parameters
  ----------
  N_x: int
    Number of grid cells in x-dimension.
  N_y: int
    Number of grid cells in y-dimension.
  N_z: int
    Number of grid cells in z-dimension.
  """
  neighbor_cell_map = {} 
  for x_ind in range(N_x):
    for y_ind in range(N_y):
      for z_ind in range(N_z):
        neighbors = []
        offsets = [-1, 0, +1]
        # Note neighbors contains self!
        for x_offset in offsets:
          for y_offset in offsets:
            for z_offset in offsets:
              neighbors.append(((x_ind+x_offset) % N_x,
                                (y_ind+y_offset) % N_y,
                                (z_ind+z_offset) % N_z))
        neighbor_cell_map[(x_ind, y_ind, z_ind)] = neighbors
  return neighbor_cell_map

def get_coords(mol):
  """
  Gets coordinates in Angstrom for RDKit mol.
  """
  N = mol.GetNumAtoms()
  coords = np.zeros((N,3))

  coords_raw = [mol.GetConformer(0).GetAtomPosition(i) for i in range(N)]
  for atom in range(N):
    coords[atom,0] = coords_raw[atom].x
    coords[atom,1] = coords_raw[atom].y
    coords[atom,2] = coords_raw[atom].z
  return coords

class NeighborListAtomicCoordinates(Featurizer):
  """
  Adjacency List of neighbors in 3-space

  Neighbors determined by user-defined distance cutoff [in Angstrom].

  https://en.wikipedia.org/wiki/Cell_list
  Ref: http://www.cs.cornell.edu/ron/references/1989/Calculations%20of%20a%20List%20of%20Neighbors%20in%20Molecular%20Dynamics%20Si.pdf

  Parameters
  ----------
  neighbor_cutoff: int
    Threshold distance [Angstroms] for counting neighbors.
  """ 

  def __init__(self, max_num_neighbors=None, neighbor_cutoff=4):
    if neighbor_cutoff <= 0:
      raise ValueError("neighbor_cutoff must be positive value.")
    if max_num_neighbors is not None:
      if not isinstance(max_num_neighbors, int) or max_num_neighbors <= 0:
        raise ValueError("max_num_neighbors must be positive integer.")
    self.max_num_neighbors = max_num_neighbors
    self.neighbor_cutoff = neighbor_cutoff
    # Type of data created by this featurizer
    self.dtype = object
    self.coordinates_featurizer = AtomicCoordinates()

  def _featurize(self, mol):
    """
    Compute neighbor list.

    Parameters
    ----------
    """
    N = mol.GetNumAtoms()
    # TODO(rbharath): Should this return a list?
    bohr_coords = self.coordinates_featurizer._featurize(mol)[0]
    coords = get_coords(mol)

    x_bins, y_bins, z_bins = get_cells(coords, self.neighbor_cutoff)

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
          if dist < self.neighbor_cutoff:
            neighbor_list[atom].add((neighbor_atom, dist))
          
      # Sort neighbors by distance
      closest_neighbors = sorted(
          list(neighbor_list[atom]), key=lambda elt: elt[1])
      closest_neighbors = [nbr for (nbr, dist) in closest_neighbors]
      # Pick up to max_num_neighbors
      closest_neighbors = closest_neighbors[:self.max_num_neighbors]
      neighbor_list[atom] = closest_neighbors

        
    return (bohr_coords, neighbor_list)

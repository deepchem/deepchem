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

  def __init__(self, neighbor_cutoff=4):
    self.neighbor_cutoff = 4

  def _featurize(self, mol):
    """
    Compute neighbor list.

    Parameters
    ----------
    """
    N = mol.GetNumAtoms()
    coords = np.zeros((N,3))

    coords_raw = [mol.GetConformer(0).GetAtomPosition(i) for i in xrange(N)]
    for atom in xrange(N):
      coords[atom,0] = coords_raw[atom].x
      coords[atom,1] = coords_raw[atom].y
      coords[atom,2] = coords_raw[atom].z

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
      y_bins.append((y_current, y_current+neighbor_cutoff)
      y_current += neighbor_cutoff
    while z_current < z_max:
      z_bins.append((z_current, z_current+neighbor_cutoff)
      z_current += neighbor_cutoff

    # Associate each atom with cell it belongs to. O(N)
    cell_lists = {}
    atom_to_cell = {}
    for x_ind in range(len(x_bins)):
      for y_ind in range(len(y_bins)):
        for z_ind in range(len(z_bins)):
          cell_lists[(x_ind, y_ind, z_ind)] = []
      
    for atom in xrange(N):
      x_coord, y_coord, z_coord = coords[atom]
      x_ind, y_ind, z_ind = None, None, None
      for ind, (x_cell_min, x_cell_max) in enumerate(x_bins):
        if x_coord >= x_cell_min and x_coord < x_cell_max:
          x_ind = ind
          break
      if x_ind is None:
        raise ValueError("No x-cell found!")
      for ind, (y_cell_min, y_cell_max) in enumerate(y_bins):
        if y_coord >= y_cell_min and y_coord < y_cell_max:
          y_ind = ind
          break
      if y_ind is None:
        raise ValueError("No y-cell found!")
      for ind, (z_cell_min, z_cell_max) in enumerate(z_bins):
        if z_coord >= z_cell_min and z_coord < z_cell_max:
          z_ind = ind
          break
      if z_ind is None:
        raise ValueError("No z-cell found!")
      cell_lists[(x_ind, y_ind, z_ind)].append(atom)
      atom_to_cell[atom] = (x_ind, y_ind, z_ind)

    # Associate each cell with its neighbor cells. Assumes periodic boundary
    # conditions, so does wrapround. O(constant)
    neighbor_cells = {} 
    N_x, N_y, N_z = len(x_bins), len(y_bins), len(z_bins)
    for x_ind in range(N_x):
      for y_ind in range(N_y):
        for z_ind in range(N_z):
          neighbors = []
          offsets = [-1, 0, +1]
          for x_offset in offsets:
            for y_offset in offsets:
              for z_offset in offsets:
                neighbors.append(((x_ind+offset) % N_x,
                                  (y_ind+offset) % N_y,
                                  (z_ind+offset) % N_z))

    # 

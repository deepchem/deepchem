"""
Atomic coordinate featurizer.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Joseph Gomes and Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "LGPL v2.1+"

import numpy as np
import mdtraj
from deepchem.feat import Featurizer
from deepchem.feat import ComplexFeaturizer
from deepchem.feat.grid_featurizer import load_molecule 
from deepchem.feat.grid_featurizer import merge_molecules

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
    coords_in_bohr = [mol.GetConformer(0).GetAtomPosition(i).__idiv__(0.52917721092)
                      for i in range(N)]

    for atom in range(N):
      coords[atom,0] = coords_in_bohr[atom].x
      coords[atom,1] = coords_in_bohr[atom].y
      coords[atom,2] = coords_in_bohr[atom].z

    coords = [coords]
    return coords

def compute_neighbor_list(coords, neighbor_cutoff, max_num_neighbors, periodic_box_size):
  """Computes a neighbor list from atom coordinates."""
  N = coords.shape[0]
  traj = mdtraj.Trajectory(coords.reshape((1, N, 3)), None)
  box_size = None
  if periodic_box_size is not None:
    box_size = np.array(periodic_box_size)
    traj.unitcell_vectors = np.array([[[box_size[0], 0, 0], [0, box_size[1], 0], [0, 0, box_size[2]]]], dtype=np.float32)
  neighbors = mdtraj.geometry.compute_neighborlist(traj, neighbor_cutoff)
  neighbor_list = {}
  for i in range(N):
    if max_num_neighbors is not None and len(neighbors[i]) > max_num_neighbors:
      delta = coords[i]-coords.take(neighbors[i], axis=0)
      if box_size is not None:
        delta -= np.round(delta/box_size)*box_size
      dist = np.linalg.norm(delta, axis=1)
      sorted_neighbors = list(zip(dist, neighbors[i]))
      sorted_neighbors.sort()
      neighbor_list[i] = [sorted_neighbors[j][1] for j in range(max_num_neighbors)]
    else:
      neighbor_list[i] = list(neighbors[i])
  return neighbor_list

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
  neighbor_cutoff: float
    Threshold distance [Angstroms] for counting neighbors.
  periodic_box_size: 3 element array
    Dimensions of the periodic box in Angstroms, or None to not use periodic boundary conditions
  """ 

  def __init__(self, max_num_neighbors=None, neighbor_cutoff=4, periodic_box_size=None):
    if neighbor_cutoff <= 0:
      raise ValueError("neighbor_cutoff must be positive value.")
    if max_num_neighbors is not None:
      if not isinstance(max_num_neighbors, int) or max_num_neighbors <= 0:
        raise ValueError("max_num_neighbors must be positive integer.")
    self.max_num_neighbors = max_num_neighbors
    self.neighbor_cutoff = neighbor_cutoff
    self.periodic_box_size = periodic_box_size
    # Type of data created by this featurizer
    self.dtype = object
    self.coordinates_featurizer = AtomicCoordinates()

  def _featurize(self, mol):
    """
    Compute neighbor list.

    Parameters
    ----------
      mol: rdkit Mol
        To be featurized.
    """
    N = mol.GetNumAtoms()
    # TODO(rbharath): Should this return a list?
    bohr_coords = self.coordinates_featurizer._featurize(mol)[0]
    coords = get_coords(mol)
    neighbor_list = compute_neighbor_list(coords, self.neighbor_cutoff, self.max_num_neighbors, self.periodic_box_size)
    return (bohr_coords, neighbor_list)

class NeighborListComplexAtomicCoordinates(ComplexFeaturizer):
  """
  Adjacency list of neighbors for protein-ligand complexes in 3-space.

  Neighbors dtermined by user-dfined distance cutoff.
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

  def _featurize_complex(self, mol_pdb_file, protein_pdb_file):
    """
    Compute neighbor list for complex.

    Parameters
    ----------
    mol_pdb: list
      Should be a list of lines of the PDB file.
    complex_pdb: list
      Should be a list of lines of the PDB file.
    """
    mol_coords, mol = load_molecule(mol_pdb_file)
    protein_coords, protein_mol = load_molecule(protein_pdb_file)
    system_coords, system_mol = merge_molecules(
        mol_coords, mol, protein_coords, protein_mol)
    
    system_neighbor_list = compute_neighbor_list(
        system_coords, self.neighbor_cutoff, self.max_num_neighbors, None)

    return (system_coords, system_neighbor_list)

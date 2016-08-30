"""
Test atomic coordinates and neighbor lists.
"""
import numpy as np
import unittest
from rdkit import Chem
from deepchem.utils import conformers
from deepchem.featurizers.atomic_coordinates import get_cells
from deepchem.featurizers.atomic_coordinates import put_atoms_in_cells
from deepchem.featurizers.atomic_coordinates import AtomicCoordinates
from deepchem.featurizers.atomic_coordinates import NeighborListAtomicCoordinates

class TestAtomicCoordinates(unittest.TestCase):
  """
  Test AtomicCoordinates.
  """
  def setUp(self):
    """
    Set up tests.
    """
    smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
    mol = Chem.MolFromSmiles(smiles)
    engine = conformers.ConformerGenerator(max_conformers=1)
    self.mol = engine.generate_conformers(mol)
    assert self.mol.GetNumConformers() > 0

  def test_atomic_coordinates(self):
    """
    Simple test that atomic coordinates returns ndarray of right shape.
    """
    N = self.mol.GetNumAtoms()
    atomic_coords_featurizer = AtomicCoordinates()
    # TODO(rbharath, joegomes): Why does AtomicCoordinates return a list? Is
    # this expected behavior? Need to think about API.
    coords = atomic_coords_featurizer._featurize(self.mol)[0]
    assert isinstance(coords, np.ndarray)
    assert coords.shape == (N, 3)

  def test_get_cells(self):
    """
    Test that coordinates are split into cell appropriately.
    """
    # The coordinates span the cube of side-length 2 set on (-1, 1)
    coords = np.array(
        [[1., 1., 1.],
         [-1., -1., -1.]])
    # Set cell size (neighbor_cutoff) at 1 angstrom.
    neighbor_cutoff = 1
    # We should get 2 bins in each dimension
    x_bins, y_bins, z_bins = get_cells(coords, neighbor_cutoff)

    # Check bins are lists
    assert isinstance(x_bins, list)
    assert isinstance(y_bins, list)
    assert isinstance(z_bins, list)

    assert len(x_bins) == 2
    assert x_bins ==[(-1.0, 0.0), (0.0, 1.0)] 
    assert len(y_bins) == 2
    assert y_bins ==[(-1.0, 0.0), (0.0, 1.0)] 
    assert len(z_bins) == 2
    assert z_bins ==[(-1.0, 0.0), (0.0, 1.0)] 


  def test_put_atoms_in_cells(self):
    """
    Test that atoms are placed into correct cells.
    """
    # As in previous example, coordinates span size-2 cube on (-1, 1)
    coords = np.array(
        [[1., 1., 1.],
         [-1., -1., -1.]])
    # Set cell size (neighbor_cutoff) at 1 angstrom.
    neighbor_cutoff = 1
    # We should get 2 bins in each dimension
    x_bins, y_bins, z_bins = get_cells(coords, neighbor_cutoff)

    cell_to_atoms, atom_to_cell = put_atoms_in_cells(
        coords, x_bins, y_bins, z_bins)

    # Both cell_to_atoms and atom_to_cell are dictionaries
    assert isinstance(cell_to_atoms, dict)
    assert isinstance(atom_to_cell, dict)

    # atom_to_cell should be of len 2 since 2 atoms
    assert len(atom_to_cell) == 2

    # cell_to_atoms should be of len 8 since 8 cells total.
    assert len(cell_to_atoms) == 8

    # We have two atoms. The first is in highest corner (1,1,1)
    # Second atom should be in lowest corner (0, 0, 0)
    assert atom_to_cell[0] == (1, 1, 1)
    assert atom_to_cell[1] == (0, 0, 0)

    # (1,1,1) should contain atom 0. (0, 0, 0) should contain atom 1.
    # Everything else should be an empty list
    for cell, atoms in cell_to_atoms.items():
      if cell == (1, 1, 1):
        assert atoms == [0]
      elif cell == (0, 0, 0):
        assert atoms == [1]
      else:
        assert atoms == []

  def test_neighbor_list_shape(self):
    """
    Simple test that Neighbor Lists have right shape.
    """
    N = self.mol.GetNumAtoms()
    nblist_featurizer = NeighborListAtomicCoordinates()
    nblist = nblist_featurizer._featurize(self.mol)
    assert isinstance(nblist, dict)
    assert len(nblist.keys()) == N
    print("nblist")
    print(nblist)
    for (atom, neighbors) in nblist.items():
      assert isinstance(atom, int)
      assert isinstance(neighbors, list)
      assert len(neighbors) <= N

  def test_neighbor_list_extremes(self):
    """
    Test Neighbor Lists with large/small boxes.
    """
    pass

    


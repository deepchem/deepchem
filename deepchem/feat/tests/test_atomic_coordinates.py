"""
Test atomic coordinates and neighbor lists.
"""
import os
import numpy as np
import unittest
from rdkit import Chem
from deepchem.utils import conformers
from deepchem.feat.atomic_coordinates import get_coords
from deepchem.feat.atomic_coordinates import NeighborListAtomicCoordinates
from deepchem.feat.atomic_coordinates import ComplexNeighborListFragmentAtomicCoordinates
from deepchem.utils import rdkit_util


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
    self.atomic_coords_featurizer = NeighborListAtomicCoordinates(max_num_atoms=1000, max_num_neighbors=10,
                                                         neighbor_cutoff=12.0)

  def test_atomic_coordinates(self):
    """
    Simple test that atomic coordinates returns ndarray of right shape.
    """
    N = 1000
    # TODO(rbharath, joegomes): Why does AtomicCoordinates return a list? Is
    # this expected behavior? Need to think about API.
    coords = self.atomic_coords_featurizer._featurize(self.mol)[0]
    assert isinstance(coords, np.ndarray)
    assert coords.shape == (self.atomic_coords_featurizer.max_num_atoms, 3)

  def test_neighbor_list_shape(self):
    """
    Simple test that Neighbor Lists have right shape.
    """
    N = self.mol.GetNumAtoms()
    coords = get_coords(self.mol)
    nblist = self.atomic_coords_featurizer._featurize(self.mol)[1]
    assert isinstance(nblist, dict)
    assert len(nblist.keys()) == N
    for (atom, neighbors) in nblist.items():
      assert isinstance(atom, int)
      assert isinstance(neighbors, list)
      assert len(neighbors) <= N

    # Do a manual distance computation and make 
    for i in range(N):
      for j in range(N):
        dist = np.linalg.norm(coords[i] - coords[j])
        print("Distance(%d, %d) = %f" % (i, j, dist))
        if dist < self.atomic_coords_featurizer.neighbor_cutoff and i != j:
          assert j in nblist[i]
        else:
          assert j not in nblist[i]

  def test_neighbor_list_extremes(self):
    """
    Test Neighbor Lists with large/small boxes.
    """
    N = self.mol.GetNumAtoms()

    # Test with cutoff 0 angstroms. There should be no neighbors in this case.
    nblist_featurizer = NeighborListAtomicCoordinates(max_num_atoms=1000, max_num_neighbors=10, neighbor_cutoff=.1)
    nblist = nblist_featurizer._featurize(self.mol)[1]
    for atom in range(N):
      assert len(nblist[atom]) == 0

    # Test with cutoff 100 angstroms. Everything should be neighbors now.
    nblist_featurizer = NeighborListAtomicCoordinates(max_num_atoms=1000, max_num_neighbors=1000, neighbor_cutoff=100)
    nblist = nblist_featurizer._featurize(self.mol)[1]
    for atom in range(N):
      assert len(nblist[atom]) == N - 1

  def test_neighbor_list_max_num_neighbors(self):
    """
    Test that neighbor lists return only max_num_neighbors.
    """
    N = self.mol.GetNumAtoms()

    max_num_neighbors = 1
    nblist_featurizer = NeighborListAtomicCoordinates(max_num_atoms=1000,
                                                      max_num_neighbors=max_num_neighbors,
                                                      neighbor_cutoff=100)
    nblist = nblist_featurizer._featurize(self.mol)[1]

    for atom in range(N):
      assert len(nblist[atom]) <= max_num_neighbors

    # Do a manual distance computation and ensure that selected neighbor is
    # closest since we set max_num_neighbors = 1
    coords = get_coords(self.mol)
    for i in range(N):
      closest_dist = np.inf
      closest_nbr = None
      for j in range(N):
        if i == j:
          continue
        dist = np.linalg.norm(coords[i] - coords[j])
        print("Distance(%d, %d) = %f" % (i, j, dist))
        if dist < closest_dist:
          closest_dist = dist
          closest_nbr = j
      print("Closest neighbor to %d is %d" % (i, closest_nbr))
      print("Distance: %f" % closest_dist)
      if closest_dist < nblist_featurizer.neighbor_cutoff:
        assert nblist[i] == [closest_nbr]
      else:
        assert nblist[i] == []

  def test_complex_featurization_simple(self):
    """Test Neighbor List computation on protein-ligand complex."""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    ligand_file = os.path.join(dir_path, "data/3zso_ligand_hyd.pdb")
    protein_file = os.path.join(dir_path, "data/3zso_protein.pdb")
    max_num_neighbors = 4
    complex_featurizer = ComplexNeighborListFragmentAtomicCoordinates(50, 3000, 3500, 10)

    m1 = rdkit_util.load_molecule(ligand_file, add_hydrogens=False, calc_charges=False)
    m2 = rdkit_util.load_molecule(ligand_file, add_hydrogens=False, calc_charges=False)
    _, _, _, _, _, _, system_coords, system_neighbor_list, _ = complex_featurizer._featurize_complex(
      ligand_file, protein_file)

    N = system_coords.shape[0]
    assert len(system_neighbor_list.keys()) == complex_featurizer.complex_num_atoms
    for atom in range(N):
      assert len(system_neighbor_list[atom]) <= max_num_neighbors

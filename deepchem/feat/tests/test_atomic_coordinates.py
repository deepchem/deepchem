"""
Test atomic coordinates and neighbor lists.
"""
import os
import logging
import numpy as np
import unittest
from deepchem.utils import conformers
from deepchem.feat.atomic_coordinates import get_coords
from deepchem.feat.atomic_coordinates import AtomicCoordinates
from deepchem.feat.atomic_coordinates import NeighborListAtomicCoordinates
from deepchem.feat.atomic_coordinates import NeighborListComplexAtomicCoordinates
from deepchem.feat.atomic_coordinates import ComplexNeighborListFragmentAtomicCoordinates

logger = logging.getLogger(__name__)


class TestAtomicCoordinates(unittest.TestCase):
  """
  Test AtomicCoordinates.
  """

  def setUp(self):
    """
    Set up tests.
    """
    smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
    from rdkit import Chem
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

  def test_neighbor_list_shape(self):
    """
    Simple test that Neighbor Lists have right shape.
    """
    nblist_featurizer = NeighborListAtomicCoordinates()
    N = self.mol.GetNumAtoms()
    coords = get_coords(self.mol)

    nblist_featurizer = NeighborListAtomicCoordinates()
    nblist = nblist_featurizer._featurize(self.mol)[1]
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
        logger.info("Distance(%d, %d) = %f" % (i, j, dist))
        if dist < nblist_featurizer.neighbor_cutoff and i != j:
          assert j in nblist[i]
        else:
          assert j not in nblist[i]

  def test_neighbor_list_extremes(self):
    """
    Test Neighbor Lists with large/small boxes.
    """
    N = self.mol.GetNumAtoms()

    # Test with cutoff 0 angstroms. There should be no neighbors in this case.
    nblist_featurizer = NeighborListAtomicCoordinates(neighbor_cutoff=.1)
    nblist = nblist_featurizer._featurize(self.mol)[1]
    for atom in range(N):
      assert len(nblist[atom]) == 0

    # Test with cutoff 100 angstroms. Everything should be neighbors now.
    nblist_featurizer = NeighborListAtomicCoordinates(neighbor_cutoff=100)
    nblist = nblist_featurizer._featurize(self.mol)[1]
    for atom in range(N):
      assert len(nblist[atom]) == N - 1

  def test_neighbor_list_max_num_neighbors(self):
    """
    Test that neighbor lists return only max_num_neighbors.
    """
    N = self.mol.GetNumAtoms()

    max_num_neighbors = 1
    nblist_featurizer = NeighborListAtomicCoordinates(max_num_neighbors)
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
        logger.info("Distance(%d, %d) = %f" % (i, j, dist))
        if dist < closest_dist:
          closest_dist = dist
          closest_nbr = j
      logger.info("Closest neighbor to %d is %d" % (i, closest_nbr))
      logger.info("Distance: %f" % closest_dist)
      if closest_dist < nblist_featurizer.neighbor_cutoff:
        assert nblist[i] == [closest_nbr]
      else:
        assert nblist[i] == []

  def test_neighbor_list_periodic(self):
    """Test building a neighbor list with periodic boundary conditions."""
    cutoff = 4.0
    box_size = np.array([10.0, 8.0, 9.0])
    N = self.mol.GetNumAtoms()
    coords = get_coords(self.mol)
    featurizer = NeighborListAtomicCoordinates(
        neighbor_cutoff=cutoff, periodic_box_size=box_size)
    neighborlist = featurizer._featurize(self.mol)[1]
    expected_neighbors = [set() for i in range(N)]
    for i in range(N):
      for j in range(i):
        delta = coords[i] - coords[j]
        delta -= np.round(delta / box_size) * box_size
        if np.linalg.norm(delta) < cutoff:
          expected_neighbors[i].add(j)
          expected_neighbors[j].add(i)
    for i in range(N):
      assert (set(neighborlist[i]) == expected_neighbors[i])

  def test_complex_featurization_simple(self):
    """Test Neighbor List computation on protein-ligand complex."""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    ligand_file = os.path.join(dir_path, "data/3zso_ligand_hyd.pdb")
    protein_file = os.path.join(dir_path, "data/3zso_protein.pdb")
    max_num_neighbors = 4
    complex_featurizer = NeighborListComplexAtomicCoordinates(max_num_neighbors)

    system_coords, system_neighbor_list = complex_featurizer._featurize_complex(
        ligand_file, protein_file)

    N = system_coords.shape[0]
    assert len(system_neighbor_list.keys()) == N
    for atom in range(N):
      assert len(system_neighbor_list[atom]) <= max_num_neighbors


# TODO(rbharath): This test will be uncommented in the next PR up on the docket.
#  def test_full_complex_featurization(self):
#    """Unit test for ComplexNeighborListFragmentAtomicCoordinates."""
#    dir_path = os.path.dirname(os.path.realpath(__file__))
#    ligand_file = os.path.join(dir_path, "data/3zso_ligand_hyd.pdb")
#    protein_file = os.path.join(dir_path, "data/3zso_protein.pdb")
#    # Pulled from PDB files. For larger datasets with more PDBs, would use
#    # max num atoms instead of exact.
#    frag1_num_atoms = 44  # for ligand atoms
#    frag2_num_atoms = 2336  # for protein atoms
#    complex_num_atoms = 2380  # in total
#    max_num_neighbors = 4
#    # Cutoff in angstroms
#    neighbor_cutoff = 4
#    complex_featurizer = ComplexNeighborListFragmentAtomicCoordinates(
#        frag1_num_atoms, frag2_num_atoms, complex_num_atoms, max_num_neighbors,
#        neighbor_cutoff)
#    (frag1_coords, frag1_neighbor_list, frag1_z, frag2_coords,
#     frag2_neighbor_list, frag2_z, complex_coords,
#     complex_neighbor_list, complex_z) = complex_featurizer._featurize_complex(
#         ligand_file, protein_file)
#
#    assert frag1_coords.shape == (frag1_num_atoms, 3)
#    self.assertEqual(
#        sorted(list(frag1_neighbor_list.keys())), list(range(frag1_num_atoms)))
#    self.assertEqual(frag1_z.shape, (frag1_num_atoms,))
#
#    self.assertEqual(frag2_coords.shape, (frag2_num_atoms, 3))
#    self.assertEqual(
#        sorted(list(frag2_neighbor_list.keys())), list(range(frag2_num_atoms)))
#    self.assertEqual(frag2_z.shape, (frag2_num_atoms,))
#
#    self.assertEqual(complex_coords.shape, (complex_num_atoms, 3))
#    self.assertEqual(
#        sorted(list(complex_neighbor_list.keys())),
#        list(range(complex_num_atoms)))
#    self.assertEqual(complex_z.shape, (complex_num_atoms,))

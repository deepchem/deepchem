"""
Test rdkit_grid_featurizer module.
"""
import os
import unittest

import numpy as np
np.random.seed(123)

from rdkit.Chem import MolFromMolFile

from deepchem.feat import rdkit_grid_featurizer as rgf


class TestHelperFunctions(unittest.TestCase):
  """
  Test functions defined in rdkit_grid_featurizer module.
  """
  def setUp(self):
    # TODO test more formats for ligand
    current_dir = os.path.dirname(os.path.realpath(__file__))
    package_dir = os.path.dirname(os.path.dirname(current_dir))
    self.protein_file = os.path.join(package_dir, 'dock', 'tests', '1jld_protein.pdb')
    self.ligand_file = os.path.join(package_dir, 'dock', 'tests', '1jld_ligand.sdf')

  def test_get_ligand_filetype(self):

    supported_extensions = ['mol2', 'sdf', 'pdb', 'pdbqt']
    # some users might try to read smiles with this function
    unsupported_extensions = ['smi', 'ism']

    for extension in supported_extensions:
      fname = 'molecule.%s' % extension
      self.assertEqual(rgf.get_ligand_filetype(fname), extension)

    for extension in unsupported_extensions:
      fname = 'molecule.%s' % extension
      self.assertRaises(ValueError, rgf.get_ligand_filetype, fname)

  def test_load_molecule(self):
    for add_hydrogens in (True, False):
      for calc_charges in (True, False):
        mol_xyz, mol_rdk = rgf.load_molecule(self.ligand_file, add_hydrogens, calc_charges)
        num_atoms = mol_rdk.GetNumAtoms()
        self.assertEqual(mol_xyz.shape, (num_atoms, 3))

  def test_generate_random__unit_vector(self):
    for _ in range(100):
      u = rgf.generate_random__unit_vector()
      self.assertEqual(u.shape, (3,))
      self.assertAlmostEqual(np.linalg.norm(u), 1.0)

  def test_generate_random_rotation_matrix(self):
    for _ in range(100):
      m = rgf.generate_random_rotation_matrix()
      self.assertEqual(m.shape, (3, 3))

  def test_rotate_molecules(self):
    # check if distances do not change
    vectors = np.random.rand(4, 2, 3)
    norms = np.linalg.norm(vectors[:, 1] - vectors[:, 0], axis=1)
    vectors_rot = np.array(rgf.rotate_molecules(vectors))
    norms_rot = np.linalg.norm(vectors_rot[:, 1] - vectors_rot[:, 0], axis=1)
    self.assertTrue(np.allclose(norms, norms_rot))

    # check if it works for molecules with different numbers of atoms
    coords = [np.random.rand(n, 3) for n in (10, 20, 40, 100)]
    coords_rot = rgf.rotate_molecules(coords)
    self.assertEqual(len(coords), len(coords_rot))

  def test_compute_pairwise_distances(self):
    n1 = 10
    n2 = 50
    coords1 = np.random.rand(n1, 3)
    coords2 = np.random.rand(n2, 3)

    distance = rgf.compute_pairwise_distances(coords1, coords2)
    self.assertTrue((distance >= 0).all())
    # random coords between 0 and 1, so the max possible distance in sqrt(2)
    self.assertTrue((distance <= 2.0 ** 0.5).all())

  def test_unit_vector(self):
    for _ in range(10):
      vector = np.random.rand(3)
      norm_vector = rgf.unit_vector(vector)
      self.assertAlmostEqual(np.linalg.norm(norm_vector), 1.0)

  def test_angle_between(self):
    for _ in range(10):
      v1 = np.random.rand(3,)
      v2 = np.random.rand(3,)
      angle = rgf.angle_between(v1, v2)
      self.assertLessEqual(angle, np.pi)
      self.assertGreaterEqual(angle, 0.0)

  def test_hash_ecfp(self):
    for power in (2, 16, 64):
      for _ in range(10):
        # FIXME strings generation is not controlled by random seed
        string = os.urandom(10).decode('latin1')
        string_hash = rgf.hash_ecfp(string, power)
        self.assertLess(string_hash, 2**power)
        self.assertGreaterEqual(string_hash, 0)

  def test_hash_ecfp_pair(self):
    for power in (2, 16, 64):
      for _ in range(10):
        # FIXME strings generation is not controlled by random seed
        string1 = os.urandom(10).decode('latin1')
        string2 = os.urandom(10).decode('latin1')
        pair_hash = rgf.hash_ecfp_pair((string1, string2), power)
        self.assertLess(pair_hash, 2**power)
        self.assertGreaterEqual(pair_hash, 0)

  def test_compute_all_ecfp(self):
    mol = MolFromMolFile(self.ligand_file)
    num_atoms = mol.GetNumAtoms()
    for degree in range(1, 4):
      # TODO test if dict contains smiles

      ecfp_all = rgf.compute_all_ecfp(mol, degree=degree)
      self.assertIsInstance(ecfp_all, dict)
      self.assertEqual(len(ecfp_all), num_atoms)
      self.assertEqual(list(ecfp_all.keys()), list(range(num_atoms)))

      num_ind = np.random.choice(range(1, num_atoms))
      indices = list(np.random.choice(range(num_atoms), num_ind, replace=False))

      ecfp_selected = rgf.compute_all_ecfp(mol, indices=indices, degree=degree)
      print(indices, list(ecfp_selected.keys()))
      self.assertIsInstance(ecfp_selected, dict)
      self.assertEqual(len(ecfp_selected), num_ind)
      self.assertEqual(sorted(ecfp_selected.keys()), sorted(indices))

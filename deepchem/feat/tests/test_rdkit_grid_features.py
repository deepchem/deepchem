"""
Test rdkit_grid_featurizer module.
"""
import os
from six import integer_types
import unittest

import numpy as np
np.random.seed(123)

from rdkit.Chem import MolFromMolFile
from rdkit.Chem.AllChem import Mol, ComputeGasteigerCharges

from deepchem.feat import rdkit_grid_featurizer as rgf


def random_string(length, chars=None):
  import string
  if chars is None:
    chars = list(string.ascii_letters + string.ascii_letters + '()[]+-.=#@/\\')
  return ''.join(np.random.choice(chars, length))


class TestHelperFunctions(unittest.TestCase):
  """
  Test functions defined in rdkit_grid_featurizer module.
  """

  def setUp(self):
    # TODO test more formats for ligand
    current_dir = os.path.dirname(os.path.realpath(__file__))
    package_dir = os.path.dirname(os.path.dirname(current_dir))
    self.protein_file = os.path.join(package_dir, 'dock', 'tests',
                                     '1jld_protein.pdb')
    self.ligand_file = os.path.join(package_dir, 'dock', 'tests',
                                    '1jld_ligand.sdf')

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
    # adding hydrogens and charges is tested in dc.utils
    for add_hydrogens in (True, False):
      for calc_charges in (True, False):
        mol_xyz, mol_rdk = rgf.load_molecule(self.ligand_file, add_hydrogens,
                                             calc_charges)
        num_atoms = mol_rdk.GetNumAtoms()
        self.assertIsInstance(mol_xyz, np.ndarray)
        self.assertIsInstance(mol_rdk, Mol)
        self.assertEqual(mol_xyz.shape, (num_atoms, 3))

  def test_generate_random__unit_vector(self):
    for _ in range(100):
      u = rgf.generate_random__unit_vector()
      # 3D vector with unit length
      self.assertEqual(u.shape, (3,))
      self.assertAlmostEqual(np.linalg.norm(u), 1.0)

  def test_generate_random_rotation_matrix(self):
    # very basic test, we check if rotations actually work in test_rotate_molecules
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
    self.assertEqual(distance.shape, (n1, n2))
    self.assertTrue((distance >= 0).all())
    # random coords between 0 and 1, so the max possible distance in sqrt(2)
    self.assertTrue((distance <= 2.0**0.5).all())

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
        string = random_string(10)
        string_hash = rgf.hash_ecfp(string, power)
        self.assertIsInstance(string_hash, integer_types)
        self.assertLess(string_hash, 2**power)
        self.assertGreaterEqual(string_hash, 0)

  def test_hash_ecfp_pair(self):
    for power in (2, 16, 64):
      for _ in range(10):
        string1 = random_string(10)
        string2 = random_string(10)
        pair_hash = rgf.hash_ecfp_pair((string1, string2), power)
        self.assertIsInstance(pair_hash, integer_types)
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
      indices = list(np.random.choice(num_atoms, num_ind, replace=False))

      ecfp_selected = rgf.compute_all_ecfp(mol, indices=indices, degree=degree)
      self.assertIsInstance(ecfp_selected, dict)
      self.assertEqual(len(ecfp_selected), num_ind)
      self.assertEqual(sorted(ecfp_selected.keys()), sorted(indices))

  def test_featurize_binding_pocket_ecfp(self):
    prot_xyz, prot_rdk = rgf.load_molecule(self.protein_file)
    lig_xyz, lig_rdk = rgf.load_molecule(self.ligand_file)
    distance = rgf.compute_pairwise_distances(
        protein_xyz=prot_xyz, ligand_xyz=lig_xyz)

    # check if results are the same if we provide precomputed distances
    prot_dict, lig_dict = rgf.featurize_binding_pocket_ecfp(
        prot_xyz,
        prot_rdk,
        lig_xyz,
        lig_rdk,)
    prot_dict_dist, lig_dict_dist = rgf.featurize_binding_pocket_ecfp(
        prot_xyz, prot_rdk, lig_xyz, lig_rdk, pairwise_distances=distance)
    # ...but first check if we actually got two dicts
    self.assertIsInstance(prot_dict, dict)
    self.assertIsInstance(lig_dict, dict)

    self.assertEqual(prot_dict, prot_dict_dist)
    self.assertEqual(lig_dict, lig_dict_dist)

    # check if we get less features with smaller distance cutoff
    prot_dict_d2, lig_dict_d2 = rgf.featurize_binding_pocket_ecfp(
        prot_xyz,
        prot_rdk,
        lig_xyz,
        lig_rdk,
        cutoff=2.0,)
    prot_dict_d6, lig_dict_d6 = rgf.featurize_binding_pocket_ecfp(
        prot_xyz,
        prot_rdk,
        lig_xyz,
        lig_rdk,
        cutoff=6.0,)
    self.assertLess(len(prot_dict_d2), len(prot_dict))
    # ligands are typically small so all atoms might be present
    self.assertLessEqual(len(lig_dict_d2), len(lig_dict))
    self.assertGreater(len(prot_dict_d6), len(prot_dict))
    self.assertGreaterEqual(len(lig_dict_d6), len(lig_dict))

    # check if using different ecfp_degree changes anything
    prot_dict_e3, lig_dict_e3 = rgf.featurize_binding_pocket_ecfp(
        prot_xyz,
        prot_rdk,
        lig_xyz,
        lig_rdk,
        ecfp_degree=3,)
    self.assertNotEqual(prot_dict_e3, prot_dict)
    self.assertNotEqual(lig_dict_e3, lig_dict)

  def test_compute_splif_features_in_range(self):
    prot_xyz, prot_rdk = rgf.load_molecule(self.protein_file)
    lig_xyz, lig_rdk = rgf.load_molecule(self.ligand_file)
    prot_num_atoms = prot_rdk.GetNumAtoms()
    lig_num_atoms = lig_rdk.GetNumAtoms()
    distance = rgf.compute_pairwise_distances(
        protein_xyz=prot_xyz, ligand_xyz=lig_xyz)

    for bins in ((0, 2), (2, 3)):
      splif_dict = rgf.compute_splif_features_in_range(
          prot_rdk,
          lig_rdk,
          distance,
          bins,)

      self.assertIsInstance(splif_dict, dict)
      for (prot_idx, lig_idx), ecfp_pair in splif_dict.items():

        for idx in (prot_idx, lig_idx):
          self.assertIsInstance(idx, (int, np.int64))
        self.assertGreaterEqual(prot_idx, 0)
        self.assertLess(prot_idx, prot_num_atoms)
        self.assertGreaterEqual(lig_idx, 0)
        self.assertLess(lig_idx, lig_num_atoms)

        for ecfp in ecfp_pair:
          ecfp_idx, ecfp_frag = ecfp.split(',')
          ecfp_idx = int(ecfp_idx)
          self.assertGreaterEqual(ecfp_idx, 0)
          # TODO upperbound?

  def test_featurize_splif(self):
    prot_xyz, prot_rdk = rgf.load_molecule(self.protein_file)
    lig_xyz, lig_rdk = rgf.load_molecule(self.ligand_file)
    distance = rgf.compute_pairwise_distances(
        protein_xyz=prot_xyz, ligand_xyz=lig_xyz)

    bins = [(1, 2), (2, 3)]

    dicts = rgf.featurize_splif(
        prot_xyz,
        prot_rdk,
        lig_xyz,
        lig_rdk,
        contact_bins=bins,
        pairwise_distances=distance,
        ecfp_degree=2)
    expected_dicts = [
        rgf.compute_splif_features_in_range(
            prot_rdk, lig_rdk, distance, c_bin, ecfp_degree=2) for c_bin in bins
    ]
    self.assertIsInstance(dicts, list)
    self.assertEqual(dicts, expected_dicts)

  def test_convert_atom_to_voxel(self):
    # 20 points with coords between -5 and 5, centered at 0
    coords_range = 10
    xyz = (np.random.rand(20, 3) - 0.5) * coords_range
    for idx in np.random.choice(20, 6):
      for box_width in (10, 20, 40):
        for voxel_width in (0.5, 1, 2):
          voxel = rgf.convert_atom_to_voxel(xyz, idx, box_width, voxel_width)
          self.assertIsInstance(voxel, list)
          self.assertEqual(len(voxel), 1)
          self.assertIsInstance(voxel[0], np.ndarray)
          self.assertEqual(voxel[0].shape, (3,))
          self.assertIs(voxel[0].dtype, np.dtype('int'))
          # indices are positive
          self.assertTrue((voxel[0] >= 0).all())
          # coordinates were properly translated and scaled
          self.assertTrue(
              (voxel[0] < (box_width + coords_range) / 2.0 / voxel_width).all())
          self.assertTrue(
              np.allclose(voxel[0],
                          np.floor((xyz[idx] + box_width / 2.0) / voxel_width)))

    # for coordinates outside of the box function should properly transform them
    # to indices and warn the user
    for args in ((np.array([[0, 1, 6]]), 0, 10, 1.0), (np.array([[0, 4, -6]]),
                                                       0, 10, 1.0)):
      # TODO check if function warns. There is assertWarns method in unittest,
      # but it is not implemented in 2.7 and buggy in 3.5 (issue 29620)
      voxel = rgf.convert_atom_to_voxel(*args)
      self.assertTrue(
          np.allclose(voxel[0], np.floor((args[0] + args[2] / 2.0) / args[3])))

  def test_convert_atom_pair_to_voxel(self):
    # 20 points with coords between -5 and 5, centered at 0
    coords_range = 10
    xyz1 = (np.random.rand(20, 3) - 0.5) * coords_range
    xyz2 = (np.random.rand(20, 3) - 0.5) * coords_range
    # 3 pairs of indices
    for idx1, idx2 in np.random.choice(20, (3, 2)):
      for box_width in (10, 20, 40):
        for voxel_width in (0.5, 1, 2):
          v1 = rgf.convert_atom_to_voxel(xyz1, idx1, box_width, voxel_width)
          v2 = rgf.convert_atom_to_voxel(xyz2, idx2, box_width, voxel_width)
          v_pair = rgf.convert_atom_pair_to_voxel((xyz1, xyz2), (idx1, idx2),
                                                  box_width, voxel_width)
          self.assertEqual(len(v_pair), 2)
          self.assertTrue((v1 == v_pair[0]).all())
          self.assertTrue((v2 == v_pair[1]).all())

  def test_compute_charge_dictionary(self):
    for fname in (self.ligand_file, self.protein_file):
      _, mol = rgf.load_molecule(fname)
      ComputeGasteigerCharges(mol)
      charge_dict = rgf.compute_charge_dictionary(mol)
      self.assertEqual(len(charge_dict), mol.GetNumAtoms())
      for i in range(mol.GetNumAtoms()):
        self.assertIn(i, charge_dict)
        self.assertIsInstance(charge_dict[i], (float, int))


class TestRdkitGridFeaturizer(unittest.TestCase):
  """
  Test RdkitGridFeaturizer class defined in rdkit_grid_featurizer module.
  """

  def setUp(self):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    package_dir = os.path.dirname(os.path.dirname(current_dir))
    self.protein_file = os.path.join(package_dir, 'dock', 'tests',
                                     '1jld_protein.pdb')
    self.ligand_file = os.path.join(package_dir, 'dock', 'tests',
                                    '1jld_ligand.sdf')

  def test_voxelize(self):
    prot_xyz, prot_rdk = rgf.load_molecule(self.protein_file)
    lig_xyz, lig_rdk = rgf.load_molecule(self.ligand_file)

    centroid = rgf.compute_centroid(lig_xyz)
    prot_xyz = rgf.subtract_centroid(prot_xyz, centroid)
    lig_xyz = rgf.subtract_centroid(lig_xyz, centroid)

    prot_ecfp_dict, lig_ecfp_dict = (rgf.featurize_binding_pocket_ecfp(
        prot_xyz, prot_rdk, lig_xyz, lig_rdk))

    box_w = 20
    f_power = 5

    rgf_featurizer = rgf.RdkitGridFeaturizer(
        box_width=box_w, ecfp_power=f_power)

    prot_tensor = rgf_featurizer._voxelize(
        rgf.convert_atom_to_voxel,
        rgf.hash_ecfp,
        prot_xyz,
        feature_dict=prot_ecfp_dict,
        channel_power=f_power)
    self.assertEqual(prot_tensor.shape, tuple([box_w] * 3 + [2**f_power]))
    all_features = prot_tensor.sum()
    # protein is too big for the box, some features should be missing
    self.assertGreater(all_features, 0)
    self.assertLess(all_features, prot_rdk.GetNumAtoms())

    lig_tensor = rgf_featurizer._voxelize(
        rgf.convert_atom_to_voxel,
        rgf.hash_ecfp,
        lig_xyz,
        feature_dict=lig_ecfp_dict,
        channel_power=f_power)
    self.assertEqual(lig_tensor.shape, tuple([box_w] * 3 + [2**f_power]))
    all_features = lig_tensor.sum()
    # whole ligand should fit in the box
    self.assertEqual(all_features, lig_rdk.GetNumAtoms())

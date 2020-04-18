import tempfile
import unittest
import os
import shutil

import numpy as np

from deepchem.utils import rdkit_util


class TestRdkitUtil(unittest.TestCase):

  def setUp(self):
    # TODO test more formats for ligand
    current_dir = os.path.dirname(os.path.realpath(__file__))
    self.protein_file = os.path.join(
        current_dir, '../../feat/tests/data/3ws9_protein_fixer_rdkit.pdb')
    self.ligand_file = os.path.join(current_dir,
                                    '../../feat/tests/data/3ws9_ligand.sdf')

  def test_load_complex(self):
    complexes = rdkit_util.load_complex(
        (self.protein_file, self.ligand_file),
        add_hydrogens=False,
        calc_charges=False)
    assert len(complexes) == 2

  def test_load_molecule(self):
    # adding hydrogens and charges is tested in dc.utils
    from rdkit.Chem.AllChem import Mol
    for add_hydrogens in (True, False):
      for calc_charges in (True, False):
        mol_xyz, mol_rdk = rdkit_util.load_molecule(self.ligand_file,
                                                    add_hydrogens, calc_charges)
        num_atoms = mol_rdk.GetNumAtoms()
        self.assertIsInstance(mol_xyz, np.ndarray)
        self.assertIsInstance(mol_rdk, Mol)
        self.assertEqual(mol_xyz.shape, (num_atoms, 3))

  def test_get_xyz_from_mol(self):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    ligand_file = os.path.join(current_dir, "../../dock/tests/1jld_ligand.sdf")

    xyz, mol = rdkit_util.load_molecule(
        ligand_file, calc_charges=False, add_hydrogens=False)
    xyz2 = rdkit_util.get_xyz_from_mol(mol)

    equal_array = np.all(xyz == xyz2)
    assert equal_array

  def test_add_hydrogens_to_mol(self):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    ligand_file = os.path.join(current_dir, "../../dock/tests/1jld_ligand.sdf")
    xyz, mol = rdkit_util.load_molecule(
        ligand_file, calc_charges=False, add_hydrogens=False)
    original_hydrogen_count = 0
    for atom_idx in range(mol.GetNumAtoms()):
      atom = mol.GetAtoms()[atom_idx]
      if atom.GetAtomicNum() == 1:
        original_hydrogen_count += 1

    assert mol is not None
    mol = rdkit_util.add_hydrogens_to_mol(mol, is_protein=False)
    assert mol is not None
    after_hydrogen_count = 0
    for atom_idx in range(mol.GetNumAtoms()):
      atom = mol.GetAtoms()[atom_idx]
      if atom.GetAtomicNum() == 1:
        after_hydrogen_count += 1
    assert after_hydrogen_count >= original_hydrogen_count

  def test_apply_pdbfixer(self):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    ligand_file = os.path.join(current_dir, "../../dock/tests/1jld_ligand.sdf")
    xyz, mol = rdkit_util.load_molecule(
        ligand_file, calc_charges=False, add_hydrogens=False)
    original_hydrogen_count = 0
    for atom_idx in range(mol.GetNumAtoms()):
      atom = mol.GetAtoms()[atom_idx]
      if atom.GetAtomicNum() == 1:
        original_hydrogen_count += 1

    assert mol is not None
    mol = rdkit_util.apply_pdbfixer(mol, hydrogenate=True, is_protein=False)
    assert mol is not None
    after_hydrogen_count = 0
    for atom_idx in range(mol.GetNumAtoms()):
      atom = mol.GetAtoms()[atom_idx]
      if atom.GetAtomicNum() == 1:
        after_hydrogen_count += 1
    assert after_hydrogen_count >= original_hydrogen_count

  def test_compute_charges(self):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    ligand_file = os.path.join(current_dir, "../../dock/tests/1jld_ligand.sdf")
    xyz, mol = rdkit_util.load_molecule(
        ligand_file, calc_charges=False, add_hydrogens=True)
    rdkit_util.compute_charges(mol)

    has_a_charge = False
    for atom_idx in range(mol.GetNumAtoms()):
      atom = mol.GetAtoms()[atom_idx]
      value = atom.GetProp(str("_GasteigerCharge"))
      if value != 0:
        has_a_charge = True
    assert has_a_charge

  def test_rotate_molecules(self):
    # check if distances do not change
    vectors = np.random.rand(4, 2, 3)
    norms = np.linalg.norm(vectors[:, 1] - vectors[:, 0], axis=1)
    vectors_rot = np.array(rdkit_util.rotate_molecules(vectors))
    norms_rot = np.linalg.norm(vectors_rot[:, 1] - vectors_rot[:, 0], axis=1)
    self.assertTrue(np.allclose(norms, norms_rot))

    # check if it works for molecules with different numbers of atoms
    coords = [np.random.rand(n, 3) for n in (10, 20, 40, 100)]
    coords_rot = rdkit_util.rotate_molecules(coords)
    self.assertEqual(len(coords), len(coords_rot))

  def test_compute_pairwise_distances(self):
    n1 = 10
    n2 = 50
    coords1 = np.random.rand(n1, 3)
    coords2 = np.random.rand(n2, 3)

    distance = rdkit_util.compute_pairwise_distances(coords1, coords2)
    self.assertEqual(distance.shape, (n1, n2))
    self.assertTrue((distance >= 0).all())
    # random coords between 0 and 1, so the max possible distance in sqrt(2)
    self.assertTrue((distance <= 2.0**0.5).all())

    # check if correct distance metric was used
    coords1 = np.array([[0, 0, 0], [1, 0, 0]])
    coords2 = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]])
    distance = rdkit_util.compute_pairwise_distances(coords1, coords2)
    self.assertTrue((distance == [[1, 2, 3], [0, 1, 2]]).all())

  def test_load_molecule(self):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    ligand_file = os.path.join(current_dir, "../../dock/tests/1jld_ligand.sdf")
    xyz, mol = rdkit_util.load_molecule(
        ligand_file, calc_charges=False, add_hydrogens=False)
    assert xyz is not None
    assert mol is not None

  def test_write_molecule(self):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    ligand_file = os.path.join(current_dir, "../../dock/tests/1jld_ligand.sdf")
    xyz, mol = rdkit_util.load_molecule(
        ligand_file, calc_charges=False, add_hydrogens=False)

    with tempfile.TemporaryDirectory() as tmp:
      outfile = os.path.join(tmp, "mol.sdf")
      rdkit_util.write_molecule(mol, outfile)

      xyz, mol2 = rdkit_util.load_molecule(
          outfile, calc_charges=False, add_hydrogens=False)

    assert mol.GetNumAtoms() == mol2.GetNumAtoms()
    for atom_idx in range(mol.GetNumAtoms()):
      atom1 = mol.GetAtoms()[atom_idx]
      atom2 = mol.GetAtoms()[atom_idx]
      assert atom1.GetAtomicNum() == atom2.GetAtomicNum()

  def test_merge_molecules_xyz(self):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    ligand_file = os.path.join(current_dir, "../../dock/tests/1jld_ligand.sdf")
    xyz, mol = rdkit_util.load_molecule(
        ligand_file, calc_charges=False, add_hydrogens=False)
    merged = rdkit_util.merge_molecules_xyz([xyz, xyz])
    for i in range(len(xyz)):
      first_atom_equal = np.all(xyz[i] == merged[i])
      second_atom_equal = np.all(xyz[i] == merged[i + len(xyz)])
      assert first_atom_equal
      assert second_atom_equal

  def test_merge_molecules(self):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    ligand_file = os.path.join(current_dir, "../../dock/tests/1jld_ligand.sdf")
    xyz, mol = rdkit_util.load_molecule(
        ligand_file, calc_charges=False, add_hydrogens=False)
    num_mol_atoms = mol.GetNumAtoms()
    # self.ligand_file is for 3ws9_ligand.sdf
    oth_xyz, oth_mol = rdkit_util.load_molecule(
        self.ligand_file, calc_charges=False, add_hydrogens=False)
    num_oth_mol_atoms = oth_mol.GetNumAtoms()
    merged = rdkit_util.merge_molecules([mol, oth_mol])
    merged_num_atoms = merged.GetNumAtoms()
    assert merged_num_atoms == num_mol_atoms + num_oth_mol_atoms

  def test_merge_molecular_fragments(self):
    pass

  def test_strip_hydrogens(self):
    pass


class TestPiInteractions(unittest.TestCase):

  def setUp(self):
    current_dir = os.path.dirname(os.path.realpath(__file__))

    # simple flat ring
    from rdkit.Chem import MolFromSmiles
    from rdkit.Chem.rdDepictor import Compute2DCoords
    self.cycle4 = MolFromSmiles('C1CCC1')
    #self.cycle4.Compute2DCoords()
    Compute2DCoords(self.cycle4)

    # load and sanitize two real molecules
    _, self.prot = rdkit_util.load_molecule(
        os.path.join(current_dir,
                     '../../feat/tests/3ws9_protein_fixer_rdkit.pdb'),
        add_hydrogens=False,
        calc_charges=False,
        sanitize=True)

    _, self.lig = rdkit_util.load_molecule(
        os.path.join(current_dir, '../../feat/tests/3ws9_ligand.sdf'),
        add_hydrogens=False,
        calc_charges=False,
        sanitize=True)

  def test_compute_ring_center(self):
    self.assertTrue(
        np.allclose(rdkit_util.compute_ring_center(self.cycle4, range(4)), 0))

  def test_compute_ring_normal(self):
    normal = rdkit_util.compute_ring_normal(self.cycle4, range(4))
    self.assertTrue(
        np.allclose(np.abs(normal / np.linalg.norm(normal)), [0, 0, 1]))

  def test_is_pi_parallel(self):
    ring1_center = np.array([0.0, 0.0, 0.0])
    ring2_center_true = np.array([4.0, 0.0, 0.0])
    ring2_center_false = np.array([10.0, 0.0, 0.0])
    ring1_normal_true = np.array([1.0, 0.0, 0.0])
    ring1_normal_false = np.array([0.0, 1.0, 0.0])

    for ring2_normal in (np.array([2.0, 0, 0]), np.array([-3.0, 0, 0])):
      # parallel normals
      self.assertTrue(
          rdkit_util.is_pi_parallel(ring1_center, ring1_normal_true,
                                    ring2_center_true, ring2_normal))
      # perpendicular normals
      self.assertFalse(
          rdkit_util.is_pi_parallel(ring1_center, ring1_normal_false,
                                    ring2_center_true, ring2_normal))
      # too far away
      self.assertFalse(
          rdkit_util.is_pi_parallel(ring1_center, ring1_normal_true,
                                    ring2_center_false, ring2_normal))

  def test_is_pi_t(self):
    ring1_center = np.array([0.0, 0.0, 0.0])
    ring2_center_true = np.array([4.0, 0.0, 0.0])
    ring2_center_false = np.array([10.0, 0.0, 0.0])
    ring1_normal_true = np.array([0.0, 1.0, 0.0])
    ring1_normal_false = np.array([1.0, 0.0, 0.0])

    for ring2_normal in (np.array([2.0, 0, 0]), np.array([-3.0, 0, 0])):
      # perpendicular normals
      self.assertTrue(
          rdkit_util.is_pi_t(ring1_center, ring1_normal_true, ring2_center_true,
                             ring2_normal))
      # parallel normals
      self.assertFalse(
          rdkit_util.is_pi_t(ring1_center, ring1_normal_false,
                             ring2_center_true, ring2_normal))
      # too far away
      self.assertFalse(
          rdkit_util.is_pi_t(ring1_center, ring1_normal_true,
                             ring2_center_false, ring2_normal))

  def test_compute_pi_stack(self):
    # order of the molecules shouldn't matter
    dicts1 = rdkit_util.compute_pi_stack(self.prot, self.lig)
    dicts2 = rdkit_util.compute_pi_stack(self.lig, self.prot)
    for i, j in ((0, 2), (1, 3)):
      self.assertEqual(dicts1[i], dicts2[j])
      self.assertEqual(dicts1[j], dicts2[i])

    # with this criteria we should find both types of stacking
    for d in rdkit_util.compute_pi_stack(
        self.lig, self.prot, dist_cutoff=7, angle_cutoff=40.):
      self.assertGreater(len(d), 0)

  def test_is_cation_pi(self):
    cation_position = np.array([[2.0, 0.0, 0.0]])
    ring_center_true = np.array([4.0, 0.0, 0.0])
    ring_center_false = np.array([10.0, 0.0, 0.0])
    ring_normal_true = np.array([1.0, 0.0, 0.0])
    ring_normal_false = np.array([0.0, 1.0, 0.0])

    # parallel normals
    self.assertTrue(
        rdkit_util.is_cation_pi(cation_position, ring_center_true,
                                ring_normal_true))
    # perpendicular normals
    self.assertFalse(
        rdkit_util.is_cation_pi(cation_position, ring_center_true,
                                ring_normal_false))
    # too far away
    self.assertFalse(
        rdkit_util.is_cation_pi(cation_position, ring_center_false,
                                ring_normal_true))

  def test_compute_cation_pi(self):
    # TODO(rbharath): find better example, currently dicts are empty
    dicts1 = rdkit_util.compute_cation_pi(self.prot, self.lig)
    dicts2 = rdkit_util.compute_cation_pi(self.lig, self.prot)

  def test_compute_binding_pocket_cation_pi(self):
    # TODO find better example, currently dicts are empty
    prot_dict, lig_dict = rdkit_util.compute_binding_pocket_cation_pi(
        self.prot, self.lig)

    exp_prot_dict, exp_lig_dict = rdkit_util.compute_cation_pi(
        self.prot, self.lig)
    add_lig, add_prot = rdkit_util.compute_cation_pi(self.lig, self.prot)
    for exp_dict, to_add in ((exp_prot_dict, add_prot), (exp_lig_dict,
                                                         add_lig)):
      for atom_idx, count in to_add.items():
        if atom_idx not in exp_dict:
          exp_dict[atom_idx] = count
        else:
          exp_dict[atom_idx] += count

    self.assertEqual(prot_dict, exp_prot_dict)
    self.assertEqual(lig_dict, exp_lig_dict)

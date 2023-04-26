import tempfile
import unittest
import os

import numpy as np

from deepchem.utils import rdkit_utils


class TestRdkitUtil(unittest.TestCase):

    def setUp(self):
        # TODO test more formats for ligand
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.protein_file = os.path.join(
            current_dir, '../../feat/tests/data/3ws9_protein_fixer_rdkit.pdb')
        self.ligand_file = os.path.join(
            current_dir, '../../feat/tests/data/3ws9_ligand.sdf')

    def test_load_complex(self):
        complexes = rdkit_utils.load_complex(
            (self.protein_file, self.ligand_file),
            add_hydrogens=False,
            calc_charges=False)
        assert len(complexes) == 2

    def test_load_molecule(self):
        # adding hydrogens and charges is tested in dc.utils
        from rdkit.Chem.AllChem import Mol
        for add_hydrogens in (True, False):
            for calc_charges in (True, False):
                mol_xyz, mol_rdk = rdkit_utils.load_molecule(
                    self.ligand_file, add_hydrogens, calc_charges)
                num_atoms = mol_rdk.GetNumAtoms()
                self.assertIsInstance(mol_xyz, np.ndarray)
                self.assertIsInstance(mol_rdk, Mol)
                self.assertEqual(mol_xyz.shape, (num_atoms, 3))

    def test_get_xyz_from_mol(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        ligand_file = os.path.join(current_dir,
                                   "../../dock/tests/1jld_ligand.sdf")

        xyz, mol = rdkit_utils.load_molecule(ligand_file,
                                             calc_charges=False,
                                             add_hydrogens=False)
        xyz2 = rdkit_utils.get_xyz_from_mol(mol)

        equal_array = np.all(xyz == xyz2)
        assert equal_array

    def test_add_hydrogens_to_mol(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        ligand_file = os.path.join(current_dir,
                                   "../../dock/tests/1jld_ligand.sdf")
        xyz, mol = rdkit_utils.load_molecule(ligand_file,
                                             calc_charges=False,
                                             add_hydrogens=False)
        original_hydrogen_count = 0
        for atom_idx in range(mol.GetNumAtoms()):
            atom = mol.GetAtoms()[atom_idx]
            if atom.GetAtomicNum() == 1:
                original_hydrogen_count += 1

        assert mol is not None
        mol = rdkit_utils.add_hydrogens_to_mol(mol, is_protein=False)
        assert mol is not None
        after_hydrogen_count = 0
        for atom_idx in range(mol.GetNumAtoms()):
            atom = mol.GetAtoms()[atom_idx]
            if atom.GetAtomicNum() == 1:
                after_hydrogen_count += 1
        assert after_hydrogen_count >= original_hydrogen_count

    def test_apply_pdbfixer(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        ligand_file = os.path.join(current_dir,
                                   "../../dock/tests/1jld_ligand.sdf")
        xyz, mol = rdkit_utils.load_molecule(ligand_file,
                                             calc_charges=False,
                                             add_hydrogens=False)
        original_hydrogen_count = 0
        for atom_idx in range(mol.GetNumAtoms()):
            atom = mol.GetAtoms()[atom_idx]
            if atom.GetAtomicNum() == 1:
                original_hydrogen_count += 1

        assert mol is not None
        mol = rdkit_utils.apply_pdbfixer(mol,
                                         hydrogenate=True,
                                         is_protein=False)
        assert mol is not None
        after_hydrogen_count = 0
        for atom_idx in range(mol.GetNumAtoms()):
            atom = mol.GetAtoms()[atom_idx]
            if atom.GetAtomicNum() == 1:
                after_hydrogen_count += 1
        assert after_hydrogen_count >= original_hydrogen_count

    def test_compute_charges(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        ligand_file = os.path.join(current_dir,
                                   "../../dock/tests/1jld_ligand.sdf")
        xyz, mol = rdkit_utils.load_molecule(ligand_file,
                                             calc_charges=False,
                                             add_hydrogens=True)
        rdkit_utils.compute_charges(mol)

        has_a_charge = False
        for atom_idx in range(mol.GetNumAtoms()):
            atom = mol.GetAtoms()[atom_idx]
            value = atom.GetProp(str("_GasteigerCharge"))
            if value != 0:
                has_a_charge = True
        assert has_a_charge

    def test_load_molecule2(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        ligand_file = os.path.join(current_dir,
                                   "../../dock/tests/1jld_ligand.sdf")
        xyz, mol = rdkit_utils.load_molecule(ligand_file,
                                             calc_charges=False,
                                             add_hydrogens=False)
        assert xyz is not None
        assert mol is not None

    def test_write_molecule(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        ligand_file = os.path.join(current_dir,
                                   "../../dock/tests/1jld_ligand.sdf")
        xyz, mol = rdkit_utils.load_molecule(ligand_file,
                                             calc_charges=False,
                                             add_hydrogens=False)

        with tempfile.TemporaryDirectory() as tmp:
            outfile = os.path.join(tmp, "mol.sdf")
            rdkit_utils.write_molecule(mol, outfile)

            xyz, mol2 = rdkit_utils.load_molecule(outfile,
                                                  calc_charges=False,
                                                  add_hydrogens=False)

        assert mol.GetNumAtoms() == mol2.GetNumAtoms()
        for atom_idx in range(mol.GetNumAtoms()):
            atom1 = mol.GetAtoms()[atom_idx]
            atom2 = mol.GetAtoms()[atom_idx]
            assert atom1.GetAtomicNum() == atom2.GetAtomicNum()

    def test_merge_molecules_xyz(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        ligand_file = os.path.join(current_dir,
                                   "../../dock/tests/1jld_ligand.sdf")
        xyz, mol = rdkit_utils.load_molecule(ligand_file,
                                             calc_charges=False,
                                             add_hydrogens=False)
        merged = rdkit_utils.merge_molecules_xyz([xyz, xyz])
        for i in range(len(xyz)):
            first_atom_equal = np.all(xyz[i] == merged[i])
            second_atom_equal = np.all(xyz[i] == merged[i + len(xyz)])
            assert first_atom_equal
            assert second_atom_equal

    def test_merge_molecules(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        ligand_file = os.path.join(current_dir,
                                   "../../dock/tests/1jld_ligand.sdf")
        xyz, mol = rdkit_utils.load_molecule(ligand_file,
                                             calc_charges=False,
                                             add_hydrogens=False)
        num_mol_atoms = mol.GetNumAtoms()
        # self.ligand_file is for 3ws9_ligand.sdf
        oth_xyz, oth_mol = rdkit_utils.load_molecule(self.ligand_file,
                                                     calc_charges=False,
                                                     add_hydrogens=False)
        num_oth_mol_atoms = oth_mol.GetNumAtoms()
        merged = rdkit_utils.merge_molecules([mol, oth_mol])
        merged_num_atoms = merged.GetNumAtoms()
        assert merged_num_atoms == num_mol_atoms + num_oth_mol_atoms

    def test_merge_molecular_fragments(self):
        pass

    def test_strip_hydrogens(self):
        pass

    def test_all_shortest_pairs(self):
        from rdkit import Chem
        mol = Chem.MolFromSmiles("CN=C=O")
        valid_dict = {
            (0, 1): (0, 1),
            (0, 2): (0, 1, 2),
            (0, 3): (0, 1, 2, 3),
            (1, 2): (1, 2),
            (1, 3): (1, 2, 3),
            (2, 3): (2, 3)
        }
        assert rdkit_utils.compute_all_pairs_shortest_path(mol) == valid_dict

    def test_pairwise_ring_info(self):
        from rdkit import Chem
        mol = Chem.MolFromSmiles("c1ccccc1")
        predict_dict = rdkit_utils.compute_pairwise_ring_info(mol)
        assert all(pair == [(6, True)] for pair in predict_dict.values())
        mol = Chem.MolFromSmiles("c1c2ccccc2ccc1")
        predict_dict = rdkit_utils.compute_pairwise_ring_info(mol)
        assert all(pair == [(6, True)] for pair in predict_dict.values())
        mol = Chem.MolFromSmiles("CN=C=O")
        predict_dict = rdkit_utils.compute_pairwise_ring_info(mol)
        assert not predict_dict

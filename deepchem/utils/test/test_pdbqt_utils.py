import unittest
import os
import tempfile
from deepchem.utils import rdkit_utils
from deepchem.utils import pdbqt_utils


class TestPDBQTUtils(unittest.TestCase):

    def setUp(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.protein_file = os.path.join(current_dir,
                                         "../../dock/tests/1jld_protein.pdb")
        self.ligand_file = os.path.join(
            current_dir,
            "../../dock/tests/dichlorophenyl_sulfanyl_methyl_phosponic_acid.sdf"
        )

    def test_pdbqt_to_pdb(self):
        """Test that a PDBQT molecule can be converted back in to PDB."""
        xyz, mol = rdkit_utils.load_molecule(self.protein_file,
                                             calc_charges=False,
                                             add_hydrogens=False)
        with tempfile.TemporaryDirectory() as tmp:
            out_pdb = os.path.join(tmp, "mol.pdb")
            out_pdbqt = os.path.join(tmp, "mol.pdbqt")

            rdkit_utils.write_molecule(mol, out_pdb, is_protein=True)
            rdkit_utils.write_molecule(mol, out_pdbqt, is_protein=True)

            pdb_block = pdbqt_utils.pdbqt_to_pdb(out_pdbqt)
            from rdkit import Chem  # type: ignore
            pdb_mol = Chem.MolFromPDBBlock(pdb_block,
                                           sanitize=False,
                                           removeHs=False)

            xyz, pdbqt_mol = rdkit_utils.load_molecule(out_pdbqt,
                                                       add_hydrogens=False,
                                                       calc_charges=False)

        assert pdb_mol.GetNumAtoms() == pdbqt_mol.GetNumAtoms()
        for atom_idx in range(pdb_mol.GetNumAtoms()):
            atom1 = pdb_mol.GetAtoms()[atom_idx]
            atom2 = pdbqt_mol.GetAtoms()[atom_idx]
            assert atom1.GetAtomicNum() == atom2.GetAtomicNum()

    def test_convert_mol_to_pdbqt(self):
        """Test that a ligand molecule can be coverted to PDBQT."""
        from rdkit import Chem
        xyz, mol = rdkit_utils.load_molecule(self.ligand_file,
                                             calc_charges=False,
                                             add_hydrogens=False)
        with tempfile.TemporaryDirectory() as tmp:
            outfile = os.path.join(tmp, "mol.pdbqt")
            writer = Chem.PDBWriter(outfile)
            writer.write(mol)
            writer.close()
            pdbqt_utils.convert_mol_to_pdbqt(mol, outfile)
            pdbqt_xyz, pdbqt_mol = rdkit_utils.load_molecule(
                outfile, add_hydrogens=False, calc_charges=False)
        assert pdbqt_mol.GetNumAtoms() == pdbqt_mol.GetNumAtoms()
        for atom_idx in range(pdbqt_mol.GetNumAtoms()):
            atom1 = pdbqt_mol.GetAtoms()[atom_idx]
            atom2 = pdbqt_mol.GetAtoms()[atom_idx]
            assert atom1.GetAtomicNum() == atom2.GetAtomicNum()

    def test_convert_protein_to_pdbqt(self):
        """Test a protein in a PDB can be converted to PDBQT."""
        from rdkit import Chem
        xyz, mol = rdkit_utils.load_molecule(self.protein_file,
                                             calc_charges=False,
                                             add_hydrogens=False)
        with tempfile.TemporaryDirectory() as tmp:
            outfile = os.path.join(tmp, "mol.pdbqt")
            writer = Chem.PDBWriter(outfile)
            writer.write(mol)
            writer.close()
            pdbqt_utils.convert_protein_to_pdbqt(mol, outfile)
            pdbqt_xyz, pdbqt_mol = rdkit_utils.load_molecule(
                outfile, add_hydrogens=False, calc_charges=False)
        assert pdbqt_mol.GetNumAtoms() == pdbqt_mol.GetNumAtoms()
        for atom_idx in range(pdbqt_mol.GetNumAtoms()):
            atom1 = pdbqt_mol.GetAtoms()[atom_idx]
            atom2 = pdbqt_mol.GetAtoms()[atom_idx]
            assert atom1.GetAtomicNum() == atom2.GetAtomicNum()

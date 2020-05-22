import unittest
import os
from deepchem.utils import rdkit_util
from deepchem.utils import pdbqt_utils

class TestPDBQTUtils(unittest.TestCase):

  def test_pdbqt_to_pdb(self):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    protein_file = os.path.join(current_dir,
                                "../../dock/tests/1jld_protein.pdb")
    xyz, mol = rdkit_util.load_molecule(
        protein_file, calc_charges=False, add_hydrogens=False)
    with tempfile.TemporaryDirectory() as tmp:
      out_pdb = os.path.join(tmp, "mol.pdb")
      out_pdbqt = os.path.join(tmp, "mol.pdbqt")

      rdkit_util.write_molecule(mol, out_pdb)
      rdkit_util.write_molecule(mol, out_pdbqt, is_protein=True)

      pdb_block = pdbqt_utils.pdbqt_to_pdb(out_pdbqt)
      from rdkit import Chem
      pdb_mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)

      xyz, pdbqt_mol = rdkit_util.load_molecule(
          out_pdbqt, add_hydrogens=False, calc_charges=False)

    assert_equal(pdb_mol.GetNumAtoms(), pdbqt_mol.GetNumAtoms())
    for atom_idx in range(pdb_mol.GetNumAtoms()):
      atom1 = pdb_mol.GetAtoms()[atom_idx]
      atom2 = pdbqt_mol.GetAtoms()[atom_idx]
      assert_equal(atom1.GetAtomicNum(), atom2.GetAtomicNum())

  def test_convert_mol_to_pdbqt(self):
    # TODO
    pass

  def test_convert_protein_to_pdbqt(self):
    # TODO
    pass

  def test_pdbqt_ligand_writer(self):
    # TODO
    pass

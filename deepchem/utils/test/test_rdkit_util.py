import tempfile
import unittest
import os
import shutil

from nose.tools import assert_equal
import numpy as np
from nose.tools import assert_false
from nose.tools import assert_true

from deepchem.utils import rdkit_util
from rdkit import Chem


class TestRdkitUtil(unittest.TestCase):

  def test_get_xyz_from_mol(self):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    ligand_file = os.path.join(current_dir, "../../dock/tests/1jld_ligand.sdf")

    xyz, mol = rdkit_util.load_molecule(
        ligand_file, calc_charges=False, add_hydrogens=False)
    xyz2 = rdkit_util.get_xyz_from_mol(mol)

    equal_array = np.all(xyz == xyz2)
    assert_true(equal_array)

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

    mol = rdkit_util.add_hydrogens_to_mol(mol)
    after_hydrogen_count = 0
    for atom_idx in range(mol.GetNumAtoms()):
      atom = mol.GetAtoms()[atom_idx]
      if atom.GetAtomicNum() == 1:
        after_hydrogen_count += 1
    assert_true(after_hydrogen_count >= original_hydrogen_count)

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
    assert_true(has_a_charge)

  def test_load_molecule(self):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    ligand_file = os.path.join(current_dir, "../../dock/tests/1jld_ligand.sdf")
    xyz, mol = rdkit_util.load_molecule(
        ligand_file, calc_charges=False, add_hydrogens=False)
    assert_true(xyz is not None)
    assert_true(mol is not None)

  def test_write_molecule(self):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    ligand_file = os.path.join(current_dir, "../../dock/tests/1jld_ligand.sdf")
    xyz, mol = rdkit_util.load_molecule(
        ligand_file, calc_charges=False, add_hydrogens=False)

    outfile = "/tmp/mol.sdf"
    rdkit_util.write_molecule(mol, outfile)

    xyz, mol2 = rdkit_util.load_molecule(
        outfile, calc_charges=False, add_hydrogens=False)

    assert_equal(mol.GetNumAtoms(), mol2.GetNumAtoms())
    for atom_idx in range(mol.GetNumAtoms()):
      atom1 = mol.GetAtoms()[atom_idx]
      atom2 = mol.GetAtoms()[atom_idx]
      assert_equal(atom1.GetAtomicNum(), atom2.GetAtomicNum())
    os.remove(outfile)

  def test_pdbqt_to_pdb(self):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    protein_file = os.path.join(current_dir,
                                "../../dock/tests/1jld_protein.pdb")
    xyz, mol = rdkit_util.load_molecule(
        protein_file, calc_charges=False, add_hydrogens=False)
    out_pdb = "/tmp/mol.pdb"
    out_pdbqt = "/tmp/mol.pdbqt"

    rdkit_util.write_molecule(mol, out_pdb)
    rdkit_util.write_molecule(mol, out_pdbqt, is_protein=True)

    pdb_block = rdkit_util.pdbqt_to_pdb(out_pdbqt)
    pdb_mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)

    xyz, pdbqt_mol = rdkit_util.load_molecule(
        out_pdbqt, add_hydrogens=False, calc_charges=False)

    assert_equal(pdb_mol.GetNumAtoms(), pdbqt_mol.GetNumAtoms())
    for atom_idx in range(pdb_mol.GetNumAtoms()):
      atom1 = pdb_mol.GetAtoms()[atom_idx]
      atom2 = pdbqt_mol.GetAtoms()[atom_idx]
      assert_equal(atom1.GetAtomicNum(), atom2.GetAtomicNum())
    os.remove(out_pdb)
    os.remove(out_pdbqt)

  def test_merge_molecules_xyz(self):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    ligand_file = os.path.join(current_dir, "../../dock/tests/1jld_ligand.sdf")
    xyz, mol = rdkit_util.load_molecule(
        ligand_file, calc_charges=False, add_hydrogens=False)
    merged = rdkit_util.merge_molecules_xyz(xyz, xyz)
    for i in range(len(xyz)):
      first_atom_equal = np.all(xyz[i] == merged[i])
      second_atom_equal = np.all(xyz[i] == merged[i + len(xyz)])
      assert_true(first_atom_equal)
      assert_true(second_atom_equal)

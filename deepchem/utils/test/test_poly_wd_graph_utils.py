import unittest
from deepchem.utils import handle_hydrogen
from deepchem.utils import make_polymer_mol
from deepchem.utils import parse_polymer_rules
from deepchem.utils import tag_atoms_in_repeating_unit
from deepchem.utils import onek_encoding_unk
from deepchem.utils import generate_atom_features
from deepchem.utils import generate_bond_features
from deepchem.utils import FeaturizationParameters
from deepchem.utils import remove_wildcard_atoms
from rdkit import Chem


class WDGraphUtilsTest(unittest.TestCase):
    """
    Tests for polymer weighted directed graph util functions
    """

    def test_converts_smiles_without_hydrogens(self):
        smiles = 'C'
        mol = handle_hydrogen(smiles, keep_h=False, add_h=False)
        assert mol.GetNumAtoms() == 1
        assert all(atom.GetSymbol() != 'H' for atom in mol.GetAtoms())

    def test_converts_smiles_with_hydrogens(self):
        smiles = 'C'
        mol = handle_hydrogen(smiles, keep_h=False, add_h=True)
        assert mol.GetNumAtoms() == 5

    def test_converts_smiles_keeping_hydrogens(self):
        smiles = 'C([H])=O'
        mol = handle_hydrogen(smiles, keep_h=False, add_h=False)
        assert mol.GetNumAtoms() == 2
        mol = handle_hydrogen(smiles, keep_h=True, add_h=False)
        assert mol.GetNumAtoms() == 3
        mol = handle_hydrogen(smiles, keep_h=True, add_h=True)
        assert mol.GetNumAtoms() == 4

    def test_polymer_mol_with_metadata(self):
        # checking value error for incorrect number of fragments
        with self.assertRaises(ValueError):
            mol = make_polymer_mol('C.C', [1], True, False)

        mol = make_polymer_mol(
            'C.C',
            [1, 1],
            True,
            False,
        )
        for atom in mol.GetAtoms():
            _ = atom.GetDoubleProp('w_frag')

    def test_parse_invalid_polymer_rules(self):
        with self.assertRaises(ValueError):
            _ = parse_polymer_rules(["1-2:0.5"])

        with self.assertRaises(ValueError):
            _ = parse_polymer_rules(["1-2:0.3:0.3:0.3"])

        with self.assertRaises(ValueError):
            _ = parse_polymer_rules(["1-2-3:0.5:0.5"])

        with self.assertRaises(ValueError):
            _ = parse_polymer_rules(["1:1.0:0.0"])

    def test_parse_dop(self):
        _, dop = parse_polymer_rules(["1-2:0.5:0.5~10"])
        assert dop == 2

    def test_tag_atoms_in_repeating_unit(self):
        mol, _ = tag_atoms_in_repeating_unit(Chem.MolFromSmiles('[1*]CC.C[2*]'))
        assert mol.GetAtomWithIdx(0).GetBoolProp('core') is False
        assert mol.GetAtomWithIdx(1).GetBoolProp('core') is True
        assert mol.GetAtomWithIdx(0).GetProp('R') == ''
        assert mol.GetAtomWithIdx(1).GetProp('R') == '1*'
        assert mol.GetAtomWithIdx(3).GetProp('R') == '2*'

    def test_onek_encoding_unk(self):
        assert len(onek_encoding_unk(1, [0, 1, 2])) == 4
        assert onek_encoding_unk(0, [0, 1, 2]) == [1, 0, 0, 0]
        assert onek_encoding_unk(69, [0, 1, 2]) == [0, 0, 0, 1]

    def test_generate_atom_n_bond_features(self):
        mol = Chem.MolFromSmiles("CC")
        PARAMS = FeaturizationParameters()
        for atom in mol.GetAtoms():
            assert len(generate_atom_features(atom, PARAMS=PARAMS)) == 133
        for bond in mol.GetBonds():
            assert len(generate_bond_features(bond, PARAMS=PARAMS)) == 14

    def test_removing_wildcard_atoms(self):
        mol = Chem.MolFromSmiles("[*]CC[*]")
        rwmol = Chem.RWMol(mol)
        rwmol = remove_wildcard_atoms(rwmol)
        assert Chem.MolToSmiles(rwmol) == "CC"
        # not altering the atom if there is no wildcard
        mol = Chem.MolFromSmiles("CC")
        rwmol = Chem.RWMol(mol)
        rwmol = remove_wildcard_atoms(rwmol)
        assert Chem.MolToSmiles(rwmol) == "CC"

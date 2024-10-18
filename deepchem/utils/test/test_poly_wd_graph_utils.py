import unittest
from deepchem.utils import handle_hydrogen
from deepchem.utils import make_polymer_mol
from deepchem.utils import parse_polymer_rules
from deepchem.utils import tag_atoms_in_repeating_unit
from deepchem.utils import onek_encoding_unk
from deepchem.utils import remove_wildcard_atoms
from deepchem.utils import PolyWDGStringValidator
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


class PolyWDGStringValidateTest(unittest.TestCase):
    """
    Tests for validator class to validate polymer weighted string representation
    """

    def test_get_parsed_vals(self):
        with self.assertRaises(ValueError):
            datapoint = "[1*]C.C[2*]|<1-2:0.5:0.5"
            _ = PolyWDGStringValidator.get_parsed_vals(datapoint)

        datapoint = "[1*]C.C[2*]|0.5|0.5|<1-2:0.5:0.5"
        monomer_mols, fragments, polymer_rules = PolyWDGStringValidator.get_parsed_vals(
            datapoint)
        assert monomer_mols == "[1*]C.C[2*]"
        assert fragments == ["0.5", "0.5"]
        assert polymer_rules == "<1-2:0.5:0.5"

    def test_get_polymer_rules(self):
        with self.assertRaises(ValueError):
            polymer_rules = "1-2:0.5:0.5"
            _ = PolyWDGStringValidator.get_polymer_rules(polymer_rules)

        polymer_rules = "<1-2:0.5:0.5"
        polymer_rules = PolyWDGStringValidator.get_polymer_rules(polymer_rules)
        assert polymer_rules == ["1-2:0.5:0.5"]

    def test_valid_validate_function(self):
        datapoint = "[1*]C.C[2*]|0.5|0.5|<1-2:0.5:0.5"
        assert PolyWDGStringValidator().validate(datapoint)

    def test_invalid_validate_function(self):
        # test for _validate_fragments
        with self.assertRaises(ValueError):
            datapoint = "[1*]C.C[2*]|0.5|<1-2:0.5:0.5"
            _ = PolyWDGStringValidator().validate(datapoint)

        # test for _validate_wildcards
        with self.assertRaises(ValueError):
            datapoint = "[1*]C.C[3*]|0.5|0.5|<1-2:0.5:0.5"
            _ = PolyWDGStringValidator().validate(datapoint)

        # test for _validate_wildcards
        with self.assertRaises(ValueError):
            datapoint = "[1*]C.C[2*]|0.5|0.5|<1-3:0.5:0.5"
            _ = PolyWDGStringValidator().validate(datapoint)

    def test_invalid_polymer_rules(self):
        # test for _validate_polymer_rules
        with self.assertRaises(ValueError):
            datapoint = "[1*]C.C[2*]|0.5|0.5|<1-3:0.5"
            _ = PolyWDGStringValidator().validate(datapoint)

        # test for _validate_polymer_rules
        with self.assertRaises(ValueError):
            datapoint = "[1*]C.C[2*]|0.5|0.5|<13:0.5:0.5"
            _ = PolyWDGStringValidator().validate(datapoint)

        # test for _validate_polymer_rules
        with self.assertRaises(ValueError):
            datapoint = "[1*]C.C[2*]|0.5|0.5|<13-:0.5:0.5"
            _ = PolyWDGStringValidator().validate(datapoint)

        # test for _validate_polymer_rules
        with self.assertRaises(ValueError):
            datapoint = "[1*]C.C[2*]|0.5|0.5|<1-3-5:0.5:0.5"
            _ = PolyWDGStringValidator().validate(datapoint)

        # test for _validate_polymer_rules
        with self.assertRaises(ValueError):
            datapoint = "[1*]C.C[2*]|0.5|0.5|<1-q:0.5:0.5"
            _ = PolyWDGStringValidator().validate(datapoint)

        # test for _validate_polymer_rules
        with self.assertRaises(ValueError):
            datapoint = "[1*]C.C[2*]|0.5|0.5|<1-69:0.5:0.5"
            _ = PolyWDGStringValidator().validate(datapoint)

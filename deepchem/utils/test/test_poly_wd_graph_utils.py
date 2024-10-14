import unittest
from deepchem.utils import PolyWDGStringValidator


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

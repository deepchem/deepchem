import pytest
from deepchem.utils.poly_converters import PSMILES2WDGConverter, WDG2PSMILESConverter

# Sample data for testing
psmiles = "*CCCC*"
metadata = {
    "indicies": [2, 3],
    "seq_index": 4,
    "residue": "0.5|0.5|",
    "smiles_type": "SMARTS"
}
wd_graph_string = "[*:1]CC[*:2].[*:3]CC[*:4]|0.5|0.5|<1-3:0.5:0.5<1-4:0.5:0.5<2-3:0.5:0.5<2-4:0.5:0.5"


# Test PSMILES2WDGConverter class
class TestPSMILES2WDGConverter:

    def test_index_wildcards(self):
        converter = PSMILES2WDGConverter()
        result = converter.index_wildcards("*CC(*)C")
        assert result == "[1*]CC([2*])C"

    def test_add_indicies_to_smiles_from_meta(self):
        converter = PSMILES2WDGConverter()
        result = converter.add_indicies_to_smiles_from_meta("*CCCC*", 3)
        assert result == "*CC**CC*"

    def test_make_wdgraph_string_from_meta(self):
        converter = PSMILES2WDGConverter()
        result = converter.make_wdgraph_string_from_meta("*CC**CC*", [3, 4])
        assert "." in result  # Check if the bond was broken

    def test_convert_smiles_to_SMARTS(self):
        converter = PSMILES2WDGConverter()
        result = converter.convert_smiles_to_SMARTS("C*CC*")
        assert result == "C[*:1]CC[*:2]"

    def test_compose_from_meta(self):
        converter = PSMILES2WDGConverter()
        result = converter.compose_from_meta(psmiles, metadata)
        assert "|" in result  # Check if the final composed result includes metadata

    def test_convert(self):
        converter = PSMILES2WDGConverter()
        result = converter.convert(psmiles, metadata)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_call(self):
        converter = PSMILES2WDGConverter()
        result = converter([psmiles], [metadata])
        assert isinstance(result, list)
        assert len(result) > 0


# Test WDG2PSMILESConverter class
class TestWDG2PSMILESConverter:

    def test_get_wildcard_bond_indecies(self):
        converter = WDG2PSMILESConverter()
        with pytest.raises(ValueError):
            converter.get_wildcard_bond_indecies("CCO")

    def test_convert_smiles_part(self):
        converter = WDG2PSMILESConverter()
        result = converter.convert_smiles_part("*CC*.*CC*")
        assert isinstance(result, tuple)
        assert len(result) == 4  # Check if the method returns 4 values

    def test_replace_SMARTS(self):
        converter = WDG2PSMILESConverter()
        result = converter.replace_SMARTS("[*:1]CC[*:2]")
        assert result == "*CC*"

    def test_convert(self):
        converter = WDG2PSMILESConverter()
        result, metadata = converter.convert(wd_graph_string)
        assert isinstance(result, str)
        assert isinstance(metadata, dict)

    def test_call(self):
        converter = WDG2PSMILESConverter()
        result, metadata = converter([wd_graph_string])
        assert isinstance(result, list)
        assert len(result) > 0

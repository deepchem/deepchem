"""
Tests for ProteinBackboneFeaturizer.
"""
import unittest
import tempfile
import os
import numpy as np
import deepchem as dc

try:
    from Bio.PDB import PDBParser  # noqa: F401
    has_biopython = True
except ImportError:
    has_biopython = False


class TestProteinBackboneFeaturizer(unittest.TestCase):
    """Test ProteinBackboneFeaturizer class."""

    def setUp(self):
        """Create minimal PDB file for testing."""
        # Minimal PDB with 3 residues
        self.pdb_content = (
            "ATOM      1  N   ALA A   1"
            "       0.000   0.000   0.000  1.00  0.00           N\n"
            "ATOM      2  CA  ALA A   1"
            "       1.458   0.000   0.000  1.00  0.00           C\n"
            "ATOM      3  C   ALA A   1"
            "       2.009   1.420   0.000  1.00  0.00           C\n"
            "ATOM      4  N   GLY A   2"
            "       1.500   2.000   1.000  1.00  0.00           N\n"
            "ATOM      5  CA  GLY A   2"
            "       2.000   3.350   1.000  1.00  0.00           C\n"
            "ATOM      6  C   GLY A   2"
            "       1.500   4.000   2.000  1.00  0.00           C\n"
            "ATOM      7  N   VAL A   3"
            "       1.000   3.500   3.000  1.00  0.00           N\n"
            "ATOM      8  CA  VAL A   3"
            "       0.500   4.000   4.000  1.00  0.00           C\n"
            "ATOM      9  C   VAL A   3"
            "       1.000   5.400   4.000  1.00  0.00           C\n"
            "END\n"
        )
        self.pdb_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.pdb', delete=False)
        self.pdb_file.write(self.pdb_content)
        self.pdb_file.close()

    def tearDown(self):
        """Clean up temporary file."""
        if os.path.exists(self.pdb_file.name):
            os.unlink(self.pdb_file.name)

    @unittest.skipIf(not has_biopython, "BioPython not installed")
    def test_featurizer_init(self):
        """Test featurizer initialization."""
        featurizer = dc.feat.ProteinBackboneFeaturizer(max_length=128)
        assert featurizer.max_length == 128

    @unittest.skipIf(not has_biopython, "BioPython not installed")
    def test_featurize_single_pdb(self):
        """Test featurizing a single PDB file."""
        featurizer = dc.feat.ProteinBackboneFeaturizer()
        features = featurizer.featurize([self.pdb_file.name])

        # Base class featurize returns np.ndarray
        assert isinstance(features, np.ndarray)
        assert len(features) == 1  # One protein

        # Each element is an (L, 3, 3) array
        coords = features[0]
        assert coords.shape == (3, 3, 3)  # 3 residues, 3 atoms, 3 xyz

    @unittest.skipIf(not has_biopython, "BioPython not installed")
    def test_backbone_coordinates(self):
        """Test that backbone coordinates are correctly extracted."""
        featurizer = dc.feat.ProteinBackboneFeaturizer()
        features = featurizer.featurize([self.pdb_file.name])

        coords = features[0]  # (L, 3, 3)

        # Check first residue N atom coordinates
        np.testing.assert_array_almost_equal(
            coords[0, 0], [0.0, 0.0, 0.0])

        # Check first residue CA atom coordinates
        np.testing.assert_array_almost_equal(
            coords[0, 1], [1.458, 0.0, 0.0])

    @unittest.skipIf(not has_biopython, "BioPython not installed")
    def test_truncation(self):
        """Test that long proteins are truncated."""
        # PDB format has strict column positions:
        # cols 1-6: record, 7-11: serial, 13-16: name,
        # 17: altLoc, 18-20: resName, 22: chainID, 23-26: resSeq,
        # 31-38: x, 39-46: y, 47-54: z
        lines = []
        atoms = [("N", "N"), ("CA", "C"), ("C", "C")]
        serial = 1
        for res_i in range(20):
            for atom_name, element in atoms:
                x = float(res_i) * 3.8
                line = (
                    f"ATOM  {serial:5d} {atom_name:^4s}"
                    f" ALA A{res_i + 1:4d}    "
                    f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}"
                    f"  1.00  0.00           {element}\n")
                lines.append(line)
                serial += 1
        lines.append("END\n")
        long_pdb_content = "".join(lines)

        pdb_file2 = tempfile.NamedTemporaryFile(
            mode='w', suffix='.pdb', delete=False)
        pdb_file2.write(long_pdb_content)
        pdb_file2.close()

        try:
            featurizer = dc.feat.ProteinBackboneFeaturizer(max_length=10)
            features = featurizer.featurize([pdb_file2.name])

            # Should be truncated to max_length
            assert features[0].shape[0] == 10
        finally:
            os.unlink(pdb_file2.name)

    @unittest.skipIf(not has_biopython, "BioPython not installed")
    def test_invalid_pdb(self):
        """Test handling of invalid PDB files."""
        invalid_pdb = tempfile.NamedTemporaryFile(
            mode='w', suffix='.pdb', delete=False)
        invalid_pdb.write("This is not a valid PDB file\n")
        invalid_pdb.close()

        try:
            featurizer = dc.feat.ProteinBackboneFeaturizer()
            features = featurizer.featurize([invalid_pdb.name])

            # Should return empty array for failed featurization
            assert features[0].size == 0
        finally:
            os.unlink(invalid_pdb.name)

    def test_requires_biopython(self):
        """Test that error is raised if BioPython is not installed."""
        if has_biopython:
            self.skipTest("BioPython is installed")

        with self.assertRaises(ImportError):
            dc.feat.ProteinBackboneFeaturizer()


if __name__ == '__main__':
    unittest.main()

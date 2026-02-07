"""
Tests for ProteinBackboneFeaturizer.
"""
import unittest
import tempfile
import os
import numpy as np
import deepchem as dc

try:
    from Bio.PDB import PDBParser
    has_biopython = True
except ImportError:
    has_biopython = False


class TestProteinBackboneFeaturizer(unittest.TestCase):
    """Test ProteinBackboneFeaturizer class."""

    def setUp(self):
        """Create minimal PDB file for testing."""
        # Minimal PDB with 3 residues
        self.pdb_content = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  N   GLY A   2       1.500   2.000   1.000  1.00  0.00           N
ATOM      5  CA  GLY A   2       2.000   3.350   1.000  1.00  0.00           C
ATOM      6  C   GLY A   2       1.500   4.000   2.000  1.00  0.00           C
ATOM      7  N   VAL A   3       1.000   3.500   3.000  1.00  0.00           N
ATOM      8  CA  VAL A   3       0.500   4.000   4.000  1.00  0.00           C
ATOM      9  C   VAL A   3       1.000   5.400   4.000  1.00  0.00           C
END
"""
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

        # Should have shape (1, L, 3, 3)
        # where L is number of residues (3 in this case)
        assert features.shape[0] == 1  # One protein
        assert features.shape[1] == 3  # Three residues
        assert features.shape[2] == 3  # N, CA, C atoms
        assert features.shape[3] == 3  # x, y, z coordinates

    @unittest.skipIf(not has_biopython, "BioPython not installed")
    def test_backbone_coordinates(self):
        """Test that backbone coordinates are correctly extracted."""
        featurizer = dc.feat.ProteinBackboneFeaturizer()
        features = featurizer.featurize([self.pdb_file.name])

        coords = features[0]  # (L, 3, 3)

        # Check first residue N atom coordinates
        np.testing.assert_array_almost_equal(coords[0, 0], [0.0, 0.0, 0.0])

        # Check first residue CA atom coordinates
        np.testing.assert_array_almost_equal(coords[0, 1], [1.458, 0.0, 0.0])

    @unittest.skipIf(not has_biopython, "BioPython not installed")
    def test_truncation(self):
        """Test that long proteins are truncated."""
        # Create a longer PDB file
        long_pdb_content = ""
        for i in range(20):
            n = i * 3 + 1
            long_pdb_content += f"ATOM  {n:5d}  N   ALA A{i+1:4d}       {float(i):.3f}   0.000   0.000  1.00  0.00           N\n"
            long_pdb_content += f"ATOM  {n+1:5d}  CA  ALA A{i+1:4d}       {float(i)+1:.3f}   0.000   0.000  1.00  0.00           C\n"
            long_pdb_content += f"ATOM  {n+2:5d}  C   ALA A{i+1:4d}       {float(i)+2:.3f}   0.000   0.000  1.00  0.00           C\n"
        long_pdb_content += "END\n"

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
        # Create invalid PDB file
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

import os
import tempfile
import unittest
import numpy as np

from deepchem.feat.ProteinMPNN_featurizer import (ProteinStructureData,
                                                  _MapperProteinMPNN,
                                                  ProteinMPNNFeaturizer,
                                                  AMINO_ACID_ALPHABET)


class TestProteinMPNNFeaturizer(unittest.TestCase):
    """
    Tests for ProteinMPNNFeaturizer and its utility classes.
    """

    def setUp(self):
        """
        Set up temporary directory and dummy PDB files for testing.
        """

        self.test_dir = tempfile.TemporaryDirectory()
        self.valid_pdb_path = os.path.join(self.test_dir.name, "valid.pdb")
        self.missing_atom_pdb_path = os.path.join(self.test_dir.name,
                                                  "missing_atom.pdb")

        # 1. Create a valid PDB with 2 residues across 2 chains
        valid_pdb_content = (
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N  \n"
            "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C  \n"
            "ATOM      3  C   ALA A   1       2.009   1.424   0.000  1.00  0.00           C  \n"
            "ATOM      4  O   ALA A   1       1.319   2.441   0.000  1.00  0.00           O  \n"
            "ATOM      5  N   GLY B   2       3.326   1.488   0.000  1.00  0.00           N  \n"
            "ATOM      6  CA  GLY B   2       4.015   2.766   0.000  1.00  0.00           C  \n"
            "ATOM      7  C   GLY B   2       5.516   2.517   0.000  1.00  0.00           C  \n"
            "ATOM      8  O   GLY B   2       6.155   3.421   0.000  1.00  0.00           O  \n"
        )
        with open(self.valid_pdb_path, 'w') as f:
            f.write(valid_pdb_content)

        # 2. Create an invalid PDB missing the 'O' atom on the first residue
        missing_atom_content = (
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N  \n"
            "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C  \n"
            "ATOM      3  C   ALA A   1       2.009   1.424   0.000  1.00  0.00           C  \n"
        )
        with open(self.missing_atom_pdb_path, 'w') as f:
            f.write(missing_atom_content)

    def tearDown(self):
        """
        Clean up the temporary directory after tests finish.
        """
        self.test_dir.cleanup()

    def protein_structure_data_valid(self):
        """Test standard initialization and default mask generation."""
        L = 5
        coords = np.random.randn(L, 4, 3).astype(np.float32)
        seq = 'ACDEF'

        structure = ProteinStructureData(backbone_coords=coords, sequence=seq)

        self.assertEqual(structure.num_residues, L)
        self.assertEqual(structure.backbone_coords.shape, (L, 4, 3))

        self.assertTrue(
            np.array_equal(structure.chain_mask, np.ones(L, dtype=np.float32)))
        self.assertTrue(
            np.array_equal(structure.chain_encoding, np.ones(L,
                                                             dtype=np.int32)))
        self.assertTrue(
            np.array_equal(structure.residue_idx, np.arange(L, dtype=np.int32)))
        self.assertTrue(
            np.array_equal(structure.mask, np.ones(L, dtype=np.float32)))

    def protein_structure_data_length_mismatch(self):
        """Test that passing mismatched lengths raises a ValueError."""
        coords = np.random.randn(4, 4, 3).astype(np.float32)
        seq = 'ACDEF'  #

        with self.assertRaises(ValueError):
            ProteinStructureData(backbone_coords=coords, sequence=seq)

    def _mapper_tokenization_and_masking(self):
        """Test sequence tokenization ('X' handling) and NaN coordinate masking."""
        L = 3
        coords = np.random.randn(L, 4, 3).astype(np.float32)

        coords[1, 0, 0] = np.nan
        seq = 'ACZ'  # 'Z' is unknown and should map to 'X' (index 20)

        structure = ProteinStructureData(backbone_coords=coords, sequence=seq)
        mapper = _MapperProteinMPNN(structure)

        X, S, mask, chain_M, residue_idx, chain_encoding = mapper.values

        self.assertFalse(np.isnan(X).any())

        self.assertEqual(S[0], AMINO_ACID_ALPHABET.index('A'))
        self.assertEqual(S[1], AMINO_ACID_ALPHABET.index('C'))
        self.assertEqual(S[2], len(AMINO_ACID_ALPHABET) - 1)  # Maps to 'X'

        # residue 1 should be 0 due to NaN
        self.assertEqual(mask[0], 1.0)
        self.assertEqual(mask[1], 0.0)
        self.assertEqual(mask[2], 1.0)

    def _featurizer_default(self):
        """Test standard featurization of a PDB file."""

        featurizer = ProteinMPNNFeaturizer()
        features = featurizer.featurize([self.valid_pdb_path])

        self.assertEqual(len(features), 1)
        feat_dict = features[0][0]

        expected_L = 2  # ALA and GLY

        self.assertIn('X', feat_dict)
        self.assertEqual(feat_dict['X'].shape, (expected_L, 4, 3))
        self.assertEqual(feat_dict['S'].shape, (expected_L,))

        # Both chains should be designable by default
        self.assertTrue(np.all(feat_dict['chain_M'] == 1.0))

        # Sequence should be AG
        self.assertEqual(feat_dict['S'][0], AMINO_ACID_ALPHABET.index('A'))
        self.assertEqual(feat_dict['S'][1], AMINO_ACID_ALPHABET.index('G'))

        # Chains should be encoded distinctly (1 and 2)
        self.assertEqual(feat_dict['chain_encoding'][0], 1)
        self.assertEqual(feat_dict['chain_encoding'][1], 2)

    def test_featurizer_specific_design_chains(self):
        """Test restricting the design_chains to a single chain."""
        featurizer = ProteinMPNNFeaturizer(design_chains=['A'])
        features = featurizer.featurize([self.valid_pdb_path])

        feat_dict = features[0][0]

        # Chain_M should be 1.0 for A (index 0) and 0.0 for B (index 1)
        self.assertEqual(feat_dict['chain_M'][0], 1.0)
        self.assertEqual(feat_dict['chain_M'][1], 0.0)

    def test_featurizer_missing_atom_raises_error(self):
        """Test that the custom atom check raises a ValueError for missing backbone atoms."""
        featurizer = ProteinMPNNFeaturizer()

        with self.assertRaisesRegex(
                ValueError, "Datapoint must be a valid file path to a PDB."):
            featurizer._featurize([self.missing_atom_pdb_path])

    def test_featurizer_invalid_file_path(self):
        """Test that an invalid file path properly raises an exception."""
        featurizer = ProteinMPNNFeaturizer()

        with self.assertRaises(ValueError):
            featurizer._featurize(["/this/path/does/not/exist.pdb"])

"""
Tests that FASTA files can be loaded.
"""
import os
import unittest

import deepchem as dc
from deepchem.feat.molecule_featurizers import OneHotFeaturizer


class TestFASTALoader(unittest.TestCase):
    """
    Test FASTALoader
    """

    def setUp(self):
        super(TestFASTALoader, self).setUp()
        self.current_dir = os.path.dirname(os.path.abspath(__file__))

    def test_legacy_fasta_one_hot(self):
        input_file = os.path.join(self.current_dir, "example.fasta")
        loader = dc.data.FASTALoader(legacy=True)
        sequences = loader.create_dataset(input_file)

        # example.fasta contains 3 sequences each of length 58.
        # The one-hot encoding turns base-pairs into vectors of length 5 (ATCGN).
        # There is one "image channel".

        assert sequences.X.shape == (3, 5, 58, 1)

    def test_fasta_one_hot(self):
        input_file = os.path.join(self.current_dir, "example.fasta")
        loader = dc.data.FASTALoader(legacy=False)
        sequences = loader.create_dataset(input_file)

        # Due to FASTALoader redesign, expected shape is now (3, 58, 5)

        assert sequences.X.shape == (3, 58, 5)

    def test_fasta_one_hot_big(self):
        protein = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            '*', '-'
        ]
        input_file = os.path.join(self.current_dir, "uniprot_truncated.fasta")
        loader = dc.data.FASTALoader(OneHotFeaturizer(charset=protein,
                                                      max_length=1165),
                                     legacy=False)
        sequences = loader.create_dataset(input_file)

        assert sequences.X.shape

    # TODO: test with full uniprot file once sharding support is added.

import os
import unittest
import pickle
import tempfile
import numpy as np
import deepchem as dc
from deepchem.feat import GeneformerFeaturizer  # Importing from the library now!


class TestGeneformerFeaturizer(unittest.TestCase):

    def setUp(self):
        super(TestGeneformerFeaturizer, self).setUp()

        # Create dummy dictionaries
        self.token_dict = {
            "<pad>": 0,
            "<cls>": 2,
            "<eos>": 3,
            "ENSG_A": 10,
            "ENSG_B": 11,
            "ENSG_C": 12
        }
        self.median_dict = {"ENSG_A": 10.0, "ENSG_B": 0.5, "ENSG_C": 1.0}

        self.token_file_fd, self.token_file_path = tempfile.mkstemp()
        with os.fdopen(self.token_file_fd, 'wb') as f:
            pickle.dump(self.token_dict, f)

        self.median_file_fd, self.median_file_path = tempfile.mkstemp()
        with os.fdopen(self.median_file_fd, 'wb') as f:
            pickle.dump(self.median_dict, f)

        self.gene_names = ["ENSG_A", "ENSG_B", "ENSG_C"]

    def tearDown(self):
        super(TestGeneformerFeaturizer, self).tearDown()
        if os.path.exists(self.token_file_path):
            os.remove(self.token_file_path)
        if os.path.exists(self.median_file_path):
            os.remove(self.median_file_path)

    def test_rank_value_encoding_logic(self):
        """CRITICAL TEST: Does it rank by (Count/Total)/Median?"""
        featurizer = GeneformerFeaturizer(
            gene_median_file=self.median_file_path,
            token_dictionary_file=self.token_file_path,
            gene_names=self.gene_names,
            max_length=5,
            model_version="V1")
        # Input: [Gene A=50, Gene B=10, Gene C=0]
        # Rank: B (11) > A (10)
        tokens = featurizer.featurize([np.array([50, 10, 0])])
        self.assertEqual(tokens[0][0], 11)
        self.assertEqual(tokens[0][1], 10)

    def test_truncation(self):
        """Test if it cuts off genes when max_length is small."""
        featurizer = GeneformerFeaturizer(
            gene_median_file=self.median_file_path,
            token_dictionary_file=self.token_file_path,
            gene_names=self.gene_names,
            max_length=1)
        tokens = featurizer.featurize([np.array([50, 10, 5])])

        # This will pass now because Step 1 has the fix!
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens.shape, (1, 1))
        self.assertEqual(tokens[0][0], 11)

import os
import unittest
import deepchem as dc
import pysam
import numpy as np


class TestCRAMLoader(unittest.TestCase):
    """
    Tests for CRAMLoader and CRAMFeaturizer
    """

    def setUp(self):
        super(TestCRAMLoader, self).setUp()
        self.current_dir = os.path.dirname(os.path.abspath(__file__))

    def test_cram_loader_with_single_file(self):
        """
        Tests CRAMLoader with a single CRAM file.
        """
        cram_file_path = os.path.join(self.current_dir, "ex1.cram")
        loader = dc.data.CRAMLoader()
        dataset = loader.create_dataset(cram_file_path)

        # Perform assertions based on the expected structure of your dataset
        # For example, check the shape of X, y, and other attributes
        assert dataset.X.shape == (396, 7)
        # Add more assertions based on your specific use case

    def test_cram_loader_with_multiple_files(self):
        """
        Tests CRAMLoader with multiple CRAM files.
        """
        cram_files = [
            os.path.join(self.current_dir, "ex1.cram"),
            os.path.join(self.current_dir, "ex1.cram")
        ]
        loader = dc.data.CRAMLoader()
        dataset = loader.create_dataset(cram_files)

        # Perform assertions based on the expected structure of your dataset
        # For example, check the shape of X, y, and other attributes
        assert dataset.X.shape == (792, 7)
        # Add more assertions based on your specific use case

    def test_cram_featurizer(self):
        """
        Tests CRAMFeaturizer.
        """
        cram_featurizer = dc.feat.CRAMFeaturizer(max_records=5)
        cram_file_path = os.path.join(self.current_dir, "ex1.cram")
        cramfile = pysam.AlignmentFile(cram_file_path, "rc")
        dataset = cram_featurizer.get_features(cramfile)

        # Perform assertions based on the expected structure of your features
        # For example, check the shape of the features array
        assert dataset.shape == (5, 7)
        # Add more assertions based on your specific use case


if __name__ == "__main__":
    unittest.main()

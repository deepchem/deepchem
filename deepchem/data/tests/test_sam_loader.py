import os
import unittest
import deepchem as dc
import pysam
import numpy as np


class TestSAMLoader(unittest.TestCase):
    """
    Tests for SAMLoader and SAMFeaturizer
    """

    def setUp(self):
        super(TestSAMLoader, self).setUp()
        self.current_dir = os.path.dirname(os.path.abspath(__file__))

    def test_sam_loader_with_single_file(self):
        """
        Tests SAMLoader with a single SAM file.
        """
        sam_file_path = os.path.join(self.current_dir, "example.sam")
        loader = dc.data.SAMLoader()
        dataset = loader.create_dataset(sam_file_path)

        # Perform assertions based on the expected structure of your dataset
        # For example, check the shape of X, y, and other attributes
        assert dataset.X.shape == (12, 7)
        # Add more assertions based on your specific use case

    def test_sam_loader_with_multiple_files(self):
        """
        Tests SAMLoader with multiple SAM files.
        """
        sam_files = [
            os.path.join(self.current_dir, "example.sam"),
            os.path.join(self.current_dir, "example.sam")
        ]
        loader = dc.data.SAMLoader()
        dataset = loader.create_dataset(sam_files)

        # Perform assertions based on the expected structure of your dataset
        # For example, check the shape of X, y, and other attributes
        assert dataset.X.shape == (24, 7)
        # Add more assertions based on your specific use case

    def test_sam_featurizer(self):
        """
        Tests SAMFeaturizer.
        """
        sam_featurizer = dc.feat.SAMFeaturizer(max_records=5)
        sam_file_path = os.path.join(self.current_dir, "example.sam")
        samfile = pysam.AlignmentFile(sam_file_path, "r")
        dataset = sam_featurizer.get_features(samfile)

        # Perform assertions based on the expected structure of your features
        # For example, check the shape of the features array
        assert dataset.shape == (5, 7)
        # Add more assertions based on your specific use case


if __name__ == "__main__":
    unittest.main()

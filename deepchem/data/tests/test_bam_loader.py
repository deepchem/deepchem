import os
import unittest
import deepchem as dc
import pysam
import numpy as np


class TestBAMLoader(unittest.TestCase):
    """
    Tests for BAMLoader and BAMFeaturizer
    """

    def setUp(self):
        super(TestBAMLoader, self).setUp()
        self.current_dir = os.path.dirname(os.path.abspath(__file__))

    def test_bam_loader_with_single_file(self):
        """
        Tests BAMLoader with a single BAM file.
        """
        bam_file_path = os.path.join(self.current_dir, "example.bam")
        loader = dc.data.BAMLoader()
        dataset = loader.create_dataset(bam_file_path)

        # Perform assertions based on the expected structure of your dataset
        # For example, check the shape of X, y, and other attributes
        assert dataset.X.shape == (396, 7)
        # Add more assertions based on your specific use case

    def test_bam_loader_with_multiple_files(self):
        """
        Tests BAMLoader with multiple BAM files.
        """
        bam_files = [
            os.path.join(self.current_dir, "example.bam"),
            os.path.join(self.current_dir, "example.bam")
        ]
        loader = dc.data.BAMLoader()
        dataset = loader.create_dataset(bam_files)

        # Perform assertions based on the expected structure of your dataset
        # For example, check the shape of X, y, and other attributes
        assert dataset.X.shape == (792, 7)
        # Add more assertions based on your specific use case

    def test_bam_featurizer(self):
        """
        Tests BAMFeaturizer.
        """
        bam_featurizer = dc.feat.BAMFeaturizer(max_records=5)
        bam_file_path = os.path.join(self.current_dir, "example.bam")
        bamfile = pysam.AlignmentFile(bam_file_path, "rb")
        dataset = bam_featurizer.get_features(bamfile)

        # Perform assertions based on the expected structure of your features
        # For example, check the shape of the features array
        assert dataset.shape == (5, 7)
        # Add more assertions based on your specific use case


if __name__ == "__main__":
    unittest.main()

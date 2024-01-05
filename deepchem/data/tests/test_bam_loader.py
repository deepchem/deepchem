import os
import unittest
import deepchem as dc
try:
    import pysam
except ImportError:
    print("Error: Unable to import pysam. Please make sure it is installed.")
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

        assert dataset.X.shape == (396, 7)

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

        assert dataset.X.shape == (792, 7)

    def test_bam_featurizer(self):
        """
        Tests BAMFeaturizer.
        """
        bam_featurizer = dc.feat.BAMFeaturizer(max_records=5)
        bam_file_path = os.path.join(self.current_dir, "example.bam")
        bamfile = pysam.AlignmentFile(bam_file_path, "rb")
        dataset = bam_featurizer.get_features(bamfile)

        assert dataset.shape == (5, 7)


if __name__ == "__main__":
    unittest.main()

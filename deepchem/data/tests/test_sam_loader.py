import os
import unittest
import deepchem as dc
import logging

logger = logging.getLogger(__name__)

try:
    import pysam
except ImportError as e:
    logger.warning(
        f'Skipped loading biological sequence featurized, missing a dependency. {e}'
    )


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

        assert dataset.X.shape == (12, 7)

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

        assert dataset.X.shape == (24, 7)

    def test_sam_featurizer(self):
        """
        Tests SAMFeaturizer.
        """
        sam_featurizer = dc.feat.SAMFeaturizer(max_records=5)
        sam_file_path = os.path.join(self.current_dir, "example.sam")
        samfile = pysam.AlignmentFile(sam_file_path, "r")
        dataset = sam_featurizer._featurize(samfile)

        assert dataset.shape == (5, 7)


if __name__ == "__main__":
    unittest.main()

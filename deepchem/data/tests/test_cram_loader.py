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
        cram_file_path = os.path.join(self.current_dir, "example.cram")
        loader = dc.data.CRAMLoader()
        dataset = loader.create_dataset(cram_file_path)

        assert dataset.X.shape == (5, 7)

    def test_cram_loader_with_multiple_files(self):
        """
        Tests CRAMLoader with multiple CRAM files.
        """
        cram_files = [
            os.path.join(self.current_dir, "example.cram"),
            os.path.join(self.current_dir, "example.cram")
        ]
        loader = dc.data.CRAMLoader()
        dataset = loader.create_dataset(cram_files)

        assert dataset.X.shape == (10, 7)

    def test_cram_featurizer(self):
        """
        Tests CRAMFeaturizer.
        """
        cram_featurizer = dc.feat.CRAMFeaturizer(max_records=5)
        cram_file_path = os.path.join(self.current_dir, "example.cram")
        cramfile = pysam.AlignmentFile(cram_file_path, "rc")
        dataset = cram_featurizer._featurize(cramfile)

        assert dataset.shape == (5, 7)


if __name__ == "__main__":
    unittest.main()

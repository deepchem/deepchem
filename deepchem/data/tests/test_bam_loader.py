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

        assert dataset.X.shape == (396, 9)

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

        assert dataset.X.shape == (792, 9)

    def test_bam_featurizer(self):
        """
        Tests BAMFeaturizer.
        """
        bam_featurizer = dc.feat.BAMFeaturizer(max_records=5)
        bam_file_path = os.path.join(self.current_dir, "example.bam")
        bamfile = pysam.AlignmentFile(bam_file_path, "rb")
        dataset = bam_featurizer._featurize(bamfile)

        assert dataset.shape == (5, 9)

    def test_bam_featurizer_with_pileup(self):
        """
        Tests BAMFeaturizer with pileup generation.
        """
        bam_featurizer = dc.feat.BAMFeaturizer(max_records=5, get_pileup=True)
        bam_file_path = os.path.join(self.current_dir, "example.bam")
        bamfile = pysam.AlignmentFile(bam_file_path, "rb")
        dataset = bam_featurizer._featurize(bamfile)

        assert dataset.shape == (5, 10)


if __name__ == "__main__":
    unittest.main()

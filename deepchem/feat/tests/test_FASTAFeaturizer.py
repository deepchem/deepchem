import os
import unittest
import deepchem as dc
import logging

logger = logging.getLogger(__name__)

try:
    import pysam
except ImportError as e:
    logger.warning(
        f'Skipped loading biological sequence featurizer, missing a dependency. {e}'
    )


class TestFASTAFeaturizer(unittest.TestCase):

    def setUp(self):
        super(TestFASTAFeaturizer, self).setUp()
        self.current_dir = os.path.dirname(os.path.abspath(__file__))

    def test_fasta_featurizer_with_single_file(self):

        fasta_file_path = os.path.join(self.current_dir, "example.fasta")
        feat = dc.feat.FASTAFeaturizer()
        features = feat.featurize([fasta_file_path])

        assert features.shape == (1, 3, 2)

    def test_fasta_featurizer_with_multiple_files(self):

        fasta_file_path = os.path.join(self.current_dir, "example.fasta")
        feat = dc.feat.FASTAFeaturizer()
        features = feat.featurize([fasta_file_path, fasta_file_path])

        assert features.shape == (2, 3, 2)

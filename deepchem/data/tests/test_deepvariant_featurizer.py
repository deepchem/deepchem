import os
import unittest
import deepchem as dc
import logging

logger = logging.getLogger(__name__)


class TestRealignerFeaturizer(unittest.TestCase):

    def setUp(self):
        super(TestRealignerFeaturizer, self).setUp()
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.featurizer = dc.feat.RealignerFeaturizer()

    def test_candidate_regions_length(self):
        bam_file_path = os.path.join(self.current_dir, "example.bam")
        fasta_file_path = os.path.join(self.current_dir, "sample.fa")
        datapoint = (bam_file_path, fasta_file_path)
        candidate_regions, _ = self.featurizer._featurize(datapoint)

        # Assert the number of candidate regions
        self.assertEqual(len(candidate_regions), 54)

    def test_reads_length(self):
        bam_file_path = os.path.join(self.current_dir, "example.bam")
        fasta_file_path = os.path.join(self.current_dir, "sample.fa")
        datapoint = (bam_file_path, fasta_file_path)
        _, reads = self.featurizer._featurize(datapoint)

        # Assert the number of reads
        self.assertEqual(len(reads), 33988)


if __name__ == "__main__":
    unittest.main()

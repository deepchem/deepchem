import os
import unittest
import deepchem as dc
import logging


logger = logging.getLogger(__name__)


class TestCandidateVariantFeaturizer(unittest.TestCase):

    def setUp(self):
        super(TestCandidateVariantFeaturizer, self).setUp()
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.featurizer = dc.feat.CandidateVariantFeaturizer()

    def test_candidate_variants(self):
        """
        Tests candidate windows generation from a BAM file and a reference genome.
        """
        bam_file_path = os.path.join(self.current_dir, "example.bam")
        fasta_file_path = os.path.join(self.current_dir, "sample.fa")
        datapoint = (bam_file_path, fasta_file_path)
        features = self.featurizer.featurize([datapoint])
        candidate_Windows = features[0]

        # Assert the number of reads
        self.assertEqual(len(candidate_Windows), 222)
        self.assertEqual(candidate_Windows[0].tolist(), ['chr1', 2, 'N', 'T', 2, 2])
        self.assertEqual(candidate_Windows[1].tolist(), ['chr1', 3, 'N', 'C', 2, 3])


if __name__ == "__main__":
    unittest.main()
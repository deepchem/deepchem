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

    def test_candidate_windows(self):
        bam_file_path = os.path.join(self.current_dir, "example.bam")
        fasta_file_path = os.path.join(self.current_dir, "sample.fa")
        datapoint = (bam_file_path, fasta_file_path)
        candidate_windows = self.featurizer._featurize(datapoint)

        # Assert the number of reads
        self.assertEqual(len(candidate_windows), 15)
        self.assertEqual(candidate_windows[13][0], 'chr2')
        self.assertEqual(candidate_windows[13][1], 136)
        self.assertEqual(candidate_windows[13][2], 137)
        self.assertEqual(candidate_windows[13][3], 102)
        self.assertEqual(candidate_windows[13][4], 21)


if __name__ == "__main__":
    unittest.main()

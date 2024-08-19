import os
import unittest
import deepchem as dc
import logging

logger = logging.getLogger(__name__)

try:
    import pysam
    import torch
    import dgl
except ImportError as e:
    logger.warning(
        f'Skipped loading deepvariant featurizer, missing a dependency. {e}'
    )


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
        self.assertEqual(len(candidate_windows), 14)
        self.assertEqual(candidate_windows[13],
                         ('chr2', 193, 197, 18, 21,
                          ['GTAAATTGATTTGAATTTTATTTCTTGGTAATGAGG']))



if __name__ == "__main__":
    unittest.main()
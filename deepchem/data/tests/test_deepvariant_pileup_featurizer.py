import os
import unittest
import deepchem as dc
import logging
import numpy as np

logger = logging.getLogger(__name__)


class TestPileupFeaturizer(unittest.TestCase):

    def setUp(self):
        super(TestPileupFeaturizer, self).setUp()
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.featurizer = dc.feat.PileupFeaturizer()

    def test_pileup(self):
        """
        Tests pileup generation.
        """
        bam_file_path = os.path.join(self.current_dir, "example.bam")
        fasta_file_path = os.path.join(self.current_dir, "sample.fa")
        candidate_path = os.path.join(self.current_dir,
                                      "candidate_variants.npy")
        candidate_variants = np.load(candidate_path, allow_pickle=True)
        datapoint = (bam_file_path, fasta_file_path, candidate_variants)
        features = self.featurizer.featurize([datapoint])
        image_dataset = features[0]

        self.assertEqual(len(image_dataset), len(candidate_variants))
        self.assertEqual(image_dataset.X[0].shape, (6, 100, 221))


if __name__ == "__main__":
    unittest.main()

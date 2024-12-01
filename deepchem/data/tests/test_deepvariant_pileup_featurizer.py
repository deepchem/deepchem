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
        windows_haplotypes_path = os.path.join(self.current_dir,
                                               "windows_haplotypes.npy")
        windows_haplotypes = np.load(windows_haplotypes_path, allow_pickle=True)
        fasta_file_path = os.path.join(self.current_dir, "sample.fa")
        datapoint = (windows_haplotypes, fasta_file_path)
        image_dataset = self.featurizer._featurize(datapoint)

        # Assert the number of reads
        self.assertEqual(len(image_dataset), 15)
        self.assertEqual(image_dataset.X[0].shape, (299, 299, 6))


if __name__ == "__main__":
    unittest.main()

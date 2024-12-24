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
        height = 299
        width = 299
        num_channels = 6
        datapoint = (windows_haplotypes, fasta_file_path, height, width,
                     num_channels)
        features = self.featurizer.featurize([datapoint])
        image_dataset = features[0]

        # Assert the number of reads
        self.assertEqual(len(image_dataset), 15)
        self.assertEqual(image_dataset.X[0].shape, (299, 299, 6))


if __name__ == "__main__":
    unittest.main()

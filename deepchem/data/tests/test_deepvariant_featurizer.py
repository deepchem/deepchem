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

    def test_windows_haplotypes(self):
        """
        Tests haplotype windows generation from a BAM file and a reference genome.
        """
        bam_file_path = os.path.join(self.current_dir, "example.bam")
        fasta_file_path = os.path.join(self.current_dir, "sample.fa")
        datapoint = (bam_file_path, fasta_file_path)
        features = self.featurizer.featurize([datapoint])
        windows_haplotypes = features[0]

        # Assert the number of reads
        self.assertEqual(len(windows_haplotypes), 53)
        self.assertEqual(windows_haplotypes[0]['span'], ('chr1', 3, 5))
        self.assertEqual(windows_haplotypes[1]['span'], ('chr1', 9, 20))


if __name__ == "__main__":
    unittest.main()

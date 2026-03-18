import os
import unittest
from deepchem.data.data_loader import FASTQLoader


class TestFASTQLoader(unittest.TestCase):
    """
  Test FASTQLoader
  """

    def setUp(self):
        super(TestFASTQLoader, self).setUp()
        self.current_dir = os.path.dirname(os.path.abspath(__file__))

    def test_fastq_one_hot(self):
        input_file = os.path.join(self.current_dir, "sample1.fastq")
        loader = FASTQLoader()
        sequences = loader.create_dataset(input_file)
        # Default file contains 4 sequences each of length 192 (excluding the end of line character '\n').
        # The one-hot encoding turns base-pairs into vectors of length 5 (ATCGN).
        # Expected shape is now (4, 192, 5)
        assert sequences.X.shape == (4, 192, 5)

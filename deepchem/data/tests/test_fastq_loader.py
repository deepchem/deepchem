import os
import unittest
from deepchem.feat.molecule_featurizers import OneHotFeaturizer
from deepchem.data.data_loader import FASTQLoader


class TestFASTQLoader(unittest.TestCase):
  """
  Test FASTQLoader
  """

  def setUp(self):
    super(TestFASTQLoader, self).setUp()
    self.current_dir = os.path.abspath(os.curdir)

  def test_fastq_one_hot(self, verbose_x=False):
    input_file = os.path.join(self.current_dir, "sample2.fastq")
    loader = FASTQLoader()
    sequences = loader.create_dataset(input_file)
    # Default file contains 4 sequences each of length 192 (excluding the end of line character '\n').
    # The one-hot encoding turns base-pairs into vectors of length 5 (ATCGN).
    # Expected shape is now (4, 192, 5)
    assert sequences.X.shape == (4, 192, 5)

    if verbose_x:  # To see the featurized version of X
      print(sequences.X)

  def test_fastq_one_hot_big(self, verbose_x=False):
    protein = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
        'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '*', '-'
    ]
    input_file = os.path.join(self.current_dir, 'sample1.fastq')
    loader = FASTQLoader(OneHotFeaturizer(charset=protein, max_length=1000))
    sequences = loader.create_dataset(input_file)
    # Default file contains 4437 sequences each of variable length
    # The one-hot encoding turns base-pairs into vectors of length 29 (equal to length of characters in the protein variable).
    # The max_length specified during initialization of the OneHotFeaturizer pads/truncates sequences to an equal length.
    # In this case, that length is 1000.
    # Expected shape is now (4437, 1000, 29)
    assert sequences.X.shape == (4437, 1000, 29)
    if verbose_x:
      print(sequences.X)

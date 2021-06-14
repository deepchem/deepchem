import unittest
import numpy as np

from deepchem.feat import ProteinTokenizer

class TestProteinTokenizer(unittest.TestCase):
  """
  Test ProteinTokenizer
  """

  def test_protein_tokenizer(self):
    """
    Test correct protein to integer conversion and untransform
    """
    ref_seq = np.array(["[CLS] A B C D E F G [SEP]"])
    tokenizer = ProteinTokenizer()
    int_seq = tokenizer(ref_seq)
    assert np.all(int_seq == [[0, 1, 2, 3, 4, 5, 6]])

    # untransform
    seq = tokenizer.untransform(int_seq)
    assert (seq == "ABCDEFG")

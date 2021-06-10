import unittest

from deepchem.feat import ProteinTokenizer

class TestProteinTokenizer(unittest.TestCase):
  """
  Test ProteinTokenizer
  """

  def test_protein_tokenizer(self):
    """
    Test correct protein to integer conversion and untransform
    """
    ref_seq = "Met Ser Arg Gly Asp Glu Stop"
    ref_int_seq = (-1, 14, 18, 19, 24, 22, 23, 4) # First digit indicates sequence is protein
    tokenizer = ProteinTokenizer()
    int_seq = tokenizer(ref_seq)
    assert ref_int_seq == int_seq

    # untransform
    seq = tokenizer.untransform(int_seq)
    assert ref_seq == seq

import unittest

from deepchem.feat import RNATokenizer

class TestRNATokenizer(unittest.TestCase):
  def test_RNA_Tokenization(self):
    """
    Test correct RNA to integer conversion and untransform
    """
    ref_seq = "AUGAGUAGGGGUGAUGAGUAG"
    ref_int_seq = (35, 44, 47, 60, 56, 59, 11) # Last digit indicates sequence is RNA
    tokenizer = RNATokenizer()
    int_seq = tokenizer(ref_seq)
    assert ref_int_seq == int_seq

    # untransform
    seq = tokenizer.untransform(int_seq)
    assert ref_seq == seq

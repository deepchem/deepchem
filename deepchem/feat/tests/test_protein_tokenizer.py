import unittest
import numpy as np

from deepchem.feat import ProteinTokenizer


class TestProteinTokenizer(unittest.TestCase):
  """
  Test ProteinTokenizer
  """

  def test_protein_tokenizer_single_sequence(self):
    """
    Test correct protein to integer conversion and untransform
    """
    ref_seq = np.array(["AMC g $4[CLS] A B $C D E Fg[SEP] E R"])
    tokenizer = ProteinTokenizer()
    int_seq = tokenizer(ref_seq)
    assert np.all(int_seq == [[0, 1, 2, 3, 4, 5, 6]])

    # untransform
    seq = tokenizer.untransform(int_seq)
    assert (seq == ("[CLS]ABCDEFG[SEP]",))

  def test_protein_tokenizer_multiple_sequences(self):
    """
    Test correct protein to integer conversion and untransform for multiple
    FASTA strings.
    """
    ref_seq = np.array(
        ["[CLS] A B C D E F G [SEP] EH", "ABC[CLS] H I J K L [SEP]"])
    tokenizer = ProteinTokenizer()
    int_seq = tokenizer(ref_seq)
    ref_int_seq = []
    ref_int_seq.append(np.array([0, 1, 2, 3, 4, 5, 6]))
    ref_int_seq.append(np.array([7, 8, 9, 10, 11]))
    ref_int_seq_asarray = np.asarray(ref_int_seq)
    assert str(int_seq) == str(ref_int_seq_asarray)

    # untransform
    seq = tokenizer.untransform(int_seq)
    assert (seq == ("[CLS]ABCDEFG[SEP]", "[CLS]HIJKL[SEP]"))

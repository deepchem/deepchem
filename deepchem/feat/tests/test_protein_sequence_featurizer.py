import unittest

from deepchem.feat import CodonFeaturizer

class TestCodonFeaturizer(unittest.TestCase):
  """
  Test CodonFeaturizer
  """

  def test_protein_sequence_featurization(self):
    """
    Test correct protein to integer conversion and untransform
    """
    ref_seq = "Met Ser Arg Gly Asp Glu Stop"
    ref_int_seq = (4, 6, 20, 23, 16, 17, 12)
    featurizer = ProteinSequenceFeaturizer()
    int_seq = featurizer(ref_seq)
    assert ref_int_seq == int_seq

    # untransform
    seq = featurizer.untransform(int_seq)
    assert ref_seq == seq

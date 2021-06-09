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
    ref_int_seq = (-1, 14, 18, 19, 24, 22, 23, 4) # First digit indicates sequence is protein
    featurizer = CodonFeaturizer()
    int_seq = featurizer(ref_seq)
    assert ref_int_seq == int_seq

    # untransform
    seq = featurizer.untransform(int_seq)
    assert ref_seq == seq

  def test_RNA_featurization(self):
    """
    Test correct RNA to integer conversion and untransform
    """
    ref_seq = "AUGAGUAGGGGUGAUGAGUAG"
    ref_int_seq = (-2, 35, 44, 47, 60, 56, 59, 11) # Last digit indicates sequence is RNA
    featurizer = CodonFeaturizer()
    int_seq = featurizer(ref_seq)
    assert ref_int_seq == int_seq

    # untransform
    seq = featurizer.untransform(int_seq)
    assert ref_seq == seq


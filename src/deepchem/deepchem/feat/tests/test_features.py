"""
Test featurizer class.
"""
import unittest

from deepchem.feat import ConvMolFeaturizer, CircularFingerprint
from deepchem.feat.basic import MolecularWeight


class TestFeaturizer(unittest.TestCase):
  """
  Tests for Featurizer.
  """

  def setUp(self):
    """
    Set up tests.
    """
    smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
    from rdkit import Chem
    self.mol = Chem.MolFromSmiles(smiles)

  def test_featurizer(self):
    """
    Test basic functionality of Featurizer.
    """
    f = MolecularWeight()
    rval = f([self.mol])
    assert rval.shape == (1, 1)

  def test_flatten_conformers(self):
    """
    Calculate molecule-level features for a multiconformer molecule.
    """
    f = MolecularWeight()
    rval = f([self.mol])
    assert rval.shape == (1, 1)

  def test_convmol_hashable(self):
    featurizer1 = ConvMolFeaturizer(atom_properties=['feature'])
    featurizer2 = ConvMolFeaturizer(atom_properties=['feature'])
    featurizer3 = ConvMolFeaturizer()

    d = set()
    d.add(featurizer1)
    d.add(featurizer2)
    d.add(featurizer3)

    self.assertEqual(2, len(d))
    featurizers = [featurizer1, featurizer2, featurizer3]

    for featurizer in featurizers:
      self.assertTrue(featurizer in featurizers)

  def test_circularfingerprint_hashable(self):
    featurizer1 = CircularFingerprint()
    featurizer2 = CircularFingerprint()
    featurizer3 = CircularFingerprint(size=5)

    d = set()
    d.add(featurizer1)
    d.add(featurizer2)
    d.add(featurizer3)

    self.assertEqual(2, len(d))
    featurizers = [featurizer1, featurizer2, featurizer3]

    for featurizer in featurizers:
      self.assertTrue(featurizer in featurizers)

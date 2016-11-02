"""
Test featurizer class.
"""
import numpy as np
import unittest

from rdkit import Chem

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

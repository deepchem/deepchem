"""
Test basic molecular features.
"""
import numpy as np
import unittest

from rdkit import Chem

from deepchem.feat.basic import MolecularWeight, RDKitDescriptors


class TestMolecularWeight(unittest.TestCase):
  """
  Test MolecularWeight.
  """
  def setUp(self):
    """
    Set up tests.
    """
    smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
    self.mol = Chem.MolFromSmiles(smiles)
    self.engine = MolecularWeight()

  def testMW(self):
    """
    Test MW.
    """
    assert np.allclose(self.engine([self.mol]), 180, atol=0.1)


class TestRDKitDescriptors(unittest.TestCase):
  """
  Test RDKitDescriptors.
  """
  def setUp(self):
    """
    Set up tests.
    """
    smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
    self.mol = Chem.MolFromSmiles(smiles)
    self.engine = RDKitDescriptors()

  def testRDKitDescriptors(self):
    """
    Test simple descriptors.
    """
    descriptors = self.engine([self.mol])
    assert np.allclose(
      descriptors[0, self.engine.descriptors.index('ExactMolWt')], 180,
      atol=0.1)

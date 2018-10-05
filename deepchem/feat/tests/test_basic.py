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
    self.smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
    self.mol = Chem.MolFromSmiles(self.smiles)
    self.engine = MolecularWeight()

  def testMW(self):
    """
    Test MW.
    """
    # mols as input argument
    assert np.allclose(self.engine([self.mol]), 180, atol=0.1)

    # smiles as input argument
    assert np.allclose(self.engine(smiles=[self.smiles]), 180, atol=0.1)


class TestRDKitDescriptors(unittest.TestCase):
  """
  Test RDKitDescriptors.
  """

  def setUp(self):
    """
    Set up tests.
    """
    self.smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
    self.mol = Chem.MolFromSmiles(self.smiles)
    self.engine = RDKitDescriptors()

  def testRDKitDescriptors(self):
    """
    Test simple descriptors.
    """
    # mols as input argument
    descriptors_from_mols = self.engine([self.mol])
    assert np.allclose(
        descriptors_from_mols[0, self.engine.descriptors.index('ExactMolWt')],
        180,
        atol=0.1)

    # smiles as input argument
    descriptors_from_smiles = self.engine(smiles=[self.smiles])
    assert np.allclose(
        descriptors_from_smiles[0, self.engine.descriptors.index('ExactMolWt')],
        180,
        atol=0.1)

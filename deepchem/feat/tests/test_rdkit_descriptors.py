"""
Test basic molecular features.
"""
import numpy as np
import unittest

from deepchem.feat import RDKitDescriptors


class TestRDKitDescriptors(unittest.TestCase):
  """
  Test RDKitDescriptors.
  """

  def setUp(self):
    """
    Set up tests.
    """
    from rdkit import Chem
    smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
    self.mol = Chem.MolFromSmiles(smiles)
    self.featurizer = RDKitDescriptors()

  def test_rdkit_descriptors(self):
    """
    Test simple descriptors.
    """
    descriptors = self.featurizer([self.mol])
    assert descriptors.shape == (1, 200)
    assert np.allclose(
        descriptors[0, self.featurizer.descriptors.index('ExactMolWt')],
        180,
        atol=0.1)

  def test_rdkit_descriptors_on_smiles(self):
    """
    Test invocation on raw smiles.
    """
    descriptors = self.featurizer('CC(=O)OC1=CC=CC=C1C(=O)O')
    assert descriptors.shape == (1, 200)
    assert np.allclose(
        descriptors[0, self.featurizer.descriptors.index('ExactMolWt')],
        180,
        atol=0.1)

  def test_rdkit_descriptors_on_mol(self):
    """
    Test invocation on RDKit mol.
    """
    descriptors = self.featurizer(self.mol)
    assert descriptors.shape == (1, 200)
    assert np.allclose(
        descriptors[0, self.featurizer.descriptors.index('ExactMolWt')],
        180,
        atol=0.1)

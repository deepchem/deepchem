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
    featurizer = RDKitDescriptors()
    descriptors = featurizer([self.mol])
    assert descriptors.shape == (1, len(featurizer.descriptors))
    assert np.allclose(descriptors[0,
                                   featurizer.descriptors.index('ExactMolWt')],
                       180,
                       atol=0.1)

  def test_rdkit_descriptors_on_smiles(self):
    """
    Test invocation on raw smiles.
    """
    featurizer = RDKitDescriptors()
    descriptors = featurizer('CC(=O)OC1=CC=CC=C1C(=O)O')
    assert descriptors.shape == (1, len(featurizer.descriptors))
    assert np.allclose(descriptors[0,
                                   featurizer.descriptors.index('ExactMolWt')],
                       180,
                       atol=0.1)

  def test_rdkit_descriptors_with_use_fragment(self):
    """
    Test with use_fragment
    """
    from rdkit.Chem import Descriptors
    featurizer = RDKitDescriptors(use_fragment=False)
    descriptors = featurizer(self.mol)
    assert descriptors.shape == (1, len(featurizer.descriptors))
    all_descriptors = Descriptors.descList
    assert len(featurizer.descriptors) < len(all_descriptors)
    assert np.allclose(descriptors[0,
                                   featurizer.descriptors.index('ExactMolWt')],
                       180,
                       atol=0.1)

  def test_rdkit_descriptors_with_use_bcut2d(self):
    """
    Test with use_bcut2d
    """
    from rdkit.Chem import Descriptors
    featurizer = RDKitDescriptors(use_bcut2d=False)
    descriptors = featurizer(self.mol)
    assert descriptors.shape == (1, len(featurizer.descriptors))
    all_descriptors = Descriptors.descList
    assert len(featurizer.descriptors) < len(all_descriptors)

    with self.assertRaises(ValueError):
      featurizer.descriptors.index('BCUT2D_MWHI')

    assert np.allclose(descriptors[0,
                                   featurizer.descriptors.index('ExactMolWt')],
                       180,
                       atol=0.1)

  def test_rdkit_descriptors_normalized(self):
    """
    Test with normalization
    """
    featurizer = RDKitDescriptors(is_normalized=True)
    assert featurizer.normalized_desc != {}

    descriptors = featurizer(self.mol)
    assert descriptors.shape == (1, len(featurizer.descriptors))

    # no normalized feature value should be greater than 1.0
    assert len(np.where(descriptors > 1.0)[0]) == 0

    assert np.allclose(descriptors[0,
                                   featurizer.descriptors.index('ExactMolWt')],
                       0.0098,
                       atol=0.0001)

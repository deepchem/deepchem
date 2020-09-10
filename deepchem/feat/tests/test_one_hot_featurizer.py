import unittest

import numpy as np

from deepchem.feat import OneHotFeaturizer
from deepchem.feat.molecule_featurizers.one_hot_featurizer import ZINC_CHARSET


class TestOneHotFeaturizert(unittest.TestCase):
  """
  Test OneHotFeaturizer.
  """

  def test_onehot_featurizer(self):
    """
    Test simple one hot encoding.
    """
    from rdkit import Chem
    length = len(ZINC_CHARSET) + 1
    smiles = 'CC(=O)Oc1ccccc1C(=O)O'
    mol = Chem.MolFromSmiles(smiles)
    featurizer = OneHotFeaturizer()
    feature = featurizer([mol])
    assert feature.shape == (1, 100, length)

    # untranform
    undo_smiles = featurizer.untransform(feature[0])
    assert smiles == undo_smiles

  def test_onehot_featurizer_with_max_length(self):
    """
    Test one hot encoding with max_length.
    """
    from rdkit import Chem
    length = len(ZINC_CHARSET) + 1
    smiles = 'CC(=O)Oc1ccccc1C(=O)O'
    mol = Chem.MolFromSmiles(smiles)
    featurizer = OneHotFeaturizer(max_length=120)
    feature = featurizer([mol])
    assert feature.shape == (1, 120, length)

    # untranform
    undo_smiles = featurizer.untransform(feature[0])
    assert smiles == undo_smiles

  def test_correct_transformation(self):
    """
    Test correct one hot encoding.
    """
    from rdkit import Chem
    charset = ['C', 'N', '=', ')', '(', 'O']
    smiles = 'CN=C=O'
    mol = Chem.MolFromSmiles(smiles)
    featurizer = OneHotFeaturizer(charset=charset, max_length=100)
    feature = featurizer([mol])
    assert np.allclose(feature[0][0], np.array([1, 0, 0, 0, 0, 0, 0]))
    assert np.allclose(feature[0][1], np.array([0, 1, 0, 0, 0, 0, 0]))
    assert np.allclose(feature[0][2], np.array([0, 0, 1, 0, 0, 0, 0]))
    assert np.allclose(feature[0][3], np.array([1, 0, 0, 0, 0, 0, 0]))
    assert np.allclose(feature[0][4], np.array([0, 0, 1, 0, 0, 0, 0]))
    assert np.allclose(feature[0][5], np.array([0, 0, 0, 0, 0, 1, 0]))
    # untranform
    undo_smiles = featurizer.untransform(feature[0])
    assert smiles == undo_smiles

import unittest

import numpy as np

from deepchem.feat import OneHotFeaturizer
from deepchem.feat.molecule_featurizers.one_hot_featurizer import ZINC_CHARSET


class TestOneHotFeaturizer(unittest.TestCase):
    """
    Test OneHotFeaturizer.
    """

    def test_onehot_featurizer_arbitrary(self):
        """
        Test simple one hot encoding for arbitrary string.
        """
        string = "abcdefghijklmnopqrstuvwxyzwebhasw"
        charset = "abcdefghijklmnopqrstuvwxyz"
        length = len(charset) + 1
        defaultMaxLength = 100
        featurizer = OneHotFeaturizer(charset)
        feature = featurizer([string])  # Implicit call to featurize()
        assert feature.shape == (1, defaultMaxLength, length)
        # untransform
        undo_string = featurizer.untransform(feature[0])
        assert string == undo_string

    def test_onehot_featurizer_SMILES(self):
        """
        Test simple one hot encoding for SMILES strings.
        """
        from rdkit import Chem
        length = len(ZINC_CHARSET) + 1
        smiles = 'CC(=O)Oc1ccccc1C(=O)O'
        mol = Chem.MolFromSmiles(smiles)
        featurizer = OneHotFeaturizer()
        feature = featurizer([mol])
        defaultMaxLength = 100
        assert feature.shape == (1, defaultMaxLength, length)
        # untranform
        undo_smiles = featurizer.untransform(feature[0])
        assert smiles == undo_smiles

    def test_onehot_featurizer_arbitrary_with_max_length(self):
        """
        Test one hot encoding with max_length.
        """
        string = "abcdefghijklmnopqrstuvwxyzvewqmc"
        charset = "abcdefghijklmnopqrstuvwxyz"
        length = len(charset) + 1
        featurizer = OneHotFeaturizer(charset, max_length=120)
        feature = featurizer([string])
        assert feature.shape == (1, 120, length)
        # untranform
        undo_string = featurizer.untransform(feature[0])
        assert string == undo_string

    def test_onehot_featurizer_SMILES_with_max_length(self):
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

    def test_correct_transformation_SMILES(self):
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

    def test_correct_transformation_arbitrary(self):
        """
        Test correct one hot encoding.
        """
        charset = "1234567890"
        string = "12345"
        featurizer = OneHotFeaturizer(charset=charset, max_length=100)
        feature = featurizer([string])
        assert np.allclose(feature[0][0],
                           np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        assert np.allclose(feature[0][1],
                           np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        assert np.allclose(feature[0][2],
                           np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
        assert np.allclose(feature[0][3],
                           np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]))
        assert np.allclose(feature[0][4],
                           np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]))
        assert "This test case has not yet been written."

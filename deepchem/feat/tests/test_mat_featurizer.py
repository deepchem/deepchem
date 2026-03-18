import unittest
from deepchem.feat import MATFeaturizer
import numpy as np


class TestMATFeaturizer(unittest.TestCase):
    """
    Test MATFeaturizer.
    """

    def setUp(self):
        """
        Set up tests.
        """
        from rdkit import Chem
        smiles = 'CC'
        self.mol = Chem.MolFromSmiles(smiles)

    def test_mat_featurizer(self):
        """
        Test featurizer.py
        """
        featurizer = MATFeaturizer()
        out = featurizer.featurize(self.mol)
        assert isinstance(out, np.ndarray)
        assert (out[0].node_features.shape == (3, 36))
        assert (out[0].adjacency_matrix.shape == (3, 3))
        assert (out[0].distance_matrix.shape == (3, 3))
        expected_node_features = np.array(
            [[
                1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0.
            ],
             [
                 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
                 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
                 0., 0., 0., 0.
             ],
             [
                 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
                 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
                 0., 0., 0., 0.
             ]])
        expected_adj_matrix = np.array([[0., 0., 0.], [0., 0., 1.],
                                        [0., 1., 0.]])
        expected_dist_matrix = np.array([[1.e+06, 1.e+06, 1.e+06],
                                         [1.e+06, 0.e+00, 1.e+00],
                                         [1.e+06, 1.e+00, 0.e+00]])
        assert (np.array_equal(out[0].node_features, expected_node_features))
        assert (np.array_equal(out[0].adjacency_matrix, expected_adj_matrix))
        assert (np.array_equal(out[0].distance_matrix, expected_dist_matrix))

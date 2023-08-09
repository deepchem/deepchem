"""
Test for MXMNet Featurizer class.
"""

import deepchem as dc
import numpy as np
import unittest

edge_index = {
    "C1=CC=NC=C1":
        np.asarray([[0, 1, 1, 5, 5, 2, 2, 4, 4, 3, 3, 0],
                    [1, 0, 5, 1, 2, 5, 4, 2, 3, 4, 0, 3]]),
    "CC(=O)C":
        np.asarray([[0, 3, 3, 2, 3, 1], [3, 0, 2, 3, 1, 3]]),
    "C":
        np.empty((2, 0), dtype=int)
}

node_features = {
    "C1=CC=NC=C1": np.asarray([[1.], [1.], [2.], [1.], [1.], [1.]]),
    "CC(=O)C": np.asarray([[1.], [1.], [3.], [1.]]),
    "C": np.asarray([[1.]])
}

node_pos = np.asarray([[-1.2700e-02, 1.0858e+00, 8.0000e-03],
                       [2.2000e-03, -6.0000e-03, 2.0000e-03],
                       [1.0117e+00, 1.4638e+00, 3.0000e-04],
                       [-5.4080e-01, 1.4475e+00, -8.7660e-01],
                       [-5.2380e-01, 1.4379e+00, 9.0640e-01]])


class TestMXMNetFeaturizer(unittest.TestCase):
    """
    Test MXMNetFeaturizer.
    """

    def setUp(self):
        """
        Set up tests.
        """
        self.smiles = ["C1=CC=NC=C1", "CC(=O)C", "C", "CP"]
        self.edge_index = list(edge_index.values())
        self.node_features = list(node_features.values())

    def test_featurizer_ring(self):
        """
        Test for featurization of "C1=CC=NC=C1" using `MXMNetFeaturizer` class.
        """

        featurizer = dc.feat.molecule_featurizers.mxmnet_featurizer.MXMNetFeaturizer(
        )
        graph_feat = featurizer.featurize(self.smiles)
        assert len(graph_feat) == 4

        assert graph_feat[0].num_nodes == 6
        assert graph_feat[0].num_node_features == 1
        assert graph_feat[0].node_features.shape == (6, 1)
        assert graph_feat[0].num_edges == 12
        assert (graph_feat[0].node_features == self.node_features[0]).all()

        assert (graph_feat[0].edge_index == self.edge_index[0]).all()

    def test_featurizer_general_case(self):
        """
        Test for featurization of "CC(=O)C" using `MXMNetFeaturizer` class.
        """

        featurizer = dc.feat.molecule_featurizers.mxmnet_featurizer.MXMNetFeaturizer(
        )
        graph_feat = featurizer.featurize(self.smiles)
        assert len(graph_feat) == 4

        assert graph_feat[1].num_nodes == 4
        assert graph_feat[1].num_node_features == 1
        assert graph_feat[1].node_features.shape == (4, 1)
        assert graph_feat[1].num_edges == 6
        assert (graph_feat[1].node_features == self.node_features[1]).all()

        assert (graph_feat[1].edge_index == self.edge_index[1]).all()

    def test_featurizer_single_atom(self):
        """
        Test for featurization of "C" using `MXMNetFeaturizer` class.
        """

        featurizer = dc.feat.molecule_featurizers.mxmnet_featurizer.MXMNetFeaturizer(
        )
        graph_feat = featurizer.featurize(self.smiles)
        assert len(graph_feat) == 4

        assert graph_feat[2].num_nodes == 1
        assert graph_feat[2].num_node_features == 1
        assert graph_feat[2].node_features.shape == (1, 1)
        assert graph_feat[2].num_edges == 0
        assert (graph_feat[2].node_features == self.node_features[2]).all()

        assert (graph_feat[2].edge_index == self.edge_index[2]).all()

    def test_featurizer_other_atom(self):
        """
        Test for featurization of "CP" using `MXMNetFeaturizer` class.
        Since the smile contains P which is not supported by featurizer, the featurization process terminates and the featurizer returns an empty numpy array.
        """

        featurizer = dc.feat.molecule_featurizers.mxmnet_featurizer.MXMNetFeaturizer(
        )
        graph_feat = featurizer.featurize(self.smiles)
        assert len(graph_feat) == 4

        assert graph_feat[3].shape == (0,)

    def test_node_pos_features(self):
        """
        Test for featurization of "C" using `MXMNetFeaturizer` class.
        It checks whether node_pos_features are handled properly.
        """
        smile = ['C']
        pos_x1 = [np.array([-0.0127, 0.0022, 1.0117, -0.5408, -0.5238])]
        pos_y1 = [np.array([1.0858, -0.0060, 1.4638, 1.4475, 1.4379])]
        pos_z1 = [np.array([0.0080, 0.0020, 0.0003, -0.8766, 0.9064])]

        featurizer = dc.feat.molecule_featurizers.mxmnet_featurizer.MXMNetFeaturizer(
            is_adding_hs=True)
        graph_feat = featurizer.featurize(smile,
                                          pos_x=pos_x1,
                                          pos_y=pos_y1,
                                          pos_z=pos_z1)
        assert isinstance(graph_feat[0].node_pos_features, np.ndarray)
        assert np.allclose(graph_feat[0].node_pos_features, node_pos, atol=1e-3)

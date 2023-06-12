"""
Test for MXMNet Featurizer class.
"""

from deepchem.feat.molecule_featurizers.mxmnet_featurizer import MXMNetFeaturizer
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


class Test_MXMNet_Featurizer(unittest.TestCase):
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

        featurizer = MXMNetFeaturizer()
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

        featurizer = MXMNetFeaturizer()
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

        featurizer = MXMNetFeaturizer()
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

        featurizer = MXMNetFeaturizer()
        graph_feat = featurizer.featurize(self.smiles)
        assert len(graph_feat) == 4

        assert graph_feat[3].shape == (0,)

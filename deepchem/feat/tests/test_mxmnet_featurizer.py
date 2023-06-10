"""
Test for MXMNet Featurizer class.
"""

from deepchem.feat.molecule_featurizers.mxmnet_featurizer import MXMNetFeaturizer
import numpy as np

edge_index_original_ordering = {
    "C1=CC=NC=C1":
        np.asarray([[0, 1, 1, 5, 5, 2, 2, 4, 4, 3, 3, 0],
                    [1, 0, 5, 1, 2, 5, 4, 2, 3, 4, 0, 3]]),
    "CC(=O)C":
        np.asarray([[0, 3, 3, 2, 3, 1], [3, 0, 2, 3, 1, 3]]),
    "C":
        np.empty((2, 0), dtype=int)
}

node_features_original = {
    "C1=CC=NC=C1": np.asarray([[1.], [1.], [2.], [1.], [1.], [1.]]),
    "CC(=O)C": np.asarray([[1.], [1.], [3.], [1.]]),
    "C": np.asarray([[1]])
}

# Set up tests.
smiles = ["C1=CC=NC=C1", "CC(=O)C", "C", "CP"]
edge_index_original_order = list(edge_index_original_ordering.values())
node_features_original_order = list(node_features_original.values())


def test_featurizer_ring():
    """
    Test for featurization of "C1=CC=NC=C1" using `MXMNetFeaturizer` class.
    """

    featurizer = MXMNetFeaturizer()
    graph_feat = featurizer.featurize(smiles)
    assert len(graph_feat) == 4

    assert graph_feat[0].num_nodes == 6
    assert graph_feat[0].num_node_features == 1
    assert graph_feat[0].node_features.shape == (6, 1)
    assert graph_feat[0].num_edges == 12
    assert (
        graph_feat[0].node_features == node_features_original_order[0]).all()

    assert (graph_feat[0].edge_index == edge_index_original_order[0]).all()


def test_featurizer_general_case():
    """
    Test for featurization of "CC(=O)C" using `MXMNetFeaturizer` class.
    """

    featurizer = MXMNetFeaturizer()
    graph_feat = featurizer.featurize(smiles)
    assert len(graph_feat) == 4

    assert graph_feat[1].num_nodes == 4
    assert graph_feat[1].num_node_features == 1
    assert graph_feat[1].node_features.shape == (4, 1)
    assert graph_feat[1].num_edges == 6
    assert (
        graph_feat[1].node_features == node_features_original_order[1]).all()

    assert (graph_feat[1].edge_index == edge_index_original_order[1]).all()


def test_featurizer_single_atom():
    """
    Test for featurization of "C" using `MXMNetFeaturizer` class.
    """

    featurizer = MXMNetFeaturizer()
    graph_feat = featurizer.featurize(smiles)
    assert len(graph_feat) == 4

    assert graph_feat[2].num_nodes == 1
    assert graph_feat[2].num_node_features == 1
    assert graph_feat[2].node_features.shape == (1, 1)
    assert graph_feat[2].num_edges == 0
    assert (
        graph_feat[2].node_features == node_features_original_order[2]).all()

    assert (graph_feat[2].edge_index == edge_index_original_order[2]).all()


def test_featurizer_other_atom():
    """
    Test for featurization of "CP" using `MXMNetFeaturizer` class.
    """

    featurizer = MXMNetFeaturizer()
    graph_feat = featurizer.featurize(smiles)
    assert len(graph_feat) == 4

    assert graph_feat[3].shape == (0,)

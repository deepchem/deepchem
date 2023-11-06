"""
Test for DMPNN Featurizer class.
"""

from deepchem.feat.molecule_featurizers.dmpnn_featurizer import DMPNNFeaturizer, GraphConvConstants
import numpy as np
import pytest

edge_index_orignal_ordering = {
    "C1=CC=NC=C1":
        np.asarray([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0],
                    [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5]]),
    "CC(=O)C":
        np.asarray([[0, 1, 1, 2, 1, 3], [1, 0, 2, 1, 3, 1]]),
    "C":
        np.empty((2, 0), dtype=int)
}

# Set up tests.
smiles = ["C1=CC=NC=C1", "CC(=O)C", "C"]
edge_index_orignal_order = list(edge_index_orignal_ordering.values())

# Set up testing parameters.
Test1_params = {
    'features_generators': None,
    'is_adding_hs': False,
    'use_original_atom_ranks': False
}
Test2_params = {
    'features_generators': None,
    'is_adding_hs': False,
    'use_original_atom_ranks': True
}
Test3_params = {
    'features_generators': None,
    'is_adding_hs': True,
    'use_original_atom_ranks': False
}
Test4_params = {
    'features_generators': ['morgan'],
    'is_adding_hs': False,
    'use_original_atom_ranks': False
}
Test5_params = {
    'features_generators': ['morgan'],
    'is_adding_hs': True,
    'use_original_atom_ranks': False
}


@pytest.mark.parametrize(
    'test_parameters',
    [Test1_params, Test2_params, Test3_params, Test4_params, Test5_params])
def test_featurizer_ring(test_parameters):
    """
    Test for featurization of "C1=CC=NC=C1" using `DMPNNFeaturizer` class.
    """
    features_generators, is_adding_hs, use_original_atom_ranks = test_parameters.values(
    )
    featurizer = DMPNNFeaturizer(
        features_generators=features_generators,
        is_adding_hs=is_adding_hs,
        use_original_atom_ranks=use_original_atom_ranks)
    graph_feat = featurizer.featurize(smiles)
    assert len(graph_feat) == 3

    if is_adding_hs:
        assert graph_feat[0].num_nodes == 11
        assert graph_feat[0].num_node_features == GraphConvConstants.ATOM_FDIM
        assert graph_feat[0].node_features.shape == (
            11, GraphConvConstants.ATOM_FDIM)
        assert graph_feat[0].num_edges == 22
        assert graph_feat[0].num_edge_features == GraphConvConstants.BOND_FDIM
        assert graph_feat[0].edge_features.shape == (
            22, GraphConvConstants.BOND_FDIM)
    else:
        assert graph_feat[0].num_nodes == 6
        assert graph_feat[0].num_node_features == GraphConvConstants.ATOM_FDIM
        assert graph_feat[0].node_features.shape == (
            6, GraphConvConstants.ATOM_FDIM)
        assert graph_feat[0].num_edges == 12
        assert graph_feat[0].num_edge_features == GraphConvConstants.BOND_FDIM
        assert graph_feat[0].edge_features.shape == (
            12, GraphConvConstants.BOND_FDIM)

    if features_generators:
        assert len(graph_feat[0].global_features
                  ) == 2048  # for `morgan` features generator
        nonzero_features_indicies = graph_feat[0].global_features.nonzero()[0]
        if is_adding_hs:
            assert len(nonzero_features_indicies) == 10
        else:
            assert len(nonzero_features_indicies) == 9
    else:
        assert graph_feat[0].global_features.size == 0

    if use_original_atom_ranks:
        assert (graph_feat[0].edge_index == edge_index_orignal_order[0]).all()
    else:
        if np.array_equal(graph_feat[0].edge_index.shape,
                          edge_index_orignal_order[0]):
            assert (graph_feat[0].edge_index !=
                    edge_index_orignal_order[0]).any()


@pytest.mark.parametrize(
    'test_parameters',
    [Test1_params, Test2_params, Test3_params, Test4_params, Test5_params])
def test_featurizer_general_case(test_parameters):
    """
    Test for featurization of "CC(=O)C" using `DMPNNFeaturizer` class.
    """
    features_generators, is_adding_hs, use_original_atom_ranks = test_parameters.values(
    )
    featurizer = DMPNNFeaturizer(
        features_generators=features_generators,
        is_adding_hs=is_adding_hs,
        use_original_atom_ranks=use_original_atom_ranks)
    graph_feat = featurizer.featurize(smiles)
    assert len(graph_feat) == 3

    if is_adding_hs:
        assert graph_feat[1].num_nodes == 10
        assert graph_feat[1].num_node_features == GraphConvConstants.ATOM_FDIM
        assert graph_feat[1].node_features.shape == (
            10, GraphConvConstants.ATOM_FDIM)
        assert graph_feat[1].num_edges == 18
        assert graph_feat[1].num_edge_features == GraphConvConstants.BOND_FDIM
        assert graph_feat[1].edge_features.shape == (
            18, GraphConvConstants.BOND_FDIM)

    else:
        assert graph_feat[1].num_nodes == 4
        assert graph_feat[1].num_node_features == GraphConvConstants.ATOM_FDIM
        assert graph_feat[1].node_features.shape == (
            4, GraphConvConstants.ATOM_FDIM)
        assert graph_feat[1].num_edges == 6
        assert graph_feat[1].num_edge_features == GraphConvConstants.BOND_FDIM
        assert graph_feat[1].edge_features.shape == (
            6, GraphConvConstants.BOND_FDIM)

    if features_generators:
        assert len(graph_feat[1].global_features
                  ) == 2048  # for `morgan` features generator
        nonzero_features_indicies = graph_feat[1].global_features.nonzero()[0]
        if is_adding_hs:
            assert len(nonzero_features_indicies) == 10
        else:
            assert len(nonzero_features_indicies) == 6
    else:
        assert graph_feat[1].global_features.size == 0

    if use_original_atom_ranks:
        assert (graph_feat[1].edge_index == edge_index_orignal_order[1]).all()
    else:
        if np.array_equal(graph_feat[1].edge_index.shape,
                          edge_index_orignal_order[1]):
            assert (graph_feat[1].edge_index !=
                    edge_index_orignal_order[1]).any()


@pytest.mark.parametrize(
    'test_parameters',
    [Test1_params, Test2_params, Test3_params, Test4_params, Test5_params])
def test_featurizer_single_atom(test_parameters):
    """
    Test for featurization of "C" using `DMPNNFeaturizer` class.
    """
    features_generators, is_adding_hs, use_original_atom_ranks = test_parameters.values(
    )
    featurizer = DMPNNFeaturizer(
        features_generators=features_generators,
        is_adding_hs=is_adding_hs,
        use_original_atom_ranks=use_original_atom_ranks)
    graph_feat = featurizer.featurize(smiles)
    assert len(graph_feat) == 3

    if is_adding_hs:
        assert graph_feat[2].num_nodes == 5
        assert graph_feat[2].num_node_features == GraphConvConstants.ATOM_FDIM
        assert graph_feat[2].node_features.shape == (
            5, GraphConvConstants.ATOM_FDIM)
        assert graph_feat[2].num_edges == 8
        assert graph_feat[2].num_edge_features == GraphConvConstants.BOND_FDIM
        assert graph_feat[2].edge_features.shape == (
            8, GraphConvConstants.BOND_FDIM)
    else:
        assert graph_feat[2].num_nodes == 1
        assert graph_feat[2].num_node_features == GraphConvConstants.ATOM_FDIM
        assert graph_feat[2].node_features.shape == (
            1, GraphConvConstants.ATOM_FDIM)
        assert graph_feat[2].num_edges == 0
        assert graph_feat[2].num_edge_features == GraphConvConstants.BOND_FDIM
        assert graph_feat[2].edge_features.shape == (
            0, GraphConvConstants.BOND_FDIM)

    if features_generators:
        assert len(graph_feat[2].global_features
                  ) == 2048  # for `morgan` features generator
        nonzero_features_indicies = graph_feat[2].global_features.nonzero()[0]
        if is_adding_hs:
            assert len(nonzero_features_indicies) == 4
        else:
            assert len(nonzero_features_indicies) == 1
    else:
        assert graph_feat[2].global_features.size == 0

    if use_original_atom_ranks:
        assert (graph_feat[2].edge_index == edge_index_orignal_order[2]).all()
    else:
        if np.array_equal(graph_feat[2].edge_index.shape,
                          edge_index_orignal_order[2]):
            # the atom order for 'C' is same in case of canonical and original ordering
            assert (
                graph_feat[2].edge_index == edge_index_orignal_order[2]).all()

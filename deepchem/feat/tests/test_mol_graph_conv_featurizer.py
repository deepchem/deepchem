import unittest

from deepchem.feat import MolGraphConvFeaturizer


class TestMolGraphConvFeaturizer(unittest.TestCase):

  def test_default_featurizer(self):
    smiles = ["C1=CC=CN=C1", "O=C(NCc1cc(OC)c(O)cc1)CCCC/C=C/C(C)C"]
    featurizer = MolGraphConvFeaturizer()
    graph_feat = featurizer.featurize(smiles)
    assert len(graph_feat) == 2

    # assert "C1=CC=CN=C1"
    assert graph_feat[0].num_nodes == 6
    assert graph_feat[0].num_node_features == 39
    assert graph_feat[0].num_edges == 12
    assert graph_feat[0].num_edge_features == 11

    # assert "O=C(NCc1cc(OC)c(O)cc1)CCCC/C=C/C(C)C"
    assert graph_feat[1].num_nodes == 22
    assert graph_feat[1].num_node_features == 39
    assert graph_feat[1].num_edges == 44
    assert graph_feat[1].num_edge_features == 11

  def test_featurizer_with_self_loop(self):
    smiles = ["C1=CC=CN=C1", "O=C(NCc1cc(OC)c(O)cc1)CCCC/C=C/C(C)C"]
    featurizer = MolGraphConvFeaturizer(add_self_edges=True)
    graph_feat = featurizer.featurize(smiles)
    assert len(graph_feat) == 2

    # assert "C1=CC=CN=C1"
    assert graph_feat[0].num_nodes == 6
    assert graph_feat[0].num_node_features == 39
    assert graph_feat[0].num_edges == 12 + 6
    assert graph_feat[0].num_edge_features == 11

    # assert "O=C(NCc1cc(OC)c(O)cc1)CCCC/C=C/C(C)C"
    assert graph_feat[1].num_nodes == 22
    assert graph_feat[1].num_node_features == 39
    assert graph_feat[1].num_edges == 44 + 22
    assert graph_feat[1].num_edge_features == 11

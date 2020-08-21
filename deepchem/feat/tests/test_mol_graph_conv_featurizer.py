import unittest

from deepchem.feat import MolGraphConvFeaturizer


# TODO: Add more test cases
class TestMolGraphConvFeaturizer(unittest.TestCase):
  def test_default_featurizer(self):
    smiles = ["C1=CC=CN=C1", "O=C(NCc1cc(OC)c(O)cc1)CCCC/C=C/C(C)C"]
    featurizer = MolGraphConvFeaturizer()
    graph_feat = featurizer.featurize(smiles)
    assert len(graph_feat) == 2

    # assert "C1=CC=CN=C1"
    assert graph_feat[0].num_nodes == 6
    assert graph_feat[0].num_node_features == 25
    assert graph_feat[0].num_edges == 12
    assert graph_feat[0].num_edge_features == 13

    # assert "O=C(NCc1cc(OC)c(O)cc1)CCCC/C=C/C(C)C"
    assert graph_feat[1].num_nodes == 22
    assert graph_feat[1].num_node_features == 25
    assert graph_feat[1].num_edges == 44
    assert graph_feat[1].num_edge_features == 13

  def test_mpnn_style_featurizer(self):
    smiles = ["C1=CC=CN=C1", "O=C(NCc1cc(OC)c(O)cc1)CCCC/C=C/C(C)C"]
    featurizer = MolGraphConvFeaturizer(use_mpnn_style=True)
    graph_feat = featurizer.featurize(smiles)
    assert len(graph_feat) == 2

    # assert "C1=CC=CN=C1"
    assert graph_feat[0].num_nodes == 6
    assert graph_feat[0].num_node_features == 17
    assert graph_feat[0].num_edges == 12
    assert graph_feat[0].num_edge_features == 5

    # assert "O=C(NCc1cc(OC)c(O)cc1)CCCC/C=C/C(C)C"
    assert graph_feat[1].num_nodes == 22
    assert graph_feat[1].num_node_features == 17
    assert graph_feat[1].num_edges == 44
    assert graph_feat[1].num_edge_features == 5

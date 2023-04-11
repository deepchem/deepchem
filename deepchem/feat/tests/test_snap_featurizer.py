from deepchem.feat.molecule_featurizers.snap_featurizer import SNAPFeaturizer, EgoGraphFeaturizer


def test_snap_featurizer():
    smiles = ["C1=CC=NC=C1", "CC(=O)C", "C"]
    featurizer = SNAPFeaturizer()
    features = featurizer.featurize(smiles)
    assert len(features) == 3
    assert features[0].node_features.shape == (6, 2)
    assert features[1].edge_index.shape == (2, 6)
    assert features[2].edge_features.shape == (0, 2)


def test_ego_graph_featurizer():
    import networkx as nx
    from deepchem.feat.graph_data import GraphData
    # Create ego graphs

    g1 = nx.Graph()
    g1.add_node("1")
    g1.add_node("2")
    g1.add_edge("1", "2", w1=1, w2=2, w3=3, w4=4, w5=5, w6=6, w7=7)

    g2 = nx.Graph()
    g2.add_node("3")
    g2.add_node("4")
    g2.add_edge("3", "4", w1=1, w2=2, w3=3, w4=4, w5=5, w6=6, w7=7)

    g3 = nx.Graph()
    g3.add_node("5")

    graphs = [g1, g2, g3]
    center_ids = ["1", "3", "5"]

    # Featurize ego graphs
    featurizer = EgoGraphFeaturizer()
    features = featurizer.featurize(zip(graphs, center_ids))

    # Check the number of features and their shapes
    assert len(features) == 3
    assert isinstance(features[0], GraphData)
    assert features[0].node_features.shape == (2, 1)
    assert features[0].edge_index.shape == (2, 2)
    assert features[0].edge_features.shape == (2, 9)
    assert features[1].node_features.shape == (2, 1)
    assert features[1].edge_index.shape == (2, 2)
    assert features[1].edge_features.shape == (2, 9)
    assert features[2].shape == (0,)

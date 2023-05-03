from deepchem.feat.molecule_featurizers.conformer_featurizer import ConformerFeaturizer


def test_conformer_featurizer():
    smiles = ["C1=CC=NC=C1", "CC(=O)C", "C"]
    featurizzer = ConformerFeaturizer()
    features = featurizzer.featurize(smiles)
    assert len(features) == 3  # 3 molecules
    assert features[0].node_features.shape[1] == 9  # 9 atom features
    # every edge index has a feature
    assert all([
        graph.edge_index.shape[1] == graph.edge_features.shape[0]
        for graph in features
    ])
    assert features[2].edge_features.shape[1] == 3  # 3 bond features

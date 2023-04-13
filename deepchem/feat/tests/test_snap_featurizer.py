from deepchem.feat.molecule_featurizers.snap_featurizer import SNAPFeaturizer


def test_snap_featurizer():
    smiles = ["C1=CC=NC=C1", "CC(=O)C", "C"]
    featurizer = SNAPFeaturizer()
    features = featurizer.featurize(smiles)
    assert len(features) == 3
    assert features[0].node_features.shape == (6, 2)
    assert features[1].edge_index.shape == (2, 6)
    assert features[2].edge_features.shape == (0, 2)

def test_conformer_featurizer():
    from deepchem.feat.molecule_featurizers.conformer_featurizer import RDKitConformerFeaturizer
    from deepchem.feat.graph_data import BatchGraphData
    import numpy as np
    smiles = ["C1=CC=NC=C1", "CC(=O)C", "C"]
    featurizer = RDKitConformerFeaturizer(num_conformers=2)
    features_list = featurizer.featurize(smiles)
    features = BatchGraphData(np.concatenate(features_list).ravel())
    assert features.num_edge_features == 3  # 3 bond features
    assert features.num_node_features == 9  # 9 atom features
    assert features.num_nodes == len(features.graph_index)
    assert features.num_edges == 96

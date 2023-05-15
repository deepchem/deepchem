def test_AtomEncoder():
    import torch

    from deepchem.feat.molecule_featurizers.conformer_featurizer import (
        full_atom_feature_dims,)
    from deepchem.models.torch_models.pna_gnn import AtomEncoder

    atom_encoder = AtomEncoder(emb_dim=32)

    num_samples = 10

    # Create input tensor with values within full_atom_feature_dims
    graph_features = torch.stack([
        torch.randint(low=0, high=dim, size=(num_samples,))
        for dim in full_atom_feature_dims
    ],
                                 dim=1)
    atom_embeddings = atom_encoder(graph_features)
    assert atom_embeddings.shape == (num_samples, 32)


def test_BondEncoder():
    import torch

    from deepchem.feat.molecule_featurizers.conformer_featurizer import (
        full_bond_feature_dims,)
    from deepchem.models.torch_models.pna_gnn import BondEncoder

    bond_encoder = BondEncoder(emb_dim=32)

    num_samples = 10

    # Create input tensor with values within full_bond_feature_dims
    graph_features = torch.stack([
        torch.randint(low=0, high=dim, size=(num_samples,))
        for dim in full_bond_feature_dims
    ],
                                 dim=1)
    bond_embeddings = bond_encoder(graph_features)
    assert bond_embeddings.shape == (num_samples, 32)

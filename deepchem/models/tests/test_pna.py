def test_AtomEncoder():
    import numpy as np
    import torch

    from deepchem.feat.graph_data import BatchGraphData
    from deepchem.feat.molecule_featurizers.conformer_featurizer import (
        RDKitConformerFeaturizer,)
    from deepchem.models.torch_models.pna_gnn import AtomEncoder

    atom_encoder = AtomEncoder(emb_dim=32)
    # simulate features from RDKitConformerFeaturizer
    graph_features = torch.tensor([[5., 0., 3., 5., 0., 0., 1., 1., 1.],
                                   [5., 0., 3., 5., 0., 0., 1., 1., 1.]])
    atom_embeddings = atom_encoder(graph_features)


def test_BondEncoder():
    import torch

    from deepchem.models.torch_models.pna_gnn import BondEncoder

    bond_encoder = BondEncoder(emb_dim=32)
    bond_features = torch.randn(3, 3)
    bond_embeddings = bond_encoder(bond_features)


def test_PNAlayer():
    pass


test_AtomEncoder()

def test_AtomEncoder():
    import torch

    from deepchem.models.torch_models.pna_gnn import AtomEncoder

    atom_encoder = AtomEncoder(emb_dim=32)
    atom_features = torch.randn(3, 9)
    atom_embeddings = atom_encoder(atom_features)

def test_BondEncoder():
    import torch

    from deepchem.models.torch_models.pna_gnn import BondEncoder

    bond_encoder = BondEncoder(emb_dim=32)
    bond_features = torch.randn(3, 3)
    bond_embeddings = bond_encoder(bond_features)

def test_PNAlayer():
    pass


test_AtomEncoder()
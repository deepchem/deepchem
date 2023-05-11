def test_AtomEncoder():
    import torch

    from deepchem.models.torch_models.pna_gnn import AtomEncoder
    from deepchem.feat.molecule_featurizers.conformer_featurizer import RDKitConformerFeaturizer
    
    atom_encoder = AtomEncoder(emb_dim=32)
    smiles = ["C1=CC=NC=C1", "CC(=O)C", "C"]
    featurizer = RDKitConformerFeaturizer(num_conformers=5, rmsd_cutoff=1)
    features_list = featurizer.featurize(smiles)
    features = BatchGraphData(np.concatenate(features_list).ravel())
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
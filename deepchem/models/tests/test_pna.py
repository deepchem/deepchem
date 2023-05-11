def test_pna():
    import torch

    from deepchem.models.torch_models.pna_gnn import PNA
    # Create a PNA model.
    model = PNA(hidden_dim=16,
                target_dim=1,
                aggregators=['mean', 'sum'],
                scalers=['identity'],
                readout_aggregators=['mean'])

    # Check that the model can be forward-propagated.
    x = torch.randn(3, 3)
    y = model(x)
    assert y.shape == (3, 1)


def test_pnagnn():
    import dgl
    import torch

    from deepchem.models.torch_models.pna_gnn import PNAGNN
    # Create a PNAGNN model.
    model = PNAGNN(hidden_dim=16,
                   aggregators=['mean', 'sum'],
                   scalers=['identity'])

    # Check that the model can be forward-propagated.
    g = dgl.graph(([0, 1, 2], [1, 2, 0]))
    g.ndata['x'] = torch.randn(3, 3)
    g.edata['edge_attr'] = torch.randn(3, 3)
    y = model(g)
    assert y.shape == (3, 1)


def test_AtomEncoder():
    import torch

    from deepchem.models.torch_models.pna_gnn import AtomEncoder

    atom_encoder = AtomEncoder(emb_dim=32)
    atom_features = torch.tensor([[1, 6, 0], [2, 7, 1]])
    atom_embeddings = atom_encoder(atom_features)

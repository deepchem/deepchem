def test_Net3DLayer():
    import dgl
    import numpy as np
    import torch

    from deepchem.models.torch_models.gnn3d import Net3DLayer
    g = dgl.graph(([0, 1], [1, 2]))
    g.ndata['feat'] = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    g.edata['d'] = torch.tensor([[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]])

    hidden_dim = 3
    batch_norm = True
    batch_norm_momentum = 0.1
    dropout = 0.1
    net3d_layer = Net3DLayer(edge_dim=hidden_dim,
                             hidden_dim=hidden_dim,
                             batch_norm=batch_norm,
                             batch_norm_momentum=batch_norm_momentum,
                             dropout=dropout)

    output_graph = net3d_layer(g)

    assert output_graph.number_of_nodes() == g.number_of_nodes()
    assert output_graph.number_of_edges() == g.number_of_edges()

    output_feats = output_graph.ndata['feat'].detach().numpy()
    assert output_feats.shape == (3, 3)
    assert not np.allclose(output_feats, g.ndata['feat'].detach().numpy())

    output_edge_feats = output_graph.edata['d'].detach().numpy()
    assert output_edge_feats.shape == (2, 3)
    assert not np.allclose(output_edge_feats, g.edata['d'].detach().numpy())

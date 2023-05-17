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


def get_regression_dataset():
    import os

    import numpy as np

    import deepchem as dc
    from deepchem.feat.molecule_featurizers.conformer_featurizer import (
        RDKitConformerFeaturizer,)
    np.random.seed(123)
    featurizer = RDKitConformerFeaturizer(num_conformers=2)
    dir = os.path.dirname(os.path.abspath(__file__))

    input_file = os.path.join(dir, 'assets/example_regression.csv')
    loader = dc.data.CSVLoader(tasks=["outcome"],
                               feature_field="smiles",
                               featurizer=featurizer)
    dataset = loader.create_dataset(input_file)
    metric = dc.metrics.Metric(dc.metrics.mean_absolute_error,
                               mode="regression")
    return dataset, metric


def test_net3d():
    # import dgl
    import numpy as np

    # import torch
    # # Create a DGL graph
    # graph = dgl.graph(([0, 1], [1, 2]))
    # graph.ndata['feat'] = torch.tensor([[1., 2., 3.], [4., 5., 6.],
    #                                     [7., 8., 9.]])
    # graph.edata['d'] = torch.tensor([[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]])
    from deepchem.feat.graph_data import BatchGraphData
    from deepchem.models.torch_models.gnn3d import Net3D
    # from deepchem.models.torch_models.pna_gnn import PNAGNN
    import torch
    data, _ = get_regression_dataset()
    features = BatchGraphData(np.concatenate(data.X).ravel())
    graph = features.to_dgl_graph()
    target_dim = 2

    # Instantiate a Net3D model
    net3d = Net3D(hidden_dim=3,
                  target_dim=target_dim,
                  readout_aggregators=['sum', 'mean'])

    graph.edata['d'] = torch.norm(graph.ndata['x'][graph.edges()[0]] -
                                  graph.ndata['x'][graph.edges()[1]],
                                  p=2,
                                  dim=-1).unsqueeze(-1).detach()
    # Perform a forward pass on the graph
    output = net3d(graph)

    # Check the output shape and values
    assert output.shape[1] == target_dim



# test_Net3DLayer()
test_net3d()

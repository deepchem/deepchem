import pytest


@pytest.mark.pytorch
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
    featurizer = RDKitConformerFeaturizer(num_conformers=2, rmsd_cutoff=3)
    dir = os.path.dirname(os.path.abspath(__file__))

    input_file = os.path.join(dir, 'assets/example_regression.csv')
    loader = dc.data.CSVLoader(tasks=["outcome"],
                               feature_field="smiles",
                               featurizer=featurizer)
    dataset = loader.create_dataset(input_file)
    metric = dc.metrics.Metric(dc.metrics.mean_absolute_error,
                               mode="regression")
    return dataset, metric


@pytest.mark.pytorch
def test_net3d():
    import numpy as np

    from deepchem.feat.graph_data import BatchGraphData
    from deepchem.models.torch_models.gnn3d import Net3D
    data, _ = get_regression_dataset()
    features = BatchGraphData(np.concatenate(data.X))
    graph = features.to_dgl_graph()
    target_dim = 2

    net3d = Net3D(hidden_dim=3,
                  target_dim=target_dim,
                  readout_aggregators=['sum', 'mean'])

    output = net3d(graph)

    assert output.shape[1] == target_dim


@pytest.mark.pytorch
def test_InfoMax3DModular():
    from deepchem.models.torch_models.gnn3d import InfoMax3DModular

    data, _ = get_regression_dataset()

    model = InfoMax3DModular(hidden_dim=64,
                             target_dim=10,
                             aggregators=['sum', 'mean', 'max'],
                             readout_aggregators=['sum', 'mean'],
                             scalers=['identity'])

    loss1 = model.fit(data, nb_epoch=1)
    loss2 = model.fit(data, nb_epoch=9)
    assert loss1 > loss2

import os
import pytest
from flaky import flaky


@pytest.mark.torch
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


def get_classification_dataset():
    import numpy as np
    import deepchem as dc
    from deepchem.feat.molecule_featurizers.conformer_featurizer import RDKitConformerFeaturizer
    dir = os.path.dirname(os.path.abspath(__file__))

    featurizer = RDKitConformerFeaturizer()
    input_file = os.path.join(dir, 'assets/example_classification.csv')
    loader = dc.data.CSVLoader(tasks=["outcome"],
                               feature_field="smiles",
                               featurizer=featurizer)
    dataset = loader.create_dataset(input_file)
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score,
                               np.mean,
                               mode="classification")
    return dataset, metric


def get_regression_dataset():
    import numpy as np
    import deepchem as dc
    from deepchem.feat.molecule_featurizers.conformer_featurizer import (
        RDKitConformerFeaturizer,)

    np.random.seed(123)
    featurizer = RDKitConformerFeaturizer()
    dir = os.path.dirname(os.path.abspath(__file__))

    input_file = os.path.join(dir, 'assets/example_regression.csv')
    loader = dc.data.CSVLoader(tasks=["outcome"],
                               feature_field="smiles",
                               featurizer=featurizer)
    dataset = loader.create_dataset(input_file)
    metric = dc.metrics.Metric(dc.metrics.mean_absolute_error,
                               mode="regression")
    return dataset, metric


@pytest.mark.torch
def test_net3d():
    from deepchem.models.torch_models.gnn3d import Net3D
    import dgl
    data, _ = get_regression_dataset()
    graphs = dgl.batch([conformer.to_dgl_graph() for conformer in data.X])
    target_dim = 2

    net3d = Net3D(hidden_dim=3,
                  target_dim=target_dim,
                  readout_aggregators=['sum', 'mean'])

    output = net3d(graphs)

    assert output.shape[1] == target_dim


def compare_weights(key, model1, model2):
    import torch
    return torch.all(
        torch.eq(model1.components[key].weight,
                 model2.components[key].weight)).item()


@flaky
@pytest.mark.torch
def testInfoMax3DModular():
    import torch
    from deepchem.models.torch_models.gnn3d import InfoMax3DModular
    torch.manual_seed(456)
    data, _ = get_regression_dataset()

    model = InfoMax3DModular(hidden_dim=64,
                             target_dim=10,
                             aggregators=['sum', 'mean', 'max'],
                             readout_aggregators=['sum', 'mean'],
                             scalers=['identity'],
                             device=torch.device('cpu'),
                             task='pretraining',
                             learning_rate=0.00001)

    loss1 = model.fit(data, nb_epoch=1)
    loss2 = model.fit(data, nb_epoch=9)
    assert loss1 > loss2


@pytest.mark.torch
def testInfoMax3DModularSaveReload():
    import torch
    from deepchem.models.torch_models.gnn3d import InfoMax3DModular

    data, _ = get_regression_dataset()
    model = InfoMax3DModular(hidden_dim=64,
                             target_dim=10,
                             aggregators=['sum', 'mean', 'max'],
                             readout_aggregators=['sum', 'mean'],
                             scalers=['identity'],
                             device=torch.device('cpu'),
                             task='pretraining')

    model.fit(data, nb_epoch=1)
    model2 = InfoMax3DModular(hidden_dim=64,
                              target_dim=10,
                              aggregators=['sum', 'mean', 'max'],
                              readout_aggregators=['sum', 'mean'],
                              scalers=['identity'],
                              task='pretraining')

    model2.load_from_pretrained(model_dir=model.model_dir)
    assert model.components.keys() == model2.components.keys()
    keys_with_weights = [
        key for key in model.components.keys()
        if hasattr(model.components[key], 'weight')
    ]
    assert all(compare_weights(key, model, model2) for key in keys_with_weights)


@flaky
@pytest.mark.torch
def testInfoMax3DModularRegression():
    import torch
    from deepchem.models.torch_models.gnn3d import InfoMax3DModular

    data, metric = get_regression_dataset()

    model = InfoMax3DModular(hidden_dim=64,
                             aggregators=['sum', 'mean', 'max'],
                             readout_aggregators=['sum', 'mean'],
                             scalers=['identity'],
                             task='regression',
                             n_tasks=1,
                             device=torch.device('cpu'))

    model.fit(data, nb_epoch=100)
    scores = model.evaluate(data, [metric])
    print(scores)
    assert scores['mean_absolute_error'] < 0.5


@flaky
@pytest.mark.torch
def testInfoMax3DModularClassification():
    import torch
    from deepchem.models.torch_models.gnn3d import InfoMax3DModular
    torch.manual_seed(1)
    data, metric = get_classification_dataset()

    model = InfoMax3DModular(hidden_dim=128,
                             aggregators=['sum', 'mean', 'max'],
                             readout_aggregators=['sum', 'mean'],
                             scalers=['identity'],
                             task='classification',
                             n_tasks=1,
                             n_classes=2,
                             device=torch.device('cpu'))

    model.fit(data, nb_epoch=10)
    scores = model.evaluate(data, [metric])
    assert scores['mean-roc_auc_score'] > 0.7


@pytest.mark.torch
def test_infomax3d_load_from_pretrained(tmpdir):
    import torch
    from deepchem.models.torch_models.gnn3d import InfoMax3DModular
    pretrain_model = InfoMax3DModular(hidden_dim=64,
                                      target_dim=10,
                                      device=torch.device('cpu'),
                                      task='pretraining',
                                      model_dir=tmpdir)
    pretrain_model._ensure_built()
    pretrain_model.save_checkpoint()
    pretrain_model_state_dict = pretrain_model.model.state_dict()

    finetune_model = InfoMax3DModular(hidden_dim=64,
                                      target_dim=10,
                                      device=torch.device('cpu'),
                                      task='classification',
                                      n_classes=2,
                                      n_tasks=1)
    finetune_model_old_state_dict = finetune_model.model.state_dict()
    # Finetune model weights should not match before loading from pretrained model
    for key, value in pretrain_model_state_dict.items():
        assert not torch.allclose(value, finetune_model_old_state_dict[key])
    finetune_model.load_from_pretrained(pretrain_model, components=['model2d'])
    finetune_model_new_state_dict = finetune_model.model.state_dict()

    # Finetune model weights should match after loading from pretrained model
    for key, value in pretrain_model_state_dict.items():
        assert torch.allclose(value, finetune_model_new_state_dict[key])

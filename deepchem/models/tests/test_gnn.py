import os
import pytest
import numpy as np
from flaky import flaky
import deepchem as dc
from deepchem.feat.molecule_featurizers import SNAPFeaturizer


def get_regression_dataset():
    np.random.seed(123)
    featurizer = SNAPFeaturizer()
    dir = os.path.dirname(os.path.abspath(__file__))

    input_file = os.path.join(dir, 'assets/example_regression.csv')
    loader = dc.data.CSVLoader(tasks=["outcome"],
                               feature_field="smiles",
                               featurizer=featurizer)
    dataset = loader.create_dataset(input_file)
    metric = dc.metrics.Metric(dc.metrics.mean_absolute_error,
                               mode="regression")
    return dataset, metric


def compare_weights(key, model1, model2):
    import torch
    return torch.all(
        torch.eq(model1.components[key].weight,
                 model2.components[key].weight)).item()


def get_multitask_regression_dataset():
    featurizer = SNAPFeaturizer()
    dir = os.path.dirname(os.path.abspath(__file__))

    input_file = os.path.join(dir, 'assets/multitask_regression.csv')
    loader = dc.data.CSVLoader(tasks=['task0', 'task1', 'task2'],
                               feature_field="smiles",
                               featurizer=featurizer)
    dataset = loader.create_dataset(input_file)
    metric = dc.metrics.Metric(dc.metrics.mean_absolute_error,
                               mode="regression")
    return dataset, metric


@pytest.mark.torch
def get_multitask_classification_dataset():
    featurizer = SNAPFeaturizer()
    dir = os.path.dirname(os.path.abspath(__file__))

    input_file = os.path.join(dir, 'assets/multitask_example.csv')
    loader = dc.data.CSVLoader(tasks=['task0', 'task1', 'task2'],
                               feature_field="smiles",
                               featurizer=featurizer)
    dataset = loader.create_dataset(input_file)
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score,
                               np.mean,
                               mode="classification")
    return dataset, metric


@pytest.mark.torch
def test_GNN_load_from_pretrained():
    from deepchem.models.torch_models.gnn import GNNModular

    dataset, _ = get_regression_dataset()
    model = GNNModular(task="edge_pred")
    model.fit(dataset, nb_epoch=1)
    model2 = GNNModular(task="edge_pred")
    model2.load_from_pretrained(model_dir=model.model_dir)
    assert model.components.keys() == model2.components.keys()
    keys_with_weights = [
        key for key in model.components.keys()
        if hasattr(model.components[key], 'weight')
    ]
    assert all(compare_weights(key, model, model2) for key in keys_with_weights)


@pytest.mark.torch
def test_gnn_reload(tmpdir):
    import torch
    from deepchem.models.torch_models.gnn import GNNModular
    model_config = {
        'gnn_type': 'gin',
        'num_layers': 3,
        'emb_dim': 64,
        'task': 'regression',
        'mask_edge': True,
        'model_dir': tmpdir,
        'device': torch.device('cpu')
    }
    old_model = GNNModular(**model_config)
    old_model._ensure_built()
    old_model.save_checkpoint()
    old_model_state = old_model.model.state_dict()

    new_model = GNNModular(**model_config)
    new_model.restore()
    new_model_state = new_model.model.state_dict()
    for key in new_model_state.keys():
        assert torch.allclose(old_model_state[key], new_model_state[key])


@pytest.mark.torch
def test_GNN_edge_pred():
    """Tests the unsupervised edge prediction task"""
    from deepchem.models.torch_models.gnn import GNNModular

    dataset, _ = get_regression_dataset()
    model = GNNModular(task="edge_pred")
    loss1 = model.fit(dataset, nb_epoch=5)
    loss2 = model.fit(dataset, nb_epoch=5)
    assert loss2 < loss1


@pytest.mark.torch
def test_GNN_node_masking():
    """Tests the unsupervised node masking task"""
    from deepchem.models.torch_models.gnn import GNNModular

    dataset, _ = get_regression_dataset()
    model = GNNModular(task="mask_nodes", device="cpu")
    loss1 = model.fit(dataset, nb_epoch=5)
    loss2 = model.fit(dataset, nb_epoch=5)
    assert loss2 < loss1


@pytest.mark.torch
def test_GNN_edge_masking():
    """Tests the unsupervised node masking task"""
    from deepchem.models.torch_models.gnn import GNNModular

    dataset, _ = get_regression_dataset()
    model = GNNModular(task="mask_edges")
    loss1 = model.fit(dataset, nb_epoch=5)
    loss2 = model.fit(dataset, nb_epoch=5)
    assert loss2 < loss1


@pytest.mark.torch
def test_GNN_regression():
    from deepchem.models.torch_models.gnn import GNNModular

    dataset, metric = get_regression_dataset()
    model = GNNModular(task="regression")
    model.fit(dataset, nb_epoch=100)
    scores = model.evaluate(dataset, [metric])
    assert scores['mean_absolute_error'] < 0.2


@pytest.mark.torch
def test_GNN_multitask_regression():
    from deepchem.models.torch_models.gnn import GNNModular

    dataset, metric = get_multitask_regression_dataset()
    model = GNNModular(task="regression", gnn_type="gcn", num_tasks=3)
    model.fit(dataset, nb_epoch=100)
    scores = model.evaluate(dataset, [metric])
    assert scores['mean_absolute_error'] < 0.2


@pytest.mark.torch
def test_GNN_multitask_classification():
    from deepchem.models.torch_models.gnn import GNNModular

    dataset, metric = get_multitask_classification_dataset()
    model = GNNModular(task="classification", gnn_type='sage', num_tasks=3)
    model.fit(dataset, nb_epoch=200)
    scores = model.evaluate(dataset, [metric])
    assert scores['mean-roc_auc_score'] >= 0.8


@pytest.mark.torch
def test_GNN_infomax():
    from deepchem.models.torch_models.gnn import GNNModular

    dataset, _ = get_regression_dataset()
    model = GNNModular(task="infomax", gnn_type='gat')
    loss1 = model.fit(dataset, nb_epoch=5)
    loss2 = model.fit(dataset, nb_epoch=5)
    assert loss2 < loss1


@flaky(max_runs=3, min_passes=1)
@pytest.mark.torch
def test_GNN_context_pred():
    from deepchem.models.torch_models.gnn import GNNModular

    dataset, _ = get_regression_dataset()
    model = GNNModular(task="context_pred",
                       context_mode="skipgram",
                       jump_knowledge="concat")
    loss1 = model.fit(dataset, nb_epoch=5)
    loss2 = model.fit(dataset, nb_epoch=5)
    assert loss2 < loss1

    model = GNNModular(task="context_pred",
                       context_mode="cbow",
                       jump_knowledge="last")
    loss1 = model.fit(dataset, nb_epoch=5)
    loss2 = model.fit(dataset, nb_epoch=5)
    assert loss2 < loss1

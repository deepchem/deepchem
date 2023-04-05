import os
import pytest
import numpy as np
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
def test_GNN_edge_pred():
    """Tests the unsupervised edge prediction task"""
    from deepchem.models.torch_models.gnn import GNNModular

    dataset, _ = get_regression_dataset()
    model = GNNModular(task="edge_pred")
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
    assert scores['mean_absolute_error'] < 0.1


@pytest.mark.torch
def test_GNN_multitask_regression():
    from deepchem.models.torch_models.gnn import GNNModular

    dataset, metric = get_multitask_regression_dataset()
    model = GNNModular(task="regression", num_tasks=3)
    model.fit(dataset, nb_epoch=100)
    scores = model.evaluate(dataset, [metric])
    assert scores['mean_absolute_error'] < 0.1

test_GNN_multitask_regression()
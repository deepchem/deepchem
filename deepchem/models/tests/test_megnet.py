"""
Test for MEGNetModel
"""
import pytest
import tempfile
import numpy as np
from flaky import flaky
import deepchem as dc
from deepchem.utils.fake_data_generator import FakeGraphGenerator as FGG

try:
    from deepchem.models.torch_models import MEGNetModel
    # When pytest runs without pytorch in the environment (ex: as in tensorflow workflow),
    # the above import raises a ModuleNotFoundError. It is safe to ignore it
    # since the below tests only run in an environment with pytorch installed.
except ModuleNotFoundError:
    pass


@flaky(max_runs=3, min_passes=1)
@pytest.mark.torch
def test_megnet_overfit():
    fgg = FGG(avg_n_nodes=10,
              n_node_features=5,
              avg_degree=4,
              n_edge_features=3,
              global_features=4,
              num_classes=5,
              task='graph')
    graphs = fgg.sample(n_graphs=100)

    model = MEGNetModel(n_node_features=5,
                        n_edge_features=3,
                        n_global_features=4,
                        n_blocks=3,
                        is_undirected=True,
                        residual_connection=True,
                        mode='classification',
                        n_classes=5,
                        batch_size=16,
                        device='cpu')
    metric = dc.metrics.Metric(dc.metrics.accuracy_score, mode="classification")

    model.fit(graphs, nb_epoch=100)
    scores = model.evaluate(graphs, [metric], n_classes=5)
    assert scores['accuracy_score'] == 1.0


@flaky(max_runs=3, min_passes=1)
@pytest.mark.torch
def test_megnet_classification():
    fgg = FGG(avg_n_nodes=10,
              n_node_features=5,
              avg_degree=4,
              n_edge_features=3,
              global_features=4,
              num_classes=10)
    graphs = fgg.sample(n_graphs=200)

    model = MEGNetModel(n_node_features=5,
                        n_edge_features=3,
                        n_global_features=4,
                        n_blocks=3,
                        is_undirected=True,
                        residual_connection=True,
                        mode='classification',
                        n_classes=10,
                        batch_size=16,
                        device='cpu')
    metric = dc.metrics.Metric(dc.metrics.accuracy_score, mode="classification")

    model.fit(graphs, nb_epoch=50)
    scores = model.evaluate(graphs, [metric], n_classes=10)
    assert scores['accuracy_score'] > 0.9


@pytest.mark.torch
def test_megnet_regression():
    # TODO The test is skipped as FakeGraphGenerator has to be updated
    # to generate regression labels
    return


@pytest.mark.torch
def test_megnet_reload():
    fgg = FGG(avg_n_nodes=10,
              n_node_features=5,
              avg_degree=4,
              n_edge_features=3,
              global_features=4,
              num_classes=3)
    graphs = fgg.sample(n_graphs=10)
    test_graphs = fgg.sample(n_graphs=10)

    model_dir = tempfile.mkdtemp()
    model = MEGNetModel(n_node_features=5,
                        n_edge_features=3,
                        n_global_features=4,
                        n_blocks=3,
                        is_undirected=True,
                        residual_connection=True,
                        mode='classification',
                        n_classes=3,
                        batch_size=16,
                        device='cpu',
                        model_dir=model_dir)

    model.fit(graphs, nb_epoch=10)

    reloaded_model = MEGNetModel(n_node_features=5,
                                 n_edge_features=3,
                                 n_global_features=4,
                                 n_blocks=3,
                                 is_undirected=True,
                                 residual_connection=True,
                                 mode='classification',
                                 n_classes=3,
                                 batch_size=16,
                                 device='cpu',
                                 model_dir=model_dir)
    reloaded_model.restore()
    orig_predict = model.predict(test_graphs)
    reloaded_predict = reloaded_model.predict(test_graphs)
    assert np.all(orig_predict == reloaded_predict)

import pytest
import numpy as np
import deepchem as dc
from deepchem.feat.molecule_featurizers import MolGraphConvFeaturizer
from deepchem.data import NumpyDataset
from deepchem.models.torch_models.infograph import InfoGraphModel
from deepchem.molnet import load_bace_classification, load_delaney
try:
    import torch
except:
    pass


@pytest.mark.torch
def get_dataset(mode='classification', num_tasks=1):
    np.random.seed(123)
    data_points = 20
    featurizer = MolGraphConvFeaturizer(use_edges=True)

    if mode == 'classification':
        tasks, all_dataset, transformers = load_bace_classification(featurizer)
    else:
        tasks, all_dataset, transformers = load_delaney(featurizer)

    train, valid, test = all_dataset
    w = torch.ones(size=(data_points, num_tasks)).float()

    if mode == 'classification':
        y = torch.randint(0, 2, size=(data_points, num_tasks)).float()
        metric = dc.metrics.Metric(dc.metrics.roc_auc_score,
                                   np.mean,
                                   mode="classification")
    else:
        y = np.random.normal(size=(data_points, num_tasks))
        metric = dc.metrics.Metric(dc.metrics.mean_absolute_error,
                                   mode="regression")

    ds = NumpyDataset(train.X[:data_points], y, w, train.ids[:data_points])

    return tasks, ds, transformers, metric


@pytest.mark.torch
def test_infograph_regression():
    tasks, dataset, transformers, metric = get_dataset('regression')
    num_feat = max(
        [dataset.X[i].num_node_features for i in range(len(dataset))])
    edge_dim = max(
        [dataset.X[i].num_edge_features for i in range(len(dataset))])
    dim = 64

    model = InfoGraphModel(num_feat,
                      edge_dim,
                      dim,
                      use_unsup_loss=False,
                      separate_encoder=False,
                      batch_size=10)

    model.fit(dataset, nb_epoch=1000)
    scores = model.evaluate(dataset, [metric], transformers)
    assert scores['mean_absolute_error'] < 0.1


@pytest.mark.torch
def test_infograph_classification():

    tasks, dataset, transformers, metric = get_dataset('classification')

    num_feat = max(
        [dataset.X[i].num_node_features for i in range(len(dataset))])
    edge_dim = max(
        [dataset.X[i].num_edge_features for i in range(len(dataset))])
    dim = 64

    model = InfoGraphModel(num_feat,
                      edge_dim,
                      dim,
                      use_unsup_loss=False,
                      separate_encoder=False,
                      batch_size=10)

    model.fit(dataset, nb_epoch=1000)
    scores = model.evaluate(dataset, [metric], transformers)
    assert scores['mean-roc_auc_score'] >= 0.9


@pytest.mark.torch
def test_fit_restore():
    tasks, dataset, transformers, metric = get_dataset('classification')

    num_feat = max(
        [dataset.X[i].num_node_features for i in range(len(dataset))])
    edge_dim = max(
        [dataset.X[i].num_edge_features for i in range(len(dataset))])
    dim = 64

    model = InfoGraphModel(num_feat,
                      edge_dim,
                      dim,
                      use_unsup_loss=False,
                      separate_encoder=False,
                      batch_size=10)

    model.fit(dataset, nb_epoch=1000)

    model2 = InfoGraphModel(num_feat,
                       edge_dim,
                       dim,
                       use_unsup_loss=False,
                       separate_encoder=False,
                       model_dir=model.model_dir)
    model2.fit(dataset, nb_epoch=1, restore=True)
    prediction = model2.predict_on_batch(dataset.X).reshape(-1, 1)
    assert np.allclose(dataset.y, np.round(prediction))

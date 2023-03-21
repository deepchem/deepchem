import os
import pytest
import numpy as np
import deepchem as dc
from deepchem.feat.molecule_featurizers import MolGraphConvFeaturizer


@pytest.mark.torch
def get_classification_dataset():
    np.random.seed(123)
    featurizer = MolGraphConvFeaturizer(use_edges=True)
    dir = os.path.dirname(os.path.abspath(__file__))

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
    np.random.seed(123)
    featurizer = MolGraphConvFeaturizer(use_edges=True)
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
def test_infograph_regression_semisupervised():
    from deepchem.models.torch_models.infograph import InfoGraphModel
    dataset, metric = get_regression_dataset()
    num_feat = max(
        [dataset.X[i].num_node_features for i in range(len(dataset))])
    edge_dim = max(
        [dataset.X[i].num_edge_features for i in range(len(dataset))])
    dim = 64

    model = InfoGraphModel(num_feat,
                           edge_dim,
                           dim,
                           use_unsup_loss=True,
                           separate_encoder=True)

    model.fit(dataset, nb_epoch=100)
    scores = model.evaluate(dataset, [metric])
    assert scores['mean_absolute_error'] < 0.1


@pytest.mark.torch
def test_infograph_regression_unsupervised():
    from deepchem.models.torch_models.infograph import InfoGraphModel
    dataset, metric = get_regression_dataset()
    num_feat = max(
        [dataset.X[i].num_node_features for i in range(len(dataset.X))])
    edge_dim = max(
        [dataset.X[i].num_edge_features for i in range(len(dataset.X))])
    dim = 64

    model = InfoGraphModel(num_feat,
                           edge_dim,
                           dim,
                           use_unsup_loss=True,
                           separate_encoder=False)

    model.fit(dataset, nb_epoch=100)
    scores = model.evaluate(dataset, [metric])
    assert scores['mean_absolute_error'] < 0.1


@pytest.mark.torch
def test_infograph_supervised_classification():
    from deepchem.models.torch_models.infograph import InfoGraphModel
    dataset, metric = get_classification_dataset()
    num_feat = max(
        [dataset.X[i].num_node_features for i in range(len(dataset.X))])
    edge_dim = max(
        [dataset.X[i].num_edge_features for i in range(len(dataset.X))])
    dim = 64

    model = InfoGraphModel(num_feat,
                           edge_dim,
                           dim,
                           use_unsup_loss=False,
                           separate_encoder=False)

    model.fit(dataset, nb_epoch=100)
    scores = model.evaluate(dataset, [metric])
    assert scores['mean-roc_auc_score'] >= 0.9
test_infograph_supervised_classification()

@pytest.mark.torch
def test_fit_restore():
    from deepchem.models.torch_models.infograph import InfoGraphModel
    dataset, _ = get_classification_dataset()
    num_feat = max(
        [dataset.X[i].num_node_features for i in range(len(dataset))])
    edge_dim = max(
        [dataset.X[i].num_edge_features for i in range(len(dataset))])
    dim = 64

    model = InfoGraphModel(num_feat,
                           edge_dim,
                           dim,
                           use_unsup_loss=False,
                           separate_encoder=False)

    model.fit(dataset, nb_epoch=100)

    model2 = InfoGraphModel(num_feat,
                            edge_dim,
                            dim,
                            use_unsup_loss=False,
                            separate_encoder=False,
                            model_dir=model.model_dir)
    model2.fit(dataset, nb_epoch=1, restore=True)
    prediction = model2.predict_on_batch(dataset.X).reshape(-1, 1)
    assert np.allclose(dataset.y, np.round(prediction))

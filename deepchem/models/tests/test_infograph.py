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
def test_infographstar_regression_semisupervised():
    from deepchem.models.torch_models.infograph import InfoGraphStarModel
    dataset, metric = get_regression_dataset()
    num_feat = 30
    edge_dim = 11
    dim = 64
    model = InfoGraphStarModel(num_feat,
                               edge_dim,
                               dim,
                               num_gc_layers=2,
                               training_mode='semisupervised')

    model.fit(dataset, nb_epoch=100)
    scores = model.evaluate(dataset, [metric])
    assert scores['mean_absolute_error'] < 0.1


@pytest.mark.torch
def test_infographstar_classification_semisupervised():
    from deepchem.models.torch_models.infograph import InfoGraphStarModel
    dataset, metric = get_classification_dataset()
    num_feat = 30
    edge_dim = 11
    dim = 64
    model = InfoGraphStarModel(num_feat,
                               edge_dim,
                               dim,
                               num_gc_layers=2,
                               training_mode='semisupervised')

    model.fit(dataset, nb_epoch=100)
    scores = model.evaluate(dataset, [metric])
    assert scores['mean-roc_auc_score'] >= 0.9


@pytest.mark.torch
def test_infographstar_classification_supervised():
    from deepchem.models.torch_models.infograph import InfoGraphStarModel
    dataset, metric = get_classification_dataset()
    num_feat = 30
    edge_dim = 11
    dim = 64

    model = InfoGraphStarModel(num_feat,
                               edge_dim,
                               dim,
                               training_mode='supervised')

    model.fit(dataset, nb_epoch=100)
    scores = model.evaluate(dataset, [metric])
    assert scores['mean-roc_auc_score'] >= 0.9


@pytest.mark.torch
def test_infographstar_regression_supervised():
    from deepchem.models.torch_models.infograph import InfoGraphStarModel
    dataset, metric = get_regression_dataset()
    num_feat = 30
    edge_dim = 11
    dim = 64
    model = InfoGraphStarModel(num_feat,
                               edge_dim,
                               dim,
                               num_gc_layers=3,
                               training_mode='supervised')

    model.fit(dataset, nb_epoch=100)
    scores = model.evaluate(dataset, [metric])
    assert scores['mean_absolute_error'] < 0.1

@pytest.mark.torch
def test_infograph():
    from deepchem.models.torch_models.infograph import InfoGraphModel
    import torch
    torch.manual_seed(123)
    dataset, _ = get_regression_dataset()
    num_feat = 30
    edge_dim = 11
    model = InfoGraphModel(num_feat, edge_dim)
    # first iteration loss is around 50
    loss = model.fit(dataset, nb_epoch=20)
    assert loss < 25

@pytest.mark.torch
def test_fit_restore():
    from deepchem.models.torch_models.infograph import InfoGraphStarModel
    dataset, _ = get_classification_dataset()
    num_feat = max(
        [dataset.X[i].num_node_features for i in range(len(dataset))])
    edge_dim = max(
        [dataset.X[i].num_edge_features for i in range(len(dataset))])
    dim = 64

    model = InfoGraphStarModel(num_feat,
                               edge_dim,
                               dim,
                               training_mode='supervised')

    model.fit(dataset, nb_epoch=100)

    model2 = InfoGraphStarModel(num_feat,
                                edge_dim,
                                dim,
                                training_mode='supervised',
                                model_dir=model.model_dir)
    model2.fit(dataset, nb_epoch=1, restore=True)
    prediction = model2.predict_on_batch(dataset.X).reshape(-1, 1)
    assert np.allclose(dataset.y, np.round(prediction))

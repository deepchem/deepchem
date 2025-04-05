import os
import deepchem as dc
import numpy as np
import pytest
import tempfile
from flaky import flaky
from deepchem.data import NumpyDataset, CSVLoader
from deepchem.feat import ConvMolFeaturizer
from deepchem.metrics import Metric, roc_auc_score, mean_absolute_error
from deepchem.molnet import load_bace_classification, load_delaney
from deepchem.utils.data_utils import download_url, get_data_dir
from deepchem import deepchemmap
from deepchem.models.optimizers import Optimizer,AdamW

try:
    import torch
    from deepchem.models.torch_models import GraphConvModel
    has_torch = True
except:
    has_torch = False


def get_dataset(mode='classification', featurizer='GraphConv', num_tasks=2):
    data_points = 20
    if mode == 'classification':
        tasks, all_dataset, transformers = load_bace_classification(
            featurizer, reload=False)
    else:
        tasks, all_dataset, transformers = load_delaney(featurizer,
                                                        reload=False)

    train, valid, test = all_dataset
    for _ in range(1, num_tasks):
        tasks.append("random_task")
    w = np.ones(shape=(data_points, len(tasks)))

    if mode == 'classification':
        y = np.random.randint(0, 2, size=(data_points, len(tasks)))
        metric = Metric(roc_auc_score, np.mean, mode="classification")
    else:
        y = np.random.normal(size=(data_points, len(tasks)))
        metric = Metric(mean_absolute_error, mode="regression")

    ds = NumpyDataset(train.X[:data_points], y, w, train.ids[:data_points])

    return tasks, ds, transformers, metric


@flaky
@pytest.mark.torch
def test_graph_conv_model():
    np.random.seed(5)
    torch.manual_seed(5)
    tasks, dataset, transformers, metric = get_dataset('classification',
                                                       'GraphConv')

    batch_size = 10
    model = GraphConvModel(len(tasks),
                           number_input_features=[75, 64],
                           batch_size=batch_size,
                           batch_normalize=False,
                           mode='classification',
                           device='cpu')

    model.fit(dataset, nb_epoch=20)
    scores = model.evaluate(dataset, [metric], transformers)
    assert scores['mean-roc_auc_score'] >= 0.9


@pytest.mark.torch
def test_neural_fingerprint_retrieval():
    tasks, dataset, _, _ = get_dataset('classification', 'GraphConv')

    fp_size = 3

    batch_size = 50
    model = GraphConvModel(len(tasks),
                           number_input_features=[75, 64],
                           batch_size=batch_size,
                           dense_layer_size=3,
                           mode='classification',
                           device='cpu')

    model.fit(dataset, nb_epoch=1)
    neural_fingerprints = model.predict_embedding(dataset)
    neural_fingerprints = np.array(neural_fingerprints)[:len(dataset)]
    assert (len(dataset), fp_size * 2) == neural_fingerprints.shape


@flaky
@pytest.mark.torch
def test_graph_conv_regression_model():
    np.random.seed(7)
    torch.manual_seed(7)
    tasks, dataset, transformers, metric = get_dataset('regression',
                                                       'GraphConv')

    batch_size = 10
    model = GraphConvModel(len(tasks),
                           number_input_features=[75, 64],
                           batch_size=batch_size,
                           batch_normalize=False,
                           mode='regression',
                           device='cpu')

    model.fit(dataset, nb_epoch=100)
    scores = model.evaluate(dataset, [metric], transformers)
    assert scores['mean_absolute_error'] < 0.15


@pytest.mark.torch
def test_graph_conv_regression_uncertainty():
    tasks, dataset, _, _ = get_dataset('regression', 'GraphConv')

    batch_size = 10
    model = GraphConvModel(len(tasks),
                           number_input_features=[75, 64],
                           batch_size=batch_size,
                           batch_normalize=False,
                           mode='regression',
                           dropout=0.1,
                           uncertainty=True,
                           device='cpu')

    model.fit(dataset, nb_epoch=100)

    # Predict the output and uncertainty.
    pred, std = model.predict_uncertainty(dataset)
    mean_error = np.mean(np.abs(dataset.y - pred))
    mean_value = np.mean(np.abs(dataset.y))
    mean_std = np.mean(std)
    assert mean_error < 0.5 * mean_value
    assert mean_std > 0.5 * mean_error
    assert mean_std < mean_value


@pytest.mark.torch
def test_graph_conv_model_no_task():
    tasks, dataset, _, __ = get_dataset('classification', 'GraphConv')
    batch_size = 10
    model = GraphConvModel(len(tasks),
                           number_input_features=[75, 64],
                           batch_size=batch_size,
                           batch_normalize=False,
                           mode='classification',
                           device='cpu')

    model.fit(dataset, nb_epoch=20)
    bace_url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv"
    download_url(url=bace_url, name="bace_tmp.csv")
    loader = CSVLoader(tasks=[],
                       smiles_field='mol',
                       featurizer=ConvMolFeaturizer())
    td = loader.featurize(os.path.join(get_data_dir(), "bace_tmp.csv"))
    model.predict(td)


@pytest.mark.torch
def test_graphconvmodel_reload():
    featurizer = dc.feat.ConvMolFeaturizer()
    tasks = ["outcome"]
    mols = ["C", "CO", "CC"]
    X = featurizer(mols)
    y = np.array([0, 1, 0])
    dataset = dc.data.NumpyDataset(X, y)
    batch_size = 10
    model_dir = tempfile.mkdtemp()
    model = GraphConvModel(len(tasks),
                           number_input_features=[75, 64],
                           batch_size=batch_size,
                           batch_normalize=False,
                           mode='classification',
                           model_dir=model_dir,
                           device='cpu')

    model.fit(dataset, nb_epoch=10)

    # Reload trained Model
    reloaded_model = GraphConvModel(len(tasks),
                                    number_input_features=[75, 64],
                                    batch_size=batch_size,
                                    batch_normalize=False,
                                    mode='classification',
                                    model_dir=model_dir,
                                    device='cpu')
    reloaded_model.restore()

    # Check predictions match on random sample
    predmols = ["CCCC", "CCCCCO", "CCCCC"]
    Xpred = featurizer(predmols)
    predset = dc.data.NumpyDataset(Xpred)
    origpred = model.predict(predset)
    reloadpred = reloaded_model.predict(predset)
    assert np.allclose(origpred, reloadpred)

@flaky
@pytest.mark.torch
def test_graph_conv_model_save_and_load():
    np.random.seed(5)
    torch.manual_seed(5)
    tasks, dataset, transformers, metric = get_dataset('classification',
                                                       'GraphConv')

    batch_size = 10
    model = GraphConvModel(len(tasks),
                           number_input_features=[75, 64],
                           batch_size=batch_size,
                           batch_normalize=False,
                           mode='classification',
                           device='cpu')

    model.fit(dataset, nb_epoch=20)
    scores = model.evaluate(dataset, [metric], transformers)
    assert scores['mean-roc_auc_score'] >= 0.9
    model.save_pretrained("save_Graph")
    model_1 = deepchemmap.Map.load_from_pretrained("save_graph")
    scores_1 = model_1.evaluate(dataset, [metric], transformers)

    assert scores_1['mean-roc_auc_score'] >= 0.9


@flaky
@pytest.mark.torch

def test_graph_conv_model_save_and_load_optimizer():
    np.random.seed(5)
    torch.manual_seed(5)
    tasks, dataset, transformers, metric = get_dataset('classification',
                                                       'GraphConv')
    optimizer = AdamW(learning_rate=0.0001)
    batch_size = 10
    model = GraphConvModel(len(tasks),
                           number_input_features=[75, 64],
                           batch_size=batch_size,
                           batch_normalize=False,
                           mode='classification',
                           device='cpu',
                           optimizer = optimizer)
    
    model.fit(dataset, nb_epoch=20)
    scores = model.evaluate(dataset, [metric], transformers)
    model.save_pretrained("save_Graph")
    model_1 = deepchemmap.Map.load_from_pretrained("save_graph")
    scores_1 = model_1.evaluate(dataset, [metric], transformers)

    assert scores_1['mean-roc_auc_score'] >= 0.6
    assert model.optimizer.__class__ == model_1.optimizer.__class__
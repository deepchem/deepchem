import pytest
import tempfile
from os import path, remove

import numpy as np

from deepchem.feat import CGCNNFeaturizer
from deepchem.molnet import load_perovskite, load_mp_metallicity
from deepchem.metrics import Metric, mae_score, roc_auc_score

try:
    import dgl  # noqa
    import torch  # noqa
    from deepchem.models import CGCNNModel
    has_pytorch_and_dgl = True
except:
    has_pytorch_and_dgl = False


@pytest.mark.torch
def test_cgcnn_regression():
    # load datasets
    current_dir = path.dirname(path.abspath(__file__))
    config = {
        "reload": False,
        "featurizer": CGCNNFeaturizer(),
        # disable transformer
        "transformers": [],
        "data_dir": path.join(current_dir, "assets")
    }
    tasks, datasets, transformers = load_perovskite(**config)
    train, valid, test = datasets

    n_tasks = len(tasks)
    model = CGCNNModel(n_tasks=n_tasks,
                       mode='regression',
                       batch_size=4,
                       learning_rate=0.001)

    # check train
    model.fit(train, nb_epoch=20)

    # check predict shape
    valid_preds = model.predict_on_batch(valid.X)
    assert valid_preds.shape == (2, n_tasks)
    test_preds = model.predict(test)
    assert test_preds.shape == (3, n_tasks)

    # check overfit
    regression_metric = Metric(mae_score, n_tasks=n_tasks)
    scores = model.evaluate(train, [regression_metric], transformers)
    assert scores[regression_metric.name] < 0.6

    if path.exists(path.join(current_dir, 'perovskite.json')):
        remove(path.join(current_dir, 'perovskite.json'))


@pytest.mark.torch
def test_cgcnn_classification():
    # load datasets
    current_dir = path.dirname(path.abspath(__file__))
    config = {
        "reload": False,
        "featurizer": CGCNNFeaturizer(),
        # disable transformer
        "transformers": [],
        "data_dir": path.join(current_dir, "assets")
    }
    tasks, datasets, transformers = load_mp_metallicity(**config)
    train, valid, test = datasets

    n_tasks = len(tasks)
    n_classes = 2
    model = CGCNNModel(n_tasks=n_tasks,
                       n_classes=n_classes,
                       mode='classification',
                       batch_size=4,
                       learning_rate=0.001)

    # check train
    model.fit(train, nb_epoch=20)

    # check predict shape
    valid_preds = model.predict_on_batch(valid.X)
    assert valid_preds.shape == (2, n_classes)
    test_preds = model.predict(test)
    assert test_preds.shape == (3, n_classes)

    # check overfit
    classification_metric = Metric(roc_auc_score, n_tasks=n_tasks)
    scores = model.evaluate(train, [classification_metric],
                            transformers,
                            n_classes=n_classes)
    assert scores[classification_metric.name] > 0.8

    if path.exists(path.join(current_dir, 'mp_is_metal.json')):
        remove(path.join(current_dir, 'mp_is_metal.json'))


@pytest.mark.torch
def test_cgcnn_reload():
    # load datasets
    current_dir = path.dirname(path.abspath(__file__))
    config = {
        "reload": False,
        "featurizer": CGCNNFeaturizer(),
        # disable transformer
        "transformers": [],
        "data_dir": path.join(current_dir, "assets")
    }
    tasks, datasets, transformers = load_mp_metallicity(**config)
    train, valid, test = datasets

    n_tasks = len(tasks)
    n_classes = 2
    model_dir = tempfile.mkdtemp()
    model = CGCNNModel(n_tasks=n_tasks,
                       n_classes=n_classes,
                       mode='classification',
                       model_dir=model_dir,
                       batch_size=4,
                       learning_rate=0.001)

    # check train
    model.fit(train, nb_epoch=20)

    # check predict shape
    valid_preds = model.predict_on_batch(valid.X)
    assert valid_preds.shape == (2, n_classes)
    test_preds = model.predict(test)
    assert test_preds.shape == (3, n_classes)

    # check overfit
    classification_metric = Metric(roc_auc_score, n_tasks=n_tasks)
    scores = model.evaluate(train, [classification_metric],
                            transformers,
                            n_classes=n_classes)
    assert scores[classification_metric.name] > 0.8

    # reload
    reloaded_model = CGCNNModel(n_tasks=n_tasks,
                                n_classes=n_classes,
                                mode='classification',
                                model_dir=model_dir,
                                batch_size=4,
                                learning_rate=0.001)
    reloaded_model.restore()

    original_pred = model.predict(test)
    reload_pred = reloaded_model.predict(test)
    assert np.all(original_pred == reload_pred)

    if path.exists(path.join(current_dir, 'mp_is_metal.json')):
        remove(path.join(current_dir, 'mp_is_metal.json'))

import pytest
import tempfile
from os import path
import numpy as np
from deepchem.utils import load_dataset_from_disk, download_url, untargz_file
from deepchem.metrics import Metric, mae_score

try:
    import dgl  # noqa: F401
    import torch  # noqa: F401
    from deepchem.models import LCNNModel
    has_pytorch_and_dgl = True
except:
    has_pytorch_and_dgl = False

URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/featurized_datasets/lcnn_data_feature.tar.gz"


@pytest.mark.torch
def test_lcnn_regression():

    current_dir = tempfile.mkdtemp()
    download_url(url=URL, dest_dir=current_dir)
    untargz_file(path.join(current_dir, 'lcnn_data_feature.tar.gz'),
                 current_dir)
    _, datasets, transformers = load_dataset_from_disk(
        path.join(current_dir, 'lcnn_data'))
    train, valid, test = datasets
    model = LCNNModel(mode='regression', batch_size=8, learning_rate=0.001)
    model.fit(train, nb_epoch=10)

    # check predict shape
    valid_preds = model.predict_on_batch(valid.X)
    assert valid_preds.shape == (65, 1)
    test_preds = model.predict(test)
    assert test_preds.shape == (65, 1)
    # check overfit
    regression_metric = Metric(mae_score)
    scores = model.evaluate(test, [regression_metric], transformers)
    assert scores[regression_metric.name] < 0.6


@pytest.mark.torch
def test_lcnn_reload():

    # needs change
    current_dir = tempfile.mkdtemp()
    download_url(url=URL, dest_dir=current_dir)
    untargz_file(path.join(current_dir, 'lcnn_data_feature.tar.gz'),
                 current_dir)
    tasks, datasets, transformers = load_dataset_from_disk(
        path.join(current_dir, 'lcnn_data'))
    train, valid, test = datasets
    model_dir = tempfile.mkdtemp()
    model = LCNNModel(mode='regression',
                      batch_size=8,
                      learning_rate=0.001,
                      model_dir=model_dir)
    model.fit(train, nb_epoch=10)

    # check predict shape
    valid_preds = model.predict_on_batch(valid.X)
    assert valid_preds.shape == (65, 1)
    test_preds = model.predict(test)
    assert test_preds.shape == (65, 1)
    # check overfit
    regression_metric = Metric(mae_score)
    scores = model.evaluate(test, [regression_metric], transformers)
    assert scores[regression_metric.name] < 0.6

    # reload
    reloaded_model = LCNNModel(mode='regression',
                               batch_size=8,
                               learning_rate=0.001,
                               model_dir=model_dir)
    reloaded_model.restore()

    original_pred = model.predict(test)
    reload_pred = reloaded_model.predict(test)

    assert np.all(np.abs(original_pred - reload_pred) < 0.0000001)

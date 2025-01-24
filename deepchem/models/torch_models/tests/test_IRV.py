import deepchem as dc
import numpy as np
import tempfile
import pickle
import os
import pytest

try:
    import torch
except ModuleNotFoundError:
    pass


@pytest.mark.torch
def test_IRV_multitask_classification_overfit():
    """Test IRV classifier overfits tiny data."""
    n_tasks = 5
    n_samples = 10
    n_features = 128
    K = 5

    # Generate dummy dataset
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.randint(2, size=(n_samples, n_features))
    y = np.ones((n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y, w, ids)
    IRV_transformer = dc.trans.IRVTransformer(K, n_tasks, dataset)
    dataset_trans = IRV_transformer.transform(dataset)
    classification_metric = dc.metrics.Metric(dc.metrics.accuracy_score,
                                              task_averager=np.mean)
    model = dc.models.torch_models.MultitaskIRVClassifier(n_tasks,
                                                          K,
                                                          learning_rate=0.01,
                                                          batch_size=n_samples)
    model.fit(dataset_trans)

    # Eval model on train
    scores = model.evaluate(dataset_trans, [classification_metric])
    assert scores[classification_metric.name] > .9


@pytest.mark.torch
def test_IRV_multitask_classification_reload():
    """Test IRV classifier can be reloaded."""
    n_tasks = 5
    n_samples = 20
    n_features = 128
    K = 5

    # Generate dummy dataset
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.randint(2, size=(n_samples, n_features))
    y = np.ones((n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y, w, ids)
    IRV_transformer = dc.trans.IRVTransformer(K, n_tasks, dataset)
    dataset_trans = IRV_transformer.transform(dataset)

    classification_metric = dc.metrics.Metric(dc.metrics.accuracy_score,
                                              task_averager=np.mean)
    model_dir = tempfile.mkdtemp()
    model = dc.models.torch_models.MultitaskIRVClassifier(n_tasks,
                                                          K=5,
                                                          learning_rate=0.01,
                                                          batch_size=n_samples,
                                                          model_dir=model_dir)

    # Fit trained model
    model.fit(dataset_trans)

    # Eval model on train
    scores = model.evaluate(dataset_trans, [classification_metric])
    assert scores[classification_metric.name] > .9

    # Reload Trained Model
    reloaded_model = dc.models.torch_models.MultitaskIRVClassifier(
        n_tasks,
        K=5,
        learning_rate=0.01,
        batch_size=n_samples,
        model_dir=model_dir)
    reloaded_model.restore()

    # Check predictions match on random sample
    Xpred = np.random.random(dataset_trans.X.shape)
    predset = dc.data.NumpyDataset(Xpred)
    origpred = model.predict(predset)
    reloadpred = reloaded_model.predict(predset)
    assert np.all(origpred == reloadpred)

    # Eval model on train
    scores = reloaded_model.evaluate(dataset_trans, [classification_metric])
    assert scores[classification_metric.name] > .9


@pytest.mark.torch
def test_IRV_multitask_classification_compare_with_tf_impl():
    """Compare the ouputs of tensorflow and pytorch models"""
    n_tasks = 5
    n_samples = 10
    n_features = 128
    K = 5

    # Generate dummy dataset
    np.random.seed(123)
    ids = np.arange(n_samples)
    X = np.random.randint(2, size=(n_samples, n_features))
    y = np.ones((n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y, w, ids)
    IRV_transformer = dc.trans.IRVTransformer(K, n_tasks, dataset)
    dataset_trans = IRV_transformer.transform(dataset)

    # Instantiate pytorch model
    model_torch = dc.models.torch_models.MultitaskIRVClassifier(
        n_tasks, K=5, learning_rate=0.01, batch_size=n_samples)

    tf_weights_dir = os.path.join(os.path.dirname(__file__), "assets",
                                  "IRV_tf_weights.pickle")
    # Load tensorflow weights
    with open(tf_weights_dir, 'rb') as f:
        data = pickle.load(f)

    # Copy tensorflow weights to pytorch model
    for name, param in model_torch.model.named_parameters():
        if "V" == name:
            param[0].data.fill_(torch.tensor(data[0][0], dtype=torch.float64))
            param[1].data.fill_(torch.tensor(data[0][1], dtype=torch.float64))
        elif "W" == name:
            param[0].data.fill_(data[1][0])
            param[1].data.fill_(data[1][1])
        elif "b" == name:
            param.data.fill_(data[2][0])
        elif "b2" == name:
            param.data.fill_(data[3][0])

    tf_output_dir = os.path.join(os.path.dirname(__file__), "assets",
                                 "IRV_tf_output.pickle")
    # Load tensorflow output for comparison
    with open(tf_output_dir, 'rb') as f:
        tf_output = pickle.load(f)

    # Predict pytorch outputs
    torch_output = model_torch.predict(dataset=dataset_trans,
                                       transformers=[IRV_transformer])

    # Compare pytorch and tensorflow outputs
    assert np.allclose(tf_output, torch_output, rtol=1e-5,
                       atol=1e-6), "Outputs do not match!"

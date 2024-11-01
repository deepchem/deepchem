import numpy as np
from deepchem.models.torch_models import RobustMultitask, RobustMultitaskClassifier
import deepchem as dc
import pytest
import tempfile
import os
try:
    import torch
    import torch.nn as nn

    has_torch = True
except ModuleNotFoundError:
    has_torch = False
    pass


@pytest.mark.torch
def test_robustmultitask_construction():
    """Test that RobustMultiTask Model can be constructed without crash.
    """

    model = RobustMultitask(
        n_tasks=1,
        n_features=100,
        mode="regression",
        layer_sizes=[128, 256],
    )

    assert model is not None


@pytest.mark.torch
def test_robustmultitask_forward():
    """Test that the forward pass of RobustMultiTask Model can be executed without crash
    and that the output has the correct value.
    """

    n_tasks = 1
    n_features = 100

    torch_model = RobustMultitask(n_tasks=n_tasks,
                                  n_features=n_features,
                                  layer_sizes=[1024],
                                  mode='classification')

    weights = np.load(
        os.path.join(os.path.dirname(__file__), "assets",
                     "tensorflow_robust_multitask_classifier_weights.npz"))

    move_weights(torch_model, weights)

    input_x = weights["input"]
    output = weights["output"]

    torch_out = torch_model(torch.from_numpy(input_x).float())[0]
    torch_out = torch_out.cpu().detach().numpy()
    assert np.allclose(output, torch_out,
                       atol=1e-4), "Predictions are not close"


@pytest.mark.torch
def test_robustmultitask_classifier_construction():
    """Test that RobustMultiTaskClassifier Model can be constructed without crash.
    """

    model = RobustMultitaskClassifier(
        n_tasks=1,
        n_features=100,
        layer_sizes=[128, 256],
        n_classes=2,
    )

    assert model is not None


@pytest.mark.torch
def test_robust_multitask_classification_forward():
    """Test that the forward pass of RobustMultiTask Model can be executed without crash
    and that the output has the correct value.
    """

    n_tasks = 1
    n_features = 100

    torch_model = RobustMultitaskClassifier(
        n_tasks=n_tasks,
        n_features=n_features,
        layer_sizes=[1024],
    )

    weights = np.load(
        os.path.join(os.path.dirname(__file__), "assets",
                     "tensorflow_robust_multitask_classifier_weights.npz"))

    move_weights(torch_model, weights)

    input_x = weights["input"]
    output = weights["output"]

    # Inference using TorchModel's predict() method works with NumpyDataset only. Hence we need to convert our numpy arrays to NumpyDataset.
    y = np.random.rand(input_x.shape[0], 1)
    w = np.ones((input_x.shape[0], 1))
    ids = np.arange(input_x.shape[0])
    input_x = dc.data.NumpyDataset(input_x, y, w, ids)

    torch_out = torch_model.predict(input_x)[0]  # We need output probabilities

    assert np.allclose(output, torch_out,
                       atol=1e-4), "Predictions are not close"


def move_weights(torch_model, weights):
    """Porting weights from Tensorflow to PyTorch"""

    def to_torch_param(weights):
        """Convert numpy weights to torch parameters to be used as model weights"""
        weights = weights.T
        return nn.Parameter(torch.from_numpy(weights))

    torch_weights = {
        k: to_torch_param(v) for k, v in weights.items() if k != "output"
    }

    # Shared layers
    torch_model.shared_layers[0].weight = torch_weights["shared-layers-dense-w"]
    torch_model.shared_layers[0].bias = torch_weights["shared-layers-dense-b"]

    # Task 0 - We have only one task.
    # Bypass layer
    torch_model.bypass_layers[0][0].weight = torch_weights[
        "bypass-layers-dense_1-w"]
    torch_model.bypass_layers[0][0].bias = torch_weights[
        "bypass-layers-dense_1-b"]
    # Output layer
    torch_model.output_layers[0].weight = torch_weights[
        'bypass-layers-dense_2-w']
    torch_model.output_layers[0].bias = torch_weights['bypass-layers-dense_2-b']


@pytest.mark.torch
def test_robust_multitask_classifier_overfit():
    """Test that the model can overfit simple classification datasets."""
    n_samples = 20
    n_features = 5
    n_tasks = 3

    # Generate dummy dataset
    np.random.seed(123)
    torch.manual_seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(2, size=(n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y, w, ids)

    classification_metric = dc.metrics.Metric(dc.metrics.accuracy_score)

    model = RobustMultitaskClassifier(
        n_tasks,
        n_features,
        layer_sizes=[128, 256],
        dropouts=0.2,
        weight_init_stddevs=0.02,
        bias_init_consts=0.0,
    )

    model.fit(dataset, nb_epoch=300)

    scores = model.evaluate(dataset, [classification_metric])
    assert scores[classification_metric.name] > 0.9, "Failed to overfit"


@pytest.mark.torch
def test_robust_multitask_classifier_reload():
    """Test that the model can be reloaded from disk."""

    n_samples = 20
    n_features = 5
    n_tasks = 3

    # Generate dummy dataset
    np.random.seed(123)
    torch.manual_seed(123)
    ids = np.arange(n_samples)
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(2, size=(n_samples, n_tasks))
    w = np.ones((n_samples, n_tasks))
    dataset = dc.data.NumpyDataset(X, y, w, ids)

    model_dir = tempfile.mkdtemp()

    orig_model = RobustMultitaskClassifier(n_tasks,
                                           n_features,
                                           layer_sizes=[128, 256],
                                           dropouts=0.2,
                                           weight_init_stddevs=0.02,
                                           bias_init_consts=0.0,
                                           model_dir=model_dir)

    orig_model.fit(dataset, nb_epoch=200)

    reloaded_model = RobustMultitaskClassifier(n_tasks,
                                               n_features,
                                               layer_sizes=[128, 256],
                                               dropouts=0.2,
                                               weight_init_stddevs=0.02,
                                               bias_init_consts=0.0,
                                               model_dir=model_dir)

    reloaded_model.restore()

    X_new = np.random.rand(n_samples, n_features)
    orig_preds = orig_model.predict_on_batch(X_new)
    reloaded_preds = reloaded_model.predict_on_batch(X_new)

    assert np.all(orig_preds == reloaded_preds), "Predictions are not the same"

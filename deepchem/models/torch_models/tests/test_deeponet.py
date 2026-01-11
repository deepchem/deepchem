import deepchem as dc
import pytest
import numpy as np
import tempfile

try:
    import torch
    has_torch = True
except ModuleNotFoundError:
    has_torch = False


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch is not installed")
def test_deeponet_construction():
    """Test that DeepONet base model can be constructed with correct attributes."""
    from deepchem.models.torch_models import DeepONet
    model = DeepONet(branch_input_dim=10,
                     trunk_input_dim=3,
                     branch_hidden=(64, 64),
                     trunk_hidden=(64, 64),
                     output_dim=64,
                     activation_fn='tanh')
    assert model is not None
    assert model.branch_net is not None
    assert model.trunk_net is not None
    assert model.bias is not None
    assert model.output_dim == 64


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch is not installed")
def test_deeponet_forward_shape():
    """Test that DeepONet forward pass returns the correct output shape."""
    from deepchem.models.torch_models import DeepONet
    model = DeepONet(branch_input_dim=10,
                     trunk_input_dim=3,
                     branch_hidden=(64, 64),
                     trunk_hidden=(64, 64),
                     output_dim=64)
    batch_size = 5
    branch_input = torch.randn(batch_size, 10)
    trunk_input = torch.randn(batch_size, 3)
    output = model([branch_input, trunk_input])
    assert output.shape == torch.Size([batch_size, 1])


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch is not installed")
def test_deeponetmodel_construction():
    """Test that DeepONetModel can be constructed with correct attributes."""
    from deepchem.models.torch_models import DeepONetModel
    model = DeepONetModel(branch_input_dim=10,
                          trunk_input_dim=3,
                          branch_hidden=(32, 32),
                          trunk_hidden=(32, 32),
                          output_dim=32,
                          batch_size=5)
    assert model is not None
    assert model.branch_input_dim == 10
    assert model.trunk_input_dim == 3


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch is not installed")
def test_deeponetmodel_fit_and_predict():
    """Test that DeepONetModel can fit to data and predict correct shape."""
    from deepchem.models.torch_models import DeepONetModel

    branch_input_dim = 10
    trunk_input_dim = 3
    n_samples = 20

    # Create concatenated input: X = [branch_data | trunk_data]
    branch_data = np.random.randn(n_samples,
                                  branch_input_dim).astype(np.float32)
    trunk_data = np.random.randn(n_samples, trunk_input_dim).astype(np.float32)
    X = np.concatenate([branch_data, trunk_data], axis=1)
    y = np.random.randn(n_samples, 1).astype(np.float32)

    dataset = dc.data.NumpyDataset(X, y)

    model = DeepONetModel(branch_input_dim=branch_input_dim,
                          trunk_input_dim=trunk_input_dim,
                          batch_size=5)
    model.fit(dataset, nb_epoch=1)

    pred = model.predict_on_batch(X)
    assert pred.shape == y.shape


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch is not installed")
def test_deeponetmodel_restore():
    """Test that DeepONetModel can be saved and restored with consistent predictions."""
    from deepchem.models.torch_models import DeepONetModel

    branch_input_dim = 10
    trunk_input_dim = 3
    n_samples = 20

    branch_data = np.random.randn(n_samples,
                                  branch_input_dim).astype(np.float32)
    trunk_data = np.random.randn(n_samples, trunk_input_dim).astype(np.float32)
    X = np.concatenate([branch_data, trunk_data], axis=1)
    y = np.random.randn(n_samples, 1).astype(np.float32)

    dataset = dc.data.NumpyDataset(X, y)

    model_dir = tempfile.mkdtemp()
    model = DeepONetModel(branch_input_dim=branch_input_dim,
                          trunk_input_dim=trunk_input_dim,
                          batch_size=5,
                          model_dir=model_dir)
    model.fit(dataset, nb_epoch=1)
    pred = model.predict_on_batch(X)

    # Create new instance and restore from checkpoint
    restored_model = DeepONetModel(branch_input_dim=branch_input_dim,
                                   trunk_input_dim=trunk_input_dim,
                                   batch_size=5,
                                   model_dir=model_dir)
    restored_model.restore()
    restored_pred = restored_model.predict_on_batch(X)

    assert pred.shape == restored_pred.shape
    assert np.allclose(pred, restored_pred, atol=1e-4)


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch is not installed")
def test_deeponetmodel_overfit():
    """Test that DeepONetModel can overfit on a small dataset."""
    from deepchem.models.torch_models import DeepONetModel

    branch_input_dim = 5
    trunk_input_dim = 2
    n_samples = 10

    branch_data = np.random.randn(n_samples,
                                  branch_input_dim).astype(np.float32)
    trunk_data = np.random.randn(n_samples, trunk_input_dim).astype(np.float32)
    X = np.concatenate([branch_data, trunk_data], axis=1)
    y = np.random.randn(n_samples, 1).astype(np.float32)

    dataset = dc.data.NumpyDataset(X, y)

    model = DeepONetModel(branch_input_dim=branch_input_dim,
                          trunk_input_dim=trunk_input_dim,
                          branch_hidden=(64, 64),
                          trunk_hidden=(64, 64),
                          output_dim=64,
                          batch_size=10,
                          learning_rate=1e-3)
    model.fit(dataset, nb_epoch=500)

    pred = model.predict_on_batch(X)
    regression_metric = dc.metrics.Metric(dc.metrics.mean_squared_error,
                                          mode='regression')
    mse = regression_metric.compute_metric(y, pred)

    assert mse < 0.1, f"DeepONetModel failed to overfit: MSE = {mse}"

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
def test_forward_eval():
    """Test that the HNN model returns the correct output shape in evaluation mode"""
    from deepchem.models.torch_models import HNN
    model = HNN()
    input_tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    model.eval()
    output = model(input_tensor)
    assert output.shape == torch.Size([1])


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch is not installed")
def test_symplectic_gradient_shape():
    """Test that the symplectic gradient output matches the input tensor shape"""
    from deepchem.models.torch_models import HNN
    model = HNN()
    input_tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    output = model.symplectic_gradient(input_tensor)
    assert output.shape == input_tensor.shape


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch is not installed")
def test_hnnmodel_train_and_predict():
    """Test HNNModel can fit to simple synthetic data and predict correct shape."""
    from deepchem.models.torch_models import HNNModel

    # small synthetic data (q, p) and their derivatives
    x = np.random.randn(10, 2).astype(np.float32)
    dx = np.random.randn(10, 2).astype(np.float32)

    dataset = dc.data.NumpyDataset(x, dx)

    model = HNNModel(batch_size=5)
    model.fit(dataset, nb_epoch=1)

    pred = model.predict_on_batch(x)

    assert pred.shape == dx.shape


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch is not installed")
def test_hnnmodel_restore():
    """Test HNNModel save and restore functionality produces consistent predictions."""
    from deepchem.models.torch_models import HNNModel

    # small synthetic data (q, p) and their derivatives
    x = np.random.randn(10, 2).astype(np.float32)
    dx = np.random.randn(10, 2).astype(np.float32)

    dataset = dc.data.NumpyDataset(x, dx)

    model_dir = tempfile.mkdtemp()
    model = HNNModel(batch_size=5, model_dir=model_dir)

    model.fit(dataset, nb_epoch=1)
    pred = model.predict_on_batch(x)

    # Create new instance of model and restore from checkpoint
    restored_model = HNNModel(batch_size=5, model_dir=model_dir)
    restored_model.restore()

    restored_pred = restored_model.predict_on_batch(x)

    assert pred.shape == restored_pred.shape
    assert np.allclose(pred, restored_pred, atol=1e-4)


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch is not installed")
def test_hnnmodel_overfit():
    """Test that HNNModel can overfit on a very small dataset."""
    from deepchem.models.torch_models import HNNModel

    np.random.seed(12)
    # small synthetic data (q, p) and their derivatives
    x = np.random.randn(20, 2).astype(np.float32)
    dx = np.random.randn(20, 2).astype(np.float32)

    dataset = dc.data.NumpyDataset(x, dx)

    regression_metric = dc.metrics.Metric(dc.metrics.mean_squared_error,
                                          mode='regression')

    model_dir = tempfile.mkdtemp()
    model = HNNModel(batch_size=5, model_dir=model_dir, learning_rate=1e-2)

    model.fit(dataset, nb_epoch=300)
    pred = model.predict_on_batch(x)

    score = regression_metric.compute_metric(dx, pred)

    assert score < 0.05, "HNNModel failed to overfit small dataset"

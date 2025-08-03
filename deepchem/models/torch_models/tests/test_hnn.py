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
    assert output.shape == torch.Size([1, 2])


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch is not installed")
def test_symplectic_gradient_shape():
    """Test that the symplectic gradient output matches the input tensor shape"""
    from deepchem.models.torch_models import HNN
    model = HNN()
    input_tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    output = model.symplectic_gradient(input_tensor)
    assert output.shape == torch.Size([1, 2])


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

    # small synthetic data (q, p) and their derivatives
    x = np.array([
        [-0.0847, 0.8053],
        [0.4012, 0.5271],
        [0.2978, 0.5341],
        [-0.2054, -0.3220],
        [-0.0823, -0.2020],
    ],
                 dtype=np.float32)

    dx = np.array([
        [1.2531, -0.2843],
        [1.1680, -0.5356],
        [1.0331, -0.7641],
        [-0.4815, 0.1087],
        [-0.4489, 0.2053],
    ],
                  dtype=np.float32)

    dataset = dc.data.NumpyDataset(x, dx)

    regression_metric = dc.metrics.Metric(dc.metrics.mean_squared_error,
                                          mode='regression')

    model = HNNModel(batch_size=5, learning_rate=1e-3)

    model.fit(dataset, nb_epoch=1000)
    pred = model.predict_on_batch(x)

    score = regression_metric.compute_metric(dx, pred)

    assert score < 0.06, "HNNModel failed to overfit small dataset"


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch is not installed")
def test_mass_spring_energy_conservation():
    """
        -> mass-spring equation
        H(q, p) = (p**2 / 2 * m) + (0.5 k * q**2)

        -> the partial derivatives are
        dq/dt -> ∂H/∂p = p/m
        dp/dt -> ∂H/∂q = -kq

        -> assuming m and k values as 1

        -> final values
        dq/dt = p
        dp/dt = -q

        """

    # (q, p) value pairs
    x_train = np.array(
        [[1.0, 0.0], [0.0, -1.0], [-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        dtype=np.float32)

    # calculated dq/dt and dp/dt
    dx_train = np.array(
        [[0.0, -1.0], [-1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, -1.0]],
        dtype=np.float32)

    from deepchem.models.torch_models import HNNModel

    dataset = dc.data.NumpyDataset(x_train, dx_train)

    model = HNNModel(d_hidden=(32, 32))
    model.fit(dataset, nb_epoch=2000)

    test_points = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
                           dtype=np.float32)

    energies = model.predict_hamiltonian(test_points)
    energy_std = np.std(energies)

    assert energy_std < 0.1, f"Energy not conserved: std = {energy_std}"

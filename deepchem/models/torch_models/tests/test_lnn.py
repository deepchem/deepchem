import deepchem as dc
import numpy as np
import pytest
import tempfile

try:
    import torch
    has_torch = True
except ModuleNotFoundError:
    has_torch = False


# @pytest.mark.torch
# @pytest.mark.skipif(not has_torch, reason="PyTorch is not installed")
# def test_forward_eval():
#     """Test that the LNN model returns the correct output shape in evaluation mode"""
#     from deepchem.models.torch_models import LNN
#     model = LNN(n_dof=2)
#     input_tensor = torch.tensor([[1.0, 5.0, 2.0, 3.0]], dtype=torch.float32)
#     model.eval()
#     output = model(input_tensor)
#     assert output.shape == torch.Size([1, 2])


# @pytest.mark.torch
# @pytest.mark.skipif(not has_torch, reason="PyTorch is not installed")
# def test_calculate_dynamics_shape():
#     """Test that the calculate dynamics returning correct output shape"""
#     from deepchem.models.torch_models import LNN
#     model = LNN(n_dof=2)
#     input_tensor = torch.tensor([[1.0, 5.0, 2.0, 3.0]], dtype=torch.float32)
#     output = model.calculate_dynamics(input_tensor)
#     assert output.shape == torch.Size([1, 2])


# @pytest.mark.torch
# @pytest.mark.skipif(not has_torch, reason="PyTorch is not installed")
# def test_lagrangian():
#     """Test that the lagrangian method is returning correct output shape (scalar value)"""
#     from deepchem.models.torch_models import LNN
#     model = LNN(n_dof=2)
#     input_tensor = torch.tensor([[1.0, 5.0, 2.0, 3.0]], dtype=torch.float32)
#     output = model.lagrangian(input_tensor)
#     assert output.shape == torch.Size([1])

# @pytest.mark.torch
# @pytest.mark.skipif(not has_torch, reason="PyTorch is not installed")
# def test_lnnmodel_train_and_predict():
#     """Test LNNModel can fit to simple synthetic data and predict correct shape."""
#     from deepchem.models.torch_models import LNNModel

#     # considering spring pendulum experiment example
#     # inputs -> [x, theta, x_dot, theta_dot]
#     inputs = np.random.randn(10, 4).astype(np.float32)
#     # labels -> [x_ddot, theta_ddot]
#     labels = np.random.randn(10, 2).astype(np.float32)

#     dataset = dc.data.NumpyDataset(inputs, labels)

#     model = LNNModel(n_dof=2, batch_size=5)
#     model.fit(dataset, nb_epoch=1)

#     pred = model.predict_on_batch(inputs)

#     assert pred.shape == labels.shape


# @pytest.mark.torch
# @pytest.mark.skipif(not has_torch, reason="PyTorch is not installed")
# def test_lnnmodel_restore():
#     """Test LNNModel save and restore functionality produces consistent predictions."""
#     from deepchem.models.torch_models import LNNModel

#     # inputs -> [x, theta, x_dot, theta_dot]
#     inputs = np.random.randn(10, 4).astype(np.float32)
#     # labels -> [x_ddot, theta_ddot]
#     labels = np.random.randn(10, 2).astype(np.float32)

#     dataset = dc.data.NumpyDataset(inputs, labels)

#     model_dir = tempfile.mkdtemp()
#     model = LNNModel(n_dof=2, batch_size=5, model_dir=model_dir)

#     model.fit(dataset, nb_epoch=1)
#     pred = model.predict_on_batch(inputs)

#     # Create new instance of model and restore from checkpoint
#     restored_model = LNNModel(n_dof=2, batch_size=5, model_dir=model_dir)
#     restored_model.restore()

#     restored_pred = restored_model.predict_on_batch(inputs)

#     assert pred.shape == restored_pred.shape
#     assert np.allclose(pred, restored_pred, atol=1e-4)


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch is not installed")
def test_lnnmodel_overfit():
    """Test that LNNModel can overfit on a very small dataset."""
    from deepchem.models.torch_models import LNNModel

    # inputs -> [x, theta, x_dot, theta_dot]
    inputs = np.array([
        [0.1681, -0.1578, 0.0100, 0.1712],
        [0.2179, -0.1347, 0.9779, 0.2733],
        [0.3656, -0.1052, 1.9439, 0.2995],
        [0.6101, -0.0764, 2.8944, 0.2663],
        [0.9493, -0.0522, 3.8154, 0.2109],
        [1.3796, -0.0336, 4.6957, 0.1574]
    ],
                 dtype=np.float32)

    # labels -> [x_ddot, theta_ddot]
    labels = np.array([
        [9.5541, 1.3175],
        [9.5940, 0.6434],
        [9.5125, -0.0977],
        [9.2853, -0.4924],
        [8.9339, -0.5630],
        [8.4838, -0.4825]
    ],
                  dtype=np.float32)


    dataset = dc.data.NumpyDataset(inputs, labels)

    regression_metric = dc.metrics.Metric(dc.metrics.mean_squared_error,
                                          mode='regression')

    model = LNNModel(n_dof=2, learning_rate=1e-2, d_hidden=(16, 16))

    model.fit(dataset, nb_epoch=1000)
    pred = model.predict_on_batch(inputs)

    score = regression_metric.compute_metric(labels, pred)

    assert score < 0.1, "LNNModel failed to overfit small dataset"
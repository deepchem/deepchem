import torch
import pytest


@pytest.fixture
def hnn_model():
    """Initialize an instance of the HNN model"""
    from deepchem.models.torch_models import HNN
    return HNN()


@pytest.fixture
def input_tensor():
    """Provide a sample input tensor for testing"""
    return torch.tensor([[1.0, 2.0]], dtype=torch.float32)


def test_forward_eval(hnn_model, input_tensor):
    """Test that the HNN model returns the correct output shape in evaluation mode"""
    hnn_model.eval()
    output = hnn_model(input_tensor)
    assert output.shape == torch.Size([1])


def test_symplectic_gradient_shape(hnn_model, input_tensor):
    """Test that the symplectic gradient output matches the input tensor shape"""
    output = hnn_model.symplectic_gradient(input_tensor)
    assert output.shape == input_tensor.shape

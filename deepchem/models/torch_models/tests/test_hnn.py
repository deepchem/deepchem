import torch
import pytest
from deepchem.models.torch_models import HNN



@pytest.fixture
def hnn_model():
    return HNN()


@pytest.fixture
def input_tensor():
    return torch.tensor([[1.0, 2.0]], dtype=torch.float32)


def test_forward_eval(hnn_model, input_tensor):
    hnn_model.eval()
    output = hnn_model(input_tensor)
    assert output.shape == torch.Size([1])


def test_symplectic_gradient_shape(hnn_model, input_tensor):
    output = hnn_model.symplectic_gradient(input_tensor)
    assert output.shape == input_tensor.shape

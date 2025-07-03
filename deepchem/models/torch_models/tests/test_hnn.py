import pytest

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

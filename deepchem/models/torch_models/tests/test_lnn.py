import pytest

try:
    import torch
    has_torch = True
except ModuleNotFoundError:
    has_torch = False


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch is not installed")
def test_forward_eval():
    """Test that the LNN model returns the correct output shape in evaluation mode"""
    from deepchem.models.torch_models import LNN
    model = LNN(n_dof=2)
    input_tensor = torch.tensor([[1.0, 5.0, 2.0, 3.0]], dtype=torch.float32)
    model.eval()
    output = model(input_tensor)
    assert output.shape == torch.Size([1, 2])


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch is not installed")
def test_calculate_dynamics_shape():
    """Test that the calculate dynamics returning correct output shape"""
    from deepchem.models.torch_models import LNN
    model = LNN(n_dof=2)
    input_tensor = torch.tensor([[1.0, 5.0, 2.0, 3.0]], dtype=torch.float32)
    output = model.calculate_dynamics(input_tensor)
    assert output.shape == torch.Size([1, 2])


@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch is not installed")
def test_lagrangian():
    """Test that the lagrangian method is returning correct output shape (scalar value)"""
    from deepchem.models.torch_models import LNN
    model = LNN(n_dof=2)
    input_tensor = torch.tensor([[1.0, 5.0, 2.0, 3.0]], dtype=torch.float32)
    output = model.lagrangian(input_tensor)
    assert output.shape == torch.Size([1])

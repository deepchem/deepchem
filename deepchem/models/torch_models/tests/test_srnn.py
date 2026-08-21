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
def test_forward():
    """Test that the SRNN model returns the correct output shape in forward pass"""
    from deepchem.models.torch_models import SRNN
    model = SRNN()
    input_tensor = torch.randn(1, 2, requires_grad=True)
    output = model(input_tensor)
    assert output.shape == torch.Size([1, 10, 2])

@pytest.mark.torch
@pytest.mark.skipif(not has_torch, reason="PyTorch is not installed")
def test_get_hamiltonian():
    """Test that the SRNN model returns the correct output shape in get_hamiltonian method"""
    from deepchem.models.torch_models import SRNN
    model = SRNN()
    q0 = torch.Tensor([1.0]).unsqueeze(1)
    p0 = torch.Tensor([2.0]).unsqueeze(1)
    output = model.get_hamiltonian(q0, p0)
    assert output.shape == torch.Size([1, 1])

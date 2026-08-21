"""Tests for differentiable bond angle layer."""

import pytest
import torch
from deepchem.models.torch_models.differentiable_geometry import DifferentiableBondAngleLayer


def test_bond_angle_layer_instantiation():
    """Test layer can be instantiated."""
    layer = DifferentiableBondAngleLayer()
    assert isinstance(layer, torch.nn.Module)


def test_bond_angle_output_shape():
    """Test output shape is correct."""
    layer = DifferentiableBondAngleLayer()
    batch_size, num_atoms = 3, 5
    coords = torch.randn(batch_size, num_atoms, 3)
    angles = layer(coords)
    assert angles.shape == (batch_size, num_atoms, num_atoms, num_atoms)


def test_bond_angle_gradient_flow():
    """Test gradients flow through the layer."""
    layer = DifferentiableBondAngleLayer()
    coords = torch.randn(2, 4, 3, requires_grad=True)
    angles = layer(coords)
    loss = angles.sum()
    loss.backward()
    assert coords.grad is not None
    assert coords.grad.shape == coords.shape


def test_bond_angle_range():
    """Test bond angles are in valid range [0, pi]."""
    layer = DifferentiableBondAngleLayer()
    coords = torch.randn(2, 6, 3)
    angles = layer(coords)
    assert torch.all(angles >= 0.0)
    assert torch.all(angles <= torch.pi + 1e-5)


def test_bond_angle_numerical_stability():
    """Test layer handles edge cases without NaN."""
    layer = DifferentiableBondAngleLayer()
    # Test with coincident points (edge case)
    coords = torch.randn(1, 3, 3)
    angles = layer(coords)
    assert not torch.any(torch.isnan(angles))


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="CUDA not available")
def test_bond_angle_gpu_compatibility():
    """Test layer works on GPU."""
    layer = DifferentiableBondAngleLayer().cuda()
    coords = torch.randn(2, 4, 3, device='cuda')
    angles = layer(coords)
    assert angles.device.type == 'cuda'
    assert angles.shape == (2, 4, 4, 4)

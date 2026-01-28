"""Tests for differentiable geometry operations."""

import pytest
import torch
from deepchem.models.torch_models.differentiable_geometry import DifferentiablePairwiseDistanceLayer


def test_pairwise_distance_layer_instantiation():
    """Test layer can be instantiated."""
    layer = DifferentiablePairwiseDistanceLayer()
    assert isinstance(layer, torch.nn.Module)


def test_pairwise_distance_output_shape():
    """Test output shape is correct."""
    layer = DifferentiablePairwiseDistanceLayer()
    batch_size, num_atoms = 4, 10
    coords = torch.randn(batch_size, num_atoms, 3)
    distances = layer(coords)
    assert distances.shape == (batch_size, num_atoms, num_atoms)


def test_pairwise_distance_gradient_flow():
    """Test gradients flow through the layer."""
    layer = DifferentiablePairwiseDistanceLayer()
    coords = torch.randn(2, 5, 3, requires_grad=True)
    distances = layer(coords)
    loss = distances.sum()
    loss.backward()
    assert coords.grad is not None
    assert coords.grad.shape == coords.shape


def test_pairwise_distance_symmetry():
    """Test distance matrix is symmetric."""
    layer = DifferentiablePairwiseDistanceLayer()
    coords = torch.randn(1, 8, 3)
    distances = layer(coords)
    assert torch.allclose(distances, distances.transpose(-1, -2))


def test_pairwise_distance_diagonal_zeros():
    """Test diagonal elements are zero (self-distance)."""
    layer = DifferentiablePairwiseDistanceLayer()
    coords = torch.randn(1, 6, 3)
    distances = layer(coords)
    diagonal = torch.diagonal(distances[0])
    assert torch.allclose(diagonal, torch.zeros_like(diagonal), atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="CUDA not available")
def test_pairwise_distance_gpu_compatibility():
    """Test layer works on GPU."""
    layer = DifferentiablePairwiseDistanceLayer().cuda()
    coords = torch.randn(2, 5, 3, device='cuda')
    distances = layer(coords)
    assert distances.device.type == 'cuda'

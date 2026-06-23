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


def test_pairwise_distance_custom_norm():
    """Test layer works with different p-norms."""
    coords = torch.randn(2, 5, 3)
    
    # L1 norm (Manhattan distance)
    layer_l1 = DifferentiablePairwiseDistanceLayer(p=1.0)
    distances_l1 = layer_l1(coords)
    expected_l1 = torch.cdist(coords, coords, p=1.0)
    assert torch.allclose(distances_l1, expected_l1)
    
    # L2 norm (Euclidean distance)
    layer_l2 = DifferentiablePairwiseDistanceLayer(p=2.0)
    distances_l2 = layer_l2(coords)
    expected_l2 = torch.cdist(coords, coords, p=2.0)
    assert torch.allclose(distances_l2, expected_l2)
    
    # Different norms produce different results
    assert not torch.allclose(distances_l1, distances_l2)


def test_pairwise_distance_invalid_shape_2d():
    """Test that 2D input raises ValueError."""
    layer = DifferentiablePairwiseDistanceLayer()
    coords_2d = torch.randn(10, 3)
    
    with pytest.raises(ValueError, match="Expected 3D input tensor"):
        layer(coords_2d)


def test_pairwise_distance_invalid_shape_4d():
    """Test that 4D input raises ValueError."""
    layer = DifferentiablePairwiseDistanceLayer()
    coords_4d = torch.randn(2, 5, 3, 1)
    
    with pytest.raises(ValueError, match="Expected 3D input tensor"):
        layer(coords_4d)


def test_pairwise_distance_batch_consistency():
    """Test that batch processing is consistent with single examples."""
    layer = DifferentiablePairwiseDistanceLayer()
    
    # Create batched input
    coords_batched = torch.randn(3, 7, 3)
    distances_batched = layer(coords_batched)
    
    # Process individually
    for i in range(3):
        coords_single = coords_batched[i:i+1]
        distances_single = layer(coords_single)
        assert torch.allclose(distances_batched[i], distances_single[0])


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="CUDA not available")
def test_pairwise_distance_gpu_compatibility():
    """Test layer works on GPU."""
    layer = DifferentiablePairwiseDistanceLayer().cuda()
    coords = torch.randn(2, 5, 3, device='cuda')
    distances = layer(coords)
    assert distances.device.type == 'cuda'


"""
Tests for the DifferentiableDocking module.

These tests verify the correctness and differentiability of the
Lennard-Jones based docking scoring function.
"""
import pytest
import numpy as np

try:
    import torch
    has_torch = True
except ModuleNotFoundError:
    has_torch = False


@pytest.mark.torch
def test_differentiable_docking_energy_computation():
    """Test that energy computation produces a scalar output."""
    from deepchem.models.torch_models.differentiable_docking import (
        DifferentiableDocking)

    docking = DifferentiableDocking(epsilon=1.0, sigma=2.0, cutoff=8.0)

    # Create random coordinates
    ligand_coords = torch.randn(10, 3)
    protein_coords = torch.randn(50, 3)

    energy = docking(ligand_coords, protein_coords)

    # Energy should be a scalar
    assert energy.dim() == 0
    assert energy.numel() == 1


@pytest.mark.torch
def test_differentiable_docking_gradient_flow():
    """Test that gradients flow through the energy computation.

    This is the key test proving differentiability.
    """
    from deepchem.models.torch_models.differentiable_docking import (
        DifferentiableDocking)

    docking = DifferentiableDocking(epsilon=1.0, sigma=2.0, cutoff=8.0)

    # Create coordinates with gradients enabled
    ligand_coords = torch.randn(10, 3, requires_grad=True)
    protein_coords = torch.randn(50, 3)

    # Compute energy
    energy = docking(ligand_coords, protein_coords)

    # Backward pass
    energy.backward()

    # Verify gradients exist
    assert ligand_coords.grad is not None
    assert ligand_coords.grad.shape == ligand_coords.shape

    # Gradients should not be all zeros (energy depends on positions)
    assert not torch.allclose(ligand_coords.grad,
                              torch.zeros_like(ligand_coords.grad))


@pytest.mark.torch
def test_differentiable_docking_optimization():
    """Test that pose optimization reduces energy."""
    from deepchem.models.torch_models.differentiable_docking import (
        DifferentiableDocking)

    torch.manual_seed(42)

    docking = DifferentiableDocking(epsilon=1.0, sigma=2.0, cutoff=8.0)

    # Create coordinates
    ligand_coords = torch.randn(5, 3)
    protein_coords = torch.randn(20, 3)

    # Initial energy
    initial_energy = docking(ligand_coords, protein_coords).item()

    # Optimize
    optimized_coords = docking.optimize_pose(ligand_coords,
                                             protein_coords,
                                             n_steps=50,
                                             learning_rate=0.01)

    # Final energy
    final_energy = docking(optimized_coords, protein_coords).item()

    # Energy should decrease or stay same (optimization minimizes)
    # Allow for numerical tolerance
    assert final_energy <= initial_energy + 1e-3


@pytest.mark.torch
def test_differentiable_docking_different_sizes():
    """Test handling of different input sizes."""
    from deepchem.models.torch_models.differentiable_docking import (
        DifferentiableDocking)

    docking = DifferentiableDocking()

    # Test various sizes
    test_cases = [(5, 10), (1, 100), (50, 5), (20, 20)]

    for n_ligand, n_protein in test_cases:
        ligand_coords = torch.randn(n_ligand, 3)
        protein_coords = torch.randn(n_protein, 3)

        energy = docking(ligand_coords, protein_coords)

        assert energy.dim() == 0, f"Failed for sizes ({n_ligand}, {n_protein})"


@pytest.mark.torch
def test_differentiable_docking_cutoff():
    """Test that cutoff properly masks distant interactions."""
    from deepchem.models.torch_models.differentiable_docking import (
        DifferentiableDocking)

    # Use a small cutoff
    docking = DifferentiableDocking(cutoff=5.0)

    # Place ligand and protein far apart (beyond cutoff)
    ligand_coords = torch.zeros(1, 3)
    protein_coords = torch.tensor([[100.0, 0.0, 0.0]])

    energy = docking(ligand_coords, protein_coords)

    # Energy should be zero when atoms are beyond cutoff
    assert torch.abs(energy) < 1e-6


@pytest.mark.torch
def test_differentiable_docking_learnable_params():
    """Test that learnable parameters work correctly."""
    from deepchem.models.torch_models.differentiable_docking import (
        DifferentiableDocking)

    docking = DifferentiableDocking(epsilon=1.0, sigma=2.0, learnable=True)

    # Check that epsilon and sigma are parameters
    param_names = [name for name, _ in docking.named_parameters()]
    assert 'epsilon' in param_names
    assert 'sigma' in param_names

    # Verify gradients flow to parameters
    ligand_coords = torch.randn(5, 3)
    protein_coords = torch.randn(10, 3)

    energy = docking(ligand_coords, protein_coords)
    energy.backward()

    assert docking.epsilon.grad is not None
    assert docking.sigma.grad is not None


@pytest.mark.torch
def test_pairwise_distances():
    """Test pairwise distance computation."""
    from deepchem.models.torch_models.differentiable_docking import (
        pairwise_distances)

    # Simple test case
    coords1 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    coords2 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    dists = pairwise_distances(coords1, coords2)

    # Expected distances:
    # (0,0,0) to (1,0,0) = 1.0
    # (0,0,0) to (0,1,0) = 1.0
    # (1,0,0) to (1,0,0) = 0.0 (with epsilon for stability)
    # (1,0,0) to (0,1,0) = sqrt(2)

    assert dists.shape == (2, 2)
    assert torch.isclose(dists[0, 0], torch.tensor(1.0), atol=1e-3)
    assert torch.isclose(dists[0, 1], torch.tensor(1.0), atol=1e-3)
    assert dists[1, 0] < 0.01  # Small due to epsilon
    assert torch.isclose(dists[1, 1],
                         torch.tensor(np.sqrt(2), dtype=torch.float32),
                         atol=1e-3)

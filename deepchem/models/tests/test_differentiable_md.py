import pytest
import torch
from deepchem.models.torch_models.differentiable_md import DifferentiableMD


class TestDifferentiableMD:
    """Test the DifferentiableMD class."""

    def test_step_shapes(self):
        """Test that step returns correct shapes."""
        md = DifferentiableMD(dt=0.01)

        batch_size = 2
        n_atoms = 3
        positions = torch.randn(batch_size, n_atoms, 3)
        velocities = torch.randn(batch_size, n_atoms, 3)
        forces = torch.randn(batch_size, n_atoms, 3)
        mass = torch.ones(n_atoms)

        new_pos, new_vel = md.step(positions, velocities, forces, mass)

        assert new_pos.shape == positions.shape
        assert new_vel.shape == velocities.shape

    def test_autograd(self):
        """Test that gradients flow through the step."""
        md = DifferentiableMD(dt=0.01)

        positions = torch.randn(2, 3, 3, requires_grad=True)
        velocities = torch.randn(2, 3, 3, requires_grad=True)
        forces = torch.randn(2, 3, 3, requires_grad=True)
        mass = torch.ones(3)

        new_pos, new_vel = md.step(positions, velocities, forces, mass)

        # Compute some loss
        loss = (new_pos ** 2).sum() + (new_vel ** 2).sum()
        loss.backward()

        assert positions.grad is not None
        assert velocities.grad is not None
        assert forces.grad is not None

    def test_scalar_mass(self):
        """Test with scalar mass."""
        md = DifferentiableMD(dt=0.01)

        positions = torch.randn(2, 3, 3)
        velocities = torch.randn(2, 3, 3)
        forces = torch.randn(2, 3, 3)
        mass = torch.tensor(1.0)

        new_pos, new_vel = md.step(positions, velocities, forces, mass)

        assert new_pos.shape == positions.shape
        assert new_vel.shape == velocities.shape

    def test_no_batch(self):
        """Test with single molecule (no batch dimension)."""
        md = DifferentiableMD(dt=0.01)

        positions = torch.randn(3, 3)
        velocities = torch.randn(3, 3)
        forces = torch.randn(3, 3)
        mass = torch.ones(3)

        new_pos, new_vel = md.step(positions, velocities, forces, mass)

        assert new_pos.shape == positions.shape
        assert new_vel.shape == velocities.shape
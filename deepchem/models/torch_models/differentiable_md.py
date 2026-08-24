import torch
from typing import Optional


class DifferentiableMD:
    """A differentiable molecular dynamics simulator using Velocity Verlet integration.

    This class implements a single time step of molecular dynamics simulation
    that is compatible with PyTorch's autograd for gradient computation.
    """

    def __init__(self, dt: float = 0.01):
        """Initialize the MD simulator.

        Parameters
        ----------
        dt : float
            Time step for integration.
        """
        self.dt = dt

    def step(self, positions: torch.Tensor, velocities: torch.Tensor,
             forces: torch.Tensor, mass: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform one step of Velocity Verlet integration.

        Parameters
        ----------
        positions : torch.Tensor
            Current positions, shape (batch_size, n_atoms, 3) or (n_atoms, 3)
        velocities : torch.Tensor
            Current velocities, same shape as positions
        forces : torch.Tensor
            Current forces, same shape as positions
        mass : torch.Tensor
            Masses, shape (n_atoms,) or scalar

        Returns
        -------
        new_positions : torch.Tensor
            Updated positions
        new_velocities : torch.Tensor
            Updated velocities
        """
        dt = self.dt

        # Ensure mass has correct shape for broadcasting
        if mass.dim() == 0:  # scalar
            mass = mass.expand_as(positions)
        elif mass.shape[-1] == positions.shape[-2]:  # (n_atoms,)
            mass = mass.unsqueeze(-1).expand_as(positions)
        else:
            raise ValueError("Mass shape incompatible with positions")

        # Half step velocity
        v_half = velocities + 0.5 * dt * forces / mass

        # Update positions
        new_positions = positions + dt * v_half

        # Note: In a full MD simulation, forces would be recomputed here
        # based on the new positions. For this prototype, we assume
        # forces are provided and constant for the step.
        new_forces = forces

        # Full step velocity
        new_velocities = v_half + 0.5 * dt * new_forces / mass

        return new_positions, new_velocities
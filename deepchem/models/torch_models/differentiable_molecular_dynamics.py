import torch
import torch.nn as nn
from deepchem.models.torch_models.torch_model import TorchModel


class VelocityVerlet(nn.Module):
  """
    Implements Velocity Verlet integration for molecular dynamics.

    This module performs a single step or multiple steps of Velocity Verlet integration.

    Parameters
    ----------
    time_step: float
        The time step for the integration.
    mass: float
        The mass of the particles.
    force_fn: callable
        A function that takes positions `r` and returns forces `F`.
        The function should take a tensor of shape (batch_size, n_particles, n_dimensions)
        and return a tensor of the same shape.
    """

  def __init__(self, time_step=0.001, mass=1.0, force_fn=None):
    super(VelocityVerlet, self).__init__()
    self.time_step = time_step
    self.mass = mass
    self.force_fn = force_fn

  def forward(self, r, v, steps=1, force_fn=None):
    """
        Performs MD simulation.

        Parameters
        ----------
        r: torch.Tensor
            Initial positions. Shape (Batch, N_atoms, 3)
        v: torch.Tensor
            Initial velocities. Shape (Batch, N_atoms, 3)
        steps: int
            Number of integration steps.
        force_fn: Callable
            Function that takes (r) and returns forces (F).
            If not provided, uses the one from __init__.

        Returns
        -------
        r_final: torch.Tensor
        v_final: torch.Tensor
        """
    if force_fn is None:
      if self.force_fn is None:
        raise ValueError("force_fn must be provided.")
      force_fn = self.force_fn

    dt = self.time_step
    m = self.mass

    # If inputs are not tensors, convert them
    if not isinstance(r, torch.Tensor):
      r = torch.tensor(r)
    if not isinstance(v, torch.Tensor):
      v = torch.tensor(v)

    # Initial force computation
    f = force_fn(r)
    a = f / m

    for _ in range(steps):
      # v(t + 0.5*dt) = v(t) + 0.5 * a(t) * dt
      v_half = v + 0.5 * a * dt

      # r(t + dt) = r(t) + v(t + 0.5*dt) * dt
      r_next = r + v_half * dt

      # a(t + dt) = F(r(t + dt)) / m
      f_next = force_fn(r_next)
      a_next = f_next / m

      # v(t + dt) = v(t + 0.5*dt) + 0.5 * a(t + dt) * dt
      v_next = v_half + 0.5 * a_next * dt

      r = r_next
      v = v_next
      a = a_next

    return r, v


class MolecularDynamics(TorchModel):
  """
    Differentiable Molecular Dynamics wrapper.

    This class wraps the VelocityVerlet module in a TorchModel.
    """

  def __init__(self,
               time_step=0.001,
               mass=1.0,
               force_fn=None,
               loss=None,
               **kwargs):
    model = VelocityVerlet(time_step=time_step, mass=mass, force_fn=force_fn)
    # loss is optional for simulation
    super(MolecularDynamics, self).__init__(model, loss=loss, **kwargs)


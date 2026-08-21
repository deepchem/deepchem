import torch
from typing import Callable, Tuple


Tensor = torch.Tensor


def harmonic_force(x: Tensor, k: float = 1.0) -> Tensor:
    """
    Harmonic force: F = -k*x

    Parameters
    ----------
    x: Tensor
        Positions tensor
    k: float
        Spring constant

    Returns
    -------
    Tensor
        Force tensor
    """
    return -k * x


def velocity_verlet_step(
    x: Tensor,
    v: Tensor,
    m: Tensor,
    dt: float,
    force_fn: Callable[[Tensor], Tensor],
) -> Tuple[Tensor, Tensor]:
    """
    Perform one Velocity-Verlet integration step.

    Returns
    -------
    (x_new, v_new)
    """
    a = force_fn(x) / m

    x_new = x + v * dt + 0.5 * a * dt**2

    a_new = force_fn(x_new) / m

    v_new = v + 0.5 * (a + a_new) * dt

    return x_new, v_new


def simulate_dynamics(
    x0: Tensor,
    v0: Tensor,
    m: Tensor,
    dt: float,
    steps: int,
    force_fn: Callable[[Tensor], Tensor],
) -> Tensor:
    """
    Simulate dynamics using Velocity-Verlet.

    Returns
    -------
    Tensor
        Trajectory tensor of shape (steps, ...)
    """
    x, v = x0, v0
    traj = []

    for _ in range(steps):
        x, v = velocity_verlet_step(x, v, m, dt, force_fn)
        traj.append(x)

    return torch.stack(traj)

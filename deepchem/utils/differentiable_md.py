import torch
from typing import Callable, Tuple

Tensor = torch.Tensor


class VelocityVerletIntegrator:
    """
    Differentiable Velocity-Verlet integrator.

    Supports multi-particle systems.
    All operations are autograd-compatible.
    """

    def __init__(self, dt: float):
        self.dt = dt

    def step(
        self,
        x: Tensor,
        v: Tensor,
        m: Tensor,
        force_fn: Callable[[Tensor], Tensor],
    ) -> Tuple[Tensor, Tensor]:

        dt = self.dt

        a = force_fn(x) / m.unsqueeze(-1)

        x_new = x + v * dt + 0.5 * a * dt**2

        a_new = force_fn(x_new) / m.unsqueeze(-1)

        v_new = v + 0.5 * (a + a_new) * dt

        return x_new, v_new

    def simulate(
        self,
        x0: Tensor,
        v0: Tensor,
        m: Tensor,
        steps: int,
        force_fn: Callable[[Tensor], Tensor],
    ) -> Tensor:

        x, v = x0, v0
        traj = [x]

        for _ in range(steps):
            x, v = self.step(x, v, m, force_fn)
            traj.append(x)

        return torch.stack(traj)

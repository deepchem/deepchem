import torch
from deepchem.utils.differentiable_md import VelocityVerletIntegrator


def simple_harmonic_force(x):
    return -x


def test_gradients_flow():

    N, D = 5, 3

    x0 = torch.randn(N, D, requires_grad=True)
    v0 = torch.zeros(N, D)
    m = torch.ones(N)

    integrator = VelocityVerletIntegrator(dt=0.01)

    traj = integrator.simulate(
        x0, v0, m,
        steps=50,
        force_fn=simple_harmonic_force
    )

    loss = traj[-1].pow(2).sum()
    loss.backward()

    assert x0.grad is not None
    assert torch.isfinite(x0.grad).all()

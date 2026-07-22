import torch
from deepchem.utils.differentiable_md import (
    simulate_dynamics,
    harmonic_force,
)


def test_differentiable_md_grad():
    x0 = torch.tensor([1.0, 0.0], requires_grad=True)
    v0 = torch.tensor([0.0, 1.0])
    m = torch.tensor(1.0)

    traj = simulate_dynamics(
        x0, v0, m,
        dt=0.01,
        steps=50,
        force_fn=harmonic_force
    )

    loss = traj[-1].pow(2).sum()
    loss.backward()

    assert x0.grad is not None
    assert torch.isfinite(x0.grad).all()

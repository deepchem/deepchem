"""Tests for differentiable Rys quadrature (spherical.py)."""
import torch
from deepchem.utils.analytical_integrators_torch.spherical import rys_roots, boys


def test_moment_equations():
    """Roots/weights must satisfy sum_k w_k * t_k^m = F_m(x) where t=u/(1+u)."""
    nroots = 3
    x = torch.tensor(2.5, dtype=torch.float64)
    roots, weights = rys_roots(nroots, x)
    t = roots / (1.0 + roots)
    f = boys(2 * nroots - 1, x.item())
    for m in range(2 * nroots):
        lhs = (weights * t ** m).sum().item()
        assert abs(lhs - float(f[m])) < 1e-12, f"Moment {m} failed"


def test_nroots_1():
    """Single-root special case."""
    x = torch.tensor(1.0, dtype=torch.float64)
    roots, weights = rys_roots(1, x)
    assert roots.shape == (1,)
    assert weights.shape == (1,)
    assert roots[0].item() > 0


def test_gradient_exists():
    """rys_roots must produce non-zero gradients w.r.t. x."""
    x = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    roots, weights = rys_roots(2, x)
    loss = roots.sum() + weights.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.item() != 0.0


def test_gradient_finite_difference():
    """Autograd gradient should match finite differences."""
    x = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    nroots = 2
    roots, weights = rys_roots(nroots, x)
    loss = roots.sum() + weights.sum()
    loss.backward()
    analytic = x.grad.item()

    eps = 1e-6
    with torch.no_grad():
        r1, w1 = rys_roots(nroots, torch.tensor(2.0 + eps, dtype=torch.float64))
        r0, w0 = rys_roots(nroots, torch.tensor(2.0 - eps, dtype=torch.float64))
    fd = ((r1.sum() + w1.sum()) - (r0.sum() + w0.sum())).item() / (2 * eps)

    assert abs(analytic - fd) < 1e-5, f"analytic={analytic}, fd={fd}"

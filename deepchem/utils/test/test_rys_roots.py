"""Tests for Rys quadrature (spherical.py)."""
import torch
from deepchem.utils.analytical_integrators_torch.spherical import rys_roots, boys


def test_moment_equations():
    """Roots/weights must satisfy sum_k w_k * t_k^m = F_m(x) where t=u/(1+u)."""
    nroots = 3
    x = 2.5
    roots, weights = rys_roots(nroots, x)
    t = [r / (1.0 + r) for r in roots]
    f = boys(2 * nroots - 1, x)
    for m in range(2 * nroots):
        lhs = sum(weights[k] * t[k] ** m for k in range(nroots))
        assert abs(lhs - float(f[m])) < 1e-12, f"Moment {m} failed"


def test_nroots_1():
    """Single-root special case."""
    x = 1.0
    roots, weights = rys_roots(1, x)
    assert len(roots) == 1
    assert len(weights) == 1
    assert roots[0] > 0


def test_tensor_input():
    """rys_roots should accept a torch scalar tensor as x."""
    x = torch.tensor(2.5, dtype=torch.float64)
    roots, weights = rys_roots(3, x)
    assert len(roots) == 3
    assert len(weights) == 3

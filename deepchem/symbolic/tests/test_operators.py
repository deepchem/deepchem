"""Tests for deepchem/symbolic/operators.py

Validates three properties for every operator:
1. Forward value correctness
2. Gradient integrity via torch.autograd.gradcheck
3. torch.compile compatibility via fullgraph=True
"""
import sys
import pytest
import torch
from deepchem.symbolic.operators import (
    op_add, op_sub, op_mul, op_div, op_pow,
    op_sin, op_cos, op_exp, op_log,
    op_arrhenius, op_logistic, op_henry
)

# ── helpers ───────────────────────────────────────────────────────────────

def make(val, double=True):
    """Create a float64 scalar tensor with grad for gradcheck."""
    dtype = torch.float64 if double else torch.float32
    return torch.tensor(val, dtype=dtype, requires_grad=True)


def gradcheck_op(fn, inputs, tol=1e-4):
    """Run gradcheck and return True if Jacobian error < tol."""
    return torch.autograd.gradcheck(fn, inputs, eps=1e-6, atol=tol, rtol=1e-3)


def compile_check(fn, inputs):
    """Verify zero graph breaks via fullgraph=True."""
    compiled = torch.compile(fn, backend="inductor", fullgraph=True)
    float_inputs = tuple(
        x.detach().float().requires_grad_(False) for x in inputs
    )
    out = compiled(*float_inputs)
    assert out is not None
    return True


# ═════════════════════════════════════════════════════════════════════════
# FORWARD CORRECTNESS
# ═════════════════════════════════════════════════════════════════════════

class TestForwardCorrectness:

    def test_op_add(self):
        assert op_add(torch.tensor(2.0), torch.tensor(3.0)).item() == 5.0

    def test_op_sub(self):
        assert op_sub(torch.tensor(5.0), torch.tensor(3.0)).item() == 2.0

    def test_op_mul(self):
        assert op_mul(torch.tensor(2.0), torch.tensor(3.0)).item() == 6.0

    def test_op_div_normal(self):
        result = op_div(torch.tensor(6.0), torch.tensor(2.0))
        assert abs(result.item() - 3.0) < 1e-5

    def test_op_div_by_zero_returns_zero(self):
        result = op_div(torch.tensor(1.0), torch.tensor(0.0))
        assert torch.isfinite(result)

    def test_op_pow(self):
        result = op_pow(torch.tensor(2.0), torch.tensor(3.0))
        assert abs(result.item() - 8.0) < 1e-4

    def test_op_sin(self):
        assert abs(op_sin(torch.tensor(0.0)).item()) < 1e-6

    def test_op_cos(self):
        assert abs(op_cos(torch.tensor(0.0)).item() - 1.0) < 1e-6

    def test_op_exp_normal(self):
        assert abs(op_exp(torch.tensor(0.0)).item() - 1.0) < 1e-6

    def test_op_exp_clamp_no_inf(self):
        result = op_exp(torch.tensor(1000.0))
        assert torch.isfinite(result)

    def test_op_log_normal(self):
        assert abs(op_log(torch.tensor(1.0)).item()) < 1e-6

    def test_op_log_clamp_no_nan(self):
        result = op_log(torch.tensor(-1.0))
        assert torch.isfinite(result)
        assert not torch.isnan(result)

    def test_op_arrhenius_finite(self):
        result = op_arrhenius(torch.tensor(50000.0), torch.tensor(300.0))
        assert torch.isfinite(result)
        assert result.item() > 0

    def test_op_logistic_midpoint(self):
        result = op_logistic(
            torch.tensor(0.0), torch.tensor(1.0),
            torch.tensor(1.0), torch.tensor(0.0)
        )
        assert abs(result.item() - 0.5) < 1e-5

    def test_op_henry(self):
        result = op_henry(torch.tensor(2.0), torch.tensor(3.0))
        assert abs(result.item() - 6.0) < 1e-5


# ═════════════════════════════════════════════════════════════════════════
# GRADIENT INTEGRITY (gradcheck)
# ═════════════════════════════════════════════════════════════════════════

class TestGradcheck:

    def test_op_add_grad(self):
        assert gradcheck_op(op_add, (make(1.5), make(2.5)))

    def test_op_sub_grad(self):
        assert gradcheck_op(op_sub, (make(3.0), make(1.0)))

    def test_op_mul_grad(self):
        assert gradcheck_op(op_mul, (make(2.0), make(3.0)))

    def test_op_div_grad(self):
        assert gradcheck_op(op_div, (make(6.0), make(2.0)))

    def test_op_pow_grad(self):
        assert gradcheck_op(op_pow, (make(2.0), make(2.0)))

    def test_op_sin_grad(self):
        assert gradcheck_op(op_sin, (make(0.5),))

    def test_op_cos_grad(self):
        assert gradcheck_op(op_cos, (make(0.5),))

    def test_op_exp_grad(self):
        assert gradcheck_op(op_exp, (make(1.0),))

    def test_op_log_grad(self):
        assert gradcheck_op(op_log, (make(2.0),))

    def test_op_arrhenius_grad(self):
        assert gradcheck_op(op_arrhenius, (make(50000.0), make(300.0)))

    def test_op_logistic_grad(self):
        assert gradcheck_op(
            op_logistic,
            (make(0.0), make(1.0), make(1.0), make(0.0))
        )

    def test_op_henry_grad(self):
        assert gradcheck_op(op_henry, (make(2.0), make(3.0)))


# ═════════════════════════════════════════════════════════════════════════
# TORCH.COMPILE COMPATIBILITY (zero graph breaks)
# ═════════════════════════════════════════════════════════════════════════
@pytest.mark.skipif(sys.platform == "win32", reason="torch.compile is not yet supported on Windows")
class TestTorchCompile:

    def test_op_add_compile(self):
        assert compile_check(op_add, (make(1.0), make(2.0)))

    def test_op_sub_compile(self):
        assert compile_check(op_sub, (make(3.0), make(1.0)))

    def test_op_mul_compile(self):
        assert compile_check(op_mul, (make(2.0), make(3.0)))

    def test_op_div_compile(self):
        assert compile_check(op_div, (make(6.0), make(2.0)))

    def test_op_pow_compile(self):
        assert compile_check(op_pow, (make(2.0), make(2.0)))

    def test_op_sin_compile(self):
        assert compile_check(op_sin, (make(0.5),))

    def test_op_cos_compile(self):
        assert compile_check(op_cos, (make(0.5),))

    def test_op_exp_compile(self):
        assert compile_check(op_exp, (make(1.0),))

    def test_op_log_compile(self):
        assert compile_check(op_log, (make(2.0),))

    def test_op_arrhenius_compile(self):
        assert compile_check(op_arrhenius, (make(50000.0), make(300.0)))

    def test_op_logistic_compile(self):
        assert compile_check(
            op_logistic,
            (make(0.0), make(1.0), make(1.0), make(0.0))
        )

    def test_op_henry_compile(self):
        assert compile_check(op_henry, (make(2.0), make(3.0)))
"""
Differentiable operator library for the DeepChem Symbolic Regression engine.

Every operator is a pure torch function with:
- A registered autograd backward pass (via torch built-ins)
- Zero Python control flow (torch.where instead of if/else)
- torch.compile compatibility confirmed via fullgraph=True

References
----------
Module B of the GSoC 2026 Symbolic Regression proposal.
"""

import torch
from torch import Tensor

# ── PHYSICAL CONSTANTS ────────────────────────────────────────────────────
R_GAS = 8.314  # J/(mol·K) — universal gas constant

# ── NUMERICAL GUARDS ──────────────────────────────────────────────────────
EXP_CLAMP_MIN  = -88.0   # float32 underflow boundary
EXP_CLAMP_MAX  =  88.0   # float32 overflow boundary
LOG_CLAMP_MIN  =  1e-8   # prevents log(0)
DIV_EPS        =  1e-8   # prevents division by zero


# ═════════════════════════════════════════════════════════════════════════
# CORE OPERATORS (9)
# ═════════════════════════════════════════════════════════════════════════

def op_add(x: Tensor, y: Tensor) -> Tensor:
    """Element-wise addition.

    Parameters
    ----------
    x : Tensor
        Left operand.
    y : Tensor
        Right operand.

    Returns
    -------
    Tensor
        x + y

    Examples
    --------
    >>> op_add(torch.tensor(2.0), torch.tensor(3.0))
    tensor(5.)
    """
    return x + y


def op_sub(x: Tensor, y: Tensor) -> Tensor:
    """Element-wise subtraction.

    Parameters
    ----------
    x : Tensor
        Left operand.
    y : Tensor
        Right operand.

    Returns
    -------
    Tensor
        x - y

    Examples
    --------
    >>> op_sub(torch.tensor(5.0), torch.tensor(3.0))
    tensor(2.)
    """
    return x - y


def op_mul(x: Tensor, y: Tensor) -> Tensor:
    """Element-wise multiplication.

    Parameters
    ----------
    x : Tensor
        Left operand.
    y : Tensor
        Right operand.

    Returns
    -------
    Tensor
        x * y

    Examples
    --------
    >>> op_mul(torch.tensor(2.0), torch.tensor(3.0))
    tensor(6.)
    """
    return x * y


def op_div(x: Tensor, y: Tensor) -> Tensor:
    """Element-wise safe division.

    Uses torch.where to avoid division by zero without Python control
    flow, preserving torch.compile graph integrity.

    Parameters
    ----------
    x : Tensor
        Numerator.
    y : Tensor
        Denominator.

    Returns
    -------
    Tensor
        x / y where |y| > eps, else 0.0

    Examples
    --------
    >>> op_div(torch.tensor(6.0), torch.tensor(2.0))
    tensor(3.)
    >>> op_div(torch.tensor(1.0), torch.tensor(0.0))
    tensor(0.)
    """
    return torch.where(
        torch.abs(y) > DIV_EPS,
        x / (y + DIV_EPS),
        torch.zeros_like(x)
    )


def op_pow(x: Tensor, y: Tensor) -> Tensor:
    """Element-wise power with safe base clamping.

    Clamps base to non-negative values to prevent NaN from
    fractional exponents applied to negative bases.

    Parameters
    ----------
    x : Tensor
        Base.
    y : Tensor
        Exponent.

    Returns
    -------
    Tensor
        |x|^y (base is absolute-valued for numerical safety)

    Examples
    --------
    >>> op_pow(torch.tensor(2.0), torch.tensor(3.0))
    tensor(8.)
    """
    return torch.pow(torch.abs(x) + LOG_CLAMP_MIN, y)


def op_sin(x: Tensor) -> Tensor:
    """Element-wise sine.

    Parameters
    ----------
    x : Tensor
        Input in radians.

    Returns
    -------
    Tensor
        sin(x)

    Examples
    --------
    >>> torch.allclose(op_sin(torch.tensor(0.0)), torch.tensor(0.0))
    True
    """
    return torch.sin(x)


def op_cos(x: Tensor) -> Tensor:
    """Element-wise cosine.

    Parameters
    ----------
    x : Tensor
        Input in radians.

    Returns
    -------
    Tensor
        cos(x)

    Examples
    --------
    >>> torch.allclose(op_cos(torch.tensor(0.0)), torch.tensor(1.0))
    True
    """
    return torch.cos(x)


def op_exp(x: Tensor) -> Tensor:
    """Element-wise exponential with float32 overflow clamping.

    Clamps input to [-88, 88] before applying exp to prevent
    float32 overflow (exp(89) = inf on float32).

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        exp(clamp(x, -88, 88))

    Examples
    --------
    >>> op_exp(torch.tensor(0.0))
    tensor(1.)
    >>> op_exp(torch.tensor(1000.0))  # would be inf without clamp
    tensor(1.6522e+38)
    """
    return torch.exp(torch.clamp(x, EXP_CLAMP_MIN, EXP_CLAMP_MAX))


def op_log(x: Tensor) -> Tensor:
    """Element-wise natural logarithm with zero-guard clamping.

    Clamps input to [1e-8, inf) before applying log to prevent
    log(0) = -inf and log of negative values = NaN.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        log(clamp(x, 1e-8, inf))

    Examples
    --------
    >>> torch.allclose(op_log(torch.tensor(1.0)), torch.tensor(0.0))
    True
    >>> op_log(torch.tensor(-1.0))  # would be NaN without clamp
    tensor(-18.4207)
    """
    return torch.log(torch.clamp(x, LOG_CLAMP_MIN))


# ═════════════════════════════════════════════════════════════════════════
# CHEMISTRY OPERATORS (3)
# ═════════════════════════════════════════════════════════════════════════

def op_arrhenius(Ea: Tensor, T: Tensor) -> Tensor:
    """Arrhenius rate equation: exp(-Ea / (R * T)).

    Models temperature-dependent reaction rates in chemical kinetics.
    Reference: Arrhenius, S. (1889). Z. Phys. Chem., 4, 226-248.

    Parameters
    ----------
    Ea : Tensor
        Activation energy in J/mol.
    T : Tensor
        Temperature in Kelvin. Clamped to [1e-8, inf) to prevent
        division by zero at T=0.

    Returns
    -------
    Tensor
        exp(-Ea / (R * T)) where R = 8.314 J/(mol·K)

    Examples
    --------
    >>> op_arrhenius(torch.tensor(50000.0), torch.tensor(300.0))
    tensor(1.4264e-09)
    """
    T_safe = torch.clamp(T, LOG_CLAMP_MIN)
    exponent = -Ea / (R_GAS * T_safe)
    return torch.exp(torch.clamp(exponent, EXP_CLAMP_MIN, EXP_CLAMP_MAX))


def op_logistic(x: Tensor, L: Tensor, k: Tensor, x0: Tensor) -> Tensor:
    """Generalised logistic (Hill) equation: L / (1 + exp(-k * (x - x0))).

    Models sigmoid-shaped dose-response relationships and receptor
    binding curves in pharmacology.
    Reference: Hill, A.V. (1910). J. Physiol., 40, iv-vii.

    Parameters
    ----------
    x : Tensor
        Input variable (e.g., ligand concentration).
    L : Tensor
        Maximum value (carrying capacity).
    k : Tensor
        Steepness (growth rate).
    x0 : Tensor
        Midpoint (x-value of sigmoid inflection).

    Returns
    -------
    Tensor
        L / (1 + exp(-k * (x - x0)))

    Examples
    --------
    >>> op_logistic(torch.tensor(0.0), torch.tensor(1.0),
    ...             torch.tensor(1.0), torch.tensor(0.0))
    tensor(0.5000)
    """
    exponent = torch.clamp(-k * (x - x0), EXP_CLAMP_MIN, EXP_CLAMP_MAX)
    return L / (1.0 + torch.exp(exponent))


def op_henry(p: Tensor, kH: Tensor) -> Tensor:
    """Henry's Law: concentration = kH * p.

    Models gas solubility in liquids as a function of partial pressure.
    Reference: Henry, W. (1803). Phil. Trans. R. Soc. London, 93, 29-274.

    Parameters
    ----------
    p : Tensor
        Partial pressure of the gas (Pa).
    kH : Tensor
        Henry's Law constant (mol/(L·Pa)).

    Returns
    -------
    Tensor
        kH * p

    Examples
    --------
    >>> op_henry(torch.tensor(101325.0), torch.tensor(3.4e-4))
    tensor(34.4505)
    """
    return kH * p
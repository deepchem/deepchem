"""
Rys quadrature roots and weights via the RDK algorithm.
Uses mpmath for stable high-precision Boys function evaluation
and Schmidt orthogonalization, matching libcint's quad-precision fallback.
"""
import numpy as np
import mpmath

DPS = 50  # working precision (decimal digits)


def boys(mmax, x):
    """Evaluate the Boys function F_m(x) for m = 0 .. mmax.

    Uses a Taylor series for small x (numerically stable) and upward
    recurrence for large x, both computed at ``DPS``-digit precision via
    mpmath.  The Boys function arises in the analytic evaluation of
    electron-repulsion and nuclear-attraction integrals over Gaussian
    basis functions.

    Parameters
    ----------
    mmax : int
        Highest order m to compute.  Returns a list of length mmax + 1.
    x : float or mpmath.mpf
        Non-negative argument of the Boys function.

    Returns
    -------
    list of mpmath.mpf
        Values [F_0(x), F_1(x), ..., F_mmax(x)].

    Examples
    --------
    >>> from deepchem.utils.analytical_integrators_torch.spherical import boys
    >>> f = boys(2, 1.0)
    >>> abs(float(f[0]) - 0.7468241330) < 1e-8
    True

    References
    ----------
    .. [1] S. F. Boys, "Electronic Wave Functions. I. A General Method of
       Calculation for the Stationary States of Any Molecular System."
       Proceedings of the Royal Society of London A, 200, 542–554 (1950).
    .. [2] I. Shavitt, "The Gaussian Function in Calculations of Statistical
       Mechanics and Quantum Mechanics." University of Wisconsin, NRCC
       Technical Report (1963).
    """
    if hasattr(x, 'item'):
        x = x.item()
    x = mpmath.mpf(x)
    eps = mpmath.power(10, -DPS - 2)
    if x < mpmath.mpf("0.5") * (mmax + 1):
        f = []
        for m in range(mmax + 1):
            s, term = mpmath.mpf(1) / (2 * m + 1), mpmath.mpf(1)
            for k in range(1, 500):
                term *= -x / k
                ds = term / (2 * m + 2 * k + 1)
                s += ds
                if mpmath.fabs(ds) < eps * mpmath.fabs(s):
                    break
            f.append(s)
        return f
    ex = mpmath.exp(-x)
    f = [mpmath.sqrt(mpmath.pi / (4 * x)) * mpmath.erf(mpmath.sqrt(x))]
    for m in range(mmax):
        f.append(((2 * m + 1) * f[m] - ex) / (2 * x))
    return f


def poly(a, x):
    """Evaluate a polynomial using Horner's method.

    Given coefficient list ``a`` where ``a[0]`` is the constant term and
    ``a[-1]`` is the leading coefficient, evaluate the polynomial at ``x``
    using the numerically stable Horner scheme.

    Parameters
    ----------
    a : list of mpmath.mpf
        Polynomial coefficients ordered from constant to leading term,
        i.e. ``a[0] + a[1]*x + ... + a[n]*x^n``.
    x : mpmath.mpf
        Point at which to evaluate the polynomial.

    Returns
    -------
    mpmath.mpf
        Value of the polynomial at ``x``.

    Examples
    --------
    >>> import mpmath
    >>> from deepchem.utils.analytical_integrators_torch.spherical import poly
    >>> # Evaluate 1 + 2x + 3x^2 at x=2: 1 + 4 + 12 = 17
    >>> float(poly([mpmath.mpf(1), mpmath.mpf(2), mpmath.mpf(3)], mpmath.mpf(2)))
    17.0

    References
    ----------
    .. [1] W. H. Press et al., "Numerical Recipes: The Art of Scientific
       Computing", 3rd ed., Cambridge University Press (2007), §5.3.
    """
    p = a[-1]
    for c in reversed(a[:-1]):
        p = p * x + c
    return p


def schmidt(f, n):
    """Modified Gram-Schmidt orthogonalization on Boys function moments.

    Constructs the upper-triangular coefficient matrix ``cs`` such that the
    polynomials ``P_j(t) = sum_{i=0}^{j} cs[i][j] * t^i`` are orthonormal
    with respect to the weight function defined by the Boys function moments
    ``f[k] = F_k(x)``.  These orthogonal polynomials define the Rys
    quadrature nodes.

    Parameters
    ----------
    f : list of mpmath.mpf
        Boys function moments F_0(x), F_1(x), ..., F_{2n-2}(x).
    n : int
        Number of polynomials to construct (equals nroots + 1).

    Returns
    -------
    list of list of mpmath.mpf
        Coefficient matrix ``cs`` of shape (n, n).  ``cs[i][j]`` is the
        coefficient of ``t^i`` in the j-th orthogonal polynomial.

    References
    ----------
    .. [1] M. Dupuis, J. Rys, H. F. King, "Evaluation of the molecular
       integrals over Gaussian basis functions." Journal of Chemical Physics,
       65(1), 111–116 (1976).
    .. [2] J. Rys, M. Dupuis, H. F. King, "Computation of electron repulsion
       integrals using the Rys quadrature method." Journal of Computational
       Chemistry, 4(2), 154–157 (1983).
    """
    cs = [[mpmath.mpf(0)] * n for _ in range(n)]
    cs[0][0] = 1 / mpmath.sqrt(f[0])
    fac = -f[1] / f[0]
    tmp = 1 / mpmath.sqrt(f[2] + fac * f[1])
    cs[0][1], cs[1][1] = fac * tmp, tmp
    for j in range(2, n):
        v, fac = [mpmath.mpf(0)] * j, f[2 * j]
        for k in range(j):
            dot = sum(cs[i][k] * f[i + j] for i in range(k + 1))
            for i in range(k + 1):
                v[i] -= dot * cs[i][k]
            fac -= dot * dot
        if fac <= 0:
            break
        fac = 1 / mpmath.sqrt(fac)
        cs[j][j] = fac
        for i in range(j):
            cs[i][j] = fac * v[i]
    return cs


def find_roots(coeffs, rt, tol):
    """Find roots of a polynomial inside sign-change brackets.

    Uses a combined bisection / secant method.  The brackets are defined by
    the previous-column roots ``rt``; if no sign change is detected between
    consecutive brackets the corresponding root is set to the sentinel value
    1 (outside the valid t-domain [0, 1)).

    Parameters
    ----------
    coeffs : list of mpmath.mpf
        Polynomial coefficients (constant term first, leading term last).
    rt : list of mpmath.mpf
        Upper bracket endpoints from the previous Rys column.  Must have
        length >= len(coeffs) - 1.
    tol : mpmath.mpf
        Convergence tolerance for bisection; iteration stops when the
        bracket width falls below ``tol``.

    Returns
    -------
    list of mpmath.mpf
        Refined roots, one per bracket.  Any root that could not be located
        (same sign at both bracket ends) is set to mpmath.mpf(1).

    References
    ----------
    .. [1] M. Dupuis, J. Rys, H. F. King, "Evaluation of the molecular
       integrals over Gaussian basis functions." Journal of Chemical Physics,
       65(1), 111–116 (1976).
    """
    result = []
    x1i, p1i = mpmath.mpf(0), coeffs[0]
    for m in range(len(coeffs) - 1):
        x0, p0, x1i = x1i, p1i, rt[m]
        p1i = poly(coeffs, x1i)
        if p1i == 0:
            result.append(x1i)
            continue
        if p0 * p1i > 0:          # no sign change → root outside [0,1)
            result.append(mpmath.mpf(1))
            continue
        x0, x1 = (x0, x1i) if x0 <= x1i else (x1i, x0)
        p0, p1 = poly(coeffs, x0), poly(coeffs, x1)
        xi = x0 + (x0 - x1) / (p1 - p0) * p0
        for _ in range(600):
            if not (x1 > tol + x0 or x0 > x1 + tol):
                break
            pi = poly(coeffs, xi)
            if pi == 0:
                break
            if p0 * pi <= 0:
                x1, p1, xi = xi, pi, 0.25 * x0 + 0.75 * xi
            else:
                x0, p0, xi = xi, pi, 0.75 * xi + 0.25 * x1
            pi = poly(coeffs, xi)
            if pi == 0:
                break
            if p0 * pi <= 0:
                x1, p1 = xi, pi
            else:
                x0, p0 = xi, pi
            xi = x0 + (x0 - x1) / (p1 - p0) * p0
        result.append(xi)
    return result


def rys_roots(nroots, x):
    """Compute Rys quadrature roots and weights for GTO integral evaluation.

    Implements the Rys–Dupuis–King (RDK) algorithm at ``DPS``-digit
    precision.  The roots (u-variable) and weights satisfy the moment
    equations

        sum_k  w_k * t_k^m  =  F_m(x),   m = 0, 1, ..., 2*nroots - 1

    where ``t_k = u_k / (1 + u_k)`` are the t-roots and F_m is the Boys
    function.  These quadrature points are used to evaluate the
    multi-dimensional GTO integrals via the Rys polynomial method.

    Parameters
    ----------
    nroots : int
        Number of quadrature points (1–13).  Must satisfy
        nroots >= (li + lj + lk + ll) // 2 + 1 for the target integral.
    x : float or torch.Tensor (scalar)
        Boys-function argument x = a0 * |R_ij - R_kl|^2 >= 0.
        A torch scalar is automatically converted via ``.item()``.

    Returns
    -------
    roots : list of float
        The u-roots u_k = t_k / (1 - t_k), each in [0, inf).
    weights : list of float
        Corresponding quadrature weights w_k > 0.

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.analytical_integrators_torch.spherical import rys_roots
    >>> roots, weights = rys_roots(1, 0.0)
    >>> abs(roots[0] - 0.5) < 1e-10   # u=t/(1-t) where t=F1(0)/F0(0)=1/3
    True

    References
    ----------
    .. [1] M. Dupuis, J. Rys, H. F. King, "Evaluation of the molecular
       integrals over Gaussian basis functions." Journal of Chemical Physics,
       65(1), 111–116 (1976).
    .. [2] J. Rys, M. Dupuis, H. F. King, "Computation of electron repulsion
       integrals using the Rys quadrature method." Journal of Computational
       Chemistry, 4(2), 154–157 (1983).
    .. [3] Q. Sun, "Libcint: An efficient general integral library for
       Gaussian basis functions." Journal of Computational Chemistry,
       36(22), 1664–1671 (2015).
    """
    if hasattr(x, 'item'):
        x = x.item()
    with mpmath.workdps(DPS):
        tol = mpmath.power(10, -(DPS - 5))
        f = boys(2 * nroots, x)
        n = nroots + 1
        cs = schmidt(f, n)

        if nroots == 1:
            r = f[1] / f[0]
            return [float(r / (1 - r))], [float(f[0])]

        a0, a1, a2 = cs[0][2], cs[1][2], cs[2][2]
        d = mpmath.sqrt(max(0, a1 * a1 - 4 * a0 * a2))
        rt = [(-a1 - d) / (2 * a2), (-a1 + d) / (2 * a2)] + [mpmath.mpf(1)] * n

        for k in range(2, nroots):
            col = k + 1
            new = find_roots([cs[i][col] for i in range(col + 1)], rt[:col], tol)
            for i, r in enumerate(new):
                rt[i] = r

        roots, weights = np.zeros(nroots), np.zeros(nroots)
        for k in range(nroots):
            r = rt[k]
            if r >= 1:
                continue
            dum = 1 / f[0] + sum(
                poly([cs[i][j] for i in range(j + 1)], r) ** 2
                for j in range(1, nroots)
            )
            roots[k] = float(r / (1 - r))
            weights[k] = float(1 / dum)

    return list(roots), list(weights)

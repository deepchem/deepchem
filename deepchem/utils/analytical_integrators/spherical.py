"""
Rys quadrature roots and weights via the RDK algorithm.
Uses mpmath for stable high-precision Boys function evaluation
and Schmidt orthogonalization, matching libcint's quad-precision fallback.
"""
import numpy as np
import mpmath

_DPS = 50  # working precision (decimal digits)


def _boys(mmax, x):
    """Boys function F_m(x) for m=0..mmax. Taylor series for small x (stable),
    upward recurrence otherwise."""
    x = mpmath.mpf(x)
    eps = mpmath.power(10, -_DPS - 2)
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


def _poly(a, x):
    """Horner evaluation; a[0]=constant, a[-1]=leading coefficient."""
    p = a[-1]
    for c in reversed(a[:-1]):
        p = p * x + c
    return p


def _schmidt(f, n):
    """Modified Gram-Schmidt orthogonalization on Boys function moments."""
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


def _find_roots(coeffs, rt, tol):
    """Bisection/secant root-finding in brackets defined by rt.
    Missing roots (same sign at both bracket ends) get sentinel value 1."""
    result = []
    x1i, p1i = mpmath.mpf(0), coeffs[0]
    for m in range(len(coeffs) - 1):
        x0, p0, x1i = x1i, p1i, rt[m]
        p1i = _poly(coeffs, x1i)
        if p1i == 0:
            result.append(x1i)
            continue
        if p0 * p1i > 0:          # no sign change → root outside [0,1)
            result.append(mpmath.mpf(1))
            continue
        x0, x1 = (x0, x1i) if x0 <= x1i else (x1i, x0)
        p0, p1 = _poly(coeffs, x0), _poly(coeffs, x1)
        xi = x0 + (x0 - x1) / (p1 - p0) * p0
        for _ in range(600):
            if not (x1 > tol + x0 or x0 > x1 + tol):
                break
            pi = _poly(coeffs, xi)
            if pi == 0:
                break
            if p0 * pi <= 0:
                x1, p1, xi = xi, pi, 0.25 * x0 + 0.75 * xi
            else:
                x0, p0, xi = xi, pi, 0.75 * xi + 0.25 * x1
            pi = _poly(coeffs, xi)
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
    """
    Rys quadrature roots and weights.

    Parameters
    ----------
    nroots : int    number of quadrature points (1–13)
    x      : float  Boys-function argument (x ≥ 0)

    Returns
    -------
    roots, weights : lists of length nroots
    """
    with mpmath.workdps(_DPS):
        tol = mpmath.power(10, -(_DPS - 5))
        f = _boys(2 * nroots, x)
        n = nroots + 1
        cs = _schmidt(f, n)

        # nroots == 1: closed-form from first moment
        if nroots == 1:
            r = f[1] / f[0]
            return [float(r / (1 - r))], [float(f[0])]

        # Seed from quadratic on column 2
        a0, a1, a2 = cs[0][2], cs[1][2], cs[2][2]
        d = mpmath.sqrt(max(0, a1 * a1 - 4 * a0 * a2))
        rt = [(-a1 - d) / (2 * a2), (-a1 + d) / (2 * a2)] + [mpmath.mpf(1)] * n

        # Refine column by column
        for k in range(2, nroots):
            col = k + 1
            new = _find_roots([cs[i][col] for i in range(col + 1)], rt[:col], tol)
            for i, r in enumerate(new):
                rt[i] = r

        # Convert t-roots → u-roots and compute weights
        roots, weights = np.zeros(nroots), np.zeros(nroots)
        for k in range(nroots):
            r = rt[k]
            if r >= 1:
                continue
            dum = 1 / f[0] + sum(
                _poly([cs[i][j] for i in range(j + 1)], r) ** 2
                for j in range(1, nroots)
            )
            roots[k] = float(r / (1 - r))
            weights[k] = float(1 / dum)

    return list(roots), list(weights)

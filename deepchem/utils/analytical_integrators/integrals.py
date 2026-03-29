import mpmath


def boys(mmax: int, x: float):
    """Boys function F_m(x) for m=0..mmax. Taylor series for small x (stable),
    upward recurrence otherwise.
    
    Examples
    --------
    >>> from deepchem.utils.analytical_integrators.integrals import boys
    >>> 

    Parameters
    ----------

    Returns
    -------
    
    """
    x = mpmath.mpf(x)
    eps = mpmath.power(10, -52)
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

import mpmath
from typing import List


def boys(mmax: int, x: float, precision: int = 52) -> List:
    r"""Compute the Boys function F_m(x) for all orders m = 0, 1, ..., mmax.

    The Boys function is defined as:

    .. math::
        F_m(x) = \int_0^1 t^{2m} e^{-x t^2} dt

    It is used when evaluating two-electron repulsion integrals and nuclear
    attraction integrals over Gaussian-type orbitals.

    Two evaluation strategies are used, switching at ``x < 0.5 * (mmax + 1)``:

    **Small x - Taylor series expansion**

    Each F_m(x) is computed independently via the convergent series:

    .. math::
        F_m(x) = \sum_{k=0}^{\infty} \frac{(-x)^k}{k! (2m + 2k + 1)}

    The series is summed until the absolute correction falls below
    :math:`10^{-52}` times the accumulated sum, with a hard cap of
    500 terms. This avoids the subtractive cancellation that affects
    the recurrence relation when x is small.

    **Large x - Analytic base case + upward recurrence**

    The base case uses the closed-form expression:

    .. math::
        F_0(x) = \frac{1}{2}\sqrt{\frac{\pi}{x}} \, \mathrm{erf}(\sqrt{x})

    Higher orders are obtained via upward recurrence:

    .. math::
        F_{m+1}(x) = \frac{(2m+1)\,F_m(x) - e^{-x}}{2x}

    This recurrence is numerically stable in the upward direction for
    large x, where F_m(x) decreases rapidly with m.


    Examples
    --------
    >>> from deepchem.utils.gaussian_integrals.integral_utils import boys
    >>> boys(1, 5.0)
    [mpf('0.39571230961051351'), mpf('0.038897436261142802')]

    Parameters
    ----------
    mmax: int
        Maximum order of the Boys function to compute. The function
        returns values for m = 0 through m = mmax (inclusive).
    x: float
        Argument of the Boys function. Physically this relates to the
        product of Gaussian exponents and squared inter-center distances.
        Must be non-negative.
    precision: int (default 52)
        Precision at which the calculation will be performed.

    Returns
    -------
    List[mpmath.mpf]
        List of length ``mmax + 1``, where element ``[m]`` is F_m(x).

    References
    ----------
    .. [1] Boys, S. F. (1950). Electronic wave functions - I. A general
       method of calculation for the stationary states of any molecular
       system. Proceedings of the Royal Society of London A, 200(1063),
       542-554. https://doi.org/10.1098/rspa.1950.0036
    """
    x = mpmath.mpf(x)
    eps = mpmath.power(10, -precision)
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

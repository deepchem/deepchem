import warnings
import torch
import numpy as np
from typing import Callable, List
from deepchem.utils.differentiation_utils.optimize.rootsolver import TerminationCondition
from deepchem.utils import ConvergenceWarning


def anderson_acc(
    fcn: Callable[..., torch.Tensor],
    x0: torch.Tensor,
    params: List,
    feat_ndims: int = 1,
    # anderson_acc parameters
    msize: int = 5,  # how many previous iterations we should save
    beta: float = 1.0,
    lmbda: float = 1e-4,  # small number to ensure invertability of the matrix
    # stopping criteria
    maxiter=None,
    f_tol=None,
    f_rtol=None,
    x_tol=None,
    x_rtol=None,
    custom_terminator=None,
    # misc options
    verbose: bool = False,
) -> torch.Tensor:
    """
    Solve the equilibrium (or fixed-point iteration) problem using Anderson acceleration.

    Examples
    --------
    >>> import torch
    >>> def fcn(x):
    ...     return x
    >>> x0 = torch.tensor([0.0], requires_grad=True)
    >>> x = anderson_acc(fcn, x0, [], 2, 10, maxiter=1000)
    >>> x
    tensor([0.], requires_grad=True)

    Parameters
    ----------
    feat_ndims: int
        The number of dimensions at the end that describe the features (i.e. non-batch dimensions)
    msize: int
        The maximum number of previous iterations we should save for the algorithm
    beta: float
        The damped or overcompensated parameters
    lmbda: float
        Small number to ensure invertability of the matrix
    maxiter: int or None
        Maximum number of iterations, or inf if it is set to None.
    f_tol: float or None
        The absolute tolerance of the norm of the output ``f - x``.
    f_rtol: float or None
        The relative tolerance of the norm of the output ``f - x``.
    x_tol: float or None
        The absolute tolerance of the norm of the input ``x``.
    x_rtol: float or None
        The relative tolerance of the norm of the input ``x``.
    verbose: bool
        Options for verbosity

    References
    ----------
    .. [1] H. F. Walker and P. Ni,
           "Anderson Acceleration for Fixed-Point Iterations".
           Siam J. Numer. Anal.
           https://users.wpi.edu/~walker/Papers/Walker-Ni,SINUM,V49,1715-1735.pdf
    """
    # x0: (..., *nfeats)
    featshape = x0.shape[-feat_ndims:]
    batch_shape = x0.shape[:-feat_ndims]
    feat_size = int(np.prod(x0.shape[-feat_ndims:]))
    dtype = x0.dtype
    device = x0.device

    if maxiter is None:
        maxiter = 100 * (feat_size + 1)

    def _ravel(x: torch.Tensor) -> torch.Tensor:
        # reshape x to have a shape of (batch_size, feats_dim)
        # x: (..., *nfeats)
        xn = x.reshape(*batch_shape, -1)  # (..., feats_tot)
        return xn

    def _unravel(xn: torch.Tensor) -> torch.Tensor:
        # xn: (..., feats_tot)
        x = xn.reshape(*batch_shape, *featshape)  # (..., *nfeats)
        return x

    def _fcn(xn: torch.Tensor) -> torch.Tensor:
        # x: (..., feats_dim)
        x = _unravel(xn)
        fn = _ravel(fcn(x, *params))
        return fn  # (..., feats_dim)

    xn = _ravel(x0)
    fn = _fcn(xn)
    xcollect = torch.zeros((*batch_shape, msize, feat_size),
                           dtype=dtype,
                           device=device)
    fcollect = torch.zeros((*batch_shape, msize, feat_size),
                           dtype=dtype,
                           device=device)
    xcollect[..., 0, :] = xn
    fcollect[..., 0, :] = fn
    xn = fn
    fn = _fcn(xn)
    xcollect[..., 1, :] = xn
    fcollect[..., 1, :] = fn

    hmat = torch.zeros((*batch_shape, msize + 1, msize + 1),
                       dtype=dtype,
                       device=device)
    y = torch.zeros((*batch_shape, msize + 1, 1), dtype=dtype, device=device)
    hmat[..., 0, 1:] = 1.0
    hmat[..., 1:, 0] = 1.0
    y[..., 0, :] = 1.0

    devnorm = (fn - xn).norm()
    stop_cond = custom_terminator if custom_terminator is not None \
        else TerminationCondition(f_tol, f_rtol, devnorm, x_tol, x_rtol)
    if devnorm == 0:
        return x0

    converge = False
    for k in range(2, maxiter):
        nsize = min(k, msize)
        g = fcollect[..., :nsize, :] - xcollect[
            ..., :nsize, :]  # (..., nsize, feat_size)
        # torch.bmm(g, g.transpose(-2, -1))
        hmat[..., 1:nsize + 1, 1:nsize + 1] = torch.einsum("...nf,...mf->...nm", g, g) + \
            lmbda * torch.eye(nsize, dtype=dtype, device=device)
        # alpha: (batch_size, nsize)
        alpha = torch.linalg.solve(hmat[..., :nsize + 1, :nsize + 1],
                                   y[..., :nsize + 1, :])[..., 1:nsize + 1, 0]
        xnew = torch.einsum("...n,...nf->...f", alpha, fcollect[..., :nsize, :]) * beta + \
            torch.einsum("...n,...nf->...f", alpha, xcollect[..., :nsize, :]) * (1 - beta)
        fnew = _fcn(xnew)
        xcollect[..., k % msize, :] = xnew
        fcollect[..., k % msize, :] = fnew

        # check the stopping condition
        to_stop = stop_cond.check(xnew, fnew - xnew, xnew - xn)
        if verbose:
            if k < 10 or k % 10 == 0 or to_stop:
                print("%6d: |dx|=%.3e, |f-x|=%.3e" % (k, (xnew - xn).norm(),
                                                      (fnew - xnew).norm()))
        # update the xn
        xn = xnew
        if to_stop:
            converge = True
            break

    if not converge:
        msg = ("The rootfinder does not converge after %d iterations.") % (
            maxiter)
        warnings.warn(ConvergenceWarning(msg))

    return _unravel(xn)

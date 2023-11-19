import functools
import numpy as np
import torch
import warnings
from typing import Sequence, Tuple, Union, Any, Mapping, Optional, Callable
from deepchem.utils.differentiation_utils import LinearOperator, MatrixLinearOperator, assert_runtime, \
    set_default_option, dummy_context_manager, get_method, normalize_bcast_dims, get_bcasted_dims, \
    ConvergenceWarning, get_np_dtype
from scipy.sparse.linalg import gmres as scipy_gmres
from deepchem.utils.differentiation_utils.optimize.root.rootsolver import broyden1


def solve(A: LinearOperator, B: torch.Tensor, E: Union[torch.Tensor, None] = None,
          M: Optional[LinearOperator] = None,
          bck_options: Mapping[str, Any] = {},
          method: Union[str, Callable, None] = None,
          **fwd_options) -> torch.Tensor:
    r"""
    Performing iterative method to solve the equation

    .. math::

        \mathbf{AX=B}

    or

    .. math::

        \mathbf{AX-MXE=B}

    where :math:`\mathbf{E}` is a diagonal matrix.
    This function can also solve batched multiple inverse equation at the
    same time by applying :math:`\mathbf{A}` to a tensor :math:`\mathbf{X}`
    with shape ``(...,na,ncols)``.
    The applied :math:`\mathbf{E}` are not necessarily identical for each column.

    Arguments
    ---------
    A: xitorch.LinearOperator
        A linear operator that takes an input ``X`` and produce the vectors in the same
        space as ``B``.
        It should have the shape of ``(*BA, na, na)``
    B: torch.Tensor
        The tensor on the right hand side with shape ``(*BB, na, ncols)``
    E: torch.Tensor or None
        If a tensor, it will solve :math:`\mathbf{AX-MXE = B}`.
        It will be regarded as the diagonal of the matrix.
        Otherwise, it just solves :math:`\mathbf{AX = B}` and ``M`` is ignored.
        If it is a tensor, it should have shape of ``(*BE, ncols)``.
    M: xitorch.LinearOperator or None
        The transformation on the ``E`` side. If ``E`` is ``None``,
        then this argument is ignored.
        If E is not ``None`` and ``M`` is ``None``, then ``M=I``.
        If LinearOperator, it must be Hermitian with shape ``(*BM, na, na)``.
    bck_options: dict
        Options of the iterative solver in the backward calculation.
    method: str or callable or None
        The method of linear equation solver. If ``None``, it will choose
        ``"cg"`` or ``"bicgstab"`` based on the matrices symmetry.
        `Note`: default method will be changed quite frequently, so if you want
        future compatibility, please specify a method.
    **fwd_options
        Method-specific options (see method below)

    Returns
    -------
    torch.Tensor
        The tensor :math:`\mathbf{X}` that satisfies :math:`\mathbf{AX-MXE=B}`.
    """
    assert_runtime(A.shape[-1] == A.shape[-2], "The linear operator A must have a square shape")
    assert_runtime(A.shape[-1] == B.shape[-2], "Mismatch shape of A & B (A: %s, B: %s)" % (A.shape, B.shape))
    assert_runtime(
        not torch.is_grad_enabled() or A.is_getparamnames_implemented,
        "The _getparamnames(self, prefix) of linear operator A must be "
        "implemented if using solve with grad enabled")
    if M is not None:
        assert_runtime(M.shape[-1] == M.shape[-2], "The linear operator M must have a square shape")
        assert_runtime(M.shape[-1] == A.shape[-1], "The shape of A & M must match (A: %s, M: %s)" % (A.shape, M.shape))
        assert_runtime(M.is_hermitian, "The linear operator M must be a Hermitian matrix")
        assert_runtime(
            not torch.is_grad_enabled() or M.is_getparamnames_implemented,
            "The _getparamnames(self, prefix) of linear operator M must be "
            "implemented if using solve with grad enabled")
    if E is not None:
        assert_runtime(E.shape[-1] == B.shape[-1],
                       "The last dimension of E & B must match (E: %s, B: %s)" % (E.shape, B.shape))
    if E is None and M is not None:
        warnings.warn("M is supplied but will be ignored because E is not supplied")

    if method is None:
        if isinstance(A, MatrixLinearOperator) and \
           (M is None or isinstance(M, MatrixLinearOperator)):
            method = "exactsolve"
        elif A.shape[-1] <= 5:  # for small matrix
            method = "exactsolve"
        else:
            is_hermit = A.is_hermitian and (M is None or M.is_hermitian)
            method = "cg" if is_hermit else "bicgstab"

    if method == "exactsolve":
        return _exactsolve(A, B, E, M)
    else:
        # get the unique parameters of A
        params = A.getlinopparams()
        mparams = M.getlinopparams() if M is not None else []
        na = len(params)
        return solve_torchfcn.apply(
            A, B, E, M, method,
            fwd_options, bck_options,
            na, *params, *mparams)

class solve_torchfcn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, E, M, method,
                fwd_options, bck_options,
                na, *all_params):
        # A: (*BA, nr, nr)
        # B: (*BB, nr, ncols)
        # E: (*BE, ncols) or None
        # M: (*BM, nr, nr) or None
        # all_params: list of tensor of any shape
        # returns: (*BABEM, nr, ncols)

        # separate the parameters for A and for M
        params = all_params[:na]
        mparams = all_params[na:]

        config = set_default_option({
        }, fwd_options)
        ctx.bck_config = set_default_option({
        }, bck_options)

        if torch.all(B == 0):  # special case
            dims = (*_get_batchdims(A, B, E, M), *B.shape[-2:])
            x = torch.zeros(dims, dtype=B.dtype, device=B.device)
        else:
            with A.uselinopparams(*params), M.uselinopparams(*mparams) if M is not None else dummy_context_manager():
                methods = {
                    "custom_exactsolve": custom_exactsolve,
                    "scipy_gmres": _wrap_gmres,
                    "broyden1": _broyden1_solve,
                    "cg": _cg,
                    "bicgstab": _bicgstab,
                    "gmres": _gmres,
                }
                method_fcn = get_method("solve", methods, method)
                x = method_fcn(A, B, E, M, **config)

        ctx.e_is_none = E is None
        ctx.A = A
        ctx.M = M
        if ctx.e_is_none:
            ctx.save_for_backward(x, *all_params)
        else:
            ctx.save_for_backward(x, E, *all_params)
        ctx.na = na
        return x

    @staticmethod
    def backward(ctx, grad_x):
        # grad_x: (*BABEM, nr, ncols)
        # x: (*BABEM, nr, ncols)
        x = ctx.saved_tensors[0]
        idx_all_params = 1 if ctx.e_is_none else 2
        all_params = ctx.saved_tensors[idx_all_params:]
        params = all_params[:ctx.na]
        mparams = all_params[ctx.na:]
        E = None if ctx.e_is_none else ctx.saved_tensors[1]

        # solve (A-biases*M)^T v = grad_x
        # this is the grad of B
        with ctx.A.uselinopparams(*params), \
             ctx.M.uselinopparams(*mparams) if ctx.M is not None else dummy_context_manager():
            AT = ctx.A.H  # (*BA, nr, nr)
            MT = ctx.M.H if ctx.M is not None else None  # (*BM, nr, nr)
            Econj = E.conj() if E is not None else None
            v = solve(AT, grad_x, Econj, MT,
                      bck_options=ctx.bck_config, **ctx.bck_config)  # (*BABEM, nr, ncols)
        grad_B = v

        # calculate the grad of matrices parameters
        with torch.enable_grad():
            params = [p.clone().requires_grad_() for p in params]
            with ctx.A.uselinopparams(*params):
                loss = -ctx.A.mm(x)  # (*BABEM, nr, ncols)

        grad_params = torch.autograd.grad((loss,), params, grad_outputs=(v,),
                                          create_graph=torch.is_grad_enabled(),
                                          allow_unused=True)

        # calculate the biases gradient
        grad_E = None
        if E is not None:
            if ctx.M is None:
                Mx = x
            else:
                with ctx.M.uselinopparams(*mparams):
                    Mx = ctx.M.mm(x)  # (*BABEM, nr, ncols)
            grad_E = torch.einsum('...rc,...rc->...c', v, Mx.conj())  # (*BABEM, ncols)

        # calculate the gradient to the biases matrices
        grad_mparams = []
        if ctx.M is not None and E is not None:
            with torch.enable_grad():
                mparams = [p.clone().requires_grad_() for p in mparams]
                lmbdax = x * E.unsqueeze(-2)
                with ctx.M.uselinopparams(*mparams):
                    mloss = ctx.M.mm(lmbdax)

            grad_mparams = torch.autograd.grad((mloss,), mparams,
                                               grad_outputs=(v,),
                                               create_graph=torch.is_grad_enabled(),
                                               allow_unused=True)

        return (None, grad_B, grad_E, None, None, None, None, None,
                *grad_params, *grad_mparams)

def custom_exactsolve(A, B, E=None,
                      M=None, **options):
    # A: (*BA, na, na)
    # B: (*BB, na, ncols)
    # E: (*BE, ncols)
    # M: (*BM, na, na)
    return _exactsolve(A, B, E, M)

# Hidden
def _wrap_gmres(A, B, E=None, M=None,
               min_eps=1e-9,
               max_niter=None,
               **unused):
    """
    Using SciPy's gmres method to solve the linear equation.

    Keyword arguments
    -----------------
    min_eps: float
        Relative tolerance for stopping conditions
    max_niter: int or None
        Maximum number of iterations. If ``None``, default to twice of the
        number of columns of ``A``.
    """
    # A: (*BA, nr, nr)
    # B: (*BB, nr, ncols)
    # E: (*BE, ncols) or None
    # M: (*BM, nr, nr) or None

    # NOTE: currently only works for batched B (1 batch dim), but unbatched A
    assert len(A.shape) == 2 and len(B.shape) == 3, "Currently only works for batched B (1 batch dim), but unbatched A"
    assert not torch.is_complex(B), "complex is not supported in gmres"

    # check the parameters
    msg = "GMRES can only do AX=B"
    assert A.shape[-2] == A.shape[-1], "GMRES can only work for square operator for now"
    assert E is None, msg
    assert M is None, msg

    nbatch, na, ncols = B.shape
    if max_niter is None:
        max_niter = 2 * na

    B = B.transpose(-1, -2)  # (nbatch, ncols, na)

    # convert the numpy/scipy
    op = A.scipy_linalg_op()
    B_np = B.detach().cpu().numpy()
    res_np = np.empty(B.shape, dtype=get_np_dtype(B.dtype))
    for i in range(nbatch):
        for j in range(ncols):
            x, info = scipy_gmres(op, B_np[i, j, :], tol=min_eps, atol=1e-12, maxiter=max_niter)
            if info > 0:
                msg = "The GMRES iteration does not converge to the desired value "\
                      "(%.3e) after %d iterations" % \
                      (min_eps, info)
                warnings.warn(ConvergenceWarning(msg))
            res_np[i, j, :] = x

    res = torch.tensor(res_np, dtype=B.dtype, device=B.device)
    res = res.transpose(-1, -2)  # (nbatch, na, ncols)
    return res

def _exactsolve(A: LinearOperator, B: torch.Tensor,
               E: Union[torch.Tensor, None],
               M: Union[LinearOperator, None]):
    """
    Solve the linear equation by contructing the full matrix of LinearOperators.

    Warnings
    --------
    * As this method construct the linear operators explicitly, it might requires
      a large memory.
    """
    # A: (*BA, na, na)
    # B: (*BB, na, ncols)
    # E: (*BE, ncols)
    # M: (*BM, na, na)
    if E is None:
        Amatrix = A.fullmatrix()  # (*BA, na, na)
        x = torch.linalg.solve(Amatrix, B)  # (*BAB, na, ncols)
    elif M is None:
        Amatrix = A.fullmatrix()
        x = _solve_ABE(Amatrix, B, E)
    else:
        Mmatrix = M.fullmatrix()  # (*BM, na, na)
        L = torch.linalg.cholesky(Mmatrix)  # (*BM, na, na)
        Linv = torch.inverse(L)  # (*BM, na, na)
        LinvT = Linv.transpose(-2, -1).conj()  # (*BM, na, na)
        A2 = torch.matmul(Linv, A.mm(LinvT))  # (*BAM, na, na)
        B2 = torch.matmul(Linv, B)  # (*BBM, na, ncols)

        X2 = _solve_ABE(A2, B2, E)  # (*BABEM, na, ncols)
        x = torch.matmul(LinvT, X2)  # (*BABEM, na, ncols)
    return x

def _solve_ABE(A: torch.Tensor, B: torch.Tensor, E: torch.Tensor):
    # A: (*BA, na, na) matrix
    # B: (*BB, na, ncols) matrix
    # E: (*BE, ncols) matrix
    na = A.shape[-1]
    BA, BB, BE = normalize_bcast_dims(A.shape[:-2], B.shape[:-2], E.shape[:-1])
    E = E.reshape(1, *BE, E.shape[-1]).transpose(0, -1)  # (ncols, *BE, 1)
    B = B.reshape(1, *BB, *B.shape[-2:]).transpose(0, -1)  # (ncols, *BB, na, 1)

    # NOTE: The line below is very inefficient for large na and ncols
    AE = A - torch.diag_embed(E.repeat_interleave(repeats=na, dim=-1), dim1=-2, dim2=-1)  # (ncols, *BAE, na, na)
    r = torch.linalg.solve(AE, B)  # (ncols, *BAEM, na, 1)
    r = r.transpose(0, -1).squeeze(0)  # (*BAEM, na, ncols)
    return r


def _cg(A: LinearOperator, B: torch.Tensor,
        E: Optional[torch.Tensor] = None,
        M: Optional[LinearOperator] = None,
        posdef: Optional[bool] = None,
        precond: Optional[LinearOperator] = None,
        max_niter: Optional[int] = None,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        eps: float = 1e-12,
        resid_calc_every: int = 10,
        verbose: bool = False,
        **unused) -> torch.Tensor:
    r"""
    Solve the linear equations using Conjugate-Gradient (CG) method.

    Keyword arguments
    -----------------
    posdef: bool or None
        Indicating if the operation :math:`\mathbf{AX-MXE}` a positive
        definite for all columns and batches.
        If None, it will be determined by power iterations.
    precond: LinearOperator or None
        LinearOperator for the preconditioning. If None, no preconditioner is
        applied.
    max_niter: int or None
        Maximum number of iteration. If None, it is set to ``int(1.5 * A.shape[-1])``
    rtol: float
        Relative tolerance for stopping condition w.r.t. norm of B
    atol: float
        Absolute tolerance for stopping condition w.r.t. norm of B
    eps: float
        Substitute the absolute zero in the algorithm's denominator with this
        value to avoid nan.
    resid_calc_every: int
        Calculate the residual in its actual form instead of substitution form
        with this frequency, to avoid rounding error accummulation.
        If your linear operator has bad numerical precision, set this to be low.
        If 0, then never calculate the residual in its actual form.
    verbose: bool
        Verbosity of the algorithm.
    """
    nr = A.shape[-1]
    ncols = B.shape[-1]
    if max_niter is None:
        max_niter = int(1.5 * nr)

    # if B is all zeros, then return zeros
    batchdims = _get_batchdims(A, B, E, M)
    if torch.allclose(B, B * 0, rtol=rtol, atol=atol):
        x0 = torch.zeros((*batchdims, nr, ncols), dtype=A.dtype, device=A.device)
        return x0

    # setup the preconditioning and the matrix problem
    precond_fcn = _setup_precond(precond)
    need_hermit = True
    A_fcn, _, B2, col_swapped = _setup_linear_problem(A, B, E, M, batchdims,
                                                      posdef, need_hermit)

    # get the stopping matrix
    B_norm = B2.norm(dim=-2, keepdim=True)  # (*BB, 1, nc)
    stop_matrix = torch.max(rtol * B_norm, atol * torch.ones_like(B_norm))  # (*BB, 1, nc)

    # prepare the initial guess (it's just all zeros)
    x0shape = (ncols, *batchdims, nr, 1) if col_swapped else (*batchdims, nr, ncols)
    xk = torch.zeros(x0shape, dtype=A.dtype, device=A.device)

    rk = B2 - A_fcn(xk)  # (*, nr, nc)
    zk = precond_fcn(rk)  # (*, nr, nc)
    pk = zk  # (*, nr, nc)
    rkzk = _dot(rk, zk)
    converge = False
    best_resid = rk.norm(dim=-2).max().item()
    best_xk = xk
    for k in range(1, max_niter + 1):
        Apk = A_fcn(pk)
        alphak = rkzk / _safedenom(_dot(pk, Apk), eps)
        xk_1 = xk + alphak * pk

        # correct the residual calculation
        if resid_calc_every != 0 and k % resid_calc_every == 0:
            rk_1 = B2 - A_fcn(xk_1)
        else:
            rk_1 = rk - alphak * Apk  # (*, nr, nc)

        # check for the stopping condition
        resid = rk_1  # B2 - A_fcn(xk_1)
        resid_norm = resid.norm(dim=-2, keepdim=True)

        max_resid_norm = resid_norm.max().item()
        if max_resid_norm < best_resid:
            best_resid = max_resid_norm
            best_xk = xk_1

        if verbose:
            if k < 10 or k % 10 == 0:
                print("%4d: |dy|=%.3e" % (k, resid_norm))

        if torch.all(resid_norm < stop_matrix):
            converge = True
            break

        zk_1 = precond_fcn(rk_1)
        rkzk_1 = _dot(rk_1, zk_1)
        betak = rkzk_1 / _safedenom(rkzk, eps)
        pk_1 = zk_1 + betak * pk

        # move to the next index
        pk = pk_1
        zk = zk_1
        xk = xk_1
        rk = rk_1
        rkzk = rkzk_1

    xk_1 = best_xk
    if not converge:
        msg = ("Convergence is not achieved after %d iterations. "
               "Max norm of best resid: %.3e") % (max_niter, best_resid)
        warnings.warn(ConvergenceWarning(msg))
    if col_swapped:
        # x: (ncols, *, nr, 1)
        xk_1 = xk_1.transpose(0, -1).squeeze(0)  # (*, nr, ncols)
    return xk_1


def _bicgstab(A: LinearOperator, B: torch.Tensor,
             E: Optional[torch.Tensor] = None,
             M: Optional[LinearOperator] = None,
             posdef: Optional[bool] = None,
             precond_l: Optional[LinearOperator] = None,
             precond_r: Optional[LinearOperator] = None,
             max_niter: Optional[int] = None,
             rtol: float = 1e-6,
             atol: float = 1e-8,
             eps: float = 1e-12,
             verbose: bool = False,
             resid_calc_every: int = 10,
             **unused) -> torch.Tensor:
    r"""
    Solve the linear equations using stabilized Biconjugate-Gradient method.

    Keyword arguments
    -----------------
    posdef: bool or None
        Indicating if the operation :math:`\mathbf{AX-MXE}` a positive
        definite for all columns and batches.
        If None, it will be determined by power iterations.
    precond_l: LinearOperator or None
        LinearOperator for the left preconditioning. If None, no
        preconditioner is applied.
    precond_r: LinearOperator or None
        LinearOperator for the right preconditioning. If None, no
        preconditioner is applied.
    max_niter: int or None
        Maximum number of iteration. If None, it is set to ``int(1.5 * A.shape[-1])``
    rtol: float
        Relative tolerance for stopping condition w.r.t. norm of B
    atol: float
        Absolute tolerance for stopping condition w.r.t. norm of B
    eps: float
        Substitute the absolute zero in the algorithm's denominator with this
        value to avoid nan.
    resid_calc_every: int
        Calculate the residual in its actual form instead of substitution form
        with this frequency, to avoid rounding error accummulation.
        If your linear operator has bad numerical precision, set this to be low.
        If 0, then never calculate the residual in its actual form.
    verbose: bool
        Verbosity of the algorithm.
    """
    nr, ncols = B.shape[-2:]
    if max_niter is None:
        max_niter = int(1.5 * nr)

    # if B is all zeros, then return zeros
    batchdims = _get_batchdims(A, B, E, M)
    if torch.allclose(B, B * 0, rtol=rtol, atol=atol):
        x0 = torch.zeros((*batchdims, nr, ncols), dtype=A.dtype, device=A.device)
        return x0

    # setup the preconditioning and the matrix problem
    precond_fcn_l = _setup_precond(precond_l)
    precond_fcn_r = _setup_precond(precond_r)
    need_hermit = False
    A_fcn, AT_fcn, B2, col_swapped = _setup_linear_problem(A, B, E, M, batchdims,
                                                           posdef, need_hermit)

    # get the stopping matrix
    B_norm = B2.norm(dim=-2, keepdim=True)  # (*BB, 1, nc)
    stop_matrix = torch.max(rtol * B_norm, atol * torch.ones_like(B_norm))  # (*BB, 1, nc)

    # prepare the initial guess (it's just all zeros)
    x0shape = (ncols, *batchdims, nr, 1) if col_swapped else (*batchdims, nr, ncols)
    xk = torch.zeros(x0shape, dtype=A.dtype, device=A.device)

    rk = B2 - A_fcn(xk)
    r0hat = rk
    rho_k = _dot(r0hat, rk)
    omega_k = torch.tensor(1.0, dtype=A.dtype, device=A.device)
    alpha: Union[float, torch.Tensor] = 1.0
    vk: Union[float, torch.Tensor] = 0.0
    pk: Union[float, torch.Tensor] = 0.0
    converge = False
    best_resid = rk.norm(dim=-2).max()
    best_xk = xk
    for k in range(1, max_niter + 1):
        rho_knew = _dot(r0hat, rk)
        omega_denom = _safedenom(omega_k, eps)
        beta = rho_knew / _safedenom(rho_k, eps) * (alpha / omega_denom)
        pk = rk + beta * (pk - omega_k * vk)
        y = precond_fcn_r(pk)
        vk = A_fcn(y)
        alpha = rho_knew / _safedenom(_dot(r0hat, vk), eps)
        h = xk + alpha * y

        s = rk - alpha * vk
        z = precond_fcn_r(s)
        t = A_fcn(z)
        Kt = precond_fcn_l(t)
        omega_k = _dot(Kt, precond_fcn_l(s)) / _safedenom(_dot(Kt, Kt), eps)
        xk = h + omega_k * z

        # correct the residual calculation regularly
        if resid_calc_every != 0 and k % resid_calc_every == 0:
            rk = B2 - A_fcn(xk)
        else:
            rk = s - omega_k * t

        # calculate the residual
        resid = rk
        resid_norm = resid.norm(dim=-2, keepdim=True)

        # save the best results
        max_resid_norm = resid_norm.max().item()
        if max_resid_norm < best_resid:
            best_resid = max_resid_norm
            best_xk = xk

        if verbose:
            if k < 10 or k % 10 == 0:
                print("%4d: |dy|=%.3e" % (k, resid_norm))

        # check for the stopping conditions
        if torch.all(resid_norm < stop_matrix):
            converge = True
            break

        rho_k = rho_knew

    xk = best_xk
    if not converge:
        msg = ("Convergence is not achieved after %d iterations. "
               "Max norm of resid: %.3e") % (max_niter, best_resid)
        warnings.warn(ConvergenceWarning(msg))
    if col_swapped:
        # x: (ncols, *, nr, 1)
        xk = xk.transpose(0, -1).squeeze(0)  # (*, nr, ncols)
    return xk


def _gmres(A: LinearOperator, B: torch.Tensor,
          E: Optional[torch.Tensor] = None,
          M: Optional[LinearOperator] = None,
          posdef: Optional[bool] = None,
          max_niter: Optional[int] = None,
          rtol: float = 1e-6,
          atol: float = 1e-8,
          eps: float = 1e-12,
          **unused) -> torch.Tensor:
    r"""
    Solve the linear equations using Generalised minial residual method.

    Keyword arguments
    -----------------
    posdef: bool or None
        Indicating if the operation :math:`\mathbf{AX-MXE}` a positive
        definite for all columns and batches.
        If None, it will be determined by power iterations.
    max_niter: int or None
        Maximum number of iteration. If None, it is set to ``int(1.5 * A.shape[-1])``
    rtol: float
        Relative tolerance for stopping condition w.r.t. norm of B
    atol: float
        Absolute tolerance for stopping condition w.r.t. norm of B
    eps: float
        Substitute the absolute zero in the algorithm's denominator with this
        value to avoid nan.
    """
    converge = False

    nr, ncols = A.shape[-1], B.shape[-1]
    if max_niter is None:
        max_niter = int(nr)

    # if B is all zeros, then return zeros
    batchdims = _get_batchdims(A, B, E, M)
    if torch.allclose(B, B * 0, rtol=rtol, atol=atol):
        x0 = torch.zeros((*batchdims, nr, ncols), dtype=A.dtype, device=A.device)
        return x0

    # setup the preconditioning and the matrix problem
    need_hermit = False
    A_fcn, AT_fcn, B2, col_swapped = _setup_linear_problem(A, B, E, M, batchdims,
                                                           posdef, need_hermit)

    # get the stopping matrix
    B_norm = B2.norm(dim=-2, keepdim=True)  # (*BB, 1, nc)
    stop_matrix = torch.max(rtol * B_norm, atol * torch.ones_like(B_norm))  # (*BB, 1, nc)

    # prepare the initial guess (it's just all zeros)
    x0shape = (ncols, *batchdims, nr, 1) if col_swapped else (*batchdims, nr, ncols)
    x0 = torch.zeros(x0shape, dtype=A.dtype, device=A.device)

    r = B2 - A_fcn(x0)  # torch.Size([*batch_dims, nr, ncols])
    best_resid = r.norm(dim=-2, keepdim=True)  # / B_norm

    best_resid = best_resid.max().item()
    best_res = x0
    q = torch.empty([max_niter] + list(r.shape), dtype=A.dtype, device=A.device)
    q[0] = r / _safedenom(r.norm(dim=-2, keepdim=True), eps)  # torch.Size([*batch_dims, nr, ncols])
    h = torch.zeros((*batchdims, ncols, max_niter + 1, max_niter), dtype=A.dtype, device=A.device)
    h = h.reshape((-1, ncols, max_niter + 1, max_niter))

    for k in range(min(nr, max_niter)):
        y = A_fcn(q[k])  # torch.Size([*batch_dims, nr, ncols])
        for j in range(k + 1):
            h[..., j, k] = _dot(q[j], y).reshape(-1, ncols)
            y = y - h[..., j, k].reshape(*batchdims, 1, ncols) * q[j]

        h[..., k + 1, k] = torch.linalg.norm(y, dim=-2)
        if torch.any(h[..., k + 1, k]) != 0 and k != max_niter - 1:
            q[k + 1] = y.reshape(-1, nr, ncols) / h[..., k + 1, k].reshape(-1, 1, ncols)
            q[k + 1] = q[k + 1].reshape(*batchdims, nr, ncols)

        b = torch.zeros((*batchdims, ncols, k + 1), dtype=A.dtype, device=A.device)
        b = b.reshape(-1, ncols, k + 1)
        b[..., 0] = torch.linalg.norm(r, dim=-2)
        rk = torch.linalg.lstsq(h[..., :k + 1, :k], b)[0]  # torch.Size([*batch_dims, max_niter])
        # Q, R = torch.linalg.qr(h[:, :k+1, :k], mode='complete')
        # result = torch.triangular_solve(torch.matmul(Q.permute(0, 2, 1), b[:, :, None])[:, :-1], R[:, :-1, :])[0]

        res = torch.empty([])
        for i in range(k):
            res = res + q[i] * rk[..., i].reshape(*batchdims, 1, ncols) + x0 if res.size() \
                else q[i] * rk[..., i].reshape(*batchdims, 1, ncols) + x0
            # res = res * B_norm

        if res.size():
            resid = B2 - A_fcn(res)
            resid_norm = resid.norm(dim=-2, keepdim=True)

            # save the best results
            max_resid_norm = resid_norm.max().item()
            if max_resid_norm < best_resid:
                best_resid = max_resid_norm
                best_res = res

            if torch.all(resid_norm < stop_matrix):
                converge = True
                break

    if not converge:
        msg = ("Convergence is not achieved after %d iterations. "
               "Max norm of resid: %.3e") % (max_niter, best_resid)
        warnings.warn(ConvergenceWarning(msg))

    res = best_res
    return res


############ general helpers ############
def _get_batchdims(A: LinearOperator, B: torch.Tensor,
                   E: Union[torch.Tensor, None],
                   M: Union[LinearOperator, None]):

    batchdims = [A.shape[:-2], B.shape[:-2]]
    if E is not None:
        batchdims.append(E.shape[:-1])
        if M is not None:
            batchdims.append(M.shape[:-2])
    return get_bcasted_dims(*batchdims)

def _setup_precond(precond: Optional[LinearOperator]) -> Callable[[torch.Tensor], torch.Tensor]:
    if isinstance(precond, LinearOperator):
        precond_fcn = lambda x: precond.mm(x)
    elif precond is None:
        precond_fcn = lambda x: x
    else:
        raise TypeError("precond can only be LinearOperator or None")
    return precond_fcn


def _setup_linear_problem(A: LinearOperator, B: torch.Tensor,
                          E: Optional[torch.Tensor], M: Optional[LinearOperator],
                          batchdims: Sequence[int],
                          posdef: Optional[bool],
                          need_hermit: bool) -> \
        Tuple[Callable[[torch.Tensor], torch.Tensor],
              Callable[[torch.Tensor], torch.Tensor],
              torch.Tensor, bool]:

    # get the linear operator (including the MXE part)
    if E is None:
        A_fcn = lambda x: A.mm(x)
        AT_fcn = lambda x: A.rmm(x)
        B_new = B
        col_swapped = False
    else:
        # A: (*BA, nr, nr) linop
        # B: (*BB, nr, ncols)
        # E: (*BE, ncols)
        # M: (*BM, nr, nr) linop
        if M is None:
            BAs, BBs, BEs = normalize_bcast_dims(A.shape[:-2], B.shape[:-2], E.shape[:-1])
        else:
            BAs, BBs, BEs, BMs = normalize_bcast_dims(A.shape[:-2], B.shape[:-2],
                                                      E.shape[:-1], M.shape[:-2])
        E = E.reshape(*BEs, *E.shape[-1:])
        E_new = E.unsqueeze(0).transpose(-1, 0).unsqueeze(-1)  # (ncols, *BEs, 1, 1)
        B = B.reshape(*BBs, *B.shape[-2:])  # (*BBs, nr, ncols)
        B_new = B.unsqueeze(0).transpose(-1, 0)  # (ncols, *BBs, nr, 1)

        def A_fcn(x):
            # x: (ncols, *BX, nr, 1)
            Ax = A.mm(x)  # (ncols, *BAX, nr, 1)
            Mx = M.mm(x) if M is not None else x  # (ncols, *BMX, nr, 1)
            MxE = Mx * E_new  # (ncols, *BMXE, nr, 1)
            return Ax - MxE

        def AT_fcn(x):
            # x: (ncols, *BX, nr, 1)
            ATx = A.rmm(x)
            MTx = M.rmm(x) if M is not None else x
            MTxE = MTx * E_new
            return ATx - MTxE

        col_swapped = True

    # estimate if it's posdef with power iteration
    if need_hermit:
        is_hermit = A.is_hermitian and (M is None or M.is_hermitian)
        if not is_hermit:
            # set posdef to False to make the operator becomes AT * A so it is
            # hermitian
            posdef = False

    # TODO: the posdef check by largest eival only works for Hermitian/symmetric
    # matrix, but it doesn't always work for non-symmetric matrix.
    # In non-symmetric case, one need to do Cholesky LDL decomposition
    if posdef is None:
        nr, ncols = B.shape[-2:]
        x0shape = (ncols, *batchdims, nr, 1) if col_swapped else (*batchdims, nr, ncols)
        x0 = torch.randn(x0shape, dtype=A.dtype, device=A.device)
        x0 = x0 / x0.norm(dim=-2, keepdim=True)
        largest_eival = _get_largest_eival(A_fcn, x0)  # (*, 1, nc)
        negeival = largest_eival <= 0

        # if the largest eigenvalue is negative, then it's not posdef
        if torch.all(negeival):
            posdef = False

        # otherwise, calculate the lowest eigenvalue to check if it's positive
        else:
            offset = torch.clamp(largest_eival, min=0.0)
            A_fcn2 = lambda x: A_fcn(x) - offset * x
            mostneg_eival = _get_largest_eival(A_fcn2, x0)  # (*, 1, nc)
            posdef = bool(torch.all(torch.logical_or(-mostneg_eival <= offset, negeival)).item())

    # get the linear operation if it is not a posdef (A -> AT.A)
    if posdef:
        return A_fcn, AT_fcn, B_new, col_swapped
    else:
        def A_new_fcn(x):
            return AT_fcn(A_fcn(x))
        B2 = AT_fcn(B_new)
        return A_new_fcn, A_new_fcn, B2, col_swapped

############ cg and bicgstab helpers ############
def _safedenom(r: torch.Tensor, eps: float) -> torch.Tensor:
    r[r == 0] = eps
    return r

def _dot(r: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    # r: (*BR, nr, nc)
    # z: (*BR, nr, nc)
    # return: (*BR, 1, nc)
    return torch.einsum("...rc,...rc->...c", r.conj(), z).unsqueeze(-2)

############ rootfinder-based ############
@functools.wraps(broyden1)
def _broyden1_solve(A, B, E=None, M=None, **options):
    return _rootfinder_solve("broyden1", A, B, E, M, **options)

def _rootfinder_solve(alg, A, B, E=None, M=None, **options):
    # using rootfinder algorithm
    nr = A.shape[-1]
    ncols = B.shape[-1]

    # set up the function for the rootfinding
    def fcn_rootfinder(xi):
        # xi: (*BX, nr*ncols)
        x = xi.reshape(*xi.shape[:-1], nr, ncols)  # (*BX, nr, ncols)
        y = A.mm(x) - B  # (*BX, nr, ncols)
        if E is not None:
            MX = M.mm(x) if M is not None else x
            MXE = MX * E.unsqueeze(-2)
            y = y - MXE  # (*BX, nr, ncols)
        y = y.reshape(*xi.shape[:-1], -1)  # (*BX, nr*ncols)
        return y

    # setup the initial guess (the batch dimension must be the largest)
    batchdims = _get_batchdims(A, B, E, M)
    x0 = torch.zeros((*batchdims, nr * ncols), dtype=A.dtype, device=A.device)

    if alg == "broyden1":
        x = broyden1(fcn_rootfinder, x0, **options)
    else:
        raise RuntimeError("Unknown method %s" % alg)
    x = x.reshape(*x.shape[:-1], nr, ncols)
    return x

def _get_largest_eival(Afcn, x):
    niter = 10
    rtol = 1e-3
    atol = 1e-6
    xnorm_prev = None
    for i in range(niter):
        x = Afcn(x)  # (*, nr, nc)
        xnorm = x.norm(dim=-2, keepdim=True)  # (*, 1, nc)

        # check if xnorm is converging
        if i > 0:
            dnorm = torch.abs(xnorm_prev - xnorm)
            if torch.all(dnorm <= rtol * xnorm + atol):
                break

        xnorm_prev = xnorm
        if i < niter - 1:
            x = x / xnorm
    return xnorm

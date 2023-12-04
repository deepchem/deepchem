import functools
from deepchem.utils.differentiation_utils import LinearOperator, MatrixLinearOperator, \
    get_and_pop_keys, dummy_context_manager, get_method, get_bcasted_dims, tallqr, \
    to_fortran_order, assert_runtime
from typing import Optional, Sequence, Tuple, Union, Callable, Any, Mapping
import torch
from deepchem.utils.misc_utils import set_default_option
from deepchem.utils.differentiation_utils.solve import solve
import warnings

def lsymeig(A: LinearOperator, neig: Optional[int] = None,
            M: Optional[LinearOperator] = None,
            bck_options: Mapping[str, Any] = {},
            method: Union[str, Callable, None] = None,
            **fwd_options) -> Tuple[torch.Tensor, torch.Tensor]:
    return symeig(A, neig, "lowest", M, method=method, bck_options=bck_options, **fwd_options)

def usymeig(A: LinearOperator, neig: Optional[int] = None,
            M: Optional[LinearOperator] = None,
            bck_options: Mapping[str, Any] = {},
            method: Union[str, Callable, None] = None,
            **fwd_options) -> Tuple[torch.Tensor, torch.Tensor]:
    return symeig(A, neig, "uppest", M, method=method, bck_options=bck_options, **fwd_options)

def symeig(A: LinearOperator, neig: Optional[int] = None,
           mode: str = "lowest", M: Optional[LinearOperator] = None,
           bck_options: Mapping[str, Any] = {},
           method: Union[str, Callable, None] = None,
           **fwd_options) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Obtain ``neig`` lowest eigenvalues and eigenvectors of a linear operator,

    .. math::

        \mathbf{AX = MXE}

    where :math:`\mathbf{A}, \mathbf{M}` are linear operators,
    :math:`\mathbf{E}` is a diagonal matrix containing the eigenvalues, and
    :math:`\mathbf{X}` is a matrix containing the eigenvectors.
    This function can handle derivatives for degenerate cases by setting non-zero
    ``degen_atol`` and ``degen_rtol`` in the backward option using the expressions
    in [1]_.

    Arguments
    ---------
    A: xitorch.LinearOperator
        The linear operator object on which the eigenpairs are constructed.
        It must be a Hermitian linear operator with shape ``(*BA, q, q)``
    neig: int or None
        The number of eigenpairs to be retrieved. If ``None``, all eigenpairs are
        retrieved
    mode: str
        ``"lowest"`` or ``"uppermost"``/``"uppest"``. If ``"lowest"``,
        it will take the lowest ``neig`` eigenpairs.
        If ``"uppest"``, it will take the uppermost ``neig``.
    M: xitorch.LinearOperator
        The transformation on the right hand side. If ``None``, then ``M=I``.
        If specified, it must be a Hermitian with shape ``(*BM, q, q)``.
    bck_options: dict
        Method-specific options for :func:`solve` which used in backpropagation
        calculation with some additional arguments for computing the backward
        derivatives:

        * ``degen_atol`` (``float`` or None): Minimum absolute difference between
          two eigenvalues to be treated as degenerate. If None, it is
          ``torch.finfo(dtype).eps**0.6``. If 0.0, no special treatment on
          degeneracy is applied. (default: None)
        * ``degen_rtol`` (``float`` or None): Minimum relative difference between
          two eigenvalues to be treated as degenerate. If None, it is
          ``torch.finfo(dtype).eps**0.4``. If 0.0, no special treatment on
          degeneracy is applied. (default: None)

        Note: the default values of ``degen_atol`` and ``degen_rtol`` are going
        to change in the future. So, for future compatibility, please specify
        the specific values.

    method: str or callable or None
        Method for the eigendecomposition. If ``None``, it will choose
        ``"exacteig"``.
    **fwd_options
        Method-specific options (see method section below).

    Returns
    -------
    tuple of tensors (eigenvalues, eigenvectors)
        It will return eigenvalues and eigenvectors with shapes respectively
        ``(*BAM, neig)`` and ``(*BAM, na, neig)``, where ``*BAM`` is the
        broadcasted shape of ``*BA`` and ``*BM``.

    References
    ----------
    .. [1] Muhammad F. Kasim,
           "Derivatives of partial eigendecomposition of a real symmetric matrix for degenerate cases".
           arXiv:2011.04366 (2020)
           `https://arxiv.org/abs/2011.04366 <https://arxiv.org/abs/2011.04366>`_
    """
    assert_runtime(A.is_hermitian, "The linear operator A must be Hermitian")
    assert_runtime(
        not torch.is_grad_enabled() or A.is_getparamnames_implemented,
        "The _getparamnames(self, prefix) of linear operator A must be "
        "implemented if using symeig with grad enabled")
    if M is not None:
        assert_runtime(M.is_hermitian, "The linear operator M must be Hermitian")
        assert_runtime(M.shape[-1] == A.shape[-1], "The shape of A & M must match (A: %s, M: %s)" % (A.shape, M.shape))
        assert_runtime(
            not torch.is_grad_enabled() or M.is_getparamnames_implemented,
            "The _getparamnames(self, prefix) of linear operator M must be "
            "implemented if using symeig with grad enabled")
    mode = mode.lower()
    if mode == "uppermost":
        mode = "uppest"
    if method is None:
        if isinstance(A, MatrixLinearOperator) and \
           (M is None or isinstance(M, MatrixLinearOperator)):
            method = "exacteig"
        else:
            # TODO: implement robust LOBPCG and put it here
            method = "exacteig"
    if neig is None:
        neig = A.shape[-1]

    if method == "exacteig":
        return _exacteig(A, neig, mode, M)
    else:
        fwd_options["method"] = method
        # get the unique parameters of A & M
        params = A.getlinopparams()
        mparams = M.getlinopparams() if M is not None else []
        na = len(params)
        return symeig_torchfcn.apply(A, neig, mode, M,
                                     fwd_options, bck_options,
                                     na, *params, *mparams)

class symeig_torchfcn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, neig, mode, M, fwd_options, bck_options, na, *amparams):
        # A: LinearOperator (*BA, q, q)
        # M: LinearOperator (*BM, q, q) or None

        # separate the sets of parameters
        params = amparams[:na]
        mparams = amparams[na:]

        config = set_default_option({
        }, fwd_options)
        ctx.bck_config = set_default_option({
            "degen_atol": None,
            "degen_rtol": None,
        }, bck_options)

        # options for calculating the backward (not for `solve`)
        alg_keys = ["degen_atol", "degen_rtol"]
        ctx.bck_alg_config = get_and_pop_keys(ctx.bck_config, alg_keys)

        method = config.pop("method")
        with A.uselinopparams(*params), M.uselinopparams(*mparams) if M is not None else dummy_context_manager():
            methods = {
                "davidson": _davidson,
                "custom_exacteig": custom_exacteig,
            }
            method_fcn = get_method("symeig", methods, method)
            evals, evecs = method_fcn(A, neig, mode, M, **config)

        # save for the backward
        # evals: (*BAM, neig)
        # evecs: (*BAM, na, neig)
        ctx.save_for_backward(evals, evecs, *amparams)
        ctx.na = na
        ctx.A = A
        ctx.M = M
        return evals, evecs

    @staticmethod
    def backward(ctx, grad_evals, grad_evecs):
        # grad_evals: (*BAM, neig)
        # grad_evecs: (*BAM, na, neig)

        # get the variables from ctx
        evals, evecs = ctx.saved_tensors[:2]
        na = ctx.na
        amparams = ctx.saved_tensors[2:]
        params = amparams[:na]
        mparams = amparams[na:]

        M = ctx.M
        A = ctx.A
        degen_atol: Optional[float] = ctx.bck_alg_config["degen_atol"]  # type: ignore
        degen_rtol: Optional[float] = ctx.bck_alg_config["degen_rtol"]  # type: ignore

        # set the default values of degen_*tol
        dtype = evals.dtype
        if degen_atol is None:
            degen_atol = torch.finfo(dtype).eps**0.6
        if degen_rtol is None:
            degen_rtol = torch.finfo(dtype).eps**0.4

        # check the degeneracy
        if degen_atol > 0 or degen_rtol > 0:
            # idx_degen: (*BAM, neig, neig)
            idx_degen, isdegenerate = _check_degen(evals, degen_atol, degen_rtol)
        else:
            isdegenerate = False
        if not isdegenerate:
            idx_degen = None

        # the loss function where the gradient will be retrieved
        # warnings: if not all params have the connection to the output of A,
        # it could cause an infinite loop because pytorch will keep looking
        # for the *params node and propagate further backward via the `evecs`
        # path. So make sure all the *params are all connected in the graph.
        with torch.enable_grad():
            params = [p.clone().requires_grad_() for p in params]
            with A.uselinopparams(*params):
                loss = A.mm(evecs)  # (*BAM, na, neig)

        # if degenerate, check the conditions for finite derivative
        if isdegenerate:
            xtg = torch.matmul(evecs.transpose(-2, -1).conj(), grad_evecs)
            req1 = idx_degen * (xtg - xtg.transpose(-2, -1).conj())
            reqtol = xtg.abs().max() * grad_evecs.shape[-2] * torch.finfo(grad_evecs.dtype).eps

            if not torch.all(torch.abs(req1) <= reqtol):
                # if the requirements are not satisfied, raises a warning
                msg = ("Degeneracy appears but the loss function seem to depend "
                       "strongly on the eigenvector. The gradient might be incorrect.\n")
                msg += "Eigenvalues:\n%s\n" % str(evals)
                msg += "Degenerate map:\n%s\n" % str(idx_degen)
                msg += "Requirements (should be all 0s):\n%s" % str(req1)
                warnings.warn(Warning(msg))

        # calculate the contributions from the eigenvalues
        gevalsA = grad_evals.unsqueeze(-2) * evecs  # (*BAM, na, neig)

        # calculate the contributions from the eigenvectors
        with M.uselinopparams(*mparams) if M is not None else dummy_context_manager():
            # orthogonalize the grad_evecs with evecs
            B = _ortho(grad_evecs, evecs, D=idx_degen, M=M, mright=False)

            # Based on test cases, complex datatype is more likely to suffer from
            # singularity error when doing the inverse. Therefore, I add a small
            # offset here to prevent that from happening
            if torch.is_complex(B):
                evals_offset = evals + 1e-14
            else:
                evals_offset = evals

            with A.uselinopparams(*params):
                gevecs = solve(A, -B, evals_offset, M, bck_options=ctx.bck_config,
                               **ctx.bck_config)  # (*BAM, na, neig)

            # orthogonalize gevecs w.r.t. evecs
            gevecsA = _ortho(gevecs, evecs, D=None, M=M, mright=True)

        # accummulate the gradient contributions
        gaccumA = gevalsA + gevecsA
        grad_params = torch.autograd.grad(
            outputs=(loss,),
            inputs=params,
            grad_outputs=(gaccumA,),
            create_graph=torch.is_grad_enabled(),
        )

        grad_mparams = []
        if ctx.M is not None:
            with torch.enable_grad():
                mparams = [p.clone().requires_grad_() for p in mparams]
                with M.uselinopparams(*mparams):
                    mloss = M.mm(evecs)  # (*BAM, na, neig)
            gevalsM = -gevalsA * evals.unsqueeze(-2)
            gevecsM = -gevecsA * evals.unsqueeze(-2)

            # the contribution from the parallel elements
            gevecsM_par = (-0.5 * torch.einsum("...ae,...ae->...e", grad_evecs, evecs.conj())
                           ).unsqueeze(-2) * evecs  # (*BAM, na, neig)

            gaccumM = gevalsM + gevecsM + gevecsM_par
            grad_mparams = torch.autograd.grad(
                outputs=(mloss,),
                inputs=mparams,
                grad_outputs=(gaccumM,),
                create_graph=torch.is_grad_enabled(),
            )

        return (None, None, None, None, None, None, None, *grad_params, *grad_mparams)


def custom_exacteig(A, neig, mode, M=None, **options):
    return _exacteig(A, neig, mode, M)


def _check_degen(evals: torch.Tensor, degen_atol: float, degen_rtol: float) -> \
        Tuple[torch.Tensor, bool]:
    # evals: (*BAM, neig)

    # get the index of degeneracies
    neig = evals.shape[-1]
    evals_diff = torch.abs(evals.unsqueeze(-2) - evals.unsqueeze(-1))  # (*BAM, neig, neig)
    degen_thrsh = degen_atol + degen_rtol * torch.abs(evals).unsqueeze(-1)
    idx_degen = (evals_diff < degen_thrsh).to(evals.dtype)
    isdegenerate = bool(torch.sum(idx_degen) > torch.numel(evals))
    return idx_degen, isdegenerate

def _ortho(A: torch.Tensor, B: torch.Tensor, *,
           D: Optional[torch.Tensor] = None,
           M: Optional[LinearOperator] = None,
           mright: bool = False) -> torch.Tensor:
    # orthogonalize every column in A w.r.t. columns in B
    # D is the degeneracy map, if None, it is identity matrix
    # M is the overlap matrix (in LinearOperator)
    # mright indicates whether to operate M at the right or at the left

    # shapes:
    # A: (*BAM, na, neig)
    # B: (*BAM, na, neig)
    if D is None:
        # contracted using opt_einsum
        str1 = "...rc,...rc->...c"
        Bconj = B.conj()
        if M is None:
            return A - torch.einsum(str1, A, Bconj).unsqueeze(-2) * B
        elif mright:
            return A - torch.einsum(str1, M.mm(A), Bconj).unsqueeze(-2) * B
        else:
            return A - M.mm(torch.einsum(str1, A, Bconj).unsqueeze(-2) * B)
    else:
        BH = B.transpose(-2, -1).conj()
        if M is None:
            DBHA = D * torch.matmul(BH, A)
            return A - torch.matmul(B, DBHA)
        elif mright:
            DBHA = D * torch.matmul(BH, M.mm(A))
            return A - torch.matmul(B, DBHA)
        else:
            DBHA = D * torch.matmul(BH, A)
            return A - M.mm(torch.matmul(B, DBHA))


def _exacteig(A: LinearOperator, neig: int,
             mode: str, M: Optional[LinearOperator]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Eigendecomposition using explicit matrix construction.
    No additional option for this method.

    Warnings
    --------
    * As this method construct the linear operators explicitly, it might requires
      a large memory.
    """
    Amatrix = A.fullmatrix()  # (*BA, q, q)
    if M is None:
        # evals, evecs = torch.linalg.eigh(Amatrix, eigenvectors=True)  # (*BA, q), (*BA, q, q)
        evals, evecs = _degen_symeig.apply(Amatrix)  # (*BA, q, q)
        return _take_eigpairs(evals, evecs, neig, mode)
    else:
        Mmatrix = M.fullmatrix()  # (*BM, q, q)

        # M decomposition to make A symmetric
        # it is done this way to make it numerically stable in avoiding
        # complex eigenvalues for (near-)degenerate case
        L = torch.linalg.cholesky(Mmatrix)  # (*BM, q, q)
        Linv = torch.inverse(L)  # (*BM, q, q)
        LinvT = Linv.transpose(-2, -1).conj()  # (*BM, q, q)
        A2 = torch.matmul(Linv, torch.matmul(Amatrix, LinvT))  # (*BAM, q, q)

        # calculate the eigenvalues and eigenvectors
        # (the eigvecs are normalized in M-space)
        # evals, evecs = torch.linalg.eigh(A2, eigenvectors=True)  # (*BAM, q, q)
        evals, evecs = _degen_symeig.apply(A2)  # (*BAM, q, q)
        evals, evecs = _take_eigpairs(evals, evecs, neig, mode)  # (*BAM, neig) and (*BAM, q, neig)
        evecs = torch.matmul(LinvT, evecs)
        return evals, evecs

# temporary solution to https://github.com/pytorch/pytorch/issues/47599
class _degen_symeig(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        eival, eivec = torch.linalg.eigh(A)
        ctx.save_for_backward(eival, eivec)
        return eival, eivec

    @staticmethod
    def backward(ctx, grad_eival, grad_eivec):
        eival, eivec = ctx.saved_tensors
        min_threshold = torch.finfo(eival.dtype).eps ** 0.6
        eivect = eivec.transpose(-2, -1).conj()

        # remove the degenerate part
        # see https://arxiv.org/pdf/2011.04366.pdf
        if grad_eivec is not None:
            # take the contribution from the eivec
            F = eival.unsqueeze(-2) - eival.unsqueeze(-1)
            idx = torch.abs(F) <= min_threshold
            F[idx] = float("inf")

            F = F.pow(-1)
            F = F * torch.matmul(eivect, grad_eivec)
            result = torch.matmul(eivec, torch.matmul(F, eivect))
        else:
            result = torch.zeros_like(eivec)

        # calculate the contribution from the eival
        if grad_eival is not None:
            result += torch.matmul(eivec, grad_eival.unsqueeze(-1) * eivect)

        # symmetrize to reduce numerical instability
        result = (result + result.transpose(-2, -1).conj()) * 0.5
        return result

def _davidson(A: LinearOperator, neig: int,
             mode: str,
             M: Optional[LinearOperator] = None,
             max_niter: int = 1000,
             nguess: Optional[int] = None,
             v_init: str = "randn",
             max_addition: Optional[int] = None,
             min_eps: float = 1e-6,
             verbose: bool = False,
             **unused) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Using Davidson method for large sparse matrix eigendecomposition [2]_.

    Arguments
    ---------
    max_niter: int
        Maximum number of iterations
    v_init: str
        Mode of the initial guess (``"randn"``, ``"rand"``, ``"eye"``)
    max_addition: int or None
        Maximum number of new guesses to be added to the collected vectors.
        If None, set to ``neig``.
    min_eps: float
        Minimum residual error to be stopped
    verbose: bool
        Option to be verbose

    References
    ----------
    .. [2] P. Arbenz, "Lecture Notes on Solving Large Scale Eigenvalue Problems"
           http://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter12.pdf
    """
    # TODO: optimize for large linear operator and strict min_eps
    # Ideas:
    # (1) use better strategy to get the estimate on eigenvalues
    # (2) use restart strategy

    if nguess is None:
        nguess = neig
    if max_addition is None:
        max_addition = neig

    # get the shape of the transformation
    na = A.shape[-1]
    if M is None:
        bcast_dims = A.shape[:-2]
    else:
        bcast_dims = get_bcasted_dims(A.shape[:-2], M.shape[:-2])
    dtype = A.dtype
    device = A.device

    prev_eigvals = None
    prev_eigvalT = None
    stop_reason = "max_niter"
    shift_is_eigvalT = False
    idx = torch.arange(neig).unsqueeze(-1)  # (neig, 1)

    # set up the initial guess
    V = _set_initial_v(v_init.lower(), dtype, device,
                       bcast_dims, na, nguess,
                       M=M)  # (*BAM, na, nguess)

    best_resid: Union[float, torch.Tensor] = float("inf")
    AV = A.mm(V)
    for i in range(max_niter):
        VT = V.transpose(-2, -1)  # (*BAM,nguess,na)
        # Can be optimized by saving AV from the previous iteration and only
        # operate AV for the new V. This works because the old V has already
        # been orthogonalized, so it will stay the same
        # AV = A.mm(V) # (*BAM,na,nguess)
        T = torch.matmul(VT, AV)  # (*BAM,nguess,nguess)

        # eigvals are sorted from the lowest
        # eval: (*BAM, nguess), evec: (*BAM, nguess, nguess)
        eigvalT, eigvecT = torch.linalg.eigh(T)
        eigvalT, eigvecT = _take_eigpairs(eigvalT, eigvecT, neig, mode)  # (*BAM, neig) and (*BAM, nguess, neig)

        # calculate the eigenvectors of A
        eigvecA = torch.matmul(V, eigvecT)  # (*BAM, na, neig)

        # calculate the residual
        AVs = torch.matmul(AV, eigvecT)  # (*BAM, na, neig)
        LVs = eigvalT.unsqueeze(-2) * eigvecA  # (*BAM, na, neig)
        if M is not None:
            LVs = M.mm(LVs)
        resid = AVs - LVs  # (*BAM, na, neig)

        # print information and check convergence
        max_resid = resid.abs().max()
        if prev_eigvalT is not None:
            deigval = eigvalT - prev_eigvalT
            max_deigval = deigval.abs().max()
            if verbose:
                print("Iter %3d (guess size: %d): resid: %.3e, devals: %.3e" %
                      (i + 1, nguess, max_resid, max_deigval))  # type:ignore

        if max_resid < best_resid:
            best_resid = max_resid
            best_eigvals = eigvalT
            best_eigvecs = eigvecA
        if max_resid < min_eps:
            break
        if AV.shape[-1] == AV.shape[-2]:
            break
        prev_eigvalT = eigvalT

        # apply the preconditioner
        t = -resid  # (*BAM, na, neig)

        # orthogonalize t with the rest of the V
        t = to_fortran_order(t)
        Vnew = torch.cat((V, t), dim=-1)
        if Vnew.shape[-1] > Vnew.shape[-2]:
            Vnew = Vnew[..., :Vnew.shape[-2]]
        nadd = Vnew.shape[-1] - V.shape[-1]
        nguess = nguess + nadd
        if M is not None:
            MV_ = M.mm(Vnew)
            V, R = tallqr(Vnew, MV=MV_)
        else:
            V, R = tallqr(Vnew)
        AVnew = A.mm(V[..., -nadd:])  # (*BAM,na,nadd)
        AVnew = to_fortran_order(AVnew)
        AV = torch.cat((AV, AVnew), dim=-1)

    eigvals = best_eigvals  # (*BAM, neig)
    eigvecs = best_eigvecs  # (*BAM, na, neig)
    return eigvals, eigvecs

def _set_initial_v(vinit_type: str,
                   dtype: torch.dtype, device: torch.device,
                   batch_dims: Sequence,
                   na: int,
                   nguess: int,
                   M: Optional[LinearOperator] = None) -> torch.Tensor:

    torch.manual_seed(12421)
    if vinit_type == "eye":
        nbatch = functools.reduce(lambda x, y: x * y, batch_dims, 1)
        V = torch.eye(na, nguess, dtype=dtype, device=device).unsqueeze(
            0).repeat(nbatch, 1, 1).reshape(*batch_dims, na, nguess)
    elif vinit_type == "randn":
        V = torch.randn((*batch_dims, na, nguess), dtype=dtype, device=device)
    elif vinit_type == "random" or vinit_type == "rand":
        V = torch.rand((*batch_dims, na, nguess), dtype=dtype, device=device)
    else:
        raise ValueError("Unknown v_init type: %s" % vinit_type)

    # orthogonalize V
    if isinstance(M, LinearOperator):
        V, R = tallqr(V, MV=M.mm(V))
    else:
        V, R = tallqr(V)
    return V

def _take_eigpairs(eival, eivec, neig, mode):
    # eival: (*BV, na)
    # eivec: (*BV, na, na)
    if mode == "lowest":
        eival = eival[..., :neig]
        eivec = eivec[..., :neig]
    else:  # uppest
        eival = eival[..., -neig:]
        eivec = eivec[..., -neig:]
    return eival, eivec

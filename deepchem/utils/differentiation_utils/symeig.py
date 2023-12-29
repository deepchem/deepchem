import functools
from deepchem.utils.differentiation_utils import LinearOperator, get_bcasted_dims
from deepchem.utils.pytorch_utils import tallqr, to_fortran_order
from typing import Optional, Sequence, Tuple, Union
import torch


def custom_exacteig(A: LinearOperator,
                    neig: int,
                    mode: str,
                    M: Optional[LinearOperator] = None,
                    **options):
    """Customized exact eigendecomposition method.

    Parameters
    ----------
    A: LinearOperator
        The linear operator to be diagonalized. It should be Hermitian.
    neig: int
        Number of eigenvalues and eigenvectors to be calculated.
    mode: str
        Mode of the eigenvalues to be calculated (``"lowest"``, ``"uppest"``)
    M: Optional[LinearOperator]
        The overlap matrix. If None, identity matrix is used.
    **options: dict
        Additional options for the eigendecomposition method.

    Returns
    -------
    evals: torch.Tensor
        Eigenvalues of the linear operator.
    evecs: torch.Tensor
        Eigenvectors of the linear operator.

    """
    return _exacteig(A, neig, mode, M)


def _check_degen(evals: torch.Tensor, degen_atol: float, degen_rtol: float) -> \
        Tuple[torch.Tensor, bool]:
    """Check whether there are degeneracies in the eigenvalues.

    Parameters
    ----------
    evals: torch.Tensor 
        Eigenvalues of the linear operator.
    degen_atol: float
        Absolute tolerance for degeneracy.
    degen_rtol: float
        Relative tolerance for degeneracy.

    Returns
    -------
    idx_degen: torch.Tensor
        Index of degeneracies.
    isdegenerate: bool
        Whether there are degeneracies.

    """
    neig = evals.shape[-1]
    evals_diff = torch.abs(evals.unsqueeze(-2) -
                           evals.unsqueeze(-1))  # (*BAM, neig, neig)
    degen_thrsh = degen_atol + degen_rtol * torch.abs(evals).unsqueeze(-1)
    idx_degen = (evals_diff < degen_thrsh).to(evals.dtype)
    isdegenerate = bool(torch.sum(idx_degen) > torch.numel(evals))
    return idx_degen, isdegenerate


def _ortho(A: torch.Tensor,
           B: torch.Tensor,
           *,
           D: Optional[torch.Tensor] = None,
           M: Optional[LinearOperator] = None,
           mright: bool = False) -> torch.Tensor:
    """Orthogonalize A w.r.t. B.

    Parameters
    ----------
    A: torch.Tensor
        First tensor. Shape: (*BAM, na, neig)
    B: torch.Tensor
        Second tensor. Shape: (*BAM, na, neig)
    D: Optional[torch.Tensor]
        Degeneracy map. If None, it is identity matrix.
    M: Optional[LinearOperator]
        The overlap matrix. If None, identity matrix is used.
    mright: bool
        Whether to operate M at the right or at the left.

    Returns
    -------
    torch.Tensor
        The Orthogonalized A.

    """
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


def _exacteig(A: LinearOperator, neig: int, mode: str,
              M: Optional[LinearOperator]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Eigendecomposition using explicit matrix construction.
    No additional option for this method.

    Parameters
    ----------
    A: LinearOperator
        The linear operator to be diagonalized. It should be Hermitian.
    neig: int
        Number of eigenvalues and eigenvectors to be calculated.
    mode: str
        Mode of the eigenvalues to be calculated (``"lowest"``, ``"uppest"``)
    M: Optional[LinearOperator]
        The overlap matrix. If None, identity matrix is used.

    Returns
    -------
    evals: torch.Tensor
        Eigenvalues of the linear operator.
    evecs: torch.Tensor
        Eigenvectors of the linear operator.

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
        evals, evecs = _take_eigpairs(evals, evecs, neig,
                                      mode)  # (*BAM, neig) and (*BAM, q, neig)
        evecs = torch.matmul(LinvT, evecs)
        return evals, evecs


class _degen_symeig(torch.autograd.Function):
    """temporary solution to 1. https://github.com/pytorch/pytorch/issues/47599
    2. https://github.com/pytorch/pytorch/issues/57272

    Grad on torch.symeig gives the wrong value if there are repeated eigenvalues
    (degeneracy exists), even though the grad should be finite.

    """

    @staticmethod
    def forward(ctx, A):
        """Calculate the eigenvalues and eigenvectors of a Hermitian matrix.

        Parameters
        ----------
        A: torch.Tensor
            The matrix to be diagonalized. It should be Hermitian.

        Returns
        -------
        eival: torch.Tensor
            Eigenvalues of the matrix.
        eivec: torch.Tensor
            Eigenvectors of the matrix.

        """
        eival, eivec = torch.linalg.eigh(A)
        ctx.save_for_backward(eival, eivec)
        return eival, eivec

    @staticmethod
    def backward(ctx, grad_eival, grad_eivec):
        """Calculate the gradient of the eigenvalues and eigenvectors of a Hermitian matrix.

        Parameters
        ----------
        grad_eival: torch.Tensor
            Gradient of the eigenvalues.
        grad_eivec: torch.Tensor
            Gradient of the eigenvectors.

        """
        eival, eivec = ctx.saved_tensors
        min_threshold = torch.finfo(eival.dtype).eps**0.6
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


def _davidson(A: LinearOperator,
              neig: int,
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
    A: LinearOperator
        The linear operator to be diagonalized. It should be Hermitian.
    neig: int
        Number of eigenvalues and eigenvectors to be calculated.
    mode: str
        Mode of the eigenvalues to be calculated (``"lowest"``, ``"uppest"``)
    M: Optional[LinearOperator]
        The overlap matrix. If None, identity matrix is used.
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
    V = _set_initial_v(v_init.lower(),
                       dtype,
                       device,
                       bcast_dims,
                       na,
                       nguess,
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
        eigvalT, eigvecT = _take_eigpairs(
            eigvalT, eigvecT, neig,
            mode)  # (*BAM, neig) and (*BAM, nguess, neig)

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
                   dtype: torch.dtype,
                   device: torch.device,
                   batch_dims: Sequence,
                   na: int,
                   nguess: int,
                   M: Optional[LinearOperator] = None) -> torch.Tensor:
    """Set the initial guess for the eigenvectors.

    Parameters
    ----------
    vinit_type: str
        Mode of the initial guess (``"randn"``, ``"rand"``, ``"eye"``)
    dtype: torch.dtype
        Data type of the initial guess.
    device: torch.device
        Device of the initial guess.
    batch_dims: Sequence
        Batch dimensions of the initial guess.
    na: int
        Number of basis functions.
    nguess: int
        Number of initial guesses.
    M: Optional[LinearOperator]
        The overlap matrix. If None, identity matrix is used.

    Returns
    -------
    V: torch.Tensor
        Initial guess for the eigenvectors.

    """

    torch.manual_seed(12421)
    if vinit_type == "eye":
        nbatch = functools.reduce(lambda x, y: x * y, batch_dims, 1)
        V = torch.eye(na, nguess, dtype=dtype,
                      device=device).unsqueeze(0).repeat(nbatch, 1, 1).reshape(
                          *batch_dims, na, nguess)
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


def _take_eigpairs(eival: torch.Tensor, eivec: torch.Tensor, neig: int,
                   mode: str):
    """Take the eigenpairs from the eigendecomposition.

    Parameters
    ----------
    eival: torch.Tensor
        Eigenvalues of the linear operator.
    eivec: torch.Tensor
        Eigenvectors of the linear operator.
    neig: int
        Number of eigenvalues and eigenvectors to be calculated.
    mode: str
        Mode of the eigenvalues to be calculated (``"lowest"``, ``"uppest"``)

    Returns
    -------
    eival: torch.Tensor
        Eigenvalues of the linear operator.
    eivec: torch.Tensor
        Eigenvectors of the linear operator.

    """
    # eival: (*BV, na)
    # eivec: (*BV, na, na)
    if mode == "lowest":
        eival = eival[..., :neig]
        eivec = eivec[..., :neig]
    else:
        eival = eival[..., -neig:]
        eivec = eivec[..., -neig:]
    return eival, eivec

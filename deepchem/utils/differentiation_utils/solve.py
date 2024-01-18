import numpy as np
import torch
import warnings
from typing import Union, Optional, Callable
from deepchem.utils.differentiation_utils import LinearOperator, normalize_bcast_dims, get_bcasted_dims
from deepchem.utils import ConvergenceWarning, get_np_dtype
from scipy.sparse.linalg import gmres as scipy_gmres


# Hidden
def wrap_gmres(A, B, E=None, M=None, min_eps=1e-9, max_niter=None, **unused):
    """
    Using SciPy's gmres method to solve the linear equation.

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.differentiation_utils import LinearOperator
    >>> A = LinearOperator.m(torch.tensor([[1., 2], [3, 4]]))
    >>> B = torch.tensor([[[5., 6], [7, 8]]])
    >>> wrap_gmres(A, B, None, None)
    tensor([[[-3.0000, -4.0000],
             [ 4.0000,  5.0000]]])

    Parameters
    ----------
    A: LinearOperator
        The linear operator A to be solved. Shape: (*BA, na, na)
    B: torch.Tensor
        Batched matrix B. Shape: (*BB, na, ncols)
    E: torch.Tensor or None
        Batched vector E. Shape: (*BE, ncols)
    M: LinearOperator or None
        The linear operator M. Shape: (*BM, na, na)
    min_eps: float
        Relative tolerance for stopping conditions
    max_niter: int or None
        Maximum number of iterations. If ``None``, default to twice of the
        number of columns of ``A``.

    Returns
    -------
    torch.Tensor
        The Solution matrix X. Shape: (*BBE, na, ncols)

    """

    # NOTE: currently only works for batched B (1 batch dim), but unbatched A
    assert len(A.shape) == 2 and len(
        B.shape
    ) == 3, "Currently only works for batched B (1 batch dim), but unbatched A"
    assert not torch.is_complex(B), "complex is not supported in gmres"

    # check the parameters
    msg = "GMRES can only do AX=B"
    assert A.shape[-2] == A.shape[
        -1], "GMRES can only work for square operator for now"
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
            x, info = scipy_gmres(op,
                                  B_np[i, j, :],
                                  tol=min_eps,
                                  atol=1e-12,
                                  maxiter=max_niter)
            if info > 0:
                msg = "The GMRES iteration does not converge to the desired value "\
                      "(%.3e) after %d iterations" % \
                      (min_eps, info)
                warnings.warn(ConvergenceWarning(msg))
            res_np[i, j, :] = x

    res = torch.tensor(res_np, dtype=B.dtype, device=B.device)
    res = res.transpose(-1, -2)  # (nbatch, na, ncols)
    return res


def exactsolve(A: LinearOperator, B: torch.Tensor, E: Union[torch.Tensor, None],
               M: Union[LinearOperator, None]):
    """
    Solve the linear equation by contructing the full matrix of LinearOperators.

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.differentiation_utils import LinearOperator
    >>> A = LinearOperator.m(torch.tensor([[1., 2], [3, 4]]))
    >>> B = torch.tensor([[5., 6], [7, 8]])
    >>> exactsolve(A, B, None, None)
    tensor([[-3., -4.],
            [ 4.,  5.]])

    Parameters
    ----------
    A: LinearOperator
        The linear operator A to be solved. Shape: (*BA, na, na)
    B: torch.Tensor
        Batched matrix B. Shape: (*BB, na, ncols)
    E: torch.Tensor or None
        Batched vector E. Shape: (*BE, ncols)
    M: LinearOperator or None
        The linear operator M. Shape: (*BM, na, na)

    Returns
    -------
    torch.Tensor
        The Solution matrix X. Shape: (*BBE, na, ncols)

    Warnings
    --------
    * As this method construct the linear operators explicitly, it might requires
      a large memory.

    """
    if E is None:
        Amatrix = A.fullmatrix()
        x = torch.linalg.solve(Amatrix, B)
    elif M is None:
        Amatrix = A.fullmatrix()
        x = solve_ABE(Amatrix, B, E)
    else:
        Mmatrix = M.fullmatrix()
        L = torch.linalg.cholesky(Mmatrix)
        Linv = torch.inverse(L)
        LinvT = Linv.transpose(-2, -1).conj()
        A2 = torch.matmul(Linv, A.mm(LinvT))
        B2 = torch.matmul(Linv, B)

        X2 = solve_ABE(A2, B2, E)
        x = torch.matmul(LinvT, X2)
    return x


def solve_ABE(A: torch.Tensor, B: torch.Tensor, E: torch.Tensor):
    """ Solve the linear equation AX = B - diag(E)X.

    Examples
    --------
    >>> import torch
    >>> A = torch.tensor([[1., 2], [3, 4]])
    >>> B = torch.tensor([[5., 6], [7, 8]])
    >>> E = torch.tensor([1., 2])
    >>> solve_ABE(A, B, E)
    tensor([[-0.1667,  0.5000],
            [ 2.5000,  3.2500]])

    Parameters
    ----------
    A: torch.Tensor
        The batched matrix A. Shape: (*BA, na, na)
    B: torch.Tensor
        The batched matrix B. Shape: (*BB, na, ncols)
    E: torch.Tensor
        The batched vector E. Shape: (*BE, ncols)

    Returns
    -------
    torch.Tensor
        The batched matrix X.

    """
    na = A.shape[-1]
    BA, BB, BE = normalize_bcast_dims(A.shape[:-2], B.shape[:-2], E.shape[:-1])
    E = E.reshape(1, *BE, E.shape[-1]).transpose(0, -1)  # (ncols, *BE, 1)
    B = B.reshape(1, *BB, *B.shape[-2:]).transpose(0, -1)  # (ncols, *BB, na, 1)

    # NOTE: The line below is very inefficient for large na and ncols
    AE = A - torch.diag_embed(E.repeat_interleave(repeats=na, dim=-1),
                              dim1=-2,
                              dim2=-1)  # (ncols, *BAE, na, na)
    r = torch.linalg.solve(AE, B)  # (ncols, *BAEM, na, 1)
    r = r.transpose(0, -1).squeeze(0)  # (*BAEM, na, ncols)
    return r


# general helpers
def get_batchdims(A: LinearOperator, B: torch.Tensor,
                  E: Union[torch.Tensor, None], M: Union[LinearOperator, None]):
    """Get the batch dimensions of the linear operator and the matrix B

    Examples
    --------
    >>> from deepchem.utils.differentiation_utils import MatrixLinearOperator
    >>> import torch
    >>> A = MatrixLinearOperator(torch.randn(4, 3, 3), True)
    >>> B = torch.randn(3, 3, 2)
    >>> get_batchdims(A, B, None, None)
    [4]

    Parameters
    ----------
    A: LinearOperator
        The linear operator. It can be a batched linear operator.
    B: torch.Tensor
        The matrix B. It can be a batched matrix.
    E: Union[torch.Tensor, None]
        The matrix E. It can be a batched matrix.
    M: Union[LinearOperator, None]
        The linear operator M. It can be a batched linear operator.

    Returns
    -------
    List[int]
        The batch dimensions of the linear operator and the matrix B

    """

    batchdims = [A.shape[:-2], B.shape[:-2]]
    if E is not None:
        batchdims.append(E.shape[:-1])
        if M is not None:
            batchdims.append(M.shape[:-2])
    return get_bcasted_dims(*batchdims)


def setup_precond(
    precond: Optional[LinearOperator] = None
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Setup the preconditioning function

    Examples
    --------
    >>> from deepchem.utils.differentiation_utils import MatrixLinearOperator
    >>> import torch
    >>> A = MatrixLinearOperator(torch.randn(4, 3, 3), True)
    >>> B = torch.randn(4, 3, 2)
    >>> cond = setup_precond(A)
    >>> cond(B).shape
    torch.Size([4, 3, 2])

    Parameters
    ----------
    precond: Optional[LinearOperator]
        The preconditioning linear operator. If None, no preconditioning is
        applied.

    Returns
    -------
    Callable[[torch.Tensor], torch.Tensor]
        The preconditioning function. It takes a tensor and returns a tensor.

    """
    if isinstance(precond, LinearOperator):

        def precond_fcn(x):
            return precond.mm(x)
    elif precond is None:

        def precond_fcn(x):
            return x
    else:
        raise TypeError("precond can only be LinearOperator or None")
    return precond_fcn


def dot(r: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Dot product of two vectors. r and z must have the same shape.
    Then sums it up across the last dimension.

    Examples
    --------
    >>> import torch
    >>> r = torch.tensor([[1, 2], [3, 4]])
    >>> z = torch.tensor([[5, 6], [7, 8]])
    >>> dot(r, z)
    tensor([[26, 44]])

    Parameters
    ----------
    r: torch.Tensor
        The first vector. Shape: (*BR, nr, nc)
    z: torch.Tensor
        The second vector. Shape: (*BR, nr, nc)

    Returns
    -------
    torch.Tensor
        The dot product of r and z. Shape: (*BR, 1, nc)

    """
    return torch.einsum("...rc,...rc->...c", r.conj(), z).unsqueeze(-2)

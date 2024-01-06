import torch
from typing import Optional, Sequence
from deepchem.utils.differentiation_utils import LinearOperator
import functools
from deepchem.utils.pytorch_utils import tallqr


def _set_initial_v(vinit_type: str,
                   dtype: torch.dtype,
                   device: torch.device,
                   batch_dims: Sequence,
                   na: int,
                   nguess: int,
                   M: Optional[LinearOperator] = None) -> torch.Tensor:
    """Set the initial guess for the eigenvectors.

    Examples
    --------
    >>> import torch
    >>> vinit_type = "eye"
    >>> dtype = torch.float64
    >>> device = torch.device("cpu")
    >>> batch_dims = (2, 3)
    >>> na = 4
    >>> nguess = 2
    >>> M = None
    >>> V = _set_initial_v(vinit_type, dtype, device, batch_dims, na, nguess, M)
    >>> V
    tensor([[[[1., 0.],
              [0., 1.],
              [0., 0.],
              [0., 0.]],
    <BLANKLINE>
             [[1., 0.],
              [0., 1.],
              [0., 0.],
              [0., 0.]],
    <BLANKLINE>
             [[1., 0.],
              [0., 1.],
              [0., 0.],
              [0., 0.]]],
    <BLANKLINE>
    <BLANKLINE>
            [[[1., 0.],
              [0., 1.],
              [0., 0.],
              [0., 0.]],
    <BLANKLINE>
             [[1., 0.],
              [0., 1.],
              [0., 0.],
              [0., 0.]],
    <BLANKLINE>
             [[1., 0.],
              [0., 1.],
              [0., 0.],
              [0., 0.]]]], dtype=torch.float64)

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
    M: Optional[LinearOperator] (default None)
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

    Examples
    --------
    >>> import torch
    >>> eival = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    >>> eivec = torch.tensor([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
    ...                       [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]])
    >>> neig = 2
    >>> mode = "lowest"
    >>> eival, eivec = _take_eigpairs(eival, eivec, neig, mode)
    >>> eival
    tensor([[1., 2.],
            [4., 5.]])
    >>> eivec
    tensor([[[1., 2.],
             [4., 5.],
             [7., 8.]],
    <BLANKLINE>
            [[1., 2.],
             [4., 5.],
             [7., 8.]]])

    Parameters
    ----------
    eival: torch.Tensor
        Eigenvalues of the linear operator. Shape: ``(*BV, na)``.
    eivec: torch.Tensor
        Eigenvectors of the linear operator. Shape: ``(*BV, na, na)``.
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
    if mode == "lowest":
        eival = eival[..., :neig]
        eivec = eivec[..., :neig]
    else:
        eival = eival[..., -neig:]
        eivec = eivec[..., -neig:]
    return eival, eivec

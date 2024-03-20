import math
import torch
from typing import Union, Optional, Tuple

eps = 1e-12
ZType = Union[int, float, torch.Tensor]

# safe operations


def safepow(a: torch.Tensor,
            p: torch.Tensor,
            eps: float = 1e-12) -> torch.Tensor:
    """Safely calculate the power of a tensor with a small eps to avoid nan.

    Examples
    --------
    >>> import torch
    >>> a = torch.tensor([1e-35, 2e-40])
    >>> p = torch.tensor([2., 3])
    >>> safepow(a, p)
    tensor([1.0000e-24, 1.0000e-36])
    >>> a**p
    tensor([0., 0.])

    Parameters
    ----------
    a: torch.Tensor
        Base tensor on which to calculate the power. Must be positive.
    p: torch.Tensor
        Power tensor, by which to calculate the power.
    eps: float (default 1e-12)
        The eps to add to the base tensor.

    Returns
    -------
    torch.Tensor
        The result tensor.

    Raises
    ------
    RuntimeError
        If the base tensor contains negative values.

    """
    if torch.any(a < 0):
        raise RuntimeError("safepow only works for positive base")
    base = torch.sqrt(a * a + eps * eps)  # soft clip
    return base**p


def safenorm(a: torch.Tensor, dim: int, eps: float = 1e-15) -> torch.Tensor:
    """
    Calculate the 2-norm safely. The square root of the inner product of a
    vector with itself.

    Examples
    --------
    >>> import torch
    >>> a = torch.tensor([1e-35, 2e-40])
    >>> safenorm(a, 0)
    tensor(1.4142e-15)
    >>> a.norm()
    tensor(0.)


    Parameters
    ----------
    a: torch.Tensor
        The tensor to calculate the norm.
    dim: int
        The dimension to calculate the norm.
    eps: float (default 1e-15)
        The eps to add to the base tensor.

    Returns
    -------
    torch.Tensor
        The result tensor.

    """
    return torch.sqrt(torch.sum(a * a + eps * eps, dim=dim))


# occupation number gradients


def occnumber(
    a: ZType,
    n: Optional[int] = None,
    dtype: torch.dtype = torch.double,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """Occupation number (maxed at 1) where the total sum of the
    output equals to a with length of the output is n.

    Examples
    --------
    >>> import torch
    >>> occnumber(torch.tensor(2.5), 3, torch.double, torch.device('cpu'))
    tensor([1.0000, 1.0000, 0.5000], dtype=torch.float64)
    >>> occnumber(2.5)
    tensor([1.0000, 1.0000, 0.5000], dtype=torch.float64)

    Parameters
    ----------
    a: ZType
        Total sum of the output
    n: Optional[int] (default None)
        Length of the output
    dtype: torch.dtype (default torch.double)
        Data type of the output
    device: torch.device (default torch.device('cpu'))
        Device of the output

    Returns
    -------
    torch.Tensor
        The constructed occupation number

    """

    if isinstance(a, torch.Tensor):
        assert a.numel() == 1
        floor_a, ceil_a = get_floor_and_ceil(a.item())
    else:  # int or float
        floor_a, ceil_a = get_floor_and_ceil(a)

    # get the length of the tensor output
    if n is None:
        nlength = ceil_a
    else:
        nlength = n
        assert nlength >= ceil_a, "The length of occupation number must be at least %d" % ceil_a

    if isinstance(a, torch.Tensor):
        res = _OccNumber.apply(a, floor_a, ceil_a, nlength, dtype, device)
    else:
        res = _construct_occ_number(a,
                                    floor_a,
                                    ceil_a,
                                    nlength,
                                    dtype=dtype,
                                    device=device)
    return res


def _construct_occ_number(a: float, floor_a: int, ceil_a: int, nlength: int,
                          dtype: torch.dtype,
                          device: torch.device) -> torch.Tensor:
    """Construct the occupation number (maxed at 1) where the total sum of the
    output equals to a with length of the output is nlength.

    Examples
    --------
    >>> import torch
    >>> _construct_occ_number(2.5, 2, 3, 3, torch.double, torch.device('cpu'))
    tensor([1.0000, 1.0000, 0.5000], dtype=torch.float64)

    Parameters
    ----------
    a: float
        Total sum of the output
    floor_a: int
        Floor of a
    ceil_a: int
        Ceiling of a
    nlength: int
        Length of the output
    dtype: torch.dtype
        Data type of the output
    device: torch.device
        Device of the output

    Returns
    -------
    torch.Tensor
        The constructed occupation number

    """
    res = torch.zeros(nlength, dtype=dtype, device=device)
    res[:floor_a] = 1
    if ceil_a > floor_a:
        res[ceil_a - 1] = a - floor_a
    return res


class _OccNumber(torch.autograd.Function):
    """Construct the occupation number (maxed at 1) where the total sum of the
    output equals to a with length of the output is nlength.

    Examples
    --------
    >>> import torch
    >>> a = torch.tensor(2.5)
    >>> _OccNumber.apply(a, 2, 3, 3, torch.double, torch.device('cpu'))
    tensor([1.0000, 1.0000, 0.5000], dtype=torch.float64)

    """

    @staticmethod
    def forward(  # type: ignore
            self, a: torch.Tensor, floor_a: int, ceil_a: int, nlength: int,
            dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Forward pass of the occupation number.

        Parameters
        ----------
        a: torch.Tensor
            Total sum of the output
        floor_a: int
            Floor of a
        ceil_a: int
            Ceiling of a
        nlenght: int
            Length of the output
        dtype: torch.dtype
            Data type of the output
        device: torch.device
            Device of the output

        Returns
        -------
        torch.Tensor
            The constructed occupation number

        """
        res = _construct_occ_number(float(a.item()),
                                    floor_a,
                                    ceil_a,
                                    nlength,
                                    dtype=dtype,
                                    device=device)
        self.ceil_a = ceil_a
        return res

    @staticmethod
    def backward(self, grad_res: torch.Tensor):  # type: ignore
        """Backward pass of the occupation number.

        Parameters
        ----------
        grad_res: torch.Tensor
            Gradient of the output

        Returns
        -------
        Tuple[torch.Tensor, None, None, None, None, None]
            Gradient of the input

        """
        grad_a = grad_res[self.ceil_a - 1]
        return (grad_a,) + (None,) * 5


def get_floor_and_ceil(aa: Union[int, float]) -> Tuple[int, int]:
    """get the ceiling and flooring of aa.

    Examples
    --------
    >>> get_floor_and_ceil(2.5)
    (2, 3)

    Parameters
    ----------
    aa: Union[int, float]
        The input number

    Returns
    -------
    Tuple[int, int]
        The flooring and ceiling of aa

    """
    if isinstance(aa, int):
        ceil_a: int = aa
        floor_a: int = aa
    else:  # floor
        ceil_a = int(math.ceil(aa))
        floor_a = int(math.floor(aa))
    return floor_a, ceil_a


# other tensor ops


def safe_cdist(a: torch.Tensor,
               b: torch.Tensor,
               add_diag_eps: bool = False,
               diag_inf: bool = False):
    """L2 pairwise distance of a and b. The diagonal is either
    replaced with a small eps or infinite.

    Examples
    --------
    >>> import torch
    >>> a = torch.tensor([[1., 2], [3, 4]])
    >>> b = torch.tensor([[1., 2], [3, 4]])
    >>> safe_cdist(a, b)
    tensor([[0.0000, 2.8284],
            [2.8284, 0.0000]])
    >>> safe_cdist(a, b, add_diag_eps=True)
    tensor([[1.4142e-12, 2.8284e+00],
            [2.8284e+00, 1.4142e-12]])
    >>> safe_cdist(a, b, diag_inf=True)
    tensor([[   inf, 2.8284],
            [2.8284,    inf]])

    Parameters
    ----------
    a: torch.Tensor
        First Tensor. Shape: (`*BA`, na, ndim)
    n: torch.Tensor
        Second Tensor. Shape: (`*BB`, nb, ndim)

    Returns
    -------
    torch.Tensor
        Pairwise distance. Shape: (`*BAB`, na, nb)

    """
    square_mat = a.shape[-2] == b.shape[-2]

    dtype = a.dtype
    device = a.device
    ab = a.unsqueeze(-2) - b.unsqueeze(-3)  # (*BAB, na, nb, ndim)

    # add the diagonal with a small eps to safeguard from nan
    if add_diag_eps:
        if not square_mat:
            raise ValueError(
                "Enabling add_diag_eps for non-square result matrix is invalid")
        ab = ab + torch.eye(ab.shape[-2], dtype=dtype,
                            device=device).unsqueeze(-1) * eps

    ab = ab.norm(dim=-1)  # (*BAB, na, nb)

    # replace the diagonal with infinite (usually used for coulomb matrix)
    if diag_inf:
        if not square_mat:
            raise ValueError(
                "Enabling diag_inf for non-square result matrix is invalid")

        infdiag = torch.eye(ab.shape[-1], dtype=dtype, device=device)
        idiag = infdiag.diagonal()
        idiag[:] = float("inf")
        ab = ab + infdiag

    return ab


def safedenom(r: torch.Tensor, eps: float) -> torch.Tensor:
    """Avoid division by zero by replacing zero elements with eps.

    Used in CG and BiCGStab.

    Examples
    --------
    >>> import torch
    >>> r = torch.tensor([1e-11, 0])
    >>> safedenom(r, 1e-12)
    tensor([1.0000e-11, 1.0000e-12])

    Parameters
    ----------
    r: torch.Tensor
        The residual vector
    eps: float
        The minimum value to avoid division by zero

    Returns
    -------
    r: torch.Tensor
        The residual vector with zero elements replaced by eps

    """
    r[r == 0] = eps
    return r

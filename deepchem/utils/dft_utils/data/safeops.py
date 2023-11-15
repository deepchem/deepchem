import math
import torch
from typing import Optional, Tuple, Union
from deepchem.utils.dft_utils import ZType

eps = 1e-12


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
        First Tensor. Shape: (*BA, na, ndim)
    n: torch.Tensor
        Second Tensor. Shape: (*BB, nb, ndim)

    Returns
    -------
    torch.Tensor
        Pairwise distance. Shape: (*BAB, na, nb)

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

import math
import torch
from typing import Union, Optional, Tuple
from deepchem.utils.dft_utils import ZType

eps = 1e-12

# safe operations

def safepow(a: torch.Tensor, p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Safely calculate the power of a tensor with a small eps to avoid nan.

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
    return base ** p

def safenorm(a: torch.Tensor, dim: int, eps: float = 1e-15) -> torch.Tensor:
    """
    Calculate the 2-norm safely. The square root of the inner product of a
    vector with itself.

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

def occnumber(a: ZType,
              n: Optional[int] = None,
              dtype: torch.dtype = torch.double,
              device: torch.device = torch.device('cpu')) -> torch.Tensor:
    # returns the occupation number (maxed at 1) where the total sum of the
    # output equals to a with length of the output is n

    def _get_floor_and_ceil(aa: Union[int, float]) -> Tuple[int, int]:
        # get the ceiling and flooring of aa
        if isinstance(aa, int):
            ceil_a: int = aa
            floor_a: int = aa
        else:  # floor
            ceil_a = int(math.ceil(aa))
            floor_a = int(math.floor(aa))
        return floor_a, ceil_a

    if isinstance(a, torch.Tensor):
        assert a.numel() == 1
        floor_a, ceil_a = _get_floor_and_ceil(a.item())
    else:  # int or float
        floor_a, ceil_a = _get_floor_and_ceil(a)

    # get the length of the tensor output
    if n is None:
        nlength = ceil_a
    else:
        nlength = n
        assert nlength >= ceil_a, "The length of occupation number must be at least %d" % ceil_a

    if isinstance(a, torch.Tensor):
        res = _OccNumber.apply(a, floor_a, ceil_a, nlength, dtype, device)
    else:
        res = _construct_occ_number(a, floor_a, ceil_a, nlength, dtype=dtype, device=device)
    return res

def _construct_occ_number(a: float, floor_a: int, ceil_a: int, nlength: int,
                          dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    res = torch.zeros(nlength, dtype=dtype, device=device)
    res[:floor_a] = 1
    if ceil_a > floor_a:
        res[ceil_a - 1] = a - floor_a
    return res

class _OccNumber(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor,  # type: ignore
                floor_a: int, ceil_a: int, nlength: int,
                dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        res = _construct_occ_number(float(a.item()), floor_a, ceil_a, nlength, dtype=dtype, device=device)
        ctx.ceil_a = ceil_a
        return res

    @staticmethod
    def backward(ctx, grad_res: torch.Tensor):  # type: ignore
        grad_a = grad_res[ctx.ceil_a - 1]
        return (grad_a,) + (None,) * 5

########################## other tensor ops ##########################
def safe_cdist(a: torch.Tensor, b: torch.Tensor, add_diag_eps: bool = False,
               diag_inf: bool = False):
    # returns the L2 pairwise distance of a and b
    # a: (*BA, na, ndim)
    # b: (*BB, nb, ndim)
    # returns: (*BAB, na, nb)
    square_mat = a.shape[-2] == b.shape[-2]

    dtype = a.dtype
    device = a.device
    ab = a.unsqueeze(-2) - b.unsqueeze(-3)  # (*BAB, na, nb, ndim)

    # add the diagonal with a small eps to safeguard from nan
    if add_diag_eps:
        if not square_mat:
            raise ValueError("Enabling add_diag_eps for non-square result matrix is invalid")
        ab = ab + torch.eye(ab.shape[-2], dtype=dtype, device=device).unsqueeze(-1) * eps

    ab = ab.norm(dim=-1)  # (*BAB, na, nb)

    # replace the diagonal with infinite (usually used for coulomb matrix)
    if diag_inf:
        if not square_mat:
            raise ValueError("Enabling diag_inf for non-square result matrix is invalid")

        infdiag = torch.eye(ab.shape[-1], dtype=dtype, device=device)
        idiag = infdiag.diagonal()
        idiag[:] = float("inf")
        ab = ab + infdiag

    return ab

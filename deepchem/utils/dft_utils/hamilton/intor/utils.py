import ctypes
import ctypes.util
import dqclibs
import numpy as np
import torch
from deepchem.utils.dft_utils import LibcintWrapper
from typing import List
import scipy

# CONSTANTS
NDIM = 3

CINT = dqclibs.CINT
CGTO = dqclibs.CGTO
CPBC = dqclibs.CPBC
# CVHF = dqclibs.CVHF
CSYMM = dqclibs.CSYMM

c_null_ptr = ctypes.POINTER(ctypes.c_void_p)


def np2ctypes(a: np.ndarray) -> ctypes.c_void_p:
    """Get the ctypes of the numpy ndarray

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1, 2, 3])
    >>> ctype = np2ctypes(a)

    Parameters
    ----------
    a : np.ndarray
        Numpy ndarray

    Returns
    -------
    ctypes.c_void_p
        ctypes of the numpy ndarray

    """
    return a.ctypes.data_as(ctypes.c_void_p)


def int2ctypes(a: int) -> ctypes.c_int:
    """Convert the python's integer to ctypes' integer

    Examples
    --------
    >>> a = 1
    >>> ctype = int2ctypes(a)

    Parameters
    ----------
    a : int
        Python's integer

    Returns
    -------
    ctypes.c_int
        ctypes' integer

    """
    return ctypes.c_int(a)


def unweighted_coul_ft(gvgrids: torch.Tensor) -> torch.Tensor:
    # Returns the unweighted fourier transform of the coulomb kernel: 4*pi/|gv|^2
    # If |gv| == 0, then it is 0.
    # gvgrids: (ngv, ndim)
    # returns: (ngv,)
    gnorm2 = torch.einsum("xd,xd->x", gvgrids, gvgrids)
    gnorm2[gnorm2 < 1e-12] = float("inf")
    coulft = 4 * np.pi / gnorm2
    return coulft

def estimate_ovlp_rcut(precision: float, coeffs: torch.Tensor, alphas: torch.Tensor) -> float:
    # estimate the rcut for lattice sum to achieve the given precision
    # it is estimated based on the overlap integral
    langmom = 1
    C = (coeffs * coeffs + 1e-200) * (2 * langmom + 1) * alphas / precision
    r0 = torch.tensor(20.0, dtype=coeffs.dtype, device=coeffs.device)
    for i in range(2):
        r0 = torch.sqrt(2.0 * torch.log(C * (r0 * r0 * alphas) ** (langmom + 1) + 1.) / alphas)
    rcut = float(torch.max(r0).detach())
    return rcut

def estimate_g_cutoff(precision: float, coeffs: torch.Tensor, alphas: torch.Tensor) -> float:
    # g-point cut off estimation based on cubic lattice
    # based on _estimate_ke_cutoff from pyscf
    # https://github.com/pyscf/pyscf/blob/c9aa2be600d75a97410c3203abf35046af8ca615/pyscf/pbc/gto/cell.py#L498

    langmom = 1
    log_k0 = 3 + torch.log(alphas) / 2
    l2fac2 = scipy.special.factorial2(langmom * 2 + 1)
    a = precision * l2fac2 ** 2 * (4 * alphas) ** (langmom * 2 + 1) / (128 * np.pi ** 4 * coeffs ** 4)
    log_rest = torch.log(a)
    Ecut = 2 * alphas * (log_k0 * (4 * langmom + 3) - log_rest)
    Ecut[Ecut <= 0] = .5
    log_k0 = .5 * torch.log(Ecut * 2)
    Ecut = 2 * alphas * (log_k0 * (4 * langmom + 3) - log_rest)
    Ecut[Ecut <= 0] = .5
    Ecut_max = float(torch.max(Ecut).detach())

    # KE ~ 1/2 * g^2
    gcut = (2 * Ecut_max) ** 0.5
    return gcut

def get_gcut(precision: float, wrappers: List[LibcintWrapper], reduce: str = "min") -> float:
    # get the G-point cut-off from the given wrappers where the FT
    # eval/integration is going to be performed
    gcuts: List[float] = []
    for wrapper in wrappers:
        # TODO: using params here can be confusing because wrapper.params
        # returns all parameters (even if it is a subset)
        coeffs, alphas, _ = wrapper.params
        gcut_wrap = estimate_g_cutoff(precision, coeffs, alphas)
        gcuts.append(gcut_wrap)
    if len(gcuts) == 1:
        return gcuts[0]
    if reduce == "min":
        return min(*gcuts)
    elif reduce == "max":
        return max(*gcuts)
    else:
        raise ValueError("Unknown reduce: %s" % reduce)

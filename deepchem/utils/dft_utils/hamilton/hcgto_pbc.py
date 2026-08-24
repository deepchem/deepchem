from typing import List
import numpy as np
import scipy
import torch

from deepchem.utils.dft_utils import LibcintWrapper


def estimate_g_cutoff(precision: float, coeffs: torch.Tensor,
                      alphas: torch.Tensor) -> float:
    """Estimate g-point cutoff for periodic boundary condition calculations.
    
    This function estimates the g-point cutoff based on cubic lattice calculations,
    adapted from the _estimate_ke_cutoff function in PySCF. The cutoff determines
    the maximum reciprocal lattice vectors needed for accurate Fourier transform
    evaluations in periodic systems.
        
    Examples
    --------
    >>> import torch
    >>> coeffs = torch.tensor([1.0, 0.5])
    >>> alphas = torch.tensor([0.5, 1.0])
    >>> gcut = estimate_g_cutoff(1e-8, coeffs, alphas)
    >>> gcut
    12.076707448347365
        
    Parameters
    ----------
    precision : float
        Desired precision for the calculation.
    coeffs : torch.Tensor
        Contraction coefficients of the Gaussian basis functions.
    alphas : torch.Tensor
        Exponential parameters of the Gaussian basis functions.
        
    Returns
    -------
    float
        Estimated g-point cutoff value.

    References
    ----------
    PySCF implementation: https://github.com/pyscf/pyscf/blob/c9aa2be600d75a97410c3203abf35046af8ca615/pyscf/pbc/gto/cell.py#L498
    """
    langmom = 1
    log_k0 = 3 + torch.log(alphas) / 2
    l2fac2 = scipy.special.factorial2(langmom * 2 + 1)
    a = precision * l2fac2**2 * (4 * alphas)**(langmom * 2 +
                                               1) / (128 * np.pi**4 * coeffs**4)
    log_rest = torch.log(a)
    Ecut = 2 * alphas * (log_k0 * (4 * langmom + 3) - log_rest)
    Ecut[Ecut <= 0] = .5
    log_k0 = .5 * torch.log(Ecut * 2)
    Ecut = 2 * alphas * (log_k0 * (4 * langmom + 3) - log_rest)
    Ecut[Ecut <= 0] = .5
    Ecut_max = float(torch.max(Ecut).detach())

    # KE ~ 1/2 * g^2
    gcut = (2 * Ecut_max)**0.5
    return gcut


def get_gcut(precision: float,
             wrappers: List[LibcintWrapper],
             reduce: str = "min") -> float:
    """Get the G-point cutoff from LibcintWrapper objects for periodic calculations.
    
    This function calculates the appropriate G-point cutoff values for Fourier transform
    evaluations and integrations in periodic boundary condition calculations. It can
    handle multiple wrapper objects and combine their cutoffs using different reduction
    strategies.
    
    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.dft_utils import LibcintWrapper, AtomCGTOBasis, loadbasis
    >>> # Create a simple hydrogen atom with STO-3G basis
    >>> dtype = torch.double
    >>> pos = torch.tensor([0.0, 0.0, 0.0], dtype=dtype)
    >>> atomz = 1
    >>> basis = loadbasis("%d:%s" % (atomz, "STO-3G"), dtype=dtype, requires_grad=False)
    >>> atombasis = AtomCGTOBasis(atomz=atomz, bases=basis, pos=pos)
    >>> wrapper = LibcintWrapper([atombasis], spherical=True)
    >>> # Get G-point cutoff with minimum reduction
    >>> gcut = get_gcut(1e-8, [wrapper], reduce="min")
    >>> gcut
    23.426791297774784

    Parameters
    ----------
    precision : float
        Desired precision for the calculation.
    wrappers : List[LibcintWrapper]
        List of LibcintWrapper objects containing basis function parameters 
        (coefficients, exponents, etc.).
    reduce : str, optional
        Reduction strategy for multiple wrappers. Options are "min" (default) 
        to take the minimum cutoff, or "max" to take the maximum cutoff.
            
    Returns
    -------
    float
        G-point cutoff value based on the specified reduction strategy.
        
    Raises
    ------
    ValueError
        If an unknown reduction strategy is specified.
        
    Notes
    -----
    The function uses estimate_g_cutoff internally for each wrapper to determine
    individual cutoff values before applying the reduction strategy.

    """
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

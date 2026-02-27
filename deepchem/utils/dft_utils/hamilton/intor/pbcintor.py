from __future__ import annotations
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, Union, Dict, Tuple
from deepchem.utils.dft_utils import LibcintWrapper
from deepchem.utils.dft_utils.hamilton.intor.molintor import _check_and_set
from deepchem.utils.dft_utils.hamilton.intor.utils import NDIM


@dataclass
class PBCIntOption:
    """PBCIntOption is a class that contains parameters for the PBC integrals.

    Examples
    --------
    >>> from deepchem.utils.dft_utils import PBCIntOption
    >>> pbc = PBCIntOption()
    >>> pbc.get_default()
    PBCIntOption(precision=1e-08, kpt_diff_tol=1e-06)

    Attributes
    ----------
    precision: float (default 1e-8)
        Precision of the integral to limit the lattice sum.
    kpt_diff_tol: float (default 1e-6)
        Difference between k-points to be regarded as the same.

    """
    precision: float = 1e-8
    kpt_diff_tol: float = 1e-6

    @staticmethod
    def get_default(
        lattsum_opt: Optional[Union[PBCIntOption,
                                    Dict]] = None) -> PBCIntOption:
        """Get the default PBCIntOption object.

        Parameters
        ----------
        lattsum_opt: Optional[Union[PBCIntOption, Dict]]
            The lattice sum option. If it is a dictionary, then it will be
            converted to a PBCIntOption object. If it is None, then just use
            the default value of PBCIntOption.

        Returns
        -------
        PBCIntOption
            The default PBCIntOption object.

        """
        if lattsum_opt is None:
            return PBCIntOption()
        elif isinstance(lattsum_opt, dict):
            return PBCIntOption(**lattsum_opt)
        else:
            return lattsum_opt


# helper functions
def get_default_options(options: Optional[PBCIntOption] = None) -> PBCIntOption:
    """if options is None, then set the default option.
    otherwise, just return the input options.

    Examples
    --------
    >>> from deepchem.utils.dft_utils import get_default_options
    >>> get_default_options()
    PBCIntOption(precision=1e-08, kpt_diff_tol=1e-06)

    Parameters
    ----------
    options: Optional[PBCIntOption]
        Input options object

    Returns
    -------
    PBCIntOption
        The options object

    """
    if options is None:
        options1 = PBCIntOption()
    else:
        options1 = options
    return options1


def get_default_kpts(kpts: Optional[torch.Tensor], dtype: torch.dtype,
                     device: torch.device) -> torch.Tensor:
    """if kpts is None, then set the default kpts (k = zeros)
    otherwise, just return the input kpts in the correct dtype and device

    Examples
    --------
    >>> from deepchem.utils.dft_utils import get_default_kpts
    >>> get_default_kpts(torch.tensor([[1, 1, 1]]), torch.float64, 'cpu')
    tensor([[1., 1., 1.]], dtype=torch.float64)

    Parameters
    ----------
    kpts: Optional[torch.Tensor]
        Input k-points
    dtype: torch.dtype
        The dtype of the kpts
    device: torch.device
        Device on which the tensord are located. Ex: cuda, cpu

    Returns
    -------
    torch.Tensor
        Default k-points

    """
    if kpts is None:
        kpts1 = torch.zeros((1, NDIM), dtype=dtype, device=device)
    else:
        kpts1 = kpts.to(dtype).to(device)
        assert kpts1.ndim == 2
        assert kpts1.shape[-1] == NDIM
    return kpts1


def _check_and_set_pbc(wrapper: LibcintWrapper,
                       other: Optional[LibcintWrapper]) -> LibcintWrapper:
    """Check and set the `other` parameter for PBC integrals.

    This function verifies that the `other` parameter is compatible with the `wrapper`
    for periodic boundary condition calculations, then returns the appropriate
    `other` parameter (sets to `wrapper` if it is `None`).

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.dft_utils.hamilton.intor.pbcintor import _check_and_set_pbc
    >>> from deepchem.utils.dft_utils import LibcintWrapper, AtomCGTOBasis, loadbasis
    >>> from deepchem.utils.dft_utils.hamilton.intor.lattice import Lattice
    >>> # Create a shared lattice
    >>> a = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]], dtype=torch.float64)
    >>> lattice = Lattice(a)
    >>> # Create two atoms with shared basis
    >>> pos1 = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    >>> pos2 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    >>> basis = loadbasis("1:STO-3G", dtype=torch.float64, requires_grad=False)
    >>> atom1 = AtomCGTOBasis(atomz=1, bases=basis, pos=pos1)
    >>> atom2 = AtomCGTOBasis(atomz=1, bases=basis, pos=pos2)
    >>> # Create a single wrapper and get subsets (ensures same parent)
    >>> combined_wrapper = LibcintWrapper([atom1, atom2], spherical=True, lattice=lattice)
    >>> wrapper = combined_wrapper[:1]  # First atom
    >>> other_wrapper = combined_wrapper[1:]  # Second atom
    >>> # Case 1: other is None - returns wrapper
    >>> result1 = _check_and_set_pbc(wrapper, None)
    >>> result1 is wrapper
    True
    >>> # Case 2: other is provided and compatible (same lattice)
    >>> result2 = _check_and_set_pbc(wrapper, other_wrapper)
    >>> result2 is other_wrapper
    True
    >>> print(result2.lattice is wrapper.lattice)
    True

    Parameters
    ----------
    wrapper : LibcintWrapper
        Primary wrapper object containing lattice information.
    other : Optional[LibcintWrapper]
        Secondary wrapper object to be checked for compatibility.

    Returns
    -------
    LibcintWrapper
        The validated `other` parameter.

    Raises
    ------
    AssertionError
        If the lattice of `other` is not the same as `wrapper`.
    """
    other1 = _check_and_set(wrapper, other)
    assert other1.lattice is wrapper.lattice
    return other1


def _concat_atm_bas_env(*wrappers: LibcintWrapper) -> Tuple[np.ndarray, ...]:
    """Concatenate atm, bas, and env arrays from multiple LibcintWrapper objects.

    This function combines the libcint integral parameters (atm, bas, env) from
    multiple wrappers into unified arrays, with proper offset adjustments for
    atom indices, basis function indices, and environment parameters.

    Parameters
    ----------
    *wrappers : LibcintWrapper
        Variable number of LibcintWrapper objects to concatenate. Must provide
        at least 2 wrappers.

    Returns
    -------
    Tuple[np.ndarray, ...]
        - all_atm: Concatenated atm array with adjusted atom coordinate indices
        - all_bas: Concatenated bas array with adjusted atom and parameter indices
        - all_env: Concatenated env array
        - ao_loc: Combined AO location array for shell indexing

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.dft_utils import LibcintWrapper, AtomCGTOBasis, loadbasis, Lattice
    >>> from deepchem.utils.dft_utils.hamilton.intor.pbcintor import _concat_atm_bas_env
    >>> # Create two atoms with the same basis
    >>> pos1 = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    >>> pos2 = torch.tensor([1.5, 0.0, 0.0], dtype=torch.float64)
    >>> basis = loadbasis("1:STO-3G", dtype=torch.float64, requires_grad=False)
    >>> atom1 = AtomCGTOBasis(atomz=1, bases=basis, pos=pos1)
    >>> atom2 = AtomCGTOBasis(atomz=1, bases=basis, pos=pos2)
    >>> # Create wrappers for each atom
    >>> wrapper1 = LibcintWrapper([atom1], spherical=True, lattice=None)
    >>> wrapper2 = LibcintWrapper([atom2], spherical=True, lattice=None)
    >>> # Concatenate atm, bas, env arrays
    >>> atm, bas, env, ao_loc = _concat_atm_bas_env(wrapper1, wrapper2)
    >>> atm.shape  # (2, 6) - 2 atoms with 6 fields each
    (2, 6)
    >>> bas.shape  # (2, 8) - 2 shells with 8 fields each
    (2, 8)
    >>> env.shape  # Combined environment array
    (60,)
    >>> ao_loc.shape  # (3,) - 3 shells -> 3 AO locations
    (3,)

    """
    assert len(wrappers) >= 2

    PTR_COORD = 1
    PTR_ZETA = 3
    ATOM_OF = 0
    PTR_EXP = 5
    PTR_COEFF = 6

    atm0, bas0, env0 = wrappers[0].atm_bas_env
    atms = [atm0]
    bass = [bas0]
    envs = [env0]
    ao_locs = [wrappers[0].full_shell_to_aoloc]

    offset = len(env0)
    natm_offset = len(atm0)
    for i in range(1, len(wrappers)):

        wrapper = wrappers[i]
        atm, bas, env = wrapper.atm_bas_env
        atm = np.copy(atm)
        bas = np.copy(bas)
        atm[:, PTR_COORD] += offset
        atm[:, PTR_ZETA] += offset
        bas[:, ATOM_OF] += natm_offset
        bas[:, PTR_EXP] += offset
        bas[:, PTR_COEFF] += offset

        ao_locs.append(wrapper.full_shell_to_aoloc[1:] + ao_locs[-1][-1])
        atms.append(atm)
        bass.append(bas)
        envs.append(env)

        offset += len(env)
        natm_offset += len(atm)

    all_atm = np.asarray(np.vstack(atms), dtype=np.int32)
    all_bas = np.asarray(np.vstack(bass), dtype=np.int32)
    all_env = np.hstack(envs)
    ao_loc = np.concatenate(ao_locs)

    return (all_atm, all_bas, all_env, ao_loc)

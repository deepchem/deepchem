from typing import Optional, List, Tuple, Callable, Dict, no_type_check
import ctypes
import copy
import operator
from functools import reduce
import numpy as np
import torch
from deepchem.utils.dft_utils import LibcintWrapper
from deepchem.utils.dft_utils.hamilton.intor.utils import np2ctypes, int2ctypes, NDIM, CINT, CGTO
from deepchem.utils.dft_utils.hamilton.intor.namemgr import IntorNameManager


def int1e(shortname: str,
          wrapper: LibcintWrapper,
          other: Optional[LibcintWrapper] = None,
          *,
          rinv_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Calculate 2-centre 1-electron integrals.

    Examples
    --------
    >>> from deepchem.utils.dft_utils import AtomCGTOBasis, LibcintWrapper, loadbasis
    >>> dtype = torch.double
    >>> d = 1.0
    >>> pos_requires_grad = True
    >>> pos1 = torch.tensor([0.1 * d,  0.0 * d,  0.2 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos2 = torch.tensor([0.0 * d,  1.0 * d, -0.4 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos3 = torch.tensor([0.2 * d, -1.4 * d, -0.9 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> poss = [pos1, pos2, pos3]
    >>> atomzs = [1, 1, 1]
    >>> allbases = [
    ...     loadbasis("%d:%s" % (max(atomz, 1), "3-21G"), dtype=dtype, requires_grad=False)
    ...     for atomz in atomzs
    ... ]
    >>> atombases = [
    ...     AtomCGTOBasis(atomz=atomzs[i], bases=allbases[i], pos=poss[i])
    ...     for i in range(len(allbases))
    ... ]
    >>> env = LibcintWrapper(atombases, True, None)
    >>> int1e("r0", env).shape
    torch.Size([3, 6, 6])
    >>> int1e("r0r0", env).shape
    torch.Size([9, 6, 6])
    >>> int1e("r0r0r0", env).shape
    torch.Size([27, 6, 6])

    Parameters
    ----------
    shortname : str
        Short name of the integral.
    wrapper : LibcintWrapper
        Wrapper object containing the atomic information.
    other : Optional[LibcintWrapper], optional
        Other wrapper object if necessary, by default None.
    rinv_pos : Optional[torch.Tensor], optional
        Tensor containing positions if shortname contains 'rinv', by default None.

    Returns
    -------
    torch.Tensor
        Tensor containing the calculated integrals.
    """
    # check and set the other parameters
    other1 = _check_and_set(wrapper, other)

    # set the rinv_pos arguments
    if "rinv" in shortname:
        assert isinstance(
            rinv_pos, torch.Tensor), "The keyword rinv_pos must be specified"
    else:
        # don't really care, it will be ignored
        rinv_pos = torch.zeros(1, dtype=wrapper.dtype, device=wrapper.device)

    return _Int2cFunction.apply(*wrapper.params, rinv_pos, [wrapper, other1],
                                IntorNameManager("int1e", shortname))


def int2c2e(shortname: str,
            wrapper: LibcintWrapper,
            other: Optional[LibcintWrapper] = None) -> torch.Tensor:
    """
    Calculate 2-centre 2-electron integrals where the `wrapper` and `other1` correspond
    to the first electron, and `other2` corresponds to another electron.
    The returned indices are sorted based on `wrapper`, `other1`, and `other2`.
    The available shortname: "ar12", "ipip1"

    Examples
    --------
    >>> from deepchem.utils.dft_utils import AtomCGTOBasis, LibcintWrapper, loadbasis
    >>> dtype = torch.double
    >>> d = 1.0
    >>> pos_requires_grad = True
    >>> pos1 = torch.tensor([0.1 * d,  0.0 * d,  0.2 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos2 = torch.tensor([0.0 * d,  1.0 * d, -0.4 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos3 = torch.tensor([0.2 * d, -1.4 * d, -0.9 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> poss = [pos1, pos2, pos3]
    >>> atomzs = [1, 1, 1]
    >>> allbases = [
    ...     loadbasis("%d:%s" % (max(atomz, 1), "3-21G"), dtype=dtype, requires_grad=False)
    ...     for atomz in atomzs
    ... ]
    >>> atombases = [
    ...     AtomCGTOBasis(atomz=atomzs[i], bases=allbases[i], pos=poss[i])
    ...     for i in range(len(allbases))
    ... ]
    >>> env = LibcintWrapper(atombases, True, None)
    >>> int2c2e("ipip1", env).shape
    torch.Size([3, 3, 6, 6])

    Parameters
    ----------
    shortname : str
        Short name of the integral.
    wrapper : LibcintWrapper
        Wrapper object containing the atomic information.
    other : Optional[LibcintWrapper], optional
        Other wrapper object if necessary, by default None.

    Returns
    -------
    torch.Tensor
        Tensor containing the calculated integrals.
    """

    # don't really care, it will be ignored
    rinv_pos = torch.zeros(1, dtype=wrapper.dtype, device=wrapper.device)

    # check and set the others
    otherw = _check_and_set(wrapper, other)
    return _Int2cFunction.apply(*wrapper.params, rinv_pos, [wrapper, otherw],
                                IntorNameManager("int2c2e", shortname))


def int3c2e(shortname: str,
            wrapper: LibcintWrapper,
            other1: Optional[LibcintWrapper] = None,
            other2: Optional[LibcintWrapper] = None) -> torch.Tensor:
    """
    Calculate 3-centre 2-electron integrals where the `wrapper` and `other1` correspond
    to the first electron, and `other2` corresponds to another electron.
    The returned indices are sorted based on `wrapper`, `other1`, and `other2`.
    The available shortname: "ar12"

    Examples
    --------
    >>> from deepchem.utils.dft_utils import AtomCGTOBasis, LibcintWrapper, loadbasis
    >>> dtype = torch.double
    >>> d = 1.0
    >>> pos_requires_grad = True
    >>> pos1 = torch.tensor([0.1 * d,  0.0 * d,  0.2 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos2 = torch.tensor([0.0 * d,  1.0 * d, -0.4 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos3 = torch.tensor([0.2 * d, -1.4 * d, -0.9 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> poss = [pos1, pos2, pos3]
    >>> atomzs = [1, 1, 1]
    >>> allbases = [
    ...     loadbasis("%d:%s" % (max(atomz, 1), "3-21G"), dtype=dtype, requires_grad=False)
    ...     for atomz in atomzs
    ... ]
    >>> atombases = [
    ...     AtomCGTOBasis(atomz=atomzs[i], bases=allbases[i], pos=poss[i])
    ...     for i in range(len(allbases))
    ... ]
    >>> env = LibcintWrapper(atombases, True, None)
    >>> int3c2e("ar12", env).shape
    torch.Size([6, 6, 6])

    Parameters
    ----------
    shortname : str
        Short name of the integral.
    wrapper : LibcintWrapper
        Wrapper object containing the atomic information.
    other1 : Optional[LibcintWrapper], optional
        Other wrapper object if necessary for the first electron, by default None.
    other2 : Optional[LibcintWrapper], optional
        Other wrapper object if necessary for the second electron, by default None.

    Returns
    -------
    torch.Tensor
        Tensor containing the calculated integrals.
    """

    # check and set the others
    other1w = _check_and_set(wrapper, other1)
    other2w = _check_and_set(wrapper, other2)
    return _Int3cFunction.apply(*wrapper.params, [wrapper, other1w, other2w],
                                IntorNameManager("int3c2e", shortname))


def int2e(shortname: str,
          wrapper: LibcintWrapper,
          other1: Optional[LibcintWrapper] = None,
          other2: Optional[LibcintWrapper] = None,
          other3: Optional[LibcintWrapper] = None) -> torch.Tensor:
    """
    Calculate 4-centre 2-electron integrals where the `wrapper` and `other1` correspond
    to the first electron, and `other2` and `other3` correspond to another
    electron.
    The returned indices are sorted based on `wrapper`, `other1`, `other2`, and `other3`.
    The available shortname: "ar12b"

    Examples
    --------
    >>> from deepchem.utils.dft_utils import AtomCGTOBasis, LibcintWrapper, loadbasis
    >>> dtype = torch.double
    >>> d = 1.0
    >>> pos_requires_grad = True
    >>> pos1 = torch.tensor([0.1 * d,  0.0 * d,  0.2 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos2 = torch.tensor([0.0 * d,  1.0 * d, -0.4 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos3 = torch.tensor([0.2 * d, -1.4 * d, -0.9 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> poss = [pos1, pos2, pos3]
    >>> atomzs = [1, 1, 1]
    >>> allbases = [
    ...     loadbasis("%d:%s" % (max(atomz, 1), "3-21G"), dtype=dtype, requires_grad=False)
    ...     for atomz in atomzs
    ... ]
    >>> atombases = [
    ...     AtomCGTOBasis(atomz=atomzs[i], bases=allbases[i], pos=poss[i])
    ...     for i in range(len(allbases))
    ... ]
    >>> env = LibcintWrapper(atombases, True, None)
    >>> int2e("ar12b", env).shape
    torch.Size([6, 6, 6, 6])

    Parameters
    ----------
    shortname : str
        Short name of the integral.
    wrapper : LibcintWrapper
        Wrapper object containing the atomic information.
    other1 : Optional[LibcintWrapper], optional
        Other wrapper object if necessary for the first electron, by default None.
    other2 : Optional[LibcintWrapper], optional
        Other wrapper object if necessary for the second electron, by default None.
    other3 : Optional[LibcintWrapper], optional
        Other wrapper object if necessary for the third electron, by default None.

    Returns
    -------
    torch.Tensor
        Tensor containing the calculated integrals.
    """

    # check and set the others
    other1w = _check_and_set(wrapper, other1)
    other2w = _check_and_set(wrapper, other2)
    other3w = _check_and_set(wrapper, other3)
    return _Int4cFunction.apply(*wrapper.params,
                                [wrapper, other1w, other2w, other3w],
                                IntorNameManager("int2e", shortname))


# shortcuts
def overlap(wrapper: LibcintWrapper,
            other: Optional[LibcintWrapper] = None) -> torch.Tensor:
    """
    Shortcut for calculating overlap integrals.

    Examples
    --------
    >>> from deepchem.utils.dft_utils import AtomCGTOBasis, LibcintWrapper, loadbasis
    >>> dtype = torch.double
    >>> d = 1.0
    >>> pos_requires_grad = True
    >>> pos1 = torch.tensor([0.1 * d,  0.0 * d,  0.2 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos2 = torch.tensor([0.0 * d,  1.0 * d, -0.4 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos3 = torch.tensor([0.2 * d, -1.4 * d, -0.9 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> poss = [pos1, pos2, pos3]
    >>> atomzs = [1, 1, 1]
    >>> allbases = [
    ...     loadbasis("%d:%s" % (max(atomz, 1), "3-21G"), dtype=dtype, requires_grad=False)
    ...     for atomz in atomzs
    ... ]
    >>> atombases = [
    ...     AtomCGTOBasis(atomz=atomzs[i], bases=allbases[i], pos=poss[i])
    ...     for i in range(len(allbases))
    ... ]
    >>> env = LibcintWrapper(atombases, True, None)
    >>> overlap(env).shape
    torch.Size([6, 6])

    Parameters
    ----------
    wrapper : LibcintWrapper
        Wrapper object containing the atomic information.
    other : Optional[LibcintWrapper], optional
        Other wrapper object if necessary, by default None.

    Returns
    -------
    torch.Tensor
        Tensor containing the calculated overlap integrals.
    """
    return int1e("ovlp", wrapper, other=other)


def kinetic(wrapper: LibcintWrapper,
            other: Optional[LibcintWrapper] = None) -> torch.Tensor:
    """
    Shortcut for calculating kinetic energy integrals.

    Examples
    --------
    >>> from deepchem.utils.dft_utils import AtomCGTOBasis, LibcintWrapper, loadbasis
    >>> dtype = torch.double
    >>> d = 1.0
    >>> pos_requires_grad = True
    >>> pos1 = torch.tensor([0.1 * d,  0.0 * d,  0.2 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos2 = torch.tensor([0.0 * d,  1.0 * d, -0.4 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos3 = torch.tensor([0.2 * d, -1.4 * d, -0.9 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> poss = [pos1, pos2, pos3]
    >>> atomzs = [1, 1, 1]
    >>> allbases = [
    ...     loadbasis("%d:%s" % (max(atomz, 1), "3-21G"), dtype=dtype, requires_grad=False)
    ...     for atomz in atomzs
    ... ]
    >>> atombases = [
    ...     AtomCGTOBasis(atomz=atomzs[i], bases=allbases[i], pos=poss[i])
    ...     for i in range(len(allbases))
    ... ]
    >>> env = LibcintWrapper(atombases, True, None)
    >>> kinetic(env).shape
    torch.Size([6, 6])

    Parameters
    ----------
    wrapper : LibcintWrapper
        Wrapper object containing the atomic information.
    other : Optional[LibcintWrapper], optional
        Other wrapper object if necessary, by default None.

    Returns
    -------
    torch.Tensor
        Tensor containing the calculated kinetic energy integrals.
    """
    return int1e("kin", wrapper, other=other)


def nuclattr(wrapper: LibcintWrapper,
             other: Optional[LibcintWrapper] = None) -> torch.Tensor:
    """
    Shortcut for calculating nuclear attraction integrals.

    Examples
    --------
    >>> from deepchem.utils.dft_utils import AtomCGTOBasis, LibcintWrapper, loadbasis
    >>> dtype = torch.double
    >>> d = 1.0
    >>> pos_requires_grad = True
    >>> pos1 = torch.tensor([0.1 * d,  0.0 * d,  0.2 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos2 = torch.tensor([0.0 * d,  1.0 * d, -0.4 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos3 = torch.tensor([0.2 * d, -1.4 * d, -0.9 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> poss = [pos1, pos2, pos3]
    >>> atomzs = [1, 1, 1]
    >>> allbases = [
    ...     loadbasis("%d:%s" % (max(atomz, 1), "3-21G"), dtype=dtype, requires_grad=False)
    ...     for atomz in atomzs
    ... ]
    >>> atombases = [
    ...     AtomCGTOBasis(atomz=atomzs[i], bases=allbases[i], pos=poss[i])
    ...     for i in range(len(allbases))
    ... ]
    >>> env = LibcintWrapper(atombases, True, None)
    >>> nuclattr(env).shape
    torch.Size([6, 6])

    Parameters
    ----------
    wrapper : LibcintWrapper
        Wrapper object containing the atomic information.
    other : Optional[LibcintWrapper], optional
        Other wrapper object if necessary, by default None.

    Returns
    -------
    torch.Tensor
        Tensor containing the calculated nuclear attraction integrals.
    """
    if not wrapper.fracz:
        return int1e("nuc", wrapper, other=other)
    else:
        res = torch.tensor([])
        allpos_params = wrapper.params[-1]
        for i in range(wrapper.natoms):
            y = int1e("rinv", wrapper, other=other, rinv_pos=allpos_params[i]) * \
                (-wrapper.atombases[i].atomz)
            res = y if (i == 0) else (res + y)
        return res


def elrep(
    wrapper: LibcintWrapper,
    other1: Optional[LibcintWrapper] = None,
    other2: Optional[LibcintWrapper] = None,
    other3: Optional[LibcintWrapper] = None,
) -> torch.Tensor:
    """
    Calculate electron repulsion integrals with three additional wrappers.

    Examples
    --------
    >>> from deepchem.utils.dft_utils import AtomCGTOBasis, LibcintWrapper, loadbasis
    >>> dtype = torch.double
    >>> d = 1.0
    >>> pos_requires_grad = True
    >>> pos1 = torch.tensor([0.1 * d,  0.0 * d,  0.2 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos2 = torch.tensor([0.0 * d,  1.0 * d, -0.4 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos3 = torch.tensor([0.2 * d, -1.4 * d, -0.9 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> poss = [pos1, pos2, pos3]
    >>> atomzs = [1, 1, 1]
    >>> allbases = [
    ...     loadbasis("%d:%s" % (max(atomz, 1), "3-21G"), dtype=dtype, requires_grad=False)
    ...     for atomz in atomzs
    ... ]
    >>> atombases = [
    ...     AtomCGTOBasis(atomz=atomzs[i], bases=allbases[i], pos=poss[i])
    ...     for i in range(len(allbases))
    ... ]
    >>> env = LibcintWrapper(atombases, True, None)
    >>> elrep(env).shape
    torch.Size([6, 6, 6, 6])

    Parameters
    ----------
    wrapper : LibcintWrapper
        Wrapper object containing the atomic information.
    other1 : Optional[LibcintWrapper], optional
        Other wrapper object if necessary for the first electron, by default None.
    other2 : Optional[LibcintWrapper], optional
        Other wrapper object if necessary for the second electron, by default None.
    other3 : Optional[LibcintWrapper], optional
        Other wrapper object if necessary for the third electron, by default None.

    Returns
    -------
    torch.Tensor
        Tensor containing the calculated electron repulsion integrals.
    """
    return int2e("ar12b", wrapper, other1, other2, other3)


def coul2c(
    wrapper: LibcintWrapper,
    other: Optional[LibcintWrapper] = None,
) -> torch.Tensor:
    """
    Calculate 2-centre Coulomb integrals.

    Examples
    --------
    >>> from deepchem.utils.dft_utils import AtomCGTOBasis, LibcintWrapper, loadbasis
    >>> dtype = torch.double
    >>> d = 1.0
    >>> pos_requires_grad = True
    >>> pos1 = torch.tensor([0.1 * d,  0.0 * d,  0.2 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos2 = torch.tensor([0.0 * d,  1.0 * d, -0.4 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos3 = torch.tensor([0.2 * d, -1.4 * d, -0.9 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> poss = [pos1, pos2, pos3]
    >>> atomzs = [1, 1, 1]
    >>> allbases = [
    ...     loadbasis("%d:%s" % (max(atomz, 1), "3-21G"), dtype=dtype, requires_grad=False)
    ...     for atomz in atomzs
    ... ]
    >>> atombases = [
    ...     AtomCGTOBasis(atomz=atomzs[i], bases=allbases[i], pos=poss[i])
    ...     for i in range(len(allbases))
    ... ]
    >>> env = LibcintWrapper(atombases, True, None)
    >>> coul2c(env).shape
    torch.Size([6, 6])

    Parameters
    ----------
    wrapper : LibcintWrapper
        Wrapper object containing the atomic information.
    other : Optional[LibcintWrapper], optional
        Other wrapper object if necessary, by default None.

    Returns
    -------
    torch.Tensor
        Tensor containing the calculated 2-centre Coulomb integrals.
    """
    return int2c2e("r12", wrapper, other)


def coul3c(
    wrapper: LibcintWrapper,
    other1: Optional[LibcintWrapper] = None,
    other2: Optional[LibcintWrapper] = None,
) -> torch.Tensor:
    """
    Calculate 3-centre Coulomb integrals.

    Examples
    --------
    >>> from deepchem.utils.dft_utils import AtomCGTOBasis, LibcintWrapper, loadbasis
    >>> dtype = torch.double
    >>> d = 1.0
    >>> pos_requires_grad = True
    >>> pos1 = torch.tensor([0.1 * d,  0.0 * d,  0.2 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos2 = torch.tensor([0.0 * d,  1.0 * d, -0.4 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos3 = torch.tensor([0.2 * d, -1.4 * d, -0.9 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> poss = [pos1, pos2, pos3]
    >>> atomzs = [1, 1, 1]
    >>> allbases = [
    ...     loadbasis("%d:%s" % (max(atomz, 1), "3-21G"), dtype=dtype, requires_grad=False)
    ...     for atomz in atomzs
    ... ]
    >>> atombases = [
    ...     AtomCGTOBasis(atomz=atomzs[i], bases=allbases[i], pos=poss[i])
    ...     for i in range(len(allbases))
    ... ]
    >>> env = LibcintWrapper(atombases, True, None)
    >>> coul3c(env).shape
    torch.Size([6, 6, 6])

    Parameters
    ----------
    wrapper : LibcintWrapper
        Wrapper object containing the atomic information.
    other1 : Optional[LibcintWrapper], optional
        Other wrapper object if necessary for the first electron, by default None.
    other2 : Optional[LibcintWrapper], optional
        Other wrapper object if necessary for the second electron, by default None.

    Returns
    -------
    torch.Tensor
        Tensor containing the calculated 3-centre Coulomb integrals.
    """
    return int3c2e("ar12", wrapper, other1, other2)


# misc functions
def _check_and_set(wrapper: LibcintWrapper,
                   other: Optional[LibcintWrapper]) -> LibcintWrapper:
    """
    Helper function to check and set parameters if necessary.

    Parameters
    ----------
    wrapper : LibcintWrapper
        Wrapper object containing the atomic information.
    other : Optional[LibcintWrapper]
        Other wrapper object if necessary.

    Returns
    -------
    LibcintWrapper
        Other wrapper object if necessary.
    """
    # check the value and set the default value of "other" in the integrals
    if other is not None:
        atm0, bas0, env0 = wrapper.atm_bas_env
        atm1, bas1, env1 = other.atm_bas_env
        msg = (
            "Argument `other*` does not have the same parent as the wrapper. "
            "Please do `LibcintWrapper.concatenate` on those wrappers first.")
        assert id(atm0) == id(atm1), msg
        assert id(bas0) == id(bas1), msg
        assert id(env0) == id(env1), msg
    else:
        other = wrapper
    assert isinstance(other, LibcintWrapper)
    return other


class _Int2cFunction(torch.autograd.Function):
    """wrapper class to provide the gradient of the 2-centre integrals"""

    @staticmethod
    def forward(
            ctx,  # type: ignore
            allcoeffs: torch.Tensor,
            allalphas: torch.Tensor,
            allposs: torch.Tensor,
            rinv_pos: torch.Tensor,
            wrappers: List[LibcintWrapper],
            int_nmgr: IntorNameManager) -> torch.Tensor:
        """Forward calculation of the 2-centre integrals.

        Parameters
        ----------
        allcoeffs : torch.Tensor
            Coefficients of the basis functions.
        allalphas : torch.Tensor
            Exponents of the basis functions.
        allposs : torch.Tensor
            Atomic positions.
        rinv_pos : torch.Tensor
            Positions for the rinv integrals.
        wrappers : List[LibcintWrapper]
            List of wrapper objects.
        int_nmgr : IntorNameManager
            Integral name manager object.

        Returns
        -------
        torch.Tensor
            Tensor containing the calculated integrals.

        """
        assert len(wrappers) == 2

        if int_nmgr.rawopname == "rinv":
            assert rinv_pos.ndim == 1 and rinv_pos.shape[0] == NDIM
            with wrappers[0].centre_on_r(rinv_pos):
                out_tensor = Intor(int_nmgr, wrappers).calc()
        else:
            out_tensor = Intor(int_nmgr, wrappers).calc()
        ctx.save_for_backward(allcoeffs, allalphas, allposs, rinv_pos)
        ctx.other_info = (wrappers, int_nmgr)
        return out_tensor  # (..., nao0, nao1)

    @no_type_check
    @staticmethod
    def backward(ctx,
                 grad_out: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """Backward calculation of the 2-centre integrals.

        Parameters
        ----------
        grad_out : torch.Tensor
            Gradient of the output tensor.

        Returns
        -------
        Tuple[Optional[torch.Tensor], ...]
            Tuple containing the gradients of the basis coefficients, basis exponents,

        """
        # grad_out: (..., nao0, nao1)
        allcoeffs, allalphas, allposs, \
            rinv_pos = ctx.saved_tensors
        wrappers, int_nmgr = ctx.other_info

        # gradient for all atomic positions
        grad_allposs: Optional[torch.Tensor] = None
        if allposs.requires_grad:
            grad_allposs = torch.zeros_like(allposs)  # (natom, ndim)
            grad_allpossT = grad_allposs.transpose(-2, -1)  # (ndim, natom)

            # get the integrals required for the derivatives
            sname_derivs = [
                int_nmgr.get_intgl_deriv_namemgr("ip", ib) for ib in (0, 1)
            ]
            # new axes added to the dimension
            new_axes_pos = [
                int_nmgr.get_intgl_deriv_newaxispos("ip", ib) for ib in (0, 1)
            ]

            def int_fcn(wrappers, namemgr):
                return _Int2cFunction.apply(*ctx.saved_tensors, wrappers,
                                            namemgr)

            # list of tensors with shape: (ndim, ..., nao0, nao1)
            dout_dposs = _get_integrals(sname_derivs, wrappers, int_fcn,
                                        new_axes_pos)

            ndim = dout_dposs[0].shape[0]
            shape = (ndim, -1, *dout_dposs[0].shape[-2:])
            grad_out2 = grad_out.reshape(shape[1:])
            # negative because the integral calculates the nabla w.r.t. the
            # spatial coordinate, not the basis central position
            grad_dpos_i = -torch.einsum("sij,dsij->di", grad_out2,
                                        dout_dposs[0].reshape(shape))
            grad_dpos_j = -torch.einsum("sij,dsij->dj", grad_out2,
                                        dout_dposs[1].reshape(shape))

            # grad_allpossT is only a view of grad_allposs, so the operation below
            # also changes grad_allposs
            ao_to_atom0 = wrappers[0].ao_to_atom().expand(ndim, -1)
            ao_to_atom1 = wrappers[1].ao_to_atom().expand(ndim, -1)
            grad_allpossT.scatter_add_(dim=-1,
                                       index=ao_to_atom0,
                                       src=grad_dpos_i)
            grad_allpossT.scatter_add_(dim=-1,
                                       index=ao_to_atom1,
                                       src=grad_dpos_j)

            grad_allposs_nuc = torch.zeros_like(grad_allposs)
            if "nuc" == int_nmgr.rawopname:
                # allposs: (natoms, ndim)
                natoms = allposs.shape[0]
                sname_rinv = int_nmgr.shortname.replace("nuc", "rinv")
                int_nmgr_rinv = IntorNameManager(int_nmgr.int_type, sname_rinv)
                sname_derivs = [
                    int_nmgr_rinv.get_intgl_deriv_namemgr("ip", ib)
                    for ib in (0, 1)
                ]
                new_axes_pos = [
                    int_nmgr_rinv.get_intgl_deriv_newaxispos("ip", ib)
                    for ib in (0, 1)
                ]

                for i in range(natoms):
                    atomz = wrappers[0].atombases[i].atomz

                    # get the integrals
                    def int_fcn(wrappers, namemgr):
                        return _Int2cFunction.apply(allcoeffs, allalphas,
                                                    allposs, allposs[i],
                                                    wrappers, namemgr)

                    dout_datposs = _get_integrals(
                        sname_derivs, wrappers, int_fcn,
                        new_axes_pos)  # (ndim, ..., nao, nao)

                    grad_datpos = grad_out * (dout_datposs[0] + dout_datposs[1])
                    grad_datpos = grad_datpos.reshape(grad_datpos.shape[0],
                                                      -1).sum(dim=-1)
                    grad_allposs_nuc[i] = (-atomz) * grad_datpos

                grad_allposs += grad_allposs_nuc

        # gradient for the rinv_pos in rinv integral
        grad_rinv_pos: Optional[torch.Tensor] = None
        if rinv_pos.requires_grad and "rinv" == int_nmgr.rawopname:
            # rinv_pos: (ndim)
            # get the integrals for the derivatives
            sname_derivs = [
                int_nmgr.get_intgl_deriv_namemgr("ip", ib) for ib in (0, 1)
            ]
            # new axes added to the dimension
            new_axes_pos = [
                int_nmgr.get_intgl_deriv_newaxispos("ip", ib) for ib in (0, 1)
            ]

            def int_fcn(wrappers, namemgr):
                return _Int2cFunction.apply(*ctx.saved_tensors, wrappers,
                                            namemgr)

            dout_datposs = _get_integrals(sname_derivs, wrappers, int_fcn,
                                          new_axes_pos)

            grad_datpos = grad_out * (dout_datposs[0] + dout_datposs[1])
            grad_rinv_pos = grad_datpos.reshape(grad_datpos.shape[0],
                                                -1).sum(dim=-1)

        # gradient for the basis coefficients
        grad_allcoeffs: Optional[torch.Tensor] = None
        grad_allalphas: Optional[torch.Tensor] = None
        if allcoeffs.requires_grad or allalphas.requires_grad:
            # obtain the uncontracted wrapper and mapping
            # uao2aos: list of (nu_ao0,), (nu_ao1,)
            u_wrappers_tup, uao2aos_tup = zip(
                *[w.get_uncontracted_wrapper() for w in wrappers])
            u_wrappers = list(u_wrappers_tup)
            uao2aos = list(uao2aos_tup)
            u_params = u_wrappers[0].params

            # get the uncontracted (gathered) grad_out
            u_grad_out = _gather_at_dims(grad_out,
                                         mapidxs=uao2aos,
                                         dims=[-2, -1])

            # get the scatter indices
            ao2shl0 = u_wrappers[0].ao_to_shell()
            ao2shl1 = u_wrappers[1].ao_to_shell()

            # calculate the gradient w.r.t. coeffs
            if allcoeffs.requires_grad:
                grad_allcoeffs = torch.zeros_like(allcoeffs)  # (ngauss)

                # get the uncontracted version of the integral
                dout_dcoeff = _Int2cFunction.apply(
                    *u_params, rinv_pos, u_wrappers,
                    int_nmgr)  # (..., nu_ao0, nu_ao1)

                # get the coefficients and spread it on the u_ao-length tensor
                coeffs_ao0 = torch.gather(allcoeffs, dim=-1,
                                          index=ao2shl0)  # (nu_ao0)
                coeffs_ao1 = torch.gather(allcoeffs, dim=-1,
                                          index=ao2shl1)  # (nu_ao1)
                # divide done here instead of after scatter to make the 2nd gradient
                # calculation correct.
                # division can also be done after scatter for more efficient 1st grad
                # calculation, but it gives the wrong result for 2nd grad
                dout_dcoeff_i = dout_dcoeff / coeffs_ao0[:, None]
                dout_dcoeff_j = dout_dcoeff / coeffs_ao1

                # (nu_ao)
                grad_dcoeff_i = torch.einsum("...ij,...ij->i", u_grad_out,
                                             dout_dcoeff_i)
                grad_dcoeff_j = torch.einsum("...ij,...ij->j", u_grad_out,
                                             dout_dcoeff_j)
                # grad_dcoeff = grad_dcoeff_i + grad_dcoeff_j

                # scatter the grad
                grad_allcoeffs.scatter_add_(dim=-1,
                                            index=ao2shl0,
                                            src=grad_dcoeff_i)
                grad_allcoeffs.scatter_add_(dim=-1,
                                            index=ao2shl1,
                                            src=grad_dcoeff_j)

            # calculate the gradient w.r.t. alphas
            if allalphas.requires_grad:
                grad_allalphas = torch.zeros_like(allalphas)  # (ngauss)

                def u_int_fcn(wrappers, int_nmgr):
                    return _Int2cFunction.apply(*u_params, rinv_pos, wrappers,
                                                int_nmgr)

                # get the uncontracted integrals
                sname_derivs = [
                    int_nmgr.get_intgl_deriv_namemgr("rr", ib) for ib in (0, 1)
                ]
                new_axes_pos = [
                    int_nmgr.get_intgl_deriv_newaxispos("rr", ib)
                    for ib in (0, 1)
                ]
                dout_dalphas = _get_integrals(sname_derivs, u_wrappers,
                                              u_int_fcn, new_axes_pos)

                # (nu_ao)
                # negative because the exponent is negative alpha * (r-ra)^2
                grad_dalpha_i = -torch.einsum("...ij,...ij->i", u_grad_out,
                                              dout_dalphas[0])
                grad_dalpha_j = -torch.einsum("...ij,...ij->j", u_grad_out,
                                              dout_dalphas[1])
                # grad_dalpha = (grad_dalpha_i + grad_dalpha_j)  # (nu_ao)

                # scatter the grad
                grad_allalphas.scatter_add_(dim=-1,
                                            index=ao2shl0,
                                            src=grad_dalpha_i)
                grad_allalphas.scatter_add_(dim=-1,
                                            index=ao2shl1,
                                            src=grad_dalpha_j)

        return grad_allcoeffs, grad_allalphas, grad_allposs, \
            grad_rinv_pos, \
            None, None, None


class _Int3cFunction(torch.autograd.Function):
    """wrapper class for the 3-centre integrals"""

    @staticmethod
    def forward(
            ctx,  # type: ignore
            allcoeffs: torch.Tensor,
            allalphas: torch.Tensor,
            allposs: torch.Tensor,
            wrappers: List[LibcintWrapper],
            int_nmgr: IntorNameManager) -> torch.Tensor:
        """Forward calculation of the 3-centre integrals.

        Parameters
        ----------
        allcoeffs : torch.Tensor
            Coefficients of the basis functions.
        allalphas : torch.Tensor
            Exponents of the basis functions.
        allposs : torch.Tensor
            Atomic positions.
        wrappers : List[LibcintWrapper]
            List of wrapper objects.
        int_nmgr : IntorNameManager
            Integral name manager object.

        Returns
        -------
        torch.Tensor
            Tensor containing the calculated integrals.

        """

        assert len(wrappers) == 3

        out_tensor = Intor(int_nmgr, wrappers).calc()
        ctx.save_for_backward(allcoeffs, allalphas, allposs)
        ctx.other_info = (wrappers, int_nmgr)
        return out_tensor  # (..., nao0, nao1, nao2)

    @no_type_check
    @staticmethod
    def backward(
            ctx,
            grad_out) -> Tuple[Optional[torch.Tensor], ...]:  # type: ignore
        """Backward calculation of the 3-centre integrals.

        Parameters
        ----------
        grad_out : torch.Tensor
            Gradient of the output tensor.

        Returns
        -------
        Tuple[Optional[torch.Tensor], ...]
            Tuple containing the gradients of the basis coefficients, basis exponents,

        """
        # grad_out: (..., nao0, nao1, nao2)
        allcoeffs, allalphas, allposs = ctx.saved_tensors
        wrappers, int_nmgr = ctx.other_info
        naos = grad_out.shape[-3:]

        # calculate the gradient w.r.t. positions
        grad_allposs: Optional[torch.Tensor] = None
        if allposs.requires_grad:
            grad_allposs = torch.zeros_like(allposs)  # (natom, ndim)
            grad_allpossT = grad_allposs.transpose(-2, -1)  # (ndim, natom)

            sname_derivs = [
                int_nmgr.get_intgl_deriv_namemgr("ip", ib) for ib in (0, 1, 2)
            ]
            new_axes_pos = [
                int_nmgr.get_intgl_deriv_newaxispos("ip", ib)
                for ib in (0, 1, 2)
            ]

            def int_fcn(wrappers, namemgr):
                return _Int3cFunction.apply(*ctx.saved_tensors, wrappers,
                                            namemgr)

            dout_dposs = _get_integrals(sname_derivs, wrappers, int_fcn,
                                        new_axes_pos)

            # negative because the integral calculates the nabla w.r.t. the
            # spatial coordinate, not the basis central position
            ndim = dout_dposs[0].shape[0]
            shape = (ndim, -1, *naos)
            grad_out2 = grad_out.reshape(*shape[1:])
            grad_pos_a1 = -torch.einsum(
                "dzijk,zijk->di", dout_dposs[0].reshape(*shape), grad_out2)
            grad_pos_a2 = -torch.einsum(
                "dzijk,zijk->dj", dout_dposs[1].reshape(*shape), grad_out2)
            grad_pos_b1 = -torch.einsum(
                "dzijk,zijk->dk", dout_dposs[2].reshape(*shape), grad_out2)

            ao_to_atom0 = wrappers[0].ao_to_atom().expand(ndim, -1)
            ao_to_atom1 = wrappers[1].ao_to_atom().expand(ndim, -1)
            ao_to_atom2 = wrappers[2].ao_to_atom().expand(ndim, -1)
            grad_allpossT.scatter_add_(dim=-1,
                                       index=ao_to_atom0,
                                       src=grad_pos_a1)
            grad_allpossT.scatter_add_(dim=-1,
                                       index=ao_to_atom1,
                                       src=grad_pos_a2)
            grad_allpossT.scatter_add_(dim=-1,
                                       index=ao_to_atom2,
                                       src=grad_pos_b1)

        # gradients for the basis coefficients
        grad_allcoeffs: Optional[torch.Tensor] = None
        grad_allalphas: Optional[torch.Tensor] = None
        if allcoeffs.requires_grad or allalphas.requires_grad:
            # obtain the uncontracted wrapper, and expanded grad_out
            # uao2ao: (nu_ao)
            u_wrappers_tup, uao2aos_tup = zip(
                *[w.get_uncontracted_wrapper() for w in wrappers])
            u_wrappers = list(u_wrappers_tup)
            uao2aos = list(uao2aos_tup)
            u_params = u_wrappers[0].params

            # u_grad_out: (..., nu_ao0, nu_ao1, nu_ao2)
            u_grad_out = _gather_at_dims(grad_out,
                                         mapidxs=uao2aos,
                                         dims=[-3, -2, -1])

            # get the scatter indices
            ao2shl0 = u_wrappers[0].ao_to_shell()  # (nu_ao0,)
            ao2shl1 = u_wrappers[1].ao_to_shell()
            ao2shl2 = u_wrappers[2].ao_to_shell()

            # calculate the grad w.r.t. coeffs
            if allcoeffs.requires_grad:
                grad_allcoeffs = torch.zeros_like(allcoeffs)

                # (..., nu_ao0, nu_ao1, nu_ao2)
                dout_dcoeff = _Int3cFunction.apply(*u_params, u_wrappers,
                                                   int_nmgr)

                # get the coefficients and spread it on the u_ao-length tensor
                coeffs_ao0 = torch.gather(allcoeffs, dim=-1,
                                          index=ao2shl0)  # (nu_ao0)
                coeffs_ao1 = torch.gather(allcoeffs, dim=-1,
                                          index=ao2shl1)  # (nu_ao1)
                coeffs_ao2 = torch.gather(allcoeffs, dim=-1,
                                          index=ao2shl2)  # (nu_ao2)
                # dout_dcoeff_*: (..., nu_ao0, nu_ao1, nu_ao2, nu_ao3)
                dout_dcoeff_a1 = dout_dcoeff / coeffs_ao0[:, None, None]
                dout_dcoeff_a2 = dout_dcoeff / coeffs_ao1[:, None]
                dout_dcoeff_b1 = dout_dcoeff / coeffs_ao2

                # reduce the uncontracted integrations
                # grad_coeff_*: (nu_ao*)
                grad_coeff_a1 = torch.einsum("...ijk,...ijk->i", dout_dcoeff_a1,
                                             u_grad_out)
                grad_coeff_a2 = torch.einsum("...ijk,...ijk->j", dout_dcoeff_a2,
                                             u_grad_out)
                grad_coeff_b1 = torch.einsum("...ijk,...ijk->k", dout_dcoeff_b1,
                                             u_grad_out)
                # grad_coeff_all = grad_coeff_a1 + grad_coeff_a2 + grad_coeff_b1 + grad_coeff_b2

                # scatter to the coefficients
                grad_allcoeffs.scatter_add_(dim=-1,
                                            index=ao2shl0,
                                            src=grad_coeff_a1)
                grad_allcoeffs.scatter_add_(dim=-1,
                                            index=ao2shl1,
                                            src=grad_coeff_a2)
                grad_allcoeffs.scatter_add_(dim=-1,
                                            index=ao2shl2,
                                            src=grad_coeff_b1)

            if allalphas.requires_grad:
                grad_allalphas = torch.zeros_like(allalphas)  # (ngauss)

                # get the uncontracted integrals
                sname_derivs = [
                    int_nmgr.get_intgl_deriv_namemgr("rr", ib)
                    for ib in (0, 1, 2)
                ]
                new_axes_pos = [
                    int_nmgr.get_intgl_deriv_newaxispos("rr", ib)
                    for ib in (0, 1, 2)
                ]

                def u_int_fcn(wrappers, int_nmgr):
                    return _Int3cFunction.apply(*u_params, wrappers, int_nmgr)

                dout_dalphas = _get_integrals(sname_derivs, u_wrappers,
                                              u_int_fcn, new_axes_pos)

                # (nu_ao)
                # negative because the exponent is negative alpha * (r-ra)^2
                grad_alpha_a1 = -torch.einsum("...ijk,...ijk->i",
                                              dout_dalphas[0], u_grad_out)
                grad_alpha_a2 = -torch.einsum("...ijk,...ijk->j",
                                              dout_dalphas[1], u_grad_out)
                grad_alpha_b1 = -torch.einsum("...ijk,...ijk->k",
                                              dout_dalphas[2], u_grad_out)
                # grad_alpha_all = (grad_alpha_a1 + grad_alpha_a2 + grad_alpha_b1 + grad_alpha_b2)

                # scatter the grad
                grad_allalphas.scatter_add_(dim=-1,
                                            index=ao2shl0,
                                            src=grad_alpha_a1)
                grad_allalphas.scatter_add_(dim=-1,
                                            index=ao2shl1,
                                            src=grad_alpha_a2)
                grad_allalphas.scatter_add_(dim=-1,
                                            index=ao2shl2,
                                            src=grad_alpha_b1)

        return grad_allcoeffs, grad_allalphas, grad_allposs, \
            None, None, None


class _Int4cFunction(torch.autograd.Function):
    """wrapper class for the 4-centre integrals"""

    @staticmethod
    def forward(
            ctx,  # type: ignore
            allcoeffs: torch.Tensor,
            allalphas: torch.Tensor,
            allposs: torch.Tensor,
            wrappers: List[LibcintWrapper],
            int_nmgr: IntorNameManager) -> torch.Tensor:
        """Forward calculation of the 4-centre integrals.

        Parameters
        ----------
        allcoeffs : torch.Tensor
            Coefficients of the basis functions.
        allalphas : torch.Tensor
            Exponents of the basis functions.
        allposs : torch.Tensor
            Atomic positions.
        wrappers : List[LibcintWrapper]
            List of wrapper objects.
        int_nmgr : IntorNameManager
            Integral name manager object.

        Returns
        -------
        torch.Tensor
            Tensor containing the calculated integrals.

        """
        assert len(wrappers) == 4

        out_tensor = Intor(int_nmgr, wrappers).calc()
        ctx.save_for_backward(allcoeffs, allalphas, allposs)
        ctx.other_info = (wrappers, int_nmgr)
        return out_tensor  # (..., nao0, nao1, nao2, nao3)

    @no_type_check
    @staticmethod
    def backward(
            ctx,
            grad_out) -> Tuple[Optional[torch.Tensor], ...]:  # type: ignore
        """Backward calculation of the 4-centre integrals.

        Parameters
        ----------
        grad_out : torch.Tensor
            Gradient of the output tensor.

        Returns
        -------
        Tuple[Optional[torch.Tensor], ...]
            Tuple containing the gradients of the basis coefficients, basis exponents,

        """
        # grad_out: (..., nao0, nao1, nao2, nao3)
        allcoeffs, allalphas, allposs = ctx.saved_tensors
        wrappers, int_nmgr = ctx.other_info
        naos = grad_out.shape[-4:]

        # calculate the gradient w.r.t. positions
        grad_allposs: Optional[torch.Tensor] = None
        if allposs.requires_grad:
            grad_allposs = torch.zeros_like(allposs)  # (natom, ndim)
            grad_allpossT = grad_allposs.transpose(-2, -1)  # (ndim, natom)

            sname_derivs = [
                int_nmgr.get_intgl_deriv_namemgr("ip", ib) for ib in range(4)
            ]
            new_axes_pos = [
                int_nmgr.get_intgl_deriv_newaxispos("ip", ib) for ib in range(4)
            ]

            def int_fcn(wrappers, namemgr):
                return _Int4cFunction.apply(*ctx.saved_tensors, wrappers,
                                            namemgr)

            dout_dposs = _get_integrals(sname_derivs, wrappers, int_fcn,
                                        new_axes_pos)

            # negative because the integral calculates the nabla w.r.t. the
            # spatial coordinate, not the basis central position
            ndim = dout_dposs[0].shape[0]
            shape = (ndim, -1, *naos)
            grad_out2 = grad_out.reshape(*shape[1:])
            grad_pos_a1 = -torch.einsum(
                "dzijkl,zijkl->di", dout_dposs[0].reshape(*shape), grad_out2)
            grad_pos_a2 = -torch.einsum(
                "dzijkl,zijkl->dj", dout_dposs[1].reshape(*shape), grad_out2)
            grad_pos_b1 = -torch.einsum(
                "dzijkl,zijkl->dk", dout_dposs[2].reshape(*shape), grad_out2)
            grad_pos_b2 = -torch.einsum(
                "dzijkl,zijkl->dl", dout_dposs[3].reshape(*shape), grad_out2)

            ao_to_atom0 = wrappers[0].ao_to_atom().expand(ndim, -1)
            ao_to_atom1 = wrappers[1].ao_to_atom().expand(ndim, -1)
            ao_to_atom2 = wrappers[2].ao_to_atom().expand(ndim, -1)
            ao_to_atom3 = wrappers[3].ao_to_atom().expand(ndim, -1)
            grad_allpossT.scatter_add_(dim=-1,
                                       index=ao_to_atom0,
                                       src=grad_pos_a1)
            grad_allpossT.scatter_add_(dim=-1,
                                       index=ao_to_atom1,
                                       src=grad_pos_a2)
            grad_allpossT.scatter_add_(dim=-1,
                                       index=ao_to_atom2,
                                       src=grad_pos_b1)
            grad_allpossT.scatter_add_(dim=-1,
                                       index=ao_to_atom3,
                                       src=grad_pos_b2)

        # gradients for the basis coefficients
        grad_allcoeffs: Optional[torch.Tensor] = None
        grad_allalphas: Optional[torch.Tensor] = None
        if allcoeffs.requires_grad or allalphas.requires_grad:
            # obtain the uncontracted wrapper, and expanded grad_out
            # uao2ao: (nu_ao)
            u_wrappers_tup, uao2aos_tup = zip(
                *[w.get_uncontracted_wrapper() for w in wrappers])
            u_wrappers = list(u_wrappers_tup)
            uao2aos = list(uao2aos_tup)
            u_params = u_wrappers[0].params

            # u_grad_out: (..., nu_ao0, nu_ao1, nu_ao2, nu_ao3)
            u_grad_out = _gather_at_dims(grad_out,
                                         mapidxs=uao2aos,
                                         dims=[-4, -3, -2, -1])

            # get the scatter indices
            ao2shl0 = u_wrappers[0].ao_to_shell()  # (nu_ao0,)
            ao2shl1 = u_wrappers[1].ao_to_shell()
            ao2shl2 = u_wrappers[2].ao_to_shell()
            ao2shl3 = u_wrappers[3].ao_to_shell()

            # calculate the grad w.r.t. coeffs
            if allcoeffs.requires_grad:
                grad_allcoeffs = torch.zeros_like(allcoeffs)

                # (..., nu_ao0, nu_ao1, nu_ao2, nu_ao3)
                dout_dcoeff = _Int4cFunction.apply(*u_params, u_wrappers,
                                                   int_nmgr)

                # get the coefficients and spread it on the u_ao-length tensor
                coeffs_ao0 = torch.gather(allcoeffs, dim=-1,
                                          index=ao2shl0)  # (nu_ao0)
                coeffs_ao1 = torch.gather(allcoeffs, dim=-1,
                                          index=ao2shl1)  # (nu_ao1)
                coeffs_ao2 = torch.gather(allcoeffs, dim=-1,
                                          index=ao2shl2)  # (nu_ao2)
                coeffs_ao3 = torch.gather(allcoeffs, dim=-1,
                                          index=ao2shl3)  # (nu_ao3)
                # dout_dcoeff_*: (..., nu_ao0, nu_ao1, nu_ao2, nu_ao3)
                dout_dcoeff_a1 = dout_dcoeff / coeffs_ao0[:, None, None, None]
                dout_dcoeff_a2 = dout_dcoeff / coeffs_ao1[:, None, None]
                dout_dcoeff_b1 = dout_dcoeff / coeffs_ao2[:, None]
                dout_dcoeff_b2 = dout_dcoeff / coeffs_ao3

                # reduce the uncontracted integrations
                # grad_coeff_*: (nu_ao*)
                grad_coeff_a1 = torch.einsum("...ijkl,...ijkl->i",
                                             dout_dcoeff_a1, u_grad_out)
                grad_coeff_a2 = torch.einsum("...ijkl,...ijkl->j",
                                             dout_dcoeff_a2, u_grad_out)
                grad_coeff_b1 = torch.einsum("...ijkl,...ijkl->k",
                                             dout_dcoeff_b1, u_grad_out)
                grad_coeff_b2 = torch.einsum("...ijkl,...ijkl->l",
                                             dout_dcoeff_b2, u_grad_out)
                # grad_coeff_all = grad_coeff_a1 + grad_coeff_a2 + grad_coeff_b1 + grad_coeff_b2

                # scatter to the coefficients
                grad_allcoeffs.scatter_add_(dim=-1,
                                            index=ao2shl0,
                                            src=grad_coeff_a1)
                grad_allcoeffs.scatter_add_(dim=-1,
                                            index=ao2shl1,
                                            src=grad_coeff_a2)
                grad_allcoeffs.scatter_add_(dim=-1,
                                            index=ao2shl2,
                                            src=grad_coeff_b1)
                grad_allcoeffs.scatter_add_(dim=-1,
                                            index=ao2shl3,
                                            src=grad_coeff_b2)

            if allalphas.requires_grad:
                grad_allalphas = torch.zeros_like(allalphas)  # (ngauss)

                # get the uncontracted integrals
                sname_derivs = [
                    int_nmgr.get_intgl_deriv_namemgr("rr", ib)
                    for ib in range(4)
                ]
                new_axes_pos = [
                    int_nmgr.get_intgl_deriv_newaxispos("rr", ib)
                    for ib in range(4)
                ]

                def u_int_fcn(wrappers, int_nmgr):
                    return _Int4cFunction.apply(*u_params, wrappers, int_nmgr)

                dout_dalphas = _get_integrals(sname_derivs, u_wrappers,
                                              u_int_fcn, new_axes_pos)

                # (nu_ao)
                # negative because the exponent is negative alpha * (r-ra)^2
                grad_alpha_a1 = -torch.einsum("...ijkl,...ijkl->i",
                                              dout_dalphas[0], u_grad_out)
                grad_alpha_a2 = -torch.einsum("...ijkl,...ijkl->j",
                                              dout_dalphas[1], u_grad_out)
                grad_alpha_b1 = -torch.einsum("...ijkl,...ijkl->k",
                                              dout_dalphas[2], u_grad_out)
                grad_alpha_b2 = -torch.einsum("...ijkl,...ijkl->l",
                                              dout_dalphas[3], u_grad_out)
                # grad_alpha_all = (grad_alpha_a1 + grad_alpha_a2 + grad_alpha_b1 + grad_alpha_b2)

                # scatter the grad
                grad_allalphas.scatter_add_(dim=-1,
                                            index=ao2shl0,
                                            src=grad_alpha_a1)
                grad_allalphas.scatter_add_(dim=-1,
                                            index=ao2shl1,
                                            src=grad_alpha_a2)
                grad_allalphas.scatter_add_(dim=-1,
                                            index=ao2shl2,
                                            src=grad_alpha_b1)
                grad_allalphas.scatter_add_(dim=-1,
                                            index=ao2shl3,
                                            src=grad_alpha_b2)

        return grad_allcoeffs, grad_allalphas, grad_allposs, \
            None, None, None


# integrator (direct interface to libcint)


# Optimizer class
class _cintoptHandler(ctypes.c_void_p):
    """
    Handler for the CINT optimizer.

    This class handles the CINT optimizer and releases resources when the object is deleted.
    """

    def __del__(self):
        """
        Destructor for the _cintoptHandler class.

        Releases resources when the object is deleted.
        """
        try:
            CGTO().CINTdel_optimizer(ctypes.byref(self))
        except AttributeError:
            pass


class Intor(object):
    """
    Integral operator class.

    This class represents an integral operator and calculates the
    integrals based on the provided atomic information.

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.dft_utils import AtomCGTOBasis, LibcintWrapper, loadbasis
    >>> from deepchem.utils.dft_utils.hamilton.intor.namemgr import IntorNameManager
    >>> dtype = torch.double
    >>> d = 1.0
    >>> pos_requires_grad = True
    >>> pos1 = torch.tensor([0.1 * d,  0.0 * d,  0.2 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos2 = torch.tensor([0.0 * d,  1.0 * d, -0.4 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos3 = torch.tensor([0.2 * d, -1.4 * d, -0.9 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> poss = [pos1, pos2, pos3]
    >>> atomzs = [1, 1, 1]
    >>> allbases = [
    ...     loadbasis("%d:%s" % (max(atomz, 1), "3-21G"), dtype=dtype, requires_grad=False)
    ...     for atomz in atomzs
    ... ]
    >>> atombases = [
    ...     AtomCGTOBasis(atomz=atomzs[i], bases=allbases[i], pos=poss[i])
    ...     for i in range(len(allbases))
    ... ]
    >>> wrap = LibcintWrapper(atombases, True, None)
    >>> int_nmgr = IntorNameManager("int1e", "r0")
    >>> intor = Intor(int_nmgr, [wrap])
    >>> intor.calc().shape
    torch.Size([3, 6])

    """

    def __init__(self, int_nmgr: IntorNameManager,
                 wrappers: List[LibcintWrapper]):
        """
        Initialize the Integral Operator.

        Parameters
        ----------
        int_nmgr : IntorNameManager
            Integral name manager.
        wrappers : List[LibcintWrapper]
            List of LibcintWrapper objects containing the atomic information.
        """
        assert len(wrappers) > 0
        wrapper0 = wrappers[0]
        self.int_type = int_nmgr.int_type
        self.atm, self.bas, self.env = wrapper0.atm_bas_env
        self.wrapper0 = wrapper0
        self.int_nmgr = int_nmgr
        self.wrapper_uniqueness = _get_uniqueness([id(w) for w in wrappers])

        # get the operator
        opname = int_nmgr.get_intgl_name(wrapper0.spherical)
        self.op = getattr(CINT(), opname)
        self.optimizer = _get_intgl_optimizer(opname, self.atm, self.bas,
                                              self.env)

        # prepare the output
        comp_shape = int_nmgr.get_intgl_components_shape()
        self.outshape = comp_shape + tuple(w.nao() for w in wrappers)
        self.ncomp = reduce(operator.mul, comp_shape, 1)
        self.shls_slice = sum((w.shell_idxs for w in wrappers), ())
        self.integral_done = False

    def calc(self) -> torch.Tensor:
        """
        Calculate the integrals.

        Returns
        -------
        torch.Tensor
            Tensor containing the calculated integrals.
        """
        assert not self.integral_done
        self.integral_done = True
        if self.int_type == "int1e" or self.int_type == "int2c2e":
            return self._int2c()
        elif self.int_type == "int3c2e":
            return self._int3c()
        elif self.int_type == "int2e":
            return self._int4c()
        else:
            raise ValueError("Unknown integral type: %s" % self.int_type)

    def _int2c(self) -> torch.Tensor:
        """
        Calculate 2-centre integrals.

        Returns
        -------
        torch.Tensor
            Tensor containing the calculated 2-centre integrals.
        """
        # performing 2-centre integrals with libcint
        drv = CGTO().GTOint2c
        outshape = self.outshape
        out = np.empty((*outshape[:-2], outshape[-1], outshape[-2]),
                       dtype=np.float64)
        drv(
            self.op,
            out.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(self.ncomp),
            ctypes.c_int(0),  # do not assume hermitian
            (ctypes.c_int * len(self.shls_slice))(*self.shls_slice),
            np2ctypes(self.wrapper0.full_shell_to_aoloc),
            self.optimizer,
            np2ctypes(self.atm),
            int2ctypes(self.atm.shape[0]),
            np2ctypes(self.bas),
            int2ctypes(self.bas.shape[0]),
            np2ctypes(self.env))

        out = np.swapaxes(out, -2, -1)
        return self._to_tensor(out)

    def _int3c(self) -> torch.Tensor:
        """
        Calculate 3-centre integrals.

        Returns
        -------
        torch.Tensor
            Tensor containing the calculated 3-centre integrals.
        """
        # performing 3-centre integrals with libcint
        drv = CGTO().GTOnr3c_drv
        fill = CGTO().GTOnr3c_fill_s1
        outsh = self.outshape
        out = np.empty((*outsh[:-3], outsh[-1], outsh[-2], outsh[-3]),
                       dtype=np.float64)
        drv(self.op, fill, out.ctypes.data_as(ctypes.c_void_p),
            int2ctypes(self.ncomp),
            (ctypes.c_int * len(self.shls_slice))(*self.shls_slice),
            np2ctypes(self.wrapper0.full_shell_to_aoloc), self.optimizer,
            np2ctypes(self.atm), int2ctypes(self.atm.shape[0]),
            np2ctypes(self.bas), int2ctypes(self.bas.shape[0]),
            np2ctypes(self.env))

        out = np.swapaxes(out, -3, -1)
        return self._to_tensor(out)

    def _int4c(self) -> torch.Tensor:
        """
        Calculate 4-centre integrals.

        Returns
        -------
        torch.Tensor
            Tensor containing the calculated 4-centre integrals.
        """
        # performing 4-centre integrals with libcint
        symm = self.int_nmgr.get_intgl_symmetry(self.wrapper_uniqueness)
        outshape = symm.get_reduced_shape(self.outshape)

        out = np.empty(outshape, dtype=np.float64)

        drv = CGTO().GTOnr2e_fill_drv
        fill = getattr(CGTO(), "GTOnr2e_fill_%s" % symm.code)
        prescreen = ctypes.POINTER(ctypes.c_void_p)()
        drv(self.op, fill, prescreen, out.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(self.ncomp), (ctypes.c_int * 8)(*self.shls_slice),
            np2ctypes(self.wrapper0.full_shell_to_aoloc), self.optimizer,
            np2ctypes(self.atm), int2ctypes(self.atm.shape[0]),
            np2ctypes(self.bas), int2ctypes(self.bas.shape[0]),
            np2ctypes(self.env))

        out = symm.reconstruct_array(out, self.outshape)
        return self._to_tensor(out)

    def _to_tensor(self, out: np.ndarray) -> torch.Tensor:
        """
        Convert numpy array to tensor.

        Parameters
        ----------
        out : np.ndarray
            Numpy array to be converted.

        Returns
        -------
        torch.Tensor
            Tensor containing the converted array.
        """
        return torch.as_tensor(out,
                               dtype=self.wrapper0.dtype,
                               device=self.wrapper0.device)


def _get_intgl_optimizer(opname: str, atm: np.ndarray, bas: np.ndarray,
                         env: np.ndarray) -> ctypes.c_void_p:
    """
    Get the optimizer for the integral.

    Examples
    --------
    >>> import numpy as np
    >>> atm = np.array([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]])
    >>> bas = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
    >>> env = np.array([0.0, 0.0, 0.0, 0.0])
    >>> opt = _get_intgl_optimizer("int1e_ovlp", atm, bas, env)

    Parameters
    ----------
    opname : str
        Name of the integral.
    atm : np.ndarray
        Array containing atomic information.
    bas : np.ndarray
        Array containing basis set information.
    env : np.ndarray
        Array containing environmental information.

    Returns
    -------
    ctypes.c_void_p
        Optimizer for the integral.
    """
    cintopt = ctypes.POINTER(ctypes.c_void_p)()
    optname = opname.replace("_cart", "").replace("_sph", "") + "_optimizer"
    copt = getattr(CINT(), optname)
    copt(ctypes.byref(cintopt), np2ctypes(atm), int2ctypes(atm.shape[0]),
         np2ctypes(bas), int2ctypes(bas.shape[0]), np2ctypes(env))
    opt = ctypes.cast(cintopt, _cintoptHandler)
    return opt


# name derivation manager functions


def _get_integrals(int_nmgrs: List[IntorNameManager],
                   wrappers: List[LibcintWrapper],
                   int_fcn: Callable[[List[LibcintWrapper], IntorNameManager],
                                     torch.Tensor],
                   new_axes_pos: List[int]) -> List[torch.Tensor]:
    """
    Get the list of tensors of the integrals.

    Parameters
    ----------
    int_nmgrs : List[IntorNameManager]
        List of integral name managers.
    wrappers : List[LibcintWrapper]
        List of LibcintWrapper objects containing atomic information.
    int_fcn : Callable[[List[LibcintWrapper], IntorNameManager], torch.Tensor]
        Integral function that receives the name and returns the results.
    new_axes_pos : List[int]
        List specifying the position of new axes.

    Returns
    -------
    List[torch.Tensor]
        List of tensors containing the calculated integrals.
    """
    res: List[torch.Tensor] = []
    # indicating if the integral is available in the libcint-generated file
    int_avail: List[bool] = [False] * len(int_nmgrs)

    for i in range(len(int_nmgrs)):
        res_i: Optional[torch.Tensor] = None

        # check if the integral can be calculated from the previous results
        for j in range(i - 1, -1, -1):

            # check the integral names equivalence
            transpose_path = int_nmgrs[j].get_transpose_path_to(int_nmgrs[i])
            if transpose_path is not None:

                # if the swapped wrappers remain unchanged, then just use the
                # transposed version of the previous version
                # TODO: think more about this (do we need to use different
                # transpose path? e.g. transpose_path[::-1])
                twrappers = _swap_list(wrappers, transpose_path)
                if twrappers == wrappers:
                    res_i = _transpose(res[j], transpose_path)
                    permute_path = int_nmgrs[j].get_comp_permute_path(
                        transpose_path)
                    res_i = res_i.permute(*permute_path)
                    break

                # otherwise, use the swapped integral with the swapped wrappers,
                # only if the integral is available in the libcint-generated
                # files
                elif int_avail[j]:
                    res_i = int_fcn(twrappers, int_nmgrs[j])
                    res_i = _transpose(res_i, transpose_path)
                    permute_path = int_nmgrs[j].get_comp_permute_path(
                        transpose_path)
                    res_i = res_i.permute(*permute_path)
                    break

                # if the integral is not available, then continue the searching
                else:
                    continue

        if res_i is None:
            try:
                # successfully executing the line below indicates that the integral
                # is available in the libcint-generated files
                res_i = int_fcn(wrappers, int_nmgrs[i])
            except AttributeError:
                msg = "The integral %s is not available from libcint, please add it" % int_nmgrs[
                    i].fullname
                raise AttributeError(msg)

            int_avail[i] = True

        res.append(res_i)

    # move the new axes (if any) to dimension 0
    assert res_i is not None
    for i in range(len(res)):
        if new_axes_pos[i] is not None:
            res[i] = torch.movedim(res[i], new_axes_pos[i], 0)

    return res


def _transpose(a: torch.Tensor, axes: List[Tuple[int, int]]) -> torch.Tensor:
    """
    Transpose the tensor.

    Examples
    --------
    >>> import torch
    >>> a = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    >>> axes = [(0, 1), (1, 2)]
    >>> _transpose(a, axes)
    tensor([[[1, 5],
             [2, 6]],
    <BLANKLINE>
            [[3, 7],
             [4, 8]]])

    Parameters
    ----------
    a : torch.Tensor
        Tensor to be transposed.
    axes : List[Tuple[int, int]]
        List of tuples specifying the axes to be transposed.

    Returns
    -------
    torch.Tensor
        Transposed tensor.
    """
    # perform the transpose of two axes for tensor a
    for axis2 in axes:
        a = a.transpose(*axis2)
    return a


def _swap_list(a: List, swaps: List[Tuple[int, int]]) -> List:
    """
    Swap elements in the list.

    Examples
    --------
    >>> a = [1, 2, 3, 4]
    >>> swaps = [(0, 1), (2, 3)]
    >>> _swap_list(a, swaps)
    [2, 1, 4, 3]

    Parameters
    ----------
    a : List
        List containing elements.
    swaps : List[Tuple[int, int]]
        List of tuples specifying the elements to be swapped.

    Returns
    -------
    List
        List with swapped elements.
    """
    # swap the elements according to the swaps input
    res = copy.copy(a)  # shallow copy
    for idxs in swaps:
        res[idxs[0]], res[idxs[1]] = res[idxs[1]], res[
            idxs[0]]  # swap the elements
    return res


def _gather_at_dims(inp: torch.Tensor, mapidxs: List[torch.Tensor],
                    dims: List[int]) -> torch.Tensor:
    """
    Gather values based on mapping indices.

    Examples
    --------
    >>> import torch
    >>> inp = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    >>> mapidxs = [torch.tensor([[0, 1], [1, 0]]), torch.tensor([[0, 1], [1, 0]])]
    >>> dims = [-2, -1]
    >>> _gather_at_dims(inp, mapidxs, dims)
    tensor([[[1, 2],
             [4, 3]],
    <BLANKLINE>
            [[7, 8],
             [6, 5]]])

    Parameters
    ----------
    inp : torch.Tensor
        Input tensor. (..., nold, ...)
    mapidxs : List[torch.Tensor]
        List of mapping indices. (..., nnew, ...)
    dims : List[int]
        List of dimensions.

    Returns
    -------
    torch.Tensor
        Tensor with gathered values.
    """
    out = inp
    for (dim, mapidx) in zip(dims, mapidxs):
        if dim < 0:
            dim = out.ndim + dim
        map2 = mapidx[(...,) + (None,) * (out.ndim - 1 - dim)]
        map2 = map2.expand(*out.shape[:dim], -1, *out.shape[dim + 1:])
        out = torch.gather(out, dim=dim, index=map2)
    return out


def _get_uniqueness(a: List) -> List[int]:
    """
    Get the uniqueness pattern from the list.

    Examples
    --------
    >>> a = [1, 2, 3, 1, 2]
    >>> _get_uniqueness(a)
    [0, 1, 2, 0, 1]

    Parameters
    ----------
    a : List
        List of elements.

    Returns
    -------
    List[int]
        List representing the uniqueness pattern.
    """
    s: Dict = {}
    res: List[int] = []
    i = 0
    for elmt in a:
        if elmt in s:
            res.append(s[elmt])
        else:
            s[elmt] = i
            res.append(i)
            i += 1
    return res

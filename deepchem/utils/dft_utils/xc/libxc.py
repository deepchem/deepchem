import warnings
import torch
try:
    import pylibxc
except (ImportError, ModuleNotFoundError) as e:
    warnings.warn("Failed to import pylibxc. Might not be able to use xc.")
from typing import List, Tuple, Union, overload, Optional
from deepchem.utils.dft_utils import BaseXC, ValGrad, SpinParam
from deepchem.utils.dft_utils.xc.libxc_wrapper import CalcLDALibXCPol, CalcLDALibXCUnpol, \
    CalcGGALibXCPol, CalcGGALibXCUnpol, CalcMGGALibXCUnpol, CalcMGGALibXCPol


ERRMSG = "This function cannot do broadcasting. " \
         "Please make sure the inputs have the same shape."
N_VRHO = 2  # number of xc energy derivative w.r.t. density (i.e. 2: u, d)
N_VSIGMA = 3  # number of energy derivative w.r.t. contracted gradient (i.e. 3: uu, ud, dd)

class LibXCLDA(BaseXC):
    _family: int = 1
    _unpolfcn_wrapper = CalcLDALibXCUnpol
    _polfcn_wrapper = CalcLDALibXCPol

    def __init__(self, name: str) -> None:
        self.libxc_unpol = pylibxc.LibXCFunctional(name, "unpolarized")
        self.libxc_pol = pylibxc.LibXCFunctional(name, "polarized")

    @property
    def family(self) -> int:
        return self._family

    def get_vxc(self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> Union[ValGrad, SpinParam[ValGrad]]:
        """Get the exchange-correlation potential from libxc.

        Parameters
        ----------
        densinfo: Union[ValGrad, SpinParam[ValGrad]]
            density information
            densinfo.value: (*BD, nr)
            densinfo.grad: (*BD, nr, ndim)

        Returns
        -------
        Union[ValGrad, SpinParam[ValGrad]]
            Potential information
            potentialinfo.value: (*BD, nr)
            potentialinfo.grad: (*BD, nr, ndim)

        """
        libxc_inps = _prepare_libxc_input(densinfo, xcfamily=self.family)
        flatten_inps = tuple(inp.reshape(-1) for inp in libxc_inps)

        # polarized case
        if not isinstance(densinfo, ValGrad):
            # outs are (vrho,) for LDA, (vrho, vsigma) for GGA each with shape
            # (nspin, *shape)
            outs = self._calc_pol(flatten_inps, densinfo.u.value.shape, 1)

        # unpolarized case
        else:
            # outs are (vrho,) for LDA, (vrho, vsigma) for GGA each with shape
            # (*shape)
            outs = self._calc_unpol(flatten_inps, densinfo.value.shape, 1)

        potinfo = _postproc_libxc_voutput(densinfo, *outs)
        return potinfo

    def get_edensityxc(self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> \
            torch.Tensor:
        """Get the exchange-correlation energy density from libxc.

        Parameters
        ----------
        densinfo: Union[ValGrad, SpinParam[ValGrad]]
            Density information
            densinfo.value: (*BD, nr)
            densinfo.grad: (*BD, nr, ndim)

        Returns
        -------
        torch.Tensor
            Exchange-correlation energy density
            edens: (*BD, nr)

        """
        libxc_inps = _prepare_libxc_input(densinfo, xcfamily=self.family)
        flatten_inps = tuple(inp.reshape(-1) for inp in libxc_inps)

        # polarized case
        if not isinstance(densinfo, ValGrad):
            rho_u = densinfo.u.value

            edens = self._calc_pol(flatten_inps, densinfo.u.value.shape, 0)[0]  # (*BD, nr)
            edens = edens.reshape(rho_u.shape)
            return edens

        # unpolarized case
        else:
            edens = self._calc_unpol(flatten_inps, densinfo.value.shape, 0)[0]  # (*BD, nr)
            return edens

    def _calc_pol(self, flatten_inps: Tuple[torch.Tensor, ...], shape: torch.Size, deriv: int) ->\
            Tuple[torch.Tensor, ...]:
        """Calculate the polarized exchange-correlation potential from libxc.
        
        Parameters
        ----------
        flatten_inps: Tuple[torch.Tensor, ...]
            Flattened inputs for libxc
        shape: torch.Size
            Shape of the density
        deriv: int
            Derivative order

        Returns
        -------
        Tuple[torch.Tensor, ...]
            Outputs from libxc

        """

        outs = self._polfcn_wrapper.apply(*flatten_inps, deriv, self.libxc_pol)

        # tuple of (nderiv, *shape) where nderiv is the number of spin-dependent
        # values from libxc
        return tuple(out.reshape(-1, *shape) for out in outs)

    def _calc_unpol(self, flatten_inps: Tuple[torch.Tensor, ...], shape: torch.Size, deriv: int) ->\
            Tuple[torch.Tensor, ...]:
        """Calculate the unpolarized exchange-correlation potential from libxc.

        Parameters
        ----------
        flatten_inps: Tuple[torch.Tensor, ...]
            Flattened inputs for libxc
        shape: torch.Size
            Shape of the density
        deriv: int
            Derivative order

        Returns
        -------
        Tuple[torch.Tensor, ...]
            Outputs from libxc

        """

        outs = self._unpolfcn_wrapper.apply(*flatten_inps, deriv, self.libxc_unpol)

        # tuple of (*shape) where shape
        return tuple(out.reshape(shape) for out in outs)

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        return []

class LibXCGGA(LibXCLDA):
    """Generalized Gradient Approximation (GGA) wrapper for libxc."""
    _family: int = 2
    _unpolfcn_wrapper = CalcGGALibXCUnpol
    _polfcn_wrapper = CalcGGALibXCPol

class LibXCMGGA(LibXCLDA):
    """Meta-Generalized Gradient Approximation (MGGA) wrapper for libxc."""
    _family: int = 4
    _unpolfcn_wrapper = CalcMGGALibXCUnpol
    _polfcn_wrapper = CalcMGGALibXCPol

def _all_same_shape(densinfo_u: ValGrad, densinfo_d: ValGrad) -> bool:
    """Check if the shapes of the input are the same.

    Parameters
    ----------
    densinfo_u: ValGrad
        Density information for the up-spin
    densinfo_d: ValGrad
        Density information for the down-spin

    """
    return densinfo_u.value.shape == densinfo_d.value.shape

def _get_polstr(polarized: bool) -> str:
    """Get the string representation of the polarization.

    Parameters
    ----------
    polarized: bool
        If the system is polarized

    Returns
    -------
    str
        String representation of the polarization

    """
    return "polarized" if polarized else "unpolarized"

def _prepare_libxc_input(densinfo: Union[SpinParam[ValGrad], ValGrad], xcfamily: int) -> Tuple[torch.Tensor, ...]:
    """Prepare the input for libxc.

    Parameters
    ----------
    densinfo: Union[SpinParam[ValGrad], ValGrad]
        Density information
        densinfo.value: (*BD, nr)
        densinfo.grad: (*BD, nr, ndim)
    xcfamily: int
        Family of the exchange-correlation functional

    Returns
    -------
    Tuple[torch.Tensor, ...]
        Inputs for libxc

    """
    # convert the densinfo into tuple of tensors for libxc inputs
    # the elements in the tuple is arranged according to libxc manual

    sigma_einsum = "...dr,...dr->...r"
    # polarized case
    if isinstance(densinfo, SpinParam):
        rho_u = densinfo.u.value  # (*nrho)
        rho_d = densinfo.d.value

        if xcfamily == 1:  # LDA
            return (rho_u, rho_d)

        assert densinfo.u.grad is not None
        assert densinfo.d.grad is not None
        grad_u = densinfo.u.grad  # (*nrho, ndim)
        grad_d = densinfo.d.grad

        # calculate the contracted gradient
        sigma_uu = torch.einsum(sigma_einsum, grad_u, grad_u)
        sigma_ud = torch.einsum(sigma_einsum, grad_u, grad_d)
        sigma_dd = torch.einsum(sigma_einsum, grad_d, grad_d)

        if xcfamily == 2:  # GGA
            return (rho_u, rho_d, sigma_uu, sigma_ud, sigma_dd)

        assert densinfo.u.lapl is not None
        assert densinfo.d.lapl is not None
        assert densinfo.u.kin is not None
        assert densinfo.d.kin is not None
        lapl_u = densinfo.u.lapl
        lapl_d = densinfo.d.lapl
        kin_u = densinfo.u.kin
        kin_d = densinfo.d.kin

        if xcfamily == 4:  # MGGA
            return (rho_u, rho_d, sigma_uu, sigma_ud, sigma_dd, lapl_u, lapl_d, kin_u, kin_d)

    # unpolarized case
    else:
        rho = densinfo.value

        if xcfamily == 1:
            return (rho,)

        assert densinfo.grad is not None
        grad = densinfo.grad  # (*nrho, ndim)

        # contracted gradient
        sigma = torch.einsum(sigma_einsum, grad, grad)

        if xcfamily == 2:  # GGA
            return (rho, sigma)

        assert densinfo.lapl is not None
        assert densinfo.kin is not None
        lapl = densinfo.lapl
        kin = densinfo.kin

        if xcfamily == 4:  # MGGA
            return (rho, sigma, lapl, kin)

    raise RuntimeError(f"xcfamily {xcfamily} is not implemented")

def _postproc_libxc_voutput(densinfo: Union[SpinParam[ValGrad], ValGrad],
                            vrho: torch.Tensor,
                            vsigma: Optional[torch.Tensor] = None,
                            vlapl: Optional[torch.Tensor] = None,
                            vkin: Optional[torch.Tensor] = None) -> Union[SpinParam[ValGrad], ValGrad]:
    """Postprocess the output from libxc into potential information.

    Parameters
    ----------
    densinfo: Union[SpinParam[ValGrad], ValGrad]
        Density information
        densinfo.value: (..., nr)
        densinfo.grad: (..., ndim, nr)
    vrho: torch.Tensor
        Density potential
    vsigma: Optional[torch.Tensor]
        Gradient potential
    vlapl: Optional[torch.Tensor]
        Laplacian potential
    vkin: Optional[torch.Tensor]
        Kinetic potential

    Returns
    -------
    Union[SpinParam[ValGrad], ValGrad]
        Potential information

    """
    # polarized case
    if isinstance(densinfo, SpinParam):
        # vrho: (2, *BD, nr)
        # vsigma: (3, *BD, nr)
        # vlapl: (2, *BD, nr)
        # vkin: (2, *BD, nr)
        vrho_u = vrho[0]
        vrho_d = vrho[1]

        # calculate the gradient potential
        vgrad_u: Optional[torch.Tensor] = None
        vgrad_d: Optional[torch.Tensor] = None
        if vsigma is not None:
            # calculate the grad_vxc
            vgrad_u = 2 * vsigma[0].unsqueeze(-2) * densinfo.u.grad + \
                vsigma[1].unsqueeze(-2) * densinfo.d.grad  # (..., 3)
            vgrad_d = 2 * vsigma[2].unsqueeze(-2) * densinfo.d.grad + \
                vsigma[1].unsqueeze(-2) * densinfo.u.grad

        vlapl_u: Optional[torch.Tensor] = None
        vlapl_d: Optional[torch.Tensor] = None
        if vlapl is not None:
            vlapl_u = vlapl[0]
            vlapl_d = vlapl[1]

        vkin_u: Optional[torch.Tensor] = None
        vkin_d: Optional[torch.Tensor] = None
        if vkin is not None:
            vkin_u = vkin[0]
            vkin_d = vkin[1]

        potinfo_u = ValGrad(value=vrho_u, grad=vgrad_u, lapl=vlapl_u, kin=vkin_u)
        potinfo_d = ValGrad(value=vrho_d, grad=vgrad_d, lapl=vlapl_d, kin=vkin_d)
        return SpinParam(u=potinfo_u, d=potinfo_d)

    # unpolarized case
    else:
        # all are: (*BD, nr)

        # calculate the gradient potential
        if vsigma is not None:
            vsigma = 2 * vsigma.unsqueeze(-2) * densinfo.grad  # (*BD, ndim, nr)

        potinfo = ValGrad(value=vrho, grad=vsigma, lapl=vlapl, kin=vkin)
        return potinfo

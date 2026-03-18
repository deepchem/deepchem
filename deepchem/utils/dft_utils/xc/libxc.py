import warnings
import torch
try:
    import pylibxc
except (ImportError, ModuleNotFoundError) as e:
    warnings.warn(f"Failed to import pylibxc. Might not be able to use xc. {e}")
from typing import List, Tuple, Union, Optional
from deepchem.utils.dft_utils import BaseXC, ValGrad, SpinParam
from deepchem.utils.dft_utils.xc.libxc_wrapper import CalcLDALibXCPol, CalcLDALibXCUnpol, \
    CalcGGALibXCPol, CalcGGALibXCUnpol, CalcMGGALibXCUnpol, CalcMGGALibXCPol


ERRMSG = "This function cannot do broadcasting. " \
         "Please make sure the inputs have the same shape."
N_VRHO = 2  # number of xc energy derivative w.r.t. density (i.e. 2: u, d)
N_VSIGMA = 3  # number of energy derivative w.r.t. contracted gradient (i.e. 3: uu, ud, dd)


class LibXCLDA(BaseXC):
    """Local Density Approximation (LDA) wrapper for libxc.
    Local-density approximations (LDA) are a class of approximations
    to the exchange–correlation (XC) energy functional in density
    functional theory (DFT) that depend solely upon the value of the
    electronic density at each point in space.

    Examples
    --------
    >>> from deepchem.utils.dft_utils import ValGrad
    >>> import torch
    >>> # create a LDA wrapper for libxc
    >>> lda = LibXCLDA("lda_x")
    >>> # create a density information
    >>> densinfo = ValGrad(value=torch.rand(2, 3, 4), grad=torch.rand(2, 3, 4, 3))
    >>> # get the exchange-correlation potential
    >>> potinfo = lda.get_vxc(densinfo)
    >>> potinfo.value.shape
    torch.Size([2, 3, 4])
    >>> edens = lda.get_edensityxc(densinfo)
    >>> edens.shape
    torch.Size([2, 3, 4])

    Attributes
    ----------
    _family: int (default 1)
        Family of the exchange-correlation functional
    _unpolfcn_wrapper: torch.autograd.Function (default CalcLDALibXCUnpol)
        Wrapper for the unpolarized LDA functional
    _polfcn_wrapper: torch.autograd.Function (default CalcLDALibXCPol)
        Wrapper for the polarized LDA functional

    """

    def __init__(self, name: str) -> None:
        """Initialize the LDA wrapper for libxc.

        Parameters
        ----------
        name: str
            Name of the exchange-correlation functional

        """
        self.libxc_unpol = pylibxc.LibXCFunctional(name, "unpolarized")
        self.libxc_pol = pylibxc.LibXCFunctional(name, "polarized")
        self._family: int = 1
        self._unpolfcn_wrapper = CalcLDALibXCUnpol  # type: ignore
        self._polfcn_wrapper = CalcLDALibXCPol  # type: ignore

    @property
    def family(self) -> int:
        """Get the family of the exchange-correlation functional.

        Returns
        -------
        int
            Family of the exchange-correlation functional

        """
        return self._family

    def get_vxc(
        self, densinfo: Union[ValGrad, SpinParam[ValGrad]]
    ) -> Union[ValGrad, SpinParam[ValGrad]]:
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

            edens = self._calc_pol(flatten_inps, densinfo.u.value.shape,
                                   0)[0]  # (*BD, nr)
            edens = edens.reshape(rho_u.shape)
            return edens

        # unpolarized case
        else:
            edens = self._calc_unpol(flatten_inps, densinfo.value.shape,
                                     0)[0]  # (*BD, nr)
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

        outs = self._unpolfcn_wrapper.apply(*flatten_inps, deriv,
                                            self.libxc_unpol)

        # tuple of (*shape) where shape
        return tuple(out.reshape(shape) for out in outs)

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        return []


class LibXCGGA(LibXCLDA):
    """Generalized Gradient Approximation (GGA) wrapper for libxc.

    GGA can correct the overestimated binding energy of LDA in
    molecules and solids and extend the processing system to the
    energy and structure of the hydrogen bond system. The
    approximation greatly improves the calculation results of the
    energy related to electrons and exchange.

    Examples
    --------
    >>> from deepchem.utils.dft_utils import ValGrad
    >>> import torch
    >>> # create a GGA wrapper for libxc
    >>> gga = LibXCGGA("gga_c_pbe")
    >>> # create a density information
    >>> n = 2
    >>> rho_u = torch.rand((n,), dtype=torch.float64).requires_grad_()
    >>> grad_u = torch.rand((3, n), dtype=torch.float64).requires_grad_()
    >>> densinfo = ValGrad(value=rho_u, grad=grad_u)
    >>> # get the exchange-correlation potential
    >>> potinfo = gga.get_vxc(densinfo)
    >>> potinfo.value.shape
    torch.Size([2])

    Attributes
    ----------
    _family: int (default 2)
        Family of the exchange-correlation functional
    _unpolfcn_wrapper: torch.autograd.Function (default CalcGGALibXCUnpol)
        Wrapper for the unpolarized GGA functional
    _polfcn_wrapper: torch.autograd.Function (default CalcGGALibXCPol)
        Wrapper for the polarized GGA functional

    """

    def __init__(self, name: str) -> None:
        """Initialize the LDA wrapper for libxc.

        Parameters
        ----------
        name: str
            Name of the exchange-correlation functional

        """
        self.libxc_unpol = pylibxc.LibXCFunctional(name, "unpolarized")
        self.libxc_pol = pylibxc.LibXCFunctional(name, "polarized")
        self._family: int = 2
        self._unpolfcn_wrapper = CalcGGALibXCUnpol  # type: ignore
        self._polfcn_wrapper = CalcGGALibXCPol  # type: ignore


class LibXCMGGA(LibXCLDA):
    """Meta-Generalized Gradient Approximation (MGGA) wrapper for libxc.

    Meta-GGAs typically improve upon the accuracy of GGAs as they
    additionally take into account the local kinetic energy density.
    This allows meta-GGAs to more accurately treat different chemical
    bonds (eg. covalent, metallic, and weak) compared to LDAs and GGAs.

    Examples
    --------
    >>> from deepchem.utils.dft_utils import ValGrad
    >>> import torch
    >>> # create a MGGA wrapper for libxc
    >>> mgga = LibXCMGGA("mgga_x_scan")
    >>> # create a density information
    >>> n = 2
    >>> rho_u = torch.rand((n,), dtype=torch.float64).requires_grad_()
    >>> grad_u = torch.rand((3, n), dtype=torch.float64).requires_grad_()
    >>> lapl_u = torch.rand((n,), dtype=torch.float64).requires_grad_()
    >>> kin_u = torch.rand((n,), dtype=torch.float64).requires_grad_()
    >>> densinfo = ValGrad(value=rho_u, grad=grad_u, lapl=lapl_u, kin=kin_u)
    >>> # get the exchange-correlation potential
    >>> potinfo = mgga.get_vxc(densinfo)
    >>> potinfo.value.shape
    torch.Size([2])

    Attributes
    ----------
    _family: int (default 4)
        Family of the exchange-correlation functional
    _unpolfcn_wrapper: torch.autograd.Function (default CalcMGGALibXCUnpol)
        Wrapper for the unpolarized MGGA functional
    _polfcn_wrapper: torch.autograd.Function (default CalcMGGALibXCPol)
        Wrapper for the polarized MGGA functional

    """

    def __init__(self, name: str) -> None:
        """Initialize the LDA wrapper for libxc.

        Parameters
        ----------
        name: str
            Name of the exchange-correlation functional

        """
        self.libxc_unpol = pylibxc.LibXCFunctional(name, "unpolarized")
        self.libxc_pol = pylibxc.LibXCFunctional(name, "polarized")
        self._family: int = 4
        self._unpolfcn_wrapper = CalcMGGALibXCUnpol  # type: ignore
        self._polfcn_wrapper = CalcMGGALibXCPol  # type: ignore


def _prepare_libxc_input(densinfo: Union[SpinParam[ValGrad], ValGrad],
                         xcfamily: int) -> Tuple[torch.Tensor, ...]:
    """Prepare the input for libxc.

    Examples
    --------
    >>> from deepchem.utils.dft_utils import ValGrad, SpinParam
    >>> import torch
    >>> # create a density information
    >>> n = 2
    >>> rho_u = torch.rand((n,), dtype=torch.float64).requires_grad_()
    >>> grad_u = torch.rand((3, n), dtype=torch.float64).requires_grad_()
    >>> lapl_u = torch.rand((n,), dtype=torch.float64).requires_grad_()
    >>> kin_u = torch.rand((n,), dtype=torch.float64).requires_grad_()
    >>> rho_d = torch.rand((n,), dtype=torch.float64).requires_grad_()
    >>> grad_d = torch.rand((3, n), dtype=torch.float64).requires_grad_()
    >>> lapl_d = torch.rand((n,), dtype=torch.float64).requires_grad_()
    >>> kin_d = torch.rand((n,), dtype=torch.float64).requires_grad_()
    >>> densinfo = SpinParam(u=ValGrad(value=rho_u, grad=grad_u, lapl=lapl_u, kin=kin_u), d=ValGrad(value=rho_d, grad=grad_d, lapl=lapl_d, kin=kin_d))
    >>> # prepare the input for libxc
    >>> inputs = _prepare_libxc_input(densinfo, 4)
    >>> len(inputs)
    9

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
            return (rho_u, rho_d, sigma_uu, sigma_ud, sigma_dd, lapl_u, lapl_d,
                    kin_u, kin_d)

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


def _postproc_libxc_voutput(
        densinfo: Union[SpinParam[ValGrad], ValGrad],
        vrho: torch.Tensor,
        vsigma: Optional[torch.Tensor] = None,
        vlapl: Optional[torch.Tensor] = None,
        vkin: Optional[torch.Tensor] = None
) -> Union[SpinParam[ValGrad], ValGrad]:
    """Postprocess the output from libxc into potential information.

    Examples
    --------
    >>> from deepchem.utils.dft_utils import ValGrad
    >>> import torch
    >>> # create a density information
    >>> n = 2
    >>> rho_u = torch.rand((n,), dtype=torch.float64).requires_grad_()
    >>> grad_u = torch.rand((3, n), dtype=torch.float64).requires_grad_()
    >>> lapl_u = torch.rand((n,), dtype=torch.float64).requires_grad_()
    >>> kin_u = torch.rand((n,), dtype=torch.float64).requires_grad_()
    >>> densinfo = ValGrad(value=rho_u, grad=grad_u, lapl=lapl_u, kin=kin_u)
    >>> # postprocess the output from libxc
    >>> potinfo = _postproc_libxc_voutput(densinfo, torch.rand((n,), dtype=torch.float64).requires_grad_())
    >>> potinfo.value.shape
    torch.Size([2])

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

        potinfo_u = ValGrad(value=vrho_u,
                            grad=vgrad_u,
                            lapl=vlapl_u,
                            kin=vkin_u)
        potinfo_d = ValGrad(value=vrho_d,
                            grad=vgrad_d,
                            lapl=vlapl_d,
                            kin=vkin_d)
        return SpinParam(u=potinfo_u, d=potinfo_d)

    # unpolarized case
    else:
        # all are: (*BD, nr)

        # calculate the gradient potential
        if vsigma is not None:
            vsigma = 2 * vsigma.unsqueeze(-2) * densinfo.grad  # (*BD, ndim, nr)

        potinfo = ValGrad(value=vrho, grad=vsigma, lapl=vlapl, kin=vkin)
        return potinfo

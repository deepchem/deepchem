import torch
from typing import Union, List
from deepchem.utils.dft_utils.xc.base_xc import BaseXC
from deepchem.utils.dft_utils.data.datastruct import ValGrad, SpinParam
from deepchem.utils import safepow


class PyTorchLDA(BaseXC):
    """Local Density Approximation (LDA) XC functional.

    LDA assumes that, at each point in space, the electrons behave like a uniform
    electron gas with the same local density. The exchange–correlation energy
    at that point is therefore taken from the known solution of a homogeneous
    electron gas.

    In a real material the electron density varies from point to point. But, LDA
    provides a simple and computationally efficient approximation for exchange–correlation
    effects. It is often used as a baseline functional because it is fast, numerically
    stable, and works reasonably well for systems with slowly varying electron densities
    such as bulk solids.

    Examples
    --------
    >>> from deepchem.utils.dft_utils import Mol, KS
    >>> from deepchem.utils.dft_utils.xc.pytorch_xc import PyTorchLDA
    >>> moldesc = "H 0 0 -0.74; H 0 0 0.74"
    >>> basis = "sto-3g"
    >>> system = Mol(moldesc=moldesc, basis=basis)
    >>> xc_dc = PyTorchLDA("lda_x")
    >>> ks = KS(system, xc_dc, variational=False)
    >>> _ = ks.run()
    >>> ks.energy().item()
    -1.023726577309795

    Notes
    -----
    - Available functionals:
      1. lda_x
    - Users can use the functinals implemented in this class or they can also create
      their own functionals by creating a subclass of PyTorchLDA class and implimenting
      the desired functional.
    - `getparamnames` method needs to be implemented in all the subclasses of BaseXC.
    - `init` method usage `getattr` builtin function to find the implemented functional.
      so there should be no need to update it in normal situation.
    """

    def __init__(self, name: str = "lda_x"):
        """Initialize the PyTorchLDA XC functional.

        Parameters
        ----------
        name: str (default "lda_x")
            The name of the LDA functional.
        """
        super().__init__()
        self.name = name
        self.functional = getattr(self, self.name)

    @property
    def family(self) -> int:
        """Returns the family identifier number of the XC.

        Returns
        -------
        int
            It returns 1 for LDA based on BaseXC.
        """
        return 1

    def lda_x(self, n: torch.Tensor) -> torch.Tensor:
        """Calculates the LDA_X exchange energy density based on [1].

        :math: E_x(n) = - C_x * n^(4/3)
        :math: C_x = (3/4) * (3/pi)^(1/3)

        This function evaluates the Dirac exchange energy density for an electron
        gas. In this approximation, the exchange energy at each point in space is
        assumed to be the same as that of a homogeneous electron gas with the same
        local electron density.

        Examples
        --------
        >>> from deepchem.utils.dft_utils import Mol, KS
        >>> from deepchem.utils.dft_utils.xc.pytorch_xc import PyTorchLDA
        >>> moldesc = "H 0 0 -0.74; H 0 0 0.74"
        >>> basis = "sto-3g"
        >>> system = Mol(moldesc=moldesc, basis=basis)
        >>> xc_dc = PyTorchLDA("lda_x")
        >>> ks = KS(system, xc_dc, variational=False)
        >>> _ = ks.run()
        >>> ks.energy().item()
        -1.023726577309795

        Parameters
        ----------
        n: torch.Tensor
            Electron density of the system.

        Returns
        -------
        torch.Tensor
            Exchange energy density.

        References
        ----------
        [1].. Dirac, P. a. M. (1930). Note on exchange phenomena in the Thomas
              Atom. Mathematical Proceedings of the Cambridge Philosophical Society,
              26(3), 376–385. https://doi.org/10.1017/s0305004100016108
        """
        C_x = torch.tensor(0.75 * ((3.0 / torch.pi)**(1.0 / 3.0)))
        n_safe = torch.clamp(n, min=1e-12)
        return -C_x * safepow(n_safe, 4.0 / 3.0)

    def get_edensityxc(
            self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> torch.Tensor:
        """Returns the xc energy density (energy per unit volume).

        Examples
        --------
        >>> import torch
        >>> from deepchem.utils.dft_utils.xc.pytorch_xc import PyTorchLDA
        >>> from deepchem.utils.dft_utils.data.datastruct import ValGrad
        >>> xc = PyTorchLDA()
        >>> n = torch.tensor([0.5, 1.0, 2.0], requires_grad=True)
        >>> densinfo = ValGrad(value=n)
        >>> edensity = xc.get_edensityxc(densinfo)
        >>> edensity.shape
        torch.Size([3])

        Parameters
        ----------
        densinfo: Union[ValGrad, SpinParam[ValGrad]]
            Local density information of electrons.

        Returns
        -------
        torch.Tensor
            The energy density of the XC.
        """
        if isinstance(densinfo, ValGrad):
            n = densinfo.value
            ex = self.functional(n)
            return ex
        else:
            nu = densinfo.u.value
            nd = densinfo.d.value
            ex = self.functional(nu) + self.functional(nd)
            return ex

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        """List tensor names that affect the output of the method.

        Parameters
        ----------
        methodname: str
            Name of the method of the class.
        prefix: str, optional
            Prefix to be appended in front of the parameters name.
            This usually contains the dots.

        Returns
        -------
        List[str]
            Sequence of name of parameters affecting the output of the method.
        """
        return []

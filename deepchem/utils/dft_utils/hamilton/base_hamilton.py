"""
Derived from: https://github.com/diffqc/dqc/blob/master/dqc/hamilton/base_hamilton.py
"""
from __future__ import annotations
import torch
from abc import abstractmethod, abstractproperty
from typing import List, Optional, Union, Tuple
from deepchem.utils.dft_utils import BaseGrid, BaseXC, BaseDF, SpinParam
from deepchem.utils.differentiation_utils import EditableModule, LinearOperator


class BaseHamilton(EditableModule):
    """
    Hamilton is a class that provides the LinearOperator of the Hamiltonian
    components.

    The Hamiltonian represents the total energy operator for a system of
    interacting electrons. The Kohn-Sham DFT approach introduces a set of
    fictitious non-interacting electrons that move in an effective potential.
    The total energy functional, which includes the kinetic energy of these
    fictitious electrons and their interaction with an effective potential
    (including the electron-electron interaction), is minimized to obtain the
    ground-state electronic structure.

    The Kohn-Sham Hamiltonian is a key component of this approach, representing
    the operator that governs the evolution of the Kohn-Sham orbitals. It
    includes terms for the kinetic energy of electrons, the external potential
    (usually from nuclei), and the exchange-correlation potential that accounts
    for the electron-electron interactions.

    The Fock matrix represents the one-electron part of the Hamiltonian matrix. Its
    components include kinetic energy, nuclear attraction, and electron-electron
    repulsion integrals. The Fock matrix is pivotal in solving the electronic
    Schrödinger equation and determining the electronic structure of molecular
    systems.

    Examples
    --------
    >>> from deepchem.utils.dft_utils import BaseHamilton
    >>> class MyHamilton(BaseHamilton):
    ...    def __init__(self):
    ...        self._nao = 2
    ...        self._kpts = torch.tensor([[0.0, 0.0, 0.0]])
    ...        self._df = None
    ...    @property
    ...    def nao(self):
    ...        return self._nao
    ...    @property
    ...    def kpts(self):
    ...        return self._kpts
    ...    @property
    ...    def df(self):
    ...        return self._df
    ...    def build(self):
    ...        return self
    ...    def get_nuclattr(self):
    ...        return torch.ones((1, 1, self.nao, self.nao))
    >>> ham = MyHamilton()
    >>> hamilton = ham.build()
    >>> hamilton.get_nuclattr()
    tensor([[[[1., 1.],
              [1., 1.]]]])

    """

    # properties
    @abstractproperty
    def nao(self) -> int:
        """Number of atomic orbital basis.

        Returns
        -------
        int
            Number of atomic orbital basis.

        """
        pass

    @abstractproperty
    def kpts(self) -> torch.Tensor:
        """List of k-points in the Hamiltonian.

        Returns
        -------
        torch.Tensor
            List of k-points in the Hamiltonian. Shape: (nkpts, ndim)

        """
        pass

    @abstractproperty
    def df(self) -> Optional[BaseDF]:
        """
        Returns the density fitting object (if any) attached to this
        Hamiltonian object.

        Returns
        -------
        Optional[BaseDF]
            Returns the density fitting object (if any) attached to this
            Hamiltonian object.

        """
        pass

    # setups
    @abstractmethod
    def build(self):
        """
        Construct the elements needed for the Hamiltonian.
        Heavy-lifting operations should be put here.

        """
        pass

    @abstractmethod
    def setup_grid(self, grid: BaseGrid, xc: Optional[BaseXC] = None) -> None:
        """
        Setup the basis (with its grad) in the spatial grid and prepare the
        gradient of atomic orbital according to the ones required by the xc.
        If xc is not given, then only setup the grid with ao (without any
        gradients of ao)

        Parameters
        ----------
        grid: BaseGrid
            Grid used to setup this Hamilton.
        xc: Optional[BaseXC] (default None)
            Exchange Corelation functional of this Hamiltonian.

        """
        pass

    # fock matrix components
    @abstractmethod
    def get_nuclattr(self) -> LinearOperator:
        """LinearOperator of the nuclear Coulomb attraction.

        Nuclear Coulomb attraction is the electrostatic force binding electrons
        to a nucleus. Positively charged protons attract negatively charged
        electrons, creating stability in quantum systems. This force plays a
        fundamental role in determining the structure and behavior of atoms,
        contributing significantly to the overall potential energy in atomic
        physics.

        Returns
        -------
        LinearOperator
            LinearOperator of the nuclear Coulomb attraction. Shape: (`*BH`, nao, nao)

        """
        pass

    @abstractmethod
    def get_kinnucl(self) -> LinearOperator:
        """
        Returns the LinearOperator of the one-electron operator (i.e. kinetic
        and nuclear attraction). Action of a LinearOperator on a function is a
        linear transformation. In the case of one-electron operators, these
        transformations are essential for solving the Schrödinger equation and
        understanding the behavior of electrons in an atomic or molecular system.

        Returns
        -------
        LinearOperator
            LinearOperator of the one-electron operator. Shape: (`*BH`, nao, nao)

        """
        pass

    @abstractmethod
    def get_overlap(self) -> LinearOperator:
        """
        Returns the LinearOperator representing the overlap of the basis.
        The overlap of the basis refers to the degree to which atomic or
        molecular orbitals in a quantum mechanical system share common space.

        Returns
        -------
        LinearOperator
            LinearOperator representing the overlap of the basis.
            Shape: (`*BH`, nao, nao)

        """
        pass

    @abstractmethod
    def get_elrep(self, dm: torch.Tensor) -> LinearOperator:
        """
        Obtains the LinearOperator of the Coulomb electron repulsion operator.
        Known as the J-matrix.

        In the context of electronic structure theory, it accounts for the
        repulsive interaction between electrons in a many-electron system. The
        J-matrix elements involve the Coulombic interactions between pairs of
        electrons, influencing the total energy and behavior of the system.

        Parameters
        ----------
        dm: torch.Tensor
            Density matrix. Shape: (`*BD`, nao, nao)

        Returns
        -------
        LinearOperator
            LinearOperator of the Coulomb electron repulsion operator.
            Shape: (`*BDH`, nao, nao)

        """
        pass

    @abstractmethod
    def get_exchange(
        self, dm: Union[torch.Tensor, SpinParam[torch.Tensor]]
    ) -> Union[LinearOperator, SpinParam[LinearOperator]]:
        """
        Obtains the LinearOperator of the exchange operator.
        It is -0.5 * K where K is the K matrix obtained from 2-electron integral.

        Exchange operator is a mathematical representation of the exchange
        interaction between identical particles, such as electrons. The
        exchange operator quantifies the effect of interchanging the
        positions of two particles.

        Parameters
        ----------
        dm: Union[torch.Tensor, SpinParam[torch.Tensor]]
            Density matrix. Shape: (`*BD`, nao, nao)

        Returns
        -------
        Union[LinearOperator, SpinParam[LinearOperator]]
            LinearOperator of the exchange operator. Shape: (`*BDH`, nao, nao)

        """
        pass

    @abstractmethod
    def get_vext(self, vext: torch.Tensor) -> LinearOperator:
        r"""
        Returns a LinearOperator of the external potential in the grid.

        .. math::
            \mathbf{V}_{ij} = \int b_i(\mathbf{r}) V(\mathbf{r}) b_j(\mathbf{r})\ d\mathbf{r}

        External potential energy that a particle experiences in a discretized
        space or grid. In quantum mechanics or computational physics, when
        solving for the behavior of particles, an external potential is often
        introduced to represent the influence of external forces.

        Parameters
        ----------
        vext: torch.Tensor
            External potential in the grid. Shape: (`*BR`, ngrid)

        Returns
        -------
        LinearOperator
            LinearOperator of the external potential in the grid. Shape: (`*BRH`, nao, nao)

        """
        pass

    @abstractmethod
    def get_vxc(
        self, dm: Union[torch.Tensor, SpinParam[torch.Tensor]]
    ) -> Union[LinearOperator, SpinParam[LinearOperator]]:
        """
        Returns a LinearOperator for the exchange-correlation potential

        The exchange-correlation potential combines two effects:

        1. Exchange potential: Arises from the antisymmetry of the electron
        wave function. It quantifies the tendency of electrons to avoid each
        other due to their indistinguishability.

        2. Correlation potential: Accounts for the electron-electron
        correlation effects that arise from the repulsion between electrons.

        TODO: check if what we need for Meta-GGA involving kinetics and for
        exact-exchange

        Parameters
        ----------
        dm: Union[torch.Tensor, SpinParam[torch.Tensor]]
            Density matrix. Shape: (`*BD`, nao, nao)

        Returns
        -------
        Union[LinearOperator, SpinParam[LinearOperator]]
            LinearOperator for the exchange-correlation potential. Shape: (`*BDH`, nao, nao)

        """
        pass

    # interface to dm
    @abstractmethod
    def ao_orb2dm(self, orb: torch.Tensor,
                  orb_weight: torch.Tensor) -> torch.Tensor:
        """Convert the atomic orbital to the density matrix.

        Parameters
        ----------
        orb: torch.Tensor
            Atomic orbital. Shape: (*BO, nao, norb)
        orb_weight: torch.Tensor
            Orbital weight. Shape: (*BW, norb)

        Returns
        -------
        torch.Tensor
            Density matrix. Shape: (*BOWH, nao, nao)

        """
        pass

    @abstractmethod
    def aodm2dens(self, dm: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        """Get the density value in the Cartesian coordinate.

        Parameters
        ----------
        dm: torch.Tensor
            Density matrix. Shape: (`*BD`, nao, nao)
        xyz: torch.Tensor
            Cartesian coordinate. Shape: (`*BR`, ndim)

        Returns
        -------
        torch.Tensor
            Density value in the Cartesian coordinate. Shape: (`*BRD`)

        """
        pass

    # energy of the Hamiltonian
    @abstractmethod
    def get_e_hcore(self, dm: torch.Tensor) -> torch.Tensor:
        """
        Get the energy from the one-electron Hamiltonian. The input is total
        density matrix.

        Parameters
        ----------
        dm: torch.Tensor
            Total Density matrix.

        Returns
        -------
        torch.Tensor
            Energy from the one-electron Hamiltonian.

        """
        pass

    @abstractmethod
    def get_e_elrep(self, dm: torch.Tensor) -> torch.Tensor:
        """
        Get the energy from the electron repulsion. The input is total density
        matrix.

        Parameters
        ----------
        dm: torch.Tensor
            Total Density matrix.

        Returns
        -------
        torch.Tensor
            Energy from the one-electron Hamiltonian.

        """
        pass

    @abstractmethod
    def get_e_exchange(
            self, dm: Union[torch.Tensor,
                            SpinParam[torch.Tensor]]) -> torch.Tensor:
        """Get the energy from the exact exchange.

        Parameters
        ----------
        dm: Union[torch.Tensor, SpinParam[torch.Tensor]]
            Density matrix.

        Returns
        -------
        torch.Tensor
            Energy from the exact exchange.

        """
        pass

    @abstractmethod
    def get_e_xc(
            self, dm: Union[torch.Tensor,
                            SpinParam[torch.Tensor]]) -> torch.Tensor:
        """
        Returns the exchange-correlation energy using the xc object given in
        ``.setup_grid()``

        Parameters
        ----------
        dm: Union[torch.Tensor, SpinParam[torch.Tensor]]
            Density matrix. Shape: (`*BD`, nao, nao)

        Returns
        -------
        torch.Tensor
            Exchange-correlation energy.

        """
        pass

    # free parameters for variational method
    @abstractmethod
    def ao_orb_params2dm(
        self,
        ao_orb_params: torch.Tensor,
        ao_orb_coeffs: torch.Tensor,
        orb_weight: torch.Tensor,
        with_penalty: Optional[float] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Convert the atomic orbital free parameters (parametrized in such a way
        so it is not bounded) to the density matrix.

        Parameters
        ----------
        ao_orb_params: torch.Tensor
            The tensor that parametrized atomic orbital in an unbounded space.
        ao_orb_coeffs: torch.Tensor
            The tensor that helps ``ao_orb_params`` in describing the orbital.
            The difference with ``ao_orb_params`` is that ``ao_orb_coeffs`` is
            not differentiable and not to be optimized in variational method.
        orb_weight: torch.Tensor
            The orbital weights.
        with_penalty: float or None
            If a float, it returns a tuple of tensors where the first element is
            ``dm``, and the second element is the penalty multiplied by the
            penalty weights. The penalty is to compensate the overparameterization
            of ``ao_orb_params``, stabilizing the Hessian for gradient calculation.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            The density matrix from the orbital parameters and (if ``with_penalty``)
            the penalty of the overparameterization of ``ao_orb_params``.

        Notes
        -----
        * The penalty should be 0 if ``ao_orb_params`` is from ``dm2ao_orb_params``.
        * The density matrix should be recoverable when put through ``dm2ao_orb_params``
          and ``ao_orb_params2dm``.

        """
        pass

    @abstractmethod
    def dm2ao_orb_params(
            self, dm: torch.Tensor, norb: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Convert from the density matrix to the orbital parameters.
        The map is not one-to-one, but instead one-to-many where there might
        be more than one orbital parameters to describe the same density matrix.
        For restricted systems, only one of the ``dm`` (``dm.u`` or ``dm.d``) is
        sufficient.

        Parameters
        ----------
        dm: torch.Tensor
            The density matrix.
        norb: int
            The number of orbitals for the system.

        Returns
        -------
        tuple of 2 torch.Tensor
            The atomic orbital parameters for the first returned value and the
            atomic orbital coefficients for the second value.

        """
        pass

    # Editable module
    @abstractmethod
    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        """Return the paramnames

        Parameters
        ----------
        methodname: str
            The name of the method.
        prefix: str (default "")
            The prefix of the paramnames.

        Returns
        -------
        List[str]
            The paramnames.

        """
        pass

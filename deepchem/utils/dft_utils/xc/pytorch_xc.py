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


class PyTorchGGA(BaseXC):
    """Generalized Gradient Approximation (GGA) XC functional.

    GGA improves upon LDA by incorporating the local gradient of the electron
    density in addition to the local density value. This allows GGA to better
    capture the inhomogeneities of real electron densities, giving improved
    accuracy for molecules, surfaces, and systems with rapidly varying densities.

    The PBE (Perdew-Burke-Ernzerhof) exchange functional implemented here uses
    an enhancement factor F(s) that modulates the LDA exchange energy based on
    the dimensionless reduced density gradient s = |∇ρ| / (2*(3π²)^(1/3)*ρ^(4/3)).

    Examples
    --------
    >>> from deepchem.utils.dft_utils import Mol, KS
    >>> from deepchem.utils.dft_utils.xc.pytorch_xc import PyTorchGGA
    >>> moldesc = "H 0 0 -0.74; H 0 0 0.74"
    >>> basis = "sto-3g"
    >>> system = Mol(moldesc=moldesc, basis=basis)
    >>> xc_dc = PyTorchGGA("gga_x_pbe")
    >>> ks = KS(system, xc_dc, variational=False)
    >>> _ = ks.run()

    Notes
    -----
    - Available functionals:
      1. gga_x_pbe
    - Users can create their own GGA functionals by subclassing PyTorchGGA and
      implementing a method that accepts (n, sigma) and returns the energy density.
    - ``getparamnames`` must be implemented in all subclasses of BaseXC.
    - ``__init__`` uses ``getattr`` to resolve the functional by name, so no changes
      to ``__init__`` are needed when adding new functionals.

    References
    ----------
    .. [1] Perdew, J. P., Burke, K., & Ernzerhof, M. (1996).
       Generalized Gradient Approximation Made Simple.
       Physical Review Letters, 77(18), 3865–3868.
       https://doi.org/10.1103/PhysRevLett.77.3865
    """

    def __init__(self, name: str = "gga_x_pbe"):
        """Initialize the PyTorchGGA XC functional.

        Parameters
        ----------
        name : str (default "gga_x_pbe")
            The name of the GGA functional to use.
        """
        super().__init__()
        self.name = name
        self.functional = getattr(self, self.name)

    @property
    def family(self) -> int:
        """Returns the family identifier number of the XC functional.

        Returns
        -------
        int
            Returns 2 for GGA, as defined in BaseXC.
        """
        return 2

    def gga_x_pbe(self, n: torch.Tensor,
                  sigma: torch.Tensor) -> torch.Tensor:
        """Calculates the PBE GGA exchange energy density based on [1].

        The PBE exchange energy density is:

        .. math::

            \\varepsilon_x(n, \\sigma) = A_x \\cdot n^{4/3} \\cdot F_x(s)

        where the LDA prefactor is:

        .. math::

            A_x = -\\frac{3}{4} \\left(\\frac{3}{\\pi}\\right)^{1/3}

        the PBE enhancement factor is:

        .. math::

            F_x(s) = 1 + \\kappa - \\frac{\\kappa^2}{\\kappa + \\mu s^2}

        and the dimensionless reduced density gradient (with libxc spin-channel
        scaling for the unpolarized case) is:

        .. math::

            s^2 = \\left(X_{2S} \\cdot 2^{1/3}\\right)^2
                  \\frac{\\sigma}{n^{8/3}}, \\quad
            X_{2S} = \\frac{1}{2\\,(6\\pi^2)^{1/3}}

        Parameters
        ----------
        n : torch.Tensor, shape (*BD, nr)
            Electron density ρ(r).  All values must be > 0.
        sigma : torch.Tensor, shape (*BD, nr)
            Contracted density gradient σ = |∇ρ|².
            This is the scalar dot product ∇ρ · ∇ρ, already summed over
            spatial dimensions.  All values must be ≥ 0.

        Returns
        -------
        torch.Tensor, shape (*BD, nr)
            PBE exchange energy density ε_x · ρ (volumetric, per unit volume).

        References
        ----------
        .. [1] Perdew, J. P., Burke, K., & Ernzerhof, M. (1996).
           Generalized Gradient Approximation Made Simple.
           Physical Review Letters, 77(18), 3865–3868.
           https://doi.org/10.1103/PhysRevLett.77.3865
        """
        # ── PBE parameters ────────────────────────────────────────────────────
        # LDA exchange prefactor  A_x = -(3/4)*(3/π)^(1/3)
        A_x = torch.tensor(-0.75 * (3.0 / torch.pi) ** (1.0 / 3.0),
                           dtype=n.dtype, device=n.device)

        # X2S = 1 / (2*(6π²)^(1/3))  converts libxc x → PBE s
        X2S = torch.tensor(
            1.0 / (2.0 * (6.0 * float(torch.pi) ** 2) ** (1.0 / 3.0)),
            dtype=n.dtype, device=n.device)

        # PBE κ and μ  (PRL 77, 3865, Table I)
        kappa = torch.tensor(0.8040,              dtype=n.dtype, device=n.device)
        mu    = torch.tensor(0.2195149727645171,  dtype=n.dtype, device=n.device)

        # Spin-channel prefactor for s² in the unpolarized case.
        # For spin=0 each channel carries ρ_σ = ρ/2 and |∇ρ_σ| = |∇ρ|/2, so:
        #   x_σ = |∇ρ_σ| / ρ_σ^(4/3) = 2^(1/3) · √σ / ρ^(4/3)
        # After spin-summing the prefactor survives only in s²:
        #   s² = (X2S · 2^(1/3))² · σ / ρ^(8/3)
        x2s_sfac_sq = (X2S * 2.0 ** (1.0 / 3.0)) ** 2

        # ── density power terms ───────────────────────────────────────────────
        n_safe = torch.clamp(n, min=1e-18)
        n43 = safepow(n_safe, 4.0 / 3.0)   # ρ^(4/3)
        n83 = n43 * n43                     # ρ^(8/3)

        # ── reduced gradient squared ──────────────────────────────────────────
        sigma_safe = torch.clamp(sigma, min=0.0)
        s2 = x2s_sfac_sq * sigma_safe / n83

        # ── PBE enhancement factor  F(s) = 1 + κ − κ²/(κ + μ·s²) ────────────
        denom = kappa + mu * s2
        F = 1.0 + kappa - kappa ** 2 / denom

        # ── exchange energy density  ε_x · ρ = A_x · ρ^(4/3) · F(s²) ─────────
        return A_x * n43 * F

    def get_edensityxc(
            self,
            densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> torch.Tensor:
        """Returns the XC energy density (energy per unit volume).

        For GGA, ``densinfo.grad`` carries the **gradient vector** of ρ with
        shape ``(*BD, ndim, nr)`` (not the scalar σ).  This method computes
        σ = |∇ρ|² = sum over the ``ndim`` axis so that the autograd graph
        through ``densinfo.grad`` is preserved for ``get_vxc``.

        For the spin-polarized case the input is a ``SpinParam[ValGrad]``
        with separate up/down components, and the spin-scaling relation is
        applied:

        .. math::

            E_x[\\rho_\\uparrow, \\rho_\\downarrow]
            = \\tfrac{1}{2}\\,E_x^{\\mathrm{unp}}[2\\rho_\\uparrow,\\,
              4\\sigma_{\\uparrow\\uparrow}]
            + \\tfrac{1}{2}\\,E_x^{\\mathrm{unp}}[2\\rho_\\downarrow,\\,
              4\\sigma_{\\downarrow\\downarrow}]

        Examples
        --------
        >>> import torch
        >>> from deepchem.utils.dft_utils.xc.pytorch_xc import PyTorchGGA
        >>> from deepchem.utils.dft_utils.data.datastruct import ValGrad
        >>> xc = PyTorchGGA()
        >>> n     = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
        >>> # grad shape is (ndim, nr) = (3, 3) as passed by the KS engine
        >>> grad  = torch.zeros(3, 3, dtype=torch.float64)
        >>> densinfo = ValGrad(value=n, grad=grad)
        >>> edensity = xc.get_edensityxc(densinfo)
        >>> edensity.shape
        torch.Size([3])

        Parameters
        ----------
        densinfo : Union[ValGrad, SpinParam[ValGrad]]
            Local density and gradient information.

            - ``ValGrad.value`` : electron density ρ, shape ``(*BD, nr)``
            - ``ValGrad.grad``  : gradient vector ∇ρ, shape ``(*BD, ndim, nr)``

        Returns
        -------
        torch.Tensor, shape (*BD, nr)
            The XC energy density ε_xc · ρ at each grid point.
        """
        if isinstance(densinfo, ValGrad):
            n = densinfo.value

            # densinfo.grad is the gradient *vector* ∇ρ with shape (*BD, ndim, nr).
            # sigma = |∇ρ|² = sum_i (∂ρ/∂x_i)²  summed over the spatial dim.
            # We keep this as an in-graph operation so autograd can compute
            # d(edensity)/d(densinfo.grad) correctly in get_vxc.
            if densinfo.grad is not None:
                sigma = (densinfo.grad ** 2).sum(dim=-2)   # (*BD, nr)
            else:
                sigma = torch.zeros_like(n)

            return self.functional(n, sigma)

        else:
            # Spin-polarized: apply the spin-scaling relation per channel.
            # Each channel's effective density is doubled and its contracted
            # gradient is quadrupled (ρ_eff = 2ρ_σ, σ_eff = 4σ_σσ); the
            # total is halved so that the integral over space is correct.
            nu = densinfo.u.value
            nd = densinfo.d.value

            if densinfo.u.grad is not None:
                sigma_u = (densinfo.u.grad ** 2).sum(dim=-2)
            else:
                sigma_u = torch.zeros_like(nu)

            if densinfo.d.grad is not None:
                sigma_d = (densinfo.d.grad ** 2).sum(dim=-2)
            else:
                sigma_d = torch.zeros_like(nd)

            ex_u = self.functional(2.0 * nu, 4.0 * sigma_u)
            ex_d = self.functional(2.0 * nd, 4.0 * sigma_d)
            return 0.5 * (ex_u + ex_d)

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        """List tensor names that affect the output of the given method.

        Parameters
        ----------
        methodname : str
            Name of the method of this class.
        prefix : str, optional
            Prefix to prepend to each parameter name (usually contains dots).

        Returns
        -------
        List[str]
            Empty because all constants are created as local tensors inside
            the functional and carry no learnable parameters.
        """
        return []


import torch
from typing import Union, List
from deepchem.utils.dft_utils.xc.base_xc import BaseXC
from deepchem.utils.dft_utils.data.datastruct import ValGrad, SpinParam
from deepchem.utils import safepow


class PyTorchMGGA(BaseXC):
    """Meta-Generalized Gradient Approximation (MGGA) XC functional.

    Meta-GGA extends GGA by additionally depending on the local kinetic energy
    density τ(r) = (1/2) Σ_i |∇ψ_i|². This extra ingredient allows the
    functional to distinguish single-orbital regions (where τ = τ_W, the
    von Weizsäcker KED) from slowly-varying or iso-orbital regions, giving
    improved accuracy for atomisation energies, barrier heights, and solids.

    The TPSS (Tao-Perdew-Staroverov-Scuseria) exchange functional implemented
    here is a fully non-empirical meta-GGA that satisfies a large number of
    exact constraints and is the reference for libxc ``mgga_x_tpss`` (id=202).

    Examples
    --------
    >>> from deepchem.utils.dft_utils import Mol, KS
    >>> from deepchem.utils.dft_utils.xc.pytorch_xc import PyTorchMGGA
    >>> moldesc = "H 0 0 -0.74; H 0 0 0.74"
    >>> basis = "sto-3g"
    >>> system = Mol(moldesc=moldesc, basis=basis)
    >>> xc_dc = PyTorchMGGA("mgga_x_tpss")
    >>> ks = KS(system, xc_dc, variational=False)
    >>> _ = ks.run()

    Notes
    -----
    - Available functionals:
      1. mgga_x_tpss
    - ``densinfo.grad`` has shape ``(*BD, nr, ndim)`` — ``nr`` before ``ndim``.
    - ``densinfo.kin``  has shape ``(*BD, nr)`` and carries τ(r), the
      positive-definite KS kinetic energy density.
    - ``getparamnames`` must be implemented in all subclasses of BaseXC.

    References
    ----------
    .. [1] Tao, J., Perdew, J. P., Staroverov, V. N., & Scuseria, G. E. (2003).
       Climbing the Density Functional Ladder: Nonempirical Meta-Generalized
       Gradient Approximation Designed for Molecules and Solids.
       Physical Review Letters, 91(14), 146401.
       https://doi.org/10.1103/PhysRevLett.91.146401
    """

    _SIGMA_EINSUM = "...dr,...dr->...r"

    def __init__(self, name: str = "mgga_x_tpss"):
        """Initialize the PyTorchMGGA XC functional.

        Parameters
        ----------
        name : str (default "mgga_x_tpss")
            The name of the MGGA functional to use.
        """
        super().__init__()
        self.name = name
        self.functional = getattr(self, self.name)

    @property
    def family(self) -> int:
        """Returns the family identifier number of the XC functional.

        Returns
        -------
        int
            Returns 4 for Meta-GGA, as defined in BaseXC.
        """
        return 4

    def mgga_x_tpss(self, n: torch.Tensor, sigma: torch.Tensor,
                    tau: torch.Tensor) -> torch.Tensor:
        """Calculates the TPSS meta-GGA exchange energy density based on [1].

        The TPSS exchange energy density is:

        .. math::

            \\varepsilon_x(n,\\sigma,\\tau) = A_x \\cdot n^{4/3}
                                              \\cdot F_x(p, \\alpha)

        where :math:`A_x = -(3/4)(3/\\pi)^{1/3}` is the LDA prefactor,
        :math:`p = s^2` is the reduced gradient squared, and

        .. math::

            \\alpha = \\frac{\\tau - \\tau_W}{\\tau_{\\mathrm{TF}}}

        is the iso-orbital indicator (0 in single-orbital, 1 in slowly-varying
        limit) with :math:`\\tau_W = \\sigma/(8n)` and
        :math:`\\tau_{\\mathrm{TF}} = (3/10)(3\\pi^2)^{2/3} n^{5/3}`.

        The enhancement factor is built in two stages:

        1. Compute :math:`q_b = (9/20)(\\alpha-1)/(1+b\\alpha(\\alpha-1)) + 2p/3`
        2. Compute :math:`x(p,\\alpha)` via Eq. (9) of the TPSS paper
        3. Apply the PBE-like clip: :math:`F_x = 1 + \\kappa_p - \\kappa_p^2/(\\kappa_p+x)`

        Parameters
        ----------
        n : torch.Tensor, shape (*BD, nr)
            Electron density ρ(r). All values must be > 0.
        sigma : torch.Tensor, shape (*BD, nr)
            Contracted density gradient σ = |∇ρ|². All values must be ≥ 0.
        tau : torch.Tensor, shape (*BD, nr)
            KS kinetic energy density τ(r) = (1/2)Σ|∇ψ_i|². All values > 0.

        Returns
        -------
        torch.Tensor, shape (*BD, nr)
            TPSS exchange energy density ε_x · ρ (volumetric, per unit volume).

        References
        ----------
        .. [1] Tao, J., Perdew, J. P., Staroverov, V. N., & Scuseria, G. E.
           (2003). Physical Review Letters, 91(14), 146401.
           https://doi.org/10.1103/PhysRevLett.91.146401
        """
        # ── TPSS parameters (PRL 91, 146401, Table I) ─────────────────────────
        kp = torch.tensor(0.804,    dtype=n.dtype, device=n.device)  # κ_p
        b  = torch.tensor(0.40,    dtype=n.dtype, device=n.device)   # b
        c  = torch.tensor(1.59096, dtype=n.dtype, device=n.device)   # c
        e  = torch.tensor(1.537,   dtype=n.dtype, device=n.device)   # e
        mu_GE = torch.tensor(10.0 / 81.0, dtype=n.dtype, device=n.device)

        # LDA exchange prefactor  A_x = -(3/4)*(3/π)^(1/3)
        A_x = torch.tensor(-0.75 * (3.0 / torch.pi) ** (1.0 / 3.0),
                           dtype=n.dtype, device=n.device)

        # ── safe inputs ───────────────────────────────────────────────────────
        n_safe   = torch.clamp(n,   min=1e-18)
        tau_safe = torch.clamp(tau, min=1e-18)
        sig_safe = torch.clamp(sigma, min=0.0)

        # ── density powers ────────────────────────────────────────────────────
        n43 = safepow(n_safe, 4.0 / 3.0)   # ρ^(4/3)
        n83 = n43 * n43                     # ρ^(8/3)

        # ── Thomas-Fermi KED ──────────────────────────────────────────────────
        # τ_TF = (3/10)*(3π²)^(2/3) * ρ^(5/3)
        tau_TF = (3.0 / 10.0) * (3.0 * torch.pi ** 2) ** (2.0 / 3.0) \
                 * safepow(n_safe, 5.0 / 3.0)

        # ── von Weizsäcker KED ────────────────────────────────────────────────
        # τ_W = σ / (8ρ)  (lower bound on τ; exact for one-orbital systems)
        tau_W = sig_safe / (8.0 * n_safe)

        # ── iso-orbital indicator  α = (τ − τ_W) / τ_TF  ∈ [0, ∞) ──────────
        # α = 0 : single-orbital region (e.g. one-electron density)
        # α = 1 : slowly-varying uniform electron gas limit
        alpha = torch.clamp((tau_safe - tau_W) / (tau_TF + 1e-30), min=0.0)

        # ── reduced gradient squared  p = s²  ────────────────────────────────
        # Uses the same libxc spin-channel convention as GGA_X_PBE:
        #   s² = (X2S·2^(1/3))²·σ/ρ^(8/3)
        X2S = torch.tensor(
            1.0 / (2.0 * (6.0 * float(torch.pi) ** 2) ** (1.0 / 3.0)),
            dtype=n.dtype, device=n.device)
        x2s_sfac_sq = (X2S * 2.0 ** (1.0 / 3.0)) ** 2
        p = x2s_sfac_sq * sig_safe / n83

        # ── TPSS Eq. (7): q_b ─────────────────────────────────────────────────
        # q_b = (9/20)*(α−1)/(1 + b*α*(α−1)) + (2/3)*p
        qb = (9.0 / 20.0) * (alpha - 1.0) \
             / (1.0 + b * alpha * (alpha - 1.0)) \
             + (2.0 / 3.0) * p

        # ── TPSS Eq. (8)/(9): x(p, α) numerator and denominator ─────────────
        #
        # The square-root argument  Θ = sqrt(0.5*(0.6p)² + 0.5*(9qb/20)²)
        # appears in both the numerator and denominator of x.
        theta = torch.sqrt(0.5 * (0.6 * p) ** 2 + 0.5 * (9.0 * qb / 20.0) ** 2)

        # Numerator of x (Eq. 9):
        #   (10/81 + c·z²(1−z)²)·p  +  (146/2025)·p²
        #   − (73/405)·p·Θ  +  e·(10/81)²·p²
        # where z ≡ τ_W/τ = α/(α+p/tau_TF·...) — but in PRL 91 the
        # variable used inside c-term is the local spin-channel ratio
        # z = τ_W/τ (which equals alpha/(alpha+1) in the UEG limit,
        # but pointwise z = tau_W/tau_safe for self-consistent use).
        z = tau_W / tau_safe   # pointwise iso-orbital ratio (not same as α)
        z = torch.clamp(z, max=1.0)

        num_x = (mu_GE + c * z ** 2 * (1.0 - z) ** 2) * p \
                + (146.0 / 2025.0) * p * p \
                - (73.0 / 405.0) * p * theta \
                + e * mu_GE ** 2 * p * p

        # Denominator of x (Eq. 9):
        den_x = 1.0 + theta + e * mu_GE ** 2 * p * p

        x_tpss = num_x / den_x

        # ── TPSS Eq. (13): PBE-like enhancement factor ────────────────────────
        # F_x = 1 + κ_p − κ_p² / (κ_p + x)
        Fx = 1.0 + kp - kp ** 2 / (kp + x_tpss)

        return A_x * n43 * Fx

    def get_edensityxc(
            self,
            densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> torch.Tensor:
        """Returns the XC energy density (energy per unit volume).

        For MGGA, ``densinfo.kin`` carries τ(r) and ``densinfo.grad`` carries
        ∇ρ with shape ``(*BD, nr, ndim)``.  sigma is computed via einsum,
        preserving the autograd graph needed by ``get_vxc``.

        For the spin-polarized case the spin-scaling relation is:

        .. math::

            E_x[\\rho_\\uparrow, \\rho_\\downarrow]
            = \\tfrac{1}{2}\\,E_x^{\\mathrm{unp}}[2\\rho_\\uparrow,\\,
              4\\sigma_{\\uparrow\\uparrow},\\, 2\\tau_\\uparrow]
            + \\tfrac{1}{2}\\,E_x^{\\mathrm{unp}}[2\\rho_\\downarrow,\\,
              4\\sigma_{\\downarrow\\downarrow},\\, 2\\tau_\\downarrow]

        Examples
        --------
        >>> import torch
        >>> from deepchem.utils.dft_utils.xc.pytorch_xc import PyTorchMGGA
        >>> from deepchem.utils.dft_utils.data.datastruct import ValGrad
        >>> xc   = PyTorchMGGA()
        >>> n    = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
        >>> grad = torch.zeros(3, 3, dtype=torch.float64)  # (nr=3, ndim=3)
        >>> tau  = torch.tensor([0.1, 0.3, 0.8], dtype=torch.float64)
        >>> densinfo = ValGrad(value=n, grad=grad, kin=tau)
        >>> xc.get_edensityxc(densinfo).shape
        torch.Size([3])

        Parameters
        ----------
        densinfo : Union[ValGrad, SpinParam[ValGrad]]
            - ``ValGrad.value`` : ρ, shape ``(*BD, nr)``
            - ``ValGrad.grad``  : ∇ρ, shape ``(*BD, nr, ndim)``
            - ``ValGrad.kin``   : τ, shape ``(*BD, nr)``

        Returns
        -------
        torch.Tensor, shape (*BD, nr)
        """
        if isinstance(densinfo, ValGrad):
            n = densinfo.value

            sigma = torch.einsum(self._SIGMA_EINSUM,
                                 densinfo.grad, densinfo.grad) \
                if densinfo.grad is not None else torch.zeros_like(n)

            # Fall back to τ_W lower bound if τ is not provided
            tau = densinfo.kin if densinfo.kin is not None \
                else sigma / (8.0 * torch.clamp(n, min=1e-18))

            return self.functional(n, sigma, tau)

        else:
            nu = densinfo.u.value
            nd = densinfo.d.value

            sigma_u = torch.einsum(self._SIGMA_EINSUM,
                                   densinfo.u.grad, densinfo.u.grad) \
                if densinfo.u.grad is not None else torch.zeros_like(nu)
            sigma_d = torch.einsum(self._SIGMA_EINSUM,
                                   densinfo.d.grad, densinfo.d.grad) \
                if densinfo.d.grad is not None else torch.zeros_like(nd)

            tau_u = densinfo.u.kin if densinfo.u.kin is not None \
                else sigma_u / (8.0 * torch.clamp(nu, min=1e-18))
            tau_d = densinfo.d.kin if densinfo.d.kin is not None \
                else sigma_d / (8.0 * torch.clamp(nd, min=1e-18))

            # τ scales like density: τ_eff = 2·τ_σ per spin channel
            ex_u = self.functional(2.0 * nu, 4.0 * sigma_u, 2.0 * tau_u)
            ex_d = self.functional(2.0 * nd, 4.0 * sigma_d, 2.0 * tau_d)
            return 0.5 * (ex_u + ex_d)

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        """List tensor names that affect the output of the given method.

        Parameters
        ----------
        methodname : str
            Name of the method of this class.
        prefix : str, optional
            Prefix to prepend to each parameter name.

        Returns
        -------
        List[str]
            Empty — all constants are local tensors with no learnable params.
        """
        return []

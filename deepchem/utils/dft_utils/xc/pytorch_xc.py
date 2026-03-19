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

import torch
from typing import Union, List
from deepchem.utils.dft_utils.xc.base_xc import BaseXC
from deepchem.utils.dft_utils.data.datastruct import ValGrad, SpinParam
from deepchem.utils import safepow


class PyTorchGGA(BaseXC):
    """Generalized Gradient Approximation (GGA) XC functional.

    GGA improves upon LDA by incorporating the local gradient of the electron
    density in addition to the local density value. This allows GGA to better
    capture the inhomogeneities of real electron densities, giving improved
    accuracy for molecules, surfaces, and systems with rapidly varying densities.

    The PBE (Perdew-Burke-Ernzerhof) exchange functional implemented here
    matches libxc ``GGA_X_PBE`` (id=101) to machine precision.

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
    - Available functionals: ``gga_x_pbe``
    - ``densinfo.grad`` has shape ``(*BD, ndim, nr)`` — spatial dimensions
      ``ndim`` come **before** grid points ``nr``, matching DeepChem's
      internal layout (see ``hcgto.py`` lines 718–753).
    - sigma is computed as ``torch.einsum("...dr,...dr->...r", grad, grad)``,
      which contracts over the ``ndim`` axis (``d``) and preserves the
      autograd graph needed by ``BaseXC.get_vxc``.

    References
    ----------
    .. [1] Perdew, J. P., Burke, K., & Ernzerhof, M. (1996).
       Generalized Gradient Approximation Made Simple.
       Physical Review Letters, 77(18), 3865–3868.
       https://doi.org/10.1103/PhysRevLett.77.3865
    """

    # grad layout in DeepChem: (*BD, ndim, nr)
    # contracting 'd' (ndim) and keeping 'r' (nr) gives sigma of shape (*BD, nr)
    _SIGMA_EINSUM = "...dr,...dr->...r"

    def __init__(self, name: str = "gga_x_pbe"):
        """Initialize the PyTorchGGA XC functional.

        Parameters
        ----------
        name : str (default "gga_x_pbe")
            Name of the GGA functional to use.
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
        """PBE GGA exchange energy density matching libxc GGA_X_PBE (id=101).

        .. math::

            \\varepsilon_x = A_x \\cdot n^{4/3} \\cdot F_x(s)

        where :math:`A_x = -(3/4)(3/\\pi)^{1/3}` and

        .. math::

            F_x(s) = 1 + \\kappa - \\frac{\\kappa^2}{\\kappa + \\mu s^2},
            \\quad
            s^2 = (X_{2S} \\cdot 2^{1/3})^2 \\frac{\\sigma}{n^{8/3}}

        The :math:`2^{1/3}` factor comes from libxc's spin-channel scaling:
        for spin=0 each channel has :math:`\\rho_\\sigma = \\rho/2` and
        :math:`|\\nabla\\rho_\\sigma| = |\\nabla\\rho|/2`, giving
        :math:`x_\\sigma = 2^{1/3}\\sqrt{\\sigma}/\\rho^{4/3}`.

        Parameters
        ----------
        n : torch.Tensor, shape (*BD, nr)
            Electron density ρ(r). All values > 0.
        sigma : torch.Tensor, shape (*BD, nr)
            Contracted density gradient σ = |∇ρ|². All values ≥ 0.

        Returns
        -------
        torch.Tensor, shape (*BD, nr)
            PBE exchange energy density ε_x · ρ  [Ha/bohr³].
        """
        # LDA exchange prefactor  A_x = -(3/4)*(3/π)^(1/3)
        A_x   = torch.tensor(-0.75*(3.0/torch.pi)**(1.0/3.0),
                              dtype=n.dtype, device=n.device)
        # X2S = 1/(2*(6π²)^(1/3)), spin-scaled: (X2S·2^(1/3))²
        X2S   = torch.tensor(1.0/(2.0*(6.0*float(torch.pi)**2)**(1.0/3.0)),
                              dtype=n.dtype, device=n.device)
        kappa = torch.tensor(0.8040,              dtype=n.dtype, device=n.device)
        mu    = torch.tensor(0.2195149727645171,  dtype=n.dtype, device=n.device)
        x2s2  = (X2S * 2.0**(1.0/3.0))**2       # (X2S·2^(1/3))²

        n_safe    = torch.clamp(n, min=1e-18)
        n43       = safepow(n_safe, 4.0/3.0)
        n83       = n43 * n43
        s2        = x2s2 * torch.clamp(sigma, min=0.0) / n83
        denom     = kappa + mu * s2
        Fx        = 1.0 + kappa - kappa**2 / denom
        return A_x * n43 * Fx

    def get_edensityxc(
            self,
            densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> torch.Tensor:
        """Returns the XC energy density (energy per unit volume).

        ``densinfo.grad`` has shape ``(*BD, ndim, nr)`` — the ``ndim``
        (spatial) axis comes before ``nr`` (grid points).  sigma is computed
        via einsum ``"...dr,...dr->...r"`` which contracts over ``d=ndim``
        and keeps ``r=nr``.  This preserves the autograd graph so that
        ``BaseXC.get_vxc`` can differentiate w.r.t. ``densinfo.grad``.

        For spin-polarized inputs the spin-scaling relation is applied:

        .. math::

            E_x[\\rho_\\uparrow, \\rho_\\downarrow]
            = \\tfrac12 E_x^\\text{unp}[2\\rho_\\uparrow,\\ 4\\sigma_{\\uparrow\\uparrow}]
            + \\tfrac12 E_x^\\text{unp}[2\\rho_\\downarrow,\\ 4\\sigma_{\\downarrow\\downarrow}]

        Examples
        --------
        >>> import torch
        >>> from deepchem.utils.dft_utils.xc.pytorch_xc import PyTorchGGA
        >>> from deepchem.utils.dft_utils.data.datastruct import ValGrad
        >>> xc   = PyTorchGGA()
        >>> n    = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
        >>> grad = torch.zeros(3, 3, dtype=torch.float64)  # (ndim=3, nr=3)
        >>> densinfo = ValGrad(value=n, grad=grad)
        >>> xc.get_edensityxc(densinfo).shape
        torch.Size([3])

        Parameters
        ----------
        densinfo : Union[ValGrad, SpinParam[ValGrad]]
            - ``ValGrad.value`` : ρ, shape ``(*BD, nr)``
            - ``ValGrad.grad``  : ∇ρ, shape ``(*BD, ndim, nr)``

        Returns
        -------
        torch.Tensor, shape (*BD, nr)
        """
        if isinstance(densinfo, ValGrad):
            n = densinfo.value
            sigma = torch.einsum(self._SIGMA_EINSUM,
                                 densinfo.grad, densinfo.grad) \
                if densinfo.grad is not None else torch.zeros_like(n)
            return self.functional(n, sigma)
        else:
            nu, nd = densinfo.u.value, densinfo.d.value
            sigma_u = torch.einsum(self._SIGMA_EINSUM,
                                   densinfo.u.grad, densinfo.u.grad) \
                if densinfo.u.grad is not None else torch.zeros_like(nu)
            sigma_d = torch.einsum(self._SIGMA_EINSUM,
                                   densinfo.d.grad, densinfo.d.grad) \
                if densinfo.d.grad is not None else torch.zeros_like(nd)
            return 0.5 * (self.functional(2.0*nu, 4.0*sigma_u) +
                          self.functional(2.0*nd, 4.0*sigma_d))

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        """Returns an empty list — all constants are local tensors."""
        return []


import torch
from typing import Union, List
from deepchem.utils.dft_utils.xc.base_xc import BaseXC
from deepchem.utils.dft_utils.data.datastruct import ValGrad, SpinParam
from deepchem.utils import safepow


class PyTorchMGGA(BaseXC):
    """Meta-Generalized Gradient Approximation (MGGA) XC functional.

    Meta-GGA extends GGA by additionally depending on the local kinetic energy
    density τ(r) = (1/2)Σ|∇ψᵢ|².  This allows the functional to distinguish
    single-orbital regions (τ = τ_W) from slowly-varying or iso-orbital regions.

    The TPSS (Tao-Perdew-Staroverov-Scuseria) exchange functional implemented
    here matches libxc ``MGGA_X_TPSS`` (id=202) to machine precision.

    The implementation follows the exact maple source ``tpss_x.mpl`` from
    libxc, with the following critical implementation notes:

    1. All intermediate quantities are expressed in terms of ``xs²`` and ``ts``
       directly — **``sqrt(sigma)`` is never taken**.  This avoids NaN gradients
       at grid points where ``sigma = 0`` (e.g. on symmetry axes), which would
       otherwise propagate through autograd into the Fock matrix and trigger
       the Hermitian check failure.
    2. The denominator of ``qb`` uses ``sqrt(1 + b·α·(α−1))``.
    3. The numerator contains ``(146/2025)·qb²`` (not ``p²``).
    4. The denominator is ``(1 + sqrt(e)·p)²``.
    5. ``MU_GE = 10/81`` is used in all terms **except** the last ``p³`` term
       which uses ``params_a_mu = 0.21951``.

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
    - Available functionals: ``mgga_x_tpss``
    - ``densinfo.grad`` has shape ``(*BD, ndim, nr)`` — ``ndim`` before ``nr``.
    - ``densinfo.kin``  has shape ``(*BD, nr)`` and carries τ(r).
    - Derivatives (``vrho``, ``vsigma``, ``vtau``) are obtained via autograd
      through ``BaseXC.get_vxc`` — no manual derivative formulas are needed.

    References
    ----------
    .. [1] Tao, J., Perdew, J. P., Staroverov, V. N., & Scuseria, G. E. (2003).
       Climbing the Density Functional Ladder: Nonempirical Meta-Generalized
       Gradient Approximation Designed for Molecules and Solids.
       Physical Review Letters, 91(14), 146401.
       https://doi.org/10.1103/PhysRevLett.91.146401
    .. [2] libxc source: ``maple/tpss_x.mpl``, ``src/mgga_x_tpss.c``
    """

    _SIGMA_EINSUM = "...dr,...dr->...r"

    def __init__(self, name: str = "mgga_x_tpss"):
        """Initialize the PyTorchMGGA XC functional.

        Parameters
        ----------
        name : str (default "mgga_x_tpss")
            Name of the MGGA functional to use.
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
        """TPSS meta-GGA exchange energy density matching libxc MGGA_X_TPSS (id=202).

        Following the exact libxc maple source ``tpss_x.mpl``:

        .. math::

            \\varepsilon_x = A_x \\cdot n^{4/3} \\cdot F_x

        where :math:`F_x = 1 + \\kappa - \\kappa^2 / (\\kappa + f_x)`.

        **Key implementation detail:** all intermediates are expressed in terms
        of :math:`x_s^2 = 2^{2/3}\\sigma/n^{8/3}` and
        :math:`t_s = 2^{2/3}\\tau/n^{5/3}`.  No ``sqrt(sigma)`` is ever taken,
        avoiding NaN autograd gradients at zero-gradient grid points.

        Intermediate variables:

        .. math::

            p       &= X_{2S}^2\\,x_s^2 \\\\
            z       &= x_s^2\\,/\\,(8\\,t_s) \\equiv \\tau_W/\\tau \\\\
            \\alpha  &= (t_s - x_s^2/8)\\,/\\,K_c,\\quad
                        K_c=(3/10)(6\\pi^2)^{2/3} \\\\
            q_b     &= \\tfrac{9}{20}(\\alpha-1)\\,/\\,
                        \\sqrt{1+b\\,\\alpha(\\alpha-1)}
                        + \\tfrac{2}{3}p \\\\
            s_i     &= \\sqrt{\\tfrac{1}{2}(\\tfrac{9}{25}z^2+p^2)}

        Numerator (from ``tpss_fxnum`` in the maple source):

        .. math::

            \\mathrm{num} =
              \\Bigl(\\mu_\\mathrm{GE} +
              \\frac{c\\,z^2}{(1+z^2)^2}\\Bigr)p
              + \\frac{146}{2025}q_b^2
              - \\frac{73}{405}q_b\\,s_i
              + \\frac{\\mu_\\mathrm{GE}^2}{\\kappa}p^2
              + 2\\sqrt{e}\\,\\mu_\\mathrm{GE}\\tfrac{9}{25}z^2
              + e\\,\\mu_\\mathrm{TPSS}\\,p^3

        where :math:`\\mu_\\mathrm{GE} = 10/81` (libxc global GE2 constant) and
        :math:`\\mu_\\mathrm{TPSS} = 0.21951` (TPSS parameter, last term only).

        Denominator: :math:`\\mathrm{den} = (1 + \\sqrt{e}\\,p)^2`.

        Parameters
        ----------
        n : torch.Tensor, shape (*BD, nr)
            Electron density ρ(r). All values > 0.
        sigma : torch.Tensor, shape (*BD, nr)
            Contracted density gradient σ = |∇ρ|². All values ≥ 0.
        tau : torch.Tensor, shape (*BD, nr)
            KS kinetic energy density τ(r) = (1/2)Σ|∇ψᵢ|². All values > 0.

        Returns
        -------
        torch.Tensor, shape (*BD, nr)
            TPSS exchange energy density ε_x · ρ  [Ha/bohr³].
        """
        # ── Constants ─────────────────────────────────────────────────────────
        kappa   = torch.tensor(0.8040,    dtype=n.dtype, device=n.device)
        b       = torch.tensor(0.40,      dtype=n.dtype, device=n.device)
        c       = torch.tensor(1.59096,   dtype=n.dtype, device=n.device)
        e       = torch.tensor(1.537,     dtype=n.dtype, device=n.device)
        # MU_GE: libxc global GE2 exchange coefficient = 10/81
        # Used in ALL numerator terms EXCEPT the last p^3 term.
        MU_GE   = torch.tensor(10.0/81.0, dtype=n.dtype, device=n.device)
        # mu_tpss: TPSS parameter = 0.21951, used ONLY in the e*mu*p^3 term.
        mu_tpss = torch.tensor(0.21951,   dtype=n.dtype, device=n.device)

        # X2S = 1/(2*(6π²)^(1/3));  K_FACTOR_C = (3/10)*(6π²)^(2/3) [spin-scaled TF]
        X2S        = torch.tensor(
            1.0 / (2.0*(6.0*float(torch.pi)**2)**(1.0/3.0)),
            dtype=n.dtype, device=n.device)
        K_FACTOR_C = torch.tensor(
            (3.0/10.0)*(6.0*float(torch.pi)**2)**(2.0/3.0),
            dtype=n.dtype, device=n.device)
        A_x = torch.tensor(
            -0.75*(3.0/float(torch.pi))**(1.0/3.0),
            dtype=n.dtype, device=n.device)

        # ── Safe inputs ───────────────────────────────────────────────────────
        n_safe   = torch.clamp(n,     min=1e-18)
        tau_safe = torch.clamp(tau,   min=1e-18)
        sig_safe = torch.clamp(sigma, min=0.0)

        # ── Spin-scaled squared x-variable and t-variable ─────────────────────
        # xs^2 = 2^(2/3) * sigma / rho^(8/3)
        # ts   = 2^(2/3) * tau   / rho^(5/3)
        #
        # IMPORTANT: xs^2 is computed directly from sigma — we NEVER take
        # sqrt(sigma).  This prevents NaN gradients at sigma=0 grid points
        # where d(sqrt(sigma))/d(sigma) = 1/(2*sqrt(sigma)) -> inf.
        rho83  = safepow(n_safe, 8.0/3.0)
        rho53  = safepow(n_safe, 5.0/3.0)
        xs2    = 2.0**(2.0/3.0) * sig_safe / rho83      # xs^2
        ts     = 2.0**(2.0/3.0) * tau_safe / rho53       # ts

        # ── TPSS dimensionless variables ──────────────────────────────────────
        # p = X2S^2 * xs^2
        p = X2S**2 * xs2

        # z = xs^2 / (8*ts) = tau_W / tau  (no sqrt needed)
        z = xs2 / (8.0 * ts)

        # alpha = (ts - xs^2/8) / K_FACTOR_C  (>= 0 by construction)
        alpha = torch.clamp((ts - xs2 / 8.0) / K_FACTOR_C, min=0.0)

        # ── qb: libxc Eq.(7) ─────────────────────────────────────────────────
        # qb = (9/20)*(alpha-1) / sqrt(1 + b*alpha*(alpha-1)) + (2/3)*p
        # Note sqrt, NOT plain division.
        # 1 + b*alpha*(alpha-1) >= 1 - b/4 = 0.9 for b=0.40 (always positive).
        denom_qb = torch.sqrt(
            torch.clamp(1.0 + b * alpha * (alpha - 1.0), min=1e-30))
        qb = (9.0/20.0) * (alpha - 1.0) / denom_qb + (2.0/3.0) * p

        # ── Inner sqrt argument: si = sqrt(0.5*(9/25*z^2 + p^2)) ─────────────
        # clamp to min=1e-30 (not 0.0) so autograd never evaluates 1/(2*sqrt(0)).
        # At sigma=0: z=p=0 so the argument is 0; sqrt(0) has gradient
        # d/dx[sqrt(x)] = 1/(2*sqrt(x)) -> inf, producing NaN in the Fock matrix.
        # The clamp floor 1e-30 is numerically negligible (sqrt(1e-30)=1e-15 and
        # it is multiplied by qb which is O(1), giving a contribution ~1e-15).
        si = torch.sqrt(torch.clamp(0.5*(9.0/25.0*z**2 + p**2), min=1e-30))

        # ── tpss_fxnum (Eq.10 in libxc maple source) ──────────────────────────
        # MU_GE (10/81) appears everywhere; mu_tpss (0.21951) only in p^3 term.
        num = (MU_GE + c * z**2 / (1.0 + z**2)**2) * p \
            + (146.0/2025.0) * qb**2 \
            - (73.0/405.0)   * qb * si \
            + MU_GE**2 / kappa * p**2 \
            + 2.0 * torch.sqrt(e) * MU_GE * (9.0/25.0) * z**2 \
            + e * mu_tpss * p**3

        # ── tpss_fxden ────────────────────────────────────────────────────────
        den = (1.0 + torch.sqrt(e) * p)**2

        # ── Enhancement factor and volumetric energy density ──────────────────
        fx  = num / den
        Fx  = 1.0 + kappa - kappa**2 / (kappa + fx)
        return A_x * safepow(n_safe, 4.0/3.0) * Fx

    def get_edensityxc(
            self,
            densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> torch.Tensor:
        """Returns the XC energy density (energy per unit volume).

        ``densinfo.grad`` has shape ``(*BD, ndim, nr)`` — spatial dimensions
        ``ndim`` come **before** grid points ``nr``, matching DeepChem's
        layout (see ``hcgto.py``).  sigma is computed via
        ``torch.einsum("...dr,...dr->...r", grad, grad)`` which:

        - contracts over ``d=ndim`` (the spatial axis) and keeps ``r=nr``
        - produces sigma of shape ``(*BD, nr)`` — **no** ``sqrt`` is taken,
          so autograd gradients are well-defined everywhere including at
          grid points where ``sigma = 0``.

        For the spin-polarized case the spin-scaling relation is applied:

        .. math::

            E_x[\\rho_\\uparrow,\\rho_\\downarrow]
            = \\tfrac12 E_x^\\text{unp}[2\\rho_\\uparrow,\\ 4\\sigma_{\\uparrow\\uparrow},
              \\ 2\\tau_\\uparrow]
            + \\tfrac12 E_x^\\text{unp}[2\\rho_\\downarrow,\\ 4\\sigma_{\\downarrow\\downarrow},
              \\ 2\\tau_\\downarrow]

        Examples
        --------
        >>> import torch
        >>> from deepchem.utils.dft_utils.xc.pytorch_xc import PyTorchMGGA
        >>> from deepchem.utils.dft_utils.data.datastruct import ValGrad
        >>> xc   = PyTorchMGGA()
        >>> n    = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
        >>> grad = torch.zeros(3, 3, dtype=torch.float64)  # (ndim=3, nr=3)
        >>> tau  = torch.tensor([0.1, 0.3, 0.8], dtype=torch.float64)
        >>> densinfo = ValGrad(value=n, grad=grad, kin=tau)
        >>> xc.get_edensityxc(densinfo).shape
        torch.Size([3])

        Parameters
        ----------
        densinfo : Union[ValGrad, SpinParam[ValGrad]]
            - ``ValGrad.value`` : ρ, shape ``(*BD, nr)``
            - ``ValGrad.grad``  : ∇ρ, shape ``(*BD, ndim, nr)``
            - ``ValGrad.kin``   : τ, shape ``(*BD, nr)``

        Returns
        -------
        torch.Tensor, shape (*BD, nr)
        """
        if isinstance(densinfo, ValGrad):
            n = densinfo.value
            # sigma = |∇ρ|² via einsum — no sqrt, safe autograd at sigma=0
            sigma = torch.einsum(self._SIGMA_EINSUM,
                                 densinfo.grad, densinfo.grad) \
                if densinfo.grad is not None else torch.zeros_like(n)
            tau = densinfo.kin if densinfo.kin is not None \
                else sigma / (8.0 * torch.clamp(n, min=1e-18))
            return self.functional(n, sigma, tau)
        else:
            nu, nd = densinfo.u.value, densinfo.d.value
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
            return 0.5 * (self.functional(2.0*nu, 4.0*sigma_u, 2.0*tau_u) +
                          self.functional(2.0*nd, 4.0*sigma_d, 2.0*tau_d))

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        """Returns an empty list — all constants are local tensors."""
        return []

from __future__ import annotations

from typing import Optional, Union
import torch

from deepchem.utils.dft_utils import ValGrad, SpinParam
from deepchem.utils.dft_utils.xc.base_xc import BaseXC

Tensor = torch.Tensor
DensInfoType = Union[ValGrad, SpinParam[ValGrad]]




class PyTorchLDA(BaseXC):
    family = 1

    def __init__(self, name: str, eps: float = 1e-30):
        self.name = name.lower()
        self.eps = eps
        allowed = {"lda_x"}
        if self.name not in allowed:
            raise ValueError(
                f"Unknown LDA functional '{name}'. Supported: {sorted(allowed)}"
            )

    def get_edensityxc(self, densinfo: DensInfoType) -> Tensor:
        c_x = -0.75 * (3.0 / torch.pi)**(1.0 / 3.0)

        if isinstance(densinfo, SpinParam):
            rho_u = torch.clamp(densinfo.u.value, min=self.eps)
            rho_d = torch.clamp(densinfo.d.value, min=self.eps)
            return c_x * (2.0**(1.0 / 3.0)) * (
                rho_u.pow(4.0 / 3.0) + rho_d.pow(4.0 / 3.0)
            )

        rho = torch.clamp(densinfo.value, min=self.eps)
        return c_x * rho.pow(4.0 / 3.0)

    def getparamnames(self, methodname: str, prefix: str = ""):
        if methodname in ["get_edensityxc", "get_vxc"]:
            return []
        raise KeyError(f"Unknown methodname: {methodname}")

class PyTorchGGA(BaseXC):
    family = 2

    def __init__(self, name: str, eps: float = 1e-30):
        self.name = name.lower()
        self.eps = eps
        allowed = {"gga_x_pbe"}
        if self.name not in allowed:
            raise ValueError(
                f"Unknown GGA functional '{name}'. Supported: {sorted(allowed)}"
            )

    def _sigma_from_grad(self, grad: Optional[Tensor], ref: Tensor) -> Tensor:
        if grad is None:
            return torch.zeros_like(ref)
        return torch.sum(grad**2, dim=-1)

    def _pbe_enhancement(self, rho: Tensor, sigma: Tensor) -> Tensor:
        rho = torch.clamp(rho, min=self.eps)
        sigma = torch.clamp(sigma, min=0.0)

        mu = 0.2195149727645171
        kappa = 0.804

        kf = (3.0 * torch.pi**2 * rho)**(1.0 / 3.0)
        s = torch.sqrt(sigma) / (2.0 * kf * rho + self.eps)
        return 1.0 + kappa - kappa / (1.0 + mu * s**2 / kappa)

    def get_edensityxc(self, densinfo: DensInfoType) -> Tensor:
        c_x = -0.75 * (3.0 / torch.pi)**(1.0 / 3.0)

        if isinstance(densinfo, SpinParam):
            pref = 2.0**(1.0 / 3.0)

            rho_u = torch.clamp(densinfo.u.value, min=self.eps)
            rho_d = torch.clamp(densinfo.d.value, min=self.eps)

            sigma_u = self._sigma_from_grad(densinfo.u.grad, rho_u)
            sigma_d = self._sigma_from_grad(densinfo.d.grad, rho_d)

            ex_lda_u = c_x * pref * rho_u.pow(4.0 / 3.0)
            ex_lda_d = c_x * pref * rho_d.pow(4.0 / 3.0)

            fx_u = self._pbe_enhancement(2.0 * rho_u, 4.0 * sigma_u)
            fx_d = self._pbe_enhancement(2.0 * rho_d, 4.0 * sigma_d)

            return 0.5 * ex_lda_u * fx_u + 0.5 * ex_lda_d * fx_d

        rho = torch.clamp(densinfo.value, min=self.eps)
        sigma = self._sigma_from_grad(densinfo.grad, rho)

        ex_lda = c_x * rho.pow(4.0 / 3.0)
        fx = self._pbe_enhancement(rho, sigma)
        return ex_lda * fx

    def getparamnames(self, methodname: str, prefix: str = ""):
        if methodname in ["get_edensityxc", "get_vxc"]:
            return []
        raise KeyError(f"Unknown methodname: {methodname}")

class PyTorchMGGA(BaseXC):
    family = 4

    def __init__(self, name: str, eps: float = 1e-30):
        self.name = name.lower()
        self.eps = eps
        allowed = {"mgga_x_tpss"}
        if self.name not in allowed:
            raise ValueError(
                f"Unknown MGGA functional '{name}'. Supported: {sorted(allowed)}"
            )

    def _sigma_from_grad(self, grad: Optional[Tensor], ref: Tensor) -> Tensor:
        if grad is None:
            return torch.zeros_like(ref)
        return torch.sum(grad**2, dim=-1)

    def _tau_from_densinfo(self, densinfo: ValGrad, ref: Tensor) -> Tensor:
        if densinfo.kin is None:
            return torch.zeros_like(ref)
        return densinfo.kin

    def _pbe_enhancement(self, rho: Tensor, sigma: Tensor) -> Tensor:
        rho = torch.clamp(rho, min=self.eps)
        sigma = torch.clamp(sigma, min=0.0)

        mu = 0.2195149727645171
        kappa = 0.804

        kf = (3.0 * torch.pi**2 * rho)**(1.0 / 3.0)
        s = torch.sqrt(sigma) / (2.0 * kf * rho + self.eps)
        return 1.0 + kappa - kappa / (1.0 + mu * s**2 / kappa)

    def _tau_correction(self, rho: Tensor, sigma: Tensor, tau: Tensor) -> Tensor:
        rho = torch.clamp(rho, min=self.eps)
        sigma = torch.clamp(sigma, min=0.0)
        tau = torch.clamp(tau, min=0.0)

        tau_w = sigma / (8.0 * rho + self.eps)
        tau_unif = (3.0 / 10.0) * (3.0 * torch.pi**2)**(2.0 / 3.0) * rho.pow(5.0 / 3.0)
        alpha = (tau - tau_w) / (tau_unif + self.eps)

        c = 0.15
        return 1.0 + c * alpha / (1.0 + alpha.abs())

    def get_edensityxc(self, densinfo: DensInfoType) -> Tensor:
        c_x = -0.75 * (3.0 / torch.pi)**(1.0 / 3.0)

        if isinstance(densinfo, SpinParam):
            pref = 2.0**(1.0 / 3.0)

            rho_u = torch.clamp(densinfo.u.value, min=self.eps)
            rho_d = torch.clamp(densinfo.d.value, min=self.eps)

            sigma_u = self._sigma_from_grad(densinfo.u.grad, rho_u)
            sigma_d = self._sigma_from_grad(densinfo.d.grad, rho_d)

            tau_u = self._tau_from_densinfo(densinfo.u, rho_u)
            tau_d = self._tau_from_densinfo(densinfo.d, rho_d)

            ex_lda_u = c_x * pref * rho_u.pow(4.0 / 3.0)
            ex_lda_d = c_x * pref * rho_d.pow(4.0 / 3.0)

            fx_u = self._pbe_enhancement(2.0 * rho_u, 4.0 * sigma_u) * \
                   self._tau_correction(2.0 * rho_u, 4.0 * sigma_u, 2.0 * tau_u)
            fx_d = self._pbe_enhancement(2.0 * rho_d, 4.0 * sigma_d) * \
                   self._tau_correction(2.0 * rho_d, 4.0 * sigma_d, 2.0 * tau_d)

            return 0.5 * ex_lda_u * fx_u + 0.5 * ex_lda_d * fx_d

        rho = torch.clamp(densinfo.value, min=self.eps)
        sigma = self._sigma_from_grad(densinfo.grad, rho)
        tau = self._tau_from_densinfo(densinfo, rho)

        ex_lda = c_x * rho.pow(4.0 / 3.0)
        fx_gga = self._pbe_enhancement(rho, sigma)
        fx_tau = self._tau_correction(rho, sigma, tau)

        return ex_lda * fx_gga * fx_tau

    def getparamnames(self, methodname: str, prefix: str = ""):
        if methodname in ["get_edensityxc", "get_vxc"]:
            return []
        raise KeyError(f"Unknown methodname: {methodname}")
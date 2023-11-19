from abc import abstractmethod
from typing import List
import torch
import xitorch as xt

class BaseOrbConverter(xt.EditableModule):
    """
    Converting the orbital from the original orbital which is orthogonal in
    the overlap-metric to a new basis.
    """
    @abstractmethod
    def nao(self) -> int:
        """
        Returns the number of atomic orbital in the new orbital basis.
        """
        pass

    @abstractmethod
    def convert_ortho_orb(self, orb: torch.Tensor) -> torch.Tensor:
        """
        Convert the orthogonal orbital into the new orbital basis sets.
        The new orbital basis sets are already orthogonal, so we don't need
        to do anything.
        """
        pass

    @abstractmethod
    def unconvert_to_ortho_dm(self, dm: torch.Tensor) -> torch.Tensor:
        """
        Unconvert (convert-back) the density matrix built from the orbitals in
        the new basis sets to the density matrix built from orthogonal orbitals.
        The orthogonal orbitals should be equal to the one referred to in function
        ``convert_ortho_orb``.
        """
        pass

    @abstractmethod
    def convert2(self, mat: torch.Tensor) -> torch.Tensor:
        """
        Convert the last 2 dimensions of the matrix with shape (..., nao, nao)
        into the new orbital basis sets with shape (..., nao2, nao2)
        """
        pass

    @abstractmethod
    def convert4(self, mat: torch.Tensor) -> torch.Tensor:
        """
        Convert the last 4 dimensions of the matrix with shape (..., nao, nao, nao, nao)
        into the new orbital basis sets with shape (..., nao2, nao2, nao2, nao2).
        """
        pass

    @abstractmethod
    def unconvert_dm(self, dm: torch.Tensor) -> torch.Tensor:
        """
        Convert back the density matrix from the density matrix in the new orbital
        basis with shape (..., nao2, nao2) to the original orbital basis
        (..., nao, nao)
        """
        pass

    @abstractmethod
    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        pass

class OrbitalOrthogonalizer(BaseOrbConverter):
    """
    Convert orbital to another type of orbital by orthogonalizing the basis sets.
    """
    def __init__(self, ovlp: torch.Tensor, threshold: float = 1e-6):
        ovlp_eival, ovlp_eivec = xt.linalg.symeig(xt.LinearOperator.m(ovlp, is_hermitian=True))
        acc_idx = ovlp_eival > threshold
        orthozer = ovlp_eivec[..., acc_idx] * (ovlp_eival[acc_idx]) ** (-0.5)  # (nao, nao2)
        self._orthozer = orthozer

    def nao(self) -> int:
        return self._orthozer.shape[-1]

    def convert_ortho_orb(self, orb: torch.Tensor) -> torch.Tensor:
        """
        Convert the orthogonal orbital into the new orbital basis sets.
        The new orbital basis sets are already orthogonal, so we don't need
        to do anything.
        """
        return orb

    def unconvert_to_ortho_dm(self, dm: torch.Tensor) -> torch.Tensor:
        return dm

    def convert2(self, mat: torch.Tensor) -> torch.Tensor:
        """
        Convert the last 2 dimensions of the matrix with shape (..., nao, nao)
        into the new orbital basis sets with shape (..., nao2, nao2)
        """
        res = self._orthozer.transpose(-2, -1).conj() @ mat @ self._orthozer
        return res

    def convert4(self, mat: torch.Tensor) -> torch.Tensor:
        """
        Convert the last 4 dimensions of the matrix with shape (..., nao, nao, nao, nao)
        into the new orbital basis sets with shape (..., nao2, nao2, nao2, nao2).
        """
        orthozer = self._orthozer
        res = torch.einsum("...ijkl,...im,...jn,...kp,...lq->...mnpq",
                           mat, orthozer, orthozer, orthozer, orthozer)
        return res

    def unconvert_dm(self, dm: torch.Tensor) -> torch.Tensor:
        """
        Convert back the density matrix from the density matrix in the new orbital
        basis with shape (..., nao2, nao2) to the original orbital basis
        (..., nao, nao)
        """
        dm = torch.einsum("...kl,ik,jl->...ij", dm, self._orthozer, self._orthozer.conj())
        return dm

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        if methodname in ["convert2", "convert4", "unconvert_dm"]:
            return [prefix + "_orthozer"]
        elif methodname in ["convert_ortho_orb", "unconvert_to_ortho_dm"]:
            return []
        else:
            raise KeyError(f"Unknown method {methodname}")

class IdentityOrbConverter(BaseOrbConverter):
    """
    Not converting the orbital
    """
    def __init__(self, ovlp: torch.Tensor):
        ovlp_eival, ovlp_eivec = xt.linalg.symeig(xt.LinearOperator.m(ovlp, is_hermitian=True))
        self._inv_sqrt_ovlp = (ovlp_eivec * ovlp_eival ** (-0.5)) @ ovlp_eivec.transpose(-2, -1).conj()
        self._sqrt_ovlp = (ovlp_eivec * ovlp_eival ** (0.5)) @ ovlp_eivec.transpose(-2, -1).conj()
        ovlp2 = (ovlp_eivec * ovlp_eival) @ ovlp_eivec.transpose(-2, -1).conj()
        self._nao = ovlp.shape[-1]

    def nao(self) -> int:
        return self._nao

    def convert_ortho_orb(self, orb: torch.Tensor) -> torch.Tensor:
        return self._inv_sqrt_ovlp @ orb

    def unconvert_to_ortho_dm(self, dm: torch.Tensor) -> torch.Tensor:
        return self._sqrt_ovlp @ dm @ self._sqrt_ovlp

    def convert2(self, mat: torch.Tensor) -> torch.Tensor:
        return mat

    def convert4(self, mat: torch.Tensor) -> torch.Tensor:
        return mat

    def unconvert_dm(self, dm: torch.Tensor) -> torch.Tensor:
        return dm

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        if methodname in ["convert2", "convert4", "unconvert_dm"]:
            return []
        elif methodname == "convert_ortho_orb":
            return [prefix + "_inv_sqrt_ovlp"]
        elif methodname == "unconvert_to_ortho_dm":
            return [prefix + "_sqrt_ovlp"]
        else:
            raise KeyError(f"Unknown method {methodname}")

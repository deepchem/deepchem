from typing import List
import torch
from deepchem.utils.differentiation_utils import EditableModule, LinearOperator, symeig

class OrbitalOrthogonalizer(EditableModule):
    """
    Convert orbital to another type of orbital by orthogonalizing the basis sets.
    """
    def __init__(self, ovlp: torch.Tensor, threshold: float = 1e-6):
        ovlp_eival, ovlp_eivec = symeig(LinearOperator.m(ovlp, is_hermitian=True))
        acc_idx = ovlp_eival > threshold
        orthozer = ovlp_eivec[..., acc_idx] * (ovlp_eival[acc_idx]) ** (-0.5)  # (nao, nao2)
        self._orthozer = orthozer

    def nao(self) -> int:
        return self._orthozer.shape[-1]

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
        dm = torch.einsum("...kl,ik,jl->ij", dm, self._orthozer, self._orthozer.conj())
        return dm

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        return [prefix + "_orthozer"]

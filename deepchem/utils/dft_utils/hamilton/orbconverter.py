import torch
from typing import List
from deepchem.utils.differentiation_utils import EditableModule, LinearOperator, symeig


class OrbitalOrthogonalizer(EditableModule):
    """Convert orbital to another type of orbital by orthogonalizing the basis sets.

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.dft_utils import OrbitalOrthogonalizer
    >>> ovlp = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
    >>> orthozer = OrbitalOrthogonalizer(ovlp)
    >>> orthozer.nao()
    2
    >>> mat = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
    >>> orthozer.convert2(mat)
    tensor([[1.0000, 0.0000],
            [0.0000, 1.0000]])

    """

    def __init__(self, ovlp: torch.Tensor, threshold: float = 1e-6):
        """Initialize the orbital orthogonalizer.

        Parameters
        ----------
        ovlp: torch.Tensor
            Overlap matrix of the original orbital basis sets.
        threshold: float
            Threshold to determine the accuracy of the overlap matrix.
        """
        ovlp_eival, ovlp_eivec = symeig(
            LinearOperator.m(ovlp, is_hermitian=True))
        acc_idx = ovlp_eival > threshold
        orthozer = ovlp_eivec[..., acc_idx] * (ovlp_eival[acc_idx])**(
            -0.5)  # (nao, nao2)
        self._orthozer = orthozer

    def nao(self) -> int:
        """Return the number of original atomic orbitals.

        Returns
        -------
        int
            Number of original atomic orbitals.
        """
        return self._orthozer.shape[-1]

    def convert2(self, mat: torch.Tensor) -> torch.Tensor:
        """
        Convert the last 2 dimensions of the matrix with shape (..., nao, nao)
        into the new orbital basis sets with shape (..., nao2, nao2)

        Parameters
        ----------
        mat: torch.Tensor
            Matrix to be converted.

        Returns
        -------
        torch.Tensor
            Converted matrix.
        """
        res = self._orthozer.transpose(-2, -1).conj() @ mat @ self._orthozer
        return res

    def convert4(self, mat: torch.Tensor) -> torch.Tensor:
        """
        Convert the last 4 dimensions of the matrix with shape (..., nao, nao, nao, nao)
        into the new orbital basis sets with shape (..., nao2, nao2, nao2, nao2).

        Parameters
        ----------
        mat: torch.Tensor
            Matrix to be converted.

        Returns
        -------
        torch.Tensor
            Converted matrix.
        """
        orthozer = self._orthozer
        res = torch.einsum("...ijkl,...im,...jn,...kp,...lq->...mnpq", mat,
                           orthozer, orthozer, orthozer, orthozer)
        return res

    def unconvert_dm(self, dm: torch.Tensor) -> torch.Tensor:
        """
        Convert back the density matrix from the density matrix in the new orbital
        basis with shape (..., nao2, nao2) to the original orbital basis
        (..., nao, nao)

        Parameters
        ----------
        dm: torch.Tensor
            Density matrix in the new orbital basis.

        Returns
        -------
        torch.Tensor
            Density matrix in the original orbital basis.
        """
        dm = torch.einsum("...kl,ik,jl->ij", dm, self._orthozer,
                          self._orthozer.conj())
        return dm

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        """Get the parameter names of the orbital orthogonalizer.

        Parameters
        ----------
        methodname: str
            Method name.
        prefix: str
            Prefix of the parameter names.

        Returns
        -------
        List[str]
            List of parameter names.
        """
        return [prefix + "_orthozer"]

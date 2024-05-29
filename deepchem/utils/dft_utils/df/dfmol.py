import torch
from typing import List, Optional
from deepchem.utils.differentiation_utils import LinearOperator
from deepchem.utils.dft_utils.hamilton.orbconverter import OrbitalOrthogonalizer
from deepchem.utils.dft_utils import LibcintWrapper, coul2c, coul3c, overlap, BaseDF, config
from deepchem.utils.dft_utils.data.datastruct import DensityFitInfo
from deepchem.utils import get_memory

class DFMol(BaseDF):
    """
    DFMol represents the class of density fitting for an isolated molecule.
    """
    def __init__(self, dfinfo: DensityFitInfo, wrapper: LibcintWrapper,
                 orthozer: Optional[OrbitalOrthogonalizer] = None):
        self.dfinfo = dfinfo
        self.wrapper = wrapper
        self._is_built = False
        self._precompute_elmat = True
        self._orthozer = orthozer

    def build(self) -> BaseDF:
        self._is_built = True

        # construct the matrix used to calculate the electron repulsion for
        # density fitting method
        method = self.dfinfo.method
        auxbasiswrapper = LibcintWrapper(self.dfinfo.auxbasis,
                                               spherical=self.wrapper.spherical)
        basisw, auxbw = LibcintWrapper.concatenate(self.wrapper, auxbasiswrapper)

        if method == "coulomb":
            print("Calculating the 2e2c integrals")
            j2c = coul2c(auxbw)  # (nxao, nxao)
            print("Calculating the 2e3c integrals")
            j3c = coul3c(basisw, other1=basisw,
                               other2=auxbw)  # (nao, nao, nxao)
        elif method == "overlap":
            j2c = overlap(auxbw)  # (nxao, nxao)
            # TODO: implement overlap3c
            raise NotImplementedError(
                "Density fitting with overlap minimization is not implemented")
        self._j2c = j2c  # (nxao, nxao)
        self._j3c = j3c  # (nao, nao, nxao)
        print("Precompute matrix for density fittings")
        self._inv_j2c = torch.inverse(j2c)

        # if the memory is too big, then don't precompute elmat
        if get_memory(j3c) > config.THRESHOLD_MEMORY:
            self._precompute_elmat = False
        else:
            self._precompute_elmat = True
            self._el_mat = torch.matmul(j3c, self._inv_j2c)  # (nao, nao, nxao)

        print("Density fitting done")
        return self

    def get_elrep(self, dm: torch.Tensor) -> LinearOperator:
        # dm: (*BD, nao, nao)
        # elrep_mat: (nao, nao, nao, nao)
        # return: (*BD, nao, nao)

        # convert the dm into the original cgto basis
        if self._orthozer is not None:
            dm = self._orthozer.unconvert_dm(dm)

        if self._precompute_elmat:
            df_coeffs = torch.einsum("...ij,ijk->...k", dm, self._el_mat)  # (*BD, nxao)
        else:
            temp = torch.einsum("...ij,ijl->...l", dm, self._j3c)
            df_coeffs = torch.einsum("...l,lk->...k", temp, self._inv_j2c)  # (*BD, nxao)

        mat = torch.einsum("...k,ijk->...ij", df_coeffs, self._j3c)  # (*BD, nao, nao)
        mat = (mat + mat.transpose(-2, -1)) * 0.5
        if self._orthozer is not None:
            mat = self._orthozer.convert2(mat)
        return LinearOperator.m(mat, is_hermitian=True)

    @property
    def j2c(self) -> torch.Tensor:
        return self._j2c

    @property
    def j3c(self) -> torch.Tensor:
        return self._j3c

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        if methodname == "get_elrep":
            if self._precompute_elmat:
                params = [prefix + "_el_mat", prefix + "_j3c"]
            else:
                params = [prefix + "_inv_j2c", prefix + "_j3c"]
            if self._orthozer is not None:
                pfix = prefix + "_orthozer."
                params += self._orthozer.getparamnames("unconvert_dm", prefix=pfix) + \
                    self._orthozer.getparamnames("convert2", prefix=pfix)
            return params
        else:
            raise KeyError("getparamnames has no %s method" % methodname)

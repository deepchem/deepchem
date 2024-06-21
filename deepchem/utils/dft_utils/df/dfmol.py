import torch
from typing import List, Optional
from deepchem.utils.differentiation_utils import LinearOperator
from deepchem.utils.dft_utils.hamilton.orbconverter import OrbitalOrthogonalizer
from deepchem.utils.dft_utils import LibcintWrapper, coul2c, coul3c, overlap, BaseDF, config
from deepchem.utils.dft_utils.data.datastruct import DensityFitInfo
from deepchem.utils import get_memory
import logging

logger = logging.getLogger(__name__)


class DFMol(BaseDF):
    """
    DFMol represents the class of density fitting for an isolated molecule.
    
    Density fitting is a standard technique in quantum chemistry as it
    helps to accelerate certain parts of a calculation, such as the
    computation of the electron repulsion energy, without significant
    loss of accuracy.

    Examples
    --------
    >>> from deepchem.utils.dft_utils.hamilton.intor.lcintwrap import LibcintWrapper
    >>> from deepchem.utils.dft_utils import OrbitalOrthogonalizer
    >>> from deepchem.utils.dft_utils import DFMol
    >>> from deepchem.utils.dft_utils.data.datastruct import DensityFitInfo

    >>> wrapper = LibcintWrapper("sto-3g")
    >>> dfinfo = DensityFitInfo("coulomb", "weigend")
    >>> orthozer = OrbitalOrthogonalizer(wrapper)
    >>> dfmol = DFMol(dfinfo, wrapper, orthozer)
    >>> dfmol.build()
    Density fitting done
    >>> dm = torch.rand(2, 5, 5)
    >>> elrep = dfmol.get_elrep(dm)
    >>> elrep
    LinearOperator with shape=(2, 5, 5) and dtype=float64

    """
    def __init__(self, dfinfo: DensityFitInfo, wrapper: LibcintWrapper,
                 orthozer: Optional[OrbitalOrthogonalizer] = None):
        """Initializes the DFMol Class.
        
        Parameters
        ----------
        dfinfo: DensityFitInfo
            Info about DF Method name and Auxiliary Basis Set.
        wrapper: LibcintWrapper
            Python wrapper for storing info of environment and
            parameters for the integrals calculation.
        orthozer: Optional[OrbitalOrthogonalizer] (default None)
            OrbitalOrthogonalizer for converting orbital to another
            type of orbital by orthogonalizing the basis sets.

        """
        self.dfinfo = dfinfo
        self.wrapper = wrapper
        self._is_built = False
        self._precompute_elmat = True
        self._orthozer = orthozer

    def build(self) -> BaseDF:
        """Build the Density Fitting Object

        Returns
        -------
        BaseDF
            The constructed density fitting object."""
        self._is_built = True

        # construct the matrix used to calculate the electron repulsion for
        # density fitting method
        method = self.dfinfo.method
        auxbasiswrapper = LibcintWrapper(self.dfinfo.auxbasis,
                                               spherical=self.wrapper.spherical)
        basisw, auxbw = LibcintWrapper.concatenate(self.wrapper, auxbasiswrapper)

        if method == "coulomb":
            logger.info("Calculating the 2e2c integrals")
            j2c = coul2c(auxbw)  # (nxao, nxao)
            logger.info("Calculating the 2e3c integrals")
            j3c = coul3c(basisw, other1=basisw,
                               other2=auxbw)  # (nao, nao, nxao)
        elif method == "overlap":
            j2c = overlap(auxbw)  # (nxao, nxao)
            # TODO: implement overlap3c
            raise NotImplementedError(
                "Density fitting with overlap minimization is not implemented")
        self._j2c = j2c  # (nxao, nxao)
        self._j3c = j3c  # (nao, nao, nxao)
        logger.info("Precompute matrix for density fittings")
        self._inv_j2c = torch.inverse(j2c)

        # if the memory is too big, then don't precompute elmat
        if get_memory(j3c) > config.THRESHOLD_MEMORY:
            self._precompute_elmat = False
        else:
            self._precompute_elmat = True
            self._el_mat = torch.matmul(j3c, self._inv_j2c)  # (nao, nao, nxao)

        logger.info("Density fitting done")
        return self

    def get_elrep(self, dm: torch.Tensor) -> LinearOperator:
        """
        Construct the electron repulsion linear operator from the given density
        matrix using the density fitting method.

        Parameters
        ----------
        dm : torch.Tensor
            Density matrix. (*BD, nao, nao)

        Returns
        -------
        LinearOperator
            The electron repulsion linear operator.

        """

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
        """Returns the 2-centre 2-electron integrals of the auxiliary basis.

        Returns
        -------
        torch.Tensor
            2-centre 2-electron integrals of the auxiliary basis.

        """
        return self._j2c

    @property
    def j3c(self) -> torch.Tensor:
        """
        Return the 3-centre 2-electron integrals of the auxiliary basis and the
        basis.

        Returns
        -------
        torch.Tensor
            3-centre 2-electron integrals of the auxiliary basis and the basis.

        """
        return self._j3c

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        """
        This method should list tensor names that affect the output of the
        method with name indicated in ``methodname``.

        Parameters
        ---------
        methodname: str
            The name of the method of the class.
        prefix: str (default="")
            The prefix to be appended in front of the parameters name.
            This usually contains the dots.

        Returns
        -------
        List[str]
            Sequence of name of parameters affecting the output of the method.

        """
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

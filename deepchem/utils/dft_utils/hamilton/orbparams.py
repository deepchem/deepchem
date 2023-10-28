"""
Derived from: https://github.com/diffqc/dqc/blob/master/dqc/hamilton/orbparams.py
"""
from typing import List
import torch

__all__ = ["BaseOrbParams", "QROrbParams", "MatExpOrbParams"]


class BaseOrbParams(object):
    """Class that provides free-parameterization of orthogonal orbitals.

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.dft_utils import BaseOrbParams
    >>> class MyOrbParams(BaseOrbParams):
    ...     @staticmethod
    ...     def params2orb(params, coeffs, with_penalty):
    ...         return params, coeffs
    ...     @staticmethod
    ...     def orb2params(orb):
    ...         return orb, torch.tensor([0], dtype=orb.dtype, device=orb.device)
    >>> params = torch.randn(3, 4, 5)
    >>> coeffs = torch.randn(3, 4, 5)
    >>> with_penalty = 0.1
    >>> orb, penalty = MyOrbParams.params2orb(params, coeffs, with_penalty)
    >>> params2, coeffs2 = MyOrbParams.orb2params(orb)
    >>> torch.allclose(params, params2)
    True

    """

    @staticmethod
    def params2orb(  # type: ignore[empty-body]
            params: torch.Tensor,
            coeffs: torch.Tensor,
            with_penalty: float = 0.0) -> List[torch.Tensor]:
        """
        Convert the parameters & coefficients to the orthogonal orbitals.
        ``params`` is the tensor to be optimized in variational method, while
        ``coeffs`` is a tensor that is needed to get the orbital, but it is not
        optimized in the variational method.

        Parameters
        ----------
        params: torch.Tensor
            The free parameters to be optimized.
        coeffs: torch.Tensor
            The coefficients to get the orthogonal orbitals.
        with_penalty: float (default 0.0)
            If not 0.0, return the penalty term for the free parameters.

        Returns
        -------
        orb: torch.Tensor
            The orthogonal orbitals.
        penalty: torch.Tensor
            The penalty term for the free parameters. If ``with_penalty`` is 0.0,
            this is not returned.

        """
        pass

    @staticmethod
    def orb2params(  # type: ignore[empty-body]
            orb: torch.Tensor) -> List[torch.Tensor]:
        """
        Get the free parameters from the orthogonal orbitals. Returns ``params``
        and ``coeffs`` described in ``params2orb``.

        Parameters
        ----------
        orb: torch.Tensor
            The orthogonal orbitals.

        Returns
        -------
        params: torch.Tensor
            The free parameters to be optimized.
        coeffs: torch.Tensor
            The coefficients to get the orthogonal orbitals.

        """
        pass


class QROrbParams(BaseOrbParams):
    """
    Orthogonal orbital parameterization using QR decomposition.
    The orthogonal orbital is represented by:

    P = QR

    Where Q is the parameters defining the rotation of the orthogonal tensor,
    and R is the coefficients tensor.

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.dft_utils import QROrbParams
    >>> params = torch.randn(3, 3)
    >>> coeffs = torch.randn(4, 3)
    >>> with_penalty = 0.1
    >>> orb, penalty = QROrbParams.params2orb(params, coeffs, with_penalty)
    >>> params2, coeffs2 = QROrbParams.orb2params(orb)

    """

    @staticmethod
    def params2orb(params: torch.Tensor,
                   coeffs: torch.Tensor,
                   with_penalty: float = 0.0) -> List[torch.Tensor]:
        """
        Convert the parameters & coefficients to the orthogonal orbitals.
        ``params`` is the tensor to be optimized in variational method, while
        ``coeffs`` is a tensor that is needed to get the orbital, but it is not
        optimized in the variational method.

        Parameters
        ----------
        params: torch.Tensor
            The free parameters to be optimized.
        coeffs: torch.Tensor
            The coefficients to get the orthogonal orbitals.
        with_penalty: float (default 0.0)
            If not 0.0, return the penalty term for the free parameters.

        Returns
        -------
        orb: torch.Tensor
            The orthogonal orbitals.
        penalty: torch.Tensor
            The penalty term for the free parameters. If ``with_penalty`` is 0.0,
            this is not returned.

        """
        orb, _ = torch.linalg.qr(params)
        if with_penalty == 0.0:
            return [orb]
        else:
            # QR decomposition's solution is not unique in a way that every column
            # can be multiplied by -1 and it still a solution
            # So, to remove the non-uniqueness, we will make the sign of the sum
            # positive.
            s1 = torch.sign(orb.sum(dim=-2, keepdim=True))  # (*BD, 1, norb)
            s2 = torch.sign(params.sum(dim=-2, keepdim=True))
            penalty = torch.mean((orb * s1 - params * s2)**2) * with_penalty
            return [orb, penalty]

    @staticmethod
    def orb2params(orb: torch.Tensor) -> List[torch.Tensor]:
        """
        Get the free parameters from the orthogonal orbitals. Returns ``params``
        and ``coeffs`` described in ``params2orb``.

        Parameters
        ----------
        orb: torch.Tensor
            The orthogonal orbitals.

        Returns
        -------
        params: torch.Tensor
            The free parameters to be optimized.
        coeffs: torch.Tensor
            The coefficients to get the orthogonal orbitals.

        """
        coeffs = torch.tensor([0], dtype=orb.dtype, device=orb.device)
        return [orb, coeffs]


class MatExpOrbParams(BaseOrbParams):
    """
    Orthogonal orbital parameterization using matrix exponential.
    The orthogonal orbital is represented by:

        P = matrix_exp(Q) @ C

    where C is an orthogonal coefficient tensor, and Q is the parameters defining
    the rotation of the orthogonal tensor.

    Examples
    --------
    >>> from deepchem.utils.dft_utils import MatExpOrbParams
    >>> params = torch.randn(3, 3)
    >>> coeffs = torch.randn(4, 3)
    >>> with_penalty = 0.1
    >>> orb, penalty = MatExpOrbParams.params2orb(params, coeffs, with_penalty)
    >>> params2, coeffs2 = MatExpOrbParams.orb2params(orb)

    """

    @staticmethod
    def params2orb(params: torch.Tensor,
                   coeffs: torch.Tensor,
                   with_penalty: float = 0.0) -> List[torch.Tensor]:
        """
        Convert the parameters & coefficients to the orthogonal orbitals.
        ``params`` is the tensor to be optimized in variational method, while
        ``coeffs`` is a tensor that is needed to get the orbital, but it is not
        optimized in the variational method.

        Parameters
        ----------
        params: torch.Tensor
            The free parameters to be optimized. (*, nparams)
        coeffs: torch.Tensor
            The coefficients to get the orthogonal orbitals. (*, nao, norb)
        with_penalty: float (default 0.0)
            If not 0.0, return the penalty term for the free parameters.

        Returns
        -------
        orb: torch.Tensor
            The orthogonal orbitals.
        penalty: torch.Tensor
            The penalty term for the free parameters. If ``with_penalty`` is 0.0,
            this is not returned.

        """
        nao = coeffs.shape[-2]
        norb = coeffs.shape[-1]  # noqa: F841
        nparams = params.shape[-1]
        bshape = params.shape[:-1]

        # construct the rotation parameters
        triu_idxs = torch.triu_indices(nao, nao, offset=1)[..., :nparams]
        rotmat = torch.zeros((*bshape, nao, nao),
                             dtype=params.dtype,
                             device=params.device)
        rotmat[..., triu_idxs[0], triu_idxs[1]] = params
        rotmat = rotmat - rotmat.transpose(-2, -1).conj()

        # calculate the orthogonal orbital
        ortho_orb = torch.matrix_exp(rotmat) @ coeffs

        if with_penalty != 0.0:
            penalty = torch.zeros((1,),
                                  dtype=params.dtype,
                                  device=params.device)
            return [ortho_orb, penalty]
        else:
            return [ortho_orb]

    @staticmethod
    def orb2params(orb: torch.Tensor) -> List[torch.Tensor]:
        """
        Get the free parameters from the orthogonal orbitals. Returns ``params``
        and ``coeffs`` described in ``params2orb``.

        Parameters
        ----------
        orb: torch.Tensor
            The orthogonal orbitals.

        Returns
        -------
        params: torch.Tensor
            The free parameters to be optimized.
        coeffs: torch.Tensor
            The coefficients to get the orthogonal orbitals.

        """
        # orb: (*, nao, norb)
        nao = orb.shape[-2]
        norb = orb.shape[-1]
        nparams = norb * (nao - norb) + norb * (norb - 1) // 2

        # the orbital becomes the coefficients while params is all zeros (no rotation)
        coeffs = orb
        params = torch.zeros((*orb.shape[:-2], nparams),
                             dtype=orb.dtype,
                             device=orb.device)
        return [params, coeffs]

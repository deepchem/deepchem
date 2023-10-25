from typing import Tuple
import torch
import numpy as np
from scipy.special import erfcinv


class Lattice(object):
    """
    Lattice is an object that describe the periodicity of the lattice.
    Note that this object does not know about atoms.
    For the integrated object between the lattice and atoms, please see Sol

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.dft_utils import Lattice
    >>> a = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    >>> lattice = Lattice(a)
    >>> lattice.lattice_vectors()
    tensor([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])
    >>> lattice.recip_vectors()
    tensor([[6.2832, 0.0000, 0.0000],
            [0.0000, 6.2832, 0.0000],
            [0.0000, 0.0000, 6.2832]])
    >>> lattice.volume() # volume of the unit cell
    tensor(1.)
    >>> lattice.get_lattice_ls(1.0) # get the neighboring lattice vectors
    tensor([[ 0.,  0., -1.],
            [ 0., -1.,  0.],
            [-1.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 1.,  0.,  0.],
            [ 0.,  1.,  0.],
            [ 0.,  0.,  1.]])
    >>> lattice.get_gvgrids(6.0) # get the neighboring G-vectors
    (tensor([[ 0.0000,  0.0000, -6.2832],
            [ 0.0000, -6.2832,  0.0000],
            [-6.2832,  0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000],
            [ 6.2832,  0.0000,  0.0000],
            [ 0.0000,  6.2832,  0.0000],
            [ 0.0000,  0.0000,  6.2832]]), tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]))
    >>> lattice.estimate_ewald_eta(1e-5) # estimate the ewald's sum eta
    1.8

    """

    def __init__(self, a: torch.Tensor):
        """Initialize the lattice object.

        2D or 1D repetition are not implemented yet

        Parameters
        ----------
        a: torch.Tensor
            The lattice vectors with shape (ndim, ndim) with ndim == 3

        """
        assert a.ndim == 2
        assert a.shape[0] == 3
        assert a.shape[-1] == 3
        self.ndim = a.shape[0]
        self.a = a
        self.device = a.device
        self.dtype = a.dtype

    def lattice_vectors(self) -> torch.Tensor:
        """Returns the 3D lattice vectors (nv, ndim) with nv == 3"""
        return self.a

    def recip_vectors(self) -> torch.Tensor:
        """
        Returns the 3D reciprocal vectors with norm == 2 * pi with shape (nv, ndim)
        with nv == 3

        Note: ``torch.det(self.a)`` should not be equal to zero.
        """
        return torch.inverse(self.a.transpose(-2, -1)) * (2 * np.pi)

    def volume(self) -> torch.Tensor:
        """Returns the volume of a lattice."""
        return torch.det(self.a)

    @property
    def params(self) -> Tuple[torch.Tensor, ...]:
        """Returns the list of parameters of this object"""
        return (self.a,)

    def get_lattice_ls(self,
                       rcut: float,
                       exclude_zeros: bool = False) -> torch.Tensor:
        """
        Returns a tensor that contains the coordinates of the neighboring
        lattices.

        Parameters
        ----------
        rcut: float
            The threshold of the distance from the main cell to be included
            in the neighbor.
        exclude_zeros: bool (default: False)
            If True, then it will exclude the vector that are all zeros.

        Returns
        -------
        ls: torch.Tensor
            Tensor with size `(nb, ndim)` containing the coordinates of the
            neighboring cells.

        """
        a = self.lattice_vectors()
        return self._generate_lattice_vectors(a,
                                              rcut,
                                              exclude_zeros=exclude_zeros)

    def get_gvgrids(
            self,
            gcut: float,
            exclude_zeros: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a tensor that contains the coordinate in reciprocal space of the
        neighboring Brillouin zones.

        Parameters
        ----------
        gcut: float
            Cut off for generating the G-points.
        exclude_zeros: bool (default: False)
            If True, then it will exclude the vector that are all zeros.

        Returns
        -------
        gvgrids: torch.Tensor
            Tensor with size `(ng, ndim)` containing the G-coordinates of the
            Brillouin zones.
        weights: torch.Tensor
            Tensor with size `(ng)` representing the weights of the G-points.

        """
        b = self.recip_vectors()
        gvgrids = self._generate_lattice_vectors(b,
                                                 gcut,
                                                 exclude_zeros=exclude_zeros)

        # 1 / cell.vol == det(b) / (2 pi)^3
        weights = torch.zeros(gvgrids.shape[0],
                              dtype=self.dtype,
                              device=self.device)
        weights = weights + torch.abs(torch.det(b)) / (2 * np.pi)**3
        return gvgrids, weights

    def estimate_ewald_eta(self, precision: float) -> float:
        """estimate the ewald's sum eta for nuclei interaction energy the
        precision is assumed to be relative precision this formula is obtained
        by estimating the sum as an integral.

        Parameters
        ----------
        precision: float
            The precision of the ewald's sum.

        Returns
        -------
        eta: float
            The estimated eta.

        """
        vol = float(self.volume().detach())
        eta0 = np.sqrt(np.pi) / vol**(1. / 3)
        eta = eta0 * erfcinv(0.5 * precision) / erfcinv(precision)
        return round(eta * 10) / 10  # round to 1 d.p.

    def _generate_lattice_vectors(self, a: torch.Tensor, rcut: float,
                                  exclude_zeros: bool) -> torch.Tensor:
        """generate the lattice vectors of multiply of a within the radius rcut.

        Parameters
        ----------
        a: torch.Tensor
            The lattice vectors with shape (ndim, ndim) with ndim == 3
        rcut: float
            The threshold of the distance from the main cell to be included
            in the neighbor.
        exclude_zeros: bool (default: False)
            If True, then it will exclude the vector that are all zeros.

        Returns
        -------
        ls: torch.Tensor
            Tensor with size `(nb, ndim)` containing the coordinates of the
            neighboring cells.

        Derived From pyscf:
        https://github.com/pyscf/pyscf/blob/e6c569932d5bab5e49994ae3dd365998fc5202b5/pyscf/pbc/tools/pbc.py#L473

        """
        b = torch.inverse(a.transpose(-2, -1))
        heights_inv = torch.norm(b, dim=-1).detach().numpy()  # (ndim)
        nimgs = (rcut * heights_inv + 1.1).astype(np.int32)  # (ndim)

        n1_0 = torch.arange(-nimgs[0],
                            nimgs[0] + 1,
                            dtype=torch.int32,
                            device=self.device)  # (nimgs2,)
        n1_1 = torch.arange(-nimgs[1],
                            nimgs[1] + 1,
                            dtype=torch.int32,
                            device=self.device)  # (nimgs2,)
        n1_2 = torch.arange(-nimgs[2],
                            nimgs[2] + 1,
                            dtype=torch.int32,
                            device=self.device)  # (nimgs2,)
        ls = n1_2[:, None] * a[0, :]  # (nimgs2, ndim)
        ls = ls + n1_1[:, None, None] * a[1, :]  # (nimgs2, nimgs2, ndim)
        ls = ls + n1_0[:, None, None,
                       None] * a[2, :]  # (nimgs2, nimgs2, nimgs2, ndim)
        ls = ls.view(-1, ls.shape[-1])  # (nb, ndim)

        # drop ls that has norm > rcut * 1.05
        ls = ls[ls.norm(dim=-1) <= rcut * 1.05, :]  # (nb2, ndim)

        if exclude_zeros:
            ls = ls[torch.any(ls != 0, dim=-1), :]

        return ls

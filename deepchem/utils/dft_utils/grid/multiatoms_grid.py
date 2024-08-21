import torch
import numpy as np
from typing import List, Optional, Tuple
from deepchem.utils.dft_utils import Lattice, BaseGrid, LebedevGrid


class BeckeGrid(BaseGrid):
    """
    Using Becke's scheme to construct the 3D grid consists of multiple 3D grids
    centered on each atom.

    Examples
    --------
    >>> from deepchem.utils.dft_utils import LebedevGrid, BeckeGrid, RadialGrid
    >>> grid = RadialGrid(100, grid_integrator="chebyshev", grid_transform="logm3")
    >>> atomgrid = [LebedevGrid(grid, 3), LebedevGrid(grid, 3)]
    >>> atompos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
    >>> grid = BeckeGrid(atomgrid, atompos)
    >>> grid.get_rgrid().shape
    torch.Size([1200, 3])
    >>> grid.get_dvolume().shape
    torch.Size([1200])

    """

    def __init__(self,
                 atomgrid: List[LebedevGrid],
                 atompos: torch.Tensor,
                 atomradii: Optional[torch.Tensor] = None,
                 ratom_adjust: str = "becke") -> None:
        """Initialize the Becke grid.

        Parameters
        ----------
        atomgrid: List[LebedevGrid]
            List of Lebedev grids for each atom. (natoms)
        atompos: torch.Tensor
            Position of each atom. (natoms, ndim)
        atomradii: Optional[torch.Tensor] (default None)
            Radii of each atom. (natoms,) or None
        ratom_adjust: str (default "becke")
            Adjustment method for the atom radii. Available: ['becke', 'treutler']

        """
        assert atompos.shape[0] == len(atomgrid), \
            "The lengths of atomgrid and atompos must be the same"
        assert len(atomgrid) > 0
        self._dtype = atomgrid[0].dtype
        self._device = atomgrid[0].device

        # construct the grid points positions, weights, and the index of grid corresponding to each atom
        rgrids, self._rgrid, dvol_atoms = _construct_rgrids(atomgrid, atompos)

        # calculate the integration weights
        weights_atoms = _get_atom_weights(rgrids,
                                          atompos,
                                          atomradii=atomradii,
                                          ratom_adjust=ratom_adjust)  # (ngrid,)
        self._dvolume = dvol_atoms * weights_atoms

    @property
    def dtype(self):
        """Return the data type of the grid.

        Returns
        -------
        torch.dtype
            Data type of the grid.

        """
        return self._dtype

    @property
    def device(self):
        """device of the grid points

        Returns
        -------
        torch.device
            device of the grid points

        """
        return self._device

    @property
    def coord_type(self):
        """type of the coordinate returned in get_rgrid.

        Returns
        -------
        str
            type of the coordinate returned in get_rgrid. It can be 'cartesian'
            or 'spherical'.

        """
        return "cart"

    def get_dvolume(self):
        """Obtain the torch.tensor containing the dV elements for the integration.

        Returns
        -------
        torch.tensor (*BG, ngrid)
            The dV elements for the integration. *BG is the length of the BaseGrid.

        """
        return self._dvolume

    def get_rgrid(self) -> torch.Tensor:
        """
        Returns the grid points position in the specified coordinate in
        self.coord_type.

        Returns
        -------
        torch.tensor (*BG, ngrid, ndim)
            The grid points position. *BG is the length of the BaseGrid.

        """
        return self._rgrid

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        """
        Return a list with the parameter names corresponding to the given method
        (methodname)

        Returns
        -------
        List[str]
            List of parameter names of methodname

        """
        if methodname == "get_rgrid":
            return [prefix + "_rgrid"]
        elif methodname == "get_dvolume":
            return [prefix + "_dvolume"]
        else:
            raise KeyError("Invalid methodname: %s" % methodname)


class PBCBeckeGrid(BaseGrid):
    """
    Use Becke's scheme to construct the 3D grid in a periodic cell. It is similar
    to non-pbc BeckeGrid, but in this case, only grid points inside the lattice
    are considered, and atoms corresponds to each grid points are involved in
    calculating the weights.

    Examples
    --------
    >>> from deepchem.utils.dft_utils import LebedevGrid, PBCBeckeGrid, RadialGrid, Lattice
    >>> grid = RadialGrid(100, grid_integrator="chebyshev", grid_transform="logm3")
    >>> atomgrid = [LebedevGrid(grid, 3), LebedevGrid(grid, 3)]
    >>> atompos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
    >>> a = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=torch.float64)
    >>> lattice = Lattice(a)
    >>> grid = PBCBeckeGrid(atomgrid, atompos, lattice)
    >>> grid.get_rgrid().shape
    torch.Size([720, 3])
    >>> grid.get_dvolume().shape
    torch.Size([720])

    """

    def __init__(self,
                 atomgrid: List[LebedevGrid],
                 atompos: torch.Tensor,
                 lattice: Lattice,
                 ratom_adjust: str = "becke"):
        """Initialize the PBCBecke grid.

        Parameters
        ----------
        atomgrid: List[LebedevGrid]
            List of Lebedev grids for each atom. (natoms)
        atompos: torch.Tensor
            Position of each atom. (natoms, ndim)
        lattice: Lattice
            Lattice object for the periodic cell.
        ratom_adjust: str (default "becke")
            Adjustment method for the atom radii. Available: ['becke', 'treutler']

        """
        assert atompos.shape[0] == len(atomgrid), \
            "The lengths of atomgrid and atompos must be the same"
        assert len(atomgrid) > 0
        self._dtype = atomgrid[0].dtype
        self._device = atomgrid[0].device

        # get the normalized coordinates
        a = lattice.lattice_vectors()  # (nlvec=ndim, ndim)
        b = lattice.recip_vectors() / (
            2 * np.pi)  # (ndim, ndim) just the inverse of lattice vector.T

        new_atompos_lst: List[torch.Tensor] = []
        new_rgrids: List[torch.Tensor] = []
        new_dvols: List[torch.Tensor] = []
        for ia, atomgr in enumerate(atomgrid):
            atpos = atompos[ia]
            rgrid = atomgr.get_rgrid() + atpos  # (natgrid, ndim)
            dvols = atomgr.get_dvolume()  # (natgrid)

            # ugrid is the normalized coordinate
            ugrid = torch.einsum("cd,gd->gc", b, rgrid)  # (natgrid, ndim)

            # get the shift required to make the grid point inside the lattice
            ns = -ugrid.floor().to(
                torch.int
            )  # (natgrid, ndim) # ratoms + ns @ a will be the new atompos

            # ns_unique: (nunique, ndim), ns_unique_idx: (natgrid,), ns_count: (nunique)
            ns_unique, ns_unique_idx, ns_count = torch.unique(
                ns, dim=0, return_inverse=True, return_counts=True)

            # ignoring the shifts with only not more than 8 points (following pyscf)
            significant_uniq_idx = ns_count > 8  # (nunique)
            significant_idx = significant_uniq_idx[ns_unique_idx]  # (natgrid,)
            ns_unique = ns_unique[significant_uniq_idx, :]  # (nunique2, ndim)
            ls_unique = torch.matmul(ns_unique.to(a.dtype),
                                     a)  # (nunique2, ndim)

            # flag the unaccepted points with -1
            flag = -1
            ns_unique_idx[~significant_idx] = flag  # (natgrid)

            # get the coordinate inside the lattice
            ls = torch.matmul(ns.to(a.dtype), a)
            rg = rgrid + ls  # (natgrid, ndim)

            # get the new atom pos
            new_atpos = atompos[ia] + ls_unique  # (nunique2, ndim)
            new_atompos_lst.append(new_atpos)

            # separate the grid points that corresponds to the different atoms
            for idx in torch.unique(ns_unique_idx):
                if idx == flag:
                    continue
                at_idx = ns_unique_idx == idx
                new_rgrids.append(rg[at_idx, :])  # list of (natgrid2, ndim)
                new_dvols.append(dvols[at_idx])  # list of (natgrid2,)

        self._rgrid = torch.cat(new_rgrids, dim=0)  # (ngrid, ndim)
        dvol_atoms = torch.cat(new_dvols, dim=0)  # (ngrid)
        new_atompos = torch.cat(new_atompos_lst, dim=0)  # (nnewatoms, ndim)
        watoms = _get_atom_weights(new_rgrids,
                                   new_atompos,
                                   ratom_adjust=ratom_adjust)  # (ngrid,)
        self._dvolume = dvol_atoms * watoms

    @property
    def dtype(self):
        """Return the data type of the grid.

        Returns
        -------
        torch.dtype
            Data type of the grid.

        """
        return self._dtype

    @property
    def device(self):
        """device of the grid points

        Returns
        -------
        torch.device
            device of the grid points

        """
        return self._device

    @property
    def coord_type(self):
        """type of the coordinate returned in get_rgrid.

        Returns
        -------
        str
            type of the coordinate returned in get_rgrid. It can be 'cartesian'
            or 'spherical'.

        """
        return "cart"

    def get_dvolume(self):
        """Obtain the torch.tensor containing the dV elements for the integration.

        Returns
        -------
        torch.tensor (*BG, ngrid)
            The dV elements for the integration. *BG is the length of the BaseGrid.

        """
        return self._dvolume

    def get_rgrid(self) -> torch.Tensor:
        """
        Returns the grid points position in the specified coordinate in
        self.coord_type.

        Returns
        -------
        torch.tensor (*BG, ngrid, ndim)
            The grid points position. *BG is the length of the BaseGrid.

        """
        return self._rgrid

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        """
        Return a list with the parameter names corresponding to the given method
        (methodname)

        Returns
        -------
        List[str]
            List of parameter names of methodname

        """
        if methodname == "get_rgrid":
            return [prefix + "_rgrid"]
        elif methodname == "get_dvolume":
            return [prefix + "_dvolume"]
        else:
            raise KeyError("Invalid methodname: %s" % methodname)


def _construct_rgrids(atomgrid: List[LebedevGrid], atompos: torch.Tensor) \
        -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Construct the grid positions in a 2D tensor, the weights per isolated atom

    Examples
    --------
    >>> from deepchem.utils.dft_utils import LebedevGrid, RadialGrid
    >>> from deepchem.utils.dft_utils.grid.multiatoms_grid import _construct_rgrids
    >>> grid = RadialGrid(100, grid_integrator="chebyshev", grid_transform="logm3")
    >>> atomgrid = [LebedevGrid(grid, 3), LebedevGrid(grid, 3)]
    >>> atompos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
    >>> allpos_lst, rgrid, dvol_atoms = _construct_rgrids(atomgrid, atompos)
    >>> len(allpos_lst)
    2

    Parameters
    ----------
    atomgrid: List[LebedevGrid]
        List of Lebedev grids for each atom. (natoms)
    atompos: torch.Tensor
        Position of each atom. (natoms, ndim)

    Returns
    -------
    Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]
        List of grid positions for each atom, the concatenated grid
        positions, and the dV elements.

    """
    allpos_lst = [
        # rgrid: (ngrid[i], ndim), pos: (ndim,)
        (gr.get_rgrid() + pos) \
        for (gr, pos) in zip(atomgrid, atompos)]
    rgrid = torch.cat(allpos_lst, dim=0)  # (ngrid, ndim)

    # calculate the dvol for an isolated atom
    dvol_atoms = torch.cat([gr.get_dvolume() for gr in atomgrid], dim=0)

    return allpos_lst, rgrid, dvol_atoms


def _get_atom_weights(rgrids: List[torch.Tensor],
                      atompos: torch.Tensor,
                      atomradii: Optional[torch.Tensor] = None,
                      ratom_adjust: str = "becke") -> torch.Tensor:
    """Calculate the weights for each grid point due to the atoms

    Examples
    --------
    >>> from deepchem.utils.dft_utils import LebedevGrid, RadialGrid
    >>> from deepchem.utils.dft_utils.grid.multiatoms_grid import _get_atom_weights
    >>> grid = RadialGrid(100, grid_integrator="chebyshev", grid_transform="logm3")
    >>> atomgrid = [LebedevGrid(grid, 3), LebedevGrid(grid, 3)]
    >>> atompos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
    >>> rgrids, _, _ = _construct_rgrids(atomgrid, atompos)
    >>> w = _get_atom_weights(rgrids, atompos)
    >>> w.shape
    torch.Size([1200])

    Parameters
    ----------
    rgrids: List[torch.Tensor]
        List of grid positions for each atom, with length natoms
        consisting of absolute position of the grids (natoms)
    atompos: torch.Tensor
        Position of each atom. (natoms, ndim)
    atomradii: Optional[torch.Tensor] (default None)
        Radii of each atom. (natoms,) or None
    ratom_adjust: str (default "becke")
        Adjustment method for the atom radii. Available: ['becke', 'treutler']

    Returns
    -------
    torch.Tensor
        The weights for each grid point. (ngrid,)

    """
    assert len(rgrids) == atompos.shape[0]
    dtype = atompos.dtype
    device = atompos.device

    natoms = atompos.shape[0]
    rdatoms = atompos - atompos.unsqueeze(1)  # (natoms, natoms, ndim)
    # add the diagonal to stabilize the gradient calculation
    rdatoms = rdatoms + torch.eye(rdatoms.shape[0],
                                  dtype=rdatoms.dtype,
                                  device=rdatoms.device).unsqueeze(-1)
    ratoms = torch.norm(rdatoms, dim=-1)  # (natoms, natoms)

    # calculate the distortion due to heterogeneity
    # (Appendix in Becke's https://doi.org/10.1063/1.454033)
    if atomradii is not None:
        if ratom_adjust == "becke":
            rad = atomradii
        elif ratom_adjust == "treutler":
            # https://aip.scitation.org/doi/pdf/10.1063/1.469408 eq (13)
            rad = atomradii**0.5
        else:
            msg = "Unknown atom adjustment: %s. Available: ['becke', 'treutler']" % ratom_adjust
            raise ValueError(msg)
        # chiij = rad / rad.unsqueeze(1)  # (natoms, natoms)
        uij = (rad - rad.unsqueeze(1)) / \
              (rad + rad.unsqueeze(1))
        aij = torch.clamp(uij / (uij * uij - 1), min=-0.45,
                          max=0.45)  # (natoms, natoms)
        aij = aij.unsqueeze(-1)  # (natoms, natoms, 1)

    xyz_full = torch.cat(rgrids, dim=0)
    # xyz_full: (ngrid, ndim)
    # cdist is more efficient but produces nan in second grad
    # rgatoms = torch.cdist(atompos, xyz, p=2.0)  # (natoms, ngrid)
    w_list: List[torch.Tensor] = []
    ioff = 0
    for ia in range(natoms):
        # concatenate the grid to save memory
        iend = ioff + rgrids[ia].shape[0]
        xyz = xyz_full[ioff:iend, :]

        rgatoms = torch.norm(xyz - atompos.unsqueeze(1),
                             dim=-1)  # (natoms, ngrid)
        mu_ij = (rgatoms - rgatoms.unsqueeze(1))  # (natoms, natoms, ngrid)
        mu_ij /= ratoms.unsqueeze(-1)  # (natoms, natoms, ngrid)

        if atomradii is not None:
            # mu_ij += aij * (1 - mu_ij * mu_ij)
            mu_ij2 = mu_ij * mu_ij
            mu_ij2 -= 1
            mu_ij2 *= (-aij)
            mu_ij2 += mu_ij
            mu_ij = mu_ij2

        # making mu_ij sparse for efficiency
        # threshold: mu_ij < 0.65 (s > 1e-3), mu_ij < 0.74 (s > 1e-4)
        nnz_idx_bool = torch.all(mu_ij < 0.74,
                                 dim=0).unsqueeze(0)  # (1, natoms, ngrid)
        mu_ij_nnz = mu_ij[:, nnz_idx_bool.squeeze(0)].reshape(
            -1)  # (natoms * nnz_col)
        nnz_idx0 = torch.nonzero(nnz_idx_bool).unsqueeze(0)  # (1, nnz_col, 3)
        nnz_col_idx = nnz_idx0[0, :, 1:].transpose(-2, -1)  # (2, nnz_col)
        nnz_idx_atm = torch.zeros((natoms, 1, 3),
                                  dtype=nnz_idx0.dtype,
                                  device=nnz_idx0.device)  # (natoms, 1, 3)
        nnz_idx_atm[:, 0, 0] = torch.arange(natoms,
                                            dtype=nnz_idx0.dtype,
                                            device=nnz_idx0.device)
        nnz_idx = (nnz_idx0 + nnz_idx_atm).reshape(
            -1, 3)  # (nnz = natoms * nnz_col, 3)
        atom_diag_idx = nnz_idx[:, 0] == nnz_idx[:, 1]  # (nnz,)

        f = mu_ij_nnz
        for _ in range(3):
            # f = 0.5 * f * (3 - f * f)
            f2 = f.clone()
            f2 *= f
            f2 -= 3
            f2 *= f
            f2 *= (-0.5)
            f = f2

        # small epsilon to avoid nan in the gradient
        # s = 0.5 * (1. + 1e-12 - f)  # (nnz,)
        s = f
        s -= (1 + 1e-12)
        s *= (-0.5)

        # s += 0.5 * torch.eye(natoms)
        s[atom_diag_idx] += 0.5
        s = s.reshape(natoms, -1)  # (natoms, nnz_col)
        psparse = s.prod(dim=0)  # (nnz_col,)

        # densify and normalize p
        p = torch.zeros((natoms, mu_ij.shape[-1]), dtype=dtype,
                        device=device)  # (natoms, ngrid)
        p[nnz_col_idx[0], nnz_col_idx[1]] = psparse
        p = p / p.sum(dim=0, keepdim=True)  # (natoms, ngrid)

        # save the grid
        w_list.append(p[ia])
        ioff = iend

    w = torch.cat(w_list, dim=-1)  # (ngrid)
    return w

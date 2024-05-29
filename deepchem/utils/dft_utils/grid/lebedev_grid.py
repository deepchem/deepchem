import os
from typing import List, Sequence, Dict
import torch
import numpy as np
from deepchem.utils.dft_utils import BaseGrid, RadialGrid

__all__ = ["LebedevGrid", "TruncatedLebedevGrid"]

class LebedevLoader(object):
    # load the lebedev points and save the cache to save time
    caches: Dict[int, np.ndarray] = {}

    @classmethod
    def load(cls, prec: int) -> np.ndarray:
        if prec not in cls.caches:
            # load the lebedev grid points
            dset_path = os.path.join(os.path.split(__file__)[0],
                                     "lebedevquad", "lebedev_%03d.txt" % prec)
            assert os.path.exists(dset_path), "The dataset lebedev_%03d.txt does not exist" % prec
            lebedev_dsets = np.loadtxt(dset_path)
            lebedev_dsets[:, :2] *= (np.pi / 180)  # convert the angles to radians
            # save to the cache
            cls.caches[prec] = lebedev_dsets

        return cls.caches[prec]

class LebedevGrid(BaseGrid):
    """
    Using Lebedev predefined angular points + radial grid to form 3D grid.
    """

    def __init__(self, radgrid: RadialGrid, prec: int) -> None:
        self._dtype = radgrid.dtype
        self._device = radgrid.device

        assert (prec % 2 == 1) and (3 <= prec <= 131),\
            "Precision must be an odd number between 3 and 131"

        # load the Lebedev grid points
        lebedev_dsets = torch.tensor(LebedevLoader.load(prec), dtype=self._dtype, device=self._device)
        wphitheta = lebedev_dsets[:, -1]  # (nphitheta)
        phi = lebedev_dsets[:, 0]
        theta = lebedev_dsets[:, 1]

        # get the radial grid
        assert radgrid.coord_type == "radial"
        r = radgrid.get_rgrid().unsqueeze(-1)  # (nr, 1)

        # get the cartesian coordinate
        rsintheta = r * torch.sin(theta)
        x = (rsintheta * torch.cos(phi)).view(-1, 1)  # (nr * nphitheta, 1)
        y = (rsintheta * torch.sin(phi)).view(-1, 1)
        z = (r * torch.cos(theta)).view(-1, 1)
        xyz = torch.cat((x, y, z), dim=-1)  # (nr * nphitheta, ndim)
        self._xyz = xyz

        # calculate the dvolume (integration weights)
        dvol_rad = radgrid.get_dvolume().unsqueeze(-1)  # (nr, 1)
        self._dvolume = (dvol_rad * wphitheta).view(-1)  # (nr * nphitheta)

    def get_rgrid(self) -> torch.Tensor:
        return self._xyz

    def get_dvolume(self) -> torch.Tensor:
        return self._dvolume

    @property
    def coord_type(self) -> str:
        return "cart"

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        if methodname == "get_rgrid":
            return [prefix + "_xyz"]
        elif methodname == "get_dvolume":
            return [prefix + "_dvolume"]
        else:
            raise KeyError("Invalid methodname: %s" % methodname)

class TruncatedLebedevGrid(LebedevGrid):
    # A class to represent the truncated lebedev grid
    # It is represented by various radial grid (usually the sliced ones)
    # with different precisions
    def __init__(self, radgrids: Sequence[RadialGrid], precs: Sequence[int]):
        assert len(radgrids) == len(precs)
        assert len(precs) > 0
        self.lebedevs = [LebedevGrid(radgrid, prec) for (radgrid, prec) in zip(radgrids, precs)]
        grid0 = self.lebedevs[0]

        # set the variables to be used in the properties
        self._dtype = grid0.dtype
        self._device = grid0.device
        self._xyz = torch.cat(tuple(grid.get_rgrid() for grid in self.lebedevs), dim=0)
        self._dvolume = torch.cat(tuple(grid.get_dvolume() for grid in self.lebedevs), dim=0)

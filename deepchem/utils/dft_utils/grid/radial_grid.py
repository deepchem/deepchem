from __future__ import annotations
from abc import abstractmethod
import torch
import numpy as np
from typing import Union, Tuple
from deepchem.utils.dft_utils import BaseGrid


class RadialGrid(BaseGrid):
    """
    Grid for radially symmetric system. This grid consists of two specifiers:
    * grid_integrator, and
    * grid_transform

    grid_integrator is to specify how to perform an integration on a fixed
    interval from -1 to 1.

    grid_transform is to transform the integration from the coordinate of
    grid_integrator to the actual coordinate.
    """

    def __init__(self,
                 ngrid: int,
                 grid_integrator: str = "chebyshev",
                 grid_transform: Union[str, BaseGridTransform] = "logm3",
                 dtype: torch.dtype = torch.float64,
                 device: torch.device = torch.device('cpu')):
        self._dtype = dtype
        self._device = device
        grid_transform_obj = get_grid_transform(grid_transform)

        # get the location and weights of the integration in its original
        # coordinate
        _x, _w = get_xw_integration(ngrid, grid_integrator)
        x = torch.as_tensor(_x, dtype=dtype, device=device)
        w = torch.as_tensor(_w, dtype=dtype, device=device)
        r = grid_transform_obj.x2r(x)  # (ngrid,)

        # get the coordinate in Cartesian
        r1 = r.unsqueeze(-1)  # (ngrid, 1)
        self.rgrid = r1
        # r1_zeros = torch.zeros_like(r1)
        # self.rgrid = torch.cat((r1, r1_zeros, r1_zeros), dim = -1)

        # integration element
        drdx = grid_transform_obj.get_drdx(x)
        vol_elmt = 4 * np.pi * r * r  # (ngrid,)
        dr = drdx * w
        self.dvolume = vol_elmt * dr  # (ngrid,)

    @property
    def coord_type(self):
        return "radial"

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    def get_dvolume(self) -> torch.Tensor:
        return self.dvolume

    def get_rgrid(self) -> torch.Tensor:
        return self.rgrid

    def __getitem__(self, key: Union[int, slice]) -> RadialGrid:
        if isinstance(key, slice):
            return SlicedRadialGrid(self, key)
        else:
            raise KeyError("Indexing for RadialGrid is not defined")

    def getparamnames(self, methodname: str, prefix: str = ""):
        if methodname == "get_dvolume":
            return [prefix + "dvolume"]
        elif methodname == "get_rgrid":
            return [prefix + "rgrid"]
        else:
            raise KeyError("getparamnames for %s is not set" % methodname)


def get_xw_integration(n: int, s0: str) -> Tuple[np.ndarray, np.ndarray]:
    # returns ``n`` points of integration from -1 to 1 and its integration
    # weights

    s = s0.lower()
    if s == "chebyshev":
        # generate the x and w from chebyshev polynomial
        # https://doi.org/10.1063/1.475719 eq (9) & (10)
        np1 = n + 1.
        icount = np.arange(n, 0, -1)
        ipn1 = icount * np.pi / np1
        sin_ipn1 = np.sin(ipn1)
        sin_ipn1_2 = sin_ipn1 * sin_ipn1
        xcheb = (np1 - 2 * icount) / np1 + 2 / np.pi * \
                (1 + 2. / 3 * sin_ipn1 * sin_ipn1) * np.cos(ipn1) * sin_ipn1
        wcheb = 16. / (3 * np1) * sin_ipn1_2 * sin_ipn1_2
        return xcheb, wcheb

    elif s == "chebyshev2":
        # generate the x and w from chebyshev polynomial of the 2nd order
        # from Handbook of Mathematical Functions (Abramowitz & Stegun) p. 889
        # note that wcheb should not have sin^2, but only sin
        np1 = n + 1.0
        icount = np.arange(n, 0, -1)
        ipn1 = icount * np.pi / np1
        sin_ipn1 = np.sin(ipn1)
        xcheb = np.cos(ipn1)
        wcheb = np.pi / np1 * sin_ipn1
        return xcheb, wcheb

    elif s == "uniform":
        x = np.linspace(-1, 1, n)
        w = np.ones(n) * (x[1] - x[0])
        w[0] *= 0.5
        w[-1] *= 0.5
        return x, w
    else:
        avail = ["chebyshev", "chebyshev2", "uniform"]
        raise RuntimeError("Unknown grid_integrator: %s. Available: %s" %
                           (s0, avail))


class SlicedRadialGrid(RadialGrid):
    # Internal class to represent the sliced radial grid
    def __init__(self, obj: RadialGrid, key: slice):
        self._dtype = obj.dtype
        self._device = obj.device
        self.dvolume = obj.dvolume[key]
        self.rgrid = obj.rgrid[key]


# Grid Transformations


class BaseGridTransform(object):

    @abstractmethod
    def x2r(self, x: torch.Tensor) -> torch.Tensor:
        # transform from x (coordinate from -1 to 1) to r coordinate (0 to inf)
        pass

    @abstractmethod
    def get_drdx(self, x: torch.Tensor) -> torch.Tensor:
        # returns the dr/dx
        pass


class DE2Transformation(BaseGridTransform):
    # eq (31) in https://link.springer.com/article/10.1007/s00214-011-0985-x
    def __init__(self,
                 alpha: float = 1.0,
                 rmin: float = 1e-7,
                 rmax: float = 20):
        assert rmin < 1.0
        self.alpha = alpha
        self.xmin = -np.log(-np.log(rmin))  # approximate for small r
        self.xmax = np.log(rmax) / alpha  # approximate for large r

    def x2r(self, x: torch.Tensor) -> torch.Tensor:
        # x is from [-1, 1]
        # xnew is from [xmin, xmax]
        xnew = 0.5 * (x * (self.xmax - self.xmin) + (self.xmax + self.xmin))
        # r is approximately from [rmin, rmax]
        r = torch.exp(self.alpha * xnew - torch.exp(-xnew))
        return r

    def get_drdx(self, x: torch.Tensor) -> torch.Tensor:
        r = self.x2r(x)
        xnew = 0.5 * (x * (self.xmax - self.xmin) + (self.xmax + self.xmin))
        return r * (self.alpha + torch.exp(-xnew)) * (0.5 *
                                                      (self.xmax - self.xmin))


class LogM3Transformation(BaseGridTransform):
    # eq (12) in https://aip.scitation.org/doi/pdf/10.1063/1.475719
    def __init__(self, ra: float = 1.0, eps: float = 1e-15):
        self.ra = ra
        self.eps = eps
        self.ln2 = np.log(2.0 + eps)

    def x2r(self, x: torch.Tensor) -> torch.Tensor:
        return self.ra * (1 - torch.log1p(-x + self.eps) / self.ln2)

    def get_drdx(self, x: torch.Tensor) -> torch.Tensor:
        return self.ra / self.ln2 / (1 - x + self.eps)


class TreutlerM4Transformation(BaseGridTransform):
    # eq (19) in https://doi.org/10.1063/1.469408
    def __init__(self, xi: float = 1.0, alpha: float = 0.6, eps: float = 1e-15):
        self._xi = xi
        self._alpha = alpha
        self._ln2 = np.log(2.0 + eps)
        self._eps = eps

    def x2r(self, x: torch.Tensor) -> torch.Tensor:
        a = 1.0 + self._eps
        r = self._xi / self._ln2 * (a + x) ** self._alpha * \
            (self._ln2 - torch.log1p(-x + self._eps))
        return r

    def get_drdx(self, x: torch.Tensor) -> torch.Tensor:
        a = 1.0 + self._eps
        fac = self._xi / self._ln2 * (a + x)**self._alpha
        r1 = fac / (1 - x + self._eps)
        r2 = fac * self._alpha / (a + x) * (self._ln2 -
                                            torch.log1p(-x + self._eps))
        return r2 + r1


def get_grid_transform(s0: Union[str, BaseGridTransform]) -> BaseGridTransform:
    # return the grid transformation object from the input
    if isinstance(s0, BaseGridTransform):
        return s0
    else:
        s = s0.lower()
        if s == "logm3":
            return LogM3Transformation()
        elif s == "de2":
            return DE2Transformation()
        elif s == "treutlerm4":
            return TreutlerM4Transformation()
        else:
            raise RuntimeError("Unknown grid transformation: %s" % s0)

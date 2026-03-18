from __future__ import annotations
from abc import abstractmethod
import torch
import numpy as np
from typing import Union, Tuple
from deepchem.utils.dft_utils import BaseGrid


class RadialGrid(BaseGrid):
    """
    Grid for radially symmetric system. This grid consists grid_integrator
    and grid_transform specifiers.

    grid_integrator is to specify how to perform an integration on a fixed
    interval from -1 to 1.

    grid_transform is to transform the integration from the coordinate of
    grid_integrator to the actual coordinate.

    Examples
    --------
    >>> grid = RadialGrid(100, grid_integrator="chebyshev",
    ...                   grid_transform="logm3")
    >>> grid.get_rgrid().shape
    torch.Size([100, 1])
    >>> grid.get_dvolume().shape
    torch.Size([100])

    """

    def __init__(self,
                 ngrid: int,
                 grid_integrator: str = "chebyshev",
                 grid_transform: Union[str, BaseGridTransform] = "logm3",
                 dtype: torch.dtype = torch.float64,
                 device: torch.device = torch.device('cpu')):
        """Initialize the RadialGrid.

        Parameters
        ----------
        ngrid: int
            Number of grid points.
        grid_integrator: str (default "chebyshev")
            The grid integrator to use. Available options are "chebyshev",
            "chebyshev2", and "uniform".
        grid_transform: Union[str, BaseGridTransform] (default "logm3")
            The grid transformation to use. Available options are "logm3",
            "de2", and "treutlerm4".
        dtype: torch.dtype, optional (default torch.float64)
            The data type to use for the grid.
        device: torch.device, optional (default torch.device('cpu'))
            The device to use for the grid.

        """
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

        # integration element
        drdx = grid_transform_obj.get_drdx(x)
        vol_elmt = 4 * np.pi * r * r  # (ngrid,)
        dr = drdx * w
        self.dvolume = vol_elmt * dr  # (ngrid,)

    @property
    def coord_type(self):
        """Returns the coordinate type of the grid.

        Returns
        -------
        str
            The coordinate type of the grid. For RadialGrid, this is "radial".

        """
        return "radial"

    @property
    def dtype(self):
        """Returns the data type of the grid.

        Returns
        -------
        torch.dtype
            The data type of the grid.

        """
        return self._dtype

    @property
    def device(self):
        """Returns the device of the grid.

        Returns
        -------
        torch.device
            The device of the grid.

        """
        return self._device

    def get_dvolume(self) -> torch.Tensor:
        """Returns the integration element of the grid.

        Returns
        -------
        torch.Tensor
            The integration element of the grid.

        """
        return self.dvolume

    def get_rgrid(self) -> torch.Tensor:
        """Returns the grid points.

        Returns
        -------
        torch.Tensor
            The grid points.

        """
        return self.rgrid

    def __getitem__(self, key: Union[int, slice]) -> RadialGrid:
        """Returns a sliced RadialGrid.

        Parameters
        ----------
        key: Union[int, slice]
            The index or slice to use for slicing the grid.

        Returns
        -------
        RadialGrid
            The sliced RadialGrid.

        """
        if isinstance(key, slice):
            return SlicedRadialGrid(self, key)
        else:
            raise KeyError("Indexing for RadialGrid is not defined")

    def getparamnames(self, methodname: str, prefix: str = ""):
        """Returns the parameter names for the given method.

        Parameters
        ----------
        methodname: str
            The name of the method.
        prefix: str, optional (default "")
            The prefix to use for the parameter names.

        Returns
        -------
        List[str]
            The parameter names for the given method.

        """
        if methodname == "get_dvolume":
            return [prefix + "dvolume"]
        elif methodname == "get_rgrid":
            return [prefix + "rgrid"]
        else:
            raise KeyError("getparamnames for %s is not set" % methodname)


def get_xw_integration(n: int, s0: str) -> Tuple[np.ndarray, np.ndarray]:
    """returns ``n`` points of integration from -1 to 1 and its integration
    weights

    Examples
    --------
    >>> x, w = get_xw_integration(100, "chebyshev")
    >>> x.shape
    (100,)
    >>> w.shape
    (100,)

    Parameters
    ----------
    n: int
        Number of grid points.
    s0: str
        The grid integrator to use. Available options are `chebyshev`,
        `chebyshev2`, and `uniform`.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The integration points and weights.

    References
    ----------
    .. [1] chebyshev polynomial eq (9) & (10) https://doi.org/10.1063/1.475719
    .. [2] Handbook of Mathematical Functions (Abramowitz & Stegun) p. 889

    """

    s = s0.lower()
    if s == "chebyshev":
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
    """Internal class to represent the sliced radial grid"""

    def __init__(self, obj: RadialGrid, key: slice):
        """Initialize the SlicedRadialGrid.

        Parameters
        ----------
        obj: RadialGrid
            The original RadialGrid.
        key: slice
            The slice to use for slicing the grid.

        """
        self._dtype = obj.dtype
        self._device = obj.device
        self.dvolume = obj.dvolume[key]
        self.rgrid = obj.rgrid[key]


# Grid Transformations


class BaseGridTransform(object):
    """Base class for grid transformation
    Grid transformation is to transform the integration from the coordinate of
    grid_integrator to the actual coordinate.

    It is used as a base class for other grid transformations.
    x2r and get_drdx are abstract methods that need to be implemented.

    """

    @abstractmethod
    def x2r(self, x: torch.Tensor) -> torch.Tensor:
        """Transform from x to r coordinate

        Parameters
        ----------
        x: torch.Tensor
            The coordinate from -1 to 1.

        Returns
        -------
        r: torch.Tensor
            The coordinate from 0 to inf.

        """
        pass

    @abstractmethod
    def get_drdx(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the dr/dx

        Parameters
        ----------
        x: torch.Tensor
            The coordinate from -1 to 1.

        Returns
        -------
        drdx: torch.Tensor
            The dr/dx.

        """
        pass


class DE2Transformation(BaseGridTransform):
    """Double exponential formula grid transformation

    Examples
    --------
    >>> x = torch.linspace(-1, 1, 100)
    >>> r = DE2Transformation().x2r(x)
    >>> r.shape
    torch.Size([100])
    >>> drdx = DE2Transformation().get_drdx(x)
    >>> drdx.shape
    torch.Size([100])

    References
    ----------
    .. [1] eq (31) in https://link.springer.com/article/10.1007/s00214-011-0985-x

    """

    def __init__(self,
                 alpha: float = 1.0,
                 rmin: float = 1e-7,
                 rmax: float = 20):
        assert rmin < 1.0
        self.alpha = alpha
        self.xmin = -np.log(-np.log(rmin))  # approximate for small r
        self.xmax = np.log(rmax) / alpha  # approximate for large r

    def x2r(self, x: torch.Tensor) -> torch.Tensor:
        """Transform from x to r coordinate

        Parameters
        ----------
        x: torch.Tensor
            The coordinate from -1 to 1.

        Returns
        -------
        r: torch.Tensor
            The coordinate from 0 to inf.

        """
        # xnew is from [xmin, xmax]
        xnew = 0.5 * (x * (self.xmax - self.xmin) + (self.xmax + self.xmin))
        # r is approximately from [rmin, rmax]
        r = torch.exp(self.alpha * xnew - torch.exp(-xnew))
        return r

    def get_drdx(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the dr/dx

        Parameters
        ----------
        x: torch.Tensor
            The coordinate from -1 to 1.

        Returns
        -------
        drdx: torch.Tensor
            The dr/dx.

        """
        r = self.x2r(x)
        xnew = 0.5 * (x * (self.xmax - self.xmin) + (self.xmax + self.xmin))
        return r * (self.alpha + torch.exp(-xnew)) * (0.5 *
                                                      (self.xmax - self.xmin))


class LogM3Transformation(BaseGridTransform):
    """LogM3 grid transformation

    Examples
    --------
    >>> x = torch.linspace(-1, 1, 100)
    >>> r = LogM3Transformation().x2r(x)
    >>> r.shape
    torch.Size([100])
    >>> drdx = LogM3Transformation().get_drdx(x)
    >>> drdx.shape
    torch.Size([100])

    References
    ----------
    .. [1] eq (12) in https://aip.scitation.org/doi/pdf/10.1063/1.475719

    """

    def __init__(self, ra: float = 1.0, eps: float = 1e-15):
        """Initialize the LogM3Transformation.

        Parameters
        ----------
        ra: float (default 1.0)
            The parameter to control the range of the grid.
        eps: float (default 1e-15)
            The parameter to avoid numerical instability.

        """
        self.ra = ra
        self.eps = eps
        self.ln2 = np.log(2.0 + eps)

    def x2r(self, x: torch.Tensor) -> torch.Tensor:
        """Transform from x to r coordinate

        Parameters
        ----------
        x: torch.Tensor
            The coordinate from -1 to 1.

        Returns
        -------
        torch.Tensor
            The coordinate from 0 to inf.

        """
        return self.ra * (1 - torch.log1p(-x + self.eps) / self.ln2)

    def get_drdx(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the dr/dx

        Parameters
        ----------
        x: torch.Tensor
            The coordinate from -1 to 1.

        Returns
        -------
        torch.Tensor
            The dr/dx.

        """
        return self.ra / self.ln2 / (1 - x + self.eps)


class TreutlerM4Transformation(BaseGridTransform):
    """Treutler M4 grid transformation

    Examples
    --------
    >>> x = torch.linspace(-1, 1, 100)
    >>> r = TreutlerM4Transformation().x2r(x)
    >>> r.shape
    torch.Size([100])
    >>> drdx = TreutlerM4Transformation().get_drdx(x)
    >>> drdx.shape
    torch.Size([100])

    References
    ----------
    .. [1] eq (19) in https://doi.org/10.1063/1.469408

    """

    def __init__(self, xi: float = 1.0, alpha: float = 0.6, eps: float = 1e-15):
        """Initialize the TreutlerM4Transformation.

        Parameters
        ----------
        xi: float (default 1.0)
            The parameter to control the range of the grid.
        alpha: float (default 0.6)
            The parameter to control the range of the grid.
        eps: float (default 1e-15)
            The parameter to avoid numerical instability.

        """
        self._xi = xi
        self._alpha = alpha
        self._ln2 = np.log(2.0 + eps)
        self._eps = eps

    def x2r(self, x: torch.Tensor) -> torch.Tensor:
        """Transform from x to r coordinate

        Parameters
        ----------
        x: torch.Tensor
            The coordinate from -1 to 1.

        Returns
        -------
        torch.Tensor
            The coordinate from 0 to inf.

        """
        a = 1.0 + self._eps
        r = self._xi / self._ln2 * (a + x) ** self._alpha * \
            (self._ln2 - torch.log1p(-x + self._eps))
        return r

    def get_drdx(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the dr/dx

        Parameters
        ----------
        x: torch.Tensor
            The coordinate from -1 to 1.

        Returns
        -------
        torch.Tensor
            The dr/dx.

        """
        a = 1.0 + self._eps
        fac = self._xi / self._ln2 * (a + x)**self._alpha
        r1 = fac / (1 - x + self._eps)
        r2 = fac * self._alpha / (a + x) * (self._ln2 -
                                            torch.log1p(-x + self._eps))
        return r2 + r1


def get_grid_transform(s0: Union[str, BaseGridTransform]) -> BaseGridTransform:
    """grid transformation object from the input

    Examples
    --------
    >>> transform = get_grid_transform("logm3")
    >>> transform.x2r(torch.tensor([0.5]))
    tensor([2.])

    Parameters
    ----------
    s0: Union[str, BaseGridTransform]
        The grid transformation to use. Available options are `logm3`,
        `de2`, and `treutlerm4`.

    Returns
    -------
    BaseGridTransform
        The grid transformation object.

    Raises
    ------
    RuntimeError
        If the input is not a valid grid transformation.

    """
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

from deepchem.utils.differentiation_utils import EditableModule
import torch
from abc import abstractmethod, abstractproperty
from typing import List


class BaseGrid(EditableModule):
    """
    BaseGrid is a class that regulates the integration points over the spatial
    dimensions.

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.dft_utils import BaseGrid
    >>> class Grid(BaseGrid):
    ...     def __init__(self):
    ...         super(Grid, self).__init__()
    ...         self.ngrid = 10
    ...         self.ndim = 3
    ...         self.dvolume = torch.ones(self.ngrid, dtype=self.dtype, device=self.device)
    ...         self.rgrid = torch.ones((self.ngrid, self.ndim), dtype=self.dtype, device=self.device)
    ...     def get_dvolume(self):
    ...         return self.dvolume
    ...     def get_rgrid(self):
    ...         return self.rgrid
    >>> grid = Grid()
    >>> grid.get_dvolume()
    tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    >>> grid.get_rgrid()
    tensor([[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]])

    References
    ----------
    Kasim, Muhammad F., and Sam M. Vinko. "Learning the exchange-correlation
    functional from nature with fully differentiable density functional theory."
    Physical Review Letters 127.12 (2021): 126403.
    https://github.com/diffqc/dqc/blob/0fe821fc92cb3457fb14f6dff0c223641c514ddb/dqc/grid/base_grid.py

    """

    @abstractproperty
    def dtype(self) -> torch.dtype:
        """dtype of the grid points.

        Returns
        -------
        torch.dtype
            dtype of the grid points

        """
        pass

    @abstractproperty
    def device(self) -> torch.device:
        """device of the grid points

        Returns
        -------
        torch.device
            device of the grid points

        """
        pass

    @abstractproperty
    def coord_type(self) -> str:
        """type of the coordinate returned in get_rgrid. It can be 'cartesian'
        or 'spherical'.

        Returns
        -------
        str
            type of the coordinate returned in get_rgrid. It can be 'cartesian'
        or 'spherical'.

        """
        pass

    @abstractmethod
    def get_dvolume(self) -> torch.Tensor:
        """Obtain the torch.tensor containing the dV elements for the integration.

        Returns
        -------
        torch.tensor (*BG, ngrid)
            The dV elements for the integration. *BG is the length of the BaseGrid.

        """
        pass

    @abstractmethod
    def get_rgrid(self) -> torch.Tensor:
        """
        Returns the grid points position in the specified coordinate in
        self.coord_type.

        Returns
        -------
        torch.tensor (*BG, ngrid, ndim)
            The grid points position. *BG is the length of the BaseGrid.

        """
        pass

    @abstractmethod
    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        """
        Return a list with the parameter names corresponding to the given method
        (methodname)

        Returns
        -------
        List[str]
            List of parameter names of methodname

        """
        pass

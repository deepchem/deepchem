from typing import Any, Callable, Union
import torch
from abc import abstractmethod
from deepchem.utils.differentiation_utils.grad import jac

# taking most of the part from SciPy


class Jacobian(object):
    """Base class for the Jacobians used in rootfinder algorithms.

    A Jacobian can best be defined as a determinant which is defined
    for a finite number of functions of the same number of variables
    in which each row consists of the first partial derivatives of
    the same function with respect to each of the variables.

    References
    ----------
    [1].. https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant
    [2].. Kasim, Muhammad & Vinko, Sam. (2020). xi$-torch: differentiable scientific computing library.

    """

    @abstractmethod
    def setup(self, x0: torch.Tensor, y0: torch.Tensor, func: Callable):
        """Setup the Jacobian for the rootfinder."""
        pass

    @abstractmethod
    def solve(self, v: torch.Tensor, tol: Any = 0):
        """Solve the linear system `J dx = v`."""
        pass

    @abstractmethod
    def update(self, x: torch.Tensor, y: torch.Tensor):
        """Update the Jacobian approximation."""
        pass


class BroydenFirst(Jacobian):
    """
    Approximating the Jacobian based on Broyden's first approximation.

    Examples
    --------
    >>> from deepchem.utils.differentiation_utils.optimize.jacobian import BroydenFirst
    >>> jacobian = BroydenFirst()
    >>> x0 = torch.tensor([1.0, 1.0], requires_grad=True)
    >>> def func(x):
    ...     return torch.tensor([x[0]**2 + x[1]**2 - 1.0, x[0] - x[1]])
    >>> y0 = func(x0)
    >>> v = torch.tensor([1.0, 1.0])
    >>> jacobian.setup(x0, y0, func)
    >>> jacobian.solve(v)
    tensor([-0.7071, -0.7071], grad_fn=<MulBackward0>)

    References
    ----------
    [1].. B.A. van der Rotten, PhD thesis,
        "A limited memory Broyden method to solve high-dimensional
        systems of nonlinear equations". Mathematisch Instituut,
        Universiteit Leiden, The Netherlands (2003).

    """

    def __init__(self,
                 alpha: Union[torch.Tensor, None] = None,
                 uv0: Any = None,
                 max_rank: Union[float, None] = None):
        """The initial guess of inverse Jacobian is `-alpha * I + u v^T`.
        `max_rank` indicates the maximum rank of the Jacoabian before
        reducing it

        Parameters
        ----------
        alpha : Union[torch.Tensor, None]
            The initial guess of inverse Jacobian is `-alpha * I`.
            If None, it is set to `-1.0`.
        uv0 : tuple, optional
            The initial guess of the inverse Jacobian.
        max_rank : Union[float, None]
            The maximum rank of the Jacobian before reducing it.
            If None, it is set to `inf`.

        """
        self.alpha = alpha
        self.uv0 = uv0
        self.max_rank = max_rank

    def setup(self, x0: torch.Tensor, y0: torch.Tensor, func: Callable):
        """Setup the Jacobian for the rootfinder.

        Parameters
        ----------
        x0
            The initial guess of the root.
        y0
            The function value at the initial guess.
        func
            The function to find the root.

        """
        self.x_prev = x0
        self.y_prev = y0

        if self.max_rank is None:
            self.max_rank = float('inf')

        if self.alpha is None:
            normy0 = torch.norm(y0)
            ones = torch.ones_like(normy0)
            if normy0:
                self.alpha = 0.5 * torch.max(torch.norm(x0), ones) / normy0
            else:
                self.alpha = ones

        if self.uv0 == "svd":
            self.uv0 = _get_svd_uv0(func, x0)

        # setup the approximate inverse Jacobian
        self.Gm = LowRankMatrix(-self.alpha, self.uv0, "restart")

    def _reduce(self):
        """
        reduce the size of Gm
        initially it was a lambda function, but it causes a leak, so
        I arranged into a method to remove the leak
        """
        self.Gm.reduce(self.max_rank)

    def solve(self, v: torch.Tensor, tol=0) -> torch.Tensor:
        """Solve the linear system `J dx = v`.

        Parameters
        ----------
        v: torch.Tensor
            The right-hand side of the linear system.
        tol: torch.Tensor
            The tolerance for the linear system.

        Returns
        -------
        res: torch.Tensor
            The solution of the linear system.

        """
        res = self.Gm.mv(v)
        return res

    def update(self, x: torch.Tensor, y: torch.Tensor):
        """Update the Jacobian approximation.

        Parameters
        ----------
        x: torch.Tensor
            The current point.
        y: torch.Tensor
            The function value at the current point.

        """
        dy = y - self.y_prev
        dx = x - self.x_prev
        # update Gm
        self._update(x, y, dx, dy, dx.norm(), dy.norm())

        self.y_prev = y
        self.x_prev = x

    def _update(self, x: torch.Tensor, y: torch.Tensor, dx: torch.Tensor,
                dy: torch.Tensor, dxnorm: torch.Tensor, dynorm: torch.Tensor):
        """Update the Jacobian approximation.

        Parameters
        ----------
        x: torch.Tensor
            The current point.
        y: torch.Tensor
            The function value at the current point.
        dx: torch.Tensor
            The difference between the current point and the previous point.
        dy: torch.Tensor
            The difference between the function value at the current point
            and the previous point.
        dxnorm: torch.Tensor
            The norm of `dx`.
        dynorm: torch.Tensor
            The norm of `dy`.

        """
        # keep the rank small
        self._reduce()

        v = self.Gm.rmv(dx)
        c = dx - self.Gm.mv(dy)
        d = v / torch.dot(dy, v)
        self.Gm = self.Gm.append(c, d)


class BroydenSecond(BroydenFirst):
    """
    Inverse Jacobian approximation based on Broyden's second method.

    Examples
    --------
    >>> from deepchem.utils.differentiation_utils.optimize.jacobian import BroydenSecond
    >>> jacobian = BroydenSecond()
    >>> x0 = torch.tensor([1.0, 1.0], requires_grad=True)
    >>> def func(x):
    ...     return torch.tensor([x[0]**2 + x[1]**2 - 1.0, x[0] - x[1]])
    >>> y0 = func(x0)
    >>> v = torch.tensor([1.0, 1.0])
    >>> jacobian.setup(x0, y0, func)
    >>> jacobian.solve(v)
    tensor([-0.7071, -0.7071], grad_fn=<MulBackward0>)

    References
    ----------
    [1] B.A. van der Rotten, PhD thesis,
        "A limited memory Broyden method to solve high-dimensional
        systems of nonlinear equations". Mathematisch Instituut,
        Universiteit Leiden, The Netherlands (2003).

    """

    def _update(self, x: torch.Tensor, y: torch.Tensor, dx: torch.Tensor,
                dy: torch.Tensor, dxnorm: torch.Tensor, dynorm: torch.Tensor):
        """Update the Jacobian approximation.

        Parameters
        ----------
        x: torch.Tensor
            The current point.
        y: torch.Tensor
            The function value at the current point.
        dx: torch.Tensor
            The difference between the current point and the previous point.
        dy: torch.Tensor
            The difference between the function value at the current point
            and the previous point.
        dxnorm: torch.Tensor
            The norm of `dx`.
        dynorm: torch.Tensor
            The norm of `dy`.

        """
        # keep the rank small
        self._reduce()

        v = dy
        c = dx - self.Gm.mv(dy)
        d = v / (dynorm * dynorm)
        self.Gm = self.Gm.append(c, d)


class LinearMixing(Jacobian):
    """ Approximating the Jacobian based on linear mixing.
    It acts as a simple check for the functionality of the rootfinder.

    Examples
    --------
    >>> from deepchem.utils.differentiation_utils.optimize.jacobian import LinearMixing
    >>> jacobian = LinearMixing()
    >>> x0 = torch.tensor([1.0, 1.0], requires_grad=True)
    >>> def func(x):
    ...     return torch.tensor([x[0]**2 + x[1]**2 - 1.0, x[0] - x[1]])
    >>> y0 = func(x0)
    >>> v = torch.tensor([1.0, 1.0])
    >>> jacobian.setup(x0, y0, func)
    >>> jacobian.solve(v)
    tensor([1., 1.])

    """

    def __init__(self, alpha: Union[float, None] = None):
        """The initial guess of inverse Jacobian is ``-alpha * I``

        Parameters
        ----------
        alpha : float, optional
            The initial guess of inverse Jacobian is ``-alpha * I``.
            If None, it is set to ``-1.0``.

        """
        if alpha is None:
            alpha = -1.0
        self.alpha = alpha

    def setup(self, x0: torch.Tensor, y0: torch.Tensor, func: Callable):
        """Setup the Jacobian for the rootfinder.

        Parameters
        ----------
        x0: torch.Tensor
            The initial guess of the root.
        y0: torch.Tensor
            The function value at the initial guess.
        func: Callable
            The function to find the root.

        """
        pass

    def solve(self, v: torch.Tensor, tol=0) -> torch.Tensor:
        """Solve the linear system `J dx = v`.

        Parameters
        ----------
        v: torch.Tensor
            The right-hand side of the linear system.
        tol
            The tolerance for the linear system.

        """
        return -v * self.alpha

    def update(self, x: torch.Tensor, y: torch.Tensor):
        """Update the Jacobian approximation.

        Parameters
        ----------
        x: torch.Tensor
            The current point.
        y: torch.Tensor
            The function value at the current point.

        """
        pass


class LowRankMatrix(object):
    """represents a matrix of `\alpha * I + \sum_n c_n d_n^T`

    Examples
    --------
    >>> from deepchem.utils.differentiation_utils.optimize.jacobian import LowRankMatrix
    >>> import torch
    >>> alpha = 1.0
    >>> uv0 = (torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0]))
    >>> reduce_method = "restart"
    >>> matrix = LowRankMatrix(alpha, uv0, reduce_method)
    >>> v = torch.tensor([1.0, 1.0])
    >>> matrix.mv(v)
    tensor([3., 3.])
    >>> matrix.rmv(v)
    tensor([3., 3.])

    """

    def __init__(self, alpha: torch.Tensor, uv0, reduce_method: str):
        """initialize the matrix

        Parameters
        ----------
        alpha : torch.Tensor
            The coefficient of the identity matrix
        uv0 : tuple
            The initial guess of the inverse Jacobian
        reduce_method : str
            The method to reduce the rank of the matrix

        """
        self.alpha = alpha
        if uv0 is None:
            self.cns = []
            self.dns = []
        else:
            cn0, dn0 = uv0
            self.cns = [cn0]
            self.dns = [dn0]
        self.reduce_method = {"restart": 0, "simple": 1}[reduce_method]

    def mv(self, v: torch.Tensor) -> torch.Tensor:
        """multiply the matrix with a vector

        Parameters
        ----------
        v: torch.Tensor
            Vector to multiply

        Returns
        -------
        res: torch.Tensor
            Result of the multiplication

        """
        res = self.alpha * v
        for i in range(len(self.dns)):
            res += self.cns[i] * torch.dot(self.dns[i], v)
        return res

    def rmv(self, v: torch.Tensor) -> torch.Tensor:
        """multiply the transpose of the matrix with a vector

        Parameters
        ----------
        v: torch.Tensor
            Vector to multiply

        Returns
        -------
        res: torch.Tensor
            Result of the multiplication

        """
        res = self.alpha * v
        for i in range(len(self.dns)):
            res += self.dns[i] * torch.dot(self.cns[i], v)
        return res

    def append(self, c: torch.Tensor, d: torch.Tensor):
        """append a rank-1 matrix to the matrix

        Parameters
        ----------
        c: torch.Tensor
            The first vector
        d: torch.Tensor
            The second vector

        Returns
        -------
        res: Union['LowRankMatrix', 'FullRankMatrix']
            The matrix after appending the rank-1 matrix

        """
        self.cns.append(c)
        self.dns.append(d)
        if len(self.cns) >= torch.numel(c):
            return FullRankMatrix(self.alpha, self.cns, self.dns)
        return self

    def reduce(self, max_rank: int, **otherparams):
        """reduce the rank of the matrix

        Parameters
        ----------
        max_rank : int
            The maximum rank of the matrix
        otherparams
            Other parameters

        """
        if len(self.cns) > max_rank:
            if self.reduce_method == 0:  # restart
                del self.cns[:]
                del self.dns[:]
            elif self.reduce_method == 1:  # simple
                n = len(self.cns)
                del self.cns[:n - max_rank]
                del self.dns[:n - max_rank]


class FullRankMatrix(object):
    """represents a full rank matrix of `\alpha * I + \sum_n c_n d_n^T`

    Examples
    --------
    >>> from deepchem.utils.differentiation_utils.optimize.jacobian import FullRankMatrix
    >>> import torch
    >>> alpha = 1.0
    >>> cns = [torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0])]
    >>> dns = [torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0])]
    >>> matrix = FullRankMatrix(alpha, cns, dns)
    >>> v = torch.tensor([1.0, 1.0])
    >>> matrix.mv(v)
    tensor([5., 5.])
    >>> matrix.rmv(v)
    tensor([5., 5.])

    """

    def __init__(self, alpha: torch.Tensor, cns: Any, dns: Any):
        """initialize the matrix

        Parameters
        ----------
        alpha : torch.Tensor
            Coefficient of the identity matrix
        cns : List
            List of the first vectors
        dns : List
            List of the second vectors

        """
        size = torch.numel(cns[0])
        dtype, device = cns[0].dtype, cns[0].device
        self.mat = torch.eye(size, dtype=dtype, device=device)
        self.mat *= alpha
        for i in range(len(cns)):
            self.mat += torch.ger(cns[i], dns[i])

    def mv(self, v: torch.Tensor) -> torch.Tensor:
        """multiply the matrix with a vector

        Parameters
        ----------
        v: torch.Tensor
            The vector to multiply

        Returns
        -------
        res: torch.Tensor
            The result of the multiplication

        """
        res = torch.matmul(self.mat, v)
        return res

    def rmv(self, v: torch.Tensor) -> torch.Tensor:
        """multiply the transpose of the matrix with a vector

        Parameters
        ----------
        v: torch.Tensor
            The vector to multiply

        Returns
        -------
        torch.Tensor
            The result of the multiplication

        """
        return torch.matmul(self.mat.T, v)

    def append(self, c: torch.Tensor, d: torch.Tensor):
        """append a rank-1 matrix to the matrix

        Parameters
        ----------
        c: torch.Tensor
            The first vector
        d: torch.Tensor
            The second vector

        Returns
        -------
        FullRankMatrix
            The matrix after appending the rank-1 matrix

        """
        self.mat += torch.ger(c, d)
        return self

    def reduce(self, max_rank: int, **kwargs):
        """reduce the rank of the matrix

        Parameters
        ----------
        max_rank : int
            The maximum rank of the matrix
        otherparams
            Other parameters

        """
        pass


def _get_svd_uv0(func: Callable, x0: torch.Tensor) -> tuple:
    """get the initial guess of the inverse Jacobian from the first Jacobian

    Parameters
    ----------
    func: Callable
        The function to find the root
    x0: torch.Tensor
        The initial guess of the root

    Returns
    -------
    uv0: tuple
        The initial guess of the inverse Jacobian

    """
    from deepchem.utils.differentiation_utils import svd
    # raise RuntimeError
    fjac = jac(func, (x0.clone().requires_grad_(),), idxs=[0])[0]
    # u: (n, 1), s: (1,), vh: (1, n)
    u, s, vh = svd(fjac, k=1, mode="lowest", method="davidson", min_eps=1e-3)
    sinv_sqrt = 1. / torch.sqrt(torch.clamp(s, min=0.1))
    uv0 = (sinv_sqrt * vh.squeeze(-2), sinv_sqrt * u.squeeze(-1))
    return uv0

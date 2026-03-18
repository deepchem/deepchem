from __future__ import annotations
from typing import Sequence, Optional, List, Union
import warnings
import torch
from abc import abstractmethod
from contextlib import contextmanager
from scipy.sparse.linalg import LinearOperator as spLinearOperator
from deepchem.utils.differentiation_utils import EditableModule, get_bcasted_dims
from deepchem.utils import shape2str, indent

__all__ = ["LinearOperator"]


class LinearOperator(EditableModule):
    """
    ``LinearOperator`` is a base class designed to behave as a linear operator
    without explicitly determining the matrix. This ``LinearOperator`` should
    be able to operate as batched linear operators where its shape is
    ``(B1,B2,...,Bb,p,q)`` with ``B*`` as the (optional) batch dimensions.
    For a user-defined class to behave as ``LinearOperator``, it must use
    ``LinearOperator`` as one of the parent and it has to have ``._mv()``
    method implemented and ``._getparamnames()`` if used in xitorch's
    functionals with torch grad enabled.

    Examples
    --------
    >>> import torch
    >>> seed = torch.manual_seed(100)
    >>> class MyLinOp(LinearOperator):
    ...     def __init__(self, shape):
    ...         super(MyLinOp, self).__init__(shape)
    ...         self.param = torch.rand(shape)
    ...     def _getparamnames(self, prefix=""):
    ...         return [prefix + "param"]
    ...     def _mv(self, x):
    ...         return torch.matmul(self.param, x)
    ...     def _rmv(self, x):
    ...         return torch.matmul(self.param.transpose(-2,-1).conj(), x)
    ...     def _mm(self, x):
    ...         return torch.matmul(self.param, x)
    ...     def _rmm(self, x):
    ...         return torch.matmul(self.param.transpose(-2,-1).conj(), x)
    ...     def _fullmatrix(self):
    ...         return self.param
    >>> linop = MyLinOp((1,3,1,2))
    >>> print(linop)
    LinearOperator (MyLinOp) with shape (1, 3, 1, 2), dtype = torch.float32, device = cpu
    >>> x = torch.rand(1,3,2,2)
    >>> linop.mv(x)
    tensor([[[[0.1991, 0.1011]],
    <BLANKLINE>
             [[0.3764, 0.5742]],
    <BLANKLINE>
             [[1.0345, 1.1802]]]])
    >>> x = torch.rand(1,3,1,1)
    >>> linop.rmv(x)
    tensor([[[[0.0250],
              [0.1827]],
    <BLANKLINE>
             [[0.0794],
              [0.1463]],
    <BLANKLINE>
             [[0.1207],
              [0.1345]]]])
    >>> x = torch.rand(1,3,2,2)
    >>> linop.mm(x)
    tensor([[[[0.8891, 0.4243]],
    <BLANKLINE>
             [[0.4856, 0.3128]],
    <BLANKLINE>
             [[0.6601, 0.9532]]]])
    >>> x = torch.rand(1,3,1,2)
    >>> linop.rmm(x)
    tensor([[[[0.0473, 0.0019],
              [0.3455, 0.0138]],
    <BLANKLINE>
             [[0.0580, 0.2504],
              [0.1069, 0.4614]],
    <BLANKLINE>
             [[0.4779, 0.1102],
              [0.5326, 0.1228]]]])
    >>> linop.fullmatrix()
    tensor([[[[0.1117, 0.8158]],
    <BLANKLINE>
             [[0.2626, 0.4839]],
    <BLANKLINE>
             [[0.6765, 0.7539]]]])

    """
    _implementation_checked = False
    _is_mv_implemented = False
    _is_mm_implemented = False
    _is_rmv_implemented = False
    _is_rmm_implemented = False
    _is_fullmatrix_implemented = False
    _is_gpn_implemented = False

    def __new__(self, *args, **kwargs):
        """Check the implemented functions in the class."""
        if not self._implementation_checked:
            self._is_mv_implemented = self._check_if_implemented("_mv")
            self._is_mm_implemented = self._check_if_implemented("_mm")
            self._is_rmv_implemented = self._check_if_implemented("_rmv")
            self._is_rmm_implemented = self._check_if_implemented("_rmm")
            self._is_fullmatrix_implemented = self._check_if_implemented(
                "_fullmatrix")
            self._is_gpn_implemented = self._check_if_implemented(
                "_getparamnames")

            self._implementation_checked = True

            if not self._is_mv_implemented:
                raise RuntimeError(
                    "LinearOperator must have at least _mv(self) "
                    "method implemented")
        return super(LinearOperator, self).__new__(self)

    @classmethod
    def m(cls, mat: torch.Tensor, is_hermitian: Optional[bool] = None):
        """
        Class method to wrap a matrix into ``LinearOperator``.

        Parameters
        ----------
        mat: torch.Tensor
            Matrix to be wrapped in the ``LinearOperator``.
        is_hermitian: bool or None
            Indicating if the matrix is Hermitian. If ``None``, the symmetry
            will be checked. If supplied as a bool, there is no check performed.

        Returns
        -------
        LinearOperator
            Linear operator object that represents the matrix.

        Example
        -------
        >>> import torch
        >>> from deepchem.utils.differentiation_utils import LinearOperator
        >>> seed = torch.manual_seed(100)
        >>> mat = torch.rand(1,3,1,2)  # 1x2 matrix with (1,3) batch dimensions
        >>> linop = LinearOperator.m(mat)
        >>> print(linop)
        MatrixLinearOperator with shape (1, 3, 1, 2):
           tensor([[[[0.1117, 0.8158]],
        <BLANKLINE>
                    [[0.2626, 0.4839]],
        <BLANKLINE>
                    [[0.6765, 0.7539]]]])

        """
        if is_hermitian is None:
            if mat.shape[-2] != mat.shape[-1]:
                is_hermitian = False
            else:
                is_hermitian = torch.allclose(mat, mat.transpose(-2, -1).conj())
        elif is_hermitian:
            # check the hermitian
            if not torch.allclose(mat, mat.transpose(-2, -1).conj()):
                raise RuntimeError(
                    "The linear operator is indicated to be hermitian, but the matrix is not"
                )

        return MatrixLinearOperator(mat, is_hermitian)

    @classmethod
    def _check_if_implemented(self, methodname: str) -> bool:
        """Check if the method is implemented in the class.

        Parameters
        ----------
        methodname : str
            The method name to be checked

        """
        this_method = getattr(self, methodname)
        base_method = getattr(LinearOperator, methodname)
        return this_method is not base_method

    def __init__(self,
                 shape: Sequence[int],
                 is_hermitian: bool = False,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 _suppress_hermit_warning: bool = False) -> None:
        """Initialize the ``LinearOperator``.

        Parameters
        ----------
        shape : Sequence[int]
            The shape of the linear operator.
        is_hermitian : bool
            Whether the linear operator is Hermitian.
        dtype : torch.dtype or None
            The dtype of the linear operator.
        device : torch.device or None
            The device of the linear operator.
        _suppress_hermit_warning : bool
            Whether to suppress the warning when the linear operator is
            Hermitian but the ``.rmv()`` or ``.rmm()`` is implemented.

        """

        super(LinearOperator, self).__init__()
        if len(shape) < 2:
            raise RuntimeError("The shape must have at least 2 dimensions")
        self._shape = shape
        self._batchshape = list(shape[:-2])
        self._is_hermitian = is_hermitian
        self._dtype = dtype if dtype is not None else torch.float32
        self._device = device if device is not None else torch.device("cpu")
        if is_hermitian and shape[-1] != shape[-2]:
            raise RuntimeError(
                "The object is indicated as Hermitian, but the shape is not square"
            )

        # check which methods are implemented
        if not _suppress_hermit_warning and self._is_hermitian and \
           (self._is_rmv_implemented or self._is_rmm_implemented):
            warnings.warn(
                "The LinearOperator is Hermitian with implemented "
                "rmv or rmm. We will use the mv and mm methods "
                "instead",
                stacklevel=2)

    def __repr__(self) -> str:
        """Representation of the ``LinearOperator``.

        Returns
        -------
        shape: Sequence[int]
            The shape of the linear operator.
        dtype: torch.dtype
            The dtype of the linear operator.
        device: torch.device
            The device of the linear operator.

        """
        return "LinearOperator (%s) with shape %s, dtype = %s, device = %s" % \
            (self.__class__.__name__, shape2str(self.shape), self.dtype, self.device)

    @abstractmethod
    def _getparamnames(self, prefix: str = "") -> List[str]:
        """
        List the self's parameters that affecting the ``LinearOperator``.
        This is for the derivative purpose.

        Parameters
        ----------
        prefix: str
            The prefix to be appended in front of the parameters name.
            This usually contains the dots.

        Returns
        -------
        List[str]
            List of parameter names (including the prefix) that affecting
            the ``LinearOperator``.

        """
        return []

    @abstractmethod
    def _mv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to be implemented for matrix-vector multiplication.
        Required for all ``LinearOperator`` objects.

        Parameters
        ----------
        x: torch.tensor
            Vector with shape ``(...,q)`` where the linear operation is operated on.

        Returns
        -------
        torch.tensor
            The result of the linear operation with shape ``(...,p)``

        """
        pass

    # @abstractmethod
    def _rmv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to be implemented for transposed matrix-vector
        multiplication. Optional. If not implemented, it will use the
        adjoint trick to compute ``.rmv()``. Usually implemented for
        efficiency reasons.

        Parameters
        ----------
        x: torch.tensor
            Vector with shape ``(...,q)`` where the linear operation is operated on.

        Returns
        -------
        torch.tensor
            The result of the linear operation with shape ``(...,p)``

        """
        raise NotImplementedError()

    # @abstractmethod # (optional)
    def _mm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to be implemented for matrix-matrix multiplication.
        If not implemented, then it uses batched version of matrix-vector
        multiplication. Usually this is implemented for efficiency reasons.

        Parameters
        ----------
        x: torch.tensor
            Vector with shape ``(...,q)`` where the linear operation is operated on.

        Returns
        -------
        torch.tensor
            The result of the linear operation with shape ``(...,p)``

        """
        raise NotImplementedError()

    # @abstractmethod
    def _rmm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to be implemented for transposed matrix-matrix
        multiplication. If not implemented, then it uses batched version
        of transposed matrix-vector multiplication. Usually this is
        implemented for efficiency reasons.

        Parameters
        ----------
        x: torch.tensor
            Vector with shape ``(...,q)`` where the linear operation is operated on.

        Returns
        -------
        torch.tensor
            The result of the linear operation with shape ``(...,p)``

        """
        raise NotImplementedError()

    # @abstractmethod
    def _fullmatrix(self) -> torch.Tensor:
        """Return the full matrix representation of the linear operator."""
        raise NotImplementedError()

    def getlinopparams(self) -> Sequence[torch.Tensor]:
        """Get the parameters that affects most of the methods (i.e. mm, mv, rmm, rmv)."""
        return self.getuniqueparams("mm")

    @contextmanager
    def uselinopparams(self, *params):
        """Context manager to temporarily set the parameters that affects most of
        the methods (i.e. mm, mv, rmm, rmv)."""
        methodname = "mm"
        try:
            _orig_params_ = self.getuniqueparams(methodname)
            self.setuniqueparams(methodname, *params)
            yield self
        finally:
            self.setuniqueparams(methodname, *_orig_params_)

    # implemented functions
    def mv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the matrix-vector operation to vector ``x`` with shape ``(...,q)``.
        The batch dimensions of ``x`` need not be the same as the batch dimensions
        of the ``LinearOperator``, but it must be broadcastable.

        Parameters
        ----------
        x: torch.tensor
            The vector with shape ``(...,q)`` where the linear operation is operated on

        Returns
        -------
        y: torch.tensor
            The result of the linear operation with shape ``(...,p)``

        """
        self._assert_if_init_executed()
        if x.shape[-1] != self.shape[-1]:
            raise RuntimeError(
                "Cannot operate .mv on shape %s. Expected (...,%d)" %
                (str(tuple(x.shape)), self.shape[-1]))

        return self._mv(x)

    def mm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the matrix-matrix operation to matrix ``x`` with shape ``(...,q,r)``.
        The batch dimensions of ``x`` need not be the same as the batch dimensions
        of the ``LinearOperator``, but it must be broadcastable.

        Parameters
        ----------
        x: torch.tensor
            The matrix with shape ``(...,q,r)`` where the linear operation is
            operated on.

        Returns
        -------
        y: torch.tensor
            The result of the linear operation with shape ``(...,p,r)``

        """
        self._assert_if_init_executed()
        if x.shape[-2] != self.shape[-1]:
            raise RuntimeError(
                "Cannot operate .mm on shape %s. Expected (...,%d,*)" %
                (str(tuple(x.shape)), self.shape[-1]))

        xbatchshape = list(x.shape[:-2])
        if self._is_mm_implemented:
            return self._mm(x)
        else:
            # use batched mv as mm

            # move the last dimension to the very first dimension to be broadcasted
            if len(xbatchshape) < len(self._batchshape):
                xbatchshape = [1] * (len(self._batchshape) -
                                     len(xbatchshape)) + xbatchshape
            x1 = x.reshape(1, *xbatchshape, *x.shape[-2:])
            xnew = x1.transpose(0, -1).squeeze(-1)  # (r,...,q)

            # apply batched mv and restore the initial shape
            ynew = self._mv(xnew)  # (r,...,p)
            y = ynew.unsqueeze(-1).transpose(0, -1).squeeze(0)  # (...,p,r)
            return y

    def rmv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the matrix-vector adjoint operation to vector ``x`` with shape ``(...,p)``,
        i.e. ``A^H x``.
        The batch dimensions of ``x`` need not be the same as the batch dimensions
        of the ``LinearOperator``, but it must be broadcastable.

        Parameters
        ----------
        x: torch.tensor
            The vector of shape ``(...,p)`` where the adjoint linear operation is operated at.

        Returns
        -------
        y: torch.tensor
            The result of the adjoint linear operation with shape ``(...,q)``

        """
        self._assert_if_init_executed()
        if x.shape[-1] != self.shape[-2]:
            raise RuntimeError(
                "Cannot operate .rmv on shape %s. Expected (...,%d)" %
                (str(tuple(x.shape)), self.shape[-2]))

        if self._is_hermitian:
            return self._mv(x)
        elif not self._is_rmv_implemented:
            return self._adjoint_rmv(x)
        return self._rmv(x)

    def rmm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the matrix-matrix adjoint operation to matrix ``x`` with shape ``(...,p,r)``,
        i.e. ``A^H X``.
        The batch dimensions of ``x`` need not be the same as the batch dimensions
        of the ``LinearOperator``, but it must be broadcastable.

        Parameters
        ----------
        x: torch.Tensor
            The matrix of shape ``(...,p,r)`` where the adjoint linear operation is operated on.

        Returns
        -------
        y: torch.Tensor
            The result of the adjoint linear operation with shape ``(...,q,r)``.

        """
        self._assert_if_init_executed()
        if x.shape[-2] != self.shape[-2]:
            raise RuntimeError(
                "Cannot operate .rmm on shape %s. Expected (...,%d,*)" %
                (str(tuple(x.shape)), self.shape[-2]))

        if self._is_hermitian:
            return self.mm(x)

        xbatchshape = list(x.shape[:-2])
        if self._is_rmm_implemented:
            return self._rmm(x)
        else:
            # use batched mv as mm
            rmv = self._rmv if self._is_rmv_implemented else self.rmv

            # move the last dimension to the very first dimension to be broadcasted
            if len(xbatchshape) < len(self._batchshape):
                xbatchshape = [1] * (len(self._batchshape) -
                                     len(xbatchshape)) + xbatchshape
            x1 = x.reshape(1, *xbatchshape, *x.shape[-2:])  # (1,...,p,r)
            xnew = x1.transpose(0, -1).squeeze(-1)  # (r,...,p)

            # apply batched mv and restore the initial shape
            ynew = rmv(xnew)  # (r,...,q)
            y = ynew.unsqueeze(-1).transpose(0, -1).squeeze(0)  # (...,q,r)
            return y

    def fullmatrix(self) -> torch.Tensor:
        """Full matrix representation of the linear operator."""
        if self._is_fullmatrix_implemented:
            return self._fullmatrix()
        else:
            self._assert_if_init_executed()
            nq = self._shape[-1]
            V = torch.eye(nq, dtype=self._dtype, device=self._device)  # (nq,nq)
            return self.mm(V)  # (B1,B2,...,Bb,np,nq)

    def scipy_linalg_op(self):
        """Return the scipy.sparse.linalg.LinearOperator object of the linear operator."""

        def to_tensor(x):
            return torch.tensor(x, dtype=self.dtype, device=self.device)

        return spLinearOperator(
            shape=self.shape,
            matvec=lambda v: self.mv(to_tensor(v)).detach().cpu().numpy(),
            rmatvec=lambda v: self.rmv(to_tensor(v)).detach().cpu().numpy(),
            matmat=lambda v: self.mm(to_tensor(v)).detach().cpu().numpy(),
            rmatmat=lambda v: self.rmm(to_tensor(v)).detach().cpu().numpy(),
        )

    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        """Get the parameter names that affects the method."""

        if methodname in ["mv", "rmv", "mm", "rmm", "fullmatrix"]:
            return self._getparamnames(prefix=prefix)
        else:
            raise KeyError("getparamnames for method %s is not implemented" %
                           methodname)

    @property
    def H(self):
        """
        Returns a LinearOperator representing the Hermite / transposed of the
        self LinearOperator.

        Returns
        -------
        LinearOperator
            The Hermite / transposed LinearOperator

        """
        if self._is_hermitian:
            return self
        return AdjointLinearOperator(self)

    # special functions
    def matmul(self, b: LinearOperator, is_hermitian: bool = False):
        """
        Returns a LinearOperator representing `self @ b`.

        Examples
        --------
        >>> import torch
        >>> seed = torch.manual_seed(100)
        >>> class MyLinOp(LinearOperator):
        ...     def __init__(self, shape):
        ...         super(MyLinOp, self).__init__(shape)
        ...         self.param = torch.rand(shape)
        ...     def _getparamnames(self, prefix=""):
        ...         return [prefix + "param"]
        ...     def _mv(self, x):
        ...         return torch.matmul(self.param, x)
        >>> linop1 = MyLinOp((1,3,1,2))
        >>> linop2 = MyLinOp((1,3,2,1))
        >>> linop = linop1.matmul(linop2)
        >>> print(linop)
        MatmulLinearOperator with shape (1, 3, 1, 1) of:
         * LinearOperator (MyLinOp) with shape (1, 3, 1, 2), dtype = torch.float32, device = cpu
         * LinearOperator (MyLinOp) with shape (1, 3, 2, 1), dtype = torch.float32, device = cpu
        >>> x = torch.rand(1,3,1,1)
        >>> linop.mv(x)
        tensor([[[[0.0458]],
        <BLANKLINE>
                 [[0.0880]],
        <BLANKLINE>
                 [[0.2664]]]])

        Parameters
        ----------
        b: LinearOperator
            Other linear operator
        is_hermitian: bool
            Flag to indicate if the resulting LinearOperator is Hermitian.

        Returns
        -------
        LinearOperator
            LinearOperator representing `self @ b`

        """
        if self.shape[-1] != b.shape[-2]:
            raise RuntimeError("Mismatch shape of matmul operation: %s and %s" %
                               (self.shape, b.shape))
        if isinstance(self, MatrixLinearOperator) and isinstance(
                b, MatrixLinearOperator):
            return LinearOperator.m(self.fullmatrix() @ b.fullmatrix(),
                                    is_hermitian=is_hermitian)
        return MatmulLinearOperator(self, b, is_hermitian=is_hermitian)

    def __add__(self, b: LinearOperator):
        """Addition with another linear operator.

        Examples
        --------
        >>> class Operator(LinearOperator):
        ...     def __init__(self, mat: torch.Tensor, is_hermitian: bool) -> None:
        ...         super(Operator, self).__init__(
        ...             shape=mat.shape,
        ...             is_hermitian=is_hermitian,
        ...             dtype=mat.dtype,
        ...             device=mat.device,
        ...             _suppress_hermit_warning=True,
        ...         )
        ...         self.mat = mat
        ...     def _mv(self, x: torch.Tensor) -> torch.Tensor:
        ...         return torch.matmul(self.mat, x.unsqueeze(-1)).squeeze(-1)
        ...     def _mm(self, x: torch.Tensor) -> torch.Tensor:
        ...         return torch.matmul(self.mat, x)
        ...     def _rmv(self, x: torch.Tensor) -> torch.Tensor:
        ...         return torch.matmul(self.mat.transpose(-3, -1).conj(), x.unsqueeze(-1)).squeeze(-1)
        ...     def _rmm(self, x: torch.Tensor) -> torch.Tensor:
        ...         return torch.matmul(self.mat.transpose(-2, -1).conj(), x)
        ...     def _fullmatrix(self) -> torch.Tensor:
        ...         return self.mat
        ...     def _getparamnames(self, prefix: str = "") -> List[str]:
        ...         return [prefix + "mat"]
        >>> op = Operator(torch.tensor([[1, 2.],
        ...                             [3, 4]]), is_hermitian=False)
        >>> x = torch.tensor([[2, 2],
        ...                   [1, 2.]])
        >>> op.mm(x)
        tensor([[ 4.,  6.],
                [10., 14.]])
        >>> op2 = op + op
        >>> op2.mm(x)
        tensor([[ 8., 12.],
                [20., 28.]])

        Parameters
        ----------
        b: LinearOperator
            The linear operator to be added.

        Returns
        -------
        LinearOperator
            The result of the addition.

        """
        assert isinstance(
            b, LinearOperator
        ), "Only addition with another LinearOperator is supported"
        if self.shape[-2:] != b.shape[-2:]:
            raise RuntimeError("Mismatch shape of add operation: %s and %s" %
                               (self.shape, b.shape))
        if isinstance(self, MatrixLinearOperator) and isinstance(
                b, MatrixLinearOperator):
            return LinearOperator.m(self.fullmatrix() + b.fullmatrix())
        return AddLinearOperator(self, b)

    def __sub__(self, b: LinearOperator):
        """Subtraction with another linear operator.

        Examples
        --------
        >>> class Operator(LinearOperator):
        ...     def __init__(self, mat: torch.Tensor, is_hermitian: bool) -> None:
        ...         super(Operator, self).__init__(
        ...             shape=mat.shape,
        ...             is_hermitian=is_hermitian,
        ...             dtype=mat.dtype,
        ...             device=mat.device,
        ...             _suppress_hermit_warning=True,
        ...         )
        ...         self.mat = mat
        ...     def _mv(self, x: torch.Tensor) -> torch.Tensor:
        ...         return torch.matmul(self.mat, x.unsqueeze(-1)).squeeze(-1)
        ...     def _mm(self, x: torch.Tensor) -> torch.Tensor:
        ...         return torch.matmul(self.mat, x)
        ...     def _rmv(self, x: torch.Tensor) -> torch.Tensor:
        ...         return torch.matmul(self.mat.transpose(-3, -1).conj(), x.unsqueeze(-1)).squeeze(-1)
        ...     def _rmm(self, x: torch.Tensor) -> torch.Tensor:
        ...         return torch.matmul(self.mat.transpose(-2, -1).conj(), x)
        ...     def _fullmatrix(self) -> torch.Tensor:
        ...         return self.mat
        ...     def _getparamnames(self, prefix: str = "") -> List[str]:
        ...         return [prefix + "mat"]
        >>> op = Operator(torch.tensor([[1, 2.],
        ...                             [3, 4]]), is_hermitian=False)
        >>> op1 = Operator(torch.tensor([[0, 1.],
        ...                              [1, 2]]), is_hermitian=False)
        >>> x = torch.tensor([[2, 2],
        ...                   [1, 2.]])
        >>> op.mm(x)
        tensor([[ 4.,  6.],
                [10., 14.]])
        >>> op2 = op - op1
        >>> op2.mm(x)
        tensor([[3., 4.],
                [6., 8.]])

        Parameters
        ----------
        b: LinearOperator
            The linear operator to be subtracted.

        Returns
        -------
        LinearOperator
            The result of the subtraction.

        """

        assert isinstance(
            b, LinearOperator
        ), "Only subtraction with another LinearOperator is supported"
        if self.shape[-2:] != b.shape[-2:]:
            raise RuntimeError("Mismatch shape of add operation: %s and %s" %
                               (self.shape, b.shape))
        if isinstance(self, MatrixLinearOperator) and isinstance(
                b, MatrixLinearOperator):
            return LinearOperator.m(self.fullmatrix() - b.fullmatrix())
        return AddLinearOperator(self, b, -1)

    def __rsub__(self, b: LinearOperator):
        return b.__sub__(self)

    def __mul__(self, f: Union[int, float]):
        if not (isinstance(f, int) or isinstance(f, float)):
            raise TypeError(
                "LinearOperator multiplication only supports integer or floating point"
            )
        if isinstance(self, MatrixLinearOperator):
            return LinearOperator.m(self.fullmatrix() * f)
        return MulLinearOperator(self, f)

    def __rmul__(self, f: Union[int, float]):
        return self.__mul__(f)

    # properties
    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the linear operator."""
        return self._dtype

    @property
    def device(self) -> torch.device:
        """The device of the linear operator."""
        return self._device

    @property
    def shape(self) -> Sequence[int]:
        """The shape of the linear operator."""
        return self._shape

    @property
    def is_hermitian(self) -> bool:
        """Whether the linear operator is Hermitian."""
        return self._is_hermitian

    # implementation
    @property
    def is_mv_implemented(self) -> bool:
        """Whether the ``.mv()`` method is implemented."""
        return True

    @property
    def is_mm_implemented(self) -> bool:
        """Whether the ``.mm()`` method is implemented."""
        return self._is_mm_implemented

    @property
    def is_rmv_implemented(self) -> bool:
        """Whether the ``.rmv()`` method is implemented."""
        return self._is_rmv_implemented

    @property
    def is_rmm_implemented(self) -> bool:
        """Whether the ``.rmm()`` method is implemented."""
        return self._is_rmm_implemented

    @property
    def is_fullmatrix_implemented(self) -> bool:
        """Whether the ``.fullmatrix()`` method is implemented."""
        return self._is_fullmatrix_implemented

    @property
    def is_getparamnames_implemented(self) -> bool:
        """Whether the ``._getparamnames()`` method is implemented."""
        return self._is_gpn_implemented

    # private functions
    def _adjoint_rmv(self, xt: torch.Tensor) -> torch.Tensor:
        """calculate the right matvec multiplication by using the adjoint trick.

        Parameters
        ----------
        xt: torch.tensor
            The vector of shape ``(...,p)`` where the adjoint linear operation is operated at.

        Returns
        -------
        torch.tensor
            The result of the adjoint linear operation with shape ``(...,q)``

        """
        # xt: (*BY, p)
        # xdummy: (*BY, q)

        BY = xt.shape[:-1]
        BA = self.shape[:-2]
        BAY = get_bcasted_dims(BY, BA)

        # calculate y = Ax
        p, q = self.shape[-2:]
        xdummy = torch.zeros((*BAY, q), dtype=xt.dtype,
                             device=xt.device).requires_grad_()
        with torch.enable_grad():
            y = self.mv(xdummy)  # (*BAY, p)

        # calculate (dL/dx)^T = A^T (dL/dy)^T with (dL/dy)^T = xt
        xt2 = xt.contiguous().expand_as(y)  # (*BAY, p)
        res = torch.autograd.grad(
            y, xdummy, grad_outputs=xt2,
            create_graph=torch.is_grad_enabled())[0]  # (*BAY, q)
        return res

    def _assert_if_init_executed(self):
        if not hasattr(self, "_shape"):
            raise RuntimeError("super().__init__ must be executed first")


# Helper Classes
class MatrixLinearOperator(LinearOperator):
    """Class method to wrap a matrix into ``LinearOperator``.
    It is a standard linear operator, used in many operations.

    Examples
    --------
    >>> import torch
    >>> seed = torch.manual_seed(100)
    >>> mat = torch.rand(3, 2)
    >>> linop = MatrixLinearOperator(mat, is_hermitian=False)
    >>> print(linop)
    MatrixLinearOperator with shape (3, 2):
       tensor([[0.1117, 0.8158],
               [0.2626, 0.4839],
               [0.6765, 0.7539]])
    >>> x = torch.rand(2, 2)
    >>> linop.mm(x)
    tensor([[0.1991, 0.1011],
            [0.1696, 0.0684],
            [0.3345, 0.1180]])
    >>> x = torch.rand(3, 2)
    >>> linop.mv(x)
    tensor([[0.6137, 0.3879, 0.6369],
            [0.7220, 0.5680, 1.0753],
            [0.7821, 0.5460, 0.9626]])

    """

    def __init__(self, mat: torch.Tensor, is_hermitian: bool) -> None:
        """Initialize the ``MatrixLinearOperator``.

        Parameters
        ----------
        mat : torch.Tensor
            The matrix to be wrapped.
        is_hermitian : bool
            Indicating if the matrix is Hermitian. If ``None``, the symmetry
            will be checked. If supplied as a bool, there is no check performed.

        """

        super(MatrixLinearOperator, self).__init__(
            shape=mat.shape,
            is_hermitian=is_hermitian,
            dtype=mat.dtype,
            device=mat.device,
            _suppress_hermit_warning=True,
        )
        self.mat = mat

    def __repr__(self):
        """Representation of the ``MatrixLinearOperator``.

        Returns
        -------
        str
            The representation of the ``MatrixLinearOperator``.

        """
        return "MatrixLinearOperator with shape %s:\n   %s" % \
            (shape2str(self.shape), indent(self.mat.__repr__(), 3))

    def _mv(self, x: torch.Tensor) -> torch.Tensor:
        """Matrix-vector multiplication.

        Parameters
        ----------
        x: torch.Tensor
            The vector with shape ``(...,q)`` where the linear operation is
            operated on.

        Returns
        -------
        torch.Tensor
            The result of the linear operation with shape ``(...,p)``

        """
        return torch.matmul(self.mat, x.unsqueeze(-1)).squeeze(-1)

    def _mm(self, x: torch.Tensor) -> torch.Tensor:
        """Matrix-matrix multiplication.

        Parameters
        ----------
        x: torch.Tensor
            The matrix with shape ``(...,q,r)`` where the linear operation is
            operated on.

        Returns
        -------
        torch.Tensor
            The result of the linear operation with shape ``(...,p,r)``

        """
        return torch.matmul(self.mat, x)

    def _rmv(self, x: torch.Tensor) -> torch.Tensor:
        """Matrix-vector adjoint multiplication.

        Parameters
        ----------
        x: torch.Tensor
            The vector with shape ``(...,p)`` where the adjoint linear operation is
            operated on.

        Returns
        -------
        torch.Tensor
            The result of the adjoint linear operation with shape ``(...,q)``

        """
        return torch.matmul(self.mat.transpose(-2, -1).conj(),
                            x.unsqueeze(-1)).squeeze(-1)

    def _rmm(self, x: torch.Tensor) -> torch.Tensor:
        """Matrix-matrix adjoint multiplication.

        Parameters
        ----------
        x: torch.Tensor
            The matrix with shape ``(...,p,r)`` where the adjoint linear operation is
            operated on.

        Returns
        -------
        torch.Tensor
            The result of the adjoint linear operation with shape ``(...,q,r)``

        """
        return torch.matmul(self.mat.transpose(-2, -1).conj(), x)

    def _fullmatrix(self) -> torch.Tensor:
        """Full matrix representation of the linear operator.

        Returns
        -------
        torch.Tensor
            The full matrix representation of the linear operator.

        """
        return self.mat

    def _getparamnames(self, prefix: str = "") -> List[str]:
        """Get the parameter names that affects the method.

        Parameters
        ----------
        prefix: str
            The prefix to be appended in front of the parameters name.
            This usually contains the dots.

        Returns
        -------
        List[str]
            List of parameter names (including the prefix) that affecting
            the ``LinearOperator``.

        """
        return [prefix + "mat"]


class AddLinearOperator(LinearOperator):
    """Adds two linear operators.

    Examples
    --------
    >>> import torch
    >>> seed = torch.manual_seed(100)
    >>> class MyLinOp(LinearOperator):
    ...     def __init__(self, shape):
    ...         super(MyLinOp, self).__init__(shape)
    ...         self.param = torch.rand(shape)
    ...     def _getparamnames(self, prefix=""):
    ...         return [prefix + "param"]
    ...     def _mv(self, x):
    ...         return torch.matmul(self.param, x)
    ...     def _rmv(self, x):
    ...         return torch.matmul(self.param.transpose(-2,-1).conj(), x)
    ...     def _mm(self, x):
    ...         return torch.matmul(self.param, x)
    ...     def _rmm(self, x):
    ...         return torch.matmul(self.param.transpose(-2,-1).conj(), x)
    ...     def _fullmatrix(self):
    ...         return self.param
    >>> linop1 = MyLinOp((1,3,1,2))
    >>> linop2 = MyLinOp((1,3,1,2))
    >>> linop = AddLinearOperator(linop1, linop2)
    >>> print(linop)
    AddLinearOperator with shape (1, 3, 1, 2) of:
     * LinearOperator (MyLinOp) with shape (1, 3, 1, 2), dtype = torch.float32, device = cpu
     * LinearOperator (MyLinOp) with shape (1, 3, 1, 2), dtype = torch.float32, device = cpu
    >>> x = torch.rand(1,3,2,2)
    >>> linop.mv(x)
    tensor([[[[0.6256, 1.0689]],
    <BLANKLINE>
             [[0.6039, 0.5380]],
    <BLANKLINE>
             [[0.9702, 2.1129]]]])
    >>> x = torch.rand(1,3,1,1)
    >>> linop.rmv(x)
    tensor([[[[0.1662],
              [0.3813]],
    <BLANKLINE>
             [[0.4460],
              [0.5705]],
    <BLANKLINE>
             [[0.5942],
              [1.1089]]]])
    >>> x = torch.rand(1,2,2,1)
    >>> linop.mm(x)
    tensor([[[[0.7845],
              [0.5439]]],
    <BLANKLINE>
    <BLANKLINE>
            [[[0.6518],
              [0.4318]]],
    <BLANKLINE>
    <BLANKLINE>
            [[[1.4336],
              [0.9796]]]])

    """

    def __init__(self, a: LinearOperator, b: LinearOperator, mul: int = 1):
        """Initialize the ``AddLinearOperator``.

        Parameters
        ----------
        a: LinearOperator
            The first linear operator to be added.
        b: LinearOperator
            The second linear operator to be added.
        mul: int
            The multiplier of the second linear operator. Default to 1.
            If -1, then the second linear operator will be subtracted.

        """
        shape = (*get_bcasted_dims(a.shape[:-2], b.shape[:-2]), a.shape[-2],
                 b.shape[-1])
        is_hermitian = a.is_hermitian and b.is_hermitian
        super(AddLinearOperator, self).__init__(
            shape=shape,
            is_hermitian=is_hermitian,
            dtype=a.dtype,
            device=a.device,
            _suppress_hermit_warning=True,
        )
        self.a = a
        self.b = b
        assert mul == 1 or mul == -1
        self.mul = mul

    def __repr__(self):
        """Representation of the ``AddLinearOperator``.

        Returns
        -------
        str
            The representation of the ``AddLinearOperator``.

        """
        return "AddLinearOperator with shape %s of:\n * %s\n * %s" % \
            (shape2str(self.shape),
             indent(self.a.__repr__(), 3),
             indent(self.b.__repr__(), 3))

    def _mv(self, x: torch.Tensor) -> torch.Tensor:
        """Matrix-vector multiplication.

        Parameters
        ----------
        x: torch.Tensor
            The vector with shape ``(...,q)`` where the linear operation is operated on

        Returns
        -------
        torch.Tensor
            The result of the linear operation with shape ``(...,p)``

        """
        return self.a._mv(x) + self.mul * self.b._mv(x)

    def _rmv(self, x: torch.Tensor) -> torch.Tensor:
        """Transposed matrix-vector multiplication.

        Parameters
        ----------
        x: torch.Tensor
            The vector of shape ``(...,p)`` where the adjoint linear operation is operated at.

        Returns
        -------
        torch.Tensor
            The result of the adjoint linear operation with shape ``(...,q)``

        """
        return self.a.rmv(x) + self.mul * self.b.rmv(x)

    def _getparamnames(self, prefix: str = "") -> List[str]:
        """Get the parameter names that affects most of the methods (i.e. mm, mv, rmm, rmv).

        Parameters
        ----------
        prefix: str
            The prefix to be appended in front of the parameters name.
            This usually contains the dots.

        Returns
        -------
        List[str]
            List of parameter names (including the prefix) that affecting
            the ``LinearOperator``.

        """
        return self.a._getparamnames(prefix=prefix + "a.") + \
            self.b._getparamnames(prefix=prefix + "b.")


class MulLinearOperator(LinearOperator):
    """Multiply a linear operator with a scalar.

    Examples
    --------
    >>> import torch
    >>> seed = torch.manual_seed(100)
    >>> class MyLinOp(LinearOperator):
    ...     def __init__(self, shape):
    ...         super(MyLinOp, self).__init__(shape)
    ...         self.param = torch.rand(shape)
    ...     def _getparamnames(self, prefix=""):
    ...         return [prefix + "param"]
    ...     def _mv(self, x):
    ...         return torch.matmul(self.param, x)
    >>> linop = MyLinOp((1,3,1,2))
    >>> print(linop)
    LinearOperator (MyLinOp) with shape (1, 3, 1, 2), dtype = torch.float32, device = cpu
    >>> x = torch.rand(1,3,2,2)
    >>> linop.mv(x)
    tensor([[[[0.1991, 0.1011]],
    <BLANKLINE>
             [[0.3764, 0.5742]],
    <BLANKLINE>
             [[1.0345, 1.1802]]]])
    >>> linop2 = linop * 2
    >>> linop2.mv(x)
    tensor([[[[0.3981, 0.2022]],
    <BLANKLINE>
             [[0.7527, 1.1485]],
    <BLANKLINE>
             [[2.0691, 2.3603]]]])

    """

    def __init__(self, a: LinearOperator, f: Union[int, float]):
        """Initialize the MulLinearOperator.

        Parameters
        ----------
        a: LinearOperator
            Linear operator to be multiplied.
        f: Union[int, float]
            Integer or floating point number to be multiplied.

        """
        shape = a.shape
        is_hermitian = a.is_hermitian
        super(MulLinearOperator, self).__init__(
            shape=shape,
            is_hermitian=is_hermitian,
            dtype=a.dtype,
            device=a.device,
            _suppress_hermit_warning=True,
        )
        self.a = a
        self.f = f

    def __repr__(self):
        """Representation of the ``MulLinearOperator``.

        Returns
        -------
        str
            The representation of the ``MulLinearOperator``.

        """
        return "MulLinearOperator with shape %s of: \n * %s\n * %s" % \
            (shape2str(self.shape),
             indent(self.a.__repr__(), 3),
             indent(self.f.__repr__(), 3))

    def _mv(self, x: torch.Tensor) -> torch.Tensor:
        """Matrix-vector multiplication.

        Parameters
        ----------
        x: torch.Tensor
            The vector with shape ``(...,q)`` where the linear operation is
            operated on.

        Returns
        -------
        torch.Tensor
            The result of the linear operation with shape ``(...,p)``

        """
        return self.a._mv(x) * self.f

    def _rmv(self, x: torch.Tensor) -> torch.Tensor:
        """Transposed matrix-vector multiplication.

        Parameters
        ----------
        x: torch.Tensor
            The vector of shape ``(...,p)`` where the adjoint linear operation
            is operated at.

        Returns
        -------
        torch.Tensor
            The result of the adjoint linear operation with shape ``(...,q)``

        """
        return self.a._rmv(x) * self.f

    def _getparamnames(self, prefix: str = "") -> List[str]:
        """Get the parameter names that affects most of the methods.
        (i.e. mm, mv, rmm, rmv).

        Parameters
        ----------
        prefix: str
            The prefix to be appended in front of the parameters name.
            This usually contains the dots.

        Returns
        -------
        List[str]
            List of parameter names (including the prefix) that affecting
            the ``LinearOperator``.

        """
        pnames = self.a._getparamnames(prefix=prefix + "a.")
        return pnames


class MatmulLinearOperator(LinearOperator):
    """
    Matrix-matrix multiplication of two linear operators.

    Examples
    --------
    >>> import torch
    >>> seed = torch.manual_seed(100)
    >>> class MyLinOp(LinearOperator):
    ...     def __init__(self, shape):
    ...         super(MyLinOp, self).__init__(shape)
    ...         self.param = torch.rand(shape)
    ...     def _getparamnames(self, prefix=""):
    ...         return [prefix + "param"]
    ...     def _mv(self, x):
    ...         return torch.matmul(self.param, x)
    >>> linop1 = MyLinOp((1,3,2,2))
    >>> linop2 = MyLinOp((1,3,2,2))
    >>> linop = MatmulLinearOperator(linop1, linop2)
    >>> print(linop)
    MatmulLinearOperator with shape (1, 3, 2, 2) of:
     * LinearOperator (MyLinOp) with shape (1, 3, 2, 2), dtype = torch.float32, device = cpu
     * LinearOperator (MyLinOp) with shape (1, 3, 2, 2), dtype = torch.float32, device = cpu
    >>> x = torch.rand(1,2,2,1)
    >>> linop.mm(x)
    tensor([[[[0.7998],
              [0.8016]],
    <BLANKLINE>
             [[0.6515],
              [0.6835]]],
    <BLANKLINE>
    <BLANKLINE>
            [[[0.9251],
              [1.1611]],
    <BLANKLINE>
             [[0.2781],
              [0.3609]]],
    <BLANKLINE>
    <BLANKLINE>
            [[[0.2591],
              [0.2376]],
    <BLANKLINE>
             [[0.8009],
              [0.8087]]]])

    """

    def __init__(self,
                 a: LinearOperator,
                 b: LinearOperator,
                 is_hermitian: bool = False):
        """Initialize the ``MatmulLinearOperator``.

        Parameters
        ----------
        a: LinearOperator
            The first linear operator to be multiplied.
        b: LinearOperator
            The second linear operator to be multiplied.
        is_hermitian: bool
            Whether the result is Hermitian. Default to False.

        """
        shape = (*get_bcasted_dims(a.shape[:-2], b.shape[:-2]), a.shape[-2],
                 b.shape[-1])
        super(MatmulLinearOperator, self).__init__(
            shape=shape,
            is_hermitian=is_hermitian,
            dtype=a.dtype,
            device=a.device,
            _suppress_hermit_warning=True,
        )
        self.a = a
        self.b = b

    def __repr__(self):
        """Representation of the ``MatmulLinearOperator``.

        Returns
        -------
        str
            The representation of the ``MatmulLinearOperator``.

        """
        return "MatmulLinearOperator with shape %s of:\n * %s\n * %s" % \
            (shape2str(self.shape),
             indent(self.a.__repr__(), 3),
             indent(self.b.__repr__(), 3))

    def _mv(self, x: torch.Tensor) -> torch.Tensor:
        """Matrix-vector multiplication.

        Parameters
        ----------
        x: torch.Tensor
            The vector with shape ``(...,q)`` where the linear operation is
            operated on.

        Returns
        -------
        torch.Tensor
            The result of the linear operation with shape ``(...,p)``

        """
        return self.a._mv(self.b._mv(x))

    def _rmv(self, x: torch.Tensor) -> torch.Tensor:
        """Transposed matrix-vector multiplication.

        Parameters
        ----------
        x: torch.Tensor
            The vector of shape ``(...,p)`` where the adjoint linear operation
            is operated at.

        Returns
        -------
        torch.Tensor
            The result of the adjoint linear operation with shape ``(...,q)``

        """
        return self.b.rmv(self.a.rmv(x))

    def _getparamnames(self, prefix: str = "") -> List[str]:
        return self.a._getparamnames(prefix=prefix + "a.") + \
            self.b._getparamnames(prefix=prefix + "b.")


class AdjointLinearOperator(LinearOperator):
    """Adjoint of a LinearOperator.

    This is used to calculate the adjoint of a LinearOperator without
    explicitly constructing the adjoint matrix. This is useful when the
    adjoint matrix is not explicitly constructed, e.g. when the LinearOperator
    is a function of other parameters.

    Examples
    --------
    >>> import torch
    >>> seed = torch.manual_seed(100)
    >>> class MyLinOp(LinearOperator):
    ...     def __init__(self, shape):
    ...         super(MyLinOp, self).__init__(shape)
    ...         self.param = torch.rand(shape)
    ...     def _getparamnames(self, prefix=""):
    ...         return [prefix + "param"]
    ...     def _mv(self, x):
    ...         return torch.matmul(self.param, x)
    ...     def _rmv(self, x):
    ...         return torch.matmul(self.param.transpose(-2,-1).conj(), x)
    >>> linop = MyLinOp((1,3,1,2))
    >>> print(linop)
    LinearOperator (MyLinOp) with shape (1, 3, 1, 2), dtype = torch.float32, device = cpu
    >>> x = torch.rand(1,3,1,1)
    >>> linop.rmv(x)
    tensor([[[[0.0293],
              [0.2143]],
    <BLANKLINE>
             [[0.0112],
              [0.0207]],
    <BLANKLINE>
             [[0.1407],
              [0.1568]]]])
    >>> linop2 = linop.H
    >>> linop2.mv(x)
    tensor([[[[0.0293],
              [0.2143]],
    <BLANKLINE>
             [[0.0112],
              [0.0207]],
    <BLANKLINE>
             [[0.1407],
              [0.1568]]]])

    """

    def __init__(self, obj: LinearOperator):
        """Initialize the ``AdjointLinearOperator``.

        Parameters
        ----------
        obj: LinearOperator
            The linear operator to be adjointed.

        """
        super(AdjointLinearOperator, self).__init__(
            shape=(*obj.shape[:-2], obj.shape[-1], obj.shape[-2]),
            is_hermitian=obj.is_hermitian,
            dtype=obj.dtype,
            device=obj.device,
            _suppress_hermit_warning=True,
        )
        self.obj = obj

    def __repr__(self):
        return "AdjointLinearOperator with shape %s of:\n - %s" % \
            (shape2str(self.shape), indent(self.obj.__repr__(), 3))

    def _mv(self, x: torch.Tensor) -> torch.Tensor:
        """Matrix-vector multiplication.

        Parameters
        ----------
        x: torch.Tensor
            The vector with shape ``(...,q)`` where the linear operation is
            operated on.

        Returns
        -------
        torch.Tensor
            The result of the linear operation with shape ``(...,p)``

        """
        if not self.obj.is_rmv_implemented:
            raise RuntimeError(
                "The ._rmv of must be implemented to call .H.mv()")
        return self.obj._rmv(x)

    def _rmv(self, x: torch.Tensor) -> torch.Tensor:
        """Matrix-vector adjoint multiplication.

        Parameters
        ----------
        x: torch.Tensor
            The vector with shape ``(...,p)`` where the adjoint linear operation is
            operated on.

        Returns
        -------
        torch.Tensor
            The result of the adjoint linear operation with shape ``(...,q)``

        """
        return self.obj._mv(x)

    def _getparamnames(self, prefix: str = "") -> List[str]:
        """Get the parameter names that affects the method.

        Parameters
        ----------
        prefix: str
            The prefix to be appended in front of the parameters name.
            This usually contains the dots.

        Returns
        -------
        List[str]
            List of parameter names (including the prefix) that affecting
            the ``LinearOperator``.

        """
        return self.obj._getparamnames(prefix=prefix + "obj.")

    @property
    def H(self):
        """Adjoint of the linear operator.

        Returns
        -------
        LinearOperator
            Adjoint of the linear operator.

        """
        return self.obj

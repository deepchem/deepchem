from abc import abstractmethod
from typing import Tuple
import numpy as np
from deepchem.utils.dft_utils.hamilton.intor.utils import CSYMM, np2ctypes, int2ctypes


class BaseSymmetry(object):
    """ Base class for symmetry operations.

    This class defines the interface for symmetry operations.

    Examples
    --------
    >>> class SemNew(BaseSymmetry):
    ...     def get_reduced_shape(self, orig_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    ...         return orig_shape
    ...     @property
    ...     def code(self) -> str:
    ...         return "sn"
    ...     def reconstruct_array(self, arr: np.ndarray, orig_shape: Tuple[int, ...]) -> np.ndarray:
    ...         return arr
    >>> sym = SemNew()
    >>> sym.get_reduced_shape((2, 3, 4))
    (2, 3, 4)
    >>> sym.code
    'sn'

    """

    @abstractmethod
    def get_reduced_shape(self, orig_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Get reduced shape from original shape.

        Parameters
        ----------
        orig_shape : Tuple[int, ...]
            Original shape of the array.

        Returns
        -------
        Tuple[int, ...]
            Reduced shape of the array.
        """
        pass

    @property
    @abstractmethod
    def code(self) -> str:
        """
        Short code for this symmetry.

        Returns
        -------
        str
            Code representing the symmetry.
        """
        pass

    @abstractmethod
    def reconstruct_array(self, arr: np.ndarray,
                          orig_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Reconstruct full array from reduced symmetrized array.

        Parameters
        ----------
        arr : np.ndarray
            Reduced symmetrized array.
        orig_shape : Tuple[int, ...]
            Original shape of the array.

        Returns
        -------
        np.ndarray
            Reconstructed full array.
        """
        pass


class S1Symmetry(BaseSymmetry):
    """ Class for S1 symmetry.

    This class defines the operations for S1 symmetry. (no symmetry)

    Examples
    --------
    >>> sym = S1Symmetry()
    >>> sym.get_reduced_shape((2, 3, 4))
    (2, 3, 4)
    >>> sym.code
    's1'
    >>> sym.reconstruct_array(np.array([1, 2, 3]), (3,))
    array([1, 2, 3])

    """

    def get_reduced_shape(self, orig_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Get reduced shape from original shape (no symmetry).

        Parameters
        ----------
        orig_shape : Tuple[int, ...]
            Original shape of the array.

        Returns
        -------
        Tuple[int, ...]
            Reduced shape (same as original).
        """
        return orig_shape

    @property
    def code(self) -> str:
        """
        Short code for S1 symmetry.

        Returns
        -------
        str
            Code "s1".
        """
        return "s1"

    def reconstruct_array(self, arr: np.ndarray,
                          orig_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Reconstruct full array from reduced symmetrized array (no symmetry).

        Parameters
        ----------
        arr : np.ndarray
            Reduced symmetrized array.
        orig_shape : Tuple[int, ...]
            Original shape of the array.

        Returns
        -------
        np.ndarray
            Reconstructed full array (same as reduced array).
        """
        return arr


class S4Symmetry(BaseSymmetry):
    """ Class for S4 symmetry.

    This class defines the operations for S4 symmetry. (ijkl) == (jikl) == (ijlk) == (jilk)

    Examples
    --------
    >>> sym = S4Symmetry()
    >>> sym.get_reduced_shape((3, 3, 4, 4))
    (6, 10)
    >>> sym.code
    's4'
    >>> sym.reconstruct_array(np.random.rand(2, 3, 4, 4), (3, 3, 4, 4)).shape
    (3, 3, 4, 4)

    """

    def get_reduced_shape(self, orig_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Get reduced shape from original shape for S4 symmetry.
        Returned shape would be (..., i(j+1)/2, k(l+1)/2)

        Parameters
        ----------
        orig_shape : Tuple[int, ...]
            Original shape of the array.

        Returns
        -------
        Tuple[int, ...]
            Reduced shape for S4 symmetry.
        """
        self.__check_orig_shape(orig_shape)

        batchshape = orig_shape[:-4]
        ijshape = orig_shape[-4] * (orig_shape[-3] + 1) // 2
        klshape = orig_shape[-2] * (orig_shape[-1] + 1) // 2
        return (*batchshape, ijshape, klshape)

    @property
    def code(self) -> str:
        """
        Short code for S4 symmetry.

        Returns
        -------
        str
            Code "s4".
        """
        return "s4"

    def reconstruct_array(self, arr: np.ndarray,
                          orig_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Reconstruct full array from reduced symmetrized array for S4 symmetry. (..., ij/2, kl/2)

        Parameters
        ----------
        arr : np.ndarray
            Reduced symmetrized array.
        orig_shape : Tuple[int, ...]
            Original shape of the array.

        Returns
        -------
        np.ndarray
            Reconstructed full array for S4 symmetry.
        """
        self.__check_orig_shape(orig_shape)

        out = np.zeros(orig_shape, dtype=arr.dtype)
        fcn = CSYMM().fills4
        fcn(np2ctypes(out), np2ctypes(arr), int2ctypes(orig_shape[-4]),
            int2ctypes(orig_shape[-2]))
        return out

    def __check_orig_shape(self, orig_shape: Tuple[int, ...]):
        """
        Check if original shape satisfies requirements for S4 symmetry.

        Parameters
        ----------
        orig_shape : Tuple[int, ...]
            Original shape of the array.

        Raises
        ------
        AssertionError
            If original shape does not satisfy requirements for S4 symmetry.
        """
        assert len(orig_shape) >= 4
        assert orig_shape[-4] == orig_shape[-3]
        assert orig_shape[-2] == orig_shape[-1]

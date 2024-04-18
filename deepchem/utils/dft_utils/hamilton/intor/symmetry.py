from abc import abstractmethod
from typing import Tuple
import numpy as np
from deepchem.utils.dft_utils.hamilton.intor.utils import CSYMM, np2ctypes, int2ctypes


class BaseSymmetry(object):
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
    def reconstruct_array(self, arr: np.ndarray, orig_shape: Tuple[int, ...]) -> np.ndarray:
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
    # no symmetry
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

    def reconstruct_array(self, arr: np.ndarray, orig_shape: Tuple[int, ...]) -> np.ndarray:
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
    # (...ijkl) == (...jikl) == (...ijlk) == (...jilk)
    def get_reduced_shape(self, orig_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Get reduced shape from original shape for S4 symmetry.

        Parameters
        ----------
        orig_shape : Tuple[int, ...]
            Original shape of the array.

        Returns
        -------
        Tuple[int, ...]
            Reduced shape for S4 symmetry.
        """
        # returned shape would be (..., i(j+1)/2, k(l+1)/2)
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

    def reconstruct_array(self, arr: np.ndarray, orig_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Reconstruct full array from reduced symmetrized array for S4 symmetry.

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
        # reconstruct full array
        # arr: (..., ij/2, kl/2)
        self.__check_orig_shape(orig_shape)

        out = np.zeros(orig_shape, dtype=arr.dtype)
        fcn = CSYMM().fills4
        fcn(np2ctypes(out), np2ctypes(arr),
            int2ctypes(orig_shape[-4]), int2ctypes(orig_shape[-2]))
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

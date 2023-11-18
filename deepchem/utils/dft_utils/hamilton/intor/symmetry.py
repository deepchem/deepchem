from abc import abstractmethod, abstractproperty
from typing import Tuple
import numpy as np
from deepchem.utils.dft_utils.hamilton.intor.utils import CSYMM, np2ctypes, int2ctypes

class BaseSymmetry(object):
    @abstractmethod
    def get_reduced_shape(self, orig_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Get the reduced shape from the original shape
        """
        pass

    @abstractproperty
    def code(self) -> str:
        """
        Short code for this symmetry
        """
        pass

    @abstractmethod
    def reconstruct_array(self, arr: np.ndarray, orig_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Reconstruct the full array from the reduced symmetrized array.
        """
        pass

class S1Symmetry(BaseSymmetry):
    # no symmetry
    def get_reduced_shape(self, orig_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return orig_shape

    @property
    def code(self) -> str:
        return "s1"

    def reconstruct_array(self, arr: np.ndarray, orig_shape: Tuple[int, ...]) -> np.ndarray:
        return arr

class S4Symmetry(BaseSymmetry):
    # (...ijkl) == (...jikl) == (...ijlk) == (...jilk)
    def get_reduced_shape(self, orig_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        # the returned shape would be (..., i(j+1)/2, k(l+1)/2)
        self.__check_orig_shape(orig_shape)

        batchshape = orig_shape[:-4]
        ijshape = orig_shape[-4] * (orig_shape[-3] + 1) // 2
        klshape = orig_shape[-2] * (orig_shape[-1] + 1) // 2
        return (*batchshape, ijshape, klshape)

    @property
    def code(self) -> str:
        return "s4"

    def reconstruct_array(self, arr: np.ndarray, orig_shape: Tuple[int, ...]) -> np.ndarray:
        # reconstruct the full array
        # arr: (..., ij/2, kl/2)
        self.__check_orig_shape(orig_shape)

        out = np.zeros(orig_shape, dtype=arr.dtype)
        fcn = CSYMM().fills4
        fcn(np2ctypes(out), np2ctypes(arr),
            int2ctypes(orig_shape[-4]), int2ctypes(orig_shape[-2]))
        return out

    def __check_orig_shape(self, orig_shape: Tuple[int, ...]):
        assert len(orig_shape) >= 4
        assert orig_shape[-4] == orig_shape[-3]
        assert orig_shape[-2] == orig_shape[-1]

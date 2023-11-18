import ctypes
import ctypes.util
from deepchem.utils.dft_utils.hamilton.intor import dftlib
import numpy as np

# contains functions and constants that are used specifically for
# dqc.hamilton.intor files (no dependance on other files in dqc.hamilton.intor
# is required)

__all__ = ["NDIM", "CINT", "CGTO", "CPBC", "CSYMM", "c_null_ptr", "np2ctypes", "int2ctypes"]

# CONSTANTS
NDIM = 3

CINT = dftlib.CINT
CGTO = dftlib.CGTO
CPBC = dftlib.CPBC
# CVHF = dftlib.CVHF
CSYMM = dftlib.CSYMM

c_null_ptr = ctypes.POINTER(ctypes.c_void_p)

def np2ctypes(a: np.ndarray) -> ctypes.c_void_p:
    # get the ctypes of the numpy ndarray
    return a.ctypes.data_as(ctypes.c_void_p)

def int2ctypes(a: int) -> ctypes.c_int:
    # convert the python's integer to ctypes' integer
    return ctypes.c_int(a)

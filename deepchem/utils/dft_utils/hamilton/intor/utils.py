import ctypes
import ctypes.util
import dqclibs
import numpy as np

# CONSTANTS
NDIM = 3

CINT = dqclibs.CINT
CGTO = dqclibs.CGTO
CPBC = dqclibs.CPBC
# CVHF = dqclibs.CVHF
CSYMM = dqclibs.CSYMM

c_null_ptr = ctypes.POINTER(ctypes.c_void_p)


def np2ctypes(a: np.ndarray) -> ctypes.c_void_p:
    """Get the ctypes of the numpy ndarray

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1, 2, 3])
    >>> ctype = np2ctypes(a)

    Parameters
    ----------
    a : np.ndarray
        Numpy ndarray

    Returns
    -------
    ctypes.c_void_p
        ctypes of the numpy ndarray

    """
    return a.ctypes.data_as(ctypes.c_void_p)


def int2ctypes(a: int) -> ctypes.c_int:
    """Convert the python's integer to ctypes' integer

    Examples
    --------
    >>> a = 1
    >>> ctype = int2ctypes(a)

    Parameters
    ----------
    a : int
        Python's integer

    Returns
    -------
    ctypes.c_int
        ctypes' integer

    """
    return ctypes.c_int(a)

import re
import warnings

try:
    import pylibxc
except (ImportError, ModuleNotFoundError) as e:
    pylibxc = None
    warnings.warn(f"Failed to import pylibxc. Might not be able to use xc. {e}")

from deepchem.utils.dft_utils import BaseXC, LibXCLDA, LibXCGGA, LibXCMGGA
from deepchem.utils.dft_utils.xc.pytorch_xc import (
    PyTorchLDA,
    PyTorchGGA,
    PyTorchMGGA,
)


def get_libxc(name: str) -> BaseXC:
    """
    Get the XC object of libxc based on its libxc name.

    Parameters
    ----------
    name: str
        Full libxc name, e.g. "gga_c_pbe"

    Returns
    -------
    BaseXC
        XC object wrapping the requested libxc functional.
    """
    if pylibxc is None:
        raise ImportError("pylibxc is not available, so get_libxc() cannot be used.")

    obj = pylibxc.LibXCFunctional(name, "unpolarized")
    family = obj.get_family()
    del obj

    if family == 1:  # LDA
        return LibXCLDA(name)
    elif family == 2:  # GGA
        return LibXCGGA(name)
    elif family == 4:  # MGGA
        return LibXCMGGA(name)
    else:
        raise NotImplementedError(
            f"LibXC wrapper for family {family} has not been implemented"
        )


def get_pytorch_xc(name: str) -> BaseXC:
    """
    Get native PyTorch XC functionals.

    Supported
    ---------
    - lda_x
    - gga_x_pbe
    - mgga_x_tpss
    """
    lname = name.lower()

    if lname == "lda_x":
        return PyTorchLDA(lname)
    elif lname == "gga_x_pbe":
        return PyTorchGGA(lname)
    elif lname == "mgga_x_tpss":
        return PyTorchMGGA(lname)
    else:
        raise ValueError(f"Unknown native PyTorch XC functional: {name}")


def get_xc(xcstr: str) -> BaseXC:
    """
    Return the XC object based on xcstr.

    Behavior
    --------
    1. If xcstr is a supported native PyTorch XC name, return that.
    2. Otherwise, interpret xcstr as a LibXC expression, e.g.
       "lda_x + gga_c_pbe".

    Parameters
    ----------
    xcstr: str
        XC name or LibXC expression.

    Returns
    -------
    BaseXC
        XC object based on the given expression.
    """
    xcstr_strip = xcstr.strip()

    # First try native PyTorch XC names
    try:
        return get_pytorch_xc(xcstr_strip)
    except ValueError:
        pass

    # Otherwise use old LibXC-expression behavior
    pattern = r"([a-zA-Z_$][a-zA-Z_$0-9]*)"
    new_xcstr = re.sub(pattern, r'get_libxc("\1")', xcstr_strip)

    glob = {"get_libxc": get_libxc}
    return eval(new_xcstr, glob)
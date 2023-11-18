import os
import sys
import ctypes
import ctypes.util
from typing import Callable, Dict, Any

__all__ = ["CINT", "CGTO", "CPBC", "CSYMM"]

# libraries
_ext = "dylib" if sys.platform == "darwin" else "so"
_libcint_relpath = f"libcint.{_ext}"
_libcgto_relpath = f"libcgto.{_ext}"
_libcpbc_relpath = f"libpbc.{_ext}"
# _libcvhf_relpath = f"libcvhf.{_ext}"
_libcsymm_relpath = f"libsymm.{_ext}"

_libs: Dict[str, Any] = {}

def _library_loader(name: str, relpath: str) -> Callable:
    curpath = os.path.dirname(os.path.abspath(__file__))
    path = os.path.abspath(os.path.join(curpath, relpath))

    # load the library and cache the handler
    def fcn():
        if name not in _libs:
            try:
                _libs[name] = ctypes.cdll.LoadLibrary(path)
            except OSError as e:
                path2 = ctypes.util.find_library(name)
                if path2 is None:
                    raise e
                _libs[name] = ctypes.cdll.LoadLibrary(path2)
        return _libs[name]
    return fcn

CINT = _library_loader("cint", _libcint_relpath)
CGTO = _library_loader("cgto", _libcgto_relpath)
CPBC = _library_loader("cpbc", _libcpbc_relpath)
# CVHF = _library_loader("CVHF", _libcvhf_relpath)
CSYMM = _library_loader("symm", _libcsymm_relpath)

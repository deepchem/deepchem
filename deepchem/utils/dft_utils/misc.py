from typing import Callable, overload, TypeVar, Any, Mapping, Dict
import functools
import torch
import copy
import scipy.special
from deepchem.utils.dft_utils.config import config

T = TypeVar('T')
K = TypeVar('K')

def set_default_option(defopt: Dict, opt: Dict) -> Dict:
    # return a dictionary based on the options and if no item from option,
    # take it from defopt

    # make a shallow copy to detach the results from defopt
    res = copy.copy(defopt)
    res.update(opt)
    return res

def memoize_method(fcn: Callable[[Any], T]) -> Callable[[Any], T]:
    # alternative for lru_cache for memoizing a method without any arguments
    # lru_cache can produce memory leak for a method
    # this can be known by running test_ks_mem.py individually

    cachename = "__cch_" + fcn.__name__

    @functools.wraps(fcn)
    def new_fcn(self) -> T:
        if cachename in self.__dict__:
            return self.__dict__[cachename]
        else:
            res = fcn(self)
            self.__dict__[cachename] = res
            return res

    return new_fcn

def get_option(name: str, s: K, options: Mapping[K, T]) -> T:
    # get the value from dictionary of options, if not found, then raise an error
    if s in options:
        return options[s]
    else:
        raise ValueError(f"Unknown {name}: {s}. The available options are: {str(list(options.keys()))}")

@overload
def gaussian_int(n: int, alpha: float) -> float:
    ...

@overload
def gaussian_int(n: int, alpha: torch.Tensor) -> torch.Tensor:
    ...

def gaussian_int(n, alpha):
    # int_0^inf x^n exp(-alpha x^2) dx
    n1 = (n + 1) * 0.5
    return scipy.special.gamma(n1) / (2 * alpha ** n1)

class _Logger(object):
    def log(self, s: str, vlevel: int = 0):
        """
        Print the string ``s`` if the verbosity level exceeds ``vlevel``.
        """
        if config.VERBOSE > vlevel:
            print(s)

logger = _Logger()

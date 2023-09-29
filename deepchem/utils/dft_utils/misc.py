from typing import Callable, overload, TypeVar, Any, Mapping, Dict
import functools
import torch
import copy
import scipy.special
from deepchem.utils.dft_utils.config import config

T = TypeVar('T')
K = TypeVar('K')


def set_default_option(default_option: Dict, option: Dict) -> Dict:
    """Returns a dictionary based on the options and if no item from option,
    take it from defopt make a shallow copy to detach the results from defopt.

    Examples
    --------
    >>> from deepchem.utils.dft_utils.misc import set_default_option
    >>> # b gets overwritten and c is added
    >>> set_default_option({"a": 1, "b": 2}, {"b": 3, "c": 4})
    {'a': 1, 'b': 3, 'c': 4}

    Parameters
    ----------
    default_option: Dict
        Default options
    option: Dict
        Options

    Returns
    -------
    res: Dict
        Dictionary based on the options and if no item from option,
        take it from defopt
    """
    result = copy.copy(default_option)
    result.update(option)
    return result


def memoize_method(fcn: Callable[[Any], T]) -> Callable[[Any], T]:
    """Alternative for lru_cache for memoizing a method without any arguments
    lru_cache can produce memory leak for a method this can be known by
    running test_ks_mem.py individually.

    Examples
    --------
    >>> from deepchem.utils.dft_utils.misc import memoize_method
    >>> class MyClass:
    ...     @memoize_method
    ...     def fcn(self):
    ...         print("fcn called")
    ...         return 1
    >>> obj = MyClass()
    >>> obj.fcn()
    fcn called
    1
    >>> obj.fcn()
    1

    Parameters
    ----------
    fcn: Callable[[Any], T]
        Function to memoize

    Returns
    -------
    new_fcn: Callable[[Any], T]
        Memoized function

    """

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
    """Gets the value from dictionary of options,
    if not found, then raise an error.

    Examples
    --------
    >>> from deepchem.utils.dft_utils.misc import get_option
    >>> get_option("test", "a", {"a": 1, "b": 2})
    1

    Parameters
    ----------
    name: str
        Name of option
    s: K
        Key to look up
    options: Mapping[K, T]
        Mapping of options

    Returns
    -------
    value: T
        Value of option

    """
    if s in options:
        return options[s]
    else:
        raise ValueError(
            f"Unknown {name}: {s}. The available options are: {str(list(options.keys()))}"
        )


@overload
def gaussian_int(n: int, alpha: float) -> float:
    ...


@overload
def gaussian_int(n: int, alpha: torch.Tensor) -> torch.Tensor:
    ...


def gaussian_int(n, alpha):
    """
    Implements: int_0^inf x^n exp(-alpha x^2) dx

    Examples
    --------
    >>> from deepchem.utils.dft_utils.misc import gaussian_int
    >>> gaussian_int(1, 1.0)
    0.5
    >>> gaussian_int(1, torch.tensor(1.0))
    tensor(0.5000)

    Parameters
    ----------
    n:
        Power of x
    alpha:
        Parameter for gaussian

    Returns
    -------
    value:
        Value of integral

    """
    n1 = (n + 1) * 0.5
    return scipy.special.gamma(n1) / (2 * alpha**n1)


class _Logger(object):
    """Logger class for DFT utilities.

    Examples
    --------
    >>> from deepchem.utils.dft_utils.misc import logger
    >>> logger.log("test", 2)
    >>> logger.log("test", -1)
    test

    """

    def log(self, s: str, vlevel: int = 0):
        """
        Print the string ``s`` if the verbosity level exceeds ``vlevel``.

        Parameters
        ----------
        s: str
            String to print
        vlevel: int
            Verbosity level

        """
        if config.VERBOSE > vlevel:
            print(s)


logger = _Logger()

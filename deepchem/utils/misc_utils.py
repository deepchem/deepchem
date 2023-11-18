import torch
import scipy.special
from typing import Union, TypeVar, Callable, Any, Dict
import copy
import functools

T = TypeVar('T')
K = TypeVar('K')


def gaussian_int(
        n: int, alpha: Union[float,
                             torch.Tensor]) -> Union[float, torch.Tensor]:
    """int_0^inf x^n exp(-alpha x^2) dx

    Examples
    --------
    >>> gaussian_int(5, 1.0)
    1.0

    Parameters
    ----------
    n: int
        The order of the integral
    alpha: Union[float, torch.Tensor]
        The parameter of the gaussian

    Returns
    -------
    Union[float, torch.Tensor]
        The value of the integral

    """
    n1 = (n + 1) * 0.5
    return scipy.special.gamma(n1) / (2 * alpha**n1)


def set_default_option(defopt: Dict, opt: Dict) -> Dict:
    """Dictionary based on the options and if no item from option,
    take it from defopt make a shallow copy to detach the results from defopt

    Examples
    --------
    >>> set_default_option({"a": 1, "b": 0}, {"b": 2})
    {'a': 1, 'b': 2}

    Parameters
    ----------
    defopt: Dict
        Default options
    opt: Dict
        New options

    Returns
    -------
    Dict
        Updated options

    """
    res = copy.copy(defopt)
    res.update(opt)
    return res


def memoize_method(fcn: Callable[[Any], T]) -> Callable[[Any], T]:
    """alternative for lru_cache for memoizing a method without any arguments
    lru_cache can produce memory leak for a method

    Examples
    --------
    >>> class A:
    ...     @memoize_method
    ...     def f(self):
    ...         print("f")
    ...         return 1
    >>> a = A()
    >>> a.f()
    f
    1
    >>> a.f()
    1

    Parameters
    ----------
    fcn: Callable[[Any], T]
        The function to memoize

    Returns
    -------
    Callable[[Any], T]
        The memoized function

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

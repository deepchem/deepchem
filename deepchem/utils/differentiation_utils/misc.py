import contextlib
import copy
from typing import Mapping, Callable, Union, Dict, List


def set_default_option(defopt: Dict, opt: Dict) -> Dict:
    """return a dictionary based on the options and if no item from option,
    take it from defopt make a shallow copy to detach the results from defopt

    Examples
    --------
    >>> set_default_option({'a': 1, 'b': 2}, {'a': 3})
    {'a': 3, 'b': 2}

    Parameters
    ----------
    defopt: dict
        Default options
    opt: dict
        Options

    Returns
    -------
    dict
        Merged options

    """
    res = copy.copy(defopt)
    res.update(opt)
    return res


def get_and_pop_keys(dct: Dict, keys: List) -> Dict:
    """Get and pop keys from a dictionary

    Examples
    --------
    >>> get_and_pop_keys({'a': 1, 'b': 2}, ['a'])
    {'a': 1}

    Parameters
    ----------
    dct: dict
        Dictionary to pop from
    keys: list
        Keys to pop

    Returns
    -------
    dict
        Dictionary containing the popped keys

    """
    res = {}
    for k in keys:
        res[k] = dct.pop(k)
    return res


def get_method(algname: str, methods: Mapping[str, Callable],
               method: Union[str, Callable]) -> Callable:
    """Get a method from a dictionary of methods

    Examples
    --------
    >>> get_method('foo', {'bar': lambda: 1}, 'bar')()
    1

    Parameters
    ----------
    algname: str
        Name of the algorithm
    methods: dict
        Dictionary of methods
    method: str or callable
        Method to get

    Returns
    -------
    callable
        The method

    """

    if isinstance(method, str):
        methodname = method.lower()
        if methodname in methods:
            return methods[methodname]
        else:
            raise RuntimeError("Unknown %s method: %s" % (algname, method))
    elif hasattr(method, "__call__"):
        return method
    elif method is None:
        assert False, "Internal assert failed, method in get_method is not supposed" \
            "to be None. If this shows, then the corresponding function fails to " \
            "set the default method"
    else:
        raise TypeError(
            "Invalid method type: %s. Only str and callable are accepted." %
            type(method))


@contextlib.contextmanager
def dummy_context_manager():
    """Dummy context manager"""
    yield None


def assert_runtime(cond, msg=""):
    """Assert at runtime

    Examples
    --------
    >>> assert_runtime(False, "This is a test")
    Traceback (most recent call last):
    ...
    RuntimeError: This is a test

    Parameters
    ----------
    cond: bool
        Condition to assert
    msg: str
        Message to raise if condition is not met

    Raises
    ------
    RuntimeError
        If condition is not met

    """
    if not cond:
        raise RuntimeError(msg)

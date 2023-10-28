"""
Utility functions for getting, setting, and deleting attributes of an object.
Derived From: https://github.com/xitorch/xitorch/blob/master/xitorch/_utils/attr.py
"""
import re
import ast
from typing import Callable, Any

__all__ = ["get_attr", "set_attr", "del_attr"]

# pattern to split the names, e.g. "model.params[1]" into ["model", "params", "[1]"]
sp = re.compile(r"\[{0,1}[\"']{0,1}\w+[\"']{0,1}\]{0,1}")


def get_attr(obj: object, name: str):
    """Get the attribute of an object.

    Examples
    --------
    >>> from deepchem.utils.attribute_utils import get_attr
    >>> class MyClass:
    ...     def __init__(self):
    ...         self.a = 1
    ...         self.b = 2
    >>> obj = MyClass()
    >>> get_attr(obj, "a")
    1

    Parameters
    ----------
    obj : object
        The object to get the attribute from.
    name : str
        The name of the attribute.
    Returns
    -------
    val : object
        The value of the attribute.

    """
    return _get_attr(obj, _preproc_name(name))


def set_attr(obj: object, name: str, val: object):
    """Set the attribute of an object.

    Examples
    --------
    >>> from deepchem.utils import set_attr
    >>> class MyClass:
    ...     def __init__(self):
    ...         self.a = 1
    ...         self.b = 2
    >>> obj = MyClass()
    >>> set_attr(obj, "a", 3)
    >>> set_attr(obj, "c", 4)
    >>> obj.a
    3
    >>> obj.c
    4

    Parameters
    ----------
    obj : object
        The object to set the attribute to.
    name : str
        The name of the attribute.
    val : object
        The value to set the attribute to.

    Returns
    -------

    """
    return _set_attr(obj, _preproc_name(name), val)


def del_attr(obj: Any, name: str):
    """Delete the attribute of an object.

    Examples
    --------
    >>> from deepchem.utils import del_attr
    >>> class MyClass:
    ...     def __init__(self):
    ...         self.a = 1
    ...         self.b = 2
    >>> obj = MyClass()
    >>> del_attr(obj, "a")
    >>> try:
    ...     obj.a
    ... except AttributeError:
    ...     print("AttributeError")
    AttributeError

    """
    return _del_attr(obj, _preproc_name(name))


def _get_attr(obj: object, names: str):
    """Helper function for `get_attr`. Gets the attribute of an object.

    Parameters
    ----------
    obj : object
        The object to get the attribute from.
    names : str
        The name of the attribute.

    Returns
    -------
    val : object
        The value of the attribute.

    """
    attrfcn = lambda obj, name: getattr(obj, name)  # noqa: E731
    dictfcn = lambda obj, key: obj.__getitem__(key)  # noqa: E731
    listfcn = lambda obj, key: obj.__getitem__(key)  # noqa: E731
    return _traverse_attr(obj, names, attrfcn, dictfcn, listfcn)


def _set_attr(obj, names, val):
    """Helper function for `set_attr`. Sets the attribute of an object.

    Parameters
    ----------
    obj : object
        The object to set the attribute to.
    names : str
        The name of the attribute.
    val : object
        The value to set the attribute to.

    """
    attrfcn = lambda obj, name: setattr(obj, name, val)  # noqa: E731
    dictfcn = lambda obj, key: obj.__setitem__(key, val)  # noqa: E731
    listfcn = lambda obj, key: obj.__setitem__(key, val)  # noqa: E731
    return _traverse_attr(obj, names, attrfcn, dictfcn, listfcn)


def _del_attr(obj: Any, names: str):
    """Helper function for `del_attr`. Deletes the attribute of an object.

    Parameters
    ----------
    obj : object
        The object to delete the attribute from.
    names : str
        The name of the attribute.

    """
    attrfcn = lambda obj, name: delattr(obj, name)  # noqa: E731
    dictfcn = lambda obj, key: obj.__delitem__(key)  # noqa: E731

    def listfcn(obj: Any, key):
        obj.__delitem__(key)
        obj.insert(key, None)  # to preserve the length

    return _traverse_attr(obj, names, attrfcn, dictfcn, listfcn)


def _preproc_name(name: str):
    """Preprocess the name of the attribute.

    Examples
    --------
    >>> from deepchem.utils.attribute_utils import _preproc_name
    >>> _preproc_name("alpha.params[1]")
    ['alpha', 'params', '[1]']

    Parameters
    ----------
    name : str
        The name of the attribute.

    Returns
    -------
    names : str
        The preprocessed name of the attribute.

    """
    return sp.findall(name)


def _traverse_attr(obj: object, names: str, attrfcn: Callable,
                   dictfcn: Callable, listfcn: Callable):
    """Traverse the attribute of an object.

    Parameters
    ----------
    obj : object
        Object to traverse the attribute from.
    names : str
        Name of the attribute.
    attrfcn : Callable
        Function to get the attribute.
    dictfcn : Callable
        Function to get the dictionary.
    listfcn : Callable
        Function to get the list.

    Returns
    -------
    val : object
        The value of the attribute.

    """
    if len(names) == 1:
        return _applyfcn(obj, names[0], attrfcn, dictfcn, listfcn)
    else:
        return _applyfcn(_get_attr(obj, names[:-1]), names[-1], attrfcn,
                         dictfcn, listfcn)


def _applyfcn(obj: object, name: str, attrfcn: Callable, dictfcn: Callable,
              listfcn: Callable):
    """Apply the function to the attribute of an object.

    Parameters
    ----------
    obj : object
        Object to apply the function to.
    name : str
        Name of the attribute.
    attrfcn : Callable
        Function to get the attribute.
    dictfcn : Callable
        Function to get the dictionary.
    listfcn : Callable
        Function to get the list.

    Returns
    -------
    val : object
        The value of the attribute.

    """
    if name[0] == "[":
        key = ast.literal_eval(name[1:-1])
        if isinstance(obj, dict):
            return dictfcn(obj, key)
        elif isinstance(obj, list):
            return listfcn(obj, key)
        else:
            msg = "The parameter with [] must be either a dictionary or a list. "
            msg += "Got type: %s" % type(obj)
            raise TypeError(msg)
    else:
        return attrfcn(obj, name)

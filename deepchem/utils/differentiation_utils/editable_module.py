"""
EditableModule Utils
Derived From: https://github.com/xitorch/xitorch/blob/master/xitorch/_core/editable_module.py
"""
import torch
from typing import List, Callable, Any

torch_float_type = [torch.float32, torch.float, torch.float64, torch.float16]


# traversing functions
def _traverse_obj(obj: Any,
                  prefix: str,
                  action: Callable,
                  crit: Callable,
                  max_depth: int = 20,
                  exception_ids=None):
    """
    Traverse an object to get/set variables that are accessible through the object.
    The object can be a torch.nn.Module, a class instance, or an iterable object.
    The action is performed on the object that satisfies the criteria.

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.differentiation_utils.editable_module import _traverse_obj
    >>> class A:
    ...     def __init__(self):
    ...         self.a = 2
    ...         self.b = torch.tensor(3.0)
    ...         self.c = torch.tensor(4.0)
    ...         self.d = torch.tensor(5.0)
    ...
    >>> a = A()
    >>> def action(elmt, name, objdict, key):
    ...     print(name, elmt)
    ...
    >>> def crit(elmt):
    ...     return isinstance(elmt, torch.Tensor) and elmt.dtype in torch_float_type
    ...
    >>> _traverse_obj(a, "", action, crit)
    b tensor(3.)
    c tensor(4.)
    d tensor(5.)

    Parameters
    ----------
    obj: Any
        The object user wants to traverse down
    prefix: str
        Prefix of the name of the collected tensors.
    action: Callable
        The action to be performed on the object.
    crit: Callable
        The criteria to be met to perform the action.
    max_depth: int (default=20)
        Maximum recursive depth to avoid infinitely running program.
        If the maximum depth is reached, then raise a RecursionError.
    exception_ids: Set[int] (default=None)
        Set of ids of objects that are already traversed to avoid infinite loop.

    Raises
    ------
    RecursionError
        If the maximum depth is reached.
    RuntimeError
        If the object is not iterable or keyable.

    """
    if exception_ids is None:
        # None is set as default arg to avoid expanding list for multiple
        # invokes of _get_tensors without exception_ids argument
        exception_ids = set()

    if isinstance(obj, torch.nn.Module):
        generators = [obj._parameters.items(), obj._modules.items()]
        name_format = "{prefix}{key}"
        objdicts = [obj._parameters, obj._modules]
    elif hasattr(obj, "__dict__"):
        generators = [obj.__dict__.items()]
        name_format = "{prefix}{key}"
        objdicts = [obj.__dict__]
    elif hasattr(obj, "__iter__"):
        generators = [obj.items() if isinstance(obj, dict) else enumerate(obj)]
        name_format = "{prefix}[{key}]"
        objdicts = [obj]
    else:
        raise RuntimeError("The object must be iterable or keyable")

    for generator, objdict in zip(generators, objdicts):
        for key, elmt in generator:
            name = name_format.format(prefix=prefix, key=key)
            if crit(elmt):
                action(elmt, name, objdict, key)
                continue

            hasdict = hasattr(elmt, "__dict__")
            hasiter = hasattr(elmt, "__iter__")
            if hasdict or hasiter:
                # add exception to avoid infinite loop if there is a mutual dependant on objects
                if id(elmt) in exception_ids:
                    continue
                else:
                    exception_ids.add(id(elmt))

                if max_depth > 0:
                    _traverse_obj(elmt,
                                  action=action,
                                  crit=crit,
                                  prefix=name + "." if hasdict else name,
                                  max_depth=max_depth - 1,
                                  exception_ids=exception_ids)
                else:
                    raise RecursionError("Maximum number of recursion reached")


def _get_tensors(obj: Any, prefix="", max_depth=20):
    """
    Collect all tensors in an object recursively and return the tensors as well
    as their "names" (names meaning the address, e.g. "self.a[0].elmt").

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.differentiation_utils.editable_module import _get_tensors
    >>> class A:
    ...     def __init__(self):
    ...         self.a = 2
    ...         self.b = torch.tensor(3.0)
    ...         self.c = torch.tensor(4.0)
    ...         self.d = torch.tensor(5.0)
    ...
    >>> a = A()
    >>> _get_tensors(a)
    ([tensor(3.), tensor(4.), tensor(5.)], ['b', 'c', 'd'])

    Parameters
    ----------
    obj: Any
        The object user wants to traverse down
    prefix: str (default="")
        Prefix of the name of the collected tensors.
    max_depth: int (default=20)
        Maximum recursive depth to avoid infinitely running program.
        If the maximum depth is reached, then raise a RecursionError.

    Returns
    -------
    res: list[torch.Tensor]
        Sequence of tensors collected recursively in the object.
    name: list[str]
        Sequence of names of the collected tensors.

    """

    # get the tensors recursively towards torch.nn.Module
    res = []
    names = []

    def action(elmt, name, objdict, key):
        res.append(elmt)
        names.append(name)

    # traverse down the object to collect the tensors
    def crit(elmt):
        return isinstance(elmt, torch.Tensor) and elmt.dtype in torch_float_type

    _traverse_obj(obj,
                  action=action,
                  crit=crit,
                  prefix=prefix,
                  max_depth=max_depth)
    return res, names


def _set_tensors(obj: Any, all_params: List[torch.Tensor], max_depth: int = 20):
    """
    Set the tensors in an object to new tensor object listed in `all_params`.

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.differentiation_utils.editable_module import _set_tensors
    >>> class A:
    ...     def __init__(self):
    ...         self.a = 2
    ...         self.b = torch.tensor(3.0)
    ...         self.c = torch.tensor(4.0)
    ...         self.d = torch.tensor(5.0)
    ...
    >>> a = A()
    >>> _set_tensors(a, [torch.tensor(6.0), torch.tensor(7.0), torch.tensor(8.0)])
    >>> a.b
    tensor(6.)
    >>> a.c
    tensor(7.)

    Parameters
    ----------
    obj: an instance
        The object user wants to traverse down
    all_params: List[torch.Tensor]
        Sequence of tensors to be put in the object.
    max_depth: int (default=20)
        Maximum recursive depth to avoid infinitely running program.
        If the maximum depth is reached, then raise a RecursionError.

    """

    def action(elmt, name, objdict, key):
        objdict[key] = all_params.pop(0)

    # traverse down the object to collect the tensors
    def crit(elmt):
        return isinstance(elmt, torch.Tensor) and elmt.dtype in torch_float_type

    _traverse_obj(obj, action=action, crit=crit, prefix="", max_depth=max_depth)

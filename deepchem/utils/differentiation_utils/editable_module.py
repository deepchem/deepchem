"""
EditableModule Utils
Derived From: https://github.com/xitorch/xitorch/blob/master/xitorch/_core/editable_module.py
"""
import inspect
import warnings
from abc import abstractmethod
import copy
import torch
from typing import Sequence, Union, Dict, List, Callable, Any
from deepchem.utils.attribute_utils import get_attr, set_attr, del_attr

__all__ = ["EditableModule"]

torch_float_type = [torch.float32, torch.float, torch.float64, torch.float16]


class EditableModule(object):
    """EditableModule is a base class to enable classes that it inherits be
    converted to pure functions for higher order derivatives purpose.

    Usage
    -----
    To use this class, the user must implement the ``getparamnames`` method
    which returns a list of tensor names that affect the output of the method
    with name indicated in ``methodname``.

    Used in:

    - Classes of Density Functional Theory (DFT).

    - It can also be used in other classes that need to be converted to pure
      functions for higher order derivatives purpose.

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.differentiation_utils import EditableModule
    >>> class A(EditableModule):
    ...     def __init__(self, a):
    ...         self.b = a*a
    ...
    ...     def mult(self, x):
    ...         return self.b * x
    ...
    ...     def getparamnames(self, methodname, prefix=""):
    ...         if methodname == "mult":
    ...             return [prefix+"b"]
    ...         else:
    ...             raise KeyError()
    >>> a = torch.tensor(2.0).requires_grad_()
    >>> x = torch.tensor(0.4).requires_grad_()
    >>> alpha = A(a)
    >>> alpha.mult(x)
    tensor(1.6000, grad_fn=<MulBackward0>)
    >>> alpha.getparamnames("mult")
    ['b']
    >>> alpha.assertparams(alpha.mult, x)
    "mult" method check done

    """

    def getparams(self, methodname: str) -> Sequence[torch.Tensor]:
        """Returns a list of tensor parameters used in the object's operations.
        Requires the ``getparamnames`` method to be implemented.

        Parameters
        ----------
        methodname: str
            The name of the method of the class.

        Returns
        -------
        Sequence[torch.Tensor]
            Sequence of tensors that are involved in the specified method of the
            object.

        """

        paramnames = self.cached_getparamnames(methodname)
        return [get_attr(self, name) for name in paramnames]

    def setparams(self, methodname: str, *params) -> int:
        """Set the input parameters to the object's parameters to make a copy of
        the operations.

        Parameters
        ----------
        methodname: str
            The name of the method of the class.
        *params:
            The parameters to be set to the object's parameters.

        Returns
        -------
        int
            The number of parameters that are set to the object's parameters.

        """
        paramnames = self.cached_getparamnames(methodname)
        for name, val in zip(paramnames, params):
            try:
                set_attr(self, name, val)
            except TypeError:  # failed because val should be param
                del_attr(self, name)
                set_attr(self, name, val)

        return len(params)

    def cached_getparamnames(self,
                             methodname: str,
                             refresh: bool = False) -> List[str]:
        """getparamnames, but cached, so it is only called once

        Parameters
        ----------
        methodname: str
            The name of the method of the class.
        refresh: bool
            If True, the cache is refreshed.

        Returns
        -------
        List[str]
            Sequence of name of parameters affecting the output of the method.

        """
        if not hasattr(self, "_paramnames_"):
            self._paramnames_: Dict[str, List[str]] = {}

        if methodname not in self._paramnames_:
            self._paramnames_[methodname] = self.getparamnames(methodname)
        return self._paramnames_[methodname]

    @abstractmethod
    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        """
        This method should list tensor names that affect the output of the
        method with name indicated in ``methodname``.
        If the ``methodname`` is not on the list in this function, it should
        raise ``KeyError``.

        Parameters
        ---------
        methodname: str
            The name of the method of the class.
        prefix: str
            The prefix to be appended in front of the parameters name.
            This usually contains the dots.

        Returns
        -------
        List[str]
            Sequence of name of parameters affecting the output of the method.

        Raises
        ------
        KeyError
            If the list in this function does not contain ``methodname``.

        """
        pass

    def getuniqueparams(self,
                        methodname: str,
                        onlyleaves: bool = False) -> List[torch.Tensor]:
        """Returns the list of unique parameters involved in the method
        specified by `methodname`.

        Parameters
        ----------
        methodname: str
            Name of the method where the returned parameters play roles.
        onlyleaves: bool
            If True, only returns leaf tensors. Otherwise, returns all tensors.

        Returns
        -------
        List[torch.Tensor]
            List of tensors that are involved in the specified method of the
            object.

        """
        allparams = self.getparams(methodname)
        idxs = self._get_unique_params_idxs(methodname, allparams)
        if onlyleaves:
            return [allparams[i] for i in idxs if allparams[i].is_leaf]
        else:
            return [allparams[i] for i in idxs]

    def setuniqueparams(self, methodname: str, *uniqueparams) -> int:
        """Set the input parameters to the object's parameters to make a copy of
        the operations. The input parameters are unique parameters, i.e. they
        are not necessarily the same tensors as the object's parameters.

        Note: This function can only be run after running getuniqueparams.

        Parameters
        ----------
        methodname: str
            The name of the method of the class.
        *uniqueparams:
            The parameters to be set to the object's parameters. The number of
            parameters must be the same as the number of unique parameters
            returned by ``getuniqueparams``.

        Returns
        -------
        int
            The number of parameters that are set to the object's parameters.

        """
        nparams = self._number_of_params[methodname]
        allparams = [None for _ in range(nparams)]
        maps = self._unique_params_maps[methodname]

        for j in range(len(uniqueparams)):
            jmap = maps[j]
            p = uniqueparams[j]
            for i in jmap:
                allparams[i] = p

        return self.setparams(methodname, *allparams)

    def _get_unique_params_idxs(
            self,
            methodname: str,
            allparams: Union[Sequence[torch.Tensor],
                             None] = None) -> Sequence[int]:
        """Returns the list of unique parameters involved in the method

        Parameters
        ----------
        methodname: str
            Name of the method where the returned parameters play roles.
        allparams: list of tensors
            List of tensors that are involved in the specified method of the
            object.

        Returns
        -------
        Sequence[int]
            List of indices of the unique parameters in the list of all parameters.

        """

        if not hasattr(self, "_unique_params_idxs"):
            self._unique_params_idxs = {}  # type: Dict[str,Sequence[int]]
            self._unique_params_maps = {}
            self._number_of_params = {}

        if methodname in self._unique_params_idxs:
            return self._unique_params_idxs[methodname]
        if allparams is None:
            allparams = self.getparams(methodname)

        # get the unique ids
        ids = []  # type: List[int]
        idxs = []
        idx_map = []  # type: List[List[int]]
        for i in range(len(allparams)):
            param = allparams[i]
            id_param = id(param)

            # search the id if it has been added to the list
            try:
                jfound = ids.index(id_param)
                idx_map[jfound].append(i)
                continue
            except ValueError:
                pass

            ids.append(id_param)
            idxs.append(i)
            idx_map.append([i])

        self._number_of_params[methodname] = len(allparams)
        self._unique_params_idxs[methodname] = idxs
        self._unique_params_maps[methodname] = idx_map
        return idxs

    # debugging
    def assertparams(self, method: Callable, *args, **kwargs):
        """
        Perform a rigorous check on the implemented ``getparamnames``
        in the class for a given method and its parameters as well as keyword
        Parameters.
        It raises warnings if there are missing or excess parameters in the
        ``getparamnames`` implementation.

        Parameters
        ---------
        method: Callable
            The method of this class to be tested
        *args:
            Parameters of the method
        **kwargs:
            Keyword parameters of the method

        """
        # check the method input
        if not inspect.ismethod(method):
            raise TypeError("The input method must be a method")
        methodself = method.__self__
        if methodself is not self:
            raise RuntimeError(
                "The method does not belong to the same instance")

        methodname = method.__name__

        # assert if the method preserve the float tensors of the object
        self.__assert_method_preserve(method, *args, **kwargs)
        self.__assert_get_correct_params(
            method, *args,
            **kwargs)  # check if getparams returns the correct tensors
        print('"%s" method check done' % methodname)

    def __assert_method_preserve(self, method, *args, **kwargs):
        """This method assert if method does not change the float tensor
        parameters of the object (i.e. it preserves the state of the object).

        Parameters
        ----------
        method: Callable
            The method of this class to be tested

        Raises
        ------
        KeyError
            If the method does not preserve the float tensors of the object.

        """

        all_params0, names0 = _get_tensors(self)
        all_params0 = [p.clone() for p in all_params0]
        method(*args, **kwargs)
        all_params1, names1 = _get_tensors(self)

        # now assert if all_params0 == all_params1
        clsname = method.__self__.__class__.__name__
        methodname = method.__name__
        msg = "The method %s.%s does not preserve the object's float tensors: \n" % (
            clsname, methodname)
        if len(all_params0) != len(all_params1):
            msg += "The number of parameters changed:\n"
            msg += "* number of object's parameters before: %d\n" % len(
                all_params0)
            msg += "* number of object's parameters after : %d\n" % len(
                all_params1)
            raise KeyError(msg)

        for pname, p0, p1 in zip(names0, all_params0, all_params1):
            if p0.shape != p1.shape:
                msg += "The shape of %s changed\n" % pname
                msg += "* (before) %s.shape: %s\n" % (pname, p0.shape)
                msg += "* (after ) %s.shape: %s\n" % (pname, p1.shape)
                raise KeyError(msg)
            if not torch.allclose(p0, p1):
                msg += "The value of %s changed\n" % pname
                msg += "* (before) %s: %s\n" % (pname, p0)
                msg += "* (after ) %s: %s\n" % (pname, p1)
                raise KeyError(msg)

    def __assert_get_correct_params(self, method, *args, **kwargs):
        """This function perform checks if the getparams on the method returns
        the correct tensors

        Parameters
        ----------
        method: Callable
            The method of this class to be tested

        Raises
        ------
        KeyError
            If the method does not return the correct tensors.

        """

        methodname = method.__name__
        clsname = method.__self__.__class__.__name__

        # get all tensor parameters in the object
        all_params, all_names = _get_tensors(self)

        def _get_tensor_name(param):
            for i in range(len(all_params)):
                if id(all_params[i]) == id(param):
                    return all_names[i]
            return None

        # get the parameter tensors used in the operation and the tensors specified by the developer
        oper_names, oper_params = self.__list_operating_params(
            method, *args, **kwargs)
        user_names = self.getparamnames(method.__name__)
        user_params = [get_attr(self, name) for name in user_names]
        user_params_id = [id(p) for p in user_params]
        oper_params_id = [id(p) for p in oper_params]
        user_params_id_set = set(user_params_id)
        oper_params_id_set = set(oper_params_id)

        # check if the userparams contains non-tensor
        for i in range(len(user_params)):
            param = user_params[i]
            if (not isinstance(param, torch.Tensor)) or \
               (isinstance(param, torch.Tensor) and param.dtype not in torch_float_type):
                msg = "Parameter %s is a non-floating point tensor" % user_names[
                    i]
                raise KeyError(msg)

        # check if there are missing parameters (present in operating params, but not in the user params)
        missing_names = []
        for i in range(len(oper_names)):
            if oper_params_id[i] not in user_params_id_set:
                # if oper_names[i] not in user_names:
                missing_names.append(oper_names[i])
        # if there are missing parameters, give a warning (because the program
        # can still run correctly, e.g. missing parameters are parameters that
        # are never set to require grad)
        if len(missing_names) > 0:
            msg = "getparams for %s.%s does not include: %s" % (
                clsname, methodname, ", ".join(missing_names))
            warnings.warn(msg, stacklevel=3)

        # check if there are excessive parameters (present in the user params, but not in the operating params)
        excess_names = []
        for i in range(len(user_names)):
            if user_params_id[i] not in oper_params_id_set:
                # if user_names[i] not in oper_names:
                excess_names.append(user_names[i])
        # if there are excess parameters, give warnings
        if len(excess_names) > 0:
            msg = "getparams for %s.%s has excess parameters: %s" % \
                (clsname, methodname, ", ".join(excess_names))
            warnings.warn(msg, stacklevel=3)

    def __list_operating_params(self, method, *args, **kwargs):
        """Sequence the tensors used in executing the method by calling the method
        and see which parameters are connected in the backward graph"""

        # get all the tensors recursively
        all_tensors, all_names = _get_tensors(self)

        # copy the tensors and require them to be differentiable
        copy_tensors0 = [
            tensor.clone().detach().requires_grad_() for tensor in all_tensors
        ]
        copy_tensors = copy.copy(copy_tensors0)
        _set_tensors(self, copy_tensors)

        # run the method and see which one has the gradients
        output = method(*args, **kwargs)
        if not isinstance(output, torch.Tensor):
            raise RuntimeError(
                "The method to be asserted must have a tensor output")
        output = output.sum()
        grad_tensors = torch.autograd.grad(output,
                                           copy_tensors0,
                                           retain_graph=True,
                                           allow_unused=True)

        # return the original tensor
        all_tensors_copy = copy.copy(all_tensors)
        _set_tensors(self, all_tensors_copy)

        names = []
        params = []
        for i, grad in enumerate(grad_tensors):
            if grad is None:
                continue
            names.append(all_names[i])
            params.append(all_tensors[i])

        return names, params


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
    >>> from deepchem.utils.differentiation_utils.editable_module import _traverse_obj, torch_float_type
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
        if isinstance(obj, dict):
            generators = [obj.items()]
        else:
            generators = [enumerate(obj)]  # type: ignore[list-item]
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

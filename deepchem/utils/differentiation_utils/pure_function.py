import torch
import inspect
from typing import Callable, List, Tuple, Union
from deepchem.utils.attribute_utils import set_attr, del_attr
from deepchem.utils.differentiation_utils import EditableModule
from deepchem.utils.misc_utils import Uniquifier
from contextlib import contextmanager
from abc import abstractmethod


class PureFunction(object):
    """
    PureFunction class wraps methods to make it stateless and expose the pure
    function to take inputs of the original inputs (`params`) and the object's
    states (`objparams`).
    For functions, this class only acts as a thin wrapper.

    Restore stack stores list of (objparams, identical) everytime the objparams
    are set, it will store the old objparams and indication if the old and new
    objparams are identical.

    For Using this Class we first need to implement `_get_all_obj_params_init`
    and `_set_all_obj_params`.

    """

    def __init__(self, fcntocall: Callable):
        """Initialize the PureFunction.

        Parameters
        ----------
        fcntocall: Callable
            The function to be wrapped

        """
        self._state_change_allowed = True
        self._allobjparams = self._get_all_obj_params_init()
        self._uniq = Uniquifier(self._allobjparams)
        self._cur_objparams = self._uniq.get_unique_objs()
        self._fcntocall = fcntocall
        self._restore_stack: List[Tuple[List, bool]] = []

    def __call__(self, *params):
        """Call the wrapped function with the current object parameters and
        the input parameters.

        Parameters
        ----------
        params: tuple
            The input parameters of the wrapped function

        Returns
        -------
        Any
            The output of the wrapped function

        """
        return self._fcntocall(*params)

    @abstractmethod
    def _get_all_obj_params_init(self) -> List:
        """Get the initial object parameters.

        Returns
        -------
        List
            The initial object parameters

        """
        pass

    @abstractmethod
    def _set_all_obj_params(self, allobjparams: List):
        """Set the object parameters.

        Parameters
        ----------
        allobjparams: List
            The object parameters to be set

        """
        pass

    def objparams(self) -> List:
        """Get the current object parameters.

        Returns
        -------
        List
            The current object parameters

        """
        return self._cur_objparams

    def set_objparams(self, objparams: List):
        """Set the object parameters.

        Parameters
        ----------
        objparams: List
            The object parameters to be set

        TODO: check if identical with current object parameters

        """
        identical = _check_identical_objs(objparams, self._cur_objparams)
        self._restore_stack.append((self._cur_objparams, identical))
        if not identical:
            allobjparams = self._uniq.map_unique_objs(objparams)
            self._set_all_obj_params(allobjparams)
            self._cur_objparams = list(objparams)

    def restore_objparams(self):
        """Restore the object parameters to the previous state."""
        old_objparams, identical = self._restore_stack.pop(-1)
        if not identical:
            allobjparams = self._uniq.map_unique_objs(old_objparams)
            self._set_all_obj_params(allobjparams)
            self._cur_objparams = old_objparams

    @contextmanager
    def useobjparams(self, objparams: List):
        """Context manager to temporarily set the object parameters.

        Parameters
        ----------
        objparams: List
            The object parameters to be set temporarily

        """
        if not self._state_change_allowed:
            raise RuntimeError("The state change is disabled")
        try:
            self.set_objparams(objparams)
            yield
        finally:
            self.restore_objparams()

    @contextmanager
    def disable_state_change(self):
        """Context manager to temporarily disable the state change."""
        try:
            prev_status = self._state_change_allowed
            self._state_change_allowed = False
            yield
        finally:
            self._state_change_allowed = prev_status


class FunctionPureFunction(PureFunction):
    """Implementation of PureFunction for functions.
    It just acts as a thin wrapper for the function.

    Examples
    --------
    >>> def fcn(x, y):
    ...     return x + y
    >>> pfunc = FunctionPureFunction(fcn)
    >>> pfunc(1, 2)
    3

    """

    def _get_all_obj_params_init(self) -> List:
        """Get the initial object parameters.

        Returns
        -------
        List
            The initial object parameters

        """
        return []

    def _set_all_obj_params(self, objparams: List):
        """Set the object parameters.

        Parameters
        ----------
        objparams: List
            The object parameters to be set

        """
        pass


class EditableModulePureFunction(PureFunction):
    """Implementation of PureFunction for EditableModule.

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.differentiation_utils import EditableModule, get_pure_function
    >>> class A(EditableModule):
    ...     def __init__(self, a):
    ...         self.b = a*a
    ...     def mult(self, x):
    ...         return self.b * x
    ...     def getparamnames(self, methodname, prefix=""):
    ...         if methodname == "mult":
    ...             return [prefix+"b"]
    ...         else:
    ...             raise KeyError()
    >>> B = A(4)
    >>> m = get_pure_function(B.mult)
    >>> m.set_objparams([3])
    >>> m(2)
    6

    """

    def __init__(self, obj: EditableModule, method: Callable):
        """Initialize the EditableModulePureFunction.

        Parameters
        ----------
        obj: EditableModule
            The object to be wrapped
        method: Callable
            The method to be wrapped

        """
        self.obj = obj
        self.method = method
        super().__init__(method)

    def _get_all_obj_params_init(self) -> List:
        """Get the initial object parameters.

        Returns
        -------
        List
            The initial object parameters

        """
        return list(self.obj.getparams(self.method.__name__))

    def _set_all_obj_params(self, allobjparams: List):
        """Set the object parameters.

        Parameters
        ----------
        allobjparams: List
            The object parameters to be set

        """
        self.obj.setparams(self.method.__name__, *allobjparams)


class TorchNNPureFunction(PureFunction):
    """Implementation of PureFunction for torch.nn.Module.

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.differentiation_utils import get_pure_function
    >>> class A(torch.nn.Module):
    ...     def __init__(self, a):
    ...         super().__init__()
    ...         self.b = torch.nn.Parameter(torch.tensor(a*a))
    ...     def forward(self, x):
    ...         return self.b * x
    >>> B = A(4.)
    >>> m = get_pure_function(B.forward)
    >>> m.set_objparams([3.])
    >>> m(2)
    6.0

    """

    def __init__(self, obj: torch.nn.Module, method: Callable):
        """Initialize the TorchNNPureFunction.

        Parameters
        ----------
        obj: torch.nn.Module
            Object to be wrapped
        method: Callable
            Method to be wrapped

        """
        self.obj = obj
        self.method = method
        super().__init__(method)

    def _get_all_obj_params_init(self) -> List:
        """get the tensors in the torch.nn.Module to be used as params

        Returns
        -------
        List
            The initial object parameters

        """
        named_params = list(self.obj.named_parameters())
        if len(named_params) == 0:
            paramnames: List[str] = []
            obj_params: List[Union[torch.Tensor, torch.nn.Parameter]] = []
        else:
            paramnames_temp, obj_params_temp = zip(*named_params)
            paramnames = list(paramnames_temp)
            obj_params = list(obj_params_temp)
        self.names = paramnames
        return obj_params

    def _set_all_obj_params(self, objparams: List):
        """Set the object parameters.

        Parameters
        ----------
        objparams: List
            The object parameters to be set

        """
        for (name, param) in zip(self.names, objparams):
            del_attr(
                self.obj, name
            )  # delete required in case the param is not a torch.nn.Parameter
            set_attr(self.obj, name, param)


def _check_identical_objs(objs1: List, objs2: List) -> bool:
    """Check if the two lists of objects are identical.

    Examples
    --------
    >>> l1 = [2, 2, 3]
    >>> l2 = [1, 2, 3]
    >>> _check_identical_objs(l1, l2)
    False

    Parameters
    ----------
    objs1: List
        The first list of objects
    objs2: List
        The second list of objects

    Returns
    -------
    bool
        True if the two lists of objects are identical, False otherwise

    """
    for obj1, obj2 in zip(objs1, objs2):
        if id(obj1) != id(obj2):
            return False
    return True


def get_pure_function(fcn) -> PureFunction:
    """Get the pure function form of the function or method ``fcn``.

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.differentiation_utils import get_pure_function
    >>> def fcn(x, y):
    ...     return x + y
    >>> pfunc = get_pure_function(fcn)
    >>> pfunc(1, 2)
    3

    Parameters
    ----------
    fcn: function or method
        Function or method to be converted into a ``PureFunction`` by exposing
        the hidden parameters affecting its outputs.

    Returns
    -------
    PureFunction
        The pure function wrapper

    """

    errmsg = "The input function must be a function, a method of " \
        "torch.nn.Module, a method of xitorch.EditableModule, or a sibling method"

    if isinstance(fcn, PureFunction):
        return fcn

    elif inspect.isfunction(fcn) or isinstance(fcn, torch.jit.ScriptFunction):
        return FunctionPureFunction(fcn)

    # if it is a method from an object, unroll the parameters and add
    # the object's parameters as well
    elif inspect.ismethod(fcn) or hasattr(fcn, "__call__"):
        if inspect.ismethod(fcn):
            obj = fcn.__self__
        else:
            obj = fcn
            fcn = fcn.__call__

        if isinstance(obj, EditableModule):
            return EditableModulePureFunction(obj, fcn)
        elif isinstance(obj, torch.nn.Module):
            return TorchNNPureFunction(obj, fcn)
        else:
            raise RuntimeError(errmsg)

    else:
        raise RuntimeError(errmsg)

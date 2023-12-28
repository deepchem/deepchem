import torch
import inspect
from typing import Callable, List, Tuple, Union, Sequence
from deepchem.utils.attribute_utils import set_attr, del_attr
from deepchem.utils.differentiation_utils import EditableModule
from deepchem.utils.misc_utils import Uniquifier
from contextlib import contextmanager
from abc import abstractmethod

__all__ = ["get_pure_function", "make_sibling"]


class PureFunction(object):
    """
    PureFunction class wraps methods to make it stateless and expose the pure
    function to take inputs of the original inputs (`params`) and the object's
    states (`objparams`).
    For functions, this class only acts as a thin wrapper.
    
    Restore stack stores list of (objparams, identical) everytime the objparams
    are set, it will store the old objparams and indication if the old and new
    objparams are identical.
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
    def _get_all_obj_params_init(self) -> List:
        return []

    def _set_all_obj_params(self, objparams: List):
        pass

class EditableModulePureFunction(PureFunction):
    def __init__(self, obj: EditableModule, method: Callable):
        self.obj = obj
        self.method = method
        super().__init__(method)

    def _get_all_obj_params_init(self) -> List:
        return list(self.obj.getparams(self.method.__name__))

    def _set_all_obj_params(self, allobjparams: List):
        self.obj.setparams(self.method.__name__, *allobjparams)

class TorchNNPureFunction(PureFunction):
    def __init__(self, obj: torch.nn.Module, method: Callable):
        self.obj = obj
        self.method = method
        super().__init__(method)

    def _get_all_obj_params_init(self) -> List:
        # get the tensors in the torch.nn.Module to be used as params
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
        for (name, param) in zip(self.names, objparams):
            del_attr(self.obj, name)  # delete required in case the param is not a torch.nn.Parameter
            set_attr(self.obj, name, param)

class SingleSiblingPureFunction(PureFunction):
    def __init__(self, fcn: Callable, fcntocall: Callable):
        self.pfunc = get_pure_function(fcn)
        super().__init__(fcntocall)

    def _get_all_obj_params_init(self) -> List:
        return self.pfunc._get_all_obj_params_init()

    def _set_all_obj_params(self, allobjparams: List):
        self.pfunc._set_all_obj_params(allobjparams)

class MultiSiblingPureFunction(PureFunction):
    def __init__(self, fcns: Sequence[Callable], fcntocall: Callable):
        self.pfuncs = [get_pure_function(fcn) for fcn in fcns]
        self.npfuncs = len(self.pfuncs)
        super().__init__(fcntocall)

    def _get_all_obj_params_init(self) -> List:
        res: List[Union[torch.Tensor, torch.nn.Parameter]] = []
        self.cumsum_idx = [0] * (self.npfuncs + 1)
        for i, pfunc in enumerate(self.pfuncs):
            objparams = pfunc._get_all_obj_params_init()
            res = res + objparams
            self.cumsum_idx[i + 1] = self.cumsum_idx[i] + len(objparams)
        return res

    def _set_all_obj_params(self, allobjparams: List):
        for i, pfunc in enumerate(self.pfuncs):
            pfunc._set_all_obj_params(allobjparams[self.cumsum_idx[i]:self.cumsum_idx[i + 1]])

def _check_identical_objs(objs1: List, objs2: List) -> bool:
    for obj1, obj2 in zip(objs1, objs2):
        if id(obj1) != id(obj2):
            return False
    return True

def get_pure_function(fcn) -> PureFunction:
    """
    Get the pure function form of the function or method ``fcn``.

    Arguments
    ---------
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

def make_sibling(*pfuncs) -> Callable[[Callable], PureFunction]:
    """
    Used as a decor to mark the decorated function as a sibling method of the
    input ``pfunc``.
    Sibling method is a method that is virtually belong to the same object, but
    behaves differently.
    Changing the state of the decorated function will also change the state of
    ``pfunc`` and its other siblings.
    """
    if len(pfuncs) == 0:
        raise TypeError("At least 1 function is required as the argument")
    elif len(pfuncs) == 1:
        return lambda fcn: SingleSiblingPureFunction(pfuncs[0], fcntocall=fcn)
    else:
        return lambda fcn: MultiSiblingPureFunction(pfuncs, fcntocall=fcn)

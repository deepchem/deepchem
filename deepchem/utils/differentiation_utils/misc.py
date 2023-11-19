import contextlib
import copy
from typing import Mapping, Callable, Optional, Union, Dict, List
try:
    import torch
except ImportError:
    pass
import numpy as np


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


def assert_type(cond, msg=""):
    if not cond:
        raise TypeError(msg)


def tallqr(V, MV=None):
    # faster QR for tall and skinny matrix
    # V: (*BV, na, nguess)
    # MV: (*BM, na, nguess) where M is the basis to make Q M-orthogonal
    # if MV is None, then MV=V
    if MV is None:
        MV = V
    VTV = torch.matmul(V.transpose(-2, -1), MV)  # (*BMV, nguess, nguess)
    R = torch.linalg.cholesky(VTV.transpose(-2, -1).conj()).transpose(-2, -1).conj()  # (*BMV, nguess, nguess)
    Rinv = torch.inverse(R)  # (*BMV, nguess, nguess)
    Q = torch.matmul(V, Rinv)
    return Q, R


def to_fortran_order(V):
    # V: (...,nrow,ncol)
    # outV: (...,nrow,ncol)

    # check if it is in C-contiguous
    if V.is_contiguous():
        # return V.set_(V.storage(), V.storage_offset(), V.size(), tuple(reversed(V.stride())))
        return V.transpose(-2, -1).contiguous().transpose(-2, -1)
    elif V.transpose(-2, -1).is_contiguous():
        return V
    else:
        raise RuntimeError("Only the last two dimensions can be made Fortran order.")


def get_np_dtype(dtype: torch.dtype):
    # return the corresponding numpy dtype from the input pytorch's tensor dtype
    if dtype == torch.float32:
        return np.float32
    elif dtype == torch.float64:
        return np.float64
    elif dtype == torch.complex64:
        return np.complex64
    elif dtype == torch.complex128:
        return np.complex128
    else:
        raise TypeError("Unknown type: %s" % dtype)

# Warnings
class UnimplementedError(Exception):
    pass

class GetSetParamsError(Exception):
    pass

class ConvergenceWarning(Warning):
    """
    Warning to be raised if the convergence of an algorithm is not achieved
    """
    pass

class MathWarning(Warning):
    """
    Raised if there are mathematical conditions that are not satisfied
    """
    pass


class Uniquifier(object):
    def __init__(self, allobjs: List):
        self.nobjs = len(allobjs)

        id2idx: Dict[int, int] = {}
        unique_objs: List[int] = []
        unique_idxs: List[int] = []
        nonunique_map_idxs: List[int] = [-self.nobjs * 2] * self.nobjs
        num_unique = 0
        for i, obj in enumerate(allobjs):
            id_obj = id(obj)
            if id_obj in id2idx:
                nonunique_map_idxs[i] = id2idx[id_obj]
                continue
            id2idx[id_obj] = num_unique
            unique_objs.append(obj)
            nonunique_map_idxs[i] = num_unique
            unique_idxs.append(i)
            num_unique += 1

        self.unique_objs = unique_objs
        self.unique_idxs = unique_idxs
        self.nonunique_map_idxs = nonunique_map_idxs
        self.num_unique = num_unique
        self.all_unique = self.nobjs == self.num_unique

    def get_unique_objs(self, allobjs: Optional[List] = None) -> List:
        if allobjs is None:
            return self.unique_objs
        assert_runtime(len(allobjs) == self.nobjs, "The allobjs must have %d elements" % self.nobjs)
        if self.all_unique:
            return allobjs
        return [allobjs[i] for i in self.unique_idxs]

    def map_unique_objs(self, uniqueobjs: List) -> List:
        assert_runtime(len(uniqueobjs) == self.num_unique, "The uniqueobjs must have %d elements" % self.num_unique)
        if self.all_unique:
            return uniqueobjs
        return [uniqueobjs[idx] for idx in self.nonunique_map_idxs]

class TensorNonTensorSeparator(object):
    """
    Class that provides function to separate/combine tensors and nontensors
    parameters.
    """

    def __init__(self, params, varonly=True):
        """
        Params is a list of tensor or non-tensor to be splitted into
        tensor/non-tensor
        """
        self.tensor_idxs = []
        self.tensor_params = []
        self.nontensor_idxs = []
        self.nontensor_params = []
        self.nparams = len(params)
        for (i, p) in enumerate(params):
            if isinstance(p, torch.Tensor) and ((varonly and p.requires_grad) or (not varonly)):
                self.tensor_idxs.append(i)
                self.tensor_params.append(p)
            else:
                self.nontensor_idxs.append(i)
                self.nontensor_params.append(p)
        self.alltensors = len(self.tensor_idxs) == self.nparams

    def get_tensor_params(self):
        return self.tensor_params

    def ntensors(self):
        return len(self.tensor_idxs)

    def nnontensors(self):
        return len(self.nontensor_idxs)

    def reconstruct_params(self, tensor_params, nontensor_params=None):
        if nontensor_params is None:
            nontensor_params = self.nontensor_params
        if len(tensor_params) + len(nontensor_params) != self.nparams:
            raise ValueError(
                "The total length of tensor and nontensor params "
                "do not match with the expected length: %d instead of %d" %
                (len(tensor_params) + len(nontensor_params), self.nparams))
        if self.alltensors:
            return tensor_params

        params = [None for _ in range(self.nparams)]
        for nidx, p in zip(self.nontensor_idxs, nontensor_params):
            params[nidx] = p
        for idx, p in zip(self.tensor_idxs, tensor_params):
            params[idx] = p
        return params

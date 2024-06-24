"""
Utilities for miscellaneous tasks.
"""
from __future__ import annotations
import contextlib
from typing import Optional, List, Callable, Dict, Any, Tuple
import warnings
import torch
import numpy as np
import h5py
from typing import Dict, List, Optional, Callable, TypeVar, Any, Mapping
import functools


def indent(s, nspace):
    """Gives indentation of the second line and next lines.
    It is used to format the string representation of an object.
    Which might be containing multiples objects in it.
    Usage: LinearOperator

    Parameters
    ----------
    s: str
        The string to be indented.
    nspace: int
        The number of spaces to be indented.

    Returns
    -------
    str
        The indented string.

    """
    spaces = " " * nspace
    lines = [spaces + c if i > 0 else c for i, c in enumerate(s.split("\n"))]
    return "\n".join(lines)


def shape2str(shape):
    """Convert the shape to string representation.
    It also nicely formats the shape to be readable.

    Parameters
    ----------
    shape: Sequence[int]
        The shape to be converted to string representation.

    Returns
    -------
    str
        The string representation of the shape.

    """
    return "(%s)" % (", ".join([str(s) for s in shape]))


# Warnings
class UnimplementedError(Exception):
    """
    Raised if a method is not implemented.
    """
    pass


class GetSetParamsError(Exception):
    """
    Raised if there is an error in getting or setting parameters.
    """
    pass


class ConvergenceWarning(Warning):
    """
    Warning to be raised if the convergence of an algorithm is not achieved.
    """
    pass


class MathWarning(Warning):
    """
    Raised if there are mathematical conditions that are not satisfied.
    """
    pass


class Uniquifier(object):
    """
    Identifies and tracks unique objects within a list, even if they are
    duplicates based on internal memory addresses (using id()).
    It Optimizes operations involving unique objects by avoiding redundant
    processing.

    Examples
    --------
    >>> from deepchem.utils import Uniquifier
    >>> a = 1
    >>> b = 2
    >>> c = 3
    >>> d = 1
    >>> u = Uniquifier([a, b, c, a, d])
    >>> u.get_unique_objs()
    [1, 2, 3]

    """

    def __init__(self, allobjs: List):
        """Initialize the uniquifier.

        Parameters
        ----------
        allobjs: List
            The list of objects to be uniquified.

        """
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
        """Get the unique objects.

        Parameters
        ----------
        allobjs: Optional[List]
            The list of objects to be uniquified.

        Returns
        -------
        List
            The list of unique objects.

        """
        if allobjs is None:
            return self.unique_objs
        assert len(
            allobjs
        ) == self.nobjs, "The allobjs must have %d elements" % self.nobjs
        if self.all_unique:
            return allobjs
        return [allobjs[i] for i in self.unique_idxs]

    def map_unique_objs(self, uniqueobjs: List) -> List:
        """Map the unique objects to the original objects.

        Parameters
        ----------
        uniqueobjs: List
            The list of unique objects.

        """
        assert len(
            uniqueobjs
        ) == self.num_unique, "The uniqueobjs must have %d elements" % self.num_unique
        if self.all_unique:
            return uniqueobjs
        return [uniqueobjs[idx] for idx in self.nonunique_map_idxs]


T = TypeVar('T')


def memoize_method(fcn: Callable[[Any], T]) -> Callable[[Any], T]:
    """Memoize a method without any arguments using a cache in the object

    Examples
    --------
    >>> class A:
    ...     @memoize_method
    ...     def foo(self):
    ...         print("foo")
    ...         return 1
    >>> a = A()
    >>> a.foo()
    foo
    1
    >>> a.foo()
    1

    Parameters
    ----------
    fcn: callable
        Function to memoize

    Returns
    -------
    callable
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


class Cache(object):
    """
    Internal object to store/load cache for heavy-load calculations.
    For a class to be able to use cache, it has to have cache as its members.
    The same cache object can also be passed to other object either via
        `.add_prefix` method or just passed as it is.
    The list of cacheable parameters are constructed within the class.
    The file to store cache also set within the class via `set` method, if no
        file is set, then the cache does not work.
    To load/store cache, the context handler, `.open()` needs to be called.
    We use the magic method `cache` here to load/store cache, i.e. if the given
        dataset is not in the cache, then it is calculated then stored.
    Otherwise, the dataset is just loaded from the cache file.
    To avoid loading the wrong cache (e.g. the same file name is set for two
        different objects), we provide `check_signature` method which will
        raise a warning if the signature from the cached file and the input of
        `check_signature` do not match.
    """
    def __init__(self):
        """Initialize the cache object."""
        self._cacheable_pnames: List[str] = []
        self._fname: Optional[str] = None
        self._pnames_to_cache: Optional[List[str]] = None
        self._fhandler: Optional[h5py.File] = None

    def set(self, fname: str, pnames: Optional[List[str]] = None):
        """set up the cache

        Parameters
        ----------
        fname : str
            File name to store the cache
        pnames : Optional[List[str]], optional
            List of parameter names to be cached, by default None

        Raises
        ------
        RuntimeError
            If the cache has been set before
        """
        self._fname = fname
        self._pnames_to_cache = pnames

    def cache(self, pname: str, fcn: Callable[[], torch.Tensor]) -> torch.Tensor:
        """cache the result of the function

        Parameters
        ----------
        pname : str
            Parameter name to be cached
        fcn : Callable[[], torch.Tensor]
            Function to be cached

        Returns
        -------
        torch.Tensor
            Result of the function

        """
        # if not has been set, then just calculate and return
        if not self.isset():
            return fcn()

        # if the param is not to be cached, then just calculate and return
        if not self._pname_to_cache(pname):
            return fcn()

        # get the dataset name
        dset_name = self._pname2dsetname(pname)

        # check if dataset name is in the file (cache exist)
        file = self._get_file_handler()
        if dset_name in file:
            # retrieve the dataset if it has been computed before
            res = self._load_dset(dset_name, file)
        else:
            # if not in the file, then compute the tensor and save the cache
            res = fcn()
            self._save_dset(dset_name, file, res)
        return res

    def cache_multi(self, pnames: List[str], fcn: Callable[[], Tuple[torch.Tensor, ...]]) \
            -> Tuple[torch.Tensor, ...]:
        """Cache the result of the function for multiple parameters

        Parameters
        ----------
        pnames : List[str]
            List of parameter names to be cached
        fcn : Callable[[], Tuple[torch.Tensor, ...]]
            Function to be cached

        Returns
        -------
        Tuple[torch.Tensor, ...]
            Tuple of results of the function

        """
        if not self.isset():
            return fcn()

        # if any of the pnames not to be cached, then just calculate and return
        if not all([self._pname_to_cache(pname) for pname in pnames]):
            return fcn()

        # get the dataset names
        dset_names = [self._pname2dsetname(pname) for pname in pnames]

        # get the file handler
        file = self._get_file_handler()

        # check if all the dataset is in the cache file, otherwise, just evaluate
        all_dset_names_in_file = all([dset_name in file for dset_name in dset_names])
        if all_dset_names_in_file:
            all_res = tuple(self._load_dset(dset_name, file) for dset_name in dset_names)
        else:
            all_res = fcn()
            for dset_name, res in zip(dset_names, all_res):
                self._save_dset(dset_name, file, res)
        return all_res

    @contextlib.contextmanager
    def open(self):
        """Open the cache file

        Yields
        -------
        h5py.File
            File handler

        """
        try:
            if self._fname is not None:
                self._fhandler = h5py.File(self._fname, "a")
            yield self._fhandler
        finally:
            if self._fname is not None:
                self._fhandler.close()
                self._fhandler = None

    def add_prefix(self, prefix: str) -> Cache:
        """Return a Cache object that will add the prefix for every input of
        parameter names

        Parameters
        ----------
        prefix : str
            Prefix to be added to the parameter names

        Returns
        -------
        Cache
            Cache object with the prefix added to the parameter names

        """
        prefix = _normalize_prefix(prefix)
        return _PrefixedCache(self, prefix)

    def add_cacheable_params(self, pnames: List[str]):
        """Add the cacheable parameter names

        Parameters
        ----------
        pnames : List[str]
            List of parameter names to be cached

        """
        self._cacheable_pnames.extend(pnames)

    def get_cacheable_params(self) -> List[str]:
        """return the cacheable parameter names

        Returns
        -------
        List[str]
            List of cacheable parameter names

        """
        return self._cacheable_pnames

    def check_signature(self, sig_dict: Dict[str, Any], _groupname: Optional[str] = "/"):
        """Executed while the file is opened, if there is no signature, then add
        the signature, if there is a signature in the file, then match it.
        If they do not match, print a warning, otherwise, do nothing
        _groupname is internal parameter within this class, should not be
         specified by classes other than Cache and its inheritances.

        Parameters
        ----------
        sig_dict : Dict[str, Any]
            Dictionary of signature
        _groupname : Optional[str] (default "/")
            Group name in the h5py file

        """
        if not self.isset():
            return

        fh = self._get_file_handler()
        sig_attrname = "signature"

        # get the group
        if _groupname in fh:
            group = fh[_groupname]
        else:
            group = fh.create_group(_groupname)

        # create the signature string
        sig_str = "\n\n".join(["%s:\n%s" % (k, str(v)) for (k, v) in sig_dict.items()])

        # if the signature does not exist, then write the signature as an attribute
        if sig_attrname not in group.attrs:
            group.attrs[sig_attrname] = str(sig_str)

        # if the signature exist, then check it, if it does not match, raise a warning
        else:
            cached_sig = str(group.attrs[sig_attrname])
            if cached_sig != sig_str:
                msg = ("Mismatch of the cached signature.\nCached signature:\n%s\n"
                       "-----------------------\n"
                       "Current signature:\n%s" % (cached_sig, sig_str))
                warnings.warn(msg)

        group.attrs["signature"] = str(sig_str)

    @staticmethod
    def get_dummy() -> Cache:
        """Returns a dummy cache that does not do anything

        Returns
        -------
        Cache
            Dummy cache object

        """
        return _DummyCache()

    def _pname_to_cache(self, pname: str) -> bool:
        """Check if the input parameter name is to be cached

        Parameters
        ----------
        pname : str
            Parameter name to be checked

        Returns
        -------
        bool
            Indicator whether the parameter name is to be cached

        """
        return (self._pnames_to_cache is None) or (pname in self._pnames_to_cache)

    def _pname2dsetname(self, pname: str) -> str:
        """Convert the parameter name to dataset name

        Parameters
        ----------
        pname : str
            Parameter name to be converted to dataset name

        Returns
        -------
        str
            Dataset name

        """
        return pname.replace(".", "/")

    def _get_file_handler(self) -> h5py.File:
        """Return the file handler, if the file is not opened yet,
        then raise an error

        Returns
        -------
        h5py.File
            File handler

        """
        if self._fhandler is None:
            msg = "The cache file has not been opened yet, please use .open() before reading/writing to the cache"
            raise RuntimeError(msg)
        else:
            return self._fhandler

    def isset(self) -> bool:
        """Returns the indicator whether the cache object has been set

        Returns
        -------
        bool
            Indicator whether the cache object has been set

        """
        return self._fname is not None

    def _load_dset(self, dset_name: str, fhandler: h5py.File) -> torch.Tensor:
        """Load the dataset from the file handler (check is performed outside)

        Parameters
        ----------
        dset_name : str
            Dataset name
        fhandler : h5py.File
            File handler for the dataset

        """
        dset_np = np.asarray(fhandler[dset_name])
        dset = torch.as_tensor(dset_np)
        return dset

    def _save_dset(self, dset_name: str, fhandler: h5py.File, dset: torch.Tensor):
        """Save res to the h5py in the dataset name

        Parameters
        ----------
        dset_name : str
            Dataset name
        fhandler : h5py.File
            File handler for the dataset
        dset : torch.Tensor
            Tensor to be saved

        """
        fhandler[dset_name] = dset.detach()

class _PrefixedCache(Cache):
    """This class adds a prefix to every parameter names input"""
    def __init__(self, obj: Cache, prefix: str):
        """Initialize the PrefixedCache object

        Parameters
        ----------
        obj : Cache
            Cache object to be prefixed
        prefix : str
            Prefix to be added to the parameter names

        """
        self._obj = obj
        self._prefix = prefix

    def set(self, fname: str, pnames: Optional[List[str]] = None):
        """set must only be done in the parent object, not in the children objects

        Parameters
        ----------
        fname : str
            File name to store the cache
        pnames : Optional[List[str]] (default None)
            List of parameter names to be cached, by default None

        """
        raise RuntimeError("Cache.set() must be done on non-prefixed cache")

    def cache(self, pname: str, fcn: Callable[[], torch.Tensor]) -> torch.Tensor:
        """Cache the result of the function

        Parameters
        ----------
        pname : str
            Parameter name to be cached
        fcn : Callable[[], torch.Tensor]
            Function to be cached

        Returns
        -------
        torch.Tensor
            Result of the function

        """
        return self._obj.cache(self._prefixed(pname), fcn)

    def cache_multi(self, pnames: List[str], fcn: Callable[[], Tuple[torch.Tensor, ...]]) \
            -> Tuple[torch.Tensor, ...]:
        """Cache the result of the function for multiple parameters

        Parameters
        ----------
        pnames : List[str]
            List of parameter names to be cached
        fcn : Callable[[], Tuple[torch.Tensor, ...]]
            Function to be cached

        Returns"""
        ppnames = [self._prefixed(pname) for pname in pnames]
        return self._obj.cache_multi(ppnames, fcn)

    @contextlib.contextmanager
    def open(self):
        """Open the cache file

        Yields
        -------
        h5py.File
            File handler

        """
        with self._obj.open() as f:
            try:
                yield f
            finally:
                pass

    def add_prefix(self, prefix: str) -> Cache:
        """Return a deeper prefixed object

        Parameters
        ----------
        prefix : str
            Prefix to be added to the parameter names

        Returns
        -------
        Cache
            Cache object with the prefix added to the parameter names

        """
        prefix = self._prefixed(_normalize_prefix(prefix))
        return self._obj.add_prefix(prefix)

    def add_cacheable_params(self, pnames: List[str]):
        """Add the cacheable parameter names

        Parameters
        ----------
        pnames : List[str]
            List of parameter names to be cached

        """
        pnames = [self._prefixed(pname) for pname in pnames]
        self._obj.add_cacheable_params(pnames)

    def get_cacheable_params(self) -> List[str]:
        """This can only be done on the root cache (non-prefixed) to avoid
        confusion about which name should be provided (absolute or relative)

        Returns
        -------
        List[str]
            List of cacheable parameter names

        Raises
        ------
        RuntimeError
            If the method is called on the prefixed cache

        """
        raise RuntimeError("Cache.get_cacheable_params() must be done on non-prefixed cache")

    def check_signature(self, sig_dict: Dict[str, Any], _groupname: Optional[str] = "/"):
        """use the prefix as the groupname to do signature check of the root object

        Parameters
        ----------
        sig_dict : Dict[str, Any]
            Dictionary of signature
        _groupname : Optional[str] (default "/")
            Group name in the h5py file

        """
        if not self.isset():
            return

        groupname = "/" + self._prefix.replace(".", "/")
        if groupname.endswith("/"):
            groupname = groupname[:-1]

        self._obj.check_signature(sig_dict, _groupname=groupname)

    def isset(self) -> bool:
        """Returns the indicator whether the cache object has been set

        Returns
        -------
        bool
            Indicator whether the cache object has been set

        """
        return self._obj.isset()

    def _prefixed(self, pname: str) -> str:
        """Returns the prefixed name

        Parameters
        ----------
        pname : str
            Parameter name to be prefixed

        Returns
        -------
        str
            Prefixed parameter name

        """
        return self._prefix + pname

class _DummyCache(Cache):
    """This class just an interface of cache without doing anything"""
    def __init__(self):
        """Initialize the DummyCache object."""
        pass

    def set(self, fname: str, pnames: Optional[List[str]] = None):
        """set up the cache

        Parameters
        ----------
        fname : str
            File name to store the cache
        pnames : Optional[List[str]], optional
            List of parameter names to be cached, by default None

        """
        pass

    def cache(self, pname: str, fcn: Callable[[], torch.Tensor]) -> torch.Tensor:
        """cache the result of the function

        Parameters
        ----------
        pname : str
            Parameter name to be cached
        fcn : Callable[[], torch.Tensor]
            Function to be cached

        Returns
        -------
        torch.Tensor
            Result of the function

        """
        return fcn()

    def cache_multi(self, pnames: List[str], fcn: Callable[[], Tuple[torch.Tensor, ...]]) \
            -> Tuple[torch.Tensor, ...]:
        """Cache the result of the function for multiple parameters

        Parameters
        ----------
        pnames : List[str]
            List of parameter names to be cached
        fcn : Callable[[], Tuple[torch.Tensor, ...]]
            Function to be cached

        Returns
        -------
        Tuple[torch.Tensor, ...]
            Tuple of results of the function

        """
        return fcn()

    @contextlib.contextmanager
    def open(self):
        """Open the cache file

        Yields
        -------
        h5py.File
            File handler

        """
        try:
            yield None
        finally:
            pass

    def add_prefix(self, prefix: str) -> Cache:
        """Return a Cache object that will add the prefix for every input of
        parameter names

        Parameters
        ----------
        prefix : str
            Prefix to be added to the parameter names

        Returns
        -------
        Cache
            Cache object with the prefix added to the parameter names

        """
        return self

    def add_cacheable_params(self, pnames: List[str]):
        """Add the cacheable parameter names

        Parameters
        ----------
        pnames : List[str]
            List of parameter names to be cached

        """
        pass

    def get_cacheable_params(self) -> List[str]:
        """return the cacheable parameter names

        Returns
        -------
        List[str]
            List of cacheable parameter names

        """
        return []

    def check_signature(self, sig_dict: Dict[str, Any], _groupname: Optional[str] = "/"):
        """Executed while the file is opened, if there is no signature, then add
        the signature, if there is a signature in the file, then match it.
        If they do not match, print a warning, otherwise, do nothing
        _groupname is internal parameter within this class, should not be
         specified by classes other than Cache and its inheritances.

        Parameters
        ----------
        sig_dict : Dict[str, Any]
            Dictionary of signature
        _groupname : Optional[str] (default "/")
            Group name in the h5py file

        """
        pass

    def isset(self):
        """Returns the indicator whether the cache object has been set

        Returns
        -------
        bool
            Indicator whether the cache object has been set

        """
        return False

def _normalize_prefix(prefix: str) -> str:
    """Added a dot at the end of prefix if it is not so.

    Parameters
    ----------
    prefix : str
        Prefix to be normalized

    Returns
    -------
    str
        Normalized prefix

    """
    if not prefix.endswith("."):
        prefix = prefix + "."
    return prefix


T = TypeVar('T')
K = TypeVar('K')


def get_option(name: str, s: K, options: Mapping[K, T]) -> T:
    """Get the value from dictionary of options, if not found, then raise an error

    Parameters
    ----------
    name : str
        Name of the option
    s : K
        Key to be searched
    options : Mapping[K, T]
        Dictionary of options

    Returns
    -------
    T
        Value of the option

    """
    if s in options:
        return options[s]
    else:
        raise ValueError(f"Unknown {name}: {s}. The available options are: {str(list(options.keys()))}")


def estimate_ovlp_rcut(precision: float, coeffs: torch.Tensor, alphas: torch.Tensor) -> float:
    """estimate the rcut for lattice sum to achieve the given precision
    it is estimated based on the overlap integral
    
    Parameters
    ----------
    precision : float
        Precision to be achieved
    coeffs : torch.Tensor
        Coefficients of the basis functions
    alphas : torch.Tensor
        Alpha values of the basis functions

    Returns
    -------
    float
        Estimated rcut

    """
    langmom = 1
    C = (coeffs * coeffs + 1e-200) * (2 * langmom + 1) * alphas / precision
    r0 = torch.tensor(20.0, dtype=coeffs.dtype, device=coeffs.device)
    for i in range(2):
        r0 = torch.sqrt(2.0 * torch.log(C * (r0 * r0 * alphas) ** (langmom + 1) + 1.) / alphas)
    rcut = float(torch.max(r0).detach())
    return rcut

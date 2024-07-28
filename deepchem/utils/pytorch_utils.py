"""Utility functions for working with PyTorch."""
import contextlib
import warnings
import scipy
import torch
from typing import Any, Callable, Dict, Optional, Sequence, Union, List, Generator, Tuple
import numpy as np
try:
    import h5py
except Exception:
    warnings.warn("h5py is not installed, cache will not work.")


def get_activation(fn: Union[Callable, str]):
    """Get a PyTorch activation function, specified either directly or as a string.

    This function simplifies allowing users to specify activation functions by name.
    If a function is provided, it is simply returned unchanged.  If a string is provided,
    the corresponding function in torch.nn.functional is returned.
    """
    if isinstance(fn, str):
        return getattr(torch.nn.functional, fn)
    return fn


def unsorted_segment_sum(data: torch.Tensor, segment_ids: torch.Tensor,
                         num_segments: int) -> torch.Tensor:
    """Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

    Parameters
    ----------
    data: torch.Tensor
        A tensor whose segments are to be summed.
    segment_ids: torch.Tensor
        The segment indices tensor.
    num_segments: int
        The number of segments.

    Returns
    -------
    tensor: torch.Tensor

    Examples
    --------
    >>> segment_ids = torch.Tensor([0, 1, 0]).to(torch.int64)
    >>> data = torch.Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [4, 3, 2, 1]])
    >>> num_segments = 2
    >>> result = unsorted_segment_sum(data=data,
    ...                               segment_ids=segment_ids,
    ...                               num_segments=num_segments)
    >>> data.shape[0]
    3
    >>> segment_ids.shape[0]
    3
    >>> len(segment_ids.shape)
    1
    >>> result
    tensor([[5., 5., 5., 5.],
            [5., 6., 7., 8.]])

    """

    if len(segment_ids.shape) != 1:
        raise AssertionError("segment_ids have be a 1-D tensor")

    if data.shape[0] != segment_ids.shape[0]:
        raise AssertionError(
            "segment_ids should be the same size as dimension 0 of input.")

    s = torch.prod(torch.tensor(data.shape[1:])).long()
    segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0],
                                                        *data.shape[1:])

    # data.shape and segment_ids.shape should be equal
    assert data.shape == segment_ids.shape
    shape: List[int] = [num_segments] + list(data.shape[1:])
    tensor: torch.Tensor = torch.zeros(*shape).scatter_add(
        0, segment_ids, data.float())
    tensor = tensor.type(data.dtype)
    return tensor


def segment_sum(data: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
    """ This function computes the sum of values along segments within a tensor. It is useful when you have a tensor with segment IDs and you want to compute the sum of values for each segment.
    This function is analogous to tf.segment_sum. (https://www.tensorflow.org/api_docs/python/tf/math/segment_sum).

    Parameters
    ----------
    data: torch.Tensor
        A pytorch tensor containing the values to be summed. It can have any shape, but its rank (number of dimensions) should be at least 1.
    segment_ids: torch.Tensor
        A 1-D tensor containing the indices for the segmentation. The segments can be any non-negative integer values, but they must be sorted in non-decreasing order.

    Returns
    -------
    out_tensor: torch.Tensor
        Tensor with the same shape as data, where each value corresponds to the sum of values within the corresponding segment.

    Examples
    --------
    >>> data = torch.Tensor([[1, 2, 3, 4], [4, 3, 2, 1], [5, 6, 7, 8]])
    >>> segment_ids = torch.Tensor([0, 0, 1]).to(torch.int64)
    >>> result = segment_sum(data=data, segment_ids=segment_ids)
    >>> data.shape[0]
    3
    >>> segment_ids.shape[0]
    3
    >>> len(segment_ids.shape)
    1
    >>> result
    tensor([[5., 5., 5., 5.],
            [5., 6., 7., 8.]])

    """
    if not all(segment_ids[i] <= segment_ids[i + 1]
               for i in range(len(segment_ids) - 1)):
        raise AssertionError("elements of segment_ids must be sorted")

    if len(segment_ids.shape) != 1:
        raise AssertionError("segment_ids have be a 1-D tensor")

    if data.shape[0] != segment_ids.shape[0]:
        raise AssertionError(
            "segment_ids should be the same size as dimension 0 of input.")

    num_segments = len(torch.unique(segment_ids))
    out_tensor = unsorted_segment_sum(data, segment_ids, num_segments)

    return out_tensor


def chunkify(a: torch.Tensor, dim: int, maxnumel: int) -> \
        Generator[Tuple[torch.Tensor, int, int], None, None]:
    """Splits the tensor `a` into several chunks of size `maxnumel` along the
    dimension given by `dim`.

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.pytorch_utils import chunkify
    >>> a = torch.arange(10)
    >>> for chunk, istart, iend in chunkify(a, 0, 3):
    ...     print(chunk, istart, iend)
    tensor([0, 1, 2]) 0 3
    tensor([3, 4, 5]) 3 6
    tensor([6, 7, 8]) 6 9
    tensor([9]) 9 12

    Parameters
    ----------
    a: torch.Tensor
        The big tensor to be splitted into chunks.
    dim: int
        The dimension where the tensor would be splitted.
    maxnumel: int
        Maximum number of elements in a chunk.

    Returns
    -------
    chunks: Generator[Tuple[torch.Tensor, int, int], None, None]
        A generator that yields a tuple of three elements: the chunk tensor, the
        starting index of the chunk and the ending index of the chunk.

    """
    dim = a.ndim + dim if dim < 0 else dim

    numel = a.numel()
    dimnumel = a.shape[dim]
    nondimnumel = numel // dimnumel
    if maxnumel < nondimnumel:
        msg = "Cannot split the tensor of shape %s along dimension %s with maxnumel %d" % \
              (a.shape, dim, maxnumel)
        raise RuntimeError(msg)

    csize = min(maxnumel // nondimnumel, dimnumel)
    ioffset = 0
    lslice = (slice(None, None, None),) * dim
    rslice = (slice(None, None, None),) * (a.ndim - dim - 1)
    while ioffset < dimnumel:
        iend = ioffset + csize
        chunks = a[(lslice + (slice(ioffset, iend, None),) +
                    rslice)], ioffset, iend
        yield chunks
        ioffset = iend


def get_memory(a: torch.Tensor) -> int:
    """Returns the size of the tensor in bytes.

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.pytorch_utils import get_memory
    >>> a = torch.randn(100, 100, dtype=torch.float64)
    >>> get_memory(a)
    80000

    Parameters
    ----------
    a: torch.Tensor
        The tensor to be measured.

    Returns
    -------
    size: int
        The size of the tensor in bytes.

    """
    size = a.numel() * a.element_size()
    return size


def gaussian_integral(
        n: int, alpha: Union[float,
                             torch.Tensor]) -> Union[float, torch.Tensor]:
    """Performs the gaussian integration.

    Examples
    --------
    >>> gaussian_integral(5, 1.0)
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


class TensorNonTensorSeparator(object):
    """
    Class that provides function to separate/combine tensors and nontensors
    parameters.

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.pytorch_utils import TensorNonTensorSeparator
    >>> a = torch.tensor([1.,2,3])
    >>> b = 4.
    >>> c = torch.tensor([5.,6,7], requires_grad=True)
    >>> params = [a, b, c]
    >>> separator = TensorNonTensorSeparator(params)
    >>> tensor_params = separator.get_tensor_params()
    >>> tensor_params
    [tensor([5., 6., 7.], requires_grad=True)]

    """

    def __init__(self, params: Sequence, varonly: bool = True):
        """Initialize the TensorNonTensorSeparator.

        Parameters
        ----------
        params: Sequence
            A list of tensor or non-tensor parameters.
        varonly: bool
            If True, only tensor parameters with requires_grad=True will be
            returned. Otherwise, all tensor parameters will be returned.

        """
        self.tensor_idxs = []
        self.tensor_params = []
        self.nontensor_idxs = []
        self.nontensor_params = []
        self.nparams = len(params)
        for (i, p) in enumerate(params):
            if isinstance(p, torch.Tensor) and ((varonly and p.requires_grad) or
                                                (not varonly)):
                self.tensor_idxs.append(i)
                self.tensor_params.append(p)
            else:
                self.nontensor_idxs.append(i)
                self.nontensor_params.append(p)
        self.alltensors = len(self.tensor_idxs) == self.nparams

    def get_tensor_params(self):
        """Returns a list of tensor parameters.

        Returns
        -------
        List[torch.Tensor]
            A list of tensor parameters.

        """
        return self.tensor_params

    def ntensors(self):
        """Returns the number of tensor parameters.

        Returns
        -------
        int
            The number of tensor parameters.

        """
        return len(self.tensor_idxs)

    def nnontensors(self):
        """Returns the number of nontensor parameters.

        Returns
        -------
        int
            The number of nontensor parameters.

        """
        return len(self.nontensor_idxs)

    def reconstruct_params(self, tensor_params, nontensor_params=None):
        """Reconstruct the parameters from tensor and nontensor parameters.

        Parameters
        ----------
        tensor_params: List[torch.Tensor]
            A list of tensor parameters.
        nontensor_params: Optional[List]
            A list of nontensor parameters. If None, the original nontensor
            parameters will be used.

        Returns
        -------
        List
            A list of parameters.

        """
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


def tallqr(V, MV=None):
    """QR decomposition for tall and skinny matrix.

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.pytorch_utils import tallqr
    >>> V = torch.randn(3, 2)
    >>> Q, R = tallqr(V)
    >>> Q.shape
    torch.Size([3, 2])
    >>> R.shape
    torch.Size([2, 2])
    >>> torch.allclose(Q @ R, V)
    True

    Parameters
    ----------
    V: torch.Tensor
        V is a matrix to be decomposed. (`*BV`, na, nguess)
    MV: torch.Tensor
        (`*BM`, na, nguess) where M is the basis to make Q M-orthogonal
        if MV is None, then MV=V (default=None)

    Returns
    -------
    Q: torch.Tensor
        The Orthogonal Part. Shape: (`*BV`, na, nguess)
    R: torch.Tensor
        The (`*BM`, nguess, nguess) where M is the basis to make Q M-orthogonal

    """
    if MV is None:
        MV = V
    VTV = torch.matmul(V.transpose(-2, -1), MV)  # (*BMV, nguess, nguess)
    R = torch.linalg.cholesky(VTV.transpose(-2, -1).conj()).transpose(
        -2, -1).conj()  # (*BMV, nguess, nguess)
    Rinv = torch.inverse(R)  # (*BMV, nguess, nguess)
    Q = torch.matmul(V, Rinv)
    return Q, R


def to_fortran_order(V):
    """Convert a tensor to Fortran order. (The last two dimensions are made Fortran order.)
    Fortran order/ array is a special case in which all elements of an array are stored in
    column-major order.

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.pytorch_utils import to_fortran_order
    >>> V = torch.randn(3, 2)
    >>> V.is_contiguous()
    True
    >>> V = to_fortran_order(V)
    >>> V.is_contiguous()
    False
    >>> V.shape
    torch.Size([3, 2])
    >>> V = torch.randn(3, 2).transpose(-2, -1)
    >>> V.is_contiguous()
    False
    >>> V = to_fortran_order(V)
    >>> V.is_contiguous()
    False
    >>> V.shape
    torch.Size([2, 3])

    Parameters
    ----------
    V: torch.Tensor
        V is a matrix to be converted. (`*BV`, na, nguess)

    Returns
    -------
    outV: torch.Tensor
        (`*BV`, nguess, na)

    """
    if V.is_contiguous():
        # return V.set_(V.storage(), V.storage_offset(), V.size(), tuple(reversed(V.stride())))
        return V.transpose(-2, -1).contiguous().transpose(-2, -1)
    elif V.transpose(-2, -1).is_contiguous():
        return V
    else:
        raise RuntimeError(
            "Only the last two dimensions can be made Fortran order.")


def get_np_dtype(dtype: torch.dtype) -> Any:
    """corresponding numpy dtype from the input pytorch's tensor dtype

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.pytorch_utils import get_np_dtype
    >>> get_np_dtype(torch.float32)
    <class 'numpy.float32'>
    >>> get_np_dtype(torch.float64)
    <class 'numpy.float64'>

    Parameters
    ----------
    dtype: torch.dtype
        pytorch's tensor dtype

    Returns
    -------
    np.dtype
        corresponding numpy dtype

    """
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


def unsorted_segment_max(data: torch.Tensor, segment_ids: torch.Tensor,
                         num_segments: int) -> torch.Tensor:
    """Computes the maximum along segments of a tensor. Analogous to tf.unsorted_segment_max.

    Parameters
    ----------
    data: torch.Tensor
        A tensor whose segments are to be maximized.
    segment_ids: torch.Tensor
        The segment indices tensor.
    num_segments: int
        The number of segments.

    Returns
    -------
    tensor: torch.Tensor

    Examples
    --------
    >>> segment_ids = torch.Tensor([0, 1, 0]).to(torch.int64)
    >>> data = torch.Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [4, 3, 2, 1]])
    >>> num_segments = 2
    >>> result = unsorted_segment_max(data=data,
    ...                               segment_ids=segment_ids,
    ...                               num_segments=num_segments)
    >>> data.shape[0]
    3
    >>> segment_ids.shape[0]
    3
    >>> len(segment_ids.shape)
    1
    >>> result
    tensor([[4., 3., 3., 4.],
            [5., 6., 7., 8.]])

    """
    if len(segment_ids.shape) != 1:
        raise AssertionError("segment_ids have to be a 1-D tensor")

    if data.shape[0] != segment_ids.shape[0]:
        raise AssertionError(
            "segment_ids should be the same size as dimension 0 of input.")

    # Initialize the tensor to hold the maximum values for each segment
    shape = [num_segments] + list(data.shape[1:])
    tensor = torch.full(shape, float('-inf'), dtype=data.dtype)

    # Create an expanded segment_ids tensor to match data shape
    expanded_segment_ids = segment_ids.unsqueeze(-1).expand(-1, *data.shape[1:])

    # Update the maximum values for each segment
    for i in range(num_segments):
        mask = expanded_segment_ids == i
        tensor[i] = torch.max(data.masked_fill(~mask, float('-inf')), dim=0)[0]

    return tensor


def estimate_ovlp_rcut(precision: float, coeffs: torch.Tensor,
                       alphas: torch.Tensor) -> float:
    """Estimate the rcut for lattice sum to achieve the given precision
    it is estimated based on the overlap integral

    Examples
    --------
    >>> from deepchem.utils import estimate_ovlp_rcut
    >>> precision = 1e-6
    >>> coeffs = torch.tensor([1.0, 2.0, 3.0])
    >>> alphas = torch.tensor([1.0, 2.0, 3.0])
    >>> estimate_ovlp_rcut(precision, coeffs, alphas)
    6.7652716636657715

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
        r0 = torch.sqrt(
            2.0 * torch.log(C *
                            (r0 * r0 * alphas)**(langmom + 1) + 1.) / alphas)
    rcut = float(torch.max(r0).detach())
    return rcut


def normalize_prefix(prefix: str) -> str:
    """Added a dot at the end of prefix if it is not so.

    Examples
    --------
    >>> _normalize_prefix("prefix")
    'prefix.'
    >>> _normalize_prefix("prefix.")
    'prefix.'

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

    Examples
    --------
    >>> class A:
    ...     def __init__(self):
    ...         self.cache = Cache.get_dummy()
    ...     def foo(self, x):
    ...         return self.cache.cache("foo", lambda: x * x)
    >>> a = A()
    >>> a.foo(2)
    4
    >>> a.foo(2)
    4

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

    def cache(self, pname: str, fcn: Callable[[],
                                              torch.Tensor]) -> torch.Tensor:
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
        all_dset_names_in_file = all(
            [dset_name in file for dset_name in dset_names])
        if all_dset_names_in_file:
            all_res = tuple(
                self._load_dset(dset_name, file) for dset_name in dset_names)
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

    def add_prefix(self, prefix: str) -> 'Cache':
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
        prefix = normalize_prefix(prefix)
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

    def check_signature(self,
                        sig_dict: Dict[str, Any],
                        _groupname: Optional[str] = "/"):
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
        sig_str = "\n\n".join(
            ["%s:\n%s" % (k, str(v)) for (k, v) in sig_dict.items()])

        # if the signature does not exist, then write the signature as an attribute
        if sig_attrname not in group.attrs:
            group.attrs[sig_attrname] = str(sig_str)

        # if the signature exist, then check it, if it does not match, raise a warning
        else:
            cached_sig = str(group.attrs[sig_attrname])
            if cached_sig != sig_str:
                msg = (
                    "Mismatch of the cached signature.\nCached signature:\n%s\n"
                    "-----------------------\n"
                    "Current signature:\n%s" % (cached_sig, sig_str))
                warnings.warn(msg)

        group.attrs["signature"] = str(sig_str)

    @staticmethod
    def get_dummy() -> 'Cache':
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
        return (self._pnames_to_cache is None) or (pname
                                                   in self._pnames_to_cache)

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

    def _save_dset(self, dset_name: str, fhandler: h5py.File,
                   dset: torch.Tensor):
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
    """This class adds a prefix to every parameter names input

    Examples
    --------
    >>> cache = Cache.get_dummy()
    >>> pcache = _PrefixedCache(cache, "prefix.")
    >>> pcache.cache("foo", lambda: 1)
    1
    >>> pcache.cache("foo", lambda: 2)
    2

    """

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

    def cache(self, pname: str, fcn: Callable[[],
                                              torch.Tensor]) -> torch.Tensor:
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
        prefix = self._prefixed(normalize_prefix(prefix))
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
        raise RuntimeError(
            "Cache.get_cacheable_params() must be done on non-prefixed cache")

    def check_signature(self,
                        sig_dict: Dict[str, Any],
                        _groupname: Optional[str] = "/"):
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
    """This class just an interface of cache without doing anything.

    Examples
    --------
    >>> cache = _DummyCache()
    >>> cache.cache("foo", lambda: 1)
    1
    >>> cache.cache("foo", lambda: 2)
    2

    """

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

    def cache(self, pname: str, fcn: Callable[[],
                                              torch.Tensor]) -> torch.Tensor:
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

    def check_signature(self,
                        sig_dict: Dict[str, Any],
                        _groupname: Optional[str] = "/"):
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


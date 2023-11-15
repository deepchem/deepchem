"""Utility functions for working with PyTorch."""

import torch
from typing import Callable, Union, List, Generator, Tuple


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

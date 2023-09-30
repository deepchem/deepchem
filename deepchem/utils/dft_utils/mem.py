"""
Density functional theory Memory utilities.
"""
from typing import Generator, Tuple

import torch

__all__ = ["chunkify", "get_memory", "get_dtype_memsize"]


def chunkify(a: torch.Tensor, dim: int, maxnumel: int) -> \
        Generator[Tuple[torch.Tensor, int, int], None, None]:
    """Splits the tensor `a` into several chunks of size `maxnumel` along the
    dimension given by `dim`.

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.dft_utils.mem import chunkify
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
    >>> from deepchem.utils.dft_utils.mem import get_memory
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
    size = a.numel() * get_dtype_memsize(a)
    return size


def get_dtype_memsize(a: torch.Tensor) -> int:
    """Returns the size of each element in the tensor in bytes.

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.dft_utils.mem import get_dtype_memsize
    >>> a = torch.randn(100, 100, dtype=torch.float64)
    >>> get_dtype_memsize(a)
    8

    Parameters
    ----------
    a: torch.Tensor
        The tensor to be measured.

    Returns
    -------
    size: int
        The size of each element in the tensor in bytes.

    """
    if a.dtype == torch.float64 or a.dtype == torch.int64:
        size = 8
    elif a.dtype == torch.float32 or a.dtype == torch.int32:
        size = 4
    elif a.dtype == torch.bool:
        size = 1
    else:
        raise TypeError("Unknown tensor type: %s" % a.dtype)
    return size

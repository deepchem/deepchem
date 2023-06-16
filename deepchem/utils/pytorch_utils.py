"""Utility functions for working with PyTorch."""

import torch
from typing import Callable, Union, List


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
                                  segment_ids=segment_ids,
                                  num_segments=num_segments)
    >>> result
    tensor([[5., 5., 5., 5.],
        [5., 6., 7., 8.]])

    """
    # length of segment_ids.shape should be 1
    assert len(segment_ids.shape) == 1

    # Shape of segment_ids should be equal to first dimension of data
    assert segment_ids.shape[-1] == data.shape[0]

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

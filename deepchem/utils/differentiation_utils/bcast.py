from typing import Tuple
import torch


# Broadcast Utilities
def normalize_bcast_dims(*shapes):
    """
    Normalize the lengths of the input shapes to have the same length.
    The shapes are padded at the front by 1 to make the lengths equal.

    Examples
    --------
    >>> normalize_bcast_dims([1, 2, 3], [2, 3])
    [[1, 2, 3], [1, 2, 3]]

    Parameters
    ----------
    shapes: List[List[int]]
        The shapes to normalize.

    Returns
    -------
    List[List[int]]
        The normalized shapes.

    """
    maxlens = max([len(shape) for shape in shapes])
    res = [[1] * (maxlens - len(shape)) + list(shape) for shape in shapes]
    return res


def get_bcasted_dims(*shapes):
    """Return the broadcasted shape of the given shapes.

    Examples
    --------
    >>> get_bcasted_dims([1, 2, 5], [2, 3, 4])
    [2, 3, 5]

    Parameters
    ----------
    shapes: List[List[int]]
        The shapes to broadcast.

    Returns
    -------
    List[int]
        The broadcasted shape.

    """
    shapes = normalize_bcast_dims(*shapes)
    return [max(*a) for a in zip(*shapes)]


def match_dim(*xs: torch.Tensor,
              contiguous: bool = False) -> Tuple[torch.Tensor, ...]:
    """match the N-1 dimensions of x and xq for searchsorted and gather with dim=-1

    Examples
    --------
    >>> x = torch.randn(10, 5)
    >>> xq = torch.randn(10, 3)
    >>> x_new, xq_new = match_dim(x, xq)
    >>> x_new.shape
    torch.Size([10, 5])
    >>> xq_new.shape
    torch.Size([10, 3])

    """
    orig_shapes = tuple(x.shape[:-1] for x in xs)
    shape = tuple(get_bcasted_dims(*orig_shapes))
    xs_new = tuple(x.expand(shape + (x.shape[-1],)) for x in xs)
    if contiguous:
        xs_new = tuple(x.contiguous() for x in xs_new)
    return xs_new

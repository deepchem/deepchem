"""
Utilities for miscellaneous tasks.
"""


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

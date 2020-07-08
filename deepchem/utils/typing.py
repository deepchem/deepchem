"""Type annotations that are widely used in DeepChem"""

from typing import Sequence, Tuple, TypeVar, Union

T = TypeVar("T")
OneOrMany = Union[T, Sequence[T]]
Shape = Tuple[int, ...]

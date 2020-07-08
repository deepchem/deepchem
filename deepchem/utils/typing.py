"""Type annotations that are widely used in DeepChem"""

from typing import Callable, List, Sequence, Tuple, TypeVar, Union

T = TypeVar("T")
ActivationFn = Union[Callable, str]
LossFunction = Callable[[List, List, List], float]
OneOrMany = Union[T, Sequence[T]]
Shape = Tuple[int, ...]

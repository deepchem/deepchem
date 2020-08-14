"""Type annotations that are widely used in DeepChem"""

from typing import Any, Callable, List, Sequence, Tuple, TypeVar, Union

T = TypeVar("T")

# An activation function for a Keras layer: either a TensorFlow function or the name of a standard activation
KerasActivationFn = Union[Callable, str]

# A loss function for use with KerasModel or TorchModel: f(outputs, labels, weights)
LossFn = Callable[[List, List, List], Any]

# A single value of some type, or multiple values of that type
OneOrMany = Union[T, Sequence[T]]

# The shape of a NumPy array
Shape = Tuple[int, ...]

# type of RDKit object
RDKitMol = Any
RDKitAtom = Any

# type of Pymatgen object
PymatgenStructure = Any
PymatgenComposition = Any

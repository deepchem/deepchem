"""Normalizing flows for transforming probability distributions using PyTorch.
"""

try:
    from deepchem.models.torch_models.flows import NormalizingFlow  # noqa
except ModuleNotFoundError as e:
    raise ImportError(
        f'These classes require PyTorch to be installed. Original error: {e}'
    ) from e

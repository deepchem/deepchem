"""Interface for reinforcement learning."""
# flake8: noqa
try:
    from deepchem.rl.torch_rl.torch_a2c import A2C  # noqa: F401
except ModuleNotFoundError:
    pass

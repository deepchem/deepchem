"""Interface for reinforcement learning."""
# flake8: noqa
try:
    from deepchem.rl.torch_rl.torch_a2c import A2CLossDiscrete, A2CLossContinuous, A2C  # noqa: F401
    from deepchem.rl.torch_rl.torch_ppo import PPOLoss, PPO  # noqa: F401
except ModuleNotFoundError:
    pass

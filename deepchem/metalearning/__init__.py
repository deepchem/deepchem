# flake8: noqa
try:
    from deepchem.metalearning.maml import MAML, MetaLearner
    from deepchem.metalearning.torch_maml import TorchMAML, TorchMetaLearner
except ModuleNotFoundError:
    pass

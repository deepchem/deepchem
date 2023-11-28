# flake8: noqa
import logging

logger = logging.getLogger(__name__)
try:
    from deepchem.metalearning.maml import MAML, MetaLearner
    from deepchem.metalearning.torch_maml import MAML, MetaLearner
except ModuleNotFoundError as e:
    logger.warning(
        f'Skipped loading some PyTorch models, missing a dependency. {e}')

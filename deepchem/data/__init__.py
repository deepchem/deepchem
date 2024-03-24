"""
Gathers all datasets in one place for convenient imports
"""
# flake8: noqa
import logging

logger = logging.getLogger(__name__)

# TODO(rbharath): Get rid of * import
from deepchem.data.datasets import pad_features
from deepchem.data.datasets import pad_batch
from deepchem.data.datasets import Dataset
from deepchem.data.datasets import NumpyDataset
from deepchem.data.datasets import DiskDataset
from deepchem.data.datasets import ImageDataset
from deepchem.data.datasets import sparsify_features
from deepchem.data.datasets import densify_features
from deepchem.data.supports import *
from deepchem.data.data_loader import DataLoader
from deepchem.data.data_loader import CSVLoader
from deepchem.data.data_loader import UserCSVLoader
from deepchem.data.data_loader import JsonLoader
from deepchem.data.data_loader import SDFLoader
from deepchem.data.data_loader import FASTALoader
from deepchem.data.data_loader import FASTQLoader
from deepchem.data.data_loader import ImageLoader
from deepchem.data.data_loader import InMemoryLoader
try:
    from deepchem.data.data_loader import SAMLoader
    from deepchem.data.data_loader import BAMLoader
    from deepchem.data.data_loader import CRAMLoader
except ImportError as e:
    logger.warning(
        f'Skipped loading biological sequence loaders, missing a dependency. {e}'
    )

"""
Gathers all datasets in one place for convenient imports
"""
# flake8: noqa

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
from deepchem.data.data_loader import FragmentLoader
from deepchem.data.data_loader import CSVLoader
from deepchem.data.data_loader import CSVFragmentLoader
from deepchem.data.data_loader import UserCSVLoader
from deepchem.data.data_loader import JsonLoader
from deepchem.data.data_loader import SDFLoader
from deepchem.data.data_loader import SDFFragmentLoader
from deepchem.data.data_loader import FASTALoader
from deepchem.data.data_loader import ImageLoader
from deepchem.data.data_loader import InMemoryLoader

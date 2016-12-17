"""
Gathers all datasets in one place for convenient imports
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

# TODO(rbharath): Get rid of * import
from deepchem.data.datasets import pad_features
from deepchem.data.datasets import pad_batch
from deepchem.data.datasets import Dataset
from deepchem.data.datasets import NumpyDataset
from deepchem.data.datasets import DiskDataset
from deepchem.data.datasets import sparsify_features
from deepchem.data.datasets import densify_features
from deepchem.data.supports import *
from deepchem.data.data_loader import DataLoader
from deepchem.data.data_loader import CSVLoader
from deepchem.data.data_loader import UserCSVLoader
from deepchem.data.data_loader import SDFLoader
import deepchem.data.tests

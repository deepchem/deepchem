"""
Train support-based models.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import warnings
import numpy as np
import tensorflow as tf
import sys
import time
from deepchem.models import Model
from deepchem.data import pad_batch
from deepchem.data import NumpyDataset
from deepchem.metrics import to_one_hot
from deepchem.metrics import from_one_hot
from deepchem.nn import model_ops
from deepchem.data import SupportGenerator
from deepchem.data import EpisodeGenerator
from deepchem.data import get_task_dataset
from deepchem.data import get_single_task_test
from deepchem.data import get_task_dataset_minus_support

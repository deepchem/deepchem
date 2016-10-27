"""
Contains an abstract base class that supports chemically aware data splits.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

import tempfile
import numpy as np
from rdkit import Chem
from deepchem.utils import ScaffoldGenerator
from deepchem.utils.save import log
from deepchem.datasets import NumpyDataset
from deepchem.featurizers.featurize import load_data
from deepchem.splits import Splitter

class TaskSplitter(Splitter):
  """
  Provides a simple interface for splitting datasets task-wise.

  For some learning problems, the training and test datasets should
  have different tasks entirely. This is a different paradigm from the
  usual Splitter, which ensures that split datasets have different
  datapoints, not different tasks.
  """

  def __init__(self):
    "Creates Task Splitter object."
    pass

  def train_valid_test_split(self, dataset, frac_train=.8, frac_valid=.1,
                             frac_test=.1):
    """Performs a train/valid/test split of the tasks for dataset.

    Parameters
    ----------
    dataset: deepchem.datasets.Dataset
      Dataset to be split
    frac_train: float, optional
      Proportion of tasks to be put into train. Rounded to nearest int.
    frac_valid: float, optional
      Proportion of tasks to be put into valid. Rounded to nearest int.
    frac_test: float, optional
      Proportion of tasks to be put into test. Rounded to nearest int.
    """
    n_tasks = len(dataset.get_task_names())
    n_train = np.round(frac_train * n_tasks)
    n_valid = np.round(frac_valid * n_tasks)
    n_test = np.round(frac_test * n_tasks)
    if n_train + n_valid + n_test != n_tasks:
      raise ValueError("Train/Valid/Test fractions don't split tasks evenly.")
    ########################################### DEBUG
    print("train_valid_test_split")
    print("n_train, n_valid, n_test")
    print(n_train, n_valid, n_test)
    ########################################### DEBUG

    X, y, w, ids = dataset.X, dataset.y, dataset.w, dataset.ids
    
    train_dataset = NumpyDataset(X, y[:,:n_train], w[:,:n_train], ids)
    valid_dataset = NumpyDataset(
        X, y[:,n_train:n_train+n_valid], w[:,n_train:n_train+n_valid], ids)
    test_dataset = NumpyDataset(
        X, y[:,n_train+n_valid:], w[:,n_train+n_valid:], ids)
    ########################################### DEBUG
    print("train_dataset.get_task_names()")
    print(train_dataset.get_task_names())
    print("valid_dataset.get_task_names()")
    print(valid_dataset.get_task_names())
    print("test_dataset.get_task_names()")
    print(test_dataset.get_task_names())
    ########################################### DEBUG
    return train_dataset, valid_dataset, test_dataset

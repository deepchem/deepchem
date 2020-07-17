"""
Contains an abstract base class that supports chemically aware data splits.
"""
import tempfile
import numpy as np
import logging
from deepchem.utils import ScaffoldGenerator
from deepchem.data import NumpyDataset
from deepchem.utils.save import load_data
from deepchem.splits import Splitter
from deepchem.utils.data import datasetify

logger = logging.getLogger(__name__)

def merge_fold_datasets(fold_datasets):
  """Merges fold datasets together.

  Assumes that fold_datasets were outputted from k_fold_split.
  Specifically, assumes that each dataset contains the same
  datapoints, listed in the same ordering.

  Parameters
  ----------
  fold_dataset: list[dc.data.Dataset]
    Each entry of this list should be a `dc.data.Dataset` object.
  """
  if not len(fold_datasets):
    return None

  # All datasets share features and identifiers by assumption.
  X = fold_datasets[0].X
  ids = fold_datasets[0].ids

  ys, ws = [], []
  for fold_dataset in fold_datasets:
    ys.append(fold_dataset.y)
    ws.append(fold_dataset.w)
  y = np.concatenate(ys, axis=1)
  w = np.concatenate(ws, axis=1)
  return NumpyDataset(X, y, w, ids)


class TaskSplitter(Splitter):
  """
  Provides a simple interface for splitting datasets task-wise.

  For some learning problems, the training and test datasets
  should have different tasks entirely. This is a different
  paradigm from the usual Splitter, which ensures that split
  datasets have different datapoints, not different tasks.
  """

  def __init__(self, *args, **kwargs):
    """Creates Task Splitter object."""
    super(TaskSplitter, self).__init__(*args, **kwargs)

  def train_valid_test_split(self,
                             dataset,
                             frac_train=.8,
                             frac_valid=.1,
                             frac_test=.1,
                             seed=None):
    """Performs a train/valid/test split of the tasks for dataset.

    If split is uneven, spillover goes to test.

    Parameters
    ----------
    dataset: data-like object. 
      Dataset to do a k-fold split on. This should either be of type
      `dc.data.Dataset` or a type that `dc.utils.data.datasetify` can
      convert into a `Dataset`.
    frac_train: float, optional
      Proportion of tasks to be put into train. Rounded to nearest int.
    frac_valid: float, optional
      Proportion of tasks to be put into valid. Rounded to nearest int.
    frac_test: float, optional
      Proportion of tasks to be put into test. Rounded to nearest int.
    seed: int, optional
      Random seed to make the split deterministic 
    """
    np.random.seed(seed)
    dataset = datasetify(dataset)
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1)
    n_tasks = len(dataset.get_task_names())
    n_train = int(np.round(frac_train * n_tasks))
    n_valid = int(np.round(frac_valid * n_tasks))
    n_test = n_tasks - n_train - n_valid

    X, y, w, ids = dataset.X, dataset.y, dataset.w, dataset.ids

    train_dataset = NumpyDataset(X, y[:, :n_train], w[:, :n_train], ids)
    valid_dataset = NumpyDataset(X, y[:, n_train:n_train + n_valid],
                                 w[:, n_train:n_train + n_valid], ids)
    test_dataset = NumpyDataset(X, y[:, n_train + n_valid:],
                                w[:, n_train + n_valid:], ids)
    return train_dataset, valid_dataset, test_dataset

  def k_fold_split(self, dataset, K, seed=None):
    """Performs a K-fold split of the tasks for dataset.

    If split is uneven, spillover goes to last fold.

    Parameters
    ----------
    dataset: data like object. 
      Dataset to be split. This should either be of type
      `dc.data.Dataset` or a type that `dc.utils.data.datasetify` can
      convert into a `Dataset`.
    K: int
      Number of splits to be made
    seed: int, optional
      Random seed to make the split deterministic 
    """
    if seed is not None:
      np.random.seed(seed)
    dataset = datasetify(dataset)
    n_tasks = len(dataset.get_task_names())
    n_per_fold = int(np.round(n_tasks / float(K)))
    if K * n_per_fold != n_tasks:
      logger.info("Assigning extra tasks to last fold due to uneven split")

    X, y, w, ids = dataset.X, dataset.y, dataset.w, dataset.ids

    fold_datasets = []
    for fold in range(K):
      if fold != K - 1:
        fold_tasks = range(fold * n_per_fold, (fold + 1) * n_per_fold)
      else:
        fold_tasks = range(fold * n_per_fold, n_tasks)
      if len(w.shape) == 1:
        w_tasks = w
      elif w.shape[1] == 1:
        w_tasks = w[:, 0]
      else:
        w_tasks = w[:, fold_tasks]
      fold_datasets.append(NumpyDataset(X, y[:, fold_tasks], w_tasks, ids))
    return fold_datasets

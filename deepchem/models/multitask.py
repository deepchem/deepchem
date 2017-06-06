"""
Convenience class that lets singletask models fit on multitask data.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import sklearn
import tempfile
import numpy as np
import shutil
from deepchem.utils.save import log
from deepchem.models import Model
from deepchem.data import DiskDataset
from deepchem.trans import undo_transforms


class SingletaskToMultitask(Model):
  """
  Convenience class to let singletask models be fit on multitask data.

  Warning: This current implementation is only functional for sklearn models.
  """

  def __init__(self, tasks, model_builder, model_dir=None, verbose=True):
    super(SingletaskToMultitask, self).__init__(
        self, model_dir=model_dir, verbose=verbose)
    self.tasks = tasks
    self.task_model_dirs = {}
    self.model_builder = model_builder
    log("About to initialize singletask to multitask model", self.verbose)
    for task in self.tasks:
      task_model_dir = os.path.join(self.model_dir, str(task))
      if not os.path.exists(task_model_dir):
        os.makedirs(task_model_dir)
      log("Initializing directory for task %s" % task, self.verbose)
      self.task_model_dirs[task] = task_model_dir

  def _create_task_datasets(self, dataset):
    """Make directories to hold data for tasks"""
    task_data_dirs = []
    for task in self.tasks:
      task_data_dir = os.path.join(self.model_dir, str(task) + "_data")
      if os.path.exists(task_data_dir):
        shutil.rmtree(task_data_dir)
      os.makedirs(task_data_dir)
      task_data_dirs.append(task_data_dir)
    task_datasets = self._to_singletask(dataset, task_data_dirs)
    for task, task_dataset in zip(self.tasks, task_datasets):
      log("Dataset for task %s has shape %s" %
          (task, str(task_dataset.get_shape())), self.verbose)
    return task_datasets

  @staticmethod
  def _to_singletask(dataset, task_dirs):
    """Transforms a multitask dataset to a collection of singletask datasets."""
    tasks = dataset.get_task_names()
    assert len(tasks) == len(task_dirs)
    log("Splitting multitask dataset into singletask datasets", dataset.verbose)
    task_datasets = [
        DiskDataset.create_dataset([], task_dirs[task_num], [task])
        for (task_num, task) in enumerate(tasks)
    ]
    #task_metadata_rows = {task: [] for task in tasks}
    for shard_num, (X, y, w, ids) in enumerate(dataset.itershards()):
      log("Processing shard %d" % shard_num, dataset.verbose)
      basename = "dataset-%d" % shard_num
      for task_num, task in enumerate(tasks):
        log("\tTask %s" % task, dataset.verbose)
        w_task = w[:, task_num]
        y_task = y[:, task_num]

        # Extract those datapoints which are present for this task
        X_nonzero = X[w_task != 0]
        num_datapoints = X_nonzero.shape[0]
        y_nonzero = np.reshape(y_task[w_task != 0], (num_datapoints, 1))
        w_nonzero = np.reshape(w_task[w_task != 0], (num_datapoints, 1))
        ids_nonzero = ids[w_task != 0]

        task_datasets[task_num].add_shard(X_nonzero, y_nonzero, w_nonzero,
                                          ids_nonzero)

    return task_datasets

  def fit(self, dataset, **kwargs):
    """
    Updates all singletask models with new information.

    Warning: This current implementation is only functional for sklearn models.
    """
    if not isinstance(dataset, DiskDataset):
      raise ValueError('SingletaskToMultitask only works with DiskDatasets')
    log("About to create task-specific datasets", self.verbose)
    task_datasets = self._create_task_datasets(dataset)
    for ind, task in enumerate(self.tasks):
      log("Fitting model for task %s" % task, self.verbose)
      task_model = self.model_builder(self.task_model_dirs[task])
      task_model.fit(task_datasets[ind], **kwargs)
      task_model.save()

  def predict_on_batch(self, X):
    """
    Concatenates results from all singletask models.
    """
    n_tasks = len(self.tasks)
    n_samples = X.shape[0]
    y_pred = np.zeros((n_samples, n_tasks))
    for ind, task in enumerate(self.tasks):
      task_model = self.model_builder(self.task_model_dirs[task])
      task_model.reload()

      y_pred[:, ind] = task_model.predict_on_batch(X)
    return y_pred

  def predict(self, dataset, transformers=[]):
    """
    Prediction for multitask models.
    """
    n_tasks = len(self.tasks)
    n_samples = len(dataset)
    y_pred = np.zeros((n_samples, n_tasks))
    for ind, task in enumerate(self.tasks):
      task_model = self.model_builder(self.task_model_dirs[task])
      task_model.reload()

      y_pred[:, ind] = task_model.predict(dataset, [])
    y_pred = undo_transforms(y_pred, transformers)
    return y_pred

  def predict_proba_on_batch(self, X, n_classes=2):
    """
    Concatenates results from all singletask models.
    """
    n_tasks = len(self.tasks)
    n_samples = X.shape[0]
    y_pred = np.zeros((n_samples, n_tasks, n_classes))
    for ind, task in enumerate(self.tasks):
      task_model = self.model_builder(self.task_model_dirs[task])
      task_model.reload()

      y_pred[:, ind] = task_model.predict_proba_on_batch(X)
    return y_pred

  def predict_proba(self, dataset, transformers=[], n_classes=2):
    """
    Concatenates results from all singletask models.
    """
    n_tasks = len(self.tasks)
    n_samples = len(dataset)
    y_pred = np.zeros((n_samples, n_tasks, n_classes))
    for ind, task in enumerate(self.tasks):
      task_model = self.model_builder(self.task_model_dirs[task])
      task_model.reload()

      y_pred[:, ind] = np.squeeze(
          task_model.predict_proba(dataset, transformers, n_classes))
    return y_pred

  def save(self):
    """Save all models

    TODO(rbharath): Saving is not yet supported for this model.
    """
    pass

  def reload(self):
    """Load all models"""
    # Loading is done on-the-fly
    pass

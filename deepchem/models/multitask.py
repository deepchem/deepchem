"""
Convenience class that lets singletask models fit on multitask data.
"""
import os
import sklearn
import tempfile
import numpy as np
import shutil
import logging
from deepchem.models import Model
from deepchem.data import DiskDataset
from deepchem.trans import undo_transforms

logger = logging.getLogger(__name__)


class SingletaskToMultitask(Model):
  """
  Convenience class to let singletask models be fit on multitask data.

  Warning: This current implementation is only functional for sklearn models.
  """

  def __init__(self, tasks, model_builder, model_dir=None):
    super(SingletaskToMultitask, self).__init__(self, model_dir=model_dir)
    self.tasks = tasks
    self.task_model_dirs = {}
    self.model_builder = model_builder
    logger.info("About to initialize singletask to multitask model")
    for task in self.tasks:
      task_model_dir = os.path.join(self.model_dir, str(task))
      if not os.path.exists(task_model_dir):
        os.makedirs(task_model_dir)
      logger.info("Initializing directory for task %s" % task)
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
      logger.info("Dataset for task %s has shape %s" %
                  (task, str(task_dataset.get_shape())))
    return task_datasets

  @staticmethod
  def _to_singletask(dataset, task_dirs):
    """Transforms a multitask dataset to a collection of singletask datasets."""
    tasks = dataset.get_task_names()
    assert len(tasks) == len(task_dirs)
    logger.info("Splitting multitask dataset into singletask datasets")
    task_datasets = [
        DiskDataset.create_dataset([], task_dirs[task_num], [task])
        for (task_num, task) in enumerate(tasks)
    ]
    #task_metadata_rows = {task: [] for task in tasks}
    for shard_num, (X, y, w, ids) in enumerate(dataset.itershards()):
      logger.info("Processing shard %d" % shard_num)
      basename = "dataset-%d" % shard_num
      for task_num, task in enumerate(tasks):
        logger.info("\tTask %s" % task)
        if len(w.shape) == 1:
          w_task = w
        elif w.shape[1] == 1:
          w_task = w[:, 0]
        else:
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
    logger.info("About to create task-specific datasets")
    task_datasets = self._create_task_datasets(dataset)
    for ind, task in enumerate(self.tasks):
      logger.info("Fitting model for task %s" % task)
      task_model = self.model_builder(self.task_model_dirs[task])
      task_model.fit(task_datasets[ind], **kwargs)
      task_model.save()

  def predict_on_batch(self, X):
    """
    Concatenates results from all singletask models.
    """
    n_tasks = len(self.tasks)
    n_samples = X.shape[0]
    y_preds = []
    for ind, task in enumerate(self.tasks):
      task_model = self.model_builder(self.task_model_dirs[task])
      task_model.reload()

      y_preds.append(task_model.predict_on_batch(X))
    y_pred = np.stack(y_preds, axis=1)
    return y_pred

  def predict(self, dataset, transformers=[]):
    """
    Prediction for multitask models.
    """
    n_tasks = len(self.tasks)
    n_samples = len(dataset)
    y_preds = []
    for ind, task in enumerate(self.tasks):
      task_model = self.model_builder(self.task_model_dirs[task])
      task_model.reload()

      y_preds.append(task_model.predict(dataset, []))
    y_pred = np.stack(y_preds, axis=1)
    y_pred = undo_transforms(y_pred, transformers)
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

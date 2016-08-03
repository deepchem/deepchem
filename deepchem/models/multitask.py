"""
Convenience class that lets singletask models fit on multitask data.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
from deepchem.utils.save import log
from deepchem.models import Model
import sklearn

class SingletaskToMultitask(Model):
  """
  Convenience class to let singletask models be fit on multitask data.

  Warning: This current implementation is only functional for sklearn models. 
  """
  def __init__(self, tasks, task_types, model_params, model_dir, model_builder,
               store_in_memory=False, verbosity=None):
    self.tasks = tasks
    self.task_types = task_types
    self.model_params = model_params
    self.models = {}
    self.model_dir = model_dir
    # If models are TF models, they don't use up RAM, so can keep in memory
    self.task_models = {}
    self.task_model_dirs = {}
    self.model_builder = model_builder
    self.verbosity = verbosity
    self.store_in_memory = store_in_memory
    log("About to initialize singletask to multitask model",
        self.verbosity, "high")
    if not os.path.exists(self.model_dir):
      os.makedirs(self.model_dir)
    self.fit_transformers = False
    for task in self.tasks:
      task_type = self.task_types[task]
      task_model_dir = os.path.join(self.model_dir, str(task))
      if not os.path.exists(task_model_dir):
        os.makedirs(task_model_dir)
      log("Initializing model for task %s" % task,
          self.verbosity, "high")
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
    task_datasets = dataset.to_singletask(task_data_dirs)
    if self.verbosity is not None:
      for task, task_dataset in zip(self.tasks, task_datasets):
        log("Dataset for task %s has shape %s"
            % (task, str(task_dataset.get_shape())), self.verbosity)
    return task_datasets
   
      
  def fit(self, dataset):
    """
    Updates all singletask models with new information.

    Warning: This current implementation is only functional for sklearn models. 
    """
    log("About to create task-specific datasets", self.verbosity, "high")
    task_datasets = self._create_task_datasets(dataset)
    for ind, task in enumerate(self.tasks):
      log("Fitting model for task %s" % task, self.verbosity, "high")
      #y_task = task_datasets[ind].get_labels()
      #X_task, y_task, w_task, ids_task = task_datasets[ind].to_numpy()
      task_model = self.model_builder(
          [task], {task: self.task_types[task]}, self.model_params,
          self.task_model_dirs[task],
          verbosity=self.verbosity)
      #if y_task.size > 0:
        #task_model.raw_model.fit(X_task, np.ravel(y_task))
      task_model.fit(task_datasets[ind])
      #else:
      #  print("No labels for task %s" % task)
      #  print("Fitting on dummy dataset.")
      #  X_task_fake = np.zeros_like(X)
      #  y_task_fake = np.zeros_like(w_task)
      #  task_model.raw_model.fit(X_task_fake, y_task_fake)
      task_model.save()
      if self.store_in_memory:
        self.task_models[task] = task_model

  def predict_on_batch(self, X):
    """
    Concatenates results from all singletask models.
    """
    n_tasks = len(self.tasks)
    n_samples = X.shape[0]
    y_pred = np.zeros((n_samples, n_tasks))
    for ind, task in enumerate(self.tasks):
      task_type = self.task_types[task]
      if self.store_in_memory:
        task_model = self.task_models[task]
      else:
        task_model = self.model_builder(
            [task], {task: self.task_types[task]}, self.model_params,
            self.task_model_dirs[task],
            verbosity=self.verbosity)
        task_model.reload()

      if task_type == "classification":
        y_pred[:, ind] = task_model.predict_on_batch(X)
      elif task_type == "regression":
        y_pred[:, ind] = task_model.predict_on_batch(X)
      else:
        raise ValueError("Invalid task_type")
    return y_pred

  def predict_proba_on_batch(self, X, n_classes=2):
    """
    Concatenates results from all singletask models.
    """
    n_tasks = len(self.tasks)
    n_samples = X.shape[0]
    y_pred = np.zeros((n_samples, n_tasks, n_classes))
    for ind, task in enumerate(self.tasks):
      if self.store_in_memory:
        task_model = self.task_models[task]
      else:
        task_model = self.model_builder(
            [task], {task: self.task_types[task]}, self.model_params,
            self.task_model_dirs[task],
            verbosity=self.verbosity)
        task_model.reload()

      y_pred[:, ind] = task_model.predict_proba_on_batch(X)
    return y_pred

  def save(self):
    """Save all models"""
    # Saving is done on-the-fly
    pass

  def reload(self):
    """Load all models"""
    # Loading is done on-the-fly
    pass

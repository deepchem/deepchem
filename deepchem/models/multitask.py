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
# DEBUG
import sklearn

class SingletaskToMultitask(Model):
  """
  Convenience class to let singletask models be fit on multitask data.

  Warning: This current implementation is only functional for sklearn models. 
  """
  def __init__(self, tasks, task_types, model_params, model_dir, model_builder,
               verbosity=None):
    self.tasks = tasks
    self.task_types = task_types
    self.model_params = model_params
    self.models = {}
    self.model_dir = model_dir
    self.task_model_dirs = {}
    self.model_builder = model_builder
    self.verbosity = verbosity
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
      
  def fit(self, dataset):
    """
    Updates all singletask models with new information.

    Warning: This current implementation is only functional for sklearn models. 
    """
    X, y, w, _ = dataset.to_numpy()
    for ind, task in enumerate(self.tasks):
      log("Fitting model for task %s" % task, self.verbosity, "high")
      y_task = y[:, ind]
      w_task = w[:, ind]
      X_task = X[w_task != 0, :]
      y_task = y_task[w_task != 0]
      task_model = self.model_builder(
          [task], {task: self.task_types[task]}, self.model_params,
          self.task_model_dirs[task],
          verbosity=self.verbosity)
      #################################### DEBUG
      if y_task.size > 0:
      #################################### DEBUG
        task_model.raw_model.fit(X_task, y_task)
      #################################### DEBUG
      else:
        print("No labels for task %s" % task)
        print("Fitting on dummy dataset.")
        X_task_fake = np.zeros_like(X)
        y_task_fake = np.zeros_like(w_task)
        print("X.shape, y.shape, w.shape, y_task.shape, w_task.shape, y_task_fake.shape")
        print(X.shape, y.shape, w.shape, y_task.shape, w_task.shape, y_task_fake.shape)
        task_model.raw_model.fit(X_task_fake, y_task_fake)
      #################################### DEBUG
      task_model.save()

  def predict_on_batch(self, X):
    """
    Concatenates results from all singletask models.
    """
    n_tasks = len(self.tasks)
    n_samples = X.shape[0]
    y_pred = np.zeros((n_samples, n_tasks))
    for ind, task in enumerate(self.tasks):
      task_type = self.task_types[task]
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
      ############################################### DEBUG
    return y_pred

  def predict_proba_on_batch(self, X, n_classes=2):
    """
    Concatenates results from all singletask models.
    """
    n_tasks = len(self.tasks)
    n_samples = X.shape[0]
    y_pred = np.zeros((n_samples, n_tasks, n_classes))
    for ind, task in enumerate(self.tasks):
      task_model = self.model_builder([task], {task: self.task_types[task]}, self.model_params,
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

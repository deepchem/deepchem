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
    self.verbosity = verbosity
    log("About to initialize singletask to multitask model",
        self.verbosity, "high")
    if not os.path.exists(self.model_dir):
      os.makedirs(self.model_dir)
    self.fit_transformers = False
    for task in self.tasks:
      task_type = self.task_types[task]
      task_model_dir = os.path.join(self.model_dir, task)
      if not os.path.exists(task_model_dir):
        os.makedirs(task_model_dir)
      log("Initializing model for task %s" % task,
          self.verbosity, "high")
      self.models[task] = model_builder([tasks], task_types, model_params,
                                        task_model_dir,
                                        verbosity=verbosity)
      
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
      # DEBUG
      X_task = X[w_task != 0, :]
      y_task = y_task[w_task != 0]
      #self.models[task].raw_model.fit(X, y_task, w_task)
      self.models[task].raw_model.fit(X_task, y_task)

      y_pred_task = self.models[task].raw_model.predict(X_task)
      print("Train ROC-AUC for %s: %f"
            % (task, sklearn.metrics.roc_auc_score(y_task.astype(int),
                                                   y_pred_task)))

  def predict_on_batch(self, X):
    """
    Concatenates results from all singletask models.
    """
    N_tasks = len(self.tasks)
    N_samples = X.shape[0]
    y_pred = np.zeros((N_samples, N_tasks))
    for ind, task in enumerate(self.tasks):
      # DEBUG: ONLY FOR DEBUGGING
      #print("ind, task")
      #print(ind, task)
      #print("self.models[task]")
      #print(self.models[task])
      #print("type(self.models[task].predict_on_batch(X))")
      #print(type(self.models[task].predict_on_batch(X)))
      y_pred[:, ind] = self.models[task].predict_on_batch(X)[:, 0]
      #return self.models[task].predict_on_batch(X)[:, 0]
      #y_pred[task] = self.models[task].predict_on_batch(X)
    return y_pred

  def save(self):
    """Save all models"""
    for task in self.tasks:
      log("Saving model for task %s" % task, self.verbosity, "high")
      self.models[task].save()

  def load(self):
    """Load all models"""
    for task in self.tasks:
      log("Loading model for task %s" % task, self.verbosity, "high")
      self.models[task].load()

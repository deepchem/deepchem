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

class SingletaskToMultitask(Model):
  """
  Convenience class to let singletask models be fit on multitask data.

  Warning: This current implementation is only functional for sklearn models. 
  """
  def __init__(self, task_types, model_params, model_dir, model_builder,
               verbosity=None):
    print("ENTERING SingletaskToMultitask")
    print("verbosity")
    print(verbosity)
    self.task_types = task_types
    self.model_params = model_params
    self.models = {}
    self.model_dir = model_dir
    self.verbosity = verbosity
    print("verbosity")
    print(verbosity)
    log("About to initialize singletask to multitask model",
        self.verbosity, "high")
    if not os.path.exists(self.model_dir):
      os.makedirs(self.model_dir)
    self.fit_transformers = False
    for task, task_type in self.task_types.iteritems():
      task_model_dir = os.path.join(self.model_dir, task)
      if not os.path.exists(task_model_dir):
        os.makedirs(task_model_dir)
      log("Initializing model for task %s" % task,
          self.verbosity, "high")
      self.models[task] = model_builder(task_types, model_params,
                                        task_model_dir,
                                        verbosity=verbosity)
      
  def fit(self, dataset):
    """
    Updates all singletask models with new information.

    Warning: This current implementation is only functional for sklearn models. 
    """
    X, y, _, _ = dataset.to_numpy()
    for ind, task in enumerate(self.task_types.keys()):
      log("Fitting model for task %s" % task, self.verbosity, "high")
      y_task = y[:, ind]
      self.models[task].raw_model.fit(X, y_task)

  def predict_on_batch(self, X):
    """
    Concatenates results from all singletask models.
    """
    N_tasks = len(self.task_types.keys())
    N_samples = X.shape[0]
    y_pred = np.zeros((N_samples, N_tasks))
    for ind, task in enumerate(self.task_types.keys()):
      y_pred[:, ind] = self.models[task].predict_on_batch(X)
    return y_pred

  def save(self):
    """Save all models"""
    for task in self.task_types.keys():
      log("Saving model for task %s" % task, self.verbosity, "high")
      self.models[task].save()

  def load(self):
    """Load all models"""
    for task in self.task_types.keys():
      log("Loading model for task %s" % task, self.verbosity, "high")
      self.models[task].load()

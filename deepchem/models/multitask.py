"""
Convenience class that lets singletask models fit on multitask data.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
from deepchem.models import Model


class SingletaskToMultitask(Model):
  """
  Convenience class to let singletask models be fit on multitask data.

  Warning: This current implementation is only functional for sklearn models. 
  """
  def __init__(self, task_types, model_params, model_builder, verbosity=None):
    self.task_types = task_types
    self.model_params = model_params
    self.models = {}
    self.fit_transformers = False
    for task, task_type in self.task_types.iteritems():
      self.models[task] = model_builder(task_types, model_params,
                                        verbosity=verbosity)
      
  def fit(self, dataset):
    """
    Updates all singletask models with new information.

    Warning: This current implementation is only functional for sklearn models. 
    """
    X, y, _, _ = dataset.to_numpy()
    for ind, task in enumerate(self.task_types.keys()):
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

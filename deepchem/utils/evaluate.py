"""
Utility functions to evaluate models on datasets.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import warnings
from deepchem.utils.save import log
import pandas as pd
import sklearn

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2015, Stanford University"
__license__ = "LGPL"

def relative_difference(x, y):
  """Compute the relative difference between x and y"""
  return np.abs(x-y)/np.abs(max(x, y))

def threshold_predictions(y, threshold):
  y_out = np.zeros_like(y)
  for ind, pred in enumerate(y):
    y_out[ind] = 1 if pred > threshold else 0
  return y_out

# TODO(rbharath): This is now simple enough that we should probably get rid of
# Evaluator object to avoid clutter.
class Evaluator(object):
  """Class that evaluates a model on a given dataset."""

  def __init__(self, model, dataset, transformers, verbosity=False):
    self.model = model
    self.dataset = dataset
    self.transformers = transformers
    self.task_names = dataset.get_task_names()
    self.task_type = model.get_task_type().lower()
    self.verbosity = verbosity

  def compute_model_performance(self, metrics, csv_out, stats_file, threshold=None):
    """
    Computes statistics of model on test data and saves results to csv.
    """
    y = self.dataset.get_labels()
    w = self.dataset.get_weights()
    y_pred = self.model.predict(self.dataset, self.transformers)
    multitask_scores = {}

    print("y.shape, y_pred.shape")
    print(y.shape, y_pred.shape)
    # Compute multitask metrics
    for metric in metrics:
      multitask_scores[metric.name] = metric.compute_metric(y, y_pred, w)

    return multitask_scores

"""
Utility functions to evaluate models on datasets.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import csv
import numpy as np
import warnings
import pandas as pd
import sklearn
from deepchem.utils.save import log
from deepchem.transformers import undo_transforms

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

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

  def output_statistics(self, scores, stats_out):
    """
    Write computed stats to file.
    """
    with open(stats_out, "wb") as statsfile:
      statsfile.write(str(scores) + "\n")

  def output_predictions(self, y_preds, csv_out):
    """
    Writes predictions to file.

    Args:
      y_preds: np.ndarray
      csvfile: Open file object.
    """
    mol_ids = self.dataset.get_ids()
    n_tasks = len(self.task_names)
    y_preds = np.reshape(y_preds, (len(y_preds), n_tasks))
    assert len(y_preds) == len(mol_ids)
    with open(csv_out, "wb") as csvfile:
      csvwriter = csv.writer(csvfile)
      csvwriter.writerow(["Compound"] + self.dataset.get_task_names())
      for mol_id, y_pred in zip(mol_ids, y_preds):
        csvwriter.writerow([mol_id] + list(y_pred))

  def compute_model_performance(self, metrics, csv_out=None, stats_out=None,
                                threshold=None):
    """
    Computes statistics of model on test data and saves results to csv.
    """
    y = self.dataset.get_labels()
    y = undo_transforms(y, self.transformers)
    w = self.dataset.get_weights()

    if not len(metrics):
      return {}
    else:
      mode = metrics[0].mode
    if mode == "classification":
      y_pred = self.model.predict_proba(self.dataset, self.transformers)
      y_pred_print = self.model.predict(self.dataset, self.transformers).astype(int)
    else:
      y_pred = self.model.predict(self.dataset, self.transformers)
      y_pred_print = y_pred
    multitask_scores = {}

    if csv_out is not None:
      log("Saving predictions to %s" % csv_out, self.verbosity)
      self.output_predictions(y_pred_print, csv_out)

    # Compute multitask metrics
    for metric in metrics:
      multitask_scores[metric.name] = metric.compute_metric(y, y_pred, w)
    
    if stats_out is not None:
      log("Saving stats to %s" % stats_out, self.verbosity)
      self.output_statistics(multitask_scores, stats_out)
  
    return multitask_scores

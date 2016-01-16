"""
Utility functions to evaluate models on datasets.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import warnings
from deepchem.utils.dataset import Dataset
from deepchem.utils.dataset import load_from_disk
from deepchem.models import Model 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import pandas as pd
import sys

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2015, Stanford University"
__license__ = "LGPL"

def undo_normalization(y, y_means, y_stds):
  """Undo the applied normalization transform."""
  return y * y_stds + y_means

def undo_transform(y, y_means, y_stds, output_transforms):
  """Undo transforms on y_pred, W_pred."""
  if not isinstance(output_transforms, list):
    output_transforms = [output_transforms]
  if (output_transforms == [""] or output_transforms == ['']
    or output_transforms == []):
    return y
  elif output_transforms == ["log"]:
    return np.exp(y)
  elif output_transforms == ["normalize"]:
    return undo_normalization(y, y_means, y_stds)
  elif output_transforms == ["log", "normalize"]:
    return np.exp(undo_normalization(y, y_means, y_stds))
  else:
    raise ValueError("Unsupported output transforms %s." % str(output_transforms))

def compute_roc_auc_scores(y, y_pred, w):
  """Transforms the results dict into roc-auc-scores and prints scores.

  Parameters
  ----------
  results: dict
  task_types: dict
    dict mapping task names to output type. Each output type must be either
    "classification" or "regression".
  """
  try:
    score = roc_auc_score(y, y_pred, sample_weight=w)
  except Exception:
    warnings.warn("ROC AUC score calculation failed.")
    score = 0.5
  return score

class Evaluator(object):
  """Class that evaluates a model on a given dataset."""

  def __init__(self, model, dataset, verbose=False):
    self.model = model
    self.dataset = dataset
    self.task_names = dataset.get_task_names()
    # TODO(rbharath): This is a hack based on fact that multi-tasktype models
    # aren't supported.
    self.task_type = model.task_types.itervalues().next()
    self.output_transforms = dataset.get_output_transforms()
    self.verbose = verbose

  def compute_model_performance(self, csv_out, stats_file):
    """
    Computes statistics of model on test data and saves results to csv.
    """
    pred_y_df = self.model.predict(self.dataset)
    if self.verbose:
      print("Saving predictions to %s" % csv_out)
    pred_y_df.to_csv(csv_out)

    if self.task_type == "classification":
      colnames = ["task_name", "roc_auc_score", "matthews_corrcoef", "recall_score", "accuracy_score"]
    elif self.task_type == "regression":
      colnames = ["task_name", "r2_score", "rms_error"]
    else:
      raise ValueError("Unrecognized task type: %s" % self.task_type)

    performance_df = pd.DataFrame(columns=colnames)
    print("compute_model_performance()")
    y_means = pred_y_df.iterrows().next()[1]["y_means"]
    y_stds = pred_y_df.iterrows().next()[1]["y_stds"]

    for i, task_name in enumerate(self.task_names):
      y = pred_y_df[task_name]
      y_pred = pred_y_df["%s_pred" % task_name]
      w = pred_y_df["%s_weight" % task_name]
      
      y = undo_transform(y, y_means, y_stds, self.output_transforms)
      y_pred = undo_transform(y_pred, y_means, y_stds, self.output_transforms)

      if self.task_type == "classification":
        y, y_pred = y[w.nonzero()], y_pred[w.nonzero()][:, 1]
        auc = compute_roc_auc_scores(y, y_pred, w)
        mcc = matthews_corrcoef(y, np.around(y_pred))
        recall = recall_score(y, np.around(y_pred))
        accuracy = accuracy_score(y, np.around(y_pred))
        performance_df.loc[i] = [task_name, auc, mcc, recall, accuracy]

      elif self.task_type == "regression":
        r2s = r2_score(y, y_pred)
        rms = np.sqrt(mean_squared_error(y, y_pred))
        performance_df.loc[i] = [task_name, r2s, rms]

    print("Saving model performance scores to %s" % stats_file)
    performance_df.to_csv(stats_file)
    
    print("Model performance scores:")
    print(performance_df)

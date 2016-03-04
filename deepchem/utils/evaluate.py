"""
Utility functions to evaluate models on datasets.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import warnings
from deepchem.utils.save import log
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import pandas as pd

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2015, Stanford University"
__license__ = "LGPL"

def to_one_hot(y):
  """Transforms label vector into one-hot encoding.

  Turns y into vector of shape [n_samples, 2] (assuming binary labels).

  y: np.ndarray
    A vector of shape [n_samples, 1]
  """
  n_samples = np.shape(y)[0]
  y_hot = np.zeros((n_samples, 2))
  for index, val in enumerate(y):
    if val == 0:
      y_hot[index] = np.array([1, 0])
    elif val == 1:
      y_hot[index] = np.array([0, 1])
  return y_hot

def threshold_predictions(y, threshold):
  y_out = np.zeros_like(y)
  for ind, pred in enumerate(y):
    y_out[ind] = 1 if pred > threshold else 0
  return y_out

def compute_roc_auc_scores(y, y_pred):
  """Transforms the results dict into roc-auc-scores and prints scores.

  Parameters
  ----------
  results: dict
  task_types: dict
    dict mapping task names to output type. Each output type must be either
    "classification" or "regression".
  """
  try:
    score = roc_auc_score(y, y_pred)
  except ValueError:
    warnings.warn("ROC AUC score calculation failed.")
    score = 0.5
  return score

class Evaluator(object):
  """Class that evaluates a model on a given dataset."""

  def __init__(self, model, dataset, transformers, verbose=False):
    self.model = model
    self.dataset = dataset
    self.transformers = transformers
    self.task_names = dataset.get_task_names()
    self.task_type = model.get_task_type()
    self.verbose = verbose

  def compute_model_performance(self, csv_out, stats_file, threshold=None):
    """
    Computes statistics of model on test data and saves results to csv.
    """
    pred_y_df = self.model.predict(self.dataset, self.transformers)

    task_type = self.task_type
    if threshold is not None:
      task_type = "classification"

    if task_type == "classification":
      colnames = ["task_name", "roc_auc_score", "matthews_corrcoef",
                  "recall_score", "accuracy_score"]
    elif task_type == "regression":
      colnames = ["task_name", "r2_score", "rms_error"]
    else:
      raise ValueError("Unrecognized task type: %s" % task_type)

    performance_df = pd.DataFrame(columns=colnames)

    for i, task_name in enumerate(self.task_names):
      print("task_name")
      print(task_name)
      print("pred_y_df.keys()")
      print(pred_y_df.keys())
      y = pred_y_df[task_name].values
      y_pred = pred_y_df["%s_pred" % task_name].values
      if threshold is not None:
        # TODO(rbharath): This is a hack. More structured approach?
        y = pred_y_df[task_name+"_raw"].values
        y_pred = threshold_predictions(y_pred, threshold)
      w = pred_y_df["%s_weight" % task_name].values

      if task_type == "classification":
        y, y_pred = y[w.nonzero()].astype(int), y_pred[w.nonzero()].astype(int)
        # Sometimes all samples have zero weight. In this case, continue.
        if not len(y):
          continue
        auc = compute_roc_auc_scores(y, y_pred)
        mcc = matthews_corrcoef(y, y_pred)
        recall = recall_score(y, y_pred)
        accuracy = accuracy_score(y, y_pred)
        performance_df.loc[i] = [task_name, auc, mcc, recall, accuracy]

      elif task_type == "regression":
        try:
          r2s = r2_score(y, y_pred)
          rms = np.sqrt(mean_squared_error(y, y_pred))
        except ValueError:
          r2s = np.nan
          rms = np.nan
        performance_df.loc[i] = [task_name, r2s, rms]

    log("Saving predictions to %s" % csv_out, self.verbose)
    pred_y_df.to_csv(csv_out)
    log("Saving model performance scores to %s" % stats_file, self.verbose)
    performance_df.to_csv(stats_file)

    return pred_y_df, performance_df

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

def from_one_hot(y, axis=1):
  """Transorms label vector from one-hot encoding.

  y: np.ndarray
    A vector of shape [n_samples, num_classes]
  """
  return np.argmax(y, axis=axis)

def threshold_predictions(y, threshold):
  y_out = np.zeros_like(y)
  for ind, pred in enumerate(y):
    y_out[ind] = 1 if pred > threshold else 0
  return y_out

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
    #pred_y_df = self.model.predict(self.dataset, self.transformers)
    y = self.dataset.get_labels()
    w = self.dataset.get_weights()
    y_pred = self.model.predict(self.dataset, self.transformers)
    multitask_scores = {}

    colnames = ["task_name"] + [metric.name for metric in metrics]
    performance_df = pd.DataFrame(columns=colnames)

    #nonempty_tasks, ys, y_preds, ws = [], [], [], []
    #if self.task_type == "classification":
    #  y = y.astype(int)
    #print("compute_model_performance()")
    #y = y[w.flatten() != 0, :]
    #y = to_one_hot(y)
    #y_pred = y_pred[w.flatten() != 0, :]
    #print("y.shape, y_pred.shape")
    #print(y.shape, y_pred.shape)
    #
    #print("sklearn.metrics.roc_auc_score(y, y_pred)")
    #print(sklearn.metrics.roc_auc_score(y, y_pred))
    #for task, task_name in enumerate(self.task_names):
    #  #y = pred_y_df[task_name].values
    #  #y_pred = pred_y_df["%s_pred" % task_name].values
    #  y_task = y[:, task]
    #  y_pred_task = y_pred[:, task]
    #  #if threshold is not None:
    #  #  # TODO(rbharath): This is a hack. More structured approach?
    #  #  y = pred_y_df[task_name+"_raw"].values
    #  #  y_pred = threshold_predictions(y_pred, threshold)
    #  #w = pred_y_df["%s_weight" % task_name].values
    #  w_task = w[:, task]

    #  y_task, y_pred_task = y_task[w_task != 0, :], y_pred_task[w_task != 0, :]
    #  #if self.task_type == "classification":
    #  #  y_task, y_pred_task = y.astype(int), y_pred.astype(int)
    #    # Sometimes all samples have zero weight. In this case, continue.
    #    #if not len(y):
    #    #  continue
    #  #nonempty_tasks.append(task_name)
    #  #ys.append(y)
    #  #y_preds.append(y_pred)
    #  #ws.append(w)

    # Compute multitask metrics
    for metric in metrics:
      multitask_scores[metric.name] = metric.compute_metric(y, y_pred, w)
    print("multitask_scores")
    print(multitask_scores)

    #all_scores = []
    #for metric in metrics:
    #  if not metric.is_multitask:
    #    all_scores.append(metric.compute_metric(ys, y_preds))
    # Note that all_scores will be of shape num_singletask_metrics x num_tasks
    #all_scores = np.array(all_scores)
    # If there are any singletask_metrics
    #if all_scores.shape[0] > 0:
    #  nonzero_ind = 0
    #  for i, task_name in enumerate(self.task_names):
    #    if task_name in nonempty_tasks:
    #      performance_df.loc[i] = [task_name] + list(all_scores[:, nonzero_ind])
    #      nonzero_ind += 1

    #log("Saving predictions to %s" % csv_out, self.verbosity)
    #pred_y_df.to_csv(csv_out)
    #log("Saving model performance scores to %s" % stats_file, self.verbosity)
    #performance_df.to_csv(stats_file)

    #return pred_y_df, performance_df, multitask_scores
    return multitask_scores

#!/usr/bin/python
#
# Copyright 2015 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluation metrics."""

import numpy as np
import warnings
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def compute_metrics(num_tasks, y_true, y_pred, metric):
  """Compute a performance metric for each task.

  Args:
    y_true: A list of arrays containing true values for each task.
    y_pred: A list of arrays containing predicted values for each task.
    metric: Must be a class that inherits from Metric 

  Returns:
    A numpy array containing metric values for each task.
  """
  computed_metrics = []
  for task in xrange(num_tasks):
    yt = y_true[task]
    yp = y_pred[task]
    try:
      metric_value = metric.compute(yt, yp)
    except (AssertionError, ValueError) as e:
      warnings.warn("Error calculating metric %s for task %d: %s"
                    % (metric_str, task, e))
      metric_value = np.nan
    computed_metrics.append(metric_value)
  return computed_metrics

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

def kappa_score(y_true, y_pred):
  """Calculate Cohen's kappa for classification tasks.

  See https://en.wikipedia.org/wiki/Cohen%27s_kappa

  Note that this implementation of Cohen's kappa expects binary labels.

  Args:
    y_true: Numpy array containing true values.
    y_pred: Numpy array containing predicted values.

  Returns:
    kappa: Numpy array containing kappa for each classification task.

  Raises:
    AssertionError: If y_true and y_pred are not the same size, or if class
      labels are not in [0, 1].
  """
  assert len(y_true) == len(y_pred), 'Number of examples does not match.'
  yt = np.asarray(y_true, dtype=int)
  yp = np.asarray(y_pred, dtype=int)
  assert np.array_equal(np.unique(yt), [0, 1]), (
      'Class labels must be binary: %s' % np.unique(yt))
  observed_agreement = np.true_divide(np.count_nonzero(np.equal(yt, yp)),
                                      len(yt))
  expected_agreement = np.true_divide(
      np.count_nonzero(yt == 1) * np.count_nonzero(yp == 1) +
      np.count_nonzero(yt == 0) * np.count_nonzero(yp == 0),
      len(yt) ** 2)
  kappa = np.true_divide(observed_agreement - expected_agreement,
                         1.0 - expected_agreement)
  return kappa

class Metric(object):
  """Wrapper class for computing user-defined metrics."""

  def __init__(self, metric, name=None, threshold=None):
    """
    Args:
      metric: function that takes args y_true, y_pred (in that order) and
              computes desired score.
    """
    self.metric = metric
    if name is None:
      self.name = self.metric.__name__
    else:
      self.name = name
    self.threshold = threshold

  def compute_metric(self, y_true, y_pred):
    """Compute a metric value.

    Args:
      y_true: A list of arrays containing true values for each task.
      y_pred: A list of arrays containing predicted values for each task.

    Returns:
      Float metric value.

    Raises:
      NotImplementedError: If metric_str is not in METRICS.
    """
    if self.threshold is not None:
      y_pred = np.greater(y_pred, threshold)
    try:
      metric_value = self.metric(y_true, y_pred)
    except (AssertionError, ValueError) as e:
      warnings.warn("Error calculating metric %s: %s"
                    % (self.name, e))
      metric_value = np.nan
    return metric_value 

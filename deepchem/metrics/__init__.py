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

import collections


import numpy as np
from sklearn import metrics

def compute_metric(num_tasks, y_true, y_pred, metric_str, threshold=0.5):
  """Compute a performance metric for each task.

  Args:
    y_true: A list of arrays containing true values for each task.
    y_pred: A list of arrays containing predicted values for each task.
    metric_str: String description of the metric to compute. Must be in
      metrics.METRICS.
    threshold: Float threshold to apply to probabilities for positive/negative
      class assignment.

  Returns:
    A numpy array containing metric values for each task.
  """
  computed_metrics = []
  for task in xrange(num_tasks):
    yt = y_true[task]
    yp = y_pred[task]
    try:
      metric_value = compute_metric(yt, yp, metric_str,
                                    threshold=threshold)
    except (AssertionError, ValueError) as e:
      warnings.warn('Error calculating metric %s for task %d: %s'
                    % (metric_str, task, e))
      metric_value = np.nan
    computed_metrics.append(metric_value)
  return computed_metrics

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


def compute_metric(y_true, y_pred, metric_str, threshold=0.5):
  """Compute a metric value.

  Args:
    y_true: A list of arrays containing true values for each task.
    y_pred: A list of arrays containing predicted values for each task.
    metric_str: String description of the metric to compute. Must be in
      biology_metrics.METRICS.
    threshold: Float threshold to apply to probabilities for positive/negative
      class assignment.

  Returns:
    Float metric value.

  Raises:
    NotImplementedError: If metric_str is not in METRICS.
  """
  if metric_str not in METRICS:
    raise NotImplementedError('Unsupported metric %s' % metric_str)
  metric_tuple = METRICS[metric_str]
  if metric_tuple.threshold:
    y_pred = np.greater(y_pred, threshold)
  return metric_tuple.func(y_true, y_pred)


class Metric(collections.namedtuple('MetricTuple', ['func', 'threshold'])):
  """A named tuple used to organize model evaluation metrics.

  Args:
    func: Function to call. Should take true and predicted values (in that
      order) and compute the metric.
    threshold: Boolean indicating whether float values should be converted to
      binary labels prior to computing the metric, e.g. accuracy.
  """

METRICS = {
  'accuracy': Metric(metrics.accuracy_score, True),
  'auc': Metric(metrics.roc_auc_score, False),
  'kappa': Metric(kappa_score, True),
  'r2': Metric(metrics.r2_score, False),
}

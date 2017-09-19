"""Evaluation metrics."""

import numpy as np
import warnings
from deepchem.utils.save import log
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from scipy.stats import pearsonr


def to_one_hot(y, n_classes=2):
  """Transforms label vector into one-hot encoding.

  Turns y into vector of shape [n_samples, 2] (assuming binary labels).

  y: np.ndarray
    A vector of shape [n_samples, 1]
  """
  n_samples = np.shape(y)[0]
  y_hot = np.zeros((n_samples, n_classes))
  y_hot[np.arange(n_samples), y.astype(np.int64)] = 1
  return y_hot


def from_one_hot(y, axis=1):
  """Transorms label vector from one-hot encoding.

  y: np.ndarray
    A vector of shape [n_samples, num_classes]
  """
  return np.argmax(y, axis=axis)


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


def balanced_accuracy_score(y, y_pred):
  """Computes balanced accuracy score."""
  num_positive = float(np.count_nonzero(y))
  num_negative = float(len(y) - num_positive)
  pos_weight = num_negative / num_positive
  weights = np.ones_like(y)
  weights[y != 0] = pos_weight
  return accuracy_score(y, y_pred, sample_weight=weights)


def pearson_r2_score(y, y_pred):
  """Computes Pearson R^2 (square of Pearson correlation)."""
  return pearsonr(y, y_pred)[0]**2


def prc_auc_score(y, y_pred):
  """Compute area under precision-recall curve"""
  assert y_pred.shape == y.shape
  assert y_pred.shape[1] == 2
  precision, recall, _ = precision_recall_curve(y[:, 1], y_pred[:, 1])
  return auc(recall, precision)


def rms_score(y_true, y_pred):
  """Computes RMS error."""
  return np.sqrt(mean_squared_error(y_true, y_pred))


def mae_score(y_true, y_pred):
  """Computes MAE."""
  return mean_absolute_error(y_true, y_pred)


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
  assert np.array_equal(
      np.unique(yt), [0,
                      1]), ('Class labels must be binary: %s' % np.unique(yt))
  observed_agreement = np.true_divide(
      np.count_nonzero(np.equal(yt, yp)), len(yt))
  expected_agreement = np.true_divide(
      np.count_nonzero(yt == 1) * np.count_nonzero(yp == 1) +
      np.count_nonzero(yt == 0) * np.count_nonzero(yp == 0), len(yt)**2)
  kappa = np.true_divide(observed_agreement - expected_agreement,
                         1.0 - expected_agreement)
  return kappa


class Metric(object):
  """Wrapper class for computing user-defined metrics."""

  def __init__(self,
               metric,
               task_averager=None,
               name=None,
               threshold=None,
               verbose=True,
               mode=None,
               compute_energy_metric=False):
    """
    Args:
      metric: function that takes args y_true, y_pred (in that order) and
              computes desired score.
      task_averager: If not None, should be a function that averages metrics
              across tasks. For example, task_averager=np.mean. If task_averager
              is provided, this task will be inherited as a multitask metric.
    """
    self.metric = metric
    self.task_averager = task_averager
    self.is_multitask = (self.task_averager is not None)
    if name is None:
      if not self.is_multitask:
        self.name = self.metric.__name__
      else:
        self.name = self.task_averager.__name__ + "-" + self.metric.__name__
    else:
      self.name = name
    self.verbose = verbose
    self.threshold = threshold
    if mode is None:
      if self.metric.__name__ in [
          "roc_auc_score", "matthews_corrcoef", "recall_score",
          "accuracy_score", "kappa_score", "precision_score",
          "balanced_accuracy_score", "prc_auc_score"
      ]:
        mode = "classification"
      elif self.metric.__name__ in [
          "pearson_r2_score", "r2_score", "mean_squared_error",
          "mean_absolute_error", "rms_score", "mae_score"
      ]:
        mode = "regression"
      else:
        raise ValueError("Must specify mode for new metric.")
    assert mode in ["classification", "regression"]
    self.mode = mode
    # The convention used is that the first task is the metric.
    # TODO(rbharath, joegomes): This doesn't seem like it should be hard-coded as
    # an option in the Metric class. Instead, this should be possible to move into
    # user-space as a custom task_averager function.
    self.compute_energy_metric = compute_energy_metric

  def compute_metric(self,
                     y_true,
                     y_pred,
                     w=None,
                     n_classes=2,
                     filter_nans=True,
                     per_task_metrics=False):
    """Compute a performance metric for each task.

    Parameters
    ----------
    y_true: np.ndarray
      An np.ndarray containing true values for each task.
    y_pred: np.ndarray
      An np.ndarray containing predicted values for each task.
    w: np.ndarray, optional
      An np.ndarray containing weights for each datapoint.
    n_classes: int, optional
      Number of classes in data for classification tasks.
    filter_nans: bool, optional
      Remove NaN values in computed metrics
    per_task_metrics: bool, optional
      If true, return computed metric for each task on multitask dataset.

    Returns
    -------
    A numpy nd.array containing metric values for each task.
    """
    if len(y_true.shape) > 1:
      n_samples, n_tasks = y_true.shape[0], y_true.shape[1]
    else:
      n_samples, n_tasks = y_true.shape[0], 1
    if self.mode == "classification":
      y_pred = np.reshape(y_pred, (n_samples, n_tasks, n_classes))
    else:
      y_pred = np.reshape(y_pred, (n_samples, n_tasks))
    y_true = np.reshape(y_true, (n_samples, n_tasks))
    if w is None or len(w) == 0:
      w = np.ones_like(y_true)
    assert y_true.shape[0] == y_pred.shape[0] == w.shape[0]
    computed_metrics = []
    for task in range(n_tasks):
      y_task = y_true[:, task]
      if self.mode == "regression":
        y_pred_task = y_pred[:, task]
      else:
        y_pred_task = y_pred[:, task, :]
      w_task = w[:, task]

      metric_value = self.compute_singletask_metric(y_task, y_pred_task, w_task)
      computed_metrics.append(metric_value)
    log("computed_metrics: %s" % str(computed_metrics), self.verbose)
    if n_tasks == 1:
      computed_metrics = computed_metrics[0]
    if not self.is_multitask:
      return computed_metrics
    else:
      if filter_nans:
        computed_metrics = np.array(computed_metrics)
        computed_metrics = computed_metrics[~np.isnan(computed_metrics)]
      if self.compute_energy_metric:
        # TODO(rbharath, joegomes): What is this magic number?
        force_error = self.task_averager(computed_metrics[1:]) * 4961.47596096
        print("Force error (metric: np.mean(%s)): %f kJ/mol/A" % (self.name,
                                                                  force_error))
        return computed_metrics[0]
      elif not per_task_metrics:
        return self.task_averager(computed_metrics)
      else:
        return self.task_averager(computed_metrics), computed_metrics

  def compute_singletask_metric(self, y_true, y_pred, w):
    """Compute a metric value.

    Args:
      y_true: A list of arrays containing true values for each task.
      y_pred: A list of arrays containing predicted values for each task.

    Returns:
      Float metric value.

    Raises:
      NotImplementedError: If metric_str is not in METRICS.
    """
    y_true = np.array(np.squeeze(y_true[w != 0]))
    y_pred = np.array(np.squeeze(y_pred[w != 0]))

    if len(y_true.shape) == 0:
      n_samples = 1
    else:
      n_samples = y_true.shape[0]
    # If there are no nonzero examples, metric is ill-defined.
    if not y_true.size:
      return np.nan

    y_true = np.reshape(y_true, (n_samples,))
    if self.mode == "classification":
      n_classes = y_pred.shape[-1]
      # TODO(rbharath): This has been a major source of bugs. Is there a more
      # robust characterization of which metrics require class-probs and which
      # don't?
      if "roc_auc_score" in self.name or "prc_auc_score" in self.name:
        y_true = to_one_hot(y_true).astype(int)
        y_pred = np.reshape(y_pred, (n_samples, n_classes))
      else:
        y_true = y_true.astype(int)
        # Reshape to handle 1-d edge cases
        y_pred = np.reshape(y_pred, (n_samples, n_classes))
        y_pred = from_one_hot(y_pred)
    else:
      y_pred = np.reshape(y_pred, (n_samples,))

    if self.threshold is not None:
      y_pred = np.greater(y_pred, threshold)
    try:
      metric_value = self.metric(y_true, y_pred)
    except (AssertionError, ValueError) as e:
      warnings.warn("Error calculating metric %s: %s" % (self.name, e))
      metric_value = np.nan
    return metric_value

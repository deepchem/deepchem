"""Evaluation metrics."""

import numpy as np
import warnings
import sklearn.metrics
import logging
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


def to_one_hot(y, n_classes=2):
  """Transforms label vector into one-hot encoding.

  Turns y into vector of shape `(n_samples, n_classes)` with a one-hot
  encoding. 

  Parameters
  ----------
  y: np.ndarray
    A vector of shape `(n_samples, 1)`

  Returns
  -------
  A numpy.ndarray of shape `(n_samples, n_classes)`.
  """
  n_samples = np.shape(y)[0]
  y_hot = np.zeros((n_samples, n_classes))
  y_hot[np.arange(n_samples), y.astype(np.int64)] = 1
  return y_hot


def from_one_hot(y, axis=1):
  """Transorms label vector from one-hot encoding.

  Parameters
  ----------
  y: np.ndarray
    A vector of shape `(n_samples, num_classes)`
  axis: int, optional (default 1)
    The axis with one-hot encodings to reduce on.

  Returns
  -------
  A numpy.ndarray of shape `(n_samples,)`
  """
  return np.argmax(y, axis=axis)


def _ensure_one_hot(y):
  """If neceessary, convert class labels to one-hot encoding."""
  if len(y.shape) == 1:
    return to_one_hot(y)
  return y


def _ensure_class_labels(y):
  """If necessary, convert one-hot encoding to class labels."""
  if len(y.shape) == 2:
    return from_one_hot(y)
  return y


def roc_auc_score(y, y_pred):
  """Area under the receiver operating characteristic curve."""
  if y.shape != y_pred.shape:
    y = _ensure_one_hot(y)
  return sklearn.metrics.roc_auc_score(y, y_pred)


def accuracy_score(y, y_pred):
  """Compute accuracy score

  Computes accuracy score for classification tasks. Works for both
  binary and multiclass classification.

  Parameters
  ----------
  y: np.ndarray
    Of shape `(N_samples,)`
  y_pred: np.ndarray
    Of shape `(N_samples,)`

  Returns
  -------
  score: float
    The fraction of correctly classified samples. A number between 0
    and 1.
  """
  y = _ensure_class_labels(y)
  y_pred = _ensure_class_labels(y_pred)
  return sklearn.metrics.accuracy_score(y, y_pred)


def balanced_accuracy_score(y, y_pred):
  """Computes balanced accuracy score."""
  num_positive = float(np.count_nonzero(y))
  num_negative = float(len(y) - num_positive)
  pos_weight = num_negative / num_positive
  weights = np.ones_like(y)
  weights[y != 0] = pos_weight
  return sklearn.metrics.balanced_accuracy_score(
      y, y_pred, sample_weight=weights)


def pearson_r2_score(y, y_pred):
  """Computes Pearson R^2 (square of Pearson correlation)."""
  return pearsonr(y, y_pred)[0]**2


def jaccard_index(y, y_pred):
  """Computes Jaccard Index which is the Intersection Over Union metric which is commonly used in image segmentation tasks

  Parameters
  ----------
  y: ground truth array
  y_pred: predicted array
  """
  return jaccard_score(y, y_pred)


def pixel_error(y, y_pred):
  """An error metric in case y, y_pred are images.

  Defined as 1 - the maximal F-score of pixel similarity, or squared
  Euclidean distance between the original and the result labels.

  Parameters
  ----------
  y: np.ndarray
    ground truth array
  y_pred: np.ndarray
    predicted array
  """
  return 1 - f1_score(y, y_pred)


def prc_auc_score(y, y_pred):
  """Compute area under precision-recall curve"""
  if y.shape != y_pred.shape:
    y = _ensure_one_hot(y)
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

  Parameters
  ----------
  y_true: np.ndarray
    Numpy array containing true values.
  y_pred: np.ndarray
    Numpy array containing predicted values.

  Returns
  -------
  kappa: np.ndarray
    Numpy array containing kappa for each classification task.

  Raises
  ------
  AssertionError: If y_true and y_pred are not the same size, or if
  class labels are not in [0, 1].
  """
  assert len(y_true) == len(y_pred), 'Number of examples does not match.'
  yt = np.asarray(y_true, dtype=int)
  yp = np.asarray(y_pred, dtype=int)
  assert np.array_equal(
      np.unique(yt),
      [0, 1]), ('Class labels must be binary: %s' % np.unique(yt))
  observed_agreement = np.true_divide(
      np.count_nonzero(np.equal(yt, yp)), len(yt))
  expected_agreement = np.true_divide(
      np.count_nonzero(yt == 1) * np.count_nonzero(yp == 1) +
      np.count_nonzero(yt == 0) * np.count_nonzero(yp == 0),
      len(yt)**2)
  kappa = np.true_divide(observed_agreement - expected_agreement,
                         1.0 - expected_agreement)
  return kappa


def bedroc_score(y_true, y_pred, alpha=20.0):
  """BEDROC metric implemented according to Truchon and Bayley that modifies
  the ROC score by allowing for a factor of early recognition

  Parameters
  ----------
  y_true (array_like):
    Binary class labels. 1 for positive class, 0 otherwise
  y_pred (array_like):
    Predicted labels
  alpha (float), default 20.0:
    Early recognition parameter

  Returns
  -------
  float: Value in [0, 1] that indicates the degree of early recognition

  Notes
  -----
  The original paper by Truchon et al. is located at
  https://pubs.acs.org/doi/pdf/10.1021/ci600426e
  """

  assert len(y_true) == len(y_pred), 'Number of examples do not match'

  assert np.array_equal(
      np.unique(y_true).astype(int),
      [0, 1]), ('Class labels must be binary: %s' % np.unique(y_true))

  from rdkit.ML.Scoring.Scoring import CalcBEDROC

  yt = np.asarray(y_true)
  yp = np.asarray(y_pred)

  yt = yt.flatten()
  yp = yp[:, 1].flatten()  # Index 1 because one_hot predictions

  scores = list(zip(yt, yp))
  scores = sorted(scores, key=lambda pair: pair[1], reverse=True)

  return CalcBEDROC(scores, 0, alpha)


class Metric(object):
  """Wrapper class for computing user-defined metrics.

  There are a variety of different metrics this class aims to support.
  At the most simple, metrics for classification and regression that
  assume that values to compare are scalars. More complicated, there
  may perhaps be two image arrays that need to be compared.

  The `Metric` class provides a wrapper for standardizing the API
  around different classes of metrics that may be useful for DeepChem
  models. The implementation provides a few non-standard conveniences
  such as built-in support for multitask and multiclass metrics, and
  support for multidimensional outputs.
  """

  def __init__(self,
               metric,
               task_averager=None,
               name=None,
               threshold=None,
               mode=None,
               compute_energy_metric=False):
    """
    Parameters
    ----------
    metric: function
      function that takes args y_true, y_pred (in that order) and
      computes desired score.
    task_averager: function, optional
      If not None, should be a function that averages metrics across
      tasks. For example, task_averager=np.mean. If task_averager is
      provided, this task will be inherited as a multitask metric.
    name: str, optional
      Name of this metric
    threshold: float, optional
      Used for binary metrics and is the threshold for the positive
      class
    mode: str, optional
      Must be either classification or regression.
    compute_energy_metric: TODO(rbharath): Should this be removed? 
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
    self.threshold = threshold
    if mode is None:
      if self.metric.__name__ in [
          "roc_auc_score", "matthews_corrcoef", "recall_score",
          "accuracy_score", "kappa_score", "precision_score",
          "balanced_accuracy_score", "prc_auc_score", "f1_score", "bedroc_score"
      ]:
        mode = "classification"
      elif self.metric.__name__ in [
          "pearson_r2_score", "r2_score", "mean_squared_error",
          "mean_absolute_error", "rms_score", "mae_score", "pearsonr"
      ]:
        mode = "regression"
      else:
        raise ValueError("Must specify mode for new metric.")
    assert mode in ["classification", "regression"]
    if self.metric.__name__ in [
        "accuracy_score", "balanced_accuracy_score", "recall_score",
        "matthews_corrcoef", "precision_score", "f1_score"
    ] and threshold is None:
      self.threshold = 0.5
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
    n_samples = y_true.shape[0]
    expected_dims = (3 if self.mode == "classification" else 2)
    if len(y_pred.shape) < expected_dims:
      n_tasks = 1
      y_true = np.expand_dims(y_true, 1)
      y_pred = np.expand_dims(y_pred, 1)
    else:
      n_tasks = y_pred.shape[1]
    if w is None or len(w) == 0:
      w = np.ones((n_samples, n_tasks))
    computed_metrics = []
    for task in range(n_tasks):
      y_task = y_true[:, task]
      y_pred_task = y_pred[:, task]
      if len(w.shape) == 1:
        w_task = w
      elif w.shape[1] == 1:
        w_task = w[:, 0]
      else:
        w_task = w[:, task]

      metric_value = self.compute_singletask_metric(y_task, y_pred_task, w_task)
      computed_metrics.append(metric_value)
    logger.info("computed_metrics: %s" % str(computed_metrics))
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

    Parameters
    ----------
    y_true: list
      A list of arrays containing true values for each task.
    y_pred: list
      A list of arrays containing predicted values for each task.

    Returns
    -------
    Float metric value.

    Raises
    ------
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
    if self.threshold is not None and len(y_pred.shape) == 1:
      y_pred = np.expand_dims(y_pred, 0)
    if self.threshold is not None:
      y_pred = y_pred[:, 1]
      y_pred = np.greater(y_pred, self.threshold)
    if len(y_true.shape) == 0:
      y_true = np.expand_dims(y_true, 0)
    if len(y_pred.shape) == 0:
      y_pred = np.expand_dims(y_pred, 0)
    try:
      metric_value = self.metric(y_true, y_pred)
    except (AssertionError, ValueError) as e:
      warnings.warn("Error calculating metric %s: %s" % (self.name, e))
      metric_value = np.nan
    return metric_value

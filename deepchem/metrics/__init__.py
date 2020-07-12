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


def threshold_predictions(y, threshold=0.5):
  """Threshold predictions from classification model.

  Parameters
  ----------
  y: np.ndarray
    Must have shape `(N, n_classes)` and be class probabilities.
  threshold: float, optional (Default 0.5)
    The threshold probability for the positive class. Note that this
    threshold will only be applied for binary classifiers (where
    `n_classes==2`). If specified for multiclass problems, will be
    ignored.

  Returns
  -------
  y_out: np.ndarray
    Of shape `(N,)` with class predictions as integers ranging from 0
    to `n_classes-1`.
  """
  if not isinstance(y, np.ndarray) or not len(y.shape) == 2:
    raise ValueError("y must be a ndarray of shape (N, n_classes)")
  N = y.shape[0]
  n_classes = y.shape[1]
  if not np.allclose(np.sum(y, axis=1), np.ones(N)):
    raise ValueError(
        "y must be a class probability matrix with rows summing to 1.")
  if n_classes != 2:
    y_out = np.argmax(y, axis=1)
    return y_out
  else:
    y_out = np.where(y[:, 1] >= threshold, np.ones(N), np.zeros(N))
    return y_out


def normalize_weight_shape(w, n_samples, n_tasks):
  """A utility function to correct the shape of the weight array.

  This utility function is used to normalize the shapes of a given
  weight array. 

  Parameters
  ----------
  w: np.ndarray
    `w` can be `None` or a scalar or a `np.ndarray` of shape
    `(n_samples,)` or of shape `(n_samples, n_tasks)`. If `w` is a
    scalar, it's assumed to be the same weight for all samples/tasks.
  n_samples: int
    The number of samples in the dataset. If `w` is not None, we should
    have `n_samples = w.shape[0]` if `w` is a ndarray
  n_tasks: int
    The number of tasks. If `w` is 2d ndarray, then we should have
    `w.shape[1] == n_tasks`.

  Returns
  -------
  w_out: np.ndarray
    Array of shape `(n_samples, n_tasks)`
  """
  if w is None:
    w_out = np.ones((n_samples, n_tasks))
  elif isinstance(w, np.ndarray):
    if len(w.shape) == 0:
      # scalar case
      w_out = w * np.ones((n_samples, n_tasks))
    elif len(w.shape) == 1:
      if len(w) != n_samples:
        raise ValueError("Length of w isn't n_samples")
      # per-example case
      # This is a little arcane but it repeats w across tasks.
      w_out = np.tile(w, (n_tasks, 1)).T
    elif len(w.shape) == 2:
      if w.shape == (n_samples, 1):
        # If w.shape == (n_samples, 1) handle it as 1D
        w = np.squeeze(w, axis=1)
        w_out = np.tile(w, (n_tasks, 1)).T
      elif w.shape != (n_samples, n_tasks):
        raise ValueError("Shape for w doens't match (n_samples, n_tasks)")
      else:
        # w.shape == (n_samples, n_tasks)
        w_out = w
    else:
      raise ValueError("w must be of dimension 1, 2, or 3")
  else:
    # scalar case
    w_out = w * np.ones((n_samples, n_tasks))
  return w_out


def normalize_prediction_shape(y, mode=None, n_classes=None):
  """A utility function to correct the shape of the input array.

  The metric computation classes expect that inputs for classification
  have the uniform shape `(N, n_tasks, n_classes)` and inputs for
  regression have the uniform shape `(N, n_tasks)`. This function
  normalizes the provided input array to have the desired shape.

  Examples
  --------
  >>> import numpy as np
  >>> y = np.random.rand(10)
  >>> y_out = normalize_prediction_shape(y, "regression")
  >>> y_out.shape
  (10, 1)

  Parameters
  ----------
  y: np.ndarray
    If `mode=="classification"`, `y` is an array of shape `(N,)` or
    `(N, n_classes)` or `(N, n_tasks, n_classes)`. If `y` is an array of shape
    `(N,)` in order to impute the number of classes correctly, `y`
    must take values from `0` to `n_classes-1` as integers. If
    `mode=="regression"`, `y` is an array of shape `(N,)` or `(N,
    n_tasks)`or `(N, n_tasks, 1)`. In the edge case where `N == 1`,
    `y` may be a scalar. If `mode` is None, then `y` can be of any
    shape and is returned unchanged.
  mode: str, optional (default None)
    If `mode` is "classification" or "regression", attempts to apply
    data transformations. For other modes, performs no transformations
    to data and returns as-is.
  n_classes: int, optional
    If specified use this as the number of classes. Else will try to
    impute it as `n_classes = max(y) + 1` for arrays and as
    `n_classes=2` for the case of scalars. Note this parameter only
    has value if `mode=="classification"`

  Returns
  -------
  y_out: np.ndarray
    If `mode=="classification"`, `y_out` is an array of shape `(N,
    n_tasks, n_classes)`. If `mode=="regression"`, `y_out` is an array
    of shape `(N, n_tasks)`.
  """
  if mode == "classification":
    if n_classes is None:
      if isinstance(y, np.ndarray):
        # Find number of classes. Note that `y` must have values in
        # range 0 to n_classes - 1
        n_classes = np.amax(y) + 1
      else:
        # scalar case
        n_classes = 2
    if isinstance(y, np.ndarray):
      if len(y.shape) == 1:
        # y_hot is of shape (N, n_classes)
        y_hot = to_one_hot(y, n_classes=n_classes)
        # Insert task dimension
        y_out = np.expand_dims(y_hot, 1)
      elif len(y.shape) == 2:
        # Insert a task dimension
        n_tasks = 1
        y_out = np.expand_dims(y, 1)
      elif len(y.shape) == 3:
        y_out = y
      else:
        raise ValueError(
            "y must be an array of dimension 1, 2, or 3 for classification problems."
        )
    else:
      # In this clase, y is a scalar. We assume that `y` is binary
      # since it's hard to do anything else in this case.
      y = np.array(y)
      y = np.reshape(y, (1,))
      y = to_one_hot(y, n_classes=n_classes)
      y_out = np.expand_dims(y, 1)
  elif mode == "regression":
    if isinstance(y, np.ndarray):
      if len(y.shape) == 1:
        # Insert a task dimension
        n_tasks = 1
        y_out = np.expand_dims(y, 1)
      elif len(y.shape) == 2:
        y_out = y
      elif len(y.shape) == 3:
        if y.shape[-1] != 1:
          raise ValueError(
              "y must a float scalar or a ndarray of shape `(N,)` or `(N, n_tasks)` or `(N, n_tasks, 1)` for regression problems."
          )
        y_out = np.squeeze(y, axis=-1)
      else:
        raise ValueError(
            "y must a float scalar or a ndarray of shape `(N,)` or `(N, n_tasks)` or `(N, n_tasks, 1)` for regression problems."
        )
    else:
      # In this clase, y is a scalar.
      try:
        y = float(y)
      except TypeError:
        raise ValueError(
            "y must a float scalar or a ndarray of shape `(N,)` or `(N, n_tasks)` or `(N, n_tasks, 1)` for regression problems."
        )
      y = np.array(y)
      y_out = np.reshape(y, (1, 1))
  else:
    # If mode isn't classification or regression don't perform any
    # transformations.
    raise ValueError("mode must be either classification or regression.")
  return y_out


def to_one_hot(y, n_classes=2):
  """Transforms label vector into one-hot encoding.

  Turns y into vector of shape `(n_samples, n_classes)` with a one-hot
  encoding. Assumes that `y` takes values from `0` to `n_classes - 1`.

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

  The `Metric` class provides a wrapper for standardizing the API
  around different classes of metrics that may be useful for DeepChem
  models. The implementation provides a few non-standard conveniences
  such as built-in support for multitask and multiclass metrics.

  There are a variety of different metrics this class aims to support.
  Metrics for classification and regression that assume that values to
  compare are scalars are supported.

  At present, this class doesn't support metric computation on models
  which don't present scalar outputs. For example, if you have a
  generative model which predicts images or molecules, you will need
  to write a custom evaluation and metric setup.
  """

  def __init__(self,
               metric,
               task_averager=None,
               name=None,
               threshold=None,
               mode=None,
               compute_energy_metric=None):
    """
    Parameters
    ----------
    metric: function
      Function that takes args y_true, y_pred (in that order) and
      computes desired score. If sample weights are to be considered,
      `metric` may take in an additional keyword argument
      `sample_weight`.
    task_averager: function, optional (default, np.mean)
      If not None, should be a function that averages metrics across
      tasks. 
    name: str, optional (default None)
      Name of this metric
    threshold: float, optional (default None) (DEPRECATED)
      Used for binary metrics and is the threshold for the positive
      class.
    mode: str, optional (default None)
      Should usually be "classification" or "regression."
    compute_energy_metric: bool, optional (default None) (DEPRECATED)
      Deprecated metric. Will be removed in a future version of
      DeepChem. Do not use.
    """
    if threshold is not None:
      logger.warn(
          "threshold is deprecated and will be removed in a future version of DeepChem. Set threshold in compute_metric instead"
      )
    if compute_energy_metric is not None:
      self.compute_energy_metric = compute_energy_metric
      logger.warn(
          "compute_energy_metric is deprecated and will be removed in a future version of DeepChem."
      )
    else:
      self.compute_energy_metric = False
    self.metric = metric
    if task_averager is None:
      self.task_averager = np.mean
    else:
      self.task_averager = task_averager
    if name is None:
      if task_averager is None:
        if hasattr(self.metric, '__name__'):
          self.name = self.metric.__name__
        else:
          self.name = "unknown metric"
      else:
        if hasattr(self.metric, '__name__'):
          self.name = task_averager.__name__ + "-" + self.metric.__name__
        else:
          self.name = "unknown metric"
    else:
      self.name = name
    if mode is None:
      # These are some smart defaults
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
        logger.info(
            "Could not detect mode of classifier. Check your results carefully."
        )
      self.mode = mode

  def compute_metric(self,
                     y_true,
                     y_pred,
                     w=None,
                     n_classes=2,
                     filter_nans=False,
                     per_task_metrics=False,
                     use_sample_weights=False,
                     threshold=None):
    """Compute a performance metric for each task.

    Parameters
    ----------
    y_true: np.ndarray
      An np.ndarray containing true values for each task. Must be of
      shape `(N, n_tasks, n_classes)` if a classification metric, else
      must be of shape `(N, n_tasks)` if a regression metric.
    y_pred: np.ndarray
      An np.ndarray containing predicted values for each task. Must be
      of shape `(N, n_tasks, n_classes)` if a classification metric,
      else must be of shape `(N, n_tasks)` if a regression metric.
    w: np.ndarray, optional
      An np.ndarray containing weights for each datapoint. If
      specified,  must be of shape `(N, n_tasks)`.
    n_classes: int, optional
      Number of classes in data for classification tasks.
    filter_nans: bool, optional (default False) (DEPRECATED)
      Remove NaN values in computed metrics
    per_task_metrics: bool, optional
      If true, return computed metric for each task on multitask dataset.
    use_sample_weights: bool, optional (default False)
      If set, use per-sample weights `w`.
    threshold: float or bool, optional (default None)
      If set, apply a thresholding operation to values. This option isj
      only sensible on classification tasks. If float, this will be
      applied as a binary classification value. If bool, then
      thresholding will be applied to a multiclass prediction and will
      pick the maximum probability class.

    Returns
    -------
    A numpy nd.array containing metric values for each task.
    """
    # TODO: How about non standard shapes?
    y_true = normalize_prediction_shape(
        y_true, mode=self.mode, n_classes=n_classes)
    y_pred = normalize_prediction_shape(
        y_pred, mode=self.mode, n_classes=n_classes)
    # This is safe now because of normalization above
    n_samples = y_true.shape[0]
    n_tasks = y_pred.shape[1]
    w = normalize_weight_shape(w, n_samples, n_tasks)
    computed_metrics = []
    for task in range(n_tasks):
      y_task = y_true[:, task]
      y_pred_task = y_pred[:, task]
      w_task = w[:, task]
      if threshold is not None:
        y_task = threshold_predictions(y_task, threshold=threshold)
        y_task = to_one_hot(y_task, n_classes=n_classes)
        y_pred_task = threshold_predictions(y_pred_task, threshold=threshold)
        y_pred_task = to_one_hot(y_pred_task, n_classes=n_classes)

      metric_value = self.compute_singletask_metric(
          y_task,
          y_pred_task,
          w_task,
          n_samples=n_samples,
          use_sample_weights=use_sample_weights)
      computed_metrics.append(metric_value)
    logger.info("computed_metrics: %s" % str(computed_metrics))
    if n_tasks == 1:
      computed_metrics = computed_metrics[0]

    # DEPRECATED. WILL BE REMOVED IN NEXT DEEPCHEM VERSION
    if filter_nans:
      computed_metrics = np.array(computed_metrics)
      computed_metrics = computed_metrics[~np.isnan(computed_metrics)]
    # DEPRECATED. WILL BE REMOVED IN NEXT DEEPCHEM VERSION
    if self.compute_energy_metric:
      force_error = self.task_averager(computed_metrics[1:]) * 4961.47596096
      logger.info("Force error (metric: np.mean(%s)): %f kJ/mol/A" %
                  (self.name, force_error))
      return computed_metrics[0]
    elif not per_task_metrics:
      return self.task_averager(computed_metrics)
    else:
      return self.task_averager(computed_metrics), computed_metrics

  def compute_singletask_metric(self,
                                y_true,
                                y_pred,
                                w=None,
                                n_samples=None,
                                use_sample_weights=False):
    """Compute a metric value.

    Parameters
    ----------
    y_true: `np.ndarray`
      True values array. This array must be of shape `(N,
      n_classes)` if classification and `(N,)` if regression.
    y_pred: `np.ndarray`
      Predictions array. This array must be of shape `(N, n_classes)`
      if classification and `(N,)` if regression.
    w: `np.ndarray`, optional (default None)
      Sample weight array. This array must be of shape `(N,)`
    n_samples: int, optional (default None)
      The number of samples in the dataset. This is `N`
    use_sample_weights: bool, optional (default False)
      If set, use per-sample weights `w`.

    Returns
    -------
    metric_value: float
      The computed value of the metric.
    """
    if n_samples is None:
      n_samples = len(y_true)
    if use_sample_weights:
      metric_value = self.metric(y_true, y_pred, sample_weight=w)
    else:
      metric_value = self.metric(y_true, y_pred)
    return metric_value

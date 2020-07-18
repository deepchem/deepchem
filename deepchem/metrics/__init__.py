"""Evaluation metrics."""

import numpy as np
import warnings
import sklearn.metrics
import logging
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


def threshold_predictions(y, threshold=None):
  """Threshold predictions from classification model.

  Parameters
  ----------
  y: np.ndarray
    Must have shape `(N, n_classes)` and be class probabilities.
  threshold: float, optional (default 0.5)
    The threshold probability for the positive class. Note that this
    threshold will only be applied for binary classifiers (where
    `n_classes==2`). If specified for multiclass problems, will be
    ignored. If `threshold` is None, and `n_classes==2` then a default
    threshold of 0.5 will be applied.

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
  if threshold is None and n_classes == 2:
    logger.info("Using default threshold of 0.5 for binary dataset.")
    threshold = 0.5
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

  Examples
  --------
  >>> import numpy as np
  >>> w_out = dc.metrics.normalize_weight_shape(None, n_samples, n_tasks)
  >>> (w_out == np.ones((n_samples, n_tasks))).all()
  True

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


def normalize_labels_shape(y, mode=None, n_tasks=None, n_classes=None):
  """A utility function to correct the shape of the labels.

  Parameters
  ----------
  y: np.ndarray
    `y` is an array of shape `(N,)` or `(N, n_tasks)` or `(N, n_tasks, 1)`.
  mode: str, optional (default None)
    If `mode` is "classification" or "regression", attempts to apply
    data transformations. 
  n_tasks: int, optional (default 1)
    The number of tasks this class is expected to handle.
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
  if n_tasks is None:
    raise ValueError("n_tasks must be specified")
  if mode not in ["classification", "regression"]:
    raise ValueError("mode must be either classification or regression.")
  if mode == "classification" and n_classes is None:
    raise ValueError("n_classes must be specified")
  if not isinstance(y, np.ndarray):
    raise ValueError("y must be a np.ndarray")
  # Handle n_classes/n_task shape ambiguity
  if mode == "classification" and len(y.shape) == 2:
    if n_classes == y.shape[1] and n_tasks != 1 and n_classes != n_tasks:
      raise ValueError("Shape of input doesn't match expected n_tasks=1")
    elif n_classes == y.shape[1] and n_tasks == 1:
      # Add in task dimension
      y = np.expand_dims(y, 1)
  if len(y.shape) == 1 and n_tasks != 1:
    raise ValueError("n_tasks must equal 1 for a 1D set of labels.")
  if (len(y.shape) == 2 or len(y.shape) == 3) and n_tasks != y.shape[1]:
    raise ValueError(
        "Shape of input doesn't match expected n_tasks=%d" % n_tasks)
  if len(y.shape) >= 4:
    raise ValueError(
        "Labels y must be a float scalar or a ndarray of shape `(N,)` or `(N, n_tasks)` or `(N, n_tasks, 1)` for regression problems and of shape `(N,)` or `(N, n_tasks)` or `(N, n_tasks, 1)` for classification problems"
    )
  if len(y.shape) == 1:
    # Insert a task dimension (we know n_tasks=1 from above0
    y_out = np.expand_dims(y, 1)
  elif len(y.shape) == 2:
    y_out = y
  elif len(y.shape) == 3:
    # If 3D and last dimension isn't 1, assume this is one-hot encoded and return as-is.
    if y.shape[-1] != 1:
      return y
    y_out = np.squeeze(y, axis=-1)
  # Handle classification. We need to convert labels into one-hot
  # representation.
  if mode == "classification":
    all_y_task = []
    for task in range(n_tasks):
      y_task = y_out[:, task]
      y_hot = to_one_hot(y_task, n_classes=n_classes)
      y_hot = np.expand_dims(y_hot, 1)
      all_y_task.append(y_hot)
    y_out = np.concatenate(all_y_task, axis=1)
  return y_out


def normalize_prediction_shape(y, mode=None, n_tasks=None, n_classes=None):
  """A utility function to correct the shape of provided predictions.

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
    `(N, n_tasks)` or `(N, n_tasks, n_classes)`. If
    `mode=="regression"`, `y` is an array of shape `(N,)` or `(N,
    n_tasks)`or `(N, n_tasks, 1)`. 
  mode: str, optional (default None)
    If `mode` is "classification" or "regression", attempts to apply
    data transformations. 
  n_tasks: int, optional (default 1)
    The number of tasks this class is expected to handle.
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
  if n_tasks is None:
    raise ValueError("n_tasks must be specified")
  if mode == "classification" and n_classes is None:
    raise ValueError("n_classes must be specified")
  if not isinstance(y, np.ndarray):
    raise ValueError("y must be a np.ndarray")
  # Handle n_classes/n_task shape ambiguity
  if mode == "classification" and len(y.shape) == 2:
    if n_classes == y.shape[1] and n_tasks != 1 and n_classes != n_tasks:
      raise ValueError("Shape of input doesn't match expected n_tasks=1")
    elif n_classes == y.shape[1] and n_tasks == 1:
      # Add in task dimension
      y = np.expand_dims(y, 1)
  if (len(y.shape) == 2 or len(y.shape) == 3) and n_tasks != y.shape[1]:
    raise ValueError(
        "Shape of input doesn't match expected n_tasks=%d" % n_tasks)
  if len(y.shape) >= 4:
    raise ValueError(
        "Predictions y must be a float scalar or a ndarray of shape `(N,)` or `(N, n_tasks)` or `(N, n_tasks, 1)` for regression problems and of shape `(N,)` or `(N, n_tasks)` or `(N, n_tasks, n_classes)` for classification problems"
    )
  if mode == "classification":
    if n_classes is None:
      raise ValueError("n_classes must be specified.")
    if len(y.shape) == 1 or len(y.shape) == 2:
      # Make everything 2D so easy to handle
      if len(y.shape) == 1:
        y = y[:, np.newaxis]
      # Handle each task separately.
      all_y_task = []
      for task in range(n_tasks):
        y_task = y[:, task]
        # Handle continuous class probabilites of positive class for binary
        if len(np.unique(y_task)) > n_classes:
          if n_classes > 2:
            raise ValueError(
                "Cannot handle continuous probabilities for multiclass problems. Need a per-class probability"
            )
          # Fill in class 0 probabilities
          y_task = np.array([1 - y_task, y_task]).T
          # Add a task dimension to concatenate on
          y_task = np.expand_dims(y_task, 1)
          all_y_task.append(y_task)
        # Handle binary labels
        else:
          # make y_hot of shape (N, n_classes)
          y_task = to_one_hot(y_task, n_classes=n_classes)
          # Add a task dimension to concatenate on
          y_task = np.expand_dims(y_task, 1)
          all_y_task.append(y_task)
      y_out = np.concatenate(all_y_task, axis=1)
    elif len(y.shape) == 3:
      y_out = y
  elif mode == "regression":
    if len(y.shape) == 1:
      # Insert a task dimension
      y_out = np.expand_dims(y, 1)
    elif len(y.shape) == 2:
      y_out = y
    elif len(y.shape) == 3:
      if y.shape[-1] != 1:
        raise ValueError(
            "y must be a float scalar or a ndarray of shape `(N,)` or `(N, n_tasks)` or `(N, n_tasks, 1)` for regression problems."
        )
      y_out = np.squeeze(y, axis=-1)
  else:
    raise ValueError("mode must be either classification or regression.")
  return y_out


def handle_classification_mode(y,
                               classification_handling_mode=None,
                               threshold_value=None):
  """Handle classification mode.

  Transform predictions so that they have the correct classification mode.

  Parameters
  ----------
  y: np.ndarray
    Must be of shape `(N, n_tasks, n_classes)`
  classification_handling_mode: str, optional (default None)
    DeepChem models by default predict class probabilities for
    classification problems. This means that for a given singletask
    prediction, after shape normalization, the DeepChem prediction will be a
    numpy array of shape `(N, n_classes)` with class probabilities.
    `classification_handling_mode` is a string that instructs this method
    how to handle transforming these probabilities. It can take on the
    following values:

    - None: default value. Pass in `y_pred` directy into `self.metric`.
    - "threshold": Use `threshold_predictions` to threshold `y_pred`. Use
      `threshold_value` as the desired threshold.
    - "threshold-one-hot": Use `threshold_predictions` to threshold `y_pred`
      using `threshold_values`, then apply `to_one_hot` to output.
  threshold_value: float, optional (default None)
    If set, and `classification_handling_mode` is "threshold" or
    "threshold-one-hot" apply a thresholding operation to values with this
    threshold. This option isj only sensible on binary classification tasks.
    If float, this will be applied as a binary classification value.

  Returns
  -------
  y_out: np.ndarray
    If `classification_handling_mode` is None, then of shape `(N, n_tasks, n_classes)`. If `classification_handling_mode` is "threshold", then of shape `(N, n_tasks)`. If `classification_handling_mode is "threshold-one-hot", then of shape `(N, n_tasks, n_classes)"
  """
  if len(y.shape) != 3:
    raise ValueError("y must be of shape (N, n_tasks, n_classes)")
  N, n_tasks, n_classes = y.shape
  if classification_handling_mode is None:
    return y
  elif classification_handling_mode == "threshold":
    thresholded = []
    for task in range(n_tasks):
      task_array = y[:, task, :]
      # Now of shape (N,)
      task_array = threshold_predictions(task_array, threshold_value)
      # Now of shape (N, 1)
      task_array = np.expand_dims(task_array, 1)
      thresholded.append(task_array)
    # Returns shape (N, n_tasks)
    return np.concatenate(thresholded, axis=1)
  elif classification_handling_mode == "threshold-one-hot":
    thresholded = []
    for task in range(n_tasks):
      task_array = y[:, task, :]
      # Now of shape (N,)
      task_array = threshold_predictions(task_array, threshold_value)
      # Now of shape (N, n_classes)
      task_array = to_one_hot(task_array, n_classes=n_classes)
      # Now of shape (N, 1, n_classes)
      task_array = np.expand_dims(task_array, 1)
      thresholded.append(task_array)
    # Returns shape (N, n_tasks, n_classes)
    return np.concatenate(thresholded, axis=1)
  else:
    raise ValueError(
        "classification_handling_mode must be one of None, threshold, threshold-one-hot"
    )


def to_one_hot(y, n_classes=2):
  """Transforms label vector into one-hot encoding.

  Turns y into vector of shape `(N, n_classes)` with a one-hot
  encoding. Assumes that `y` takes values from `0` to `n_classes - 1`.


  Parameters
  ----------
  y: np.ndarray
    A vector of shape `(N,)` or `(N, 1)`

  Returns
  -------
  A numpy.ndarray of shape `(N, n_classes)`.
  """
  if len(y.shape) > 2:
    raise ValueError("y must be a vector of shape (N,) or (N, 1)")
  if len(y.shape) == 2 and y.shape[1] != 1:
    raise ValueError("y must be a vector of shape (N,) or (N, 1)")
  if len(np.unique(y)) > n_classes:
    raise ValueError("y has more than n_class unique elements.")
  N = np.shape(y)[0]
  y_hot = np.zeros((N, n_classes))
  y_hot[np.arange(N), y.astype(np.int64)] = 1
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


def pearson_r2_score(y, y_pred):
  """Computes Pearson R^2 (square of Pearson correlation).

  Parameters
  ----------
  y: 1D array
    Of shape `(N,)
  y_pred: 1D array
    Of shape `(N,)`

  Returns
  -------
  Float value of the Pearson-R^2 score.
  """
  return pearsonr(y, y_pred)[0]**2


def jaccard_index(y, y_pred):
  """Computes Jaccard Index which is the Intersection Over Union metric which is commonly used in image segmentation tasks

  DEPRECATED: WILL BE REMOVED IN A FUTURE VERSION OF DEEEPCHEM. USE `jaccard_score` instead.

  Parameters
  ----------
  y: np.ndarray
    ground truth array
  y_pred: np.ndarray
    predicted array

  Returns
  -------
  score: float
    The jaccard index. A number between 0 and 1.
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

  Returns
  -------
  score: float
    The pixel-error. A number between 0 and 1.
  """
  return 1 - f1_score(y, y_pred)


def prc_auc_score(y, y_pred):
  """Compute area under precision-recall curve

  Parameters
  ----------
  y: np.ndarray
    Of shape `(N, n_classes)` or `(N,)` with true labels 
  y_pred: np.ndarray
    Of shape `(N, n_classes)` with class probabilities.

  Returns
  -------
  The area under the precision-recall curve. A number between 0 and 1.
  """
  precision, recall, _ = precision_recall_curve(y[:, 1], y_pred[:, 1])
  return auc(recall, precision)


def rms_score(y_true, y_pred):
  """Computes RMS error."""
  return np.sqrt(mean_squared_error(y_true, y_pred))


def mae_score(y_true, y_pred):
  """Computes MAE."""
  return mean_absolute_error(y_true, y_pred)


# kappa_score is an alias for `sklearn.metrics.cohen_kappa_score`
kappa_score = cohen_kappa_score


def bedroc_score(y_true, y_pred, alpha=20.0):
  """BEDROC metric implemented according to Truchon and Bayley that modifies
  the ROC score by allowing for a factor of early recognition

  Parameters
  ----------
  y_true: array_like
    Binary class labels. 1 for positive class, 0 otherwise
  y_pred: array_like
    Predicted labels
  alpha: float, optional (default 20.0)
    Early recognition parameter

  Returns
  -------
  float: Value in [0, 1] that indicates the degree of early recognition

  Note
  ----
  This function requires rdkit to be installed.

  References
  ----------
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
               n_tasks=None,
               classification_handling_mode=None,
               threshold_value=None,
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
    n_tasks: int, optional (default 1)
      The number of tasks this class is expected to handle.
    classification_handling_mode: str, optional (default None)
      DeepChem models by default predict class probabilities for
      classification problems. This means that for a given singletask
      prediction, after shape normalization, the DeepChem prediction will be a
      numpy array of shape `(N, n_classes)` with class probabilities.
      `classification_handling_mode` is a string that instructs this method
      how to handle transforming these probabilities. It can take on the
      following values:
 
      - None: default value. Pass in `y_pred` directy into `self.metric`.
      - "threshold": Use `threshold_predictions` to threshold `y_pred`. Use
        `threshold_value` as the desired threshold.
      - "threshold-one-hot": Use `threshold_predictions` to threshold `y_pred`
        using `threshold_values`, then apply `to_one_hot` to output.
    threshold_value: float, optional (default None)
      If set, and `classification_handling_mode` is "threshold" or
      "threshold-one-hot" apply a thresholding operation to values with this
      threshold. This option isj only sensible on binary classification tasks.
      If float, this will be applied as a binary classification value.
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
          "roc_auc_score",
          "matthews_corrcoef",
          "recall_score",
          "accuracy_score",
          "kappa_score",
          "cohen_kappa_score",
          "precision_score",
          "balanced_accuracy_score",
          "prc_auc_score",
          "f1_score",
          "bedroc_score",
          "jaccard_score",
          "jaccard_index",
          "pixel_error",
      ]:
        mode = "classification"
        # These are some smart defaults corresponding to sklearn's required
        # behavior
        if classification_handling_mode is None:
          if self.metric.__name__ in [
              "matthews_corrcoef", "cohen_kappa_score", "kappa_score",
              "balanced_accuracy_score", "recall_score", "jaccard_score",
              "jaccard_index", "pixel_error", "f1_score"
          ]:
            classification_handling_mode = "threshold"
          elif self.metric.__name__ in [
              "accuracy_score", "precision_score", "bedroc_score"
          ]:
            classification_handling_mode = "threshold-one-hot"
          elif self.metric.__name__ in ["roc_auc_score", "prc_auc_score"]:
            classification_handling_mode = None
      elif self.metric.__name__ in [
          "pearson_r2_score", "r2_score", "mean_squared_error",
          "mean_absolute_error", "rms_score", "mae_score", "pearsonr"
      ]:
        mode = "regression"
      else:
        raise ValueError(
            "Please specify the mode of this metric. mode must be 'regression' or 'classification'"
        )

    self.mode = mode
    self.n_tasks = n_tasks
    if classification_handling_mode not in [
        None, "threshold", "threshold-one-hot"
    ]:
      raise ValueError(
          "classification_handling_mode must be one of None, 'threshold', 'threshold_one_hot'"
      )
    self.classification_handling_mode = classification_handling_mode
    self.threshold_value = threshold_value

  def compute_metric(self,
                     y_true,
                     y_pred,
                     w=None,
                     n_tasks=None,
                     n_classes=2,
                     filter_nans=False,
                     per_task_metrics=False,
                     use_sample_weights=False,
                     **kwargs):
    """Compute a performance metric for each task.

    Parameters
    ----------
    y_true: np.ndarray
      An np.ndarray containing true values for each task. Must be of shape
      `(N,)` or `(N, n_tasks)` or `(N, n_tasks, n_classes)` if a
      classification metric. If of shape `(N, n_tasks)` values can either be
      class-labels or probabilities of the positive class for binary
      classification problems. If a regression problem, must be of shape
      `(N,)` or `(N, n_tasks)` or `(N, n_tasks, 1)` if a regression metric.
    y_pred: np.ndarray
      An np.ndarray containing predicted values for each task. Must be
      of shape `(N, n_tasks, n_classes)` if a classification metric,
      else must be of shape `(N, n_tasks)` if a regression metric.
    w: np.ndarray, optional
      An np.ndarray containing weights for each datapoint. If
      specified,  must be of shape `(N, n_tasks)`.
    n_tasks: int, optional (default 1)
      The number of tasks this class is expected to handle.
    n_classes: int, optional
      Number of classes in data for classification tasks.
    filter_nans: bool, optional (default False) (DEPRECATED)
      Remove NaN values in computed metrics
    per_task_metrics: bool, optional
      If true, return computed metric for each task on multitask dataset.
    use_sample_weights: bool, optional (default False)
      If set, use per-sample weights `w`.
    kwargs: dict
      Will be passed on to self.metric

    Returns
    
    A numpy nd.array containing metric values for each task.
    """
    # Attempt some limited shape imputation to find n_tasks
    if n_tasks is None:
      if self.n_tasks is None and isinstance(y_true, np.ndarray):
        if len(y_true.shape) == 1:
          n_tasks = 1
        elif len(y_true.shape) >= 2:
          n_tasks = y_true.shape[1]
      else:
        n_tasks = self.n_tasks
    y_true = normalize_labels_shape(
        y_true, mode=self.mode, n_tasks=n_tasks, n_classes=n_classes)
    y_pred = normalize_prediction_shape(
        y_pred, mode=self.mode, n_tasks=n_tasks, n_classes=n_classes)
    if self.mode == "classification":
      y_true = handle_classification_mode(
          y_true, self.classification_handling_mode, self.threshold_value)
      y_pred = handle_classification_mode(
          y_pred, self.classification_handling_mode, self.threshold_value)
    n_samples = y_true.shape[0]
    w = normalize_weight_shape(w, n_samples, n_tasks)
    computed_metrics = []
    for task in range(n_tasks):
      y_task = y_true[:, task]
      y_pred_task = y_pred[:, task]
      w_task = w[:, task]

      metric_value = self.compute_singletask_metric(
          y_task,
          y_pred_task,
          w_task,
          n_samples=n_samples,
          use_sample_weights=use_sample_weights,
          **kwargs)
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
                                use_sample_weights=False,
                                **kwargs):
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
    n_samples: int, optional (default None) (DEPRECATED)
      The number of samples in the dataset. This is `N`. This argument is
      ignored.
    use_sample_weights: bool, optional (default False)
      If set, use per-sample weights `w`.
    kwargs: dict
      Will be passed on to self.metric

    Returns
    -------
    metric_value: float
      The computed value of the metric.
    """
    if n_samples != None:
      logger.warning("n_samples is a deprecated argument which is ignored.")
    # Attempt to convert both into the same type
    if self.mode == "regression":
      if len(y_true.shape) != 1 or len(
          y_pred.shape) != 1 or len(y_true) != len(y_pred):
        raise ValueError(
            "For regression metrics, y_true and y_pred must both be of shape (N,)"
        )
    elif self.mode == "classification":
      pass
      #if len(y_true.shape) != 2 or len(y_pred.shape) != 2 or y_true.shape != y_pred.shape:
      #  raise ValueError("For classification metrics, y_true and y_pred must both be of shape (N, n_classes)")
    else:
      raise ValueError(
          "Only classification and regression are supported for metrics calculations."
      )
    if use_sample_weights:
      metric_value = self.metric(y_true, y_pred, sample_weight=w, **kwargs)
    else:
      metric_value = self.metric(y_true, y_pred, **kwargs)
    return metric_value

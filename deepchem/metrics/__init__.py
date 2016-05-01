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

  def __init__(self, metric, task_averager=None, name=None, threshold=None,
               verbosity=None, mode=None):
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
    self.verbosity = verbosity
    self.threshold = threshold
    if mode is None:
      if self.name in ["roc_auc_score", "matthews_corrcoef", "recall_score",
                       "accuracy_score", "kappa_score"]:
        mode = "classification"
      elif self.name in ["r2_score", "mean_squared_error",
                         "mean_absolute_error"]:
        mode = "regression"
      else:
        raise ValueError("Must specify mode for new metric.")
    assert mode in ["classification", "regression"]
    self.mode = mode

  def compute_metric(self, y_true, y_pred, w, n_classes=2):
    """Compute a performance metric for each task.

    Args:
      y_true: A list of arrays containing true values for each task.
      y_pred: A list of arrays containing predicted values for each task.
      metric: Must be a class that inherits from Metric 

    Returns:
      A numpy array containing metric values for each task.
    """
    assert y_true.shape[0] == y_pred.shape[0] == w.shape[0]
    n_samples, n_tasks = y_true.shape[0], y_true.shape[1] 
    if self.mode == "classification":
      y_pred = np.reshape(y_pred, (n_samples, n_tasks, n_classes))
    else:
      y_pred = np.reshape(y_pred, (n_samples, n_tasks))
    computed_metrics = []
    for task in xrange(n_tasks):
      y_task = y_true[:, task]
      if self.mode == "regression":
        y_pred_task = y_pred[:, task]
      else:
        y_pred_task = y_pred[:, task, :]
      w_task = w[:, task]
    
      metric_value = self.compute_singletask_metric(
          y_task, y_pred_task, w_task)
      computed_metrics.append(metric_value)
    log("computed_metrics: %s" % str(computed_metrics), self.verbosity)
    if n_tasks == 1:
      computed_metrics = computed_metrics[0]
    if not self.is_multitask:
      return computed_metrics
    else:
      return self.task_averager(computed_metrics)

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
      if "roc_auc_score" in self.name:
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
      warnings.warn("Error calculating metric %s: %s"
                    % (self.name, e))
      metric_value = np.nan
    return metric_value 

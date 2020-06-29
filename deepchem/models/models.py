"""
Contains an abstract base class that supports different ML models.
"""
import sys
import numpy as np
import pandas as pd
import joblib
import os
import shutil
import tempfile
import sklearn
import logging
from sklearn.base import BaseEstimator

from deepchem.data import Dataset, pad_features
from deepchem.trans import undo_transforms
from deepchem.utils.save import load_from_disk
from deepchem.utils.save import save_to_disk
from deepchem.utils.save import log
from deepchem.utils.evaluate import Evaluator

logger = logging.getLogger(__name__)


class Model(BaseEstimator):
  """
  Abstract base class for DeepChem models.
  """

  def __init__(self, model_instance=None, model_dir=None, **kwargs):
    """Example constructor for a model.

    This is intended only for convenience of subclass implementations
    and should not be invoked directly.

    Parameters:
    -----------
    model_instance: object
      Wrapper around ScikitLearn/Keras/Tensorflow model object.
    model_dir: str, optional (default None)
      Path to directory where model will be stored. If not specified,
      model will be stored in a temporary directory.
    """
    if self.__class__.__name__ == "Model":
      raise ValueError(
          "This constructor is for an abstract class and should never be called directly. Can only call from subclass constructors."
      )
    self.model_dir_is_temp = False
    if model_dir is not None:
      if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    else:
      model_dir = tempfile.mkdtemp()
      self.model_dir_is_temp = True
    self.model_dir = model_dir
    self.model_instance = model_instance
    self.model_class = model_instance.__class__

  def __del__(self):
    if 'model_dir_is_temp' in dir(self) and self.model_dir_is_temp:
      shutil.rmtree(self.model_dir)

  def fit_on_batch(self, X, y, w):
    """
    Updates existing model with new information.
    """
    raise NotImplementedError(
        "Each model is responsible for its own fit_on_batch method.")

  def predict_on_batch(self, X, **kwargs):
    """
    Makes predictions on given batch of new data.

    Parameters
    ----------
    X: np.ndarray
      Features
    """
    raise NotImplementedError(
        "Each model is responsible for its own predict_on_batch method.")

  def reload(self):
    """
    Reload trained model from disk.
    """
    raise NotImplementedError(
        "Each model is responsible for its own reload method.")

  @staticmethod
  def get_model_filename(model_dir):
    """
    Given model directory, obtain filename for the model itself.
    """
    return os.path.join(model_dir, "model.joblib")

  @staticmethod
  def get_params_filename(model_dir):
    """
    Given model directory, obtain filename for the model itself.
    """
    return os.path.join(model_dir, "model_params.joblib")

  def save(self):
    """Dispatcher function for saving.

    Each subclass is responsible for overriding this method.
    """
    raise NotImplementedError

  def fit(self, dataset, *args, **kwargs):
    """Fits a model on data in a Dataset object.

    This is an abstract method that should never be invoked directly.
    Concrete subclasses of this class will overwrite the `fit()`
    method. Arguments here are only provided for a suggestion of the
    API for concrete subclasses.
    """
    raise NotImplementedError

  def predict(self, dataset, transformers=[], batch_size=None):
    """Uses self to make predictions on provided Dataset object.

    Parameters
    ----------
    dataset: dc.data.Dataset
      Dataset object.
    transformers: list
      List of deepchem.transformers.Transformer
    batch_size: int, optional (default None)
      The batch size to evaluate models at.

    Returns
    -------
      y_pred: numpy ndarray of shape (n_samples,)
    """
    y_preds = []
    n_tasks = self.get_num_tasks()
    ind = 0

    for (X_batch, _, _, ids_batch) in dataset.iterbatches(
        batch_size, deterministic=True):
      n_samples = len(X_batch)
      y_pred_batch = self.predict_on_batch(X_batch)
      # Discard any padded predictions
      y_pred_batch = y_pred_batch[:n_samples]
      y_pred_batch = undo_transforms(y_pred_batch, transformers)
      y_preds.append(y_pred_batch)
    y_pred = np.concatenate(y_preds)
    return y_pred

  def evaluate(self, dataset, metrics, transformers=[], **kwargs):
    """Evaluates the performance of this model on specified dataset.

    This function uses `Evaluator` under the hood to perform model
    evaluation. As a result, it inherits the same limitations of
    `Evaluator`. Namely, that only regression and classification
    models can be evaluated in this fashion. For generator models, you
    will need to overwrite this method to perform a custom evaluation.

    Keyword arguments specified here will be passed to
    `Evalautor.compute_model_performance`.

    Parameters
    ----------
    dataset: `dc.data.Dataset`
      Dataset object.
    metrics: dc.metrics.Metric/list[dc.metrics.Metric]/function
      The set of metrics provided. This class attempts to do some
      intelligent handling of input. If a single `dc.metrics.Metric`
      object is provided or a list is provided, it will evaluate
      `self.model` on these metrics. If a function is provided, it is
      assumed to be a metric function that this method will attempt to
      wrap in a `dc.metrics.Metric` object. A metric function must
      accept two arguments, `y_true, y_pred` both of which are
      `np.ndarray` objects and return a floating point score. The
      metric function may also accept a keyword argument
      `sample_weight` to account for per-sample weights.
    transformers: list
      List of `dc.trans.Transformer` objects. These transformations
      must have been applied to `dataset` previously. The dataset will
      be untransformed for metric evaluation.

    Returns
    -------
    multitask_scores: dict
      Dictionary mapping names of metrics to metric scores.
    all_task_scores: dict, optional
      If `per_task_metrics == True` is passed as a keyword argument,
      then returns a second dictionary of scores for each task
      separately.
    """
    evaluator = Evaluator(self, dataset, transformers)
    return evaluator.compute_model_performance(metrics, **kwargs)

  def get_task_type(self):
    """
    Currently models can only be classifiers or regressors.
    """
    raise NotImplementedError

  def get_num_tasks(self):
    """
    Get number of tasks.
    """
    raise NotImplementedError

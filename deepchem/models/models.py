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
from sklearn.base import BaseEstimator

import logging
from deepchem.data import Dataset, pad_features
from deepchem.metrics import Metric
from deepchem.trans import Transformer, undo_transforms
from deepchem.utils.save import load_from_disk
from deepchem.utils.save import save_to_disk
from deepchem.utils.evaluate import Evaluator

from typing import Any, Dict, List, Optional, Sequence
from deepchem.utils.typing import OneOrMany

logger = logging.getLogger(__name__)


class Model(BaseEstimator):
  """
  Abstract base class for different ML models.
  """

  def __init__(self,
               model_instance: Optional[Any] = None,
               model_dir: Optional[str] = None,
               **kwargs) -> None:
    """Abstract class for all models.

    Parameters
    -----------
    model_instance: object
      Wrapper around ScikitLearn/Keras/Tensorflow model object.
    model_dir: str
      Path to directory where model will be stored.
    """
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

  def fit_on_batch(self, X: Sequence, y: Sequence, w: Sequence) -> float:
    """Perform a single step of training.

    Parameters
    ----------
    X: ndarray
      the inputs for the batch
    y: ndarray
      the labels for the batch
    w: ndarray
      the weights for the batch

    Returns
    -------
    the loss on the batch
    """
    raise NotImplementedError(
        "Each model is responsible for its own fit_on_batch method.")

  def predict_on_batch(self, X: Sequence):
    """
    Makes predictions on given batch of new data.

    Parameters
    ----------
    X: np.ndarray
      Features
    """
    raise NotImplementedError(
        "Each model is responsible for its own predict_on_batch method.")

  def reload(self) -> None:
    """
    Reload trained model from disk.
    """
    raise NotImplementedError(
        "Each model is responsible for its own reload method.")

  @staticmethod
  def get_model_filename(model_dir: str) -> str:
    """
    Given model directory, obtain filename for the model itself.
    """
    return os.path.join(model_dir, "model.joblib")

  @staticmethod
  def get_params_filename(model_dir: str) -> str:
    """
    Given model directory, obtain filename for the model itself.
    """
    return os.path.join(model_dir, "model_params.joblib")

  def save(self) -> None:
    """Dispatcher function for saving.

    Each subclass is responsible for overriding this method.
    """
    raise NotImplementedError

  def fit(self, dataset: Dataset, nb_epoch: int = 10) -> float:
    """
    Fits a model on data in a Dataset object.

    Parameters
    ----------
    dataset: Dataset
      the Dataset to train on
    nb_epoch: int
      the number of epochs to train for

    Returns
    -------
    the average loss over the most recent epoch
    """
    # TODO(rbharath/enf): We need a structured way to deal with potential GPU
    #                     memory overflows.
    for epoch in range(nb_epoch):
      logger.info("Starting epoch %s" % str(epoch + 1))
      losses = []
      for (X_batch, y_batch, w_batch, ids_batch) in dataset.iterbatches():
        losses.append(self.fit_on_batch(X_batch, y_batch, w_batch))
      logger.info(
          "Avg loss for epoch %d: %f" % (epoch + 1, np.array(losses).mean()))
    return np.array(losses).mean()

  def predict(self, dataset: Dataset,
              transformers: List[Transformer] = []) -> OneOrMany[np.ndarray]:
    """
    Uses self to make predictions on provided Dataset object.


    Parameters
    ----------
    dataset: dc.data.Dataset
      Dataset to make prediction on
    transformers: list of dc.trans.Transformers
      Transformers that the input data has been transformed by.  The output
      is passed through these transformers to undo the transformations.

    Returns
    -------
    a NumPy array of the model produces a single output, or a list of arrays
    if it produces multiple outputs
    """
    y_preds = []
    n_tasks = self.get_num_tasks()
    ind = 0

    for (X_batch, _, _, ids_batch) in dataset.iterbatches(deterministic=True):
      n_samples = len(X_batch)
      y_pred_batch = self.predict_on_batch(X_batch)
      # Discard any padded predictions
      y_pred_batch = y_pred_batch[:n_samples]
      y_pred_batch = undo_transforms(y_pred_batch, transformers)
      y_preds.append(y_pred_batch)
    y_pred = np.concatenate(y_preds)
    return y_pred

  def evaluate(self,
               dataset: Dataset,
               metrics: List[Metric],
               transformers: List[Transformer] = [],
               per_task_metrics: bool = False):
    """
    Evaluates the performance of this model on specified dataset.

    Parameters
    ----------
    dataset: dc.data.Dataset
      Dataset object.
    metric: deepchem.metrics.Metric
      Evaluation metric
    transformers: list
      List of deepchem.transformers.Transformer
    per_task_metrics: bool
      If True, return per-task scores.

    Returns
    -------
    dict
      Maps tasks to scores under metric.
    """
    evaluator = Evaluator(self, dataset, transformers)
    if not per_task_metrics:
      scores = evaluator.compute_model_performance(metrics)
      return scores
    else:
      scores, per_task_scores = evaluator.compute_model_performance(
          metrics, per_task_metrics=per_task_metrics)
      return scores, per_task_scores

  def get_task_type(self) -> str:
    """
    Currently models can only be classifiers or regressors.
    """
    raise NotImplementedError

  def get_num_tasks(self) -> int:
    """
    Get number of tasks.
    """
    raise NotImplementedError

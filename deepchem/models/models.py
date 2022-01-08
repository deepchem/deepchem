"""
Contains an abstract base class that supports different ML models.
"""

import os
import shutil
import tempfile
import logging
from typing import List, Optional, Sequence

import numpy as np

from deepchem.data import Dataset
from deepchem.metrics import Metric
from deepchem.trans import Transformer, undo_transforms
from deepchem.utils.evaluate import Evaluator
from deepchem.utils.typing import OneOrMany

logger = logging.getLogger(__name__)


class Model(object):
  """
  Abstract base class for DeepChem models.
  """

  def __init__(self, model=None, model_dir: Optional[str] = None,
               **kwargs) -> None:
    """Abstract class for all models.

    This is intended only for convenience of subclass implementations
    and should not be invoked directly.

    Parameters
    ----------
    model: object
      Wrapper around ScikitLearn/Keras/Tensorflow model object.
    model_dir: str, optional (default None)
      Path to directory where model will be stored. If not specified,
      model will be stored in a temporary directory.
    """
    if self.__class__.__name__ == "Model":
      raise ValueError(
          "This constructor is for an abstract class and should never be called directly."
          "Can only call from subclass constructors.")

    self.model_dir_is_temp = False
    if model_dir is not None:
      if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    else:
      model_dir = tempfile.mkdtemp()
      self.model_dir_is_temp = True
    self.model_dir = model_dir
    self.model = model
    self.model_class = model.__class__

  def __del__(self):
    if 'model_dir_is_temp' in dir(self) and self.model_dir_is_temp:
      shutil.rmtree(self.model_dir)

  def fit_on_batch(self, X: Sequence, y: Sequence, w: Sequence):
    """Perform a single step of training.

    Parameters
    ----------
    X: np.ndarray
      the inputs for the batch
    y: np.ndarray
      the labels for the batch
    w: np.ndarray
      the weights for the batch
    """
    raise NotImplementedError(
        "Each model is responsible for its own fit_on_batch method.")

  def predict_on_batch(self, X: np.typing.ArrayLike):
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

  def fit(self, dataset: Dataset):
    """
    Fits a model on data in a Dataset object.

    Parameters
    ----------
    dataset: Dataset
      the Dataset to train on
    """
    raise NotImplementedError(
        "Each model is responsible for its own fit method.")

  def predict(self, dataset: Dataset,
              transformers: List[Transformer] = []) -> OneOrMany[np.ndarray]:
    """
    Uses self to make predictions on provided Dataset object.

    Parameters
    ----------
    dataset: Dataset
      Dataset to make prediction on
    transformers: List[Transformer]
      Transformers that the input data has been transformed by. The output
      is passed through these transformers to undo the transformations.

    Returns
    -------
    np.ndarray
      A numpy array of predictions the model produces.
    """
    y_preds = []
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
               per_task_metrics: bool = False,
               use_sample_weights: bool = False,
               n_classes: int = 2):
    """
    Evaluates the performance of this model on specified dataset.

    This function uses `Evaluator` under the hood to perform model
    evaluation. As a result, it inherits the same limitations of
    `Evaluator`. Namely, that only regression and classification
    models can be evaluated in this fashion. For generator models, you
    will need to overwrite this method to perform a custom evaluation.

    Keyword arguments specified here will be passed to
    `Evaluator.compute_model_performance`.

    Parameters
    ----------
    dataset: Dataset
      Dataset object.
    metrics: Metric / List[Metric] / function
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
    transformers: List[Transformer]
      List of `dc.trans.Transformer` objects. These transformations
      must have been applied to `dataset` previously. The dataset will
      be untransformed for metric evaluation.
    per_task_metrics: bool, optional (default False)
      If true, return computed metric for each task on multitask dataset.
    use_sample_weights: bool, optional (default False)
      If set, use per-sample weights `w`.
    n_classes: int, optional (default None)
      If specified, will use `n_classes` as the number of unique classes
      in `self.dataset`. Note that this argument will be ignored for
      regression metrics.

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
    return evaluator.compute_model_performance(
        metrics,
        per_task_metrics=per_task_metrics,
        use_sample_weights=use_sample_weights,
        n_classes=n_classes)

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

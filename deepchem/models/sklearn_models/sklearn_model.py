"""
Code for processing datasets using scikit-learn.
"""
import logging
from typing import List, Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV

from deepchem.models import Model
from deepchem.data import Dataset
from deepchem.trans import Transformer
from deepchem.utils.save import load_from_disk, save_to_disk

NON_WEIGHTED_MODELS = [
    LogisticRegression, PLSRegression, GaussianProcessRegressor, ElasticNetCV,
    LassoCV, BayesianRidge
]

logger = logging.getLogger(__name__)


class SklearnModel(Model):
  """Wrapper class that wraps scikit-learn models as DeepChem models.

  When you're working with scikit-learn and DeepChem, at times it can
  be useful to wrap a scikit-learn model as a DeepChem model. The
  reason for this might be that you want to do an apples-to-apples
  comparison of a scikit-learn model to another DeepChem model, or
  perhaps you want to use the hyperparameter tuning capabilities in
  `dc.hyper`. The `SklearnModel` class provides a wrapper around scikit-learn
  models that allows scikit-learn models to be trained on `Dataset` objects
  and evaluated with the same metrics as other DeepChem models.`

  Notes
  -----
  All `SklearnModels` perform learning solely in memory. This means that it
  may not be possible to train `SklearnModel` on large `Dataset`s.
  """

  def __init__(self,
               model_instance: BaseEstimator,
               model_dir: Optional[str] = None,
               **kwargs):
    """
    Parameters
    ----------
    model_instance: BaseEstimator
      The model instance which inherits a scikit-learn `BaseEstimator` Class.
    model_dir: str, optional (default None)
      If specified the model will be stored in this directory. Else, a
      temporary directory will be used.
    kwargs: dict
      kwargs['use_weights'] is a bool which determines if we pass weights into
      self.model_instance.fit().
    """
    super(SklearnModel, self).__init__(model_instance, model_dir, **kwargs)
    if 'use_weights' in kwargs:
      self.use_weights = kwargs['use_weights']
    else:
      self.use_weights = True
    for model_instance in NON_WEIGHTED_MODELS:
      if isinstance(self.model_instance, model_instance):
        self.use_weights = False

  # FIXME: Return type "None" of "fit" incompatible with return type "float" in supertype "Model"
  def fit(self, dataset: Dataset, **kwargs) -> None:  # type: ignore[override]
    """Fits scikit-learn model to data.

    Parameters
    ----------
    dataset: Dataset
      The `Dataset` to train this model on.
    """
    X = dataset.X
    y = np.squeeze(dataset.y)
    w = np.squeeze(dataset.w)
    # Some scikit-learn models don't use weights.
    if self.use_weights:
      self.model_instance.fit(X, y, w)
      return
    self.model_instance.fit(X, y)

  def predict_on_batch(self, X: np.ndarray) -> np.ndarray:
    """Makes predictions on batch of data.

    Parameters
    ----------
    X: np.ndarray
      A numpy array of features.

    Returns
    -------
    np.ndarray
      The value is a return value of `predict_proba` or `predict` method
      of the scikit-learn model. If the scikit-learn model has both methods,
      the value is always a return value of `predict_proba`.
    """
    try:
      # FIXME: BaseEstimator doesn't guarantee the class has `predict_proba` method.
      return self.model_instance.predict_proba(X)  # type: ignore
    except AttributeError:
      # FIXME: BaseEstimator doesn't guarantee the class has `predict` method.
      return self.model_instance.predict(X)  # type: ignore

  def predict(self, X: Dataset,
              transformers: List[Transformer] = []) -> np.ndarray:
    """Makes predictions on dataset.

    Parameters
    ----------
    dataset: Dataset
      Dataset to make prediction on.
    transformers: List[Transformer]
      Transformers that the input data has been transformed by. The output
      is passed through these transformers to undo the transformations.
    """
    return super(SklearnModel, self).predict(X, transformers)

  def save(self):
    """Saves scikit-learn model to disk using joblib."""
    save_to_disk(self.model_instance, self.get_model_filename(self.model_dir))

  def reload(self):
    """Loads scikit-learn model from joblib file on disk."""
    self.model_instance = load_from_disk(
        self.get_model_filename(self.model_dir))

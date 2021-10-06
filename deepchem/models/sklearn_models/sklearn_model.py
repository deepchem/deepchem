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
from deepchem.utils.data_utils import load_from_disk, save_to_disk
from deepchem.utils.typing import ArrayLike

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
  and evaluated with the same metrics as other DeepChem models.

  Example
  ------
  >>> import deepchem as dc
  >>> import numpy as np
  >>> from sklearn.linear_model import LinearRegression
  >>> # Generating a random data and creating a dataset
  >>> X, y = np.random.randn(5, 1), np.random.randn(5)
  >>> dataset = dc.data.NumpyDataset(X, y)
  >>> # Wrapping a Sklearn Linear Regression model using DeepChem models API
  >>> sklearn_model = LinearRegression()
  >>> dc_model = dc.models.SklearnModel(sklearn_model)
  >>> dc_model.fit(dataset)  # fitting dataset

  Notes
  -----
  All `SklearnModels` perform learning solely in memory. This means that it
  may not be possible to train `SklearnModel` on large `Dataset`s.
  """

  def __init__(self,
               model: BaseEstimator,
               model_dir: Optional[str] = None,
               **kwargs):
    """
    Parameters
    ----------
    model: BaseEstimator
      The model instance which inherits a scikit-learn `BaseEstimator` Class.
    model_dir: str, optional (default None)
      If specified the model will be stored in this directory. Else, a
      temporary directory will be used.
    model_instance: BaseEstimator (DEPRECATED)
      The model instance which inherits a scikit-learn `BaseEstimator` Class.
    kwargs: dict
      kwargs['use_weights'] is a bool which determines if we pass weights into
      self.model.fit().
    """
    if 'model_instance' in kwargs:
      model_instance = kwargs['model_instance']
      if model is not None:
        raise ValueError(
            "Can not use both model and model_instance argument at the same time."
        )
      logger.warning(
          "model_instance argument is deprecated and will be removed in a future version of DeepChem."
          "Use model argument instead.")
      model = model_instance

    super(SklearnModel, self).__init__(model, model_dir, **kwargs)
    if 'use_weights' in kwargs:
      self.use_weights = kwargs['use_weights']
    else:
      self.use_weights = True
    for model in NON_WEIGHTED_MODELS:
      if isinstance(self.model, model):
        self.use_weights = False

  def fit(self, dataset: Dataset) -> None:
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
      self.model.fit(X, y, w)
      return
    self.model.fit(X, y)

  def predict_on_batch(self, X: ArrayLike) -> np.ndarray:
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
      return self.model.predict_proba(X)
    except AttributeError:
      return self.model.predict(X)

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
    save_to_disk(self.model, self.get_model_filename(self.model_dir))

  def reload(self):
    """Loads scikit-learn model from joblib file on disk."""
    self.model = load_from_disk(self.get_model_filename(self.model_dir))

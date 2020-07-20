"""
Code for processing datasets using scikit-learn.
"""
import numpy as np
import logging
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LassoLarsCV
from deepchem.models import Model
from deepchem.utils.save import load_from_disk
from deepchem.utils.save import save_to_disk

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

  Note
  ----
  All `SklearnModels` perform learning solely in memory. This means that it
  may not be possible to train `SklearnModel` on large `Dataset`s.
  """

  def __init__(self, model_instance=None, model_dir=None, **kwargs):
    """
    Parameters
    ----------
    model_instance: `sklearn.base.BaseEstimator`
      Must be a scikit-learn `BaseEstimator Class`.
    model_dir: str, optional (default None)
      If specified the model will be stored in this directory. Else, a
      temporary directory will be used.
    kwargs: dict
      kwargs['use_weights'] is a bool which determines if we pass weights into
      self.model_instance.fit()
    """
    super(SklearnModel, self).__init__(model_instance, model_dir, **kwargs)
    if 'use_weights' in kwargs:
      self.use_weights = kwargs['use_weights']
    else:
      self.use_weights = True
    for model_instance in NON_WEIGHTED_MODELS:
      if isinstance(self.model_instance, model_instance):
        self.use_weights = False

  def fit(self, dataset, **kwargs):
    """Fits SKLearn model to data.

    Parameters
    ----------
    dataset: `Dataset`
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

  def predict_on_batch(self, X, pad_batch=False):
    """
    Makes predictions on batch of data.

    Parameters
    ----------
    X: np.ndarray
      Features
    pad_batch: bool, optional
      Ignored for Sklearn Model. Only used for Tensorflow models
      with rigid batch-size requirements.
    """
    try:
      return self.model_instance.predict_proba(X)
    except AttributeError:
      return self.model_instance.predict(X)

  def predict(self, X, transformers=[]):
    """
    Makes predictions on dataset.
    """
    return super(SklearnModel, self).predict(X, transformers)

  def save(self):
    """Saves sklearn model to disk using joblib."""
    save_to_disk(self.model_instance, self.get_model_filename(self.model_dir))

  def reload(self):
    """Loads sklearn model from joblib file on disk."""
    self.model_instance = load_from_disk(
        Model.get_model_filename(self.model_dir))

  def get_num_tasks(self):
    """Number of tasks for this model. Defaults to 1"""
    return 1

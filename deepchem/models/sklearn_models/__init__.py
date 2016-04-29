"""
Code for processing datasets using scikit-learn.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LassoLarsCV
from deepchem.models import Model
from deepchem.utils.save import load_from_disk
from deepchem.utils.save import save_to_disk

class SklearnModel(Model):
  """
  Abstract base class for different ML models.
  """
  def __init__(self, tasks, task_types, model_params, model_dir, fit_transformers=None,
               model_instance=None, initialize_raw_model=True, verbosity=None,
               mode="classification"):
    super(SklearnModel, self).__init__(
        tasks, task_types, model_params, model_dir,
        fit_transformers=fit_transformers, 
        initialize_raw_model=initialize_raw_model)
    self.model_dir = model_dir
    self.task_types = task_types
    self.model_params = model_params
    self.raw_model = model_instance
    self.verbosity = verbosity
    assert mode in ["classification", "regression"]
    self.mode = mode

  # TODO(rbharath): This does not work with very large datasets! sklearn does
  # support partial_fit, but only for some models. Might make sense to make
  # PartialSklearnModel subclass at some point to support large data models.
  # Also, use of batch_size=32 is arbitrary and kludgey
  def fit(self, dataset):
    """
    Fits SKLearn model to data.
    """
    X, y, w, _ = dataset.to_numpy()
    y, w = y.flatten(), w.flatten()
    self.raw_model.fit(X, y, w)
    y_pred_raw = self.raw_model.predict(X)

  def predict_on_batch(self, X):
    """
    Makes predictions on batch of data.
    """
    return self.raw_model.predict(X)

  def predict_proba_on_batch(self, X):
    return self.raw_model.predict_proba(X)

  def predict(self, X, transformers):
    """
    Makes predictions on dataset.
    """
    # Sets batch_size which the default impl in Model expects
    #TODO(enf/rbharath): This is kludgy. Fix later.
    if "batch_size" not in self.model_params.keys():
      self.model_params["batch_size"] = None 
    return super(SklearnModel, self).predict(X, transformers)

  def save(self):
    """Saves sklearn model to disk using joblib."""
    super(SklearnModel, self).save()
    save_to_disk(self.raw_model, self.get_model_filename(self.model_dir))

  def load(self):
    """Loads sklearn model from joblib file on disk."""
    self.raw_model = load_from_disk(Model.get_model_filename(self.model_dir))

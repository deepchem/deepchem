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

  def fit(self, dataset):
    """
    Fits SKLearn model to data.
    """
    X, y, w, _ = dataset.to_numpy()
    y, w = np.squeeze(y), np.squeeze(w)
    # Logistic regression doesn't support weights
    if not isinstance(self.model_instance, LogisticRegression):
      self.model_instance.fit(X, y, w)
    else:
      self.model_instance.fit(X, y)
    y_pred_raw = self.model_instance.predict(X)

  def predict_on_batch(self, X):
    """
    Makes predictions on batch of data.
    """
    return self.model_instance.predict(X)

  def predict_proba_on_batch(self, X):
    """
    Makes per-class predictions on batch of data.
    """
    return self.model_instance.predict_proba(X)

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
    self.model_instance = load_from_disk(Model.get_model_filename(self.model_dir))

  def get_num_tasks(self):
    """Number of tasks for this model. Defaults to 1"""
    return 1

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
from deepchem.utils.dataset import load_from_disk
from deepchem.utils.dataset import save_to_disk

class SklearnModel(Model):
  """
  Abstract base class for different ML models.
  """
  def __init__(self, task_types, model_params, 
               model_instance=RandomForestRegressor(),
               initialize_raw_model=True):
    super(SklearnModel, self).__init__(
        task_types, model_params, initialize_raw_model)
    self.task_types = task_types
    self.model_params = model_params
    self.raw_model = model_instance

  # TODO(rbharath): This does not work with very large datasets! sklearn does
  # support partial_fit, but only for some models. Might make sense to make
  # PartialSklearnModel subclass at some point to support large data models.
  def fit(self, numpy_dataset):
    """
    Fits SKLearn model to data.
    """
    Xs, ys = [], []
    for (X, y, _, _) in numpy_dataset.itershards():
      Xs.append(X)
      ys.append(y)
    X = np.concatenate(Xs)
    y = np.concatenate(ys).ravel()
    self.raw_model.fit(X, y)

  def predict_on_batch(self, X):
    """
    Makes predictions on batch of data.
    """
    return self.raw_model.predict(X)

  def predict(self, X):
    """
    Makes predictions on dataset.
    """
    # Sets batch_size which the default impl in Model expects
    #TODO(enf/rbharath): This is kludgy. Fix later.
    if "batch_size" not in self.model_params.keys():
      self.model_params["batch_size"] = 32
    return super(SklearnModel, self).predict(X)

  def save(self, out_dir):
    """Saves sklearn model to disk using joblib."""
    super(SklearnModel, self).save(out_dir)
    save_to_disk(self.raw_model, self.get_model_filename(out_dir))

  def load(self, model_dir):
    """Loads sklearn model from joblib file on disk."""
    self.raw_model = load_from_disk(Model.get_model_filename(model_dir))

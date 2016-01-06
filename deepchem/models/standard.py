"""
Code for processing datasets using scikit-learn.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LassoLarsCV
from deepchem.models import Model

class SklearnModel(Model):
  """
  Abstract base class for different ML models.
  """
  def __init__(self, task_types, model_params, initialize_raw_model=True):
    self.task_types = task_types
    self.model_params = model_params
    self.raw_model = None

  def fit_on_batch(self, X, y, w):
    """
    Updates existing model with new information.
    """
    raise NotImplementedError(
        "Each model is responsible for its own fit_on_batch method.")

  def predict_on_batch(self, X):
    """
    Makes predictions on given batch of new data.
    """
    raise NotImplementedError(
        "Each model is responsible for its own predict_on_batch method.")   

def fit_singletask_models(train_data, modeltype):
  """Fits singletask linear regression models to potency.

  Parameters
  ----------
  paths: list
    List of paths to datasets.
  modeltype: String
    A string describing the model to be trained. Options are RandomForest,
  splittype: string
    Type of split for train/test. Either random or scaffold.
  seed: int (optional)
    Seed to initialize np.random.
  output_transforms: dict
    dict mapping task names to label transform. Each output type must be either
    None or "log". Only for regression outputs.
  """
  models = {}
  import numpy as np
  X_train = train_data["features"]
  sorted_tasks = train_data["sorted_tasks"]
  for task in sorted_tasks:
    print "Building model for task %s" % task
    (y_train, W_train) = train_data[task]
    W_train = W_train.ravel()
    task_X_train = X_train[W_train.nonzero()]
    task_y_train = y_train[W_train.nonzero()]
    if modeltype == "rf_regressor":
      model = RandomForestRegressor(
          n_estimators=500, n_jobs=-1, warm_start=True, max_features="sqrt")
    elif modeltype == "rf_classifier":
      model = RandomForestClassifier(
          n_estimators=500, n_jobs=-1, warm_start=True, max_features="sqrt")
    elif modeltype == "logistic":
      model = LogisticRegression(class_weight="auto")
    elif modeltype == "linear":
      model = LinearRegression(normalize=True)
    elif modeltype == "ridge":
      model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], normalize=True)
    elif modeltype == "lasso":
      model = LassoCV(max_iter=2000, n_jobs=-1)
    elif modeltype == "lasso_lars":
      model = LassoLarsCV(max_iter=2000, n_jobs=-1)
    elif modeltype == "elastic_net":
      model = ElasticNetCV(max_iter=2000, n_jobs=-1)
    else:
      raise ValueError("Invalid model type provided.")
    model.fit(task_X_train, task_y_train.ravel())
    models[task] = model
  return models

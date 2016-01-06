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
    super(SklearnModel, self).__init__(task_types, model_params,
                                       initialize_raw_model)
    self.task_types = task_types
    self.model_params = model_params
    if initialize_raw_model:
      if self.modeltype == "rf_regressor":
        raw_model = RandomForestRegressor(
            n_estimators=500, n_jobs=-1, warm_start=True, max_features="sqrt")
      elif self.modeltype == "rf_classifier":
        raw_model = RandomForestClassifier(
            n_estimators=500, n_jobs=-1, warm_start=True, max_features="sqrt")
      elif modeltype == "logistic":
        raw_model = LogisticRegression(class_weight="auto")
      elif modeltype == "linear":
        raw_model = LinearRegression(normalize=True)
      elif modeltype == "ridge":
        raw_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], normalize=True)
      elif modeltype == "lasso":
        raw_model = LassoCV(max_iter=2000, n_jobs=-1)
      elif modeltype == "lasso_lars":
        raw_model = LassoLarsCV(max_iter=2000, n_jobs=-1)
      elif modeltype == "elastic_net":
        raw_model = ElasticNetCV(max_iter=2000, n_jobs=-1)
      else:
        raise ValueError("Invalid model type provided.")

  # TODO(rbharath): This is a partial implementation! Does not work for a
  # datasets with more than one shard. 
  def fit(self, numpy_dataset):
    """
    Fits SKLearn model to data.
    """
    for (X, y, _, _) in numpy_dataset.itershards():
      self.raw_model.fit(X, y)
      return

  def predict_on_batch(self, X):
    """
    Makes predictions on given batch of new data.
    """
    return self.raw_model.predict(X)

  def save(self, out_dir):
    """Saves sklearn model to disk using joblib."""
    super(SklearnModel, self).save(out_dir)
    joblib.dump(self.raw_model, self.get_model_filename(out_dir))

  def load(self, model_dir):
    """Loads sklearn model from joblib file on disk."""
    super(SklearnModel, self).load(model_dir)
    self.raw_model = joblib.load(self.get_model_filename(model_dir)

Model.register_model_type("logistic", SklearnModel)
Model.register_model_type("rf_classifier", SklearnModel)
Model.register_model_type("rf_regressor", SklearnModel)
Model.register_model_type("linear", SklearnModel)
Model.register_model_type("ridge", SklearnModel)
Model.register_model_type("lasso", SklearnModel)
Model.register_model_type("lasso_lars", SklearnModel)
Model.register_model_type("elastic_net", SklearnModel)


# TODO(rbharath): Need to fix singletask dataset support.
'''
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
'''

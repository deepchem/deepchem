"""
Contains an abstract base class that supports different ML models.
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

#TODO(enf/rbharath): incorporate save, load, eval, fit features into class Model.
class Model(object):
  """
  Abstract base class for different ML models.
  """
  # List of registered models
  registered_model_types = {}
  def __init__(self, task_types, model_params, initialize_raw_model=True):
    self.task_types = task_types
    self.model_params = model_params

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

  def set_raw_model(self, raw_model):
    """
    Set underlying raw model. Useful when loading from disk.
    """
    self.raw_model = raw_model

  def get_raw_model(self):
    """
    Return raw model.
    """
    return(self.raw_model)

  @staticmethod
  def model_builder(model_type, task_types, model_params,
                    initialize_raw_model=True):
    print("model_builder()")
    print("model_params")
    print(model_params)
    if model_type in Model.registered_model_types:
      model = Model.registered_model_types[model_type](
          task_types, model_params, initialize_raw_model)
    else:
      raise ValueError("model_type %s is not supported" % model_type)
    return model

  @staticmethod
  def register_model_type(model_type, model_class):
    Model.registered_model_types[model_type] = model_class


'''
def model_predictions(X, model, n_targets, task_types, modeltype="sklearn"):
  """Obtains predictions of provided model on test_set.

  Returns an ndarray of shape (n_samples, n_targets)

  TODO(rbharath): This function uses n_targets instead of
  task_transforms like everything else.

  Parameters
  ----------
  X: numpy.ndarray
    Test set data.
  model: model.
    A trained scikit-learn or keras model.
  n_targets: int
    Number of output targets
  task_types: dict
    dict mapping target names to output type. Each output type must be either
    "classification" or "regression".
  modeltype: string
    Either sklearn, keras, or keras_multitask
  """
  # Extract features for test set and make preds
  # TODO(rbharath): This change in shape should not(!) be handled here. Make
  # an upstream change so the evaluator doesn't have to worry about this.
  if len(np.shape(X)) > 2:  # Dealing with 3D data
    if len(np.shape(X)) != 5:
      raise ValueError(
          "Tensorial datatype must be of shape (n_samples, N, N, N, n_channels).")
    (n_samples, axis_length, _, _, n_channels) = np.shape(X)
    X = np.reshape(X, (n_samples, axis_length, n_channels, axis_length, axis_length))
  if modeltype == "keras-graph":
    predictions = model.predict({"input": X})
    ypreds = []
    for index in range(n_targets):
      ypreds.append(predictions["task%d" % index])
  elif modeltype == "sklearn":
    # Must be single-task (breaking multitask RFs here)
    task_type = task_types.itervalues().next()
    if task_type == "classification":
      print("model_predictions()")
      print("np.shape(X)")
      print(np.shape(X))
      ypreds = model.predict_proba(X)
    elif task_type == "regression":
      ypreds = model.predict(X)
  elif modeltype == "keras-sequential":
    ypreds = model.predict(X)
  else:
    raise ValueError("Improper modeltype.")
  if isinstance(ypreds, np.ndarray):
    ypreds = np.squeeze(ypreds)
  if not isinstance(ypreds, list):
    ypreds = [ypreds]
  return ypreds
'''

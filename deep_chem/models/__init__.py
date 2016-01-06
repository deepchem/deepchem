"""
Contains an abstract base class that supports different ML models.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import sys
import numpy as np
import pandas as pd
import joblib
import os
from keras.models import model_from_json
from deep_chem.utils.dataset import NumpyDataset
from deep_chem.utils.dataset import load_sharded_dataset
from deep_chem.utils.dataset import save_sharded_dataset

'''
def get_parameter_filename(model_dir):
  """
  Given model directory, obtain filename for stored parameters.
  """
  filename = os.path.join(model_dir, "model_params.joblib")
  return filename
'''

# TODO(rbharath): Make these instance methods...
def save_sklearn_model(model, filename):
  """Saves sklearn model to disk using joblib."""
  joblib.dump(model, filename)

def save_keras_model(model, filename):
  """Saves keras models to disk."""
  filename, _ = os.path.splitext(filename)

  # Note that keras requires the model architecture and weights to be stored
  # separately. A json file is generated that specifies the model architecture.
  # The weights will be stored in an h5 file. The pkl.gz file with store the
  # target name.
  json_filename = "%s.%s" % (filename, "json")
  h5_filename = "%s.%s" % (filename, "h5")
  # Save architecture
  json_string = model.to_json()
  with open(json_filename, "wb") as file_obj:
    file_obj.write(json_string)
  model.save_weights(h5_filename, overwrite=True)


def get_model_filename(model_dir):
  """
  Given model directory, obtain filename for the model itself.
  """
  filename = os.path.join(model_dir, "model_params.joblib")
  return filename

# TODO(rbharath): Make a static method
def get_model_type(model_name):
  """Associate each model with a model_type (used for saving/loading)."""
  if model_name in ["singletask_deep_classifier", "multitask_deep_classifier",
                    "singletask_deep_regressor", "multitask_deep_regressor"]:
    model_type = "keras-graph"
  elif model_name in ["convolutional_3D_regressor"]:
    model_type = "keras-sequential"
  elif model_name == "neural_fingerprint":
    model_type = "autograd"
  else:
    model_type = "sklearn"
  return model_type

# TODO(rbharath): Make this an instance method of Model objects.
def load_sklearn_model(filename):
  """Loads sklearn model from file on disk."""
  return joblib.load(filename)

def load_keras_model(filename):
  """Loads keras model from disk.

  Assumes that filename.json and filename.h5 respectively contain the model
  architecture and weights.
  """
  filename, _ = os.path.splitext(filename)

  json_filename = "%s.%s" % (filename, "json")
  h5_filename = "%s.%s" % (filename, "h5")

  with open(json_filename) as file_obj:
    model = model_from_json(file_obj.read())
  model.load_weights(h5_filename)
  return model

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
    """
    Factory method that initializes model of requested type.
    """
    if model_type in Model.registered_model_types:
      model = Model.registered_model_types[model_type](
          task_types, model_params, initialize_raw_model)
    else:
      raise ValueError("model_type %s is not supported" % model_type)
    return model

  @staticmethod
  def register_model_type(model_type, model_class):
    """
    Registers model types in static variable for factory/dispatchers to use.
    """
    Model.registered_model_types[model_type] = model_class

  @staticmethod
  def load_model(model_name, model_dir):
    """Dispatcher function for loading."""
    model_type = get_model_type(model_name)
    params = load_sharded_dataset(get_model_filename(model_dir))
    model = Model.model_builder(model_name, params["task_types"],
                          params["model_params"], initialize_raw_model=False)
    if model_type == "sklearn":
      raw_model = load_sklearn_model(get_model_filename(model_dir))
    elif "keras" in model_type:
      raw_model = load_keras_model(get_model_filename(model_dir))
    else:
      raise ValueError("Unsupported model_type.")
    model.set_raw_model(raw_model)
    return model

  # TODO(rbharath): This really shouldn't be a static method. Make an instance
  # method instance.
  @staticmethod
  def save_model(model, model_name, model_dir):
    """Dispatcher function for saving."""
    model_type = get_model_type(model_name)
    params = {"model_params" : model.model_params,
              "task_types" : model.task_types}
    save_sharded_dataset(params, get_model_filename(model_dir))

    raw_model = model.get_raw_model()
    if model_type == "sklearn":
      save_sklearn_model(raw_model, get_model_filename(model_dir))
    elif "keras" in model_type:
      save_keras_model(raw_model, get_model_filename(model_dir))
    else:
      raise ValueError("Unsupported model_type.")


  def fit(self, numpy_dataset):
    """
    Fits a model on data in a NumpyDataset object.
    """
    # TODO(rbharath/enf): This GPU_RAM is black magic. Needs to be removed/made
    # more general.
    MAX_GPU_RAM = float(691007488/50)
    for (X, y, w, _) in numpy_dataset.itershards():
      if sys.getsizeof(X) > MAX_GPU_RAM:
        nb_block = float(sys.getsizeof(X))/MAX_GPU_RAM
        nb_sample = np.shape(X)[0]
        interval_points = np.linspace(0,nb_sample,nb_block+1).astype(int)
        for j in range(0,len(interval_points)-1):
          indices = range(interval_points[j],interval_points[j+1])
          X_batch = X[indices,:]
          y_batch = y[indices]
          w_batch = w[indices]
          self.fit_on_batch(X_batch, y_batch, w_batch)
      else:
        self.fit_on_batch(X, y, w)

  # TODO(rbharath): What does this function do when y is not provided. Suspect
  # it breaks. Need to fix.

  # TODO(rbharath): The structure of the produced df might be
  # complicated. Better way to model?
  def predict(self, numpy_dataset):
    """
    Uses self to make predictions on provided NumpyDataset object.
    """
    task_names = numpy_dataset.get_task_names()
    pred_task_names = ["%s_pred" % task_name for task_name in task_names]
    w_task_names = ["%s_weight" % task_name for task_name in task_names]
    column_names = (['ids'] + task_names + pred_task_names + w_task_names
                           + ["y_means", "y_stds"])
    pred_y_df = pd.DataFrame(columns=column_names)

    # TODO(rbharath/enf): This is only for GPU models, and is currently depends
    # on magic numbers.
    MAX_GPU_RAM = float(691007488/50)
    for (X, y, w, ids) in numpy_dataset.itershards():
      if sys.getsizeof(X) > MAX_GPU_RAM:
        nb_block = float(sys.getsizeof(X))/MAX_GPU_RAM
        nb_sample = np.shape(X)[0]
        interval_points = np.linspace(0,nb_sample,nb_block+1).astype(int)
        y_preds = []
        for j in range(0,len(interval_points)-1):
          indices = range(interval_points[j],interval_points[j+1])
          X_batch = X[indices,:]
          y_batch = y[indices]
          w_batch = w[indices]
          y_preds.append(self.predict_on_batch(X_batch))
        y_pred = np.concatenate(y_preds)
      else:
        y_pred = self.predict_on_batch(X)
      print("model.predict()")
      print("np.shape(y)")
      print(np.shape(y))
      print("np.shape(y_pred)")
      print(np.shape(y_pred))
      y_pred = np.reshape(y_pred, np.shape(y))

      shard_df = pd.DataFrame(columns=column_names)
      shard_df['ids'] = ids
      shard_df[task_names] = y
      shard_df[pred_task_names] = y_pred
      shard_df[w_task_names] = w
      shard_df["y_means"] = numpy_dataset.get_label_means() 
      shard_df["y_stds"] = numpy_dataset.get_label_stds() 
      pred_y_df = pd.concat([pred_y_df, shard_df])

    return pred_y_df 

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

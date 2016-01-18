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
from deepchem.utils.dataset import Dataset
from deepchem.utils.dataset import load_from_disk
from deepchem.utils.dataset import save_to_disk

class Model(object):
  """
  Abstract base class for different ML models.
  """
  # List of registered models
  registered_model_types = {}
  def __init__(self, model_type, task_types, model_params,
               initialize_raw_model=True):
    self.model_type = model_type
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

  def set_raw_model(self, raw_model):
    """
    Set underlying raw model. Useful when loading from disk.
    """
    self.raw_model = raw_model

  def get_raw_model(self):
    """
    Return raw model.
    """
    return self.raw_model

  @staticmethod
  def get_model_filename(out_dir):
    """
    Given model directory, obtain filename for the model itself.
    """
    return os.path.join(out_dir, "model.joblib")

  @staticmethod
  def get_params_filename(out_dir):
    """
    Given model directory, obtain filename for the model itself.
    """
    return os.path.join(out_dir, "model_params.joblib")

  @staticmethod
  def model_builder(model_type, task_types, model_params,
                    initialize_raw_model=True):
    """
    Factory method that initializes model of requested type.
    """
    if model_type in Model.registered_model_types:
      model = Model.registered_model_types[model_type](
          model_type, task_types, model_params, initialize_raw_model)
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
  def get_task_type(model_name):
    """
    Given model type, determine if classifier or regressor.
    """
    if model_name in ["logistic", "rf_classifier", "singletask_deep_classifier",
                      "multitask_deep_classifier"]:
      return "classification"
    else:
      return "regression"

  @staticmethod
  def load(model_type, model_dir):
    """Dispatcher function for loading."""
    params = load_from_disk(Model.get_params_filename(model_dir))
    if model_type in Model.registered_model_types:
      model = Model.registered_model_types[model_type](
          model_type=params["model_type"],
          task_types=params["task_types"],
          model_params=params["model_params"])
      model.load(model_dir)
    else:
      raise ValueError("model_type %s is not supported" % model_type)
    return model

  def save(self, out_dir):
    """Dispatcher function for saving."""
    params = {"model_params" : self.model_params,
              "task_types" : self.task_types,
              "model_type": self.model_type}
    save_to_disk(params, Model.get_params_filename(out_dir))

  def fit(self, dataset):
    """
    Fits a model on data in a Dataset object.
    """
    # TODO(rbharath/enf): We need a structured way to deal with potential GPU
    #                     memory overflows.
    batch_size = self.model_params["batch_size"]
    for epoch in range(self.model_params["nb_epoch"]):
      print("Starting epoch %s" % str(epoch+1))
      for i, (X, y, w, _) in enumerate(dataset.itershards()):
        print("Training on batch-%s/epoch-%s" % (str(i+1), str(epoch+1)))
        nb_sample = np.shape(X)[0]
        interval_points = np.linspace(
            0, nb_sample, np.ceil(float(nb_sample)/batch_size)+1, dtype=int)
        for j in range(len(interval_points)-1):
          indices = range(interval_points[j], interval_points[j+1])
          X_batch = X[indices, :]
          y_batch = y[indices]
          w_batch = w[indices]
          self.fit_on_batch(X_batch, y_batch, w_batch)

  # TODO(rbharath): The structure of the produced df might be
  # complicated. Better way to model?
  def predict(self, dataset):
    """
    Uses self to make predictions on provided Dataset object.
    """
    task_names = dataset.get_task_names()
    pred_task_names = ["%s_pred" % task_name for task_name in task_names]
    w_task_names = ["%s_weight" % task_name for task_name in task_names]
    column_names = (['ids'] + task_names + pred_task_names + w_task_names
                    + ["y_means", "y_stds"])
    pred_y_df = pd.DataFrame(columns=column_names)

    batch_size = self.model_params["batch_size"]
    for (X, y, w, ids) in dataset.itershards():
      nb_sample = np.shape(X)[0]
      interval_points = np.linspace(
          0, nb_sample, np.ceil(float(nb_sample)/batch_size)+1, dtype=int)
      y_preds = []
      for j in range(len(interval_points)-1):
        indices = range(interval_points[j], interval_points[j+1])
        y_preds.append(self.predict_on_batch(X[indices, :]))
      y_pred = np.concatenate(y_preds)
      y_pred = np.reshape(y_pred, np.shape(y))

      shard_df = pd.DataFrame(columns=column_names)
      shard_df['ids'] = ids
      shard_df[task_names] = y
      shard_df[pred_task_names] = y_pred
      shard_df[w_task_names] = w
      # TODO(rbharath): This feels like a total hack. Is there a structured way
      # to deal with this instead?
      shard_df["y_means"] = list(dataset.get_label_means())[0] * np.ones(np.shape(y))
      shard_df["y_stds"] = list(dataset.get_label_stds())[0]  * np.ones(np.shape(y))
      pred_y_df = pd.concat([pred_y_df, shard_df])

    return pred_y_df

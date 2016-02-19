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
from deepchem.utils.save import log

class Model(object):
  """
  Abstract base class for different ML models.
  """
  # List of registered models
  registered_model_classes = {}
  non_sklearn_models = ["SingleTaskDNN", "MultiTaskDNN", "DockingDNN"]
  def __init__(self, task_types, model_params, model_instance=None,
               initialize_raw_model=True, verbosity="low", **kwargs):
    self.model_class = model_instance.__class__
    self.task_types = task_types
    self.model_params = model_params
    self.raw_model = None
    assert verbosity in [None, "low", "high"]
    self.low_verbosity = (verbosity == "low")
    self.high_verbosity = (verbosity == "high")

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
  def model_builder(model_instance, task_types, model_params,
                    initialize_raw_model=True):
    """
    Factory method that initializes model of requested type.
    """
    if model_instance.__class__ in non_sklearn_models:
      model = model_instance(task_types, model_params, initialize_raw_model)
    else:
      model = Model.registered_model_classes["SklearnModel"](model_instance, 
                                                       task_types, model_params,
                                                       initialize_raw_model)
    return model

  @staticmethod
  def register_model_type(model_class):
    """
    Registers model types in static variable for factory/dispatchers to use.
    """
    Model.registered_model_classes[model_class.__class__] = model_class

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
  def load(model_dir):
    """Dispatcher function for loading."""
    params = load_from_disk(Model.get_params_filename(model_dir))
    model_class = params["model_class"]
    if model_class in Model.registered_model_classes:
      model = Model.registered_model_classes[model_class](
          task_types=params["task_types"],
          model_params=params["model_params"])
      model.load(model_dir)
    else:
      model = Model.registered_model_classes["SklearnModel"](model_instance=model_class,
                           task_types=params["task_types"],
                           model_params=params["model_params"])
      model.load(model_dir)
    return model

  def save(self, out_dir):
    """Dispatcher function for saving."""
    params = {"model_params" : self.model_params,
              "task_types" : self.task_types,
              "model_class": self.__class__}
    save_to_disk(params, Model.get_params_filename(out_dir))

  def fit(self, dataset):
    """
    Fits a model on data in a Dataset object.
    """
    # TODO(rbharath/enf): We need a structured way to deal with potential GPU
    #                     memory overflows.
    batch_size = self.model_params["batch_size"]
    for epoch in range(self.model_params["nb_epoch"]):
      log("Starting epoch %s" % str(epoch+1), self.low_verbosity)
      for i, (X, y, w, _) in enumerate(dataset.itershards()):
        log("Training on shard-%s/epoch-%s" % (str(i+1), str(epoch+1)),
        self.high_verbosity)
        nb_sample = np.shape(X)[0]
        interval_points = np.linspace(
            0, nb_sample, np.ceil(float(nb_sample)/batch_size)+1, dtype=int)
        for j in range(len(interval_points)-1):
          log("Training on batch-%s/shard-%s/epoch-%s" %
              (str(j+1), str(i+1), str(epoch+1)), self.high_verbosity)
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
        y_pred_on_batch = self.predict_on_batch(X[indices, :]).reshape((len(indices),len(task_names)))
        y_preds.append(y_pred_on_batch)

      y_pred = np.concatenate(y_preds)
      y_pred = np.reshape(y_pred, np.shape(y))

      shard_df = pd.DataFrame(columns=column_names)
      shard_df['ids'] = ids
      shard_df[task_names] = y
      shard_df[pred_task_names] = y_pred
      shard_df[w_task_names] = w
      pred_y_df = pd.concat([pred_y_df, shard_df])

    return pred_y_df

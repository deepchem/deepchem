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
import tempfile
import sklearn
from deepchem.datasets import Dataset
from deepchem.transformers import undo_transforms
from deepchem.utils.save import load_from_disk
from deepchem.utils.save import save_to_disk
from deepchem.utils.save import log


class Model(object):
  """
  Abstract base class for different ML models.
  """
  def __init__(self, tasks, task_types, model_params, model_dir, fit_transformers=None,
               model_instance=None, initialize_raw_model=True, 
               verbosity=None, **kwargs):
    self.model_class = model_instance.__class__
    self.model_dir = model_dir
    if not os.path.exists(self.model_dir):
      os.makedirs(self.model_dir)
    self.tasks = tasks
    self.task_types = task_types
    self.model_params = model_params
    self.fit_transformers = fit_transformers

    self.raw_model = None
    assert verbosity in [None, "low", "high"]
    self.verbosity = verbosity

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

  def predict_proba_on_batch(self, X):
    """
    Makes predictions of class probabilities on given batch of new data.
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

  def reload(self):
    """
    Reload trained model from disk.
    """
    raise NotImplementedError(
        "Each model is responsible for its own reload method.")

  @staticmethod
  def get_model_filename(model_dir):
    """
    Given model directory, obtain filename for the model itself.
    """
    return os.path.join(model_dir, "model.joblib")

  @staticmethod
  def get_params_filename(model_dir):
    """
    Given model directory, obtain filename for the model itself.
    """
    return os.path.join(model_dir, "model_params.joblib")

  def save(self):
    """Dispatcher function for saving."""
    params = {"model_params" : self.model_params,
              "task_types" : self.task_types,
              "model_class": self.__class__}
    save_to_disk(params, Model.get_params_filename(self.model_dir))

  def fit(self, dataset):
    """
    Fits a model on data in a Dataset object.
    """
    # TODO(rbharath/enf): We need a structured way to deal with potential GPU
    #                     memory overflows.
    batch_size = self.model_params["batch_size"]
    for epoch in range(self.model_params["nb_epoch"]):
      log("Starting epoch %s" % str(epoch+1), self.verbosity)
      losses = []
      for (X_batch, y_batch, w_batch, _) in dataset.iterbatches(batch_size):
        if self.fit_transformers:
          X_batch, y_batch, w_batch = self.transform_on_batch(X_batch, y_batch,
                                            w_batch)
        losses.append(self.fit_on_batch(X_batch, y_batch, w_batch))
      log("Avg loss for epoch %d: %f"
          % (epoch+1,np.array(losses).mean()),self.verbosity)


  def transform_on_batch(self, X, y, w):
    """
    Transforms data in a 1-shard Dataset object with Transformer objects.
    """
    # Transform X, y, and w
    for transformer in self.fit_transformers:
      X, y, w = transformer.transform_on_array(X, y, w)

    return X, y, w

  def predict(self, dataset, transformers=[]):
    """
    Uses self to make predictions on provided Dataset object.

    Returns:
      y_pred: numpy ndarray of shape (n_samples,)
    """
    y_preds = []
    batch_size = self.model_params["batch_size"]
    for (X_batch, y_batch, w_batch, ids_batch) in dataset.iterbatches(batch_size):
      y_pred_batch = np.reshape(self.predict_on_batch(X_batch), y_batch.shape)
      y_pred_batch = undo_transforms(y_pred_batch, transformers)
      y_preds.append(y_pred_batch)
    y_pred = np.vstack(y_preds)
  
    # The iterbatches does padding with zero-weight examples on the last batch.
    # Remove padded examples.
    n_samples, n_tasks = len(dataset), len(self.tasks)
    y_pred = y_pred[:n_samples]
    y_pred = np.reshape(y_pred, (n_samples, n_tasks))
    return y_pred

  def predict_proba(self, dataset, transformers=[], n_classes=2):
    """
    TODO: Do transformers even make sense here?

    Returns:
      y_pred: numpy ndarray of shape (n_samples, n_classes*n_tasks)
    """
    y_preds = []
    batch_size = self.model_params["batch_size"]
    n_tasks = len(self.tasks)
    for (X_batch, y_batch, w_batch, ids_batch) in dataset.iterbatches(batch_size):
      y_pred_batch = self.predict_proba_on_batch(X_batch)
      batch_size = len(y_batch)
      y_pred_batch = np.squeeze(
          np.reshape(y_pred_batch, (batch_size, n_tasks, n_classes)))
      y_pred_batch = undo_transforms(y_pred_batch, transformers)
      y_preds.append(y_pred_batch)
    y_pred = np.vstack(y_preds)
    # The iterbatches does padding with zero-weight examples on the last batch.
    # Remove padded examples.
    n_samples, n_tasks = len(dataset), len(self.tasks)
    y_pred = y_pred[:n_samples]
    y_pred = np.reshape(y_pred, (n_samples, n_tasks, n_classes))
    return y_pred

  def get_task_type(self):
    """
    Currently models can only be classifiers or regressors.
    """
    # TODO(rbharath): This is a hack based on fact that multi-tasktype models
    # aren't supported.
    return self.task_types.itervalues().next()

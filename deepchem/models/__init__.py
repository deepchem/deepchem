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
from deepchem.datasets import Dataset
from deepchem.utils.save import load_from_disk
from deepchem.utils.save import save_to_disk
from deepchem.utils.save import log
import sklearn


# DEBUG COPY!
def to_one_hot(y):
  """Transforms label vector into one-hot encoding.

  Turns y into vector of shape [n_samples, 2] (assuming binary labels).

  y: np.ndarray
    A vector of shape [n_samples, 1]
  """
  n_samples = np.shape(y)[0]
  y_hot = np.zeros((n_samples, 2))
  for index, val in enumerate(y):
    if val == 0:
      y_hot[index] = np.array([1, 0])
    elif val == 1:
      y_hot[index] = np.array([0, 1])
  return y_hot

def undo_transforms(y, transformers):
  """Undoes all transformations applied."""
  # Note that transformers have to be undone in reversed order
  for transformer in reversed(transformers):
    if transformer.transform_y:
      y = transformer.untransform(y)
  return y

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

  # TODO(rbharath): The structure of the produced df might be
  # complicated. Better way to model?
  def predict(self, dataset, transformers):
    """
    Uses self to make predictions on provided Dataset object.
    """
    X, y, w, ids = dataset.to_numpy()
    
    #y_pred = np.reshape(self.predict_on_batch(X), y.shape)
    #y_pred = undo_transforms(y_pred, transformers)

    batch_size = self.model_params["batch_size"]
    # Have to include ys/ws since we might pad batches
    y_preds = []
    print("predict()")
    print("len(dataset)")
    print(len(dataset))
    for (X_batch, y_batch, w_batch, ids_batch) in dataset.iterbatches(batch_size):
      y_pred_batch = np.reshape(self.predict_on_batch(X_batch), y_batch.shape)
      y_pred_batch = undo_transforms(y_pred_batch, transformers)
      y_preds.append(y_pred_batch)
      #ys.append(y_batch)
      #w_preds.append(w_batch)
      print("X_batch.shape, y_batch.shape, y_pred_batch.shape")
      print(X_batch.shape, y_batch.shape, y_pred_batch.shape)
    #y = np.vstack(ys)
    y_pred = np.vstack(y_preds)
    #w_pred = np.vstack(w_preds)
  
    #X = X[w.flatten() != 0, :]
    #print("Model.predict()")
    #print("y.shape, w.shape, y_pred.shape")
    #print(y.shape, w.shape, y_pred.shape)
    #for task in xrange(num_tasks):
    #  y_task, w_task, y_pred_task = y[:, task], w[:, task], y_pred[:, task]
    #  y_task = y_task[w_task.flatten() != 0]
    #  y_task = to_one_hot(y_task)
    #  y_pred_task = y_pred_task[w_task.flatten() != 0][:, np.newaxis]

    # The iterbatches does padding with zero-weight examples on the last batch.
    # Remove padded examples.
    y_pred = y_pred[:len(dataset)]

    return y_pred

    #task_names = dataset.get_task_names()
    #pred_task_names = ["%s_pred" % task_name for task_name in task_names]
    #w_task_names = ["%s_weight" % task_name for task_name in task_names]
    #raw_task_names = [task_name+"_raw" for task_name in task_names]
    #raw_pred_task_names = [pred_task_name+"_raw" for pred_task_name in pred_task_names]
    #column_names = (['ids'] + raw_task_names + task_names
    #                + raw_pred_task_names + pred_task_names + w_task_names
    #                + ["y_means", "y_stds"])
    #pred_y_df = pd.DataFrame(columns=column_names)

    #batch_size = self.model_params["batch_size"]
    #for (X_batch, y_batch, w_batch, ids_batch) in dataset.iterbatches(batch_size):

    #  # HACK(JG): This was a hack to perform n-fold averaging of y_pred on
    #  # a given X_batch.  If fit_transformers exist, we will apply them to
    #  # X_batch 1 times and average the resulting y_pred before we undo 
    #  # transforms on y_pred and y.  In the future the averaging will be
    #  # performed n_sample times, where n_sample can be user-specified.

    #  if self.fit_transformers:

    #    y_preds = []
    #    for i in xrange(1):
    #      X_b, y_b, w_b = self.transform_on_batch(X_batch, y_batch, w_batch)
    #      y_pred = self.predict_on_batch(X_b)
    #      y_pred = np.reshape(y_pred, np.shape(y_b))
    #      y_preds.append(y_pred)

    #    y_pred = np.array(y_preds).mean(axis=0)

    #  else:

    #    # DEBUG
    #    X_batch = X_batch[w_batch.flatten() != 0, :]
    #    y_batch = y_batch[w_batch.flatten() != 0]

    #    y_pred = self.predict_on_batch(X_batch)
    #    y_pred = np.reshape(y_pred, np.shape(y_batch))

    #    # DEBUG:
    #    print("y_batch.shape, y_pred.shape")
    #    print(y_batch.shape, y_pred.shape)
    #    #y_batch_d = y_batch[w_batch != 0]
    #    #y_pred_d = y_pred[w_batch != 0]
    #    #print("y_batch_d.shape, y_pred_d.shape")
    #    #print(y_batch_d.shape, y_pred_d.shape)
    #    if np.count_nonzero(y_batch) > 0:
    #      print("sklearn.metrics.roc_auc_score(y_batch, y_pred)")
    #      print(sklearn.metrics.roc_auc_score(y_batch, y_pred))

    #  # Now undo transformations on y, y_pred

    #  y_raw, y_pred_raw = y_batch, y_pred
    #  y_batch = undo_transforms(y_batch, transformers)
    #  y_pred = undo_transforms(y_pred, transformers)

    #  batch_df = pd.DataFrame(columns=column_names)
    #  #batch_df['ids'] = ids_batch
    #  #batch_df[raw_task_names] = y_raw
    #  #batch_df[task_names] = y_batch
    #  #batch_df[raw_pred_task_names] = y_pred_raw
    #  #batch_df[pred_task_names] = y_pred
    #  #batch_df[w_task_names] = w_batch
    #  pred_y_df = pd.concat([pred_y_df, batch_df])

    #return pred_y_df

  def get_task_type(self):
    """
    Currently models can only be classifiers or regressors.
    """
    # TODO(rbharath): This is a hack based on fact that multi-tasktype models
    # aren't supported.
    return self.task_types.itervalues().next()

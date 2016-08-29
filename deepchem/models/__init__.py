"""
Contains an abstract base class that supports different ML models.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
__author__ = "Bharath Ramsundar and Joseph Gomes"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"
import sys
import numpy as np
import pandas as pd
import joblib
import os
import tempfile
import sklearn
from deepchem.datasets import Dataset
from deepchem.transformers import undo_transforms
from deepchem.transformers import undo_grad_transforms
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
    n_tasks = len(self.tasks)
    for (X_batch, y_batch, w_batch, ids_batch) in dataset.iterbatches(
        batch_size, deterministic=True):
      n_samples = len(X_batch)
      y_pred_batch = np.reshape(self.predict_on_batch(X_batch), (n_samples, n_tasks))
      y_pred_batch = undo_transforms(y_pred_batch, transformers)
      y_preds.append(y_pred_batch)
    y_pred = np.vstack(y_preds)
  
    # The iterbatches does padding with zero-weight examples on the last batch.
    # Remove padded examples.
    n_samples, n_tasks = len(dataset), len(self.tasks)
    y_pred = np.reshape(y_pred, (n_samples, n_tasks))
    # Special case to handle singletasks.
    if n_tasks == 1:
      y_pred = np.reshape(y_pred, (n_samples,)) 
    return y_pred

  def predict_grad(self, dataset, transformers=[]):
    """
    Uses self to calculate gradient on provided Dataset object.

    TODO(rbharath): Should we assume each model has meaningful gradients to
    predict? Should this be a subclass for PhysicalModel or the like?

    Returns:
      y_pred: numpy ndarray of shape (n_samples,)
    """
    grads = []
    batch_size = self.model_params["batch_size"]
    for (X_batch, y_batch, w_batch, ids_batch) in dataset.iterbatches(batch_size):
      energy_batch = self.predict_on_batch(X_batch)
      grad_batch = self.predict_grad_on_batch(X_batch)
      grad_batch = undo_grad_transforms(grad_batch, energy_batch, transformers)
      grads.append(grad_batch)
    grad = np.vstack(grads)
  
    return grad

  def evaluate_error(self, dataset, transformers=[]):
    """
    Evaluate the error in energy and gradient components, forcebalance-style.

    TODO(rbharath): This looks like it should be a subclass method for a
    PhysicalMethod class. forcebalance style errors aren't meaningful for most
    chem-informatic datasets.
    """
    y_preds = []
    y_train = []
    batch_size = self.model_params["batch_size"]
    for (X_batch, y_batch, w_batch, ids_batch) in dataset.iterbatches(batch_size):

      y_pred_batch = self.predict_on_batch(X_batch)
      y_pred_batch = np.reshape(y_pred_batch, y_batch.shape)

      y_pred_batch = undo_transforms(y_pred_batch, transformers)
      y_preds.append(y_pred_batch)

      y_batch = undo_transforms(y_batch, transformers)
      y_train.append(y_batch)

    y_pred = np.vstack(y_preds)
    y = np.vstack(y_train)

    n_samples, n_tasks = len(dataset), len(self.tasks)
    n_atoms = int((n_tasks-1)/3)

    y_pred = np.reshape(y_pred, (n_samples, n_tasks)) 
    y = np.reshape(y, (n_samples, n_tasks))
    grad = y_pred[:,1:]
    grad_train = y[:,1:]

    energy_error = y[:,0]-y_pred[:,0]
    # convert Hartree to kJ/mol
    energy_error = np.sqrt(np.mean(energy_error*energy_error))*2625.5002
 
    grad = np.reshape(grad, (n_samples, n_atoms, 3))
    grad_train = np.reshape(grad_train, (n_samples, n_atoms, 3))    
  
    grad_error = grad-grad_train
    # convert Hartree/bohr to kJ/mol/Angstrom
    grad_error = np.sqrt(np.mean(grad_error*grad_error))*4961.47596096

    print("Energy error (RMSD): %f kJ/mol" % energy_error)
    print("Grad error (RMSD): %f kJ/mol/A" % grad_error)
    
    return energy_error, grad_error

  def evaluate_error_class2(self, dataset, transformers=[]):
    """
    Evaluate the error in energy and gradient components, forcebalance-style.

    TODO(rbharath): Should be a subclass PhysicalModel method. Also, need to
    find a better name for this method (class2 doesn't tell us anything about the
    semantics of this method.
    """
    y_preds = []
    y_train = []
    grads = []
    batch_size = self.model_params["batch_size"]
    for (X_batch, y_batch, w_batch, ids_batch) in dataset.iterbatches(batch_size):

      # untransformed E is needed for undo_grad_transform
      energy_batch = self.predict_on_batch(X_batch)
      grad_batch = self.predict_grad_on_batch(X_batch)
      grad_batch = undo_grad_transforms(grad_batch, energy_batch, transformers)      
      grads.append(grad_batch)
      y_pred_batch = np.reshape(energy_batch, y_batch.shape)

      # y_pred_batch gives us the pred E and pred multitask trained gradE
      y_pred_batch = undo_transforms(y_pred_batch, transformers)
      y_preds.append(y_pred_batch)

      # undo transforms on y_batch should know how to handle E and gradE separately
      y_batch = undo_transforms(y_batch, transformers)
      y_train.append(y_batch)

    y_pred = np.vstack(y_preds)
    y = np.vstack(y_train)
    grad = np.vstack(grads)

    n_samples, n_tasks = len(dataset), len(self.tasks)
    n_atoms = int((n_tasks-1)/3)

    y_pred = np.reshape(y_pred, (n_samples, n_tasks)) 
    y = np.reshape(y, (n_samples, n_tasks))
    grad_train = y[:,1:]

    energy_error = y[:,0]-y_pred[:,0]
    energy_error = np.sqrt(np.mean(energy_error*energy_error))*2625.5002
 
    grad = np.reshape(grad, (n_samples, n_atoms, 3))
    grad_train = np.reshape(grad_train, (n_samples, n_atoms, 3))    
  
    grad_error = grad-grad_train
    grad_error = np.sqrt(np.mean(grad_error*grad_error))*4961.47596096

    print("Energy error (RMSD): %f kJ/mol" % energy_error)
    print("Grad error (RMSD): %f kJ/mol/A" % grad_error)
    
    return energy_error, grad_error

  def test_fd_grad(self, dataset, transformers=[]):
    """
    Uses self to calculate finite difference gradient on provided Dataset object.
    Currently only useful if your task is energy and self contains predict_grad_on_batch.

    TODO(rbharath): This shouldn't be a method of the Model class. Perhaps a
    method of PhysicalModel subclass. Leaving it in for time-being while refactoring
    continues.

    Returns:
      y_pred: numpy ndarray of shape (n_samples,)
    """
    y_preds = []
    batch_size = self.model_params["batch_size"]
    for (X_batch, y_batch, w_batch, ids_batch) in dataset.iterbatches(batch_size):

      for xb in X_batch:

        num_atoms = xb.shape[0]
        coords = 3

        h = 0.001
        fd_batch = []
        # Filling a new batch with displaced geometries
        for i in xrange(num_atoms):
          for j in xrange(coords):
            displace = np.zeros((num_atoms, coords))
            displace[i][j] += h/2
            fd_batch.append(xb+displace)
            fd_batch.append(xb-displace)

        fd_batch = np.asarray(fd_batch)
        # Predict energy on displaced geometry batch
        y_pred_batch = self.predict_on_batch(fd_batch)
        energy = y_pred_batch[:,0]
        y_pred_batch = undo_transforms(y_pred_batch, transformers)
        y_pred_batch = y_pred_batch[:,0]
        y_pred_batch = np.reshape(y_pred_batch, (3*num_atoms, 2))

        fd_grads = []
        # Calculate numerical gradient by centered finite difference
        for x in y_pred_batch:
          fd_grads.append((x[0]-x[1])/h)

        fd_grads = np.asarray(fd_grads)
        fd_grads = np.reshape(fd_grads, (num_atoms, coords))

        xb = np.asarray([xb])
        y_pred_batch = self.predict_grad_on_batch(xb)
        y_pred_batch = undo_grad_transforms(energy, y_pred_batch, transformers)
        # Calculate error between symbolic gradient and numerical gradient
        y_pred_batch = y_pred_batch-fd_grads
        #print(y_pred_batch)
        y_preds.append(y_pred_batch)

    y_pred = np.vstack(y_preds)
  
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
    for (X_batch, y_batch, w_batch, ids_batch) in dataset.iterbatches(
        batch_size, deterministic=True):
      y_pred_batch = self.predict_proba_on_batch(X_batch)
      batch_size = len(y_batch)
      y_pred_batch = np.reshape(y_pred_batch, (batch_size, n_tasks, n_classes))
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

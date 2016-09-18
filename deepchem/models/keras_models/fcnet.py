"""
Code for processing the Google vs-datasets using keras.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
from keras.models import Graph
from keras.models import model_from_json
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization 
from keras.optimizers import SGD
from deepchem.models.keras_models import KerasModel
from deepchem.metrics import to_one_hot

class MultiTaskDNN(Graph):
  """
  Model for multitask MLP in keras.
  
  TODO(rbharath): Port this code over to use Keras's new functional-API
  instead of using legacy Graph object.
  """
  def __init__(self, n_tasks, n_inputs, task_type, nb_layers=1, nb_hidden=1000,
               init="glorot_uniform", batchnorm=False, dropout=0.5,
               activation="relu", learning_rate=.001, decay=1e-6,
               momentum=0.9, nesterov=False, verbosity="low"):
    super(MultiTaskDNN, self).__init__()
    # Store hyperparameters
    assert task_type in ["classification", "regression"]
    self.task_type = task_type
    self.n_inputs = n_inputs
    self.nb_layers = nb_layers
    self.nb_hidden = nb_hidden
    self.init = init
    self.batchnorm = batchnorm
    self.dropout = dropout
    self.activation = activation
    self.learning_rate = learning_rate
    self.decay = decay
    self.momentum = momentum
    self.nesterov = nesterov

    self.add_input(name="input", input_shape=(self.n_inputs,))
    prev_layer = "input"
    for ind, layer in enumerate(range(self.nb_layers)):
      dense_layer_name = "dense%d" % ind
      activation_layer_name = "activation%d" % ind
      batchnorm_layer_name = "batchnorm%d" % ind
      dropout_layer_name = "dropout%d" % ind
      self.add_node(
          Dense(self.nb_hidden, init=self.init),
          name=dense_layer_name, input=prev_layer)
      prev_layer = dense_layer_name 
      if self.batchnorm:
        self.add_node(
          BatchNormalization(), input=prev_layer, name=batchnorm_layer_name)
        prev_layer = batchnorm_layer_name
      self.add_node(Activation(self.activation),
                    name=activation_layer_name, input=prev_layer)
      prev_layer = activation_layer_name
      if self.dropout > 0:
        self.add_node(Dropout(self.dropout),
                      name=dropout_layer_name,
                      input=prev_layer)
        prev_layer = dropout_layer_name
    for task in range(self.n_tasks):
      if self.task_type == "classification":
        self.add_node(
            Dense(2, init=self.init, activation="softmax"),
            name="dense_head%d" % task, input=prev_layer)
      elif self.task_type == "regression":
        self.add_node(
            Dense(1, init=self.init),
            name="dense_head%d" % task, input=prev_layer)
      self.add_output(name="task%d" % task, input="dense_head%d" % task)

    loss_dict = {}
    for task in range(self.n_tasks):
      taskname = "task%d" % ind
      if self.task_type == "classification":
        loss_dict[taskname] = "binary_crossentropy"
      elif self.task_type == "regression":
        loss_dict[taskname] = "mean_squared_error"
    sgd = SGD(lr=self.learning_rate,
              decay=self.decay,
              momentum=self.momentum,
              nesterov=self.nesterov)
    self.compile(optimizer=sgd, loss=loss_dict)

  def get_data_dict(self, X, y=None):
    """Wrap data X in dict for graph computations (Keras graph only for now)."""
    data = {}
    data["input"] = X
    for task in range(self.n_tasks):
      taskname = "task%d" % task 
      if y is not None:
        if self.task_type == "classification":
          data[taskname] = to_one_hot(y[:, task])
        elif self.task_type == "regression":
          data[taskname] = y[:, task]
    return data

  def get_sample_weight(self, w):
    """Get dictionaries needed to fit models"""
    sample_weight = {}
    for task in range(self.n_tasks):
      sample_weight["task%d" % task] = w[:, task]
    return sample_weight

  def fit_on_batch(self, X, y, w):
    """
    Updates existing model with new information.
    """
    eps = .001
    # Add eps weight to avoid minibatches with zero weight (causes theano to crash).
    w = w + eps * np.ones(np.shape(w))
    data = self.get_data_dict(X, y)
    sample_weight = self.get_sample_weight(w)
    loss = self.raw_model.train_on_batch(data, sample_weight=sample_weight)
    return loss

  def predict_on_batch(self, X):
    """
    Makes predictions on given batch of new data.
    """
    data = self.get_data_dict(X)
    y_pred_dict = self.raw_model.predict_on_batch(data)
    nb_samples = np.shape(X)[0]
    y_pred = np.zeros((nb_samples, self.n_tasks))
    for task in range(self.n_tasks):
      taskname = "task%d" % task 
      if self.task_type == "classification":
        # Class probabilities are predicted for classification outputs. Instead,
        # output the most likely class.
        y_pred_task = np.squeeze(np.argmax(y_pred_dict[taskname], axis=1))
      else:
        y_pred_task = np.squeeze(y_pred_dict[taskname])
      y_pred[:, task] = y_pred_task
    y_pred = np.squeeze(y_pred)
    return y_pred

  def predict_proba_on_batch(self, X, n_classes=2):
    """
    Makes predictions on given batch of new data.
    """
    data = self.get_data_dict(X)
    y_pred_dict = self.raw_model.predict_on_batch(data)
    n_samples = np.shape(X)[0]
    y_pred = np.zeros((n_samples, self.n_tasks, n_classes))
    for task in rand(self.n_tasks):
      taskname = "task%d" % task 
      y_pred_task = np.squeeze(y_pred_dict[taskname])
      y_pred[:, task] = y_pred_task
    y_pred = np.squeeze(y_pred)
    return y_pred

class SingleTaskDNN(MultiTaskDNN):
  """
  Abstract base class for different ML models.
  """
  def __init__(self, n_inputs, task_type, **kwargs):
    n_tasks = 1
    super(SingleTaskDNN, self).__init__(
        n_tasks, n_inputs, task_type, **kwargs)

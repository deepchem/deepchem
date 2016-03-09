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
from deepchem.utils.evaluate import to_one_hot

class MultiTaskDNN(KerasModel):
  """
  Model for multitask MLP in keras.
  """
  def __init__(self, task_types, model_params,
               initialize_raw_model=True, verbosity="low"):
    super(MultiTaskDNN, self).__init__(task_types, model_params,
                                       initialize_raw_model=initialize_raw_model,
                                       verbosity=verbosity)
    if initialize_raw_model:
      sorted_tasks = sorted(task_types.keys())
      (n_inputs,) = model_params["data_shape"]
      model = Graph()
      model.add_input(name="input", input_shape=(n_inputs,))
      prev_layer = "input"
      for ind, layer in enumerate(range(model_params["nb_layers"])):
        dense_layer_name = "dense%d" % ind
        activation_layer_name = "activation%d" % ind
        batchnorm_layer_name = "batchnorm%d" % ind
        dropout_layer_name = "dropout%d" % ind
        model.add_node(
            Dense(model_params["nb_hidden"], init=model_params["init"]),
            name=dense_layer_name, input=prev_layer)
        prev_layer = dense_layer_name 
        if model_params["batchnorm"]:
          model.add_node(
            BatchNormalization(), input=prev_layer, name=batchnorm_layer_name)
          prev_layer = batchnorm_layer_name
        model.add_node(Activation(model_params["activation"]),
                       name=activation_layer_name, input=prev_layer)
        prev_layer = activation_layer_name
        if model_params["dropout"] > 0:
          model.add_node(Dropout(model_params["dropout"]),
                         name=dropout_layer_name,
                         input=prev_layer)
          prev_layer = dropout_layer_name
      for ind, task in enumerate(sorted_tasks):
        task_type = task_types[task]
        if task_type == "classification":
          model.add_node(
              Dense(2, init=model_params["init"], activation="softmax"),
              name="dense_head%d" % ind, input=prev_layer)
        elif task_type == "regression":
          model.add_node(
              Dense(1, init=model_params["init"]),
              name="dense_head%d" % ind, input=prev_layer)
        model.add_output(name="task%d" % ind, input="dense_head%d" % ind)

      loss_dict = {}
      for ind, task in enumerate(sorted_tasks):
        task_type, taskname = task_types[task], "task%d" % ind
        if task_type == "classification":
          loss_dict[taskname] = "binary_crossentropy"
        elif task_type == "regression":
          loss_dict[taskname] = "mean_squared_error"
      sgd = SGD(lr=model_params["learning_rate"],
                decay=model_params["decay"],
                momentum=model_params["momentum"],
                nesterov=model_params["nesterov"])
      model.compile(optimizer=sgd, loss=loss_dict)
      self.raw_model = model

  def get_data_dict(self, X, y=None):
    """Wrap data X in dict for graph computations (Keras graph only for now)."""
    data = {}
    data["input"] = X
    for ind, task in enumerate(sorted(self.task_types.keys())):
      task_type, taskname = self.task_types[task], "task%d" % ind
      if y is not None:
        if task_type == "classification":
          data[taskname] = to_one_hot(y[:, ind])
        elif task_type == "regression":
          data[taskname] = y[:, ind]
    return data

  def get_sample_weight(self, w):
    """Get dictionaries needed to fit models"""
    sample_weight = {}
    for ind in range(len(sorted(self.task_types.keys()))):
      sample_weight["task%d" % ind] = w[:, ind]
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
    sorted_tasks = sorted(self.task_types.keys())
    nb_samples = np.shape(X)[0]
    nb_tasks = len(sorted_tasks)
    y_pred = np.zeros((nb_samples, nb_tasks))
    for ind, task in enumerate(sorted_tasks):
      task_type = self.task_types[task]
      taskname = "task%d" % ind
      if task_type == "classification":
        # Class probabilities are predicted for classification outputs. Instead,
        # output the most likely class.
        y_pred_task = np.squeeze(np.argmax(y_pred_dict[taskname], axis=1))
      else:
        y_pred_task = np.squeeze(y_pred_dict[taskname])
      y_pred[:, ind] = y_pred_task
    y_pred = np.squeeze(y_pred)
    return y_pred

class SingleTaskDNN(MultiTaskDNN):
  """
  Abstract base class for different ML models.
  """
  def __init__(self, task_types, model_params, initialize_raw_model=True, verbosity="low"):
    super(SingleTaskDNN, self).__init__(task_types, model_params,
                                        initialize_raw_model=initialize_raw_model,
                                        verbosity=verbosity)

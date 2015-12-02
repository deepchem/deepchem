"""
Code for processing the Google vs-datasets using keras.
"""
import numpy as np
import keras
from keras.models import Graph
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from deep_chem.utils.preprocess import to_one_hot


def fit_multitask_mlp(train_data, task_types, **training_params):
  """
  Perform stochastic gradient descent optimization for a keras multitask MLP.
  Returns AUCs, R^2 scores, and RMS values.

  Parameters
  ----------
  task_types: dict 
    dict mapping task names to output type. Each output type must be either
    "classification" or "regression".
  training_params: dict
    Aggregates keyword parameters to pass to train_multitask_model
  """
  models = {}
  # Follows convention from process_datasets that the data for multitask models
  # is grouped under key "all"
  X_train = train_data["features"]
  (y_train, W_train) = train_data["all"]
  models["all"] = train_multitask_model(X_train, y_train, W_train, task_types,
                                **training_params)
  return models

def fit_singletask_mlp(train_data, task_types, **training_params):
  """
  Perform stochastic gradient descent optimization for a keras MLP.

  task_types: dict 
    dict mapping task names to output type. Each output type must be either
    "classification" or "regression".
  output_transforms: dict 
    dict mapping task names to label transform. Each output type must be either
    None or "log". Only for regression outputs.
  training_params: dict
    Aggregates keyword parameters to pass to train_multitask_model
  """
  models = {}
  train_ids = train_data["mol_ids"]
  X_train = train_data["features"]
  sorted_tasks = train_data["sorted_tasks"]
  for index, task in enumerate(sorted_tasks):
    print "Training model %d" % index
    print "Target %s" % task
    (y_train, W_train) = train_data[task]
    flat_W_train = W_train.ravel()
    task_X_train = X_train[flat_W_train.nonzero()]
    task_y_train = y_train[flat_W_train.nonzero()]
    print "%d compounds in Train" % len(train_ids)
    models[task] = train_multitask_model(task_X_train, task_y_train, W_train,
        {task: task_types[task]}, **training_params)
  return models

def train_multitask_model(X, y, W, task_types,
  learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True, activation="relu",
  dropout=0.5, nb_epoch=20, batch_size=50, n_hidden=500,
  validation_split=0.1):
  """
  Perform stochastic gradient descent optimization for a keras multitask MLP.
  Returns a trained model.

  Parameters
  ----------
  X: np.ndarray
    Feature matrix
  y: np.ndarray
    Label matrix
  W: np.ndarray
    Weight matrix
  task_types: dict 
    dict mapping task names to output type. Each output type must be either
    "classification" or "regression".
  learning_rate: float
    Learning rate used.
  decay: float
    Learning rate decay.
  momentum: float
    Momentum used in SGD.
  nesterov: bool
    Use Nesterov acceleration
  nb_epoch: int
    maximal number of epochs to run the optimizer
  """
  eps = .001
  num_tasks = len(task_types)
  sorted_tasks = sorted(task_types.keys())
  local_task_types = task_types.copy()
  endpoints = sorted_tasks
  print "train_multitask_model()"
  print "np.shape(X)"
  print np.shape(X)
  n_inputs = len(X[0].flatten())
  # Add eps weight to avoid minibatches with zero weight (causes theano to crash).
  W = W + eps * np.ones(np.shape(W))
  print "np.shape(W)"
  print np.shape(W)
  model = Graph()
  #model.add_input(name="input", ndim=n_inputs)
  model.add_input(name="input", input_shape=(n_inputs,))
  model.add_node(
      Dense(n_hidden, init='uniform', activation=activation),
      name="dense", input="input")
  model.add_node(Dropout(dropout), name="dropout", input="dense")
  top_layer = "dropout"
  for ind, task in enumerate(endpoints):
    task_type = local_task_types[task]
    if task_type == "classification":
      model.add_node(
          Dense(2, init='uniform', activation="softmax"),
          name="dense_head%d" % ind, input=top_layer)
    elif task_type == "regression":
      model.add_node(
          Dense(1, init='uniform'),
          name="dense_head%d" % ind, input=top_layer)
    model.add_output(name="task%d" % ind, input="dense_head%d" % ind)
  data_dict, loss_dict, sample_weights = {}, {}, {}
  data_dict["input"] = X
  for ind, task in enumerate(endpoints):
    task_type = local_task_types[task]
    taskname = "task%d" % ind 
    sample_weights[taskname] = W[:,ind]
    if task_type == "classification":
      loss_dict[taskname] = "binary_crossentropy"
      data_dict[taskname] = to_one_hot(y[:,ind])
    elif task_type == "regression":
      loss_dict[taskname] = "mean_squared_error"
      data_dict[taskname] = y[:,ind]
  sgd = SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=nesterov)
  print "About to compile model!"
  model.compile(optimizer=sgd, loss=loss_dict)
  print "Done compiling. About to fit model!"
  print "validation_split: " + str(validation_split)
  model.fit(data_dict, nb_epoch=nb_epoch, batch_size=batch_size,
    validation_split=validation_split, sample_weight=sample_weights)
  return model

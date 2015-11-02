"""
Code for processing the Google vs-datasets using keras.
"""
import numpy as np
import sys
import keras
from keras.models import Graph
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from deep_chem.utils.load import load_datasets
from deep_chem.utils.load import ensure_balanced
from deep_chem.utils.preprocess import multitask_to_singletask
from deep_chem.utils.preprocess import split_dataset
from deep_chem.utils.preprocess import dataset_to_numpy
from deep_chem.utils.preprocess import to_one_hot
from deep_chem.utils.evaluate import eval_model
from deep_chem.utils.evaluate import compute_r2_scores
from deep_chem.utils.evaluate import compute_rms_scores
from deep_chem.utils.evaluate import compute_roc_auc_scores
from deep_chem.utils.load import load_and_transform_dataset


def fit_multitask_mlp(train_data, test_data, task_types, **training_params):
  """
  Perform stochastic gradient descent optimization for a keras multitask MLP.
  Returns AUCs, R^2 scores, and RMS values.

  Parameters
  ----------
  task_types: dict 
    dict mapping target names to output type. Each output type must be either
    "classification" or "regression".
  training_params: dict
    Aggregates keyword parameters to pass to train_multitask_model
  """
  (train, X_train, y_train, W_train), (test, X_test, y_test, W_test) = (
      train_data, test_data)
  model = train_multitask_model(X_train, y_train, W_train, task_types,
                                **training_params)
  results = eval_model(test, model, task_types,
      modeltype="keras_multitask")
  local_task_types = task_types.copy()
  aucs = compute_roc_auc_scores(results, local_task_types)
  if aucs:
    print "Mean AUC: %f" % np.mean(np.array(aucs.values()))
  r2s = compute_r2_scores(results, local_task_types)
  if r2s:
    print "Mean R^2: %f" % np.mean(np.array(r2s.values()))

def fit_singletask_mlp(per_task_data, task_types, num_to_train=None, **training_params):
  """
  Perform stochastic gradient descent optimization for a keras MLP.

  task_types: dict 
    dict mapping target names to output type. Each output type must be either
    "classification" or "regression".
  output_transforms: dict 
    dict mapping target names to label transform. Each output type must be either
    None or "log". Only for regression outputs.
  training_params: dict
    Aggregates keyword parameters to pass to train_multitask_model
  """
  print "ENTERING FIT_SINGLETASK_MLP"
  ret_vals = {}
  aucs, r2s, rms = {}, {}, {}
  sorted_targets = sorted(per_task_data.keys())
  if num_to_train:
    sorted_targets = sorted_targets[:num_to_train]
  for index, target in enumerate(sorted_targets):
    print "Training model %d" % index
    print "Target %s" % target
    (train, X_train, y_train, W_train), (test, X_test, y_test, W_test) = (
        per_task_data[target])
    print "len(train)"
    print len(train)
    print "len(test)"
    print len(test)
    model = train_multitask_model(X_train, y_train, W_train,
        {target: task_types[target]}, **training_params)
    results = eval_model(test, model, {target: task_types[target]}, 
                         # We run singletask models as special cases of
                         # multitask.
                         modeltype="keras_multitask")
    target_aucs = compute_roc_auc_scores(results, task_types)
    target_r2s = compute_r2_scores(results, task_types)
    target_rms = compute_rms_scores(results, task_types)

    aucs.update(target_aucs)
    r2s.update(target_r2s)
    rms.update(target_rms)
  if aucs:
    print aucs
    print "Mean AUC: %f" % np.mean(np.array(aucs.values()))
  if r2s:
    print r2s
    print "Mean R^2: %f" % np.mean(np.array(r2s.values()))

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
    dict mapping target names to output type. Each output type must be either
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
  sorted_targets = sorted(task_types.keys())
  local_task_types = task_types.copy()
  endpoints = sorted_targets
  (_, n_inputs) = np.shape(X)
  # Add eps weight to avoid minibatches with zero weight (causes theano to crash).
  W = W + eps * np.ones(np.shape(W))
  model = Graph()
  model.add_input(name="input", ndim=n_inputs)
  model.add_node(
      Dense(n_inputs, n_hidden, init='uniform', activation=activation),
      name="dense", input="input")
  model.add_node(Dropout(dropout), name="dropout", input="dense")
  top_layer = "dropout"
  for task, target in enumerate(endpoints):
    task_type = local_task_types[target]
    if task_type == "classification":
      model.add_node(
          Dense(n_hidden, 2, init='uniform', activation="softmax"),
          name="dense_head%d" % task, input=top_layer)
    elif task_type == "regression":
      model.add_node(
          Dense(n_hidden, 1, init='uniform'),
          name="dense_head%d" % task, input=top_layer)
    model.add_output(name="task%d" % task, input="dense_head%d" % task)
  data_dict, loss_dict, sample_weights = {}, {}, {}
  data_dict["input"] = X
  for task, target in enumerate(endpoints):
    task_type = local_task_types[target]
    taskname = "task%d" % task
    sample_weights[taskname] = W[:, task]
    if task_type == "classification":
      loss_dict[taskname] = "binary_crossentropy"
      data_dict[taskname] = to_one_hot(y[:,task])
    elif task_type == "regression":
      loss_dict[taskname] = "mean_squared_error"
      data_dict[taskname] = y[:,task]
  sgd = SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=nesterov)
  print "About to compile model!"
  model.compile(optimizer=sgd, loss=loss_dict)
  print "Done compiling. About to fit model!"
  print "validation_split: " + str(validation_split)
  model.fit(data_dict, nb_epoch=nb_epoch, batch_size=batch_size,
    validation_split=validation_split, sample_weight=sample_weights)
  return model

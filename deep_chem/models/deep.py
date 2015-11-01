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
from deep_chem.utils.preprocess import train_test_random_split
from deep_chem.utils.preprocess import train_test_scaffold_split
from deep_chem.utils.preprocess import dataset_to_numpy
from deep_chem.utils.preprocess import to_one_hot
from deep_chem.utils.evaluate import eval_model
from deep_chem.utils.evaluate import compute_r2_scores
from deep_chem.utils.evaluate import compute_rms_scores
from deep_chem.utils.evaluate import compute_roc_auc_scores
from deep_chem.utils.load import load_and_transform_dataset

def process_multitask(paths, input_transforms, output_transforms, feature_types,
    splittype="random", seed=None, weight_positives=False):
  """Extracts multitask datasets and splits into train/test.

  Returns a tuple of test/train datasets, fingerprints, and labels.

  TODO(rbharath): This function is ugly. Returns way too many arguments. Clean
  it up.

  Parameters
  ----------
  paths: list 
    List of paths to datasets. 
  output_transforms: dict 
    dict mapping target names to label transform. Each output type must be either
    None, "log", "normalize" or "log-normalize". Only for regression outputs.
  splittype: string
    Must be "random" or "scaffold"
  seed: int
    Seed used for random splits.
  """
  dataset = load_and_transform_dataset(paths, input_transforms, output_transforms,
			prediction_endpoint, feature_types=feature_types,
      weight_positives=weight_positives)
  sorted_targets = sorted(dataset.keys())
  if splittype == "random":
    train, test = train_test_random_split(dataset, seed=seed)
  elif splittype == "scaffold":
    train, test = train_test_scaffold_split(dataset)
  else:
    raise ValueError("Improper splittype. Must be random/scaffold.")
  X_train, y_train, W_train = dataset_to_numpy(train)
  ## TODO(rbharath): Still need to fix the failures for PCBA. Temporarily
  ## commenting out to experiment.
  #if weight_positives:
  #  print "Train set balance"
  #  ensure_balanced(y_train, W_train)
  X_test, y_test, W_test = dataset_to_numpy(test)
  #if weight_positives:
  #  print "Test set balance"
  #  ensure_balanced(y_test, W_test)
  return (train, X_train, y_train, W_train, test, X_test, y_test, W_test)

def process_singletask(paths, input_transforms, output_transforms,
		prediction_endpoint, feature_types,
		splittype="random", seed=None,
    weight_positives=True):
  """Extracts singletask datasets and splits into train/test.

  Returns a dict that maps target names to tuples.

  Parameters
  ----------
  paths: list 
    List of paths to Google vs datasets. 
  output_transforms: dict 
    dict mapping target names to label transform. Each output type must be either
    None or "log". Only for regression outputs.
  splittype: string
    Must be "random" or "scaffold"
  seed: int
    Seed used for random splits.
  """
  dataset = load_and_transform_dataset(paths, input_transforms, output_transforms,
			prediction_endpoint, feature_types=feature_types,
      weight_positives=weight_positives)
  singletask = multitask_to_singletask(dataset)
  arrays = {}
  for target in singletask:
    data = singletask[target]
    if len(data) == 0:
      continue
    if splittype == "random":
      train, test = train_test_random_split(data, seed=seed)
    elif splittype == "scaffold":
      train, test = train_test_scaffold_split(data)
    else:
      raise ValueError("Improper splittype. Must be random/scaffold.")
    X_train, y_train, W_train = dataset_to_numpy(train)
    X_test, y_test, W_test = dataset_to_numpy(test)
    arrays[target] = (train, X_train, y_train, W_train), (test, X_test, y_test, W_test)
  return arrays


def fit_multitask_mlp(paths, task_types, input_transforms, output_transforms,
                      prediction_endpoint, feature_types, splittype="random",
                      weight_positives=False, **training_params):
  """
  Perform stochastic gradient descent optimization for a keras multitask MLP.
  Returns AUCs, R^2 scores, and RMS values.

  Parameters
  ----------
  paths: list 
    List of paths to Google vs datasets. 
  task_types: dict 
    dict mapping target names to output type. Each output type must be either
    "classification" or "regression".
  output_transforms: dict 
    dict mapping target names to label transform. Each output type must be either
    None, "log", "normalize", or "log-normalize". Only for regression outputs.
  training_params: dict
    Aggregates keyword parameters to pass to train_multitask_model
  """
  (train, X_train, y_train, W_train), (test, X_test, y_test, W_test) = (
      process_multitask(paths, input_transforms, output_transforms, feature_types, splittype=splittype,
                        weight_positives=weight_positives))
  print np.shape(y_train)
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

def fit_singletask_mlp(paths, task_types, input_transforms, output_transforms,
											 prediction_endpoint,
                       feature_types,
                       splittype="random", weight_positives=True,
                       num_to_train=None, **training_params):
  """
  Perform stochastic gradient descent optimization for a keras MLP.

  paths: list 
    List of paths to Google vs datasets. 
  task_types: dict 
    dict mapping target names to output type. Each output type must be either
    "classification" or "regression".
  output_transforms: dict 
    dict mapping target names to label transform. Each output type must be either
    None or "log". Only for regression outputs.
  training_params: dict
    Aggregates keyword parameters to pass to train_multitask_model
  """
  singletasks = process_singletask(paths, input_transforms, output_transforms,
		prediction_endpoint, feature_types,
    splittype=splittype, weight_positives=weight_positives)
  ret_vals = {}
  aucs, r2s, rms = {}, {}, {}
  sorted_targets = sorted(singletasks.keys())
  if num_to_train:
    sorted_targets = sorted_targets[:num_to_train]
  for index, target in enumerate(sorted_targets):
    print "Training model %d" % index
    print "Target %s" % target
    (train, X_train, y_train, W_train), (test, X_test, y_test, W_test) = (
        singletasks[target])
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
  if rms:
    print rms
    print "Mean RMS: %f" % np.mean(np.array(rms.values()))

def train_multitask_model(X, y, W, task_types,
  learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True, activation="relu",
  dropout=0.5, nb_epoch=20, batch_size=50, n_hidden=500, n_inputs=1024,
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

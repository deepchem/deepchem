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

def process_multitask(paths, task_transforms, desc_transforms, splittype="random",
    seed=None, add_descriptors=False, weight_positives=False, desc_weight=0.5):
  """Extracts multitask datasets and splits into train/test.

  Returns a tuple of test/train datasets, fingerprints, and labels.

  TODO(rbharath): This function is ugly. Returns way too many arguments. Clean
  it up.

  Parameters
  ----------
  paths: list 
    List of paths to Google vs datasets. 
  task_transforms: dict 
    dict mapping target names to label transform. Each output type must be either
    None or "log". Only for regression outputs.
  desc_transforms: dict
    dict mapping descriptor number to transform. Each transform must be
    either None, "log", "normalize", or "log-normalize"
  splittype: string
    Must be "random" or "scaffold"
  seed: int
    Seed used for random splits.
  """
  dataset = load_and_transform_dataset(paths, task_transforms, desc_transforms,
      add_descriptors=add_descriptors, weight_positives=weight_positives)
  if splittype == "random":
    train, test = train_test_random_split(dataset, seed=seed)
  elif splittype == "scaffold":
    train, test = train_test_scaffold_split(dataset)
  else:
    raise ValueError("Improper splittype. Must be random/scaffold.")
  X_train, y_train, W_train = dataset_to_numpy(train,
      add_descriptors=add_descriptors, desc_weight=desc_weight)
  if weight_positives:
    print "Train set balance"
    ensure_balanced(y_train, W_train)
  X_test, y_test, W_test = dataset_to_numpy(test,
      add_descriptors=add_descriptors, desc_weight=desc_weight)
  if weight_positives:
    print "Test set balance"
    ensure_balanced(y_test, W_test)
  return (train, X_train, y_train, W_train, test, X_test, y_test, W_test)

def process_singletask(paths, task_transforms, desc_transforms, splittype="random", seed=None,
    add_descriptors=False, desc_weight=0.5, weight_positives=True):
  """Extracts singletask datasets and splits into train/test.

  Returns a dict that maps target names to tuples.

  Parameters
  ----------
  paths: list 
    List of paths to Google vs datasets. 
  task_transforms: dict 
    dict mapping target names to label transform. Each output type must be either
    None or "log". Only for regression outputs.
  splittype: string
    Must be "random" or "scaffold"
  seed: int
    Seed used for random splits.
  """
  dataset = load_and_transform_dataset(paths, task_transforms, desc_transforms,
      add_descriptors=add_descriptors, weight_positives=weight_positives)
  singletask = multitask_to_singletask(dataset)
  arrays = {}
  for target in singletask:
    print target
    data = singletask[target]
    print "len(data)"
    print len(data)
    # TODO(rbharath): Remove limitation after debugging.
    if len(data) == 0:
      continue
    if splittype == "random":
      train, test = train_test_random_split(data, seed=seed)
    elif splittype == "scaffold":
      train, test = train_test_scaffold_split(data)
    else:
      raise ValueError("Improper splittype. Must be random/scaffold.")
    X_train, y_train, W_train = dataset_to_numpy(train,
        add_descriptors=add_descriptors, desc_weight=desc_weight)
    X_test, y_test, W_test = dataset_to_numpy(test,
        add_descriptors=add_descriptors, desc_weight=desc_weight)
    arrays[target] = (train, X_train, y_train, W_train, test, X_test, y_test,
        W_test)
  return arrays


def fit_multitask_mlp(paths, task_types, task_transforms, desc_transforms,
                      splittype="random", add_descriptors=False, desc_weight=0.5,
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
  task_transforms: dict 
    dict mapping target names to label transform. Each output type must be either
    None or "log". Only for regression outputs.
  desc_transforms: dict
    dict mapping descriptor number to transform. Each transform must be
    either None, "log", "normalize", or "log-normalize"
  add_descriptors: bool
    Add descriptor prediction as extra task.
  training_params: dict
    Aggregates keyword parameters to pass to train_multitask_model
  """
  (train, X_train, y_train, W_train, test, X_test, y_test, W_test) = (
      process_multitask(paths, task_transforms, desc_transforms,
      splittype=splittype, add_descriptors=add_descriptors, desc_weight=desc_weight,
      weight_positives=weight_positives))
  print np.shape(y_train)
  model = train_multitask_model(X_train, y_train, W_train, task_types,
                                desc_transforms, add_descriptors=add_descriptors,
                                **training_params)
  results = eval_model(test, model, task_types, desc_transforms,
      add_descriptors=add_descriptors, modeltype="keras_multitask")
  if add_descriptors:
    local_task_types = task_types.copy()
    for desc in desc_transforms:
      local_task_types[desc] = "regression"
  else:
    local_task_types = task_types.copy()
  aucs = compute_roc_auc_scores(results, local_task_types)
  if aucs:
    print "Mean AUC: %f" % np.mean(np.array(aucs.values()))
  r2s = compute_r2_scores(results, local_task_types)
  if r2s:
    print "Mean R^2: %f" % np.mean(np.array(r2s.values()))

def fit_singletask_mlp(paths, task_types, task_transforms,
                       desc_transforms, splittype="random",
                       add_descriptors=False, desc_weight=0.5,
                       weight_positives=True, num_to_train=None, **training_params):
  """
  Perform stochastic gradient descent optimization for a keras MLP.

  paths: list 
    List of paths to Google vs datasets. 
  task_types: dict 
    dict mapping target names to output type. Each output type must be either
    "classification" or "regression".
  task_transforms: dict 
    dict mapping target names to label transform. Each output type must be either
    None or "log". Only for regression outputs.
  desc_transforms: dict
    dict mapping descriptor number to transform. Each transform must be
    either None, "log", "normalize", or "log-normalize"
  training_params: dict
    Aggregates keyword parameters to pass to train_multitask_model
  """
  singletasks = process_singletask(paths, task_transforms, desc_transforms,
    splittype=splittype, add_descriptors=add_descriptors,
    desc_weight=desc_weight, weight_positives=weight_positives)
  ret_vals = {}
  aucs, r2s, rms = {}, {}, {}
  sorted_targets = sorted(singletasks.keys())
  if num_to_train:
    sorted_targets = sorted_targets[:num_to_train]
  for index, target in enumerate(sorted_targets):
    print "Training model %d" % index
    (train, X_train, y_train, W_train, test, X_test, y_test, W_test) = (
        singletasks[target])
    model = train_multitask_model(X_train, y_train, W_train,
        {target: task_types[target]}, desc_transforms, add_descriptors=add_descriptors,
        **training_params)
    results = eval_model(test, model, {target: task_types[target]}, 
                         desc_transforms,
                         # We run singletask models as special cases of
                         # multitask.
                         modeltype="keras_multitask",
                         add_descriptors=add_descriptors)
    print "Target %s" % target
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

def train_multitask_model(X, y, W, task_types, desc_transforms, add_descriptors=False,
                      learning_rate=0.01, decay=1e-6,
                      momentum=0.9, nesterov=True, activation="relu",
                      dropout=0.5, nb_epoch=20, batch_size=50, n_hidden=500,
                      n_input=1024, validation_split=0.1):
  """
  Perform stochastic gradient descent optimization for a keras multitask MLP.
  Returns a trained model.

  TODO(rbharath): The handling of add_descriptors for semi-supervised learning
  is horrible. Refactor.

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
  desc_transforms: dict
    dict mapping descriptor number to transform. Each transform must be
    either None, "log", "normalize", or "log-normalize"
  add_descriptors: bool
    Add descriptor prediction as extra task.
  learning_rate: float
    Learning rate used.
  decay: float
    Learning rate decay.
  momentum: float
    Momentum used in SGD.
  nesterov: bool
    Use Nesterov acceleration
  n_epochs: int
    maximal number of epochs to run the optimizer
  """
  eps = .001
  num_tasks = len(task_types)
  sorted_targets = sorted(task_types.keys())
  if add_descriptors:
    sorted_descriptors = sorted(desc_transforms.keys())
    endpoints = sorted_targets + sorted_descriptors
    local_task_types = task_types.copy()
    for desc in desc_transforms:
      local_task_types[desc] = "regression"
  else:
    local_task_types = task_types.copy()
    endpoints = sorted_targets
  print "endpoints: " + str(endpoints)
  # Add eps weight to avoid minibatches with zero weight (causes theano to crash).
  W = W + eps * np.ones(np.shape(W))
  model = Graph()
  model.add_input(name="input", ndim=n_input)
  model.add_node(
      Dense(n_input, n_hidden, init='uniform', activation=activation),
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
  print "decay: " + str(decay)
  model.fit(data_dict, nb_epoch=nb_epoch, batch_size=batch_size, validation_split=validation_split,
            sample_weight=sample_weights)
  return model

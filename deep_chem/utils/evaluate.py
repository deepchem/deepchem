"""
Utility functions to evaluate models on datasets.
"""
__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2015, Stanford University"
__license__ = "LGPL"

import numpy as np
import warnings
from deep_chem.utils.preprocess import dataset_to_numpy
from deep_chem.utils.preprocess import labels_to_weights
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score

def model_predictions(test_set, model, n_targets, task_types,
    modeltype="sklearn"):
  """Obtains predictions of provided model on test_set.

  Returns a list of per-task predictions.

  TODO(rbharath): This function uses n_targets instead of
  task_transforms like everything else.

  Parameters
  ----------
  test_set: dict 
    A dictionary of type produced by load_datasets. Contains the test-set.
  model: model.
    A trained scikit-learn or keras model.
  n_targets: int
    Number of output targets
  task_types: dict 
    dict mapping target names to output type. Each output type must be either
    "classification" or "regression".
  modeltype: string
    Either sklearn, keras, or keras_multitask
  """
  # Extract features for test set and make preds
  X, _, _ = dataset_to_numpy(test_set)
  if modeltype == "keras_multitask":
    predictions = model.predict({"input": X})
    ypreds = []
    for index in range(n_targets):
      ypreds.append(predictions["task%d" % index])
  elif modeltype == "sklearn":
    # Must be single-task (breaking multitask RFs here)
    task_type = task_types.itervalues().next()
    if task_type == "classification":
      ypreds = model.predict_proba(X)
    elif task_type == "regression":
      ypreds = model.predict(X)
  elif modeltype == "keras":
    ypreds = model.predict(X)
  else:
    raise ValueError("Improper modeltype.")
  # Handle the edge case for singletask. 
  if type(ypreds) != list:
    ypreds = [ypreds]
  return ypreds
  
def eval_model(test_set, model, task_types, modeltype="sklearn"):
  """Evaluates the provided model on the test-set.

  Returns a dict which maps target-names to pairs of np.ndarrays (ytrue,
  yscore) of true labels vs. predict

  TODO(rbharath): This function is too complex. Refactor and simplify.

  Parameters
  ----------
  test_set: dict 
    A dictionary of type produced by load_datasets. Contains the test-set.
  model: model.
    A trained scikit-learn or keras model.
  task_types: dict 
    dict mapping target names to output type. Each output type must be either
    "classification" or "regression".
  modeltype: string
    Either sklearn, keras, or keras_multitask
  """
  sorted_targets = sorted(task_types.keys())
  local_task_types = task_types.copy()
  endpoints = sorted_targets
  ypreds = model_predictions(test_set, model, len(sorted_targets),
      local_task_types, modeltype=modeltype)
  results = {}
  for target in endpoints:
    results[target] = ([], [])  # (ytrue, yscore)
  # Iterate through test set data points.
  for index, smiles in enumerate(sorted(test_set.keys())):
    datapoint = test_set[smiles]
    labels = datapoint["labels"]
    for t_ind, target in enumerate(endpoints):
      task_type = local_task_types[target]
      if (task_type == "classification"
        and target in sorted_targets
        and labels[target] == -1):
        continue
      else:
        ytrue, yscore = results[target]
        if task_type == "classification":
          if labels[target] == 0:
            ytrue.append(0)
          elif labels[target] == 1:
            ytrue.append(1)
          else:
            raise ValueError("Labels must be 0/1.")
        elif target in sorted_targets and task_type == "regression":
          ytrue.append(labels[target])
        elif target not in sorted_targets and task_type == "regression":
          descriptors = datapoint["descriptors"]
          # The "target" for descriptors is simply the index in the
          # descriptor vector.
          ytrue.append(descriptors[int(target)])
        else:
          raise ValueError("task_type must be classification or regression.")
        yscore.append(ypreds[t_ind][index])
  for target in endpoints:
    ytrue, yscore = results[target]
    results[target] = (np.array(ytrue), np.array(yscore))
  return results

def compute_roc_auc_scores(results, task_types):
  """Transforms the results dict into roc-auc-scores and prints scores.

  Parameters
  ----------
  results: dict
    A dictionary of type produced by eval_model which maps target-names to
    pairs of lists (ytrue, yscore).
  task_types: dict 
    dict mapping target names to output type. Each output type must be either
    "classification" or "regression".
  """
  scores = {}
  for target in results:
    if task_types[target] != "classification":
      continue
    ytrue, yscore = results[target]
    sample_weights = labels_to_weights(ytrue)
    print "np.shape(ytrue)"
    print np.shape(ytrue)
    print "np.shape(yscore)"
    print np.shape(yscore)
    try:
      score = roc_auc_score(ytrue, yscore[:,1], sample_weight=sample_weights)
    except Exception as e:
      warnings.warn("ERROR! ROC_AUC_SCORE CALCULATION FAILED.")
      score = 0.5
    #score = roc_auc_score(ytrue, yscore, sample_weight=sample_weights)
    print "Target %s: AUC %f" % (target, score)
    scores[target] = score
  return scores

def compute_r2_scores(results, task_types):
  """Transforms the results dict into R^2 values and prints them.

  Parameters
  ----------
  results: dict
    A dictionary of type produced by eval_regression_model which maps target-names to
    pairs of lists (ytrue, yscore).
  task_types: dict 
    dict mapping target names to output type. Each output type must be either
    "classification" or "regression".
  """
  scores = {}
  for target in results:
    if task_types[target] != "regression":
      continue
    ytrue, yscore = results[target]
    score = r2_score(ytrue, yscore)
    print "Target %s: R^2 %f" % (target, score)
    scores[target] = score
  return scores

def compute_rms_scores(results, task_types):
  """Transforms the results dict into RMS values and prints them.

  Parameters
  ----------
  results: dict
    A dictionary of type produced by eval_regression_model which maps target-names to
    pairs of lists (ytrue, yscore).
  task_types: dict 
    dict mapping target names to output type. Each output type must be either
    "classification" or "regression".
  """
  scores = {}
  for target in results:
    if task_types[target] != "regression":
      continue
    ytrue, yscore = results[target]
    rms = np.sqrt(mean_squared_error(ytrue, yscore))
    print "Target %s: RMS %f" % (target, rms)
    scores[target] = rms 
  return scores

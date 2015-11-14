"""
Utility functions to evaluate models on datasets.
"""
__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2015, Stanford University"
__license__ = "LGPL"

import csv
import numpy as np
import warnings
from deep_chem.utils.preprocess import dataset_to_numpy
from deep_chem.utils.preprocess import labels_to_weights
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt

def compute_model_performance(per_task_data, task_types, models, modeltype,
    aucs=True, r2s=False, rms=False):
  """Computes statistics for model performance on test set."""
  all_results, auc_vals, r2_vals, rms_vals = {}, {}, {}, {}
  for index, target in enumerate(sorted(per_task_data.keys())):
    print "Evaluating model %d" % index
    print "Target %s" % target
    (train_ids, Xtrain, ytrain, wtrain), (test_ids, Xtest, ytest, wtest) = per_task_data[target]
    model = models[target]
    results = eval_model(test_ids, Xtest, ytest, wtest, model, {target: task_types[target]}, 
                         modeltype=modeltype)
    #print results
    all_results[target] = results[target]
    if aucs:
      auc_vals.update(compute_roc_auc_scores(results, task_types))
    if r2s: 
      r2_vals.update(compute_r2_scores(results, task_types))
    if rms:
      rms_vals.update(compute_rms_scores(results, task_types))

  if aucs:
    print "Mean AUC: %f" % np.mean(np.array(auc_vals.values()))
  if r2s:
    print "Mean R^2: %f" % np.mean(np.array(r2_vals.values()))
  if rms:
    print "Mean RMS: %f" % np.mean(np.array(rms_vals.values()))
  return all_results, aucs, r2s, rms

def model_predictions(X, model, n_targets, task_types, modeltype="sklearn"):
  """Obtains predictions of provided model on test_set.

  Returns an ndarray of shape (n_samples, n_targets) 

  TODO(rbharath): This function uses n_targets instead of
  task_transforms like everything else.

  Parameters
  ----------
  X: numpy.ndarray 
    Test set data. 
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
  # TODO(rbharath): This change in shape should not(!) be handled here. Make
  # an upstream change so the evaluator doesn't have to worry about this.
  if len(np.shape(X)) > 2:  # Dealing with 3D data
    if len(np.shape(X)) != 5:
      raise ValueError("Tensorial datatype must be of shape (n_samples, N, N, N, n_channels).")
    (n_samples, axis_length, _, _, n_channels) = np.shape(X)
    X = np.reshape(X, (n_samples, axis_length, n_channels, axis_length, axis_length))
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
  ypreds = np.reshape(ypreds, (len(ypreds), n_targets))
  return ypreds

def eval_model(ids, X, Ytrue, W, model, task_types, modeltype="sklearn"):
  """Evaluates the provided model on the test-set.

  Returns a dict which maps target-names to pairs of np.ndarrays (ytrue,
  yscore) of true labels vs. predict

  # TODO(rbharath): Multitask support is now broken. Need to use weight matrix
  # W to get rid of missing data entries for multitask data.

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
  ypreds = model_predictions(X, model, len(task_types),
      task_types, modeltype=modeltype)
  results = {}
  for target_ind, target in enumerate(sorted_targets):
    ytrue, ypred = Ytrue[:, target_ind], ypreds[:, target_ind]
    results[target] = (ids, np.squeeze(ytrue), np.squeeze(ypred))
  return results

def results_to_csv(results, out, task_type="classification"):
  """Writes results as CSV to out."""
  for target in results:
    mol_ids, ytrues, yscores= results[target]
    if task_type == "classification":
      yscores = np.around(yscores[:,1]).astype(int)
    elif task_type == "regression":
      if type(yscores[0]) == np.ndarray:
        yscores = yscores[:,0]
    with open(out, "wb") as csvfile:
      csvwriter = csv.writer(csvfile, delimiter="\t")
      csvwriter.writerow(["Ids", "True", "Model-Prediction"])
      for id, ytrue, yscore in zip(mol_ids, ytrues, yscores):
        csvwriter.writerow([id, ytrue, yscore])
    print "Writing results on test set for target %s to %s" % (target, out)
    

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
    _, ytrue, yscore = results[target]
    sample_weights = labels_to_weights(ytrue)
    try:
      score = roc_auc_score(ytrue, yscore[:,1], sample_weight=sample_weights)
    except Exception as e:
      warnings.warn("ROC AUC score calculation failed.")
      score = 0.5
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
    _, ytrue, yscore = results[target]
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
    _, ytrue, yscore = results[target]
    rms = np.sqrt(mean_squared_error(ytrue, yscore))
    print "Target %s: RMS %f" % (target, rms)
    scores[target] = rms 
  return scores

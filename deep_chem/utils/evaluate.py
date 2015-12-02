"""
Utility functions to evaluate models on datasets.
"""
from __future__ import print_function

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2015, Stanford University"
__license__ = "LGPL"

import csv
import numpy as np
import warnings
import sys
from deep_chem.utils.preprocess import labels_to_weights
from deep_chem.utils.preprocess import undo_transform_outputs
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

def compute_model_performance(raw_test_data, test_data, task_types, models,
                              modeltype, output_transforms, aucs=False,
                              r2s=False, rms=False, recall=False,
                              accuracy=False, mcc=False,
                              print_file=sys.stdout):
  """Computes statistics for model performance on test set."""
  all_results = {}
  auc_vals, mcc_vals, recall_vals, accuracy_vals = {}, {}, {}, {}
  r2_vals, rms_vals = {}, {}
  test_ids = test_data["mol_ids"]
  X_test = test_data["features"]
  for index, target in enumerate(test_data["sorted_tasks"]):
    print("Evaluating model %d" % index, file=print_file)
    print("Target %s" % target, file=print_file)
    (y_test, w_test) = test_data[target]
    (ytest_raw, _) = raw_test_data[target]
    #model = models[target]
    model = models.itervalues().next()
    results = eval_model(
        test_ids, X_test, y_test, ytest_raw, w_test, model,
        {target: task_types[target]}, modeltype=modeltype,
        output_transforms=output_transforms)
    all_results[target] = results[target]
    if aucs:
      auc_vals.update(compute_roc_auc_scores(results, task_types))
    if r2s:
      r2_vals.update(compute_r2_scores(results, task_types))
    if rms:
      rms_vals.update(compute_rms_scores(results, task_types))
    if mcc:
      mcc_vals.update(compute_matthews_corr(results, task_types))
    if recall:
      recall_vals.update(compute_recall_score(results, task_types))
    if accuracy:
      recall_vals.update(compute_accuracy_score(results, task_types))

  if aucs:
    print("Mean AUC: %f" % np.mean(np.array(auc_vals.values())), file=print_file)
  if r2s:
    print("Mean R^2: %f" % np.mean(np.array(r2_vals.values())), file=print_file)
  if rms:
    print("Mean RMS: %f" % np.mean(np.array(rms_vals.values())), file=print_file)
  if mcc:
    print("Mean MCC: %f" % np.mean(np.array(mcc_vals.values())), file=print_file)
  if recall:
    print("Mean Recall: %f" % np.mean(np.array(recall_vals.values())), file=print_file)
  if accuracy:
    print("Mean Accuracy: %f" % np.mean(np.array(accuracy_vals.values())), file=print_file)

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
      raise ValueError(
          "Tensorial datatype must be of shape (n_samples, N, N, N, n_channels).")
    (n_samples, axis_length, _, _, n_channels) = np.shape(X)
    X = np.reshape(X, (n_samples, axis_length, n_channels, axis_length, axis_length))
  if modeltype == "keras-graph":
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
  elif modeltype == "keras-sequential":
    ypreds = model.predict(X)
  else:
    raise ValueError("Improper modeltype.")
  if isinstance(ypreds, np.ndarray):
    ypreds = np.squeeze(ypreds)
  if not isinstance(ypreds, list):
    ypreds = [ypreds]
  return ypreds

def eval_model(ids, X, Ytrue, Ytrue_raw, W, model, task_types,
               output_transforms, modeltype="sklearn"):
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
  ypreds = model_predictions(
      X, model, len(task_types), task_types, modeltype=modeltype)
  results = {}
  for target_ind, target in enumerate(sorted_targets):
    ytrue_raw, _, ypred = Ytrue_raw[:, target_ind], Ytrue[:, target_ind], ypreds[target_ind]
    ypred = undo_transform_outputs(ytrue_raw, ypred, output_transforms)
    results[target] = (ids, np.squeeze(ytrue_raw), np.squeeze(ypred))
  return results

def results_to_csv(results, out, task_type="classification"):
  """Writes results as CSV to out."""
  for target in results:
    mol_ids, ytrues, yscores = results[target]
    if task_type == "classification":
      yscores = np.around(yscores[:, 1]).astype(int)
    elif task_type == "regression":
      if isinstance(yscores[0], np.ndarray):
        yscores = yscores[:, 0]
    with open(out, "wb") as csvfile:
      csvwriter = csv.writer(csvfile, delimiter="\t")
      csvwriter.writerow(["Ids", "True", "Model-Prediction"])
      for mol_id, ytrue, yscore in zip(mol_ids, ytrues, yscores):
        csvwriter.writerow([mol_id, ytrue, yscore])
    print("Writing results on test set for target %s to %s" % (target, out))


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
    print("Target %s: R^2 %f" % (target, score))
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
    print("Target %s: RMS %f" % (target, rms))
    scores[target] = rms
  return scores

def compute_roc_auc_scores(results, task_types):
  """Transforms the results dict into roc-auc-scores and prints scores.

  Parameters
  ----------
  results: dict
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
      score = roc_auc_score(ytrue, yscore[:, 1], sample_weight=sample_weights)
    except Exception:
      warnings.warn("ROC AUC score calculation failed.")
      score = 0.5
    print("Target %s: AUC %f" % (target, score))
    scores[target] = score
  return scores

def compute_matthews_corr(results, task_types):
  """Computes Matthews Correlation Coefficients."""
  scores = {}
  for target in results:
    if task_types[target] != "classification":
      continue
    _, ytrue, ypred = results[target]
    mcc = matthews_corrcoef(ytrue, np.around(ypred[:, 1]))
    print("Target %s: MCC %f" % (target, mcc))
    scores[target] = mcc
  return scores

def compute_recall_score(results, task_types):
  """Computes recall score."""
  scores = {}
  for target in results:
    if task_types[target] != "classification":
      continue
    _, ytrue, ypred = results[target]
    recall = recall_score(ytrue, np.around(ypred[:, 1]))
    print("Target %s: Recall %f" % (target, recall))
    scores[target] = recall
  return scores

def compute_accuracy_score(results, task_types):
  """Computes accuracy score."""
  scores = {}
  for target in results:
    if task_types[target] != "classification":
      continue
    _, ytrue, ypred = results[target]
    accuracy = accuracy_score(ytrue, np.around(ypred[:, 1]))
    print("Target %s: Accuracy %f" % (target, accuracy))
    scores[target] = accuracy
  return scores

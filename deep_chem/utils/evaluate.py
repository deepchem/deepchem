"""
Utility functions to evaluate models on datasets.
"""
from __future__ import print_function
from __future__ import division
import csv
import numpy as np
import warnings
import sys
from deep_chem.utils.preprocess import undo_transform_outputs
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2015, Stanford University"
__license__ = "LGPL"


def eval_trained_model(modeltype, saved_model, saved_data, task_type,
                      csv_out, stats_out, target_names):
  """Evaluates a trained model on specified data."""
  model = load_model(modeltype, saved_model)
  task_types = {target: task_type for target in target_names}

  stored_test = load_sharded_dataset(saved_data)
  test_dict = stored_test["transformed"]
  raw_test_dict = stored_test["raw"]
  output_transforms = stored_test["transforms"]["output_transform"]

  with open(stats_out, "wb") as stats_file:
    results, _, _, _ = compute_model_performance(
        raw_test_dict, test_dict, task_types, model, modeltype,
        output_transforms, aucs=compute_aucs, r2s=compute_r2s, rms=compute_rms,
        recall=compute_recall, accuracy=compute_accuracy,
        mcc=compute_matthews_corrcoef, print_file=stats_file)
  with open(stats_out, "r") as stats_file:
    print(stats_file.read())
  results_to_csv(results, csv_out, task_type=task_type)

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
    # Find nonzero rows in weight matrix
    # http://stackoverflow.com/questions/23726026/finding-which-rows-have-all-elements-as-zeros-in-a-matrix-with-numpy
    nonzero_ws = np.where(w_test.any(axis=1))[0]
    task_w_test = w_test[nonzero_ws]
    task_X_test = X_test[nonzero_ws]
    task_y_test = y_test[nonzero_ws]
    task_y_test_raw = y_test[nonzero_ws]
    model = models[target]
    # TODO(rbharath): Multitask vs. singletask is explicitly cased here and
    # below. This is awkward. More elegant way of handling this?
    if target == "all":
      eval_task_types = task_types.copy()
    else:
      eval_task_types = {target: task_types[target]}

    results = eval_model(
        test_ids, task_X_test, task_y_test, task_y_test_raw, model,
        eval_task_types, modeltype=modeltype,
        output_transforms=output_transforms)
    if target == "all":
      all_results.update(results)
    else:
      all_results[target] = results[target]
    # Classification Metrics
    if aucs:
      auc_vals.update(compute_roc_auc_scores(results, task_w_test, task_types))
    if mcc:
      mcc_vals.update(compute_matthews_corr(results, task_w_test, task_types))
    if recall:
      recall_vals.update(compute_recall_score(results, task_w_test, task_types))
    if accuracy:
      accuracy_vals.update(compute_accuracy_score(results, task_w_test, task_types))
    # Regression Metrics
    if r2s:
      r2_vals.update(compute_r2_scores(results, task_types))
    if rms:
      rms_vals.update(compute_rms_scores(results, task_types))
  # Print classification metrics
  if aucs:
    print("AUCs", file=print_file)
    print(auc_vals, file=print_file)
    print("Mean AUC: %f" % np.mean(np.array(auc_vals.values())), file=print_file)
  if mcc:
    print("MCCs", file=print_file)
    print(mcc_vals, file=print_file)
    print("Mean MCC: %f" % np.mean(np.array(mcc_vals.values())), file=print_file)
  if recall:
    print("Recalls", file=print_file)
    print(recall_vals, file=print_file)
    print("Mean Recall: %f" % np.mean(np.array(recall_vals.values())), file=print_file)
  if accuracy:
    print("Accuracies", file=print_file)
    print(accuracy_vals, file=print_file)
    print("Mean Accuracy: %f" % np.mean(np.array(accuracy_vals.values())), file=print_file)
  # Print regression metrics
  if r2s:
    print("R^2s", file=print_file)
    print(r2_vals, file=print_file)
    print("Mean R^2: %f" % np.mean(np.array(r2_vals.values())), file=print_file)
  if rms:
    print("RMSs", file=print_file)
    print(rms_vals, file=print_file)
    print("Mean RMS: %f" % np.mean(np.array(rms_vals.values())), file=print_file)

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
      print("model_predictions()")
      print("np.shape(X)")
      print(np.shape(X))
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

def eval_model(ids, X, Ytrue, Ytrue_raw, model, task_types,
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
  sorted_tasks = sorted(results.keys())
  processed_results = {}
  for task in sorted_tasks:
    mol_ids, ytrues, yscores = results[task]
    if task_type == "classification":
      yscores = np.around(yscores[:, 1]).astype(int)
    elif task_type == "regression":
      if isinstance(yscores[0], np.ndarray):
        yscores = yscores[:, 0]
    for (mol_id, ytrue, yscore) in zip(mol_ids, ytrues, yscores):
      if mol_id not in processed_results:
        processed_results[mol_id] = {}
      processed_results[mol_id][task] = (ytrue, yscore)
  with open(out, "wb") as csvfile:
    csvwriter = csv.writer(csvfile, delimiter="\t")
    colnames = ["Ids"]
    for task in sorted_tasks:
      colnames += ["%s-True" % task, "%s-Pred" % task]
    csvwriter.writerow(colnames)
    for mol_id in processed_results:
      col = [mol_id]
      for task in sorted_tasks:
        if task in processed_results[mol_id]:
          ytrue, yscore = processed_results[mol_id][task]
          # TODO(rbharath): Missing data handling is broken for multitask
          # regression models! Find a more general solution (perhaps by using
          # NaNs in place of -1)
          # Handle missing data case
          if ytrue == -1:
            (ytrue, yscore) = ("", "")
        else:
          (ytrue, yscore) = ("", "")
        col += [ytrue, yscore]
      csvwriter.writerow(col)
  print("Writing results on test set to %s" % out)

def compute_r2_scores(results, task_types):
  """Transforms the results dict into R^2 values and prints them.

  Parameters
  ----------
  results: dict
    A dictionary of type produced by eval_regression_model which maps task-names to
    pairs of lists (ytrue, yscore).
  task_types: dict
    dict mapping task names to output type. Each output type must be either
    "classification" or "regression".
  """
  scores = {}
  for task in results:
    if task_types[task] != "regression":
      continue
    _, ytrue, yscore = results[task]
    score = r2_score(ytrue, yscore)
    print("Target %s: R^2 %f" % (task, score))
    scores[task] = score
  return scores

def compute_rms_scores(results, task_types):
  """Transforms the results dict into RMS values and prints them.

  Parameters
  ----------
  results: dict
    A dictionary of type produced by eval_regression_model which maps task-names to
    pairs of lists (ytrue, yscore).
  task_types: dict
    dict mapping task names to output type. Each output type must be either
    "classification" or "regression".
  """
  scores = {}
  for task in results:
    if task_types[task] != "regression":
      continue
    _, ytrue, yscore = results[task]
    rms = np.sqrt(mean_squared_error(ytrue, yscore))
    print("Target %s: RMS %f" % (task, rms))
    scores[task] = rms
  return scores

# TODO(rbharath): The following functions have a lot of boilerplate. Refactor?
def compute_roc_auc_scores(results, weights, task_types):
  """Transforms the results dict into roc-auc-scores and prints scores.

  Parameters
  ----------
  results: dict
  task_types: dict
    dict mapping task names to output type. Each output type must be either
    "classification" or "regression".
  """
  scores = {}
  for ind, task in enumerate(sorted(results.keys())):
    if task_types[task] != "classification":
      continue
    wtrue = weights[:, ind].ravel()
    _, ytrue, yscore = results[task]
    ytrue, yscore = ytrue[wtrue.nonzero()], yscore[wtrue.nonzero()]
    try:
      score = roc_auc_score(ytrue, yscore[:, 1], sample_weight=wtrue)
    except Exception:
      warnings.warn("ROC AUC score calculation failed.")
      score = 0.5
    print("Target %s: AUC %f" % (task, score))
    scores[task] = score
  return scores

def compute_matthews_corr(results, weights, task_types):
  """Computes Matthews Correlation Coefficients."""
  scores = {}
  for ind, task in enumerate(sorted(results.keys())):
    if task_types[task] != "classification":
      continue
    wtrue = weights[:, ind].ravel()
    _, ytrue, ypred = results[task]
    ytrue, ypred = ytrue[wtrue.nonzero()], ypred[wtrue.nonzero()]

    mcc = matthews_corrcoef(ytrue, np.around(ypred[:, 1]))
    print("Target %s: MCC %f" % (task, mcc))
    scores[task] = mcc
  return scores

def compute_recall_score(results, weights, task_types):
  """Computes recall score."""
  scores = {}
  for ind, task in enumerate(sorted(results.keys())):
    if task_types[task] != "classification":
      continue
    wtrue = weights[:, ind].ravel()
    _, ytrue, ypred = results[task]
    ytrue, ypred = ytrue[wtrue.nonzero()], ypred[wtrue.nonzero()]

    recall = recall_score(ytrue, np.around(ypred[:, 1]))
    print("Target %s: Recall %f" % (task, recall))
    scores[task] = recall
  return scores

def compute_accuracy_score(results, weights, task_types):
  """Computes accuracy score."""
  scores = {}
  for ind, task in enumerate(sorted(results.keys())):
    if task_types[task] != "classification":
      continue
    wtrue = weights[:, ind].ravel()
    _, ytrue, ypred = results[task]
    ytrue, ypred = ytrue[wtrue.nonzero()], ypred[wtrue.nonzero()]
    accuracy = accuracy_score(ytrue, np.around(ypred[:, 1]))
    print("Target %s: Accuracy %f" % (task, accuracy))
    scores[task] = accuracy
  return scores

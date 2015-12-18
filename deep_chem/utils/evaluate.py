"""
Utility functions to evaluate models on datasets.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import warnings
#from deep_chem.utils.preprocess import undo_transform_outputs
from deep_chem.utils.preprocess import get_metadata_filename
from deep_chem.utils.preprocess import get_sorted_task_names
from deep_chem.utils.preprocess import get_task_type
from deep_chem.utils.save import load_model
from deep_chem.utils.save import load_sharded_dataset
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import pandas as pd

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2015, Stanford University"
__license__ = "LGPL"

def eval_trained_model(model_name, model_dir, data_dir,
                       csv_out, stats_out, split="test"):
  """Evaluates a trained model on specified data."""
  model = load_model(model_name, model_dir)
  task_type = get_task_type(model_name)
  task_names, pred_y_df = compute_y_pred(model, data_dir, csv_out, split)
  compute_model_performance(pred_y_df, task_names, task_type, stats_out)

def compute_y_pred(model, data_dir, csv_out, split):
  """
  Computes model predictions on data and stores csv to disk.
  """
  metadata_filename = get_metadata_filename(data_dir)
  metadata_df = load_sharded_dataset(metadata_filename)
  task_names = metadata_df.iterrows().next()[1]['task_names']
  pred_task_names = ["%s_pred" % task_name for task_name in task_names]
  w_task_names = ["%s_weight" % task_name for task_name in task_names]
  column_names = ['ids'] + task_names + pred_task_names + w_task_names
  pred_y_df = pd.DataFrame(columns=column_names)

  for _, row in metadata_df.iterrows():
    if row['split'] == split:
      X = load_sharded_dataset(row['X'])
      y = load_sharded_dataset(row['y'])
      w = load_sharded_dataset(row['w'])
      ids = load_sharded_dataset(row['ids'])

      y_pred = model.predict_on_batch(X)
      y_pred = np.reshape(y_pred, np.shape(y))

      mini_df = pd.DataFrame(columns=column_names)
      mini_df['ids'] = ids
      mini_df[task_names] = y
      mini_df[pred_task_names] = y_pred
      mini_df[w_task_names] = w
      pred_y_df = pd.concat([pred_y_df, mini_df])

  print("Saving predictions to %s" % csv_out)
  pred_y_df.to_csv(csv_out)
  print("Saved.")

  return task_names, pred_y_df

def compute_model_performance(pred_y_df, task_names, task_type, stats_file):
  """
  Computes statistics of model on test data and saves results to csv.
  """
  if task_type == "classification":
    colnames = ["task_name", "roc_auc_score", "matthews_corrcoef", "recall_score", "accuracy_score"]
  elif task_type == "regression":
    colnames = ["task_name", "r2_score", "rms_error"]
  else:
    raise ValueError("Unrecognized task type: %s" % task_type)

  performance_df = pd.DataFrame(columns=colnames)

  for i, task_name in enumerate(task_names):
    y = pred_y_df[task_name]
    y_pred = pred_y_df["%s_pred" % task_name]
    w = pred_y_df["%s_weight" % task_name]

    if task_type == "classification":
      y, y_pred = y[w.nonzero()], y_pred[w.nonzero()][:, 1]
      auc = compute_roc_auc_scores(y, y_pred, w)
      mcc = matthews_corrcoef(y, np.around(y_pred))
      recall = recall_score(y, np.around(y_pred))
      accuracy = accuracy_score(y, np.around(y_pred))
      performance_df.loc[i] = [task_name, auc, mcc, recall, accuracy]

    elif task_type == "regression":
      r2s = r2_score(y, y_pred)
      rms = np.sqrt(mean_squared_error(y, y_pred))
      performance_df.loc[i] = [task_name, r2s, rms]

  print("Saving model performance scores to %s" % stats_file)
  performance_df.to_csv(stats_file)
  print("Saved.")
  
  print("Model performance scores:")
  print(performance_df)

#TODO(enf/rhbarath): This might work, this might be broken.
def compute_roc_auc_scores(y, y_pred, w):
  """Transforms the results dict into roc-auc-scores and prints scores.

  Parameters
  ----------
  results: dict
  task_types: dict
    dict mapping task names to output type. Each output type must be either
    "classification" or "regression".
  """
  try:
    score = roc_auc_score(y, y_pred, sample_weight=w)
  except Exception:
    warnings.warn("ROC AUC score calculation failed.")
    score = 0.5
  return score

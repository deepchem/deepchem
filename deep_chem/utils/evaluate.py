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
from deep_chem.utils.preprocess import undo_transform
from deep_chem.utils.save import load_model
from deep_chem.utils.save import load_sharded_dataset
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import pandas as pd
import sys

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2015, Stanford University"
__license__ = "LGPL"

def eval_trained_model(model_name, model_dir, data_dir,
                       csv_out, stats_out, output_transforms, split="test"):
  """Evaluates a trained model on specified data."""
  model = load_model(model_name, model_dir)
  task_type = get_task_type(model_name)
  task_names, pred_y_df = compute_y_pred(model, data_dir, csv_out, split)
  compute_model_performance(pred_y_df, task_names, 
                            task_type, stats_out, output_transforms)

def compute_y_pred(model, data_dir, csv_out, split):
  """
  Computes model predictions on data and stores csv to disk.
  """
  metadata_filename = get_metadata_filename(data_dir)
  metadata_df = load_sharded_dataset(metadata_filename)
  task_names = metadata_df.iterrows().next()[1]['task_names']
  pred_task_names = ["%s_pred" % task_name for task_name in task_names]
  w_task_names = ["%s_weight" % task_name for task_name in task_names]
  column_names = (['ids'] + task_names + pred_task_names + w_task_names
                         + ["y_means", "y_stds"])
  pred_y_df = pd.DataFrame(columns=column_names)

  split_df = metadata_df.loc[metadata_df['split'] == split]
  nb_batch = split_df.shape[0]
  MAX_GPU_RAM = float(691007488/50)

  for i, row in split_df.iterrows():
    print("Evaluating on %s batch %d out of %d" % (split, i+1, nb_batch))
    X = load_sharded_dataset(row['X-transformed'])
    y = load_sharded_dataset(row['y-transformed'])
    w = load_sharded_dataset(row['w'])
    ids = load_sharded_dataset(row['ids'])

    if sys.getsizeof(X) > MAX_GPU_RAM:
      nb_block = float(sys.getsizeof(X))/MAX_GPU_RAM
      nb_sample = np.shape(X)[0]
      interval_points = np.linspace(0,nb_sample,nb_block+1).astype(int)
      y_preds = []
      for j in range(0,len(interval_points)-1):
        indices = range(interval_points[j],interval_points[j+1])
        X_batch = X[indices,:]
        y_batch = y[indices]
        w_batch = w[indices]
        y_preds.append(model.predict_on_batch(X_batch))
      y_pred = np.concatenate(y_preds)
    else:
      y_pred = model.predict_on_batch(X)

    y_pred = np.reshape(y_pred, np.shape(y))

    mini_df = pd.DataFrame(columns=column_names)
    mini_df['ids'] = ids
    mini_df[task_names] = y
    mini_df[pred_task_names] = y_pred
    mini_df[w_task_names] = w
    mini_df["y_means"] = split_df["y_means"]
    mini_df["y_stds"] = split_df["y_stds"]
    pred_y_df = pd.concat([pred_y_df, mini_df])

  print("Saving predictions to %s" % csv_out)
  pred_y_df.to_csv(csv_out)
  print("Saved.")

  return task_names, pred_y_df

def compute_model_performance(pred_y_df, task_names, task_type, stats_file, output_transforms):
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
  print("compute_model_performance()")
  print("pred_y_df")
  print(pred_y_df)
  y_means = pred_y_df.iterrows().next()[1]["y_means"]
  y_stds = pred_y_df.iterrows().next()[1]["y_stds"]

  for i, task_name in enumerate(task_names):
    y = pred_y_df[task_name]
    y_pred = pred_y_df["%s_pred" % task_name]
    w = pred_y_df["%s_weight" % task_name]

    y = undo_transform(y, y_means, y_stds, output_transforms)
    y_pred = undo_transform(y_pred, y_means, y_stds, output_transforms)

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

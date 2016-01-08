"""
Utility functions to evaluate models on datasets.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import warnings
from deepchem.utils.dataset import ShardedDataset
from deepchem.utils.preprocess import get_task_type
from deepchem.utils.preprocess import undo_transform
from deepchem.utils.dataset import load_from_disk
from deepchem.models import Model 
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

def eval_trained_model(model_type, model_dir, data_dir,
                       csv_out, stats_out, output_transforms, split="test"):
  """Evaluates a trained model on specified data."""
  model = Model.load(model_type, model_dir)
  task_type = get_task_type(model_type)
  test = ShardedDataset(data_dir)
  task_names = test.get_task_names()
  pred_y_df = model.predict(test)

  print("Saving predictions to %s" % csv_out)
  pred_y_df.to_csv(csv_out)
  compute_model_performance(pred_y_df, task_names, 
                            task_type, stats_out, output_transforms)


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
  y_means = pred_y_df.iterrows().next()[1]["y_means"]
  y_stds = pred_y_df.iterrows().next()[1]["y_stds"]

  print("compute_model_performance()")
  for i, task_name in enumerate(task_names):
    y = pred_y_df[task_name]
    y_pred = pred_y_df["%s_pred" % task_name]
    w = pred_y_df["%s_weight" % task_name]
    
    print("y_means")
    print(y_means)
    print("y_stds")
    print(y_stds)
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

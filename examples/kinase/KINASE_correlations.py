"""
Script that computes correlations of KINASE tasks. 
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import tempfile
import shutil
import deepchem as dc
import pandas as pd
from KINASE_datasets import load_kinase

###Load data###
np.random.seed(123)
shard_size = 2000
print("About to load KINASE data.")
KINASE_tasks, datasets, transformers = load_kinase(shard_size=shard_size)
train_dataset, valid_dataset, test_dataset = datasets

y_train = train_dataset.y
n_tasks = y_train.shape[1]

all_results = []
for task in range(n_tasks):
  y_task = y_train[:, task]
  task_results = []
  for other_task in range(n_tasks):
    if task == other_task:
      task_results.append(1.)
      continue
    y_other = y_train[:, other_task]
    r2 = dc.metrics.pearson_r2_score(y_task, y_other)
    print("r2 for %s-%s is %f" % (task, other_task, r2))
    task_results.append(r2)
  print("Task %d" % task)
  print(task_results)
  all_results.append(task_results)
print("Writing results to kinase_corr.csv")
df = pd.DataFrame(all_results)
df.to_csv("kinase_corr.csv")

"""
Script that loads random forest models trained on the sider and toxcast datasets, predicts on sweetlead,
creates covariance matrix
@Author Aneesh Pappu
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from deepchem.models.multitask import SingletaskToMultitask
from deepchem import metrics
from deepchem.metrics import Metric
from deepchem.models.sklearn_models import SklearnModel
from deepchem.splits import StratifiedSplitter, RandomSplitter
from sweetlead_datasets import load_sweet

sys.path.append('./../toxcast')
sys.path.append('./../sider')

from tox_datasets import load_tox
from sider_datasets import load_sider

"""
Load toxicity models now
"""

# Set some global variables up top
reload = False
verbosity = "high"

base_tox_data_dir = "/home/apappu/deepchem-models/toxcast_models/toxcast/toxcast_data"

tox_tasks, tox_dataset, tox_transformers = load_tox(
    base_tox_data_dir, reload=reload)

#removes directory if present -- warning
base_tox_dir = "/home/apappu/deepchem-models/toxcast_models/toxcast/toxcast_analysis"

tox_train_dir = os.path.join(base_tox_dir, "train_dataset")
tox_valid_dir = os.path.join(base_tox_dir, "valid_dataset")
tox_test_dir = os.path.join(base_tox_dir, "test_dataset")
tox_model_dir = os.path.join(base_tox_dir, "model")

tox_splitter = StratifiedSplitter()

#default split is 80-10-10 train-valid-test split
tox_train_dataset, tox_valid_dataset, tox_test_dataset = tox_splitter.train_valid_test_split(
  tox_dataset, tox_train_dir, tox_valid_dir, tox_test_dir)

# Fit Logistic Regression models
tox_task_types = {task: "classification" for task in tox_tasks}


classification_metric = Metric(metrics.roc_auc_score, np.mean,
                               verbosity=verbosity,
                               mode="classification")
params_dict = {
    "batch_size": None,
    "data_shape": tox_train_dataset.get_data_shape(),
}

def model_builder(tasks, task_types, model_params, model_dir, verbosity=None):
  return SklearnModel(tasks, task_types, model_params, model_dir,
                      model_instance=RandomForestClassifier(
                          class_weight="balanced",
                          n_estimators=500,
                          n_jobs=-1),
                      verbosity=verbosity)
tox_model = SingletaskToMultitask(tox_tasks, tox_task_types, params_dict, tox_model_dir,
                              model_builder, verbosity=verbosity)
tox_model.reload()

"""
Load sider models now
"""

base_sider_data_dir = "/home/apappu/deepchem-models/toxcast_models/sider/sider_data"

sider_tasks, sider_dataset, sider_transformers = load_sider(
    base_sider_data_dir, reload=reload)

base_sider_dir = "/home/apappu/deepchem-models/toxcast_models/sider/sider_analysis"

sider_train_dir = os.path.join(base_sider_dir, "train_dataset")
sider_valid_dir = os.path.join(base_sider_dir, "valid_dataset")
sider_test_dir = os.path.join(base_sider_dir, "test_dataset")
sider_model_dir = os.path.join(base_sider_dir, "model")

sider_splitter = RandomSplitter()
sider_train_dataset, sider_valid_dataset, sider_test_dataset = sider_splitter.train_valid_test_split(
  sider_dataset, sider_train_dir, sider_valid_dir, sider_test_dir)

# Fit Logistic Regression models
sider_task_types = {task: "classification" for task in sider_tasks}

params_dict = {
  "batch_size": None,
  "data_shape": sider_train_dataset.get_data_shape(),
}

sider_model = SingletaskToMultitask(sider_tasks, sider_task_types, params_dict, sider_model_dir,
                              model_builder, verbosity=verbosity)
sider_model.reload()

"""
Load sweetlead dataset now. Pass in dataset object and appropriate transformers to predict functions
"""

base_sweet_data_dir = "/home/apappu/deepchem-models/toxcast_models/sweetlead/sweet_data"

sweet_dataset, sweet_transformers = load_sweet(
    base_sweet_data_dir, reload=reload)

sider_predictions = sider_model.predict(sweet_dataset, sweet_transformers)

tox_predictions = tox_model.predict(sweet_dataset, sweet_transformers)

sider_dimensions = sider_predictions.shape[1]
tox_dimensions = tox_predictions.shape[1]

confusion_matrix = np.zeros(shape=(tox_dimensions, sider_dimensions))
for i in range(tox_predictions.shape[0]):
  nonzero_tox = np.nonzero(tox_predictions[i, :])
  nonzero_sider = np.nonzero(sider_predictions[i, :])
  for j in nonzero_tox[0]:
    for k in nonzero_sider[0]:
      confusion_matrix[j,k] +=1
 
df = pd.DataFrame(confusion_matrix)

df.to_csv("./tox_sider_matrix.csv")


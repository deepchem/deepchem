"""
Script that trains Sklearn multitask models on the sider dataset
@Author Bharath Ramsundar, Aneesh Pappu
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import shutil
from sklearn.ensemble import RandomForestClassifier
from deepchem.datasets import Dataset
from deepchem.models.multitask import SingletaskToMultitask
from deepchem import metrics
from deepchem.metrics import Metric
from deepchem.models.sklearn_models import SklearnModel
from deepchem.utils.evaluate import Evaluator
from sider_datasets import load_sider
from deepchem.splits import StratifiedSplitter, RandomSplitter

# Set some global variables up top
reload = False
verbosity = "high"

base_data_dir = "/home/apappu/deepchem-models/toxcast_models/sider/sider_data"

sider_tasks, sider_dataset, transformers = load_sider(
    base_data_dir, reload=reload)

#removes directory if present -- warning
base_dir = "/home/apappu/deepchem-models/toxcast_models/sider/sider_analysis"
if os.path.exists(base_dir):
  shutil.rmtree(base_dir)
if not os.path.exists(base_dir):
  os.makedirs(base_dir)
train_dir = os.path.join(base_dir, "train_dataset")
valid_dir = os.path.join(base_dir, "valid_dataset")
test_dir = os.path.join(base_dir, "test_dataset")
model_dir = os.path.join(base_dir, "model")

print("About to perform train/valid/test split.")
#train_scores_list = []
#valid_scores_list = []
#splitters = [RandomSplitter(), StratifiedSplitter()]

#for splitter in splitters:
  #default split is 80-10-10 train-valid-test split
  
splitter = RandomSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
  sider_dataset, train_dir, valid_dir, test_dir)

train_dataset.set_verbosity(verbosity)
valid_dataset.set_verbosity(verbosity)
test_dataset.set_verbosity(verbosity)

# Fit Logistic Regression models
sider_task_types = {task: "classification" for task in sider_tasks}


classification_metric = Metric(metrics.roc_auc_score, np.mean,
                               verbosity=verbosity,
                               mode="classification")
params_dict = {
    "batch_size": None,
    "data_shape": train_dataset.get_data_shape(),
}

if os.path.exists(model_dir):
  shutil.rmtree(model_dir)
os.makedirs(model_dir)
def model_builder(tasks, task_types, model_params, model_dir, verbosity=None):
  return SklearnModel(tasks, task_types, model_params, model_dir,
                      model_instance=RandomForestClassifier(
                          class_weight="balanced",
                          n_estimators=500,
                          n_jobs=-1),
                      verbosity=verbosity)
model = SingletaskToMultitask(sider_tasks, sider_task_types, params_dict, model_dir,
                              model_builder, verbosity=verbosity)

# Fit trained model
model.fit(train_dataset)
model.save()

train_evaluator = Evaluator(model, train_dataset, transformers, verbosity=verbosity)
train_scores = train_evaluator.compute_model_performance([classification_metric])

print("Train scores")
#  train_scores_list.append(train_scores)

valid_evaluator = Evaluator(model, valid_dataset, transformers, verbosity=verbosity)
valid_scores = valid_evaluator.compute_model_performance([classification_metric])

print("Validation scores")
#  valid_scores_list.append(valid_scores)
#
#for i in range(2):
#  if i == 0:
#    print("Random Splitter")
#    print("Train Scores")
#  elif i == 1:
#    print("Stratified Splitter")
#    print("Train Scores")
#  print(train_scores_list[i])
#
#  if i == 0:
#    print("Random Splitter")
#    print("Valid Scores")
#  elif i == 1:
#    print("Stratified Splitter")
#    print("Valid Scores")
#  print(valid_scores_list[i])

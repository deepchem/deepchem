"""
Script that trains Sklearn multitask models on nci dataset.

"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import shutil
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from deepchem.datasets import Dataset
from deepchem.models.multitask import SingletaskToMultitask
from deepchem import metrics
from deepchem.metrics import Metric
from deepchem.models.sklearn_models import SklearnModel
from deepchem.utils.evaluate import Evaluator
from deepchem.datasets.nci_datasets import load_nci
from deepchem.splits import RandomSplitter

np.random.seed(123)

# Set some global variables up top

reload = True
verbosity = "high"
force_transform = False 

base_data_dir = "/scratch/users/rbharath/nci_data_dir"
base_dir = "/scratch/users/rbharath/nci_analysis_dir"

nci_tasks, nci_dataset, transformers = load_nci(
    base_data_dir, reload=reload, force_transform=force_transform)

if os.path.exists(base_dir):
  shutil.rmtree(base_dir)
os.makedirs(base_dir)

train_dir = os.path.join(base_dir, "train_dataset")
valid_dir = os.path.join(base_dir, "valid_dataset")
test_dir = os.path.join(base_dir, "test_dataset")
model_dir = os.path.join(base_dir, "model")

print("About to perform train/valid/test split.")
splitter = RandomSplitter(verbosity=verbosity)
print("Performing new split.")
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
    nci_dataset, train_dir, valid_dir, test_dir)
train_dataset.set_verbosity(verbosity)
valid_dataset.set_verbosity(verbosity)
test_dataset.set_verbosity(verbosity)

# Fit Logistic Regression models
nci_task_types = {task: "regression" for task in nci_tasks}


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
                      model_instance=RandomForestRegressor(n_estimators=500),
                      verbosity=verbosity)
model = SingletaskToMultitask(nci_tasks, nci_task_types, params_dict, model_dir,
                              model_builder, verbosity=verbosity)

# Fit trained model
model.fit(train_dataset)
model.save()

train_evaluator = Evaluator(model, train_dataset, transformers, verbosity=verbosity)
train_scores = train_evaluator.compute_model_performance([classification_metric])

print("Train scores")
print(train_scores)

valid_evaluator = Evaluator(model, valid_dataset, transformers, verbosity=verbosity)
valid_scores = valid_evaluator.compute_model_performance([classification_metric])

print("Validation scores")
print(valid_scores)

"""
Script that trains Sklearn models on PDBbind dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import shutil
from sklearn.ensemble import RandomForestRegressor
from deepchem.datasets import Dataset
from deepchem.models.multitask import SingletaskToMultitask
from deepchem import metrics
from deepchem.metrics import Metric
from deepchem.models.sklearn_models import SklearnModel
from deepchem.utils.evaluate import Evaluator
from deepchem.datasets.pdbbind_datasets import load_pdbbind

np.random.seed(123)

# Set some global variables up top

reload = True
verbosity = "high"

pdbbind_dir = "/scratch/users/rbharath/deep-docking/datasets/pdbbind/"
base_data_dir = "/scratch/users/rbharath/pdbbind"

pdbbind_tasks, dataset, transformers = load_pdbbind(
    pdbbind_dir, base_data_dir, reload=reload)
print("len(dataset)")
print(len(dataset))

base_dir = "/scratch/users/rbharath/pdbbind_analysis"
if os.path.exists(base_dir):
  shutil.rmtree(base_dir)
if not os.path.exists(base_dir):
  os.makedirs(base_dir)
train_dir = os.path.join(base_dir, "train_dataset")
valid_dir = os.path.join(base_dir, "valid_dataset")
test_dir = os.path.join(base_dir, "test_dataset")
model_dir = os.path.join(base_dir, "model")

print("About to perform train/valid/test split.")
num_train = .8 * len(dataset)
X, y, w, ids = dataset.to_numpy()
print("dataset.to_numpy()")
print("X.shape, y.shape, w.shape, ids.shape")
print(X.shape, y.shape, w.shape, ids.shape)
print("Using following tasks")
print(pdbbind_tasks)
X_train, X_valid = X[:num_train], X[num_train:]
y_train, y_valid = y[:num_train], y[num_train:]
w_train, w_valid = w[:num_train], w[num_train:]
ids_train, ids_valid = ids[:num_train], ids[num_train:]

if os.path.exists(train_dir):
  shutil.rmtree(train_dir)
train_dataset = Dataset.from_numpy(train_dir, X_train, y_train,
                                   w_train, ids_train, pdbbind_tasks)

if os.path.exists(valid_dir):
  shutil.rmtree(valid_dir)
valid_dataset = Dataset.from_numpy(valid_dir, X_valid, y_valid,
                                   w_valid, ids_valid, pdbbind_tasks)

# Fit Logistic Regression models
pdbbind_task_types = {task: "regression" for task in pdbbind_tasks}


classification_metric = Metric(metrics.r2_score, verbosity=verbosity,
                               mode="regression")
params_dict = { 
    "batch_size": None,
    "data_shape": train_dataset.get_data_shape(),
}   

if os.path.exists(model_dir):
  shutil.rmtree(model_dir)
os.makedirs(model_dir)
model = SklearnModel(pdbbind_tasks, pdbbind_task_types, params_dict, model_dir,
                     model_instance=RandomForestRegressor(n_estimators=500),
                     verbosity=verbosity)

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

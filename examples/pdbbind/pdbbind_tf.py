"""
Script that trains Sklearn models on PDBbind dataset.
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

import os
import sys
import tempfile
import numpy as np
import numpy.random

import sys
import shutil
from deepchem.featurizers.featurize import DataLoader
from deepchem.hyperparameters import HyperparamOpt
from deepchem import metrics
from deepchem.metrics import Metric
from deepchem.models.tensorflow_models import TensorflowModel
from deepchem.models.tensorflow_models.fcnet import TensorflowMultiTaskRegressor
from deepchem.transformers import NormalizationTransformer
from deepchem.utils.evaluate import Evaluator
from deepchem.splits import RandomSplitter
from deepchem.featurizers.atomic_coordinates import AtomicCoordinates
from deepchem.datasets.pdbbind_datasets import load_core_pdbbind_grid
from deepchem.datasets import Dataset

verbosity = "high"
base_dir = "/scratch/users/rbharath/PDBBIND-ATOMICNET"
if os.path.exists(base_dir):
  shutil.rmtree(base_dir)
if not os.path.exists(base_dir):
  os.makedirs(base_dir)

feature_dir = os.path.join(base_dir, "feature")
train_dir = os.path.join(base_dir, "train")
valid_dir = os.path.join(base_dir, "valid")
test_dir = os.path.join(base_dir, "test")
model_dir = os.path.join(base_dir, "model")

# REPLACE WITH DOWNLOADED PDBBIND EXAMPLE
pdbbind_dir = "/scratch/users/rbharath/deep-docking/datasets/pdbbind"
pdbbind_tasks, dataset, transformers = load_core_pdbbind_grid(
    pdbbind_dir, base_dir)

print("len(dataset)")
print(len(dataset))

print("About to perform train/valid/test split.")
num_train = .8 * len(dataset)
X, y, w, ids = dataset.to_numpy()

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


classification_metric = Metric(metrics.pearson_r2_score, verbosity=verbosity,
                               mode="regression")

params_dict = { 
    "batch_size": 64,
    "nb_epoch": 20,
    "data_shape": train_dataset.get_data_shape(),
    "layer_sizes": [1000],
    "weight_init_stddevs": [.1],
    "bias_init_consts": [1.],
    "dropouts": [.25],
    "num_classification_tasks": len(pdbbind_tasks),
    "num_classes": 2,
    "penalty": .0,
    "optimizer": "momentum",
    "learning_rate": .0003,
    "momentum": .9,
}   

if os.path.exists(model_dir):
  shutil.rmtree(model_dir)
os.makedirs(model_dir)
model = TensorflowModel(pdbbind_tasks, pdbbind_task_types, params_dict, model_dir,
                        tf_class=TensorflowMultiTaskRegressor,
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

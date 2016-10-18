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
from pdbbind_datasets import load_core_pdbbind_grid
from deepchem.featurizers.featurize import DataLoader
from deepchem.hyperparameters import HyperparamOpt
from deepchem import metrics
from deepchem.metrics import Metric
from deepchem.models.tensorflow_models import TensorflowModel
from deepchem.models.tensorflow_models.fcnet import TensorflowMultiTaskRegressor
from deepchem.utils.evaluate import Evaluator
from deepchem.splits import RandomSplitter
from deepchem.featurizers.atomic_coordinates import AtomicCoordinates
from deepchem.datasets import DiskDataset

verbosity = "high"
base_dir = "/tmp/PDBBIND-ATOMICNET"
if os.path.exists(base_dir):
  shutil.rmtree(base_dir)
os.makedirs(base_dir)

feature_dir = os.path.join(base_dir, "feature")
train_dir = os.path.join(base_dir, "train")
valid_dir = os.path.join(base_dir, "valid")
test_dir = os.path.join(base_dir, "test")
model_dir = os.path.join(base_dir, "model")

# REPLACE WITH DOWNLOADED PDBBIND EXAMPLE
pdbbind_dir = "/tmp/deep-docking/datasets/pdbbind"
pdbbind_tasks, dataset, transformers = load_core_pdbbind_grid(
    pdbbind_dir, base_dir)

print("About to perform train/valid/test split.")
num_train = .8 * len(dataset)
X, y, w, ids = (dataset.X, dataset.y, dataset.w, dataset.ids)

X_train, X_valid = X[:num_train], X[num_train:]
y_train, y_valid = y[:num_train], y[num_train:]
w_train, w_valid = w[:num_train], w[num_train:]
ids_train, ids_valid = ids[:num_train], ids[num_train:]

train_dataset = DiskDataset.from_numpy(train_dir, X_train, y_train,
                                   w_train, ids_train, pdbbind_tasks)
valid_dataset = DiskDataset.from_numpy(valid_dir, X_valid, y_valid,
                                   w_valid, ids_valid, pdbbind_tasks)

classification_metric = Metric(metrics.pearson_r2_score, verbosity=verbosity,
                               mode="regression")

n_features = dataset.get_data_shape()[0]
tensorflow_model = TensorflowMultiTaskRegressor(
    len(pdbbind_tasks), n_features, model_dir, dropouts=[.25],
    learning_rate=0.0003, weight_init_stddevs=[.1],
    batch_size=64, verbosity=verbosity)
model = TensorflowModel(tensorflow_model, model_dir)

# Fit trained model
model.fit(train_dataset, nb_epoch=20)
model.save()

train_evaluator = Evaluator(model, train_dataset, transformers, verbosity=verbosity)
train_scores = train_evaluator.compute_model_performance([classification_metric])

print("Train scores")
print(train_scores)

valid_evaluator = Evaluator(model, valid_dataset, transformers, verbosity=verbosity)
valid_scores = valid_evaluator.compute_model_performance([classification_metric])

print("Validation scores")
print(valid_scores)

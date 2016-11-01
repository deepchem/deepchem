"""
Script that trains Keras multitask models on MUV dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import shutil
from muv_datasets import load_muv
from deepchem.datasets import Dataset
from deepchem import metrics
from deepchem.metrics import Metric
from deepchem.utils.evaluate import Evaluator
from deepchem.models.keras_models.fcnet import MultiTaskDNN
from deepchem.models.keras_models import KerasModel

# Set some global variables up top
np.random.seed(123)
reload = True
verbosity = "high"
<<<<<<< HEAD
model = "logistic"

base_data_dir = "/scratch/users/apappu/muv"

muv_tasks, dataset, transformers = load_muv(
    base_data_dir, reload=reload)
print("len(dataset)")
print(len(dataset))

base_dir = "/scratch/users/apappu/muv_analysis"
model_dir = os.path.join(base_dir, "model")
if os.path.exists(base_dir):
  shutil.rmtree(base_dir)
os.makedirs(base_dir)

# Load MUV data
muv_tasks, muv_datasets, transformers = load_muv(
    base_dir, reload=reload)
train_dataset, valid_dataset = muv_datasets 
n_features = 1024 


# Build model
classification_metric = Metric(metrics.roc_auc_score, np.mean,
                               verbosity=verbosity,
                               mode="classification")

keras_model = MultiTaskDNN(len(muv_tasks), n_features, "classification",
                           dropout=.25, learning_rate=.001, decay=1e-4)
model = KerasModel(keras_model, self.model_dir, verbosity=verbosity)

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

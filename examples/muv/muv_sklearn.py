"""
Script that trains Sklearn multitask models on MUV dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import shutil
from muv_datasets import load_muv
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from deepchem.data import Dataset
from deepchem.models.multitask import SingletaskToMultitask
from deepchem import metrics
from deepchem.metrics import Metric
from deepchem.models.sklearn_models import SklearnModel
from deepchem.utils.evaluate import Evaluator

np.random.seed(123)

# Set some global variables up top

reload = True
verbosity = "high"

base_dir = "/tmp/muv_sklearn"
model_dir = os.path.join(base_dir, "model")
if os.path.exists(base_dir):
  shutil.rmtree(base_dir)
os.makedirs(base_dir)

# Load MUV dataset
muv_tasks, muv_datasets, transformers = load_muv(
    base_dir, reload=reload)
(train_dataset, valid_dataset) = muv_datasets

# Fit models
classification_metric = Metric(metrics.roc_auc_score, np.mean,
                               verbosity=verbosity,
                               mode="classification")

def model_builder(model_dir):
  sklearn_model = RandomForestClassifier(
      class_weight="balanced", n_estimators=500)
  return SklearnModel(sklearn_model, model_dir)
model = SingletaskToMultitask(muv_tasks, model_builder, model_dir)

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

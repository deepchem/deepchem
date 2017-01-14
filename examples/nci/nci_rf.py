"""
Script that trains Sklearn multitask models on nci dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import shutil
from nci_datasets import load_nci
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from deepchem.data import Dataset
from deepchem.models.multitask import SingletaskToMultitask
from deepchem import metrics
from deepchem.metrics import Metric
from deepchem.models.sklearn_models import SklearnModel
from deepchem.utils.evaluate import Evaluator

np.random.seed(123)

# Set some global variables up top
verbosity = "high"

base_dir = "/tmp/nci_rf"
model_dir = os.path.join(base_dir, "model")
if os.path.exists(base_dir):
  shutil.rmtree(base_dir)
os.makedirs(base_dir)

nci_tasks, nci_dataset, transformers = load_nci(
    base_dir)

(train_dataset, valid_dataset, test_dataset) = nci_dataset

classification_metric = Metric(metrics.roc_auc_score, np.mean,
                               verbosity=verbosity,
                               mode="classification")
def model_builder(model_dir):
  sklearn_model = RandomForestRegressor(n_estimators=500)
  return SklearnModel(sklearn_model, model_dir)
model = SingletaskToMultitask(nci_tasks, model_builder, model_dir)

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

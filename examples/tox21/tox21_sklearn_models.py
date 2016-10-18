"""
Script that trains sklearn models on Tox21 dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import shutil
import numpy as np
from tox21_datasets import load_tox21
from sklearn.ensemble import RandomForestClassifier
from deepchem.utils.save import load_from_disk
from deepchem.datasets import Dataset
from deepchem.models.multitask import SingletaskToMultitask
from deepchem import metrics
from deepchem.metrics import Metric
from deepchem.models.sklearn_models import SklearnModel
from deepchem.utils.evaluate import Evaluator

# Only for debug!
np.random.seed(123)

# Set some global variables up top
verbosity = "high"

base_dir = "/tmp/tox21_sklearn"
#Make directories to store the raw and featurized datasets.
model_dir = os.path.join(base_dir, "model")
if os.path.exists(base_dir):
  shutil.rmtree(base_dir)
os.makedirs(base_dir)

# Load Tox21 dataset
tox21_tasks, tox21_datasets, transformers = load_tox21(base_dir, reload=False)
(train_dataset, valid_dataset) = tox21_datasets

# Fit models
classification_metric = Metric(metrics.roc_auc_score, np.mean,
                               verbosity=verbosity,
                               mode="classification")

def model_builder(model_dir):
  sklearn_model = RandomForestClassifier(
      class_weight="balanced", n_estimators=500)
  return SklearnModel(sklearn_model, model_dir)
model = SingletaskToMultitask(tox21_tasks, model_builder, model_dir)

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

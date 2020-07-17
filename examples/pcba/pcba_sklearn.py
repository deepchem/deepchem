"""
Script that trains Sklearn multitask models on PCBA dataset.
"""
import os
import numpy as np
import shutil
from deepchem.molnet import load_pcba
from sklearn.ensemble import RandomForestClassifier
from deepchem.models.multitask import SingletaskToMultitask
from deepchem import metrics
from deepchem.metrics import Metric
from deepchem.models.sklearn_models import SklearnModel
from deepchem.utils.evaluate import Evaluator

np.random.seed(123)

# Set some global variables up top
reload = True
is_verbose = False

base_dir = "/tmp/pcba_sklearn"
model_dir = os.path.join(base_dir, "model")
if os.path.exists(base_dir):
  shutil.rmtree(base_dir)
os.makedirs(base_dir)

pcba_tasks, pcba_datasets, transformers = load_pcba()
(train_dataset, valid_dataset, test_dataset) = pcba_datasets

classification_metric = Metric(
    metrics.roc_auc_score, np.mean, verbose=is_verbose, mode="classification")


def model_builder(model_dir):
  sklearn_model = RandomForestClassifier(
      class_weight="balanced", n_estimators=500)
  return SklearnModel(sklearn_model, model_dir)


model = SingletaskToMultitask(pcba_tasks, model_builder, model_dir)

# Fit trained model
model.fit(train_dataset)
model.save()

train_evaluator = Evaluator(
    model, train_dataset, transformers, verbose=is_verbose)
train_scores = train_evaluator.compute_model_performance(
    [classification_metric])

print("Train scores")
print(train_scores)

valid_evaluator = Evaluator(
    model, valid_dataset, transformers, verbose=is_verbose)
valid_scores = valid_evaluator.compute_model_performance(
    [classification_metric])

print("Validation scores")
print(valid_scores)

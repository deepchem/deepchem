"""
Script that trains multitask models on Tox21 dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import shutil
import numpy as np
from tox21_datasets import load_tox21
from deepchem.datasets import Dataset
from deepchem import metrics
from deepchem.metrics import Metric
from deepchem.utils.evaluate import Evaluator
from deepchem.models.tensorflow_models.fcnet import TensorflowMultiTaskClassifier
from deepchem.models.tensorflow_models import TensorflowModel

# Only for debug!
np.random.seed(123)

# Set some global variables up top
verbosity = "high"

#Make directories to store the raw and featurized datasets.
base_dir = "/tmp/tox21_tf"
data_dir = os.path.join(base_dir, "dataset")
model_dir = os.path.join(base_dir, "model")
# This is for good debug (to make sure nasty state isn't being passed around)
if os.path.exists(base_dir):
  shutil.rmtree(base_dir)
os.makedirs(base_dir)

# Load Tox21 dataset
n_features = 1024
tox21_tasks, tox21_datasets, transformers = load_tox21(data_dir, reload=False)
# Do train/valid split.
train_dataset, valid_dataset = tox21_datasets

# Fit models
classification_metric = Metric(metrics.roc_auc_score, np.mean,
                               verbosity=verbosity,
                               mode="classification")

tensorflow_model = TensorflowMultiTaskClassifier(
    len(tox21_tasks), n_features, model_dir, dropouts=[.25],
    learning_rate=0.0003, weight_init_stddevs=[1.],
    batch_size=32, verbosity=verbosity)
model = TensorflowModel(tensorflow_model, model_dir)

# Fit trained model
model.fit(train_dataset)
model.save()

train_evaluator = Evaluator(model, train_dataset, transformers,
                            verbosity=verbosity)
train_scores = train_evaluator.compute_model_performance(
    [classification_metric])

print("Train scores")
print(train_scores)

valid_evaluator = Evaluator(model, valid_dataset, transformers,
                            verbosity=verbosity)
valid_scores = valid_evaluator.compute_model_performance([classification_metric])

print("Validation scores")
print(valid_scores)

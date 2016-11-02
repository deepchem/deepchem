"""
Script that trains Tensorflow multitask models on PCBA dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import shutil
from pcba_datasets import load_pcba
from deepchem.utils.save import load_from_disk
from deepchem.data import Dataset
from deepchem import metrics
from deepchem.metrics import Metric
from deepchem.metrics import to_one_hot
from deepchem.utils.evaluate import Evaluator
from deepchem.models.tensorflow_models.fcnet import TensorflowMultiTaskClassifier
from deepchem.models.tensorflow_models import TensorflowModel

np.random.seed(123)

# Set some global variables up top

reload = True
verbosity = "high"

base_dir = "/tmp/pcba_tf"
model_dir = os.path.join(base_dir, "model")
if os.path.exists(base_dir):
  shutil.rmtree(base_dir)
os.makedirs(base_dir)

pcba_tasks, pcba_datasets, transformers = load_pcba(
    base_dir, reload=reload)
(train_dataset, valid_dataset) = pcba_datasets


classification_metric = Metric(metrics.roc_auc_score, np.mean,
                               verbosity=verbosity,
                               mode="classification")

tensorflow_model = TensorflowMultiTaskClassifier(
    len(pcba_tasks), n_features, model_dir, dropouts=[.25],
    learning_rate=0.001, weight_init_stddevs=[.1],
    batch_size=64, verbosity=verbosity)
model = TensorflowModel(tensorflow_model, model_dir)

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

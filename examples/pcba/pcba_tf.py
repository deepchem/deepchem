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

np.random.seed(123)

pcba_tasks, pcba_datasets, transformers = load_pcba()
(train_dataset, valid_dataset, test_dataset) = pcba_datasets

metric = Metric(metrics.roc_auc_score, np.mean, mode="classification")

n_features = train_dataset.get_data_shape()[0]
model_dir = None
model = TensorflowMultiTaskClassifier(
    len(pcba_tasks),
    n_features,
    model_dir,
    dropouts=[.25],
    learning_rate=0.001,
    weight_init_stddevs=[.1],
    batch_size=64,
    verbosity="high")

# Fit trained model
model.fit(train_dataset)
model.save()

train_evaluator = Evaluator(
    model, train_dataset, transformers, verbosity=verbosity)
train_scores = train_evaluator.compute_model_performance([metric])

print("Train scores")
print(train_scores)

valid_evaluator = Evaluator(
    model, valid_dataset, transformers, verbosity=verbosity)
valid_scores = valid_evaluator.compute_model_performance([metric])

print("Validation scores")
print(valid_scores)

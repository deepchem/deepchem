"""
Script that trains Tensorflow multitask models on PCBA dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import shutil
from deepchem.molnet import load_pcba
from deepchem.utils.save import load_from_disk
from deepchem.data import Dataset
from deepchem import metrics
from deepchem.metrics import Metric
from deepchem.metrics import to_one_hot
from deepchem.utils.evaluate import Evaluator
from deepchem.models import MultitaskClassifier
from deepchem.models.optimizers import ExponentialDecay

np.random.seed(123)

pcba_tasks, pcba_datasets, transformers = load_pcba()
(train_dataset, valid_dataset, test_dataset) = pcba_datasets

metric = Metric(metrics.roc_auc_score, np.mean, mode="classification")

n_features = train_dataset.get_data_shape()[0]
rate = ExponentialDecay(0.001, 0.8, 1000)
model = MultitaskClassifier(
    len(pcba_tasks),
    n_features,
    dropouts=[.25],
    learning_rate=rate,
    weight_init_stddevs=[.1],
    batch_size=64)

# Fit trained model
model.fit(train_dataset)

train_evaluator = Evaluator(model, train_dataset, transformers)
train_scores = train_evaluator.compute_model_performance([metric])

print("Train scores")
print(train_scores)

valid_evaluator = Evaluator(model, valid_dataset, transformers)
valid_scores = valid_evaluator.compute_model_performance([metric])

print("Validation scores")
print(valid_scores)

"""
Script that trains multitask models on Tox21 dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import deepchem as dc
from tox21_datasets import load_tox21
from deepchem.models.tensorgraph.tensor_graph import TensorGraph
from deepchem.models.tensorgraph.models.robust_multitask import tensorGraphMultitaskClassifier

# Only for debug!
from deepchem.models.tensorgraph.tensor_graph import TensorGraph

np.random.seed(123)

# Load Tox21 dataset
n_features = 1024
tox21_tasks, tox21_datasets, transformers = load_tox21()
train_dataset, valid_dataset, test_dataset = tox21_datasets

# Fit models
metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean,
                           mode="classification")

n_layers = 1
n_bypass_layers = 1
nb_epoch = 10
model_dir = "/tmp/multiclass"
model = tensorGraphMultitaskClassifier(
  len(tox21_tasks), train_dataset.get_data_shape()[0],
  layer_sizes=[500] * n_layers, bypass_layer_sizes=[50] * n_bypass_layers,
  model_dir=model_dir)
# Fit trained model
model.fit(train_dataset, nb_epoch=nb_epoch)
model.save()
print("saved")

#model = TensorGraph.load_from_dir(model_dir=model_dir)

print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)

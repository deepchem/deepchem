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
old = False
if old:
  model = dc.models.RobustMultitaskClassifier(
      len(tox21_tasks), train_dataset.get_data_shape()[0],
      layer_sizes=[500]*n_layers, bypass_layer_sizes=[50]*n_bypass_layers,
      dropouts=[.25]*n_layers, bypass_dropouts=[.25]*n_bypass_layers,
      weight_init_stddevs=[.02]*n_layers, bias_init_consts=[.5]*n_layers,
      bypass_weight_init_stddevs=[.02]*n_bypass_layers,
      bypass_bias_init_consts=[.5]*n_bypass_layers,
      learning_rate=.0003, penalty=.0001, penalty_type="l2",
      optimizer="adam", batch_size=100)
else:
    model = tensorGraphMultitaskClassifier(
    len(tox21_tasks), train_dataset.get_data_shape()[0],
    layer_sizes=[500] * n_layers, bypass_layer_sizes=[50] * n_bypass_layers,
    model_dir=model_dir)
# print(model.layers['LABEL0'].out_tensor.get_shape())
# print(model.layers['GUESS0'].out_tensor.get_shape())
# Fit trained model
# model.fit(train_dataset, nb_epoch=100, learning_rate=0.0003)
model.tensorboard = True
model.fit(train_dataset, nb_epoch=10)
model.save()

print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)

"""
Script that trains Tensorflow multitask models on QM9 dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import deepchem as dc
import numpy as np
from qm9_datasets import load_qm9

np.random.seed(123)
qm9_tasks, datasets, transformers = load_qm9()
train_dataset, valid_dataset, test_dataset = datasets
fit_transformers = [dc.trans.CoulombFitTransformer(train_dataset)]
regression_metric = [
    dc.metrics.Metric(dc.metrics.mean_absolute_error, mode="regression"),
    dc.metrics.Metric(dc.metrics.pearson_r2_score, mode="regression")
]
model = dc.models.TensorflowMultiTaskFitTransformRegressor(
    n_tasks=len(qm9_tasks),
    n_features=[29, 29],
    learning_rate=0.001,
    momentum=.8,
    batch_size=32,
    weight_init_stddevs=[1 / np.sqrt(400), 1 / np.sqrt(100), 1 / np.sqrt(100)],
    bias_init_consts=[0., 0., 0.],
    layer_sizes=[400, 100, 100],
    dropouts=[0.01, 0.01, 0.01],
    fit_transformers=fit_transformers,
    n_evals=10,
    seed=123)

# Fit trained model
model.fit(train_dataset, nb_epoch=50)
model.save()

train_scores = model.evaluate(train_dataset, regression_metric, transformers)
print("Train scores [kcal/mol]")
print(train_scores)

valid_scores = model.evaluate(valid_dataset, regression_metric, transformers)
print("Valid scores [kcal/mol]")
print(valid_scores)

test_scores = model.evaluate(test_dataset, regression_metric, transformers)
print("Test scores [kcal/mol]")
print(test_scores)

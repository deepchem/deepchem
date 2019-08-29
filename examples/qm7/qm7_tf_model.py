"""
Script that trains Tensorflow singletask models on QM7 dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import deepchem as dc
import numpy as np
from deepchem.molnet import load_qm7_from_mat
from deepchem.models.optimizers import ExponentialDecay

np.random.seed(123)
qm7_tasks, datasets, transformers = load_qm7_from_mat(
    split='stratified', move_mean=True)
train_dataset, valid_dataset, test_dataset = datasets
fit_transformers = [dc.trans.CoulombFitTransformer(train_dataset)]
metric = [
    dc.metrics.Metric(dc.metrics.mean_absolute_error, mode="regression"),
    dc.metrics.Metric(dc.metrics.pearson_r2_score, mode="regression")
]

rate = 0.001
model = dc.models.MultitaskFitTransformRegressor(
    n_tasks=1,
    n_features=[23, 23],
    learning_rate=rate,
    momentum=.8,
    batch_size=25,
    weight_init_stddevs=[1 / np.sqrt(400), 1 / np.sqrt(100), 1 / np.sqrt(100)],
    bias_init_consts=[0., 0., 0.],
    layer_sizes=[400, 100, 100],
    dropouts=[0.01, 0.01, 0.01],
    fit_transformers=fit_transformers,
    seed=123)

# Fit trained model
model.fit(train_dataset, nb_epoch=50)

train_scores = model.evaluate(train_dataset, metric, transformers)
valid_scores = model.evaluate(valid_dataset, metric, transformers)

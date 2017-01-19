"""
Script that trains Tensorflow singletask models on GDB7 dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import deepchem as dc
import numpy as np
from gdb7_datasets import load_gdb7

np.random.seed(123)

gdb7_tasks, datasets, transformers = load_gdb7(featurizer=dc.feat.CoulombMatrix(23))
train_dataset, valid_dataset, test_dataset = datasets
fit_transformers = [dc.trans.CoulombRandomizationFitTransformer(), dc.trans.NormalizationFitTransformer()]

regression_metric = dc.metrics.Metric(dc.metrics.mean_absolute_error, 
                                      mode="regression")
model = dc.models.TensorflowMultiTaskFitTransformRegressor(
    n_tasks=len(gdb7_tasks), n_features=23,
    learning_rate=.0002, momentum=.8, batch_size=512,
    weight_init_stddevs=[1/np.sqrt(2000),1/np.sqrt(800),1/np.sqrt(800),1/np.sqrt(1000)],
    bias_init_consts=[0.,0.,0.,0.], layer_sizes=[2000,800,800,1000], 
    dropouts=[0.1,0.1,0.1,0.1], fit_transformers=fit_transformers, n_random_samples=10, seed=123)

# Fit trained model
model.fit(train_dataset, nb_epoch=2)
model.save()

train_scores = model.evaluate(train_dataset, [regression_metric], transformers)
print("Train scores [kcal/mol]")
print(train_scores)

valid_scores = model.evaluate(valid_dataset, [regression_metric], transformers)
print("Validation scores [kcal/mol]")
print(valid_scores)

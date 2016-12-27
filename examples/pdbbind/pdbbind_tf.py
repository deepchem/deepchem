"""
Script that trains Sklearn models on PDBbind dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

import deepchem as dc
import numpy as np
from pdbbind_datasets import load_pdbbind_grid

# For stable runs 
np.random.seed(123)

pdbbind_tasks, pdbbind_datasets, transformers = load_pdbbind_grid()
train_dataset, valid_dataset, test_dataset = pdbbind_datasets 

metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)

n_features = train_dataset.X.shape[1]
model = dc.models.TensorflowMultiTaskRegressor(
    len(pdbbind_tasks), n_features, dropouts=[.25], learning_rate=0.0003,
    weight_init_stddevs=[.1], batch_size=64)

# Fit trained model
model.fit(train_dataset, nb_epoch=20)
model.save()

print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)

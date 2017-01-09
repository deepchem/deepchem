"""
Script that trains Tensorflow Progressive Multitask models on UV datasets.
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import tempfile
import shutil
import numpy as np
import deepchem as dc
from MERCK_datasets import load_uv

# Set numpy seed
np.random.seed(123)

###Load data###
shard_size = 2000
num_shards_per_batch = 4
print("About to load MERCK data.")
UV_tasks, datasets, transformers = load_uv(
    shard_size=shard_size, num_shards_per_batch=num_shards_per_batch)
train_dataset, valid_dataset, test_dataset = datasets

print("Number of compounds in train set")
print(len(train_dataset))
print("Number of compounds in validation set")
print(len(valid_dataset))
print("Number of compounds in test set")
print(len(test_dataset))

###Create model###
n_layers = 3
nb_epoch = 50
model = dc.models.ProgressiveMultitaskRegressor(
    len(UV_tasks), train_dataset.get_data_shape()[0],
    layer_sizes=[25]*n_layers, dropouts=[.25]*n_layers,
    alpha_init_stddevs=[.02]*n_layers, weight_init_stddevs=[.02]*n_layers,
    bias_init_consts=[1.]*n_layers, learning_rate=.0003,
    penalty=.0001, penalty_type="l2", optimizer="adam", batch_size=100,
    seed=123, verbosity="high")


#Use R2 classification metric
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, task_averager=np.mean)

print("Training model")
model.fit(train_dataset, nb_epoch=nb_epoch)
#model.old_fit(train_dataset, nb_epoch=nb_epoch)

train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)
#Only use for final evaluation
test_scores = model.evaluate(test_dataset, [metric], transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)

print("Test scores")
print(test_scores)

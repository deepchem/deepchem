"""
Script that trains graph-conv models on SAMPL(FreeSolv) dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.set_random_seed(123)
import deepchem as dc

# Load SAMPL(FreeSolv) dataset
SAMPL_tasks, SAMPL_datasets, transformers = dc.molnet.load_sampl(
    featurizer='GraphConv')
train_dataset, valid_dataset, test_dataset = SAMPL_datasets

# Define metric
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)

# Batch size of models
batch_size = 50
model = dc.models.GraphConvModel(len(SAMPL_tasks), mode='regression')

# Fit trained model
model.fit(train_dataset, nb_epoch=20)

print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)

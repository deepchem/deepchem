"""
Script that trains textCNN models on delaney dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.set_random_seed(123)
import deepchem as dc

# Load Delaney dataset
delaney_tasks, delaney_datasets, transformers = dc.molnet.load_delaney(
    featurizer='Raw', split='index')
train_dataset, valid_dataset, test_dataset = delaney_datasets

# Fit models
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)

char_dict, length = dc.models.TextCNNTensorGraph.build_char_dict(train_dataset)

# Batch size of models
batch_size = 64

model = dc.models.TextCNNTensorGraph(
    len(delaney_tasks),
    char_dict,
    seq_length=length,
    mode='regression',
    learning_rate=1e-3,
    batch_size=batch_size,
    use_queue=False)

# Fit trained model
model.fit(train_dataset, nb_epoch=100)

print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)

"""
Script that trains DTNN models on qm7 dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.set_random_seed(123)
import deepchem as dc
from deepchem.models.tensorgraph.optimizers import ExponentialDecay

# Load QM7 dataset
tasks, datasets, transformers = dc.molnet.load_qm7_from_mat()
train_dataset, valid_dataset, test_dataset = datasets

# Fit models
metric = [
    dc.metrics.Metric(dc.metrics.mean_absolute_error, mode="regression"),
    dc.metrics.Metric(dc.metrics.pearson_r2_score, mode="regression")
]

# Batch size of models
batch_size = 50
n_embedding = 30
n_distance = 51
distance_min = -1.
distance_max = 9.2
n_hidden = 15

rate = ExponentialDecay(0.0001, 0.97, 5000)
model = dc.models.DTNNModel(
    len(tasks),
    n_embedding=n_embedding,
    n_hidden=n_hidden,
    n_distance=n_distance,
    distance_min=distance_min,
    distance_max=distance_max,
    output_activation=False,
    batch_size=batch_size,
    learning_rate=rate,
    use_queue=False,
    mode="regression")
#model.restore()

# Fit trained model
model.fit(train_dataset, nb_epoch=3000)

train_scores = model.evaluate(train_dataset, metric, transformers)
print("Train scores [kcal/mol]")
print(train_scores)

valid_scores = model.evaluate(valid_dataset, metric, transformers)
print("Valid scores [kcal/mol]")
print(valid_scores)

test_scores = model.evaluate(test_dataset, metric, transformers)
print("Test scores [kcal/mol]")
print(test_scores)

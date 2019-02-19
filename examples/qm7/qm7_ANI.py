"""
Script that trains ANI models on qm7 dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.set_random_seed(123)
import deepchem as dc

tasks, datasets, transformers = dc.molnet.load_qm7_from_mat(
    featurizer='BPSymmetryFunctionInput', split='stratified', move_mean=False)

train, valid, test = datasets

# Batch size of models
max_atoms = 23
batch_size = 128
layer_structures = [64, 64, 32]
atom_number_cases = [1, 6, 7, 8, 16]

# Fit models
metric = [
    dc.metrics.Metric(dc.metrics.mean_absolute_error, mode="regression"),
    dc.metrics.Metric(dc.metrics.pearson_r2_score, mode="regression")
]

model = dc.models.ANIRegression(
    len(tasks),
    max_atoms,
    exp_loss=False,
    activation_fn='relu',
    layer_structures=layer_structures,
    atom_number_cases=atom_number_cases,
    batch_size=batch_size,
    learning_rate=0.001,
    use_queue=False,
    mode="regression")

model.fit(train, nb_epoch=50)

train_scores = model.evaluate(train, metric, transformers)
valid_scores = model.evaluate(valid, metric, transformers)
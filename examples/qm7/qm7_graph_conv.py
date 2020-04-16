"""
Script that trains GraphConv models on qm7 dataset.
"""
import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.random.set_seed(123)
import deepchem as dc

# Load Tox21 dataset
tasks, datasets, transformers = dc.molnet.load_qm7_from_mat(
    featurizer='GraphConv', move_mean=True)
train_dataset, valid_dataset, test_dataset = datasets

# Fit models
metric = [
    dc.metrics.Metric(dc.metrics.mean_absolute_error, mode="regression"),
    dc.metrics.Metric(dc.metrics.pearson_r2_score, mode="regression")
]

# Batch size of models
batch_size = 64

model = dc.models.GraphConvModel(
    len(tasks), batch_size=batch_size, learning_rate=0.001, mode="regression")

# Fit trained model
model.fit(train_dataset, nb_epoch=5)

print("Evaluating model")
train_scores = model.evaluate(train_dataset, metric, transformers)
valid_scores = model.evaluate(valid_dataset, metric, transformers)

"""
Script that trains graph-conv models on HOPV dataset.
"""
import numpy as np

from deepchem.models import GraphConvModel

np.random.seed(123)
import tensorflow as tf
tf.random.set_seed(123)
import deepchem as dc
from deepchem.molnet import load_hopv

# Load HOPV dataset
hopv_tasks, hopv_datasets, transformers = load_hopv(featurizer='GraphConv')
train_dataset, valid_dataset, test_dataset = hopv_datasets

# Fit models
metric = [
    dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean, mode="regression"),
    dc.metrics.Metric(
        dc.metrics.mean_absolute_error, np.mean, mode="regression")
]

# Number of features on conv-mols
n_feat = 75
# Batch size of models
batch_size = 50
model = GraphConvModel(
    len(hopv_tasks), batch_size=batch_size, mode='regression')

# Fit trained model
model.fit(train_dataset, nb_epoch=25)

print("Evaluating model")
train_scores = model.evaluate(train_dataset, metric, transformers)
valid_scores = model.evaluate(valid_dataset, metric, transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)

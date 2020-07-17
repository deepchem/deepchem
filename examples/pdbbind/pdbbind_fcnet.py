"""
Script that trains Tensorflow models on PDBbind dataset.
"""
import os
import numpy as np
import tensorflow as tf
# For stable runs
np.random.seed(123)
tf.random.set_seed(123)

import deepchem as dc
from deepchem.molnet import load_pdbbind_grid

split = "random"
subset = "full"
pdbbind_tasks, pdbbind_datasets, transformers = load_pdbbind_grid(
    split=split, subset=subset)
train_dataset, valid_dataset, test_dataset = pdbbind_datasets

metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)

current_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(current_dir, "%s_%s_DNN" % (split, subset))

n_features = train_dataset.X.shape[1]
model = dc.models.MultitaskRegressor(
    len(pdbbind_tasks),
    n_features,
    model_dir=model_dir,
    dropouts=[.25],
    learning_rate=0.0003,
    weight_init_stddevs=[.1],
    batch_size=64)

# Fit trained model
model.fit(train_dataset, nb_epoch=100)

print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)

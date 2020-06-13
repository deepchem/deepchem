"""
Script that trains Chemception models on delaney dataset.
"""
import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.random.set_seed(123)
import deepchem as dc

# Load Delaney dataset
delaney_tasks, delaney_datasets, transformers = dc.molnet.load_delaney(
    featurizer='smiles2img', split='index', img_spec="engd")
train_dataset, valid_dataset, test_dataset = delaney_datasets

# Get Metric
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)

model = dc.models.ChemCeption(
    img_spec="engd",
    n_tasks=len(delaney_tasks),
    model_dir="./model",
    mode="regression")

# Fit trained model
model.fit(train_dataset, nb_epoch=1)

print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)

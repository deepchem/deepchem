"""
Script that trains Sklearn RF models on PDBbind dataset.
"""
import os
import deepchem as dc
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from deepchem.molnet import load_pdbbind

# For stable runs
np.random.seed(123)

pdbbind_tasks, pdbbind_datasets, transformers = load_pdbbind(
    featurizer="grid", split="random", subset="core")
train_dataset, valid_dataset, test_dataset = pdbbind_datasets

metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)

sklearn_model = RandomForestRegressor(n_estimators=500)
model = dc.models.SklearnModel(sklearn_model)

# Fit trained model
print("Fitting model on train dataset")
model.fit(train_dataset)

print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)

"""
Script that trains Sklearn RF models on PDBbind dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

import os
import deepchem as dc
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from pdbbind_datasets import load_pdbbind_grid

# For stable runs 
np.random.seed(123)

split = "random"
subset = "full"
pdbbind_tasks, pdbbind_datasets, transformers = load_pdbbind_grid(
    split=split, subset=subset)
train_dataset, valid_dataset, test_dataset = pdbbind_datasets 

metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)

current_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(current_dir, "%s_%s_RF" % (split, subset))

sklearn_model = RandomForestRegressor(n_estimators=500)
model = dc.models.SklearnModel(sklearn_model, model_dir=model_dir)

# Fit trained model
print("Fitting model on train dataset")
model.fit(train_dataset)
model.save()

print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)

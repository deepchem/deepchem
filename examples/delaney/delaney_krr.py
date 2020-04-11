#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 15:04:19 2017

@author: zqwu
"""
from sklearn.kernel_ridge import KernelRidge

import numpy as np
import deepchem as dc
import tempfile

# Only for debug!
np.random.seed(123)

# Load Delaney dataset
n_features = 1024
delaney_tasks, delaney_datasets, transformers = dc.molnet.load_delaney()
train_dataset, valid_dataset, test_dataset = delaney_datasets

metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)


def model_builder(model_dir):
  sklearn_model = KernelRidge(kernel="rbf", alpha=1e-3, gamma=0.05)
  return dc.models.SklearnModel(sklearn_model, model_dir)


model_dir = tempfile.mkdtemp()
model = dc.models.SingletaskToMultitask(delaney_tasks, model_builder, model_dir)

model.fit(train_dataset)
model.save()

print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)

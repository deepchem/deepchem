"""
Script that trains Sklearn singletask models on GDB7 dataset.
"""
import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.random.set_seed(123)
import deepchem as dc
from sklearn.kernel_ridge import KernelRidge

tasks, datasets, transformers = dc.molnet.load_qm7(
    featurizer='CoulombMatrix', split='stratified', move_mean=False)

train, valid, test = datasets

regression_metric = dc.metrics.Metric(
    dc.metrics.mean_absolute_error, mode="regression")


def model_builder(model_dir):
  sklearn_model = KernelRidge(kernel="rbf", alpha=5e-4, gamma=0.008)
  return dc.models.SklearnModel(sklearn_model, model_dir)


model = dc.models.SingletaskToMultitask(tasks, model_builder)

# Fit trained model
model.fit(train)
model.save()

train_evaluator = dc.utils.evaluate.Evaluator(model, train, transformers)
train_scores = train_evaluator.compute_model_performance([regression_metric])

print("Train scores [kcal/mol]")
print(train_scores)

test_evaluator = dc.utils.evaluate.Evaluator(model, test, transformers)
test_scores = test_evaluator.compute_model_performance([regression_metric])

print("Validation scores [kcal/mol]")
print(test_scores)

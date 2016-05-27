"""
Script that trains Tensorflow multitask models on PCBA dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import shutil
from deepchem.utils.save import load_from_disk
from deepchem.datasets import Dataset
from deepchem import metrics
from deepchem.metrics import Metric
from deepchem.metrics import to_one_hot
from deepchem.utils.evaluate import Evaluator
from deepchem.datasets.pcba_datasets import load_pcba
from deepchem.models.tensorflow_models.fcnet import TensorflowMultiTaskClassifier
from deepchem.models.tensorflow_models import TensorflowModel


np.random.seed(123)

# Set some global variables up top

reload = True
verbosity = "high"

base_data_dir = "/scratch/users/rbharath/pcba"

pcba_tasks, featurized_samples, dataset, transformers = load_pcba(
    base_data_dir, reload=reload)
print("len(featurized_samples), len(dataset)")
print(len(featurized_samples), len(dataset))

base_dir = "/scratch/users/rbharath/pcba_analysis"
if os.path.exists(base_dir):
  shutil.rmtree(base_dir)
if not os.path.exists(base_dir):
  os.makedirs(base_dir)
train_dir = os.path.join(base_dir, "train_dataset")
valid_dir = os.path.join(base_dir, "valid_dataset")
test_dir = os.path.join(base_dir, "test_dataset")
model_dir = os.path.join(base_dir, "model")

print("About to perform train/valid/test split.")
num_train = .8 * len(featurized_samples)
X, y, w, ids = dataset.to_numpy()
num_tasks = 120
pcba_tasks = pcba_tasks[:num_tasks]
print("Using following tasks")
print(pcba_tasks)
X_train, X_valid = X[:num_train], X[num_train:]
y_train, y_valid = y[:num_train, :num_tasks], y[num_train:, :num_tasks]
w_train, w_valid = w[:num_train, :num_tasks], w[num_train:, :num_tasks]
ids_train, ids_valid = ids[:num_train], ids[num_train:]

if os.path.exists(train_dir):
  shutil.rmtree(train_dir)
train_dataset = Dataset.from_numpy(train_dir, X_train, y_train,
                                   w_train, ids_train, pcba_tasks)

if os.path.exists(valid_dir):
  shutil.rmtree(valid_dir)
valid_dataset = Dataset.from_numpy(valid_dir, X_valid, y_valid,
                                   w_valid, ids_valid, pcba_tasks)

# Fit Logistic Regression models
pcba_task_types = {task: "classification" for task in pcba_tasks}


classification_metric = Metric(metrics.roc_auc_score, np.mean,
                               verbosity=verbosity,
                               mode="classification")
params_dict = { 
    "batch_size": 64,
    "nb_epoch": 10,
    "data_shape": train_dataset.get_data_shape(),
    "layer_sizes": [1000],
    "weight_init_stddevs": [.1],
    "bias_init_consts": [1.],
    "dropouts": [.25],
    "num_classification_tasks": len(pcba_tasks),
    "num_classes": 2,
    "penalty": .0,
    "optimizer": "momentum",
    "learning_rate": .001,
    "momentum": .9,
}   


if os.path.exists(model_dir):
  shutil.rmtree(model_dir)
os.makedirs(model_dir)
model = TensorflowModel(pcba_tasks, pcba_task_types, params_dict, model_dir,
                        tf_class=TensorflowMultiTaskClassifier,
                        verbosity=verbosity)

# Fit trained model
model.fit(train_dataset)
model.save()

train_evaluator = Evaluator(model, train_dataset, transformers, verbosity=verbosity)
train_scores = train_evaluator.compute_model_performance([classification_metric])

print("Train scores")
print(train_scores)

valid_evaluator = Evaluator(model, valid_dataset, transformers, verbosity=verbosity)
valid_scores = valid_evaluator.compute_model_performance([classification_metric])

print("Validation scores")
print(valid_scores)

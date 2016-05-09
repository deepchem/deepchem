"""
Script that trains sklearn models on Tox21 dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import shutil
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from deepchem.utils.save import load_from_disk
from deepchem.datasets import Dataset
from deepchem.featurizers.featurize import DataFeaturizer
from deepchem.featurizers.fingerprints import CircularFingerprint
from deepchem.splits import ScaffoldSplitter
from deepchem.splits import RandomSplitter
from deepchem.datasets import Dataset
from deepchem.transformers import BalancingTransformer
from deepchem.hyperparameters import HyperparamOpt
from deepchem.models.multitask import SingletaskToMultitask
from deepchem import metrics
from deepchem.metrics import Metric
from deepchem.metrics import to_one_hot
from deepchem.models.sklearn_models import SklearnModel
from deepchem.utils.evaluate import relative_difference
from deepchem.utils.evaluate import Evaluator


# Only for debug!
np.random.seed(123)

# Set some global variables up top
reload = True
verbosity = "high"
model = "logistic"

base_dir = "/scratch/users/rbharath/tox21_analysis"
if not os.path.exists(base_dir):
  os.makedirs(base_dir)

current_dir = os.path.dirname(os.path.realpath(__file__))
#Make directories to store the raw and featurized datasets.
feature_dir = os.path.join(base_dir, "features")
samples_dir = os.path.join(base_dir, "samples")
full_dir = os.path.join(base_dir, "full_dataset")
train_dir = os.path.join(base_dir, "train_dataset")
valid_dir = os.path.join(base_dir, "valid_dataset")
test_dir = os.path.join(base_dir, "test_dataset")
model_dir = os.path.join(base_dir, "model")

# Load Tox21 dataset
print("About to load Tox21 dataset.")
dataset_file = os.path.join(
    current_dir, "../datasets/tox21.csv.gz")
dataset = load_from_disk(dataset_file)
print("Columns of dataset: %s" % str(dataset.columns.values))
print("Number of examples in dataset: %s" % str(dataset.shape[0]))

# Featurize tox21 dataset
print("About to featurize Tox21 dataset.")
featurizers = [CircularFingerprint(size=1024)]
all_tox21_tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
                   'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
# For debugging purposes
#tox21_tasks = tox21_tasks[0:1]
all_valid_scores = {}
for task in all_tox21_tasks:
  tox21_tasks = [task]

  print("Using following tasks")
  print(tox21_tasks)
  # This is for good debug (to make sure nasty state isn't being passed around)
  if os.path.exists(feature_dir):
    shutil.rmtree(feature_dir)
  featurizer = DataFeaturizer(tasks=tox21_tasks,
                              smiles_field="smiles",
                              compound_featurizers=featurizers,
                              verbosity=verbosity)
  featurized_samples = featurizer.featurize(
      dataset_file, feature_dir,
      samples_dir, shard_size=8192,
      reload=reload)

  # Generate datasets
  print("About to create datasets")
  print("tox21_tasks")
  print(tox21_tasks)

  # This is for good debug (to make sure nasty state isn't being passed around)
  if os.path.exists(full_dir):
    shutil.rmtree(full_dir)
  full_dataset = Dataset(data_dir=full_dir, samples=featurized_samples, 
                          featurizers=featurizers, tasks=tox21_tasks,
                          verbosity=verbosity, reload=reload)

  # Do train/valid split.
  num_train = 7200
  X, y, w, ids = full_dataset.to_numpy()
  X_train, X_valid = X[:num_train], X[num_train:]
  y_train, y_valid = y[:num_train], y[num_train:]
  w_train, w_valid = w[:num_train], w[num_train:]
  ids_train, ids_valid = ids[:num_train], ids[num_train:]

  # Not sure if we need to constantly delete these directories...
  if os.path.exists(train_dir):
    shutil.rmtree(train_dir)
  train_dataset = Dataset.from_numpy(train_dir, tox21_tasks, X_train, y_train,
                                     w_train, ids_train)

  if os.path.exists(valid_dir):
    shutil.rmtree(valid_dir)
  valid_dataset = Dataset.from_numpy(valid_dir, tox21_tasks, X_valid, y_valid,
                                     w_valid, ids_valid)

  # No data transformations for now
  transformers = []

  # Fit models
  tox21_task_types = {task: "Classification" for task in tox21_tasks}

  classification_metric = Metric(metrics.roc_auc_score, np.mean,
                                 verbosity=verbosity,
                                 mode="classification")
  params_dict = { 
      "batch_size": None,
      "data_shape": train_dataset.get_data_shape(),
  }   

  # This is for good debug (to make sure nasty state isn't being passed around)
  if os.path.exists(model_dir):
    shutil.rmtree(model_dir)
  def model_builder(tasks, task_types, model_params, model_dir, verbosity=None):
    return SklearnModel(tasks, task_types, model_params, model_dir,
                        #model_instance=LogisticRegression(class_weight="balanced"),
                        model_instance=RandomForestClassifier(
                            class_weight="balanced",
                            n_estimators=500),
                        verbosity=verbosity)
  model = SingletaskToMultitask(tox21_tasks, tox21_task_types, params_dict, model_dir,
                                model_builder, verbosity=verbosity)

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
  all_valid_scores[task] = valid_scores.values()[0]
print("all_valid_scores")
print(all_valid_scores)
print("np.mean(np.array(all_valid_scores.items()))")
print(np.mean(np.array(all_valid_scores.items())))

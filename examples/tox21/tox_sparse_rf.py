"""
<<<<<<< 220848a3c318019643f62a8f99876ae951287818
Script that trains Sklearn multitask models on tox dataset.
=======
Script that trains Sklearn multitask models on nci dataset.
>>>>>>> sparse split test -- rf on tox 21
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import shutil
from sklearn.linear_model import LogisticRegression
<<<<<<< 220848a3c318019643f62a8f99876ae951287818
from sklearn.ensemble import RandomForestClassifier
=======
from sklearn.ensemble import RandomForestRegressor
>>>>>>> sparse split test -- rf on tox 21
from deepchem.datasets import Dataset
from deepchem.models.multitask import SingletaskToMultitask
from deepchem import metrics
from deepchem.metrics import Metric
from deepchem.models.sklearn_models import SklearnModel
from deepchem.utils.evaluate import Evaluator
<<<<<<< 220848a3c318019643f62a8f99876ae951287818
from deepchem.datasets.tox21_datasets import load_tox21
=======
from deepchem.datasets.nci_datasets import load_nci
>>>>>>> sparse split test -- rf on tox 21
from deepchem.splits import StratifiedSplitter
from deepchem.splits import RandomSplitter

np.random.seed(123)

# Set some global variables up top

<<<<<<< 220848a3c318019643f62a8f99876ae951287818
reload = False
verbosity = "high"
force_transform = False 

base_data_dir = "./tox_data_dir"
base_dir = "./tox_analysis_dir"

tox_tasks, tox_dataset, transformers = load_tox21(
    base_data_dir, reload=reload)
=======
reload = True
verbosity = "high"
force_transform = False 

base_data_dir = "/scratch/users/rbharath/nci_data_dir"
base_dir = "/scratch/users/rbharath/nci_analysis_dir"

nci_tasks, nci_dataset, transformers = load_nci(
    base_data_dir, reload=reload, force_transform=force_transform)
>>>>>>> sparse split test -- rf on tox 21

if os.path.exists(base_dir):
  shutil.rmtree(base_dir)
os.makedirs(base_dir)

train_dir = os.path.join(base_dir, "train_dataset")
valid_dir = os.path.join(base_dir, "valid_dataset")
test_dir = os.path.join(base_dir, "test_dataset")
model_dir = os.path.join(base_dir, "model")

print("About to perform train/valid/test split for both stratified and random.")
splitters = [RandomSplitter(verbosity=verbosity), StratifiedSplitter(verbosity=verbosity)]
print("Performing splits")

<<<<<<< 220848a3c318019643f62a8f99876ae951287818
train_scores_list = []
valid_scores_list = []
for splitter in splitters:
  train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
      tox_dataset, train_dir, valid_dir, test_dir)
=======
train_scores = []
valid_scores = []
for splitter in splitters:
  train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
      nci_dataset, train_dir, valid_dir, test_dir)
>>>>>>> sparse split test -- rf on tox 21
  train_dataset.set_verbosity(verbosity)
  valid_dataset.set_verbosity(verbosity)
  test_dataset.set_verbosity(verbosity)

  # Fit Logistic Regression models
<<<<<<< 220848a3c318019643f62a8f99876ae951287818
  tox_task_types = {task: "classification" for task in tox_tasks}
=======
  nci_task_types = {task: "regression" for task in nci_tasks}

>>>>>>> sparse split test -- rf on tox 21

  classification_metric = Metric(metrics.roc_auc_score, np.mean,
                                 verbosity=verbosity,
                                 mode="classification")
  params_dict = {
      "batch_size": None,
      "data_shape": train_dataset.get_data_shape(),
  }

  if os.path.exists(model_dir):
    shutil.rmtree(model_dir)
  os.makedirs(model_dir)
  def model_builder(tasks, task_types, model_params, model_dir, verbosity=None):
    return SklearnModel(tasks, task_types, model_params, model_dir,
<<<<<<< 220848a3c318019643f62a8f99876ae951287818
                        model_instance=RandomForestClassifier(
                          class_weight="balanced",
                          n_estimators=500,
                          n_jobs = -1),
                        verbosity=verbosity)
  model = SingletaskToMultitask(tox_tasks, tox_task_types, params_dict, model_dir,
=======
                        model_instance=RandomForestRegressor(n_estimators=500),
                        verbosity=verbosity)
  model = SingletaskToMultitask(nci_tasks, nci_task_types, params_dict, model_dir,
>>>>>>> sparse split test -- rf on tox 21
                                model_builder, verbosity=verbosity)

  # Fit trained model
  model.fit(train_dataset)
  model.save()

  train_evaluator = Evaluator(model, train_dataset, transformers, verbosity=verbosity)
  train_scores = train_evaluator.compute_model_performance([classification_metric])


  valid_evaluator = Evaluator(model, valid_dataset, transformers, verbosity=verbosity)
  valid_scores = valid_evaluator.compute_model_performance([classification_metric])

<<<<<<< 220848a3c318019643f62a8f99876ae951287818
  train_scores_list.append(train_scores)
  valid_scores_list.append(valid_scores)
=======
  train_scores.append(train_scores)
  valid_scores.append(valid_scores)
>>>>>>> sparse split test -- rf on tox 21

for i in range(2):
  if i == 0:
    print("Random")
    print("---------------------")
  if i == 1:
    print("Stratified")
    print("---------------------")
  print("Train score")
<<<<<<< 220848a3c318019643f62a8f99876ae951287818
  print(train_scores_list[i])
  print("---------------------")
  print("Valid score")
  print(valid_scores_list[i])
=======
  print(train_scores[i])
  print("---------------------")
  print("Valid score")
  print(valid_scores[i])
>>>>>>> sparse split test -- rf on tox 21

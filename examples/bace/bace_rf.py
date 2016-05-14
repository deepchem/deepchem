import sys
import os
import deepchem
import tempfile, shutil
from deepchem.utils.save import load_from_disk
from deepchem.splits import SpecifiedSplitter
from deepchem.featurizers.featurize import DataFeaturizer
from deepchem.datasets import Dataset
from deepchem.transformers import NormalizationTransformer
from deepchem.transformers import ClippingTransformer
from deepchem.hyperparameters import HyperparamOpt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from deepchem.models.sklearn_models import SklearnModel
from bace_features import user_specified_features
from deepchem import metrics
from deepchem.metrics import Metric
from deepchem.utils.evaluate import Evaluator
from deepchem.datasets.bace_datasets import load_bace

#mode = "regression"
#mode = "classification"
#verbosity = "high"

def bace_rf_model(mode="classification", verbosity="high", split="20-80"):
  bace_tasks, train_dataset, valid_dataset, test_dataset, crystal_dataset, transformers = \
    load_bace(mode=mode, transform=False, split=split)

  if mode == "regression":
    metric = Metric(metrics.r2_score, verbosity=verbosity)
    model_class = RandomForestRegressor
  elif mode == "classification":
    metric = Metric(metrics.roc_auc_score, verbosity=verbosity)
    model_class = RandomForestClassifier
  else:
    raise ValueError("Invalid mode %s" % mode)

  def model_builder(tasks, task_types, params_dict, model_dir, verbosity=verbosity):
      n_estimators = params_dict["n_estimators"]
      max_features = params_dict["max_features"]
      return SklearnModel(
          tasks, task_types, params_dict, model_dir,
          model_instance=model_class(n_estimators=n_estimators,
                                     max_features=max_features))
  params_dict = {
      "n_estimators": [10, 100],
      "batch_size": [None],
      "data_shape": [train_dataset.get_data_shape()],
      "max_features": ["auto", "sqrt", "log2", None],
      }
  optimizer = HyperparamOpt(model_builder, bace_tasks,
                            {task: mode for task in bace_tasks})
  best_rf, best_rf_hyperparams, all_rf_results = optimizer.hyperparam_search(
      params_dict, train_dataset, valid_dataset, transformers,
      metric=metric)

  if len(train_dataset) > 0:
    rf_train_evaluator = Evaluator(best_rf, train_dataset, transformers)
    rf_train_score = rf_train_evaluator.compute_model_performance([metric])
    print("RF Train set %s: %s" % (metric.name, str(rf_train_score)))

  if len(valid_dataset) > 0:
    rf_valid_evaluator = Evaluator(best_rf, valid_dataset, transformers)
    rf_valid_score = rf_valid_evaluator.compute_model_performance([metric])
    print("RF Valid set %s: %s" % (metric.name, str(rf_valid_score)))

  if len(test_dataset) > 0:
    rf_test_evaluator = Evaluator(best_rf, test_dataset, transformers)
    rf_test_score = rf_test_evaluator.compute_model_performance([metric])
    print("RF Test set %s: %s" % (metric.name, str(rf_test_score)))

  if len(crystal_dataset) > 0:
    rf_crystal_evaluator = Evaluator(best_rf, crystal_dataset, transformers)
    rf_crystal_score = rf_crystal_evaluator.compute_model_performance(
        [metric])
    print("RF Crystal set %s: %s" % (metric.name, str(rf_crystal_score)))

if __name__ == "__main__":
  print("Classifier RF 20-80:")
  print("--------------------------------")
  bace_rf_model(mode="classification", verbosity="high", split="20-80")
  print("Classifier RF 80-20:")
  print("--------------------------------")
  bace_rf_model(mode="classification", verbosity="high", split="80-20")

  print("Regressor RF 20-80:")
  print("--------------------------------")
  bace_rf_model(mode="regression", verbosity="high", split="20-80")
  print("Regressor RF 80-20:")
  print("--------------------------------")
  bace_rf_model(mode="regression", verbosity="high", split="80-20")

import sys
import os
import deepchem
import tempfile, shutil
import numpy as np
import numpy.random
from deepchem.utils.save import load_from_disk
from deepchem.splits import SpecifiedSplitter
from deepchem.featurizers.featurize import DataFeaturizer
from deepchem.datasets import Dataset
from deepchem.transformers import NormalizationTransformer
from deepchem.transformers import ClippingTransformer
from deepchem.hyperparameters import HyperparamOpt
from sklearn.ensemble import RandomForestClassifier
from deepchem.models.sklearn_models import SklearnModel
from bace_features import user_specified_features
from deepchem import metrics
from deepchem.metrics import Metric
from deepchem.utils.evaluate import Evaluator
from deepchem.datasets.bace_datasets import load_bace
from deepchem.models.keras_models.fcnet import SingleTaskDNN

reload = True
verbosity = "high"
mode = "classification"
metric = Metric(metrics.roc_auc_score, verbosity=verbosity)
bace_tasks, train_dataset, valid_dataset, test_dataset, crystal_dataset, transformers \
  = load_bace(mode=mode)

def model_builder(tasks, task_types, params_dict, model_dir, verbosity=None):
    n_estimators = params_dict["n_estimators"]
    max_features = params_dict["max_features"]
    return SklearnModel(
        tasks, task_types, params_dict, model_dir, 
        model_instance=RandomForestClassifier(n_estimators=n_estimators,
                                              max_features=max_features))

params_dict = {"activation": ["relu"],
                "momentum": [.9],
                "batch_size": [50],
                "init": ["glorot_uniform"],
                "data_shape": [train_dataset.get_data_shape()],
                #"learning_rate": np.power(10., np.random.uniform(-5, -2, size=5)),
                "learning_rate": [1e-3],
                #"decay": np.power(10, np.random.uniform(-6, -4, size=5)),
                "decay": [1e-5],
                "nb_hidden": [1000],
                "nb_epoch": [40],
                "nesterov": [False],
                "dropout": [.5],
                "nb_layers": [1],
                "batchnorm": [False],
              }

optimizer = HyperparamOpt(SingleTaskDNN, bace_tasks,
                          {task: mode for task in bace_tasks})
                          verbosity=verbosity)
best_dnn, best_hyperparams, all_results = optimizer.hyperparam_search(
    params_dict, train_dataset, valid_dataset, transformers,
    metric=metric)

dnn_train_evaluator = Evaluator(best_dnn, train_dataset, transformers)            
dnn_train_roc_auc_score = dnn_train_evaluator.compute_model_performance(
    [metric])
print("DNN Train set ROC-AUC %s" % str(dnn_train_roc_auc_score))

dnn_valid_evaluator = Evaluator(best_dnn, valid_dataset, transformers)            
dnn_valid_roc_auc_score = dnn_valid_evaluator.compute_model_performance(
    [metric])
print("DNN Valid set ROC-AUC %s" % str(dnn_valid_roc_auc_score))
                                                                                             
dnn_test_evaluator = Evaluator(best_dnn, test_dataset, transformers)
dnn_test_roc_auc_score = dnn_test_evaluator.compute_model_performance(
    [metric])
print("DNN Test set ROC-AUC %s" % str(dnn_test_roc_auc_score))

dnn_crystal_evaluator = Evaluator(best_dnn, crystal_dataset, transformers)
dnn_crystal_roc_auc_score = dnn_crystal_evaluator.compute_model_performance(
    [metric])
print("DNN Crystal set ROC-AUC %s" % str(dnn_crystal_roc_auc_score))

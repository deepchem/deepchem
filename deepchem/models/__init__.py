"""
Gathers all models in one place for convenient imports
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

from deepchem.models.models import Model
from deepchem.models.sklearn_models import SklearnModel
from deepchem.models.tf_keras_models.multitask_classifier import MultitaskGraphClassifier
from deepchem.models.tf_keras_models.support_classifier import SupportGraphClassifier
from deepchem.models.multitask import SingletaskToMultitask

from deepchem.models.tensorflow_models.fcnet import TensorflowMultiTaskRegressor
from deepchem.models.tensorflow_models.fcnet import TensorflowMultiTaskClassifier
from deepchem.models.tensorflow_models.robust_multitask import RobustMultitaskRegressor
from deepchem.models.tensorflow_models.robust_multitask import RobustMultitaskClassifier
from deepchem.models.tensorflow_models.lr import TensorflowLogisticRegression
from deepchem.models.tensorflow_models.progressive_multitask import ProgressiveMultitaskRegressor

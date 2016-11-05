"""
Gathers all models in one place for convenient imports
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

from deepchem.models.models import Model
from deepchem.models.sklearn_models import SklearnModel
from deepchem.models.keras_models import KerasModel
from deepchem.models.tf_keras_models.multitask_classifier import MultitaskGraphClassifier
from deepchem.models.tf_keras_models.support_classifier import SupportGraphClassifier
from deepchem.models.tensorflow_models import TensorflowModel
from deepchem.models.multitask import SingletaskToMultitask

# TODO(rbharath): I'm not sure if these belong here or in deepchem.nn
# The issue is that these are not valid deepchem models. The solution might be
# to make inherit from Model class
from deepchem.models.keras_models.fcnet import MultiTaskDNN
from deepchem.models.tensorflow_models.fcnet import TensorflowMultiTaskRegressor
from deepchem.models.tensorflow_models.fcnet import TensorflowMultiTaskClassifier
from deepchem.models.tensorflow_models.robust_multitask import RobustMultitaskRegressor

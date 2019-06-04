"""
Gathers all models in one place for convenient imports
"""
from __future__ import division
from __future__ import unicode_literals

from deepchem.models.models import Model
from deepchem.models.keras_model import KerasModel
from deepchem.models.sklearn_models import SklearnModel
from deepchem.models.xgboost_models import XGBoostModel
from deepchem.models.multitask import SingletaskToMultitask

from deepchem.models.tensorgraph.tensor_graph import TensorGraph
from deepchem.models.tensorgraph.fcnet import MultitaskRegressor
from deepchem.models.tensorgraph.fcnet import MultitaskClassifier
from deepchem.models.tensorgraph.fcnet import MultitaskFitTransformRegressor
from deepchem.models.tensorgraph.IRV import TensorflowMultitaskIRVClassifier
from deepchem.models.tensorgraph.robust_multitask import RobustMultitaskClassifier
from deepchem.models.tensorgraph.robust_multitask import RobustMultitaskRegressor
from deepchem.models.tensorgraph.progressive_multitask import ProgressiveMultitaskRegressor, ProgressiveMultitaskClassifier
from deepchem.models.tensorgraph.models.graph_models import WeaveModel, DTNNModel, DAGModel, GraphConvModel, MPNNModel
from deepchem.models.tensorgraph.models.symmetry_function_regression import BPSymmetryFunctionRegression, ANIRegression
from deepchem.models.tensorgraph.models.scscore import ScScoreModel

from deepchem.models.tensorgraph.models.seqtoseq import SeqToSeq
from deepchem.models.tensorgraph.models.gan import GAN, WGAN
from deepchem.models.tensorgraph.models.text_cnn import TextCNNModel
from deepchem.models.tensorgraph.sequential import Sequential
from deepchem.models.tensorgraph.models.sequence_dnn import SequenceDNN
from deepchem.models.tensorgraph.models.ontology import OntologyModel, OntologyNode, create_gene_ontology
from deepchem.models.tensorgraph.models.atomic_conv import AtomicConvModel

#################### Compatibility imports for renamed TensorGraph models. Remove below with DeepChem 3.0. ####################

from deepchem.models.tensorgraph.models.text_cnn import TextCNNTensorGraph
from deepchem.models.tensorgraph.models.graph_models import WeaveTensorGraph, DTNNTensorGraph, DAGTensorGraph, GraphConvTensorGraph, MPNNTensorGraph

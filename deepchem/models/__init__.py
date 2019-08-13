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
from deepchem.models.callbacks import ValidationCallback

from deepchem.models.tensorgraph.tensor_graph import TensorGraph
from deepchem.models.fcnet import MultitaskRegressor
from deepchem.models.fcnet import MultitaskClassifier
from deepchem.models.fcnet import MultitaskFitTransformRegressor
from deepchem.models.IRV import TensorflowMultitaskIRVClassifier
from deepchem.models.robust_multitask import RobustMultitaskClassifier
from deepchem.models.robust_multitask import RobustMultitaskRegressor
from deepchem.models.progressive_multitask import ProgressiveMultitaskRegressor, ProgressiveMultitaskClassifier
from deepchem.models.graph_models import WeaveModel, DTNNModel, DAGModel, GraphConvModel, MPNNModel
from deepchem.models.tensorgraph.models.symmetry_function_regression import BPSymmetryFunctionRegression, ANIRegression
from deepchem.models.scscore import ScScoreModel

from deepchem.models.seqtoseq import SeqToSeq
from deepchem.models.gan import GAN, WGAN
from deepchem.models.text_cnn import TextCNNModel
from deepchem.models.tensorgraph.sequential import Sequential
from deepchem.models.tensorgraph.models.sequence_dnn import SequenceDNN
from deepchem.models.tensorgraph.models.ontology import OntologyModel, OntologyNode, create_gene_ontology
from deepchem.models.atomic_conv import AtomicConvModel
from deepchem.models.chemnet_models import Smiles2Vec, ChemCeption

#################### Compatibility imports for renamed TensorGraph models. Remove below with DeepChem 3.0. ####################

from deepchem.models.text_cnn import TextCNNTensorGraph
from deepchem.models.graph_models import WeaveTensorGraph, DTNNTensorGraph, DAGTensorGraph, GraphConvTensorGraph, MPNNTensorGraph

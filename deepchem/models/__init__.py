"""
Gathers all models in one place for convenient imports
"""
# flake8: noqa
from deepchem.models.models import Model
from deepchem.models.multitask import SingletaskToMultitask
from deepchem.models.wandblogger import WandbLogger
from deepchem.models.callbacks import ValidationCallback

# Tensorflow Depedency Models
try:
  from deepchem.models.keras_model import KerasModel

  from deepchem.models.IRV import MultitaskIRVClassifier
  from deepchem.models.robust_multitask import RobustMultitaskClassifier
  from deepchem.models.robust_multitask import RobustMultitaskRegressor
  from deepchem.models.progressive_multitask import ProgressiveMultitaskRegressor, ProgressiveMultitaskClassifier
  from deepchem.models.graph_models import WeaveModel, DTNNModel, DAGModel, GraphConvModel, MPNNModel
  from deepchem.models.scscore import ScScoreModel

  from deepchem.models.seqtoseq import SeqToSeq
  from deepchem.models.gan import GAN, WGAN
  from deepchem.models.molgan import BasicMolGANModel
  from deepchem.models.cnn import CNN
  from deepchem.models.text_cnn import TextCNNModel
  from deepchem.models.atomic_conv import AtomicConvModel
  from deepchem.models.chemnet_models import Smiles2Vec, ChemCeption
except ModuleNotFoundError:
  pass

# scikit-learn model
from deepchem.models.sklearn_models import SklearnModel
from deepchem.models.gbdt_models import GBDTModel

# PyTorch models
try:
  from deepchem.models.torch_models import TorchModel
  from deepchem.models.torch_models import AttentiveFP, AttentiveFPModel
  from deepchem.models.torch_models import CGCNN, CGCNNModel
  from deepchem.models.torch_models import GAT, GATModel
  from deepchem.models.torch_models import GCN, GCNModel
  from deepchem.models.torch_models import LCNN, LCNNModel
  from deepchem.models.torch_models import Pagtn, PagtnModel
  from deepchem.models.fcnet import MultitaskRegressor, MultitaskClassifier, MultitaskFitTransformRegressor
except ModuleNotFoundError:
  pass

# Jax models
try:
  from deepchem.models.jax_models import JaxModel
  from deepchem.models.jax_models import PINNModel
except ModuleNotFoundError:
  pass

#####################################################################################
# Compatibility imports for renamed XGBoost models. Remove below with DeepChem 3.0.
#####################################################################################

from deepchem.models.gbdt_models.gbdt_model import XGBoostModel

########################################################################################
# Compatibility imports for renamed TensorGraph models. Remove below with DeepChem 3.0.
########################################################################################
try:
  from deepchem.models.text_cnn import TextCNNTensorGraph
  from deepchem.models.graph_models import WeaveTensorGraph, DTNNTensorGraph, DAGTensorGraph, GraphConvTensorGraph, MPNNTensorGraph
  from deepchem.models.IRV import TensorflowMultitaskIRVClassifier
except ModuleNotFoundError:
  pass

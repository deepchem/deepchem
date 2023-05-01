# flake8:noqa
import logging

logger = logging.getLogger(__name__)

from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.torch_models.modular import ModularTorchModel
from deepchem.models.torch_models.attentivefp import AttentiveFP, AttentiveFPModel
from deepchem.models.torch_models.cgcnn import CGCNN, CGCNNModel
from deepchem.models.torch_models.gat import GAT, GATModel
from deepchem.models.torch_models.gcn import GCN, GCNModel
from deepchem.models.torch_models.infograph import InfoGraphStar, InfoGraphStarModel, InfoGraphEncoder, GINEncoder, InfoGraph, InfoGraphModel, InfoGraphEncoder
from deepchem.models.torch_models.mpnn import MPNN, MPNNModel
from deepchem.models.torch_models.lcnn import LCNN, LCNNModel
from deepchem.models.torch_models.pagtn import Pagtn, PagtnModel
from deepchem.models.torch_models.mat import MAT, MATModel
from deepchem.models.torch_models.megnet import MEGNetModel
from deepchem.models.torch_models.normalizing_flows_pytorch import NormalizingFlow
from deepchem.models.torch_models.layers import MultilayerPerceptron, CNNModule, CombineMeanStd, WeightedLinearCombo, AtomicConvolution, NeighborList
from deepchem.models.torch_models.cnn import CNN
from deepchem.models.torch_models.attention import ScaledDotProductAttention, SelfAttention
from deepchem.models.torch_models.grover import GroverModel, GroverPretrain, GroverFinetune
from deepchem.models.torch_models.readout import GroverReadout
try:
    from deepchem.models.torch_models.dmpnn import DMPNN, DMPNNModel
    from deepchem.models.torch_models.gnn import GNN, GNNHead, GNNModular
except ModuleNotFoundError as e:
    logger.warning(
        f'Skipped loading modules with pytorch-geometric dependency, missing a dependency. {e}'
    )
try:
    from deepchem.models.torch_models.hf_models import HuggingFaceModel
except ModuleNotFoundError as e:
    logger.warning(f'Skipped loading modules with transformers dependency. {e}')

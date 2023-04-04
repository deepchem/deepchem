import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRU, Linear, ReLU, Sequential
from typing import Iterable, List, Tuple
from deepchem.metrics import to_one_hot

import deepchem as dc
from deepchem.models.losses import SoftmaxCrossEntropy
from deepchem.feat.graph_data import BatchGraphData
from deepchem.models.losses import (
    GlobalMutualInformationLoss,
    LocalMutualInformationLoss,
)
from deepchem.models.torch_models.layers import MultilayerPerceptron
from deepchem.models.torch_models.modular import ModularTorchModel

try:
    from torch_geometric.nn import GINConv, NNConv, global_add_pool
    from torch_geometric.nn.aggr import Set2Set
except ImportError:
    pass


class GINEncoder(torch.nn.Module):
    """
    Graph Information Network (GIN) encoder. This is a graph convolutional network that produces encoded representations for molecular graph inputs.

    Parameters
    ----------
    num_features: int
        The number of node features
    embedding_dim: int
        The dimension of the output embedding
    num_gc_layers: int, optional (default 5)
        The number of graph convolutional layers to use

    Example
    -------
    >>> import numpy as np
    >>> from deepchem.models.torch_models.infograph import GINEncoder
    >>> from deepchem.feat.graph_data import GraphData
    >>> encoder = GINEncoder(num_features=25, embedding_dim=32)
    >>> node_features = np.random.randn(10, 25)
    >>> edge_index = np.array([[0, 1, 2], [1, 2, 3]])
    >>> edge_features = np.random.randn(3, 10)
    >>> graph_index = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> data = GraphData(node_features=node_features, edge_index=edge_index, edge_features=edge_features, graph_index=graph_index).numpy_to_torch()
    >>> embedding, intermediate_embeddings = encoder(data)
    >>> print(embedding.shape)
    torch.Size([1, 30])

    References
    ----------
    .. [1] Xu, K., Hu, W., Leskovec, J. & Jegelka, S. How Powerful are Graph Neural Networks? arXiv:1810.00826 [cs, stat] (2019).

    """

    def __init__(self,
                 num_features: int,
                 embedding_dim: int,
                 num_gc_layers: int = 5):
        dim = int(
            embedding_dim / num_gc_layers
        )  # the output dimension of this encoder is modified by the number of GC layers, so this is necessary to ensure that the output dimension is consistent with the InfoGraphEncoder
        super().__init__()

        self.num_gc_layers = num_gc_layers

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_gc_layers):
            if i == 0:
                nn = Sequential(Linear(num_features, dim), ReLU(),
                                Linear(dim, dim))
            else:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            conv = GINConv(nn)
            bn = torch.nn.BatchNorm1d(dim)

            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, data):
        """
        Encodes the input graph data.

        Parameters
        ----------
        data : BatchGraphData
            The batched input graph data.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the encoded representation and intermediate embeddings.
        """
        xs = []
        x = data.node_features
        for i in range(self.num_gc_layers):

            x = F.relu(self.convs[i](x, data.edge_index))
            x = self.bns[i](x)
            xs.append(x)

        xpool = [global_add_pool(x, data.graph_index) for x in xs]
        x = torch.cat(xpool, 1)
        xs = torch.cat(xs, 1)
        return x, xs


class InfoGraphEncoder(torch.nn.Module):
    """
    The encoder for the InfoGraph model. It is a message passing graph convolutional
    network that produces encoded representations for molecular graph inputs.

    Parameters
    ----------
    num_features: int
        Number of node features for each input
    edge_features: int
        Number of edge features for each input
    embedding_dim: int
        Dimension of the embedding

    Example
    -------
    >>> import numpy as np
    >>> from deepchem.models.torch_models.infograph import InfoGraphEncoder
    >>> from deepchem.feat.graph_data import GraphData
    >>> encoder = InfoGraphEncoder(num_features=25, edge_features=10, embedding_dim=32)
    >>> node_features = np.random.randn(10, 25)
    >>> edge_index = np.array([[0, 1, 2], [1, 2, 3]])
    >>> edge_features = np.random.randn(3, 10)
    >>> graph_index = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> data = GraphData(node_features=node_features, edge_index=edge_index, edge_features=edge_features, graph_index=graph_index).numpy_to_torch()
    >>> embedding, feature_map = encoder(data)
    >>> print(embedding.shape)
    torch.Size([1, 64])
    """

    def __init__(self, num_features, edge_features, embedding_dim):
        super().__init__()
        self.lin0 = torch.nn.Linear(num_features, embedding_dim)

        nn = Sequential(Linear(edge_features, 128), ReLU(),
                        Linear(128, embedding_dim * embedding_dim))
        self.conv = NNConv(embedding_dim,
                           embedding_dim,
                           nn,
                           aggr='mean',
                           root_weight=False)
        self.gru = GRU(embedding_dim, embedding_dim)

        self.set2set = Set2Set(embedding_dim, processing_steps=3)

    def forward(self, data):
        """
        Encode input graphs into an embedding and feature map.

        Parameters
        ----------
        data: Union[BatchGraphData, GraphData]
            Contains information about graphs.

        Returns
        -------
        torch.Tensor
            Encoded tensor of input data.
        torch.Tensor
            Feature map tensor of input data.

        """
        out = F.relu(self.lin0(data.node_features))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_features))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
            feat_map = out

        # set2set doubles the dimensionality of the embedding
        out = self.set2set(out, data.graph_index)
        return out, feat_map


class InfoGraph(nn.Module):
    """
    The nn.Module for InfoGraph. This class defines the forward pass of InfoGraph.

    References
    ----------
    1. Sun, F.-Y., Hoffmann, J., Verma, V. & Tang, J. InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Information Maximization. Preprint at http://arxiv.org/abs/1908.01000 (2020).

    Example
    -------
    >>> from deepchem.models.torch_models.infograph import InfoGraphModel
    >>> from deepchem.feat.molecule_featurizers import MolGraphConvFeaturizer
    >>> from deepchem.feat.graph_data import BatchGraphData
    >>> num_feat = 30
    >>> num_edge = 11
    >>> infographmodular = InfoGraphModel(num_feat, num_edge, 64)
    >>> smiles = ['C1=CC=CC=C1', 'C1=CC=CC=C1C2=CC=CC=C2']
    >>> featurizer = MolGraphConvFeaturizer(use_edges=True)
    >>> graphs = BatchGraphData(featurizer.featurize(smiles))
    >>> graphs = graphs.numpy_to_torch(infographmodular.device)
    >>> model = infographmodular.model
    >>> global_enc, local_enc = model(graphs)

    """

    def __init__(self, encoder, local_d, global_d, prior_d, init_emb=False):
        super().__init__()
        self.encoder = encoder
        self.local_d = local_d
        self.global_d = global_d
        self.prior_d = prior_d
        if init_emb:
            self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data):
        y, M = self.encoder(data)
        g_enc = self.global_d(y)
        l_enc = self.local_d(M)
        return g_enc, l_enc


class InfoGraphModel(ModularTorchModel):
    """
    InfoGraph is a graph convolutional model for unsupervised graph-level representation learning. The model aims to maximize the mutual information between the representations of entire graphs and the representations of substructures of different granularity.

    The unsupervised training of InfoGraph involves two encoders: one that encodes the entire graph and another that encodes substructures of different sizes. The mutual information between the two encoder outputs is maximized using a contrastive loss function.
    The model randomly samples pairs of graphs and substructures, and then maximizes their mutual information by minimizing their distance in a learned embedding space.

    This can be used for downstream tasks such as graph classification and molecular property prediction.It is implemented as a ModularTorchModel in order to facilitate transfer learning.

    References
    ----------
    1. Sun, F.-Y., Hoffmann, J., Verma, V. & Tang, J. InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Information Maximization. Preprint at http://arxiv.org/abs/1908.01000 (2020).

    Parameters
    ----------
    num_features: int
        Number of node features for each input
    edge_features: int
        Number of edge features for each input
    embedding_dim: int
        Dimension of the embedding
    num_gc_layers: int
        Number of graph convolutional layers
    prior: bool
        Whether to use a prior expectation in the loss term
    gamma: float
        Weight of the prior expectation in the loss term
    measure: str
        The divergence measure to use for the unsupervised loss. Options are 'GAN', 'JSD',
        'KL', 'RKL', 'X2', 'DV', 'H2', or 'W1'.
    average_loss: bool
        Whether to average the loss over the batch

    Example
    -------
    >>> from deepchem.models.torch_models.infograph import InfoGraphModel
    >>> from deepchem.feat import MolGraphConvFeaturizer
    >>> from deepchem.data import NumpyDataset
    >>> import torch
    >>> smiles = ["C1CCC1", "C1=CC=CN=C1"]
    >>> featurizer = MolGraphConvFeaturizer(use_edges=True)
    >>> X = featurizer.featurize(smiles)
    >>> y = torch.randint(0, 2, size=(2, 1)).float()
    >>> w = torch.ones(size=(2, 1)).float()
    >>> ds = NumpyDataset(X, y, w)
    >>> num_feat = max([ds.X[i].num_node_features for i in range(len(ds))])
    >>> edge_dim = max([ds.X[i].num_edge_features for i in range(len(ds))])
    >>> model = InfoGraphModel(num_feat, edge_dim, 15)
    >>> loss = model.fit(ds, nb_epoch=1)
    """

    def __init__(self,
                 num_features,
                 embedding_dim,
                 num_gc_layers=5,
                 prior=True,
                 gamma=.1,
                 measure='JSD',
                 average_loss=True,
                 **kwargs):
        self.num_features = num_features
        self.embedding_dim = embedding_dim * num_gc_layers
        self.num_gc_layers = num_gc_layers
        self.gamma = gamma
        self.prior = prior
        self.measure = measure
        self.average_loss = average_loss
        self.localloss = LocalMutualInformationLoss()._create_pytorch_loss(
            measure, average_loss)
        self.components = self.build_components()
        self.model = self.build_model()
        super().__init__(self.model, self.components, **kwargs)

    def build_components(self) -> dict:
        """
        Build the components of the model. InfoGraph is an unsupervised molecular graph representation learning model. It consists of an encoder, a local discriminator, a global discriminator, and a prior discriminator.

        The unsupervised loss is calculated by the mutual information in embedding representations at all layers.

        Components list, type and description:
        --------------------------------------
        encoder: GINEncoder, graph convolutional encoder

        local_d: MultilayerPerceptron, local discriminator

        global_d: MultilayerPerceptron, global discriminator

        prior_d: MultilayerPerceptron, prior discriminator
        """
        return {
            'encoder':
                GINEncoder(self.num_features, self.embedding_dim,
                           self.num_gc_layers),
            'local_d':
                MultilayerPerceptron(self.embedding_dim,
                                     self.embedding_dim, (self.embedding_dim,),
                                     skip_connection=True),
            'global_d':
                MultilayerPerceptron(self.embedding_dim,
                                     self.embedding_dim, (self.embedding_dim,),
                                     skip_connection=True),
            'prior_d':
                MultilayerPerceptron(self.embedding_dim,
                                     1, (self.embedding_dim,),
                                     activation_fn='sigmoid')
        }

    def build_model(self) -> nn.Module:
        return InfoGraph(**self.components)

    def loss_func(self, inputs, labels, weights):
        y, M = self.components['encoder'](inputs)
        g_enc = self.components['global_d'](y)
        l_enc = self.components['local_d'](M)
        local_global_loss = self.localloss(l_enc, g_enc, inputs.graph_index)
        if self.prior:
            prior = torch.rand_like(y)
            term_a = torch.log(self.components['prior_d'](prior)).mean()
            term_b = torch.log(1.0 - self.components['prior_d'](y)).mean()
            prior = -(term_a + term_b) * self.gamma
        else:
            prior = 0
        return local_global_loss + prior

    def _prepare_batch(self, batch):
        """
        Prepares the batch for the model by converting the GraphData numpy arrays to torch tensors and moving them to the device.
        """
        inputs, labels, weights = batch
        inputs = BatchGraphData(inputs[0]).numpy_to_torch(self.device)

        _, labels, weights = super()._prepare_batch(([], labels, weights))

        if (len(labels) != 0) and (len(weights) != 0):
            labels = labels[0]
            weights = weights[0]

        return inputs, labels, weights


class InfoGraphStar(torch.nn.Module):
    """
    The nn.Module for InfoGraphStar. This class defines the forward pass of InfoGraphStar.

    Parameters
    ----------
    encoder: torch.nn.Module
        The encoder for InfoGraphStar.
    unsup_encoder: torch.nn.Module
        The unsupervised encoder for InfoGraph, of identical architecture to encoder.
    ff1: torch.nn.Module
        The first feedforward layer for InfoGraphStar.
    ff2: torch.nn.Module
        The second feedforward layer for InfoGraphStar.
    fc1: torch.nn.Module
        The first fully connected layer for InfoGraphStar.
    fc2: torch.nn.Module
        The second fully connected layer for InfoGraphStar.

    References
    ----------
    .. [1] F.-Y. Sun, J. Hoffmann, V. Verma, and J. Tang, “InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Maximization.” arXiv, Jan. 17, 2020. http://arxiv.org/abs/1908.01000


    Example
    -------
    >>> import torch
    >>> import numpy as np
    >>> from deepchem.models.torch_models.infograph import InfoGraphStarModel
    >>> from deepchem.feat.molecule_featurizers import MolGraphConvFeaturizer
    >>> from deepchem.feat.graph_data import BatchGraphData
    >>> num_feat = 30
    >>> num_edge = 11
    >>> infographmodular = InfoGraphStarModel(num_feat,num_edge,64)
    >>> smiles = ['C1=CC=CC=C1', 'C1=CC=CC=C1C2=CC=CC=C2']
    >>> featurizer = MolGraphConvFeaturizer(use_edges=True)
    >>> graphs = BatchGraphData(featurizer.featurize(smiles)).numpy_to_torch(infographmodular.device)
    >>> model = infographmodular.model
    >>> output = model(graphs).cpu().detach().numpy()

    """

    def __init__(self,
                 encoder,
                 unsup_encoder,
                 ff1,
                 ff2,
                 fc1,
                 fc2,
                 local_d,
                 global_d,
                 mode='regression',
                 num_tasks=1,
                 num_classes=2,
                 init_emb=False):
        super().__init__()
        self.encoder = encoder
        self.unsup_encoder = unsup_encoder
        self.ff1 = ff1
        self.ff2 = ff2
        self.fc1 = fc1
        self.fc2 = fc2
        self.local_d = local_d
        self.global_d = global_d
        self.mode = mode
        self.num_tasks = num_tasks
        self.num_classes = num_classes
        if init_emb:
            self.init_emb()

    def init_emb(self):
        """
        Initializes the parameters of the model by setting their values from a uniform distribution and filling the bias with 0.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data):
        """
        Forward pass for InfoGraphStar.

        Parameters
        ----------
        data: Union[GraphData, BatchGraphData]
            The input data, either a single graph or a batch of graphs.
        """

        out, M = self.encoder(data)
        out = F.relu(self.fc1(out))
        pred = self.fc2(out)
        if self.mode == 'classification':
            pred = torch.reshape(pred, (-1, self.num_tasks, self.num_classes))
        return pred


class InfoGraphStarModel(ModularTorchModel):
    """
    InfographStar is a semi-supervised graph convolutional network for predicting molecular properties.
    It aims to maximize the mutual information between the graph-level representation and the
    representations of substructures of different scales. It does this by producing graph-level
    encodings and substructure encodings, and then using a discriminator to classify if they
    are from the same molecule or not.

    Supervised training is done by using the graph-level encodings to predict the target property. Semi-supervised training is done by adding a loss term that maximizes the mutual information between the graph-level encodings and the substructure encodings to the supervised loss.
    These modes can be chosen by setting the training_mode parameter.

    To conduct training in unsupervised mode, use InfoGraphModel.

    References
    ----------
    .. [1] F.-Y. Sun, J. Hoffmann, V. Verma, and J. Tang, “InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Maximization.” arXiv, Jan. 17, 2020. http://arxiv.org/abs/1908.01000

    Parameters
    ----------
    num_features: int
        Number of node features for each input
    edge_features: int
        Number of edge features for each input
    embedding_dim: int
        Dimension of the embedding
    training_mode: str
        The mode to use for training. Options are 'supervised', 'semisupervised'. For unsupervised training, use InfoGraphModel.
    measure: str
        The divergence measure to use for the unsupervised loss. Options are 'GAN', 'JSD',
        'KL', 'RKL', 'X2', 'DV', 'H2', or 'W1'.
    average_loss: bool
        Whether to average the loss over the batch

    Examples
    --------
    >>> from deepchem.models.torch_models import InfoGraphStarModel
    >>> from deepchem.feat import MolGraphConvFeaturizer
    >>> from deepchem.data import NumpyDataset
    >>> import torch
    >>> smiles = ["C1CCC1", "C1=CC=CN=C1"]
    >>> featurizer = MolGraphConvFeaturizer(use_edges=True)
    >>> X = featurizer.featurize(smiles)
    >>> y = torch.randint(0, 2, size=(2, 1)).float()
    >>> w = torch.ones(size=(2, 1)).float()
    >>> ds = NumpyDataset(X, y, w)
    >>> num_feat = max([ds.X[i].num_node_features for i in range(len(ds))])
    >>> edge_dim = max([ds.X[i].num_edge_features for i in range(len(ds))])
    >>> model = InfoGraphStarModel(num_feat, edge_dim, 15, training_mode='semisupervised')
    >>> loss = model.fit(ds, nb_epoch=1)
    """

    def __init__(self,
                 num_features,
                 edge_features,
                 embedding_dim,
                 task='supervised',
                 mode='regression',
                 num_classes=2,
                 num_tasks=1,
                 measure='JSD',
                 average_loss=True,
                 num_gc_layers=5,
                 **kwargs):

        self.edge_features = edge_features
        self.local = True
        self.prior = False
        self.gamma = .1
        self.num_features = num_features
        self.task = task
        self.mode = mode
        self.num_classes = num_classes
        if self.mode == 'regression':
            self.output_dim = num_tasks
        elif self.mode == 'classification':
            self.num_tasks = num_tasks
            self.output_dim = num_classes * num_tasks
            self.class_loss = SoftmaxCrossEntropy()._create_pytorch_loss()
        if self.task == 'supervised':
            self.embedding_dim = embedding_dim
        elif self.task == 'semisupervised':
            self.embedding_dim = embedding_dim * num_gc_layers
            self.num_gc_layers = num_gc_layers
        self.localloss = LocalMutualInformationLoss()._create_pytorch_loss(
            measure, average_loss)
        self.globalloss = GlobalMutualInformationLoss()._create_pytorch_loss(
            measure, average_loss)

        self.components = self.build_components()
        self.model = self.build_model()
        super().__init__(self.model, self.components, **kwargs)

    def build_components(self):
        """
        Builds the components of the InfoGraphStar model. InfoGraphStar works by maximizing the mutual information between the graph-level representation and the representations of substructures of different scales.

        It does this by producing graph-level encodings and substructure encodings, and then using a discriminator to classify if they are from the same molecule or not.

        The encoder is a graph convolutional network that produces the graph-level encodings and substructure encodings.

        In a supervised training mode, only 1 encoder is used and the encodings are not compared, while in a semi-supvervised training mode they are different in order to prevent negative transfer from the pretraining stage.

        The local discriminator is a multilayer perceptron that classifies if the substructure encodings are from the same molecule or not while the global discriminator classifies if the graph-level encodings are from the same molecule or not.

        Components list, type and description:
        --------------------------------------
        encoder: InfoGraphEncoder

        unsup_encoder: InfoGraphEncoder for supervised or GINEncoder for unsupervised training

        ff1: MultilayerPerceptron, feedforward network

        ff2: MultilayerPerceptron, feedforward network

        fc1: torch.nn.Linear, fully connected layer

        fc2: torch.nn.Linear, fully connected layer

        local_d: MultilayerPerceptron, local discriminator

        global_d: MultilayerPerceptron, global discriminator
        """
        if self.task == 'supervised':
            return {
                'encoder':
                    InfoGraphEncoder(self.num_features, self.edge_features,
                                     self.embedding_dim),
                'unsup_encoder':
                    InfoGraphEncoder(self.num_features, self.edge_features,
                                     self.embedding_dim),
                'ff1':
                    MultilayerPerceptron(2 * self.embedding_dim,
                                         self.embedding_dim,
                                         (self.embedding_dim,)),
                'ff2':
                    MultilayerPerceptron(2 * self.embedding_dim,
                                         self.embedding_dim,
                                         (self.embedding_dim,)),
                'fc1':
                    torch.nn.Linear(2 * self.embedding_dim, self.embedding_dim),
                'fc2':
                    torch.nn.Linear(self.embedding_dim, self.output_dim),
                'local_d':
                    MultilayerPerceptron(self.embedding_dim,
                                         self.embedding_dim,
                                         (self.embedding_dim,),
                                         skip_connection=True),
                'global_d':
                    MultilayerPerceptron(2 * self.embedding_dim,
                                         self.embedding_dim,
                                         (self.embedding_dim,),
                                         skip_connection=True)
            }
        elif self.task == 'semisupervised':
            return {
                'encoder':
                    InfoGraphEncoder(self.num_features, self.edge_features,
                                     self.embedding_dim),
                'unsup_encoder':
                    GINEncoder(self.num_features, self.embedding_dim,
                               self.num_gc_layers),
                'ff1':
                    MultilayerPerceptron(2 * self.embedding_dim,
                                         self.embedding_dim,
                                         (self.embedding_dim,)),
                'ff2':
                    MultilayerPerceptron(self.embedding_dim, self.embedding_dim,
                                         (self.embedding_dim,)),
                'fc1':
                    torch.nn.Linear(2 * self.embedding_dim, self.embedding_dim),
                'fc2':
                    torch.nn.Linear(self.embedding_dim, self.output_dim),
                'local_d':
                    MultilayerPerceptron(self.embedding_dim,
                                         self.embedding_dim,
                                         (self.embedding_dim,),
                                         skip_connection=True),
                'global_d':
                    MultilayerPerceptron(self.embedding_dim,
                                         self.embedding_dim,
                                         (self.embedding_dim,),
                                         skip_connection=True)
            }

    def build_model(self):
        """
        Builds the InfoGraph model by unpacking the components dictionary and passing them to the InfoGraph nn.module.
        """
        if self.mode == 'regression':
            return InfoGraphStar(**self.components,)
        elif self.mode == 'classification':
            return InfoGraphStar(**self.components,
                                 mode=self.mode,
                                 num_tasks=self.num_tasks,
                                 num_classes=self.num_classes)

    def loss_func(self, inputs, labels, weights):
        sup_loss = self.sup_loss(inputs, labels)
        if self.task == 'semisupervised':
            local_unsup_loss = self.local_unsup_loss(inputs)
            global_unsup_loss = self.global_unsup_loss(inputs, labels, weights)
            loss = sup_loss + local_unsup_loss + global_unsup_loss * self.learning_rate
            # original implementation also includes an option if not using a separate encoder:
            # loss = sup_loss + local_unsup_loss * self.learning_rate
            return (loss * weights).mean()
        else:
            return (sup_loss * weights).mean()

    def sup_loss(self, inputs, labels):
        if self.mode == 'regression':
            out = self.model(inputs)
            sup_loss = F.mse_loss(out, labels)
        elif self.mode == 'classification':
            out = self.model(inputs)
            out = F.softmax(out, dim=2)
            sup_loss = self.class_loss(out, labels)
        return sup_loss

    def local_unsup_loss(self, inputs):
        if self.task == 'semisupervised':
            y, M = self.components['unsup_encoder'](inputs)
        else:
            y, M = self.components['encoder'](inputs)
        g_enc = self.components['global_d'](y)
        l_enc = self.components['local_d'](M)

        if self.local:
            loss = self.localloss(l_enc, g_enc, inputs.graph_index)

        return loss

    def global_unsup_loss(self, inputs, labels, weights):
        y, M = self.components['encoder'](inputs)
        y_, M_ = self.components['unsup_encoder'](inputs)

        g_enc = self.components['ff1'](y)
        g_enc1 = self.components['ff2'](y_)

        loss = self.globalloss(g_enc, g_enc1)
        return loss

    def _prepare_batch(self, batch):
        """
        Prepares the batch for the model by converting the GraphData numpy arrays to torch tensors and moving them to the device.
        """
        inputs, labels, weights = batch
        inputs = BatchGraphData(inputs[0]).numpy_to_torch(self.device)

        _, labels, weights = super()._prepare_batch(([], labels, weights))

        if (len(labels) != 0) and (len(weights) != 0):
            labels = labels[0]
            weights = weights[0]

        return inputs, labels, weights

    def default_generator(
            self,
            dataset: dc.data.Dataset,
            epochs: int = 1,
            mode: str = 'fit',
            deterministic: bool = True,
            pad_batches: bool = True) -> Iterable[Tuple[List, List, List]]:
        for epoch in range(epochs):
            for (X_b, y_b, w_b,
                 ids_b) in dataset.iterbatches(batch_size=self.batch_size,
                                               deterministic=deterministic,
                                               pad_batches=pad_batches):
                if self.mode == 'classification' and y_b is not None:
                    y_b = to_one_hot(y_b.flatten(), self.num_classes).reshape(
                        -1, self.num_tasks, self.num_classes)
                yield ([X_b], [y_b], [w_b])

from deepchem.models.torch_models.modular import ModularTorchModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU
from deepchem.feat.graph_data import BatchGraphData
from deepchem.models.torch_models.layers import MultilayerPerceptron
import math


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
    dim: int
        Dimension of the embedding
    """

    def __init__(self, num_features, edge_features, dim):
        from torch_geometric.nn import NNConv
        from torch_geometric.nn.aggr import Set2Set

        super().__init__()
        self.lin0 = torch.nn.Linear(num_features, dim)

        nn = Sequential(Linear(edge_features, 128), ReLU(),
                        Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean', root_weight=False)
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)

    def forward(self, data):
        out = F.relu(self.lin0(data.node_features))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_features))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
            feat_map = out

        out = self.set2set(out, data.graph_index)
        return out, feat_map


class InfoGraphModel(ModularTorchModel):
    """
    Infograph is a semi-supervised graph convolutional network for predicting molecular properties.
    It aims to maximize the mutual information between the graph-level representation and the
    representations of substructures of different scales. It does this by producing graph-level
    encodings and substructure encodings, and then using a discriminator to classify if they
    are from the same molecule or not.

    To conduct training in unsupervised mode, use_unsup_loss should be True and separate_encoder
    should be set to False. For semi-supervised training, use_unsup_loss should be True and
    separate_encoder should be True. For supervised training, use_unsup_loss should be False.

    References
    ----------
    F.-Y. Sun, J. Hoffmann, V. Verma, and J. Tang, “InfoGraph: Unsupervised and Semi-supervised
    Graph-Level Representation Learning via Mutual Information Maximization.” arXiv, Jan. 17, 2020.
    http://arxiv.org/abs/1908.01000

    Parameters
    ----------
    num_features: int
        Number of node features for each input
    edge_features: int
        Number of edge features for each input
    dim: int
        Dimension of the embedding
    use_unsup_loss: bool
        Whether to use the unsupervised loss
    separate_encoder: bool
        Whether to use a separate encoder for the unsupervised loss
    measure: str
        The divergence measure to use for the unsupervised loss. Options are 'GAN', 'JSD',
        'KL', 'RKL', 'X2', 'DV', 'H2', or 'W1'.
    average_loss: bool
        Whether to average the loss over the batch

    Examples
    --------
    >>> from deepchem.models.torch_models import InfoGraphModel
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
    >>> model = InfoGraphModel(num_feat, edge_dim, 15, use_unsup_loss=True, separate_encoder=True)
    >>> loss = model.fit(ds, nb_epoch=1)
    """

    def __init__(self,
                 num_features,
                 edge_features,
                 dim,
                 use_unsup_loss=False,
                 separate_encoder=False,
                 measure='JSD',
                 average_loss=True,
                 **kwargs):
        self.embedding_dim = dim
        self.edge_features = edge_features
        self.separate_encoder = separate_encoder
        self.local = True
        self.num_features = num_features
        self.use_unsup_loss = use_unsup_loss
        self.measure = measure
        self.average_loss = average_loss

        self.components = self.build_components()
        self.model = self.build_model()
        super().__init__(self.model, self.components, **kwargs)

    def build_components(self):
        return {
            'encoder':
                InfoGraphEncoder(self.num_features, self.edge_features,
                                 self.embedding_dim),
            'unsup_encoder':
                InfoGraphEncoder(self.num_features, self.edge_features,
                                 self.embedding_dim),
            'ff1':
                MultilayerPerceptron(2 * self.embedding_dim, self.embedding_dim,
                                     (self.embedding_dim,)),
            'ff2':
                MultilayerPerceptron(2 * self.embedding_dim, self.embedding_dim,
                                     (self.embedding_dim,)),
            'fc1':
                torch.nn.Linear(2 * self.embedding_dim, self.embedding_dim),
            'fc2':
                torch.nn.Linear(self.embedding_dim, 1),
            'local_d':
                MultilayerPerceptron(self.embedding_dim,
                                     self.embedding_dim, (self.embedding_dim,),
                                     skip_connection=True),
            'global_d':
                MultilayerPerceptron(2 * self.embedding_dim,
                                     self.embedding_dim, (self.embedding_dim,),
                                     skip_connection=True),
        }

    def build_model(self):
        """
        Builds the InfoGraph model by unpacking the components dictionary and passing them to the InfoGraph nn.module.
        """
        return InfoGraph(**self.components)

    def loss_func(self, inputs, labels, weights):
        if self.use_unsup_loss:
            sup_loss = F.mse_loss(self.model(inputs), labels)
            unsup_loss = self.unsup_loss(inputs)
            if self.separate_encoder:
                unsup_sup_loss = self.unsup_sup_loss(inputs, labels, weights)
                loss = sup_loss + unsup_loss + unsup_sup_loss * self.learning_rate
            else:
                loss = sup_loss + unsup_loss * self.learning_rate
            return (loss * weights).mean()
        else:
            sup_loss = F.mse_loss(self.model(inputs), labels)
            return (sup_loss * weights).mean()

    def unsup_loss(self, inputs):
        if self.separate_encoder:
            y, M = self.components['unsup_encoder'](inputs)
        else:
            y, M = self.components['encoder'](inputs)
        g_enc = self.components['global_d'](y)
        l_enc = self.components['local_d'](M)

        if self.local:
            loss = self._local_global_loss(l_enc, g_enc, inputs.graph_index)
        return loss

    def _prepare_batch(self, batch):
        inputs, labels, weights = batch
        inputs = BatchGraphData(inputs[0])
        inputs.edge_features = torch.from_numpy(
            inputs.edge_features).float().to(self.device)
        inputs.edge_index = torch.from_numpy(inputs.edge_index).long().to(
            self.device)
        inputs.node_features = torch.from_numpy(
            inputs.node_features).float().to(self.device)
        inputs.graph_index = torch.from_numpy(inputs.graph_index).long().to(
            self.device)

        _, labels, weights = super()._prepare_batch(([], labels, weights))

        if (len(labels) != 0) and (len(weights) != 0):
            labels = labels[0]
            weights = weights[0]

        return inputs, labels, weights

    def unsup_sup_loss(self, inputs, labels, weights):
        y, M = self.components['encoder'](inputs)
        y_, M_ = self.components['unsup_encoder'](inputs)

        g_enc = self.components['ff1'](y)
        g_enc1 = self.components['ff2'](y_)

        loss = self._global_global_loss(g_enc, g_enc1)
        return loss

    def _local_global_loss(self, l_enc, g_enc, batch):
        """
        Parameters:
        ----------
        l_enc: torch.Tensor
            Local feature map from the encoder.
        g_enc: torch.Tensor
            Global features from the encoder.
        batch: torch.Tensor
            Batch tensor

        Returns:
        -------
        loss: torch.Tensor
            Local Global Loss value
        """
        num_graphs = g_enc.shape[0]
        num_nodes = l_enc.shape[0]

        pos_mask = torch.zeros((num_nodes, num_graphs)).to(self.device)
        neg_mask = torch.ones((num_nodes, num_graphs)).to(self.device)
        for nodeidx, graphidx in enumerate(batch):
            pos_mask[nodeidx][graphidx] = 1.
            neg_mask[nodeidx][graphidx] = 0.

        res = torch.mm(l_enc, g_enc.t())

        E_pos = self.get_positive_expectation(res * pos_mask)
        E_pos = (E_pos * pos_mask).sum() / pos_mask.sum()
        E_neg = self.get_negative_expectation(res * neg_mask)
        E_neg = (E_neg * neg_mask).sum() / neg_mask.sum()

        return E_neg - E_pos

    def _global_global_loss(self, g_enc, g_enc1):
        """
        Parameters:
        ----------
        g_enc: torch.Tensor
            Global features from the encoder.
        g_enc1: torch.Tensor
            Global features from the separate encoder.

        Returns:
        -------
        loss: torch.Tensor
            Global Global Loss value
        """
        num_graphs = g_enc.shape[0]

        pos_mask = torch.eye(num_graphs).to(self.device)
        neg_mask = 1 - pos_mask

        res = torch.mm(g_enc, g_enc1.t())

        E_pos = self.get_positive_expectation(res * pos_mask)
        E_pos = (E_pos * pos_mask).sum() / pos_mask.sum()
        E_neg = self.get_negative_expectation(res * neg_mask)
        E_neg = (E_neg * neg_mask).sum() / neg_mask.sum()

        return E_neg - E_pos

    def get_positive_expectation(self, p_samples):
        """Computes the positive part of a divergence / difference.

        Parameters:
        ----------
        p_samples: torch.Tensor
            Positive samples.
        average: bool
            Average the result over samples.

        Returns:
        -------
        Ep: torch.Tensor
            Positive part of the divergence / difference.
        """
        log_2 = math.log(2.)

        if self.measure == 'GAN':
            Ep = -F.softplus(-p_samples)
        elif self.measure == 'JSD':
            Ep = log_2 - F.softplus(-p_samples)
        elif self.measure == 'X2':
            Ep = p_samples**2
        elif self.measure == 'KL':
            Ep = p_samples + 1.
        elif self.measure == 'RKL':
            Ep = -torch.exp(-p_samples)
        elif self.measure == 'DV':
            Ep = p_samples
        elif self.measure == 'H2':
            Ep = 1. - torch.exp(-p_samples)
        elif self.measure == 'W1':
            Ep = p_samples
        else:
            raise ValueError('Unknown measure: {}'.format(self.measure))

        if self.average_loss:
            return Ep.mean()
        else:
            return Ep

    def get_negative_expectation(self, q_samples):
        """Computes the negative part of a divergence / difference.

        Parameters:
        ----------
        q_samples: torch.Tensor
            Negative samples.
        average: bool
            Average the result over samples.

        Returns:
        -------
        Ep: torch.Tensor
            Negative part of the divergence / difference.

        """
        log_2 = math.log(2.)

        if self.measure == 'GAN':
            Eq = F.softplus(-q_samples) + q_samples
        elif self.measure == 'JSD':
            Eq = F.softplus(-q_samples) + q_samples - log_2
        elif self.measure == 'X2':
            Eq = -0.5 * ((torch.sqrt(q_samples**2) + 1.)**2)
        elif self.measure == 'KL':
            Eq = torch.exp(q_samples)
        elif self.measure == 'RKL':
            Eq = q_samples - 1.
        elif self.measure == 'DV':
            Eq = log_sum_exp(q_samples, 0) - math.log(q_samples.size(0))
        elif self.measure == 'H2':
            Eq = torch.exp(q_samples) - 1.
        elif self.measure == 'W1':
            Eq = q_samples
        else:
            raise ValueError('Unknown measure: {}'.format(self.measure))

        if self.average_loss:
            return Eq.mean()
        else:
            return Eq


class InfoGraph(torch.nn.Module):
    """
    The nn.Module for InfoGraph. This class defines the forward pass of InfoGraph.

    Parameters
    ----------
    encoder: torch.nn.Module
        The encoder for InfoGraph.
    unsup_encoder: torch.nn.Module
        The unsupervised encoder for InfoGraph, of identical architecture to encoder.
    ff1: torch.nn.Module
        The first feedforward layer for InfoGraph.
    ff2: torch.nn.Module
        The second feedforward layer for InfoGraph.
    fc1: torch.nn.Module
        The first fully connected layer for InfoGraph.
    fc2: torch.nn.Module
        The second fully connected layer for InfoGraph.


    Example
    -------
    >>> import torch
    >>> import numpy as np
    >>> from deepchem.models.torch_models.infograph import InfoGraphModel
    >>> from deepchem.feat.molecule_featurizers import MolGraphConvFeaturizer
    >>> from deepchem.feat.graph_data import BatchGraphData
    >>> smiles = ['C1=CC=CC=C1', 'C1=CC=CC=C1C2=CC=CC=C2']
    >>> featurizer = MolGraphConvFeaturizer(use_edges=True)
    >>> graphs = BatchGraphData(featurizer.featurize(smiles))
    >>> num_feat = 30
    >>> num_edge = 11
    >>> infographmodular = InfoGraphModel(num_feat,num_edge,64,use_unsup_loss=True,separate_encoder=True)
    >>>  # convert features to torch tensors
    >>> graphs.edge_features = torch.from_numpy(graphs.edge_features).to(infographmodular.device).float()
    >>> graphs.edge_index = torch.from_numpy(graphs.edge_index).to(infographmodular.device).long()
    >>> graphs.node_features = torch.from_numpy(graphs.node_features).to(infographmodular.device).float()
    >>> graphs.graph_index = torch.from_numpy(graphs.graph_index).to(infographmodular.device).long()
    >>> model = infographmodular.model
    >>> output = model(graphs).cpu().detach().numpy()

    """

    def __init__(self, encoder, unsup_encoder, ff1, ff2, fc1, fc2, local_d,
                 global_d):
        super(InfoGraph, self).__init__()
        self.encoder = encoder
        self.unsup_encoder = unsup_encoder
        self.ff1 = ff1
        self.ff2 = ff2
        self.fc1 = fc1
        self.fc2 = fc2
        self.local_d = local_d
        self.global_d = global_d
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data):
        """
        Forward pass for InfoGraph.

        Parameters
        ----------
        data: Union[GraphData, BatchGraphData]
            The input data, either a single graph or a batch of graphs.
        """

        out, M = self.encoder(data)
        out = F.relu(self.fc1(out))
        pred = self.fc2(out)
        return pred


def log_sum_exp(x, axis=None):
    """Log sum exp function.

    Parameters
    ----------
    x: torch.Tensor
        Input tensor
    axis: int
        Axis to perform sum over

    Returns
    -------
    y: torch.Tensor
        Log sum exp of x

    """
    x_max = torch.max(x, axis)[0]
    y = torch.log((torch.exp(x - x_max)).sum(axis)) + x_max
    return y

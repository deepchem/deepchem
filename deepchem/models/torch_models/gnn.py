import random
import copy
import torch
import numpy as np
from torch_geometric.nn import GINEConv, GCNConv, GATConv, SAGEConv, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn.aggr import AttentionalAggregation, Set2Set
from torch_geometric.nn.inits import uniform
import torch.nn as nn
from torch.functional import F
from deepchem.data import Dataset
from deepchem.models.losses import SoftmaxCrossEntropy, EdgePredictionLoss, GraphNodeMaskingLoss, GraphEdgeMaskingLoss, DeepGraphInfomaxLoss, GraphContextPredLoss
from deepchem.models.torch_models import ModularTorchModel
from deepchem.feat.graph_data import BatchGraphData, GraphData, shortest_path_length
from typing import Iterable, List, Tuple
from deepchem.metrics import to_one_hot

num_node_type = 120  # including the extra mask tokens
num_chirality_tag = 3
num_edge_type = 6  # including aromatic and self-loop edge, and extra masked tokens


class GNN(torch.nn.Module):
    """
    GNN module for the GNNModular model.

    This module is responsible for the graph neural network layers in the GNNModular model.

    Parameters
    ----------
    node_type_embedding: torch.nn.Embedding
        Embedding layer for node types.
    chirality_embedding: torch.nn.Embedding
        Embedding layer for chirality tags.
    gconvs: torch.nn.ModuleList
        ModuleList of graph convolutional layers.
    batch_norms: torch.nn.ModuleList
        ModuleList of batch normalization layers.
    dropout: int
        Dropout probability.
    jump_knowledge: str
        The type of jump knowledge to use. [1] Must be one of "last", "sum", "max", "concat" or "none".
        "last": Use the node representation from the last GNN layer.
        "concat": Concatenate the node representations from all GNN layers.
        "max": Take the element-wise maximum of the node representations from all GNN layers.
        "sum": Take the element-wise sum of the node representations from all GNN layers.
    init_emb: bool
        Whether to initialize the embedding layers with Xavier uniform initialization.

    References
    ----------
    .. [1] Xu, K. et al. Representation Learning on Graphs with Jumping Knowledge Networks. Preprint at https://doi.org/10.48550/arXiv.1806.03536 (2018).

    Example
    -------
    >>> from deepchem.models.torch_models.gnn import GNNModular
    >>> from deepchem.feat.graph_data import BatchGraphData
    >>> from deepchem.feat.molecule_featurizers import SNAPFeaturizer
    >>> featurizer = SNAPFeaturizer()
    >>> smiles = ["C1=CC=CC=C1", "C1=CC=CC=C1C=O", "C1=CC=CC=C1C(=O)O"]
    >>> features = featurizer.featurize(smiles)
    >>> modular = GNNModular(emb_dim = 8, task = "edge_pred")
    >>> batched_graph = BatchGraphData(features).numpy_to_torch(device=modular.device)
    >>> gnnmodel = modular.gnn
    >>> print(gnnmodel(batched_graph)[0].shape)
    torch.Size([23, 8])

    """

    def __init__(self,
                 node_type_embedding,
                 chirality_embedding,
                 gconvs,
                 batch_norms,
                 dropout,
                 jump_knowledge,
                 init_emb=False):
        super(GNN, self).__init__()

        self.node_type_embedding = node_type_embedding
        self.chirality_embedding = chirality_embedding
        self.gconv = gconvs
        self.batch_norms = batch_norms
        self.dropout = dropout
        self.num_layer = len(gconvs)
        self.jump_knowledge = jump_knowledge

        # may mess with loading pretrained weights
        if init_emb:
            torch.nn.init.xavier_uniform_(self.node_type_embedding.weight.data)
            torch.nn.init.xavier_uniform_(self.chirality_embedding.weight.data)

    def forward(self, data: BatchGraphData):
        """
        Forward pass for the GNN module.

        Parameters
        ----------
        data: BatchGraphData
            Batched graph data.
        """

        node_feats = data.node_features[:, 0].long()  # type: ignore
        chiral_feats = data.node_features[:, 1].long()  # type: ignore
        node_emb = self.node_type_embedding(node_feats)
        chir_emb = self.chirality_embedding(chiral_feats)
        x = node_emb + chir_emb

        h_list = [x]
        for i, conv_layer in enumerate(self.gconv):
            if isinstance(conv_layer, (GINEConv, GATConv)):
                h = conv_layer(h_list[i], data.edge_index, data.edge_features)
            elif isinstance(conv_layer, (GCNConv, SAGEConv)):
                h = conv_layer(h_list[i], data.edge_index)
            h = self.batch_norms[i](h)
            h = F.dropout(F.relu(h), self.dropout, training=self.training)
            if i == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.dropout, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training=self.training)
            h_list.append(h)

        if self.jump_knowledge == "concat":
            node_representation = torch.cat(h_list, dim=1)
            # reshapes node_representation to (num_nodes, num_layers * emb_dim)
        elif self.jump_knowledge == "last":
            node_representation = h_list[-1]
        elif self.jump_knowledge == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.jump_knowledge == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        return (node_representation, data)


class GNNHead(torch.nn.Module):
    """
    Prediction head module for the GNNModular model.

    Parameters
    ----------
    pool: Union[function,torch.nn.Module]
        Pooling function or nn.Module to use
    head: torch.nn.Module
        Prediction head to use
    task: str
        The type of task. Must be one of "regression", "classification".
    num_tasks: int
        Number of tasks.
    num_classes: int
        Number of classes for classification.
    """

    def __init__(self, pool, head, task, num_tasks, num_classes):
        super().__init__()
        self.pool = pool
        self.head = head
        self.task = task
        self.num_tasks = num_tasks
        self.num_classes = num_classes

    def forward(self, data):
        """
        Forward pass for the GNN head module.

        Parameters
        ----------
        data: tuple
            A tuple containing the node representations and the input graph data.
            node_representation is a torch.Tensor created after passing input through the GNN layers.
            input_batch is the original input BatchGraphData.
        """
        node_representation, input_batch = data

        pooled = self.pool(node_representation, input_batch.graph_index)
        out = self.head(pooled)
        if self.task == "classification":
            out = torch.reshape(out, (-1, self.num_tasks, self.num_classes))
        return out


class LocalGlobalDiscriminator(nn.Module):
    """
    This discriminator module is a linear layer without bias, used to measure the similarity between local node representations (`x`) and global graph representations (`summary`).

    The goal of the discriminator is to distinguish between positive and negative pairs of local and global representations.

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.torch_models.gnn import LocalGlobalDiscriminator
    >>> discriminator = LocalGlobalDiscriminator(hidden_dim=64)
    >>> x = torch.randn(32, 64)  # Local node representations
    >>> summary = torch.randn(32, 64)  # Global graph representations
    >>> similarity_scores = discriminator(x, summary)
    >>> print(similarity_scores.shape)
    torch.Size([32])
    """

    def __init__(self, hidden_dim):
        """
        `self.weight` is a learnable weight matrix of shape `(hidden_dim, hidden_dim)`.

        nn.Parameters are tensors that require gradients and are optimized during the training process.

        Parameters
        ----------
        hidden_dim : int
            The size of the hidden dimension for the weight matrix.

        """
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)

    def forward(self, x, summary):
        """
        Computes the product of `summary` and `self.weight`, and then calculates the element-wise product of `x` and the resulting matrix `h`.
        It then sums over the `hidden_dim` dimension, resulting in a tensor of shape `(batch_size,)`, which represents the similarity scores between the local and global representations.

        Parameters
        ----------
        x : torch.Tensor
            Local node representations of shape `(batch_size, hidden_dim)`.
        summary : torch.Tensor
            Global graph representations of shape `(batch_size, hidden_dim)`.

        Returns
        -------
        torch.Tensor
            A tensor of shape `(batch_size,)`, representing the similarity scores between the local and global representations.
        """
        h = torch.matmul(summary, self.weight)
        return torch.sum(x * h, dim=1)


class GNNModular(ModularTorchModel):
    """
    Modular GNN which allows for easy swapping of GNN layers.

    Parameters
    ----------
    gnn_type: str
        The type of GNN layer to use. Must be one of "gin", "gcn", "graphsage", or "gat".
    num_layer: int
        The number of GNN layers to use.
    emb_dim: int
        The dimensionality of the node embeddings.
    num_tasks: int
        The number of tasks.
    graph_pooling: str
        The type of graph pooling to use. Must be one of "sum", "mean", "max", "attention" or "set2set".
        "sum" may cause issues with positive prediction loss.
    dropout: float, optional (default 0)
        The dropout probability.
    jump_knowledge: str, optional (default "last")
        The type of jump knowledge to use. [1] Must be one of "last", "sum", "max", or "concat".
        "last": Use the node representation from the last GNN layer.
        "concat": Concatenate the node representations from all GNN layers. This will increase the dimensionality of the node representations by a factor of `num_layer`.
        "max": Take the element-wise maximum of the node representations from all GNN layers.
        "sum": Take the element-wise sum of the node representations from all GNN layers. This may cause issues with positive prediction loss.
    task: str, optional (default "regression")
        The type of task.
        Unsupervised tasks:
        edge_pred: Edge prediction. Predicts whether an edge exists between two nodes.
        mask_nodes: Masking nodes. Predicts the masked node.
        mask_edges: Masking edges. Predicts the masked edge.
        infomax: Infomax. Maximizes mutual information between local node representations and a pooled global graph representation.
        context_pred: Context prediction. Predicts the surrounding context of a node.
        Supervised tasks:
        "regression" or "classification".
    mask_rate: float, optional (default 0.1)
        The rate at which to mask nodes or edges for mask_nodes and mask_edges tasks.
    mask_edge: bool, optional (default True)
        Whether to also mask connecting edges for mask_nodes tasks.
    context_size: int, optional (default 1)
        The size of the context to use for context prediction tasks.
    neighborhood_size: int, optional (default 3)
        The size of the neighborhood to use for context prediction tasks.
    context_mode: str, optional (default "cbow")
        The context mode to use for context prediction tasks. Must be one of "cbow" or "skipgram".
    neg_samples: int, optional (default 1)
        The number of negative samples to use for context prediction.

    Examples
    --------
    >>> import numpy as np
    >>> import deepchem as dc
    >>> from deepchem.feat.molecule_featurizers import SNAPFeaturizer
    >>> from deepchem.models.torch_models.gnn import GNNModular
    >>> featurizer = SNAPFeaturizer()
    >>> smiles = ["C1=CC=CC=C1", "C1=CC=CC=C1C=O", "C1=CC=CC=C1C(=O)O"]
    >>> features = featurizer.featurize(smiles)
    >>> dataset = dc.data.NumpyDataset(features, np.zeros(len(features)))
    >>> model = GNNModular(task="edge_pred")
    >>> loss = model.fit(dataset, nb_epoch=1)

    References
    ----------
    .. [1] Xu, K. et al. Representation Learning on Graphs with Jumping Knowledge Networks. Preprint at https://doi.org/10.48550/arXiv.1806.03536 (2018).
    .. [2] Hu, W. et al. Strategies for Pre-training Graph Neural Networks. Preprint at https://doi.org/10.48550/arXiv.1905.12265 (2020).
    """

    def __init__(self,
                 gnn_type: str = "gin",
                 num_layer: int = 3,
                 emb_dim: int = 64,
                 num_tasks: int = 1,
                 num_classes: int = 2,
                 graph_pooling: str = "mean",
                 dropout: int = 0,
                 jump_knowledge: str = "last",
                 task: str = "edge_pred",
                 mask_rate: float = .1,
                 mask_edge: bool = True,
                 context_size: int = 1,
                 neighborhood_size: int = 3,
                 context_mode: str = "cbow",
                 neg_samples: int = 1,
                 **kwargs):
        self.gnn_type = gnn_type
        self.num_layer = num_layer
        self.emb_dim = emb_dim

        self.num_tasks = num_tasks
        self.num_classes = num_classes
        if task == "classification":
            self.output_dim = num_classes * num_tasks
            self.criterion = SoftmaxCrossEntropy()._create_pytorch_loss()
        elif task == "regression":
            self.output_dim = num_tasks
            self.criterion = F.mse_loss
        elif task == "edge_pred":
            self.output_dim = num_tasks
            self.edge_pred_loss = EdgePredictionLoss()._create_pytorch_loss()
        elif task == "mask_nodes":
            self.mask_rate = mask_rate
            self.mask_edge = mask_edge
            self.node_mask_loss = GraphNodeMaskingLoss()._create_pytorch_loss(
                self.mask_edge)
        elif task == "mask_edges":
            self.mask_rate = mask_rate
            self.edge_mask_loss = GraphEdgeMaskingLoss()._create_pytorch_loss()
        elif task == "infomax":
            self.infomax_loss = DeepGraphInfomaxLoss()._create_pytorch_loss()
        elif task == "context_pred":
            self.context_size = context_size
            self.neighborhood_size = neighborhood_size
            self.neg_samples = neg_samples
            self.context_mode = context_mode
            self.context_pred_loss = GraphContextPredLoss(
            )._create_pytorch_loss(context_mode, neg_samples)

        self.graph_pooling = graph_pooling
        self.dropout = dropout
        self.jump_knowledge = jump_knowledge
        self.task = task

        self.components = self.build_components()
        self.model = self.build_model()
        super().__init__(self.model, self.components, **kwargs)

    def build_components(self):
        """
        Builds the components of the GNNModular model. It initializes the encoders, batch normalization layers, pooling layers, and head layers based on the provided configuration. The method returns a dictionary containing the following components:

        Components list, type and description:
        --------------------------------------
        node_type_embedding: torch.nn.Embedding, an embedding layer for node types.

        chirality_embedding: torch.nn.Embedding, an embedding layer for chirality tags.

        gconvs: torch_geometric.nn.conv.MessagePassing, a list of graph convolutional layers (encoders) based on the specified GNN type (GIN, GCN, or GAT).

        batch_norms: torch.nn.BatchNorm1d, a list of batch normalization layers corresponding to the encoders.

        pool: Union[function,torch_geometric.nn.aggr.Aggregation], a pooling layer based on the specified graph pooling type (sum, mean, max, attention, or set2set).

        head: nn.Linear, a linear layer for the head of the model.

        These components are then used to construct the GNN and GNN_head modules for the GNNModular model.
        """

        encoders, batch_norms = self.build_gnn(self.num_layer)
        components = {
            'node_type_embedding':
                torch.nn.Embedding(num_node_type, self.emb_dim),
            'chirality_embedding':
                torch.nn.Embedding(num_chirality_tag, self.emb_dim),
            'gconvs':
                encoders,
            'batch_norms':
                batch_norms
        }
        self.gnn = GNN(components['node_type_embedding'],
                       components['chirality_embedding'], components['gconvs'],
                       components['batch_norms'], self.dropout,
                       self.jump_knowledge)

        if self.task in ("mask_nodes", "mask_edges"):
            linear_pred_nodes = torch.nn.Linear(self.emb_dim, num_node_type -
                                                1)  # -1 to remove mask token
            linear_pred_edges = torch.nn.Linear(self.emb_dim, num_edge_type -
                                                1)  # -1 to remove mask token
            components.update({
                'linear_pred_nodes': linear_pred_nodes,
                'linear_pred_edges': linear_pred_edges
            })

        # for supervised tasks, add prediction head
        elif self.task in ("regression", "classification"):
            if self.graph_pooling == "sum":
                pool = global_add_pool
            elif self.graph_pooling == "mean":
                pool = global_mean_pool
            elif self.graph_pooling == "max":
                pool = global_max_pool
            elif self.graph_pooling == "attention":
                if self.jump_knowledge == "concat":
                    pool = AttentionalAggregation(
                        gate_nn=torch.nn.Linear((self.num_layer + 1) *
                                                self.emb_dim, 1))
                else:
                    pool = AttentionalAggregation(
                        gate_nn=torch.nn.Linear(self.emb_dim, 1))
            elif self.graph_pooling == "set2set":
                set2setiter = 3
                if self.jump_knowledge == "concat":
                    pool = Set2Set((self.num_layer + 1) * self.emb_dim,
                                   set2setiter)
                else:
                    pool = Set2Set(self.emb_dim, processing_steps=set2setiter)

            if self.graph_pooling == "set2set":
                mult = 2
            else:
                mult = 1

            if self.jump_knowledge == "concat":
                head = torch.nn.Linear(
                    mult * (self.num_layer + 1) * self.emb_dim, self.output_dim)
            else:
                head = torch.nn.Linear(mult * self.emb_dim, self.output_dim)

            components.update({'pool': pool, 'head': head})

            self.gnn_head = GNNHead(components['pool'], components['head'],
                                    self.task, self.num_tasks, self.num_classes)

        elif self.task == 'infomax':
            descrim = LocalGlobalDiscriminator(self.emb_dim)
            components.update({
                'discriminator': descrim,
                'pool': global_mean_pool
            })

        elif self.task == 'context_pred':
            if self.graph_pooling == "sum":
                pool = global_add_pool
            elif self.graph_pooling == "mean":
                pool = global_mean_pool
            elif self.graph_pooling == "max":
                pool = global_max_pool
            elif self.graph_pooling == "attention":
                raise NotImplementedError(
                    "Attentional pooling not implemented for context prediction task."
                )
            elif self.graph_pooling == "set2set":
                raise NotImplementedError(
                    "Set2set pooling not implemented for context prediction task."
                )

            if self.jump_knowledge == "concat":  # concat changes the emb_dim
                c_gconvs, c_batch_norms = self.build_gnn(self.num_layer)
            else:
                c_gconvs, c_batch_norms = self.build_gnn(
                    self.neighborhood_size - self.context_size)
            context_gnn_components = {
                'c_node_type_embedding':
                    torch.nn.Embedding(num_node_type, self.emb_dim),
                'c_chirality_embedding':
                    torch.nn.Embedding(num_chirality_tag, self.emb_dim),
                'c_gconvs':
                    c_gconvs,
                'c_batch_norms':
                    c_batch_norms
            }

            self.context_gnn = GNN(
                context_gnn_components['c_node_type_embedding'],
                context_gnn_components['c_chirality_embedding'],
                context_gnn_components['c_gconvs'],
                context_gnn_components['c_batch_norms'], self.dropout,
                self.jump_knowledge)
            components.update({'pool': pool, **context_gnn_components})

        return components

    def build_gnn(self, num_layer):
        """
        Build graph neural network encoding layers by specifying the number of GNN layers.

        Parameters
        ----------
        num_layer : int
            The number of GNN layers to be created.

        Returns
        -------
        tuple of (torch.nn.ModuleList, torch.nn.ModuleList)
            A tuple containing two ModuleLists:
            1. encoders: A ModuleList of GNN layers (currently only GIN is supported).
            2. batch_norms: A ModuleList of batch normalization layers corresponding to each GNN layer.
        """

        encoders = []
        batch_norms = []
        for layer in range(num_layer):
            if self.gnn_type == "gin":
                encoders.append(
                    GINEConv(
                        torch.nn.Linear(self.emb_dim, self.emb_dim),
                        edge_dim=2,  # edge type, edge direction
                        aggr="add"))
            elif self.gnn_type == "gcn":
                encoders.append(GCNConv(self.emb_dim, self.emb_dim))
            elif self.gnn_type == "gat":
                encoders.append(GATConv(self.emb_dim, self.emb_dim))
            elif self.gnn_type == "sage":
                encoders.append(SAGEConv(self.emb_dim, self.emb_dim))
            else:
                raise ValueError("Unsuppported GNN type.")
            batch_norms.append(torch.nn.BatchNorm1d(self.emb_dim))
        encoders = torch.nn.ModuleList(encoders)
        batch_norms = torch.nn.ModuleList(batch_norms)

        return encoders, batch_norms

    def build_model(self):
        """
        Builds the appropriate model based on the specified task.

        For the edge prediction task, the model is simply the GNN module because it is an unsupervised task and does not require a prediction head.

        Supervised tasks such as node classification and graph regression require a prediction head, so the model is a sequential module consisting of the GNN module followed by the GNN_head module.
        """
        # unsupervised tasks do not need a pred head
        if self.task in ("edge_pred", "mask_nodes", "mask_edges", "infomax",
                         "context_pred"):
            return self.gnn
        elif self.task in ("regression", "classification"):
            return torch.nn.Sequential(self.gnn, self.gnn_head)
        else:
            raise ValueError(f"Task {self.task} is not supported.")

    def loss_func(self, inputs, labels, weights):
        """
        The loss function executed in the training loop, which is based on the specified task.
        """
        if self.task == "edge_pred":
            node_emb, inputs = self.model(inputs)
            loss = self.edge_pred_loss(node_emb, inputs)
        elif self.task == "mask_nodes":
            loss = self.masked_node_loss_loader(inputs)
        elif self.task == "mask_edges":
            loss = self.masked_edge_loss_loader(inputs)
        elif self.task == "infomax":
            loss = self.infomax_loss_loader(inputs)
        elif self.task == "regression":
            loss = self.regression_loss_loader(inputs, labels)
        elif self.task == "classification":
            loss = self.classification_loss_loader(inputs, labels)
        elif self.task == "context_pred":
            loss = self.context_pred_loss_loader(inputs)
        return (loss * weights).mean()

    def regression_loss_loader(self, inputs, labels):
        out = self.model(inputs)
        reg_loss = self.criterion(out, labels)
        return reg_loss

    def classification_loss_loader(self, inputs, labels):
        out = self.model(inputs)
        out = F.softmax(out, dim=2)
        class_loss = self.criterion(out, labels)
        return class_loss

    def masked_node_loss_loader(self, inputs):
        """
        Produces the loss between the predicted node features and the true node features for masked nodes.  Set mask_edge to True to also predict the edge types for masked edges.
        """

        node_emb, inputs = self.model(inputs)
        pred_node = self.components['linear_pred_nodes'](
            node_emb[inputs.masked_node_indices])
        if self.mask_edge:
            masked_edge_index = inputs.edge_index[:,
                                                  inputs.connected_edge_indices]
            edge_rep = node_emb[masked_edge_index[0]] + node_emb[
                masked_edge_index[1]]
            pred_edge = self.components['linear_pred_edges'](edge_rep)
        else:
            pred_edge = None
        return self.node_mask_loss(pred_node, pred_edge, inputs)

    def masked_edge_loss_loader(self, inputs):
        """
        Produces the loss between the predicted edge types and the true edge types for masked edges.
        """
        node_emb, inputs = self.model(inputs)
        masked_edge_index = inputs.edge_index[:, inputs.masked_edge_idx]
        edge_emb = node_emb[masked_edge_index[0]] + node_emb[
            masked_edge_index[1]]
        pred_edge = self.components['linear_pred_edges'](edge_emb)

        return self.edge_mask_loss(pred_edge, inputs)

    def infomax_loss_loader(self, inputs):
        """
        Loss that maximizes mutual information between local node representations and a pooled global graph representation. The positive and negative scores represent the similarity between local node representations and global graph representations of simlar and dissimilar graphs, respectively.

        Parameters
        ----------
        inputs: BatchedGraphData
            BatchedGraphData object containing the node features, edge indices, and graph indices for the batch of graphs.
        """
        node_emb, inputs = self.model(inputs)
        summary_emb = torch.sigmoid(self.components['pool'](node_emb,
                                                            inputs.graph_index))
        positive_expanded_summary_emb = summary_emb[inputs.graph_index]

        shifted_summary_emb = summary_emb[cycle_index(len(summary_emb), 1)]
        negative_expanded_summary_emb = shifted_summary_emb[inputs.graph_index]

        positive_score = self.components['discriminator'](
            node_emb, positive_expanded_summary_emb)
        negative_score = self.components['discriminator'](
            node_emb, negative_expanded_summary_emb)

        return self.infomax_loss(positive_score, negative_score)

    def context_pred_loss_loader(self, inputs):
        """
        Loads the context prediction loss for the given input by taking the batched subgraph and context graphs and computing the context prediction loss for each subgraph and context graph pair.

        Parameters
        ----------
        inputs : tuple
            A tuple containing the following elements:
            - substruct_batch (BatchedGraphData): Batched subgraph, or neighborhood, graphs.
            - s_overlap (List[int]): List of overlapping subgraph node indices between the subgraph and context graphs.
            - context_graphs (BatchedGraphData): Batched context graphs.
            - c_overlap (List[int]): List of overlapping context node indices between the subgraph and context graphs.
            - overlap_size (List[int]): List of the number of overlapping nodes between the subgraph and context graphs.

        Returns
        -------
        context_pred_loss : torch.Tensor
            The context prediction loss
        """
        substruct_batch = inputs[0]
        s_overlap = inputs[1]
        context_graphs = inputs[2]
        c_overlap = inputs[3]
        overlap_size = inputs[4]

        substruct_rep = self.gnn(substruct_batch)[0][
            s_overlap]  # 0 for node representation index
        overlapped_node_rep = self.context_gnn(context_graphs)[0][
            c_overlap]  # 0 for node representation index

        context_rep = self.components['pool'](overlapped_node_rep, c_overlap)
        # negative contexts are obtained by shifting the indicies of context embeddings
        neg_context_rep = torch.cat([
            context_rep[cycle_index(len(context_rep), i + 1)]
            for i in range(self.neg_samples)
        ],
                                    dim=0)

        context_pred_loss = self.context_pred_loss(substruct_rep,
                                                   overlapped_node_rep,
                                                   context_rep, neg_context_rep,
                                                   overlap_size)

        return context_pred_loss

    def _overlap_batcher(self, substruct_graphs, s_overlap, context_graphs,
                         c_overlap):
        """
        This method provides batching for the context prediction task.

        It handles the batching of the overlapping indicies between the subgraph and context graphs.

        Parameters
        ----------
        substruct_graphs: BatchedGraphData
            Batched subgraph, or neighborhood, graphs.
        s_overlap: List[List[int]]
            List of lists of overlapping subgraph node indicies between the subgraph and context graphs.
        context_graphs: BatchedGraphData
            Batched context graphs.
        c_overlap: List[List[int]]
            List of lists of overlapping context node indicies between the subgraph and context graphs.

        Returns
        -------
        flat_s_overlap: List[int]
            List of overlapping subgraph node indicies between the subgraph and context graphs.
        flat_c_overlap: List[int]
            List of overlapping context node indicies between the subgraph and context graphs.
        overlap_size: List[int]
            List of the number of overlapping nodes between the subgraph and context graphs.
        """
        cumsum_substruct = 0
        cumsum_context = 0

        for i, (sub, context) in enumerate(zip(substruct_graphs,
                                               context_graphs)):
            num_nodes_substruct = len(sub.node_features)
            num_nodes_context = len(context.node_features)

            s_overlap[i] = [s + cumsum_substruct for s in s_overlap[i]]
            c_overlap[i] = [c + cumsum_context for c in c_overlap[i]]

            cumsum_substruct += num_nodes_substruct
            cumsum_context += num_nodes_context

        flat_s_overlap = [item for sublist in s_overlap for item in sublist]
        flat_c_overlap = [item for sublist in c_overlap for item in sublist]
        overlap_size = [len(s) for s in c_overlap]
        return flat_s_overlap, flat_c_overlap, overlap_size

    def _prepare_batch(self, batch):
        """
        Prepares the batch for the model by converting the GraphData numpy arrays to BatchedGraphData torch tensors and moving them to the device, then transforming the input to the appropriate format for the task.

        Parameters
        ----------
        batch: tuple
            A tuple containing the inputs, labels, and weights for the batch.

        Returns
        -------
        inputs: BatchGraphData
            The inputs for the batch, converted to a BatchGraphData object, moved to the device, and transformed to the appropriate format for the task.
        labels: torch.Tensor
            The labels for the batch, moved to the device.
        weights: torch.Tensor
            The weights for the batch, moved to the device.
        """
        inputs, labels, weights = batch
        if self.task in ("regression", "classification", "infomax"):
            inputs = BatchGraphData(inputs[0]).numpy_to_torch(self.device)
        if self.task == "edge_pred":
            inputs = BatchGraphData(inputs[0]).numpy_to_torch(self.device)
            inputs = negative_edge_sampler(inputs)
        elif self.task == "mask_nodes":
            inputs = BatchGraphData(inputs[0]).numpy_to_torch(self.device)
            inputs = mask_nodes(inputs, self.mask_rate)
        elif self.task == "mask_edges":
            inputs = BatchGraphData(inputs[0]).numpy_to_torch(self.device)
            inputs = mask_edges(inputs, self.mask_rate)
        elif self.task == "context_pred":
            sampled_g = [
                context_pred_sampler(graph, self.context_size,
                                     self.neighborhood_size)
                for graph in inputs[0]
            ]
            try:
                subgraphs_list = [x[0] for x in sampled_g]
                s_overlap_list = [x[1] for x in sampled_g]
                context_list = [x[2] for x in sampled_g]
                c_overlap_list = [x[3] for x in sampled_g]
            except ValueError:
                raise ValueError(
                    "Not enough nodes in graph to sample context, use a smaller context or a larger neighborhood size."
                )

            s_overlap, c_overlap, overlap_size = self._overlap_batcher(
                subgraphs_list, s_overlap_list, context_list, c_overlap_list)
            s_overlap = torch.tensor(s_overlap).to(self.device)
            c_overlap = torch.tensor(c_overlap).to(self.device)

            b_subgraphs = BatchGraphData(subgraphs_list).numpy_to_torch(
                self.device)
            b_context = BatchGraphData(context_list).numpy_to_torch(self.device)
            inputs = (b_subgraphs, s_overlap, b_context, c_overlap,
                      overlap_size)

        _, labels, weights = super()._prepare_batch(([], labels, weights))

        if (len(labels) != 0) and (len(weights) != 0):
            labels = labels[0]
            weights = weights[0]

        return inputs, labels, weights

    def default_generator(
            self,
            dataset: Dataset,
            epochs: int = 1,
            mode: str = 'fit',
            deterministic: bool = True,
            pad_batches: bool = True) -> Iterable[Tuple[List, List, List]]:
        """
        This default generator is modified from the default generator in dc.models.tensorgraph.tensor_graph.py to support multitask classification. If the task is classification, the labels y_b are converted to a one-hot encoding and reshaped according to the number of tasks and classes.
        """

        for epoch in range(epochs):
            for (X_b, y_b, w_b,
                 ids_b) in dataset.iterbatches(batch_size=self.batch_size,
                                               deterministic=deterministic,
                                               pad_batches=pad_batches):
                if self.task == 'classification' and y_b is not None:
                    y_b = to_one_hot(y_b.flatten(), self.num_classes).reshape(
                        -1, self.num_tasks, self.num_classes)
                yield ([X_b], [y_b], [w_b])


def negative_edge_sampler(input_graph: BatchGraphData):
    """
    NegativeEdge is a function that adds negative edges to the input graph data. It randomly samples negative edges (edges that do not exist in the original graph) and adds them to the input graph data.
    The number of negative edges added is equal to half the number of edges in the original graph. This is useful for tasks like edge prediction, where the model needs to learn to differentiate between existing and non-existing edges.

    Parameters
    ----------
    input_graph: dc.feat.graph_data.BatchGraphData
        The input graph data.

    Returns
    -------
    BatchGraphData
        A new BatchGraphData object with the additional attribute `negative_edge_index`.

    Example
    -------
    >>> import numpy as np
    >>> import torch
    >>> from deepchem.feat.graph_data import BatchGraphData, GraphData
    >>> from deepchem.models.torch_models.gnn import negative_edge_sampler
    >>> num_nodes_list, num_edge_list = [3, 4, 5], [2, 4, 5]
    >>> num_node_features, num_edge_features = 32, 32
    >>> edge_index_list = [
    ...     np.array([[0, 1], [1, 2]]),
    ...     np.array([[0, 1, 2, 3], [1, 2, 0, 2]]),
    ...     np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]),
    ... ]
    >>> graph_list = [
    ...     GraphData(node_features=np.random.random_sample(
    ...         (num_nodes_list[i], num_node_features)),
    ...               edge_index=edge_index_list[i],
    ...               edge_features=np.random.random_sample(
    ...                   (num_edge_list[i], num_edge_features)),
    ...               node_pos_features=None) for i in range(len(num_edge_list))
    ... ]
    >>> batched_graph = BatchGraphData(graph_list)
    >>> batched_graph = batched_graph.numpy_to_torch()
    >>> neg_sampled = negative_edge_sampler(batched_graph)
    """
    data = copy.deepcopy(input_graph)

    num_nodes = data.num_nodes
    num_edges = data.num_edges

    edge_set = set([
        str(data.edge_index[0, i].cpu().item()) + "," +
        str(data.edge_index[1, i].cpu().item())
        for i in range(data.edge_index.shape[1])
    ])

    redandunt_sample = torch.randint(0, num_nodes, (2, 5 * num_edges))
    sampled_ind = []
    sampled_edge_set = set([])
    for i in range(5 * num_edges):
        node1 = redandunt_sample[0, i].cpu().item()
        node2 = redandunt_sample[1, i].cpu().item()
        edge_str = str(node1) + "," + str(node2)
        if edge_str not in edge_set and edge_str not in sampled_edge_set and not node1 == node2:
            sampled_edge_set.add(edge_str)
            sampled_ind.append(i)
        if len(sampled_ind) == num_edges / 2:
            break

    data.negative_edge_index = redandunt_sample[:, sampled_ind]  # type: ignore

    return data


def mask_nodes(input_graph: BatchGraphData,
               mask_rate,
               masked_node_indices=None,
               mask_edge=True):
    """
    Mask nodes and their connected edges in a BatchGraphData object.

    This function assumes that the first node feature is the atomic number, for example with the SNAPFeaturizer. It will set masked nodes' features to 0.

    Parameters
    ----------
    input_graph: dc.feat.BatchGraphData
        Assume that the edge ordering is the default PyTorch geometric ordering, where the two directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                    [1, 0, 2, 1, 3, 2]])
    masked_node_indices: list, optional
        If None, then randomly samples num_nodes * mask rate number of node indices. Otherwise, a list of node indices that sets the nodes to be masked (for debugging only).
    mask_edge: bool, optional
        Will mask the edges connected to the masked nodes.

    Returns
    -------
    data: dc.feat.BatchGraphData
        Creates new attributes in the original data object:
        - data.mask_node_idx
        - data.mask_node_label
        - data.mask_edge_idx
        - data.mask_edge_label

        """
    data = copy.deepcopy(input_graph)

    if masked_node_indices is None:
        # sample x distinct nodes to be masked, based on mask rate. But
        # will sample at least 1 node
        num_nodes = data.node_features.size()[0]  # type: ignore
        sample_size = int(num_nodes * mask_rate + 1)
        masked_node_indices = random.sample(range(num_nodes), sample_size)

    # create mask node label by copying node feature of mask node
    mask_node_labels_list = []
    for node_idx in masked_node_indices:
        mask_node_labels_list.append(data.node_features[node_idx].view(1, -1))
    data.mask_node_label = torch.cat(  # type: ignore
        mask_node_labels_list, dim=0)[:, 0].long()
    data.masked_node_indices = torch.tensor(masked_node_indices)  # type: ignore

    # modify the original node feature of the masked node
    num_node_feats = data.node_features.size()[1]  # type: ignore
    for node_idx in masked_node_indices:
        data.node_features[node_idx] = torch.zeros((1, num_node_feats))
    # zeros are meant to represent the masked features. This is distinct from the
    # original implementation, where the masked features are represented by the
    # the last feature token 119.
    # link to source: https://github.com/snap-stanford/pretrain-gnns/blob/08f126ac13623e551a396dd5e511d766f9d4f8ff/chem/util.py#L241

    if mask_edge:
        # create mask edge labels by copying edge features of edges that are connected to
        # mask nodes
        connected_edge_indices = []
        for edge_idx, (u, v) in enumerate(
                data.edge_index.cpu().numpy().T):  # type: ignore
            for node_idx in masked_node_indices:
                if node_idx in set(
                    (u, v)) and edge_idx not in connected_edge_indices:
                    connected_edge_indices.append(edge_idx)

        if len(connected_edge_indices) > 0:
            # create mask edge labels by copying edge features of the edges connected to
            # the mask nodes
            mask_edge_labels_list = []
            for edge_idx in connected_edge_indices[::2]:  # because the
                # edge ordering is such that two directions of a single
                # edge occur in pairs, so to get the unique undirected
                # edge indices, we take every 2nd edge index from list
                mask_edge_labels_list.append(
                    data.edge_features[edge_idx].view(  # type: ignore
                        1, -1))

            data.mask_edge_label = torch.cat(  # type: ignore
                mask_edge_labels_list, dim=0)[:, 0].long()  # type: ignore
            # modify the original edge features of the edges connected to the mask nodes
            num_edge_feat = data.edge_features.size()[1]  # type: ignore
            for edge_idx in connected_edge_indices:
                data.edge_features[edge_idx] = torch.zeros(  # type: ignore
                    (1, num_edge_feat))  # type: ignore
            # zeros are meant to represent the masked features. This is distinct from the
            # original implementation, where the masked features are represented by the
            # the last feature token 4.
            # link to source: https://github.com/snap-stanford/pretrain-gnns/blob/08f126ac13623e551a396dd5e511d766f9d4f8ff/chem/util.py#L268

            data.connected_edge_indices = torch.tensor(  # type: ignore
                connected_edge_indices[::2])
        else:
            data.mask_edge_label = torch.empty(  # type: ignore
                (0, 2)).to(torch.int64)
            data.connected_edge_indices = torch.tensor(  # type: ignore
                connected_edge_indices).to(torch.int64)

    return data


def mask_edges(input_graph: BatchGraphData,
               mask_rate: float,
               masked_edge_indices=None):
    """
    Mask edges in a BatchGraphData object.

    This is separate from the mask_nodes function because we want to be able to mask edges without masking any nodes.

    Parameters
    ----------
    input_graph: dc.feat.BatchGraphData
        Assume that the edge ordering is the default PyTorch geometric ordering, where the two directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                      [1, 0, 2, 1, 3, 2]])
    masked_edge_indices : list, optional
        If None, then randomly sample num_edges * mask_rate + 1 number of edge indices. Otherwise should correspond to the 1st direction of an edge pair. ie all indices should be an even number

    Returns
    -------
    data: dc.feat.BatchGraphData
        Creates new attributes in the original  object:
        - data.mask_edge_idx: indices of masked edges
        - data.mask_edge_labels: corresponding ground truth edge feature for each masked edge
        - data.edge_attr: modified in place: the edge features (both directions) that correspond to the masked edges have the masked edge feature
        """
    data = copy.deepcopy(input_graph)

    if masked_edge_indices is None:
        # sample x distinct edges to be masked, based on mask rate. But
        # will sample at least 1 edge
        num_edges = int(data.edge_index.size()[1] /  # type: ignore
                        2)  # num unique edges
        sample_size = int(num_edges * mask_rate + 1)
        # during sampling, we only pick the 1st direction of a particular
        # edge pair
        masked_edge_indices = [
            2 * i for i in random.sample(range(num_edges), sample_size)
        ]

    data.masked_edge_idx = torch.tensor(  # type: ignore
        np.array(masked_edge_indices))

    # create ground truth edge features for the edges that correspond to
    # the masked indices
    mask_edge_labels_list = []
    for idx in masked_edge_indices:
        mask_edge_labels_list.append(  # yapf: disable
            data.edge_features[idx].view(  # type: ignore
                1, -1))
    data.mask_edge_label = torch.cat(  # type: ignore
        mask_edge_labels_list, dim=0)

    # created new masked edge_attr, where both directions of the masked
    # edges have masked edge type. For message passing in gcn

    # append the 2nd direction of the masked edges
    all_masked_edge_indices = masked_edge_indices + [
        i + 1 for i in masked_edge_indices
    ]
    num_edge_feat = data.edge_features.size()[1]  # type: ignore
    for idx in all_masked_edge_indices:
        data.edge_features[idx] = torch.zeros(  # type: ignore
            (1, num_edge_feat))
    # zeros are meant to represent the masked features. This is distinct from the
    # original implementation, where the masked features are represented by 0s and
    # an additional mask feature
    # link to source: https://github.com/snap-stanford/pretrain-gnns/blob/08f126ac13623e551a396dd5e511d766f9d4f8ff/bio/util.py#L101

    return data


def cycle_index(num, shift):
    """
    Creates a 1-dimensional tensor of integers with a specified length (`num`) and a cyclic shift (`shift`). The tensor starts with integers from `shift` to `num - 1`, and then wraps around to include integers from `0` to `shift - 1` at the end.

    Parameters
    ----------
    num: int
        Length of the tensor.
    shift: int
        Amount to shift the tensor by.

    Example
    -------
    >>> num = 10
    >>> shift = 3
    >>> arr = cycle_index(num, shift)
    >>> print(arr)
    tensor([3, 4, 5, 6, 7, 8, 9, 0, 1, 2])
    """
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr


def context_pred_sampler(input_graph: GraphData,
                         l1: int,
                         l2: int,
                         root_idx=None):
    """
    Generate subgraph and context graph for context prediction.

    This function takes an input graph and generates a subgraph and a context graph for context prediction. The subgraph is the entire input graph, while the context graph is a ring around the root node outside of l1 and inside of l2.

    Parameters
    ----------
    input_graph : GraphData
        The input graph for which the subgraph and context graph are to be generated.
    l1 : int
        The inner diameter of the context graph.
    l2 : int
        The outer diameter of the context graph.
    root_idx : int, optional
        The index of the root node. If not provided, a random node will be selected as the root.

    Returns
    -------
    subgraph : GraphData
        The subgraph generated from the input graph.
    s_overlap : list
        The indices of overlapping nodes between substruct and context, with respect to the substruct ordering.
    context_G : GraphData
        The context graph generated from the input graph.
    c_overlap : list
        The indices of overlapping nodes between substruct and context, with respect to the context ordering.

    Examples
    --------
    >>> import numpy as np
    >>> from deepchem.feat.graph_data import GraphData
    >>> from deepchem.models.torch_models.gnn import context_pred_sampler
    >>> x = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1], [1, 1], [0, 0]])
    >>> edge_index = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [1, 0, 3, 2, 5, 4, 7, 6]])
    >>> edge_feats = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1], [1, 1], [0, 0]])
    >>> graph_index = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    >>> data = GraphData(node_features=x,
    ...                  edge_index=edge_index,
    ...                  edge_features=edge_feats,
    ...                  graph_index=graph_index)
    >>> l1 = 1
    >>> l2 = 2
    >>> subgraph, s_overlap, context_G, c_overlap = context_pred_sampler(data, l1, l2)
    """
    data = copy.deepcopy(input_graph)

    root_idx = random.sample(range(data.num_nodes), 1)[0]

    # Take the entire graph, but can be modified to take a subgraph of k-hops from the root node
    x_substruct = data.node_features
    edge_attr_substruct = data.edge_features
    edge_index_substruct = data.edge_index

    subgraph = GraphData(node_features=x_substruct,
                         edge_features=edge_attr_substruct,
                         edge_index=edge_index_substruct)

    # Get node idx between root and the inner diameter l1
    l1_node_idxes = shortest_path_length(data, root_idx, l1).keys()
    # Get node idx between root and the outer diameter l2
    l2_node_idxes = shortest_path_length(data, root_idx, l2).keys()

    # takes a ring around the root node outside of l1 and inside of l2
    context_node_idxes = set(l1_node_idxes).symmetric_difference(
        set(l2_node_idxes))

    if len(context_node_idxes) > 0:
        context_G, context_node_map = data.subgraph(context_node_idxes)
        # Get indices of overlapping nodes between substruct and context, WRT context ordering
        s_overlap = list(context_node_idxes)
        c_overlap = [context_node_map[old_idx] for old_idx in s_overlap]
        return (subgraph, s_overlap, context_G, c_overlap)
    else:
        return (subgraph, [], None, [])

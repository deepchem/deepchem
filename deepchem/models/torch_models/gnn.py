import torch
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn.aggr import AttentionalAggregation, Set2Set
from torch.functional import F
from deepchem.models.torch_models import ModularTorchModel
from deepchem.feat.graph_data import BatchGraphData

num_atom_type = 120
num_chirality_tag = 3
# Relevant in future PRs
# num_bond_type = 6
# num_bond_direction = 3


class GNN(torch.nn.Module):
    """
    GNN module for the GNNModular model.

    This module is responsible for the graph neural network layers in the GNNModular model.

    Parameters
    ----------
    atom_type_embedding: torch.nn.Embedding
        Embedding layer for atom types.
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
    >>> batched_graph = BatchGraphData(features).numpy_to_torch(device="cuda")
    >>> modular = model = GNNModular("gin", 3, 64, 1, "attention", 0, "last", "edge_pred")
    >>> gnnmodel = modular.gnn
    >>> print(gnnmodel(batched_graph)[0].shape)
    torch.Size([23, 64])

    """

    def __init__(self,
                 atom_type_embedding,
                 chirality_embedding,
                 gconvs,
                 batch_norms,
                 dropout,
                 jump_knowledge,
                 init_emb=False):
        super(GNN, self).__init__()

        self.atom_type_embedding = atom_type_embedding
        self.chirality_embedding = chirality_embedding
        self.gconv = gconvs
        self.batch_norms = batch_norms
        self.dropout = dropout
        self.num_layer = len(gconvs)
        self.jump_knowledge = jump_knowledge

        # may mess with loading pretrained weights
        if init_emb:
            torch.nn.init.xavier_uniform_(self.atom_type_embedding.weight.data)
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
        x = self.atom_type_embedding(node_feats) + self.chirality_embedding(
            chiral_feats)

        h_list = [x]
        for i, conv_layer in enumerate(self.gconv):
            h = conv_layer(h_list[i], data.edge_index, data.edge_features)
            h = self.batch_norms[i](h)
            h = F.dropout(F.relu(h), self.dropout, training=self.training)
            if i == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.dropout, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training=self.training)
            h_list.append(h)

        # Different implementations of JK
        if self.jump_knowledge == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.jump_knowledge == "last":
            node_representation = h_list[-1]
        elif self.jump_knowledge == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.jump_knowledge == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        return node_representation, data


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

    def __init__(self, pool, head):
        super().__init__()
        self.pool = pool
        self.head = head

    def forward(self, node_representation, data):
        """
        Forward pass for the GNN head module.

        Parameters
        ----------
        node_representation: torch.Tensor
            The node representations after passing through the GNN layers.
        data: BatchGraphData
            The original input graph data.
        """

        pooled = self.pool(node_representation, data.graph_index)
        out = self.head(pooled)
        return out


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
    dropout: float, optional (default 0)
        The dropout probability.
    jump_knowledge: str, optional (default "last")
        The type of jump knowledge to use. [1] Must be one of "last", "sum", "max", "concat" or "none".
        "last": Use the node representation from the last GNN layer.
        "concat": Concatenate the node representations from all GNN layers.
        "max": Take the element-wise maximum of the node representations from all GNN layers.
        "sum": Take the element-wise sum of the node representations from all GNN layers.
    task: str, optional (default "regression")
        The type of task. Can be unsupervised tasks "edge_pred" or "node_pred" or supervised tasks like "regression" or "classification".

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
                 graph_pooling: str = "attention",
                 dropout: int = 0,
                 jump_knowledge: str = "concat",
                 task: str = "edge_pred",
                 **kwargs):
        self.gnn_type = gnn_type
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        self.dropout = dropout
        self.jump_knowledge = jump_knowledge
        self.task = task
        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.components = self.build_components()
        self.model = self.build_model()
        super().__init__(self.model, self.components, **kwargs)

    def build_components(self):
        """
        Builds the components of the GNNModular model. It initializes the encoders, batch normalization layers, pooling layers, and head layers based on the provided configuration. The method returns a dictionary containing the following components:

        Components list, type and description:
        --------------------------------------
        atom_type_embedding: torch.nn.Embedding, an embedding layer for atom types.

        chirality_embedding: torch.nn.Embedding, an embedding layer for chirality tags.

        gconvs: torch_geometric.nn.conv.MessagePassing, a list of graph convolutional layers (encoders) based on the specified GNN type (GIN, GCN, or GAT).

        batch_norms: torch.nn.BatchNorm1d, a list of batch normalization layers corresponding to the encoders.

        pool: Union[function,torch_geometric.nn.aggr.Aggregation], a pooling layer based on the specified graph pooling type (sum, mean, max, attention, or set2set).

        head: nn.Linear, a linear layer for the head of the model.

        These components are then used to construct the GNN and GNN_head modules for the GNNModular model.
        """
        encoders = []
        batch_norms = []
        for layer in range(self.num_layer):
            if self.gnn_type == "gin":
                encoders.append(
                    GINEConv(
                        torch.nn.Linear(self.emb_dim, self.emb_dim),
                        edge_dim=2,  # bond type, bond direction
                        aggr="add"))
            else:
                raise ValueError("Only GIN is supported for now")
            # Relevent for future PRs
            # elif self.gnn_type == "gcn":
            #     encoders.append(GCNConv(self.emb_dim))
            # elif self.gnn_type == "gat":
            #     encoders.append(GATConv(self.emb_dim))
            batch_norms.append(torch.nn.BatchNorm1d(self.emb_dim))
        encoders = torch.nn.ModuleList(encoders)
        batch_norms = torch.nn.ModuleList(batch_norms)

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
                pool = Set2Set((self.num_layer + 1) * self.emb_dim, set2setiter)
            else:
                pool = Set2Set(self.emb_dim, processing_steps=set2setiter)

        if self.graph_pooling == "set2set":
            mult = 2
        else:
            mult = 1

        if self.jump_knowledge == "concat":
            head = torch.nn.Linear(mult * (self.num_layer + 1) * self.emb_dim,
                                   self.num_tasks)
        else:
            head = torch.nn.Linear(mult * self.emb_dim, self.num_tasks)

        components = {
            'atom_type_embedding':
                torch.nn.Embedding(num_atom_type, self.emb_dim),
            'chirality_embedding':
                torch.nn.Embedding(num_chirality_tag, self.emb_dim),
            'gconvs':
                encoders,
            'batch_norms':
                batch_norms,
            'pool':
                pool,
            'head':
                head
        }
        self.gnn = GNN(components['atom_type_embedding'],
                       components['chirality_embedding'], components['gconvs'],
                       components['batch_norms'], self.dropout,
                       self.jump_knowledge)
        self.gnn_head = GNNHead(components['pool'], components['head'])
        return components

    def build_model(self):
        """
        Builds the appropriate model based on the specified task.

        For the edge prediction task, the model is simply the GNN module because it is an unsupervised task and does not require a prediction head.

        Supervised tasks such as node classification and graph regression require a prediction head, so the model is a sequential module consisting of the GNN module followed by the GNN_head module.
        """

        if self.task == "edge_pred":  # unsupervised task, does not need pred head
            return self.gnn
        else:
            return torch.nn.Sequential(self.gnn, self.gnn_head)

    def loss_func(self, inputs, labels, weights):
        """
        The loss function executed in the training loop, which is based on the specified task.
        """
        if self.task == "edge_pred":
            return self.edge_pred_loss(inputs, labels, weights)

    def edge_pred_loss(self, inputs, labels, weights):
        """
        The loss function for the graph edge prediction task.

        The inputs in this loss must be a BatchGraphData object transformed by the NegativeEdge molecule feature utility.
        """
        node_emb, _ = self.model(
            inputs)  # node_emb shape == [num_nodes x emb_dim]

        positive_score = torch.sum(node_emb[inputs.edge_index[0, ::2]] *
                                   node_emb[inputs.edge_index[1, ::2]],
                                   dim=1)
        negative_score = torch.sum(node_emb[inputs.negative_edge_index[0]] *
                                   node_emb[inputs.negative_edge_index[1]],
                                   dim=1)

        loss = self.criterion(
            positive_score, torch.ones_like(positive_score)) + self.criterion(
                negative_score, torch.zeros_like(negative_score))
        return (loss * weights[0]).mean()

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
        inputs = BatchGraphData(inputs[0]).numpy_to_torch(self.device)
        if self.task == "edge_pred":
            inputs = negative_edge_sampler(inputs)

        _, labels, weights = super()._prepare_batch(([], labels, weights))

        if (len(labels) != 0) and (len(weights) != 0):
            labels = labels[0]
            weights = weights[0]

        return inputs, labels, weights


def negative_edge_sampler(data: BatchGraphData):
    """
    NegativeEdge is a function that adds negative edges to the input graph data. It randomly samples negative edges (edges that do not exist in the original graph) and adds them to the input graph data.
    The number of negative edges added is equal to half the number of edges in the original graph. This is useful for tasks like edge prediction, where the model needs to learn to differentiate between existing and non-existing edges.

    Parameters
    ----------
    data: dc.feat.graph_data.BatchGraphData
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
    import torch

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

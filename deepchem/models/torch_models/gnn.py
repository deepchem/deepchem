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
                 JK,
                 init_emb=False):
        super(GNN, self).__init__()

        self.atom_type_embedding = atom_type_embedding
        self.chirality_embedding = chirality_embedding
        self.gconv = gconvs
        self.batch_norms = batch_norms
        self.dropout = dropout
        self.num_layer = len(gconvs)
        self.JK = JK

        # may mess with loading pretrained weights
        if init_emb:
            torch.nn.init.xavier_uniform_(self.atom_type_embedding.weight.data)
            torch.nn.init.xavier_uniform_(self.chirality_embedding.weight.data)

    def forward(self, data: BatchGraphData):
        """
        Forward pass for the GNN module.
        """

        x = self.atom_type_embedding(  # type: ignore
            data.node_features[:, 0].long()) + self.chirality_embedding(
                data.node_features[:, 1].long())

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
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        return node_representation, data


class GNN_head(torch.nn.Module):
    """
    Forward pass for the GNN head module.

    Parameters
    ----------
    node_representation: torch.Tensor
        The node representations after passing through the GNN layers.
    data: BatchGraphData
        The input graph data.

    Returns
    -------
    out: torch.Tensor
        The output of the GNN head module.
    """

    def __init__(self, pool, head):
        super().__init__()
        self.pool = pool
        self.head = head

    def forward(self, node_representation, data):
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
    JK: str, optional (default "last")
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
                 JK: str = "concat",
                 task: str = "edge_pred",
                 **kwargs):
        self.gnn_type = gnn_type
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        self.dropout = dropout
        self.JK = JK
        self.task = task
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.neg_edges = NegativeEdge()

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
            if self.JK == "concat":
                pool = AttentionalAggregation(
                    gate_nn=torch.nn.Linear((self.num_layer + 1) *
                                            self.emb_dim, 1))
            else:
                pool = AttentionalAggregation(
                    gate_nn=torch.nn.Linear(self.emb_dim, 1))
        elif self.graph_pooling == "set2set":
            set2setiter = 3
            if self.JK == "concat":
                pool = Set2Set((self.num_layer + 1) * self.emb_dim, set2setiter)
            else:
                pool = Set2Set(self.emb_dim, processing_steps=set2setiter)

        if self.graph_pooling == "set2set":
            mult = 2
        else:
            mult = 1

        if self.JK == "concat":
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
                       components['batch_norms'], self.dropout, self.JK)
        self.gnn_head = GNN_head(components['pool'], components['head'])
        return components

    def build_model(self):
        if self.task == "edge_pred":  # unsupervised task, does not need pred head
            return self.gnn
        else:
            return torch.nn.Sequential(self.gnn, self.gnn_head)

    def loss_func(self, inputs, labels, weights):
        if self.task == "edge_pred":
            return self.edge_pred_loss(inputs, labels, weights)

    def edge_pred_loss(self, inputs, labels, weights):
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
        Prepares the batch for the model by converting the GraphData numpy arrays to torch tensors and moving them to the device.
        """
        inputs, labels, weights = batch
        inputs = BatchGraphData(inputs[0]).numpy_to_torch(self.device)
        inputs = self.neg_edges(inputs)

        _, labels, weights = super()._prepare_batch(([], labels, weights))

        if (len(labels) != 0) and (len(weights) != 0):
            labels = labels[0]
            weights = weights[0]

        return inputs, labels, weights


class NegativeEdge:
    """
    NegativeEdge is a callable class that adds negative edges to the input graph data. It randomly samples negative edges (edges that do not exist in the original graph) and adds them to the input graph data.
    The number of negative edges added is equal to half the number of edges in the original graph. This is useful for tasks like edge prediction, where the model needs to learn to differentiate between existing and non-existing edges.
    """

    def __call__(self, data):
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

        data.negative_edge_index = redandunt_sample[:, sampled_ind]

        return data

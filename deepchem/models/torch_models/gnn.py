import torch
from torch_geometric.nn import GINConv, GCNConv, GATConv, global_add_pool, global_mean_pool, global_max_pool, Set2Set
from torch_geometric.nn.aggr import AttentionalAggregation
from torch.functional import F
from deepchem.models.torch_models import ModularTorchModel
from torch_geometric.data import Data
# from torch_scatter import scatter_add
# from torch_geometric.nn.inits import glorot, zeros

# from torch_geometric.nn.conv import CuGraphSAGEConv
# from torch_geometric.nn.models import GraphSAGE

num_atom_type = 120  # including the extra mask tokens
num_chirality_tag = 3
num_bond_type = 6
num_bond_direction = 3


class GNN(torch.nn.Module):
    """

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """

    def __init__(self,
                 num_layer,
                 emb_dim,
                 JK="last",
                 drop_ratio=0,
                 gnn_type="gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            # elif gnn_type == "graphsage":
            #     self.gnns.append(GraphSAGEConv(emb_dim))

        # List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h),
                              self.drop_ratio,
                              training=self.training)
            h_list.append(h)

        # Different implementations of Jk-concat
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

        return node_representation


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


    """

    def __init__(self, gnn_type, num_layer, emb_dim, num_tasks, graph_pooling,
                 JK, **kwargs):
        self.gnn_type = gnn_type
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        self.JK = JK
        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.components = self.build_components()
        self.model = self.build_model()
        super().__init__(self.model, self.components, **kwargs)

    def build_components(self):
        encoders = []
        batch_norms = []
        for layer in range(self.num_layer):
            if self.gnn_type == "gin":
                self.encoders.append(GINConv(self.emb_dim, aggr="add"))
            elif self.gnn_type == "gcn":
                self.encoders.append(GCNConv(self.emb_dim))
            elif self.gnn_type == "gat":
                self.encoders.append(GATConv(self.emb_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(self.emb_dim))

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

        return {
            'atom_type_embedding':
                torch.nn.Embedding(num_atom_type, self.emb_dim),
            'chirality_embedding':
                torch.nn.Embedding(num_chirality_tag, self.emb_dim),
            'gnn':
                encoders,
            'batch_norms':
                batch_norms,
            'pool':
                pool,
            'head':
                head
        }

    def build_model(self):
        return GNN(**self.components)

    def loss_func(self, inputs, labels, weights):
        #edge pred
        batch = inputs.to(self.device)
        node_emb = model(batch.x, batch.edge_index, batch.edge_attr)  #self.gnn

        positive_score = torch.sum(node_emb[batch.edge_index[0, ::2]] *
                                   node_emb[batch.edge_index[1, ::2]],
                                   dim=1)
        negative_score = torch.sum(node_emb[batch.negative_edge_index[0]] *
                                   node_emb[batch.negative_edge_index[1]],
                                   dim=1)

        optimizer.zero_grad()
        loss = criterion(positive_score,
                         torch.ones_like(positive_score)) + criterion(
                             negative_score, torch.zeros_like(negative_score))
        return (loss * weights[0]).mean()

    def default_generator(self, dataset, epochs=1, **kwargs):
        return DataLoaderAE(dataset, batch_size=32, shuffle=True)


class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat
        
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """

    def __init__(self,
                 num_layer,
                 emb_dim,
                 num_tasks,
                 JK="last",
                 drop_ratio=0,
                 graph_pooling="mean",
                 gnn_type="gin"):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type)

        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(
                    gate_nn=torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim,
                                    set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1

        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(
                self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim,
                                                     self.num_tasks)

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)

        return self.graph_pred_linear(self.pool(node_representation, batch))


# edge pred mode training code
criterion = torch.nn.BCEWithLogitsLoss()


def edge_pred(args, model, device, loader, optimizer):
    model.train()

    train_acc_accum = 0
    train_loss_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        node_emb = model(batch.x, batch.edge_index, batch.edge_attr)

        positive_score = torch.sum(node_emb[batch.edge_index[0, ::2]] *
                                   node_emb[batch.edge_index[1, ::2]],
                                   dim=1)
        negative_score = torch.sum(node_emb[batch.negative_edge_index[0]] *
                                   node_emb[batch.negative_edge_index[1]],
                                   dim=1)

        optimizer.zero_grad()
        loss = criterion(positive_score,
                         torch.ones_like(positive_score)) + criterion(
                             negative_score, torch.zeros_like(negative_score))
        loss.backward()
        optimizer.step()

        train_loss_accum += float(loss.detach().cpu().item())
        acc = (torch.sum(positive_score > 0) + torch.sum(negative_score < 0)
               ).to(torch.float32) / float(2 * len(positive_score))
        train_acc_accum += float(acc.detach().cpu().item())

    return train_acc_accum / step, train_loss_accum / step


class DataLoaderAE(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderAE, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchAE.from_data_list(data_list),
            **kwargs)


class BatchAE(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchAE, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = BatchAE()

        for key in keys:
            batch[key] = []
        batch.batch = []

        cumsum_node = 0

        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes,), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if key in ['edge_index', 'negative_edge_index']:
                    item = item + cumsum_node
                batch[key].append(item)

            cumsum_node += num_nodes

        for key in keys:
            batch[key] = torch.cat(batch[key], dim=batch.cat_dim(key))
        batch.batch = torch.cat(batch.batch, dim=-1)
        return batch.contiguous()

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1

    def cat_dim(self, key):
        return -1 if key in ["edge_index", "negative_edge_index"] else 0
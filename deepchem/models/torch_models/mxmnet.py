import math
import deepchem as dc
import torch
import torch.nn as nn
from deepchem.data import Dataset
from torch_geometric.data import Batch
from deepchem.models.torch_models import TorchModel
from deepchem.models.losses import Loss, L1Loss
from torch_geometric.nn import global_add_pool, radius
from torch_geometric.utils import remove_self_loops
from torch_sparse import SparseTensor
from typing import List, Iterable, Tuple
from deepchem.utils.pytorch_utils import get_activation


class MXMNet(nn.Module):

    def __init__(self,
                 dim,
                 n_layer,
                 cutoff,
                 num_spherical=7,
                 num_radial=6,
                 envelope_exponent=5,
                 activation_fn='silu',
                 n_tasks=1):
        super(MXMNet, self).__init__()

        self.dim = dim
        self.n_layer = n_layer
        self.cutoff = cutoff
        activation_fn = get_activation(activation_fn)

        self.embeddings = nn.Parameter(torch.ones((5, self.dim)))

        self.rbf_l = dc.models.torch_models.layers.MXMNetBesselBasisLayer(
            16, 5, envelope_exponent)
        self.rbf_g = dc.models.torch_models.layers.MXMNetBesselBasisLayer(
            16, self.cutoff, envelope_exponent)
        self.sbf = dc.models.torch_models.layers.MXMNetSphericalBasisLayer(
            num_spherical, num_radial, 5, envelope_exponent)

        self.rbf_g_mlp = dc.models.torch_models.layers.MultilayerPerceptron(
            d_input=16, d_output=self.dim, activation_fn=activation_fn)
        self.rbf_l_mlp = dc.models.torch_models.layers.MultilayerPerceptron(
            d_input=16, d_output=self.dim, activation_fn=activation_fn)

        self.sbf_1_mlp = dc.models.torch_models.layers.MultilayerPerceptron(
            d_input=num_spherical * num_radial,
            d_output=self.dim,
            activation_fn=activation_fn)
        self.sbf_2_mlp = dc.models.torch_models.layers.MultilayerPerceptron(
            d_input=num_spherical * num_radial,
            d_output=self.dim,
            activation_fn=activation_fn)

        self.global_layers = torch.nn.ModuleList()
        for layer in range(self.n_layer):
            self.global_layers.append(
                dc.models.torch_models.layers.MXMNetGlobalMessagePassing(
                    self.dim))

        self.local_layers = torch.nn.ModuleList()
        for layer in range(self.n_layer):
            self.local_layers.append(
                dc.models.torch_models.layers.MXMNetLocalMessagePassing(
                    self.dim))

        self.init()

    def init(self):
        stdv = math.sqrt(3)
        self.embeddings.data.uniform_(-stdv, stdv)

    def indices(self, edge_index, num_nodes):
        row, col = edge_index

        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(row=col,
                             col=row,
                             value=value,
                             sparse_sizes=(num_nodes, num_nodes))

        # Compute the node indices for two-hop angles
        adj_t_row = adj_t[row]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()
        mask = idx_i != idx_k
        idx_i_1, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

        idx_kj = adj_t_row.storage.value()[mask]
        idx_ji_1 = adj_t_row.storage.row()[mask]

        # Compute the node indices for one-hop angles
        adj_t_col = adj_t[col]

        num_pairs = adj_t_col.set_value(None).sum(dim=1).to(torch.long)
        idx_i_2 = row.repeat_interleave(num_pairs)
        idx_j1 = col.repeat_interleave(num_pairs)
        idx_j2 = adj_t_col.storage.col()

        idx_ji_2 = adj_t_col.storage.row()
        idx_jj = adj_t_col.storage.value()

        return idx_i_1, idx_j, idx_k, idx_kj, idx_ji_1, idx_i_2, idx_j1, idx_j2, idx_jj, idx_ji_2

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        pos = data.pos
        batch = data.batch
        x = torch.reshape(x, (-1,))
        # Initialize node embeddings
        h = torch.index_select(self.embeddings, 0, x.long())

        # Get the edges and pairwise distances in the local layer
        edge_index_l, _ = remove_self_loops(edge_index)
        j_l, i_l = edge_index_l
        dist_l = (pos[i_l] - pos[j_l]).pow(2).sum(dim=-1).sqrt()

        # Get the edges pairwise distances in the global layer
        row, col = radius(pos,
                          pos,
                          self.cutoff,
                          batch,
                          batch,
                          max_num_neighbors=500)
        edge_index_g = torch.stack([row, col], dim=0)
        edge_index_g, _ = remove_self_loops(edge_index_g)
        j_g, i_g = edge_index_g
        dist_g = (pos[i_g] - pos[j_g]).pow(2).sum(dim=-1).sqrt()

        # Compute the node indices for defining the angles
        idx_i_1, idx_j, idx_k, idx_kj, idx_ji, idx_i_2, idx_j1, idx_j2, idx_jj, idx_ji_2 = self.indices(
            edge_index_l, num_nodes=h.size(0))

        # Compute the two-hop angles
        pos_ji_1, pos_kj = pos[idx_j] - pos[idx_i_1], pos[idx_k] - pos[idx_j]
        a = (pos_ji_1 * pos_kj).sum(dim=-1)
        b = torch.cross(pos_ji_1, pos_kj).norm(dim=-1)
        angle_1 = torch.atan2(b, a)

        # Compute the one-hop angles
        pos_ji_2, pos_jj = pos[idx_j1] - pos[idx_i_2], pos[idx_j2] - pos[idx_j1]
        a = (pos_ji_2 * pos_jj).sum(dim=-1)
        b = torch.cross(pos_ji_2, pos_jj).norm(dim=-1)
        angle_2 = torch.atan2(b, a)

        # Get the RBF and SBF embeddings
        rbf_g = self.rbf_g(dist_g)
        rbf_l = self.rbf_l(dist_l)
        sbf_1 = self.sbf(dist_l, angle_1, idx_kj)
        sbf_2 = self.sbf(dist_l, angle_2, idx_jj)

        rbf_g = self.rbf_g_mlp(rbf_g)
        rbf_l = self.rbf_l_mlp(rbf_l)
        sbf_1 = self.sbf_1_mlp(sbf_1)
        sbf_2 = self.sbf_2_mlp(sbf_2)

        # Perform the message passing schemes
        node_sum = 0

        for layer in range(self.n_layer):
            h = self.global_layers[layer](h, rbf_g, edge_index_g)
            h, t = self.local_layers[layer](h, rbf_l, sbf_1, sbf_2, idx_kj,
                                            idx_ji, idx_jj, idx_ji_2,
                                            edge_index_l)
            node_sum += t

        # Readout
        output = global_add_pool(node_sum, batch)
        return output.view(-1)


class MXMNetModel(TorchModel):

    def __init__(self,
                 dim: int,
                 n_layer: int,
                 cutoff: int,
                 batch_size: int = 2,
                 mode: str = 'regression',
                 n_tasks: int = 1,
                 activation_fn: str = 'silu',
                 num_spherical: int = 7,
                 num_radial: int = 6,
                 envelope_exponent: int = 5,
                 **kwargs):

        model: nn.Module = MXMNet(activation_fn=activation_fn,
                                  dim=dim,
                                  n_layer=n_layer,
                                  n_tasks=n_tasks,
                                  cutoff=cutoff,
                                  num_spherical=num_spherical,
                                  num_radial=num_radial,
                                  envelope_exponent=envelope_exponent)

        if mode == 'regression':
            loss: Loss = L1Loss()
            output_types: List[str] = ['prediction']

        super(MXMNetModel, self).__init__(model,
                                          loss=loss,
                                          output_types=output_types,
                                          batch_size=batch_size,
                                          **kwargs)

    def _prepare_batch(
        self, batch: Tuple[List, List, List]
    ) -> Tuple[Batch, List[torch.Tensor], List[torch.Tensor]]:
        """Method to prepare pytorch-geometric batches from inputs.

        Overrides the existing ``_prepare_batch`` method to customize how model batches are
        generated from the inputs.

        .. note::
            This method requires PyTorch Geometric to be installed.

        Parameters
        ----------
        batch: Tuple[List, List, List]
            batch data from ``default_generator``

        Returns
        -------
        Tuple[Batch, List[torch.Tensor], List[torch.Tensor]]
        """
        graphs_list: List
        labels: List
        weights: List

        graphs_list, labels, weights = batch
        pyg_batch: Batch = Batch()
        pyg_batch = pyg_batch.from_data_list(graphs_list)

        _, labels, weights = super(MXMNetModel, self)._prepare_batch(
            ([], labels, weights))
        return pyg_batch, labels, weights

    def default_generator(self,
                          dataset: Dataset,
                          epochs: int = 1,
                          mode: str = 'fit',
                          deterministic: bool = True,
                          pad_batches: bool = False,
                          **kwargs) -> Iterable[Tuple[List, List, List]]:
        """Create a generator that iterates batches for a dataset.

        Overrides the existing ``default_generator`` method to customize how model inputs are
        generated from the data.

        Then data from each molecule is converted to a ``_ModData`` object and stored as list of graphs.
        The graphs are modified such that all tensors have same size in 0th dimension. (important requirement for batching)

        Parameters
        ----------
        dataset: Dataset
            the data to iterate
        epochs: int
            the number of times to iterate over the full dataset
        mode: str
            allowed values are 'fit' (called during training), 'predict' (called
            during prediction), and 'uncertainty' (called during uncertainty
            prediction)
        deterministic: bool
            whether to iterate over the dataset in order, or randomly shuffle the
            data for each epoch
        pad_batches: bool
            whether to pad each batch up to this model's preferred batch size

        Returns
        -------
        a generator that iterates batches, each represented as a tuple of lists:
        ([inputs], [outputs], [weights])
        Here, [inputs] is list of graphs.
        """
        for epoch in range(epochs):
            for (X_b, y_b, w_b,
                 ids_b) in dataset.iterbatches(batch_size=self.batch_size,
                                               deterministic=deterministic,
                                               pad_batches=pad_batches):
                pyg_graphs_list: List = []

                for graph in X_b:
                    # generate concatenated feature vector and mapping
                    pyg_graph = graph.to_pyg_graph()
                    pyg_graphs_list.append(pyg_graph)

                yield (pyg_graphs_list, [y_b], [w_b])

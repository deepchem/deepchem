import math
import deepchem as dc
import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, radius
from torch_geometric.utils import remove_self_loops
from torch_sparse import SparseTensor
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

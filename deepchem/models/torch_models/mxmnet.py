import math
import deepchem as dc
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool, radius
from torch_geometric.utils import remove_self_loops
from typing import Union, Callable, Tuple, Optional
from deepchem.utils.pytorch_utils import get_activation
from deepchem.models.torch_models.layers import MXMNetBesselBasisLayer, MXMNetSphericalBasisLayer, MultilayerPerceptron


class MXMNet(nn.Module):
    """
    Molecular Mechanics-Driven Graph Neural Network with Multiplex Graph for Molecular Structures.
    In this class, we define the global and local message passing layers and the MXMNet Model[1]_.
    We also define the forward call of this model in the forward function.

    Note:
    This model currently supports only Single Task Regression. Multi Task Regression, Single-Task Classification
    and Multi-Task Classification is not yet implemented.
    References
    ----------
    .. [1] Molecular Mechanics-Driven Graph Neural Network with Multiplex Graph for Molecular Structures https://arxiv.org/pdf/2011.07457

    Example
    -------
    >>> import deepchem as dc
    >>> import os
    >>> import tempfile
    >>> import torch
    >>> from torch_geometric.data import Data, Batch
    >>> from deepchem.feat.molecule_featurizers import MXMNetFeaturizer
    >>> QM9_TASKS = ["mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "cv", "u0", "u298",
    ...              "h298", "g298"]
    >>> dim = 10
    >>> n_layer = 6
    >>> cutoff = 5
    >>> feat = MXMNetFeaturizer()
    >>> # Get data
    >>> loader = dc.data.SDFLoader(tasks=[QM9_TASKS[0]],
    ...                            featurizer=feat,
    ...                            sanitize=True)
    >>> current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    >>> dataset_path = os.path.join(current_dir, "tests/assets/qm9_mini.sdf")
    >>> dataset = loader.create_dataset(inputs=dataset_path,
    ...                            shard_size=1)
    >>> model = MXMNet(dim=dim, n_layer=n_layer, cutoff=cutoff)
    >>> train_dir = None
    >>> if train_dir is None:
    ...    train_dir = tempfile.mkdtemp()
    >>> data = dataset.select([i for i in range(0, 1)], train_dir)
    >>> data = data.X
    >>> data = [data[i].to_pyg_graph() for i in range(1)]
    >>> pyg_batch = Batch()
    >>> pyg_batch = pyg_batch.from_data_list(data)
    >>> output = model(pyg_batch)
    >>> output[0].shape
    torch.Size([1])
    """

    def __init__(self,
                 dim: int,
                 n_layer: int,
                 cutoff: float,
                 num_spherical: int = 7,
                 num_radial: int = 6,
                 envelope_exponent: int = 5,
                 activation_fn: Union[Callable, str] = 'silu',
                 n_tasks: int = 1):
        """Initialize the MXMNet class.
        Parameters
        ----------
        dim: int
            The dimensionality of node embeddings.
        n_layer: int
            The number of message passing layers.
        cutoff: float
            The distance cutoff for edge connections.
        num_spherical: int, default 7
            The number of spherical harmonics.
        num_radial: int, default 6
            The number of radial basis functions.
        envelope_exponent: int, default 5
            The exponent for the envelope function.
        activation_fn: Union[Callable, str], default 'silu'
            The activation function name.
        n_tasks: int, default 1
            The number of prediction tasks. Only single Task regression is supported currently.
        """
        super(MXMNet, self).__init__()

        self.dim: int = dim
        self.n_tasks: int = n_tasks
        self.n_layer: int = n_layer
        self.cutoff: float = cutoff
        activation_fn = get_activation(activation_fn)

        self.embeddings: nn.Parameter = nn.Parameter(torch.ones((5, self.dim)))

        self.rbf_l: MXMNetBesselBasisLayer = dc.models.torch_models.layers.MXMNetBesselBasisLayer(
            16, 5, envelope_exponent)
        self.rbf_g: MXMNetBesselBasisLayer = dc.models.torch_models.layers.MXMNetBesselBasisLayer(
            16, self.cutoff, envelope_exponent)
        self.sbf: MXMNetSphericalBasisLayer = dc.models.torch_models.layers.MXMNetSphericalBasisLayer(
            num_spherical, num_radial, 5, envelope_exponent)

        self.rbf_g_mlp: MultilayerPerceptron = dc.models.torch_models.layers.MultilayerPerceptron(
            d_input=16, d_output=self.dim, activation_fn=activation_fn)
        self.rbf_l_mlp: MultilayerPerceptron = dc.models.torch_models.layers.MultilayerPerceptron(
            d_input=16, d_output=self.dim, activation_fn=activation_fn)

        self.sbf_1_mlp: MultilayerPerceptron = dc.models.torch_models.layers.MultilayerPerceptron(
            d_input=num_spherical * num_radial,
            d_output=self.dim,
            activation_fn=activation_fn)
        self.sbf_2_mlp: MultilayerPerceptron = dc.models.torch_models.layers.MultilayerPerceptron(
            d_input=num_spherical * num_radial,
            d_output=self.dim,
            activation_fn=activation_fn)

        self.global_layers: nn.ModuleList = torch.nn.ModuleList()
        for layer in range(self.n_layer):
            self.global_layers.append(
                dc.models.torch_models.layers.MXMNetGlobalMessagePassing(
                    self.dim))

        self.local_layers: nn.ModuleList = torch.nn.ModuleList()
        for layer in range(self.n_layer):
            self.local_layers.append(
                dc.models.torch_models.layers.MXMNetLocalMessagePassing(
                    self.dim))

        self.init()

    def init(self) -> None:
        """
        Initialize the node embeddings by setting them to random values within a predefined range.
        This method ensures that the node embeddings are initialized with random values, promoting diversity in the
        initial node representations.
        """

        stdv: float = math.sqrt(3)
        self.embeddings.data.uniform_(-stdv, stdv)

    def indices(
        self, edge_index: torch.Tensor, num_nodes: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor]:
        """
        Compute node indices for defining angles in the molecular graph.
        Parameters
        ----------
        edge_index: torch.Tensor
            The edge index tensor.
        num_nodes: int
            The number of nodes in the graph.
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Node indices for various angle calculations.
        """

        row: torch.Tensor
        col: torch.Tensor
        row, col = edge_index

        # Initialize an adjacency matrix with zeros
        adj_matrix: torch.Tensor = torch.zeros((num_nodes, num_nodes),
                                               device=row.device)

        value: torch.Tensor = torch.arange(row.size(0), device=row.device)

        for i in range(row.size(0)):
            adj_matrix[col[i], row[i]] = value[i]

        adj_t_row: torch.Tensor = adj_matrix[row]
        adj_t_row1: torch.Tensor = adj_matrix[row]

        for i in range(adj_t_row1.size(0)):
            if (adj_t_row1[i] != 0).sum() == 0:
                adj_t_row1[i][0] = 1

        # Compute the node indices for two-hop angles
        num_triplets: torch.Tensor = torch.zeros((adj_t_row.size(0)),
                                                 device=row.device)

        for i in range(adj_t_row1.size(0)):
            num_triplets[i] = (adj_t_row1[i] != 0).sum().to(torch.long)

        num_triplets = num_triplets.to(torch.long)

        idx_i: torch.Tensor = col.repeat_interleave(num_triplets)
        idx_j: torch.Tensor = row.repeat_interleave(num_triplets)
        idx_k: torch.Tensor = (adj_t_row1 != 0).nonzero()[:, 1]
        mask: torch.Tensor = idx_i != idx_k

        idx_i_1: torch.Tensor
        idx_i_1, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]
        idx_kj: torch.Tensor = adj_t_row[adj_t_row1 != 0].to(torch.long)[mask]
        idx_ji_1: torch.Tensor = (adj_t_row1 != 0).nonzero()[:, 0][mask]

        # Compute the node indices for one-hop angles
        adj_t_col: torch.Tensor = adj_matrix[col]
        adj_t_col1: torch.Tensor = adj_matrix[col]
        for i in range(adj_t_col1.size(0)):
            if (adj_t_col1[i] != 0).sum() == 0:
                adj_t_col1[i][0] = 1

        num_pairs: torch.Tensor = torch.zeros((adj_t_col.size(0)),
                                              device=row.device)
        for i in range(adj_t_col1.size(0)):
            num_pairs[i] = (adj_t_col1[i] != 0).sum().to(torch.long)

        num_pairs = num_pairs.to(torch.long)
        idx_i_2: torch.Tensor = row.repeat_interleave(num_pairs)
        idx_j1: torch.Tensor = col.repeat_interleave(num_pairs)
        idx_j2: torch.Tensor = (adj_t_col1 != 0).nonzero()[:, 1]

        idx_ji_2: torch.Tensor = (adj_t_col1 != 0).nonzero()[:, 0]
        idx_jj: torch.Tensor = adj_t_col[adj_t_col1 != 0].to(torch.long)

        return idx_i_1, idx_j, idx_k, idx_kj, idx_ji_1, idx_i_2, idx_j1, idx_j2, idx_jj, idx_ji_2

    def forward(self, pyg_batch: Batch) -> torch.Tensor:
        """
        Forward pass of the MXMNet model.
        Parameters
        ----------
        pyg_batch: Batch
            A pytorch-geometric batch containing tensors for:
                - node features
                - edge_index
                - pos
                - batch information
        Returns
        -------
        torch.Tensor
            The model's output.
        """

        x: torch.Tensor = pyg_batch.x
        edge_index: torch.Tensor = pyg_batch.edge_index
        pos: torch.Tensor = pyg_batch.pos
        batch = pyg_batch.batch
        x = torch.reshape(x, (-1,))
        # Initialize node embeddings
        h: torch.Tensor = torch.index_select(self.embeddings, 0, x.long())

        # Get the edges and pairwise distances in the local layer
        edge_index_l: torch.Tensor
        edge_index_l, _ = remove_self_loops(edge_index)
        j_l: torch.Tensor
        i_l: torch.Tensor
        j_l, i_l = edge_index_l
        dist_l: torch.Tensor = (pos[i_l] - pos[j_l]).pow(2).sum(dim=-1).sqrt()

        # Get the edges pairwise distances in the global layer
        row: torch.Tensor
        col: torch.Tensor
        row, col = radius(pos,
                          pos,
                          self.cutoff,
                          batch,
                          batch,
                          max_num_neighbors=500)
        edge_index_g: torch.Tensor = torch.stack([row, col], dim=0)
        edge_index_g, _ = remove_self_loops(edge_index_g)
        j_g: torch.Tensor
        i_g: torch.Tensor
        j_g, i_g = edge_index_g
        dist_g: torch.Tensor = (pos[i_g] - pos[j_g]).pow(2).sum(dim=-1).sqrt()

        num_nodes = h.size(0)
        # Compute the node indices for defining the angles
        idx_i_1, idx_j, idx_k, idx_kj, idx_ji, idx_i_2, idx_j1, idx_j2, idx_jj, idx_ji_2 = self.indices(
            edge_index_l, num_nodes=num_nodes)

        # Compute the two-hop angles
        pos_ji_1, pos_kj = pos[idx_j] - pos[idx_i_1], pos[idx_k] - pos[idx_j]
        a = (pos_ji_1 * pos_kj).sum(dim=-1)
        b: torch.Tensor = torch.cross(pos_ji_1, pos_kj).norm(dim=-1)
        angle_1: torch.Tensor = torch.atan2(b, a)

        # Compute the one-hop angles
        pos_ji_2, pos_jj = pos[idx_j1] - pos[idx_i_2], pos[idx_j2] - pos[idx_j1]
        a = (pos_ji_2 * pos_jj).sum(dim=-1)
        b = torch.cross(pos_ji_2, pos_jj).norm(dim=-1)
        angle_2: torch.Tensor = torch.atan2(b, a)

        # Get the RBF and SBF embeddings
        rbf_g: torch.Tensor = self.rbf_g(dist_g)
        rbf_l: torch.Tensor = self.rbf_l(dist_l)
        sbf_1: torch.Tensor = self.sbf(dist_l, angle_1, idx_kj)
        sbf_2: torch.Tensor = self.sbf(dist_l, angle_2, idx_jj)

        rbf_g = self.rbf_g_mlp(rbf_g)
        rbf_l = self.rbf_l_mlp(rbf_l)
        sbf_1 = self.sbf_1_mlp(sbf_1)
        sbf_2 = self.sbf_2_mlp(sbf_2)

        # Perform the message passing schemes
        node_sum: Optional[torch.Tensor] = None

        for layer in range(self.n_layer):
            h = self.global_layers[layer](h, rbf_g, edge_index_g)
            t: torch.Tensor
            h, t = self.local_layers[layer](h, rbf_l, sbf_1, sbf_2, idx_kj,
                                            idx_ji, idx_jj, idx_ji_2,
                                            edge_index_l)
            if (node_sum is None):
                node_sum = t
            else:
                node_sum += t

        # Readout
        output: torch.Tensor = global_add_pool(node_sum, batch)
        return output.view(-1, self.n_tasks)

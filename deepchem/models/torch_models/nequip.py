import logging
from typing import Optional
import torch
import torch.nn as nn
from deepchem.models.torch_models.layers import SE3GraphConv, Fiber

try:
    import dgl
except ImportError:
    pass


class NequIP(nn.Module):
    """
    The NequIP Neural Network Architecture.
    
    This module implements the E(3)-equivariant backbone described in the NequIP paper.
    It takes a molecular graph and outputs total potential energy and (optional) forces.
    
    Parameters
    ----------
    hidden_channels : int
        Number of channels for the hidden layers (scalar features).
    num_layers : int
        Number of interaction layers.
    max_degree : int
        Maximum degree of spherical harmonics (l=0, 1, ...).
    num_atoms : int
        Number of unique atom types (for embedding).
    """

    def __init__(self,
                 hidden_channels: int = 32,
                 num_layers: int = 3,
                 max_degree: int = 2,
                 num_atoms: int = 100,
                 atom_feature_dim: int = 32):
        super(NequIP, self).__init__()

        self.num_layers = num_layers
        self.max_degree = max_degree
        self.hidden_channels = hidden_channels

        # Embedding
        self.embedding = nn.Embedding(num_atoms, hidden_channels)

        # Define Hidden Fiber structure: scalars (l=0) + vectors (l=1) + ...
        structure = [(hidden_channels, l) for l in range(max_degree + 1)]
        self.fiber_hidden = Fiber(structure=structure)

        # Input fiber: just scalars
        self.fiber_in = Fiber(structure=[(hidden_channels, 0)])

        # Interaction Blocks
        self.blocks = nn.ModuleList()
        self.blocks.append(
            SE3GraphConv(self.fiber_in,
                         self.fiber_hidden,
                         self_interaction=True))
        for _ in range(num_layers - 1):
            self.blocks.append(
                SE3GraphConv(self.fiber_hidden,
                             self.fiber_hidden,
                             self_interaction=True))

        # Output Head: Project all features to scalars (l=0) for Energy prediction
        self.fiber_out = Fiber(structure=[(hidden_channels, 0)])
        self.final_conv = SE3GraphConv(self.fiber_hidden,
                                       self.fiber_out,
                                       self_interaction=True)

        # Readout MLP: Scalar -> Atomic Energy
        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.SiLU(),
            nn.Linear(hidden_channels, 1))

    def forward(self, g: 'dgl.DGLGraph', node_feats: torch.Tensor):
        """
        Parameters
        ----------
        g : dgl.DGLGraph
            Input graph. Must contain 'pos' in ndata if forces are required.
        node_feats : torch.Tensor
            LongTensor of atomic numbers.
            
        Returns
        -------
        total_energy : torch.Tensor (Batch, 1)
        forces : torch.Tensor (N_nodes, 3) (Optional)
        """
        # Ensure pos enables gradients if we are training
        pos = g.ndata.get('pos')
        compute_forces = (pos is not None) and (pos.requires_grad or
                                                self.training)

        if pos is not None:
            if not pos.requires_grad and self.training:
                pos.requires_grad_(True)
                g.ndata['pos'] = pos

            # Recompute edge features to ensure gradient flow for forces
            # The dataset loader creates static edge features which breaks autograd.
            # We must recompute them from the current `pos` tensor.
            src, dst = g.edges()
            edge_vecs = pos[dst] - pos[src]
            # Update the graph structure with differentiable edge vectors
            g.edata['edge_attr'] = edge_vecs

        # 1. Get Geometric Basis
        from deepchem.utils.equivariance_utils import get_equivariant_basis_and_r
        basis, r = get_equivariant_basis_and_r(g, self.max_degree)

        # 2. Embedding
        h = self.embedding(node_feats)
        h = {'0': h.unsqueeze(-1)}

        # 3. Message Passing
        for layer in self.blocks:
            h = layer(h, g, r=r, basis=basis)

        # 4. Readout
        h_final = self.final_conv(h, g, r=r, basis=basis)
        scalar_out = h_final['0'].squeeze(-1)
        atomic_energies = self.readout_mlp(scalar_out)

        # Sum atomic energies to get Total Energy per graph
        # Explicitly store in ndata to avoid dgl.sum_nodes interpreting tensor as key
        g.ndata['atomic_energies'] = atomic_energies
        total_energy_pred = dgl.sum_nodes(g, 'atomic_energies')

        if compute_forces:
            # Automatic differentiation for Forces
            # F = - grad(E)
            sum_total_energy = total_energy_pred.sum()

            grads = torch.autograd.grad(sum_total_energy,
                                        pos,
                                        create_graph=self.training,
                                        retain_graph=self.training,
                                        allow_unused=True)[0]

            if grads is None:
                forces = torch.zeros_like(pos)
            else:
                forces = -grads

            return total_energy_pred, forces

        return total_energy_pred, None

import torch
import torch.nn as nn
import torch.nn.functional as F
from deepchem.models.torch_models.layers import WLNGraphConvolution
from deepchem.models.torch_models.torch_model import TorchModel


import torch
import torch.nn as nn
from typing import Tuple

class WLN(nn.Module):
    """
    Weisfeiler-Lehman Network (WLN) for predicting chemical reactivity.

    This model implements the architecture described in "A graph-convolutional neural network model 
    for the prediction of chemical reactivity" [1]_. and It is designed to predict the likelihood of 
    changes in bond order between every pair of atoms in a reactant set.

    The model consists of three main stages:
    1.  **Reactivity Perception (Local Embeddings):** A graph convolutional network (WLNConv) 
        updates atom representations based on their local chemical environment.
    2.  **Global Attention:** A global attention mechanism computes a context vector for each atom, 
        accounting for long-range effects and reagent influences (e.g., activating reagents).
    3.  **Pairwise Scoring:** The model predicts a probability distribution over potential bond 
        changes (e.g., single, double, triple, no-bond) for every pair of atoms using local features, 
        global context, and pairwise bond features.

    References
    ----------
    .. [1] Coley, Connor W., et al. "A graph-convolutional neural network model for the prediction of chemical reactivity." Chemical Science 10.2 (2019): 370-377.

    Examples
    --------
    >>> import torch
    >>> from deepchem.models.torch_models.layers import WLN, WLNGraphConvolution
    >>> # Define hyperparameters
    >>> batch_size = 2
    >>> max_atoms = 5
    >>> atom_fdim = 10
    >>> wln_bond_fdim = 6
    >>> binary_fdim = 4
    >>> hidden_size = 32
    >>> depth = 3
    >>> num_bond_orders = 5
    >>> # Initialize model
    >>> model = WLN(atom_fdim, wln_bond_fdim, binary_fdim, hidden_size, depth, num_bond_orders)
    >>> # Create dummy input tensors
    >>> atom_features = torch.randn(batch_size, max_atoms, atom_fdim)
    >>> adj_matrix = torch.ones(batch_size, max_atoms, max_atoms)
    >>> wln_bond_features = torch.randn(batch_size, max_atoms, max_atoms, wln_bond_fdim)
    >>> binary_features = torch.randn(batch_size, max_atoms, max_atoms, binary_fdim)
    >>> atom_mask = torch.ones(batch_size, max_atoms)
    >>> # Forward pass
    >>> scores = model(atom_features, adj_matrix, wln_bond_features, binary_features, atom_mask)
    >>> print(scores.shape)
    torch.Size([2, 5, 5, 5])
    """

    def __init__(self, 
                 atom_feature_dim: int, 
                 wln_bond_fdim: int, 
                 binary_fdim: int, 
                 hidden_size: int, 
                 depth: int, 
                 num_bond_orders: int):
        """
        Initialize the WLN model.

        Parameters
        ----------
        atom_feature_dim: int
            Dimension of the initial atom feature vectors.
        wln_bond_fdim: int
            Dimension of the bond features used specifically for the Graph Convolution (WLNConv).
        binary_fdim: int
            Dimension of the binary/bond features used for the Global Attention and Pairwise Scoring components.
        hidden_size: int
            Dimension of the hidden layers (size of the atom representations).
        depth: int
            Number of message passing iterations in the graph convolution.
        num_bond_orders: int
            Number of output classes for bond changes.
        """
        super(WLN, self).__init__()
        
        # This part uses the WLN_BOND_FDIM 
        self.WLNConv = WLNGraphConvolution(atom_feature_dim, wln_bond_fdim, hidden_size, depth)
        

        # It processes the concatenation of the atom's state and the sum of its neighbors' states.
        self.local_linear = nn.Linear(hidden_size * 2, hidden_size)
        

        # Calculates attention scores alpha_vz based on atom states and pairwise features.
        self.attn_P_a = nn.Linear(hidden_size, hidden_size) # P_a
        self.attn_P_b = nn.Linear(binary_fdim, hidden_size) # P_b
        self.attn_u = nn.Linear(hidden_size, 1)             # u^T
        self.relu = nn.ReLU()
        

        # Predicts the reactivity score s_av for every pair of atoms.
        self.pair_linear = nn.Linear(hidden_size * 4 + binary_fdim, num_bond_orders)
        
        self.hidden_size = hidden_size
        self.num_bond_orders = num_bond_orders

    def forward(self, 
                atom_features: torch.Tensor, 
                adj_matrix: torch.Tensor, 
                wln_bond_features: torch.Tensor, 
                binary_features: torch.Tensor, 
                atom_mask: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass to predict pairwise bond change likelihoods.

        Parameters
        ----------
        atom_features: torch.Tensor
            Input atom features of shape `(batch_size, max_atoms, atom_fdim)`.
        adj_matrix: torch.Tensor
            Adjacency matrix of shape `(batch_size, max_atoms, max_atoms)`.
        wln_bond_features: torch.Tensor
            Bond features for the graph convolution, shape `(batch_size, max_atoms, max_atoms, wln_bond_fdim)`.
        binary_features: torch.Tensor
            Bond features for attention and scoring, shape `(batch_size, max_atoms, max_atoms, binary_fdim)`.
        atom_mask: torch.Tensor
            Mask indicating valid atoms (1 for real, 0 for padding), shape `(batch_size, max_atoms)`.

        Returns
        -------
        scores: torch.Tensor
            Predicted scores for each bond type between every pair of atoms.
            Shape `(batch_size, max_atoms, max_atoms, num_bond_orders)`.
        """
        


        h, _ = self.WLNConv(atom_features, adj_matrix, wln_bond_features, atom_mask)
        float_mask = atom_mask.unsqueeze(-1).float()
        


        sum_neigh = torch.bmm(adj_matrix.float(), h)
        local_input = torch.cat([h, sum_neigh], dim=-1)
        local_feat = self.local_linear(local_input) * float_mask
        

        B, A, D = h.shape
        h_i_exp = h.unsqueeze(2).expand(B, A, A, D)  # c_v (target atom)
        h_k_exp = h.unsqueeze(1).expand(B, A, A, D)  # c_z (context atom)
        bond_exp = binary_features                   # b_vz (bond features)
        
        Pa_cv = self.attn_P_a(h_i_exp)
        Pa_cz = self.attn_P_a(h_k_exp) # Share weights for Pa
        Pb_bvz = self.attn_P_b(bond_exp)
        
        attn_hidden = self.relu(Pa_cv + Pa_cz + Pb_bvz)
        attn_logits = self.attn_u(attn_hidden).squeeze(-1)
        attn_weights = torch.sigmoid(attn_logits) # alpha_vz
        

        context = torch.sum(attn_weights.unsqueeze(-1) * h_k_exp, dim=2) * float_mask # tilde_c_v


        local_i = local_feat.unsqueeze(2).expand(B, A, A, D)
        local_j = local_feat.unsqueeze(1).expand(B, A, A, D)
        c_i = context.unsqueeze(2).expand(B, A, A, D)
        c_j = context.unsqueeze(1).expand(B, A, A, D)
        

        pair_input = torch.cat([local_i, local_j, c_i, c_j, bond_exp], dim=-1)
        logits = self.pair_linear(pair_input)
        scores = torch.sigmoid(logits)
        

        mask_i = atom_mask.unsqueeze(1).unsqueeze(-1).float()  # [B, 1, A, 1]
        mask_j = atom_mask.unsqueeze(-1).unsqueeze(-1).float()  # [B, A, 1, 1]
        scores = scores * mask_i * mask_j
        
        return scores

class WLNScoring(TorchModel):
    def __init__(self, 
                 atom_feature_dim: int, 
                 wln_bond_fdim: int, 
                 binary_fdim: int, 
                 hidden_size: int, 
                 depth: int =3, 
                 num_bond_orders: int = 5,
                 loss: nn.modules = nn.BCELoss()):
        
        
        self.loss_fn = loss
        model = WLN(atom_feature_dim, wln_bond_fdim, binary_fdim, hidden_size, depth, num_bond_orders)
        
        super(WLNScoring,self).__init__(model = model,loss=self.loss_fn)
        
        
    def default_generator(self):
        
       "TODO : Implement custom default generator since it requires custom dataset handling (WLN featurizer)"
       pass
           
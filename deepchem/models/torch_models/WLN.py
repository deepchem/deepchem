import torch
import torch.nn as nn
import torch.nn.functional as F
from deepchem.models.torch_models.layers import WLNGraphConvolution
from deepchem.models.torch_models.torch_model import TorchModel


class WLN(nn.Module):
    """
    Combines local graph convolutions, global attention, and pairwise scoring.
    """
    def __init__(self, atom_feature_dim: int, wln_bond_fdim: int, binary_fdim: int, 
                 hidden_size: int, depth: int, num_bond_orders: int):
        super(WLN, self).__init__()
        
        # --- Local Embedding Component ---
        # This part uses the WLN_BOND_FDIM 
        self.WLNConv = WLNGraphConvolution(atom_feature_dim, wln_bond_fdim, hidden_size, depth)
        
        # --- Local Feature Calculation (Figure 2C) ---
        # This part is not explicitly in the supplement's equations but is in Figure 2C [cite: 253]
        self.local_linear = nn.Linear(hidden_size * 2, hidden_size)
        
        # --- Global Attention Component (S2.3 / Figure 2D) ---
        # This part uses the BINARY_FDIM [cite: 254, 548]
        self.attn_P_a = nn.Linear(hidden_size, hidden_size) # P_a
        self.attn_P_b = nn.Linear(binary_fdim, hidden_size) # P_b
        self.attn_u = nn.Linear(hidden_size, 1)             # u^T
        self.relu = nn.ReLU()
        
        # --- Final Prediction Component (S2.4 / Figure 2E) ---
        # This part uses local feats, global context, and binary feats [cite: 255, 554]
        self.pair_linear = nn.Linear(hidden_size * 4 + binary_fdim, num_bond_orders)
        
        self.hidden_size = hidden_size
        self.num_bond_orders = num_bond_orders

    def forward(self, atom_features, adj_matrix, wln_bond_features, binary_features, atom_mask):
        """
        *** CORRECTED Forward Pass ***
        
        Args:
            atom_features (Tensor): (B, A, atom_fdim)
            adj_matrix (Tensor): (B, A, A)
            wln_bond_features (Tensor): (B, A, A, wln_bond_fdim) - For WLNConv
            binary_features (Tensor): (B, A, A, binary_fdim) - For Attention & Scoring
            atom_mask (Tensor): (B, A)
        """
        
        # 1. Local Embeddings (c_v)
        # h is the local atom embedding, c_v in the paper [cite: 549]
        h, _ = self.WLNConv(atom_features, adj_matrix, wln_bond_features, atom_mask)
        float_mask = atom_mask.unsqueeze(-1).float()
        
        # 2. Local Features (Figure 2C)
        sum_neigh = torch.bmm(adj_matrix.float(), h)
        local_input = torch.cat([h, sum_neigh], dim=-1)
        local_feat = self.local_linear(local_input) * float_mask
        
        # 3. Global Context (tilde_c_v) (S2.3 / Figure 2D) [cite: 254, 548]
        B, A, D = h.shape
        h_i_exp = h.unsqueeze(2).expand(B, A, A, D)  # c_v
        h_k_exp = h.unsqueeze(1).expand(B, A, A, D)  # c_z
        bond_exp = binary_features                  # b_vz
        
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
        
        # Mask padded atoms
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
        
        super(WLNScoring,self).__init__(model = model,loss=self._loss_fn)
        
        
    def _loss_fn(self,outputs,labels,weights):
        
        labels_tensor: torch.Tensor = labels[0]
        outputs_tensor: torch.Tensor = outputs[0]
        loss = self.loss_fn(labels_tensor, outputs_tensor)
        return loss
import pytest
import torch
torch.manual_seed(42) 
from deepchem.models.torch_models.layers import WLNGraphConvolution

try:
    import torch
    has_torch = True
except ModuleNotFoundError:
    has_torch = False
    pass

@pytest.mark.torch
def test_output_shape():
    "test WLNGraphConvolution output shapes"
    batch_size = 2
    max_atoms = 5
    atom_dim = 10
    bond_dim = 6
    hidden_size = 32
    depth = 3
    conv = WLNGraphConvolution(atom_dim, bond_dim, hidden_size, depth)
    
    atom_features = torch.randn(batch_size, max_atoms, atom_dim)
    bond_features = torch.randn(batch_size, max_atoms, max_atoms, bond_dim)
    adj_matrix = torch.ones(batch_size, max_atoms, max_atoms)
    atom_mask = torch.ones(batch_size, max_atoms)
    atom_reps, graph_rep = conv(atom_features, adj_matrix, bond_features, atom_mask)
    
    assert atom_reps.shape == (batch_size, max_atoms, hidden_size)
    assert graph_rep.shape == (batch_size, hidden_size)

@pytest.mark.torch    
def test_graph_isomorphism():
    "test WLNGraphConvolution graph isomorphism invariance"
    batch_size = 1
    max_atoms = 4
    atom_dim = 8
    bond_dim = 5
    hidden_size = 16
    depth = 2
    
    conv = WLNGraphConvolution(atom_dim, bond_dim, hidden_size, depth)
    

    atom_features_1 = torch.randn(batch_size, max_atoms, atom_dim)
    bond_features_1 = torch.randn(batch_size, max_atoms, max_atoms, bond_dim)

    adj_matrix_1 = torch.tensor([[[0, 1, 0, 1],
                                  [1, 0, 1, 0],
                                  [0, 1, 0, 1],
                                  [1, 0, 1, 0]]], dtype=torch.float32)
    atom_mask_1 = torch.ones(batch_size, max_atoms)
    

    # Permutation indices: 0->0, 1->1, 2->3, 3->2
    perm_idx = torch.tensor([0, 1, 3, 2])
    
    atom_features_2 = atom_features_1[:, perm_idx, :]
    
    adj_matrix_2 = adj_matrix_1[:, perm_idx, :] # Permute rows
    adj_matrix_2 = adj_matrix_2[:, :, perm_idx] # Permute cols
    
    bond_features_2 = bond_features_1[:, perm_idx, :, :] # Permute rows
    bond_features_2 = bond_features_2[:, :, perm_idx, :] # Permute cols
    
    atom_mask_2 = atom_mask_1.clone() 
    
    conv.eval()
    
    with torch.no_grad():
        _, graph_rep_1 = conv(atom_features_1, adj_matrix_1, bond_features_1, atom_mask_1)
        _, graph_rep_2 = conv(atom_features_2, adj_matrix_2, bond_features_2, atom_mask_2)
    

    assert torch.allclose(graph_rep_1, graph_rep_2, atol=1e-5)
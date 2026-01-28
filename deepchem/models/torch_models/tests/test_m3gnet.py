import sys
import os
from unittest.mock import MagicMock
mock_tf = MagicMock()
mock_tf.__spec__ = MagicMock()
sys.modules["tensorflow"] = mock_tf
sys.modules["dgl"] = MagicMock() # Mock dgl to avoid DLL error

# Add project root to sys.path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from torch_geometric.data import Batch, Data
from deepchem.models.torch_models.m3gnet import M3GNet
import numpy as np

def test_m3gnet_forward():
    # 1. Create Dummy Data
    num_nodes = 5
    num_edges = 10
    
    # Random atom types (e.g. H, C, O)
    z = torch.randint(0, 10, (num_nodes,))
    
    # Random positions (3D)
    pos = torch.randn(num_nodes, 3)
    
    # Edge index (fully connected for simplicity)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Distances (edges)
    edge_attr = torch.randn(num_edges, 1) # Distances
    
    # Angles (triplets) - Dummy
    # In real M3GNet, we need actual triplets. Here we mock them.
    num_triplets = 20
    angle = torch.randn(num_triplets) 
    idx_kj = torch.randint(0, num_edges, (num_triplets,)) # Map triplet to k-j edge
    
    data = Data(x=z, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
    # Attach M3GNet specific attributes normally handled by featurizer or graph construction
    data.angle = angle
    data.idx_kj = idx_kj
    
    batch = Batch.from_data_list([data])
    
    # 2. Instantiate Model
    model = M3GNet(units=16, n_blocks=2, n_atom_types=20, n_radial=4, n_spherical=3)
    
    # 3. Forward Pass
    output = model(batch)
    
    print("Output Shape:", output.shape)
    assert output.shape == (1, 1), f"Expected (1, 1), got {output.shape}"
    print("Verification Successful!")

if __name__ == "__main__":
    test_m3gnet_forward()

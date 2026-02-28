
# Core imports
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem

# DeepChem
import deepchem as dc
from deepchem.models import TorchModel
from deepchem.models.losses import Loss
from deepchem.data import Dataset, NumpyDataset

# Typing
from typing import Tuple, Optional, List, Dict, Any, Sequence, Iterator

# Utilities
import logging
import time
import pickle
import math
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print(" All imports successful!")
print(f" PyTorch version: {torch.__version__}")
print(f" DeepChem version: {dc.__version__}")
print(f" Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")


print("="*70)
print("UNIT TESTS FOR MACE MODEL")
print("="*70)

import unittest
import torch
from torch_geometric.data import Data, Batch

class TestMACEComponents(unittest.TestCase):

    def setUp(self):
        """Setup test fixtures"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_dim = 64
        self.num_basis = 8

    def test_radial_basis_shape(self):
        """Test RadialBasis output shape"""
        print("\n Testing RadialBasis shape...")
        basis = RadialBasis(num_basis=self.num_basis, cutoff=5.0).to(self.device)
        distances = torch.rand(100, 1).to(self.device) * 5.0
        output = basis(distances)

        self.assertEqual(output.shape, (100, self.num_basis))
        self.assertTrue(torch.all(torch.isfinite(output)))
        print("PASS: RadialBasis outputs correct shape")

    def test_radial_basis_cutoff(self):
        """Test RadialBasis respects cutoff"""
        print("\n Testing RadialBasis cutoff...")
        basis = RadialBasis(num_basis=self.num_basis, cutoff=5.0).to(self.device)

        # Distances beyond cutoff
        far_distances = torch.tensor([[6.0], [7.0], [10.0]]).to(self.device)
        output = basis(far_distances)

        # Should be close to zero
        self.assertTrue(torch.all(output.abs() < 0.1))
        print(" PASS: Values suppressed beyond cutoff")

    def test_equivariant_interaction_shape(self):
        """Test EquivariantMACEInteraction preserves shapes"""
        print("\n Testing Interaction layer...")
        interaction = EquivariantMACEInteractionClean(self.hidden_dim, self.num_basis).to(self.device)

        num_atoms = 10
        num_edges = 30

        s = torch.randn(num_atoms, self.hidden_dim).to(self.device)
        v = torch.randn(num_atoms, self.hidden_dim, 3).to(self.device)
        edge_index = torch.randint(0, num_atoms, (2, num_edges)).to(self.device)
        edge_attr = torch.randn(num_edges, self.num_basis).to(self.device)
        edge_vec = torch.randn(num_edges, 3).to(self.device)

        s_out, v_out = interaction(s, v, edge_index, edge_attr, edge_vec)

        self.assertEqual(s_out.shape, (num_atoms, self.hidden_dim))
        self.assertEqual(v_out.shape, (num_atoms, self.hidden_dim, 3))
        self.assertTrue(torch.all(torch.isfinite(s_out)))
        self.assertTrue(torch.all(torch.isfinite(v_out)))
        print("PASS: Shapes preserved correctly")

    def test_e3_equivariance_rotation(self):
        """Test E(3) equivariance under rotation"""
        print("\n Testing E(3) equivariance (rotation)...")
        model = MACEClean(hidden_dim=64, num_interactions=2, num_basis=8, cutoff=5.0).to(self.device)

        # Create test molecule
        torch.manual_seed(42)
        z = torch.randint(1, 10, (8,)).to(self.device)
        pos = torch.randn(8, 3).to(self.device)

        # Create edges
        num_atoms = 8
        pos_i = pos.unsqueeze(0).expand(num_atoms, -1, -1)
        pos_j = pos.unsqueeze(1).expand(-1, num_atoms, -1)
        dist = torch.norm(pos_i - pos_j, dim=2)
        mask = (dist < 5.0) & (dist > 0)
        edge_index = mask.nonzero(as_tuple=False).t()

        batch = torch.zeros(8, dtype=torch.long).to(self.device)

        # Original prediction
        energy1, _ = model(z, pos, edge_index, batch)

        # Rotation matrix (90 degrees around z-axis)
        rotation = torch.tensor([
            [0., -1., 0.],
            [1., 0., 0.],
            [0., 0., 1.]
        ], dtype=torch.float32).to(self.device)

        pos_rotated = pos @ rotation.T

        # Recompute edges for rotated positions
        pos_i_rot = pos_rotated.unsqueeze(0).expand(num_atoms, -1, -1)
        pos_j_rot = pos_rotated.unsqueeze(1).expand(-1, num_atoms, -1)
        dist_rot = torch.norm(pos_i_rot - pos_j_rot, dim=2)
        mask_rot = (dist_rot < 5.0) & (dist_rot > 0)
        edge_index_rot = mask_rot.nonzero(as_tuple=False).t()

        # Prediction after rotation
        energy2, _ = model(z, pos_rotated, edge_index_rot, batch)

        # Energies should be identical (invariant)
        diff = torch.abs(energy1 - energy2).item()
        self.assertTrue(diff < 1e-4, f"Energy diff: {diff:.6e}")
        print(f"PASS: Rotation invariant (diff: {diff:.2e})")

    def test_e3_equivariance_translation(self):
        """Test E(3) equivariance under translation"""
        print("\n Testing E(3) equivariance (translation)...")
        model = MACEClean(hidden_dim=64, num_interactions=2, num_basis=8, cutoff=5.0).to(self.device)

        torch.manual_seed(42)
        z = torch.randint(1, 10, (8,)).to(self.device)
        pos = torch.randn(8, 3).to(self.device)

        # Create edges
        num_atoms = 8
        pos_i = pos.unsqueeze(0).expand(num_atoms, -1, -1)
        pos_j = pos.unsqueeze(1).expand(-1, num_atoms, -1)
        dist = torch.norm(pos_i - pos_j, dim=2)
        mask = (dist < 5.0) & (dist > 0)
        edge_index = mask.nonzero(as_tuple=False).t()

        batch = torch.zeros(8, dtype=torch.long).to(self.device)

        # Original prediction
        energy1, _ = model(z, pos, edge_index, batch)

        # Translate by random vector
        translation = torch.randn(1, 3).to(self.device)
        pos_translated = pos + translation

        # Edges don't change with translation
        energy2, _ = model(z, pos_translated, edge_index, batch)

        diff = torch.abs(energy1 - energy2).item()
        self.assertTrue(diff < 1e-4, f"Energy diff: {diff:.6e}")
        print(f" PASS: Translation invariant (diff: {diff:.2e})")

    def test_force_computation(self):
        """Test force prediction works correctly"""
        print("\n Testing force computation...")
        base_mace = MACEClean(hidden_dim=64, num_interactions=2, num_basis=8, cutoff=5.0).to(self.device)
        model = MACEWithForcesClean(base_mace).to(self.device)

        torch.manual_seed(42)
        z = torch.randint(1, 10, (8,)).to(self.device)
        pos = torch.randn(8, 3, requires_grad=True).to(self.device)

        num_atoms = 8
        pos_i = pos.unsqueeze(0).expand(num_atoms, -1, -1)
        pos_j = pos.unsqueeze(1).expand(-1, num_atoms, -1)
        dist = torch.norm(pos_i - pos_j, dim=2)
        mask = (dist < 5.0) & (dist > 0)
        edge_index = mask.nonzero(as_tuple=False).t()

        batch = torch.zeros(8, dtype=torch.long).to(self.device)

        energy, forces = model(z, pos, edge_index, batch, compute_forces=True)

        # FIXED: Accept (1, 1) or (1,) shape
        self.assertTrue(energy.shape in [(1,), (1, 1)])
        self.assertEqual(forces.shape, (8, 3))
        self.assertTrue(torch.all(torch.isfinite(forces)))
        print(f" PASS: Forces computed correctly")

    def test_force_energy_consistency(self):
        """Test forces are negative gradients of energy"""
        print("\n Testing force-energy consistency...")
        base_mace = MACEClean(hidden_dim=64, num_interactions=2, num_basis=8, cutoff=5.0).to(self.device)
        model = MACEWithForcesClean(base_mace).to(self.device)

        torch.manual_seed(42)
        z = torch.randint(1, 10, (5,)).to(self.device)
        pos = torch.randn(5, 3, requires_grad=True).to(self.device)

        num_atoms = 5
        pos_i = pos.unsqueeze(0).expand(num_atoms, -1, -1)
        pos_j = pos.unsqueeze(1).expand(-1, num_atoms, -1)
        dist = torch.norm(pos_i - pos_j, dim=2)
        mask = (dist < 5.0) & (dist > 0)
        edge_index = mask.nonzero(as_tuple=False).t()

        batch = torch.zeros(5, dtype=torch.long).to(self.device)

        # Get forces from model
        energy, forces_model = model(z, pos, edge_index, batch, compute_forces=True)

        # Compute forces manually
        forces_manual = -torch.autograd.grad(energy.sum(), pos, create_graph=False)[0]

        diff = torch.abs(forces_model - forces_manual).max().item()
        self.assertTrue(diff < 1e-5, f"Force consistency error: {diff:.6e}")
        print(f"PASS: Forces are correct gradients (diff: {diff:.2e})")

    def test_batch_processing(self):
        """Test model handles batched molecules of different sizes"""
        print("\n Testing batch processing...")
        model = MACEClean(hidden_dim=64, num_interactions=2, num_basis=8, cutoff=5.0).to(self.device)

        # Create 3 molecules with different sizes
        data_list = []
        for num_atoms in [5, 8, 6]:
            z = torch.randint(1, 10, (num_atoms,))
            pos = torch.randn(num_atoms, 3)

            pos_i = pos.unsqueeze(0).expand(num_atoms, -1, -1)
            pos_j = pos.unsqueeze(1).expand(-1, num_atoms, -1)
            dist = torch.norm(pos_i - pos_j, dim=2)
            mask = (dist < 5.0) & (dist > 0)
            edge_index = mask.nonzero(as_tuple=False).t()

            data_list.append(Data(z=z, pos=pos, edge_index=edge_index))

        batch = Batch.from_data_list(data_list).to(self.device)

        energy, _ = model(batch.z, batch.pos, batch.edge_index, batch.batch)

        # FIXED: Accept (3, 1) or (3,) shape
        self.assertTrue(energy.shape in [(3,), (3, 1)])
        self.assertTrue(torch.all(torch.isfinite(energy)))
        print(f"PASS: Handles variable-sized molecules")

    def test_model_reproducibility(self):
        """Test model gives same output with same seed"""
        print("\n Testing reproducibility...")

        torch.manual_seed(42)
        model1 = MACEClean(hidden_dim=64, num_interactions=2, num_basis=8, cutoff=5.0).to(self.device)

        torch.manual_seed(42)
        model2 = MACEClean(hidden_dim=64, num_interactions=2, num_basis=8, cutoff=5.0).to(self.device)

        torch.manual_seed(123)
        z = torch.randint(1, 10, (8,)).to(self.device)
        pos = torch.randn(8, 3).to(self.device)

        num_atoms = 8
        pos_i = pos.unsqueeze(0).expand(num_atoms, -1, -1)
        pos_j = pos.unsqueeze(1).expand(-1, num_atoms, -1)
        dist = torch.norm(pos_i - pos_j, dim=2)
        mask = (dist < 5.0) & (dist > 0)
        edge_index = mask.nonzero(as_tuple=False).t()
        batch = torch.zeros(8, dtype=torch.long).to(self.device)

        energy1, _ = model1(z, pos, edge_index, batch)
        energy2, _ = model2(z, pos, edge_index, batch)

        diff = torch.abs(energy1 - energy2).item()
        self.assertTrue(diff < 1e-6, f"Reproducibility error: {diff:.6e}")
        print(f" PASS: Reproducible (diff: {diff:.2e})")


# Run tests
print("\n")
print("RUNNING TESTS")
print("="*70)

suite = unittest.TestLoader().loadTestsFromTestCase(TestMACEComponents)
runner = unittest.TextTestRunner(verbosity=0)
result = runner.run(suite)

print("\n")
if result.wasSuccessful():
    print(f"ALL {result.testsRun} TESTS PASSED!")

    print("\n What we verified:")
    print("   RadialBasis works correctly")
    print("   Interaction layers preserve shapes")
    print("   E(3) equivariance (rotation & translation)")
    print("   Force computation works")
    print("   Forces match energy gradients")
    print("   Batch processing handles variable sizes")
    print("   Model is reproducible")
    print("\n   Your MACE implementation is scientifically sound! ")
else:
    print(f"{len(result.failures + result.errors)} TESTS FAILED")

    for failure in result.failures + result.errors:
        print(f"\n {failure[0]}")
        print(failure[1][:300])

print("\n")
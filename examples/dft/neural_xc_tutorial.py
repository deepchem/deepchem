"""
Tutorial: Neural Network Exchange-Correlation Functionals in DeepChem
======================================================================

This tutorial demonstrates how to use neural network-based XC functionals
(NNLDA and NNPBE) for DFT calculations in DeepChem.

Author: Utkarsh Khajuria (GSoC 2026 Preparation)
Date: December 2025
"""

import torch
import torch.nn as nn
from deepchem.models.dft.nnxc import NNLDA, NNPBE, HybridXC
try:
    from dqc.utils.datastruct import ValGrad
except:
    from deepchem.utils.dftutils import ValGrad
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("NEURAL XC FUNCTIONAL TUTORIAL - DeepChem DFT")
print("=" * 70)

# Part 1: Understanding NNLDA (Neural LDA Functional)
print("\n### Part 1: Neural LDA Functional ###\n")
print("LDA (Local Density Approximation) functionals depend only on")
print("electron density ρ(r) at each point in space.")
print("\nIn NNLDA, we replace the traditional LDA functional with a")
print("neural network that learns the exchange-correlation energy.")

# Create a simple neural network model
class SimpleXCNet(nn.Module):
    """Simple 2-layer neural network for XC functional"""
    def __init__(self, input_size=2, hidden_size=16):
        super(SimpleXCNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.net(x)

# Initialize the model
torch.manual_seed(42)
xc_model = SimpleXCNet(input_size=2, hidden_size=16)
print(f"\n✓ Created neural network with {sum(p.numel() for p in xc_model.parameters())} parameters")

# Create NNLDA functional
nnlda = NNLDA(xc_model)
print("✓ Initialized NNLDA functional")

# Create sample density information
n_points = 10  # Number of grid points
densinfo = ValGrad(value=torch.rand(n_points).requires_grad_())

# Compute exchange-correlation energy density
try:
    exc_density = nnlda.get_edensityxc(densinfo)
    print(f"✓ Computed XC energy density at {n_points} grid points")
    print(f"  Shape: {exc_density.shape}")
    print(f"  Sample values: {exc_density[:3].detach().numpy()}")
except Exception as e:
    print(f"✗ NNLDA computation failed: {e}")

# Part 2: Understanding NNPBE (Neural PBE Functional)
print("\n\n### Part 2: Neural PBE (GGA) Functional ###\n")
print("PBE is a GGA (Generalized Gradient Approximation) functional")
print("that depends on both density ρ(r) AND its gradient ∇ρ(r).")
print("\nThis captures non-local effects better than LDA.")

# Create NNPBE functional
try:
    nnpbe = NNPBE(xc_model)
    print("\n✓ Initialized NNPBE functional")
    
    # For GGA functionals, we need gradient information too
    densinfo_gga = ValGrad(
        value=torch.rand(n_points).requires_grad_(),
        grad=torch.rand(n_points, 3).requires_grad_()  # 3D gradient
    )
    
    exc_density_gga = nnpbe.get_edensityxc(densinfo_gga)
    print(f"✓ Computed GGA XC energy density")
    print(f"  Input: density + gradient at {n_points} points")
    print(f"  Output shape: {exc_density_gga.shape}")
except Exception as e:
    print(f"✗ NNPBE computation failed: {e}")

# Part 3: Hybrid Functionals
print("\n\n### Part 3: Hybrid XC Functionals ###\n")
print("HybridXC combines traditional functionals with neural networks:")
print("E_xc = α * E_xc^traditional + (1-α) * E_xc^neural")
print("\nThis allows gradual transition from known functionals to learned ones.")

try:
    # Create hybrid functional mixing LDA with neural network
    hybrid_xc = HybridXC("lda_x", xc_model, aweight0=0.5)
    print("\n✓ Created HybridXC with 50% LDA, 50% neural")
    
    exc_hybrid = hybrid_xc.get_edensityxc(densinfo)
    print(f"✓ Computed hybrid XC energy density")
    print(f"  Traditional (LDA) weight: 0.5")
    print(f"  Neural network weight: 0.5")
except Exception as e:
    print(f"✗ HybridXC computation failed: {e}")

# Part 4: Training Considerations
print("\n\n### Part 4: Training Neural XC Functionals ###\n")
print("To train these functionals, you typically:")
print("1. Collect reference DFT data (high-level calculations)")
print("2. Define loss function (energy errors, forces, etc.)")
print("3. Optimize neural network parameters")
print("4. Validate on test molecules")
print("\nSee the full XCModel class in dftxc.py for training implementation.")

print("\n" + "=" * 70)
print("TUTORIAL COMPLETE")
print("=" * 70)
print("\nNext steps:")
print("- Read the paper: arxiv.org/abs/2309.15985")
print("- Explore deepchem/models/dft/dftxc.py for training")
print("- Check deepchem/models/tests/ for more examples")
print("\nFor GSoC 2026: Consider implementing neural versions of:")
print("  • Meta-GGA functionals (SCAN, M06-L)")
print("  • Range-separated hybrids (ωB97X)")
print("  • Double-hybrid functionals (B2PLYP)")

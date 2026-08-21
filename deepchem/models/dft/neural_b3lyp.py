"""
Neural B3LYP - Dual Implementation
==================================

1. Production version (with DQC/LibXC) - for PR submissions
2. Simple version (PyTorch only) - fallback & learning

Author: Utkarsh Khajuria (@UtkarsHMer05)  
Date: December 2025
For GSoC 2026 - DeepChem DFT Support
"""

import torch
import torch.nn as nn
from typing import Union, Optional
import warnings
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__) or '.')

# Check for DQC availability
HAS_DQC = False
try:
    from dqc.utils.datastruct import ValGrad
    from dqc.api.getxc import get_xc
    from deepchem.utils.dftutils import SpinParam
    from deepchem.models.dft.nnxc import BaseNNXC
    from dqc.utils.safeops import safenorm, safepow
    HAS_DQC = True
    print("✅ DQC available - using production implementation with LibXC")
except ImportError:
    pass  # Silently fallback to simple version


if HAS_DQC:
    # ==========================================
    # PRODUCTION VERSION (with LibXC)
    # ==========================================
    class NeuralB3LYP(BaseNNXC):
        """
        Neural B3LYP - Production Implementation
        
        Integrates with DeepChem's DFT infrastructure using accurate
        LibXC functionals. This version can be used in actual DFT SCF
        calculations.
        
        Parameters
        ----------
        weight_network : torch.nn.Module
            Neural network that predicts mixing weights
        normalize_weights : bool
            Apply softmax normalization to weights
        
        Examples
        --------
        >>> import torch.nn as nn
        >>> from neural_b3lyp import NeuralB3LYP, create_weight_network
        >>> weight_net = create_weight_network([32, 16])
        >>> xc = NeuralB3LYP(weight_net)
        >>> # Use in DFT calculations...
        """
        
        def __init__(self, weight_network: nn.Module, normalize_weights: bool = True):
            super().__init__()
            self.weight_network = weight_network
            self.normalize_weights = normalize_weights
            
            # Initialize LibXC functionals (accurate implementations)
            self.lda_x = get_xc("lda_x")          # Slater exchange
            self.gga_x_b88 = get_xc("gga_x_b88")  # Becke 88
            self.lda_c_vwn = get_xc("lda_c_vwn")  # VWN correlation
            self.gga_c_lyp = get_xc("gga_c_lyp")  # LYP correlation
            
            # Traditional B3LYP weights for reference
            self.register_buffer('traditional_weights', 
                               torch.tensor([0.08, 0.72, 0.19, 0.81]))
        
        @property
        def family(self) -> int:
            """Returns 2 for GGA (uses density and gradient)"""
            return 2
        
        def _extract_features(self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> torch.Tensor:
            """
            Extract features from density information for neural network.
            
            Returns tensor of shape (*batch, n_points, 3) containing:
            - Total density (n)
            - Spin density (ξ)
            - Normalized gradient (s)
            """
            a = 6.187335452560271  # 2 * (3 * π^2)^(1/3)
            
            if isinstance(densinfo, ValGrad):  # Unpolarized
                n = densinfo.value.unsqueeze(-1)
                xi = torch.zeros_like(n)
                
                if densinfo.grad is not None:
                    s = safenorm(densinfo.grad, dim=-1).unsqueeze(-1)
                    n_offset = n + 1e-18
                    s = s / (a * safepow(n_offset, 4.0/3.0))
                else:
                    s = torch.zeros_like(n)
                    
            else:  # Polarized
                nu = densinfo.u.value.unsqueeze(-1)
                nd = densinfo.d.value.unsqueeze(-1)
                n = nu + nd
                n_offset = n + 1e-18
                xi = (nu - nd) / n_offset
                
                if densinfo.u.grad is not None and densinfo.d.grad is not None:
                    s = safenorm(densinfo.u.grad + densinfo.d.grad, dim=-1).unsqueeze(-1)
                    s = s / (a * safepow(n_offset, 4.0/3.0))
                else:
                    s = torch.zeros_like(n)
            
            return torch.cat([n, xi, s], dim=-1)
        
        def get_edensityxc(self, densinfo: Union[ValGrad, SpinParam[ValGrad]]) -> torch.Tensor:
            """
            Compute XC energy density using neural B3LYP.
            
            Parameters
            ----------
            densinfo : Union[ValGrad, SpinParam[ValGrad]]
                Density information from DQC
            
            Returns
            -------
            torch.Tensor
                XC energy density
            """
            # Get component energies from LibXC
            e_lda_x = self.lda_x.get_edensityxc(densinfo)
            e_b88 = self.gga_x_b88.get_edensityxc(densinfo)
            e_vwn = self.lda_c_vwn.get_edensityxc(densinfo)
            e_lyp = self.gga_c_lyp.get_edensityxc(densinfo)
            
            # Neural network predicts weights
            features = self._extract_features(densinfo)
            weights = self.weight_network(features)
            
            if self.normalize_weights:
                weights = torch.softmax(weights, dim=-1)
            
            # Combine with learned weights
            e_xc = (weights[..., 0] * e_lda_x + 
                    weights[..., 1] * e_b88 + 
                    weights[..., 2] * e_vwn + 
                    weights[..., 3] * e_lyp)
            
            return e_xc

else:
    # ==========================================
    # SIMPLIFIED VERSION (fallback)
    # ==========================================
    print("⚠️  DQC not available - loading simplified Neural B3LYP")
    
    try:
        # Try to import the simple version
        from neural_b3lyp_simple import NeuralB3LYP
        print("✅ Simplified Neural B3LYP loaded successfully!")
        
    except ImportError as import_error:
        # Last resort: print helpful error
        print(f"❌ Could not import neural_b3lyp_simple: {import_error}")
        print("\n" + "="*70)
        print("ERROR: Neural B3LYP requires either:")
        print("  1. DQC library: pip install dqc")
        print("  2. neural_b3lyp_simple.py in the same directory")
        print("="*70)
        
        # Create a dummy class that raises informative error
        class NeuralB3LYP:
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "Neural B3LYP requires either DQC library or neural_b3lyp_simple.py.\n"
                    "Please install DQC: pip install dqc\n"
                    "Or ensure neural_b3lyp_simple.py is in the same directory."
                )


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def create_weight_network(hidden_sizes=[32, 16], activation='tanh'):
    """
    Create a feedforward network for predicting B3LYP mixing weights.
    
    Parameters
    ----------
    hidden_sizes : list of int
        Hidden layer sizes
    activation : str
        Activation function ('tanh', 'relu', 'sigmoid')
    
    Returns
    -------
    torch.nn.Module
        Weight prediction network
    
    Examples
    --------
    >>> net = create_weight_network([64, 32, 16])
    >>> xc = NeuralB3LYP(net)
    """
    activation_fn = {
        'tanh': nn.Tanh,
        'relu': nn.ReLU,
        'sigmoid': nn.Sigmoid
    }[activation.lower()]
    
    layers = []
    input_size = 3  # density, spin_density, gradient
    
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(activation_fn())
        input_size = hidden_size
    
    # Output: 4 weights for [lda_x, b88, vwn, lyp]
    layers.append(nn.Linear(input_size, 4))
    
    return nn.Sequential(*layers)


# ==========================================
# EXPORTS
# ==========================================
__all__ = ['NeuralB3LYP', 'create_weight_network', 'HAS_DQC']


# ==========================================
# STANDALONE TEST (if run directly)
# ==========================================
if __name__ == "__main__":
    print("=" * 70)
    print("NEURAL B3LYP - STANDALONE TEST")
    print("=" * 70)
    print(f"\nDQC Available: {HAS_DQC}")
    
    if not HAS_DQC:
        print("\n⚠️  Running with simplified version (DQC not available)")
        
        # Test with simple version
        try:
            xc = NeuralB3LYP()
            density = torch.linspace(0.1, 1.0, 5)
            energy = xc(density)
            print(f"\n✅ Neural B3LYP works!")
            print(f"Energy values: {energy.detach().numpy()}")
        except Exception as e:
            print(f"\n❌ Error: {e}")
    else:
        print("\n✅ Running with production version (LibXC)")
        print("   Use test_neural_b3lyp.py for full testing")
    
    print("\n" + "=" * 70)

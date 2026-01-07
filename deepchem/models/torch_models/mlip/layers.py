import torch
import torch.nn as nn
import numpy as np

def polynomial_envelope(length: torch.Tensor, max_length: float, p: int = 5) -> torch.Tensor:
    """
    Computes the polynomial envelope function ensuring smooth cutoff.
    Matches MACE Eq. 8.
    """
    x = length
    # Normalized distance
    r_norm = x / max_length
    
    # Coefficients
    c1 = (p + 1.0) * (p + 2.0) / 2.0
    c2 = p * (p + 2.0)
    c3 = p * (p + 1.0) / 2.0
    
    term1 = c1 * torch.pow(r_norm, p)
    term2 = c2 * torch.pow(r_norm, p + 1)
    term3 = c3 * torch.pow(r_norm, p + 2)
    
    envelope = 1.0 - term1 + term2 - term3
    
    # Mask values beyond cutoff
    return envelope * (x < max_length).float()

class BesselBasis(nn.Module):
    """
    Bessel Basis function for radial expansion.
    Orthogonal sine waves expanding interatomic distances.
    """
    def __init__(self, r_max: float, num_basis: int = 8):
        super(BesselBasis, self).__init__()
        self.r_max = r_max
        self.num_basis = num_basis
        
        # Precompute frequencies (n * pi)
        n = torch.arange(1, num_basis + 1).float()
        self.register_buffer('n_pi', n * torch.pi)
        
        # Normalization factor sqrt(2/r_max)
        self.prefactor = np.sqrt(2.0 / r_max)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        # Expand for broadcasting [..., 1]
        r_expanded = r.unsqueeze(-1)
        
        # n * pi * r / r_max
        arg = self.n_pi * (r_expanded / self.r_max)
        
        # sin(x)/x (Handling division by zero for stability)
        # We add a tiny epsilon to avoid NaN at r=0, though physical r > 0 usually
        eps = 1e-8
        return self.prefactor * torch.sin(arg) / (r_expanded + eps)

class RadialEmbeddingBlock(nn.Module):
    """
    Combines Bessel Basis and Polynomial Envelope.
    """
    def __init__(self, r_max: float, num_basis: int = 8, p: int = 5):
        super(RadialEmbeddingBlock, self).__init__()
        self.bessel = BesselBasis(r_max, num_basis)
        self.r_max = r_max
        self.p = p

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        basis = self.bessel(r)
        env = polynomial_envelope(r, self.r_max, self.p)
        return basis * env.unsqueeze(-1)
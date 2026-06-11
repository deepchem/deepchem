import pytest
import numpy as np
import torch
import sys
from unittest.mock import MagicMock

# Mock transformers if it is not installed
try:
    import transformers
except ImportError:
    transformers = MagicMock()
    transformers.__path__ = []
    sys.modules["transformers"] = transformers
    sys.modules["transformers.models.auto"] = MagicMock()
    sys.modules["transformers.data"] = MagicMock()
    sys.modules["transformers.data.data_collator"] = MagicMock()

try:
    from deepchem.models.torch_models.differentiable_molecular_dynamics import MolecularDynamics
except ImportError:
    # This might happen if torch is not installed, but we should let pytest handle it via markers or just fail
    pass

@pytest.mark.torch
def test_molecular_dynamics_harmonic_oscillator():
    """
    Test velocity verlet integration with a harmonic oscillator.
    """
    # 1D harmonic oscillator: F = -k * x
    # Mass m = 1
    # k = 1
    # Angular frequency omega = sqrt(k/m) = 1
    # Expected solution: x(t) = x(0) * cos(omega*t) + v(0)/omega * sin(omega*t)
    
    k = 1.0
    m = 1.0
    
    def harmonic_force(r):
        return -k * r
        
    md = MolecularDynamics(time_step=0.01, mass=m, force_fn=harmonic_force)
    
    # Initial conditions
    # Batch size 1, 1 particle, 1 dimension
    r0 = torch.tensor([[[1.0]]], requires_grad=True)
    v0 = torch.tensor([[[0.0]]], requires_grad=True)
    
    # Run for 100 steps -> t = 1.0
    steps = 100
    
    # We call the internal model directly to pass steps, 
    # as fitting the TorchModel interface to this test is unnecessary complexity
    r_final, v_final = md.model(r0, v0, steps=steps)
    
    t_final = 0.01 * steps
    
    # Analytical solution
    x_expected = 1.0 * np.cos(t_final)
    v_expected = -1.0 * np.sin(t_final)
    
    # Check close. Velocity Verlet is 2nd order accurate.
    assert np.allclose(r_final.detach().numpy(), x_expected, atol=1e-2)
    assert np.allclose(v_final.detach().numpy(), v_expected, atol=1e-2)
    
    # Test differentiability
    # If we compute a loss on final state, gradients should flow back to initial state
    loss = (r_final - torch.tensor([[[0.0]]])).sum()
    loss.backward()
    
    assert r0.grad is not None
    assert v0.grad is not None

@pytest.mark.torch
def test_molecular_dynamics_shape():
    """Test that shapes are preserved."""
    def zero_force(r):
        return torch.zeros_like(r)
        
    md = MolecularDynamics(time_step=0.1, mass=1.0, force_fn=zero_force)
    
    batch_size = 5
    n_particles = 10
    n_dims = 3
    
    r0 = torch.randn(batch_size, n_particles, n_dims)
    v0 = torch.randn(batch_size, n_particles, n_dims)
    
    r_final, v_final = md.model(r0, v0, steps=5)
    
    assert r_final.shape == r0.shape
    assert v_final.shape == v0.shape

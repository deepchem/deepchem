import pytest
import numpy as np
import deepchem as dc
import torch

try:
    from deepchem.models.torch_models.kalman_filter import KalmanFilter
    has_torch = True
except ModuleNotFoundError:
    has_torch = False

@pytest.mark.torch
def test_kalman_filter_initialization():
    """Test that KalmanFilter can be initialized."""
    if not has_torch:
        pytest.skip("PyTorch not installed")
        
    state_dim = 2
    obs_dim = 1
    model = KalmanFilter(state_dim=state_dim, observation_dim=obs_dim)
    assert model.state_dim == state_dim
    assert model.observation_dim == obs_dim

@pytest.mark.torch
def test_kalman_filter_fit_predict():
    """Test KalmanModel fit and predict on synthetic data."""
    if not has_torch:
        pytest.skip("PyTorch not installed")

    # Generate synthetic data
    # 10 samples, 20 time steps, 1 observation dimension
    n_samples = 10
    n_steps = 20
    obs_dim = 1
    state_dim = 1
    
    # Create random observations
    X = np.random.randn(n_samples, n_steps, obs_dim).astype(np.float32)
    
    # Create dataset
    dataset = dc.data.NumpyDataset(X=X)

    model = KalmanFilter(state_dim=state_dim, observation_dim=obs_dim, learning_rate=0.01, batch_size=2)

    # Train
    loss = model.fit(dataset, nb_epoch=2)
    assert loss < 100000 # Just checks it doesn't explode
    
    # predict returns filtered_states as it is the first output type
    filtered_states = model.predict(dataset)
    
    assert filtered_states.shape == (n_samples, n_steps, state_dim)

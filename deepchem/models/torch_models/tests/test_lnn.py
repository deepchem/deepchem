"""
Tests for LagrangianNNModel.
These tests check:
1) Overfitting a tiny dataset (ensuring the network can memorize data).
2) Saving/loading the model to preserve learned parameters.
"""

import pytest
import numpy as np

import deepchem as dc
from deepchem.data import NumpyDataset
from deepchem.metrics import Metric
from sklearn.metrics import mean_squared_error

from deepchem.models.torch_models.lnn import LagrangianNNModel

import numpy as np
from scipy.integrate import solve_ivp

def single_pendulum_ode(t, y, g=9.81, L=1.0):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -(g/L) * np.sin(theta)
    return [dtheta_dt, domega_dt]

def generate_single_pendulum_data(num_samples=300, 
                                  t_span=[0, 10], 
                                  dt=0.01,
                                  theta_range=(-np.pi/2, np.pi/2),
                                  omega_range=(-1,1)):
    """
    Generate single pendulum data. 
    Returns X => (theta, omega), y => (ddtheta) as arrays.
    """
    X_data = []
    Y_data = []

    # Time steps
    t_eval = np.arange(t_span[0], t_span[1], dt)

    for _ in range(num_samples):
        # Random initial conditions
        init_theta = np.random.uniform(*theta_range)
        init_omega = np.random.uniform(*omega_range)
        sol = solve_ivp(single_pendulum_ode, t_span, [init_theta, init_omega], t_eval=t_eval)

        # Extract angles, compute numeric accelerations
        thetas = sol.y[0]   # shape (#timesteps,)
        omegas = sol.y[1]

        # Numeric derivative for accelerations
        ddtheta_approx = np.gradient(omegas, dt)

        # For each time step, store X=(theta, omega), y=ddtheta
        for i in range(len(thetas)):
            X_data.append([thetas[i], omegas[i]])
            Y_data.append([ddtheta_approx[i]])

    X_data = np.array(X_data, dtype=np.float32)
    Y_data = np.array(Y_data, dtype=np.float32)
    return X_data, Y_data


def generate_tiny_pendulum_data(num_samples=8, seed=123):
    """
    Generate a small random dataset for testing overfitting and model I/O.
    This is not physically accurate data; we use random values for (theta, omega) 
    and random 'accelerations' as labels to see if the model can memorize them.

    Parameters
    ----------
    num_samples: int
      Number of samples to generate.
    seed: int
      Random seed for reproducibility.

    Returns
    -------
    X: np.ndarray, shape (num_samples, 2)
      Fake pendulum inputs: [theta, omega].
    y: np.ndarray, shape (num_samples, 1)
      Fake 'accelerations' as labels.
    """
    rng = np.random.default_rng(seed)
    # Random angles & velocities in [-1, 1]
    X = rng.uniform(-1, 1, size=(num_samples, 2)).astype(np.float32)
    # Random accelerations in [-0.5, 0.5]
    y = rng.uniform(-0.5, 0.5, size=(num_samples, 1)).astype(np.float32)
    return X, y

@pytest.mark.torch
def test_lnn_overfit():
    """
    Overfit Test:
    Ensures LagrangianNNModel can learn a tiny dataset to a very low MSE.
    Similar approach to other Torch-based overfit tests in DeepChem.
    """
    # Generate a very small dataset of 8 samples
    X, y = generate_tiny_pendulum_data()
    dataset = NumpyDataset(X, y)

    # Create the model
    model = LagrangianNNModel(input_dim=2, hidden_dim=16, learning_rate=1e-3)

    # Attempt to overfit for 200 epochs
    model.fit(dataset, nb_epoch=200)

    # Evaluate MSE on the same dataset to see if it memorized
    mse_metric = Metric(mean_squared_error)
    scores = model.evaluate(dataset, [mse_metric])
    mse_val = scores['mean_squared_error']
    print(f"[Overfit Test] MSE: {mse_val}")

    assert mse_val < 1e-1, f"Overfit test failed. MSE={mse_val}"

@pytest.mark.torch
def test_lnn_save_reload(tmp_path):
    """
    Save & Reload Test:
    - Trains LagrangianNNModel on a small dataset
    - Saves the model (if implemented) or uses save_checkpoint
    - Reloads the model and checks that MSE remains nearly identical

    Notes
    -----
    Some TorchModel versions require overriding save() if 
    not using built-in checkpoint methods.
    """
    X, y = generate_tiny_pendulum_data(num_samples=12, seed=2023)
    dataset = NumpyDataset(X, y)

    # Model dir for temporary files
    model_dir = str(tmp_path / "lnn_model_dir")
    model = LagrangianNNModel(
        input_dim=2,
        hidden_dim=16,
        model_dir=model_dir,
        learning_rate=1e-3
    )

    # Train
    model.fit(dataset, nb_epoch=50)

    # Evaluate pre-save
    mse_metric = Metric(mean_squared_error)
    scores_before = model.evaluate(dataset, [mse_metric])
    mse_before = scores_before['mean_squared_error']
    print(f"[Save/Reload Test] MSE before save: {mse_before}")
    
    try:
        model.save()
    except NotImplementedError:
        pytest.skip("save() not implemented for this TorchModel yet.")

    # Create a new model instance and restore
    new_model = LagrangianNNModel(
        input_dim=2,
        hidden_dim=16,
        model_dir=model_dir,
        learning_rate=1e-3
    )
    try:
        new_model.restore()
    except NotImplementedError:
        # Similarly, skip if restore isn't implemented
        pytest.skip("restore() not implemented for this TorchModel yet.")

    # Evaluate post-restore
    scores_after = new_model.evaluate(dataset, [mse_metric])
    mse_after = scores_after['mean_squared_error']
    print(f"[Save/Reload Test] MSE after restore: {mse_after}")

    # We expect nearly the same MSE
    assert abs(mse_before - mse_after) < 1e-8, (
        f"Model performance changed after reload! before={mse_before}, "
        f"after={mse_after}"
    )

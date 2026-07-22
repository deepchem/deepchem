import numpy as np
import pytest
from deepchem.models.kalman_filter import KalmanFilterModel


def generate_synthetic_data(T=50, seed=0):
    """
    Generate synthetic linear motion data for Kalman Filter testing.
    Model:
        x_k = F * x_{k-1} + w_k
        z_k = H * x_k + v_k
    """
    np.random.seed(seed)

    dim_x = 2
    dim_z = 1

    F = np.array([[1, 1], [0, 1]])
    H = np.array([[1, 0]])
    Q = 1e-4 * np.eye(dim_x)
    R = np.array([[0.1]])
    P0 = np.eye(dim_x)
    x0 = np.array([0, 1])

    x_true = np.zeros((T, dim_x))
    z_obs = np.zeros((T, dim_z))
    x_true[0] = x0

    for t in range(1, T):
        x_true[t] = F @ x_true[t - 1] + np.random.multivariate_normal(
            np.zeros(dim_x), Q)
        z_obs[t] = H @ x_true[t] + np.random.normal(0, np.sqrt(R[0, 0]))

    return dim_x, dim_z, F, H, Q, R, P0, x0, x_true, z_obs


def test_filter_shapes():
    """Ensure KalmanFilter outputs have correct shapes."""
    dim_x, dim_z, F, H, Q, R, P0, x0, x_true, z_obs = generate_synthetic_data()
    model = KalmanFilterModel(dim_x, dim_z, F=F, H=H, Q=Q, R=R, P0=P0, x0=x0)
    xs, Ps = model.filter(z_obs)

    assert xs.shape == (len(z_obs), dim_x)
    assert Ps.shape == (len(z_obs), dim_x, dim_x)


def test_filter_accuracy():
    """Filtered position estimates must be close to ground truth."""
    dim_x, dim_z, F, H, Q, R, P0, x0, x_true, z_obs = generate_synthetic_data(
        T=100)
    model = KalmanFilterModel(dim_x, dim_z, F=F, H=H, Q=Q, R=R, P0=P0, x0=x0)
    xs, _ = model.filter(z_obs)

    mse = np.mean((xs[:, 0] - x_true[:, 0])**2)
    assert mse < 1.0, f"MSE too high: {mse}"


def test_predict_consistency():
    """predict() output must match filter()'s state estimates."""
    dim_x, dim_z, F, H, Q, R, P0, x0, x_true, z_obs = generate_synthetic_data()
    model = KalmanFilterModel(dim_x, dim_z, F=F, H=H, Q=Q, R=R, P0=P0, x0=x0)

    xs_full, _ = model.filter(z_obs)
    xs_pred = model.predict(z_obs)

    np.testing.assert_allclose(xs_full, xs_pred, atol=1e-6)


def test_no_filterpy_fallback(monkeypatch):
    """Ensure NumPy fallback works when filterpy is unavailable."""
    monkeypatch.setattr("deepchem.models.kalman_filter._HAS_FILTERPY", False)

    dim_x, dim_z, F, H, Q, R, P0, x0, x_true, z_obs = generate_synthetic_data()
    model = KalmanFilterModel(dim_x,
                              dim_z,
                              F=F,
                              H=H,
                              Q=Q,
                              R=R,
                              P0=P0,
                              x0=x0,
                              use_filterpy=False)

    xs, _ = model.filter(z_obs)
    assert not np.isnan(xs).any()


def test_covariance_reduction():
    """Uncertainty (covariance) should shrink as measurements accumulate."""
    dim_x, dim_z, F, H, Q, R, P0, x0, x_true, z_obs = generate_synthetic_data(
        T=80)
    model = KalmanFilterModel(dim_x, dim_z, F=F, H=H, Q=Q, R=R, P0=P0, x0=x0)
    _, Ps = model.filter(z_obs)

    assert Ps[-1, 0, 0] < P0[0, 0], "Covariance did not decrease over time"


def test_high_noise_case():
    """Kalman filter should not blow up even with very noisy measurements."""
    np.random.seed(5)
    dim_x, dim_z, F, H, Q, R, P0, x0, x_true, z_obs = generate_synthetic_data(
        T=80)

    R = np.array([[10.0]])

    model = KalmanFilterModel(dim_x, dim_z, F=F, H=H, Q=Q, R=R, P0=P0, x0=x0)
    xs, _ = model.filter(z_obs)

    assert not np.isnan(xs).any(), "Filter failed under high noise"


def test_missing_measurements():
    """KF should gracefully skip update when measurement is NaN."""
    dim_x, dim_z, F, H, Q, R, P0, x0, x_true, z_obs = generate_synthetic_data(
        T=50)

    z_obs[10] = np.nan
    z_obs[20] = np.nan

    model = KalmanFilterModel(dim_x, dim_z, F=F, H=H, Q=Q, R=R, P0=P0, x0=x0)

    xs, _ = model.filter(z_obs)

    assert np.all(np.isfinite(xs)), "Filter failed with NaN measurements"


if __name__ == "__main__":
    pytest.main([__file__])

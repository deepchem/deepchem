"""
Kalman Filter Model
--------------------------------
Implements a basic linear Kalman Filter with configurable matrices.
Supports both filterpy-based and pure NumPy implementations.

References
--------------------------------
    - Kalman, R. E. (1960). "A New Approach to Linear Filtering and Prediction Problems."
      Journal of Basic Engineering.
    - Welch, G. & Bishop, G. (1995). "An Introduction to the Kalman Filter."
      https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf
    - FilterPy documentation (optional backend):
      https://filterpy.readthedocs.io/en/latest/
"""

import numpy as np
from typing import Optional, Tuple

try:
    from filterpy.kalman import KalmanFilter
    _HAS_FILTERPY = True
except ImportError:
    _HAS_FILTERPY = False


class KalmanFilterModel:
    """
    Parameters
    ----------
    dim_x : int
        Dimension of the state vector.
    dim_z : int
        Dimension of the observation vector.
    F : np.ndarray, optional
        State transition matrix.
    H : np.ndarray, optional
        Observation matrix.
    Q : np.ndarray, optional
        Process noise covariance.
    R : np.ndarray, optional
        Measurement noise covariance.
    P0 : np.ndarray, optional
        Initial state covariance.
    x0 : np.ndarray, optional
        Initial state estimate.
    use_filterpy : bool, default=True
        Whether to use `filterpy` if available, otherwise fall back to NumPy.
    """

    def __init__(self,
                 dim_x: int,
                 dim_z: int,
                 F: Optional[np.ndarray] = None,
                 H: Optional[np.ndarray] = None,
                 Q: Optional[np.ndarray] = None,
                 R: Optional[np.ndarray] = None,
                 P0: Optional[np.ndarray] = None,
                 x0: Optional[np.ndarray] = None,
                 use_filterpy: bool = True):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.use_filterpy = use_filterpy and _HAS_FILTERPY

        self.F = np.eye(dim_x) if F is None else F
        self.H = np.eye(dim_z, dim_x) if H is None else H  # FIXED HERE
        self.Q = np.eye(dim_x) if Q is None else Q
        self.R = np.eye(dim_z) if R is None else R
        self.P0 = np.eye(dim_x) if P0 is None else P0
        self.x0 = np.zeros((dim_x,)) if x0 is None else x0

    def _build_filterpy(self):
        """Initialize a filterpy KalmanFilter object."""
        kf = KalmanFilter(dim_x=self.dim_x, dim_z=self.dim_z)
        kf.F = self.F
        kf.H = self.H
        kf.Q = self.Q
        kf.R = self.R
        kf.P = self.P0.copy()
        kf.x = self.x0.reshape((-1, 1))
        return kf

    def filter(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        T = observations.shape[0]
        xs = np.zeros((T, self.dim_x))
        Ps = np.zeros((T, self.dim_x, self.dim_x))

        if self.use_filterpy:
            kf = self._build_filterpy()
            for t in range(T):
                z = observations[t].reshape((-1, 1))
                kf.predict()
                if not np.isnan(z).any():
                    kf.update(z)
                xs[t, :] = kf.x.ravel()
                Ps[t, :, :] = kf.P

        else:
            x = self.x0.reshape((-1, 1))
            P = self.P0.copy()
            identity = np.eye(self.dim_x)

            for t in range(T):
                x = self.F @ x
                P = self.F @ P @ self.F.T + self.Q

                z = observations[t].reshape((-1, 1))
                if not np.isnan(z).any():
                    S = self.H @ P @ self.H.T + self.R
                    K = P @ self.H.T @ np.linalg.inv(S)
                    y = z - self.H @ x
                    x = x + K @ y
                    P = (identity - K @ self.H) @ P

                xs[t, :] = x.ravel()
                Ps[t, :, :] = P

        return xs, Ps

    def predict(self, observations: np.ndarray) -> np.ndarray:
        xs, _ = self.filter(observations)
        return xs

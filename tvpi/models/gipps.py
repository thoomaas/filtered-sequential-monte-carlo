import numpy as np
from .base import Model

class GippsModel(Model):
    def __init__(self):
        self._param_names = ['tau', 'a', 'b', 'b_hat', 'v0', 's0']

    @property
    def n_params(self) -> int:
        return 6

    @property
    def param_names(self) -> list[str]:
        return self._param_names

    @property
    def dynamic_mode_segmentation(self) -> bool:
        return True

    def predict_vectorized(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        # Compute model
        va, vb = self.model_core(x, theta)

        return np.minimum(va, vb)

    def compute_mode(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        # Compute model
        va, vb = self.model_core(x, theta)

        # Mode 1: Acceleration/Free-flow (va <= vb)
        # Mode 2: Braking/Following (vb < va)
        return np.where(va <= vb, 1, 2)

    def model_core(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        # x is (3, K): [vel, range, vel_lead]
        # theta is (K, 6, n_particles)

        vel = x[0]
        dist = x[1]
        vel_lead = x[2]

        # Extract parameters (add extra dimensions for broadcasting)
        tau = theta[:, 0, :]
        a = theta[:, 1, :]
        b = theta[:, 2, :]
        b_hat = theta[:, 3, :]
        v0 = theta[:, 4, :]
        s0 = theta[:, 5, :]

        # Protect denominators to avoid division by zero from bad random particles
        v0_safe = np.where(np.abs(v0) < 1e-3, 1e-3 * np.sign(v0) + (v0==0)*1e-3, v0)
        b_hat_safe = np.where(np.abs(b_hat) < 1e-3, 1e-3 * np.sign(b_hat) + (b_hat==0)*1e-3, b_hat)

        # Acceleration limit
        va = vel[:, None] + 2.5 * a * tau * (1 - vel[:, None]/v0_safe) * np.sqrt(np.maximum(0, 0.025 + vel[:, None]/v0_safe))

        # Braking/Safety limit
        inner_sqrt = b**2 * tau**2 + b * (2*(s0 - dist[:, None]) + vel[:, None]*tau + vel_lead[:, None]**2 / b_hat_safe)
        vb = b * tau + np.sqrt(np.maximum(0, inner_sqrt))

        return va, vb

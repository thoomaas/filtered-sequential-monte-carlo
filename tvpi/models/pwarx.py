import numpy as np
from .base import Model

class PWARXModel(Model):
    """
    PieceWise AutoRegressive eXogenous (PWARX) model implementation.
    The model is linear-in-parameters: y = sum(theta_m^T * [x; 1])
    """

    def __init__(self, n_params: int):
        self._n_params = n_params
        self._param_names = [f"theta_{i+1}" for i in range(n_params)]

    @property
    def n_params(self) -> int:
        return self._n_params

    @property
    def n_inputs(self) -> int:
        return self._n_params - 1

    @property
    def has_bias(self) -> bool:
        return True

    @property
    def param_names(self) -> list[str]:
        return self._param_names

    @property
    def dynamic_mode_segmentation(self) -> bool:
        return False

    def predict_vectorized(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        # Handle the bias term [x; 1] internally
        phi = np.vstack([x, np.ones((1, x.shape[1]))])
        # Re-use the fast einsum logic
        return np.einsum('kps,pk->ks', theta, phi)

    def compute_mode(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        # PWARX relies on static external clustering, this is currently unused.
        raise NotImplementedError("PWARX modes are static and provided by the dataset.")

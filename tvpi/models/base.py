from abc import ABC, abstractmethod
import numpy as np

class Model(ABC):
    """
    Abstract Base Class for physical models.
    """

    @abstractmethod
    def predict_vectorized(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Batch calculation for all particles across multiple time steps.
        Returns: (K, n_particles) - Predicted outputs
        """
        pass

    @abstractmethod
    def compute_mode(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Computes the active mode for each time step and particle.

        Args:
            x: Regressor matrix of shape (n_inputs, K)
            theta: Parameter tensor of shape (K, n_params, n_particles)

        Returns:
            modes: Integer array of shape (K, n_particles) indicating the active mode (1-indexed).
        """
        pass

    @property
    @abstractmethod
    def dynamic_mode_segmentation(self) -> bool:
        """
        True: Modes depend on parameters and must be recalculated every iteration (e.g., Gipps).
        False: Modes are static/pre-clustered from data and do not change (e.g., PWARX).
        """
        pass

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Total number of parameters."""
        pass

    @property
    @abstractmethod
    def n_inputs(self) -> int:
        """Number of input regressors expected by the model."""
        pass

    @property
    @abstractmethod
    def has_bias(self) -> bool:
        """Whether the model includes an additive bias term in its parameters."""
        pass

    @property
    @abstractmethod
    def param_names(self) -> list[str]:
        """Returns the names of the model parameters."""
        pass
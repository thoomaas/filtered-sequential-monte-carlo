import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, Any

class DataProcessor:
    """
    Handles loading from Excel files, normalization, and mode clustering.
    Direct port of MATLAB's "Data from file case".
    """

    def __init__(self, cluster_acc: float = 0.35, cluster_dec: float = -0.35):
        self.cluster_acc = cluster_acc
        self.cluster_dec = cluster_dec
        self.norm_factors = {}

    def prepare_external_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Loads data from Excel/CSV, applies delay, and performs clustering.
        config keys: 'file_path', 'y_column', 'x_columns', 'delay', 'clustering'
        """
        file_path = config['file_path']
        delay = config.get('delay', 0)

        # Load file based on extension
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        # Extract y (output)
        y_col = config['y_column']
        y_raw = df.iloc[:, y_col].values if isinstance(y_col, int) else df[y_col].values

        # Extract x (regressors)
        x_cols = config['x_columns']
        if isinstance(x_cols[0], int):
            x_raw = df.iloc[:, x_cols].values.T
        else:
            x_raw = df[x_cols].values.T

        # Apply delay (aligning x_k with y_{k+delay})
        K = len(y_raw)
        x = x_raw[:, :K-delay]
        y = y_raw[delay:]

        # Clustering
        if config.get('clustering') == 'kmeans':
            modes = self.kmeans_clustering(y, n_modes=config['n_modes'])
        else:
            modes = self.manual_clustering(y)

        return {
            'x': x,
            'y': y,
            'mode': modes,
        }

    def get_signal_stats(self, signal: np.ndarray):
        # Use percentiles to be robust to outliers
        s_min = np.percentile(signal, 0.5)
        s_max = np.percentile(signal, 99.5)
        s_range = s_max - s_min
        if s_range == 0: s_range = 1.0
        return s_min, s_range

    def manual_clustering(self, y: np.ndarray) -> np.ndarray:
        """
        Manual mode clustering based on output thresholds.
        """
        modes = np.full(y.shape, 2, dtype=int)
        modes[y < self.cluster_dec] = 1 # Dec case (MATLAB uses 1 for < dec)
        modes[y > self.cluster_acc] = 3 # Acc case (MATLAB uses 3 for > acc)
        return modes

    def kmeans_clustering(self, y: np.ndarray, n_modes: int = 3) -> np.ndarray:
        """
        Automatic clustering using K-Means.
        """
        kmeans = KMeans(n_clusters=n_modes, random_state=40)
        modes = kmeans.fit_predict(y.reshape(-1, 1)) + 1 # 1-indexed to match MATLAB
        return modes

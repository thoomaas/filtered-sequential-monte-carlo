import os
import numpy as np
import json

def load_config(config_path: str) -> dict:
    """Loads a configuration JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def compare_params(current: np.ndarray, expected: np.ndarray, rtol=1e-5, atol=1e-8) -> bool:
    """Compares two parameter arrays using numpy.allclose."""
    return np.allclose(current, expected, rtol=rtol, atol=atol)

def save_baseline(case_dir, estimates: np.ndarray, config: dict):
    """Saves the current estimates as the new 'golden' baseline."""
    os.makedirs(case_dir, exist_ok=True)
    np.save(os.path.join(case_dir, "expected.npy"), estimates)
    with open(os.path.join(case_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))

def load_baseline(case_dir):
    """Loads the expected baseline for comparison."""
    expected_path = os.path.join(case_dir, "expected.npy")
    if not os.path.exists(expected_path):
        return None
    return np.load(expected_path)

def get_case_dirs(base_dir="tests/regression/cases"):
    """Returns a list of all test case directories."""
    if not os.path.exists(base_dir):
        return []
    return [os.path.join(base_dir, d) for d in os.listdir(base_dir) 
            if os.path.isdir(os.path.join(base_dir, d))]

import sys
import os
import numpy as np
import json
import argparse
import random

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tvpi.models.pwarx import PWARXModel
from tvpi.models.gipps import GippsModel
from tvpi.core.optim import BSMC, FilteredSMC, ConstantSMC
from tvpi.data.generator import generate_theoretical_example
from tests.regression.utils import load_baseline, compare_params

def sanitize_config(config):
    """Recursively converts string keys to integers and lists to numpy arrays where expected."""
    if isinstance(config, list):
        return [sanitize_config(x) for x in config]
    if not isinstance(config, dict):
        return config
    
    new_dict = {}
    for k, v in config.items():
        new_key = int(k) if isinstance(k, str) and k.isdigit() else k
        
        # Convert specific keys to numpy arrays if they are lists
        if k in ['sigma_obs', 'sigma_p', 'theta_true', 'initial_param_range'] and isinstance(v, list):
            v = np.array(v)
        
        new_dict[new_key] = sanitize_config(v)
    return new_dict

def run_case(case_dir, update_baseline=False):
    """Runs a single test case and compares with baseline."""
    config_path = os.path.join(case_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"Error: config.json not found in {case_dir}")
        return False

    with open(config_path, 'r') as f:
        raw_config = json.load(f)

    # Sanitize config (JSON keys are always strings, but optimizer expects ints for modes)
    model_type = raw_config.get('model_type', 'PWARX')
    ident_params = sanitize_config(raw_config['identification_params'])
    model_params = sanitize_config(raw_config['model_parameters'])
    learning_data = sanitize_config(raw_config.get('learning_data', {}))
    
    # Set seed GLOBALLY before instantiating anything
    seed = ident_params.get('random_seed')
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Instantiate Model
    if model_type == 'PWARX':
        model = PWARXModel(n_params=model_params['number_params'])
    elif model_type == 'Gipps':
        model = GippsModel()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Generate Data
    if not learning_data:
        # Default fallback
        learning_data = {
            'data_type': 0,
            'sigma_noise': 0.05,
            'synth_data_time': np.arange(-2.3, 2.79, 0.01),
            'synth_param_constant': [[1, 0.5], [-1.0, 2.0]],
            'synth_param_variable_speed': [[12, 0], [6, 0]],
            'synth_param_variable_value': [[1/6.0, 1], [1/4.0, 1]],
        }
    else:
        # Ensure synth_data_time is a numpy array
        if 'synth_data_time' in learning_data:
            learning_data['synth_data_time'] = np.array(learning_data['synth_data_time'])

    data = generate_theoretical_example(model_params, learning_data, model)

    # Instantiate Scheme
    ident_type = ident_params['identification_type']
    if ident_type == 0:
        scheme = BSMC(model, ident_params, model_params)
    elif ident_type == 1:
        scheme = FilteredSMC(model, ident_params, model_params)
    elif ident_type == 2:
        scheme = ConstantSMC(model, ident_params, model_params)
    else:
        raise ValueError(f"Unsupported identification type: {ident_type}")

    # Run Scheme
    results = scheme.run(data)
    
    # Extract Estimates for comparison
    # BSMC (type 0) returns the results list directly.
    # FilteredSMC (type 1) and ConstantSMC (type 2) return a tuple (results, ...)
    if ident_type in [1, 2]:
        results_list = results[0]
    else:
        results_list = results
        
    K = len(results_list)
    n_modes = model_params['number_modes']
    n_params = model_params['number_params']
    current_estimates = np.zeros((n_modes, n_params, K))
    for k in range(K):
        for m in range(1, n_modes + 1):
            current_estimates[m-1, :, k] = results_list[k]['estimates'][m]

    if update_baseline:
        np.save(os.path.join(case_dir, "expected.npy"), current_estimates)
        print(f"Updated baseline for {case_dir}")
        return True

    expected_estimates = load_baseline(case_dir)
    if expected_estimates is None:
        print(f"Baseline not found for {case_dir}. Run with --update-baseline to create it.")
        return False

    if compare_params(current_estimates, expected_estimates):
        print(f"PASS: {case_dir}")
        return True
    else:
        print(f"FAIL: {case_dir}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("case_dir", help="Directory of the test case")
    parser.add_argument("--update-baseline", action="store_true", help="Update the 'expected' results")
    args = parser.parse_args()
    run_case(args.case_dir, args.update_baseline)

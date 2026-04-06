# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pandas",
#     "scipy",
#     "scikit-learn",
#     "matplotlib",
#     "seaborn",
#     "tqdm",
#     "openpyxl",
#     "pyinstrument"
# ]
# ///

import os
import datetime
import numpy as np

# Import TVPI Core modules
from tvpi.models.pwarx import PWARXModel
from tvpi.core.optim import FilteredSMC
from tvpi.data.generator import generate_theoretical_example
from tvpi.core.plotting import (
    plot_results,
    plot_iteration_evolution,
    plot_final_with_uncertainty,
    plot_convergence_metrics,
    plot_identification_performance
)

def main():
    print("=" * 60)
    print(" QUICKLAUNCH: PWARX Model with Filtered SMC (FSMC)")
    print("=" * 60)

    # 1. Base configuration
    model_parameters = {
        'number_modes': 2,
        'number_params': 2,
    }

    learning_data = {
        'data_type': 0, # Synthetic data
        'sigma_noise': 0.05,
        'synth_data_time': np.arange(-2.3, 2.79, 0.01),
        'synth_param_constant': [[1, 0.5], [-1.0, 2.0]],
        'synth_param_variable_speed': [[12, 0], [6, 0]],
        'synth_param_variable_value': [[1/6.0, 1], [1/4.0, 1]],
    }

    identification_params = {
        'identification_type': 1, # 1 for Filtered SMC
        'resampling_type': 1,
        'n_resample': 10,
        'sample_mult': 10,
        'initial_param_range': (-5.0, 5.0),
        'varying_ident_params': {1: (0,), 2: (0,)},
        'sigma_p': 0.05,
        'precision': 'float64',
        'likelihood_type': 'cauchy',
        'sigma_obs': 0.5,

        # Specific parameters for Filtered SMC
        'use_constant_init': 0,
        'sigma_g': {
            1: np.array([2.0, 0.0]),
            2: np.array([4.0, 0.0])
        },
        'filtered_anchors': True,
        'point_estimate_extraction_share': 0.2,
        'n_iterations': 20,
        'max_iterations': 100,
        'convergence_threshold': 0.3,
        'convergence_patience': 5,
        'sigma_window_multiplier': 3.0,
    }

    # 2. Output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"quicklaunch_fsmc_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"\n[INFO] Saving results to: {results_dir}\n")

    # 3. Model & Data Initialization
    model = PWARXModel(n_params=model_parameters['number_params'])
    model_parameters['number_params'] = model.n_params

    data = generate_theoretical_example(model_parameters, learning_data, model)
    identification_params['theta_true'] = data['theta_true']

    # 4. Identification Process
    fsmc = FilteredSMC(
        model=model,
        identification_params=identification_params,
        model_parameters=model_parameters
    )

    print("\n[INFO] Starting Filtered SMC Parameter Identification...\n")
    results, error_history, history_estimates, change_history = fsmc.run(data)

    # 5. Save & Plot Results
    fsmc.save_results(results, results_dir)

    plot_results(data, results, save_path=os.path.join(results_dir, 'fsmc_identification_results.png'))
    plot_iteration_evolution(data, history_estimates, save_path=os.path.join(results_dir, 'fsmc_identification_results_iterations.png'))
    plot_final_with_uncertainty(data, results, save_path=os.path.join(results_dir, 'fsmc_final_params_with_uncertainty.png'))
    plot_convergence_metrics(error_history, change_history, save_path=os.path.join(results_dir, 'fsmc_algorithm_convergence.png'))
    plot_identification_performance(data, results, model, save_path=os.path.join(results_dir, 'identification_performance.png'))

    print("\n[SUCCESS] Identification complete! Check the results folder for plots.")

if __name__ == "__main__":
    main()

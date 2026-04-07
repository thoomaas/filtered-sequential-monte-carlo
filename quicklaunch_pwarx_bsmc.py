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
from tvpi.core.optim import BSMC
from tvpi.data.generator import generate_theoretical_example
from tvpi.core.plotting import (
    plot_results, 
    plot_bsmc_results, 
    plot_final_with_uncertainty
)

def main():
    print("=" * 60)
    print(" QUICKLAUNCH: PWARX Model with Bayesian SMC (BSMC)")
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
        'identification_type': 0, # 0 for BSMC
        'resampling_type': 1,
        'n_resample': 100,
        'sample_mult': 100,
        'initial_param_range': (-5.0, 5.0),
        'varying_ident_params': {1: (0,), 2: (0,)},
        'sigma_p': 0.05,
        'precision': 'float64',
        'likelihood_type': 'cauchy',
        'sigma_obs': 0.5,
    }

    # 2. Output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"quicklaunch_bsmc_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"\n[INFO] Saving results to: {results_dir}\n")

    # 3. Model & Data Initialization
    model = PWARXModel(n_inputs=model_parameters['number_params'] - 1)
    model_parameters['number_params'] = model.n_params

    data = generate_theoretical_example(model_parameters, learning_data, model)
    identification_params['theta_true'] = data['theta_true']

    # 4. Identification Process
    bsmc = BSMC(
        model=model,
        identification_params=identification_params,
        model_parameters=model_parameters
    )
    
    print("\n[INFO] Starting BSMC Parameter Identification...\n")
    results = bsmc.run(data)

    # 5. Save & Plot Results
    bsmc.save_results(results, results_dir)
    
    plot_bsmc_results(data, results, save_path=os.path.join(results_dir, 'bsmc_initial_params.png'))
    plot_results(data, results, save_path=os.path.join(results_dir, 'bsmc_identification_results.png'))
    plot_final_with_uncertainty(data, results, save_path=os.path.join(results_dir, 'bsmc_final_params_with_uncertainty.png'))
    
    print("\n[SUCCESS] Identification complete! Check the results folder for plots.")

if __name__ == "__main__":
    main()

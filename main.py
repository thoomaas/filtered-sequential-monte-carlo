import os
import numpy as np
import datetime
from tvpi.data.generator import generate_theoretical_example
from tvpi.data.processor import DataProcessor
from tvpi.models.pwarx import PWARXModel
from tvpi.models.gipps import GippsModel
from tvpi.core.optim import BSMC, FilteredSMC, ConstantSMC
from tvpi.core.plotting import (
    plot_results, plot_bsmc_results, plot_final_with_uncertainty,
    plot_convergence, plot_iteration_evolution, plot_convergence_metrics,
    plot_optimization_diagnostics, plot_identification_performance
)

def main():
    ## CONFIGURATION

    # Model selection. You can add you own custom models.
    model_type = 'PWARX' # Type of model 'PWARX' or 'Gipps'
    # model_type = 'Gipps' # Type of model 'PWARX' or 'Gipps'

    # Parameters identification scheme
    identification_params = {
        # Type of parameter identification scheme
        #  0 for time-varying BSMC (Bayesian Sequential Monte-Carlo)
        #  1 for Fileterd Sequential Monte-Carlo
        #  2 for Constant parameter SMC
        'identification_type': 1,
        # Type of resampling methodology
        #  0 for 'deterministic' (Top-N) (faster and cleaner filtering but looses particles diversity). Recommended for constant parameter identification.
        #  1 for 'stochastic' with elitism (slower but keeps particles diversity). Recommended for time-varying parameter identification.
        'resampling_type': 1,
        # Number of initial particles. >=100 for BSMC, >=10 for filtered SMC. [int]
        'n_resample': 10,
        # Sample multiplier. >=100 for BSMC, >=10 for filtered SMC. [int]
        'sample_mult': 10,
        # Range of the initial particles, in the amplitude of the fitted data. [tuple (min, max)]
        'initial_param_range': (-5.0, 5.0),
        # Index of the identified parameters. [ None or dict {mode_index:(parameter_index,),}]
        'varying_ident_params': {1:(0,), 2: (0,)},
        # Standard deviation of new particles sampling (gaussian distribution). Convergence speed VS accuracy. [float or tuple]
        # Normalized to each mode's signal amplitude if a float, in the physical units in a tuple.
        'sigma_p': 0.05,
        # Numerical precision. 'float32' to reduce memory usage and improve speed on large datasets. Default: 'float64'
        'precision': 'float64',
        # Particles weighting likelihood_type options:
        #  'gaussian': fastest convergence for data with small gaussian noise. Requires good initial parameters due to fast decay to 0.
        #  'laplace': L1 optim, for occasional large outliers.
        #  'cauchy': slowest but robust to noise.
        #  'robust': abs(1/(error+1))**0.5
        'likelihood_type': 'cauchy',
        # Distribution standard deviation used for particles weighting, normalized to each mode's signal error. [float]
        'sigma_obs': 0.5,
        # Real value of parameters in the case of synthetic data generation. These values are used for non-identified parameters with synthetic data. [np.array]
        'theta_true': None,

        ########################################
        # Specific parameters for Filtered SMC #
        ########################################
        # 0 to start from scratch, 1 to first run a constant parameters indenfitication. [int]
        'use_constant_init': 0,
        # Standard deviation of the filtering process per mode and parameter. [None or dict {mode_index:(array of signmags,),}]
        'sigma_g': {
            1: np.array([2.0, 0.0]),
            2: np.array([4.0, 0.0])
        },
        # Anchor stabilization using gaussian filtering on max estimates with sigma_g. False for no, True for yes. [bool]
        'filtered_anchors': True, # Default True
        # 0.0 for max based time point estimate (like in journal publications), up to 1.0 for weighted mean of all particles. [float]
        'point_estimate_extraction_share': 0.2, # Default 0.2
        # Number of iterations of the filtered SMC parameters identification process, or 'auto' [int or string]
        'n_iterations': 20,
        # Safety limit if 'auto' is used [int]
        'max_iterations': 100,
        # Max allowed parameter change (relative to sigma_p) [float]
        'convergence_threshold': 0.3,
        # Number of consecutive stable iterations required [int]
        'convergence_patience': 5,
        # Possibility to limit the window of time smoothing (avoid using all the time steps). Faster than using all time steps. None or positive float. Default: 4.0. [float]
        'sigma_window_multiplier': 4.0,

        ########################################
        # Specific parameters for Constant SMC #
        ########################################
        'cst_ident_params': {1:(0, 1,), 2: (0, 1,)}, # Index of the identified parameters. None or dict {mode_index:(parameter_index,),}
        'cst_n_iterations': 500,        # Total number of batch optimization loops
        'cst_n_stoRegV': 500,           # Number of data samples to evaluate likelihood
        'cst_sigma_p_initial': 1.0,     # Initial noise amplitude for particle generation
        'cst_sigma_p_min': 0.0001,      # Minimum noise limit
        'cst_decay_rate': 0.99,         # Multiplier to shrink sigma_p at each iteration
        'cst_n_resample': 100,           # Number of particles per mode
        'cst_sample_mult': 10,          # Multiplier for new particle generation
        'n_pre_samples': 100,           # Noise pre-sampling
    }

    # Indentified model parameters
    model_parameters = {
        'number_modes': 2,      # Number of modes of the model
        'number_params': 2,     # Number of identified parameters of the model
    }

    # Learning data type
    learning_data = {
        'data_type': 0,     # 0 for synthetic data, 1 for external data (Excel file)
        'external_data': {
            'file_path': 'data/follow2_O_can_20151117133115_2nd_loop.xlsx_readyForSimulation_soft.xlsx',
            'y_column': 2,        # Example column index or name
            'x_columns': [2, 4],  # List of column indices or names
            'delay': 4,           # Delay in time steps
            'clustering': 'manual', # 'manual' or 'kmeans'
            'cluster_acc': 0.35,
            'cluster_dec': -0.35,
            'n_modes': 3          # For kmeans clustering
        },
        'sigma_noise': 0.05, # Synthetic data additive noise standard deviation (gaussian distribution)
        'synth_data_time': np.arange(-2.3, 2.79, 0.01), # Data time vector for synthetic data generation
        'synth_param_constant': [[1, 0.5], [-1.0, 2.0]], # Parameters constant values for data parameters generation [[mode 1], [mode 2]]
        'synth_param_variable_speed': [[12, 0], [6, 0]],    # Sinus speed (ex: theta_var[0,0,:]=np.sin(x_range * 12))
        'synth_param_variable_value': [[1/6.0, 1], [1/4.0, 1]], # Sinus amplitude (ex: theta_var[0,0,:]=np.sin(x_range * 12)/6.0)
    }

    # Generate timestamped directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"--- Identification started. Saving to: {results_dir} ---")

    ## Data importation
    if model_type == 'Gipps':
        if learning_data['data_type'] == 1:
            learning_data['external_data']['x_columns'] = [2, 4, 5]

    if learning_data['data_type'] == 1:
        processor = DataProcessor(
            cluster_acc=learning_data['external_data'].get('cluster_acc', 0.35),
            cluster_dec=learning_data['external_data'].get('cluster_dec', -0.35)
        )
        data = processor.prepare_external_data(learning_data['external_data'])

    ## Model Initialization & Configuration
    if model_type == 'Gipps':
        # Gipps is a global model -> Force 1 Mode
        model_parameters['number_modes'] = 1
        model_parameters['number_params'] = 6

        if learning_data['data_type'] == 0:
            learning_data['synth_param_constant'] = [[1.0, 1.5, -2.0, -2.5, 30.0, 2.0]]
            learning_data['synth_param_variable_speed'] = [[0, 0, 1.0, 0, 0, 0]]
            learning_data['synth_param_variable_value'] = [[0, 0, 0.5, 0, 0, 0]]
            learning_data['synth_data_time'] = np.arange(0, 60, 0.4)

        # Physical Noise Steps for Gipps: [tau, a, b, b_hat, v0, s0]
        identification_params['sigma_p'] = [0.1, 0.2, 0.2, 0.2, 1.0, 2.0]

        # For Constant SMC (Init): Identify all 6 parameters
        # It has 2 internal behavioral states (1: Accel, 2: Brake)
        # 0: tau, 1: a, 2: b, 3: b_hat, 4: v0, 5: s0
        identification_params['cst_ident_params'] = {1: (0, 1, 2, 3, 4, 5)}

        # For BSMC and Filtered SMC: Only vary parameter 'b' (index 2)
        # It has 2 internal behavioral states (1: Accel, 2: Brake)
        # 0: tau, 1: a, 2: b, 3: b_hat, 4: v0, 5: s0
        identification_params['varying_ident_params'] = {1: (2,)}

        # Add a new specific dictionary for Behavioral State visibility
        identification_params['state_visibility'] = {
            1: (0, 1, 4),       # Accel state
            2: (0, 2, 3, 5, 4)  # Brake state
        }

        # Apply smoothing filter only to the varying parameter 'b'
        # The array matches the 6 parameters. We put 5.0 at index 2, and 0.0 for fixed parameters.
        identification_params['sigma_g'] = {1: np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])}

        model = GippsModel()
        if learning_data['data_type'] == 1:
            # Override the data modes so the optimizer doesn't split the data
            data['mode'] = np.ones_like(data['mode'])
    elif model_type == 'PWARX':
        # PWARX is a hybrid model -> Modes come from data clustering
        if learning_data['data_type'] == 1:
            # PWARX is a hybrid model -> Modes come from data clustering
            model_parameters['number_modes'] = len(np.unique(data['mode']))
            n_inputs = data['x'].shape[0]

            # We must configure parameters for every mode found in the data
            pwarx_identified_cst = (0,1,2,)
            pwarx_identified_var = (0,)
            identification_params['varying_ident_params'] = {}
            identification_params['cst_ident_params'] = {}
            identification_params['sigma_g'] = {}

            for m in range(1, model_parameters['number_modes'] + 1):
                identification_params['varying_ident_params'][m] = pwarx_identified_var
                identification_params['cst_ident_params'][m] = pwarx_identified_cst
                identification_params['sigma_g'][m] = np.array([10.0])

        else:
            # For synthetic data, we use the values from the config dict
            n_inputs = model_parameters['number_params'] - 1

        model = PWARXModel(n_inputs=n_inputs)

    # Assign the exact parameter count directly from the initialized model
    model_parameters['number_params'] = model.n_params

    ## Data creation
    if learning_data['data_type']==0:
        data = generate_theoretical_example(model_parameters, learning_data, model)
        identification_params['theta_true'] = data['theta_true']

    ## Parameters identification
    if identification_params['identification_type']==0:
        # Call optimizer
        bsmc = BSMC(
            model = model,
            identification_params = identification_params,
            model_parameters = model_parameters
        )
        print("Running Initial BSMC...")
        results = bsmc.run(data)

        # Save results
        bsmc.save_results(results, results_dir)

        # BSMC Initial Plot
        plot_bsmc_results(data, results, save_path=os.path.join(results_dir, 'bsmc_initial_params.png'))
        print("BSMC initial parameters plot saved to bsmc_initial_params.png")

        # Generic Results Plot
        plot_results(data, results, save_path=os.path.join(results_dir, 'bsmc_identification_results.png'))
        print("General identification results saved to identification_results.png")

        # Final Results with Uncertainty
        plot_final_with_uncertainty(data, results, save_path=os.path.join(results_dir, 'bsmc_final_params_with_uncertainty.png'))
        print("Final parameters plot with uncertainty saved to bsmc_final_params_with_uncertainty.png")

    elif identification_params['identification_type']==1:
        theta_init = None
        if identification_params['use_constant_init']:
            # Set as constant param ident
            identification_params['identification_type'] = 2
            # Call optimizer
            cst_smc = ConstantSMC(
                model = model,
                identification_params = identification_params,
                model_parameters = model_parameters
            )
            cst_results, *_ = cst_smc.run(data)
            # Extract the final constant vector for each mode
            theta_init = cst_results[0]['estimates']
            print("-> Constant Initialization Complete.")

        # Set as constant filtered BMC
        identification_params['identification_type'] = 1
            # Call optimizer
        fsmc = FilteredSMC(
            model = model,
            identification_params = identification_params,
            model_parameters = model_parameters
        )
        print(f"Running Filtered SMC ({identification_params['n_iterations']} iterations)...")
        results, error_history, history_estimates, change_history = fsmc.run(data, theta_init)

        # Save results
        fsmc.save_results(results, results_dir)

        # Final Results with Uncertainty
        plot_final_with_uncertainty(data, results, save_path=os.path.join(results_dir, 'fsmc_final_params_with_uncertainty.png'))
        print("Final parameters plot with uncertainty saved to fsmc_final_params_with_uncertainty.png")

        # Convergence Plot
        if identification_params['n_iterations']=='auto':
            plot_convergence_metrics(error_history, change_history, threshold=identification_params['convergence_threshold'], save_path=os.path.join(results_dir, 'fsmc_algorithm_convergence.png'))
        else:
            plot_convergence(error_history, save_path=os.path.join(results_dir, 'fsmc_algorithm_convergence.png'))
        print("Algorithm convergence plot saved to fsmc_algorithm_convergence.png")

        # Generic Results Plot
        plot_results(data, results, save_path=os.path.join(results_dir, 'fsmc_identification_results.png'))
        print("General identification results saved to fsmc_identification_results.png")

        # Iterative Results Plot
        plot_iteration_evolution(data, history_estimates, save_path=os.path.join(results_dir, 'fsmc_identification_results_iterations.png'))
        print("Identification results history saved to fsmc_identification_results_iterations.png")

    elif identification_params['identification_type']==2:
        cst_smc = ConstantSMC(
            model = model,
            identification_params = identification_params,
            model_parameters = model_parameters
        )
        print(f"Running Constant SMC...")
        # Receive the histories
        results, error_history, history_estimates, change_history, sigma_history = cst_smc.run(data)

        if learning_data['data_type'] == 1: # Only if using Excel
            print("\n--- Identified Parameters (Physical Units) ---")
            # Extract from first result entry
            final_theta = results[0]['estimates']
            for m, theta in final_theta.items():
                # Data was never normalized, so theta is already physical!
                print(f"  Mode {m}: {theta}")

        cst_smc.save_results(results, results_dir)

        # Plot Results
        plot_results(data, results, save_path=os.path.join(results_dir, 'csmc_identification_results.png'))
        print("General identification results saved to csmc_identification_results.png")

        plot_iteration_evolution(data, history_estimates, save_path=os.path.join(results_dir, 'csmc_identification_results_iterations.png'))
        print("Identification results history saved to csmc_identification_results_iterations.png")

        plot_optimization_diagnostics(error_history, sigma_history, save_path=os.path.join(results_dir, 'cst_algorithm_convergence.png'))
        print("Algorithm convergence plot saved to cst_algorithm_convergence.png")


    plot_identification_performance(data, results, model, save_path=os.path.join(results_dir, 'identification_performance.png'))
    print(f"Performance plot saved to identification_performance.png")

if __name__ == "__main__":
    main()

import os
import json
import numpy as np
from typing import Dict, Any, List
from ..models.base import Model
from scipy.ndimage import convolve1d
import time
from tqdm import trange
from tvpi.data.processor import DataProcessor

class IdentificationScheme:
    def __init__(self, model, identification_params, model_parameters):
        self.model = model
        self.identification_params = identification_params
        self.n_modes = model_parameters['number_modes']
        self.n_params = model_parameters['number_params']
        self.sigma_p = identification_params['sigma_p']
        self.sigma_obs = identification_params['sigma_obs']
        self.likelihood_type = identification_params['likelihood_type']
        self.identification_type = identification_params['identification_type']
        self.resampling_type = identification_params['resampling_type']
        self.theta_true = identification_params.get('theta_true', None)
        self.dtype = np.float32 if identification_params['precision'] == 'float32' else np.float64
        self.processor = DataProcessor()

        # Run coherence check
        self._validate_dimensions()
        # Optim problem general summary
        self._display_config_summary()

    def _display_config_summary(self):
        """Displays a summary of the identification problem setup."""
        print("\n" + "="*50)
        print("IDENTIFICATION PROBLEM SUMMARY")
        print("-" * 50)
        print(f"Number of Input Regressors: {self.model.n_inputs}")
        print(f"Number of Modes:           {self.n_modes}")
        bias_str = " (incl. bias)" if self.model.has_bias else ""
        print(f"Total Parameters per Mode: {self.n_params}{bias_str}")

        print("\nFitted Parameters by Mode:")
        for m in range(1, self.n_modes + 1):
            fitted = list(self.varying_params.get(m, []))
            fixed = [p for p in range(self.n_params) if p not in fitted]

            status = f"Mode {m}: {len(fitted)} fitted {fitted}"
            if fixed:
                source = "theta_true" if self.theta_true is not None else "0.0 (default)"
                status += f" | {len(fixed)} fixed to {source}"
            print(status)

        print("-" * 50)
        print(f"Likelihood:   {self.likelihood_type} (sigma_obs: {self.sigma_obs})")
        print(f"Resampling:   {'Deterministic' if self.resampling_type==0 else 'Stochastic'}")
        print("="*50 + "\n")

    def _set_scaled_sigma(self, data: Dict[str, Any]) -> Dict[str, Any]:
        y_min, y_range = self.processor.get_signal_stats(data['y'])

        # If sigma_p is a list or array, use it directly (Custom physical units)
        if isinstance(self.sigma_p, (list, np.ndarray)):
            self.sigma_p_scaled = np.array(self.sigma_p, dtype=self.dtype)
            # Ensure it matches model params
            if len(self.sigma_p_scaled) != self.n_params:
                raise ValueError(f"sigma_p array length ({len(self.sigma_p_scaled)}) must match model n_params ({self.n_params})")
        else:
            # Fallback auto-scaling for models lacking explicit sigma_p arrays
            n_regressors = data['x'].shape[0]
            x_min = np.zeros(n_regressors)
            x_range = np.zeros(n_regressors)
            for i in range(n_regressors):
                x_min[i], x_range[i] = self.processor.get_signal_stats(data['x'][i, :])

            self.sigma_p_scaled = np.zeros(self.n_params, dtype=self.dtype)
            for i in range(min(n_regressors, self.n_params)):
                self.sigma_p_scaled[i] = self.sigma_p * (y_range / (x_range[i] + 1e-6))
            # Handle bias term if present
            if self.model.has_bias and self.n_params > n_regressors:
                self.sigma_p_scaled[-1] = self.sigma_p * y_range

        # Needed by the processor later
        self.norm_y_min, self.norm_y_range = y_min, y_range

    def _validate_dimensions(self):
        """
        Ensures identification_params are coherent with n_modes and n_params.
        Normalizes varying_params and sigma_g into standard dictionary formats.
        """
        # 1. Normalize varying_params (Dict of tuples)
        if self.identification_type==2:
            vp = self.identification_params.get('cst_ident_params')
        else:
            vp = self.identification_params.get('varying_ident_params')

        self.varying_params = {}
        if vp is not None:
            for state_or_mode, params in vp.items():
                self.varying_params[int(state_or_mode)] = tuple(p for p in params if p < self.n_params)
        else:
            # Default: all parameters vary for all modes
            for m in range(1, self.n_modes + 1):
                self.varying_params[m] = tuple(range(self.n_params))

        # 2. Normalize sigma_g (Dict of arrays)
        sg = self.identification_params.get('sigma_g')
        if sg is not None:
            self.sigma_g = {}
            if isinstance(sg, dict):
                for m in range(1, self.n_modes + 1):
                    # Get existing or default to zeros
                    orig_sig = np.atleast_1d(sg.get(m, 0.0))
                    # Pad or truncate to match n_params
                    new_sig = np.zeros(self.n_params)
                    copy_len = min(len(orig_sig), self.n_params)
                    new_sig[:copy_len] = orig_sig[:copy_len]
                    self.sigma_g[m] = new_sig
            else:
                # If a single array/value is provided, broadcast it to all modes
                template = np.atleast_1d(sg)
                new_sig = np.zeros(self.n_params)
                copy_len = min(len(template), self.n_params)
                new_sig[:copy_len] = template[:copy_len]
                self.sigma_g = {m: new_sig.copy() for m in range(1, self.n_modes + 1)}
        else:
            self.sigma_g = None

    # Add this method to both BSMC and FilteredSMC classes
    def _calculate_weights(self, error: np.ndarray) -> np.ndarray:
        if self.likelihood_type == 'gaussian':
            w = np.exp(-0.5 * (error / self.sigma_obs)**2)
        elif self.likelihood_type == 'laplace':
            w = np.exp(-np.abs(error) / (self.sigma_obs / np.sqrt(2)))
        elif self.likelihood_type == 'cauchy':
            w = 1.0 / (1.0 + (error / self.sigma_obs)**2)
        elif self.likelihood_type == 'robust':
            w = (1.0 / (np.abs(error) + 1.0))**0.5
        else:
            raise ValueError(f' /!\\ Weighting approach "{self.likelihood_type}" not supported')
        return np.maximum(w, 1e-300)

    def save_results(self, results, results_dir):
        """Saves both the identification configuration and the extracted parameters."""
        os.makedirs(results_dir, exist_ok=True)

        # 1. Save identification configuration
        with open(os.path.join(results_dir, "config.json"), "w") as f:
            json.dump(self.identification_params, f, indent=4,
                      default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))

        # 2. Extract parameters into (Modes, Params, Time) array
        K = len(results)
        final_estimates = np.zeros((self.n_modes, self.n_params, K))
        for k in range(K):
            for m in range(1, self.n_modes + 1):
                final_estimates[m-1, :, k] = results[k]['estimates'][m]

        # 3. Save as binary (.npy) and readable (.csv)
        np.save(os.path.join(results_dir, "params.npy"), final_estimates)

        flat_data = final_estimates.reshape(-1, K).T
        cols = [f"M{m+1}_P{p+1}" for m in range(self.n_modes) for p in range(self.n_params)]
        np.savetxt(os.path.join(results_dir, "params.csv"), flat_data,
                   delimiter=",", header=",".join(cols), comments='')


class BSMC(IdentificationScheme):
    """
    Standard Bayesian Sequential Monte-Carlo (BSMC) parameter estimation.
    """

    def __init__(
        self,
        model: Model,
        identification_params: dict,
        model_parameters: dict,
    ):
        super().__init__(model, identification_params, model_parameters)
        self.model = model
        self.n_resample = identification_params['n_resample']
        self.sample_mult = identification_params['sample_mult']
        self.n_sample = identification_params['n_resample'] * identification_params['sample_mult']
        self.initial_param_range = identification_params['initial_param_range']
        self.init_uncertainty = 100

        if self.varying_params is not None:
            for m, params in self.varying_params.items():
                if m < 1 or m > self.n_modes:
                    raise ValueError(f"Invalid mode index {m}")
                if any(p < 0 or p >= self.n_params for p in params):
                    raise ValueError(f"Invalid parameter index in mode {m}")

    def run(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Runs the identification scheme
        """
        # Timing
        tic=time.time()

        # Parameters
        K = data['y'].shape[0]

        # Sigma_p normalization
        self._set_scaled_sigma(data)

        # Generate initial particles for each mode
        initial_particles = {}
        for m_idx in trange(1, self.n_modes + 1):
            # Set particle vector over time to zero
            initial_particles[m_idx] = np.zeros((self.n_params, self.n_resample), dtype=self.dtype)

            # Determine which parameters is identified for this mode
            if self.varying_params is None:
                mode_varying = range(self.n_params)  # all parameters are identified
            else:
                mode_varying = self.varying_params.get(m_idx, ())

            # On each parameter
            for p_idx in range(self.n_params):
                # If the parameter must be fitted
                if p_idx in mode_varying:
                    # Random initialization for identified parameters
                    initial_particles[m_idx][p_idx, :] = (
                        self.initial_param_range[0]
                        + (self.initial_param_range[1] - self.initial_param_range[0])
                        * np.random.rand(self.n_resample).astype(self.dtype)
                    )
                else:
                    # Fixed parameter: Use provided value if available, else 0.0
                    initial_particles[m_idx][p_idx, :] = self.theta_true[m_idx - 1, p_idx, 0] if self.theta_true is not None else 0.0

        last_active = np.full(self.n_modes + 1, -1, dtype=int)

        # Initialize particles structure
        particles = [{'resampled': {}, 'estimates': {}, 'mu': 0, 'uncertainty': {}} for _ in range(K)]

        # Initial resample
        mu_0 = int(data['mode'][0])
        for i in range(1, self.n_modes + 1):
            particles[0]['resampled'][i] = initial_particles[i].copy()
            particles[0]['estimates'][i] = np.zeros(self.n_params, dtype=self.dtype)

            if i == mu_0:
                last_active[i] = 0
                dist = 0
            else:
                dist = self.init_uncertainty
            particles[0]['uncertainty'][i] = (1.0 + dist) * self.sigma_p_scaled

        for k in range(1, K):
            mu_k = int(data['mode'][k])
            particles[k]['mu'] = mu_k
            last_active[mu_k] = k

            for i in range(1, self.n_modes + 1):
                if mu_k == i:
                    # 1- Sample new particles
                    prev_resampled = particles[k-1]['resampled'][i]
                    sampled = np.repeat(prev_resampled, self.sample_mult, axis=1)

                    # Add noise only to varying parameters
                    noise = np.zeros((self.n_params, self.n_sample))
                    # Determine varying params for this mode
                    if self.varying_params is None:
                        mode_varying = range(self.n_params)
                    else:
                        mode_varying = self.varying_params.get(i, ())

                    for p_idx in mode_varying:
                        noise[p_idx, :] = np.random.normal(
                            0, self.sigma_p_scaled[p_idx], self.n_sample
                        ).astype(self.dtype)
                    sampled += noise

                    # 2- Importance weighting (with windowing if needed)
                    combined_w_norm = np.zeros(self.n_sample)
                    for p_win in range(1, self.n_params + 1):
                        k_t = max(0, k + 1 - p_win)
                        # Reshape x to (n_regressors, 1)
                        x_k = data['x'][:, k_t].reshape(-1, 1)
                        # Reshape sampled to (1, n_params, n_sample)
                        theta_exp = sampled[np.newaxis, :, :]
                        # Predict and flatten back to 1D array of shape (n_sample,)
                        y_calc = self.model.predict_vectorized(x_k, theta_exp).flatten()
                        error = (y_calc - data['y'][k_t])/ (np.abs(data['y'][k_t]) + 1e-6)
                        w = self._calculate_weights(error)
                        combined_w_norm += w / np.sum(w)

                    weights_norm = combined_w_norm / np.sum(combined_w_norm)

                    # 3- Resample
                    if self.resampling_type == 0:
                        # Top-N selection
                        indices = np.argsort(weights_norm)[::-1][:self.n_resample]
                        particles[k]['resampled'][i] = sampled[:, indices]
                    else:
                        # Stochastic (Multinomial) selection with elitism
                        # Protect the best particle
                        best_sampled_idx = np.argmax(weights_norm)
                        particles[k]['resampled'][i] = np.zeros((self.n_params, self.n_resample))
                        particles[k]['resampled'][i][:, 0] = sampled[:, best_sampled_idx]
                        # Sample the rest
                        indices = np.random.choice(self.n_sample, size=self.n_resample-1, p=weights_norm)
                        particles[k]['resampled'][i][:, 1:] = sampled[:, indices]

                    # Estimate: max weight particle
                    # Format inputs for K=1 time step
                    x_k = data['x'][:, k].reshape(-1, 1)  # (n_regressors, 1)
                    # Resampled particles shape: (n_params, n_resample) -> (1, n_params, n_resample)
                    resampled_exp = particles[k]['resampled'][i][np.newaxis, :, :]
                    # Get predictions and calculate error
                    y_resampled = self.model.predict_vectorized(x_k, resampled_exp).flatten()
                    resampled_error = (y_resampled - data['y'][k])/(np.abs(data['y'][k]) + 1e-6)
                    resampled_weights = self._calculate_weights(resampled_error)
                    best_idx = np.argmax(resampled_weights)
                    particles[k]['estimates'][i] = particles[k]['resampled'][i][:, best_idx]
                else:
                    particles[k]['resampled'][i] = particles[k-1]['resampled'][i].copy()
                    particles[k]['estimates'][i] = particles[k-1]['estimates'][i].copy()

                # Update uncertainty for ALL modes at step k
                if last_active[i] == -1: # Never seen
                    dist = k + self.init_uncertainty
                else:
                    # Time since last observation
                    dist = k - last_active[i]

                particles[k]['uncertainty'][i] = (1.0 + dist) * self.sigma_p_scaled

        toc = time.time()
        print(f"Elapsed time (core): {toc-tic:.4f}s")

        return particles


class FilteredSMC(IdentificationScheme):
    """
    Filtered Sequential Monte-Carlo with iterative smoothing.
    Direct port and vectorized implementation of fct_filteredSMC_ident.m logic.
    """

    def __init__(
        self,
        model: Model,
        identification_params: dict,
        model_parameters: dict,
    ):
        super().__init__(model, identification_params, model_parameters)
        self.model = model
        self.filtered_anchors = identification_params['filtered_anchors']
        self.n_resample = identification_params['n_resample']
        self.sample_mult = identification_params['sample_mult']
        self.n_sample = self.n_resample * self.sample_mult
        self.initial_param_range = identification_params['initial_param_range']
        self.sigma_window_multiplier = identification_params['sigma_window_multiplier']
        self.n_iterations = identification_params['n_iterations']
        self.max_iterations = identification_params['max_iterations']
        self.conv_threshold = identification_params['convergence_threshold']
        self.conv_patience = identification_params['convergence_patience']
        self.dtype = np.float32 if identification_params['precision'] == 'float32' else np.float64
        self.mode_switch_margin = 0 # Code commented and incomplete
        self.point_estimate_extraction_share = float(identification_params.get('point_estimate_extraction_share', 0.0))

        # Validation + normalization of varying_params
        if self.varying_params is not None:
            self.varying_params = {
                int(m): set(v) for m, v in self.varying_params.items()
            }
        else:
            self.varying_params = {m: set(range(self.n_params)) for m in range(1, self.n_modes + 1)}

        # Pre-calculate flat indices of varying parameters for vectorized noise injection
        self.identified_indices = []
        for m_idx in range(1, self.n_modes + 1):
            for p_idx in self.varying_params.get(m_idx, set()):
                # Store (mode_index, param_index) tuples
                self.identified_indices.append((m_idx - 1, p_idx))
        # N_total_varying is the sum of varying params across all modes
        self.n_total_varying = len(self.identified_indices)

    def run(self, data: Dict[str, Any], theta_prior=None) -> tuple[List[Dict[str, Any]], List[float]]:
        # Timing
        tic=time.time()

        # Parameters
        K = data['y'].shape[0]
        n_modes = self.n_modes
        n_params = self.n_params
        n_sample = self.n_sample
        n_resample = self.n_resample

        # Sigma_p normalization
        self._set_scaled_sigma(data)

        # Cast input data to target precision
        y_data = data['y'].astype(self.dtype)
        x_data = data['x'].astype(self.dtype)

        # Initialize Parameter Activity Mask: (n_modes, n_params, K)
        # Determines if a specific parameter is "Observable" at time k
        param_active = np.zeros((n_modes, n_params, K), dtype=bool)

        def update_activity_mask(current_state_traj):
            """Helper to update which parameters are observable based on behavior states."""
            mask = np.zeros((n_modes, n_params, K), dtype=bool)
            # Get the visibility dict (default to varying_params if not provided)
            visibility_dict = self.identification_params.get('state_visibility', self.varying_params)
            for k in range(K):
                state = int(current_state_traj[k])

                if self.model.dynamic_mode_segmentation:
                    # If model's modes visibility depends on the internal behavioral state
                    visible_p_idxs = visibility_dict.get(state, ())
                    for m_idx in range(n_modes):
                        for p in visible_p_idxs:
                            mask[m_idx, p, k] = True
                else:
                    # If the mode is active, all its parameters (fixed & varying) are observable
                    m_idx = state - 1
                    mask[m_idx, :, k] = True
            return mask

        # Initial mask setup
        if not self.model.dynamic_mode_segmentation:
            param_active = update_activity_mask(data['mode'])

        in_mode = np.any(param_active, axis=1)

        # # Alternate version with more data points
        # # Initialize Safe Identification Mask (in_mode_safe)
        # in_mode_safe = in_mode.copy()
        # # Apply margin to each mode
        # for m_idx in range(n_modes):
        #     margin = int( self.mode_switch_margin )
        #     if margin > 0:
        #         # Find transitions (0 to 1 or 1 to 0)
        #         diff = np.diff(in_mode[m_idx, :].astype(int))
        #         starts = np.where(diff == 1)[0] + 1
        #         ends = np.where(diff == -1)[0]
        #         # Shrink the 'active' regions from both sides
        #         for s in starts:
        #             in_mode_safe[m_idx, s : min(s + margin, K)] = False
        #         for e in ends:
        #             in_mode_safe[m_idx, max(0, e - margin + 1) : e + 1] = False
        #         # Handle boundaries if mode starts/ends active
        #         if in_mode[m_idx, 0]:
        #             in_mode_safe[m_idx, 0 : min(margin, K)] = False
        #         if in_mode[m_idx, -1]:
        #             in_mode_safe[m_idx, max(0, K - margin) : K] = False
        # in_mode = in_mode_safe.copy()

        # Initial Particles Initialization
        # resampled_particles: (K, n_modes, n_params, n_resample)
        resampled_particles = np.zeros((K, n_modes, n_params, n_resample), dtype=self.dtype)
        for m_idx in range(n_modes):
            m_num = m_idx + 1
            mode_varying = self.varying_params.get(m_num, set())

            # Determine the base value for this mode/parameter (Prior > True > 0.0)
            for p_idx in range(n_params):
                if theta_prior is not None and m_num in theta_prior:
                    base_val = theta_prior[m_num][p_idx]
                elif self.theta_true is not None:
                    base_val = self.theta_true[m_idx, p_idx, 0]
                else:
                    base_val = 0.0

                if p_idx in mode_varying:
                    if theta_prior is not None and (m_num) in theta_prior:
                        # Initialize around the prior using current sigma_p
                        prior_val = theta_prior[m_num][p_idx]
                        resampled_particles[:, m_idx, p_idx, :] = np.random.normal(
                            prior_val, self.sigma_p_scaled[p_idx], (K, n_resample)
                        ).astype(self.dtype)
                    else:
                        # Broad random initialization (Default)
                        resampled_particles[:, m_idx, p_idx, :] = (
                            self.initial_param_range[0]
                            + (self.initial_param_range[1] - self.initial_param_range[0])
                            * np.random.rand(n_resample).astype(self.dtype)
                        )
                else:
                    # Parameter is fixed: Strictly use the base value
                    resampled_particles[:, m_idx, p_idx, :] = base_val

        error_history = []
        change_history = []
        # History of best estimates (max weight particle) for each iteration
        # Each entry will be a numpy array of shape (n_modes, n_params, K)
        history_estimates = []

        # Main Iteration Loop
        it = 0
        stable_count = 0
        loop_limit = self.n_iterations if isinstance(self.n_iterations, int) else self.max_iterations
        avg_max_est_filter = np.zeros((n_modes, n_params, K), dtype=self.dtype) # Best estimates from the smoothing step

        while it < loop_limit:
            print(f"Iteration {it+1}/{self.n_iterations}")

            # Dynamic Mode Update if required by model
            if self.model.dynamic_mode_segmentation:
                # Use best trajectory from previous iteration to determine behavioral states
                # theta shape: (K, n_params, 1)
                consensus_theta = resampled_particles[:, 0, :, 0][:, :, np.newaxis]
                state_traj = self.model.compute_mode(x_data, consensus_theta).flatten()
                param_active = update_activity_mask(state_traj)
                in_mode = np.any(param_active, axis=1) # Shape: (n_modes, K)
                # Need to add the version with self.mode_switch_margin if ever required

            # Phase 1: Sampling
            # sampled_particles: (K, n_modes, n_params, n_sample)
            # Sample from previous iteration's resampled particles at each time step
            sampled_particles = np.repeat(resampled_particles, self.sample_mult, axis=3)

            # Add noise only to varying parameters
            if self.n_total_varying > 0:
                # Generate noise for the parameters that actually vary
                # Result shape: (K, n_total_varying, n_sample)
                noise_block = np.zeros((K, self.n_total_varying, self.n_sample), dtype=self.dtype)
                for idx, (m_idx_list, p_idx_list) in enumerate(self.identified_indices):
                    # Apply the specific physical sigma for each identified parameter
                    noise_block[:, idx, :] = np.random.normal(
                        0, self.sigma_p_scaled[p_idx_list], (K, self.n_sample)
                    ).astype(self.dtype)

                # Elitism: Ensure the first particle (which is a copy of the previous elite)
                # is not jittered. This allows the algorithm to converge to a stable
                # trajectory and prevents the constant 1-sigma jitter across iterations.
                noise_block[:, :, 0] = 0.0

                # Inject noise directly into the 4D particles tensor
                # We use tuple-based indexing to "unroll" the mode/param dimensions
                m_idxs, p_idxs = zip(*self.identified_indices)
                sampled_particles[:, m_idxs, p_idxs, :] += noise_block

            # Phase 2: Importance Weighting
            w_norm = np.zeros((K, n_modes, n_sample), dtype=self.dtype)
            for m_idx in range(n_modes):
                # theta_all shape: (K, n_params, n_sample)
                theta_all = sampled_particles[:, m_idx, :, :]
                # result shape: (K, n_sample)
                y_calc = self.model.predict_vectorized(x_data, theta_all)
                # Broadcasted error calculation: (K, n_sample)
                error = (y_calc - y_data[:, np.newaxis])/ (np.abs(y_data[:, np.newaxis]) + 1e-6)
                # Apply weighting function to the entire matrix
                w = self._calculate_weights(error)
                # Row-wise normalization (over n_sample)
                w_sum = np.sum(w, axis=1, keepdims=True)
                w_norm[:, m_idx, :] = np.where(w_sum > 0, w / w_sum, np.array([1.0 / n_sample], dtype=self.dtype))
                # Apply uniform weights for steps where mode is not active
                mask_inactive = ~in_mode[m_idx, :]
                w_norm[mask_inactive, m_idx, :] = 1.0 / n_sample

            # Phase 3: Smoothing (Support update)
            # 3a. Extract Point Estimate Profile
            raw_max_est = np.zeros((n_modes, n_params, K), dtype=self.dtype)

            # Determine number of top particles to use for consensus
            top_p = float(self.point_estimate_extraction_share)
            top_p = np.clip(top_p, 0.0, 1.0)
            n_top = max(1, int(np.round(n_sample * top_p)))

            for m_idx in range(n_modes):
                # 1. Point Extraction (Local Consensus)
                w_m = w_norm[:, m_idx, :] # (K, n_sample)
                theta_m = sampled_particles[:, m_idx, :, :] # (K, n_params, n_sample)

                if n_top == 1:
                    # Fast-path for ArgMax (MAP)
                    best_indices = np.argmax(w_m, axis=1)
                    raw_max_est[m_idx, :, :] = theta_m[np.arange(K), :, best_indices].T
                else:
                    # Top-N Weighted Mean
                    # Find indices of top N_top particles per time step
                    if n_top == n_sample:
                        top_indices = np.arange(n_sample)[np.newaxis, :].repeat(K, axis=0)
                    else:
                        top_indices = np.argpartition(-w_m, n_top - 1, axis=1)[:, :n_top]

                    # Extract top weights and normalize them
                    top_w = np.take_along_axis(w_m, top_indices, axis=1)
                    top_w_sum = np.sum(top_w, axis=1, keepdims=True)
                    top_w_sum[top_w_sum == 0] = 1.0 # Prevent division by zero
                    top_w_norm = top_w / top_w_sum

                    # Extract top particles
                    top_indices_expanded = top_indices[:, np.newaxis, :]
                    top_indices_expanded = np.broadcast_to(top_indices_expanded, (K, n_params, n_top))
                    top_theta = np.take_along_axis(theta_m, top_indices_expanded, axis=2)

                    # Compute weighted mean
                    raw_max_est[m_idx, :, :] = np.einsum('kps,ks->pk', top_theta, top_w_norm)

                # 2. Anchor Stabilization & Interpolation
                # Get sigma_g for the mode
                sig_g_mode = self.sigma_g.get(m_idx + 1) if isinstance(self.sigma_g, dict) else self.sigma_g

                for p_idx in range(n_params):
                    # Find time steps where this specific parameter was observable
                    active_indices = np.where(param_active[m_idx, p_idx, :])[0]

                    if len(active_indices) > 0:
                        s = sig_g_mode[p_idx] if sig_g_mode is not None else 0.0
                        active_vals = raw_max_est[m_idx, p_idx, active_indices]

                        # Anchor Stabilization: Smooth available points before interpolation
                        if self.filtered_anchors == True and s > 1e-6 and len(active_indices) > 1:
                            # Apply local Gaussian smoothing only on the sequence of active points
                            # Note: Near occlusions, this naturally uses "half" the kernel
                            dist_sq = (active_indices[:, np.newaxis] - active_indices[np.newaxis, :])**2
                            W_anchors = np.exp(-0.5 * dist_sq / (s**2))
                            sum_W = W_anchors.sum(axis=1)
                            sum_W[sum_W == 0] = 1.0
                            anchors = (W_anchors @ active_vals) / sum_W
                        else:
                            anchors = active_vals

                        # Mandatory Linear Interpolation: fill gaps to connect anchors
                        if len(active_indices) < K:
                            raw_max_est[m_idx, p_idx, :] = np.interp(
                                np.arange(K), active_indices, anchors
                            ).astype(self.dtype)
                        else:
                            raw_max_est[m_idx, p_idx, :] = anchors
                    else:
                        # Mode never active: Keep initial values
                        pass

            # # 3b. Temporal Gaussian Filtering of Max Estimates
            avg_max_est_filter = np.zeros((n_modes, n_params, K), dtype=self.dtype)
            if self.sigma_window_multiplier==None:
                times = np.arange(K)
                for m_idx in range(n_modes):
                    # Retrieve sigma_g for this mode
                    sig_g_mode = None
                    if isinstance(self.sigma_g, dict):
                        sig_g_mode = self.sigma_g.get(m_idx + 1)
                    elif self.sigma_g is not None:
                        sig_g_mode = self.sigma_g

                    for p_idx in range(n_params):
                        s = sig_g_mode[p_idx] if sig_g_mode is not None else 0.0
                        if s > 1e-6:
                            # Matrix-based Gaussian filtering (excluding the current time step k)
                            # A[k] = sum_{j!=k} W[k,j]*E[j] / sum_{j!=k} W[k,j]
                            dist_sq = (times[:, np.newaxis] - times[np.newaxis, :])**2
                            W = np.exp(-0.5 * dist_sq / (s**2))
                            np.fill_diagonal(W, 0)
                            sum_W = W.sum(axis=1)
                            # Handle cases where sum_W might be 0 (though unlikely with Gaussian)
                            sum_W[sum_W == 0] = 1.0
                            avg_max_est_filter[m_idx, p_idx, :] = (W @ raw_max_est[m_idx, p_idx, :]) / sum_W
                        else:
                            # No smoothing if sigma is effectively 0
                            avg_max_est_filter[m_idx, p_idx, :] = raw_max_est[m_idx, p_idx, :]

            else: # Optimized Sliding Window
                # 1. Flatten all (mode, param) pairs into a single channel dimension: (M*P, K)
                flat_raw = raw_max_est.reshape(-1, K)
                flat_avg = np.zeros_like(flat_raw, dtype=self.dtype)
                ones_vector = np.ones(K, dtype=self.dtype)
                # 2. Collect all sigma values into a flat array
                all_sigmas = []
                for m_idx in range(n_modes):
                    sig_g_mode = self.sigma_g.get(m_idx + 1) if isinstance(self.sigma_g, dict) else self.sigma_g
                    for p_idx in range(n_params):
                        all_sigmas.append(sig_g_mode[p_idx] if sig_g_mode is not None else 0.0)
                all_sigmas = np.array(all_sigmas)
                # 3. Process by groups of unique sigma values
                for s in np.unique(all_sigmas):
                    # Identify which (mode, param) channels share this sigma
                    target_indices = np.where(all_sigmas == s)[0]
                    if s <= 1e-6:
                        # No smoothing needed for this group
                        flat_avg[target_indices] = flat_raw[target_indices]
                    else:
                        # 3a. Create Kernel (Once per unique sigma)
                        half_win = int(np.ceil(s * self.sigma_window_multiplier))
                        win_range = np.arange(-half_win, half_win + 1)
                        kernel = np.exp(-0.5 * (win_range / s)**2)
                        kernel[half_win] = 0.0 # Exclude center
                        # 3b. Multi-channel convolution: (N_channels_in_group, K)
                        # convolve1d handles the batch processing in C/MKL
                        weighted_sums = convolve1d(flat_raw[target_indices], kernel, axis=1, mode='nearest', cval=0.0)
                        # 3c. Normalization (same for all channels in this group)
                        sum_W = convolve1d(ones_vector, kernel, mode='nearest', cval=0.0)
                        flat_avg[target_indices] = weighted_sums / sum_W[np.newaxis, :]
                # 4. Reshape back to original dimensions
                avg_max_est_filter = flat_avg.reshape(n_modes, n_params, K)

            # 3c. Combined Weights
            w_adj_norm = np.zeros((K, n_modes, n_sample), dtype=self.dtype)
            for m_idx in range(n_modes):
                # mu shape: (K, n_params, 1)
                mu = avg_max_est_filter[m_idx, :, :].T[:, :, np.newaxis]
                # val shape: (K, n_params, n_sample)
                val = sampled_particles[:, m_idx, :, :]

                # Gaussian log-diffs: (K, n_params, n_sample)
                safe_sigma = np.maximum(self.sigma_p_scaled, 1e-12)
                log_p_val = -0.5 * ((val - mu) / safe_sigma[:, np.newaxis])**2
                # Joint log-likelihood for the particle (sum across parameters)
                # Result shape: (K, n_sample)
                log_w2 = np.sum(log_p_val, axis=1)
                # To avoid numerical underflow, we normalize relative to the max log-likelihood per time step
                log_w2_max = np.max(log_w2, axis=1, keepdims=True)
                w2 = np.exp(log_w2 - log_w2_max)

                # Combine w_norm and w2: (K, n_sample)
                w2_norm = np.where(np.sum(w2, axis=1, keepdims=True) > 0,
                                   w2 / np.sum(w2, axis=1, keepdims=True),
                                   1.0 / n_sample)
                wa = w_norm[:, m_idx, :] * w2_norm
                wa_sum = np.sum(wa, axis=1, keepdims=True)
                w_adj_norm[:, m_idx, :] = np.where(wa_sum > 0, wa / wa_sum, 1.0 / n_sample)

            # Phase 4: Resampling & Error Calculation
            all_top_indices = np.zeros((K, n_modes, n_resample), dtype=int)
            for m_idx in range(n_modes):
                w_current = w_adj_norm[:, m_idx, :]
                if self.resampling_type == 0:
                    # Deterministic Top-N resampling
                    part_indices = np.argpartition(-w_current, n_resample, axis=1)[:, :n_resample]
                    drawn_w = np.take_along_axis(w_current, part_indices, axis=1)
                    sort_idx = np.argsort(drawn_w, axis=1)[:, ::-1]
                    all_top_indices[:, m_idx, :] = np.take_along_axis(part_indices, sort_idx, axis=1)
                else:
                    # Stochastic (Multinomial) resampling with elitism
                    cumsum_w = np.cumsum(w_current, axis=1)
                    sums = cumsum_w[:, -1, np.newaxis]
                    sums[sums == 0] = 1.0
                    cumsum_w /= sums
                    u = np.random.rand(K, n_resample, 1).astype(self.dtype)
                    indices = np.argmax(cumsum_w[:, np.newaxis, :] > u, axis=2)
                    # Get the indices of the single best particle for every time step k
                    best_sampled_indices = np.argmax(w_current, axis=1) # Shape: (K,)
                    # Replace the first resampled particle with the elite one for every k
                    indices[:, 0] = best_sampled_indices
                    drawn_w = np.take_along_axis(w_current, indices, axis=1)
                    sort_idx = np.argsort(drawn_w, axis=1)[:, ::-1]
                    all_top_indices[:, m_idx, :] = np.take_along_axis(indices, sort_idx, axis=1)

            # 1. Update all resampled particles (With Explicit Broadcasting)
            # Expand (K, M, 1, Nr) to (K, M, P, Nr) to match sampled_particles
            broadcasted_indices = np.broadcast_to(
                all_top_indices[:, :, np.newaxis, :],
                (K, n_modes, n_params, n_resample)
            )
            resampled_particles = np.take_along_axis(sampled_particles, broadcasted_indices, axis=3)

            # 2. Vectorized Iteration Error Calculation
            iteration_error = 0
            for m_idx in range(n_modes):
                # best_estimates: (K, n_params)
                best_estimates = avg_max_est_filter[m_idx, :, :].T
                # Expand from (K, n_params) -> (K, n_params, 1) to match predict_vectorized expected shape
                theta_expanded = best_estimates[:, :, np.newaxis]
                # Use the model to estimae y_est
                y_est = self.model.predict_vectorized(x_data, theta_expanded).flatten()
                # Filter by active mask and accumulate
                active_mask = in_mode[m_idx, :]
                iteration_error += np.sum(np.abs(y_data[active_mask] - y_est[active_mask]))

            error_history.append(iteration_error)

            # Store current iteration's best estimates for history
            # using the smoothed profile (avg_max_est_filter)
            history_estimates.append(avg_max_est_filter.copy())

            # Calculate the maximum absolute change across all modes, params, and time steps
            if it > 0:
                prev_estimates = history_estimates[-2]

                # 1. Vectorized Absolute Difference (M*P, K)
                flat_diff = np.abs(avg_max_est_filter - prev_estimates).reshape(-1, K)
                flat_smoothed = np.zeros_like(flat_diff)

                # 2. Collect all sigmas into a flat array to match flat_diff rows
                all_sigmas = []
                for m_idx in range(self.n_modes):
                    sig_g_m = self.sigma_g.get(m_idx + 1) if isinstance(self.sigma_g, dict) else self.sigma_g
                    for p_idx in range(self.n_params):
                        all_sigmas.append(sig_g_m[p_idx] if sig_g_m is not None else 0.0)
                all_sigmas = np.array(all_sigmas)

                # 3. Batch process by unique sigma values
                for s in np.unique(all_sigmas):
                    target_idx = np.where(all_sigmas == s)[0]
                    if s <= 1e-6:
                        flat_smoothed[target_idx] = flat_diff[target_idx]
                    elif self.sigma_window_multiplier is None:
                        # Full matrix-based smoothing (exact but slow for large K)
                        times = np.arange(K)
                        dist_sq = (times[:, np.newaxis] - times[np.newaxis, :])**2
                        W = np.exp(-0.5 * dist_sq / (s**2))
                        W_sum = W.sum(axis=1, keepdims=True)
                        W_sum[W_sum == 0] = 1.0
                        W_norm = W / W_sum
                        # Process each channel in this group
                        for idx in target_idx:
                            flat_smoothed[idx, :] = W_norm @ flat_diff[idx, :]
                    else:
                        hw = int(np.ceil(s * self.sigma_window_multiplier))
                        kernel = np.exp(-0.5 * (np.arange(-hw, hw + 1) / s)**2)
                        kernel /= kernel.sum() # Normalize for local average
                        flat_smoothed[target_idx] = convolve1d(flat_diff[target_idx], kernel, axis=1, mode='constant')

                # 4. Global Max of the windowed averages
                safe_sigma_p_scaled = np.maximum(self.sigma_p_scaled, 1e-12)
                sigma_repeat = np.repeat(safe_sigma_p_scaled[np.newaxis, :], n_modes, axis=0).reshape(-1, 1)
                flat_rel_change = flat_smoothed / sigma_repeat
                relative_change = np.max(flat_rel_change)
                # print(f"  Max Param Change: {relative_change:.5f} (Threshold: {self.conv_threshold})")
                change_history.append(relative_change)

                # Convergence check
                if self.n_iterations == 'auto':
                    if relative_change < self.conv_threshold:
                        stable_count += 1
                        if stable_count >= self.conv_patience:
                            print(f"-> Converged automatically at iteration {it+1}")
                            break
                    else:
                        stable_count = 0 # Reset patience if it spikes

            it += 1

        # Per parameter identification uncertainty calculation
        uncertainty_matrix = np.zeros((K, n_modes, n_params), dtype=self.dtype)
        for m_idx in range(n_modes):
            for p_idx in range(n_params):
                active_indices = np.where(param_active[m_idx, p_idx, :])[0]
                if len(active_indices) > 0:
                    dists = np.min(np.abs(np.arange(K)[:, np.newaxis] - active_indices[np.newaxis, :]), axis=1)
                    # Scale distance by sigma_p_scaled (n_params,)
                    # Result shape: (K, n_params)
                    uncertainty_matrix[:, m_idx, p_idx] = dists * self.sigma_p_scaled[p_idx]
                else:
                    # Mode never active: High uncertainty based on full range K
                    uncertainty_matrix[:, m_idx, p_idx] = K * self.sigma_p_scaled[p_idx]

        # Structured conversion back to List[Dict] for plotting compatibility
        final_results = []
        for k in range(K):
            step_res = {
                'resampled': {},
                'estimates': {},
                'mu': int(data['mode'][k]),
                'uncertainty': {}
            }
            for m_idx in range(n_modes):
                m_num = m_idx + 1
                step_res['resampled'][m_num] = resampled_particles[k, m_idx, :, :]
                step_res['estimates'][m_num] = avg_max_est_filter[m_idx, :, k]
                step_res['uncertainty'][m_num] = uncertainty_matrix[k, m_idx, :]
            final_results.append(step_res)

        toc = time.time()
        print(f"Elapsed time (core): {toc-tic:.4f}s")
        return final_results, error_history, history_estimates, change_history

class ConstantSMC(IdentificationScheme):
    """
    Constant parameter identification using Batch SMC.
    Direct port of MATLAB's global driver profile identification.
    """
    def __init__(self, model, identification_params, model_parameters):
        super().__init__(model, identification_params, model_parameters)
        self.n_iterations = identification_params['cst_n_iterations']
        self.n_stoRegV = identification_params['cst_n_stoRegV']
        self.sigma_p_initial = identification_params['cst_sigma_p_initial']
        self.sigma_p_min = identification_params['cst_sigma_p_min']
        self.n_resample = identification_params['cst_n_resample']
        self.sample_mult = identification_params['cst_sample_mult']
        self.n_sample = self.n_resample * self.sample_mult
        self.initial_param_range = identification_params['initial_param_range']
        self.n_pre_samples = identification_params['n_pre_samples']

        # Can be changed by user but default values should be quite good already
        self.it_packet = identification_params.get('it_packet', 10)                 # Size of data used for evaluation of local convergence
        self.sigma_p_shrink = identification_params.get('sigma_p_shrink', 0.67)      # Reduce sampling noise by 1/3
        self.shrink_threshold = identification_params.get('shrink_threshold', 3.0)    # Multiplier to the measured std dev to trigger sigma_g shrink

        # Pre-calculate indices of varying parameters for vectorized noise
        self.identified_indices = []
        for m_idx in range(1, self.n_modes + 1):
            for p_idx in self.varying_params.get(m_idx, tuple()):
                self.identified_indices.append((m_idx - 1, p_idx))

        # Top-N is prefered for constant param ident
        if self.resampling_type==1:
            print("[NOTICE] Deterministic (Top-N) particles selection is prefered for constand paramter identification\n")

    def run(self, data):
        # Timing
        tic=time.time()

        # Parameters
        K = data['y'].shape[0]
        n_modes, n_params = self.n_modes, self.n_params
        n_sample, n_resample = self.n_sample, self.n_resample
        n_select_max = self.n_stoRegV

        # Sigma_p normalization
        self._set_scaled_sigma(data)

        # Initialize history storage
        error_history = []
        history_estimates = []
        change_history = []
        sigma_history = []
        mode_indices = [np.where(data['mode'] == m+1)[0] for m in range(n_modes)]
        active_modes_mask = np.array([len(idx) > 0 for idx in mode_indices])
        current_sigma_p = self.sigma_p_initial

        # Initialize particles tensor (M, P, Nr) with zeros
        particles = np.zeros((n_modes, n_params, n_resample), dtype=self.dtype)
        for m_idx in range(1, n_modes + 1):
            varying = self.varying_params.get(m_idx, tuple())
            for p_idx in range(n_params):
                if p_idx in varying:
                    # Randomize varying parameters
                    particles[m_idx-1, p_idx, :] = np.random.uniform(
                        low=self.initial_param_range[0],
                        high=self.initial_param_range[1],
                        size=(n_resample)
                    ).astype(self.dtype)
                else:
                    # Fixed parameter: Use provided value if available, else 0.0
                    particles[m_idx-1, p_idx, :] = self.theta_true[m_idx - 1, p_idx, 0] if self.theta_true is not None else 0.0

        # Pre-allocate containers for the blocks: (PoolSize, Modes, Params, SubSampleSize)
        n_regressors = data['x'].shape[0]
        pre_x = np.zeros((self.n_pre_samples, n_modes, n_regressors, n_select_max), dtype=self.dtype)
        pre_y = np.zeros((self.n_pre_samples, n_modes, n_select_max), dtype=self.dtype)
        mode_sel_counts = np.zeros(n_modes)

        print(f"Pre-sampling {self.n_pre_samples} data blocks per mode...")
        for m in range(n_modes):
            idx = mode_indices[m]
            if len(idx) > 0:
                n_sel = min(n_select_max, len(idx))
                mode_sel_counts[m] = n_sel
                for i in range(self.n_pre_samples):
                    # Pick random points for this specific pool entry
                    s_idx = np.random.choice(idx, size=n_sel, replace=False)
                    pre_x[i, m, :, :n_sel] = data['x'][:, s_idx]
                    pre_y[i, m, :n_sel] = data['y'][s_idx]

        # Random sampler
        rng = np.random.default_rng()

        # Buffer to store estimates for stability check
        est_buffer = np.zeros((n_modes, n_params, self.it_packet), dtype=self.dtype)

        # Main Optimization Loop
        for it in trange(self.n_iterations):
            # Pick a pre-sampled block from the pool
            p_idx = int(self.n_pre_samples*rng.random())
            current_x = pre_x[p_idx]    # Shape: (n_modes, n_params, n_select_max)
            current_y = pre_y[p_idx]    # Shape: (n_modes, n_select_max)

            # Sample New Particles (as before)
            sampled = np.repeat(particles, self.sample_mult, axis=2)
            if self.identified_indices:
                noise = np.zeros((len(self.identified_indices), n_sample), dtype=self.dtype)
                for idx, (m_idx_list, p_idx_list) in enumerate(self.identified_indices):
                    # Use the float initial multiplier instead of the physical list
                    scale_factor = current_sigma_p / self.sigma_p_initial
                    noise[idx, :] = np.random.normal(
                        0, scale_factor * self.sigma_p_scaled[p_idx_list], n_sample
                    ).astype(self.dtype)
                m_idxs, p_idxs = zip(*self.identified_indices)
                sampled[m_idxs, p_idxs, :] += noise

            # Vectorized Batch Likelihood using the Model
            y_calc = np.zeros((n_modes, n_sample, n_select_max), dtype=self.dtype)
            for m in range(n_modes):
                if mode_sel_counts[m] > 0:
                    n_sel = int(mode_sel_counts[m])
                    # sampled[m] is (n_params, n_sample). Add time dimension -> (1, n_params, n_sample)
                    theta_m = np.repeat(sampled[m][np.newaxis, :, :], n_sel, axis=0)
                    x_m = current_x[m, :, :n_sel] # (n_regressors, n_select_max)
                    # Output is (n_select_max, n_sample)
                    y_pred = self.model.predict_vectorized(x_m, theta_m)
                    y_calc[m, :, :n_sel] = y_pred.T # Transpose to (n_sample, n_select_max)

            errors = (y_calc - current_y[:, np.newaxis, :])/(np.abs(current_y[:, np.newaxis, :]) + 1e-6) # (M, S, T)
            # Mask the errors so only actual data points count
            # mode_sel_counts[m] stores the actual number of points used for mode m
            batch_error = np.zeros((n_modes, n_sample), dtype=self.dtype)
            for m in range(n_modes):
                n_sel = int(mode_sel_counts[m])
                if n_sel > 0:
                    # Only average over the first n_sel columns
                    batch_error[m, :] = np.mean(np.abs(errors[m, :, :n_sel]), axis=1)
                else:
                    batch_error[m, :] = 0.0 # Or keep as is
            # Apply weighting function to every individual point in the batch
            point_weights = self._calculate_weights(errors)
            # Batch likelihood: Aggregate weights across the batch time dimension (T)
            weights = np.mean(point_weights, axis=2) # Result: (M, S)

            # Resampling
            if self.resampling_type == 0:
                # Deterministic
                top_idx = np.argsort(weights, axis=1)[:, ::-1][:, :n_resample]
                m_grid = np.arange(n_modes)[:, np.newaxis, np.newaxis]
                p_grid = np.arange(n_params)[np.newaxis, :, np.newaxis]
                particles = sampled[m_grid, p_grid, top_idx[:, np.newaxis, :]]
            else:
                # Randomized resampling with elitism
                w_sum = np.sum(weights, axis=1, keepdims=True)
                w_norm = np.where(w_sum > 0, weights / w_sum, 1.0 / n_sample)
                top_idx = np.argmax(w_norm, axis=1)[:, np.newaxis]
                for m in range(n_modes):
                    if active_modes_mask[m]:
                        # Always keep the absolute best particle as the first one
                        best_of_all_idx = np.argmax(weights[m])
                        particles[m, :, 0] = sampled[m, :, best_of_all_idx]
                        # Stochastic draw for the remaining N-1 particles to maintain diversity
                        indices = np.random.choice(n_sample, size=n_resample-1, p=w_norm[m])
                        particles[m, :, 1:] = sampled[m][:, indices]

            # Update stability check buffer with current best estimates
            buf_idx = it % self.it_packet
            est_buffer[:, :, buf_idx] = particles[:, :, 0]

            # Stability check
            # Instead of tracking the jittery 'best' particle, track the cloud center.
            # particles shape: (n_modes, n_params, n_resample)
            cloud_center = np.mean(particles, axis=2)
            buf_idx = it % self.it_packet
            est_buffer[:, :, buf_idx] = cloud_center
            # Check for Stability at every full packet
            if it > 0 and (it + 1) % self.it_packet == 0:
                # Calculate standard deviation of the cloud center over the last N iterations
                packet_std = np.std(est_buffer, axis=2)
                stable_modes = []
                for m in range(n_modes):
                    if active_modes_mask[m]:
                        # Max std ensures ALL parameters in the mode are stable
                        std_thresholds = self.shrink_threshold * (current_sigma_p / self.sigma_p_initial) * self.sigma_p_scaled
                        if np.all(packet_std[m] < std_thresholds):
                            stable_modes.append(m)
                # Shrink if all active modes are stable
                if len(stable_modes) == np.sum(active_modes_mask):
                    if current_sigma_p <= self.sigma_p_min:
                        print(f"\n-> Full convergence reached at iteration {it+1}.")
                        break
                    current_sigma_p = max(self.sigma_p_min, current_sigma_p * self.sigma_p_shrink)
                    print(f" -> All modes stable. Shrinking search radius to: {current_sigma_p:.5f}")

            # History Tracking
            # Always pick the particle with the highest weight from the SAMPLED set
            # regardless of whether we did stochastic or deterministic resampling.
            best_idx = top_idx[:, 0] # Index of best particle per mode
            m_idx_grid = np.arange(n_modes)

            # sampled shape is (M, P, S), we take (M, P)
            best_params = sampled[m_idx_grid, :, best_idx]

            # Broadcast the constant best param to all K steps for plotting
            current_it_estimates = np.repeat(best_params[:, :, np.newaxis], K, axis=2)
            history_estimates.append(current_it_estimates)

            # Sum error across active modes
            # top_idx should be (n_modes, 1) or (n_modes, n_resample)
            best_errs = np.take_along_axis(batch_error, top_idx[:, 0:1], axis=1).flatten()
            error_history.append(np.sum(best_errs[active_modes_mask]))
            sigma_history.append(current_sigma_p)

            if it > 0:
                change_history.append(np.max(np.abs(current_it_estimates - history_estimates[-2])) / self.sigma_p_initial)

        # 4. Final Results Assembly
        final_best_particles = {m+1: particles[m] for m in range(n_modes)}
        final_estimates = {m+1: particles[m, :, 0] for m in range(n_modes)}

        print("\n--- Identified Constant Parameters (Final Iteration) ---")
        for m_num, theta in final_estimates.items():
            # Only print if mode was actually active in the data
            if active_modes_mask[m_num-1]:
                params_str = ", ".join([f"{p:.5f}" for p in theta])
                print(f"  Mode {m_num}: [{params_str}]")
            else:
                print(f"  Mode {m_num}: [Inactive Mode - No Data]")
        print("----------------------------------------------------------\n")

         # 5. Build framework-compatible output sequence
        results = []
        for k in range(K):
            step_res = {'resampled': {}, 'estimates': {}, 'mu': int(data['mode'][k]), 'uncertainty': {}}
            for m in range(1, n_modes + 1):
                step_res['resampled'][m] = final_best_particles[m]
                step_res['estimates'][m] = final_estimates[m]
                step_res['uncertainty'][m] = np.std(final_best_particles[m], axis=1)
            results.append(step_res)

        print(f"Elapsed time (core): {time.time()-tic:.4f}s")
        return results, error_history, history_estimates, change_history, sigma_history
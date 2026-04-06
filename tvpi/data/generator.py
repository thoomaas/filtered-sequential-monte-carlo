import numpy as np
from typing import Dict, Any

def generate_theoretical_example(
        model_parameters,
        learning_data,
        model
    ) -> Dict[str, Any]:
    """
    Synthetic data generation

    Creating sinus wave data based on user parameters
    """
    # Parameters
    x_range = learning_data['synth_data_time']
    K = len(x_range)

    # Model definition
    n_modes = model_parameters['number_modes']
    n_params = model_parameters['number_params']

    # Check dimensions
    if np.array(learning_data['synth_param_constant']).shape[0] != n_modes:
        raise Exception("model_parameters['number_modes'] and learning_data['synth_param_constant'] mismatch")
    if np.array(learning_data['synth_param_constant']).shape[1] != n_params:
        raise Exception("model_parameters['number_params'] and learning_data['synth_param_constant'] mismatch")

    # Model parameter initialization with constant values
    theta_var = np.array(learning_data['synth_param_constant'])[:, :, np.newaxis] * np.ones(K)

    # Vectorized calculation of time-varying synthetic parameters
    speeds = np.array(learning_data['synth_param_variable_speed'])
    amplitudes = np.array(learning_data['synth_param_variable_value'])
    mask = speeds != 0 # Only apply sin where speed != 0
    theta_var += mask[..., np.newaxis] * amplitudes[..., np.newaxis] * np.sin(speeds[..., np.newaxis] * x_range)

    ## Synthetic modes (can be modified by user)
    # Current setup matches published paper
    if n_modes > 1:
        q = K // 4
        mode_nb = np.concatenate([
            1 * np.ones(q, dtype=int),
            2 * np.ones(q, dtype=int),
            1 * np.ones(q, dtype=int),
            2 * np.ones(K - 3*q, dtype=int)
        ])
        mode_nb[9:20] = 2
        mode_nb[-20:-9] = 1
    else:
        mode_nb = np.ones(K, dtype=int)

    ## Output calculation
    # Prepare the True regressor matrix 'x'
    if n_params == 6: # Gipps model
        # Time steps
        dt = x_range[1] - x_range[0] if len(x_range) > 1 else 0.1

        # 1. Simulate Leader Behavior
        # Leader starts at 25m/s and BRAKES hard at mid-point
        vel_lead = np.full(K, 25.0)
        vel_lead[K//2:] = 10.0 # Sudden drop to 10m/s

        # 2. Simulate Follower and Distance
        vel = np.zeros(K)
        dist = np.zeros(K)

        vel[0] = 20.0 # Start at 20m/s
        dist[0] = 40.0 # Start at 40m distance

        # Use a simple integration to generate realistic data
        for k in range(1, K):
            # Calculate position change
            dist[k] = dist[k-1] + (vel_lead[k-1] - vel[k-1]) * dt
            # Speed oscillates slightly to add realism
            vel[k] = vel[k-1] + 0.1 * np.sin(k * 0.05)

        x_matrix = np.vstack([vel, dist, vel_lead])
    else:
        # For the PWARX toy example, 'x' is just the time/index.
        x_matrix = x_range.reshape(1, -1)

    # Prepare the True parameter trajectory following the mode sequence
    # Shape: (Time K, Parameters P, 1 Particle)
    true_theta_traj = np.zeros((K, n_params, 1))
    for k in range(K):
        m_idx = mode_nb[k] - 1
        true_theta_traj[k, :, 0] = theta_var[m_idx, :, k]
    # Generate clean output using the model's own logic
    y_clean = model.predict_vectorized(x_matrix, true_theta_traj).flatten()

    # Additive noise
    sigma_noise = learning_data['sigma_noise']
    e_noise = np.random.normal(0, sigma_noise, K) if sigma_noise > 0 else np.zeros(K)
    y_noisy = y_clean + e_noise

    return {
        'x': x_matrix,
        'y_clean': y_clean,
        'y': y_noisy,
        'mode': mode_nb,
        'theta_true': theta_var,
    }

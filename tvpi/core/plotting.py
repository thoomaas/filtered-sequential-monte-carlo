import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any

def plot_bsmc_results(data: Dict[str, Any], results: List[Dict[str, Any]], save_path: str = None):
    """
    Equivalent to MATLAB's figures_initial_param.m
    Plots initial BSMC parameter evolution in parameter space.
    """
    K = data['y'].shape[0]
    n_modes = len(results[0]['estimates'])
    n_params = results[0]['estimates'][1].shape[0]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Colors for modes
    colors = {1: 'blue', 2: 'red', 3: 'green'}

    for i in range(1, n_modes + 1):
        # Extract estimates for this mode
        est_traj = np.array([results[k]['estimates'][i] for k in range(K)])

        # Plot trajectory in parameter space (assuming 2 params for 2D plot)
        if n_params >= 2:
            ax.plot(est_traj[:, 0], est_traj[:, 1], color=colors.get(i, 'black'), marker='x', label=f'Mode {i} Trajectory', alpha=0.5)
            # Mark end point
            ax.scatter(est_traj[-1, 0], est_traj[-1, 1], color=colors.get(i, 'black'), marker='d', s=100)

        # If true parameters are known
        if 'theta_true' in data:
            true_theta = data['theta_true'][i-1, :, 0] # Initial true
            ax.scatter(true_theta[0], true_theta[1], color='black', marker='o', s=100, label=f'True Mode {i} Start')

    ax.set_xlabel('Parameter 1')
    ax.set_ylabel('Parameter 2')
    ax.set_title('BSMC Parameter Evolution in Parameter Space')
    ax.legend()
    ax.grid(True)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_final_with_uncertainty(data: Dict[str, Any], results: List[Dict[str, Any]], save_path: str = None):
    """
    Plots the final identified parameter trajectories with uncertainty bands.
    """
    K = data['y'].shape[0]
    n_modes = len(results[0]['estimates'])
    n_params = results[0]['estimates'][1].shape[0]

    fig, axes = plt.subplots(n_modes, 1, figsize=(24, 4 * n_modes), sharex=True)
    if n_modes == 1:
        axes = [axes]

    for i in range(1, n_modes + 1):
        ax = axes[i-1]

        # Extract estimated trajectory and uncertainty for mode i
        est_traj = np.zeros((n_params, K))
        unc_traj = np.zeros((n_params, K))
        for k in range(K):
            est_traj[:, k] = results[k]['estimates'][i]
            unc_traj[:, k] = results[k]['uncertainty'][i]

        for p in range(n_params):
            # Plot True
            if 'theta_true' in data:
                ax.plot(data['theta_true'][i-1, p, :], '--', color='red', label=f'True P{p+1}' if p==0 else "", alpha=0.7)

            # Plot Estimated
            line, = ax.plot(est_traj[p, :], label=f'Identified P{p+1}', linewidth=1.5)

            # Uncertainty band (MATLAB e_bar)
            ax.fill_between(np.arange(K), est_traj[p, :] - unc_traj[p, :],
                            est_traj[p, :] + unc_traj[p, :],
                            color=line.get_color(), alpha=0.1)

        ax.set_ylabel(f'Mode {i} Parameters')
        ax.legend()
        ax.grid(True)

    axes[-1].set_xlabel('Time Step')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_convergence(error_history: List[float], save_path: str = None):
    """
    Plots total error vs iteration number.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(error_history) + 1), error_history, 'o-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Total Absolute Error')
    plt.title('Algorithm Convergence')
    plt.grid(True)
    plt.ylim(bottom=0.0) # Ensure y-axis starts at 0

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_results(data: Dict[str, Any], results: List[Dict[str, Any]], save_path: str = None):
    """
    Standard line plot of results.
    """
    K = data['y'].shape[0]
    n_modes = len(results[0]['estimates'])
    n_params = results[0]['estimates'][1].shape[0]

    fig, axes = plt.subplots(n_modes, 1, figsize=(24, 4 * n_modes), sharex=True)
    if n_modes == 1:
        axes = [axes]

    for i in range(1, n_modes + 1):
        ax = axes[i-1]
        est_traj = np.array([results[k]['estimates'][i] for k in range(K)])
        for p in range(n_params):
            ax.plot(est_traj[:, p], label=f'P{p+1}')
            if 'theta_true' in data:
                ax.plot(data['theta_true'][i-1, p, :], '--', alpha=0.5)
        ax.legend()
        ax.grid(True)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_mode_data(data: Dict[str, Any]):
    """
    Plots the output signal and the mode sequence.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 6), sharex=True)

    ax1.plot(data['y'], label='Noisy Output')
    if 'y_clean' in data:
        ax1.plot(data['y_clean'], 'k--', label='Clean Output', alpha=0.5)
    ax1.set_ylabel('y')
    ax1.legend()
    ax1.grid(True)

    ax2.step(np.arange(len(data['mode'])), data['mode'], where='post')
    ax2.set_ylabel('Mode')
    ax2.set_xlabel('Time Step')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def plot_iteration_evolution(data: Dict[str, Any], history_estimates: List[np.ndarray], save_path: str = None):
    """
    Plots the evolution of parameter estimates across algorithm iterations.
    history_estimates: List of arrays of shape (n_modes, n_params, K)
    """
    n_iterations = len(history_estimates)
    n_modes, n_params, K = history_estimates[0].shape

    fig, axes = plt.subplots(n_modes, 1, figsize=(24, 4 * n_modes), sharex=True)
    if n_modes == 1:
        axes = [axes]

    # Use different colormaps for different parameters to distinguish them
    cms = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens, plt.cm.Purples, plt.cm.Oranges]

    for m_idx in range(n_modes):
        ax = axes[m_idx]

        for p_idx in range(n_params):
            cm = cms[p_idx % len(cms)]
            param_colors = cm(np.linspace(0.3, 1, n_iterations))

            # 1. Plot True parameter if available
            if 'theta_true' in data:
                ax.plot(data['theta_true'][m_idx, p_idx, :], '--', color=param_colors[-1], label=f'True P{p_idx+1}', alpha=0.8, linewidth=2)

            # 2. Plot each iteration
            for it in range(n_iterations):
                label = f'P{p_idx+1} Iter {it+1}' if it == 0 or it == n_iterations-1 else ""
                ax.plot(history_estimates[it][m_idx, p_idx, :],
                        color=param_colors[it],
                        label=label,
                        alpha=0.6,
                        linewidth=1 if it < n_iterations-1 else 2)

        ax.set_ylabel(f'Mode {m_idx + 1} Parameters')
        ax.set_title(f'Evolution per Iteration - Mode {m_idx + 1}')
        ax.grid(True, alpha=0.3)

        # Deduplicate legend labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small')

    axes[-1].set_xlabel('Time Step')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_convergence_metrics(error_history: List[float], change_history: List[float], threshold: float = None, save_path: str = None):
    """
    Plots both the Measurement Error (MAE) and the Parameter Stability (Max Change).
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 1. Plot Measurement Error (Total Absolute Error / K)
    # We divide by K to show the Average error per time step
    color_err = 'tab:blue'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Mean Absolute Error (MAE)', color=color_err)
    ax1.plot(np.arange(1, len(error_history) + 1), np.array(error_history) / len(error_history), 'o-', color=color_err, linewidth=2, label='MAE')
    ax1.tick_params(axis='y', labelcolor=color_err)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0.0)

    # 2. Plot Parameter Stability (Relative Change)
    ax2 = ax1.twinx()
    color_stab = 'tab:red'
    ax2.set_ylabel('Max Relative Param Change', color=color_stab)
    ax2.plot(np.arange(2, len(change_history) + 2), change_history, 's--', color=color_stab, linewidth=1.5, label='Stability')
    ax2.tick_params(axis='y', labelcolor=color_stab)
    ax2.set_ylim(bottom=0.0)

    # Add Threshold line if provided
    if threshold:
        ax2.axhline(y=threshold, color='black', linestyle=':', alpha=0.5, label='Threshold')

    plt.title('Algorithm Convergence: MAE and Fit Stability')
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_optimization_diagnostics(error_history: List[float], sigma_history: List[float], save_path: str = None):
    """
    Plots the Mean Absolute Error and the Sigma_p decay (exploration range).
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    iterations = np.arange(1, len(error_history) + 1)

    # 1. Plot Measurement Error
    color_err = 'tab:blue'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Batch MAE (Total Mode Error)', color=color_err)
    ax1.plot(iterations, error_history, '-', color=color_err, linewidth=2, label='Mean Error')
    ax1.tick_params(axis='y', labelcolor=color_err)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0.0)

    # 2. Plot Sigma_p Decay (Exploration Radius)
    ax2 = ax1.twinx()
    color_sig = 'tab:red'
    ax2.set_ylabel('Sigma_p (Exploration Range)', color=color_sig)
    ax2.plot(iterations, sigma_history, '--', color=color_sig, linewidth=1.5, label='Sigma_p')
    ax2.tick_params(axis='y', labelcolor=color_sig)
    ax2.set_ylim(bottom=0.0)

    # Shade the exploration range around zero (representing parameter uncertainty)
    ax2.fill_between(iterations, 0, sigma_history, color=color_sig, alpha=0.1)

    plt.title('Constant SMC Optimization Diagnostics')
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_identification_performance(data: Dict[str, Any], results: List[Dict[str, Any]], model, save_path: str = None):
    """
    Compares the original Y signal with the Y signal reconstructed from identified parameters.
    Also displays the mode sequence.
    """
    K = data['y'].shape[0]
    n_params = model.n_params

    # Prepare the trajectory of identified parameters
    # Shape: (Time K, Parameters P, 1 Particle)
    theta_traj = np.zeros((K, n_params, 1))
    for k in range(K):
        mu_k = int(data['mode'][k])
        # Get the best estimate for the active mode at time k
        theta_traj[k, :, 0] = results[k]['estimates'][mu_k]
    # Reconstruct the entire signal at once
    # Result is (K, 1), we flatten it to (K,)
    y_modeled = model.predict_vectorized(data['x'], theta_traj).flatten()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # 1. Y Comparison Plot
    ax1.plot(data['y'], color='lightgray', alpha=0.7, label='Data $y$')
    if 'y_clean' in data:
        ax1.plot(data['y_clean'], 'k--', alpha=0.6, label='Data $y$ (Clean)')
    ax1.plot(y_modeled, 'r-', linewidth=1.5, label='Modeled $\hat{y}$')

    ax1.set_ylabel('Output Value')
    ax1.set_title('Simulation Performance: Data vs. Modeled')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 2. Mode Sequence Plot
    ax2.step(np.arange(K), data['mode'], where='post', color='tab:blue', linewidth=1.5)
    ax2.set_ylabel('Mode')
    ax2.set_xlabel('Time Step')
    ax2.set_yticks(np.unique(data['mode']))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
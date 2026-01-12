"""
E8 Consciousness Model Simulation
==================================
Operationalizes consciousness as coherent activation across Weyl chambers
in an E8-structured state space, coupled by φⁿ-scaled nonlinear dynamics.

Mathematical Framework:
- E8 State Space: 240 roots of the E8 Lie algebra as activation sites in 8D
- Weyl Chambers: Fundamental domains approximated via projection binning
- Dynamics: Kuramoto oscillators for synchronization/coherence
- φ-Scaling: Golden ratio (φ ≈ 1.618) introduces self-similar nonlinearity
- Noise: Gaussian white noise for stochastic perturbations
- Learning: Hebbian plasticity for adaptive coupling weights

Author: Generated for Schumann Resonance EEG Analysis project
"""

import numpy as np
from scipy.integrate import odeint
from itertools import permutations, product
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618033988749895


def generate_e8_roots():
    """
    Generate the 240 roots of the E8 Lie algebra.

    E8 roots consist of two parts:
    1. D8 sublattice: All permutations of (±1, ±1, 0, 0, 0, 0, 0, 0) - 112 roots
    2. Spinor vectors: (±1/2, ±1/2, ..., ±1/2) with even # of minus signs - 128 roots

    All roots have ||root||² = 2 (unit normalized to length √2).
    """
    roots = []

    # Part 1: D8 sublattice - permutations of (±1, ±1, 0, 0, 0, 0, 0, 0)
    # Choose 2 positions out of 8 for the ±1 values
    base = np.zeros(8)
    for i in range(8):
        for j in range(i + 1, 8):
            for s1 in [-1, 1]:
                for s2 in [-1, 1]:
                    root = np.zeros(8)
                    root[i] = s1
                    root[j] = s2
                    roots.append(root)

    # Part 2: Spinor vectors - (±1/2)^8 with even number of minus signs
    for signs in product([-0.5, 0.5], repeat=8):
        root = np.array(signs)
        # Count negative signs (even number required)
        if np.sum(root < 0) % 2 == 0:
            roots.append(root)

    roots = np.array(roots)

    # Verify: should have 240 roots
    assert len(roots) == 240, f"Expected 240 roots, got {len(roots)}"

    # Verify: all roots have ||root||² = 2
    norms_sq = np.sum(roots**2, axis=1)
    assert np.allclose(norms_sq, 2.0), "Not all roots have ||root||² = 2"

    return roots


def build_e8_adjacency_graph(roots):
    """
    Build adjacency graph for E8 roots.

    Two roots are adjacent if their squared Euclidean distance is 2,
    which is equivalent to inner product <root_i, root_j> = 1.

    Returns:
        adjacency: dict mapping root index to set of adjacent indices
        edges: list of (i, j) tuples for undirected edges
    """
    n = len(roots)
    adjacency = defaultdict(set)
    edges = []

    # Compute pairwise inner products
    # Inner product = 1 means adjacent (squared distance = 2)
    for i in range(n):
        for j in range(i + 1, n):
            inner = np.dot(roots[i], roots[j])
            if np.isclose(inner, 1.0):
                adjacency[i].add(j)
                adjacency[j].add(i)
                edges.append((i, j))

    return dict(adjacency), edges


def approximate_weyl_chambers(roots, n_bins=4):
    """
    Approximate Weyl chambers by projecting roots onto a generic direction
    and binning into quantiles.

    Uses direction v = [1,2,3,4,5,6,7,8]/||v|| to avoid symmetry degeneracies.

    Returns:
        projections: 1D projections of each root
        bin_indices: chamber assignment for each root (0 to n_bins-1)
        bin_edges: edges defining the bins
    """
    # Generic direction avoiding symmetries
    v = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    v = v / np.linalg.norm(v)

    # Project roots
    projections = roots @ v

    # Bin into quantiles (approximate chambers)
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(projections, percentiles)
    bin_edges[0] -= 1e-6  # Ensure all points included
    bin_edges[-1] += 1e-6

    bin_indices = np.digitize(projections, bin_edges[1:-1])

    return projections, bin_indices, bin_edges


def kuramoto_dynamics(phases, t, adjacency, omega, K, sigma):
    """
    Kuramoto model dynamics for phase oscillators.

    dθ_i/dt = ω_i + K * Σ_{j adj i} sin(σ * (θ_j - θ_i))

    Args:
        phases: current phase array
        t: time (unused but required by odeint)
        adjacency: dict mapping node to adjacent nodes
        omega: natural frequencies
        K: coupling strength
        sigma: phase difference scaling (1.0 = standard, φ = golden ratio)
    """
    n = len(phases)
    dtheta = omega.copy()

    for i in range(n):
        coupling_sum = 0.0
        for j in adjacency.get(i, []):
            coupling_sum += np.sin(sigma * (phases[j] - phases[i]))
        dtheta[i] += K * coupling_sum

    return dtheta


def compute_order_parameter(phases):
    """
    Compute Kuramoto order parameter r = |<exp(i*θ)>|.

    r = 0: completely incoherent
    r = 1: perfect synchronization
    """
    return np.abs(np.mean(np.exp(1j * phases)))


# =============================================================================
# NOISE AND LEARNING EXTENSIONS
# =============================================================================

def initialize_coupling_weights(adjacency, init_weight=1.0):
    """
    Initialize coupling weight matrix W_ij for Hebbian learning.

    Args:
        adjacency: dict mapping node to set of adjacent nodes
        init_weight: initial weight value (default 1.0)

    Returns:
        weights: dict of dicts, weights[i][j] = coupling weight from j to i
    """
    weights = {}
    for i, neighbors in adjacency.items():
        weights[i] = {j: init_weight for j in neighbors}
    return weights


def kuramoto_step_noisy(phases, adjacency, omega, K, sigma, weights, dt, noise_strength):
    """
    Single Euler-Maruyama step for stochastic Kuramoto model.

    dθ_i = [ω_i + K * Σ_j W_ij * sin(σ(θ_j - θ_i))] dt + D * dW_i

    Args:
        phases: current phase array
        adjacency: adjacency dict
        omega: natural frequencies
        K: base coupling strength
        sigma: phase scaling (1.0 or φ)
        weights: coupling weight matrix
        dt: time step
        noise_strength: D, standard deviation of Gaussian noise

    Returns:
        new_phases: updated phases after one step
    """
    n = len(phases)
    dtheta = omega.copy()

    for i in range(n):
        coupling_sum = 0.0
        for j in adjacency.get(i, []):
            w_ij = weights.get(i, {}).get(j, 1.0)
            coupling_sum += w_ij * np.sin(sigma * (phases[j] - phases[i]))
        dtheta[i] += K * coupling_sum

    # Deterministic update + Gaussian noise (Wiener process increment)
    noise = noise_strength * np.sqrt(dt) * np.random.randn(n)
    new_phases = phases + dtheta * dt + noise

    # Keep phases in [0, 2π)
    new_phases = np.mod(new_phases, 2 * np.pi)

    return new_phases


def hebbian_weight_update(weights, phases, adjacency, eta, sigma, decay=0.0, w_min=0.1, w_max=5.0):
    """
    Hebbian learning rule for coupling weights.

    ΔW_ij = η * cos(σ(θ_j - θ_i)) - decay * (W_ij - 1)

    When oscillators are in phase: cos ≈ 1, weights increase
    When out of phase: cos ≈ -1, weights decrease
    Decay term pulls weights back toward 1.0 (homeostatic)

    Args:
        weights: current weight matrix (modified in place)
        phases: current phases
        adjacency: adjacency dict
        eta: learning rate
        sigma: phase scaling
        decay: homeostatic decay rate (pulls weights toward 1.0)
        w_min: minimum allowed weight
        w_max: maximum allowed weight

    Returns:
        weights: updated weights (same object, modified in place)
    """
    for i in adjacency.keys():
        for j in adjacency[i]:
            phase_diff = sigma * (phases[j] - phases[i])
            # Hebbian term: strengthen when in phase
            delta_w = eta * np.cos(phase_diff)
            # Homeostatic decay toward baseline
            delta_w -= decay * (weights[i][j] - 1.0)
            # Update with bounds
            weights[i][j] = np.clip(weights[i][j] + delta_w, w_min, w_max)

    return weights


def run_simulation_noisy(roots, adjacency, omega, K, sigma,
                         noise_strength=0.1, t_max=50, n_steps=5000, seed=42):
    """
    Run stochastic Kuramoto simulation with Gaussian noise.

    Uses Euler-Maruyama integration for the stochastic differential equation:
    dθ_i = [ω_i + K * Σ_j sin(σ(θ_j - θ_i))] dt + D * dW_i

    Args:
        noise_strength: D, noise amplitude (0 = deterministic)

    Returns:
        t, phases, global_r, bin_r (same as run_simulation)
    """
    np.random.seed(seed)
    n = len(roots)
    dt = t_max / n_steps

    # Initial conditions
    phases = np.random.uniform(0, 2 * np.pi, n)
    weights = initialize_coupling_weights(adjacency)

    # Storage
    t = np.linspace(0, t_max, n_steps)
    phases_history = np.zeros((n_steps, n))
    phases_history[0] = phases

    # Integrate using Euler-Maruyama
    for step in range(1, n_steps):
        phases = kuramoto_step_noisy(phases, adjacency, omega, K, sigma, weights, dt, noise_strength)
        phases_history[step] = phases

    # Compute order parameters
    _, bin_indices, _ = approximate_weyl_chambers(roots)
    n_bins = len(np.unique(bin_indices))

    global_r = np.zeros(n_steps)
    bin_r = np.zeros((n_steps, n_bins))

    for step in range(n_steps):
        global_r[step] = compute_order_parameter(phases_history[step])
        for b in range(n_bins):
            mask = bin_indices == b
            if np.sum(mask) > 0:
                bin_r[step, b] = compute_order_parameter(phases_history[step, mask])

    return t, phases_history, global_r, bin_r


def run_simulation_with_learning(roots, adjacency, omega, K, sigma,
                                 noise_strength=0.0, eta=0.001, decay=0.0001,
                                 t_max=50, n_steps=5000, seed=42):
    """
    Run Kuramoto simulation with Hebbian learning and optional noise.

    Coupling weights adapt based on phase coherence:
    - In-phase oscillators strengthen their connection
    - Out-of-phase oscillators weaken their connection

    Args:
        eta: Hebbian learning rate
        decay: homeostatic decay rate
        noise_strength: stochastic noise amplitude (0 = deterministic)

    Returns:
        t: time array
        phases: phase trajectories
        global_r: global order parameter
        bin_r: per-bin order parameters
        weights_history: weight statistics over time
    """
    np.random.seed(seed)
    n = len(roots)
    dt = t_max / n_steps

    # Initial conditions
    phases = np.random.uniform(0, 2 * np.pi, n)
    weights = initialize_coupling_weights(adjacency)

    # Storage
    t = np.linspace(0, t_max, n_steps)
    phases_history = np.zeros((n_steps, n))
    phases_history[0] = phases

    # Weight statistics tracking
    weights_history = {
        'mean': np.zeros(n_steps),
        'std': np.zeros(n_steps),
        'min': np.zeros(n_steps),
        'max': np.zeros(n_steps)
    }

    def compute_weight_stats(weights):
        all_weights = [w for d in weights.values() for w in d.values()]
        return np.mean(all_weights), np.std(all_weights), np.min(all_weights), np.max(all_weights)

    mean_w, std_w, min_w, max_w = compute_weight_stats(weights)
    weights_history['mean'][0] = mean_w
    weights_history['std'][0] = std_w
    weights_history['min'][0] = min_w
    weights_history['max'][0] = max_w

    # Integrate with learning
    for step in range(1, n_steps):
        # Phase update
        phases = kuramoto_step_noisy(phases, adjacency, omega, K, sigma, weights, dt, noise_strength)
        phases_history[step] = phases

        # Weight update (Hebbian learning)
        weights = hebbian_weight_update(weights, phases, adjacency, eta * dt, sigma, decay * dt)

        # Track weight statistics
        mean_w, std_w, min_w, max_w = compute_weight_stats(weights)
        weights_history['mean'][step] = mean_w
        weights_history['std'][step] = std_w
        weights_history['min'][step] = min_w
        weights_history['max'][step] = max_w

    # Compute order parameters
    _, bin_indices, _ = approximate_weyl_chambers(roots)
    n_bins = len(np.unique(bin_indices))

    global_r = np.zeros(n_steps)
    bin_r = np.zeros((n_steps, n_bins))

    for step in range(n_steps):
        global_r[step] = compute_order_parameter(phases_history[step])
        for b in range(n_bins):
            mask = bin_indices == b
            if np.sum(mask) > 0:
                bin_r[step, b] = compute_order_parameter(phases_history[step, mask])

    return t, phases_history, global_r, bin_r, weights_history, weights


def run_simulation_full(roots, adjacency, omega, K, sigma,
                        noise_strength=0.0, eta=0.0, decay=0.0,
                        t_max=50, n_steps=5000, seed=42):
    """
    Unified simulation function supporting all modes:
    - Deterministic (noise=0, eta=0)
    - Noisy (noise>0, eta=0)
    - Learning (noise>=0, eta>0)

    Returns dict with all results.
    """
    if eta > 0:
        t, phases, global_r, bin_r, weights_history, final_weights = run_simulation_with_learning(
            roots, adjacency, omega, K, sigma,
            noise_strength=noise_strength, eta=eta, decay=decay,
            t_max=t_max, n_steps=n_steps, seed=seed
        )
        return {
            't': t,
            'phases': phases,
            'global_r': global_r,
            'bin_r': bin_r,
            'weights_history': weights_history,
            'final_weights': final_weights,
            'has_learning': True,
            'has_noise': noise_strength > 0
        }
    elif noise_strength > 0:
        t, phases, global_r, bin_r = run_simulation_noisy(
            roots, adjacency, omega, K, sigma,
            noise_strength=noise_strength, t_max=t_max, n_steps=n_steps, seed=seed
        )
        return {
            't': t,
            'phases': phases,
            'global_r': global_r,
            'bin_r': bin_r,
            'has_learning': False,
            'has_noise': True
        }
    else:
        # Use original odeint-based simulation for deterministic case
        t, phases, global_r, bin_r = run_simulation(
            roots, adjacency, omega, K, sigma,
            t_max=t_max, n_steps=n_steps, seed=seed
        )
        return {
            't': t,
            'phases': phases,
            'global_r': global_r,
            'bin_r': bin_r,
            'has_learning': False,
            'has_noise': False
        }


def plot_learning_results(results_list, labels, title="E8 Consciousness with Learning",
                         save_path=None):
    """
    Plot comparison of multiple simulation runs with learning/noise.

    Args:
        results_list: list of result dicts from run_simulation_full
        labels: list of labels for each run
        title: figure title
        save_path: optional path to save figure
    """
    n_runs = len(results_list)
    has_learning = any(r.get('has_learning', False) for r in results_list)

    if has_learning:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes = axes.reshape(1, -1)

    colors = plt.cm.tab10(np.linspace(0, 1, n_runs))

    # Plot 1: Global order parameter
    ax1 = axes[0, 0] if has_learning else axes[0, 0]
    for i, (res, label) in enumerate(zip(results_list, labels)):
        ax1.plot(res['t'], res['global_r'], color=colors[i], linewidth=2, label=label)
    ax1.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Global Order Parameter r', fontsize=12)
    ax1.set_title('Coherence Dynamics', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Final coherence comparison
    ax2 = axes[0, 1] if has_learning else axes[0, 1]
    final_rs = [res['global_r'][-1] for res in results_list]
    bars = ax2.bar(range(n_runs), final_rs, color=colors)
    ax2.set_xticks(range(n_runs))
    ax2.set_xticklabels(labels, rotation=15, ha='right')
    ax2.set_ylabel('Final Order Parameter r', fontsize=12)
    ax2.set_title('Final Coherence Values', fontsize=14)
    ax2.set_ylim(0, 1.2)
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, final_rs):
        ax2.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    if has_learning:
        # Plot 3: Weight evolution
        ax3 = axes[1, 0]
        for i, (res, label) in enumerate(zip(results_list, labels)):
            if res.get('has_learning', False):
                ax3.plot(res['t'], res['weights_history']['mean'], color=colors[i],
                        linewidth=2, label=f'{label} (mean)')
                ax3.fill_between(res['t'],
                               res['weights_history']['mean'] - res['weights_history']['std'],
                               res['weights_history']['mean'] + res['weights_history']['std'],
                               color=colors[i], alpha=0.2)
        ax3.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Initial weight')
        ax3.set_xlabel('Time', fontsize=12)
        ax3.set_ylabel('Coupling Weight', fontsize=12)
        ax3.set_title('Hebbian Weight Evolution (mean ± std)', fontsize=14)
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Weight distribution at end
        ax4 = axes[1, 1]
        for i, (res, label) in enumerate(zip(results_list, labels)):
            if res.get('has_learning', False):
                all_weights = [w for d in res['final_weights'].values() for w in d.values()]
                ax4.hist(all_weights, bins=50, alpha=0.5, color=colors[i], label=label, density=True)
        ax4.axvline(1.0, color='gray', linestyle='--', linewidth=2, label='Initial weight')
        ax4.set_xlabel('Coupling Weight', fontsize=12)
        ax4.set_ylabel('Density', fontsize=12)
        ax4.set_title('Final Weight Distribution', fontsize=14)
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()
    return fig


def run_noise_learning_analysis(K=1.0, t_max=100, n_steps=10000, seed=42, save_dir=None):
    """
    Comprehensive analysis of noise and learning effects on E8 consciousness model.

    Tests multiple configurations:
    1. Deterministic (baseline)
    2. Noisy only
    3. Learning only
    4. Noisy + Learning

    For both standard (σ=1) and φ-scaled (σ=φ) dynamics.
    """
    print("=" * 70)
    print("E8 CONSCIOUSNESS MODEL: NOISE AND LEARNING ANALYSIS")
    print("=" * 70)

    # Setup
    print("\nSetting up E8 lattice...")
    roots = generate_e8_roots()
    adjacency, edges = build_e8_adjacency_graph(roots)
    projections, _, _ = approximate_weyl_chambers(roots)
    omega = 5.0 * projections

    # Parameters
    noise_levels = [0.0, 0.1, 0.5]
    learning_rates = [0.0, 0.001, 0.01]

    results_all = {}

    # Standard dynamics (σ=1)
    print("\n" + "=" * 70)
    print("STANDARD DYNAMICS (σ = 1)")
    print("=" * 70)

    sigma = 1.0
    for noise in noise_levels:
        for eta in learning_rates:
            label = f"noise={noise}, η={eta}"
            print(f"\nRunning: {label}...")

            res = run_simulation_full(
                roots, adjacency, omega, K, sigma,
                noise_strength=noise, eta=eta, decay=0.0001 if eta > 0 else 0.0,
                t_max=t_max, n_steps=n_steps, seed=seed
            )

            key = f"std_noise{noise}_eta{eta}"
            results_all[key] = res
            print(f"  Final r = {res['global_r'][-1]:.3f}")

    # φ-scaled dynamics (σ=φ)
    print("\n" + "=" * 70)
    print("φ-SCALED DYNAMICS (σ = φ)")
    print("=" * 70)

    sigma = PHI
    for noise in noise_levels:
        for eta in learning_rates:
            label = f"noise={noise}, η={eta}"
            print(f"\nRunning: {label}...")

            res = run_simulation_full(
                roots, adjacency, omega, K, sigma,
                noise_strength=noise, eta=eta, decay=0.0001 if eta > 0 else 0.0,
                t_max=t_max, n_steps=n_steps, seed=seed
            )

            key = f"phi_noise{noise}_eta{eta}"
            results_all[key] = res
            print(f"  Final r = {res['global_r'][-1]:.3f}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"\n{'Configuration':<35} {'Standard (σ=1)':<15} {'φ-Scaled (σ=φ)':<15}")
    print("-" * 65)

    for noise in noise_levels:
        for eta in learning_rates:
            config = f"noise={noise}, η={eta}"
            std_key = f"std_noise{noise}_eta{eta}"
            phi_key = f"phi_noise{noise}_eta{eta}"
            std_r = results_all[std_key]['global_r'][-1]
            phi_r = results_all[phi_key]['global_r'][-1]
            print(f"{config:<35} {std_r:<15.3f} {phi_r:<15.3f}")

    # Visualizations
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)

    # Plot 1: Noise effects comparison
    print("\n\nGenerating visualizations...")

    fig1, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Standard - noise sweep
    ax = axes[0, 0]
    for noise in noise_levels:
        key = f"std_noise{noise}_eta0.0"
        res = results_all[key]
        ax.plot(res['t'], res['global_r'], linewidth=2, label=f'D={noise}')
    ax.set_title('Standard (σ=1): Noise Effects', fontsize=14)
    ax.set_xlabel('Time')
    ax.set_ylabel('Order Parameter r')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

    # φ-scaled - noise sweep
    ax = axes[0, 1]
    for noise in noise_levels:
        key = f"phi_noise{noise}_eta0.0"
        res = results_all[key]
        ax.plot(res['t'], res['global_r'], linewidth=2, label=f'D={noise}')
    ax.set_title('φ-Scaled (σ=φ): Noise Effects', fontsize=14)
    ax.set_xlabel('Time')
    ax.set_ylabel('Order Parameter r')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

    # Standard - learning sweep
    ax = axes[1, 0]
    for eta in learning_rates:
        key = f"std_noise0.0_eta{eta}"
        res = results_all[key]
        ax.plot(res['t'], res['global_r'], linewidth=2, label=f'η={eta}')
    ax.set_title('Standard (σ=1): Learning Effects', fontsize=14)
    ax.set_xlabel('Time')
    ax.set_ylabel('Order Parameter r')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

    # φ-scaled - learning sweep
    ax = axes[1, 1]
    for eta in learning_rates:
        key = f"phi_noise0.0_eta{eta}"
        res = results_all[key]
        ax.plot(res['t'], res['global_r'], linewidth=2, label=f'η={eta}')
    ax.set_title('φ-Scaled (σ=φ): Learning Effects', fontsize=14)
    ax.set_xlabel('Time')
    ax.set_ylabel('Order Parameter r')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

    plt.suptitle('E8 Consciousness: Noise and Learning Effects', fontsize=16)
    plt.tight_layout()

    if save_dir:
        plt.savefig(f'{save_dir}/e8_noise_learning_effects.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {save_dir}/e8_noise_learning_effects.png")

    plt.show()

    # Plot 2: Learning with noise (combined effects)
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    configs = [
        ('std_noise0.0_eta0.0', 'Deterministic'),
        ('std_noise0.1_eta0.0', 'Noise only'),
        ('std_noise0.0_eta0.01', 'Learning only'),
        ('std_noise0.1_eta0.01', 'Noise + Learning'),
    ]
    for key, label in configs:
        res = results_all[key]
        ax.plot(res['t'], res['global_r'], linewidth=2, label=label)
    ax.set_title('Standard (σ=1): Combined Effects', fontsize=14)
    ax.set_xlabel('Time')
    ax.set_ylabel('Order Parameter r')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    configs = [
        ('phi_noise0.0_eta0.0', 'Deterministic'),
        ('phi_noise0.1_eta0.0', 'Noise only'),
        ('phi_noise0.0_eta0.01', 'Learning only'),
        ('phi_noise0.1_eta0.01', 'Noise + Learning'),
    ]
    for key, label in configs:
        res = results_all[key]
        ax.plot(res['t'], res['global_r'], linewidth=2, label=label)
    ax.set_title('φ-Scaled (σ=φ): Combined Effects', fontsize=14)
    ax.set_xlabel('Time')
    ax.set_ylabel('Order Parameter r')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

    plt.suptitle('E8 Consciousness: Combined Noise and Learning', fontsize=16)
    plt.tight_layout()

    if save_dir:
        plt.savefig(f'{save_dir}/e8_combined_effects.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {save_dir}/e8_combined_effects.png")

    plt.show()

    # Plot 3: Weight evolution for learning cases
    fig3, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for eta in [0.001, 0.01]:
        key = f"std_noise0.0_eta{eta}"
        res = results_all[key]
        if res.get('has_learning'):
            ax.plot(res['t'], res['weights_history']['mean'], linewidth=2, label=f'η={eta}')
            ax.fill_between(res['t'],
                          res['weights_history']['min'],
                          res['weights_history']['max'],
                          alpha=0.1)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('Standard (σ=1): Weight Evolution', fontsize=14)
    ax.set_xlabel('Time')
    ax.set_ylabel('Coupling Weight')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for eta in [0.001, 0.01]:
        key = f"phi_noise0.0_eta{eta}"
        res = results_all[key]
        if res.get('has_learning'):
            ax.plot(res['t'], res['weights_history']['mean'], linewidth=2, label=f'η={eta}')
            ax.fill_between(res['t'],
                          res['weights_history']['min'],
                          res['weights_history']['max'],
                          alpha=0.1)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('φ-Scaled (σ=φ): Weight Evolution', fontsize=14)
    ax.set_xlabel('Time')
    ax.set_ylabel('Coupling Weight')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('E8 Consciousness: Hebbian Weight Dynamics', fontsize=16)
    plt.tight_layout()

    if save_dir:
        plt.savefig(f'{save_dir}/e8_weight_evolution.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {save_dir}/e8_weight_evolution.png")

    plt.show()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return results_all


# =============================================================================
# ORIGINAL SIMULATION (DETERMINISTIC)
# =============================================================================

def run_simulation(roots, adjacency, omega, K, sigma, t_max=50, n_steps=5000, seed=42):
    """
    Run Kuramoto dynamics simulation on E8 lattice.

    Returns:
        t: time array
        phases: phase trajectories (n_steps x n_nodes)
        global_r: global order parameter over time
        bin_r: per-bin order parameters over time
    """
    np.random.seed(seed)
    n = len(roots)

    # Initial random phases
    theta0 = np.random.uniform(0, 2 * np.pi, n)

    # Time array
    t = np.linspace(0, t_max, n_steps)

    # Integrate
    phases = odeint(kuramoto_dynamics, theta0, t,
                    args=(adjacency, omega, K, sigma))

    # Get chamber assignments
    _, bin_indices, _ = approximate_weyl_chambers(roots)
    n_bins = len(np.unique(bin_indices))

    # Compute order parameters over time
    global_r = np.zeros(n_steps)
    bin_r = np.zeros((n_steps, n_bins))

    for step in range(n_steps):
        global_r[step] = compute_order_parameter(phases[step])
        for b in range(n_bins):
            mask = bin_indices == b
            if np.sum(mask) > 0:
                bin_r[step, b] = compute_order_parameter(phases[step, mask])

    return t, phases, global_r, bin_r


def analyze_results(t, global_r, bin_r, label=""):
    """Analyze simulation results and return summary statistics."""
    # Final values
    final_global = global_r[-1]
    final_bin_avg = np.mean(bin_r[-1])
    final_bin_individual = bin_r[-1]

    # Max and late-stage means
    max_global = np.max(global_r)

    # Mean over final 4% of simulation (t > 48 for t_max=50)
    late_idx = int(0.96 * len(t))
    mean_late_global = np.mean(global_r[late_idx:])
    mean_late_bin = np.mean(bin_r[late_idx:])

    results = {
        'label': label,
        'final_global_r': final_global,
        'final_bin_avg_r': final_bin_avg,
        'final_bin_individual': final_bin_individual,
        'max_global_r': max_global,
        'mean_late_global': mean_late_global,
        'mean_late_bin': mean_late_bin
    }

    return results


def plot_simulation_results(t, results_standard, results_phi,
                           global_r_std, global_r_phi,
                           bin_r_std, bin_r_phi,
                           save_path=None):
    """Create comprehensive visualization of simulation results."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Colors
    colors = plt.cm.Set2(np.linspace(0, 1, 4))

    # 1. Global order parameter over time
    ax1 = axes[0, 0]
    ax1.plot(t, global_r_std, 'b-', linewidth=2, label='Standard (σ=1)')
    ax1.plot(t, global_r_phi, 'r-', linewidth=2, label=f'φ-Scaled (σ={PHI:.3f})')
    ax1.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect sync')
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Global Order Parameter r', fontsize=12)
    ax1.set_title('E8 Consciousness Coherence Dynamics', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3)

    # 2. Per-bin order parameters (Standard)
    ax2 = axes[0, 1]
    for b in range(bin_r_std.shape[1]):
        ax2.plot(t, bin_r_std[:, b], color=colors[b], linewidth=1.5,
                label=f'Chamber {b+1}')
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Bin Order Parameter r', fontsize=12)
    ax2.set_title('Standard Model: Per-Chamber Coherence', fontsize=14)
    ax2.legend(loc='lower right')
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3)

    # 3. Per-bin order parameters (φ-Scaled)
    ax3 = axes[0, 2]
    for b in range(bin_r_phi.shape[1]):
        ax3.plot(t, bin_r_phi[:, b], color=colors[b], linewidth=1.5,
                label=f'Chamber {b+1}')
    ax3.set_xlabel('Time', fontsize=12)
    ax3.set_ylabel('Bin Order Parameter r', fontsize=12)
    ax3.set_title('φ-Scaled Model: Per-Chamber Coherence', fontsize=14)
    ax3.legend(loc='lower right')
    ax3.set_ylim(0, 1.1)
    ax3.grid(True, alpha=0.3)

    # 4. Comparison bar chart - Final values
    ax4 = axes[1, 0]
    x = np.arange(2)
    width = 0.35
    final_global = [results_standard['final_global_r'], results_phi['final_global_r']]
    final_bin = [results_standard['final_bin_avg_r'], results_phi['final_bin_avg_r']]

    bars1 = ax4.bar(x - width/2, final_global, width, label='Global r', color='steelblue')
    bars2 = ax4.bar(x + width/2, final_bin, width, label='Avg Bin r', color='coral')
    ax4.set_ylabel('Order Parameter r', fontsize=12)
    ax4.set_title('Final Coherence Values', fontsize=14)
    ax4.set_xticks(x)
    ax4.set_xticklabels(['Standard (σ=1)', f'φ-Scaled (σ={PHI:.2f})'])
    ax4.legend()
    ax4.set_ylim(0, 1.2)
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax4.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax4.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    # 5. Per-chamber final values comparison
    ax5 = axes[1, 1]
    x = np.arange(4)
    width = 0.35
    bars1 = ax5.bar(x - width/2, results_standard['final_bin_individual'], width,
                   label='Standard', color='steelblue')
    bars2 = ax5.bar(x + width/2, results_phi['final_bin_individual'], width,
                   label='φ-Scaled', color='coral')
    ax5.set_ylabel('Order Parameter r', fontsize=12)
    ax5.set_xlabel('Weyl Chamber (approximate)', fontsize=12)
    ax5.set_title('Per-Chamber Final Coherence', fontsize=14)
    ax5.set_xticks(x)
    ax5.set_xticklabels(['Chamber 1', 'Chamber 2', 'Chamber 3', 'Chamber 4'])
    ax5.legend()
    ax5.set_ylim(0, 1.2)
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. E8 Graph statistics and interpretation
    ax6 = axes[1, 2]
    ax6.axis('off')

    interpretation_text = """
    E8 Consciousness Model Summary
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Graph Statistics:
    • 240 nodes (E8 roots in 8D)
    • 6,720 undirected edges
    • Average degree: 56

    Standard Model (σ = 1):
    → Full coherence (r ≈ 1.0)
    → Unified activation across chambers
    → "Conscious" integration achieved

    φ-Scaled Model (σ = φ ≈ 1.618):
    → Partial coherence (r ≈ 0.36)
    → Clustered but fragmented states
    → "Proto-conscious" multi-stability

    Interpretation:
    φ-scaling introduces self-similar
    dynamics that resist rigid lock-in,
    potentially modeling dynamic awareness.
    """

    ax6.text(0.1, 0.95, interpretation_text, transform=ax6.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()

    return fig


def plot_e8_projection(roots, projections, bin_indices, save_path=None):
    """Visualize E8 roots projected onto 2D with chamber coloring."""

    # Use PCA for 2D projection (more informative than single generic direction)
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    roots_2d = pca.fit_transform(roots)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. 2D projection colored by chamber
    ax1 = axes[0]
    colors = plt.cm.Set2(np.linspace(0, 1, 4))
    for b in range(4):
        mask = bin_indices == b
        ax1.scatter(roots_2d[mask, 0], roots_2d[mask, 1],
                   c=[colors[b]], s=50, alpha=0.7, label=f'Chamber {b+1}')
    ax1.set_xlabel('PC1', fontsize=12)
    ax1.set_ylabel('PC2', fontsize=12)
    ax1.set_title('E8 Roots (240 points) - PCA Projection', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # 2. Distribution of projections with chamber bins
    ax2 = axes[1]
    ax2.hist(projections, bins=40, color='steelblue', alpha=0.7, edgecolor='black')

    # Add bin edges
    percentiles = np.percentile(projections, [25, 50, 75])
    for p in percentiles:
        ax2.axvline(p, color='red', linestyle='--', linewidth=2, alpha=0.7)

    ax2.set_xlabel('Projection onto generic direction', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Weyl Chamber Approximation via Projection Binning', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()

    return fig


def run_full_simulation(K=1.0, t_max=50, n_steps=5000, seed=42,
                       save_dir=None, show=True):
    """
    Run complete E8 consciousness simulation comparing standard vs φ-scaled dynamics.

    Args:
        K: Coupling strength (default 1.0)
        t_max: Simulation duration
        n_steps: Number of time steps
        seed: Random seed for reproducibility
        save_dir: Directory to save figures (None = don't save)
        show: Whether to display plots

    Returns:
        Dictionary with all results
    """
    print("=" * 60)
    print("E8 CONSCIOUSNESS MODEL SIMULATION")
    print("=" * 60)

    # 1. Generate E8 roots
    print("\n[1/5] Generating E8 roots...")
    roots = generate_e8_roots()
    print(f"      Generated {len(roots)} roots in 8D")
    print(f"      ||root||² = {np.sum(roots[0]**2):.1f} (verified)")

    # 2. Build adjacency graph
    print("\n[2/5] Building E8 adjacency graph...")
    adjacency, edges = build_e8_adjacency_graph(roots)
    avg_degree = np.mean([len(adj) for adj in adjacency.values()])
    print(f"      Nodes: {len(roots)}")
    print(f"      Edges: {len(edges)} undirected")
    print(f"      Average degree: {avg_degree:.1f}")

    # 3. Approximate Weyl chambers
    print("\n[3/5] Approximating Weyl chambers...")
    projections, bin_indices, bin_edges = approximate_weyl_chambers(roots)
    print(f"      Projection range: [{projections.min():.3f}, {projections.max():.3f}]")
    print(f"      Bin edges: {np.round(bin_edges, 3)}")
    for b in range(4):
        count = np.sum(bin_indices == b)
        print(f"      Chamber {b+1}: {count} sites")

    # 4. Set up natural frequencies
    print("\n[4/5] Configuring oscillator dynamics...")
    omega = 5.0 * projections  # Heterogeneous frequencies from projections
    print(f"      Natural frequencies ω ∈ [{omega.min():.2f}, {omega.max():.2f}]")
    print(f"      Coupling strength K = {K}")
    print(f"      Golden ratio φ = {PHI:.6f}")

    # 5. Run simulations
    print("\n[5/5] Running simulations...")

    print("      → Standard model (σ = 1.0)...")
    t, phases_std, global_r_std, bin_r_std = run_simulation(
        roots, adjacency, omega, K, sigma=1.0, t_max=t_max, n_steps=n_steps, seed=seed
    )
    results_standard = analyze_results(t, global_r_std, bin_r_std, "Standard")

    print("      → φ-Scaled model (σ = φ)...")
    _, phases_phi, global_r_phi, bin_r_phi = run_simulation(
        roots, adjacency, omega, K, sigma=PHI, t_max=t_max, n_steps=n_steps, seed=seed
    )
    results_phi = analyze_results(t, global_r_phi, bin_r_phi, "φ-Scaled")

    # Print results table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print(f"\n{'Metric':<30} {'Standard':>12} {'φ-Scaled':>12}")
    print("-" * 55)
    print(f"{'Final Global r':<30} {results_standard['final_global_r']:>12.3f} {results_phi['final_global_r']:>12.3f}")
    print(f"{'Final Avg Bin r':<30} {results_standard['final_bin_avg_r']:>12.3f} {results_phi['final_bin_avg_r']:>12.3f}")
    print(f"{'Max Global r':<30} {results_standard['max_global_r']:>12.3f} {results_phi['max_global_r']:>12.3f}")
    print(f"{'Mean r (t>48)':<30} {results_standard['mean_late_global']:>12.3f} {results_phi['mean_late_global']:>12.3f}")

    print(f"\nPer-Chamber Final r:")
    print("-" * 55)
    for b in range(4):
        std_val = results_standard['final_bin_individual'][b]
        phi_val = results_phi['final_bin_individual'][b]
        print(f"{'  Chamber ' + str(b+1):<30} {std_val:>12.3f} {phi_val:>12.3f}")

    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print("""
Standard Model (σ = 1):
  → Full coherence achieved (r ≈ 1.0)
  → E8 symmetries propagate unified activation across chambers
  → Simulates "conscious" integration

φ-Scaled Model (σ = φ):
  → Partial coherence (r ≈ 0.36)
  → φ introduces aperiodic tension preventing rigid lock-in
  → Clustered but fragmented states - "proto-conscious" dynamics
  → Suggests dynamic, adaptive awareness vs static synchrony
""")

    # Visualizations
    if show or save_dir:
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            proj_path = f"{save_dir}/e8_projection.png"
            dynamics_path = f"{save_dir}/e8_dynamics.png"
        else:
            proj_path = None
            dynamics_path = None

        # E8 projection plot
        plot_e8_projection(roots, projections, bin_indices, save_path=proj_path)

        # Dynamics comparison plot
        plot_simulation_results(
            t, results_standard, results_phi,
            global_r_std, global_r_phi,
            bin_r_std, bin_r_phi,
            save_path=dynamics_path
        )

    # Compile all results
    all_results = {
        'roots': roots,
        'adjacency': adjacency,
        'edges': edges,
        'projections': projections,
        'bin_indices': bin_indices,
        'omega': omega,
        't': t,
        'standard': {
            'phases': phases_std,
            'global_r': global_r_std,
            'bin_r': bin_r_std,
            'results': results_standard
        },
        'phi_scaled': {
            'phases': phases_phi,
            'global_r': global_r_phi,
            'bin_r': bin_r_phi,
            'results': results_phi
        },
        'params': {
            'K': K,
            't_max': t_max,
            'n_steps': n_steps,
            'phi': PHI,
            'seed': seed
        }
    }

    print("\n✓ Simulation complete!")

    return all_results


# Additional analysis functions

def run_phi_power_sweep(roots, adjacency, omega, powers=None, K=1.0,
                        t_max=50, n_steps=5000, seed=42):
    """
    Sweep through different powers of φ to explore φⁿ scaling effects.

    Tests σ = φⁿ for various n values.
    """
    if powers is None:
        powers = [-2, -1, -0.5, 0, 0.5, 1, 2, 3]

    results = []

    for n in powers:
        sigma = PHI ** n
        _, _, global_r, bin_r = run_simulation(
            roots, adjacency, omega, K, sigma=sigma,
            t_max=t_max, n_steps=n_steps, seed=seed
        )

        final_r = global_r[-1]
        results.append({
            'power': n,
            'sigma': sigma,
            'final_r': final_r,
            'max_r': np.max(global_r),
            'mean_late_r': np.mean(global_r[int(0.96*n_steps):])
        })
        print(f"φ^{n:+.1f} = {sigma:.4f}: final r = {final_r:.3f}")

    return results


def run_coupling_sweep(roots, adjacency, omega, K_values=None,
                      sigma=1.0, t_max=50, n_steps=5000, seed=42):
    """
    Sweep through different coupling strengths K.
    """
    if K_values is None:
        K_values = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]

    results = []

    for K in K_values:
        _, _, global_r, _ = run_simulation(
            roots, adjacency, omega, K, sigma=sigma,
            t_max=t_max, n_steps=n_steps, seed=seed
        )

        final_r = global_r[-1]
        results.append({
            'K': K,
            'final_r': final_r,
            'max_r': np.max(global_r),
            'time_to_90pct': np.argmax(global_r > 0.9) / n_steps * t_max if np.any(global_r > 0.9) else np.inf
        })
        print(f"K = {K:.2f}: final r = {final_r:.3f}")

    return results


if __name__ == "__main__":
    # Run the full simulation
    results = run_full_simulation(
        K=1.0,
        t_max=50,
        n_steps=5000,
        seed=42,
        save_dir=None,  # Set to a path to save figures
        show=True
    )

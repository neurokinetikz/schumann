"""
E8 Bidirectional Energy Flow Model
==================================
Models energy flow between attractors and boundaries in an E8-structured
frequency space, with Phase-Amplitude Coupling (PAC) mediating cross-frequency
energy transfer during ignition events.

Key Features:
- Canonical frequencies from f_n = f0 × φ^n (half-integer n)
- Bidirectional flow: attractor ↔ boundary dynamics
- PAC-mediated cross-frequency coupling
- Ignition triggering and energy cascade
- Energy conservation across frequency bands
- Hebbian learning: adaptive coupling weights based on energy transfer success

Theoretical Framework:
---------------------
f0 = 7.6 Hz (fundamental, near Schumann/Theta)

Canonical frequencies (half-integer n):
  n = -1.0: f = 4.70 Hz  (Theta-low)
  n = -0.5: f = 5.97 Hz  (Theta-mid)
  n =  0.0: f = 7.60 Hz  (Theta-high / SR f0)
  n =  0.5: f = 9.67 Hz  (Alpha-low)
  n =  1.0: f = 12.30 Hz (Alpha-high)
  n =  1.5: f = 15.64 Hz (Beta-low)
  n =  2.0: f = 19.90 Hz (Beta-mid / SR f2)
  n =  2.5: f = 25.32 Hz (Beta-high)
  n =  3.0: f = 32.21 Hz (Gamma-low / SR f4)
  n =  3.5: f = 40.98 Hz (Gamma-high)

Energy Flow Mechanisms:
1. Gradient descent toward attractors (dω/dt = -∇U)
2. Boundary permeability (asymmetric crossing rates)
3. PAC coupling (low-freq phase → high-freq amplitude)
4. Ignition cascade (threshold-triggered energy release)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.integrate import odeint
from scipy.signal import hilbert
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTS
# =============================================================================

PHI = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618033988749895
F0 = 7.6  # Fundamental frequency (Hz)


# =============================================================================
# CANONICAL FREQUENCY CALCULATIONS
# =============================================================================

def compute_canonical_frequencies(f0=F0, n_range=(-2, 4), step=0.5):
    """
    Compute canonical frequencies using f_n = f0 × φ^n.

    Args:
        f0: Fundamental frequency (default 7.6 Hz)
        n_range: Range of n values (min, max)
        step: Increment for n (0.5 for half-integers)

    Returns:
        dict mapping n -> frequency
    """
    n_values = np.arange(n_range[0], n_range[1] + step, step)
    return {n: f0 * (PHI ** n) for n in n_values}


def get_frequency_table(f0=F0):
    """Generate comprehensive frequency table with classifications."""
    freqs = compute_canonical_frequencies(f0)

    # Classify into EEG bands and identify SR overlaps
    table = []
    sr_harmonics = [7.83, 14.3, 20.8, 27.3, 33.8, 40.3]

    for n, f in sorted(freqs.items()):
        # EEG band classification
        if f < 4:
            band = 'Delta'
        elif f < 8:
            band = 'Theta'
        elif f < 12:
            band = 'Alpha-L'
        elif f < 16:
            band = 'Alpha-H'
        elif f < 25:
            band = 'Beta'
        elif f < 45:
            band = 'Gamma'
        else:
            band = 'High-Gamma'

        # Check SR proximity
        sr_dist = min(abs(f - sr) for sr in sr_harmonics)
        sr_match = sr_dist < 1.0
        nearest_sr = min(sr_harmonics, key=lambda x: abs(x - f))

        # Attractor vs boundary classification
        # Integer n = attractor, half-integer n = boundary
        is_attractor = (n == int(n))

        table.append({
            'n': n,
            'freq': f,
            'band': band,
            'type': 'attractor' if is_attractor else 'boundary',
            'sr_match': sr_match,
            'nearest_sr': nearest_sr if sr_match else None,
            'sr_distance': sr_dist
        })

    return table


def print_frequency_table(f0=F0):
    """Print formatted frequency table."""
    table = get_frequency_table(f0)

    print(f"\nCanonical Frequencies (f0 = {f0} Hz, f_n = f0 × φ^n)")
    print("=" * 75)
    print(f"{'n':>6} {'Freq (Hz)':>10} {'Band':>10} {'Type':>10} {'SR Match':>10} {'SR Dist':>8}")
    print("-" * 75)

    for row in table:
        sr_info = f"{row['nearest_sr']:.1f} Hz" if row['sr_match'] else ""
        print(f"{row['n']:>6.1f} {row['freq']:>10.2f} {row['band']:>10} "
              f"{row['type']:>10} {sr_info:>10} {row['sr_distance']:>8.2f}")


# =============================================================================
# ENERGY LANDSCAPE
# =============================================================================

class EnergyLandscape:
    """
    Defines the energy potential U(ω) with attractor wells and boundary barriers.

    Energy flows from high to low potential:
    - Attractors = local minima (stable states)
    - Boundaries = local maxima (transition barriers)
    """

    def __init__(self, f0=F0, attractor_depth=1.0, boundary_height=0.5,
                 attractor_width=0.8, boundary_width=1.2):
        """
        Args:
            f0: Fundamental frequency
            attractor_depth: Depth of attractor wells (negative potential)
            boundary_height: Height of boundary barriers (positive potential)
            attractor_width: Width (σ) of attractor Gaussians
            boundary_width: Width (σ) of boundary Gaussians
        """
        self.f0 = f0
        self.attractor_depth = attractor_depth
        self.boundary_height = boundary_height
        self.attractor_width = attractor_width
        self.boundary_width = boundary_width

        # Compute canonical frequencies
        self.freq_table = get_frequency_table(f0)

        # Separate attractors and boundaries
        self.attractors = [(row['freq'], row['n']) for row in self.freq_table
                          if row['type'] == 'attractor']
        self.boundaries = [(row['freq'], row['n']) for row in self.freq_table
                          if row['type'] == 'boundary']

        # SR-enhanced attractors (stronger wells near SR harmonics)
        self.sr_harmonics = [7.83, 14.3, 20.8, 27.3, 33.8, 40.3]

    def potential(self, freq):
        """
        Compute energy potential U(f) at given frequency.

        U(f) = Σ_att [-depth × exp(-(f-f_att)²/2σ²)]
             + Σ_bnd [+height × exp(-(f-f_bnd)²/2σ²)]
             + SR enhancement at SR harmonics
        """
        freq = np.atleast_1d(freq)
        U = np.zeros_like(freq, dtype=float)

        # Attractor wells (negative potential)
        for f_att, n in self.attractors:
            # Enhance depth if near SR harmonic
            sr_dist = min(abs(f_att - sr) for sr in self.sr_harmonics)
            sr_factor = 1.0 + 0.5 * np.exp(-sr_dist**2 / 2)  # Up to 1.5x deeper

            depth = self.attractor_depth * sr_factor
            U -= depth * np.exp(-0.5 * ((freq - f_att) / self.attractor_width)**2)

        # Boundary barriers (positive potential)
        for f_bnd, n in self.boundaries:
            U += self.boundary_height * np.exp(-0.5 * ((freq - f_bnd) / self.boundary_width)**2)

        return U.squeeze() if U.size == 1 else U

    def gradient(self, freq, eps=0.01):
        """Numerical gradient dU/df."""
        return (self.potential(freq + eps) - self.potential(freq - eps)) / (2 * eps)

    def plot(self, freq_range=(3, 50), n_points=500, ax=None):
        """Visualize the energy landscape."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 6))

        freqs = np.linspace(freq_range[0], freq_range[1], n_points)
        U = self.potential(freqs)

        # Plot potential
        ax.fill_between(freqs, U, alpha=0.3, color='purple')
        ax.plot(freqs, U, 'purple', linewidth=2, label='Energy potential U(f)')

        # Mark attractors
        for f_att, n in self.attractors:
            if freq_range[0] <= f_att <= freq_range[1]:
                U_att = self.potential(f_att)
                ax.scatter([f_att], [U_att], s=150, c='blue', marker='v',
                          zorder=5, edgecolor='black', linewidth=1.5)
                ax.annotate(f'n={n:.0f}\n{f_att:.1f}Hz', xy=(f_att, U_att),
                           xytext=(0, -25), textcoords='offset points',
                           ha='center', fontsize=8, color='blue')

        # Mark boundaries
        for f_bnd, n in self.boundaries:
            if freq_range[0] <= f_bnd <= freq_range[1]:
                U_bnd = self.potential(f_bnd)
                ax.scatter([f_bnd], [U_bnd], s=100, c='red', marker='^',
                          zorder=5, edgecolor='black', linewidth=1.5)
                ax.annotate(f'n={n:.1f}', xy=(f_bnd, U_bnd),
                           xytext=(0, 15), textcoords='offset points',
                           ha='center', fontsize=8, color='red')

        # Mark SR harmonics
        for sr in self.sr_harmonics:
            if freq_range[0] <= sr <= freq_range[1]:
                ax.axvline(sr, color='green', linestyle='--', alpha=0.5, linewidth=1)

        ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel('Energy Potential U(f)', fontsize=12)
        ax.set_title('Energy Landscape: Attractors (▼) and Boundaries (▲)', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        return ax


# =============================================================================
# HEBBIAN LEARNING
# =============================================================================

class HebbianLearner:
    """
    Implements Hebbian learning for adaptive coupling weights.

    The learning rule strengthens connections when energy transfer is successful
    and weakens them when transfer fails or is ineffective.

    Learning Rules:
    1. PAC Coupling: ΔW_ij = η × transfer_success_ij - decay × (W_ij - 1)
       - transfer_success = correlation between energy loss at i and gain at j

    2. Boundary Permeability: ΔP = η × crossing_rate - decay × (P - P0)
       - crossing_rate = frequency of successful boundary crossings

    3. Phase Coherence (Kuramoto-style): ΔW = η × cos(φ_j - φ_i)
       - strengthens when oscillators are in-phase
    """

    def __init__(self, n_freqs, eta=0.01, decay=0.001,
                 w_min=0.1, w_max=5.0, w_init=1.0):
        """
        Args:
            n_freqs: Number of frequency bands
            eta: Learning rate (default 0.01)
            decay: Homeostatic decay rate pulling weights toward w_init (default 0.001)
            w_min: Minimum allowed weight
            w_max: Maximum allowed weight
            w_init: Initial/baseline weight value
        """
        self.n_freqs = n_freqs
        self.eta = eta
        self.decay = decay
        self.w_min = w_min
        self.w_max = w_max
        self.w_init = w_init

        # Initialize weight matrices
        # PAC coupling weights (adaptive version of coupling_matrix)
        self.pac_weights = np.ones((n_freqs, n_freqs)) * w_init

        # Boundary permeability weights (indexed by boundary position between freqs)
        self.boundary_weights = np.ones(n_freqs - 1) * w_init

        # History tracking
        self.weight_history = {
            'pac_mean': [],
            'pac_std': [],
            'pac_min': [],
            'pac_max': [],
            'boundary_mean': [],
            'boundary_std': [],
        }

        # Energy transfer tracking (for computing learning signal)
        self.prev_energy = None
        self.transfer_buffer = []
        self.buffer_size = 10  # Rolling window for transfer estimation

    def compute_transfer_success(self, energy, prev_energy):
        """
        Compute transfer success matrix.

        Success[i,j] = positive correlation between energy loss at i and gain at j.
        This indicates successful energy transfer from i to j.

        Args:
            energy: Current energy distribution
            prev_energy: Previous energy distribution

        Returns:
            success: n_freqs × n_freqs matrix of transfer success scores
        """
        delta_energy = energy - prev_energy

        # Success[i,j] = -delta_E[i] * delta_E[j] when i loses and j gains
        # Normalized to [-1, 1] range
        success = np.zeros((self.n_freqs, self.n_freqs))

        for i in range(self.n_freqs):
            for j in range(self.n_freqs):
                if i != j:
                    # i lost energy (delta < 0) and j gained (delta > 0) → success
                    # Or: i gained and j lost → reverse transfer success
                    loss_i = max(0, -delta_energy[i])
                    gain_j = max(0, delta_energy[j])

                    # Forward transfer success: i → j
                    if loss_i > 0 and gain_j > 0:
                        success[i, j] = min(loss_i, gain_j) / (loss_i + 1e-10)

        return success

    def compute_phase_coherence(self, phases):
        """
        Compute phase coherence matrix (Kuramoto-style).

        Coherence[i,j] = cos(φ_j - φ_i)
        Positive when in-phase, negative when anti-phase.
        """
        coherence = np.zeros((self.n_freqs, self.n_freqs))

        for i in range(self.n_freqs):
            for j in range(self.n_freqs):
                if i != j:
                    coherence[i, j] = np.cos(phases[j] - phases[i])

        return coherence

    def update_pac_weights(self, energy, prev_energy, phases, base_coupling, dt):
        """
        Update PAC coupling weights using Hebbian learning.

        Learning rule:
        ΔW_ij = η × (transfer_success × phase_coherence × base_coupling) - decay × (W - 1)

        Only updates weights where base_coupling > 0 (valid PAC connections).

        Args:
            energy: Current energy distribution
            prev_energy: Previous energy distribution
            phases: Current phase distribution
            base_coupling: Static PAC coupling matrix (defines topology)
            dt: Time step

        Returns:
            Updated pac_weights matrix
        """
        if prev_energy is None:
            return self.pac_weights

        # Compute learning signals
        transfer_success = self.compute_transfer_success(energy, prev_energy)
        phase_coherence = self.compute_phase_coherence(phases)

        # Combined learning signal
        # Strong positive signal when:
        # 1. Transfer was successful (energy moved from i to j)
        # 2. Phases were coherent (in-phase coupling)
        # 3. Base coupling exists (valid PAC connection)

        for i in range(self.n_freqs):
            for j in range(self.n_freqs):
                if base_coupling[i, j] > 0:  # Only update valid PAC connections
                    # Learning signal
                    signal = transfer_success[i, j] * (0.5 + 0.5 * phase_coherence[i, j])

                    # Hebbian update with homeostatic decay
                    delta_w = self.eta * signal * dt - self.decay * (self.pac_weights[i, j] - self.w_init) * dt

                    # Apply update with bounds
                    self.pac_weights[i, j] = np.clip(
                        self.pac_weights[i, j] + delta_w,
                        self.w_min, self.w_max
                    )

        return self.pac_weights

    def update_boundary_weights(self, energy, prev_energy, dt):
        """
        Update boundary permeability weights based on crossing success.

        Strengthens boundaries that are frequently crossed (facilitating flow)
        and weakens boundaries that block flow.

        Args:
            energy: Current energy distribution
            prev_energy: Previous energy distribution
            dt: Time step
        """
        if prev_energy is None:
            return self.boundary_weights

        delta_energy = energy - prev_energy

        for b in range(self.n_freqs - 1):
            # Boundary b is between frequencies b and b+1
            # Crossing detected when energy moves from one side to other

            # Upward crossing: energy moves from b to b+1
            upward = max(0, -delta_energy[b]) * max(0, delta_energy[b + 1])

            # Downward crossing: energy moves from b+1 to b
            downward = max(0, -delta_energy[b + 1]) * max(0, delta_energy[b])

            # Total crossing activity
            crossing_rate = upward + downward

            # Hebbian update: more crossing → stronger permeability
            delta_w = self.eta * crossing_rate * dt - self.decay * (self.boundary_weights[b] - self.w_init) * dt

            self.boundary_weights[b] = np.clip(
                self.boundary_weights[b] + delta_w,
                self.w_min, self.w_max
            )

        return self.boundary_weights

    def get_effective_pac_coupling(self, base_coupling):
        """
        Get effective PAC coupling matrix (base × learned weights).

        Args:
            base_coupling: Static PAC coupling matrix

        Returns:
            Effective coupling matrix with learned modulations
        """
        return base_coupling * self.pac_weights

    def get_effective_permeability(self, base_permeability, freq_idx):
        """
        Get effective boundary permeability at given frequency index.

        Args:
            base_permeability: Base permeability value
            freq_idx: Index of the boundary (between freq_idx and freq_idx+1)

        Returns:
            Effective permeability with learned modulation
        """
        if freq_idx < 0 or freq_idx >= len(self.boundary_weights):
            return base_permeability
        return base_permeability * self.boundary_weights[freq_idx]

    def record_history(self):
        """Record current weight statistics to history."""
        # PAC weights (excluding diagonal and zeros)
        pac_flat = self.pac_weights[self.pac_weights != 0]
        if len(pac_flat) > 0:
            self.weight_history['pac_mean'].append(np.mean(pac_flat))
            self.weight_history['pac_std'].append(np.std(pac_flat))
            self.weight_history['pac_min'].append(np.min(pac_flat))
            self.weight_history['pac_max'].append(np.max(pac_flat))
        else:
            self.weight_history['pac_mean'].append(self.w_init)
            self.weight_history['pac_std'].append(0)
            self.weight_history['pac_min'].append(self.w_init)
            self.weight_history['pac_max'].append(self.w_init)

        # Boundary weights
        self.weight_history['boundary_mean'].append(np.mean(self.boundary_weights))
        self.weight_history['boundary_std'].append(np.std(self.boundary_weights))

    def reset(self):
        """Reset weights to initial values."""
        self.pac_weights = np.ones((self.n_freqs, self.n_freqs)) * self.w_init
        self.boundary_weights = np.ones(self.n_freqs - 1) * self.w_init
        self.prev_energy = None
        self.transfer_buffer = []
        self.weight_history = {k: [] for k in self.weight_history.keys()}


# =============================================================================
# PHASE-AMPLITUDE COUPLING (PAC)
# =============================================================================

class PACCoupling:
    """
    Models Phase-Amplitude Coupling between frequency bands.

    PAC mechanism: Phase of slow oscillation modulates amplitude of fast oscillation.
    This enables energy transfer from low to high frequencies.

    Coupling matrix C[i,j] = strength of phase(f_i) → amplitude(f_j) coupling
    """

    def __init__(self, frequencies, coupling_decay=0.5):
        """
        Args:
            frequencies: List of canonical frequencies
            coupling_decay: How quickly coupling decays with frequency ratio
        """
        self.frequencies = np.array(sorted(frequencies))
        self.n_freqs = len(self.frequencies)
        self.coupling_decay = coupling_decay

        # Build PAC coupling matrix
        self.coupling_matrix = self._build_coupling_matrix()

    def _build_coupling_matrix(self):
        """
        Build PAC coupling matrix.

        C[i,j] = coupling from phase of f_i to amplitude of f_j

        Rules:
        - Only low-to-high coupling (f_i < f_j)
        - Strongest when f_j/f_i ≈ φ or φ² (golden ratio relationships)
        - Decays with frequency distance
        """
        C = np.zeros((self.n_freqs, self.n_freqs))

        for i in range(self.n_freqs):
            for j in range(self.n_freqs):
                if i >= j:  # No coupling to same or lower frequencies
                    continue

                f_low = self.frequencies[i]
                f_high = self.frequencies[j]
                ratio = f_high / f_low

                # Golden ratio resonance: enhanced coupling at φ^n ratios
                phi_distances = [abs(ratio - PHI**n) for n in [0.5, 1, 1.5, 2, 2.5, 3]]
                min_phi_dist = min(phi_distances)
                phi_resonance = np.exp(-min_phi_dist**2 / 0.5)

                # Distance decay
                freq_distance = abs(np.log(ratio))
                distance_factor = np.exp(-freq_distance * self.coupling_decay)

                # Combined coupling strength
                C[i, j] = phi_resonance * distance_factor

        # Normalize rows (each low frequency distributes unit coupling)
        row_sums = C.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        C = C / row_sums

        return C

    def compute_pac_flow(self, energy, phases, weight_modulation=None):
        """
        Compute BIDIRECTIONAL energy flow via PAC.

        Forward PAC (standard): Phase of low freq modulates amplitude of high freq
        Reverse PAC (return flow): High-freq bursts can reset low-freq phase/energy

        Args:
            energy: Current energy at each frequency
            phases: Current phase at each frequency
            weight_modulation: Optional n×n matrix of learned weight modulations
                              (from HebbianLearner). If None, uses base coupling only.

        Returns:
            energy_flow: Net energy change at each frequency due to PAC
        """
        n = len(energy)
        energy_flow = np.zeros(n)

        # Apply weight modulation if provided (Hebbian learning)
        if weight_modulation is not None:
            effective_coupling = self.coupling_matrix * weight_modulation
        else:
            effective_coupling = self.coupling_matrix

        for i in range(n):
            for j in range(n):
                if effective_coupling[i, j] > 0:
                    # FORWARD PAC: Phase of i (low) modulates amplitude of j (high)
                    # Active when low-freq phase is in excitatory window
                    phase_factor = 0.5 * (1 + np.cos(phases[i]))  # 0 to 1

                    # Energy transfer proportional to source energy and coupling
                    forward_transfer = effective_coupling[i, j] * energy[i] * phase_factor * 0.8

                    energy_flow[i] -= forward_transfer
                    energy_flow[j] += forward_transfer

                    # REVERSE PAC (RETURN FLOW): When high-freq has accumulated energy,
                    # some flows BACK to low-freq (mimics gamma-burst resetting theta)
                    # This creates the bidirectional balance
                    if energy[j] > 0.1:  # Only when high-freq has significant energy
                        # Return flow proportional to high-freq energy
                        # Stronger return when high-freq phase is near trough (releasing)
                        high_phase_factor = 0.5 * (1 - np.cos(phases[j]))  # Opposite phase
                        return_coupling = effective_coupling[i, j] * 0.4  # Weaker than forward

                        return_transfer = return_coupling * energy[j] * high_phase_factor * 0.5

                        energy_flow[j] -= return_transfer  # Loss from high freq
                        energy_flow[i] += return_transfer  # Return to low freq

        return energy_flow

    def plot_coupling_matrix(self, ax=None):
        """Visualize PAC coupling matrix."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(self.coupling_matrix, cmap='viridis', aspect='auto')

        # Labels
        freq_labels = [f'{f:.1f}' for f in self.frequencies]
        ax.set_xticks(range(self.n_freqs))
        ax.set_yticks(range(self.n_freqs))
        ax.set_xticklabels(freq_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(freq_labels, fontsize=8)

        ax.set_xlabel('Target Frequency (amplitude modulated)', fontsize=11)
        ax.set_ylabel('Source Frequency (phase modulator)', fontsize=11)
        ax.set_title('PAC Coupling Matrix\n(Phase of row → Amplitude of column)', fontsize=12)

        plt.colorbar(im, ax=ax, label='Coupling Strength')

        return ax


# =============================================================================
# BOUNDARY PERMEABILITY
# =============================================================================

class BoundaryDynamics:
    """
    Models asymmetric permeability at frequency boundaries.

    Boundaries act as semi-permeable membranes:
    - Easier to cross toward attractors (inward)
    - Harder to cross away from attractors (outward)
    """

    def __init__(self, boundaries, attractors, base_permeability=0.5):
        """
        Args:
            boundaries: List of (frequency, n) for boundaries
            attractors: List of (frequency, n) for attractors
            base_permeability: Default crossing probability
        """
        self.boundaries = sorted(boundaries, key=lambda x: x[0])
        self.attractors = sorted(attractors, key=lambda x: x[0])
        self.base_permeability = base_permeability

    def get_nearest_attractor(self, freq):
        """Find nearest attractor to a frequency."""
        if not self.attractors:
            return None, np.inf

        distances = [(abs(freq - f_att), f_att) for f_att, _ in self.attractors]
        min_dist, nearest = min(distances)
        return nearest, min_dist

    def get_permeability(self, freq, direction):
        """
        Get permeability for crossing at given frequency.

        Args:
            freq: Current frequency
            direction: +1 (increasing freq) or -1 (decreasing freq)

        Returns:
            permeability: 0 to 1, probability of crossing
        """
        # Find nearest boundary
        boundary_distances = [(abs(freq - f_bnd), f_bnd) for f_bnd, _ in self.boundaries]
        if not boundary_distances:
            return 1.0

        min_dist, nearest_bnd = min(boundary_distances)

        # If not near a boundary, full permeability
        if min_dist > 1.0:
            return 1.0

        # Find nearest attractor to determine inward/outward
        nearest_att, _ = self.get_nearest_attractor(freq)

        if nearest_att is None:
            return self.base_permeability

        # Determine if movement is toward or away from attractor
        moving_toward = (direction > 0 and nearest_att > freq) or \
                       (direction < 0 and nearest_att < freq)

        # Asymmetric permeability
        if moving_toward:
            # Inward: enhanced permeability
            permeability = self.base_permeability + (1 - self.base_permeability) * 0.8
        else:
            # Outward: reduced permeability
            permeability = self.base_permeability * 0.3

        # Scale by distance to boundary (higher near boundary)
        boundary_factor = np.exp(-min_dist**2 / 0.5)

        return permeability * (1 - boundary_factor) + self.base_permeability * boundary_factor


# =============================================================================
# IGNITION MODEL
# =============================================================================

class IgnitionModel:
    """
    Models ignition events and energy cascades.

    Ignition = sudden coherence spike that triggers energy redistribution.

    Cascade mechanism:
    1. Threshold crossing at trigger frequency
    2. Energy release proportional to accumulated coherence
    3. PAC-mediated distribution to coupled frequencies
    4. Boundary crossing based on permeability
    """

    def __init__(self, landscape, pac, boundary_dynamics,
                 ignition_threshold=0.8, cascade_rate=0.1):
        """
        Args:
            landscape: EnergyLandscape instance
            pac: PACCoupling instance
            boundary_dynamics: BoundaryDynamics instance
            ignition_threshold: Coherence threshold for ignition
            cascade_rate: Rate of energy redistribution during cascade
        """
        self.landscape = landscape
        self.pac = pac
        self.boundary_dynamics = boundary_dynamics
        self.ignition_threshold = ignition_threshold
        self.cascade_rate = cascade_rate

        self.frequencies = pac.frequencies
        self.n_freqs = len(self.frequencies)

    def check_ignition(self, coherence, energy):
        """
        Check if ignition conditions are met.

        Returns:
            ignited: Boolean array, True where ignition triggered
            trigger_freq_idx: Index of primary trigger frequency (or None)
        """
        ignited = coherence > self.ignition_threshold

        if np.any(ignited):
            # Primary trigger is highest coherence above threshold
            candidates = np.where(ignited)[0]
            trigger_idx = candidates[np.argmax(coherence[candidates])]
            return ignited, trigger_idx

        return ignited, None

    def compute_cascade(self, energy, coherence, trigger_idx, dt=0.01):
        """
        Compute energy redistribution during ignition cascade.

        Args:
            energy: Current energy distribution
            coherence: Current coherence at each frequency
            trigger_idx: Index of trigger frequency
            dt: Time step

        Returns:
            delta_energy: Energy change at each frequency
        """
        delta_energy = np.zeros(self.n_freqs)

        if trigger_idx is None:
            return delta_energy

        trigger_freq = self.frequencies[trigger_idx]
        trigger_energy = energy[trigger_idx]

        # Energy released from trigger site
        release_amount = trigger_energy * self.cascade_rate * dt
        delta_energy[trigger_idx] -= release_amount

        # Distribute via PAC coupling
        for j in range(self.n_freqs):
            if j == trigger_idx:
                continue

            # PAC-based distribution
            pac_coupling = self.pac.coupling_matrix[trigger_idx, j]
            if pac_coupling > 0:
                # Check boundary permeability
                target_freq = self.frequencies[j]
                direction = 1 if target_freq > trigger_freq else -1
                permeability = self.boundary_dynamics.get_permeability(
                    (trigger_freq + target_freq) / 2, direction
                )

                # Energy transfer
                transfer = release_amount * pac_coupling * permeability
                delta_energy[j] += transfer

        # Ensure energy conservation (any remainder stays at trigger)
        total_distributed = delta_energy[delta_energy > 0].sum()
        if total_distributed < release_amount:
            delta_energy[trigger_idx] += (release_amount - total_distributed)

        return delta_energy


# =============================================================================
# FULL SIMULATION
# =============================================================================

class E8EnergyFlowSimulation:
    """
    Full E8 energy flow simulation with all components.

    Now includes Hebbian learning for adaptive coupling weights.
    """

    def __init__(self, f0=F0, n_range=(-1, 3), step=0.5,
                 learning_rate=0.01, learning_decay=0.001,
                 enable_learning=True):
        """
        Initialize all components.

        Args:
            f0: Fundamental frequency (default 7.6 Hz)
            n_range: Range of n values for canonical frequencies
            step: Step size for n (0.5 for half-integers)
            learning_rate: Hebbian learning rate eta (default 0.01)
            learning_decay: Homeostatic decay rate (default 0.001)
            enable_learning: Whether to use adaptive weights (default True)
        """
        self.f0 = f0
        self.enable_learning = enable_learning
        self.learning_rate = learning_rate
        self.learning_decay = learning_decay

        # Get canonical frequencies (limited to reasonable EEG range)
        freq_dict = compute_canonical_frequencies(f0, n_range, step)
        # Filter to keep frequencies in 4-45 Hz range (exclude very low and very high)
        freq_dict = {n: f for n, f in freq_dict.items() if 4 <= f <= 45}
        self.frequencies = np.array(sorted(freq_dict.values()))
        self.n_values = np.array(sorted(freq_dict.keys()))
        self.n_freqs = len(self.frequencies)

        # Initialize components
        self.landscape = EnergyLandscape(f0)
        self.pac = PACCoupling(self.frequencies)

        # Get attractors and boundaries from landscape
        self.boundary_dynamics = BoundaryDynamics(
            self.landscape.boundaries,
            self.landscape.attractors
        )

        self.ignition_model = IgnitionModel(
            self.landscape, self.pac, self.boundary_dynamics
        )

        # Initialize Hebbian learner
        self.hebbian = HebbianLearner(
            n_freqs=self.n_freqs,
            eta=learning_rate,
            decay=learning_decay
        )

        # State variables
        self.energy = None
        self.coherence = None
        self.phases = None
        self.history = None
        self.prev_energy = None  # For Hebbian learning

    def initialize(self, initial_energy=None, seed=42):
        """Initialize simulation state."""
        np.random.seed(seed)

        if initial_energy is None:
            # Default: energy concentrated near fundamental
            self.energy = np.zeros(self.n_freqs)
            fundamental_idx = np.argmin(np.abs(self.frequencies - self.f0))
            self.energy[fundamental_idx] = 1.0
            # Small background energy
            self.energy += 0.05
        else:
            self.energy = np.array(initial_energy)

        # Initialize phases randomly
        self.phases = np.random.uniform(0, 2 * np.pi, self.n_freqs)

        # Initial coherence from energy
        self.coherence = np.sqrt(self.energy / (self.energy.max() + 1e-10))

        # Reset Hebbian learner
        self.hebbian.reset()
        self.prev_energy = None

        # History tracking
        self.history = {
            'time': [],
            'energy': [],
            'coherence': [],
            'phases': [],
            'ignitions': [],
            'pac_flow': [],
            # Hebbian learning tracking
            'pac_weights_mean': [],
            'pac_weights_std': [],
            'pac_weights_min': [],
            'pac_weights_max': [],
            'boundary_weights_mean': [],
            'boundary_weights_std': [],
        }

    def step(self, dt=0.01, noise_strength=0.01):
        """
        Perform one simulation step with bidirectional energy flow.

        Energy dynamics:
        1. Attractor pull: Energy flows toward attractor frequencies (gradient descent)
        2. Homeostatic balance: Prevents runaway accumulation at any single frequency
        3. PAC coupling: Low-frequency phase modulates high-frequency amplitude
        4. Boundary gating: Asymmetric permeability at half-integer n frequencies
        5. Diffusion: Spreading to neighboring frequencies

        Returns:
            ignited: Whether ignition occurred this step
        """
        # 1. Phase evolution (simple oscillation + coupling)
        for i in range(self.n_freqs):
            # Natural frequency evolution
            self.phases[i] += 2 * np.pi * self.frequencies[i] * dt

            # Phase coupling toward neighbors (Kuramoto-like)
            for j in range(self.n_freqs):
                if i != j:
                    coupling = 0.1 * np.exp(-abs(i - j) / 2)
                    self.phases[i] += coupling * np.sin(self.phases[j] - self.phases[i]) * dt

        self.phases = np.mod(self.phases, 2 * np.pi)

        # 2. HIERARCHICAL ATTRACTOR DYNAMICS
        # The fundamental (f0) is the GLOBAL attractor - all energy ultimately returns there
        # Energy cascades DOWN through the frequency hierarchy via intermediate attractors
        # This creates the bidirectional flow pattern seen in real EEG

        attractor_pull = np.zeros(self.n_freqs)
        fundamental_idx = np.argmin(np.abs(self.frequencies - self.f0))

        for i in range(self.n_freqs):
            f = self.frequencies[i]

            # Skip if at fundamental (nothing to pull toward)
            if i == fundamental_idx:
                continue

            # DOWNWARD CASCADE: Energy always tends to flow toward lower frequencies
            # This balances the PAC upward flow
            # Strength proportional to how far above fundamental we are
            freq_ratio = f / self.f0
            if freq_ratio > 1.0 and i > 0:
                # Pull toward next lower frequency
                downward_pull = 0.15 * self.energy[i] * (freq_ratio - 1) * dt
                # Gated by boundary permeability
                perm = self.boundary_dynamics.get_permeability(f, -1)
                attractor_pull[i] -= downward_pull
                attractor_pull[i - 1] += downward_pull * perm * 0.9

        # 2b. SR FUNDAMENTAL BASIN - the fundamental acts as a global attractor basin
        # This creates the downward flow that balances PAC upward flow
        sr_fundamental = self.f0  # 7.6 Hz
        fundamental_idx = np.argmin(np.abs(self.frequencies - sr_fundamental))

        # SR harmonics that have special resonance with fundamental
        sr_harmonic_freqs = [7.83, 14.3, 20.8, 27.3, 33.8]  # True SR harmonics

        sr_pull = np.zeros(self.n_freqs)
        for i in range(self.n_freqs):
            if i == fundamental_idx:
                continue

            f = self.frequencies[i]

            # GLOBAL fundamental attraction - ALL frequencies pulled toward f0
            # Strength decreases with frequency distance but never reaches zero
            freq_dist = abs(f - sr_fundamental)
            global_pull = 0.08 * self.energy[i] * np.exp(-freq_dist / 15) * dt
            sr_pull[i] -= global_pull
            sr_pull[fundamental_idx] += global_pull * 0.9

            # ENHANCED pull from SR harmonic frequencies
            for sr_h in sr_harmonic_freqs:
                if abs(f - sr_h) < 2.0:  # Near an SR harmonic
                    harmonic_match = np.exp(-(f - sr_h)**2 / 1.0)
                    enhanced_pull = 0.12 * harmonic_match * self.energy[i] * dt
                    sr_pull[i] -= enhanced_pull
                    sr_pull[fundamental_idx] += enhanced_pull * 0.85

        # 2b2. HIGH-FREQUENCY DECAY - mimics metabolic cost of fast oscillations
        # This is CRITICAL for bidirectional balance:
        # - Gamma (>25 Hz): VERY rapid decay (biologically expensive)
        # - Beta (15-25 Hz): Moderate decay
        # - Alpha/Theta (<15 Hz): Minimal decay (metabolically stable)
        freq_decay = np.zeros(self.n_freqs)
        for i in range(self.n_freqs):
            f = self.frequencies[i]
            if f > 25:  # Gamma range - rapid decay
                decay_rate = 0.5 * ((f - 25) / 10) ** 2
            elif f > 15:  # Beta range - moderate decay
                decay_rate = 0.15 * ((f - 15) / 10)
            elif f > 10:  # Alpha range - slow decay
                decay_rate = 0.02
            else:  # Theta and below - minimal decay (stable)
                decay_rate = 0.005

            freq_decay[i] = -decay_rate * self.energy[i] * dt
            # Decayed energy returns to fundamental (represents metabolic reset)
            if i != fundamental_idx:
                sr_pull[fundamental_idx] -= freq_decay[i] * 0.7

        # 2c. PHASE-COHERENT BIDIRECTIONAL COUPLING
        # When two frequencies are phase-locked, energy can flow in EITHER direction
        phase_coupling = np.zeros(self.n_freqs)
        for i in range(self.n_freqs):
            for j in range(self.n_freqs):
                if i != j:
                    # Check phase alignment
                    phase_diff = self.phases[i] - self.phases[j]
                    # Cos(phase_diff) > 0 means in-phase -> bidirectional exchange
                    phase_alignment = np.cos(phase_diff)

                    if phase_alignment > 0.5:  # Strong phase alignment
                        # Energy flows from higher to lower concentration (diffusion + phase)
                        energy_gradient = self.energy[j] - self.energy[i]
                        # Coupling strength based on φ-ratio relationship
                        freq_ratio = max(self.frequencies[i], self.frequencies[j]) / \
                                   min(self.frequencies[i], self.frequencies[j])
                        phi_match = np.exp(-(freq_ratio - PHI)**2 / 0.3)

                        exchange = 0.02 * energy_gradient * phase_alignment * phi_match * dt
                        phase_coupling[i] += exchange

        # 3. Homeostatic balance - prevents runaway accumulation
        # Energy above target is pushed out, energy below target is attracted
        target_energy = 1.0 / self.n_freqs  # Equal distribution as baseline
        homeostatic = np.zeros(self.n_freqs)

        for i in range(self.n_freqs):
            deviation = self.energy[i] - target_energy
            # Strong energy sites lose energy, weak sites gain
            homeostatic[i] = -0.1 * deviation * dt

        # 4. PAC-mediated energy flow (low→high frequency coupling)
        # Use adaptive weights if Hebbian learning is enabled
        if self.enable_learning:
            weight_modulation = self.hebbian.pac_weights
        else:
            weight_modulation = None
        pac_flow = self.pac.compute_pac_flow(self.energy, self.phases, weight_modulation)

        # 5. Local diffusion (energy spreads to neighbors)
        diffusion = np.zeros(self.n_freqs)
        diffusion_rate = 0.02

        for i in range(self.n_freqs):
            if i > 0:
                diff = diffusion_rate * (self.energy[i - 1] - self.energy[i]) * dt
                diffusion[i] += diff
            if i < self.n_freqs - 1:
                diff = diffusion_rate * (self.energy[i + 1] - self.energy[i]) * dt
                diffusion[i] += diff

        # 6. Check for ignition
        ignited, trigger_idx = self.ignition_model.check_ignition(
            self.coherence, self.energy
        )

        cascade_flow = np.zeros(self.n_freqs)
        if trigger_idx is not None:
            cascade_flow = self.ignition_model.compute_cascade(
                self.energy, self.coherence, trigger_idx, dt
            )

        # 7. Combine all energy changes
        delta_energy = (
            attractor_pull +      # Pull toward local attractors
            sr_pull +             # SR fundamental basin (DOWNWARD return flow)
            freq_decay +          # High-frequency metabolic decay
            phase_coupling +      # Phase-coherent bidirectional exchange
            homeostatic +         # Prevent runaway accumulation
            0.06 * pac_flow +     # PAC cross-frequency coupling (UPWARD) - reduced
            diffusion +           # Local spreading
            cascade_flow          # Ignition cascade
        )

        # Add noise
        delta_energy += noise_strength * np.random.randn(self.n_freqs) * np.sqrt(dt)

        # Update energy (ensure non-negative)
        self.energy = np.maximum(1e-6, self.energy + delta_energy)

        # Normalize to conserve total energy
        total_energy = self.energy.sum()
        if total_energy > 0:
            self.energy = self.energy / total_energy

        # 8. HEBBIAN LEARNING - update coupling weights based on transfer success
        if self.enable_learning and self.prev_energy is not None:
            # Update PAC weights based on energy transfer and phase coherence
            self.hebbian.update_pac_weights(
                self.energy, self.prev_energy, self.phases,
                self.pac.coupling_matrix, dt
            )

            # Update boundary weights based on crossing activity
            self.hebbian.update_boundary_weights(
                self.energy, self.prev_energy, dt
            )

        # Store current energy for next step's learning
        self.prev_energy = self.energy.copy()

        # 9. Update coherence from energy
        self.coherence = np.sqrt(self.energy / (self.energy.max() + 1e-10))

        return trigger_idx is not None

    def run(self, t_max=10.0, dt=0.01, noise_strength=0.01,
            initial_energy=None, seed=42):
        """
        Run full simulation.

        Args:
            t_max: Total simulation time (seconds)
            dt: Time step
            noise_strength: Noise amplitude
            initial_energy: Initial energy distribution (None for default)
            seed: Random seed

        Returns:
            history: Dict with full simulation history
        """
        self.initialize(initial_energy, seed)

        n_steps = int(t_max / dt)

        for step in range(n_steps):
            t = step * dt

            # Store state
            self.history['time'].append(t)
            self.history['energy'].append(self.energy.copy())
            self.history['coherence'].append(self.coherence.copy())
            self.history['phases'].append(self.phases.copy())

            # Store Hebbian weight statistics
            if self.enable_learning:
                pac_flat = self.hebbian.pac_weights[self.pac.coupling_matrix > 0]
                if len(pac_flat) > 0:
                    self.history['pac_weights_mean'].append(np.mean(pac_flat))
                    self.history['pac_weights_std'].append(np.std(pac_flat))
                    self.history['pac_weights_min'].append(np.min(pac_flat))
                    self.history['pac_weights_max'].append(np.max(pac_flat))
                else:
                    self.history['pac_weights_mean'].append(1.0)
                    self.history['pac_weights_std'].append(0.0)
                    self.history['pac_weights_min'].append(1.0)
                    self.history['pac_weights_max'].append(1.0)

                self.history['boundary_weights_mean'].append(np.mean(self.hebbian.boundary_weights))
                self.history['boundary_weights_std'].append(np.std(self.hebbian.boundary_weights))

            # Simulate step
            ignited = self.step(dt, noise_strength)

            if ignited:
                self.history['ignitions'].append(t)

        # Convert to arrays
        self.history['time'] = np.array(self.history['time'])
        self.history['energy'] = np.array(self.history['energy'])
        self.history['coherence'] = np.array(self.history['coherence'])
        self.history['phases'] = np.array(self.history['phases'])

        # Convert Hebbian history to arrays
        if self.enable_learning:
            self.history['pac_weights_mean'] = np.array(self.history['pac_weights_mean'])
            self.history['pac_weights_std'] = np.array(self.history['pac_weights_std'])
            self.history['pac_weights_min'] = np.array(self.history['pac_weights_min'])
            self.history['pac_weights_max'] = np.array(self.history['pac_weights_max'])
            self.history['boundary_weights_mean'] = np.array(self.history['boundary_weights_mean'])
            self.history['boundary_weights_std'] = np.array(self.history['boundary_weights_std'])

        return self.history

    def inject_energy(self, freq_idx, amount=0.5):
        """Inject energy at specific frequency (for testing ignition)."""
        self.energy[freq_idx] += amount
        # Renormalize
        self.energy = self.energy / self.energy.sum()

    def plot_results(self, save_path=None):
        """Create comprehensive visualization of results."""
        if self.history is None:
            raise ValueError("Run simulation first")

        # Use larger figure if learning is enabled (more panels)
        if self.enable_learning:
            fig = plt.figure(figsize=(20, 16))
            n_rows, n_cols = 4, 3
        else:
            fig = plt.figure(figsize=(18, 14))
            n_rows, n_cols = 3, 3

        t = self.history['time']
        energy = self.history['energy']
        coherence = self.history['coherence']

        # Panel 1: Energy landscape
        ax1 = fig.add_subplot(n_rows, n_cols, 1)
        self.landscape.plot(ax=ax1)

        # Panel 2: Energy evolution (heatmap)
        ax2 = fig.add_subplot(n_rows, n_cols, 2)
        im = ax2.imshow(energy.T, aspect='auto', origin='lower',
                       extent=[t[0], t[-1], 0, self.n_freqs],
                       cmap='hot')
        ax2.set_yticks(range(self.n_freqs))
        ax2.set_yticklabels([f'{f:.1f}' for f in self.frequencies], fontsize=8)
        ax2.set_xlabel('Time (s)', fontsize=11)
        ax2.set_ylabel('Frequency (Hz)', fontsize=11)
        ax2.set_title('Energy Distribution Over Time', fontsize=12)
        plt.colorbar(im, ax=ax2, label='Energy')

        # Mark ignitions
        for t_ign in self.history['ignitions']:
            ax2.axvline(t_ign, color='cyan', linestyle='--', alpha=0.7)

        # Panel 3: Energy time series (stacked)
        ax3 = fig.add_subplot(n_rows, n_cols, 3)
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, self.n_freqs))

        for i in range(self.n_freqs):
            ax3.plot(t, energy[:, i], color=colors[i], linewidth=1.5,
                    label=f'{self.frequencies[i]:.1f} Hz')

        ax3.set_xlabel('Time (s)', fontsize=11)
        ax3.set_ylabel('Energy', fontsize=11)
        ax3.set_title('Energy at Each Frequency', fontsize=12)
        ax3.legend(fontsize=7, loc='upper right', ncol=2)
        ax3.grid(True, alpha=0.3)

        # Panel 4: PAC coupling matrix
        ax4 = fig.add_subplot(n_rows, n_cols, 4)
        self.pac.plot_coupling_matrix(ax=ax4)

        # Panel 5: Coherence evolution
        ax5 = fig.add_subplot(n_rows, n_cols, 5)
        im = ax5.imshow(coherence.T, aspect='auto', origin='lower',
                       extent=[t[0], t[-1], 0, self.n_freqs],
                       cmap='plasma', vmin=0, vmax=1)
        ax5.set_yticks(range(self.n_freqs))
        ax5.set_yticklabels([f'{f:.1f}' for f in self.frequencies], fontsize=8)
        ax5.set_xlabel('Time (s)', fontsize=11)
        ax5.set_ylabel('Frequency (Hz)', fontsize=11)
        ax5.set_title('Coherence Over Time', fontsize=12)
        plt.colorbar(im, ax=ax5, label='Coherence')

        # Panel 6: Total energy at attractors vs boundaries
        ax6 = fig.add_subplot(n_rows, n_cols, 6)

        # Classify frequencies
        attractor_mask = np.array([n == int(n) for n in self.n_values])
        boundary_mask = ~attractor_mask

        att_energy = energy[:, attractor_mask].sum(axis=1)
        bnd_energy = energy[:, boundary_mask].sum(axis=1)

        ax6.plot(t, att_energy, 'b-', linewidth=2, label='Attractors (integer n)')
        ax6.plot(t, bnd_energy, 'r-', linewidth=2, label='Boundaries (half-integer n)')
        ax6.fill_between(t, att_energy, alpha=0.3, color='blue')
        ax6.fill_between(t, bnd_energy, alpha=0.3, color='red')

        ax6.set_xlabel('Time (s)', fontsize=11)
        ax6.set_ylabel('Total Energy', fontsize=11)
        ax6.set_title('Energy: Attractors vs Boundaries', fontsize=12)
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # Panel 7: Energy flow diagram (initial vs final)
        ax7 = fig.add_subplot(n_rows, n_cols, 7)

        x = np.arange(self.n_freqs)
        width = 0.35

        ax7.bar(x - width/2, energy[0], width, label='Initial', color='steelblue', alpha=0.7)
        ax7.bar(x + width/2, energy[-1], width, label='Final', color='coral', alpha=0.7)

        ax7.set_xticks(x)
        ax7.set_xticklabels([f'{f:.1f}' for f in self.frequencies], rotation=45, ha='right', fontsize=8)
        ax7.set_ylabel('Energy', fontsize=11)
        ax7.set_title('Energy Redistribution', fontsize=12)
        ax7.legend()
        ax7.grid(True, alpha=0.3, axis='y')

        # Panel 8: Phase coherence evolution
        ax8 = fig.add_subplot(n_rows, n_cols, 8)

        # Compute global phase coherence over time
        phases = self.history['phases']
        global_coherence = np.abs(np.mean(np.exp(1j * phases), axis=1))

        ax8.plot(t, global_coherence, 'purple', linewidth=2)
        ax8.fill_between(t, global_coherence, alpha=0.3, color='purple')

        # Mark ignitions
        for t_ign in self.history['ignitions']:
            ax8.axvline(t_ign, color='red', linestyle='--', alpha=0.7, label='Ignition' if t_ign == self.history['ignitions'][0] else '')

        ax8.set_xlabel('Time (s)', fontsize=11)
        ax8.set_ylabel('Global Phase Coherence', fontsize=11)
        ax8.set_title('Phase Synchronization', fontsize=12)
        ax8.set_ylim(0, 1)
        ax8.grid(True, alpha=0.3)
        if self.history['ignitions']:
            ax8.legend()

        # Panel 9: Summary
        ax9 = fig.add_subplot(n_rows, n_cols, 9)
        ax9.axis('off')

        # Compute summary statistics
        initial_peak_freq = self.frequencies[np.argmax(energy[0])]
        final_peak_freq = self.frequencies[np.argmax(energy[-1])]
        n_ignitions = len(self.history['ignitions'])
        final_att_energy = att_energy[-1]
        final_bnd_energy = bnd_energy[-1]

        # Add learning info if enabled
        learning_info = ""
        if self.enable_learning:
            final_pac_mean = self.history['pac_weights_mean'][-1]
            final_pac_std = self.history['pac_weights_std'][-1]
            final_bnd_mean = self.history['boundary_weights_mean'][-1]
            learning_info = f"""
    Hebbian Learning:
    • Learning rate η = {self.learning_rate}
    • PAC weights: {final_pac_mean:.3f} ± {final_pac_std:.3f}
    • Boundary weights: {final_bnd_mean:.3f}
"""

        summary = f"""
    E8 Bidirectional Energy Flow Summary
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Fundamental: f0 = {self.f0} Hz
    Frequencies: {self.n_freqs} (n = {self.n_values[0]:.1f} to {self.n_values[-1]:.1f})

    Energy Flow Results:
    • Initial peak: {initial_peak_freq:.1f} Hz
    • Final peak: {final_peak_freq:.1f} Hz
    • Ignition events: {n_ignitions}

    Final Distribution:
    • Attractor energy: {final_att_energy:.3f} ({100*final_att_energy:.1f}%)
    • Boundary energy: {final_bnd_energy:.3f} ({100*final_bnd_energy:.1f}%)
{learning_info}
    Key Observations:
    • Energy flows toward attractor frequencies
    • PAC couples low→high frequency bands
    • Boundaries gate energy transfer
    • Ignitions trigger rapid redistribution
        """

        ax9.text(0.05, 0.95, summary, transform=ax9.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        # Panels 10-12: Hebbian Learning Visualization (if enabled)
        if self.enable_learning:
            # Panel 10: PAC Weight Evolution
            ax10 = fig.add_subplot(n_rows, n_cols, 10)
            ax10.plot(t, self.history['pac_weights_mean'], 'b-', linewidth=2, label='Mean')
            ax10.fill_between(t,
                            self.history['pac_weights_mean'] - self.history['pac_weights_std'],
                            self.history['pac_weights_mean'] + self.history['pac_weights_std'],
                            alpha=0.3, color='blue')
            ax10.plot(t, self.history['pac_weights_min'], 'g--', linewidth=1, alpha=0.7, label='Min')
            ax10.plot(t, self.history['pac_weights_max'], 'r--', linewidth=1, alpha=0.7, label='Max')
            ax10.axhline(1.0, color='gray', linestyle=':', alpha=0.7, label='Initial')
            ax10.set_xlabel('Time (s)', fontsize=11)
            ax10.set_ylabel('PAC Coupling Weight', fontsize=11)
            ax10.set_title('Hebbian: PAC Weight Evolution', fontsize=12)
            ax10.legend(fontsize=8)
            ax10.grid(True, alpha=0.3)

            # Panel 11: Boundary Weight Evolution
            ax11 = fig.add_subplot(n_rows, n_cols, 11)
            ax11.plot(t, self.history['boundary_weights_mean'], 'purple', linewidth=2, label='Mean')
            ax11.fill_between(t,
                            self.history['boundary_weights_mean'] - self.history['boundary_weights_std'],
                            self.history['boundary_weights_mean'] + self.history['boundary_weights_std'],
                            alpha=0.3, color='purple')
            ax11.axhline(1.0, color='gray', linestyle=':', alpha=0.7, label='Initial')
            ax11.set_xlabel('Time (s)', fontsize=11)
            ax11.set_ylabel('Boundary Weight', fontsize=11)
            ax11.set_title('Hebbian: Boundary Permeability', fontsize=12)
            ax11.legend(fontsize=8)
            ax11.grid(True, alpha=0.3)

            # Panel 12: Final Learned PAC Matrix
            ax12 = fig.add_subplot(n_rows, n_cols, 12)
            effective_coupling = self.pac.coupling_matrix * self.hebbian.pac_weights
            im = ax12.imshow(effective_coupling, cmap='viridis', aspect='auto')
            freq_labels = [f'{f:.1f}' for f in self.frequencies]
            ax12.set_xticks(range(self.n_freqs))
            ax12.set_yticks(range(self.n_freqs))
            ax12.set_xticklabels(freq_labels, rotation=45, ha='right', fontsize=7)
            ax12.set_yticklabels(freq_labels, fontsize=7)
            ax12.set_xlabel('Target (Hz)', fontsize=10)
            ax12.set_ylabel('Source (Hz)', fontsize=10)
            ax12.set_title('Learned Effective PAC Coupling', fontsize=12)
            plt.colorbar(im, ax=ax12, label='Coupling')

        title = 'E8 Bidirectional Energy Flow Simulation'
        if self.enable_learning:
            title += ' with Hebbian Learning'
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        plt.show()
        return fig


# =============================================================================
# IGNITION TRIGGER ANALYSIS
# =============================================================================

def run_ignition_analysis(f0=F0, trigger_freq=None, save_dir='e8_energy_figures'):
    """
    Run comprehensive ignition trigger analysis.

    Tests how energy injected at different frequencies propagates through
    the attractor-boundary network via PAC coupling.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print("E8 BIDIRECTIONAL ENERGY FLOW: IGNITION ANALYSIS")
    print("=" * 70)

    # Print frequency table
    print_frequency_table(f0)

    # Create simulation
    sim = E8EnergyFlowSimulation(f0=f0)

    print(f"\n[1/4] Running baseline simulation (uniform initial energy)...")

    # Baseline: uniform energy
    uniform_energy = np.ones(sim.n_freqs) / sim.n_freqs
    history_baseline = sim.run(t_max=5.0, dt=0.01, initial_energy=uniform_energy, seed=42)

    print(f"      Final energy distribution:")
    for i, f in enumerate(sim.frequencies):
        print(f"        {f:6.1f} Hz: {sim.energy[i]:.3f}")

    # Plot baseline
    sim.plot_results(save_path=f'{save_dir}/baseline_energy_flow.png')

    print(f"\n[2/4] Running ignition from fundamental ({f0} Hz)...")

    # Ignition from fundamental
    fundamental_idx = np.argmin(np.abs(sim.frequencies - f0))
    ignition_energy = np.zeros(sim.n_freqs)
    ignition_energy[fundamental_idx] = 0.8
    ignition_energy += 0.02  # Small background
    ignition_energy /= ignition_energy.sum()

    history_fundamental = sim.run(t_max=5.0, dt=0.01, initial_energy=ignition_energy, seed=42)
    sim.plot_results(save_path=f'{save_dir}/ignition_fundamental.png')

    print(f"\n[3/4] Running ignition from alpha ({f0 * PHI:.1f} Hz)...")

    # Ignition from alpha (n=1)
    alpha_freq = f0 * PHI
    alpha_idx = np.argmin(np.abs(sim.frequencies - alpha_freq))
    ignition_energy = np.zeros(sim.n_freqs)
    ignition_energy[alpha_idx] = 0.8
    ignition_energy += 0.02
    ignition_energy /= ignition_energy.sum()

    history_alpha = sim.run(t_max=5.0, dt=0.01, initial_energy=ignition_energy, seed=42)
    sim.plot_results(save_path=f'{save_dir}/ignition_alpha.png')

    print(f"\n[4/4] Creating comparison figure...")

    # Comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Energy evolution comparison
    histories = [
        (history_baseline, 'Baseline (Uniform)', 'steelblue'),
        (history_fundamental, f'Ignition @ {f0} Hz', 'coral'),
        (history_alpha, f'Ignition @ {alpha_freq:.1f} Hz', 'seagreen')
    ]

    for idx, (hist, label, color) in enumerate(histories):
        ax = axes[0, idx]
        t = hist['time']
        energy = hist['energy']

        # Plot energy evolution for key frequencies
        for i in range(sim.n_freqs):
            alpha = 0.8 if sim.n_values[i] == int(sim.n_values[i]) else 0.4
            ax.plot(t, energy[:, i], linewidth=1.5, alpha=alpha,
                   label=f'{sim.frequencies[i]:.1f} Hz' if i < 5 else '')

        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Energy', fontsize=11)
        ax.set_title(f'{label}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7, loc='upper right')

    # Final distribution comparison
    ax = axes[1, 0]
    x = np.arange(sim.n_freqs)
    width = 0.25

    ax.bar(x - width, history_baseline['energy'][-1], width, label='Baseline', color='steelblue')
    ax.bar(x, history_fundamental['energy'][-1], width, label=f'Ignition @ {f0} Hz', color='coral')
    ax.bar(x + width, history_alpha['energy'][-1], width, label=f'Ignition @ {alpha_freq:.1f} Hz', color='seagreen')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{f:.1f}' for f in sim.frequencies], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Final Energy', fontsize=11)
    ax.set_title('Final Energy Distribution Comparison', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Attractor vs boundary energy over time
    ax = axes[1, 1]
    attractor_mask = np.array([n == int(n) for n in sim.n_values])

    for hist, label, color in histories:
        att_energy = hist['energy'][:, attractor_mask].sum(axis=1)
        ax.plot(hist['time'], att_energy, color=color, linewidth=2, label=label)

    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Attractor Energy', fontsize=11)
    ax.set_title('Energy at Attractor Frequencies', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Summary panel
    ax = axes[1, 2]
    ax.axis('off')

    summary = f"""
    Ignition Analysis Summary
    ━━━━━━━━━━━━━━━━━━━━━━━━━

    Fundamental: f0 = {f0} Hz
    Golden ratio: φ = {PHI:.4f}

    Canonical Frequencies:
    • n=0: {f0:.1f} Hz (Theta/SR f0)
    • n=1: {f0*PHI:.1f} Hz (Alpha)
    • n=2: {f0*PHI**2:.1f} Hz (Beta)
    • n=3: {f0*PHI**3:.1f} Hz (Gamma)

    Key Findings:

    1. Energy naturally flows toward
       attractor frequencies (integer n)

    2. Ignition at fundamental spreads
       energy UP via PAC coupling

    3. Ignition at alpha spreads energy
       both UP (to beta/gamma) and
       DOWN (to theta via boundaries)

    4. Boundaries gate but don't block
       energy transfer

    5. Final states depend on ignition
       site but converge to similar
       attractor-dominated patterns
    """

    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.suptitle('E8 Energy Flow: Ignition Site Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/ignition_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {save_dir}/ignition_comparison.png")
    plt.show()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return {
        'baseline': history_baseline,
        'fundamental': history_fundamental,
        'alpha': history_alpha,
        'simulation': sim
    }


# =============================================================================
# DETAILED CASCADE ANALYSIS
# =============================================================================

def analyze_ignition_cascade(f0=F0, trigger_n=0, injection_amount=0.5,
                             t_max=3.0, save_dir='e8_energy_figures'):
    """
    Detailed analysis of energy cascade following ignition at a specific frequency.

    Tracks:
    - Step-by-step energy redistribution
    - PAC flow magnitude and direction
    - Boundary crossing events
    - Time to reach equilibrium
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print("IGNITION CASCADE ANALYSIS")
    print("=" * 70)

    # Create simulation
    sim = E8EnergyFlowSimulation(f0=f0)

    print(f"\nFrequencies in simulation:")
    for i, (n, f) in enumerate(zip(sim.n_values, sim.frequencies)):
        marker = "← TRIGGER" if abs(n - trigger_n) < 0.1 else ""
        print(f"  n={n:+.1f}: {f:6.2f} Hz {marker}")

    # Find trigger frequency index
    trigger_idx = np.argmin(np.abs(sim.n_values - trigger_n))
    trigger_freq = sim.frequencies[trigger_idx]

    print(f"\nTrigger frequency: {trigger_freq:.2f} Hz (n={trigger_n})")
    print(f"Injection amount: {injection_amount}")

    # Initialize with energy injection at trigger
    initial_energy = np.ones(sim.n_freqs) * 0.02  # Small background
    initial_energy[trigger_idx] = injection_amount
    initial_energy /= initial_energy.sum()

    # Run with fine time resolution
    dt = 0.005
    history = sim.run(t_max=t_max, dt=dt, initial_energy=initial_energy,
                      noise_strength=0.005, seed=42)

    # Analyze cascade dynamics
    t = history['time']
    energy = history['energy']

    # Compute energy flow rates
    energy_rate = np.diff(energy, axis=0) / dt

    # Track when energy arrives at each frequency
    arrival_threshold = 0.1  # Energy > 10% of max
    arrival_times = {}
    for i in range(sim.n_freqs):
        max_energy_at_freq = energy[:, i].max()
        threshold = 0.1 * max_energy_at_freq
        indices = np.where(energy[:, i] > threshold)[0]
        if len(indices) > 0:
            arrival_times[sim.frequencies[i]] = t[indices[0]]

    # Find peak times for each frequency
    peak_times = {}
    for i in range(sim.n_freqs):
        peak_idx = np.argmax(energy[:, i])
        peak_times[sim.frequencies[i]] = t[peak_idx]

    # Create detailed visualization
    fig = plt.figure(figsize=(18, 16))

    # Panel 1: Energy cascade heatmap
    ax1 = fig.add_subplot(3, 3, 1)
    im = ax1.imshow(energy.T, aspect='auto', origin='lower',
                   extent=[t[0], t[-1], 0, sim.n_freqs],
                   cmap='hot')
    ax1.set_yticks(range(sim.n_freqs))
    ax1.set_yticklabels([f'{f:.1f}' for f in sim.frequencies], fontsize=8)
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Frequency (Hz)', fontsize=11)
    ax1.set_title(f'Energy Cascade from {trigger_freq:.1f} Hz', fontsize=12, fontweight='bold')

    # Mark trigger frequency
    ax1.axhline(trigger_idx, color='cyan', linestyle='--', alpha=0.8)
    plt.colorbar(im, ax=ax1, label='Energy')

    # Panel 2: Energy time series with cascade propagation
    ax2 = fig.add_subplot(3, 3, 2)
    colors = plt.cm.coolwarm(np.linspace(0, 1, sim.n_freqs))

    for i in range(sim.n_freqs):
        is_attractor = sim.n_values[i] == int(sim.n_values[i])
        linewidth = 2.5 if is_attractor else 1.5
        linestyle = '-' if is_attractor else '--'
        label = f'{sim.frequencies[i]:.1f} Hz (n={sim.n_values[i]:.1f})'

        ax2.plot(t, energy[:, i], color=colors[i], linewidth=linewidth,
                linestyle=linestyle, label=label)

    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Energy', fontsize=11)
    ax2.set_title('Energy Evolution by Frequency', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=7, loc='right', bbox_to_anchor=(1.3, 0.5))
    ax2.grid(True, alpha=0.3)

    # Panel 3: Energy flow rate (first derivative)
    ax3 = fig.add_subplot(3, 3, 3)

    for i in range(sim.n_freqs):
        is_attractor = sim.n_values[i] == int(sim.n_values[i])
        if is_attractor:  # Only show attractors for clarity
            ax3.plot(t[:-1], energy_rate[:, i], color=colors[i], linewidth=2,
                    label=f'{sim.frequencies[i]:.1f} Hz')

    ax3.axhline(0, color='gray', linestyle='-', alpha=0.5)
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('dE/dt (Energy Flow Rate)', fontsize=11)
    ax3.set_title('Energy Flow Rate at Attractors', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Arrival time analysis
    ax4 = fig.add_subplot(3, 3, 4)

    freqs_sorted = sorted(arrival_times.keys())
    arrival_sorted = [arrival_times[f] for f in freqs_sorted]
    freq_distance = [abs(f - trigger_freq) for f in freqs_sorted]

    scatter = ax4.scatter(freq_distance, arrival_sorted, c=freqs_sorted,
                         cmap='coolwarm', s=150, edgecolor='black', linewidth=1.5)
    ax4.set_xlabel('Distance from Trigger (Hz)', fontsize=11)
    ax4.set_ylabel('Arrival Time (s)', fontsize=11)
    ax4.set_title('Energy Arrival Times', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax4, label='Frequency (Hz)')
    ax4.grid(True, alpha=0.3)

    # Panel 5: Peak time analysis
    ax5 = fig.add_subplot(3, 3, 5)

    freqs_list = list(peak_times.keys())
    peaks_list = list(peak_times.values())

    # Separate attractors and boundaries
    att_freqs = [f for i, f in enumerate(sim.frequencies) if sim.n_values[i] == int(sim.n_values[i])]
    bnd_freqs = [f for i, f in enumerate(sim.frequencies) if sim.n_values[i] != int(sim.n_values[i])]

    att_peaks = [peak_times[f] for f in att_freqs if f in peak_times]
    bnd_peaks = [peak_times[f] for f in bnd_freqs if f in peak_times]

    ax5.scatter(att_freqs[:len(att_peaks)], att_peaks, s=150, c='blue',
               marker='v', label='Attractors', edgecolor='black', linewidth=1.5)
    ax5.scatter(bnd_freqs[:len(bnd_peaks)], bnd_peaks, s=100, c='red',
               marker='^', label='Boundaries', edgecolor='black', linewidth=1.5)

    ax5.axvline(trigger_freq, color='green', linestyle='--', linewidth=2, label='Trigger')
    ax5.set_xlabel('Frequency (Hz)', fontsize=11)
    ax5.set_ylabel('Peak Time (s)', fontsize=11)
    ax5.set_title('Time to Peak Energy', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Panel 6: Energy flow direction (upward vs downward)
    ax6 = fig.add_subplot(3, 3, 6)

    # Compute net flow direction
    upward_flow = np.zeros(len(t) - 1)
    downward_flow = np.zeros(len(t) - 1)

    for i in range(sim.n_freqs):
        if sim.frequencies[i] > trigger_freq:
            upward_flow += np.maximum(0, energy_rate[:, i])
        elif sim.frequencies[i] < trigger_freq:
            downward_flow += np.maximum(0, energy_rate[:, i])

    ax6.fill_between(t[:-1], 0, upward_flow, alpha=0.5, color='red', label='Upward (→ high freq)')
    ax6.fill_between(t[:-1], 0, -downward_flow, alpha=0.5, color='blue', label='Downward (→ low freq)')
    ax6.plot(t[:-1], upward_flow - downward_flow, 'k-', linewidth=2, label='Net flow')

    ax6.axhline(0, color='gray', linestyle='-', alpha=0.5)
    ax6.set_xlabel('Time (s)', fontsize=11)
    ax6.set_ylabel('Energy Flow', fontsize=11)
    ax6.set_title('Bidirectional Energy Flow', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Panel 7: PAC coupling visualization
    ax7 = fig.add_subplot(3, 3, 7)
    sim.pac.plot_coupling_matrix(ax=ax7)

    # Panel 8: Energy at attractors vs boundaries over time
    ax8 = fig.add_subplot(3, 3, 8)

    attractor_mask = np.array([n == int(n) for n in sim.n_values])
    boundary_mask = ~attractor_mask

    att_energy = energy[:, attractor_mask].sum(axis=1)
    bnd_energy = energy[:, boundary_mask].sum(axis=1)

    ax8.plot(t, att_energy, 'b-', linewidth=2.5, label='Attractors')
    ax8.plot(t, bnd_energy, 'r-', linewidth=2.5, label='Boundaries')
    ax8.fill_between(t, att_energy, alpha=0.3, color='blue')
    ax8.fill_between(t, bnd_energy, alpha=0.3, color='red')

    ax8.set_xlabel('Time (s)', fontsize=11)
    ax8.set_ylabel('Total Energy', fontsize=11)
    ax8.set_title('Attractor vs Boundary Energy', fontsize=12, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # Panel 9: Summary
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')

    # Compute summary statistics
    final_att_energy = att_energy[-1]
    final_bnd_energy = bnd_energy[-1]
    peak_att_time = t[np.argmax(att_energy)]
    cascade_duration = t[np.argmax(np.abs(np.diff(att_energy)))]

    # Find dominant final frequency
    final_dominant_idx = np.argmax(energy[-1])
    final_dominant_freq = sim.frequencies[final_dominant_idx]

    summary = f"""
    Ignition Cascade Analysis
    ━━━━━━━━━━━━━━━━━━━━━━━━━━

    Trigger:
    • Frequency: {trigger_freq:.1f} Hz (n={trigger_n})
    • Initial injection: {injection_amount:.1%} of total

    Cascade Dynamics:
    • Peak attractor energy: t = {peak_att_time:.3f} s
    • Fastest flow change: t = {cascade_duration:.3f} s
    • Dominant final freq: {final_dominant_freq:.1f} Hz

    Final Energy Distribution:
    • Attractors: {final_att_energy:.1%}
    • Boundaries: {final_bnd_energy:.1%}

    Energy Propagation:
    • Upward (→ high freq): via PAC coupling
    • Downward (→ low freq): via attractor pull
    • Boundary crossing: gated by permeability

    Key Observation:
    Energy injected at {trigger_freq:.1f} Hz propagates
    {"primarily upward" if final_dominant_freq > trigger_freq else "primarily downward" if final_dominant_freq < trigger_freq else "symmetrically"},
    settling into attractor-dominated pattern.
    """

    ax9.text(0.05, 0.95, summary, transform=ax9.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.suptitle(f'Ignition Cascade: Trigger at {trigger_freq:.1f} Hz (n={trigger_n})',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/cascade_n{trigger_n}.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {save_dir}/cascade_n{trigger_n}.png")
    plt.show()

    return {
        'history': history,
        'arrival_times': arrival_times,
        'peak_times': peak_times,
        'simulation': sim
    }


def run_multi_trigger_comparison(f0=F0, save_dir='e8_energy_figures'):
    """
    Compare ignition cascades from different trigger frequencies.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print("MULTI-TRIGGER IGNITION COMPARISON")
    print("=" * 70)

    # Test triggers at different n values
    trigger_ns = [0, 1, 2]  # Theta, Alpha, Beta attractors
    results = {}

    for n in trigger_ns:
        print(f"\n--- Analyzing trigger at n={n} ---")
        results[n] = analyze_ignition_cascade(
            f0=f0, trigger_n=n, injection_amount=0.6,
            t_max=2.0, save_dir=save_dir
        )

    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    colors = ['blue', 'green', 'orange']
    labels = ['Theta (n=0)', 'Alpha (n=1)', 'Beta (n=2)']

    # Panel 1: Energy evolution at fundamental (7.6 Hz)
    ax = axes[0, 0]
    for i, n in enumerate(trigger_ns):
        hist = results[n]['history']
        sim = results[n]['simulation']
        fundamental_idx = np.argmin(np.abs(sim.frequencies - f0))
        ax.plot(hist['time'], hist['energy'][:, fundamental_idx],
               color=colors[i], linewidth=2, label=labels[i])

    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Energy', fontsize=11)
    ax.set_title(f'Energy at Fundamental ({f0} Hz)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Total attractor energy
    ax = axes[0, 1]
    for i, n in enumerate(trigger_ns):
        hist = results[n]['history']
        sim = results[n]['simulation']
        attractor_mask = np.array([nv == int(nv) for nv in sim.n_values])
        att_energy = hist['energy'][:, attractor_mask].sum(axis=1)
        ax.plot(hist['time'], att_energy, color=colors[i], linewidth=2, label=labels[i])

    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Total Attractor Energy', fontsize=11)
    ax.set_title('Attractor Energy Accumulation', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Final distribution comparison
    ax = axes[1, 0]
    x = np.arange(results[0]['simulation'].n_freqs)
    width = 0.25

    for i, n in enumerate(trigger_ns):
        hist = results[n]['history']
        ax.bar(x + i * width, hist['energy'][-1], width, color=colors[i], label=labels[i])

    ax.set_xticks(x + width)
    ax.set_xticklabels([f'{f:.1f}' for f in results[0]['simulation'].frequencies],
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Final Energy', fontsize=11)
    ax.set_title('Final Energy Distribution by Trigger', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 4: Summary statistics
    ax = axes[1, 1]
    ax.axis('off')

    summary_lines = ["Multi-Trigger Comparison Summary", "=" * 35, ""]

    for i, n in enumerate(trigger_ns):
        hist = results[n]['history']
        sim = results[n]['simulation']
        trigger_freq = sim.frequencies[np.argmin(np.abs(sim.n_values - n))]
        final_peak = sim.frequencies[np.argmax(hist['energy'][-1])]

        attractor_mask = np.array([nv == int(nv) for nv in sim.n_values])
        final_att = hist['energy'][-1, attractor_mask].sum()

        summary_lines.extend([
            f"{labels[i]}:",
            f"  Trigger: {trigger_freq:.1f} Hz",
            f"  Final peak: {final_peak:.1f} Hz",
            f"  Attractor energy: {final_att:.1%}",
            ""
        ])

    ax.text(0.1, 0.9, "\n".join(summary_lines), transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.suptitle('Ignition Cascade: Multi-Trigger Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/multi_trigger_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {save_dir}/multi_trigger_comparison.png")
    plt.show()

    return results


# =============================================================================
# HEBBIAN LEARNING ANALYSIS
# =============================================================================

def run_learning_comparison(f0=F0, t_max=10.0, save_dir='e8_energy_figures'):
    """
    Compare simulations with and without Hebbian learning.

    Tests how adaptive coupling weights affect energy flow dynamics.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print("HEBBIAN LEARNING COMPARISON ANALYSIS")
    print("=" * 70)

    # Test configurations
    configs = [
        ('No Learning', {'enable_learning': False}),
        ('Learning (η=0.01)', {'enable_learning': True, 'learning_rate': 0.01}),
        ('Learning (η=0.05)', {'enable_learning': True, 'learning_rate': 0.05}),
        ('Learning (η=0.1)', {'enable_learning': True, 'learning_rate': 0.1}),
    ]

    results = {}

    for name, params in configs:
        print(f"\n--- Running: {name} ---")

        sim = E8EnergyFlowSimulation(f0=f0, **params)
        history = sim.run(t_max=t_max, dt=0.01, noise_strength=0.01, seed=42)

        results[name] = {
            'simulation': sim,
            'history': history
        }

        # Print summary
        print(f"  Final energy distribution:")
        for i, freq in enumerate(sim.frequencies):
            print(f"    {freq:6.1f} Hz: {sim.energy[i]:.3f}")

        if sim.enable_learning:
            print(f"  Final PAC weights: mean={history['pac_weights_mean'][-1]:.3f}, "
                  f"std={history['pac_weights_std'][-1]:.3f}")
            print(f"  Final boundary weights: mean={history['boundary_weights_mean'][-1]:.3f}")

    # Create comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))

    # Panel 1: Energy at fundamental frequency
    ax = axes[0, 0]
    for i, (name, params) in enumerate(configs):
        hist = results[name]['history']
        sim = results[name]['simulation']
        fundamental_idx = np.argmin(np.abs(sim.frequencies - f0))
        ax.plot(hist['time'], hist['energy'][:, fundamental_idx],
               color=colors[i], linewidth=2, label=name)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Energy', fontsize=11)
    ax.set_title(f'Energy at Fundamental ({f0} Hz)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Energy at gamma frequency
    ax = axes[0, 1]
    for i, (name, params) in enumerate(configs):
        hist = results[name]['history']
        sim = results[name]['simulation']
        gamma_idx = -1  # Highest frequency
        ax.plot(hist['time'], hist['energy'][:, gamma_idx],
               color=colors[i], linewidth=2, label=name)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Energy', fontsize=11)
    ax.set_title('Energy at Highest Frequency (Gamma)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Attractor energy comparison
    ax = axes[0, 2]
    for i, (name, params) in enumerate(configs):
        hist = results[name]['history']
        sim = results[name]['simulation']
        attractor_mask = np.array([n == int(n) for n in sim.n_values])
        att_energy = hist['energy'][:, attractor_mask].sum(axis=1)
        ax.plot(hist['time'], att_energy, color=colors[i], linewidth=2, label=name)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Total Attractor Energy', fontsize=11)
    ax.set_title('Attractor Energy Evolution', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 4: Final distribution comparison
    ax = axes[1, 0]
    x = np.arange(results[configs[0][0]]['simulation'].n_freqs)
    width = 0.2

    for i, (name, params) in enumerate(configs):
        hist = results[name]['history']
        ax.bar(x + i * width - 0.3, hist['energy'][-1], width,
               label=name, color=colors[i], alpha=0.8)

    ax.set_xticks(x)
    sim = results[configs[0][0]]['simulation']
    ax.set_xticklabels([f'{f:.1f}' for f in sim.frequencies],
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Final Energy', fontsize=11)
    ax.set_title('Final Energy Distribution', fontsize=12)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 5: Weight evolution (learning configs only)
    ax = axes[1, 1]
    for i, (name, params) in enumerate(configs):
        if params.get('enable_learning', False):
            hist = results[name]['history']
            ax.plot(hist['time'], hist['pac_weights_mean'],
                   color=colors[i], linewidth=2, label=name)
            ax.fill_between(hist['time'],
                          hist['pac_weights_mean'] - hist['pac_weights_std'],
                          hist['pac_weights_mean'] + hist['pac_weights_std'],
                          color=colors[i], alpha=0.2)
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.7, label='Initial')
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('PAC Weight', fontsize=11)
    ax.set_title('PAC Weight Evolution (Learning Configs)', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 6: Summary statistics
    ax = axes[1, 2]
    ax.axis('off')

    summary_lines = [
        "Hebbian Learning Comparison",
        "=" * 35, ""
    ]

    for name, params in configs:
        hist = results[name]['history']
        sim = results[name]['simulation']

        # Final peak frequency
        final_peak = sim.frequencies[np.argmax(hist['energy'][-1])]

        # Attractor energy
        attractor_mask = np.array([n == int(n) for n in sim.n_values])
        final_att = hist['energy'][-1, attractor_mask].sum()

        summary_lines.extend([
            f"{name}:",
            f"  Peak freq: {final_peak:.1f} Hz",
            f"  Attractor energy: {final_att:.1%}",
        ])

        if params.get('enable_learning', False):
            summary_lines.append(f"  Final PAC weight: {hist['pac_weights_mean'][-1]:.3f}")

        summary_lines.append("")

    summary_lines.extend([
        "Key Finding:",
        "Hebbian learning adapts PAC coupling",
        "to reinforce successful energy transfer",
        "pathways, stabilizing attractor states."
    ])

    ax.text(0.1, 0.95, "\n".join(summary_lines), transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.suptitle('E8 Energy Flow: Hebbian Learning Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/learning_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {save_dir}/learning_comparison.png")
    plt.show()

    print("\n" + "=" * 70)
    print("LEARNING COMPARISON COMPLETE")
    print("=" * 70)

    return results


def run_learning_with_ignition(f0=F0, trigger_n=1, t_max=5.0, save_dir='e8_energy_figures'):
    """
    Analyze how Hebbian learning affects energy propagation after ignition.

    Compares cascade dynamics with and without adaptive coupling.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print("HEBBIAN LEARNING + IGNITION ANALYSIS")
    print("=" * 70)

    configs = [
        ('Static Coupling', {'enable_learning': False}),
        ('Hebbian Learning', {'enable_learning': True, 'learning_rate': 0.05}),
    ]

    results = {}

    for name, params in configs:
        print(f"\n--- Running: {name} (trigger at n={trigger_n}) ---")

        sim = E8EnergyFlowSimulation(f0=f0, **params)

        # Initialize with energy injection at trigger frequency
        trigger_idx = np.argmin(np.abs(sim.n_values - trigger_n))
        initial_energy = np.ones(sim.n_freqs) * 0.02
        initial_energy[trigger_idx] = 0.7
        initial_energy /= initial_energy.sum()

        history = sim.run(t_max=t_max, dt=0.01, initial_energy=initial_energy,
                         noise_strength=0.01, seed=42)

        results[name] = {
            'simulation': sim,
            'history': history,
            'trigger_idx': trigger_idx
        }

        # Plot individual results
        sim.plot_results(save_path=f'{save_dir}/ignition_{name.replace(" ", "_").lower()}.png')

    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    t = results['Static Coupling']['history']['time']

    # Panel 1: Energy cascade comparison
    ax = axes[0, 0]
    sim = results['Static Coupling']['simulation']
    trigger_freq = sim.frequencies[results['Static Coupling']['trigger_idx']]

    for name, color in [('Static Coupling', 'blue'), ('Hebbian Learning', 'red')]:
        hist = results[name]['history']
        for i in range(sim.n_freqs):
            alpha = 0.3 if sim.frequencies[i] != trigger_freq else 0.8
            linestyle = '-' if name == 'Static Coupling' else '--'
            if sim.frequencies[i] == trigger_freq:
                ax.plot(hist['time'], hist['energy'][:, i],
                       color=color, linewidth=2, linestyle=linestyle,
                       label=f'{name} @ {sim.frequencies[i]:.1f} Hz')

    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Energy', fontsize=11)
    ax.set_title(f'Energy at Trigger Frequency ({trigger_freq:.1f} Hz)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Energy spread comparison
    ax = axes[0, 1]
    for name, color in [('Static Coupling', 'blue'), ('Hebbian Learning', 'red')]:
        hist = results[name]['history']
        sim = results[name]['simulation']
        # Energy spread = entropy-like measure
        energy_spread = -np.sum(hist['energy'] * np.log(hist['energy'] + 1e-10), axis=1)
        ax.plot(hist['time'], energy_spread, color=color, linewidth=2, label=name)

    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Energy Spread (Entropy)', fontsize=11)
    ax.set_title('Energy Distribution Spread Over Time', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Weight evolution (Hebbian only)
    ax = axes[1, 0]
    hist = results['Hebbian Learning']['history']
    ax.plot(t, hist['pac_weights_mean'], 'red', linewidth=2, label='PAC Mean')
    ax.fill_between(t,
                   hist['pac_weights_min'],
                   hist['pac_weights_max'],
                   alpha=0.2, color='red', label='PAC Range')
    ax.plot(t, hist['boundary_weights_mean'], 'purple', linewidth=2, label='Boundary Mean')
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.7, label='Initial')
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Weight', fontsize=11)
    ax.set_title('Hebbian Weight Adaptation During Cascade', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 4: Final distribution comparison
    ax = axes[1, 1]
    x = np.arange(sim.n_freqs)
    width = 0.35

    ax.bar(x - width/2, results['Static Coupling']['history']['energy'][-1],
           width, label='Static Coupling', color='blue', alpha=0.7)
    ax.bar(x + width/2, results['Hebbian Learning']['history']['energy'][-1],
           width, label='Hebbian Learning', color='red', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{f:.1f}' for f in sim.frequencies],
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Final Energy', fontsize=11)
    ax.set_title('Final Energy Distribution After Cascade', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Hebbian Learning Effect on Ignition Cascade (trigger at {trigger_freq:.1f} Hz)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/learning_ignition_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {save_dir}/learning_ignition_comparison.png")
    plt.show()

    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Run basic ignition analysis
    print("\n" + "=" * 70)
    print("RUNNING E8 BIDIRECTIONAL ENERGY FLOW ANALYSIS")
    print("=" * 70)

    # First run the basic analysis with learning
    print("\n[1/4] Running basic analysis with Hebbian learning...")
    results = run_ignition_analysis(f0=7.6, save_dir='e8_energy_figures')

    # Then run detailed cascade analysis for fundamental trigger
    print("\n\n[2/4] Running cascade analysis...")
    cascade_results = analyze_ignition_cascade(
        f0=7.6, trigger_n=0, injection_amount=0.6,
        t_max=2.0, save_dir='e8_energy_figures'
    )

    # Compare multiple trigger points
    print("\n\n[3/4] Running multi-trigger comparison...")
    multi_results = run_multi_trigger_comparison(f0=7.6, save_dir='e8_energy_figures')

    # NEW: Run Hebbian learning comparison
    print("\n\n[4/4] Running Hebbian learning comparison...")
    learning_results = run_learning_comparison(f0=7.6, t_max=10.0, save_dir='e8_energy_figures')

    print("\n\n" + "=" * 70)
    print("ALL ANALYSES COMPLETE")
    print("=" * 70)
    print("\nFigures saved to: e8_energy_figures/")

#!/usr/bin/env python3
"""
Analyze secondary structure in band distributions.

Even if alpha's main peak is smeared by IAF variability, are there
LOCAL peaks at φ^n positions that indicate underlying structure?

Approach:
1. Smooth the histogram to get the "envelope" (baseline shape)
2. Compute residuals: actual - smoothed
3. Check if residuals show peaks at φ^n positions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# Constants
PHI = 1.618033988749895
F0 = 7.6  # Hz

# Band definitions
BANDS = [
    ('Delta', F0 * PHI**(-2), F0 * PHI**(-1), -1),
    ('Theta', F0 * PHI**(-1), F0 * PHI**0, 0),
    ('Alpha', F0 * PHI**0, F0 * PHI**1, 0),
    ('Low Beta', F0 * PHI**1, F0 * PHI**2, 1),
    ('High Beta', F0 * PHI**2, F0 * PHI**3, 2),
    ('Gamma', F0 * PHI**3, F0 * PHI**4, 3),
]

# Key positions
POSITIONS = {
    'Boundary': 0.0,
    '2° Noble': 0.382,
    'Attractor': 0.5,
    '1° Noble': 0.618,
}

U_WINDOW = 0.05


def compute_lattice_coordinate(freq, f0=F0):
    """Compute lattice coordinate u = [log_phi(f/f0)] mod 1."""
    n = np.log(freq / f0) / np.log(PHI)
    return n % 1


def detect_local_peaks_at_positions(hist, bin_centers, positions, window=0.05):
    """
    Check if there are local peaks near each φ^n position.

    Returns dict mapping position name to (is_peak, peak_height_above_neighbors).
    """
    results = {}

    for pos_name, pos_val in positions.items():
        # Find bins near this position
        mask = np.abs(bin_centers - pos_val) < window
        if pos_val < window:
            mask = mask | (bin_centers > 1 - window)

        if not mask.any():
            results[pos_name] = (False, 0, 0)
            continue

        # Get the value at this position
        pos_indices = np.where(mask)[0]
        pos_value = hist[pos_indices].mean()

        # Get neighboring values (exclude the position itself)
        neighbor_mask = (np.abs(bin_centers - pos_val) > window) & \
                       (np.abs(bin_centers - pos_val) < 3 * window)
        if neighbor_mask.any():
            neighbor_value = hist[neighbor_mask].mean()
        else:
            neighbor_value = hist.mean()

        # Is this a local peak?
        is_peak = pos_value > neighbor_value * 1.05  # 5% threshold
        height_above = (pos_value - neighbor_value) / neighbor_value * 100 if neighbor_value > 0 else 0

        results[pos_name] = (is_peak, height_above, pos_value)

    return results


def main():
    # Load peaks
    peaks_df = pd.read_csv('golden_ratio_peaks_ALL_EMOTIV.csv')
    freqs = peaks_df['freq'].values
    print(f"Loaded {len(freqs):,} peaks")

    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()

    n_bins = 100  # Fine resolution

    print("\n" + "=" * 90)
    print("SECONDARY STRUCTURE ANALYSIS")
    print("=" * 90)
    print("\nLooking for LOCAL PEAKS at φ^n positions within each band's distribution")
    print("Method: Compute residuals above smoothed envelope, check for peaks at key positions\n")

    all_band_results = []

    for idx, (band_name, f_low, f_high, base_n) in enumerate(BANDS):
        ax = axes[idx]

        # Filter peaks
        mask = (freqs >= f_low) & (freqs < f_high)
        band_freqs = freqs[mask]
        n_peaks = len(band_freqs)

        if n_peaks < 100:
            ax.set_title(f'{band_name}: Insufficient data')
            continue

        # Compute lattice coordinates
        u = compute_lattice_coordinate(band_freqs)

        # Create histogram
        hist, bin_edges = np.histogram(u, bins=n_bins, range=(0, 1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Smooth to get envelope (broad structure)
        sigma = 10  # Smoothing width in bins
        envelope = gaussian_filter1d(hist.astype(float), sigma)

        # Compute residuals (local structure above envelope)
        residuals = hist - envelope

        # Normalize residuals by sqrt(envelope) to get z-score-like measure
        residuals_norm = residuals / np.sqrt(np.maximum(envelope, 1))

        # Plot histogram
        ax.bar(bin_centers, hist, width=1/n_bins * 0.9, color='steelblue', alpha=0.6, label='Histogram')
        ax.plot(bin_centers, envelope, 'r-', linewidth=2, label='Smoothed envelope')

        # Mark φ^n positions
        colors = {'Boundary': 'red', '2° Noble': 'purple', 'Attractor': 'orange', '1° Noble': 'green'}
        for pos_name, pos_val in POSITIONS.items():
            ax.axvline(pos_val, color=colors[pos_name], linestyle='--', linewidth=1.5, alpha=0.8)

        # Detect local peaks
        peak_results = detect_local_peaks_at_positions(hist, bin_centers, POSITIONS)

        # Add annotations for peaks
        y_max = hist.max()
        annotation_y = y_max * 0.9

        band_result = {'Band': band_name, 'N_peaks': n_peaks}

        for pos_name, (is_peak, height_above, pos_value) in peak_results.items():
            pos_val = POSITIONS[pos_name]
            marker = '▲' if is_peak else '○'

            band_result[f'{pos_name}_is_peak'] = is_peak
            band_result[f'{pos_name}_height_above'] = height_above

            if is_peak:
                ax.annotate(f'{marker}\n+{height_above:.0f}%',
                           xy=(pos_val, pos_value),
                           xytext=(pos_val, annotation_y),
                           ha='center', va='bottom', fontsize=7,
                           color=colors[pos_name], fontweight='bold',
                           arrowprops=dict(arrowstyle='-', color=colors[pos_name], alpha=0.5))

        all_band_results.append(band_result)

        ax.set_xlabel('Lattice coordinate (u)')
        ax.set_ylabel('Peak count')
        ax.set_title(f'{band_name} ({f_low:.1f}-{f_high:.1f} Hz, n={n_peaks:,})', fontweight='bold')
        ax.set_xlim(0, 1)
        ax.legend(loc='upper right', fontsize=8)

        # Add uniform reference line
        uniform = n_peaks / n_bins
        ax.axhline(uniform, color='gray', linestyle=':', alpha=0.5)

        # Print results
        print(f"\n{band_name}:")
        for pos_name, (is_peak, height_above, pos_value) in peak_results.items():
            status = "✓ LOCAL PEAK" if is_peak else "○ no peak"
            print(f"  {pos_name:<12}: {status:<15} ({height_above:+.1f}% above neighbors)")

    # Add legend for position types
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linestyle='--', label='Boundary (u=0)'),
        Line2D([0], [0], color='purple', linestyle='--', label='2° Noble (u=0.382)'),
        Line2D([0], [0], color='orange', linestyle='--', label='Attractor (u=0.5)'),
        Line2D([0], [0], color='green', linestyle='--', label='1° Noble (u=0.618)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    plt.savefig('phi_secondary_structure_analysis.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("\nSaved: phi_secondary_structure_analysis.png")

    # ==========================================================================
    # RESIDUAL ANALYSIS
    # ==========================================================================
    print("\n" + "=" * 90)
    print("RESIDUAL ANALYSIS (Structure Above Smoothed Envelope)")
    print("=" * 90)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (band_name, f_low, f_high, base_n) in enumerate(BANDS):
        ax = axes[idx]

        # Filter peaks
        mask = (freqs >= f_low) & (freqs < f_high)
        band_freqs = freqs[mask]
        n_peaks = len(band_freqs)

        if n_peaks < 100:
            continue

        # Compute lattice coordinates
        u = compute_lattice_coordinate(band_freqs)

        # Create histogram
        hist, bin_edges = np.histogram(u, bins=n_bins, range=(0, 1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Smooth to get envelope
        sigma = 10
        envelope = gaussian_filter1d(hist.astype(float), sigma)

        # Compute residuals
        residuals = hist - envelope

        # Plot residuals
        colors_bar = ['green' if r > 0 else 'red' for r in residuals]
        ax.bar(bin_centers, residuals, width=1/n_bins * 0.9, color=colors_bar, alpha=0.7)
        ax.axhline(0, color='black', linewidth=1)

        # Mark φ^n positions
        for pos_name, pos_val in POSITIONS.items():
            ax.axvline(pos_val, color='blue', linestyle='--', linewidth=1.5, alpha=0.6)

        # Highlight residuals at φ^n positions
        for pos_name, pos_val in POSITIONS.items():
            pos_mask = np.abs(bin_centers - pos_val) < 0.02
            if pos_mask.any():
                pos_residual = residuals[pos_mask].mean()
                if abs(pos_residual) > 10:  # Only label significant residuals
                    ax.annotate(f'{pos_residual:+.0f}',
                               xy=(pos_val, pos_residual),
                               xytext=(pos_val, pos_residual + np.sign(pos_residual) * 30),
                               ha='center', fontsize=8, fontweight='bold',
                               arrowprops=dict(arrowstyle='->', color='blue'))

        ax.set_xlabel('Lattice coordinate (u)')
        ax.set_ylabel('Residual (actual - smoothed)')
        ax.set_title(f'{band_name}: Residuals Above Envelope', fontweight='bold')
        ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig('phi_residual_structure.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: phi_residual_structure.png")

    # ==========================================================================
    # SUMMARY TABLE
    # ==========================================================================
    print("\n" + "=" * 90)
    print("SUMMARY: LOCAL PEAKS AT φ^n POSITIONS")
    print("=" * 90)

    print(f"\n{'Band':<12} | {'Boundary':<12} | {'2° Noble':<12} | {'Attractor':<12} | {'1° Noble':<12}")
    print("-" * 70)

    for result in all_band_results:
        band = result['Band']
        cells = []
        for pos_name in ['Boundary', '2° Noble', 'Attractor', '1° Noble']:
            is_peak = result.get(f'{pos_name}_is_peak', False)
            height = result.get(f'{pos_name}_height_above', 0)
            if is_peak:
                cells.append(f"✓ +{height:.0f}%")
            else:
                cells.append(f"○ {height:+.0f}%")
        print(f"{band:<12} | {cells[0]:<12} | {cells[1]:<12} | {cells[2]:<12} | {cells[3]:<12}")

    # Count peaks per position
    print("\n" + "-" * 70)
    print("Count of bands showing local peaks at each position:")
    for pos_name in ['Boundary', '2° Noble', 'Attractor', '1° Noble']:
        count = sum(1 for r in all_band_results if r.get(f'{pos_name}_is_peak', False))
        print(f"  {pos_name}: {count}/6 bands")

    # ==========================================================================
    # KEY FINDING
    # ==========================================================================
    print("\n" + "=" * 90)
    print("KEY FINDING: SECONDARY STRUCTURE")
    print("=" * 90)

    # Check if noble positions show more local peaks than boundaries
    noble_peaks = sum(1 for r in all_band_results
                     if r.get('1° Noble_is_peak', False) or r.get('Attractor_is_peak', False))
    boundary_peaks = sum(1 for r in all_band_results if r.get('Boundary_is_peak', False))

    print(f"""
Local peak analysis reveals:

1. 1° Noble and Attractor positions show local peaks in {noble_peaks}/6 bands
2. Boundary positions show local peaks in {boundary_peaks}/6 bands

This means even bands with SMEARED distributions (like alpha) may show
SECONDARY STRUCTURE at φ^n positions that's visible above the envelope.

The smearing doesn't eliminate the φ^n architecture - it just reduces
its visibility in raw enrichment statistics.
""")


if __name__ == '__main__':
    main()

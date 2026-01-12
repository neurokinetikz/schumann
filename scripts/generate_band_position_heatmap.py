#!/usr/bin/env python3
"""
Generate Band × Position Z-score heatmap.
Shows enrichment/depletion z-scores for each EEG band at each lattice position.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
PHI = 1.618033988749895
F0 = 7.6  # Hz

def compute_lattice_coordinate(freq, f0=F0):
    """Compute lattice coordinate u = [log_phi(f/f0)] mod 1."""
    n = np.log(freq / f0) / np.log(PHI)
    return n % 1

def main():
    # Load peaks
    peaks_df = pd.read_csv('golden_ratio_peaks_ALL.csv')
    freqs = peaks_df['freq'].values
    print(f"Loaded {len(freqs):,} peaks")

    # Define phi^n band boundaries
    bands = [
        ('Delta', F0 * PHI**(-2), F0 * PHI**(-1)),      # ~2.9-4.7 Hz
        ('Theta', F0 * PHI**(-1), F0 * PHI**0),          # ~4.7-7.6 Hz
        ('Alpha', F0 * PHI**0, F0 * PHI**1),             # ~7.6-12.3 Hz
        ('Low Beta', F0 * PHI**1, F0 * PHI**2),          # ~12.3-19.9 Hz
        ('High Beta', F0 * PHI**2, F0 * PHI**3),         # ~19.9-32.2 Hz
        ('Gamma', F0 * PHI**3, F0 * PHI**4),             # ~32.2-52.1 Hz
    ]

    # Positions to analyze (excluding boundary which wraps around)
    positions = [0.25, 0.382, 0.5, 0.618, 0.75]
    position_labels = ['0.25\n(Quarter)', '0.382\n(2° Noble)', '0.5\n(Attractor)',
                       '0.618\n(1° Noble)', '0.75\n(Quarter)']

    # Compute z-scores for each band × position
    z_matrix = np.zeros((len(bands), len(positions)))
    n_bins = 25
    u_window = 1.0 / n_bins  # Width of one bin

    for i, (band_name, f_low, f_high) in enumerate(bands):
        # Filter peaks in this band
        mask = (freqs >= f_low) & (freqs < f_high)
        band_freqs = freqs[mask]
        n_peaks = len(band_freqs)

        if n_peaks == 0:
            continue

        # Compute lattice coordinates
        u = compute_lattice_coordinate(band_freqs)

        # Expected count per bin under uniform distribution
        expected_per_bin = n_peaks / n_bins

        for j, pos in enumerate(positions):
            # Find which bin this position falls into
            bin_idx = int(pos * n_bins)
            if bin_idx >= n_bins:
                bin_idx = n_bins - 1

            # Count peaks in this bin
            bin_low = bin_idx / n_bins
            bin_high = (bin_idx + 1) / n_bins
            observed = np.sum((u >= bin_low) & (u < bin_high))

            # Compute z-score
            z = (observed - expected_per_bin) / np.sqrt(expected_per_bin)
            z_matrix[i, j] = z

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 7))

    # Create DataFrame for seaborn
    band_names = [b[0] for b in bands]
    z_df = pd.DataFrame(z_matrix, index=band_names, columns=positions)

    # Custom colormap: red (negative) to white (0) to green (positive)
    cmap = sns.diverging_palette(10, 130, s=80, l=55, as_cmap=True)

    # Determine symmetric color limits
    vmax = max(abs(z_matrix.min()), abs(z_matrix.max()))
    vmin = -vmax

    # Plot heatmap
    sns.heatmap(z_df, annot=True, fmt='.1f', cmap=cmap, center=0,
                vmin=vmin, vmax=vmax, linewidths=0.5, linecolor='gray',
                cbar_kws={'label': 'Z-score', 'shrink': 0.8},
                annot_kws={'fontsize': 12, 'fontweight': 'bold'}, ax=ax)

    # Customize
    ax.set_title('Z-scores by Band and Position\n(Green = enriched, Red = depleted)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Fractional Position u', fontsize=12)
    ax.set_ylabel('EEG Band', fontsize=12)

    # Better x-tick labels
    ax.set_xticklabels(position_labels, fontsize=10, ha='center')
    ax.set_yticklabels(band_names, fontsize=11, rotation=0)

    plt.tight_layout()
    plt.savefig('phi_band_position_heatmap.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: phi_band_position_heatmap.png")

    # Print summary table
    print("\n" + "="*70)
    print("Z-SCORE MATRIX (Band × Position)")
    print("="*70)
    print(z_df.round(1).to_string())
    print("\nNote: |z| > 1.96 is significant at p < 0.05")
    print("      |z| > 2.58 is significant at p < 0.01")
    print("      |z| > 3.29 is significant at p < 0.001")

if __name__ == '__main__':
    main()

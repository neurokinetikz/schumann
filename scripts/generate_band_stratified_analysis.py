#!/usr/bin/env python3
"""
Generate band-stratified enrichment analysis figure.
6-panel layout showing lattice coordinate distribution within each phi^n band.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

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
        ('Delta', F0 * PHI**(-2), F0 * PHI**(-1)),      # phi^-2 to phi^-1: ~2.9-4.7 Hz
        ('Theta', F0 * PHI**(-1), F0 * PHI**0),          # phi^-1 to phi^0: ~4.7-7.6 Hz
        ('Alpha', F0 * PHI**0, F0 * PHI**1),             # phi^0 to phi^1: ~7.6-12.3 Hz
        ('Low Beta', F0 * PHI**1, F0 * PHI**2),          # phi^1 to phi^2: ~12.3-19.9 Hz
        ('High Beta', F0 * PHI**2, F0 * PHI**3),         # phi^2 to phi^3: ~19.9-32.2 Hz
        ('Gamma', F0 * PHI**3, F0 * PHI**4),             # phi^3 to phi^4: ~32.2-52.1 Hz
    ]

    # Create figure - 2 rows x 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    # Key positions for vertical lines
    positions = {
        0.382: ('purple', '2° Noble'),
        0.5: ('orange', 'Attractor'),
        0.618: ('green', '1° Noble'),
    }

    # Quarter integer positions (light gray reference)
    quarter_positions = [0.25, 0.75]

    for idx, (band_name, f_low, f_high) in enumerate(bands):
        ax = axes[idx]

        # Filter peaks in this band
        mask = (freqs >= f_low) & (freqs < f_high)
        band_freqs = freqs[mask]
        n_peaks = len(band_freqs)

        # Compute lattice coordinates
        u = compute_lattice_coordinate(band_freqs)

        # Create histogram
        n_bins = 25
        counts, bin_edges = np.histogram(u, bins=n_bins, range=(0, 1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = 1.0 / n_bins

        # Compute enrichment (ratio to uniform)
        expected_per_bin = n_peaks / n_bins
        enrichment = counts / expected_per_bin

        # Plot bars
        ax.bar(bin_centers, enrichment, width=bin_width * 0.9, color='steelblue',
               alpha=0.8, edgecolor='white', linewidth=0.3)

        # Uniform expectation line
        ax.axhline(1.0, color='red', linestyle='--', linewidth=2)

        # Quarter integer reference lines (light gray)
        for qpos in quarter_positions:
            ax.axvline(qpos, color='lightgray', linestyle='--', linewidth=1.5, alpha=0.7)

        # Position markers with z-scores
        z_scores = {}
        for pos, (color, label) in positions.items():
            ax.axvline(pos, color=color, linestyle='-', linewidth=2, alpha=0.9)

            # Compute z-score at this position
            bin_idx = int(pos * n_bins)
            if bin_idx >= n_bins:
                bin_idx = n_bins - 1
            observed = counts[bin_idx]
            z = (observed - expected_per_bin) / np.sqrt(expected_per_bin)
            z_scores[pos] = z

            # Significance stars
            if abs(z) > 3.29:
                sig = '***'
            elif abs(z) > 2.58:
                sig = '**'
            elif abs(z) > 1.96:
                sig = '*'
            else:
                sig = ''

            z_scores[pos] = (z, sig, color)

        # Set y limits - auto-scale based on data with padding for labels
        y_max = enrichment.max() * 1.15  # 15% padding for labels
        ax.set_ylim(0, y_max)

        # White background box for labels
        bbox_style = dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor='none', alpha=0.85)

        # Add z-score labels with staggered heights to avoid overlap
        label_heights = {0.382: 0.95, 0.5: 0.86, 0.618: 0.95}
        for pos, (z, sig, color) in z_scores.items():
            y_frac = label_heights[pos]
            ax.text(pos, y_max * y_frac, f'z={z:.1f}{sig}',
                    ha='center', va='bottom', fontsize=8, color=color,
                    fontweight='bold', bbox=bbox_style, zorder=10)

        # Title with band info
        ax.set_title(f'{band_name} ({f_low:.1f}-{f_high:.1f} Hz)\nn={n_peaks:,}',
                     fontsize=11, fontweight='bold')

        ax.set_xlabel('Fractional position u', fontsize=10)
        ax.set_ylabel('Enrichment (vs uniform)', fontsize=10)
        ax.set_xlim(0, 1)

    # Create legend elements
    legend_elements = [
        Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Uniform (1.0)'),
        Line2D([0], [0], color='lightgray', linestyle='--', linewidth=1.5, label='Quarter (0.25, 0.75)'),
        Line2D([0], [0], color='purple', linestyle='-', linewidth=2, label='2° Noble (0.382)'),
        Line2D([0], [0], color='orange', linestyle='-', linewidth=2, label='Attractor (0.5)'),
        Line2D([0], [0], color='green', linestyle='-', linewidth=2, label='1° Noble (0.618)'),
    ]

    # Add legend to figure (outside panels, at bottom)
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, -0.02), frameon=True, fancybox=True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # Make room for legend
    plt.savefig('phi_band_stratified_analysis.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: phi_band_stratified_analysis.png")

    # Print summary statistics
    print("\n" + "="*70)
    print("BAND-STRATIFIED SUMMARY")
    print("="*70)
    for band_name, f_low, f_high in bands:
        mask = (freqs >= f_low) & (freqs < f_high)
        band_freqs = freqs[mask]
        u = compute_lattice_coordinate(band_freqs)
        n_peaks = len(band_freqs)

        # Enrichment at key positions (window = 0.05)
        u_window = 0.05
        attractor_mask = np.abs(u - 0.5) < u_window
        noble1_mask = np.abs(u - 0.618) < u_window
        noble2_mask = np.abs(u - 0.382) < u_window

        attractor_e = (attractor_mask.sum() / (2 * u_window * n_peaks) - 1) * 100
        noble1_e = (noble1_mask.sum() / (2 * u_window * n_peaks) - 1) * 100
        noble2_e = (noble2_mask.sum() / (2 * u_window * n_peaks) - 1) * 100

        print(f"\n{band_name} ({f_low:.1f}-{f_high:.1f} Hz): n={n_peaks:,}")
        print(f"  2° Noble (0.382): {noble2_e:+.1f}%")
        print(f"  Attractor (0.5):  {attractor_e:+.1f}%")
        print(f"  1° Noble (0.618): {noble1_e:+.1f}%")

if __name__ == '__main__':
    main()

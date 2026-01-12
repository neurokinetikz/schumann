#!/usr/bin/env python3
"""
Regenerate lattice coordinate distribution visualization.
Two-panel figure matching original style.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import gaussian_filter1d

# Constants
PHI = 1.618033988749895
F0 = 7.6  # Hz

def compute_lattice_coordinate(freq, f0=F0):
    """Compute lattice coordinate u = [log_phi(f/f0)] mod 1."""
    n = np.log(freq / f0) / np.log(PHI)
    return n % 1

def plot_lattice_distribution(freqs, output_path='golden_ratio_lattice_ALL_EMOTIV.png',
                              title_suffix='ALL_EMOTIV', n_sessions=951):
    """Generate two-panel lattice distribution plot matching original style."""
    # Compute lattice coordinates
    u = compute_lattice_coordinate(freqs)

    # Create two-panel figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1, 1])

    # === TOP PANEL: Histogram ===
    n_bins = 100
    counts, bin_edges, _ = ax1.hist(u, bins=n_bins, range=(0, 1),
                                     color='steelblue', alpha=0.8,
                                     edgecolor='white', linewidth=0.3)

    # Text box style with white background
    bbox_style = dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8,
                      edgecolor='none')

    # Quarter integer references (light gray)
    for qpos in [0.25, 0.75]:
        ax1.axvline(qpos, color='lightgray', linestyle='--', linewidth=1.0, alpha=0.6)
        ax1.text(qpos, counts.max() * 1.02, f'φ^{qpos}', ha='center', va='bottom',
                 fontsize=7, color='gray', bbox=bbox_style, zorder=20)

    # Key positions with labels
    positions = [
        ('Boundary\nφ^0.0', 0.0, 'darkorange'),
        ('2° Noble\nφ^0.382', 0.382, 'olive'),
        ('Attractor\nφ^0.5', 0.5, 'crimson'),
        ('1° Noble\nφ^0.618', 0.618, 'teal'),
        ('', 1.0, 'darkorange'),
    ]

    for name, pos, color in positions:
        ax1.axvline(pos, color=color, linestyle='--', linewidth=1.5, alpha=0.8)
        if name:
            ax1.text(pos, counts.max() * 1.02, name, ha='center', va='bottom',
                     fontsize=8, color=color, bbox=bbox_style, zorder=20)

    ax1.set_xlabel('Fractional n (mod 1)', fontsize=11)
    ax1.set_ylabel('Peak Count', fontsize=11)
    ax1.set_title(f'{title_suffix} ({len(freqs):,} peaks, {n_sessions} sessions)\n'
                  f'Lattice Coordinate Distribution (n = log(f/7.6)/log(φ), mod 1)',
                  fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, counts.max() * 1.12)

    # === BOTTOM PANEL: KDE ratio to uniform ===
    # Compute KDE
    kde_bins = 200
    hist_fine, edges_fine = np.histogram(u, bins=kde_bins, range=(0, 1))
    bin_centers = (edges_fine[:-1] + edges_fine[1:]) / 2
    expected_per_bin = len(u) / kde_bins

    # Smooth with gaussian filter for KDE-like appearance
    smoothed = gaussian_filter1d(hist_fine.astype(float), sigma=3)
    ratio_to_uniform = smoothed / expected_per_bin

    # Plot KDE ratio
    ax2.fill_between(bin_centers, ratio_to_uniform, alpha=0.3, color='steelblue')
    ax2.plot(bin_centers, ratio_to_uniform, color='steelblue', linewidth=1.5)

    # Uniform expectation line
    ax2.axhline(1.0, color='red', linestyle='--', linewidth=1.5,
                label='Uniform expectation')

    # Quarter integer references (light gray) in KDE panel
    for qpos in [0.25, 0.75]:
        ax2.axvline(qpos, color='lightgray', linestyle='--', linewidth=1.0, alpha=0.6)

    # Position markers and annotations
    pos_values = {
        0.0: ('Boundary', 'darkorange'),
        0.382: ('2° Noble', 'olive'),
        0.5: ('Attractor', 'crimson'),
        0.618: ('1° Noble', 'teal'),
    }

    for pos, (name, color) in pos_values.items():
        ax2.axvline(pos, color=color, linestyle='--', linewidth=1.5, alpha=0.8)
        # Find ratio at this position
        idx = np.argmin(np.abs(bin_centers - pos))
        ratio_val = ratio_to_uniform[idx]
        # Annotate with ratio value
        ax2.annotate(f'{ratio_val:.2f}x', xy=(pos, ratio_val),
                     xytext=(pos, ratio_val + 0.15),
                     ha='center', fontsize=9, fontweight='bold', color=color,
                     bbox=bbox_style, zorder=20,
                     arrowprops=dict(arrowstyle='->', color=color, lw=1))

    ax2.set_xlabel('Fractional n (mod 1)', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Kernel Density Estimate (numbers show ratio to uniform)',
                  fontsize=11, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0.4, 1.6)
    ax2.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path} ({len(freqs):,} peaks)")

    # Compute stats for return
    observed_20 = np.histogram(u, bins=20, range=(0, 1))[0]
    expected_20 = len(freqs) / 20
    chi2, p_val = stats.chisquare(observed_20, [expected_20] * 20)

    boundary_mask = (u < 0.05) | (u > 0.95)
    attractor_mask = (u > 0.45) & (u < 0.55)
    noble1_mask = (u > 0.568) & (u < 0.668)
    noble2_mask = (u > 0.332) & (u < 0.432)

    boundary_enrich = (boundary_mask.sum() / (0.10 * len(u))) - 1
    attractor_enrich = (attractor_mask.sum() / (0.10 * len(u))) - 1
    noble1_enrich = (noble1_mask.sum() / (0.10 * len(u))) - 1
    noble2_enrich = (noble2_mask.sum() / (0.10 * len(u))) - 1

    return {
        'chi2': chi2,
        'p_value': p_val,
        'boundary_enrich': boundary_enrich,
        'attractor_enrich': attractor_enrich,
        'noble1_enrich': noble1_enrich,
        'noble2_enrich': noble2_enrich
    }


if __name__ == '__main__':
    # Load ALL peaks (includes MUSE + Emotiv combined dataset)
    peaks_df = pd.read_csv('golden_ratio_peaks_ALL.csv')
    freqs = peaks_df['freq'].values

    # Count sessions from summary file
    summary_df = pd.read_csv('golden_ratio_summary_ALL.csv')
    n_sessions = len(summary_df)

    print(f"Loaded {len(freqs):,} peaks from {n_sessions} sessions")
    print(f"Frequency range: {freqs.min():.1f} - {freqs.max():.1f} Hz")

    # Generate main lattice plot for ALL (combined) dataset
    results = plot_lattice_distribution(freqs, 'golden_ratio_lattice_ALL.png', 'ALL', n_sessions)

    print(f"\nResults:")
    print(f"  χ² = {results['chi2']:,.0f}")
    print(f"  Boundary enrichment: {results['boundary_enrich']*100:+.1f}%")
    print(f"  Attractor enrichment: {results['attractor_enrich']*100:+.1f}%")
    print(f"  1° Noble enrichment: {results['noble1_enrich']*100:+.1f}%")
    print(f"  2° Noble enrichment: {results['noble2_enrich']*100:+.1f}%")

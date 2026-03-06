#!/usr/bin/env python3
"""
Regenerate combined modes_vs_nobles visualization using continuous GED data.

This merges the continuous GED peaks (which have no boundary gaps) with
the existing analysis to show the full frequency distribution.
"""

import sys
sys.path.insert(0, './lib')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os

from phi_frequency_model import PHI, F0, POSITION_OFFSETS, get_default_phi_table
from peak_distribution_analysis import ANALYSIS_BANDS, POSITION_COLORS

def load_continuous_peaks(continuous_file: str, band_file: str = None) -> pd.DataFrame:
    """
    Load continuous GED peaks and optionally merge with band-based peaks.

    Parameters
    ----------
    continuous_file : str
        Path to continuous GED peaks CSV
    band_file : str, optional
        Path to band-based GED peaks CSV for comparison

    Returns
    -------
    pd.DataFrame
        Combined peak data
    """
    continuous_df = pd.read_csv(continuous_file)
    print(f"Loaded {len(continuous_df)} continuous GED peaks")

    if band_file and os.path.exists(band_file):
        band_df = pd.read_csv(band_file)
        print(f"Loaded {len(band_df)} band-based GED peaks for comparison")

        # Don't merge - just use continuous for now
        # Could deduplicate later if needed

    return continuous_df


def get_all_predictions(max_freq: float = 45.0) -> list:
    """Get all φⁿ predictions including boundaries."""
    predictions = []

    # Map bands to octaves
    BAND_OCTAVES = {
        'theta': -1,
        'alpha': 0,
        'beta_low': 1,
        'beta_high': 2,
        'gamma': 3
    }

    for band, info in ANALYSIS_BANDS.items():
        octave = BAND_OCTAVES.get(band, 0)
        band_range = info['freq_range']

        # Add boundary at band start (φ^n integer)
        boundary_freq = F0 * (PHI ** octave)
        if boundary_freq <= max_freq and boundary_freq >= 4.0:
            predictions.append({
                'frequency': boundary_freq,
                'position_type': 'boundary',
                'band': band,
                'octave': octave,
                'offset': 0.0
            })

        # Add all position types within the band
        for pos_type, offset in POSITION_OFFSETS.items():
            freq = F0 * (PHI ** (octave + offset))
            if band_range[0] < freq < min(band_range[1], max_freq):
                predictions.append({
                    'frequency': freq,
                    'position_type': pos_type,
                    'band': band,
                    'octave': octave,
                    'offset': offset
                })

    return sorted(predictions, key=lambda x: x['frequency'])


def plot_combined_continuous(
    peaks_df: pd.DataFrame,
    modes_df: pd.DataFrame,
    output_path: str,
    max_freq: float = 45.0
):
    """
    Create combined visualization using continuous GED data.

    Parameters
    ----------
    peaks_df : pd.DataFrame
        Continuous GED peaks with 'frequency' and 'band' columns
    modes_df : pd.DataFrame
        Detected modes from phi_025_modes.csv
    output_path : str
        Output path for figure
    max_freq : float
        Maximum frequency to display
    """
    # Filter to max frequency
    peaks_df = peaks_df[peaks_df['frequency'] <= max_freq].copy()
    freqs = peaks_df['frequency'].values

    # Get predictions
    predictions = get_all_predictions(max_freq)

    # Create figure
    fig, ax = plt.subplots(figsize=(18, 7))

    # 1. Band shading
    band_colors = {
        'theta': '#ffcccc',
        'alpha': '#ccffcc',
        'beta_low': '#ccccff',
        'beta_high': '#ffffcc',
        'gamma': '#ffccff'
    }

    for band, info in ANALYSIS_BANDS.items():
        f_lo, f_hi = info['freq_range']
        if f_lo < max_freq:
            ax.axvspan(f_lo, min(f_hi, max_freq), alpha=0.15,
                      color=band_colors.get(band, 'gray'), zorder=0)
            # Label
            mid = (f_lo + min(f_hi, max_freq)) / 2
            ax.text(mid, 0.002, band.replace('_', ' ').title(),
                   ha='center', va='bottom', fontsize=10, alpha=0.7)

    # 2. Histogram
    bins = np.linspace(4.5, max_freq, 200)
    ax.hist(freqs, bins=bins, density=True, alpha=0.4, color='gray',
            edgecolor='none', label='Distribution')

    # 3. KDE
    kde = gaussian_kde(freqs, bw_method=0.03)
    freq_grid = np.linspace(4.5, max_freq, 1000)
    kde_vals = kde(freq_grid)
    ax.plot(freq_grid, kde_vals, 'b-', linewidth=2, label='KDE', alpha=0.8)
    ax.fill_between(freq_grid, kde_vals, alpha=0.2, color='blue')

    # 4. Detected modes (red lines)
    for _, row in modes_df.iterrows():
        freq = row['frequency']
        if freq <= max_freq:
            ax.axvline(freq, color='red', linewidth=2, alpha=0.8, zorder=10)
            # Triangle marker at top
            ax.plot(freq, kde(freq)[0], 'rv', markersize=8, zorder=11)

    # 5. φⁿ predictions (colored dashed lines)
    added_labels = set()
    for pred in predictions:
        freq = pred['frequency']
        ptype = pred['position_type']
        color = POSITION_COLORS.get(ptype, 'gray')

        # Boundaries get solid lines, others dashed
        if ptype == 'boundary':
            linestyle = '-'
            linewidth = 2.5
            alpha = 0.9
        else:
            linestyle = '--'
            linewidth = 1.5
            alpha = 0.7

        label = ptype if ptype not in added_labels else None
        ax.axvline(freq, color=color, linestyle=linestyle, linewidth=linewidth,
                  alpha=alpha, label=label, zorder=5)
        added_labels.add(ptype)

    # Formatting
    ax.set_xlim(4.5, max_freq)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)

    # Title with stats
    n_modes = len(modes_df[modes_df['frequency'] <= max_freq])
    n_preds = len(predictions)
    ax.set_title(
        f'GED Peak Distribution vs φⁿ Predictions — Continuous Sweep (No Boundary Gaps)\n'
        f'{len(peaks_df):,} GED peaks | {n_modes} detected modes | {n_preds} φⁿ predictions (including boundaries)',
        fontsize=13, fontweight='bold'
    )

    # Legend
    ax.legend(loc='upper right', fontsize=9, ncol=2)

    # Grid
    ax.grid(True, alpha=0.3, axis='x')

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    # Print boundary coverage
    print("\nBoundary coverage (continuous sweep):")
    boundaries = [4.70, 7.60, 12.30, 19.90, 32.19]
    for bf in boundaries:
        if bf <= max_freq:
            near = np.sum((freqs >= bf - 0.5) & (freqs <= bf + 0.5))
            print(f"  φ^n at {bf:.2f} Hz: {near} peaks")


def compare_coverage(continuous_df: pd.DataFrame, band_df: pd.DataFrame, max_freq: float = 45.0):
    """Compare coverage between continuous and band-based detection."""
    cont_freqs = continuous_df[continuous_df['frequency'] <= max_freq]['frequency'].values
    band_freqs = band_df[band_df['frequency'] <= max_freq]['frequency'].values

    print("\n" + "="*60)
    print("COVERAGE COMPARISON: Continuous vs Band-Based")
    print("="*60)

    boundaries = [7.60, 12.30, 19.90, 32.19]

    print(f"\n{'Boundary':>12} {'Continuous':>12} {'Band-based':>12} {'Improvement':>12}")
    print("-"*52)

    for bf in boundaries:
        cont_near = np.sum((cont_freqs >= bf - 0.5) & (cont_freqs <= bf + 0.5))
        band_near = np.sum((band_freqs >= bf - 0.5) & (band_freqs <= bf + 0.5))
        improvement = cont_near - band_near
        print(f"{bf:>12.2f} {cont_near:>12} {band_near:>12} {improvement:>+12}")

    print(f"\n{'Total peaks':>12} {len(cont_freqs):>12} {len(band_freqs):>12}")


if __name__ == '__main__':
    # Paths
    continuous_file = 'exports_peak_distribution/physf_ged/continuous/ged_peaks_continuous.csv'
    band_file = 'exports_peak_distribution/physf_ged/ged_peaks.csv'
    modes_file = 'exports_peak_distribution/physf_ged/phi_025/phi_025_modes.csv'
    output_dir = 'exports_peak_distribution/physf_ged/noble_alignment'

    os.makedirs(output_dir, exist_ok=True)

    # Check if continuous data exists
    if not os.path.exists(continuous_file):
        print(f"ERROR: Continuous GED peaks not found at {continuous_file}")
        print("Run the continuous GED detection first.")
        sys.exit(1)

    # Load data
    continuous_df = load_continuous_peaks(continuous_file, band_file)
    modes_df = pd.read_csv(modes_file)

    # Generate visualization
    output_path = f"{output_dir}/modes_vs_nobles_continuous.png"
    plot_combined_continuous(continuous_df, modes_df, output_path, max_freq=45.0)

    # Compare coverage if band-based data exists
    if os.path.exists(band_file):
        band_df = pd.read_csv(band_file)
        compare_coverage(continuous_df, band_df)

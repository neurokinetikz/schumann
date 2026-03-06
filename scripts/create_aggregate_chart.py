#!/usr/bin/env python3
"""
Create aggregate GED peak distribution chart combining all datasets.
Uses the same style as create_clean_modes_chart.py (logphi style).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from phi_frequency_model import PHI, POSITION_OFFSETS as BASE_OFFSETS

# Constants
F0 = 7.60  # Base frequency

# Extended position offsets including noble_5, noble_6 and their inverses
PHI_INV = 1 / PHI  # 0.618033...
POSITION_OFFSETS = {
    **BASE_OFFSETS,
    'noble_5':     PHI_INV ** 5,       # ≈ 0.0902
    'noble_6':     PHI_INV ** 6,       # ≈ 0.0557
    'inv_noble_5': 1 - PHI_INV ** 5,   # ≈ 0.9098
    'inv_noble_6': 1 - PHI_INV ** 6,   # ≈ 0.9443
}

# Analysis bands
ANALYSIS_BANDS = {
    'theta': {'freq_range': (4.7, 7.6), 'octave': -1, 'color': '#ffcccc'},
    'alpha': {'freq_range': (7.6, 12.3), 'octave': 0, 'color': '#ccffcc'},
    'beta_low': {'freq_range': (12.3, 19.9), 'octave': 1, 'color': '#ccccff'},
    'beta_high': {'freq_range': (19.9, 32.19), 'octave': 2, 'color': '#ffffcc'},
    'gamma': {'freq_range': (32.19, 45.0), 'octave': 3, 'color': '#ffccff'}
}

# All position types with styling
POSITION_STYLES = {
    'boundary': {'color': '#e74c3c', 'linestyle': '-', 'linewidth': 2.0, 'alpha': 0.8},
    'attractor': {'color': '#27ae60', 'linestyle': '--', 'linewidth': 1.5, 'alpha': 0.6},
    'noble_1': {'color': '#3498db', 'linestyle': '--', 'linewidth': 1.2, 'alpha': 0.5},
    'noble_2': {'color': '#9b59b6', 'linestyle': ':', 'linewidth': 1.0, 'alpha': 0.4},
    'noble_3': {'color': '#f39c12', 'linestyle': ':', 'linewidth': 1.0, 'alpha': 0.4},
    'noble_4': {'color': '#1abc9c', 'linestyle': ':', 'linewidth': 1.0, 'alpha': 0.4},
    'noble_5': {'color': '#16a085', 'linestyle': ':', 'linewidth': 0.8, 'alpha': 0.3},
    'noble_6': {'color': '#2c3e50', 'linestyle': ':', 'linewidth': 0.8, 'alpha': 0.3},
    'inv_noble_3': {'color': '#e67e22', 'linestyle': ':', 'linewidth': 1.0, 'alpha': 0.35},
    'inv_noble_4': {'color': '#95a5a6', 'linestyle': ':', 'linewidth': 1.0, 'alpha': 0.35},
    'inv_noble_5': {'color': '#7f8c8d', 'linestyle': ':', 'linewidth': 0.8, 'alpha': 0.3},
    'inv_noble_6': {'color': '#bdc3c7', 'linestyle': ':', 'linewidth': 0.8, 'alpha': 0.3},
}


def freq_to_phi_exp(f):
    """Convert frequency to φ-exponent: n = log(f/F0) / log(PHI)"""
    return np.log(f / F0) / np.log(PHI)


def phi_exp_to_freq(n):
    """Convert φ-exponent to frequency."""
    return F0 * (PHI ** n)


def get_all_predictions(max_freq=45.0):
    """Get all φⁿ predictions including all noble positions."""
    predictions = []

    for band, info in ANALYSIS_BANDS.items():
        octave = info['octave']
        band_range = info['freq_range']

        # Boundary at band start
        boundary_freq = F0 * (PHI ** octave)
        if 4.0 <= boundary_freq <= max_freq:
            predictions.append({
                'frequency': boundary_freq,
                'position_type': 'boundary',
                'band': band,
                'label': f'φ^{octave}'
            })

        # All position types within the band
        for pos_type, offset in POSITION_OFFSETS.items():
            freq = F0 * (PHI ** (octave + offset))
            if band_range[0] < freq < min(band_range[1], max_freq):
                predictions.append({
                    'frequency': freq,
                    'position_type': pos_type,
                    'band': band,
                    'label': None
                })

    return sorted(predictions, key=lambda x: x['frequency'])


def create_logphi_chart(peaks_df, output_path, title_suffix="", max_freq=45.0):
    """Create log-φ scale chart with equal-width φ-octaves (matching reference style)."""

    # Filter data
    peaks_df = peaks_df[peaks_df['frequency'] <= max_freq].copy()
    freqs = peaks_df['frequency'].values

    # Get φ-exponents for all peaks
    phi_exps = freq_to_phi_exp(freqs)

    # Get predictions with φ-exponents
    predictions = get_all_predictions(max_freq)
    for pred in predictions:
        pred['phi_exp'] = freq_to_phi_exp(pred['frequency'])

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 6))

    # Define φ-exponent range (theta to gamma: -1 to ~3)
    n_min = -1.0
    n_max = freq_to_phi_exp(max_freq)

    # 1. Band shading (equal width in φ-exponent space)
    for band, info in ANALYSIS_BANDS.items():
        n_lo = info['octave']
        n_hi = info['octave'] + 1
        if n_lo < n_max:
            ax.axvspan(n_lo, min(n_hi, n_max), alpha=0.08,
                      color=info['color'], zorder=0)

    # 2. Histogram in φ-exponent space (counts, not density)
    bins_per_octave = 30
    phi_bins = np.linspace(n_min, n_max, int((n_max - n_min) * bins_per_octave) + 1)
    counts, bin_edges, patches = ax.hist(phi_exps, bins=phi_bins, alpha=0.6,
                                          color='#b0b0b0', edgecolor='none',
                                          zorder=1)

    # 3. All φⁿ predictions (vertical lines with labels at top)
    ymax = counts.max() * 1.15
    ax.set_ylim(0, ymax)

    # Position type labels at top (name + φ offset)
    position_labels = {
        'boundary': 'Boundary φ⁰',
        'noble_6': 'Noble₆ φ⁻⁶',
        'noble_5': 'Noble₅ φ⁻⁵',
        'noble_4': 'Noble₄ φ⁻⁴',
        'noble_3': 'Noble₃ φ⁻³',
        'noble_2': 'Noble₂ φ⁻²',
        'attractor': 'Attractor φ⁰·⁵',
        'noble_1': 'Noble₁ φ⁻¹',
        'inv_noble_3': 'Inv₃ 1-φ⁻³',
        'inv_noble_4': 'Inv₄ 1-φ⁻⁴',
        'inv_noble_5': 'Inv₅ 1-φ⁻⁵',
        'inv_noble_6': 'Inv₆ 1-φ⁻⁶',
    }

    # Add prediction lines with top labels
    bbox_style = dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.85,
                      edgecolor='none')

    for pred in predictions:
        n = pred['phi_exp']
        ptype = pred['position_type']
        freq = pred['frequency']
        style = POSITION_STYLES.get(ptype, POSITION_STYLES['boundary'])

        # Vertical line
        ax.axvline(n, color=style['color'], linestyle=style['linestyle'],
                  linewidth=style['linewidth'], alpha=style['alpha'], zorder=5)

        # Add label at top (frequency and position type)
        label_text = f"{freq:.1f}"
        ax.text(n, ymax * 0.98, label_text,
               ha='center', va='top', fontsize=7, fontweight='bold',
               color=style['color'], alpha=0.9, rotation=90,
               bbox=bbox_style)

    # Add position type legend at very top (single row)
    unique_types_in_order = ['boundary', 'noble_6', 'noble_5', 'noble_4', 'noble_3', 'noble_2',
                             'attractor', 'noble_1', 'inv_noble_3', 'inv_noble_4', 'inv_noble_5', 'inv_noble_6']

    legend_x_start = 0.0
    legend_spacing = 0.075

    for i, ptype in enumerate(unique_types_in_order):
        if ptype in POSITION_STYLES:
            style = POSITION_STYLES[ptype]
            x_pos = legend_x_start + i * legend_spacing
            ax.annotate(position_labels.get(ptype, ptype),
                       xy=(x_pos, 1.08), xycoords='axes fraction',
                       fontsize=7, color=style['color'], ha='left',
                       fontweight='bold')

    # 4. Band labels inside chart at bottom
    band_positions = {
        'theta': (-0.5, 'Theta'),
        'alpha': (0.5, 'Alpha'),
        'beta_low': (1.5, 'Beta Low'),
        'beta_high': (2.5, 'Beta High'),
        'gamma': (3.25, 'Gamma')
    }
    for band, (mid_n, label) in band_positions.items():
        if mid_n < n_max:
            ax.text(mid_n, ymax * 0.03, label,
                   ha='center', va='bottom', fontsize=11, fontweight='bold',
                   color='#555555')

    # 5. Secondary x-axis labels showing frequency values at integer φ-exponents
    for n in range(-1, 4):
        freq = phi_exp_to_freq(n)
        if freq <= max_freq:
            ax.text(n, ymax * 1.02, f'{freq:.1f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold',
                   color='#333333')

    # Formatting
    ax.set_xlim(n_min, n_max)
    ax.set_ylim(0, ymax)

    ax.set_xlabel(f'φ-exponent (n)        f = {F0:.2f} × φⁿ Hz', fontsize=12, fontweight='bold')
    ax.set_ylabel('Peak Count', fontsize=12, fontweight='bold')

    # X-axis ticks at integer φ-exponents
    ax.set_xticks(range(-1, 4))
    ax.tick_params(axis='x', which='both', length=0)
    ax.tick_params(axis='y', which='both', length=0)

    # Title
    ax.set_title(
        f'GED Peak Distribution vs φⁿ Predictions — Log-φ Scale (F0={F0:.2f} Hz)\n'
        f'{len(peaks_df):,} peaks | Equal-width φ-octaves{title_suffix}',
        fontsize=13, fontweight='bold', pad=40
    )

    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def load_dataset(csv_path):
    """Load a GED peaks CSV file."""
    df = pd.read_csv(csv_path)
    # Handle different column names
    if 'frequency' not in df.columns and 'freq_hz' in df.columns:
        df['frequency'] = df['freq_hz']
    return df


def main():
    """Main entry point."""
    # Dataset paths
    datasets = {
        'PhySF': 'exports_peak_distribution/physf_ged/truly_continuous/ged_peaks_truly_continuous.csv',
        'MPENG': 'exports_peak_distribution/mpeng_ged/truly_continuous/ged_peaks_truly_continuous.csv',
        'Emotions': 'exports_peak_distribution/emotions_ged/truly_continuous/ged_peaks_truly_continuous.csv',
    }

    # Output directory
    output_dir = 'exports_peak_distribution/aggregate_truly_continuous'
    os.makedirs(output_dir, exist_ok=True)

    # Load and combine all datasets
    all_data = []
    dataset_stats = {}

    for name, path in datasets.items():
        if os.path.exists(path):
            df = load_dataset(path)
            df['dataset'] = name
            all_data.append(df)
            sessions = df['session'].nunique() if 'session' in df.columns else 'N/A'
            dataset_stats[name] = {'peaks': len(df), 'sessions': sessions}
            print(f"Loaded {name}: {len(df):,} peaks from {sessions} sessions")
        else:
            print(f"Warning: {path} not found")

    if not all_data:
        print("No data loaded!")
        return

    combined = pd.concat(all_data, ignore_index=True)
    total_peaks = len(combined)
    print(f"\nTotal: {total_peaks:,} peaks")

    # Create aggregate chart
    print("\nCreating aggregate chart...")
    title_suffix = f"\n(PhySF + MPENG + Emotions)"
    create_logphi_chart(
        combined,
        os.path.join(output_dir, 'aggregate_modes_logphi_f0_760.png'),
        title_suffix=title_suffix
    )

    # Create individual charts with consistent styling
    for name, path in datasets.items():
        if os.path.exists(path):
            print(f"\nCreating {name} individual chart...")
            df = load_dataset(path)
            create_logphi_chart(
                df,
                os.path.join(output_dir, f'{name.lower()}_modes_logphi_f0_760.png'),
                title_suffix=f" — {name}"
            )

    print("\nDone!")
    print(f"\nGenerated charts in: {output_dir}/")


if __name__ == '__main__':
    main()

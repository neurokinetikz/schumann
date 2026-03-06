#!/usr/bin/env python3
"""
Create a clean, readable modes vs predictions chart.
"""

import sys
sys.path.insert(0, './lib')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os

from phi_frequency_model import PHI, F0, POSITION_OFFSETS as BASE_OFFSETS

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


def create_clean_chart(peaks_df, modes_df, output_path, max_freq=45.0):
    """Create a clean, readable visualization with log-φ scale x-axis."""

    # Filter data
    peaks_df = peaks_df[peaks_df['frequency'] <= max_freq].copy()
    freqs = peaks_df['frequency'].values

    # Convert to φ-exponent: n = log(f/F0) / log(PHI)
    def freq_to_phi_exp(f):
        return np.log(f / F0) / np.log(PHI)

    def phi_exp_to_freq(n):
        return F0 * (PHI ** n)

    # Get φ-exponents for all peaks
    phi_exps = freq_to_phi_exp(freqs)

    # Get ALL predictions (including all noble positions)
    predictions = get_all_predictions(max_freq)
    for pred in predictions:
        pred['phi_exp'] = freq_to_phi_exp(pred['frequency'])

    # Create figure with better proportions
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

    # 2. Histogram in φ-exponent space (subtle)
    bins_per_octave = 30
    phi_bins = np.linspace(n_min, n_max, int((n_max - n_min) * bins_per_octave) + 1)
    ax.hist(phi_exps, bins=phi_bins, density=True, alpha=0.25, color='#34495e',
            edgecolor='none', label='Peak distribution', zorder=1)

    # 3. KDE curve in φ-exponent space (smooth)
    kde = gaussian_kde(phi_exps, bw_method=0.08)
    phi_grid = np.linspace(n_min, n_max, 500)
    kde_vals = kde(phi_grid)
    ax.plot(phi_grid, kde_vals, color='#2c3e50', linewidth=2.5,
            label='Density (KDE)', zorder=3)
    ax.fill_between(phi_grid, kde_vals, alpha=0.15, color='#2c3e50', zorder=2)

    # 4. Detected modes (triangle markers only - no vertical lines)
    mode_phi_exps_plotted = []
    for _, row in modes_df.iterrows():
        freq = row['frequency']
        if freq <= max_freq:
            mode_phi_exps_plotted.append(freq_to_phi_exp(freq))

    # Add mode markers at KDE curve
    if mode_phi_exps_plotted:
        mode_heights = [kde(n)[0] for n in mode_phi_exps_plotted]
        ax.scatter(mode_phi_exps_plotted, mode_heights,
                  marker='v', s=100, c='#c0392b', edgecolor='white',
                  linewidth=1.5, zorder=10, label=f'Detected modes (n={len(mode_phi_exps_plotted)})')

    # 5. All φⁿ predictions (subtle reference lines for all positions)
    added_types = set()
    for pred in predictions:
        n = pred['phi_exp']
        ptype = pred['position_type']
        style = POSITION_STYLES[ptype]

        # Add to legend only once per type
        label = ptype.replace('_', ' ').title() if ptype not in added_types else None
        added_types.add(ptype)

        ax.axvline(n, color=style['color'], linestyle=style['linestyle'],
                  linewidth=style['linewidth'], alpha=style['alpha'],
                  label=label, zorder=5)

        # Add frequency label for boundaries only
        if pred['label']:
            ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0
            ax.text(n, ymax * 0.95, f"{pred['frequency']:.1f}",
                   ha='center', va='top', fontsize=8, fontweight='bold',
                   color=style['color'], alpha=0.9)

    # 6. Band labels at bottom
    band_positions = {
        'theta': (-0.5, 'Theta'),
        'alpha': (0.5, 'Alpha'),
        'beta_low': (1.5, 'Beta Low'),
        'beta_high': (2.5, 'Beta High'),
        'gamma': (3.25, 'Gamma')
    }
    ymin = ax.get_ylim()[0]
    for band, (mid_n, label) in band_positions.items():
        if mid_n < n_max:
            ax.text(mid_n, ymin + 0.01, label,
                   ha='center', va='bottom', fontsize=10, fontweight='bold',
                   color='#7f8c8d', alpha=0.8)

    # 7. Add frequency annotations at integer φ-exponents (top of chart)
    ymax = ax.get_ylim()[1]
    for n in range(-1, 4):
        freq = phi_exp_to_freq(n)
        if freq <= max_freq:
            ax.text(n, ymax * 1.02, f'{freq:.1f} Hz',
                   ha='center', va='bottom', fontsize=9, fontweight='bold',
                   color='#333333')

    # Formatting
    ax.set_xlim(n_min, n_max)
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 1.08)
    ax.set_xlabel(f'φ-exponent (n)        f = {F0:.2f} × φⁿ Hz', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')

    # X-axis ticks at integer φ-exponents
    ax.set_xticks(range(-1, 4))
    ax.tick_params(axis='x', which='both', length=0)
    ax.tick_params(axis='y', which='both', length=0)

    # Title
    ax.set_title(
        f'Peak Distribution vs φⁿ Predictions — Log-φ Scale (F0={F0:.2f} Hz)\n'
        f'{len(peaks_df):,} peaks | {len(mode_phi_exps_plotted)} detected modes | Equal-width φ-octaves',
        fontsize=13, fontweight='bold', pad=15
    )

    # Legend (bottom right)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.95,
             edgecolor='#bdc3c7', fancybox=True)

    # No grid - prediction lines provide reference
    ax.set_axisbelow(True)

    # Remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_logphi_chart(peaks_df, modes_df, output_path, max_freq=45.0):
    """Create log-φ scale chart with equal-width φ-octaves."""

    # Filter data
    peaks_df = peaks_df[peaks_df['frequency'] <= max_freq].copy()
    freqs = peaks_df['frequency'].values

    # Convert to φ-exponent: n = log(f/F0) / log(PHI)
    def freq_to_phi_exp(f):
        return np.log(f / F0) / np.log(PHI)

    def phi_exp_to_freq(n):
        return F0 * (PHI ** n)

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

    # 2. Histogram in φ-exponent space, normalized by Hz-width per bin
    # Equal bins in φ-exponent space span different Hz widths, so raw counts
    # show an increasing trend that obscures the actual lattice signal.
    # Dividing by Hz-width gives peaks/Hz — a flat line under uniform density.
    bins_per_octave = 60
    phi_bins = np.linspace(n_min, n_max, int((n_max - n_min) * bins_per_octave) + 1)
    counts, bin_edges = np.histogram(phi_exps, bins=phi_bins)

    # Compute Hz-width for each bin
    bin_lo_hz = phi_exp_to_freq(bin_edges[:-1])
    bin_hi_hz = phi_exp_to_freq(bin_edges[1:])
    bin_width_hz = bin_hi_hz - bin_lo_hz

    # Density = counts / Hz-width (peaks per Hz)
    density = counts / np.where(bin_width_hz > 0, bin_width_hz, 1e-6)

    # Plot as bar chart
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width_phi = bin_edges[1] - bin_edges[0]
    ax.bar(bin_centers, density, width=bin_width_phi, alpha=0.6,
           color='#b0b0b0', edgecolor='none', zorder=1)

    # Overlay uniform null expectation (flat line = no lattice structure)
    total_peaks = len(phi_exps)
    freq_lo = phi_exp_to_freq(n_min)
    freq_hi = phi_exp_to_freq(n_max)
    uniform_density = total_peaks / (freq_hi - freq_lo)
    ax.axhline(uniform_density, color='#000000', linestyle='--', linewidth=2.0,
               alpha=0.7, zorder=6, label=f'Uniform null ({uniform_density:.0f} peaks/Hz)')

    # 3. All φⁿ predictions (vertical lines with labels at top)
    ymax = density.max() * 1.15
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

    # 5. Secondary x-axis labels showing frequency values
    # Add frequency annotations at integer φ-exponents
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
    ax.set_ylabel('Peak Density (peaks / Hz)', fontsize=12, fontweight='bold')

    # X-axis ticks at integer φ-exponents
    ax.set_xticks(range(-1, 4))
    ax.tick_params(axis='x', which='both', length=0)
    ax.tick_params(axis='y', which='both', length=0)

    # Title
    ax.set_title(
        f'Peak Density vs φⁿ Predictions — Log-φ Scale (F0={F0:.2f} Hz)\n'
        f'{len(peaks_df):,} peaks | Hz-normalized | Equal-width φ-octaves',
        fontsize=13, fontweight='bold', pad=40
    )

    # Legend for null line
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95,
             edgecolor='#bdc3c7', fancybox=True)

    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_comparison_chart(band_df, continuous_df, modes_df, output_path, max_freq=45.0):
    """Create side-by-side comparison of band-based vs continuous detection."""

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    predictions = get_all_predictions(max_freq)

    for ax, (df, title, color) in zip(axes, [
        (band_df, f'Band-by-Band Detection ({len(band_df):,} peaks) — Gaps at Boundaries', '#3498db'),
        (continuous_df, f'Continuous Sweep Detection ({len(continuous_df):,} peaks) — No Gaps', '#27ae60')
    ]):
        df = df[df['frequency'] <= max_freq].copy()
        freqs = df['frequency'].values

        # Band shading
        for band, info in ANALYSIS_BANDS.items():
            f_lo, f_hi = info['freq_range']
            if f_lo < max_freq:
                ax.axvspan(f_lo, min(f_hi, max_freq), alpha=0.08,
                          color=info['color'], zorder=0)

        # Histogram
        bins = np.linspace(4.5, max_freq, 100)
        ax.hist(freqs, bins=bins, density=True, alpha=0.3, color=color,
                edgecolor='none', zorder=1, label='Distribution')

        # KDE
        kde = gaussian_kde(freqs, bw_method=0.04)
        freq_grid = np.linspace(4.5, max_freq, 500)
        kde_vals = kde(freq_grid)
        ax.plot(freq_grid, kde_vals, color=color, linewidth=2.5, zorder=3, label='KDE')
        ax.fill_between(freq_grid, kde_vals, alpha=0.2, color=color, zorder=2)

        # All position type lines (subtle)
        added_types = set()
        for pred in predictions:
            ptype = pred['position_type']
            style = POSITION_STYLES[ptype]
            # Add label only once per type
            label = ptype.replace('_', ' ').title() if ptype not in added_types else None
            added_types.add(ptype)
            ax.axvline(pred['frequency'], color=style['color'],
                      linewidth=style['linewidth'], linestyle=style['linestyle'],
                      alpha=style['alpha'] * 0.7, zorder=4, label=label)

        # Detected modes (triangles only)
        mode_freqs = [row['frequency'] for _, row in modes_df.iterrows() if row['frequency'] <= max_freq]
        if mode_freqs:
            mode_y = [kde(f)[0] for f in mode_freqs]
            ax.scatter(mode_freqs, mode_y, marker='v', s=60, c='#c0392b',
                      edgecolor='white', linewidth=1, zorder=10,
                      label=f'Modes (n={len(mode_freqs)})')

        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim(4.5, max_freq)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', which='both', length=0)

        # X-axis ticks: major every 1 Hz, minor every 0.5 Hz
        # No tick lines (length=0) to avoid clutter with prediction vertical lines
        from matplotlib.ticker import MultipleLocator
        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax.tick_params(axis='x', which='both', length=0)

        # Legend bottom right (multiple columns for all entries)
        ax.legend(loc='lower right', fontsize=7, framealpha=0.9, ncol=2)

    axes[1].set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')

    # Add annotations for boundary gaps
    for bf in [7.60, 12.30, 19.90, 32.19]:
        axes[0].annotate('gap', xy=(bf, 0), xytext=(bf, 0.005),
                        ha='center', fontsize=8, color='red', alpha=0.7)

    plt.suptitle('Comparison: Band-Based vs Continuous GED Detection',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def run_charts(continuous_file, output_dir, custom_f0=None, band_file=None, modes_file=None, max_freq=45.0):
    """
    Generate charts with optional custom F0 and paths.

    Parameters
    ----------
    continuous_file : str
        Path to continuous GED peaks CSV
    output_dir : str
        Output directory for charts
    custom_f0 : float, optional
        Custom F0 value (if None, uses default 7.60)
    band_file : str, optional
        Path to band-based peaks CSV (for comparison chart)
    modes_file : str, optional
        Path to modes CSV
    max_freq : float
        Maximum frequency to include
    """
    global F0

    # Override F0 if custom value provided
    original_f0 = F0
    if custom_f0 is not None:
        F0 = custom_f0
        print(f"Using custom F0: {F0:.4f} Hz")
    else:
        print(f"Using default F0: {F0:.2f} Hz")

    os.makedirs(output_dir, exist_ok=True)

    # Load continuous data
    print(f"Loading peaks from: {continuous_file}")
    continuous_df = pd.read_csv(continuous_file)
    print(f"Loaded {len(continuous_df):,} peaks")

    # Load modes if available
    if modes_file and os.path.exists(modes_file):
        modes_df = pd.read_csv(modes_file)
        print(f"Loaded {len(modes_df)} modes")
    else:
        # Create empty modes DataFrame
        modes_df = pd.DataFrame(columns=['band', 'mode_idx', 'frequency', 'ratio_to_next'])
        print("No modes file provided, skipping mode markers")

    # Create log-φ scale chart
    output_name = f"modes_vs_predictions_logphi"
    if custom_f0:
        output_name += f"_f0_{custom_f0:.2f}".replace('.', '')
    print(f"\nCreating log-φ scale chart with F0={F0:.4f}...")
    create_logphi_chart(
        continuous_df, modes_df,
        f"{output_dir}/{output_name}.png",
        max_freq=max_freq
    )

    # Create clean chart
    print("\nCreating clean chart...")
    create_clean_chart(
        continuous_df, modes_df,
        f"{output_dir}/modes_vs_predictions_clean.png",
        max_freq=max_freq
    )

    # Create comparison chart if band file available
    if band_file and os.path.exists(band_file):
        print(f"\nLoading band-based peaks from: {band_file}")
        band_df = pd.read_csv(band_file)
        print(f"Band-based peaks: {len(band_df):,}")

        print("\nCreating comparison chart...")
        create_comparison_chart(
            band_df, continuous_df, modes_df,
            f"{output_dir}/band_vs_continuous_comparison.png",
            max_freq=max_freq
        )

    # Restore original F0
    F0 = original_f0

    print("\nDone!")


if __name__ == '__main__':
    # Parse command-line arguments
    # Usage: python create_clean_modes_chart.py [peaks_file] [output_dir] [f0]
    if len(sys.argv) >= 4:
        continuous_file = sys.argv[1]
        output_dir = sys.argv[2]
        custom_f0 = float(sys.argv[3])
        band_file = None
        modes_file = None
    elif len(sys.argv) >= 3:
        continuous_file = sys.argv[1]
        output_dir = sys.argv[2]
        custom_f0 = None
        band_file = None
        modes_file = None
    elif len(sys.argv) == 2:
        continuous_file = sys.argv[1]
        output_dir = os.path.dirname(continuous_file)
        custom_f0 = None
        band_file = None
        modes_file = None
    else:
        # Default PhySF paths
        band_file = 'exports_peak_distribution/physf_ged/ged_peaks.csv'
        continuous_file = 'exports_peak_distribution/physf_ged/continuous_v2/ged_peaks_continuous.csv'
        modes_file = 'exports_peak_distribution/physf_ged/phi_025/phi_025_modes.csv'
        output_dir = 'exports_peak_distribution/physf_ged/noble_alignment'
        custom_f0 = None

    print("=" * 60)
    print("GED Peak Distribution Chart Generator")
    print("=" * 60)
    print(f"Peaks file: {continuous_file}")
    print(f"Output dir: {output_dir}")
    if custom_f0:
        print(f"Custom F0: {custom_f0:.4f} Hz")
    print("=" * 60)

    run_charts(
        continuous_file=continuous_file,
        output_dir=output_dir,
        custom_f0=custom_f0,
        band_file=band_file,
        modes_file=modes_file,
        max_freq=45.0
    )

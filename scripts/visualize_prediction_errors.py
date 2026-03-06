#!/usr/bin/env python3
"""
Visualize gedBounds Prediction Error Analysis
==============================================

Creates 4 visualization figures showing how detected gedBounds
boundaries align with predicted φⁿ positions.

Outputs:
- prediction_scatter.png       # Detected vs Predicted
- error_by_position_type.png   # Box plot comparison
- frequency_alignment.png      # Stem plot with φⁿ grid
- error_histogram.png          # Distribution of errors
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'exports_peak_distribution' / 'true_gedbounds'
FIG_DIR = DATA_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df_errors = pd.read_csv(DATA_DIR / 'prediction_error_analysis.csv')
df_comprehensive = pd.read_csv(DATA_DIR / 'comprehensive_phi_comparison.csv')
df_phi_ref = pd.read_csv(DATA_DIR / 'phi_positions_reference.csv')

# Color palette for position types
COLORS = {
    'noble_1': '#27ae60',    # Green (most noble)
    'noble_2': '#2ecc71',    # Light green
    'noble_3': '#1abc9c',    # Teal
    'noble_4': '#16a085',    # Dark teal
    'boundary': '#e74c3c',   # Red
    'attractor': '#3498db',  # Blue
    'inv_noble_3': '#9b59b6', # Purple
    'inv_noble_4': '#8e44ad'  # Dark purple
}

# Position type display names
POSITION_NAMES = {
    'noble_1': '1° Noble (φ⁻¹)',
    'noble_2': '2° Noble',
    'noble_3': '3° Noble',
    'noble_4': '4° Noble',
    'boundary': 'Boundary (n∈ℤ)',
    'attractor': 'Attractor',
    'inv_noble_3': '3° Inv Noble',
    'inv_noble_4': '4° Inv Noble'
}


def plot_prediction_scatter():
    """Figure 1: Detected vs Predicted Scatter Plot"""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot perfect prediction line
    ax.plot([4, 46], [4, 46], 'k--', alpha=0.4, linewidth=2, label='Perfect prediction (y=x)')

    # Add ±0.5 Hz bands
    ax.fill_between([4, 46], [3.5, 45.5], [4.5, 46.5], alpha=0.1, color='green', label='±0.5 Hz band')

    # Plot each point by position type
    for pos_type, color in COLORS.items():
        subset = df_errors[df_errors['position_type'] == pos_type]
        if len(subset) > 0:
            ax.scatter(subset['nearest_phi_freq'], subset['detected_freq'],
                      c=color, s=120, alpha=0.8,
                      label=POSITION_NAMES.get(pos_type, pos_type),
                      edgecolors='white', linewidth=1.5, zorder=5)

    # Annotate best matches (error < 0.15 Hz)
    for _, row in df_errors[df_errors['distance_hz'] < 0.15].iterrows():
        ax.annotate(f"Δ={row['distance_hz']:.3f} Hz",
                   (row['nearest_phi_freq'], row['detected_freq']),
                   textcoords="offset points", xytext=(8, 8), fontsize=8,
                   arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    # Statistics annotation
    mean_err = df_errors['distance_hz'].mean()
    median_err = df_errors['distance_hz'].median()
    pct_05 = (df_errors['distance_hz'] <= 0.5).mean() * 100

    stats_text = (f"N = {len(df_errors)} boundaries\n"
                  f"Mean Error: {mean_err:.3f} Hz\n"
                  f"Median Error: {median_err:.3f} Hz\n"
                  f"{pct_05:.1f}% within 0.5 Hz")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Predicted φⁿ Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Detected Boundary Frequency (Hz)', fontsize=12)
    ax.set_title('gedBounds Detection vs φⁿ Prediction\n(0.1 Hz Resolution)', fontsize=14)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(4, 46)
    ax.set_ylim(4, 46)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'prediction_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIG_DIR / 'prediction_scatter.png'}")


def plot_error_by_position_type():
    """Figure 2: Error by Position Type Box/Violin Plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Prepare data grouped by position type
    position_order = ['noble_1', 'noble_2', 'noble_3', 'noble_4',
                      'attractor', 'boundary', 'inv_noble_3', 'inv_noble_4']

    data_by_type = []
    labels = []
    colors = []

    for pos in position_order:
        subset = df_errors[df_errors['position_type'] == pos]['distance_hz']
        if len(subset) > 0:
            data_by_type.append(subset.values)
            labels.append(POSITION_NAMES.get(pos, pos))
            colors.append(COLORS[pos])

    # Box plot
    bp = ax1.boxplot(data_by_type, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax1.set_ylabel('Absolute Error (Hz)', fontsize=12)
    ax1.set_xlabel('Position Type', fontsize=12)
    ax1.set_title('Prediction Error by φⁿ Position Type', fontsize=14)
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='0.5 Hz threshold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend()

    # Bar plot of mean error by type
    means = [np.mean(d) for d in data_by_type]
    stds = [np.std(d) for d in data_by_type]
    x_pos = np.arange(len(labels))

    bars = ax2.bar(x_pos, means, yerr=stds, capsize=4, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.set_ylabel('Mean Absolute Error (Hz)', fontsize=12)
    ax2.set_xlabel('Position Type', fontsize=12)
    ax2.set_title('Mean Prediction Error ± Std by Position Type', fontsize=14)
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='0.5 Hz threshold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add count annotations
    for i, d in enumerate(data_by_type):
        ax2.annotate(f'n={len(d)}', (i, means[i] + stds[i] + 0.05),
                    ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'error_by_position_type.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIG_DIR / 'error_by_position_type.png'}")


def plot_frequency_alignment():
    """Figure 3: Frequency Alignment Stem Plot with φⁿ Grid"""
    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot φⁿ reference positions as background
    for _, row in df_phi_ref.iterrows():
        pos_type = row['position_type']
        freq = row['frequency']
        color = COLORS.get(pos_type, 'gray')

        # Vertical lines for φⁿ positions
        if pos_type == 'boundary':
            ax.axvline(x=freq, color=color, alpha=0.6, linewidth=2, linestyle='-')
        elif 'noble' in pos_type:
            ax.axvline(x=freq, color=color, alpha=0.3, linewidth=1, linestyle='--')
        else:
            ax.axvline(x=freq, color=color, alpha=0.2, linewidth=1, linestyle=':')

    # Plot detected boundaries as stems
    detected = df_errors['detected_freq'].values
    predicted = df_errors['nearest_phi_freq'].values
    errors = df_errors['distance_hz'].values
    pos_types = df_errors['position_type'].values

    # Stem plot - detected frequencies
    markerline, stemlines, baseline = ax.stem(detected, np.ones(len(detected)),
                                               linefmt='k-', markerfmt='ko', basefmt=' ')
    plt.setp(stemlines, linewidth=1.5, alpha=0.7)
    plt.setp(markerline, markersize=8)

    # Draw arrows from detected to predicted
    for det, pred, err, pos in zip(detected, predicted, errors, pos_types):
        color = COLORS.get(pos, 'gray')
        arrow_style = '->' if err < 0.5 else '-|>'
        ax.annotate('', xy=(pred, 0.5), xytext=(det, 1),
                   arrowprops=dict(arrowstyle=arrow_style, color=color,
                                  lw=1.5, alpha=0.6))
        # Label with error
        if err < 0.2:
            ax.text(det, 1.05, f'{err:.2f}', ha='center', fontsize=7, rotation=90)

    # Add band labels
    bands = {'theta': (4.5, 7.6), 'alpha': (7.6, 12.3), 'beta_low': (12.3, 19.9),
             'beta_high': (19.9, 32.2), 'gamma': (32.2, 45)}
    band_colors = {'theta': '#f1c40f', 'alpha': '#e67e22', 'beta_low': '#e74c3c',
                   'beta_high': '#9b59b6', 'gamma': '#3498db'}

    for band, (start, end) in bands.items():
        mid = (start + end) / 2
        ax.axhspan(-0.15, -0.05, xmin=(start-4.5)/40.5, xmax=(end-4.5)/40.5,
                   alpha=0.3, color=band_colors[band])
        ax.text(mid, -0.1, band.replace('_', ' ').title(), ha='center', fontsize=9)

    # Legend for position types
    legend_patches = [mpatches.Patch(color=COLORS[pt], label=POSITION_NAMES[pt], alpha=0.6)
                     for pt in COLORS.keys()]
    ax.legend(handles=legend_patches, loc='upper right', ncol=2, fontsize=8)

    ax.set_xlim(4.5, 45)
    ax.set_ylim(-0.2, 1.3)
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('', fontsize=12)
    ax.set_title('Detected Boundaries (stems) vs φⁿ Positions (vertical lines)\nArrows show mapping to nearest φⁿ position', fontsize=14)
    ax.set_yticks([])
    ax.grid(True, alpha=0.2, axis='x')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'frequency_alignment.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIG_DIR / 'frequency_alignment.png'}")


def plot_error_histogram():
    """Figure 4: Error Distribution Histogram"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    errors = df_errors['distance_hz'].values
    signed_errors = df_errors['signed_error_hz'].values

    # Absolute error histogram
    bins = np.arange(0, max(errors) + 0.1, 0.1)
    n, bins_out, patches = ax1.hist(errors, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')

    # Color bars by threshold
    for i, patch in enumerate(patches):
        if bins[i] <= 0.2:
            patch.set_facecolor('#27ae60')  # Green - excellent
        elif bins[i] <= 0.5:
            patch.set_facecolor('#f39c12')  # Orange - good
        else:
            patch.set_facecolor('#e74c3c')  # Red - poor

    # Add threshold lines
    ax1.axvline(x=0.2, color='green', linestyle='--', linewidth=2, label='0.2 Hz (excellent)')
    ax1.axvline(x=0.5, color='orange', linestyle='--', linewidth=2, label='0.5 Hz (good)')

    # Statistics
    pct_02 = (errors <= 0.2).mean() * 100
    pct_05 = (errors <= 0.5).mean() * 100
    pct_10 = (errors <= 1.0).mean() * 100

    stats_text = (f"N = {len(errors)}\n"
                  f"Mean: {errors.mean():.3f} Hz\n"
                  f"Median: {np.median(errors):.3f} Hz\n"
                  f"≤0.2 Hz: {pct_02:.1f}%\n"
                  f"≤0.5 Hz: {pct_05:.1f}%\n"
                  f"≤1.0 Hz: {pct_10:.1f}%")
    ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax1.set_xlabel('Absolute Error (Hz)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Distribution of Prediction Errors', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')

    # Signed error histogram (to check for bias)
    bins2 = np.arange(min(signed_errors) - 0.1, max(signed_errors) + 0.1, 0.1)
    ax2.hist(signed_errors, bins=bins2, edgecolor='black', alpha=0.7, color='steelblue')
    ax2.axvline(x=0, color='red', linestyle='-', linewidth=2, label='Zero (no bias)')
    ax2.axvline(x=signed_errors.mean(), color='orange', linestyle='--', linewidth=2,
                label=f'Mean: {signed_errors.mean():.3f} Hz')

    ax2.set_xlabel('Signed Error (Detected - Predicted) (Hz)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Signed Error Distribution\n(Checking for systematic bias)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'error_histogram.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIG_DIR / 'error_histogram.png'}")


def plot_comprehensive_summary():
    """Bonus Figure: 4-panel summary combining key insights"""
    fig = plt.figure(figsize=(16, 12))

    # Panel 1: Scatter (top-left)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot([4, 46], [4, 46], 'k--', alpha=0.4, linewidth=2)
    for pos_type, color in COLORS.items():
        subset = df_errors[df_errors['position_type'] == pos_type]
        if len(subset) > 0:
            ax1.scatter(subset['nearest_phi_freq'], subset['detected_freq'],
                       c=color, s=80, alpha=0.7, label=POSITION_NAMES.get(pos_type, pos_type)[:10])
    ax1.set_xlabel('Predicted φⁿ (Hz)')
    ax1.set_ylabel('Detected (Hz)')
    ax1.set_title('Detected vs Predicted')
    ax1.set_xlim(4, 46)
    ax1.set_ylim(4, 46)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Panel 2: Error by type (top-right)
    ax2 = fig.add_subplot(2, 2, 2)
    position_order = ['noble_1', 'noble_2', 'noble_3', 'noble_4',
                      'attractor', 'boundary', 'inv_noble_3', 'inv_noble_4']
    means = []
    labels = []
    colors = []
    for pos in position_order:
        subset = df_errors[df_errors['position_type'] == pos]['distance_hz']
        if len(subset) > 0:
            means.append(subset.mean())
            labels.append(POSITION_NAMES.get(pos, pos)[:8])
            colors.append(COLORS[pos])

    bars = ax2.barh(labels, means, color=colors, alpha=0.7, edgecolor='black')
    ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Mean Abs Error (Hz)')
    ax2.set_title('Error by Position Type')
    ax2.grid(True, alpha=0.3, axis='x')

    # Panel 3: Histogram (bottom-left)
    ax3 = fig.add_subplot(2, 2, 3)
    errors = df_errors['distance_hz'].values
    bins = np.arange(0, max(errors) + 0.15, 0.15)
    ax3.hist(errors, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
    ax3.axvline(x=0.5, color='orange', linestyle='--', linewidth=2)
    pct_05 = (errors <= 0.5).mean() * 100
    ax3.text(0.95, 0.95, f'{pct_05:.0f}% ≤ 0.5 Hz', transform=ax3.transAxes,
            ha='right', va='top', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white'))
    ax3.set_xlabel('Absolute Error (Hz)')
    ax3.set_ylabel('Count')
    ax3.set_title('Error Distribution')
    ax3.grid(True, alpha=0.3, axis='y')

    # Panel 4: Dataset comparison (bottom-right)
    ax4 = fig.add_subplot(2, 2, 4)
    # Use AGGREGATE HIGH sensitivity for comparison
    agg_high = df_comprehensive[(df_comprehensive['dataset'] == 'AGGREGATE') &
                                 (df_comprehensive['sensitivity'] == 'HIGH')]
    pct_by_type = agg_high.groupby('position_type')['match_0.5hz'].mean() * 100

    types_present = [pt for pt in position_order if pt in pct_by_type.index]
    pcts = [pct_by_type[pt] for pt in types_present]
    cols = [COLORS[pt] for pt in types_present]

    ax4.bar([POSITION_NAMES[pt][:8] for pt in types_present], pcts, color=cols, alpha=0.7, edgecolor='black')
    ax4.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax4.set_ylabel('% Within 0.5 Hz')
    ax4.set_title('Match Rate by Position Type')
    ax4.tick_params(axis='x', rotation=45)
    ax4.set_ylim(0, 105)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle('gedBounds Prediction Error Analysis Summary\n(2,950 sessions, 0.1 Hz resolution)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'prediction_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIG_DIR / 'prediction_summary.png'}")


if __name__ == '__main__':
    print("Creating prediction error visualizations...")
    print(f"Data: {len(df_errors)} unique detected frequencies")
    print(f"Reference: {len(df_phi_ref)} φⁿ positions")
    print()

    plot_prediction_scatter()
    plot_error_by_position_type()
    plot_frequency_alignment()
    plot_error_histogram()
    plot_comprehensive_summary()

    print("\n" + "="*60)
    print("All visualizations complete!")
    print(f"Output directory: {FIG_DIR}")

#!/usr/bin/env python3
"""Regenerate peak distribution charts with legend in lower right."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PHI = 1.6180339887
F0 = 7.6

def create_peak_distribution_chart(peak_freqs_arr, title, subtitle, output_path, xlim=48):
    """Create a peak frequency distribution chart with φ^n overlays."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Histogram with fine bins (0.309 Hz resolution)
    bins_fine = np.arange(1, xlim + 3, 0.309)
    n_hist, bins_out, patches = ax.hist(peak_freqs_arr, bins=bins_fine, color='steelblue',
                                         edgecolor='white', alpha=0.85, linewidth=0.5)

    max_height = max(n_hist) if len(n_hist) > 0 else 1

    # Integer φ^n boundaries for EEG band shading
    phi_bounds = {
        -1: F0 * (PHI ** -1),  # ~4.7 Hz
        0: F0,                  # 7.6 Hz
        1: F0 * PHI,            # ~12.3 Hz
        2: F0 * (PHI ** 2),     # ~19.9 Hz
        3: F0 * (PHI ** 3),     # ~32.2 Hz
        4: F0 * (PHI ** 4),     # ~52.1 Hz
    }

    # EEG band shading using integer φ^n as boundaries (subtle muted colors)
    bands = [
        (1, phi_bounds[-1], 'Delta', '#d0d8e0'),
        (phi_bounds[-1], phi_bounds[0], 'Theta', '#e0dcd0'),
        (phi_bounds[0], phi_bounds[1], 'Alpha', '#d0e0d4'),
        (phi_bounds[1], phi_bounds[2], 'Beta-L', '#e0d4d0'),
        (phi_bounds[2], phi_bounds[3], 'Beta-H', '#dcd0e0'),
        (phi_bounds[3], min(xlim, phi_bounds[4]), 'Gamma', '#d0dce0'),
    ]

    for low, high, band_name, color in bands:
        if high > xlim:
            high = xlim
        ax.axvspan(low, high, alpha=0.25, color=color, zorder=0)

    # Band labels flush at top using axes transform
    for low, high, band_name, color in bands:
        if high > xlim:
            high = xlim
        mid = (low + high) / 2
        if mid < xlim:
            ax.text(mid, 0.99, band_name, ha='center', va='top',
                    transform=ax.get_xaxis_transform(),
                    fontsize=8, color='#444444', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.75,
                             edgecolor=color, linewidth=1.0))

    # φ^n reference lines with labels - including ALL position types
    position_types = {
        'boundary':      {'offset': 0.0,   'color': '#cc8800', 'style': '-',  'alpha': 0.6, 'lw': 1.5, 'label': 'Boundary (φⁿ)'},
        'quarter':       {'offset': 0.25,  'color': '#8888cc', 'style': '--', 'alpha': 0.3, 'lw': 0.8, 'label': '¼ (k+0.25)'},
        'noble_2':       {'offset': 0.382, 'color': '#88aa44', 'style': '--', 'alpha': 0.5, 'lw': 1.0, 'label': '2° Noble (k+0.382)'},
        'attractor':     {'offset': 0.5,   'color': '#cc4444', 'style': '--', 'alpha': 0.5, 'lw': 1.2, 'label': 'Attractor (k+0.5)'},
        'noble_1':       {'offset': 0.618, 'color': '#22aa88', 'style': ':',  'alpha': 0.6, 'lw': 1.5, 'label': '1° Noble (k+0.618)'},
        'three_quarter': {'offset': 0.75,  'color': '#888888', 'style': '--', 'alpha': 0.4, 'lw': 0.8, 'label': '¾ (k+0.75)'},
    }

    # Draw lines for each position type with legend entries
    legend_handles = []
    for ptype, props in position_types.items():
        first_line = True
        for k in range(-2, 5):  # Integer base positions
            n = k + props['offset']
            freq = F0 * (PHI ** n)
            if 1 < freq < xlim:
                line = ax.axvline(freq, color=props['color'], linestyle=props['style'],
                                  alpha=props['alpha'], linewidth=props['lw'], zorder=2)
                if first_line:
                    line.set_label(props['label'])
                    legend_handles.append(line)
                    first_line = False

                # Add frequency labels for key positions (boundary, attractor, nobles)
                if ptype in ['boundary', 'attractor', 'noble_1']:
                    if props['offset'] == 0:
                        if k == 0:
                            label = f'f₀\n{freq:.1f}'
                        else:
                            label = f'φ^{k}\n{freq:.1f}'
                    else:
                        label = f'{freq:.1f}'
                    ax.text(freq, 0.95, label, ha='center', va='top',
                           transform=ax.get_xaxis_transform(),
                           fontsize=6, color=props['color'], alpha=0.8)

    # Add legend in LOWER RIGHT
    ax.legend(handles=legend_handles, loc='lower right', fontsize=8, framealpha=0.9)

    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Peak Count', fontsize=12)
    ax.set_title(f'{title}\n{subtitle}', fontsize=13, fontweight='bold')
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, max_height * 1.15)
    ax.set_xticks(np.arange(0, xlim + 5, 5))
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    # Regenerate PRIMARY dataset chart
    print("Loading primary dataset peaks...")
    primary_df = pd.read_csv("papers/golden_ratio_peaks_ALL.csv")
    primary_freqs = primary_df['freq'].values
    n_sessions = primary_df['session'].nunique()
    n_electrodes = primary_df['electrode'].nunique() if 'electrode' in primary_df.columns else 19

    print(f"Primary: {len(primary_freqs)} peaks, {n_sessions} sessions")
    create_peak_distribution_chart(
        primary_freqs,
        'FOOOF Peak Frequency Distribution - ALL',
        f'({len(primary_freqs)} peaks across {n_sessions} sessions, {n_electrodes} electrodes)',
        'golden_ratio_peaks_ALL.png',
        xlim=48
    )

    # Regenerate EMOTIONS dataset chart
    print("\nLoading emotions dataset peaks...")
    emotions_df = pd.read_csv("golden_ratio_peaks_EMOTIONS.csv")
    emotions_freqs = emotions_df['freq'].values
    # Filter to same range as primary for comparison
    emotions_freqs_filtered = emotions_freqs[emotions_freqs <= 48]
    n_sessions_emo = emotions_df['session'].nunique()

    print(f"Emotions: {len(emotions_freqs)} peaks total, {len(emotions_freqs_filtered)} <= 48 Hz, {n_sessions_emo} sessions")
    create_peak_distribution_chart(
        emotions_freqs_filtered,
        'FOOOF Peak Frequency Distribution - EMOTIONS',
        f'({len(emotions_freqs_filtered)} peaks across {n_sessions_emo} sessions)',
        'golden_ratio_peaks_EMOTIONS.png',
        xlim=48
    )

    # Copy to papers/images/
    import shutil
    shutil.copy('golden_ratio_peaks_ALL.png', 'papers/images/')
    shutil.copy('golden_ratio_peaks_EMOTIONS.png', 'papers/images/')
    print("\nCopied both charts to papers/images/")

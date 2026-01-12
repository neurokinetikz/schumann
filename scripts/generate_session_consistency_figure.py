#!/usr/bin/env python3
"""
Generate Session-Level Consistency figure for the paper.
8-panel layout: 6 histograms (top) + bar chart + scatter (bottom).
Includes all position types: Boundary, Quarter (0.25, 0.75), 2° Noble, Attractor, 1° Noble.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.stats import wilcoxon

# Constants
PHI = 1.618033988749895
F0 = 7.6  # Hz

def compute_lattice_coordinate(freq, f0=F0):
    """Compute lattice coordinate u = [log_phi(f/f0)] mod 1."""
    n = np.log(freq / f0) / np.log(PHI)
    return n % 1

def compute_session_enrichments(peaks_df, f0=F0, freq_range=(4, 50), min_peaks=20):
    """Compute enrichment at each position for each session."""
    mask = (peaks_df['freq'] >= freq_range[0]) & (peaks_df['freq'] <= freq_range[1])
    df = peaks_df[mask].copy()

    # Determine session column
    session_col = None
    for col in ['session', 'file', 'filename', 'source']:
        if col in df.columns:
            session_col = col
            break

    if session_col is None:
        if 'file' in df.columns:
            session_col = 'file'
        else:
            df['session'] = df.index // 500
            session_col = 'session'

    results = []
    u_window = 0.05

    # All 6 positions: Boundary, Quarter, 2° Noble, Attractor, 1° Noble, Quarter
    positions = [0.0, 0.25, 0.382, 0.5, 0.618, 0.75]

    for session, group in df.groupby(session_col):
        if len(group) < min_peaks:
            continue

        freqs = group['freq'].values
        u = compute_lattice_coordinate(freqs, f0)
        n_peaks = len(u)

        session_data = {'session': session, 'n_peaks': n_peaks}

        for pos in positions:
            if pos == 0.0:
                # Boundary wraps around 0/1
                in_window = (np.abs(u - 0.0) < u_window) | (np.abs(u - 1.0) < u_window)
                expected_frac = 4 * u_window  # Two windows (at 0 and 1)
            else:
                in_window = np.abs(u - pos) < u_window
                expected_frac = 2 * u_window
            observed_frac = in_window.sum() / n_peaks
            enrichment = observed_frac / expected_frac
            session_data[f'E_{pos}'] = enrichment

        results.append(session_data)

    return pd.DataFrame(results)

def main():
    # Load peaks
    peaks_df = pd.read_csv('golden_ratio_peaks_ALL.csv')
    print(f"Loaded {len(peaks_df):,} peaks")

    # Compute session enrichments
    session_df = compute_session_enrichments(peaks_df)
    n_sessions = len(session_df)
    mean_peaks = session_df['n_peaks'].mean()
    print(f"Computed enrichments for {n_sessions} sessions (mean {mean_peaks:.0f} peaks/session)")

    # Create figure with 8 panels (6 top, 2 bottom)
    fig = plt.figure(figsize=(18, 10))

    # GridSpec: 2 rows, 12 columns for flexible positioning
    gs = GridSpec(2, 12, figure=fig, hspace=0.32, wspace=0.5,
                  left=0.04, right=0.98, top=0.91, bottom=0.08)

    # Colors for each position type
    colors = {
        0.0: '#d62728',     # red - Boundary
        0.25: '#7f7f7f',    # gray - Quarter
        0.382: '#FFA500',   # orange - 2° Noble
        0.5: '#2CA02C',     # green - Attractor
        0.618: '#9467BD',   # purple - 1° Noble
        0.75: '#7f7f7f',    # gray - Quarter
    }

    labels = {
        0.0: 'Boundary',
        0.25: 'Quarter',
        0.382: '2° Noble',
        0.5: 'Attractor',
        0.618: '1° Noble',
        0.75: 'Quarter',
    }

    positions_list = [0.0, 0.25, 0.382, 0.5, 0.618, 0.75]

    # === TOP ROW: 6 Histograms ===
    for idx, pos in enumerate(positions_list):
        ax = fig.add_subplot(gs[0, idx*2:(idx+1)*2])

        data = session_df[f'E_{pos}'].values
        mean_val = data.mean()
        median_val = np.median(data)

        # Wilcoxon test
        stat, p_val = wilcoxon(data - 1.0)

        # Histogram
        ax.hist(data, bins=25, color=colors[pos], alpha=0.7,
                edgecolor='white', linewidth=0.5)

        # Null line
        ax.axvline(1.0, color='black', linestyle='--', linewidth=1.5, label='Null')

        # Mean
        ax.axvline(mean_val, color='darkgreen', linestyle='-', linewidth=1.5,
                   label=f'Mean={mean_val:.2f}')

        # Title
        p_str = f'{p_val:.1e}' if p_val < 0.001 else f'{p_val:.3f}'
        sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else ''))
        ax.set_title(f'{labels[pos]} (u={pos})\np={p_str} {sig}',
                     fontsize=9, fontweight='bold')

        ax.set_xlabel('$E_s$', fontsize=8)
        if idx == 0:
            ax.set_ylabel('Count', fontsize=9)
        ax.legend(fontsize=6, loc='upper right')

        # Set x limits based on data
        xlim_max = max(2.2, data.max() * 1.1)
        ax.set_xlim(0.2, xlim_max)
        ax.tick_params(labelsize=7)

    # === BOTTOM ROW: 2 panels ===
    # Bar chart (left half)
    ax_bar = fig.add_subplot(gs[1, 0:6])

    means = [session_df[f'E_{p}'].mean() for p in positions_list]
    stds = [session_df[f'E_{p}'].std() for p in positions_list]
    n = len(session_df)
    ci95 = [1.96 * s / np.sqrt(n) for s in stds]

    x_pos = np.arange(6)
    bars = ax_bar.bar(x_pos, means, yerr=ci95, capsize=4,
                      color=[colors[p] for p in positions_list],
                      alpha=0.85, edgecolor='black', linewidth=1.0)

    ax_bar.axhline(1.0, color='black', linestyle='--', linewidth=2, label='Null (E=1.0)')

    # Significance markers
    for i, (m, ci) in enumerate(zip(means, ci95)):
        if m - ci > 1.0:
            ax_bar.text(i, m + ci + 0.02, '***', ha='center', fontsize=10, fontweight='bold')
        elif m + ci < 1.0:
            ax_bar.text(i, m + ci + 0.02, '***', ha='center', fontsize=10, fontweight='bold', color='red')

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels([f'{labels[p]}\n(u={p})' for p in positions_list], fontsize=9)
    ax_bar.set_ylabel('Mean Session Enrichment', fontsize=11)
    ax_bar.set_title('Comparison of All Position Enrichments (with 95% CI)', fontsize=11, fontweight='bold')
    ax_bar.legend(fontsize=9, loc='upper right')
    ax_bar.set_ylim(0, 1.65)

    # Scatter plot (right half)
    ax_scatter = fig.add_subplot(gs[1, 6:12])

    x_data = session_df['E_0.5'].values
    y_data = session_df['E_0.618'].values

    ax_scatter.scatter(x_data, y_data, alpha=0.35, s=25, c='steelblue', edgecolors='none')

    # Correlation
    r, p_corr = stats.pearsonr(x_data, y_data)

    # y=x line
    lims = [0.2, 2.8]
    ax_scatter.plot(lims, lims, 'k--', alpha=0.5, linewidth=1.5, label='y=x')

    ax_scatter.set_xlabel('Enrichment at u=0.5 (Attractor)', fontsize=11)
    ax_scatter.set_ylabel('Enrichment at u=0.618 (1° Noble)', fontsize=11)
    ax_scatter.set_title(f'Per-Session Enrichment Correlation\nr = {r:.3f}',
                         fontsize=11, fontweight='bold')
    ax_scatter.set_xlim(0.2, 2.8)
    ax_scatter.set_ylim(0.2, 2.8)
    ax_scatter.set_aspect('equal')
    ax_scatter.legend(fontsize=9, loc='upper left')

    # Main title
    fig.suptitle(f'Session-Level Consistency Analysis\n({n_sessions} sessions, {mean_peaks:.0f} peaks/session mean)',
                 fontsize=13, fontweight='bold', y=0.98)

    # Save
    plt.savefig('phi_session_level_enrichment.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: phi_session_level_enrichment.png")

    # Print summary
    print(f"\n{'='*70}")
    print("SESSION-LEVEL RESULTS")
    print(f"{'='*70}")
    print(f"N sessions: {n_sessions}")
    print(f"Mean peaks/session: {mean_peaks:.0f}")
    print()
    for pos in positions_list:
        mean = session_df[f'E_{pos}'].mean()
        std = session_df[f'E_{pos}'].std()
        ci = 1.96 * std / np.sqrt(n)
        stat, p = wilcoxon(session_df[f'E_{pos}'].values - 1.0)
        d = (mean - 1.0) / std  # Cohen's d
        pct_above = 100 * (session_df[f'E_{pos}'] > 1).mean()
        print(f"{labels[pos]:12s} (u={pos:5.3f}): E = {mean:.3f} [{mean-ci:.3f}, {mean+ci:.3f}], d = {d:+.3f}, {pct_above:.1f}% > 1")

if __name__ == '__main__':
    main()

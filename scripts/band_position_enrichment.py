#!/usr/bin/env python3
"""
Per-Band, Per-Position Enrichment Analysis
============================================

Computes enrichment at each of 12 lattice positions within each of 5
frequency bands. Produces:
  1. CSV table: band x position -> enrichment%, n_peaks, p-value
  2. Heatmap figure
  3. Optional overlay on logphi density chart

Usage:
    python scripts/band_position_enrichment.py --input path/to/peaks.csv
    python scripts/band_position_enrichment.py --input path/to/peaks.csv --overlay
    python scripts/band_position_enrichment.py --input path/to/peaks.csv --n-perm 2000
"""

import os
import sys
import argparse
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from phi_frequency_model import PHI, F0
from analyze_aggregate_enrichment import (
    compute_lattice_coordinate,
    compute_enrichment_at_positions,
    EXTENDED_OFFSETS,
)
from noble_boundary_dissociation import ANALYSIS_BANDS

# Position display order (by offset value)
POSITION_ORDER = [
    'boundary', 'noble_6', 'noble_5', 'noble_4', 'noble_3', 'noble_2',
    'attractor', 'noble_1', 'inv_noble_3', 'inv_noble_4', 'inv_noble_5', 'inv_noble_6',
]

# Band display order
BAND_ORDER = ['theta', 'alpha', 'beta_low', 'beta_high', 'gamma']

# Nice labels
POSITION_LABELS = {
    'boundary':    'Boundary (0.000)',
    'noble_6':     'Noble₆ (0.056)',
    'noble_5':     'Noble₅ (0.090)',
    'noble_4':     'Noble₄ (0.146)',
    'noble_3':     'Noble₃ (0.236)',
    'noble_2':     'Noble₂ (0.382)',
    'attractor':   'Attractor (0.500)',
    'noble_1':     'Noble₁ (0.618)',
    'inv_noble_3': 'Inv₃ (0.764)',
    'inv_noble_4': 'Inv₄ (0.854)',
    'inv_noble_5': 'Inv₅ (0.910)',
    'inv_noble_6': 'Inv₆ (0.944)',
}

BAND_LABELS = {
    'theta':     'Theta\n(4.7–7.6)',
    'alpha':     'Alpha\n(7.6–12.3)',
    'beta_low':  'Beta Low\n(12.3–19.9)',
    'beta_high': 'Beta High\n(19.9–32.2)',
    'gamma':     'Gamma\n(32.2–45.0)',
}


# =========================================================================
# CORE ANALYSIS
# =========================================================================

def compute_band_position_enrichment(freqs, f0=F0, bands=None, offsets=None, window=0.05):
    """
    Compute enrichment at each position within each band.

    Returns DataFrame with columns:
        band, position, offset, n_peaks_band, n_peaks_window,
        enrichment_pct, observed_frac, expected_frac
    """
    if bands is None:
        bands = ANALYSIS_BANDS
    if offsets is None:
        offsets = EXTENDED_OFFSETS

    rows = []
    for band_name in BAND_ORDER:
        if band_name not in bands:
            continue
        f_lo, f_hi = bands[band_name]
        band_freqs = freqs[(freqs >= f_lo) & (freqs < f_hi)]
        n_band = len(band_freqs)

        if n_band < 10:
            # Too few peaks — fill with NaN
            for pos_name in POSITION_ORDER:
                rows.append({
                    'band': band_name,
                    'position': pos_name,
                    'offset': offsets.get(pos_name, np.nan),
                    'n_peaks_band': n_band,
                    'n_peaks_window': 0,
                    'enrichment_pct': np.nan,
                    'observed_frac': np.nan,
                    'expected_frac': 2 * window,
                })
            continue

        lattice = compute_lattice_coordinate(band_freqs, f0)
        enrichment = compute_enrichment_at_positions(lattice, offsets, window)

        for pos_name in POSITION_ORDER:
            if pos_name not in enrichment:
                continue
            e = enrichment[pos_name]
            rows.append({
                'band': band_name,
                'position': pos_name,
                'offset': e['offset'],
                'n_peaks_band': n_band,
                'n_peaks_window': e['n_peaks'],
                'enrichment_pct': e['enrichment_pct'],
                'observed_frac': e['observed_frac'],
                'expected_frac': e['expected_frac'],
            })

    return pd.DataFrame(rows)


def permutation_test_band_position(freqs, f0=F0, bands=None, offsets=None,
                                    window=0.05, n_perm=1000, seed=42):
    """
    Phase-rotation permutation test for each band x position cell.

    Returns DataFrame with enrichment + perm_z, perm_p columns.
    """
    if bands is None:
        bands = ANALYSIS_BANDS
    if offsets is None:
        offsets = EXTENDED_OFFSETS

    rng = np.random.default_rng(seed)

    # First compute observed enrichment
    obs_df = compute_band_position_enrichment(freqs, f0, bands, offsets, window)

    # Build null distributions per band
    null_enrichments = {}  # (band, position) -> list of null enrichments

    for band_name in BAND_ORDER:
        if band_name not in bands:
            continue
        f_lo, f_hi = bands[band_name]
        band_freqs = freqs[(freqs >= f_lo) & (freqs < f_hi)]
        n_band = len(band_freqs)
        if n_band < 10:
            continue

        lattice = compute_lattice_coordinate(band_freqs, f0)

        # Initialize storage
        for pos_name in POSITION_ORDER:
            null_enrichments[(band_name, pos_name)] = []

        # Phase-rotation permutations
        for _ in range(n_perm):
            theta = rng.uniform(0, 1)
            rotated = (lattice + theta) % 1.0
            null_result = compute_enrichment_at_positions(rotated, offsets, window)
            for pos_name in POSITION_ORDER:
                if pos_name in null_result:
                    null_enrichments[(band_name, pos_name)].append(
                        null_result[pos_name]['enrichment_pct']
                    )

    # Compute z-scores and p-values
    perm_z_list = []
    perm_p_list = []

    for _, row in obs_df.iterrows():
        key = (row['band'], row['position'])
        if key in null_enrichments and len(null_enrichments[key]) > 0:
            null = np.array(null_enrichments[key])
            null_mean = np.mean(null)
            null_std = np.std(null)
            obs_val = row['enrichment_pct']

            if null_std > 0:
                z = (obs_val - null_mean) / null_std
                p = np.mean(null >= obs_val)  # one-tailed
            else:
                z = 0.0
                p = 0.5
            perm_z_list.append(z)
            perm_p_list.append(p)
        else:
            perm_z_list.append(np.nan)
            perm_p_list.append(np.nan)

    obs_df['perm_z'] = perm_z_list
    obs_df['perm_p'] = perm_p_list

    return obs_df


def bootstrap_band_position_ci(freqs, f0=F0, bands=None, offsets=None,
                                 window=0.05, n_boot=1000, ci=0.95, seed=42):
    """
    Bootstrap 95% CIs for each band x position enrichment.
    """
    if bands is None:
        bands = ANALYSIS_BANDS
    if offsets is None:
        offsets = EXTENDED_OFFSETS

    rng = np.random.default_rng(seed)
    alpha = (1 - ci) / 2

    ci_data = {}  # (band, position) -> (ci_lower, ci_upper)

    for band_name in BAND_ORDER:
        if band_name not in bands:
            continue
        f_lo, f_hi = bands[band_name]
        band_freqs = freqs[(freqs >= f_lo) & (freqs < f_hi)]
        n_band = len(band_freqs)
        if n_band < 10:
            for pos_name in POSITION_ORDER:
                ci_data[(band_name, pos_name)] = (np.nan, np.nan)
            continue

        lattice = compute_lattice_coordinate(band_freqs, f0)

        # Storage
        boot_samples = {pos: [] for pos in POSITION_ORDER}

        for _ in range(n_boot):
            idx = rng.choice(n_band, size=n_band, replace=True)
            boot_coords = lattice[idx]
            boot_result = compute_enrichment_at_positions(boot_coords, offsets, window)
            for pos_name in POSITION_ORDER:
                if pos_name in boot_result:
                    boot_samples[pos_name].append(boot_result[pos_name]['enrichment_pct'])

        for pos_name in POSITION_ORDER:
            samples = np.array(boot_samples[pos_name])
            if len(samples) > 0:
                ci_data[(band_name, pos_name)] = (
                    np.percentile(samples, alpha * 100),
                    np.percentile(samples, (1 - alpha) * 100),
                )
            else:
                ci_data[(band_name, pos_name)] = (np.nan, np.nan)

    return ci_data


def fdr_correct(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR correction. Returns boolean array of significance."""
    p = np.array(p_values, dtype=float)
    valid = ~np.isnan(p)
    significant = np.zeros(len(p), dtype=bool)

    if valid.sum() == 0:
        return significant

    valid_p = p[valid]
    n = len(valid_p)
    sorted_idx = np.argsort(valid_p)
    sorted_p = valid_p[sorted_idx]

    # BH threshold
    thresholds = alpha * np.arange(1, n + 1) / n
    below = sorted_p <= thresholds

    if below.any():
        max_idx = np.max(np.where(below))
        reject = np.zeros(n, dtype=bool)
        reject[sorted_idx[:max_idx + 1]] = True
        significant[valid] = reject

    return significant


# =========================================================================
# VISUALIZATION
# =========================================================================

def plot_band_position_heatmap(results_df, output_path, title=None):
    """
    Band x Position heatmap of enrichment percentages.
    """
    # Pivot to matrix
    pivot = results_df.pivot(index='position', columns='band', values='enrichment_pct')

    # Reorder
    pivot = pivot.reindex(index=POSITION_ORDER, columns=BAND_ORDER)

    # Labels
    row_labels = [POSITION_LABELS.get(p, p) for p in pivot.index]
    col_labels = [BAND_LABELS.get(b, b) for b in pivot.columns]

    # Significance markers
    annot_matrix = pivot.copy().astype(str)
    if 'significant' in results_df.columns:
        sig_pivot = results_df.pivot(index='position', columns='band', values='significant')
        sig_pivot = sig_pivot.reindex(index=POSITION_ORDER, columns=BAND_ORDER)

        pval_pivot = results_df.pivot(index='position', columns='band', values='perm_p')
        pval_pivot = pval_pivot.reindex(index=POSITION_ORDER, columns=BAND_ORDER)

        for i, pos in enumerate(pivot.index):
            for j, band in enumerate(pivot.columns):
                val = pivot.loc[pos, band]
                if pd.isna(val):
                    annot_matrix.loc[pos, band] = ''
                    continue
                p = pval_pivot.loc[pos, band] if not pd.isna(pval_pivot.loc[pos, band]) else 1.0
                stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                annot_matrix.loc[pos, band] = f'{val:.1f}{stars}'
    else:
        for i, pos in enumerate(pivot.index):
            for j, band in enumerate(pivot.columns):
                val = pivot.loc[pos, band]
                annot_matrix.loc[pos, band] = f'{val:.1f}' if not pd.isna(val) else ''

    # Color limits: symmetric around 0
    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))
    vmax = min(vmax, 100)  # Cap at 100%

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        pivot.astype(float),
        annot=annot_matrix.values,
        fmt='',
        cmap='RdBu_r',
        center=0,
        vmin=-vmax,
        vmax=vmax,
        xticklabels=col_labels,
        yticklabels=row_labels,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'Enrichment (%)'},
        ax=ax,
    )

    # Group separators
    group_boundaries = [0.5, 6.5, 7.5]  # After boundary, after attractor, after noble_1
    for y in group_boundaries:
        ax.axhline(y, color='black', linewidth=1.5)

    if title is None:
        n_peaks = results_df['n_peaks_band'].sum() // len(POSITION_ORDER)
        title = f'Per-Band Per-Position Enrichment (φ-lattice)\n{int(n_peaks):,} peaks total'
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)

    ax.set_xlabel('')
    ax.set_ylabel('')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Heatmap saved: {output_path}")


def plot_band_position_bar(results_df, output_path, title=None):
    """
    Grouped bar chart: for each band, show enrichment at all 12 positions.
    """
    fig, axes = plt.subplots(1, 5, figsize=(20, 6), sharey=True)

    colors = {
        'boundary': '#e74c3c',
        'attractor': '#27ae60',
        'noble_1': '#3498db', 'noble_2': '#9b59b6', 'noble_3': '#f39c12',
        'noble_4': '#1abc9c', 'noble_5': '#16a085', 'noble_6': '#2c3e50',
        'inv_noble_3': '#e67e22', 'inv_noble_4': '#95a5a6',
        'inv_noble_5': '#7f8c8d', 'inv_noble_6': '#bdc3c7',
    }

    for ax, band_name in zip(axes, BAND_ORDER):
        band_data = results_df[results_df['band'] == band_name]
        band_data = band_data.set_index('position').reindex(POSITION_ORDER)

        vals = band_data['enrichment_pct'].values
        bar_colors = [colors.get(p, '#888888') for p in POSITION_ORDER]
        short_labels = [p.replace('inv_noble_', 'inv').replace('noble_', 'n').replace('boundary', 'bnd').replace('attractor', 'att') for p in POSITION_ORDER]

        bars = ax.bar(range(len(vals)), vals, color=bar_colors, edgecolor='none', alpha=0.8)

        # Significance markers
        if 'perm_p' in band_data.columns:
            for i, (v, p_val) in enumerate(zip(vals, band_data['perm_p'].values)):
                if not np.isnan(p_val) and p_val < 0.05:
                    ax.text(i, v + (1 if v >= 0 else -3), '*',
                           ha='center', fontsize=12, fontweight='bold')

        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xticks(range(len(short_labels)))
        ax.set_xticklabels(short_labels, rotation=90, fontsize=7)

        f_lo, f_hi = ANALYSIS_BANDS[band_name]
        n = int(band_data['n_peaks_band'].iloc[0]) if len(band_data) > 0 else 0
        ax.set_title(f'{band_name.replace("_", " ").title()}\n({f_lo}–{f_hi} Hz, n={n:,})',
                    fontsize=10, fontweight='bold')

    axes[0].set_ylabel('Enrichment (%)', fontsize=11, fontweight='bold')

    if title is None:
        title = 'Per-Position Enrichment by Band (φ-lattice)'
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Bar chart saved: {output_path}")


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Per-band, per-position enrichment analysis for φ-lattice'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Path to peaks CSV (with frequency column)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: same as input)')
    parser.add_argument('--window', type=float, default=0.05,
                        help='Half-window width in lattice coords (default: 0.05)')
    parser.add_argument('--n-perm', type=int, default=1000,
                        help='Number of phase-rotation permutations (default: 1000)')
    parser.add_argument('--n-boot', type=int, default=1000,
                        help='Number of bootstrap resamples (default: 1000)')
    parser.add_argument('--overlay', action='store_true',
                        help='Generate logphi overlay chart')
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.dirname(args.input)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("PER-BAND, PER-POSITION ENRICHMENT ANALYSIS")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {output_dir}")
    print(f"Window: {args.window}")
    print(f"Permutations: {args.n_perm}")
    print(f"Bootstraps: {args.n_boot}")
    print(f"F0: {F0} Hz")

    # Load peaks
    t0 = time.time()
    df = pd.read_csv(args.input)
    freq_col = 'frequency' if 'frequency' in df.columns else 'freq'
    freqs = df[freq_col].dropna().values.astype(float)
    freqs = freqs[(freqs >= 1.0) & (freqs <= 50.0)]
    print(f"\nLoaded {len(freqs):,} peaks ({time.time()-t0:.1f}s)")

    # Step 1: Observed enrichment
    print("\n--- Computing observed enrichment ---")
    t1 = time.time()
    results = compute_band_position_enrichment(freqs, F0, window=args.window)
    print(f"  Done ({time.time()-t1:.1f}s)")

    # Step 2: Permutation test
    print(f"\n--- Phase-rotation permutation test ({args.n_perm} perms) ---")
    t2 = time.time()
    results = permutation_test_band_position(freqs, F0, window=args.window, n_perm=args.n_perm)
    print(f"  Done ({time.time()-t2:.1f}s)")

    # Step 3: FDR correction
    results['significant'] = fdr_correct(results['perm_p'].values)
    n_sig = results['significant'].sum()
    print(f"  FDR significant: {n_sig}/{len(results)} cells")

    # Step 4: Bootstrap CIs
    print(f"\n--- Bootstrap CIs ({args.n_boot} resamples) ---")
    t3 = time.time()
    ci_data = bootstrap_band_position_ci(freqs, F0, window=args.window, n_boot=args.n_boot)
    ci_lower = []
    ci_upper = []
    for _, row in results.iterrows():
        key = (row['band'], row['position'])
        if key in ci_data:
            ci_lower.append(ci_data[key][0])
            ci_upper.append(ci_data[key][1])
        else:
            ci_lower.append(np.nan)
            ci_upper.append(np.nan)
    results['ci_lower'] = ci_lower
    results['ci_upper'] = ci_upper
    print(f"  Done ({time.time()-t3:.1f}s)")

    # Save CSV
    csv_path = os.path.join(output_dir, 'band_position_enrichment.csv')
    results.to_csv(csv_path, index=False)
    print(f"\n  CSV saved: {csv_path}")

    # Print summary table
    print("\n--- ENRICHMENT SUMMARY ---")
    for band_name in BAND_ORDER:
        band_data = results[results['band'] == band_name]
        n_band = int(band_data['n_peaks_band'].iloc[0]) if len(band_data) > 0 else 0
        print(f"\n  {band_name.upper()} ({n_band:,} peaks):")
        for _, row in band_data.iterrows():
            sig = '*' if row.get('significant', False) else ' '
            e = row['enrichment_pct']
            if pd.isna(e):
                print(f"    {row['position']:>15s}:     N/A")
            else:
                p = row.get('perm_p', np.nan)
                print(f"    {row['position']:>15s}: {e:+7.1f}% {sig} (p={p:.3f}, n={int(row['n_peaks_window'])})")

    # Generate visualizations
    print("\n--- Generating visualizations ---")

    heatmap_path = os.path.join(output_dir, 'band_position_heatmap.png')
    plot_band_position_heatmap(results, heatmap_path)

    bar_path = os.path.join(output_dir, 'band_position_bars.png')
    plot_band_position_bar(results, bar_path)

    total_time = time.time() - t0
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)")


if __name__ == '__main__':
    main()

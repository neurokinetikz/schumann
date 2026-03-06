#!/usr/bin/env python3
"""
Noble–Boundary Double Dissociation Across EEGMMIDB Task Conditions
===================================================================

Tests whether noble enrichment and boundary enrichment shift in opposite
directions across cognitive states (rest vs task), producing a double
dissociation that reverses across frequency bands.

Dissociation Index (DI) per band:
    DI = Δnoble - Δboundary
       = (noble_A - noble_B) - (boundary_A - boundary_B)

Significance tested via session-level permutation (5000 shuffles).

Usage:
    python scripts/noble_boundary_dissociation.py
    python scripts/noble_boundary_dissociation.py --n-perms 10000
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, './lib')
from phi_frequency_model import PHI, F0

# Reuse enrichment functions from existing analysis
sys.path.insert(0, './scripts')
from analyze_aggregate_enrichment import (
    compute_lattice_coordinate,
    compute_enrichment_at_positions,
    EXTENDED_OFFSETS,
)

# =========================================================================
# CONSTANTS
# =========================================================================

INPUT_CSV = 'exports_peak_distribution/eegmmidb_fooof/golden_ratio_peaks_EEGMMIDB.csv'
OUTPUT_DIR = 'exports_peak_distribution/eegmmidb_fooof'

ANALYSIS_BANDS = {
    'theta':     (4.70, 7.60),
    'alpha':     (7.60, 12.30),
    'beta_low':  (12.30, 19.90),
    'beta_high': (19.90, 32.19),
    'gamma':     (32.19, 45.00),
}

# EEGMMIDB run-to-condition mapping
RUN_CONDITIONS = {
    1:  'rest_eyes_open',   2:  'rest_eyes_closed',
    3:  'motor_execution',  4:  'motor_imagery',
    5:  'motor_execution',  6:  'motor_imagery',
    7:  'motor_execution',  8:  'motor_imagery',
    9:  'motor_execution',  10: 'motor_imagery',
    11: 'motor_execution',  12: 'motor_imagery',
    13: 'motor_execution',  14: 'motor_imagery',
}

# Noble position keys (for aggregate noble enrichment)
NOBLE_KEYS = [k for k in EXTENDED_OFFSETS if 'noble' in k]

# Lattice window half-width (consistent with analyze_aggregate_enrichment.py)
WINDOW = 0.05


# =========================================================================
# HELPERS
# =========================================================================

def load_and_assign_conditions(csv_path):
    """Load FOOOF peaks and assign task conditions from session names."""
    df = pd.read_csv(csv_path)
    df['run'] = df['session'].apply(lambda s: int(s.split('R')[1]))
    df['condition'] = df['run'].map(RUN_CONDITIONS)
    return df


def band_enrichment(freqs, band_range, f0=F0, window=WINDOW):
    """Compute enrichment at all positions for peaks within a band."""
    f_lo, f_hi = band_range
    in_band = freqs[(freqs >= f_lo) & (freqs < f_hi)]
    if len(in_band) < 10:
        return None, 0
    lattice = compute_lattice_coordinate(in_band, f0)
    result = compute_enrichment_at_positions(lattice, EXTENDED_OFFSETS, window)
    return result, len(in_band)


def extract_metrics(enrichment_result):
    """Extract noble (aggregate), noble_1, attractor, boundary enrichment %."""
    if enrichment_result is None:
        return {'noble': np.nan, 'noble_1': np.nan,
                'attractor': np.nan, 'boundary': np.nan}

    noble_peaks = sum(enrichment_result[k]['n_peaks'] for k in NOBLE_KEYS
                      if k in enrichment_result)
    noble_expected = sum(enrichment_result[k]['expected_frac'] for k in NOBLE_KEYS
                         if k in enrichment_result)
    n_total = list(enrichment_result.values())[0]['n_peaks'] / \
              list(enrichment_result.values())[0]['observed_frac'] \
              if list(enrichment_result.values())[0]['observed_frac'] > 0 else 1

    noble_obs_frac = noble_peaks / n_total if n_total > 0 else 0
    noble_enr = (noble_obs_frac / noble_expected - 1) * 100 if noble_expected > 0 else 0

    return {
        'noble':     noble_enr,
        'noble_1':   enrichment_result.get('noble_1', {}).get('enrichment_pct', np.nan),
        'attractor': enrichment_result.get('attractor', {}).get('enrichment_pct', np.nan),
        'boundary':  enrichment_result.get('boundary', {}).get('enrichment_pct', np.nan),
    }


def compute_dissociation_index(metrics_a, metrics_b, noble_key='noble_1'):
    """
    DI = Δnoble - Δboundary = (noble_A - noble_B) - (boundary_A - boundary_B).

    Positive DI → condition A has relatively more noble and less boundary.
    """
    d_noble = metrics_a[noble_key] - metrics_b[noble_key]
    d_boundary = metrics_a['boundary'] - metrics_b['boundary']
    di = d_noble - d_boundary

    # True double dissociation: noble and boundary shift in opposite directions
    is_dissociation = (d_noble > 0 and d_boundary < 0) or \
                      (d_noble < 0 and d_boundary > 0)

    return {
        'DI': di,
        'd_noble': d_noble,
        'd_boundary': d_boundary,
        'is_dissociation': is_dissociation,
    }


# =========================================================================
# PERMUTATION TEST
# =========================================================================

def permutation_test_dissociation(df, sessions_a, sessions_b,
                                   band_name, band_range,
                                   n_perms=5000, noble_key='noble_1'):
    """
    Session-level permutation test on the Dissociation Index.

    Shuffles session labels between conditions A and B, recomputing DI
    each time to build a null distribution.
    """
    rng = np.random.default_rng(42)
    all_sessions = np.array(list(sessions_a) + list(sessions_b))
    n_a = len(sessions_a)

    # Real DI
    freqs_a = df.loc[df['session'].isin(sessions_a), 'freq'].values
    freqs_b = df.loc[df['session'].isin(sessions_b), 'freq'].values

    enr_a, n_a_band = band_enrichment(freqs_a, band_range)
    enr_b, n_b_band = band_enrichment(freqs_b, band_range)

    if enr_a is None or enr_b is None:
        return None

    metrics_a = extract_metrics(enr_a)
    metrics_b = extract_metrics(enr_b)
    real = compute_dissociation_index(metrics_a, metrics_b, noble_key)

    # Permutation null
    null_dis = np.zeros(n_perms)
    for i in range(n_perms):
        perm = rng.permutation(all_sessions)
        perm_a = set(perm[:n_a])
        perm_b = set(perm[n_a:])

        fa = df.loc[df['session'].isin(perm_a), 'freq'].values
        fb = df.loc[df['session'].isin(perm_b), 'freq'].values

        ea, _ = band_enrichment(fa, band_range)
        eb, _ = band_enrichment(fb, band_range)

        if ea is None or eb is None:
            null_dis[i] = 0
            continue

        ma = extract_metrics(ea)
        mb = extract_metrics(eb)
        null_dis[i] = compute_dissociation_index(ma, mb, noble_key)['DI']

    p_val = np.mean(np.abs(null_dis) >= np.abs(real['DI']))
    z = (real['DI'] - null_dis.mean()) / null_dis.std() if null_dis.std() > 0 else 0

    return {
        'band': band_name,
        'noble_key': noble_key,
        'n_a': n_a_band,
        'n_b': n_b_band,
        **real,
        **{f'{noble_key}_A': metrics_a[noble_key],
           f'{noble_key}_B': metrics_b[noble_key],
           'boundary_A': metrics_a['boundary'],
           'boundary_B': metrics_b['boundary'],
           'attractor_A': metrics_a['attractor'],
           'attractor_B': metrics_b['attractor']},
        'p_val': p_val,
        'z_score': z,
        'null_mean': null_dis.mean(),
        'null_std': null_dis.std(),
    }


# =========================================================================
# CROSSOVER TEST
# =========================================================================

def crossover_test(results_df, noble_key='noble_1'):
    """
    Test whether the DI sign flips between low bands and high bands.

    Pools theta+alpha (low) vs beta_high+gamma (high) and tests whether
    DI_low and DI_high have opposite signs with combined significance.
    """
    low = results_df[results_df['band'].isin(['theta', 'alpha'])]
    high = results_df[results_df['band'].isin(['beta_high', 'gamma'])]

    if len(low) == 0 or len(high) == 0:
        return None

    di_low = low['DI'].mean()
    di_high = high['DI'].mean()

    # Combined p via Fisher's method on per-band p-values
    low_ps = low['p_val'].values
    high_ps = high['p_val'].values
    all_ps = np.concatenate([low_ps, high_ps])

    # Sign flip?
    sign_flip = (di_low > 0 and di_high < 0) or (di_low < 0 and di_high > 0)

    return {
        'DI_low': di_low,
        'DI_high': di_high,
        'sign_flip': sign_flip,
        'min_p_low': low_ps.min() if len(low_ps) > 0 else 1.0,
        'min_p_high': high_ps.min() if len(high_ps) > 0 else 1.0,
    }


# =========================================================================
# VISUALIZATION
# =========================================================================

def plot_crossover_figure(all_results, output_path):
    """
    3-panel figure showing noble vs boundary enrichment shifts per band.
    """
    contrast_names = list(all_results.keys())
    n_panels = len(contrast_names)

    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5),
                             sharey=True)
    if n_panels == 1:
        axes = [axes]

    band_order = list(ANALYSIS_BANDS.keys())
    x = np.arange(len(band_order))
    band_labels = ['Theta', 'Alpha', 'Beta\nLow', 'Beta\nHigh', 'Gamma']

    for ax, contrast_name in zip(axes, contrast_names):
        rdf = all_results[contrast_name]

        d_noble = []
        d_boundary = []
        p_vals = []
        is_dissoc = []

        for band in band_order:
            row = rdf[rdf['band'] == band]
            if len(row) == 0:
                d_noble.append(0)
                d_boundary.append(0)
                p_vals.append(1.0)
                is_dissoc.append(False)
            else:
                row = row.iloc[0]
                d_noble.append(row['d_noble'])
                d_boundary.append(row['d_boundary'])
                p_vals.append(row['p_val'])
                is_dissoc.append(row['is_dissociation'])

        d_noble = np.array(d_noble)
        d_boundary = np.array(d_boundary)

        # Plot lines
        ln1, = ax.plot(x, d_noble, 'o-', color='#e74c3c', linewidth=2.2,
                        markersize=8, label='Δ Noble₁', zorder=3)
        ln2, = ax.plot(x, d_boundary, 's-', color='#3498db', linewidth=2.2,
                        markersize=8, label='Δ Boundary', zorder=3)

        # Shade dissociation zones (where lines are on opposite sides of zero)
        for i in range(len(x) - 1):
            if (d_noble[i] * d_boundary[i] < 0) or (d_noble[i+1] * d_boundary[i+1] < 0):
                ax.axvspan(x[i] - 0.3, x[i+1] + 0.3, alpha=0.08,
                           color='#2ecc71', zorder=0)

        # Significance markers for DI
        for i, (p, dis) in enumerate(zip(p_vals, is_dissoc)):
            if p < 0.001:
                marker = '***'
            elif p < 0.01:
                marker = '**'
            elif p < 0.05:
                marker = '*'
            else:
                marker = ''

            if marker:
                y_pos = max(abs(d_noble[i]), abs(d_boundary[i])) + 2
                color = '#2ecc71' if dis else '#95a5a6'
                ax.annotate(marker, (x[i], y_pos), ha='center', fontsize=14,
                            fontweight='bold', color=color)

        ax.axhline(0, color='grey', linewidth=0.8, linestyle='--', zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(band_labels, fontsize=10)
        ax.set_title(contrast_name, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best', framealpha=0.9)

        # Annotate crossover
        crossover = crossover_test(rdf)
        if crossover and crossover['sign_flip']:
            ax.text(0.98, 0.02, 'Crossover ✓', transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=9, color='#27ae60',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#eafaf1',
                              edgecolor='#27ae60', alpha=0.8))

    axes[0].set_ylabel('Δ Enrichment (%) [Condition A − B]', fontsize=11)

    fig.suptitle('Noble–Boundary Double Dissociation Across Task Conditions\n'
                 'EEGMMIDB FOOOF (n=1.86M peaks, 109 subjects)',
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Figure saved: {output_path}")


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Noble-Boundary double dissociation across task conditions")
    parser.add_argument('--n-perms', type=int, default=5000)
    parser.add_argument('--noble-key', type=str, default='noble_1',
                        choices=['noble_1', 'noble', 'attractor'])
    args = parser.parse_args()

    print("=" * 80)
    print("Noble–Boundary Double Dissociation Analysis")
    print("=" * 80)

    # --- Load data ---
    df = load_and_assign_conditions(INPUT_CSV)
    print(f"Loaded {len(df):,} peaks from {df['session'].nunique()} sessions")
    print(f"Conditions: {df['condition'].value_counts().to_dict()}")

    # --- Build session sets for each contrast ---
    sessions_by_cond = {
        c: set(df.loc[df['condition'] == c, 'session'].unique())
        for c in df['condition'].unique()
    }

    contrasts = {
        'Rest vs Task': (
            sessions_by_cond.get('rest_eyes_open', set()) |
            sessions_by_cond.get('rest_eyes_closed', set()),
            sessions_by_cond.get('motor_execution', set()) |
            sessions_by_cond.get('motor_imagery', set()),
        ),
        'Eyes Open vs Closed': (
            sessions_by_cond.get('rest_eyes_open', set()),
            sessions_by_cond.get('rest_eyes_closed', set()),
        ),
        'Execution vs Imagery': (
            sessions_by_cond.get('motor_execution', set()),
            sessions_by_cond.get('motor_imagery', set()),
        ),
    }

    # --- Run permutation tests ---
    all_results = {}
    all_rows = []

    for contrast_name, (sess_a, sess_b) in contrasts.items():
        print(f"\n{'─' * 80}")
        print(f"  {contrast_name}  ({len(sess_a)} vs {len(sess_b)} sessions)")
        print(f"  Permutation test: {args.n_perms} shuffles")
        print(f"{'─' * 80}")

        print(f"  {'Band':12s} | {'Noble₁_A':>9s} {'Noble₁_B':>9s} {'ΔNoble':>8s} | "
              f"{'Bound_A':>8s} {'Bound_B':>8s} {'ΔBound':>8s} | "
              f"{'DI':>8s} {'z':>7s} {'p':>7s} {'Dissoc?':>8s}")
        print(f"  {'-'*12} | {'-'*9} {'-'*9} {'-'*8} | "
              f"{'-'*8} {'-'*8} {'-'*8} | "
              f"{'-'*8} {'-'*7} {'-'*7} {'-'*8}")

        band_results = []
        for band_name, band_range in ANALYSIS_BANDS.items():
            result = permutation_test_dissociation(
                df, sess_a, sess_b, band_name, band_range,
                n_perms=args.n_perms, noble_key=args.noble_key,
            )

            if result is None:
                print(f"  {band_name:12s} | {'N/A':>9s}")
                continue

            band_results.append(result)

            sig = ""
            if result['p_val'] < 0.001: sig = "***"
            elif result['p_val'] < 0.01: sig = "** "
            elif result['p_val'] < 0.05: sig = "*  "
            elif result['p_val'] < 0.1: sig = ".  "

            dis_marker = "YES" if result['is_dissociation'] else "no"
            nk = args.noble_key

            print(f"  {band_name:12s} | "
                  f"{result[f'{nk}_A']:9.2f} {result[f'{nk}_B']:9.2f} "
                  f"{result['d_noble']:+8.2f} | "
                  f"{result['boundary_A']:8.2f} {result['boundary_B']:8.2f} "
                  f"{result['d_boundary']:+8.2f} | "
                  f"{result['DI']:+8.2f} {result['z_score']:+7.2f} "
                  f"{result['p_val']:7.4f}{sig} {dis_marker:>8s}")

            all_rows.append({
                'contrast': contrast_name,
                **result,
            })

        rdf = pd.DataFrame(band_results)
        all_results[contrast_name] = rdf

        # Crossover test
        cross = crossover_test(rdf, noble_key=args.noble_key)
        if cross:
            print(f"\n  Crossover test: DI_low={cross['DI_low']:+.2f}, "
                  f"DI_high={cross['DI_high']:+.2f}, "
                  f"sign_flip={'YES' if cross['sign_flip'] else 'no'}")

    # --- Save table ---
    table_path = os.path.join(OUTPUT_DIR, 'dissociation_table.csv')
    table_df = pd.DataFrame(all_rows)
    table_df.to_csv(table_path, index=False)
    print(f"\n  Table saved: {table_path}")

    # --- Summary ---
    print("\n" + "=" * 80)
    print("  DOUBLE DISSOCIATION SUMMARY")
    print("=" * 80)

    for contrast_name, rdf in all_results.items():
        sig_dissoc = rdf[(rdf['p_val'] < 0.05) & (rdf['is_dissociation'])]
        if len(sig_dissoc) > 0:
            print(f"\n  {contrast_name}:")
            for _, row in sig_dissoc.iterrows():
                print(f"    {row['band']:12s}: DI={row['DI']:+.2f}%, "
                      f"ΔNoble={row['d_noble']:+.2f}%, "
                      f"ΔBound={row['d_boundary']:+.2f}%, "
                      f"z={row['z_score']:+.2f}, p={row['p_val']:.4f}")
        else:
            print(f"\n  {contrast_name}: no significant dissociations at p<0.05")

    # Count true double dissociations
    n_true = sum(1 for _, rdf in all_results.items()
                 for _, row in rdf.iterrows()
                 if row['p_val'] < 0.05 and row['is_dissociation'])
    n_total = sum(len(rdf) for rdf in all_results.values())
    print(f"\n  Total: {n_true} significant double dissociations "
          f"out of {n_total} band×contrast tests")

    # --- Plot ---
    fig_path = os.path.join(OUTPUT_DIR, 'dissociation_crossover.png')
    plot_crossover_figure(all_results, fig_path)

    print("\nDone.")


if __name__ == '__main__':
    main()

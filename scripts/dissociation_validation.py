#!/usr/bin/env python3
"""
Rigorous Validation of Noble–Boundary Dissociation Claims
==========================================================

Addresses 6 critical concerns raised in peer review:
  1. FDR correction for multiple comparisons
  2. Within-octave density peak fitting (noble_1 vs attractor)
  3. 2×2 factorial decomposition of gamma (EO/task confound)
  4. Spectral resolution caveat for alpha IAF
  5. Null distribution diagnostics (symmetry, normality)
  6. Summary figure

Usage:
    python scripts/dissociation_validation.py
    python scripts/dissociation_validation.py --n-perms 10000
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, shapiro, skew, kurtosis
from scipy.optimize import minimize_scalar

sys.path.insert(0, './lib')
sys.path.insert(0, './scripts')

from phi_frequency_model import PHI, F0
from analyze_aggregate_enrichment import (
    compute_lattice_coordinate, compute_enrichment_at_positions, EXTENDED_OFFSETS
)
from noble_boundary_dissociation import (
    band_enrichment, extract_metrics, compute_dissociation_index,
    load_and_assign_conditions, permutation_test_dissociation,
    ANALYSIS_BANDS, RUN_CONDITIONS, WINDOW,
)

# =========================================================================
# CONSTANTS
# =========================================================================

INPUT_CSV = 'exports_peak_distribution/eegmmidb_fooof/golden_ratio_peaks_EEGMMIDB.csv'
DI_TABLE = 'exports_peak_distribution/eegmmidb_fooof/dissociation_table.csv'
OUTPUT_DIR = 'exports_peak_distribution/eegmmidb_fooof'

NOBLE_KEYS = [k for k in EXTENDED_OFFSETS if 'noble' in k]

# PSD parameters — must match run_eegmmidb_full_pipeline.py
# (nperseg_sec=4.0 at fs=160 → nperseg=640, Δf=0.25 Hz)
SFREQ = 160.0
NPERSEG = 640
FREQ_RESOLUTION = SFREQ / NPERSEG  # 0.25 Hz


# =========================================================================
# ANALYSIS 1: FDR CORRECTION
# =========================================================================

def fdr_correction(p_values, q=0.05):
    """
    Benjamini-Hochberg FDR correction.

    Falls back to manual implementation if statsmodels unavailable.
    """
    try:
        from statsmodels.stats.multitest import multipletests
        reject, p_adj, _, _ = multipletests(p_values, alpha=q, method='fdr_bh')
        return reject, p_adj
    except ImportError:
        # Manual BH procedure
        n = len(p_values)
        sorted_idx = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_idx]
        thresholds = np.arange(1, n + 1) / n * q

        p_adj = np.zeros(n)
        reject = np.zeros(n, dtype=bool)

        # Adjusted p-values (step-up)
        cummin = sorted_p[-1] * n / n
        p_adj_sorted = np.zeros(n)
        for i in range(n - 1, -1, -1):
            raw = sorted_p[i] * n / (i + 1)
            cummin = min(cummin, raw)
            p_adj_sorted[i] = min(cummin, 1.0)

        p_adj[sorted_idx] = p_adj_sorted
        reject = p_adj < q

        return reject, p_adj


def run_fdr_analysis():
    """Apply FDR correction to DI permutation p-values."""
    print("=" * 80)
    print("  ANALYSIS 1: FDR CORRECTION (Benjamini-Hochberg, q=0.05)")
    print("=" * 80)

    di_df = pd.read_csv(DI_TABLE)
    p_values = di_df['p_val'].values
    reject, p_adj = fdr_correction(p_values, q=0.05)

    di_df['p_fdr'] = p_adj
    di_df['reject_fdr'] = reject

    print(f"\n  {'Contrast':25s} {'Band':12s} {'p_uncorr':>10s} {'p_FDR':>10s} {'Survives':>10s}")
    print(f"  {'-'*25} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")

    for _, row in di_df.sort_values('p_val').iterrows():
        survives = "YES" if row['reject_fdr'] else "no"
        sig = ""
        if row['p_val'] < 0.001: sig = "***"
        elif row['p_val'] < 0.01: sig = "** "
        elif row['p_val'] < 0.05: sig = "*  "

        print(f"  {row['contrast']:25s} {row['band']:12s} "
              f"{row['p_val']:10.4f}{sig} {row['p_fdr']:10.4f} {survives:>10s}")

    n_survive = reject.sum()
    print(f"\n  {n_survive} of {len(p_values)} tests survive FDR correction")

    fdr_path = os.path.join(OUTPUT_DIR, 'validation_fdr_table.csv')
    di_df.to_csv(fdr_path, index=False)
    print(f"  Saved: {fdr_path}")

    return di_df


# =========================================================================
# ANALYSIS 2: WITHIN-OCTAVE DENSITY PEAK FITTING
# =========================================================================

def find_kde_mode(lattice_coords, bw=0.02):
    """Find the mode (peak) of the KDE on [0, 1)."""
    if len(lattice_coords) < 20:
        return np.nan

    kde = gaussian_kde(lattice_coords, bw_method=bw)
    result = minimize_scalar(lambda x: -kde(x)[0], bounds=(0.01, 0.99),
                             method='bounded')
    return result.x


def bootstrap_mode(lattice_coords, n_boot=1000, bw=0.02, seed=42):
    """Bootstrap the KDE mode location."""
    rng = np.random.default_rng(seed)
    modes = []
    n = len(lattice_coords)

    for _ in range(n_boot):
        boot = lattice_coords[rng.choice(n, size=n, replace=True)]
        mode = find_kde_mode(boot, bw=bw)
        if np.isfinite(mode):
            modes.append(mode)

    modes = np.array(modes)
    return {
        'mode_mean': modes.mean(),
        'mode_median': np.median(modes),
        'mode_std': modes.std(),
        'ci_lower': np.percentile(modes, 2.5),
        'ci_upper': np.percentile(modes, 97.5),
        'n_boot': len(modes),
    }


def run_mode_analysis(df):
    """Test whether within-octave density peaks at noble_1 or attractor."""
    print("\n" + "=" * 80)
    print("  ANALYSIS 2: WITHIN-OCTAVE KDE MODE LOCATION")
    print(f"  Testing: Is the density peak at noble_1 (0.618) or attractor (0.500)?")
    print("=" * 80)

    rows = []

    for band_name, (f_lo, f_hi) in ANALYSIS_BANDS.items():
        freqs = df['freq'].values
        in_band = freqs[(freqs >= f_lo) & (freqs < f_hi)]

        if len(in_band) < 50:
            continue

        lattice = compute_lattice_coordinate(in_band, F0)
        mode = find_kde_mode(lattice)
        boot = bootstrap_mode(lattice, n_boot=1000)

        includes_attractor = boot['ci_lower'] <= 0.500 <= boot['ci_upper']
        includes_noble1 = boot['ci_lower'] <= 0.618 <= boot['ci_upper']

        dist_to_attractor = abs(mode - 0.500)
        dist_to_noble1 = abs(mode - 0.618)
        closer_to = 'attractor' if dist_to_attractor < dist_to_noble1 else 'noble_1'

        row = {
            'band': band_name, 'n_peaks': len(in_band),
            'mode': mode,
            **boot,
            'includes_attractor': includes_attractor,
            'includes_noble1': includes_noble1,
            'dist_to_attractor': dist_to_attractor,
            'dist_to_noble1': dist_to_noble1,
            'closer_to': closer_to,
        }
        rows.append(row)

        verdict = []
        if includes_attractor:
            verdict.append("CI includes 0.500")
        if includes_noble1:
            verdict.append("CI includes 0.618")
        if not includes_attractor and not includes_noble1:
            verdict.append("CI excludes both")

        print(f"\n  {band_name.upper()} ({len(in_band):,} peaks):")
        print(f"    Mode = {mode:.4f}  [95% CI: {boot['ci_lower']:.4f} – {boot['ci_upper']:.4f}]")
        print(f"    Distance to attractor (0.500): {dist_to_attractor:.4f}")
        print(f"    Distance to noble_1  (0.618): {dist_to_noble1:.4f}")
        print(f"    Closer to: {closer_to}")
        print(f"    {' | '.join(verdict)}")

    # Per-condition mode analysis
    print(f"\n  --- Per-condition mode (Rest vs Task) ---")
    for cond_name, cond_vals in [('rest', ['rest_eyes_open', 'rest_eyes_closed']),
                                  ('task', ['motor_execution', 'motor_imagery'])]:
        cond_freqs = df.loc[df['condition'].isin(cond_vals), 'freq'].values
        for band_name, (f_lo, f_hi) in ANALYSIS_BANDS.items():
            in_band = cond_freqs[(cond_freqs >= f_lo) & (cond_freqs < f_hi)]
            if len(in_band) < 50:
                continue
            lattice = compute_lattice_coordinate(in_band, F0)
            mode = find_kde_mode(lattice)
            rows.append({
                'band': f'{band_name}_{cond_name}', 'n_peaks': len(in_band),
                'mode': mode, 'mode_mean': np.nan, 'mode_median': np.nan,
                'mode_std': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan,
                'n_boot': 0,
                'includes_attractor': np.nan, 'includes_noble1': np.nan,
                'dist_to_attractor': abs(mode - 0.500),
                'dist_to_noble1': abs(mode - 0.618),
                'closer_to': 'attractor' if abs(mode - 0.500) < abs(mode - 0.618) else 'noble_1',
            })
            print(f"    {band_name:12s} [{cond_name:4s}]: mode={mode:.4f}, "
                  f"closer to {'attractor' if abs(mode - 0.500) < abs(mode - 0.618) else 'noble_1'}")

    mode_df = pd.DataFrame(rows)
    mode_path = os.path.join(OUTPUT_DIR, 'validation_mode_locations.csv')
    mode_df.to_csv(mode_path, index=False)
    print(f"\n  Saved: {mode_path}")

    return mode_df


# =========================================================================
# ANALYSIS 3: GAMMA 3-CELL DECOMPOSITION
# =========================================================================

def run_gamma_decomposition(df, n_perms=5000):
    """
    Decompose gamma DI into EO/task components.

    3-cell design: rest-EO (R01), rest-EC (R02), task (R03-R14).
    Tests: (a) rest-EO vs task (motor effect, matched visual)
           (b) rest-EO vs rest-EC (visual effect, no motor)
    """
    print("\n" + "=" * 80)
    print("  ANALYSIS 3: GAMMA 3-CELL DECOMPOSITION (EO/task confound)")
    print("=" * 80)

    sessions_by_cond = {
        c: set(df.loc[df['condition'] == c, 'session'].unique())
        for c in df['condition'].unique()
    }

    cells = {
        'rest_EO': sessions_by_cond.get('rest_eyes_open', set()),
        'rest_EC': sessions_by_cond.get('rest_eyes_closed', set()),
        'task':    sessions_by_cond.get('motor_execution', set()) |
                   sessions_by_cond.get('motor_imagery', set()),
    }

    gamma_range = ANALYSIS_BANDS['gamma']

    comparisons = [
        ('rest-EO vs Task (motor effect, matched EO)', cells['rest_EO'], cells['task']),
        ('rest-EO vs rest-EC (visual effect, no motor)', cells['rest_EO'], cells['rest_EC']),
        ('rest-EC vs Task (motor + visual confound)', cells['rest_EC'], cells['task']),
    ]

    results = []

    for label, sess_a, sess_b in comparisons:
        result = permutation_test_dissociation(
            df, sess_a, sess_b, 'gamma', gamma_range,
            n_perms=n_perms, noble_key='noble_1',
        )

        if result is None:
            print(f"\n  {label}: N/A")
            continue

        sig = ""
        if result['p_val'] < 0.001: sig = "***"
        elif result['p_val'] < 0.01: sig = "** "
        elif result['p_val'] < 0.05: sig = "*  "

        dis = "YES" if result['is_dissociation'] else "no"

        print(f"\n  {label}")
        print(f"    A: {len(sess_a)} sessions | B: {len(sess_b)} sessions")
        print(f"    Noble₁_A={result['noble_1_A']:+.1f}%  Noble₁_B={result['noble_1_B']:+.1f}%  "
              f"ΔNoble={result['d_noble']:+.1f}%")
        print(f"    Bound_A={result['boundary_A']:+.1f}%   Bound_B={result['boundary_B']:+.1f}%   "
              f"ΔBound={result['d_boundary']:+.1f}%")
        print(f"    DI={result['DI']:+.1f}%  z={result['z_score']:+.2f}  "
              f"p={result['p_val']:.4f}{sig}  Dissociation: {dis}")

        results.append({'comparison': label, **result})

    # Interpretation
    print(f"\n  --- Interpretation ---")
    if len(results) >= 2:
        motor_p = results[0]['p_val']
        visual_p = results[1]['p_val']

        if motor_p < 0.05 and visual_p >= 0.05:
            print("  -> Motor effect is real (rest-EO vs task significant, visual not)")
        elif motor_p >= 0.05 and visual_p < 0.05:
            print("  -> Gamma signal is VISUAL, not motor (visual significant, motor not)")
        elif motor_p < 0.05 and visual_p < 0.05:
            print("  -> Both motor and visual contribute independently")
        else:
            print("  -> Neither reaches significance in isolation — original effect is fragile")

    return results


# =========================================================================
# ANALYSIS 4: SPECTRAL RESOLUTION CAVEAT
# =========================================================================

def run_spectral_resolution_analysis(df):
    """Document spectral resolution limitations for alpha claims."""
    print("\n" + "=" * 80)
    print("  ANALYSIS 4: SPECTRAL RESOLUTION CAVEAT")
    print(f"  SFREQ={SFREQ} Hz, nperseg={NPERSEG} → Δf={FREQ_RESOLUTION:.4f} Hz")
    print("=" * 80)

    alpha_freqs = df.loc[(df['freq'] >= 7.6) & (df['freq'] < 12.3), 'freq'].values

    print(f"\n  Alpha peaks: {len(alpha_freqs):,}")
    print(f"  Mean: {alpha_freqs.mean():.3f} Hz")
    print(f"  Median: {np.median(alpha_freqs):.3f} Hz")

    # KDE mode of alpha frequencies
    if len(alpha_freqs) > 100:
        kde = gaussian_kde(alpha_freqs, bw_method=0.1)
        x = np.linspace(7.6, 12.3, 1000)
        alpha_mode = x[np.argmax(kde(x))]
        print(f"  KDE mode: {alpha_mode:.3f} Hz")

    # Key frequency comparisons
    targets = [
        ('noble_1', 10.23),
        ('attractor', 9.67),
        ('round_10', 10.00),
    ]

    print(f"\n  Can the spectral resolution distinguish key frequencies?")
    for i, (name_a, freq_a) in enumerate(targets):
        for name_b, freq_b in targets[i+1:]:
            delta = abs(freq_a - freq_b)
            resolvable = "YES" if delta > FREQ_RESOLUTION else "NO"
            print(f"    {name_a} ({freq_a:.2f}) vs {name_b} ({freq_b:.2f}): "
                  f"Δ={delta:.3f} Hz — {resolvable} (Δf={FREQ_RESOLUTION:.3f})")

    # Overlap analysis: peaks near 10.23 that are also near 10.0
    near_noble1 = alpha_freqs[(alpha_freqs >= 10.23 - FREQ_RESOLUTION) &
                               (alpha_freqs <= 10.23 + FREQ_RESOLUTION)]
    near_round10 = alpha_freqs[(alpha_freqs >= 10.0 - FREQ_RESOLUTION) &
                                (alpha_freqs <= 10.0 + FREQ_RESOLUTION)]
    overlap = np.intersect1d(
        np.where((alpha_freqs >= 10.23 - FREQ_RESOLUTION) &
                 (alpha_freqs <= 10.23 + FREQ_RESOLUTION)),
        np.where((alpha_freqs >= 10.0 - FREQ_RESOLUTION) &
                 (alpha_freqs <= 10.0 + FREQ_RESOLUTION))
    )

    overlap_frac = len(overlap) / max(len(near_noble1), 1)
    print(f"\n  Peaks within ±{FREQ_RESOLUTION:.3f} Hz of noble_1 (10.23): {len(near_noble1):,}")
    print(f"  Peaks within ±{FREQ_RESOLUTION:.3f} Hz of round 10.0:    {len(near_round10):,}")
    print(f"  Overlap fraction: {overlap_frac:.1%}")
    print(f"  -> {'CONFOUNDED' if overlap_frac > 0.5 else 'Partially distinguishable'}: "
          f"noble_1 (10.23 Hz) and round 10.0 Hz claims are "
          f"{'indistinguishable' if overlap_frac > 0.5 else 'partially separable'} "
          f"at this spectral resolution")

    # Check for PSD grid clustering
    psd_grid = np.arange(0, 80, FREQ_RESOLUTION)
    alpha_grid = psd_grid[(psd_grid >= 7.6) & (psd_grid < 12.3)]
    print(f"\n  PSD grid points in alpha: {len(alpha_grid)}")
    print(f"  Grid points near noble_1: "
          f"{alpha_grid[np.abs(alpha_grid - 10.23) < FREQ_RESOLUTION/2]}")
    print(f"  Grid points near attractor: "
          f"{alpha_grid[np.abs(alpha_grid - 9.67) < FREQ_RESOLUTION/2]}")


# =========================================================================
# ANALYSIS 5: NULL DISTRIBUTION DIAGNOSTICS
# =========================================================================

def run_null_diagnostics(df, n_perms=5000):
    """Check null distribution properties for all DI tests."""
    print("\n" + "=" * 80)
    print("  ANALYSIS 5: NULL DISTRIBUTION DIAGNOSTICS")
    print("=" * 80)

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
        'EO vs EC': (
            sessions_by_cond.get('rest_eyes_open', set()),
            sessions_by_cond.get('rest_eyes_closed', set()),
        ),
        'Exec vs Imagery': (
            sessions_by_cond.get('motor_execution', set()),
            sessions_by_cond.get('motor_imagery', set()),
        ),
    }

    rng = np.random.default_rng(42)
    diag_rows = []
    null_arrays = {}  # Store for plotting

    print(f"\n  {'Contrast':20s} {'Band':12s} | {'Mean':>8s} {'Median':>8s} {'Std':>7s} "
          f"{'Skew':>7s} {'Kurt':>7s} {'Shap_p':>8s} {'Symm?':>6s}")
    print(f"  {'-'*20} {'-'*12} | {'-'*8} {'-'*8} {'-'*7} "
          f"{'-'*7} {'-'*7} {'-'*8} {'-'*6}")

    for cname, (sess_a, sess_b) in contrasts.items():
        all_sessions = np.array(list(sess_a) + list(sess_b))
        n_a = len(sess_a)

        for band_name, band_range in ANALYSIS_BANDS.items():
            # Build null distribution
            null_dis = np.zeros(n_perms)
            for i in range(n_perms):
                perm = rng.permutation(all_sessions)
                perm_a = set(perm[:n_a])
                perm_b = set(perm[n_a:])

                fa = df.loc[df['session'].isin(perm_a), 'freq'].values
                fb = df.loc[df['session'].isin(perm_b), 'freq'].values

                ea, _ = band_enrichment(fa, band_range)
                eb, _ = band_enrichment(fb, band_range)

                if ea is not None and eb is not None:
                    ma = extract_metrics(ea)
                    mb = extract_metrics(eb)
                    null_dis[i] = compute_dissociation_index(ma, mb, 'noble_1')['DI']

            null_key = f"{cname}_{band_name}"
            null_arrays[null_key] = null_dis

            sk = skew(null_dis)
            ku = kurtosis(null_dis)
            # Shapiro-Wilk on subsample (max 5000)
            shap_sample = null_dis[:min(5000, len(null_dis))]
            _, shap_p = shapiro(shap_sample)
            symmetric = abs(sk) < 0.5

            diag_rows.append({
                'contrast': cname, 'band': band_name,
                'null_mean': null_dis.mean(), 'null_median': np.median(null_dis),
                'null_std': null_dis.std(),
                'skewness': sk, 'kurtosis': ku,
                'shapiro_p': shap_p,
                'symmetric': symmetric,
                'n_a': n_a, 'n_b': len(sess_b),
            })

            sym_marker = "OK" if symmetric else "SKEW"
            print(f"  {cname:20s} {band_name:12s} | "
                  f"{null_dis.mean():8.3f} {np.median(null_dis):8.3f} {null_dis.std():7.3f} "
                  f"{sk:+7.3f} {ku:+7.3f} {shap_p:8.4f} {sym_marker:>6s}")

    diag_df = pd.DataFrame(diag_rows)
    diag_path = os.path.join(OUTPUT_DIR, 'validation_null_diagnostics.csv')
    diag_df.to_csv(diag_path, index=False)
    print(f"\n  Saved: {diag_path}")

    # Summary
    n_skewed = (~diag_df['symmetric']).sum()
    print(f"\n  {n_skewed} of {len(diag_df)} null distributions show |skewness| > 0.5")
    if n_skewed > 0:
        skewed = diag_df[~diag_df['symmetric']]
        for _, row in skewed.iterrows():
            print(f"    {row['contrast']:20s} {row['band']:12s}: skew={row['skewness']:+.3f}")

    return diag_df, null_arrays


# =========================================================================
# ANALYSIS 6: SUMMARY FIGURE
# =========================================================================

def plot_validation_figure(fdr_df, mode_df, gamma_results, null_arrays, output_path):
    """4-panel validation figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Panel A: FDR-corrected DI significance ---
    ax = axes[0, 0]
    di_main = fdr_df[fdr_df['contrast'] == 'Rest vs Task'].copy()
    if len(di_main) > 0:
        bands = list(ANALYSIS_BANDS.keys())
        x = np.arange(len(bands))
        di_vals = []
        colors = []
        for b in bands:
            row = di_main[di_main['band'] == b]
            if len(row) > 0:
                di_vals.append(row.iloc[0]['DI'])
                if row.iloc[0]['reject_fdr']:
                    colors.append('#e74c3c')
                elif row.iloc[0]['p_val'] < 0.05:
                    colors.append('#f39c12')
                else:
                    colors.append('#95a5a6')
            else:
                di_vals.append(0)
                colors.append('#95a5a6')

        bars = ax.bar(x, di_vals, color=colors, edgecolor='white', linewidth=0.5)
        ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
        ax.set_xticks(x)
        ax.set_xticklabels(['Theta', 'Alpha', 'Beta\nLow', 'Beta\nHigh', 'Gamma'],
                           fontsize=9)
        ax.set_ylabel('Dissociation Index (%)', fontsize=10)
        ax.set_title('A. Rest vs Task DI\n(red=FDR sig, orange=uncorrected only, grey=n.s.)',
                      fontsize=10, fontweight='bold')

    # --- Panel B: Within-octave KDE mode per band ---
    ax = axes[0, 1]
    mode_main = mode_df[~mode_df['band'].str.contains('_')].copy()  # Exclude per-condition
    if len(mode_main) > 0:
        bands = mode_main['band'].values
        x = np.arange(len(bands))
        modes = mode_main['mode'].values
        ci_lo = mode_main['ci_lower'].values
        ci_hi = mode_main['ci_upper'].values

        ax.errorbar(x, modes, yerr=[modes - ci_lo, ci_hi - modes],
                     fmt='ko', capsize=5, capthick=1.5, markersize=8, linewidth=1.5)
        ax.axhline(0.500, color='#3498db', linewidth=1.5, linestyle='--',
                    label='Attractor (0.500)')
        ax.axhline(0.618, color='#e74c3c', linewidth=1.5, linestyle='--',
                    label='Noble₁ (0.618)')
        ax.set_xticks(x)
        ax.set_xticklabels([b.replace('_', '\n') for b in bands], fontsize=9)
        ax.set_ylabel('KDE Mode (lattice coordinate)', fontsize=10)
        ax.set_title('B. Within-Octave Density Peak Location\n(black dots = mode ± 95% CI)',
                      fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')
        ax.set_ylim(0.35, 0.75)

    # --- Panel C: Gamma 3-cell decomposition ---
    ax = axes[1, 0]
    if gamma_results:
        labels = [r['comparison'].split('(')[0].strip() for r in gamma_results]
        dis = [r['DI'] for r in gamma_results]
        p_vals = [r['p_val'] for r in gamma_results]

        x = np.arange(len(labels))
        colors = ['#e74c3c' if p < 0.05 else '#95a5a6' for p in p_vals]
        ax.barh(x, dis, color=colors, edgecolor='white', height=0.5)
        ax.axvline(0, color='grey', linewidth=0.8, linestyle='--')

        for i, (d, p) in enumerate(zip(dis, p_vals)):
            sig = ""
            if p < 0.01: sig = "**"
            elif p < 0.05: sig = "*"
            ax.text(d + (1 if d >= 0 else -1), i,
                    f'DI={d:+.1f}% p={p:.3f}{sig}', va='center', fontsize=8)

        ax.set_yticks(x)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Dissociation Index (%)', fontsize=10)
        ax.set_title('C. Gamma Decomposition\n(red = p<0.05, grey = n.s.)',
                      fontsize=10, fontweight='bold')

    # --- Panel D: Null distribution for gamma Rest-vs-Task ---
    ax = axes[1, 1]
    null_key = 'Rest vs Task_gamma'
    if null_key in null_arrays:
        null = null_arrays[null_key]

        # Real DI value from the FDR table
        gamma_row = fdr_df[(fdr_df['contrast'] == 'Rest vs Task') &
                           (fdr_df['band'] == 'gamma')]
        real_di = gamma_row.iloc[0]['DI'] if len(gamma_row) > 0 else None

        ax.hist(null, bins=60, color='#bdc3c7', edgecolor='white',
                density=True, alpha=0.8, label='Null distribution')

        if real_di is not None:
            ax.axvline(real_di, color='#e74c3c', linewidth=2.5,
                        label=f'Observed DI = {real_di:.1f}%')

        ax.axvline(0, color='grey', linewidth=0.8, linestyle='--')

        # Annotate skewness
        sk = skew(null)
        ax.text(0.02, 0.95, f'skew={sk:+.3f}\nn={len(null)}',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Dissociation Index (%)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title('D. Null Distribution (Gamma, Rest vs Task)\n5,000 session-label permutations',
                      fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)

    fig.suptitle('Validation of Noble–Boundary Dissociation Claims\n'
                 'EEGMMIDB FOOOF (n=1.86M peaks, 109 subjects)',
                 fontsize=13, fontweight='bold', y=1.01)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n  Figure saved: {output_path}")


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Rigorous validation of noble-boundary dissociation claims")
    parser.add_argument('--n-perms', type=int, default=5000)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print("  RIGOROUS VALIDATION OF NOBLE–BOUNDARY DISSOCIATION CLAIMS")
    print("  Addressing 6 critical peer-review concerns")
    print("=" * 80)

    df = load_and_assign_conditions(INPUT_CSV)
    print(f"  Loaded {len(df):,} peaks, {df['session'].nunique()} sessions\n")

    # Analysis 1: FDR
    fdr_df = run_fdr_analysis()

    # Analysis 2: Mode location
    mode_df = run_mode_analysis(df)

    # Analysis 3: Gamma decomposition
    gamma_results = run_gamma_decomposition(df, n_perms=args.n_perms)

    # Analysis 4: Spectral resolution
    run_spectral_resolution_analysis(df)

    # Analysis 5: Null diagnostics
    diag_df, null_arrays = run_null_diagnostics(df, n_perms=args.n_perms)

    # Analysis 6: Summary figure
    fig_path = os.path.join(OUTPUT_DIR, 'dissociation_validation.png')
    plot_validation_figure(fdr_df, mode_df, gamma_results, null_arrays, fig_path)

    # Final verdict
    print("\n" + "=" * 80)
    print("  FINAL VERDICT")
    print("=" * 80)

    # 1. FDR
    n_fdr = fdr_df['reject_fdr'].sum() if 'reject_fdr' in fdr_df.columns else 0
    print(f"  1. FDR correction: {n_fdr}/15 tests survive BH at q=0.05")

    # 2. Mode location
    main_modes = mode_df[~mode_df['band'].str.contains('_')]
    n_closer_noble = (main_modes['closer_to'] == 'noble_1').sum()
    n_closer_attr = (main_modes['closer_to'] == 'attractor').sum()
    print(f"  2. KDE mode: {n_closer_noble}/5 bands closer to noble_1, "
          f"{n_closer_attr}/5 closer to attractor")

    # 3. Gamma decomposition
    if gamma_results:
        motor_p = gamma_results[0]['p_val']
        visual_p = gamma_results[1]['p_val'] if len(gamma_results) > 1 else 1.0
        print(f"  3. Gamma decomposition: motor-effect p={motor_p:.4f}, "
              f"visual-effect p={visual_p:.4f}")

    # 4. Spectral resolution
    print(f"  4. Spectral resolution: Δf={FREQ_RESOLUTION:.3f} Hz — "
          f"noble_1 vs attractor resolvable (Δ=0.56), "
          f"noble_1 vs round-10 NOT resolvable (Δ=0.23)")

    # 5. Null symmetry
    n_skewed = (~diag_df['symmetric']).sum()
    print(f"  5. Null symmetry: {len(diag_df) - n_skewed}/15 have |skew| < 0.5")

    print("\n  Done.")


if __name__ == '__main__':
    main()

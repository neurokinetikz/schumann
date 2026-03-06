#!/usr/bin/env python3
"""
Fast completion of dissociation validation Analyses 3, 5, 6.

The original script's Analysis 5 bottleneck: df.isin() on 1.86M rows × 75K iterations.
Fix: pre-build session→freq dict, concatenate arrays per permutation.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
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
    load_and_assign_conditions, ANALYSIS_BANDS, RUN_CONDITIONS, WINDOW,
)

INPUT_CSV = 'exports_peak_distribution/eegmmidb_fooof/golden_ratio_peaks_EEGMMIDB.csv'
DI_TABLE = 'exports_peak_distribution/eegmmidb_fooof/dissociation_table.csv'
FDR_TABLE = 'exports_peak_distribution/eegmmidb_fooof/validation_fdr_table.csv'
MODE_TABLE = 'exports_peak_distribution/eegmmidb_fooof/validation_mode_locations.csv'
OUTPUT_DIR = 'exports_peak_distribution/eegmmidb_fooof'

NOBLE_KEYS = [k for k in EXTENDED_OFFSETS if 'noble' in k]
# PSD parameters — must match run_eegmmidb_full_pipeline.py
# (nperseg_sec=4.0 at fs=160 → nperseg=640, Δf=0.25 Hz)
SFREQ = 160.0
NPERSEG = 640
FREQ_RESOLUTION = SFREQ / NPERSEG  # 0.25 Hz


def build_session_freq_index(df):
    """Pre-build session -> freq array dict. This is the key optimization."""
    print("  Building session→freq index...", flush=True)
    idx = {}
    for sess, grp in df.groupby('session'):
        idx[sess] = grp['freq'].values
    print(f"  Built index for {len(idx)} sessions", flush=True)
    return idx


def fast_permutation_test_di(session_freq_idx, sess_a, sess_b, band_name, band_range,
                              n_perms=2000, noble_key='noble_1', seed=42):
    """
    Fast permutation test using pre-built session→freq index.
    Returns (observed_di, null_array, result_dict).
    """
    all_sessions = np.array(list(sess_a) + list(sess_b))
    n_a = len(sess_a)

    # Observed DI
    fa = np.concatenate([session_freq_idx[s] for s in sess_a if s in session_freq_idx])
    fb = np.concatenate([session_freq_idx[s] for s in sess_b if s in session_freq_idx])

    ea, na = band_enrichment(fa, band_range)
    eb, nb = band_enrichment(fb, band_range)

    if ea is None or eb is None:
        return None, None, None

    ma = extract_metrics(ea)
    mb = extract_metrics(eb)
    obs_result = compute_dissociation_index(ma, mb, noble_key)
    obs_di = obs_result['DI']

    # Null distribution
    rng = np.random.default_rng(seed)
    null_dis = np.zeros(n_perms)

    for i in range(n_perms):
        perm = rng.permutation(all_sessions)
        perm_a = perm[:n_a]
        perm_b = perm[n_a:]

        pa = np.concatenate([session_freq_idx[s] for s in perm_a if s in session_freq_idx])
        pb = np.concatenate([session_freq_idx[s] for s in perm_b if s in session_freq_idx])

        ea_p, _ = band_enrichment(pa, band_range)
        eb_p, _ = band_enrichment(pb, band_range)

        if ea_p is not None and eb_p is not None:
            ma_p = extract_metrics(ea_p)
            mb_p = extract_metrics(eb_p)
            null_dis[i] = compute_dissociation_index(ma_p, mb_p, noble_key)['DI']

    # P-value and z-score
    p_val = np.mean(np.abs(null_dis) >= np.abs(obs_di))
    null_std = null_dis.std()
    z = (obs_di - null_dis.mean()) / null_std if null_std > 0 else 0.0

    result = {
        **obs_result,
        'noble_1_A': ma.get('noble_1', np.nan),
        'noble_1_B': mb.get('noble_1', np.nan),
        'boundary_A': ma.get('boundary', np.nan),
        'boundary_B': mb.get('boundary', np.nan),
        'attractor_A': ma.get('attractor', np.nan),
        'attractor_B': mb.get('attractor', np.nan),
        'p_val': p_val,
        'z_score': z,
        'null_mean': null_dis.mean(),
        'null_std': null_std,
        'n_a': na, 'n_b': nb,
    }

    return obs_di, null_dis, result


# =========================================================================
# ANALYSIS 3: GAMMA 3-CELL DECOMPOSITION
# =========================================================================

def run_gamma_decomposition(session_freq_idx, sessions_by_cond, n_perms=5000):
    """Decompose gamma DI into EO/task components using fast index."""
    print("\n" + "=" * 80, flush=True)
    print("  ANALYSIS 3: GAMMA 3-CELL DECOMPOSITION (EO/task confound)", flush=True)
    print("=" * 80, flush=True)

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
        print(f"\n  {label}  (A={len(sess_a)}, B={len(sess_b)})", flush=True)
        obs_di, null, result = fast_permutation_test_di(
            session_freq_idx, sess_a, sess_b, 'gamma', gamma_range,
            n_perms=n_perms, noble_key='noble_1',
        )

        if result is None:
            print(f"    N/A", flush=True)
            continue

        sig = ""
        if result['p_val'] < 0.001: sig = "***"
        elif result['p_val'] < 0.01: sig = "** "
        elif result['p_val'] < 0.05: sig = "*  "

        dis = "YES" if result['is_dissociation'] else "no"

        print(f"    Noble1_A={result['noble_1_A']:+.1f}%  Noble1_B={result['noble_1_B']:+.1f}%  "
              f"dNoble={result['d_noble']:+.1f}%", flush=True)
        print(f"    Bound_A={result['boundary_A']:+.1f}%   Bound_B={result['boundary_B']:+.1f}%   "
              f"dBound={result['d_boundary']:+.1f}%", flush=True)
        print(f"    DI={result['DI']:+.1f}%  z={result['z_score']:+.2f}  "
              f"p={result['p_val']:.4f}{sig}  Dissociation: {dis}", flush=True)

        results.append({'comparison': label, **result})

    # Interpretation
    print(f"\n  --- Interpretation ---", flush=True)
    if len(results) >= 2:
        motor_p = results[0]['p_val']
        visual_p = results[1]['p_val']

        if motor_p < 0.05 and visual_p >= 0.05:
            print("  -> Motor effect is real (rest-EO vs task significant, visual not)", flush=True)
        elif motor_p >= 0.05 and visual_p < 0.05:
            print("  -> Gamma signal is VISUAL, not motor (visual significant, motor not)", flush=True)
        elif motor_p < 0.05 and visual_p < 0.05:
            print("  -> Both motor and visual contribute independently", flush=True)
        else:
            print("  -> Neither reaches significance in isolation — original effect is fragile", flush=True)

    return results


# =========================================================================
# ANALYSIS 4: SPECTRAL RESOLUTION (just print, no permutation)
# =========================================================================

def run_spectral_resolution_analysis(df):
    """Document spectral resolution limitations for alpha claims."""
    print("\n" + "=" * 80, flush=True)
    print("  ANALYSIS 4: SPECTRAL RESOLUTION CAVEAT", flush=True)
    print(f"  SFREQ={SFREQ} Hz, nperseg={NPERSEG} -> df={FREQ_RESOLUTION:.4f} Hz", flush=True)
    print("=" * 80, flush=True)

    alpha_freqs = df.loc[(df['freq'] >= 7.6) & (df['freq'] < 12.3), 'freq'].values

    print(f"\n  Alpha peaks: {len(alpha_freqs):,}", flush=True)
    print(f"  Mean: {alpha_freqs.mean():.3f} Hz", flush=True)
    print(f"  Median: {np.median(alpha_freqs):.3f} Hz", flush=True)

    if len(alpha_freqs) > 100:
        kde = gaussian_kde(alpha_freqs, bw_method=0.1)
        x = np.linspace(7.6, 12.3, 1000)
        alpha_mode = x[np.argmax(kde(x))]
        print(f"  KDE mode: {alpha_mode:.3f} Hz", flush=True)

    targets = [('noble_1', 10.23), ('attractor', 9.67), ('round_10', 10.00)]

    print(f"\n  Can the spectral resolution distinguish key frequencies?", flush=True)
    for i, (name_a, freq_a) in enumerate(targets):
        for name_b, freq_b in targets[i+1:]:
            delta = abs(freq_a - freq_b)
            resolvable = "YES" if delta > FREQ_RESOLUTION else "NO"
            print(f"    {name_a} ({freq_a:.2f}) vs {name_b} ({freq_b:.2f}): "
                  f"d={delta:.3f} Hz -- {resolvable} (df={FREQ_RESOLUTION:.3f})", flush=True)

    near_noble1 = alpha_freqs[(alpha_freqs >= 10.23 - FREQ_RESOLUTION) &
                               (alpha_freqs <= 10.23 + FREQ_RESOLUTION)]
    near_round10 = alpha_freqs[(alpha_freqs >= 10.0 - FREQ_RESOLUTION) &
                                (alpha_freqs <= 10.0 + FREQ_RESOLUTION)]
    overlap_idx = np.where(
        (alpha_freqs >= max(10.23 - FREQ_RESOLUTION, 10.0 - FREQ_RESOLUTION)) &
        (alpha_freqs <= min(10.23 + FREQ_RESOLUTION, 10.0 + FREQ_RESOLUTION))
    )[0]

    overlap_frac = len(overlap_idx) / max(len(near_noble1), 1)
    print(f"\n  Peaks within +/-{FREQ_RESOLUTION:.3f} Hz of noble_1 (10.23): {len(near_noble1):,}", flush=True)
    print(f"  Peaks within +/-{FREQ_RESOLUTION:.3f} Hz of round 10.0:    {len(near_round10):,}", flush=True)
    print(f"  Overlap fraction: {overlap_frac:.1%}", flush=True)
    print(f"  -> {'CONFOUNDED' if overlap_frac > 0.5 else 'Partially distinguishable'}", flush=True)

    psd_grid = np.arange(0, 80, FREQ_RESOLUTION)
    alpha_grid = psd_grid[(psd_grid >= 7.6) & (psd_grid < 12.3)]
    print(f"\n  PSD grid points in alpha: {len(alpha_grid)}", flush=True)
    print(f"  Grid points near noble_1: "
          f"{alpha_grid[np.abs(alpha_grid - 10.23) < FREQ_RESOLUTION/2]}", flush=True)
    print(f"  Grid points near attractor: "
          f"{alpha_grid[np.abs(alpha_grid - 9.67) < FREQ_RESOLUTION/2]}", flush=True)


# =========================================================================
# ANALYSIS 5: NULL DISTRIBUTION DIAGNOSTICS (FAST)
# =========================================================================

def run_null_diagnostics_fast(session_freq_idx, sessions_by_cond, n_perms=2000):
    """Fast null distribution diagnostics using pre-built index."""
    print("\n" + "=" * 80, flush=True)
    print("  ANALYSIS 5: NULL DISTRIBUTION DIAGNOSTICS (FAST)", flush=True)
    print(f"  Using {n_perms} permutations with pre-indexed sessions", flush=True)
    print("=" * 80, flush=True)

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
    null_arrays = {}

    print(f"\n  {'Contrast':20s} {'Band':12s} | {'Mean':>8s} {'Median':>8s} {'Std':>7s} "
          f"{'Skew':>7s} {'Kurt':>7s} {'Shap_p':>8s} {'Symm?':>6s}", flush=True)
    print(f"  {'-'*20} {'-'*12} | {'-'*8} {'-'*8} {'-'*7} "
          f"{'-'*7} {'-'*7} {'-'*8} {'-'*6}", flush=True)

    for cname, (sess_a, sess_b) in contrasts.items():
        all_sessions = np.array(list(sess_a) + list(sess_b))
        n_a = len(sess_a)

        for band_name, band_range in ANALYSIS_BANDS.items():
            null_dis = np.zeros(n_perms)
            for i in range(n_perms):
                perm = rng.permutation(all_sessions)
                perm_a = perm[:n_a]
                perm_b = perm[n_a:]

                # Fast: concatenate from index instead of .isin()
                pa = np.concatenate([session_freq_idx[s] for s in perm_a
                                     if s in session_freq_idx])
                pb = np.concatenate([session_freq_idx[s] for s in perm_b
                                     if s in session_freq_idx])

                ea, _ = band_enrichment(pa, band_range)
                eb, _ = band_enrichment(pb, band_range)

                if ea is not None and eb is not None:
                    ma = extract_metrics(ea)
                    mb = extract_metrics(eb)
                    null_dis[i] = compute_dissociation_index(ma, mb, 'noble_1')['DI']

            null_key = f"{cname}_{band_name}"
            null_arrays[null_key] = null_dis

            sk = skew(null_dis)
            ku = kurtosis(null_dis)
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
                  f"{sk:+7.3f} {ku:+7.3f} {shap_p:8.4f} {sym_marker:>6s}", flush=True)

    diag_df = pd.DataFrame(diag_rows)
    diag_path = os.path.join(OUTPUT_DIR, 'validation_null_diagnostics.csv')
    diag_df.to_csv(diag_path, index=False)
    print(f"\n  Saved: {diag_path}", flush=True)

    n_skewed = (~diag_df['symmetric']).sum()
    print(f"\n  {n_skewed} of {len(diag_df)} null distributions show |skewness| > 0.5", flush=True)

    return diag_df, null_arrays


# =========================================================================
# ANALYSIS 6: SUMMARY FIGURE
# =========================================================================

def plot_validation_figure(fdr_df, mode_df, gamma_results, null_arrays, output_path):
    """4-panel validation figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Panel A: FDR-corrected DI significance ---
    ax = axes[0, 0]
    bands = list(ANALYSIS_BANDS.keys())

    for contrast, style, offset in [
        ('Rest vs Task', {'color': '#2c3e50', 'alpha': 0.9}, -0.2),
        ('Eyes Open vs Closed', {'color': '#2980b9', 'alpha': 0.7}, 0.0),
        ('Execution vs Imagery', {'color': '#8e44ad', 'alpha': 0.7}, 0.2),
    ]:
        di_sub = fdr_df[fdr_df['contrast'] == contrast]
        x = np.arange(len(bands))
        di_vals = []
        edge_colors = []
        for b in bands:
            row = di_sub[di_sub['band'] == b]
            if len(row) > 0:
                di_vals.append(row.iloc[0]['DI'])
                if row.iloc[0].get('reject_fdr', False):
                    edge_colors.append('#e74c3c')
                elif row.iloc[0]['p_val'] < 0.05:
                    edge_colors.append('#f39c12')
                else:
                    edge_colors.append('none')
            else:
                di_vals.append(0)
                edge_colors.append('none')

        bars = ax.bar(x + offset, di_vals, 0.2, color=style['color'],
                       alpha=style['alpha'], label=contrast.replace(' vs ', '\nvs '),
                       edgecolor=edge_colors, linewidth=2)

    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.set_xticks(np.arange(len(bands)))
    ax.set_xticklabels([b.replace('_', '\n') for b in bands], fontsize=9)
    ax.set_ylabel('Dissociation Index (%)', fontsize=10)
    ax.set_title('A. DI by Contrast and Band\n(orange border = p<0.05 uncorr; red = FDR)',
                  fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')

    # --- Panel B: Within-octave KDE mode per band ---
    ax = axes[0, 1]
    mode_main = mode_df[~mode_df['band'].str.contains('_')].copy()
    if len(mode_main) > 0:
        band_order = list(ANALYSIS_BANDS.keys())
        mode_main = mode_main.set_index('band').reindex(band_order).reset_index()
        x = np.arange(len(mode_main))
        modes = mode_main['mode'].values
        ci_lo = mode_main['ci_lower'].values
        ci_hi = mode_main['ci_upper'].values

        ax.errorbar(x, modes, yerr=[modes - ci_lo, ci_hi - modes],
                     fmt='ko', capsize=5, capthick=1.5, markersize=8, linewidth=1.5)
        ax.axhline(0.500, color='#3498db', linewidth=1.5, linestyle='--',
                    label='Attractor (0.500)')
        ax.axhline(0.618, color='#e74c3c', linewidth=1.5, linestyle='--',
                    label='Noble_1 (0.618)')
        ax.set_xticks(x)
        ax.set_xticklabels([b.replace('_', '\n') for b in mode_main['band']], fontsize=9)
        ax.set_ylabel('KDE Mode (lattice coordinate)', fontsize=10)
        ax.set_title('B. Within-Octave Density Peak Location\n(black dots = mode +/- 95% CI)',
                      fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')
        ax.set_ylim(0.3, 1.0)

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
            offset = max(abs(d) * 0.1, 2)
            ax.text(d + (offset if d >= 0 else -offset), i,
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

        gamma_row = fdr_df[(fdr_df['contrast'] == 'Rest vs Task') &
                           (fdr_df['band'] == 'gamma')]
        real_di = gamma_row.iloc[0]['DI'] if len(gamma_row) > 0 else None

        ax.hist(null, bins=60, color='#bdc3c7', edgecolor='white',
                density=True, alpha=0.8, label='Null distribution')

        if real_di is not None:
            ax.axvline(real_di, color='#e74c3c', linewidth=2.5,
                        label=f'Observed DI = {real_di:.1f}%')

        ax.axvline(0, color='grey', linewidth=0.8, linestyle='--')

        sk = skew(null)
        ax.text(0.02, 0.95, f'skew={sk:+.3f}\nn={len(null)}',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Dissociation Index (%)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title('D. Null Distribution (Gamma, Rest vs Task)\nsession-label permutations',
                      fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)

    fig.suptitle('Validation of Noble-Boundary Dissociation Claims\n'
                 'EEGMMIDB FOOOF (n=1.86M peaks, 109 subjects)',
                 fontsize=13, fontweight='bold', y=1.01)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n  Figure saved: {output_path}", flush=True)


# =========================================================================
# MAIN
# =========================================================================

def main():
    import time

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    n_perms_gamma = 5000
    n_perms_null = 2000  # Sufficient for diagnostics

    print("=" * 80, flush=True)
    print("  FAST COMPLETION: Analyses 3, 4, 5, 6", flush=True)
    print(f"  Gamma decomposition: {n_perms_gamma} perms", flush=True)
    print(f"  Null diagnostics: {n_perms_null} perms (sufficient for skew/normality)", flush=True)
    print("=" * 80, flush=True)

    # Load data
    t0 = time.time()
    df = load_and_assign_conditions(INPUT_CSV)
    print(f"  Loaded {len(df):,} peaks, {df['session'].nunique()} sessions "
          f"({time.time()-t0:.1f}s)", flush=True)

    # Build fast index
    t0 = time.time()
    session_freq_idx = build_session_freq_index(df)
    print(f"  Index built ({time.time()-t0:.1f}s)", flush=True)

    # Session sets
    sessions_by_cond = {
        c: set(df.loc[df['condition'] == c, 'session'].unique())
        for c in df['condition'].unique()
    }

    # Load already-completed results
    fdr_df = pd.read_csv(FDR_TABLE)
    mode_df = pd.read_csv(MODE_TABLE)
    print(f"\n  Loaded FDR table ({len(fdr_df)} rows) and mode table ({len(mode_df)} rows)", flush=True)

    # Analysis 3: Gamma decomposition
    t0 = time.time()
    gamma_results = run_gamma_decomposition(session_freq_idx, sessions_by_cond,
                                             n_perms=n_perms_gamma)
    print(f"\n  Analysis 3 done ({time.time()-t0:.1f}s)", flush=True)

    # Analysis 4: Spectral resolution
    run_spectral_resolution_analysis(df)

    # Analysis 5: Null diagnostics (FAST)
    t0 = time.time()
    diag_df, null_arrays = run_null_diagnostics_fast(session_freq_idx, sessions_by_cond,
                                                      n_perms=n_perms_null)
    print(f"\n  Analysis 5 done ({time.time()-t0:.1f}s)", flush=True)

    # Analysis 6: Summary figure
    fig_path = os.path.join(OUTPUT_DIR, 'dissociation_validation.png')
    plot_validation_figure(fdr_df, mode_df, gamma_results, null_arrays, fig_path)

    # Save gamma results
    if gamma_results:
        gamma_df = pd.DataFrame(gamma_results)
        gamma_path = os.path.join(OUTPUT_DIR, 'validation_gamma_decomposition.csv')
        gamma_df.to_csv(gamma_path, index=False)
        print(f"  Saved: {gamma_path}", flush=True)

    # Final verdict
    print("\n" + "=" * 80, flush=True)
    print("  FINAL VERDICT", flush=True)
    print("=" * 80, flush=True)

    n_fdr = fdr_df['reject_fdr'].sum() if 'reject_fdr' in fdr_df.columns else 0
    print(f"  1. FDR correction: {n_fdr}/15 tests survive BH at q=0.05", flush=True)

    main_modes = mode_df[~mode_df['band'].str.contains('_')]
    n_closer_noble = (main_modes['closer_to'] == 'noble_1').sum()
    n_closer_attr = (main_modes['closer_to'] == 'attractor').sum()
    print(f"  2. KDE mode: {n_closer_noble}/5 bands closer to noble_1, "
          f"{n_closer_attr}/5 closer to attractor", flush=True)

    if gamma_results:
        motor_p = gamma_results[0]['p_val']
        visual_p = gamma_results[1]['p_val'] if len(gamma_results) > 1 else 1.0
        print(f"  3. Gamma decomposition: motor-effect p={motor_p:.4f}, "
              f"visual-effect p={visual_p:.4f}", flush=True)

    print(f"  4. Spectral resolution: df={FREQ_RESOLUTION:.3f} Hz -- "
          f"noble_1 vs attractor resolvable (d=0.56), "
          f"noble_1 vs round-10 NOT resolvable (d=0.23)", flush=True)

    n_skewed = (~diag_df['symmetric']).sum()
    print(f"  5. Null symmetry: {len(diag_df) - n_skewed}/15 have |skew| < 0.5", flush=True)

    print("\n  Done.", flush=True)


if __name__ == '__main__':
    main()

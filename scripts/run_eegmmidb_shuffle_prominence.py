#!/usr/bin/env python3
"""
Within-Band Shuffle Test & Prominence Decomposition (EEGMMIDB)
================================================================

Tests whether phi-lattice structural score reflects genuine within-band
spectral organization or is an artifact of inter-band peak density clustering.

Key insight: phi-octave band boundaries map to u=0 in lattice coordinates,
so uniform-in-frequency within a phi-octave = uniform-in-u. The null for
N band-peaks is N draws from U(0,1), making each permutation ~0.1ms.

Phases:
  1. Within-band shuffle (phi, f0=8.5) — does SS survive?
  2. Per-band leave-one-out — which band drives the signal?
  3. Prominence decomposition — is SS prominence-dependent?
  4. Multi-base comparison under shuffle null
  5. Repeat at f0=7.6

Usage:
    python scripts/run_eegmmidb_shuffle_prominence.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, './lib')
sys.path.insert(0, './scripts')

from phi_frequency_model import PHI
from ratio_specificity import lattice_coordinate, _enrichment_at_offset
from structural_phi_specificity import (
    natural_positions, compute_structural_score, BASES
)

# =========================================================================
# CONSTANTS
# =========================================================================

DEFAULT_INPUT_CSV = 'exports_peak_distribution/eegmmidb_fooof/golden_ratio_peaks_EEGMMIDB.csv'
DEFAULT_OUTPUT_DIR = 'exports_peak_distribution/eegmmidb_fooof/shuffle_prominence'

F0_VALUES = [8.5, 7.6]
N_PERM_SHUFFLE = 10_000
N_PERM_LOO = 5_000
N_BOOTSTRAP = 1_000
WINDOW = 0.05
SEED = 42


def make_phi_bands(f0, freq_ceil=50.0):
    """Phi-octave band boundaries, gamma capped at freq_ceil."""
    return {
        'theta':     (f0 / PHI, f0),
        'alpha':     (f0, f0 * PHI),
        'beta_low':  (f0 * PHI, f0 * PHI ** 2),
        'beta_high': (f0 * PHI ** 2, f0 * PHI ** 3),
        'gamma':     (f0 * PHI ** 3, freq_ceil),
    }


def band_u_range(lo, hi, f0, base):
    """Compute the u-coordinate span for a frequency band.

    For a full phi-octave [f0*b^n, f0*b^(n+1)], u spans [0, 1).
    For a truncated band (gamma), u spans [0, log_b(hi/lo) mod 1).
    """
    span = np.log(hi / lo) / np.log(base)
    return span % 1.0 if abs(span - round(span)) > 1e-9 else 1.0


# =========================================================================
# PHASE 1: WITHIN-BAND SHUFFLE TEST
# =========================================================================

def within_band_shuffle_null(freqs, f0, base=PHI, n_perm=10_000, seed=42,
                              freq_ceil=50.0, window=WINDOW):
    """Test SS under within-band shuffle null.

    For each permutation: draw n_band peaks from U(0, u_range) per band,
    concatenate, compute SS. For full phi-octave bands, u_range = 1.0.
    """
    # Observed
    u_obs = lattice_coordinate(freqs, f0, base)
    u_obs = u_obs[np.isfinite(u_obs)]
    positions = natural_positions(base)
    SS_obs, enrich_obs = compute_structural_score(u_obs, positions, window)

    # Also compute global phase-rotation null for comparison
    rng = np.random.default_rng(seed)
    null_global = np.empty(n_perm)
    for i in range(n_perm):
        theta = rng.uniform(0, 1)
        u_rot = (u_obs + theta) % 1.0
        null_global[i], _ = compute_structural_score(u_rot, positions, window)

    z_global = (SS_obs - null_global.mean()) / null_global.std()
    p_global = (null_global >= SS_obs).mean()

    # Within-band shuffle null
    bands = make_phi_bands(f0, freq_ceil)
    freqs_arr = np.asarray(freqs)

    # Precompute band membership and u-ranges
    band_info = []
    for bname, (lo, hi) in bands.items():
        mask = (freqs_arr >= lo) & (freqs_arr < hi)
        n_band = mask.sum()
        u_range = band_u_range(lo, hi, f0, base)
        band_info.append((bname, n_band, u_range))

    null_shuffle = np.empty(n_perm)
    for i in range(n_perm):
        u_parts = []
        for bname, n_band, u_range in band_info:
            if n_band > 0:
                u_parts.append(rng.uniform(0, u_range, size=n_band))
        u_null = np.concatenate(u_parts)
        null_shuffle[i], _ = compute_structural_score(u_null, positions, window)

    z_shuffle = (SS_obs - null_shuffle.mean()) / null_shuffle.std()
    p_shuffle = (null_shuffle >= SS_obs).mean()

    return {
        'SS_obs': SS_obs,
        'enrichments_obs': enrich_obs,
        'z_global': z_global,
        'p_global': p_global,
        'null_global_mean': null_global.mean(),
        'null_global_std': null_global.std(),
        'z_shuffle': z_shuffle,
        'p_shuffle': p_shuffle,
        'null_shuffle_mean': null_shuffle.mean(),
        'null_shuffle_std': null_shuffle.std(),
        'null_shuffle': null_shuffle,
        'null_global': null_global,
        'band_info': band_info,
        'n_peaks': len(u_obs),
    }


# =========================================================================
# PHASE 2: PER-BAND LEAVE-ONE-OUT
# =========================================================================

def per_band_leave_one_out(freqs, f0, base=PHI, n_perm=5_000, seed=42,
                            freq_ceil=50.0, window=WINDOW):
    """Shuffle one band at a time, keep others real. Measure SS drop."""
    u_obs = lattice_coordinate(freqs, f0, base)
    u_obs = u_obs[np.isfinite(u_obs)]
    positions = natural_positions(base)
    SS_obs, _ = compute_structural_score(u_obs, positions, window)

    bands = make_phi_bands(f0, freq_ceil)
    freqs_arr = np.asarray(freqs)
    rng = np.random.default_rng(seed)

    results = []
    for bname, (lo, hi) in bands.items():
        mask = (freqs_arr >= lo) & (freqs_arr < hi)
        n_band = mask.sum()
        u_range = band_u_range(lo, hi, f0, base)

        # u-coordinates for all peaks
        u_all = lattice_coordinate(freqs_arr, f0, base)
        u_real = u_all[np.isfinite(u_all)]

        # Indices within the finite-u array that belong to this band
        freqs_finite = freqs_arr[np.isfinite(u_all)]
        band_idx = np.where((freqs_finite >= lo) & (freqs_finite < hi))[0]
        other_idx = np.where(~((freqs_finite >= lo) & (freqs_finite < hi)))[0]

        null_scores = np.empty(n_perm)
        for i in range(n_perm):
            u_perm = u_real.copy()
            u_perm[band_idx] = rng.uniform(0, u_range, size=len(band_idx))
            null_scores[i], _ = compute_structural_score(u_perm, positions, window)

        SS_shuffled_mean = null_scores.mean()
        SS_drop = SS_obs - SS_shuffled_mean
        SS_drop_pct = (SS_drop / SS_obs * 100) if SS_obs != 0 else 0.0
        p_val = (null_scores >= SS_obs).mean()

        results.append({
            'band': bname,
            'n_peaks': n_band,
            'SS_obs': SS_obs,
            'SS_shuffled_mean': SS_shuffled_mean,
            'SS_shuffled_std': null_scores.std(),
            'SS_drop': SS_drop,
            'SS_drop_pct': SS_drop_pct,
            'p_value': p_val,
        })

    return pd.DataFrame(results)


# =========================================================================
# PHASE 3: PROMINENCE DECOMPOSITION
# =========================================================================

def prominence_decomposition(freqs, powers, f0, base=PHI, n_deciles=10,
                              n_bootstrap=1_000, seed=42, window=WINDOW):
    """Compute SS per power decile with bootstrap CIs."""
    freqs = np.asarray(freqs, dtype=float)
    powers = np.asarray(powers, dtype=float)
    positions = natural_positions(base)
    rng = np.random.default_rng(seed)

    # Compute decile edges
    decile_edges = np.percentile(powers, np.linspace(0, 100, n_deciles + 1))

    results = []
    for d in range(n_deciles):
        lo_p, hi_p = decile_edges[d], decile_edges[d + 1]
        if d < n_deciles - 1:
            mask = (powers >= lo_p) & (powers < hi_p)
        else:
            mask = (powers >= lo_p) & (powers <= hi_p)

        f_d = freqs[mask]
        n_d = len(f_d)
        if n_d < 10:
            results.append({
                'decile': d + 1, 'n_peaks': n_d,
                'power_lo': lo_p, 'power_hi': hi_p,
                'SS': np.nan, 'SS_ci_lo': np.nan, 'SS_ci_hi': np.nan,
                'E_boundary': np.nan, 'E_attractor': np.nan,
                'E_noble_1': np.nan, 'E_noble_2': np.nan,
            })
            continue

        u_d = lattice_coordinate(f_d, f0, base)
        u_d = u_d[np.isfinite(u_d)]
        SS_d, enrich_d = compute_structural_score(u_d, positions, window)

        # Bootstrap CI
        boot_ss = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.integers(0, len(u_d), size=len(u_d))
            boot_ss[b], _ = compute_structural_score(u_d[idx], positions, window)

        results.append({
            'decile': d + 1, 'n_peaks': n_d,
            'power_lo': lo_p, 'power_hi': hi_p,
            'SS': SS_d,
            'SS_ci_lo': np.percentile(boot_ss, 2.5),
            'SS_ci_hi': np.percentile(boot_ss, 97.5),
            'E_boundary': enrich_d.get('boundary', np.nan),
            'E_attractor': enrich_d.get('attractor', np.nan),
            'E_noble_1': enrich_d.get('noble', np.nan),
            'E_noble_2': enrich_d.get('noble_2', np.nan),
        })

    return pd.DataFrame(results)


def prominence_decomposition_per_band(freqs, powers, f0, base=PHI,
                                       n_deciles=5, n_bootstrap=1_000,
                                       seed=42, freq_ceil=50.0, window=WINDOW):
    """Prominence decomposition within each phi-octave band."""
    freqs = np.asarray(freqs, dtype=float)
    powers = np.asarray(powers, dtype=float)
    bands = make_phi_bands(f0, freq_ceil)
    positions = natural_positions(base)
    rng = np.random.default_rng(seed)

    all_results = []
    for bname, (lo, hi) in bands.items():
        bmask = (freqs >= lo) & (freqs < hi)
        f_band = freqs[bmask]
        p_band = powers[bmask]

        if len(f_band) < 20:
            continue

        decile_edges = np.percentile(p_band, np.linspace(0, 100, n_deciles + 1))

        for d in range(n_deciles):
            lo_p, hi_p = decile_edges[d], decile_edges[d + 1]
            if d < n_deciles - 1:
                mask = (p_band >= lo_p) & (p_band < hi_p)
            else:
                mask = (p_band >= lo_p) & (p_band <= hi_p)

            f_d = f_band[mask]
            n_d = len(f_d)
            if n_d < 10:
                all_results.append({
                    'band': bname, 'decile': d + 1, 'n_peaks': n_d,
                    'power_lo': lo_p, 'power_hi': hi_p,
                    'SS': np.nan, 'SS_ci_lo': np.nan, 'SS_ci_hi': np.nan,
                })
                continue

            u_d = lattice_coordinate(f_d, f0, base)
            u_d = u_d[np.isfinite(u_d)]
            SS_d, _ = compute_structural_score(u_d, positions, window)

            boot_ss = np.empty(n_bootstrap)
            for b in range(n_bootstrap):
                idx = rng.integers(0, len(u_d), size=len(u_d))
                boot_ss[b], _ = compute_structural_score(u_d[idx], positions, window)

            all_results.append({
                'band': bname, 'decile': d + 1, 'n_peaks': n_d,
                'power_lo': lo_p, 'power_hi': hi_p,
                'SS': SS_d,
                'SS_ci_lo': np.percentile(boot_ss, 2.5),
                'SS_ci_hi': np.percentile(boot_ss, 97.5),
            })

    return pd.DataFrame(all_results)


# =========================================================================
# PHASE 4: MULTI-BASE SHUFFLE COMPARISON
# =========================================================================

def multi_base_shuffle_test(freqs, f0_values, bases, n_perm=10_000, seed=42,
                             freq_ceil=50.0, window=WINDOW):
    """Within-band shuffle + global phase-rotation for each (f0, base)."""
    rng_base = np.random.default_rng(seed)
    rows = []

    for f0 in f0_values:
        bands = make_phi_bands(f0, freq_ceil)
        freqs_arr = np.asarray(freqs, dtype=float)

        # Precompute band info (shared across bases for same f0)
        band_counts = []
        for bname, (lo, hi) in bands.items():
            mask = (freqs_arr >= lo) & (freqs_arr < hi)
            n_band = mask.sum()
            # For non-phi bases, band boundaries are still phi-octave.
            # u_range depends on the actual base being tested.
            band_counts.append((bname, lo, hi, n_band))

        for bkey, (bval, blabel) in bases.items():
            rng = np.random.default_rng(rng_base.integers(0, 2**32))

            u_obs = lattice_coordinate(freqs_arr, f0, bval)
            u_obs_finite = u_obs[np.isfinite(u_obs)]
            positions = natural_positions(bval)
            SS_obs, enrich_obs = compute_structural_score(u_obs_finite, positions, window)

            # Global null
            null_global = np.empty(n_perm)
            for i in range(n_perm):
                theta = rng.uniform(0, 1)
                u_rot = (u_obs_finite + theta) % 1.0
                null_global[i], _ = compute_structural_score(u_rot, positions, window)

            z_global = (SS_obs - null_global.mean()) / null_global.std() if null_global.std() > 0 else 0.0
            p_global = (null_global >= SS_obs).mean()

            # Within-band shuffle null
            # Compute u_range for each band under this base
            band_info_base = []
            for bname, lo, hi, n_band in band_counts:
                u_range = band_u_range(lo, hi, f0, bval)
                band_info_base.append((bname, n_band, u_range))

            null_shuffle = np.empty(n_perm)
            for i in range(n_perm):
                u_parts = []
                for bname, n_band, u_range in band_info_base:
                    if n_band > 0:
                        u_parts.append(rng.uniform(0, u_range, size=n_band))
                u_null = np.concatenate(u_parts)
                null_shuffle[i], _ = compute_structural_score(u_null, positions, window)

            z_shuffle = (SS_obs - null_shuffle.mean()) / null_shuffle.std() if null_shuffle.std() > 0 else 0.0
            p_shuffle = (null_shuffle >= SS_obs).mean()

            rows.append({
                'f0': f0,
                'base_key': bkey,
                'base_label': blabel,
                'base_value': bval,
                'n_positions': len(positions),
                'SS_obs': SS_obs,
                'z_global': z_global,
                'p_global': p_global,
                'z_shuffle': z_shuffle,
                'p_shuffle': p_shuffle,
                'null_shuffle_mean': null_shuffle.mean(),
                'null_shuffle_std': null_shuffle.std(),
                **{f'E_{k}': v for k, v in enrich_obs.items()},
            })

            print(f"  f0={f0}, {blabel:>4s}: SS={SS_obs:+7.1f}  "
                  f"z_global={z_global:+5.1f}  z_shuffle={z_shuffle:+5.1f}  "
                  f"p_shuffle={p_shuffle:.4f}")

    return pd.DataFrame(rows)


# =========================================================================
# FIGURES
# =========================================================================

def fig_shuffle_test(results_by_f0, outpath):
    """Figure 1: Observed SS vs shuffle null distribution."""
    fig, axes = plt.subplots(2, len(results_by_f0), figsize=(6 * len(results_by_f0), 8))
    if len(results_by_f0) == 1:
        axes = axes.reshape(-1, 1)

    for col, (f0, res) in enumerate(results_by_f0.items()):
        # Top: histogram of shuffle null
        ax = axes[0, col]
        ax.hist(res['null_shuffle'], bins=80, alpha=0.6, color='steelblue',
                density=True, label='Within-band shuffle null')
        ax.axvline(res['SS_obs'], color='red', lw=2, label=f"Observed SS={res['SS_obs']:.1f}")
        ax.axvline(res['null_shuffle_mean'], color='steelblue', ls='--', lw=1,
                   label=f"Null mean={res['null_shuffle_mean']:.1f}")
        ax.set_xlabel('Structural Score')
        ax.set_ylabel('Density')
        ax.set_title(f'f₀ = {f0} Hz')
        ax.legend(fontsize=8)

        # Bottom: z-score comparison
        ax = axes[1, col]
        z_vals = [res['z_global'], res['z_shuffle']]
        labels = ['Global\nphase-rotation', 'Within-band\nshuffle']
        colors = ['#4CAF50', '#FF9800']
        bars = ax.bar(labels, z_vals, color=colors, width=0.5, edgecolor='black')
        ax.axhline(2.0, color='gray', ls='--', lw=1, alpha=0.5)
        ax.axhline(-2.0, color='gray', ls='--', lw=1, alpha=0.5)
        for bar, z in zip(bars, z_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'z={z:.1f}', ha='center', fontsize=10, fontweight='bold')
        ax.set_ylabel('z-score')
        ax.set_title(f'Null comparison (f₀={f0})')

    fig.suptitle('Within-Band Shuffle Test (φ-lattice, EEGMMIDB)', fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")


def fig_band_contributions(loo_df, f0, outpath):
    """Figure 2: Per-band SS drop from leave-one-out."""
    fig, ax = plt.subplots(figsize=(8, 5))

    bands = loo_df['band'].values
    drops = loo_df['SS_drop'].values
    n_peaks = loo_df['n_peaks'].values
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    bars = ax.bar(range(len(bands)), drops, color=colors[:len(bands)],
                  edgecolor='black', width=0.6)

    # Add peak count labels
    for i, (bar, n) in enumerate(zip(bars, n_peaks)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'n={n:,}', ha='center', fontsize=8, color='gray')

    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([f"{b}\n({n:,} peaks)" for b, n in zip(bands, n_peaks)],
                       fontsize=9)
    ax.set_ylabel('SS drop when band shuffled')
    ax.set_title(f'Per-Band Leave-One-Out (φ, f₀={f0} Hz)\n'
                 f'Higher = band contributes more to lattice structure')
    ax.axhline(0, color='black', lw=0.5)

    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")


def fig_prominence(agg_df, band_df, f0, outpath):
    """Figure 3: Prominence decomposition."""
    has_band = band_df is not None and len(band_df) > 0
    band_names = band_df['band'].unique() if has_band else []
    n_bands = len(band_names)

    fig, axes = plt.subplots(1 + (1 if has_band else 0), 1,
                              figsize=(10, 5 + (4 if has_band else 0)),
                              gridspec_kw={'height_ratios': [2] + ([1.5] if has_band else [])})
    if not hasattr(axes, '__len__'):
        axes = [axes]

    # Top: aggregate
    ax = axes[0]
    valid = agg_df.dropna(subset=['SS'])
    ax.fill_between(valid['decile'], valid['SS_ci_lo'], valid['SS_ci_hi'],
                    alpha=0.2, color='steelblue')
    ax.plot(valid['decile'], valid['SS'], 'o-', color='steelblue', lw=2,
            markersize=6)
    ax.axhline(0, color='gray', ls='--', lw=1)
    ax.set_xlabel('Power decile (1=weakest, 10=strongest)')
    ax.set_ylabel('Structural Score')
    ax.set_title(f'Prominence Decomposition (φ, f₀={f0} Hz)')
    ax.set_xticks(valid['decile'].values)

    # Bottom: per-band
    if has_band:
        ax = axes[1]
        colors = {'theta': '#1f77b4', 'alpha': '#ff7f0e', 'beta_low': '#2ca02c',
                  'beta_high': '#d62728', 'gamma': '#9467bd'}
        for bname in band_names:
            bdata = band_df[band_df['band'] == bname].dropna(subset=['SS'])
            if len(bdata) > 0:
                ax.plot(bdata['decile'], bdata['SS'], 'o-', label=bname,
                        color=colors.get(bname, 'gray'), lw=1.5, markersize=4)
                ax.fill_between(bdata['decile'], bdata['SS_ci_lo'], bdata['SS_ci_hi'],
                                alpha=0.1, color=colors.get(bname, 'gray'))
        ax.axhline(0, color='gray', ls='--', lw=1)
        ax.set_xlabel('Power decile within band')
        ax.set_ylabel('Structural Score')
        ax.set_title('Per-Band Prominence')
        ax.legend(fontsize=8, ncol=5)
        ax.set_xticks(range(1, 6))

    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")


# =========================================================================
# MAIN
# =========================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Within-band shuffle & prominence tests')
    parser.add_argument('--input', type=str, default=None,
                        help='Override input CSV path')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override output directory')
    args = parser.parse_args()

    INPUT_CSV = args.input or DEFAULT_INPUT_CSV
    OUTPUT_DIR = args.output_dir or DEFAULT_OUTPUT_DIR
    FIG_DIR = os.path.join(OUTPUT_DIR, 'figures')

    t0 = time.time()
    os.makedirs(FIG_DIR, exist_ok=True)

    # --- LOAD ---
    print(f"Loading peaks from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    freqs = df['freq'].values
    powers = df['power'].values
    print(f"  {len(df):,} peaks, freq range [{freqs.min():.1f}, {freqs.max():.1f}] Hz")
    print(f"  Power range [{powers.min():.4f}, {powers.max():.4f}]")

    # --- PHASE 1 & 5: Within-band shuffle at both f0 values ---
    print("\n" + "="*70)
    print("PHASE 1 & 5: Within-band shuffle test")
    print("="*70)

    results_by_f0 = {}
    shuffle_rows = []

    for f0 in F0_VALUES:
        print(f"\n  f0 = {f0} Hz ...")
        res = within_band_shuffle_null(freqs, f0, PHI, N_PERM_SHUFFLE, SEED,
                                        freq_ceil=50.0)
        results_by_f0[f0] = res

        print(f"    SS_obs = {res['SS_obs']:.1f}")
        print(f"    Global:  z = {res['z_global']:+.1f}, p = {res['p_global']:.4f}")
        print(f"    Shuffle: z = {res['z_shuffle']:+.1f}, p = {res['p_shuffle']:.4f}")
        print(f"    Shuffle null mean = {res['null_shuffle_mean']:.1f} ± {res['null_shuffle_std']:.1f}")
        print(f"    Band peak counts: {res['band_info']}")

        shuffle_rows.append({
            'f0': f0, 'base': 'phi', 'n_peaks': res['n_peaks'],
            'SS_obs': res['SS_obs'],
            'z_global': res['z_global'], 'p_global': res['p_global'],
            'z_shuffle': res['z_shuffle'], 'p_shuffle': res['p_shuffle'],
            'null_shuffle_mean': res['null_shuffle_mean'],
            'null_shuffle_std': res['null_shuffle_std'],
            'null_global_mean': res['null_global_mean'],
            'null_global_std': res['null_global_std'],
            **{f'E_{k}': v for k, v in res['enrichments_obs'].items()},
        })

    shuffle_df = pd.DataFrame(shuffle_rows)
    shuffle_df.to_csv(os.path.join(OUTPUT_DIR, 'shuffle_test_results.csv'), index=False)
    print(f"\n  Saved: shuffle_test_results.csv")

    fig_shuffle_test(results_by_f0, os.path.join(FIG_DIR, 'fig_shuffle_test.png'))

    # --- PHASE 2: Per-band leave-one-out (f0=8.5 only) ---
    print("\n" + "="*70)
    print("PHASE 2: Per-band leave-one-out (f0=8.5)")
    print("="*70)

    loo_df = per_band_leave_one_out(freqs, 8.5, PHI, N_PERM_LOO, SEED, freq_ceil=50.0)
    print(loo_df.to_string(index=False))
    loo_df.to_csv(os.path.join(OUTPUT_DIR, 'band_leave_one_out.csv'), index=False)

    fig_band_contributions(loo_df, 8.5, os.path.join(FIG_DIR, 'fig_band_contributions.png'))

    # --- PHASE 3: Prominence decomposition (f0=8.5) ---
    print("\n" + "="*70)
    print("PHASE 3: Prominence decomposition (f0=8.5)")
    print("="*70)

    print("\n  Aggregate (10 deciles)...")
    prom_agg = prominence_decomposition(freqs, powers, 8.5, PHI, n_deciles=10,
                                         n_bootstrap=N_BOOTSTRAP, seed=SEED)
    print(prom_agg[['decile', 'n_peaks', 'SS', 'SS_ci_lo', 'SS_ci_hi']].to_string(index=False))
    prom_agg.to_csv(os.path.join(OUTPUT_DIR, 'prominence_aggregate.csv'), index=False)

    print("\n  Per-band (5 deciles)...")
    prom_band = prominence_decomposition_per_band(freqs, powers, 8.5, PHI,
                                                    n_deciles=5, n_bootstrap=N_BOOTSTRAP,
                                                    seed=SEED, freq_ceil=50.0)
    print(prom_band[['band', 'decile', 'n_peaks', 'SS', 'SS_ci_lo', 'SS_ci_hi']].to_string(index=False))
    prom_band.to_csv(os.path.join(OUTPUT_DIR, 'prominence_per_band.csv'), index=False)

    fig_prominence(prom_agg, prom_band, 8.5, os.path.join(FIG_DIR, 'fig_prominence_decomposition.png'))

    # --- PHASE 4: Multi-base shuffle comparison ---
    print("\n" + "="*70)
    print("PHASE 4: Multi-base shuffle comparison (9 bases × 2 f0)")
    print("="*70)

    multi_df = multi_base_shuffle_test(freqs, F0_VALUES, BASES, N_PERM_SHUFFLE,
                                        SEED, freq_ceil=50.0)
    multi_df.to_csv(os.path.join(OUTPUT_DIR, 'multi_base_shuffle.csv'), index=False)

    # Print ranking tables
    for f0 in F0_VALUES:
        sub = multi_df[multi_df['f0'] == f0].sort_values('z_shuffle', ascending=False)
        print(f"\n  Rankings at f0={f0} Hz:")
        print(f"  {'Base':>6s} {'SS_obs':>8s} {'z_global':>9s} {'z_shuffle':>10s} {'p_shuffle':>10s}")
        for _, row in sub.iterrows():
            print(f"  {row['base_label']:>6s} {row['SS_obs']:+8.1f} "
                  f"{row['z_global']:+9.1f} {row['z_shuffle']:+10.1f} "
                  f"{row['p_shuffle']:10.4f}")

    # --- SUMMARY ---
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"DONE in {elapsed:.0f} seconds")
    print(f"Output: {OUTPUT_DIR}/")
    print(f"{'='*70}")

    # Key results summary
    print("\n=== KEY RESULTS ===")
    for f0 in F0_VALUES:
        r = results_by_f0[f0]
        print(f"\nf0={f0}: SS={r['SS_obs']:.1f}")
        print(f"  Global null:  z={r['z_global']:+.1f}, p={r['p_global']:.4f}")
        print(f"  Shuffle null: z={r['z_shuffle']:+.1f}, p={r['p_shuffle']:.4f}")
        verdict = "SURVIVES" if r['p_shuffle'] < 0.05 else "FAILS"
        print(f"  Verdict: {verdict} within-band shuffle")

    # Prominence summary
    valid_prom = prom_agg.dropna(subset=['SS'])
    if len(valid_prom) >= 2:
        r_prom, p_prom = stats.spearmanr(valid_prom['decile'], valid_prom['SS'])
        print(f"\nProminence-SS correlation: rho={r_prom:.3f}, p={p_prom:.4f}")
        trend = "INCREASING" if r_prom > 0.3 else ("DECREASING" if r_prom < -0.3 else "FLAT")
        print(f"  Trend: {trend} (strongest peaks {'more' if r_prom > 0 else 'less'} lattice-aligned)")


if __name__ == '__main__':
    main()

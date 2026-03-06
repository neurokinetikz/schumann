#!/usr/bin/env python3
"""
Phase 7: Base Specificity on LEMON
===================================

Tests whether the phi-lattice age and cognitive effects appear at alternative
exponential bases. Key question: is RWT ~ z_alpha (the sole FDR survivor)
phi-specific, or does any reasonable band decomposition work?

Two tests per base:
  (A) Global within-band shuffle z (using base-specific octave bands)
      → correlate with age
  (B) Fixed-alpha z (peaks in [f0, f0*phi] Hz, same peaks for all bases)
      → correlate with RWT (partial r controlling age, and within-young)

Usage:
    python scripts/run_lemon_base_specificity.py
"""

import os
import sys
import time
import logging
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from ratio_specificity import lattice_coordinate, PHI
from structural_phi_specificity import natural_positions
from continuous_compliance import (
    SIGMA_DEFAULT,
    weighted_structural_score,
    _apply_weight_transform,
    null_kernel_density,
)

warnings.filterwarnings('ignore', category=RuntimeWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# Paths
EXPORTS = os.path.join(os.path.dirname(__file__), '..', 'exports_lemon')
PEAKS_85HZ = os.path.join(EXPORTS, 'per_subject_85hz')
OUT_DIR = os.path.join(EXPORTS, 'amplitude_weighted')
FIG_DIR = os.path.join(OUT_DIR, 'figures')

SIGMA = SIGMA_DEFAULT
F0 = 8.5

# Bases to test
BASES = {
    '1.4':  1.4,
    '√2':   np.sqrt(2),
    '3/2':  1.5,
    'φ':    PHI,
    '1.7':  1.7,
    '1.8':  1.8,
    '2':    2.0,
    'e':    np.e,
    'π':    np.pi,
}

# Phi-alpha band (fixed Hz range for all bases)
ALPHA_LO = F0
ALPHA_HI = F0 * PHI  # ~13.75 Hz


def make_base_bands(f0, freq_ceil, base):
    """Create octave bands for an arbitrary exponential base.

    Each band spans one full base-octave: [f0*base^n, f0*base^(n+1)].
    Starts below f0 and goes up to freq_ceil.
    """
    bands = {}
    # Go down: sub-bass octaves
    lo = f0 / base
    if lo >= 1.0:
        bands['sub_alpha'] = (lo, f0)

    # Go up
    names = ['alpha_eq', 'beta_eq', 'gamma_eq', 'high_gamma_eq']
    for i, name in enumerate(names):
        lo = f0 * (base ** i)
        hi = f0 * (base ** (i + 1))
        if lo >= freq_ceil:
            break
        hi = min(hi, freq_ceil)
        bands[name] = (lo, hi)
        if hi >= freq_ceil:
            break

    return bands


def load_peaks(sid):
    """Load per-subject peak CSV from 85 Hz extraction."""
    path = os.path.join(PEAKS_85HZ, f'{sid}_peaks.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty or 'freq' not in df.columns:
        return None
    return df


def available_subjects():
    """List subject IDs with 85 Hz peak CSVs."""
    sids = []
    for f in sorted(os.listdir(PEAKS_85HZ)):
        if f.endswith('_peaks.csv') and f.startswith('sub-'):
            sid = f.replace('_peaks.csv', '')
            sids.append(sid)
    return sids


def within_band_shuffle_at_base(freqs, powers, f0, base, sigma, n_perm=1000,
                                 seed=42, freq_ceil=85.0):
    """Within-band shuffle z-score at an arbitrary base.

    Uses base-specific octave bands and base-specific lattice positions.
    """
    freqs = np.asarray(freqs, dtype=float)
    powers = np.asarray(powers, dtype=float)

    mask = (freqs > 0) & np.isfinite(freqs) & np.isfinite(powers)
    freqs = freqs[mask]
    powers = powers[mask]

    if len(freqs) == 0:
        return np.nan

    u = lattice_coordinate(freqs, f0, base)
    valid = np.isfinite(u)
    u = u[valid]
    freqs_v = freqs[valid]
    powers_v = powers[valid]

    if len(u) == 0:
        return np.nan

    w = _apply_weight_transform(powers_v, 'rank')
    positions = natural_positions(base)

    # Observed
    ss_obs, _ = weighted_structural_score(u, positions, sigma, w)

    # Assign bands
    bands = make_base_bands(f0, freq_ceil, base)
    band_labels = np.full(len(freqs_v), -1, dtype=int)
    band_list = list(bands.items())
    for bi, (bname, (lo, hi)) in enumerate(band_list):
        in_band = (freqs_v >= lo) & (freqs_v < hi)
        band_labels[in_band] = bi

    # Assign orphans to nearest band
    outside = band_labels == -1
    if outside.any():
        for idx in np.where(outside)[0]:
            f = freqs_v[idx]
            best_dist = np.inf
            best_bi = 0
            for bi, (bname, (lo, hi)) in enumerate(band_list):
                d = abs(f - (lo + hi) / 2)
                if d < best_dist:
                    best_dist = d
                    best_bi = bi
            band_labels[idx] = best_bi

    # Null
    rng = np.random.default_rng(seed)
    null_scores = np.zeros(n_perm)
    for pi in range(n_perm):
        u_shuf = u.copy()
        for bi in range(len(band_list)):
            in_band = band_labels == bi
            if in_band.sum() > 0:
                u_shuf[in_band] = rng.uniform(0, 1, size=in_band.sum())
        null_scores[pi], _ = weighted_structural_score(
            u_shuf, positions, sigma, w)

    null_mean = null_scores.mean()
    null_std = null_scores.std()
    z = (ss_obs - null_mean) / null_std if null_std > 0 else 0.0
    return z


def alpha_band_z_at_base(freqs, powers, f0, base, sigma, n_perm=1000,
                          seed=42):
    """Shuffle z-score for peaks in the fixed phi-alpha Hz range [f0, f0*phi].

    Same peaks for all bases — only the lattice mapping and positions change.
    Since all peaks are in one band, shuffle = uniform [0, 1) for all.
    """
    freqs = np.asarray(freqs, dtype=float)
    powers = np.asarray(powers, dtype=float)

    # Select alpha-range peaks
    mask = ((freqs >= ALPHA_LO) & (freqs < ALPHA_HI) &
            np.isfinite(freqs) & np.isfinite(powers))
    freqs = freqs[mask]
    powers = powers[mask]

    if len(freqs) < 3:
        return np.nan

    u = lattice_coordinate(freqs, f0, base)
    valid = np.isfinite(u)
    u = u[valid]
    powers_v = powers[valid]

    if len(u) < 3:
        return np.nan

    w = _apply_weight_transform(powers_v, 'rank')
    positions = natural_positions(base)

    ss_obs, _ = weighted_structural_score(u, positions, sigma, w)

    # Single-band shuffle: all peaks get uniform [0,1)
    rng = np.random.default_rng(seed)
    null_scores = np.zeros(n_perm)
    for pi in range(n_perm):
        u_shuf = rng.uniform(0, 1, size=len(u))
        null_scores[pi], _ = weighted_structural_score(
            u_shuf, positions, sigma, w)

    null_mean = null_scores.mean()
    null_std = null_scores.std()
    z = (ss_obs - null_mean) / null_std if null_std > 0 else 0.0
    return z


def partial_corr(x, y, z):
    """Partial correlation r(x,y | z). Returns (r, p, n)."""
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    n = len(x)
    if n < 10:
        return np.nan, np.nan, n

    # Residualize
    from numpy.polynomial.polynomial import polyfit, polyval
    cx = np.polyfit(z, x, 1)
    cy = np.polyfit(z, y, 1)
    rx = x - np.polyval(cx, z)
    ry = y - np.polyval(cy, z)
    r, p = stats.pearsonr(rx, ry)
    return r, p, n


def main():
    t_start = time.time()
    os.makedirs(FIG_DIR, exist_ok=True)

    # Load behavioral data
    sf = pd.read_csv(os.path.join(EXPORTS, 'subject_features.csv'))
    mb = pd.read_csv(os.path.join(EXPORTS, 'master_behavioral.csv'))
    features = sf.merge(mb, on='subject_id', how='inner')
    features = features[features['excluded'] != True].copy()
    log.info(f"Loaded {len(features)} subjects")

    sids = available_subjects()
    log.info(f"Found {len(sids)} subjects with 85 Hz peaks")

    n_perm = 1000

    # Benchmark
    test_peaks = load_peaks(sids[0])
    t0 = time.time()
    _ = within_band_shuffle_at_base(
        test_peaks['freq'].values, test_peaks['power'].values,
        F0, PHI, SIGMA, n_perm=100)
    bench = time.time() - t0
    est = bench * (n_perm / 100) * len(sids) * len(BASES)
    log.info(f"Benchmark: {bench:.2f}s for 100 perms, 1 subj, 1 base")
    log.info(f"Estimated total: {est/60:.1f} min for {len(sids)} subj × "
             f"{len(BASES)} bases × {n_perm} perms")

    # ========================================================================
    # Compute z-scores at each base
    # ========================================================================
    results = {bname: [] for bname in BASES}
    alpha_results = {bname: [] for bname in BASES}

    for i, sid in enumerate(sids):
        peaks = load_peaks(sid)
        if peaks is None:
            for bname in BASES:
                results[bname].append({'subject_id': sid, 'z_global': np.nan})
                alpha_results[bname].append({'subject_id': sid, 'z_alpha': np.nan})
            continue

        freqs = peaks['freq'].values
        powers = peaks['power'].values

        for bname, bval in BASES.items():
            # Global z
            z_g = within_band_shuffle_at_base(
                freqs, powers, F0, bval, SIGMA, n_perm=n_perm, seed=42)
            results[bname].append({'subject_id': sid, 'z_global': z_g})

            # Fixed-alpha z
            z_a = alpha_band_z_at_base(
                freqs, powers, F0, bval, SIGMA, n_perm=n_perm, seed=42)
            alpha_results[bname].append({'subject_id': sid, 'z_alpha': z_a})

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t_start
            log.info(f"  [{i+1}/{len(sids)}] {elapsed/60:.1f} min elapsed")

    # ========================================================================
    # Assemble per-base DataFrames and test correlations
    # ========================================================================
    summary_rows = []

    for bname, bval in BASES.items():
        df_g = pd.DataFrame(results[bname])
        df_a = pd.DataFrame(alpha_results[bname])
        df = df_g.merge(df_a, on='subject_id')
        df = df.merge(features[['subject_id', 'age_midpoint', 'age_group', 'RWT']],
                       on='subject_id', how='left')

        n_positions = len(natural_positions(bval))

        # (A) z_global ~ age
        valid = df.dropna(subset=['age_midpoint', 'z_global'])
        if len(valid) > 10:
            r_age, p_age = stats.pearsonr(valid['age_midpoint'], valid['z_global'])
        else:
            r_age, p_age = np.nan, np.nan

        # (B) z_alpha ~ RWT (partial, controlling age)
        valid2 = df.dropna(subset=['age_midpoint', 'z_alpha', 'RWT'])
        if len(valid2) > 10:
            r_rwt_partial, p_rwt_partial, n_rwt = partial_corr(
                valid2['z_alpha'].values, valid2['RWT'].values,
                valid2['age_midpoint'].values)
        else:
            r_rwt_partial, p_rwt_partial, n_rwt = np.nan, np.nan, 0

        # (C) z_alpha ~ RWT within young only
        young = df[(df['age_group'] == 'young')].dropna(
            subset=['z_alpha', 'RWT'])
        if len(young) > 10:
            r_rwt_young, p_rwt_young = stats.pearsonr(
                young['z_alpha'], young['RWT'])
        else:
            r_rwt_young, p_rwt_young = np.nan, np.nan

        # (D) z_global ~ RWT partial
        valid3 = df.dropna(subset=['age_midpoint', 'z_global', 'RWT'])
        if len(valid3) > 10:
            r_rwt_global_partial, p_rwt_global_partial, _ = partial_corr(
                valid3['z_global'].values, valid3['RWT'].values,
                valid3['age_midpoint'].values)
        else:
            r_rwt_global_partial, p_rwt_global_partial = np.nan, np.nan

        # Fraction individually significant
        n_sig = (df['z_global'].dropna() > 1.96).sum()
        n_total = df['z_global'].dropna().shape[0]
        frac_sig = n_sig / n_total if n_total > 0 else 0

        # Alpha-eq band Hz range for this base
        alpha_eq_lo = F0
        alpha_eq_hi = F0 * bval

        row = {
            'base_name': bname,
            'base_value': round(bval, 4),
            'n_positions': n_positions,
            'alpha_eq_range': f'{alpha_eq_lo:.1f}-{alpha_eq_hi:.1f}',
            'z_global_age_r': round(r_age, 3),
            'z_global_age_p': round(p_age, 4),
            'z_alpha_RWT_partial_r': round(r_rwt_partial, 3) if not np.isnan(r_rwt_partial) else np.nan,
            'z_alpha_RWT_partial_p': round(p_rwt_partial, 4) if not np.isnan(p_rwt_partial) else np.nan,
            'z_alpha_RWT_young_r': round(r_rwt_young, 3) if not np.isnan(r_rwt_young) else np.nan,
            'z_alpha_RWT_young_p': round(p_rwt_young, 4) if not np.isnan(p_rwt_young) else np.nan,
            'z_global_RWT_partial_r': round(r_rwt_global_partial, 3) if not np.isnan(r_rwt_global_partial) else np.nan,
            'z_global_RWT_partial_p': round(p_rwt_global_partial, 4) if not np.isnan(p_rwt_global_partial) else np.nan,
            'frac_individually_sig': round(frac_sig, 3),
            'mean_z_global': round(df['z_global'].mean(), 3),
        }
        summary_rows.append(row)
        log.info(f"  {bname:>4}: z~age r={r_age:+.3f} p={p_age:.4f} | "
                 f"z_alpha~RWT partial r={r_rwt_partial:+.3f} p={p_rwt_partial:.4f} | "
                 f"young r={r_rwt_young:+.3f} p={p_rwt_young:.4f}")

    summary_df = pd.DataFrame(summary_rows)

    # FDR on the alpha-RWT partial p-values
    valid_ps = summary_df['z_alpha_RWT_partial_p'].dropna()
    if len(valid_ps) > 1:
        _, p_fdr, _, _ = multipletests(valid_ps.values, alpha=0.05, method='fdr_bh')
        summary_df.loc[valid_ps.index, 'z_alpha_RWT_partial_p_fdr'] = p_fdr

    # FDR on age p-values
    valid_age_ps = summary_df['z_global_age_p'].dropna()
    if len(valid_age_ps) > 1:
        _, p_fdr_age, _, _ = multipletests(valid_age_ps.values, alpha=0.05, method='fdr_bh')
        summary_df.loc[valid_age_ps.index, 'z_global_age_p_fdr'] = p_fdr_age

    # Save
    out_path = os.path.join(OUT_DIR, 'base_specificity.csv')
    summary_df.to_csv(out_path, index=False)
    log.info(f"Saved {out_path}")

    # ========================================================================
    # Print formatted table
    # ========================================================================
    print("\n" + "=" * 100)
    print("BASE SPECIFICITY RESULTS")
    print("=" * 100)

    print(f"\n{'Base':>6} {'Val':>6} {'#Pos':>4} | "
          f"{'z~age r':>8} {'p':>8} {'FDR':>8} | "
          f"{'z_α~RWT r':>10} {'p':>8} {'FDR':>8} | "
          f"{'young r':>8} {'p':>8} | "
          f"{'%sig':>5} {'mean_z':>7}")
    print("-" * 100)

    for _, row in summary_df.sort_values('base_value').iterrows():
        phi_marker = ' ←' if row['base_name'] == 'φ' else ''
        p_fdr_age = row.get('z_global_age_p_fdr', np.nan)
        p_fdr_rwt = row.get('z_alpha_RWT_partial_p_fdr', np.nan)
        print(f"{row['base_name']:>6} {row['base_value']:>6.3f} {row['n_positions']:>4} | "
              f"{row['z_global_age_r']:>+8.3f} {row['z_global_age_p']:>8.4f} "
              f"{p_fdr_age:>8.4f} | "
              f"{row['z_alpha_RWT_partial_r']:>+10.3f} {row['z_alpha_RWT_partial_p']:>8.4f} "
              f"{p_fdr_rwt:>8.4f} | "
              f"{row['z_alpha_RWT_young_r']:>+8.3f} {row['z_alpha_RWT_young_p']:>8.4f} | "
              f"{row['frac_individually_sig']:>5.1%} {row['mean_z_global']:>+7.2f}"
              f"{phi_marker}")

    # ========================================================================
    # Figure: base comparison
    # ========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sorted_df = summary_df.sort_values('base_value')
    x = range(len(sorted_df))
    labels = sorted_df['base_name'].values

    # Panel 1: z_global ~ age
    ax = axes[0]
    colors = ['tab:orange' if n == 'φ' else 'steelblue' for n in labels]
    ax.bar(x, sorted_df['z_global_age_r'].values, color=colors)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel('r(z_global, age)')
    ax.set_title('Age correlation by base')

    # Panel 2: z_alpha ~ RWT (partial)
    ax = axes[1]
    vals = sorted_df['z_alpha_RWT_partial_r'].values
    ax.bar(x, vals, color=colors)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel('partial r(z_alpha, RWT | age)')
    ax.set_title('RWT prediction by base (partial)')

    # Panel 3: z_alpha ~ RWT within young
    ax = axes[2]
    vals_y = sorted_df['z_alpha_RWT_young_r'].values
    ax.bar(x, vals_y, color=colors)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel('r(z_alpha, RWT) within young')
    ax.set_title('RWT prediction within young')

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig_base_specificity.png')
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved {fig_path}")

    total = time.time() - t_start
    log.info(f"Total runtime: {total/60:.1f} min")


if __name__ == '__main__':
    main()

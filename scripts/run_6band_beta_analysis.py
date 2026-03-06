#!/usr/bin/env python3
"""
6-Band Dominant-Peak Lattice Analysis — Including Beta Sub-Bands
================================================================

Adds beta_low (13-20 Hz, phi-octave n=+1) and beta_high (20-32 Hz, phi-octave n=+2)
to the standard 4-band dominant-peak analysis. Uses existing overlap-trim peak CSVs
from EEGMMIDB and LEMON — no re-extraction needed.

Reports both 4-band (original) and 6-band (with beta) results for comparison.

Data sources:
  EEGMMIDB: exports_eegmmidb/per_subject_overlap_trim_f07.83/S*_peaks.csv (N=109)
  LEMON:    exports_lemon/per_subject_overlap_trim_f07.83/sub-*_peaks.csv (N=202)
"""

import sys
import os
import glob
import numpy as np
import pandas as pd
from scipy import stats
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from phi_replication import (
    F0, PHI,
    POSITIONS_DEG2, PHI_POSITIONS, POSITIONS_14, BASES,
    KDE_BANDWIDTH, N_PERMUTATIONS, AGE_BINS,
    lattice_coord, circ_dist, min_lattice_dist,
    nearest_position_name, positions_for_base,
    density_at_position,
)

# ══════════════════════════════════════════════════════════════════════════
# BAND DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════

BANDS_4 = {
    'delta': (1.0, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'gamma': (30.0, 45.0),
}

BANDS_6 = {
    'delta':     (1.0,  4.0),
    'theta':     (4.0,  8.0),
    'alpha':     (8.0,  13.0),
    'beta_low':  (13.0, 20.0),   # phi-octave n=+1 (12.3-19.9 Hz)
    'beta_high': (20.0, 32.0),   # phi-octave n=+2 (19.9-32.2 Hz)
    'gamma':     (32.0, 45.0),   # adjusted from (30,45) for contiguity
}

# ══════════════════════════════════════════════════════════════════════════
# DATA PATHS
# ══════════════════════════════════════════════════════════════════════════

DATASETS = {
    'EEGMMIDB': {
        'peak_dir': 'exports_eegmmidb/per_subject_overlap_trim_f07.83',
        'pattern': 'S*_peaks.csv',
        'condition': 'combined (R01+R02)',
    },
    'LEMON': {
        'peak_dir': 'exports_lemon/per_subject_overlap_trim_f07.83',
        'pattern': 'sub-*_peaks.csv',
        'condition': 'EC',
    },
}

OUT_DIR = 'exports_6band'


# ══════════════════════════════════════════════════════════════════════════
# DOMINANT PEAK COMPUTATION
# ══════════════════════════════════════════════════════════════════════════

def compute_dominant_peaks(peaks_df, bands, subject_id=None):
    """Extract strongest peak per band, compute lattice coordinates.

    Parameters
    ----------
    peaks_df : DataFrame with columns [freq, power, ...]
    bands : dict of {band_name: (lo, hi)}
    subject_id : str

    Returns
    -------
    dict with per-band freq/power/u/d/nearest, mean_d, n_bands
    """
    if len(peaks_df) == 0:
        return None

    row = {'subject': subject_id}
    n_bands = 0

    for band_name, (lo, hi) in bands.items():
        bp = peaks_df[(peaks_df['freq'] >= lo) & (peaks_df['freq'] < hi)]
        if len(bp) > 0:
            idx = bp['power'].idxmax()
            freq = bp.loc[idx, 'freq']
            power = bp.loc[idx, 'power']
            u = lattice_coord(freq)
            d = min_lattice_dist(u, POSITIONS_DEG2)
            nearest = nearest_position_name(u, POSITIONS_DEG2)

            row[f'{band_name}_freq'] = freq
            row[f'{band_name}_power'] = power
            row[f'{band_name}_u'] = u
            row[f'{band_name}_d'] = d
            row[f'{band_name}_nearest'] = nearest
            n_bands += 1
        else:
            row[f'{band_name}_freq'] = np.nan
            row[f'{band_name}_power'] = np.nan
            row[f'{band_name}_u'] = np.nan
            row[f'{band_name}_d'] = np.nan
            row[f'{band_name}_nearest'] = 'none'

    row['n_bands'] = n_bands
    ds = [row[f'{b}_d'] for b in bands
          if not np.isnan(row.get(f'{b}_d', np.nan))]
    row['mean_d'] = np.mean(ds) if ds else np.nan

    return row


def load_dataset(peak_dir, pattern):
    """Load all per-subject peak CSVs from a directory."""
    base = os.path.join(os.path.dirname(__file__), '..', peak_dir)
    files = sorted(glob.glob(os.path.join(base, pattern)))
    subjects = []
    for f in files:
        basename = os.path.basename(f)
        sub_id = basename.replace('_peaks.csv', '')
        peaks = pd.read_csv(f)
        subjects.append((sub_id, peaks))
    return subjects


# ══════════════════════════════════════════════════════════════════════════
# STATISTICS SUITE
# ══════════════════════════════════════════════════════════════════════════

def run_statistics(dom_df, bands, label=''):
    """Full statistical suite, parameterized on band definitions."""
    n_bands_expected = len(bands)
    band_names = list(bands.keys())

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  N={len(dom_df)} subjects, {n_bands_expected} bands")
    print(f"{'='*70}")

    # Filter to subjects with all bands
    valid = dom_df[dom_df['n_bands'] == n_bands_expected].copy()
    n_valid = len(valid)
    n_total = len(dom_df)

    print(f"\n  Complete {n_bands_expected}-band data: {n_valid}/{n_total} subjects")
    for band_name in bands:
        n_with = dom_df[f'{band_name}_freq'].notna().sum()
        print(f"    {band_name}: {n_with}/{n_total} subjects with peak")

    # Fallback if few complete subjects
    min_complete = n_bands_expected - 1
    if n_valid < 30:
        valid_fallback = dom_df[dom_df['n_bands'] >= min_complete].copy()
        print(f"  (Fallback to >={min_complete} bands: {len(valid_fallback)}/{n_total})")
        analysis_df = valid_fallback
    else:
        analysis_df = valid

    if len(analysis_df) < 10:
        print("  WARNING: Too few subjects for analysis!")
        return {}

    results = {'label': label, 'n_total': n_total, 'n_valid': len(analysis_df),
               'n_bands': n_bands_expected}

    # Recompute lattice coordinates
    for band_name in bands:
        freq_col = f'{band_name}_freq'
        if freq_col in analysis_df.columns:
            analysis_df[f'{band_name}_u'] = analysis_df[freq_col].apply(
                lambda f: lattice_coord(f) if pd.notna(f) else np.nan)
            analysis_df[f'{band_name}_d'] = analysis_df[f'{band_name}_u'].apply(
                lambda u: min_lattice_dist(u, POSITIONS_DEG2) if pd.notna(u) else np.nan)
            analysis_df[f'{band_name}_nearest'] = analysis_df[f'{band_name}_u'].apply(
                lambda u: nearest_position_name(u, POSITIONS_DEG2) if pd.notna(u) else 'none')

    # Recompute mean_d
    band_d_cols = [f'{b}_d' for b in bands if f'{b}_d' in analysis_df.columns]
    analysis_df['mean_d'] = analysis_df[band_d_cols].mean(axis=1)

    # ── 1. Population-level alignment ──
    print(f"\n  --- Population-Level Alignment (f0={F0} Hz) ---")
    pop_freqs = {}
    for band_name in bands:
        freqs = analysis_df[f'{band_name}_freq'].dropna()
        if len(freqs) > 0:
            med = freqs.median()
            pop_freqs[band_name] = med
            u = lattice_coord(med)
            d = min_lattice_dist(u, POSITIONS_DEG2)
            near = nearest_position_name(u, POSITIONS_DEG2)
            print(f"    {band_name:>10s}: median={med:.2f} Hz, u={u:.3f}, "
                  f"d={d:.3f} -> {near}")

    pop_ds = [min_lattice_dist(lattice_coord(f), POSITIONS_DEG2)
              for f in pop_freqs.values()]
    pop_mean_d = np.mean(pop_ds)
    print(f"    Population mean_d = {pop_mean_d:.4f}")

    # Null: 100K permutations
    n_perm = 100_000
    null_pop = np.zeros(n_perm)
    for i in range(n_perm):
        ds = []
        for band_name, (lo, hi) in bands.items():
            if band_name in pop_freqs:
                ds.append(min_lattice_dist(
                    lattice_coord(np.random.uniform(lo, hi)), POSITIONS_DEG2))
        null_pop[i] = np.mean(ds)
    p_pop = (null_pop <= pop_mean_d).mean()
    print(f"    Uniform null: p = {p_pop:.4f}")
    results['pop_mean_d'] = pop_mean_d
    results['p_pop'] = p_pop

    # ── 2. Per-subject alignment ──
    print(f"\n  --- Per-Subject Alignment ---")
    mean_ds = analysis_df['mean_d'].values
    obs_mean = mean_ds.mean()
    obs_sd = mean_ds.std()

    null_expected = np.mean([min_lattice_dist(np.random.uniform(0, 1), POSITIONS_DEG2)
                             for _ in range(100_000)])

    t_stat, p_ttest = stats.ttest_1samp(mean_ds, null_expected)
    cohen_d = (null_expected - obs_mean) / obs_sd if obs_sd > 0 else 0.0

    try:
        w_stat, p_wilcox = stats.wilcoxon(mean_ds - null_expected, alternative='less')
    except Exception:
        p_wilcox = np.nan

    print(f"    Observed mean_d: {obs_mean:.4f} +/- {obs_sd:.4f}")
    print(f"    Expected (null): {null_expected:.4f}")
    print(f"    t = {t_stat:.2f}, p = {p_ttest:.2e}")
    print(f"    Wilcoxon p = {p_wilcox:.2e}")
    print(f"    Cohen's d = {cohen_d:.3f}")

    # Count individually significant
    n_sig = 0
    null_ref = np.array([np.mean([min_lattice_dist(np.random.uniform(0, 1), POSITIONS_DEG2)
                                   for _ in range(n_bands_expected)])
                          for _ in range(10_000)])
    p5 = np.percentile(null_ref, 5)
    for md in mean_ds:
        if md <= p5:
            n_sig += 1
    print(f"    Individually significant (p<0.05): {n_sig}/{len(analysis_df)} "
          f"({100*n_sig/len(analysis_df):.1f}% vs 5% expected)")

    results['obs_mean_d'] = obs_mean
    results['null_expected_d'] = null_expected
    results['cohen_d'] = cohen_d
    results['p_ttest'] = p_ttest
    results['p_wilcox'] = p_wilcox
    results['n_individually_sig'] = n_sig

    # ── 3. Per-band breakdown ──
    print(f"\n  --- Per-Band Breakdown ---")
    for band_name, (lo, hi) in bands.items():
        ds = analysis_df[f'{band_name}_d'].dropna().values
        if len(ds) < 5:
            print(f"    {band_name:>10s}: too few peaks ({len(ds)})")
            continue
        band_null = np.array([min_lattice_dist(
            lattice_coord(np.random.uniform(lo, hi)), POSITIONS_DEG2)
                              for _ in range(10_000)])
        ks_stat, ks_p = stats.ks_2samp(ds, band_null, alternative='less')
        band_d = (band_null.mean() - ds.mean()) / ds.std() if ds.std() > 0 else 0
        nearests = analysis_df[f'{band_name}_nearest'].value_counts()
        top_pos = nearests.index[0] if len(nearests) > 0 else 'none'
        top_pct = 100 * nearests.iloc[0] / nearests.sum() if len(nearests) > 0 else 0
        print(f"    {band_name:>10s}: mean_d={ds.mean():.4f}, null={band_null.mean():.4f}, "
              f"KS p={ks_p:.3e}, d={band_d:.2f}  | top: {top_pos} ({top_pct:.0f}%)")

    # ── 4. Cross-base comparison ──
    print(f"\n  --- Cross-Base Comparison (9 bases, degree-3) ---")
    base_results = {}

    band_freqs = {}
    for band_name in bands:
        vals = analysis_df[f'{band_name}_freq'].dropna().values
        band_freqs[band_name] = vals

    for base_name, base_val in BASES.items():
        positions = positions_for_base(base_val, degree=3)
        seg_ds = []
        for _, srow in analysis_df.iterrows():
            band_ds = []
            for band_name in bands:
                freq = srow[f'{band_name}_freq']
                if pd.isna(freq):
                    continue
                u = lattice_coord(freq, f0=F0, base=base_val)
                d = min_lattice_dist(u, positions)
                band_ds.append(d)
            if band_ds:
                seg_ds.append(np.mean(band_ds))
        seg_ds = np.array(seg_ds)

        rng = np.random.RandomState(42)
        n_perm = 5000
        null_means = np.empty(n_perm)
        for perm_i in range(n_perm):
            perm_ds = []
            for band_name, freqs_arr in band_freqs.items():
                if len(freqs_arr) == 0:
                    continue
                shuffled = rng.uniform(0, 1, len(freqs_arr))
                dists = np.array([min_lattice_dist(u, positions) for u in shuffled])
                perm_ds.append(dists.mean())
            null_means[perm_i] = np.mean(perm_ds) if perm_ds else np.nan

        null_mean = np.nanmean(null_means)
        null_sd = np.nanstd(null_means)
        z_score = (null_mean - seg_ds.mean()) / null_sd if null_sd > 0 else 0.0
        p_val = np.mean(null_means <= seg_ds.mean())

        base_results[base_name] = {
            'mean_d': seg_ds.mean(), 'median_d': np.median(seg_ds),
            'sd_d': seg_ds.std(), 'n_positions': len(positions),
            'null_mean': null_mean, 'z_score': z_score, 'p_value': p_val,
            'values': seg_ds,
        }

    ranking_z = sorted(base_results.items(), key=lambda x: -x[1]['z_score'])
    print(f"\n    Rank by z-score:")
    for rank, (bname, br) in enumerate(ranking_z, 1):
        marker = ' <-' if bname == 'phi' else ''
        print(f"    {rank}. {bname:6s}: z={br['z_score']:+.2f}, mean_d={br['mean_d']:.4f}, "
              f"null={br['null_mean']:.4f}, p={br['p_value']:.4f} "
              f"({br['n_positions']} pos){marker}")

    phi_rank_z = next(i+1 for i, (name, _) in enumerate(ranking_z) if name == 'phi')
    results['phi_rank'] = phi_rank_z
    results['base_results'] = base_results

    # ── 5. Noble contribution ──
    shared_pos = {'boundary': 0.0, 'attractor': 0.5}
    shared_ds = []
    for _, srow in analysis_df.iterrows():
        band_ds = []
        for band_name in bands:
            freq = srow[f'{band_name}_freq']
            if pd.isna(freq):
                continue
            u = lattice_coord(freq)
            d = min_lattice_dist(u, shared_pos)
            band_ds.append(d)
        if band_ds:
            shared_ds.append(np.mean(band_ds))
    shared_ds = np.array(shared_ds)

    full_mean = base_results['phi']['mean_d']
    shared_mean = shared_ds.mean()
    noble_contrib = (1 - full_mean / shared_mean) * 100 if shared_mean > 0 else 0
    print(f"\n  --- Noble Position Contribution ---")
    print(f"    Full phi (deg-3): mean_d = {full_mean:.4f}")
    print(f"    Shared only (2 pos): mean_d = {shared_mean:.4f}")
    print(f"    Noble contribution: {noble_contrib:.1f}%")
    results['noble_contrib'] = noble_contrib

    return results


def run_14position_enrichment(dom_df, bands, n_perm=N_PERMUTATIONS):
    """14-position enrichment, parameterized on bands."""
    band_names = list(bands.keys())

    print(f"\n{'='*70}")
    print(f"  14-Position Enrichment Table ({len(bands)} bands)")
    print(f"{'='*70}")

    band_u = {}
    all_u = []
    for band in band_names:
        col = f'{band}_u'
        if col in dom_df.columns:
            vals = dom_df[col].dropna().values
            band_u[band] = vals
            all_u.extend(vals)
    all_u = np.array(all_u)

    if len(all_u) == 0:
        print("  WARNING: No u values found!")
        return pd.DataFrame()

    sorted_pos = sorted(POSITIONS_14.items(), key=lambda x: x[1])
    rows = []

    for pos_name, pos_val in sorted_pos:
        row = {'position': pos_name, 'u': pos_val}

        # Combined enrichment
        observed = density_at_position(all_u, pos_val)
        null_densities = np.array([density_at_position(
            np.random.uniform(0, 1, len(all_u)), pos_val) for _ in range(n_perm)])
        null_mean = null_densities.mean()
        null_std = null_densities.std()
        z_score = (observed - null_mean) / null_std if null_std > 0 else 0.0
        p_value = np.mean(null_densities >= observed)
        enrichment_pct = (observed / null_mean - 1) * 100 if null_mean > 0 else 0.0

        row['enrichment_pct'] = enrichment_pct
        row['z_score'] = z_score
        row['p_value'] = p_value

        # Per-band breakdown
        for band in band_names:
            if band in band_u and len(band_u[band]) > 0:
                obs_b = density_at_position(band_u[band], pos_val)
                null_b = np.array([density_at_position(
                    np.random.uniform(0, 1, len(band_u[band])), pos_val)
                    for _ in range(min(n_perm, 5000))])
                null_b_mean = null_b.mean()
                null_b_std = null_b.std()
                row[f'{band}_z'] = ((obs_b - null_b_mean) / null_b_std
                                    if null_b_std > 0 else 0.0)
                row[f'{band}_enrich'] = ((obs_b / null_b_mean - 1) * 100
                                         if null_b_mean > 0 else 0.0)
            else:
                row[f'{band}_z'] = np.nan
                row[f'{band}_enrich'] = np.nan

        rows.append(row)

    enrichment_df = pd.DataFrame(rows)

    # Print header
    band_abbrevs = {'delta': 'd', 'theta': 'th', 'alpha': 'a',
                    'beta_low': 'bL', 'beta_high': 'bH', 'gamma': 'g'}
    hdr_bands = '  '.join(f'{band_abbrevs.get(b, b[:2]):>5s}' for b in band_names)
    print(f"\n  {'Position':<15} {'u':>6} {'Enrich':>8} {'z':>7}   {hdr_bands}")
    sep_bands = '  '.join('-----' for _ in band_names)
    print(f"  {'-'*15} {'-'*6} {'-'*8} {'-'*7}   {sep_bands}")

    for _, r in enrichment_df.iterrows():
        band_vals = '  '.join(
            f"{r.get(f'{b}_z', np.nan):>+5.1f}" for b in band_names)
        verdict = '***' if r['z_score'] > 2 else ('* ' if r['z_score'] > 1
                  else ('vvv' if r['z_score'] < -2
                  else ('v  ' if r['z_score'] < -1 else '   ')))
        print(f"  {r['position']:<15} {r['u']:>6.3f} {r['enrichment_pct']:>+7.1f}% "
              f"{r['z_score']:>+7.1f}   {band_vals}  {verdict}")

    pullers = enrichment_df[enrichment_df['z_score'] > 1.0]
    pushers = enrichment_df[enrichment_df['z_score'] < -1.0]
    print(f"\n  Pullers (z>1): {', '.join(pullers['position'].values)}")
    print(f"  Pushers (z<-1): {', '.join(pushers['position'].values)}")

    return enrichment_df


# ══════════════════════════════════════════════════════════════════════════
# COMPARISON: 4-BAND vs 6-BAND
# ══════════════════════════════════════════════════════════════════════════

def compare_4_vs_6(dom_df_4, dom_df_6, label=''):
    """Side-by-side comparison of 4-band and 6-band results."""
    print(f"\n{'='*70}")
    print(f"  4-BAND vs 6-BAND COMPARISON: {label}")
    print(f"{'='*70}")

    # Merge on subject
    merged = dom_df_4[['subject', 'mean_d']].merge(
        dom_df_6[['subject', 'mean_d']], on='subject', suffixes=('_4', '_6'))

    print(f"\n  Matched subjects: {len(merged)}")
    print(f"  4-band mean_d: {merged['mean_d_4'].mean():.4f} +/- {merged['mean_d_4'].std():.4f}")
    print(f"  6-band mean_d: {merged['mean_d_6'].mean():.4f} +/- {merged['mean_d_6'].std():.4f}")

    r, p = stats.pearsonr(merged['mean_d_4'], merged['mean_d_6'])
    print(f"  Correlation: r={r:.3f}, p={p:.2e}")

    diff = merged['mean_d_6'] - merged['mean_d_4']
    t, p_diff = stats.ttest_rel(merged['mean_d_6'], merged['mean_d_4'])
    print(f"  Difference: {diff.mean():+.4f} +/- {diff.std():.4f} (t={t:.2f}, p={p_diff:.3e})")

    if diff.mean() > 0:
        print(f"  --> Beta bands DILUTE alignment (6-band mean_d higher)")
    else:
        print(f"  --> Beta bands STRENGTHEN alignment (6-band mean_d lower)")

    return merged


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def process_dataset(name, config):
    """Process one dataset: load peaks, compute 4-band + 6-band, run stats."""
    print(f"\n\n{'#'*70}")
    print(f"  DATASET: {name} ({config['condition']})")
    print(f"{'#'*70}")

    subjects = load_dataset(config['peak_dir'], config['pattern'])
    print(f"  Loaded {len(subjects)} subjects")

    if len(subjects) == 0:
        print("  ERROR: No peak files found!")
        return None, None

    # Compute dominant peaks for both band sets
    rows_6 = []
    rows_4 = []
    for sub_id, peaks_df in subjects:
        row_6 = compute_dominant_peaks(peaks_df, BANDS_6, subject_id=sub_id)
        row_4 = compute_dominant_peaks(peaks_df, BANDS_4, subject_id=sub_id)
        if row_6:
            rows_6.append(row_6)
        if row_4:
            rows_4.append(row_4)

    dom_df_6 = pd.DataFrame(rows_6)
    dom_df_4 = pd.DataFrame(rows_4)
    print(f"  6-band: {len(dom_df_6)} subjects, 4-band: {len(dom_df_4)} subjects")

    # Output directory
    out = os.path.join(os.path.dirname(__file__), '..', OUT_DIR, name)
    os.makedirs(out, exist_ok=True)

    # Run 6-band statistics
    stats_6 = run_statistics(dom_df_6, BANDS_6, label=f'{name} [6-band]')
    enrichment_6 = run_14position_enrichment(dom_df_6, BANDS_6)

    # Run 4-band statistics (for comparison / verification)
    stats_4 = run_statistics(dom_df_4, BANDS_4, label=f'{name} [4-band verification]')

    # 4 vs 6 comparison
    comparison = compare_4_vs_6(dom_df_4, dom_df_6, label=name)

    # Save outputs
    dom_df_6.to_csv(os.path.join(out, 'per_subject_dominant_peaks_6band.csv'), index=False)
    dom_df_4.to_csv(os.path.join(out, 'per_subject_dominant_peaks_4band.csv'), index=False)

    if len(enrichment_6) > 0:
        enrichment_6.to_csv(os.path.join(out, '14position_enrichment_6band.csv'), index=False)

    if comparison is not None:
        comparison.to_csv(os.path.join(out, '4band_vs_6band_comparison.csv'), index=False)

    # Save cross-base comparison
    if 'base_results' in stats_6:
        base_rows = []
        for bname, br in stats_6['base_results'].items():
            base_rows.append({
                'base': bname, 'mean_d': br['mean_d'], 'median_d': br['median_d'],
                'sd_d': br['sd_d'], 'n_positions': br['n_positions'],
                'z_score': br['z_score'], 'p_value': br['p_value'],
            })
        pd.DataFrame(base_rows).to_csv(
            os.path.join(out, 'cross_base_comparison_6band.csv'), index=False)

    # Save summary
    summary_rows = []
    for label_name, s in [('6-band', stats_6), ('4-band', stats_4)]:
        if s:
            summary_rows.append({
                'analysis': label_name,
                'n_valid': s.get('n_valid'),
                'obs_mean_d': s.get('obs_mean_d'),
                'null_expected_d': s.get('null_expected_d'),
                'cohen_d': s.get('cohen_d'),
                'p_ttest': s.get('p_ttest'),
                'p_wilcox': s.get('p_wilcox'),
                'phi_rank': s.get('phi_rank'),
                'pop_mean_d': s.get('pop_mean_d'),
                'p_pop': s.get('p_pop'),
                'noble_contrib': s.get('noble_contrib'),
            })
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(out, 'summary_4band_vs_6band.csv'), index=False)

    print(f"\n  Results saved to {out}/")
    return dom_df_6, dom_df_4


def main():
    t0 = time.time()
    print("6-Band Dominant-Peak Lattice Analysis")
    print(f"Bands: {list(BANDS_6.keys())}")
    print(f"f0 = {F0} Hz, base = phi = {PHI:.6f}")
    print(f"Output: {OUT_DIR}/")

    for name, config in DATASETS.items():
        process_dataset(name, config)

    elapsed = time.time() - t0
    print(f"\n\nDone! Total time: {elapsed:.0f}s")


if __name__ == '__main__':
    main()

"""
Per-position alignment analysis at each degree level.

Tests whether individual lattice positions carry significant alignment signal,
specifically whether degree-3 positions (u=0.236, u=0.764) individually show
significant peaks nearby, or whether the signal is concentrated only at the
4 core degree-2 positions.

This addresses the reviewer concern: is the position-count decomposition
(d: 0.40 → 0.28 → 0.07) evidence of genuine signal dilution, or just
the coarse-partition effect?
"""

import numpy as np
import pandas as pd
from scipy import stats
import sys
import os

# ── Constants ────────────────────────────────────────────────────────
PHI = (1 + np.sqrt(5)) / 2
F0 = 7.83
BANDS = ['delta', 'theta', 'alpha', 'gamma']

# Position sets at each degree
POSITIONS_DEG2 = {
    'boundary':  0.000,
    'noble_2':   1 - 1/PHI,       # 0.382
    'attractor': 0.500,
    'noble_1':   1/PHI,            # 0.618
}

POSITIONS_DEG3 = {
    **POSITIONS_DEG2,
    'noble_3':     (1/PHI)**3,     # 0.236
    'inv_noble_3': 1 - (1/PHI)**3, # 0.764
}

POSITIONS_DEG4 = {
    **POSITIONS_DEG3,
    'noble_4':     (1/PHI)**4,     # 0.146
    'inv_noble_4': 1 - (1/PHI)**4, # 0.854
}

POSITIONS_DEG5 = {
    **POSITIONS_DEG4,
    'noble_5':     (1/PHI)**5,     # 0.090
    'inv_noble_5': 1 - (1/PHI)**5, # 0.910
}

POSITIONS_DEG6 = {
    **POSITIONS_DEG5,
    'noble_6':     (1/PHI)**6,     # 0.056
    'inv_noble_6': 1 - (1/PHI)**6, # 0.944
}

POSITIONS_DEG7 = {
    **POSITIONS_DEG6,
    'noble_7':     (1/PHI)**7,     # 0.034
    'inv_noble_7': 1 - (1/PHI)**7, # 0.966
}


def circ_dist(a, b):
    """Circular distance on [0, 1)."""
    d = abs(a - b)
    return min(d, 1 - d)


def density_at_position(u_values, pos, bandwidth=0.03):
    """
    Compute KDE density at a specific position from observed u values.
    Returns density relative to uniform expectation (1.0 = uniform).
    """
    dists = np.array([circ_dist(u, pos) for u in u_values])
    # Gaussian kernel density at position
    weights = np.exp(-0.5 * (dists / bandwidth)**2)
    density = weights.mean()
    # Uniform expectation under same kernel
    # For uniform on [0,1), the expected kernel density at any point is
    # approximately the integral of the Gaussian kernel = bandwidth * sqrt(2*pi) / 1
    # but we normalize by comparing to shuffled null instead
    return density


def permutation_test_position(u_values, position, n_perm=10000, bandwidth=0.03):
    """
    Test whether observed u values cluster near a specific position
    more than expected by chance (uniform distribution on [0,1)).
    """
    observed = density_at_position(u_values, position, bandwidth)

    null_densities = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = np.random.uniform(0, 1, len(u_values))
        null_densities[i] = density_at_position(shuffled, position, bandwidth)

    p_value = np.mean(null_densities >= observed)
    z_score = (observed - null_densities.mean()) / null_densities.std()
    enrichment_pct = (observed / null_densities.mean() - 1) * 100

    return {
        'observed_density': observed,
        'null_mean': null_densities.mean(),
        'null_std': null_densities.std(),
        'z_score': z_score,
        'p_value': p_value,
        'enrichment_pct': enrichment_pct,
    }


def fraction_near_position(u_values, position, radius=0.05):
    """Fraction of u values within `radius` of a position."""
    dists = np.array([circ_dist(u, position) for u in u_values])
    return np.mean(dists < radius)


def per_position_analysis(df, dataset_name):
    """
    For each position at each degree, compute:
    1. What fraction of dominant peaks land near it
    2. KDE enrichment with permutation p-value
    3. Per-band breakdown
    """
    print(f"\n{'='*80}")
    print(f"DATASET: {dataset_name} (N={len(df)})")
    print(f"{'='*80}")

    # Collect all u values across bands
    all_u = []
    band_u = {}
    for band in BANDS:
        col = f'{band}_u'
        if col in df.columns:
            vals = df[col].dropna().values
            band_u[band] = vals
            all_u.extend(vals)
    all_u = np.array(all_u)

    print(f"\nTotal u values: {len(all_u)} across {len(band_u)} bands")

    # ── Per-position analysis at each degree ─────────────────────────
    for deg_name, positions in [('Degree-2 (4 pos)', POSITIONS_DEG2),
                                 ('Degree-3 (6 pos)', POSITIONS_DEG3),
                                 ('Degree-4 (8 pos)', POSITIONS_DEG4),
                                 ('Degree-5 (10 pos)', POSITIONS_DEG5),
                                 ('Degree-6 (12 pos)', POSITIONS_DEG6),
                                 ('Degree-7 (14 pos)', POSITIONS_DEG7)]:
        print(f"\n{'─'*60}")
        print(f"  {deg_name}")
        print(f"{'─'*60}")
        print(f"  {'Position':<15} {'u':>6} {'Frac<0.05':>10} {'Enrichment':>12} {'z':>8} {'p':>8}")
        print(f"  {'─'*15} {'─'*6} {'─'*10} {'─'*12} {'─'*8} {'─'*8}")

        for pos_name, pos_val in sorted(positions.items(), key=lambda x: x[1]):
            frac = fraction_near_position(all_u, pos_val, radius=0.05)
            result = permutation_test_position(all_u, pos_val, n_perm=10000)

            print(f"  {pos_name:<15} {pos_val:>6.3f} {frac:>10.3f} "
                  f"{result['enrichment_pct']:>+11.1f}% "
                  f"{result['z_score']:>8.1f} {result['p_value']:>8.4f}")

        # Report only the NEW positions at this degree
        if deg_name == 'Degree-3 (6 pos)':
            print(f"\n  ** Degree-3 additions (noble_3 at 0.236, inv_noble_3 at 0.764):")
            for pos_name in ['noble_3', 'inv_noble_3']:
                pos_val = positions[pos_name]
                # Per-band breakdown
                print(f"     {pos_name} (u={pos_val:.3f}):")
                for band in BANDS:
                    if band in band_u:
                        frac_b = fraction_near_position(band_u[band], pos_val, radius=0.05)
                        res_b = permutation_test_position(band_u[band], pos_val, n_perm=5000)
                        print(f"       {band:>8}: frac={frac_b:.3f}, "
                              f"enrichment={res_b['enrichment_pct']:+.1f}%, "
                              f"z={res_b['z_score']:.1f}, p={res_b['p_value']:.4f}")

    # ── Direct test: do degree-3 positions carry signal? ─────────────
    print(f"\n{'─'*60}")
    print(f"  CRITICAL TEST: Degree-3 positions individually significant?")
    print(f"{'─'*60}")

    # For each subject, compute distance to ONLY the two new positions
    # and test if this is smaller than uniform expectation
    deg3_new = {'noble_3': (1/PHI)**3, 'inv_noble_3': 1 - (1/PHI)**3}

    # Expected distance to nearest of 2 positions on [0,1)
    # Positions at 0.236 and 0.764 create gaps:
    # 0.236, 0.764-0.236=0.528, 1.0-0.764+0.236=0.472
    sorted_new = sorted(deg3_new.values())
    gaps = [sorted_new[0], sorted_new[1]-sorted_new[0], 1.0-sorted_new[1]+sorted_new[0]]
    # Wait, for 2 positions on the circle, there are 2 gaps
    gaps_circ = [sorted_new[1] - sorted_new[0], 1.0 - sorted_new[1] + sorted_new[0]]
    d_null_2pos = sum(g**2 for g in gaps_circ) / 4.0

    subject_d_new = []
    for _, row in df.iterrows():
        band_ds = []
        for band in BANDS:
            u_col = f'{band}_u'
            if u_col in row and pd.notna(row[u_col]):
                u = row[u_col]
                d_min = min(circ_dist(u, p) for p in deg3_new.values())
                band_ds.append(d_min)
        if len(band_ds) >= 3:
            subject_d_new.append(np.mean(band_ds))

    subject_d_new = np.array(subject_d_new)

    t_stat, t_p = stats.ttest_1samp(subject_d_new, d_null_2pos)
    cohens_d_new = (d_null_2pos - subject_d_new.mean()) / subject_d_new.std()

    print(f"  Null expectation (2 positions at 0.236, 0.764): d_null = {d_null_2pos:.4f}")
    print(f"  Observed mean d to degree-3 positions only:     d_obs  = {subject_d_new.mean():.4f}")
    print(f"  t = {t_stat:.3f}, p = {t_p:.2e}")
    print(f"  Cohen's d = {cohens_d_new:.3f}")
    print(f"  Direction: {'CLOSER than null (signal)' if subject_d_new.mean() < d_null_2pos else 'FARTHER than null (no signal)'}")

    # ── Compare: degree-2 positions alone ────────────────────────────
    sorted_d2 = sorted(POSITIONS_DEG2.values())
    gaps_d2 = []
    for i in range(len(sorted_d2)):
        next_i = (i + 1) % len(sorted_d2)
        if next_i == 0:
            gap = 1.0 - sorted_d2[i] + sorted_d2[0]
        else:
            gap = sorted_d2[next_i] - sorted_d2[i]
        gaps_d2.append(gap)
    d_null_d2 = sum(g**2 for g in gaps_d2) / 4.0

    subject_d_core = []
    for _, row in df.iterrows():
        band_ds = []
        for band in BANDS:
            u_col = f'{band}_u'
            if u_col in row and pd.notna(row[u_col]):
                u = row[u_col]
                d_min = min(circ_dist(u, p) for p in POSITIONS_DEG2.values())
                band_ds.append(d_min)
        if len(band_ds) >= 3:
            subject_d_core.append(np.mean(band_ds))

    subject_d_core = np.array(subject_d_core)
    t_stat_core, t_p_core = stats.ttest_1samp(subject_d_core, d_null_d2)
    cohens_d_core = (d_null_d2 - subject_d_core.mean()) / subject_d_core.std()

    print(f"\n  Comparison — degree-2 positions only:")
    print(f"  Null expectation (4 positions): d_null = {d_null_d2:.4f}")
    print(f"  Observed mean d:                d_obs  = {subject_d_core.mean():.4f}")
    print(f"  t = {t_stat_core:.3f}, p = {t_p_core:.2e}")
    print(f"  Cohen's d = {cohens_d_core:.3f}")

    return {
        'dataset': dataset_name,
        'N': len(df),
        'deg3_new_d_obs': subject_d_new.mean(),
        'deg3_new_d_null': d_null_2pos,
        'deg3_new_cohens_d': cohens_d_new,
        'deg3_new_p': t_p,
        'deg2_d_obs': subject_d_core.mean(),
        'deg2_d_null': d_null_d2,
        'deg2_cohens_d': cohens_d_core,
        'deg2_p': t_p_core,
    }


# ── Load datasets ───────────────────────────────────────────────────
print("Loading datasets...")

results = []

# LEMON
lemon_path = 'exports_lemon/per_subject_overlap_trim_f07.83/dominant_peak/per_subject_dominant_peaks.csv'
if os.path.exists(lemon_path):
    df_lemon = pd.read_csv(lemon_path)
    results.append(per_position_analysis(df_lemon, 'LEMON'))
else:
    print(f"LEMON not found at {lemon_path}")

# EEGMMIDB
eegmmidb_path = 'exports_eegmmidb/per_subject_overlap_trim_f07.83/dominant_peak/per_subject_dominant_peaks.csv'
if os.path.exists(eegmmidb_path):
    df_eeg = pd.read_csv(eegmmidb_path)
    results.append(per_position_analysis(df_eeg, 'EEGMMIDB'))
else:
    print(f"EEGMMIDB not found at {eegmmidb_path}")

# Dortmund EC-pre (primary condition)
dort_path = '/Volumes/T9/dortmund_data/lattice_results_ot/dortmund_ot_dominant_peaks_EyesClosed_pre.csv'
if os.path.exists(dort_path):
    df_dort = pd.read_csv(dort_path)
    results.append(per_position_analysis(df_dort, 'Dortmund EC-pre'))
else:
    print(f"Dortmund not found at {dort_path}")

# ── Summary table ───────────────────────────────────────────────────
if results:
    print(f"\n\n{'='*80}")
    print("SUMMARY: Per-Position Signal at Each Degree")
    print(f"{'='*80}")
    print(f"\n{'Dataset':<20} {'Deg-2 d':>8} {'Deg-2 d_null':>12} {'Deg-2 Cohen d':>14} {'Deg-2 p':>12}")
    print(f"{'─'*20} {'─'*8} {'─'*12} {'─'*14} {'─'*12}")
    for r in results:
        print(f"{r['dataset']:<20} {r['deg2_d_obs']:>8.4f} {r['deg2_d_null']:>12.4f} "
              f"{r['deg2_cohens_d']:>14.3f} {r['deg2_p']:>12.2e}")

    print(f"\n{'Dataset':<20} {'Deg3-new d':>10} {'Deg3-new null':>14} {'Deg3-new Cohen d':>16} {'Deg3-new p':>12}")
    print(f"{'─'*20} {'─'*10} {'─'*14} {'─'*16} {'─'*12}")
    for r in results:
        print(f"{r['dataset']:<20} {r['deg3_new_d_obs']:>10.4f} {r['deg3_new_d_null']:>14.4f} "
              f"{r['deg3_new_cohens_d']:>16.3f} {r['deg3_new_p']:>12.2e}")

    print(f"\nInterpretation:")
    for r in results:
        if r['deg3_new_cohens_d'] > 0.05 and r['deg3_new_p'] < 0.05:
            print(f"  {r['dataset']}: Degree-3 positions INDIVIDUALLY significant "
                  f"(d={r['deg3_new_cohens_d']:.3f}, p={r['deg3_new_p']:.2e})")
            print(f"    → Supports dilution interpretation over coarse-partition")
        elif r['deg3_new_cohens_d'] > 0 and r['deg3_new_p'] >= 0.05:
            print(f"  {r['dataset']}: Degree-3 positions trend toward significance "
                  f"(d={r['deg3_new_cohens_d']:.3f}, p={r['deg3_new_p']:.2e})")
            print(f"    → Ambiguous: consistent with weak signal or coarse-partition")
        else:
            print(f"  {r['dataset']}: Degree-3 positions NOT significant "
                  f"(d={r['deg3_new_cohens_d']:.3f}, p={r['deg3_new_p']:.2e})")
            print(f"    → Supports coarse-partition interpretation")

print("\nDone.")

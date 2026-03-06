#!/usr/bin/env python3
"""
Audit: Cross-Base Fairness in Dominant-Peak Lattice Analysis
=============================================================

Tests whether phi's rank-#1 status across datasets is genuine or an artifact
of asymmetric position counts in the cross-base comparison code.

Bug found: Dortmund/Bonn/overlap-trim scripts give phi 8 hardcoded positions
while other bases get 4-6 from positions_for_base(). This gives phi a large
geometric advantage (expected null d ≈ 0.032 vs 0.042-0.065).

This script:
  1a. Enumerates exact position counts per base under both approaches
  1b. Re-runs cross-base comparison with FAIR positions on existing Dortmund data
  1c. Canonical-frequency diagnostic (would textbook peaks explain the result?)
  1d. Random-position-set null (are phi's positions special?)
  1e. Band-shuffled null (do standard band definitions encode the result?)

Uses ONLY saved per-subject CSVs — no re-extraction needed.

Usage:
    python scripts/audit_crossbase_fairness.py
"""

import numpy as np
import pandas as pd
from scipy import stats
import os
import sys

# ── Constants ──────────────────────────────────────────────────────────────

PHI = (1 + np.sqrt(5)) / 2
F0 = 7.83

BANDS = {
    'delta': (1.0, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'gamma': (30.0, 45.0),
}

# The BUGGY 8-position set used in Dortmund/Bonn scripts
PHI_POSITIONS_8 = {
    'boundary':    0.000,
    'noble_4':     PHI**-4,
    'noble_3':     PHI**-3,
    'noble_2':     1 - 1/PHI,
    'attractor':   0.500,
    'noble_1':     1/PHI,
    'inv_noble_3': 1 - PHI**-3,
    'inv_noble_4': 1 - PHI**-4,
}

# The FAIR 4-position set used in EEGMMIDB
PHI_POSITIONS_4 = {
    'boundary':  0.000,
    'noble_2':   1 - 1/PHI,  # 0.382
    'attractor': 0.500,
    'noble_1':   1/PHI,      # 0.618
}

# 9 comparison bases
BASES = {
    'phi':   PHI,
    '1.4':   1.4,
    'sqrt2': np.sqrt(2),
    '3/2':   1.5,
    '1.7':   1.7,
    '1.8':   1.8,
    '2':     2.0,
    'e':     np.e,
    'pi':    np.pi,
}

# Data paths
DORTMUND_RESULTS = '/Volumes/T9/dortmund_data/lattice_results_ot'
DORTMUND_LONG = '/Volumes/T9/dortmund_data/lattice_results_longitudinal'
BONN_RESULTS = '/Volumes/T9/bonn_data/lattice_results'
EEGMMIDB_RESULTS = 'exports_eegmmidb/per_subject_overlap_trim_f07.83/dominant_peak'
LEMON_RESULTS = 'exports_lemon/per_subject_overlap_trim_f07.83/dominant_peak'


# ── Position generators ───────────────────────────────────────────────────

def positions_for_base_dortmund(base):
    """Dortmund-style: degree 3, no inverse of higher degrees. Used for non-phi bases in buggy code."""
    pos = {'boundary': 0.0, 'attractor': 0.5}
    inv = 1.0 / base
    if inv > 0.02 and abs(inv - 0.5) > 0.02:
        pos['noble'] = inv
    if abs(1 - inv) > 0.02 and abs(1 - inv - 0.5) > 0.02 and (1 - inv) > 0.02:
        pos['inv_noble'] = 1 - inv
    inv2 = inv ** 2
    if inv2 > 0.02 and abs(inv2 - 0.5) > 0.02:
        if all(abs(inv2 - v) > 0.02 for v in pos.values()):
            pos['noble_2'] = inv2
    inv3 = inv ** 3
    if inv3 > 0.02 and abs(inv3 - 0.5) > 0.02:
        if all(abs(inv3 - v) > 0.02 for v in pos.values()):
            pos['noble_3'] = inv3
    return pos


def positions_degree3_symmetric(base):
    """Degree 3 symmetric: forward AND inverse positions through degree 3.
    No special-casing for any base. This is the PRIMARY fair comparison.

    Generates: boundary, attractor, inv^k, 1-inv^k for k=1,2,3
    Filtered for uniqueness (>0.02 separation) and valid range."""
    MIN_SEP = 0.02
    inv = 1.0 / base
    pos = {'boundary': 0.0, 'attractor': 0.5}

    def _try_add(name, val):
        if val < MIN_SEP or val > 1 - MIN_SEP:
            return
        if abs(val - 0.5) < MIN_SEP:
            return
        if all(abs(val - v) > MIN_SEP for v in pos.values()):
            pos[name] = val

    # Degree 1
    _try_add('noble', inv)
    _try_add('inv_noble', 1 - inv)
    # Degree 2
    _try_add('noble_2', inv ** 2)
    _try_add('inv_noble_2', 1 - inv ** 2)
    # Degree 3
    _try_add('noble_3', inv ** 3)
    _try_add('inv_noble_3', 1 - inv ** 3)

    return pos


def positions_for_base_eegmmidb(base):
    """EEGMMIDB-style: degree 2 + inverses. For phi returns the 4-position set."""
    if abs(base - PHI) < 1e-6:
        return dict(PHI_POSITIONS_4)
    inv = 1.0 / base
    pos = {'boundary': 0.0, 'attractor': 0.5}
    if abs(inv) > 0.02 and abs(inv - 0.5) > 0.02:
        pos['noble'] = inv
    if abs(1 - inv) > 0.02 and abs(1 - inv - 0.5) > 0.02:
        pos['inv_noble'] = 1 - inv
    inv2 = inv ** 2
    if inv2 > 0.02 and abs(inv2 - 0.5) > 0.02 and all(abs(inv2 - v) > 0.02 for v in pos.values()):
        pos['noble_2'] = inv2
    inv2c = 1 - inv2
    if 0.02 < inv2c < 0.98 and abs(inv2c - 0.5) > 0.02 and all(abs(inv2c - v) > 0.02 for v in pos.values()):
        pos['inv_noble_2'] = inv2c
    return pos


def positions_degree2_universal(base):
    """Degree 2 for ALL bases (no special-casing phi). Most conservative."""
    inv = 1.0 / base
    pos = {'boundary': 0.0, 'attractor': 0.5}
    if abs(inv) > 0.02 and abs(inv - 0.5) > 0.02:
        pos['noble'] = inv
    if abs(1 - inv) > 0.02 and abs(1 - inv - 0.5) > 0.02 and (1 - inv) > 0.02:
        pos['inv_noble'] = 1 - inv
    inv2 = inv ** 2
    if inv2 > 0.02 and abs(inv2 - 0.5) > 0.02:
        if all(abs(inv2 - v) > 0.02 for v in pos.values()):
            pos['noble_2'] = inv2
    inv2c = 1 - inv2
    if 0.02 < inv2c < 0.98 and abs(inv2c - 0.5) > 0.02:
        if all(abs(inv2c - v) > 0.02 for v in pos.values()):
            pos['inv_noble_2'] = inv2c
    return pos


def positions_degree4_symmetric(base):
    """Degree 4 symmetric: forward AND inverse positions through degree 4.
    No special-casing for any base.

    Generates: boundary, attractor, inv^k, 1-inv^k for k=1,2,3,4
    Filtered for uniqueness (>0.02 separation) and valid range."""
    MIN_SEP = 0.02
    inv = 1.0 / base
    pos = {'boundary': 0.0, 'attractor': 0.5}

    def _try_add(name, val):
        if val < MIN_SEP or val > 1 - MIN_SEP:
            return
        if abs(val - 0.5) < MIN_SEP:
            return
        if all(abs(val - v) > MIN_SEP for v in pos.values()):
            pos[name] = val

    # Degree 1
    _try_add('noble', inv)
    _try_add('inv_noble', 1 - inv)
    # Degree 2
    _try_add('noble_2', inv ** 2)
    _try_add('inv_noble_2', 1 - inv ** 2)
    # Degree 3
    _try_add('noble_3', inv ** 3)
    _try_add('inv_noble_3', 1 - inv ** 3)
    # Degree 4
    _try_add('noble_4', inv ** 4)
    _try_add('inv_noble_4', 1 - inv ** 4)

    return pos


def positions_fixed_n(base, n_target, max_degree=20):
    """Generate exactly n_target positions for any base by adding degrees
    until the target count is reached. This ensures every base gets the
    SAME number of positions, eliminating the coverage confound entirely.

    Priority order: boundary, attractor, then (inv^k, 1-inv^k) for k=1,2,3,...
    Stops adding as soon as n_target is reached. If a base can't reach
    n_target (positions collapse or fall outside [0.02, 0.98]), returns
    whatever it achieved with a warning."""
    MIN_SEP = 0.02
    inv = 1.0 / base
    pos = {'boundary': 0.0, 'attractor': 0.5}

    def _try_add(name, val):
        if len(pos) >= n_target:
            return False
        if val < MIN_SEP or val > 1 - MIN_SEP:
            return False
        if abs(val - 0.5) < MIN_SEP:
            return False
        if all(abs(val - v) > MIN_SEP for v in pos.values()):
            pos[name] = val
            return True
        return False

    for k in range(1, max_degree + 1):
        if len(pos) >= n_target:
            break
        _try_add(f'noble_{k}', inv ** k)
        if len(pos) >= n_target:
            break
        _try_add(f'inv_noble_{k}', 1 - inv ** k)

    return pos


# ── Lattice math ──────────────────────────────────────────────────────────

def lattice_coord(freq, f0=F0, base=PHI):
    if freq <= 0 or f0 <= 0:
        return np.nan
    return (np.log(freq / f0) / np.log(base)) % 1.0


def min_dist(u, positions):
    if np.isnan(u):
        return np.nan
    d_min = 0.5
    for p in positions.values():
        d = abs(u - p)
        d = min(d, 1 - d)
        if d < d_min:
            d_min = d
    return d_min


def nearest_pos(u, positions):
    if np.isnan(u):
        return 'none'
    best_name = 'boundary'
    d_min = 0.5
    for name, p in positions.items():
        d = abs(u - p)
        d = min(d, 1 - d)
        if d < d_min:
            d_min = d
            best_name = name
    return best_name


def expected_d_for_positions(positions):
    """Analytical expected mean distance under uniform distribution on [0,1].
    E[d] = sum(gap_i^2 / 4) where gaps are distances between sorted adjacent positions on the circle."""
    vals = sorted(positions.values())
    n = len(vals)
    if n == 0:
        return 0.25  # uniform on [0,1]
    gaps = [vals[i+1] - vals[i] for i in range(n-1)]
    gaps.append(1.0 - vals[-1] + vals[0])  # wrap-around gap
    return sum(g**2 / 4 for g in gaps)


def compute_mean_d_for_subject(freqs_dict, base, positions):
    """Compute mean lattice distance for a subject's dominant peak frequencies."""
    ds = []
    for band_name, freq in freqs_dict.items():
        if np.isnan(freq):
            continue
        u = lattice_coord(freq, f0=F0, base=base)
        d = min_dist(u, positions)
        ds.append(d)
    if len(ds) == 4:
        return np.mean(ds)
    return np.nan


# ── Data loading ──────────────────────────────────────────────────────────

def load_dortmund_condition(results_dir, condition_csv):
    """Load a Dortmund per-subject dominant peaks CSV."""
    path = os.path.join(results_dir, condition_csv)
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    return df


def extract_freqs(df):
    """Extract per-subject frequency dicts from a dominant peaks DataFrame."""
    subjects = []
    for _, row in df.iterrows():
        freqs = {}
        for band in BANDS:
            col = f'{band}_freq'
            if col in row and pd.notna(row[col]):
                freqs[band] = row[col]
            else:
                freqs[band] = np.nan
        n_valid = sum(1 for v in freqs.values() if not np.isnan(v))
        if n_valid == 4:
            subjects.append(freqs)
    return subjects


# ═══════════════════════════════════════════════════════════════════════════
# 1a. ENUMERATE POSITION COUNTS
# ═══════════════════════════════════════════════════════════════════════════

def audit_1a_position_counts():
    print("=" * 80)
    print("1a. POSITION COUNTS PER BASE UNDER EACH APPROACH")
    print("=" * 80)

    approaches = {
        'BUGGY (Dortmund)': lambda b: PHI_POSITIONS_8 if abs(b - PHI) < 1e-6 else positions_for_base_dortmund(b),
        'DEGREE-3 SYMMETRIC': positions_degree3_symmetric,
        'DEGREE-4 SYMMETRIC': positions_degree4_symmetric,
        'FIXED N=6': lambda b: positions_fixed_n(b, 6),
        'FIXED N=8': lambda b: positions_fixed_n(b, 8),
    }

    for approach_name, pos_fn in approaches.items():
        print(f"\n  --- {approach_name} ---")
        for base_name, base_val in sorted(BASES.items(), key=lambda x: x[1]):
            positions = pos_fn(base_val)
            exp_d = expected_d_for_positions(positions)
            sorted_pos = sorted(positions.items(), key=lambda x: x[1])
            pos_str = ', '.join(f"{name}={val:.3f}" for name, val in sorted_pos)
            max_deg = max((int(n.split('_')[-1]) for n in positions if n.startswith('noble_') or n.startswith('inv_noble_')), default=0)
            print(f"    {base_name:6s} ({base_val:.4f}): {len(positions)} pos (deg≤{max_deg}), "
                  f"E[d]={exp_d:.4f}  [{pos_str}]")


# ═══════════════════════════════════════════════════════════════════════════
# 1b. FAIR CROSS-BASE RE-ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def run_crossbase(subjects_freqs, pos_fn, label=""):
    """Run cross-base comparison with a given position function.
    Returns dict of base_name -> (mean_d_array, expected_d, n_positions)."""
    results = {}
    for base_name, base_val in BASES.items():
        positions = pos_fn(base_val)
        exp_d = expected_d_for_positions(positions)

        mean_ds = []
        for freqs in subjects_freqs:
            md = compute_mean_d_for_subject(freqs, base_val, positions)
            if not np.isnan(md):
                mean_ds.append(md)

        mean_ds = np.array(mean_ds)
        results[base_name] = {
            'mean_d': mean_ds.mean() if len(mean_ds) > 0 else np.nan,
            'median_d': np.median(mean_ds) if len(mean_ds) > 0 else np.nan,
            'sd_d': mean_ds.std() if len(mean_ds) > 0 else np.nan,
            'n_positions': len(positions),
            'expected_d': exp_d,
            'norm_d': mean_ds.mean() / exp_d if len(mean_ds) > 0 and exp_d > 0 else np.nan,
            'values': mean_ds,
            'n_subjects': len(mean_ds),
        }
    return results


def print_crossbase(results, label=""):
    """Print cross-base results with ranking."""
    ranking = sorted(results.items(), key=lambda x: x[1]['mean_d'])
    print(f"\n  Ranking by raw mean_d ({label}):")
    print(f"  {'Rank':>4s}  {'Base':>6s}  {'mean_d':>8s}  {'E[d]':>8s}  {'norm_d':>8s}  {'N_pos':>5s}  {'N_subj':>6s}")
    for rank, (bname, r) in enumerate(ranking, 1):
        marker = ' <--' if bname == 'phi' else ''
        print(f"  {rank:>4d}  {bname:>6s}  {r['mean_d']:>8.4f}  {r['expected_d']:>8.4f}  "
              f"{r['norm_d']:>8.4f}  {r['n_positions']:>5d}  {r['n_subjects']:>6d}{marker}")

    # Coverage-normalized ranking
    ranking_norm = sorted(results.items(), key=lambda x: x[1]['norm_d'])
    print(f"\n  Ranking by NORMALIZED d (observed/expected) ({label}):")
    print(f"  {'Rank':>4s}  {'Base':>6s}  {'norm_d':>8s}  {'mean_d':>8s}  {'E[d]':>8s}  {'N_pos':>5s}")
    for rank, (bname, r) in enumerate(ranking_norm, 1):
        marker = ' <--' if bname == 'phi' else ''
        print(f"  {rank:>4d}  {bname:>6s}  {r['norm_d']:>8.4f}  {r['mean_d']:>8.4f}  "
              f"{r['expected_d']:>8.4f}  {r['n_positions']:>5d}{marker}")

    # Paired tests: phi vs each (raw and normalized)
    phi_vals = results['phi']['values']
    phi_exp = results['phi']['expected_d']
    print(f"\n  Paired t-tests (phi vs each, raw mean_d):")
    for bname, r in sorted(results.items(), key=lambda x: x[1]['mean_d']):
        if bname == 'phi':
            continue
        other_vals = r['values']
        if len(phi_vals) == len(other_vals) and len(phi_vals) > 5:
            t, p = stats.ttest_rel(phi_vals, other_vals, alternative='less')
            wins = (phi_vals < other_vals).mean()
            sig = '*' if p < 0.05 else ' '
            print(f"    phi vs {bname:>6s}: Δd={other_vals.mean()-phi_vals.mean():+.4f}, "
                  f"phi wins {wins*100:.1f}%, p={p:.3e} {sig}")


def audit_1b_fair_crossbase():
    print("\n" + "=" * 80)
    print("1b. FAIR CROSS-BASE RE-ANALYSIS ON EXISTING DATA")
    print("=" * 80)

    approaches = {
        'BUGGY (8 phi positions)':
            lambda b: PHI_POSITIONS_8 if abs(b - PHI) < 1e-6 else positions_for_base_dortmund(b),
        'DEGREE-3 SYMMETRIC':
            positions_degree3_symmetric,
        'DEGREE-4 SYMMETRIC':
            positions_degree4_symmetric,
        'FIXED N=6 (equal coverage)':
            lambda b: positions_fixed_n(b, 6),
        'FIXED N=8 (equal coverage)':
            lambda b: positions_fixed_n(b, 8),
    }

    # Load Dortmund EyesClosed pre (primary replication target)
    datasets = {}

    # Dortmund ses-1
    for cond in ['EyesClosed_pre', 'EyesOpen_pre']:
        csv_name = f'dortmund_ot_dominant_peaks_{cond}.csv'
        df = load_dortmund_condition(DORTMUND_RESULTS, csv_name)
        if df is not None:
            subjs = extract_freqs(df)
            datasets[f'Dortmund_{cond}'] = subjs
            print(f"\n  Loaded {len(subjs)} subjects for Dortmund {cond}")

    # Dortmund ses-2
    for cond in ['EyesClosed_pre', 'EyesOpen_pre']:
        csv_name = f'dortmund_ses2_dominant_peaks_{cond}.csv'
        df = load_dortmund_condition(DORTMUND_LONG, csv_name)
        if df is not None:
            subjs = extract_freqs(df)
            datasets[f'Dortmund_ses2_{cond}'] = subjs
            print(f"  Loaded {len(subjs)} subjects for Dortmund ses-2 {cond}")

    # Bonn
    bonn_path = os.path.join(BONN_RESULTS, 'bonn_dominant_peaks_all.csv')
    if os.path.isfile(bonn_path):
        bonn_df = pd.read_csv(bonn_path)
        for set_name in bonn_df['set'].unique():
            sdf = bonn_df[bonn_df['set'] == set_name]
            subjs = extract_freqs(sdf)
            if subjs:
                datasets[f'Bonn_{set_name}'] = subjs
                print(f"  Loaded {len(subjs)} segments for Bonn set {set_name}")

    # EEGMMIDB
    eegmmidb_path = os.path.join(EEGMMIDB_RESULTS, 'per_subject_dominant_peaks.csv')
    if os.path.isfile(eegmmidb_path):
        df = pd.read_csv(eegmmidb_path)
        subjs = extract_freqs(df)
        datasets['EEGMMIDB'] = subjs
        print(f"  Loaded {len(subjs)} subjects for EEGMMIDB")

    # LEMON
    lemon_path = os.path.join(LEMON_RESULTS, 'per_subject_dominant_peaks.csv')
    if os.path.isfile(lemon_path):
        df = pd.read_csv(lemon_path)
        subjs = extract_freqs(df)
        datasets['LEMON'] = subjs
        print(f"  Loaded {len(subjs)} subjects for LEMON")

    # Run all approaches on primary dataset (Dortmund_EyesClosed_pre)
    primary_key = 'Dortmund_EyesClosed_pre'
    if primary_key in datasets:
        print(f"\n{'='*80}")
        print(f"PRIMARY ANALYSIS: {primary_key} (N={len(datasets[primary_key])})")
        print(f"{'='*80}")

        for approach_name, pos_fn in approaches.items():
            print(f"\n  ===== {approach_name} =====")
            results = run_crossbase(datasets[primary_key], pos_fn, label=approach_name)
            print_crossbase(results, label=approach_name)

    # Run degree-3 symmetric on ALL datasets
    print(f"\n{'='*80}")
    print("CROSS-DATASET COMPARISON UNDER DEGREE-3 SYMMETRIC")
    print(f"{'='*80}")

    fix_b = positions_degree3_symmetric
    for ds_name, subjs in sorted(datasets.items()):
        print(f"\n  ── {ds_name} (N={len(subjs)}) ──")
        results = run_crossbase(subjs, fix_b, label=ds_name)
        ranking = sorted(results.items(), key=lambda x: x[1]['mean_d'])
        phi_rank = [name for name, _ in ranking].index('phi') + 1
        phi_norm_rank = sorted(results.items(), key=lambda x: x[1]['norm_d'])
        phi_norm_rank_pos = [name for name, _ in phi_norm_rank].index('phi') + 1
        print(f"    Phi rank (raw): {phi_rank}/9, mean_d={results['phi']['mean_d']:.4f}")
        print(f"    Phi rank (normalized): {phi_norm_rank_pos}/9, norm_d={results['phi']['norm_d']:.4f}")
        print(f"    Top 3 (raw): ", end='')
        for i, (bname, r) in enumerate(ranking[:3]):
            print(f"{bname}={r['mean_d']:.4f}", end='  ')
        print()
        print(f"    Top 3 (norm): ", end='')
        for i, (bname, r) in enumerate(phi_norm_rank[:3]):
            print(f"{bname}={r['norm_d']:.4f}", end='  ')
        print()

    # ── FIXED N=6 cross-dataset (EQUAL COVERAGE) ──
    print(f"\n{'='*80}")
    print("CROSS-DATASET COMPARISON: FIXED N=6 (equal coverage, no normalization needed)")
    print(f"{'='*80}")

    for ds_name, subjs in sorted(datasets.items()):
        print(f"\n  ── {ds_name} (N={len(subjs)}) ──")
        pos_fn = lambda b: positions_fixed_n(b, 6)
        results = run_crossbase(subjs, pos_fn, label=ds_name)
        ranking = sorted(results.items(), key=lambda x: x[1]['mean_d'])
        phi_rank = [name for name, _ in ranking].index('phi') + 1
        # Verify all have 6 positions
        counts = {bn: r['n_positions'] for bn, r in results.items()}
        print(f"    Position counts: {counts}")
        print(f"    Phi rank: {phi_rank}/9, mean_d={results['phi']['mean_d']:.4f}")
        print(f"    Full ranking: ", end='')
        for i, (bname, r) in enumerate(ranking):
            marker = '*' if bname == 'phi' else ''
            print(f"{i+1}.{bname}{marker}={r['mean_d']:.4f}", end='  ')
        print()

    # ── FIXED N=8 cross-dataset (EQUAL COVERAGE) ──
    print(f"\n{'='*80}")
    print("CROSS-DATASET COMPARISON: FIXED N=8 (equal coverage, no normalization needed)")
    print(f"{'='*80}")

    for ds_name, subjs in sorted(datasets.items()):
        print(f"\n  ── {ds_name} (N={len(subjs)}) ──")
        pos_fn = lambda b: positions_fixed_n(b, 8)
        results = run_crossbase(subjs, pos_fn, label=ds_name)
        ranking = sorted(results.items(), key=lambda x: x[1]['mean_d'])
        phi_rank = [name for name, _ in ranking].index('phi') + 1
        counts = {bn: r['n_positions'] for bn, r in results.items()}
        print(f"    Position counts: {counts}")
        print(f"    Phi rank: {phi_rank}/9, mean_d={results['phi']['mean_d']:.4f}")
        print(f"    Full ranking: ", end='')
        for i, (bname, r) in enumerate(ranking):
            marker = '*' if bname == 'phi' else ''
            print(f"{i+1}.{bname}{marker}={r['mean_d']:.4f}", end='  ')
        print()

    # ── Summary table: fixed-N comparison ──
    print(f"\n{'='*80}")
    print("SUMMARY: PHI RANK ACROSS APPROACHES AND DATASETS")
    print(f"{'='*80}")

    summary_fns = {
        'D3-sym': positions_degree3_symmetric,
        'D3-norm': positions_degree3_symmetric,  # report norm rank
        'N=6': lambda b: positions_fixed_n(b, 6),
        'N=8': lambda b: positions_fixed_n(b, 8),
    }
    print(f"\n  {'Dataset':<30s}  {'D3 raw':>7s} {'D3 norm':>7s}  {'N=6':>7s} {'N=8':>7s}")

    for ds_name, subjs in sorted(datasets.items()):
        row = f"  {ds_name:<30s}"
        # D3 raw + norm
        r3 = run_crossbase(subjs, positions_degree3_symmetric, label='')
        rank_raw = sorted(r3.items(), key=lambda x: x[1]['mean_d'])
        phi_r3_raw = [n for n, _ in rank_raw].index('phi') + 1
        rank_norm = sorted(r3.items(), key=lambda x: x[1]['norm_d'])
        phi_r3_norm = [n for n, _ in rank_norm].index('phi') + 1
        row += f"  {phi_r3_raw}/9    {phi_r3_norm}/9  "
        # N=6
        r6 = run_crossbase(subjs, lambda b: positions_fixed_n(b, 6), label='')
        rank6 = sorted(r6.items(), key=lambda x: x[1]['mean_d'])
        phi_r6 = [n for n, _ in rank6].index('phi') + 1
        row += f"  {phi_r6}/9  "
        # N=8
        r8 = run_crossbase(subjs, lambda b: positions_fixed_n(b, 8), label='')
        rank8 = sorted(r8.items(), key=lambda x: x[1]['mean_d'])
        phi_r8 = [n for n, _ in rank8].index('phi') + 1
        row += f"  {phi_r8}/9  "
        print(row)

    return datasets


# ═══════════════════════════════════════════════════════════════════════════
# 1c. CANONICAL-FREQUENCY DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════════

def audit_1c_canonical_frequencies():
    print("\n" + "=" * 80)
    print("1c. CANONICAL-FREQUENCY DIAGNOSTIC")
    print("    Would textbook EEG peaks (2, 6, 10, 37 Hz) explain the alignment?")
    print("=" * 80)

    textbook = {'delta': 2.0, 'theta': 6.0, 'alpha': 10.0, 'gamma': 37.0}

    pos_fn = positions_degree3_symmetric

    print(f"\n  Using degree-3 symmetric positions:")
    print(f"  {'Base':>6s}  {'N_pos':>5s}  {'E[d]':>8s}  {'textbook_d':>10s}  {'norm':>8s}  {'verdict':>10s}")
    for base_name, base_val in sorted(BASES.items(), key=lambda x: x[1]):
        positions = pos_fn(base_val)
        exp_d = expected_d_for_positions(positions)
        obs_d = compute_mean_d_for_subject(textbook, base_val, positions)
        norm = obs_d / exp_d if exp_d > 0 else np.nan
        verdict = 'TRIVIAL' if norm < 0.6 else 'BORDERLINE' if norm < 0.85 else 'GENUINE'
        print(f"  {base_name:>6s}  {len(positions):>5d}  {exp_d:>8.4f}  {obs_d:>10.4f}  "
              f"{norm:>8.3f}  {verdict:>10s}")

    # Also with 8-position phi
    print(f"\n  With BUGGY 8-position phi:")
    positions_8 = PHI_POSITIONS_8
    exp_d_8 = expected_d_for_positions(positions_8)
    obs_d_8 = compute_mean_d_for_subject(textbook, PHI, positions_8)
    norm_8 = obs_d_8 / exp_d_8
    print(f"  phi (8 pos): E[d]={exp_d_8:.4f}, textbook_d={obs_d_8:.4f}, norm={norm_8:.3f}")

    # Per-band detail for phi (degree-3 symmetric)
    print(f"\n  Per-band detail (phi, degree-3 symmetric positions):")
    positions_phi = pos_fn(PHI)
    for band, freq in textbook.items():
        u = lattice_coord(freq, f0=F0, base=PHI)
        d = min_dist(u, positions_phi)
        pos_name = nearest_pos(u, positions_phi)
        print(f"    {band}: freq={freq} Hz, u={u:.4f}, d={d:.4f}, nearest={pos_name}")

    # Alpha sweep
    print(f"\n  Alpha sweep (8-13 Hz, phi degree-3 symmetric):")
    for alpha_hz in np.arange(8.0, 13.1, 0.5):
        freqs = {'delta': 2.0, 'theta': 6.0, 'alpha': alpha_hz, 'gamma': 37.0}
        md = compute_mean_d_for_subject(freqs, PHI, positions_phi)
        exp = expected_d_for_positions(positions_phi)
        print(f"    alpha={alpha_hz:5.1f} Hz: mean_d={md:.4f}, norm={md/exp:.3f}")


# ═══════════════════════════════════════════════════════════════════════════
# 1d. RANDOM-POSITION-SET NULL
# ═══════════════════════════════════════════════════════════════════════════

def audit_1d_random_positions(datasets):
    print("\n" + "=" * 80)
    print("1d. RANDOM-POSITION-SET NULL")
    print("    Are phi's 4 specific positions special, or would any 4 positions work?")
    print("=" * 80)

    # Use actual observed median peaks from Dortmund EyesClosed_pre
    primary_key = 'Dortmund_EyesClosed_pre'
    if primary_key not in datasets:
        print("  [SKIPPED: No Dortmund data available]")
        return

    subjs = datasets[primary_key]
    # Compute population median frequencies
    all_delta = [s['delta'] for s in subjs if not np.isnan(s['delta'])]
    all_theta = [s['theta'] for s in subjs if not np.isnan(s['theta'])]
    all_alpha = [s['alpha'] for s in subjs if not np.isnan(s['alpha'])]
    all_gamma = [s['gamma'] for s in subjs if not np.isnan(s['gamma'])]

    med_freqs = {
        'delta': np.median(all_delta),
        'theta': np.median(all_theta),
        'alpha': np.median(all_alpha),
        'gamma': np.median(all_gamma),
    }
    print(f"  Median peak frequencies: δ={med_freqs['delta']:.2f}, θ={med_freqs['theta']:.2f}, "
          f"α={med_freqs['alpha']:.2f}, γ={med_freqs['gamma']:.2f}")

    # Phi's degree-3 symmetric performance
    phi_pos = positions_degree3_symmetric(PHI)
    phi_d = compute_mean_d_for_subject(med_freqs, PHI, phi_pos)
    phi_exp = expected_d_for_positions(phi_pos)
    print(f"\n  Phi ({len(phi_pos)} positions, degree-3 sym): mean_d = {phi_d:.4f}, E[d] = {phi_exp:.4f}, norm = {phi_d/phi_exp:.3f}")

    n_phi_pos = len(phi_pos)

    # Generate random position sets (same count as phi) and compute mean_d
    np.random.seed(42)
    n_random = 10_000
    random_ds = np.zeros(n_random)
    random_norms = np.zeros(n_random)

    for i in range(n_random):
        # Generate same number of random positions as phi has
        rand_pos_vals = np.sort(np.random.uniform(0, 1, n_phi_pos))
        rand_pos = {f'p{j}': v for j, v in enumerate(rand_pos_vals)}
        rand_exp = expected_d_for_positions(rand_pos)

        # Compute mean_d for median frequencies using phi as the base
        ds = []
        for band, freq in med_freqs.items():
            u = lattice_coord(freq, f0=F0, base=PHI)
            d = min_dist(u, rand_pos)
            ds.append(d)
        random_ds[i] = np.mean(ds)
        random_norms[i] = random_ds[i] / rand_exp if rand_exp > 0 else np.nan

    p_raw = (random_ds <= phi_d).mean()
    p_norm = (random_norms <= phi_d / phi_exp).mean()
    z_raw = (phi_d - random_ds.mean()) / random_ds.std() if random_ds.std() > 0 else 0

    print(f"\n  Random 4-position null (N={n_random}):")
    print(f"    Null mean_d: {random_ds.mean():.4f} ± {random_ds.std():.4f}")
    print(f"    Phi mean_d:  {phi_d:.4f}")
    print(f"    p (raw):     {p_raw:.4f}  (fraction of random ≤ phi)")
    print(f"    z (raw):     {z_raw:.2f}")
    print(f"    p (norm):    {p_norm:.4f}")

    # Also test per-subject distribution
    print(f"\n  Per-subject random position null:")
    n_subj_random = min(1000, n_random)
    phi_per_subj = np.array([compute_mean_d_for_subject(s, PHI, phi_pos) for s in subjs])
    phi_per_subj = phi_per_subj[~np.isnan(phi_per_subj)]

    rand_better_count = 0
    for i in range(n_subj_random):
        rand_pos_vals = np.sort(np.random.uniform(0, 1, n_phi_pos))
        rand_pos = {f'p{j}': v for j, v in enumerate(rand_pos_vals)}
        rand_per_subj = np.array([compute_mean_d_for_subject(s, PHI, rand_pos) for s in subjs])
        rand_per_subj = rand_per_subj[~np.isnan(rand_per_subj)]
        if rand_per_subj.mean() <= phi_per_subj.mean():
            rand_better_count += 1

    p_subj = rand_better_count / n_subj_random
    print(f"    N subjects: {len(phi_per_subj)}")
    print(f"    Phi per-subj mean_d: {phi_per_subj.mean():.4f}")
    print(f"    Random sets with lower mean_d: {rand_better_count}/{n_subj_random} ({p_subj:.4f})")


# ═══════════════════════════════════════════════════════════════════════════
# 1e. BAND-SHUFFLED NULL
# ═══════════════════════════════════════════════════════════════════════════

def audit_1e_band_shuffled(datasets):
    print("\n" + "=" * 80)
    print("1e. BAND-SHUFFLED NULL")
    print("    Do standard band definitions (1-4, 4-8, 8-13, 30-45) encode the result?")
    print("=" * 80)

    primary_key = 'Dortmund_EyesClosed_pre'
    if primary_key not in datasets:
        print("  [SKIPPED: No Dortmund data available]")
        return

    # We can't re-extract peaks with different bands from saved data.
    # But we CAN test: for the OBSERVED peak frequencies, does shifting
    # bands change which peak gets selected? The dominant peak is the
    # strongest in each band — different band boundaries would select
    # different peaks from the FOOOF pool.
    #
    # Since we only have the dominant peaks (not the full FOOOF pool),
    # we test a weaker version: given the 4 observed dominant frequencies,
    # how sensitive is mean_d to the LATTICE COORDINATE STRUCTURE
    # implied by different band definitions?
    #
    # Actually, the band definitions affect peak SELECTION, not lattice math.
    # The lattice coordinate u = log_phi(f/f0) mod 1 is independent of bands.
    # What matters is which FREQUENCIES end up in the analysis.
    #
    # So the real test is: do the observed peak frequencies have INHERENTLY
    # special phi-lattice coordinates, regardless of how they were selected?

    subjs = datasets[primary_key]
    phi_pos = positions_degree3_symmetric(PHI)
    phi_exp = expected_d_for_positions(phi_pos)

    # Get the observed per-subject mean_d with phi
    phi_ds = np.array([compute_mean_d_for_subject(s, PHI, phi_pos) for s in subjs])
    phi_ds = phi_ds[~np.isnan(phi_ds)]
    obs_mean_d = phi_ds.mean()

    print(f"\n  Observed phi mean_d: {obs_mean_d:.4f} (E[d]={phi_exp:.4f}, norm={obs_mean_d/phi_exp:.3f})")

    # Frequency shuffle null: for each permutation, randomly reassign
    # each subject's 4 frequencies to 4 bands (destroying band-specific
    # frequency structure while preserving the individual frequencies)
    np.random.seed(42)
    n_perm = 10_000
    shuffled_ds = np.zeros(n_perm)

    all_subj_freqs = []
    for s in subjs:
        fs = [s[b] for b in ['delta', 'theta', 'alpha', 'gamma'] if not np.isnan(s[b])]
        if len(fs) == 4:
            all_subj_freqs.append(fs)

    for i in range(n_perm):
        perm_ds = []
        for fs in all_subj_freqs:
            # The frequencies stay the same — we just compute mean_d
            # (since lattice coord doesn't depend on band assignment,
            # this tests whether our 4 specific frequencies are special)
            # Actually, mean_d = mean of d for each frequency.
            # Shuffling band assignment doesn't change this — d is computed
            # per frequency independently.
            pass

        # A more meaningful null: sample 4 frequencies per subject
        # from the COMBINED pool of all subjects' peaks (breaking
        # the within-subject correlation structure)
        for _ in range(len(all_subj_freqs)):
            random_subj_idx = np.random.randint(0, len(all_subj_freqs), 4)
            random_band_idx = np.random.randint(0, 4, 4)
            fake_freqs = [all_subj_freqs[si][bi] for si, bi in zip(random_subj_idx, random_band_idx)]
            ds = [min_dist(lattice_coord(f, f0=F0, base=PHI), phi_pos) for f in fake_freqs]
            perm_ds.append(np.mean(ds))

        shuffled_ds[i] = np.mean(perm_ds)

    p_shuffle = (shuffled_ds <= obs_mean_d).mean()
    z_shuffle = (obs_mean_d - shuffled_ds.mean()) / shuffled_ds.std() if shuffled_ds.std() > 0 else 0

    print(f"\n  Cross-subject frequency shuffle null (N={n_perm}):")
    print(f"    Null mean_d: {shuffled_ds.mean():.4f} ± {shuffled_ds.std():.4f}")
    print(f"    Observed:    {obs_mean_d:.4f}")
    print(f"    p:           {p_shuffle:.4f}")
    print(f"    z:           {z_shuffle:.2f}")

    # Alternative band definitions test:
    # Same widths (3, 4, 5, 15 Hz), shifted boundaries
    alt_bands_list = [
        ('Standard',    {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'gamma': (30, 45)}),
        ('Shift+1',     {'delta': (2, 5), 'theta': (5, 9), 'alpha': (9, 14), 'gamma': (31, 46)}),
        ('Shift+2',     {'delta': (3, 6), 'theta': (6, 10), 'alpha': (10, 15), 'gamma': (32, 47)}),
        ('Shift-1',     {'delta': (1, 4), 'theta': (3, 7), 'alpha': (7, 12), 'gamma': (29, 44)}),
        ('Narrow',      {'delta': (1, 3), 'theta': (5, 7), 'alpha': (9, 12), 'gamma': (33, 42)}),
        ('Wide',        {'delta': (1, 5), 'theta': (3, 9), 'alpha': (7, 14), 'gamma': (28, 48)}),
    ]

    print(f"\n  NOTE: Cannot re-extract peaks with different bands from saved data.")
    print(f"  The band definitions affect peak SELECTION (which peak is dominant),")
    print(f"  not the lattice coordinate computation. A true test would require")
    print(f"  re-running FOOOF extraction, which takes hours per dataset.")
    print(f"  The tests above (random positions + cross-subject shuffle)")
    print(f"  address the same question from different angles.")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("AUDIT: CROSS-BASE FAIRNESS IN PHI-LATTICE ANALYSIS")
    print("=" * 80)
    print(f"f₀ = {F0} Hz, φ = {PHI:.10f}")
    print(f"Bands: {BANDS}")
    print()

    # 1a: Position counts
    audit_1a_position_counts()

    # 1b: Fair cross-base (also loads all data)
    datasets = audit_1b_fair_crossbase()

    # 1c: Canonical frequencies
    audit_1c_canonical_frequencies()

    # 1d: Random position null
    audit_1d_random_positions(datasets)

    # 1e: Band-shuffled null
    audit_1e_band_shuffled(datasets)

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)

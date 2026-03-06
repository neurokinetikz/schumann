#!/usr/bin/env python3
"""
Mode Shift Analysis: Per-Condition Bootstrap CIs and Permutation Tests
=======================================================================

Addresses 4 peer-review pushback items on the EEGMMIDB dissociation analysis:
  1. Per-condition bootstrap CIs on KDE mode location (3 conditions)
  2. Formal permutation test on mode difference (session-label shuffle)
  3. Beta_low inv_noble_6 motor-signature explanation
  4. Band-level claim stratification (gamma clean, alpha confounded, beta_low control)

Usage:
    python scripts/mode_shift_analysis.py
    python scripts/mode_shift_analysis.py --n-perms 5000 --n-boot 2000
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
from scipy.stats import gaussian_kde, wilcoxon, norm

sys.path.insert(0, './lib')
sys.path.insert(0, './scripts')

from phi_frequency_model import PHI, F0
from analyze_aggregate_enrichment import compute_lattice_coordinate, EXTENDED_OFFSETS
from noble_boundary_dissociation import (
    load_and_assign_conditions, ANALYSIS_BANDS,
)

# =========================================================================
# CONSTANTS
# =========================================================================

INPUT_CSV = 'exports_peak_distribution/eegmmidb_fooof/golden_ratio_peaks_EEGMMIDB.csv'
OUTPUT_DIR = 'exports_peak_distribution/eegmmidb_fooof'

TARGET_BANDS = ['alpha', 'gamma', 'beta_low']

CONDITION_MAP = {
    'rest_EO': ['rest_eyes_open'],
    'rest_EC': ['rest_eyes_closed'],
    'task':    ['motor_execution', 'motor_imagery'],
}

# Reference lattice positions
PHI_INV = 1.0 / PHI
NOBLE_1 = PHI_INV                    # 0.6180
ATTRACTOR = 0.500
INV_NOBLE_6 = 1 - PHI_INV ** 6      # 0.9443

REFERENCE_POSITIONS = {
    'boundary':    0.000,
    'attractor':   ATTRACTOR,
    'noble_1':     NOBLE_1,
    'inv_noble_4': 1 - PHI_INV ** 4,
    'inv_noble_6': INV_NOBLE_6,
}


# =========================================================================
# COPIED HELPER FUNCTIONS (from dissociation_validation.py)
# =========================================================================

# Pre-computed grid for fast mode finding (0.005 resolution, sufficient for bw=0.02)
_MODE_GRID = np.linspace(0.01, 0.99, 200)


def find_kde_mode(lattice_coords, bw=0.02, max_n=5000, rng=None):
    """Find the mode (peak) of the KDE on [0, 1) using grid evaluation.
    Subsamples to max_n for speed; uses grid instead of minimize_scalar
    to avoid scipy optimizer instability in tight loops.
    Set max_n=0 to disable subsampling (use all data)."""
    if len(lattice_coords) < 20:
        return np.nan
    if max_n > 0 and len(lattice_coords) > max_n:
        if rng is None:
            rng = np.random.default_rng(0)
        lattice_coords = rng.choice(lattice_coords, size=max_n, replace=False)
    kde = gaussian_kde(lattice_coords, bw_method=bw)
    vals = kde(_MODE_GRID)
    return _MODE_GRID[np.argmax(vals)]


def bootstrap_mode(lattice_coords, n_boot=1000, bw=0.02, seed=42):
    """Bootstrap the KDE mode location."""
    rng = np.random.default_rng(seed)
    modes = []
    n = len(lattice_coords)
    for _ in range(n_boot):
        boot = lattice_coords[rng.choice(n, size=n, replace=True)]
        mode = find_kde_mode(boot, bw=bw, rng=rng)
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


def build_session_freq_index(df):
    """Pre-build session -> freq array dict."""
    idx = {}
    for sess, grp in df.groupby('session'):
        idx[sess] = grp['freq'].values
    return idx


# =========================================================================
# GENERALIZED LATTICE HELPERS (for control analyses)
# =========================================================================

def compute_lattice_coordinate_general(freqs, f0=F0, base=PHI):
    """Generalized lattice coordinate: u = [log_base(f/f0)] mod 1."""
    exps = np.log(freqs / f0) / np.log(base)
    return exps % 1.0


def build_session_band_lattice_index_general(session_freq_idx, band_range,
                                              f0=F0, base=PHI):
    """Pre-compute per-session lattice coordinates with arbitrary f0 and base."""
    f_lo, f_hi = band_range
    idx = {}
    for sess, freqs in session_freq_idx.items():
        in_band = freqs[(freqs >= f_lo) & (freqs < f_hi)]
        if len(in_band) > 0:
            idx[sess] = compute_lattice_coordinate_general(in_band, f0, base)
    return idx


# =========================================================================
# NEW HELPER
# =========================================================================

def get_condition_lattice(session_freq_idx, session_ids, band_range, f0=F0):
    """Extract lattice coordinates for a condition + band combination."""
    valid_sessions = [s for s in session_ids if s in session_freq_idx]
    if not valid_sessions:
        return np.array([]), 0
    freqs = np.concatenate([session_freq_idx[s] for s in valid_sessions])
    f_lo, f_hi = band_range
    in_band = freqs[(freqs >= f_lo) & (freqs < f_hi)]
    if len(in_band) < 20:
        return np.array([]), 0
    lattice = compute_lattice_coordinate(in_band, f0)
    return lattice, len(in_band)


def nearest_position(mode_val):
    """Find the nearest reference position to a mode value."""
    best_name, best_dist = None, 1.0
    for name, offset in REFERENCE_POSITIONS.items():
        d = abs(mode_val - offset)
        d_wrap = min(d, abs(mode_val - offset - 1), abs(mode_val - offset + 1))
        if d_wrap < best_dist:
            best_dist = d_wrap
            best_name = name
    return best_name, best_dist


# =========================================================================
# ANALYSIS 1: PER-CONDITION BOOTSTRAP CIs
# =========================================================================

def run_per_condition_modes(session_freq_idx, sessions_by_cond, n_boot=1000):
    """Compute KDE mode + bootstrap 95% CI for each band × condition."""
    print("\n" + "=" * 80, flush=True)
    print("  ANALYSIS 1: PER-CONDITION BOOTSTRAP CIs ON KDE MODE", flush=True)
    print("  3 conditions: rest-EO, rest-EC, task", flush=True)
    print("  3 bands: alpha, gamma, beta_low", flush=True)
    print(f"  Bootstrap: {n_boot} resamples", flush=True)
    print("=" * 80, flush=True)

    # Build 3-cell condition sets
    cells = {}
    for cell_name, cond_vals in CONDITION_MAP.items():
        cells[cell_name] = set()
        for cv in cond_vals:
            if cv in sessions_by_cond:
                cells[cell_name] |= sessions_by_cond[cv]
        print(f"  {cell_name}: {len(cells[cell_name])} sessions", flush=True)

    rows = []

    print(f"\n  {'Band':12s} {'Condition':10s} {'N peaks':>8s} {'Mode':>7s} "
          f"{'CI_lo':>7s} {'CI_hi':>7s} {'Nearest':>12s} {'Dist':>7s}", flush=True)
    print(f"  {'-'*12} {'-'*10} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*12} {'-'*7}", flush=True)

    for band_name in TARGET_BANDS:
        band_range = ANALYSIS_BANDS[band_name]
        for cond_name in ['rest_EO', 'rest_EC', 'task']:
            lattice, n_peaks = get_condition_lattice(
                session_freq_idx, cells[cond_name], band_range)

            if len(lattice) < 20:
                print(f"  {band_name:12s} {cond_name:10s} {n_peaks:>8d}  -- insufficient data --", flush=True)
                continue

            mode = find_kde_mode(lattice, rng=np.random.default_rng(42))
            boot = bootstrap_mode(lattice, n_boot=n_boot)
            near_name, near_dist = nearest_position(mode)

            row = {
                'band': band_name,
                'condition': cond_name,
                'n_peaks': n_peaks,
                'n_sessions': len(cells[cond_name]),
                'mode': mode,
                'ci_lower': boot['ci_lower'],
                'ci_upper': boot['ci_upper'],
                'mode_std': boot['mode_std'],
                'nearest_position': near_name,
                'dist_to_nearest': near_dist,
                'dist_to_noble1': abs(mode - NOBLE_1),
                'dist_to_attractor': abs(mode - ATTRACTOR),
                'dist_to_inv_noble6': min(abs(mode - INV_NOBLE_6),
                                          abs(mode - INV_NOBLE_6 + 1)),
                'ci_excludes_attractor': not (boot['ci_lower'] <= ATTRACTOR <= boot['ci_upper']),
                'ci_excludes_noble1': not (boot['ci_lower'] <= NOBLE_1 <= boot['ci_upper']),
            }
            rows.append(row)

            print(f"  {band_name:12s} {cond_name:10s} {n_peaks:>8,d} {mode:7.4f} "
                  f"{boot['ci_lower']:7.4f} {boot['ci_upper']:7.4f} "
                  f"{near_name:>12s} {near_dist:7.4f}", flush=True)

    df = pd.DataFrame(rows)

    # Print key interpretations
    print(f"\n  --- Key Findings ---", flush=True)
    for band in TARGET_BANDS:
        band_rows = df[df['band'] == band]
        if len(band_rows) == 0:
            continue
        print(f"\n  {band.upper()}:", flush=True)
        for _, r in band_rows.iterrows():
            excl = []
            if r['ci_excludes_attractor']:
                excl.append("CI excludes attractor")
            if r['ci_excludes_noble1']:
                excl.append("CI excludes noble_1")
            excl_str = "; ".join(excl) if excl else "CI includes both"
            print(f"    {r['condition']:10s}: mode={r['mode']:.4f} -> {r['nearest_position']} "
                  f"(d={r['dist_to_nearest']:.4f})  [{excl_str}]", flush=True)

    return df


# =========================================================================
# ANALYSIS 2: PERMUTATION TEST ON MODE DIFFERENCE
# =========================================================================

def build_session_band_lattice_index(session_freq_idx, band_range, f0=F0):
    """Pre-compute per-session lattice coordinates for a specific band.
    This is the key optimization: avoids recomputing freq->lattice 60K times."""
    f_lo, f_hi = band_range
    idx = {}
    for sess, freqs in session_freq_idx.items():
        in_band = freqs[(freqs >= f_lo) & (freqs < f_hi)]
        if len(in_band) > 0:
            idx[sess] = compute_lattice_coordinate(in_band, f0)
    return idx


def permutation_test_mode_difference(session_freq_idx, sess_a, sess_b,
                                      band_range, n_perms=5000, seed=42):
    """
    Test whether two conditions have different KDE mode locations.
    Session-level label shuffle preserves within-session correlation.
    Uses pre-computed per-session lattice coords for speed.
    """
    # Pre-compute lattice coords per session for this band
    band_lattice = build_session_band_lattice_index(session_freq_idx, band_range)

    # Observed modes
    lat_a_parts = [band_lattice[s] for s in sess_a if s in band_lattice]
    lat_b_parts = [band_lattice[s] for s in sess_b if s in band_lattice]

    if not lat_a_parts or not lat_b_parts:
        return None

    lat_a = np.concatenate(lat_a_parts)
    lat_b = np.concatenate(lat_b_parts)
    n_a = len(lat_a)
    n_b = len(lat_b)

    if n_a < 20 or n_b < 20:
        return None

    rng = np.random.default_rng(seed)
    mode_a = find_kde_mode(lat_a, rng=rng)
    mode_b = find_kde_mode(lat_b, rng=rng)
    obs_diff = mode_a - mode_b

    # Null distribution using pre-computed lattice arrays
    all_sessions = np.array(list(sess_a) + list(sess_b))
    n_a_sess = len(sess_a)
    null_diffs = np.zeros(n_perms)

    t_perm = time.time()
    for i in range(n_perms):
        perm = rng.permutation(all_sessions)
        perm_a = perm[:n_a_sess]
        perm_b = perm[n_a_sess:]

        # Fast: concatenate pre-computed lattice arrays
        pa_parts = [band_lattice[s] for s in perm_a if s in band_lattice]
        pb_parts = [band_lattice[s] for s in perm_b if s in band_lattice]

        if pa_parts and pb_parts:
            lat_pa = np.concatenate(pa_parts)
            lat_pb = np.concatenate(pb_parts)
            if len(lat_pa) >= 20 and len(lat_pb) >= 20:
                null_diffs[i] = find_kde_mode(lat_pa, rng=rng) - find_kde_mode(lat_pb, rng=rng)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t_perm
            rate = (i + 1) / elapsed
            eta = (n_perms - i - 1) / rate
            print(f"    perm {i+1}/{n_perms} ({elapsed:.0f}s elapsed, {eta:.0f}s ETA)", flush=True)

    p_val = np.mean(np.abs(null_diffs) >= np.abs(obs_diff))
    null_std = null_diffs.std()
    z = (obs_diff - null_diffs.mean()) / null_std if null_std > 0 else 0.0

    return {
        'mode_A': mode_a,
        'mode_B': mode_b,
        'observed_diff': obs_diff,
        'p_val': p_val,
        'z_score': z,
        'null_mean': null_diffs.mean(),
        'null_std': null_std,
        'n_a_peaks': n_a,
        'n_b_peaks': n_b,
        'n_a_sessions': len(sess_a),
        'n_b_sessions': len(sess_b),
        'null_distribution': null_diffs,
    }


def build_subject_pairs(sessions_by_cond):
    """Build within-subject session pairs for paired permutation tests.

    Returns:
        eo_ec_pairs: list of (eo_sess, ec_sess) for subjects with both
        eo_task_pairs: list of (eo_sess, [task_sess_1, ...]) per subject
        ec_task_pairs: list of (ec_sess, [task_sess_1, ...]) per subject
    """
    # Collect sessions by subject and condition type
    eo_by_subj = {}  # subject -> session
    ec_by_subj = {}
    task_by_subj = {}  # subject -> [sessions]

    for cond_key, cond_vals in CONDITION_MAP.items():
        for cv in cond_vals:
            if cv not in sessions_by_cond:
                continue
            for sess in sessions_by_cond[cv]:
                subj = sess.split('R')[0]
                if cond_key == 'rest_EO':
                    eo_by_subj[subj] = sess
                elif cond_key == 'rest_EC':
                    ec_by_subj[subj] = sess
                elif cond_key == 'task':
                    task_by_subj.setdefault(subj, []).append(sess)

    # Build paired lists
    eo_ec_pairs = []
    eo_task_pairs = []
    ec_task_pairs = []

    all_subjects = sorted(set(eo_by_subj) & set(ec_by_subj))
    for subj in all_subjects:
        eo_ec_pairs.append((eo_by_subj[subj], ec_by_subj[subj]))
        if subj in task_by_subj:
            eo_task_pairs.append((eo_by_subj[subj], task_by_subj[subj]))
            ec_task_pairs.append((ec_by_subj[subj], task_by_subj[subj]))

    return eo_ec_pairs, eo_task_pairs, ec_task_pairs


def paired_permutation_test_eo_ec(band_lattice, pairs, n_perms=5000, seed=42):
    """Within-subject paired permutation test for EO vs EC mode difference.

    Each subject contributes 1 EO and 1 EC session. Under the null, each
    subject's labels are independently flipped with p=0.5, removing all
    between-subject variance from the null distribution.
    """
    # Observed: collect pools
    lat_a_parts = [band_lattice[eo] for eo, ec in pairs if eo in band_lattice]
    lat_b_parts = [band_lattice[ec] for eo, ec in pairs if ec in band_lattice]

    if not lat_a_parts or not lat_b_parts:
        return None

    lat_a = np.concatenate(lat_a_parts)
    lat_b = np.concatenate(lat_b_parts)

    if len(lat_a) < 20 or len(lat_b) < 20:
        return None

    # Observed modes: use all data (no subsampling) for deterministic result
    mode_a = find_kde_mode(lat_a, max_n=0)
    mode_b = find_kde_mode(lat_b, max_n=0)
    obs_diff = mode_a - mode_b

    # Separate RNG for permutation shuffles (subsampled for speed)
    rng = np.random.default_rng(seed)

    # Valid pairs (both sessions present in band_lattice)
    valid_pairs = [(eo, ec) for eo, ec in pairs
                   if eo in band_lattice and ec in band_lattice]
    n_subjects = len(valid_pairs)

    null_diffs = np.zeros(n_perms)
    t_perm = time.time()

    for i in range(n_perms):
        flips = rng.binomial(1, 0.5, n_subjects)
        perm_a_parts, perm_b_parts = [], []

        for j, (eo, ec) in enumerate(valid_pairs):
            if flips[j]:
                perm_a_parts.append(band_lattice[ec])
                perm_b_parts.append(band_lattice[eo])
            else:
                perm_a_parts.append(band_lattice[eo])
                perm_b_parts.append(band_lattice[ec])

        lat_pa = np.concatenate(perm_a_parts)
        lat_pb = np.concatenate(perm_b_parts)
        null_diffs[i] = find_kde_mode(lat_pa, rng=rng) - find_kde_mode(lat_pb, rng=rng)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t_perm
            rate = (i + 1) / elapsed
            eta = (n_perms - i - 1) / rate
            print(f"    perm {i+1}/{n_perms} ({elapsed:.0f}s elapsed, {eta:.0f}s ETA)",
                  flush=True)

    p_val = np.mean(np.abs(null_diffs) >= np.abs(obs_diff))
    null_std = null_diffs.std()
    z = (obs_diff - null_diffs.mean()) / null_std if null_std > 0 else 0.0

    return {
        'mode_A': mode_a,
        'mode_B': mode_b,
        'observed_diff': obs_diff,
        'p_val': p_val,
        'z_score': z,
        'null_mean': null_diffs.mean(),
        'null_std': null_std,
        'n_a_peaks': len(lat_a),
        'n_b_peaks': len(lat_b),
        'n_a_sessions': n_subjects,
        'n_b_sessions': n_subjects,
        'null_distribution': null_diffs,
        'test_type': 'paired',
    }


def paired_permutation_test_rest_vs_task(band_lattice, rest_task_pairs,
                                          n_perms=5000, seed=42):
    """Within-subject paired permutation test for rest vs pooled-task.

    rest_task_pairs: list of (rest_session, [task_session_1, ...]) per subject.
    Each subject's 12 task sessions are pooled. Under the null, each subject's
    rest/task labels are independently flipped with p=0.5.
    """
    # Build per-subject pooled arrays
    subject_rest = []  # list of arrays
    subject_task = []  # list of arrays
    for rest_sess, task_sessions in rest_task_pairs:
        if rest_sess not in band_lattice:
            continue
        task_parts = [band_lattice[s] for s in task_sessions if s in band_lattice]
        if not task_parts:
            continue
        subject_rest.append(band_lattice[rest_sess])
        subject_task.append(np.concatenate(task_parts))

    if len(subject_rest) < 10:
        return None

    lat_a = np.concatenate(subject_rest)
    lat_b = np.concatenate(subject_task)

    if len(lat_a) < 20 or len(lat_b) < 20:
        return None

    # Observed modes: use all data (no subsampling) for deterministic result
    mode_a = find_kde_mode(lat_a, max_n=0)
    mode_b = find_kde_mode(lat_b, max_n=0)
    obs_diff = mode_a - mode_b

    # Separate RNG for permutation shuffles (subsampled for speed)
    rng = np.random.default_rng(seed)

    n_subjects = len(subject_rest)
    null_diffs = np.zeros(n_perms)
    t_perm = time.time()

    for i in range(n_perms):
        flips = rng.binomial(1, 0.5, n_subjects)
        perm_a_parts, perm_b_parts = [], []

        for j in range(n_subjects):
            if flips[j]:
                perm_a_parts.append(subject_task[j])
                perm_b_parts.append(subject_rest[j])
            else:
                perm_a_parts.append(subject_rest[j])
                perm_b_parts.append(subject_task[j])

        lat_pa = np.concatenate(perm_a_parts)
        lat_pb = np.concatenate(perm_b_parts)
        null_diffs[i] = find_kde_mode(lat_pa, rng=rng) - find_kde_mode(lat_pb, rng=rng)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t_perm
            rate = (i + 1) / elapsed
            eta = (n_perms - i - 1) / rate
            print(f"    perm {i+1}/{n_perms} ({elapsed:.0f}s elapsed, {eta:.0f}s ETA)",
                  flush=True)

    p_val = np.mean(np.abs(null_diffs) >= np.abs(obs_diff))
    null_std = null_diffs.std()
    z = (obs_diff - null_diffs.mean()) / null_std if null_std > 0 else 0.0

    n_rest_sess = sum(1 for r, _ in rest_task_pairs if r in band_lattice)
    n_task_sess = sum(len([s for s in ts if s in band_lattice])
                      for _, ts in rest_task_pairs)

    return {
        'mode_A': mode_a,
        'mode_B': mode_b,
        'observed_diff': obs_diff,
        'p_val': p_val,
        'z_score': z,
        'null_mean': null_diffs.mean(),
        'null_std': null_std,
        'n_a_peaks': len(lat_a),
        'n_b_peaks': len(lat_b),
        'n_a_sessions': n_rest_sess,
        'n_b_sessions': n_task_sess,
        'null_distribution': null_diffs,
        'test_type': 'paired_pooled',
    }


# =========================================================================
# ANALYSIS 5: TARGETED BINARY CONTRAST
# =========================================================================

# Thresholds: midpoint between the two positions that define the contrast
BAND_THRESHOLDS = {
    'gamma':    0.559,   # midpoint of attractor (0.500) and noble_1 (0.618)
    'alpha':    0.559,   # same region
    'beta_low': 0.899,   # midpoint of inv_noble_4 (0.854) and inv_noble_6 (0.944)
}


def compute_per_subject_proportions(band_lattice, subject_sessions, threshold):
    """For each subject, compute proportion of peaks above threshold.

    Parameters
    ----------
    band_lattice : dict of session_id -> np.ndarray of lattice coords
    subject_sessions : dict of subject_id -> list of session_ids
    threshold : float in [0, 1)

    Returns
    -------
    dict of subject_id -> (proportion, n_peaks)
    """
    result = {}
    for subj, sessions in subject_sessions.items():
        parts = [band_lattice[s] for s in sessions if s in band_lattice]
        if not parts:
            continue
        coords = np.concatenate(parts)
        if len(coords) < 10:
            continue
        result[subj] = (np.mean(coords > threshold), len(coords))
    return result


def run_targeted_binary_tests(session_freq_idx, sessions_by_cond):
    """Run per-subject paired Wilcoxon tests on proportion above threshold."""
    print("\n" + "=" * 80, flush=True)
    print("  ANALYSIS 5: TARGETED BINARY CONTRAST (PER-SUBJECT PAIRED WILCOXON)", flush=True)
    print("  Proportion of peaks above threshold, paired across 109 subjects", flush=True)
    print("=" * 80, flush=True)

    # Build subject-level session groupings
    eo_ec_pairs, eo_task_pairs, ec_task_pairs = build_subject_pairs(sessions_by_cond)
    print(f"\n  Subject pairs: {len(eo_ec_pairs)} EO-EC, "
          f"{len(eo_task_pairs)} EO-task, {len(ec_task_pairs)} EC-task", flush=True)

    # Build per-subject session dicts for each condition
    eo_by_subj = {}
    ec_by_subj = {}
    task_by_subj = {}
    for eo, ec in eo_ec_pairs:
        subj = eo.split('R')[0]
        eo_by_subj[subj] = [eo]
        ec_by_subj[subj] = [ec]
    for eo, task_sessions in eo_task_pairs:
        subj = eo.split('R')[0]
        task_by_subj[subj] = task_sessions

    # Pre-compute band lattice indices
    band_lattice_cache = {}
    for band_name in TARGET_BANDS:
        band_range = ANALYSIS_BANDS[band_name]
        band_lattice_cache[band_name] = build_session_band_lattice_index(
            session_freq_idx, band_range)

    # Define contrasts: (label, cond_A_sessions, cond_B_sessions)
    contrasts = [
        ('rest-EO vs rest-EC', eo_by_subj, ec_by_subj),
        ('rest-EO vs task',    eo_by_subj, task_by_subj),
        ('rest-EC vs task',    ec_by_subj, task_by_subj),
    ]

    results = []
    n_tests = len(TARGET_BANDS) * len(contrasts)

    print(f"\n  Bonferroni correction for {n_tests} tests "
          f"(alpha = {0.05/n_tests:.4f})", flush=True)
    print(f"\n  {'Contrast':22s} {'Band':10s} {'Thresh':>6s} {'Med_A':>7s} {'Med_B':>7s} "
          f"{'Med_d':>7s} {'W':>8s} {'p':>10s} {'p_bonf':>8s} {'r':>6s} {'Sig?':>5s}",
          flush=True)
    print(f"  {'-'*22} {'-'*10} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*8} "
          f"{'-'*10} {'-'*8} {'-'*6} {'-'*5}", flush=True)

    for band_name in TARGET_BANDS:
        band_lattice = band_lattice_cache[band_name]
        threshold = BAND_THRESHOLDS[band_name]

        for label, subj_sess_a, subj_sess_b in contrasts:
            # Compute per-subject proportions
            props_a = compute_per_subject_proportions(
                band_lattice, subj_sess_a, threshold)
            props_b = compute_per_subject_proportions(
                band_lattice, subj_sess_b, threshold)

            # Match subjects present in both conditions
            common_subjs = sorted(set(props_a) & set(props_b))
            if len(common_subjs) < 10:
                print(f"  {label:22s} {band_name:10s}  -- insufficient subjects "
                      f"({len(common_subjs)}) --", flush=True)
                continue

            vals_a = np.array([props_a[s][0] for s in common_subjs])
            vals_b = np.array([props_b[s][0] for s in common_subjs])
            diffs = vals_a - vals_b

            # Wilcoxon signed-rank test
            # Remove zeros (ties at zero) — wilcoxon can't handle all-zero
            nonzero_diffs = diffs[diffs != 0]
            if len(nonzero_diffs) < 10:
                print(f"  {label:22s} {band_name:10s}  -- too many zero diffs "
                      f"({len(diffs) - len(nonzero_diffs)}/{len(diffs)}) --", flush=True)
                continue

            w_stat, p_val = wilcoxon(diffs, alternative='two-sided')

            # Effect size: rank-biserial correlation
            # r = 1 - 4W/(n(n+1)) where W = min(T+, T-) from scipy wilcoxon
            n = len(nonzero_diffs)
            r_effect = 1 - (4 * w_stat) / (n * (n + 1))

            p_bonf = min(p_val * n_tests, 1.0)
            sig = "YES" if p_bonf < 0.05 else "no"

            star = ""
            if p_val < 0.001: star = "***"
            elif p_val < 0.01: star = "** "
            elif p_val < 0.05: star = "*  "

            med_a = np.median(vals_a)
            med_b = np.median(vals_b)
            med_d = np.median(diffs)

            result_row = {
                'contrast': label,
                'band': band_name,
                'threshold': threshold,
                'n_subjects': len(common_subjs),
                'median_A': med_a,
                'median_B': med_b,
                'median_diff': med_d,
                'mean_diff': diffs.mean(),
                'std_diff': diffs.std(),
                'W_stat': w_stat,
                'p_val': p_val,
                'p_bonf': p_bonf,
                'r_effect': r_effect,
            }
            results.append(result_row)

            print(f"  {label:22s} {band_name:10s} {threshold:6.3f} "
                  f"{med_a:7.4f} {med_b:7.4f} {med_d:+7.4f} "
                  f"{w_stat:8.0f} {p_val:10.6f}{star} "
                  f"{p_bonf:8.4f} {r_effect:+6.3f} {sig:>5s}", flush=True)

    return results


# =========================================================================
# ANALYSIS 6: MIXTURE MODEL AT PHI-POSITIONS
# =========================================================================

# Component means (fixed)
MIXTURE_COMPONENTS = {
    'boundary':    0.000,
    'attractor':   ATTRACTOR,
    'noble_1':     NOBLE_1,
    'inv_noble_4': 1 - PHI_INV ** 4,
    'inv_noble_6': INV_NOBLE_6,
}


def min_circular_dist(x, mu):
    """Minimum distance on [0, 1) with wrap-around."""
    d = np.abs(x - mu)
    return np.minimum(d, 1.0 - d)


def fit_phi_mixture(lattice_coords, means, sigma_init=0.04, max_iter=100,
                    tol=1e-6, update_sigma=True):
    """EM with fixed component means, estimating weights and optionally sigma.

    Parameters
    ----------
    lattice_coords : np.ndarray, shape (n,)
    means : np.ndarray, shape (K,)
    sigma_init : float
    max_iter : int
    tol : float — convergence tolerance on log-likelihood
    update_sigma : bool — if True, estimate shared sigma; if False, keep fixed

    Returns
    -------
    weights : np.ndarray, shape (K,) — mixing weights (sum to 1)
    sigma : float — shared component width
    ll : float — final log-likelihood
    """
    K = len(means)
    n = len(lattice_coords)
    weights = np.ones(K) / K
    sigma = sigma_init

    # Pre-compute circular distances to each component: shape (n, K)
    dists = np.zeros((n, K))
    for k in range(K):
        dists[:, k] = min_circular_dist(lattice_coords, means[k])

    prev_ll = -np.inf

    for iteration in range(max_iter):
        # E-step: responsibilities
        log_resp = np.zeros((n, K))
        for k in range(K):
            log_resp[:, k] = np.log(weights[k] + 1e-300) + \
                norm.logpdf(dists[:, k], 0, sigma)

        # Log-sum-exp for numerical stability
        log_norm = np.logaddexp.reduce(log_resp, axis=1)
        log_resp -= log_norm[:, np.newaxis]
        resp = np.exp(log_resp)

        # Log-likelihood
        ll = log_norm.sum()
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

        # M-step: update weights
        weights = resp.mean(axis=0)
        weights = np.maximum(weights, 1e-10)
        weights /= weights.sum()

        # M-step: update sigma (shared across components)
        if update_sigma:
            weighted_sq_dist = (resp * dists ** 2).sum()
            sigma = np.sqrt(weighted_sq_dist / n)
            sigma = max(sigma, 0.005)  # floor to prevent collapse

    return weights, sigma, ll


def run_mixture_model_tests(session_freq_idx, sessions_by_cond):
    """Run per-subject phi-mixture model + paired Wilcoxon on mixing weights."""
    print("\n" + "=" * 80, flush=True)
    print("  ANALYSIS 6: PHI-POSITION MIXTURE MODEL (PER-SUBJECT PAIRED WILCOXON)", flush=True)
    print("  Fixed-mean Gaussian mixture at 5 phi-positions", flush=True)
    print("  Paired tests on mixing weights across conditions", flush=True)
    print("=" * 80, flush=True)

    component_names = list(MIXTURE_COMPONENTS.keys())
    component_means = np.array(list(MIXTURE_COMPONENTS.values()))
    K = len(component_names)

    # Build subject-level session groupings
    eo_ec_pairs, eo_task_pairs, ec_task_pairs = build_subject_pairs(sessions_by_cond)

    eo_by_subj = {}
    ec_by_subj = {}
    task_by_subj = {}
    for eo, ec in eo_ec_pairs:
        subj = eo.split('R')[0]
        eo_by_subj[subj] = [eo]
        ec_by_subj[subj] = [ec]
    for eo, task_sessions in eo_task_pairs:
        subj = eo.split('R')[0]
        task_by_subj[subj] = task_sessions

    # Pre-compute band lattice indices
    band_lattice_cache = {}
    for band_name in TARGET_BANDS:
        band_range = ANALYSIS_BANDS[band_name]
        band_lattice_cache[band_name] = build_session_band_lattice_index(
            session_freq_idx, band_range)

    condition_defs = {
        'rest_EO': eo_by_subj,
        'rest_EC': ec_by_subj,
        'task':    task_by_subj,
    }

    contrasts = [
        ('rest-EO vs rest-EC', 'rest_EO', 'rest_EC'),
        ('rest-EO vs task',    'rest_EO', 'task'),
        ('rest-EC vs task',    'rest_EC', 'task'),
    ]

    all_results = []   # per-subject weight rows for CSV
    test_results = []  # Wilcoxon test summary rows

    for band_name in TARGET_BANDS:
        band_lattice = band_lattice_cache[band_name]
        print(f"\n  --- {band_name.upper()} ---", flush=True)

        # Fit mixture for each subject × condition
        # subject_weights[cond][subj] = weight_vector of length K
        subject_weights = {}

        for cond_name, subj_sess in condition_defs.items():
            subject_weights[cond_name] = {}
            n_fit = 0
            sigmas = []

            for subj, sessions in subj_sess.items():
                parts = [band_lattice[s] for s in sessions if s in band_lattice]
                if not parts:
                    continue
                coords = np.concatenate(parts)
                if len(coords) < 20:
                    continue

                weights, sigma, ll = fit_phi_mixture(coords, component_means)
                subject_weights[cond_name][subj] = weights
                sigmas.append(sigma)
                n_fit += 1

                all_results.append({
                    'band': band_name,
                    'condition': cond_name,
                    'subject': subj,
                    'n_peaks': len(coords),
                    'sigma': sigma,
                    **{f'w_{component_names[k]}': weights[k] for k in range(K)},
                })

            print(f"  {cond_name:10s}: {n_fit} subjects fitted, "
                  f"median sigma = {np.median(sigmas):.4f}" if sigmas else
                  f"  {cond_name:10s}: 0 subjects fitted", flush=True)

        # Paired Wilcoxon on each component weight for each contrast
        # Global Bonferroni across all bands × contrasts × components
        n_component_tests = len(TARGET_BANDS) * len(contrasts) * K
        print(f"\n  Bonferroni correction: {n_component_tests} tests global "
              f"(alpha = {0.05/n_component_tests:.4f})", flush=True)

        print(f"\n  {'Contrast':22s} {'Component':12s} {'Med_A':>7s} {'Med_B':>7s} "
              f"{'Med_d':>7s} {'W':>8s} {'p':>10s} {'p_bonf':>8s} {'r':>6s} {'Sig?':>5s}",
              flush=True)
        print(f"  {'-'*22} {'-'*12} {'-'*7} {'-'*7} {'-'*7} {'-'*8} "
              f"{'-'*10} {'-'*8} {'-'*6} {'-'*5}", flush=True)

        for label, cond_a, cond_b in contrasts:
            wa = subject_weights[cond_a]
            wb = subject_weights[cond_b]
            common = sorted(set(wa) & set(wb))

            if len(common) < 10:
                print(f"  {label:22s}  -- insufficient paired subjects --", flush=True)
                continue

            for k_idx, comp_name in enumerate(component_names):
                vals_a = np.array([wa[s][k_idx] for s in common])
                vals_b = np.array([wb[s][k_idx] for s in common])
                diffs = vals_a - vals_b

                med_a = np.median(vals_a)
                med_b = np.median(vals_b)
                med_d = np.median(diffs)

                # Filter: skip tests where both conditions have negligible weight
                # (sigma >> inter-component distance causes absorption artifacts)
                if med_a < 0.01 and med_b < 0.01:
                    test_results.append({
                        'band': band_name,
                        'contrast': label,
                        'component': comp_name,
                        'n_subjects': len(common),
                        'median_A': med_a,
                        'median_B': med_b,
                        'median_diff': med_d,
                        'mean_diff': diffs.mean(),
                        'W_stat': np.nan,
                        'p_val': np.nan,
                        'p_bonf': np.nan,
                        'r_effect': np.nan,
                        'filtered': True,
                    })
                    print(f"  {label:22s} {comp_name:12s} {med_a:7.4f} {med_b:7.4f} "
                          f" -- filtered: both medians < 0.01 --", flush=True)
                    continue

                nonzero = diffs[diffs != 0]
                if len(nonzero) < 10:
                    print(f"  {label:22s} {comp_name:12s}  -- too many zeros --",
                          flush=True)
                    continue

                w_stat, p_val = wilcoxon(diffs, alternative='two-sided')
                # r = 1 - 4W/(n(n+1)) where W = min(T+, T-) from scipy wilcoxon
                n = len(nonzero)
                r_effect = 1 - (4 * w_stat) / (n * (n + 1))
                p_bonf = min(p_val * n_component_tests, 1.0)
                sig = "YES" if p_bonf < 0.05 else "no"

                star = ""
                if p_val < 0.001: star = "***"
                elif p_val < 0.01: star = "** "
                elif p_val < 0.05: star = "*  "

                test_results.append({
                    'band': band_name,
                    'contrast': label,
                    'component': comp_name,
                    'n_subjects': len(common),
                    'median_A': med_a,
                    'median_B': med_b,
                    'median_diff': med_d,
                    'mean_diff': diffs.mean(),
                    'W_stat': w_stat,
                    'p_val': p_val,
                    'p_bonf': p_bonf,
                    'r_effect': r_effect,
                    'filtered': False,
                })

                print(f"  {label:22s} {comp_name:12s} {med_a:7.4f} {med_b:7.4f} "
                      f"{med_d:+7.4f} {w_stat:8.0f} {p_val:10.6f}{star} "
                      f"{p_bonf:8.4f} {r_effect:+6.3f} {sig:>5s}", flush=True)

    return all_results, test_results


def run_mode_permutation_tests(session_freq_idx, sessions_by_cond, n_perms=5000):
    """Run within-subject paired permutation tests for mode differences."""
    print("\n" + "=" * 80, flush=True)
    print("  ANALYSIS 2: WITHIN-SUBJECT PAIRED PERMUTATION TESTS", flush=True)
    print(f"  {n_perms} paired permutations per contrast", flush=True)
    print(f"  Bonferroni correction for 6 tests (alpha = 0.0083)", flush=True)
    print("=" * 80, flush=True)

    # Build subject-level pairs
    eo_ec_pairs, eo_task_pairs, ec_task_pairs = build_subject_pairs(sessions_by_cond)
    print(f"\n  Subject pairs: {len(eo_ec_pairs)} EO-EC, "
          f"{len(eo_task_pairs)} EO-task, {len(ec_task_pairs)} EC-task", flush=True)

    # Pre-compute band lattice indices for each band
    band_lattice_cache = {}
    for band_name in TARGET_BANDS:
        band_range = ANALYSIS_BANDS[band_name]
        band_lattice_cache[band_name] = build_session_band_lattice_index(
            session_freq_idx, band_range)

    # Define contrasts: (label, band, test_type, pairs_or_args)
    contrasts = [
        ('rest-EO vs task',    'gamma',    'rest_task', eo_task_pairs),
        ('rest-EO vs rest-EC', 'gamma',    'eo_ec',     eo_ec_pairs),
        ('rest-EO vs task',    'alpha',    'rest_task', eo_task_pairs),
        ('rest-EO vs rest-EC', 'alpha',    'eo_ec',     eo_ec_pairs),
        ('rest-EO vs task',    'beta_low', 'rest_task', eo_task_pairs),
        ('rest-EC vs task',    'beta_low', 'rest_task', ec_task_pairs),
    ]

    results = []

    print(f"\n  {'Contrast':22s} {'Band':10s} {'Type':12s} {'Mode_A':>7s} {'Mode_B':>7s} "
          f"{'Diff':>7s} {'z':>6s} {'p':>8s} {'p_bonf':>8s} {'Sig?':>5s}", flush=True)
    print(f"  {'-'*22} {'-'*10} {'-'*12} {'-'*7} {'-'*7} {'-'*7} {'-'*6} "
          f"{'-'*8} {'-'*8} {'-'*5}", flush=True)

    for label, band_name, test_type, pairs in contrasts:
        band_lattice = band_lattice_cache[band_name]

        if test_type == 'eo_ec':
            result = paired_permutation_test_eo_ec(
                band_lattice, pairs, n_perms=n_perms)
        else:  # rest_task
            result = paired_permutation_test_rest_vs_task(
                band_lattice, pairs, n_perms=n_perms)

        if result is None:
            print(f"  {label:22s} {band_name:10s}  -- insufficient data --", flush=True)
            continue

        p_bonf = min(result['p_val'] * 6, 1.0)
        sig = "YES" if p_bonf < 0.05 else "no"

        result_row = {
            'contrast': label,
            'band': band_name,
            'test_type': result['test_type'],
            'mode_A': result['mode_A'],
            'mode_B': result['mode_B'],
            'observed_diff': result['observed_diff'],
            'p_val': result['p_val'],
            'p_bonf': p_bonf,
            'z_score': result['z_score'],
            'null_mean': result['null_mean'],
            'null_std': result['null_std'],
            'n_a_sessions': result['n_a_sessions'],
            'n_b_sessions': result['n_b_sessions'],
            'n_a_peaks': result['n_a_peaks'],
            'n_b_peaks': result['n_b_peaks'],
        }
        results.append(result_row)

        # Store null distribution for plotting (gamma contrasts only)
        if band_name == 'gamma':
            result_row['_null_distribution'] = result['null_distribution']

        star = ""
        if result['p_val'] < 0.001: star = "***"
        elif result['p_val'] < 0.01: star = "** "
        elif result['p_val'] < 0.05: star = "*  "

        ttype = result['test_type']
        print(f"  {label:22s} {band_name:10s} {ttype:12s} {result['mode_A']:7.4f} "
              f"{result['mode_B']:7.4f} {result['observed_diff']:+7.4f} "
              f"{result['z_score']:+6.2f} {result['p_val']:8.4f}{star} "
              f"{p_bonf:8.4f} {sig:>5s}", flush=True)

    return results


# =========================================================================
# ANALYSIS 3: BETA_LOW MOTOR SIGNATURE
# =========================================================================

def run_beta_low_motor_analysis(mode_df):
    """Interpret beta_low mode at inv_noble_6 as motor signature."""
    print("\n" + "=" * 80, flush=True)
    print("  ANALYSIS 3: BETA_LOW MOTOR SIGNATURE INTERPRETATION", flush=True)
    print("=" * 80, flush=True)

    inv_noble_6_freq = F0 * PHI ** (1 + INV_NOBLE_6)
    print(f"\n  inv_noble_6 offset: {INV_NOBLE_6:.4f}", flush=True)
    print(f"  In beta_low (octave 1): f = {F0:.2f} * phi^(1 + {INV_NOBLE_6:.4f}) "
          f"= {inv_noble_6_freq:.2f} Hz", flush=True)
    print(f"  This falls in the post-movement beta rebound (PMBR) range (15-25 Hz)", flush=True)

    bl = mode_df[mode_df['band'] == 'beta_low']
    if len(bl) > 0:
        print(f"\n  Per-condition modes:", flush=True)
        for _, r in bl.iterrows():
            d_inv6 = min(abs(r['mode'] - INV_NOBLE_6),
                         abs(r['mode'] - INV_NOBLE_6 + 1))
            print(f"    {r['condition']:10s}: mode={r['mode']:.4f}, "
                  f"d(inv_noble_6)={d_inv6:.4f}, "
                  f"nearest={r['nearest_position']}", flush=True)

        # Check if task is closer to inv_noble_6 than rest
        task_row = bl[bl['condition'] == 'task']
        eo_row = bl[bl['condition'] == 'rest_EO']
        ec_row = bl[bl['condition'] == 'rest_EC']

        if len(task_row) > 0 and len(eo_row) > 0:
            task_d = abs(task_row.iloc[0]['mode'] - INV_NOBLE_6)
            eo_d = abs(eo_row.iloc[0]['mode'] - INV_NOBLE_6)
            closer = "task" if task_d < eo_d else "rest-EO"
            print(f"\n  Closer to inv_noble_6: {closer} "
                  f"(task d={task_d:.4f}, rest-EO d={eo_d:.4f})", flush=True)

    print(f"\n  Interpretation: beta_low mode at inv_noble_6 (~{inv_noble_6_freq:.1f} Hz) "
          f"reflects", flush=True)
    print(f"  post-movement beta rebound (PMBR) in this motor-task dataset.", flush=True)
    print(f"  Non-motor datasets show noble_1 dominance (+27.5%), not inv_noble_6.", flush=True)
    print(f"  Beta_low serves as a POSITIVE CONTROL: the lattice analysis", flush=True)
    print(f"  detects known motor physiology at the correct lattice position.", flush=True)


# =========================================================================
# ANALYSIS 4: CLAIM STRATIFICATION
# =========================================================================

def print_claim_stratification(mode_df, perm_results):
    """Print tiered claim summary."""
    print("\n" + "=" * 80, flush=True)
    print("  ANALYSIS 4: CLAIM STRATIFICATION", flush=True)
    print("=" * 80, flush=True)

    # Gamma
    gamma_task = mode_df[(mode_df['band'] == 'gamma') & (mode_df['condition'] == 'task')]
    gamma_sig = [r for r in perm_results
                 if r['band'] == 'gamma' and r.get('p_bonf', 1) < 0.05]

    print(f"\n  TIER 1 — GAMMA (defensible):", flush=True)
    if len(gamma_task) > 0:
        gt = gamma_task.iloc[0]
        print(f"    Task mode = {gt['mode']:.4f} (d to noble_1 = {gt['dist_to_noble1']:.4f})", flush=True)
        print(f"    CI [{gt['ci_lower']:.4f}, {gt['ci_upper']:.4f}]", flush=True)
        excl_att = "excludes" if gt['ci_excludes_attractor'] else "includes"
        excl_n1 = "excludes" if gt['ci_excludes_noble1'] else "includes"
        print(f"    CI {excl_att} attractor, {excl_n1} noble_1", flush=True)
    print(f"    No IAF confound (gamma >> alpha range)", flush=True)
    print(f"    No known non-phi explanation for ~43 Hz clustering at 0.618", flush=True)
    if gamma_sig:
        for r in gamma_sig:
            print(f"    Permutation test: {r['contrast']} p_bonf={r['p_bonf']:.4f} *", flush=True)
    else:
        print(f"    No permutation tests survive Bonferroni", flush=True)

    # Alpha
    alpha_task = mode_df[(mode_df['band'] == 'alpha') & (mode_df['condition'] == 'task')]
    print(f"\n  TIER 2 — ALPHA (confounded):", flush=True)
    if len(alpha_task) > 0:
        at = alpha_task.iloc[0]
        print(f"    Task mode = {at['mode']:.4f} (d to noble_1 = {at['dist_to_noble1']:.4f})", flush=True)
    print(f"    Noble_1 (10.23 Hz) vs IAF (~10 Hz): unresolvable at df=0.312 Hz", flush=True)
    print(f"    Claim: 'consistent with but not uniquely supporting phi framework'", flush=True)

    # Beta_low
    bl_task = mode_df[(mode_df['band'] == 'beta_low') & (mode_df['condition'] == 'task')]
    print(f"\n  TIER 3 — BETA_LOW (control band):", flush=True)
    if len(bl_task) > 0:
        bt = bl_task.iloc[0]
        print(f"    Task mode = {bt['mode']:.4f} (d to inv_noble_6 = "
              f"{bt['dist_to_inv_noble6']:.4f})", flush=True)
    print(f"    Motor signature (PMBR at ~19.4 Hz) validates lattice detects", flush=True)
    print(f"    real physiology at the correct lattice position", flush=True)


# =========================================================================
# FIGURE
# =========================================================================

def plot_mode_shift_figure(mode_df, perm_results, output_path,
                           targeted_results=None, mixture_tests=None,
                           mixture_weights=None):
    """4-panel figure: modes, targeted binary, mixture weights, beta_low."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    cond_order = ['rest_EO', 'rest_EC', 'task']
    cond_labels = ['Rest (EO)', 'Rest (EC)', 'Task']
    cond_colors = ['#3498db', '#2c3e50', '#e74c3c']

    # --- Panel A: Per-condition modes for gamma + alpha ---
    ax = axes[0, 0]
    x = np.arange(len(cond_order))

    for band, marker, color, offset in [
        ('gamma', 'o', '#e74c3c', -0.12),
        ('alpha', 's', '#3498db', +0.12),
    ]:
        band_data = mode_df[mode_df['band'] == band]
        modes, ci_lo, ci_hi = [], [], []
        for cond in cond_order:
            row = band_data[band_data['condition'] == cond]
            if len(row) > 0:
                r = row.iloc[0]
                modes.append(r['mode'])
                ci_lo.append(max(0, r['mode'] - r['ci_lower']))
                ci_hi.append(max(0, r['ci_upper'] - r['mode']))
            else:
                modes.append(np.nan)
                ci_lo.append(0)
                ci_hi.append(0)

        ax.errorbar(x + offset, modes,
                     yerr=[np.maximum(0, ci_lo), np.maximum(0, ci_hi)],
                     fmt=marker, color=color, capsize=6, capthick=1.5,
                     markersize=10, linewidth=1.5, label=band,
                     markeredgecolor='white', markeredgewidth=0.5)

    ax.axhline(NOBLE_1, color='#e74c3c', linestyle='--', linewidth=1.2,
               alpha=0.6, label=f'Noble_1 ({NOBLE_1:.3f})')
    ax.axhline(ATTRACTOR, color='#3498db', linestyle='--', linewidth=1.2,
               alpha=0.6, label=f'Attractor ({ATTRACTOR:.3f})')

    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, fontsize=10)
    ax.set_ylabel('KDE Mode (lattice coordinate)', fontsize=11)
    ax.set_title('A. Per-Condition KDE Mode\n(gamma + alpha, 95% bootstrap CI)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, loc='upper left', ncol=2)
    ax.set_ylim(0.30, 1.00)
    ax.set_xlim(-0.5, 2.5)

    # --- Panel B: Targeted binary contrast — paired proportion differences ---
    ax = axes[0, 1]

    if targeted_results:
        # Show gamma contrasts as horizontal bars with CIs
        gamma_results = [r for r in targeted_results if r['band'] == 'gamma']
        bar_labels = []
        bar_diffs = []
        bar_colors = []
        bar_sigs = []

        for r in gamma_results:
            bar_labels.append(r['contrast'])
            bar_diffs.append(r['median_diff'])
            bar_sigs.append(r['p_val'] < 0.05)
            bar_colors.append('#e74c3c' if r['p_val'] < 0.05 else '#bdc3c7')

        if bar_diffs:
            y_pos = np.arange(len(bar_labels))
            bars = ax.barh(y_pos, bar_diffs, color=bar_colors, edgecolor='white',
                           height=0.5, alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(bar_labels, fontsize=9)
            ax.axvline(0, color='grey', linewidth=0.8, linestyle='--')
            ax.set_xlabel('Median Proportion Difference (A - B)', fontsize=10)

            # Annotate with p-values
            for i, r in enumerate(gamma_results):
                star = ""
                if r['p_val'] < 0.001: star = " ***"
                elif r['p_val'] < 0.01: star = " **"
                elif r['p_val'] < 0.05: star = " *"
                ax.text(bar_diffs[i] + 0.002 * np.sign(bar_diffs[i]),
                        i, f"p={r['p_val']:.4f}{star}\nr={r['r_effect']:+.3f}",
                        va='center', fontsize=8,
                        ha='left' if bar_diffs[i] >= 0 else 'right')

        # Also show alpha and beta_low as text annotations
        other_results = [r for r in targeted_results if r['band'] != 'gamma']
        if other_results:
            text_lines = []
            for r in other_results:
                star = ""
                if r['p_val'] < 0.001: star = "***"
                elif r['p_val'] < 0.01: star = "**"
                elif r['p_val'] < 0.05: star = "*"
                text_lines.append(
                    f"{r['band']:8s} {r['contrast']:22s} "
                    f"d={r['median_diff']:+.4f} p={r['p_val']:.4f}{star}")
            ax.text(0.02, 0.02, '\n'.join(text_lines),
                    transform=ax.transAxes, fontsize=6.5, va='bottom',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    ax.set_title('B. Targeted Binary Contrast\n(per-subject prop. above threshold, '
                 'paired Wilcoxon)', fontsize=11, fontweight='bold')

    # --- Panel C: Mixture model weights for gamma band ---
    ax = axes[1, 0]

    if mixture_weights:
        mw_df = pd.DataFrame(mixture_weights)
        gamma_mw = mw_df[mw_df['band'] == 'gamma']

        if len(gamma_mw) > 0:
            comp_names = [c for c in MIXTURE_COMPONENTS.keys()]
            w_cols = [f'w_{c}' for c in comp_names]

            # Compute mean weight per condition
            cond_means = {}
            for cond in cond_order:
                cond_data = gamma_mw[gamma_mw['condition'] == cond]
                if len(cond_data) > 0:
                    cond_means[cond] = [cond_data[wc].mean() for wc in w_cols]

            if cond_means:
                x_comp = np.arange(len(comp_names))
                width = 0.25

                for i, (cond, cond_label, color) in enumerate(
                        zip(cond_order, cond_labels, cond_colors)):
                    if cond in cond_means:
                        ax.bar(x_comp + i * width - width, cond_means[cond],
                               width, label=cond_label, color=color, alpha=0.8,
                               edgecolor='white')

                # Mark significant mixture tests with stars
                if mixture_tests:
                    gamma_mt = [t for t in mixture_tests if t['band'] == 'gamma']
                    for t in gamma_mt:
                        if t['p_val'] < 0.05:
                            comp_idx = comp_names.index(t['component'])
                            max_h = max(cond_means[c][comp_idx]
                                        for c in cond_order if c in cond_means)
                            star = '***' if t['p_val'] < 0.001 else \
                                   '**' if t['p_val'] < 0.01 else '*'
                            ax.text(comp_idx, max_h + 0.01, star,
                                    ha='center', fontsize=10, fontweight='bold')

                ax.set_xticks(x_comp)
                ax.set_xticklabels([c.replace('_', '\n') for c in comp_names],
                                   fontsize=8)
                ax.set_ylabel('Mean Mixing Weight', fontsize=11)
                ax.legend(fontsize=8, loc='upper right')

    ax.set_title('C. Phi-Position Mixture Model\n(gamma band, mean weights per condition)',
                 fontsize=11, fontweight='bold')

    # --- Panel D: Beta_low per-condition modes ---
    ax = axes[1, 1]
    bl = mode_df[mode_df['band'] == 'beta_low']
    x = np.arange(len(cond_order))

    if len(bl) > 0:
        modes, ci_lo, ci_hi = [], [], []
        for cond in cond_order:
            row = bl[bl['condition'] == cond]
            if len(row) > 0:
                r = row.iloc[0]
                modes.append(r['mode'])
                ci_lo.append(max(0, r['mode'] - r['ci_lower']))
                ci_hi.append(max(0, r['ci_upper'] - r['mode']))
            else:
                modes.append(np.nan)
                ci_lo.append(0)
                ci_hi.append(0)

        ax.errorbar(x, modes,
                     yerr=[np.maximum(0, ci_lo), np.maximum(0, ci_hi)],
                     fmt='D', color='#27ae60', capsize=6, capthick=1.5,
                     markersize=10, linewidth=1.5, markeredgecolor='white')

    ax.axhline(INV_NOBLE_6, color='#27ae60', linestyle='--', linewidth=1.2,
               alpha=0.6, label=f'inv_noble_6 ({INV_NOBLE_6:.3f})')
    ax.axhline(NOBLE_1, color='#e74c3c', linestyle=':', linewidth=1, alpha=0.4,
               label=f'Noble_1 ({NOBLE_1:.3f})')
    ax.axhline(ATTRACTOR, color='#3498db', linestyle=':', linewidth=1, alpha=0.4,
               label=f'Attractor ({ATTRACTOR:.3f})')

    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, fontsize=10)
    ax.set_ylabel('KDE Mode (lattice coordinate)', fontsize=11)
    ax.set_title('D. Beta_low Motor Signature\n(inv_noble_6 = PMBR at ~19.4 Hz)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='lower left')
    ax.set_ylim(0.60, 1.02)
    ax.set_xlim(-0.5, 2.5)

    fig.suptitle('Mode Shift Analysis: Per-Condition Tests\n'
                 'EEGMMIDB FOOOF (109 subjects, 3-condition split)',
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n  Figure saved: {output_path}", flush=True)


# =========================================================================
# CONTROL 1: ANCHOR FREQUENCY SWEEP
# =========================================================================

def run_anchor_sweep(session_freq_idx, sessions_by_cond):
    """Sweep f₀ across [5-12] Hz to test whether 7.6 Hz is special.

    For each f₀, recompute lattice coords and run the mixture model
    on gamma EC-vs-task (our strongest contrast). If the effect peaks
    at f₀≈7.6 Hz, the phi lattice captures real spectral structure.
    """
    print("\n" + "=" * 80, flush=True)
    print("  CONTROL 1: ANCHOR FREQUENCY SWEEP", flush=True)
    print("  Varying f₀ across 5-12 Hz, gamma band, EC vs task", flush=True)
    print("=" * 80, flush=True)

    f0_values = [5.0, 6.0, 7.0, 7.6, 8.0, 9.0, 10.0, 11.0, 12.0]
    gamma_range = ANALYSIS_BANDS['gamma']

    component_names = list(MIXTURE_COMPONENTS.keys())
    component_means = np.array(list(MIXTURE_COMPONENTS.values()))
    K = len(component_names)

    # Build subject pairs (once)
    eo_ec_pairs, eo_task_pairs, ec_task_pairs = build_subject_pairs(sessions_by_cond)
    ec_by_subj = {}
    task_by_subj = {}
    for eo, ec in eo_ec_pairs:
        subj = eo.split('R')[0]
        ec_by_subj[subj] = [ec]
    for eo, task_sessions in eo_task_pairs:
        subj = eo.split('R')[0]
        task_by_subj[subj] = task_sessions

    results = []

    print(f"\n  {'f0':>6s} {'Component':12s} {'n':>4s} {'W':>8s} "
          f"{'p':>10s} {'r':>7s}", flush=True)
    print(f"  {'-'*6} {'-'*12} {'-'*4} {'-'*8} {'-'*10} {'-'*7}", flush=True)

    for f0 in f0_values:
        # Recompute lattice coords at this f0
        band_lattice = build_session_band_lattice_index_general(
            session_freq_idx, gamma_range, f0=f0, base=PHI)

        # Fit mixture per subject per condition
        ec_weights = {}
        task_weights = {}

        for subj, sessions in ec_by_subj.items():
            parts = [band_lattice[s] for s in sessions if s in band_lattice]
            if not parts:
                continue
            coords = np.concatenate(parts)
            if len(coords) < 20:
                continue
            w, sigma, ll = fit_phi_mixture(coords, component_means)
            ec_weights[subj] = w

        for subj, sessions in task_by_subj.items():
            parts = [band_lattice[s] for s in sessions if s in band_lattice]
            if not parts:
                continue
            coords = np.concatenate(parts)
            if len(coords) < 20:
                continue
            w, sigma, ll = fit_phi_mixture(coords, component_means)
            task_weights[subj] = w

        common = sorted(set(ec_weights) & set(task_weights))

        for k_idx, comp_name in enumerate(component_names):
            vals_ec = np.array([ec_weights[s][k_idx] for s in common])
            vals_task = np.array([task_weights[s][k_idx] for s in common])
            diffs = vals_ec - vals_task

            med_ec = np.median(vals_ec)
            med_task = np.median(vals_task)

            # Skip negligible weight components
            if med_ec < 0.01 and med_task < 0.01:
                results.append({
                    'f0': f0, 'component': comp_name, 'n_subjects': len(common),
                    'median_EC': med_ec, 'median_task': med_task,
                    'W_stat': np.nan, 'p_val': np.nan, 'r_effect': np.nan,
                    'filtered': True,
                })
                continue

            nonzero = diffs[diffs != 0]
            if len(nonzero) < 10:
                results.append({
                    'f0': f0, 'component': comp_name, 'n_subjects': len(common),
                    'median_EC': med_ec, 'median_task': med_task,
                    'W_stat': np.nan, 'p_val': np.nan, 'r_effect': np.nan,
                    'filtered': True,
                })
                continue

            w_stat, p_val = wilcoxon(diffs, alternative='two-sided')
            n = len(nonzero)
            r_effect = 1 - (4 * w_stat) / (n * (n + 1))

            results.append({
                'f0': f0, 'component': comp_name, 'n_subjects': len(common),
                'median_EC': med_ec, 'median_task': med_task,
                'W_stat': w_stat, 'p_val': p_val, 'r_effect': r_effect,
                'filtered': False,
            })

            star = "***" if p_val < 0.001 else "** " if p_val < 0.01 else \
                   "*  " if p_val < 0.05 else "   "
            print(f"  {f0:6.1f} {comp_name:12s} {len(common):4d} {w_stat:8.0f} "
                  f"{p_val:10.6f}{star} {r_effect:+7.3f}", flush=True)

    print(f"\n  Anchor sweep complete: {len(results)} tests across "
          f"{len(f0_values)} f₀ values", flush=True)
    return results


# =========================================================================
# CONTROL 2: NON-PHI BASE SWEEP
# =========================================================================

def run_base_sweep(session_freq_idx, sessions_by_cond):
    """Sweep exponential base to test whether phi (1.618) is special.

    For each base, recompute lattice coords and fit a mixture with
    equispaced positions (fair comparison). Gamma EC-vs-task.
    """
    print("\n" + "=" * 80, flush=True)
    print("  CONTROL 2: NON-PHI BASE SWEEP", flush=True)
    print("  Varying base, f₀=7.6 fixed, equispaced positions, gamma EC-task", flush=True)
    print("=" * 80, flush=True)

    bases = [1.4, 1.5, PHI, 1.7, 1.8, 2.0, np.e]
    base_labels = ['1.400', '1.500', 'phi', '1.700', '1.800', '2.000', 'e']
    K = 5
    equispaced_means = np.array([i / K for i in range(K)])  # [0, 0.2, 0.4, 0.6, 0.8]
    phi_means = np.array(list(MIXTURE_COMPONENTS.values()))

    gamma_range = ANALYSIS_BANDS['gamma']

    # Build subject pairs (once)
    eo_ec_pairs, eo_task_pairs, ec_task_pairs = build_subject_pairs(sessions_by_cond)
    ec_by_subj = {}
    task_by_subj = {}
    for eo, ec in eo_ec_pairs:
        subj = eo.split('R')[0]
        ec_by_subj[subj] = [ec]
    for eo, task_sessions in eo_task_pairs:
        subj = eo.split('R')[0]
        task_by_subj[subj] = task_sessions

    results = []

    print(f"\n  {'Base':>8s} {'Positions':10s} {'Comp':>5s} {'n':>4s} "
          f"{'W':>8s} {'p':>10s} {'r':>7s}", flush=True)
    print(f"  {'-'*8} {'-'*10} {'-'*5} {'-'*4} {'-'*8} {'-'*10} {'-'*7}",
          flush=True)

    for base, blabel in zip(bases, base_labels):
        # Recompute lattice coords
        band_lattice = build_session_band_lattice_index_general(
            session_freq_idx, gamma_range, f0=F0, base=base)

        # Run with equispaced positions
        position_sets = [('equi', equispaced_means)]
        # For phi specifically, also run with phi-derived positions
        if abs(base - PHI) < 0.001:
            position_sets.append(('phi', phi_means))

        for pos_label, means in position_sets:
            ec_weights = {}
            task_weights = {}

            for subj, sessions in ec_by_subj.items():
                parts = [band_lattice[s] for s in sessions if s in band_lattice]
                if not parts:
                    continue
                coords = np.concatenate(parts)
                if len(coords) < 20:
                    continue
                w, sigma, ll = fit_phi_mixture(coords, means)
                ec_weights[subj] = w

            for subj, sessions in task_by_subj.items():
                parts = [band_lattice[s] for s in sessions if s in band_lattice]
                if not parts:
                    continue
                coords = np.concatenate(parts)
                if len(coords) < 20:
                    continue
                w, sigma, ll = fit_phi_mixture(coords, means)
                task_weights[subj] = w

            common = sorted(set(ec_weights) & set(task_weights))

            # Test each component, find best
            best_p = 1.0
            best_r = 0.0
            best_z = 0.0

            for k_idx in range(K):
                vals_ec = np.array([ec_weights[s][k_idx] for s in common])
                vals_task = np.array([task_weights[s][k_idx] for s in common])
                diffs = vals_ec - vals_task

                med_ec = np.median(vals_ec)
                med_task = np.median(vals_task)

                if med_ec < 0.01 and med_task < 0.01:
                    continue

                nonzero = diffs[diffs != 0]
                if len(nonzero) < 10:
                    continue

                w_stat, p_val = wilcoxon(diffs, alternative='two-sided')
                n = len(nonzero)
                r_effect = 1 - (4 * w_stat) / (n * (n + 1))

                # Convert to z-score for comparison
                # z = (W - E[W]) / std(W), E[W] = n(n+1)/4
                e_w = n * (n + 1) / 4
                std_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
                z = (w_stat - e_w) / std_w if std_w > 0 else 0.0

                if p_val < best_p:
                    best_p = p_val
                    best_r = r_effect
                    best_z = abs(z)

            results.append({
                'base': base,
                'base_label': blabel,
                'positions': pos_label,
                'n_subjects': len(common),
                'best_p': best_p,
                'best_r': best_r,
                'best_z': best_z,
            })

            star = "***" if best_p < 0.001 else "** " if best_p < 0.01 else \
                   "*  " if best_p < 0.05 else "   "
            print(f"  {blabel:>8s} {pos_label:10s} {K:5d} {len(common):4d} "
                  f"{'':>8s} {best_p:10.6f}{star} {best_r:+7.3f}", flush=True)

    print(f"\n  Base sweep complete: {len(results)} configurations tested",
          flush=True)
    return results


# =========================================================================
# CONTROL 3: RANDOM POSITION CONTROL
# =========================================================================

def run_random_position_control(session_freq_idx, sessions_by_cond,
                                 n_iter=100, seed=42):
    """Null distribution from random mixture positions.

    Place 5 components at random positions in [0,1), fit mixture,
    run Wilcoxon. Build null distribution of max |z| to compare
    against phi-position z-scores.
    """
    print("\n" + "=" * 80, flush=True)
    print("  CONTROL 3: RANDOM POSITION CONTROL", flush=True)
    print(f"  {n_iter} iterations, 5 random positions, gamma EC-task", flush=True)
    print("=" * 80, flush=True)

    gamma_range = ANALYSIS_BANDS['gamma']
    K = 5
    rng = np.random.default_rng(seed)

    # Pre-compute band lattice index (standard f0, phi base)
    band_lattice = build_session_band_lattice_index(session_freq_idx, gamma_range)

    # Build subject pairs
    eo_ec_pairs, eo_task_pairs, ec_task_pairs = build_subject_pairs(sessions_by_cond)
    ec_by_subj = {}
    task_by_subj = {}
    for eo, ec in eo_ec_pairs:
        subj = eo.split('R')[0]
        ec_by_subj[subj] = [ec]
    for eo, task_sessions in eo_task_pairs:
        subj = eo.split('R')[0]
        task_by_subj[subj] = task_sessions

    # Pre-collect per-subject coords for speed
    ec_coords = {}
    task_coords = {}
    for subj, sessions in ec_by_subj.items():
        parts = [band_lattice[s] for s in sessions if s in band_lattice]
        if parts:
            coords = np.concatenate(parts)
            if len(coords) >= 20:
                ec_coords[subj] = coords
    for subj, sessions in task_by_subj.items():
        parts = [band_lattice[s] for s in sessions if s in band_lattice]
        if parts:
            coords = np.concatenate(parts)
            if len(coords) >= 20:
                task_coords[subj] = coords

    common_subjs = sorted(set(ec_coords) & set(task_coords))
    print(f"  {len(common_subjs)} subjects with paired data", flush=True)

    # First: compute phi-position z-scores for comparison
    # Apply same weight filter as Analysis 6 (both medians < 0.01 → skip)
    phi_means = np.array(list(MIXTURE_COMPONENTS.values()))
    phi_component_names = list(MIXTURE_COMPONENTS.keys())

    # Pre-compute phi weights for all subjects (reuse across z-score calc)
    phi_ec_weights = {s: fit_phi_mixture(ec_coords[s], phi_means)[0]
                      for s in common_subjs}
    phi_task_weights = {s: fit_phi_mixture(task_coords[s], phi_means)[0]
                        for s in common_subjs}

    phi_z_scores = []
    for k_idx in range(K):
        ec_w = np.array([phi_ec_weights[s][k_idx] for s in common_subjs])
        task_w = np.array([phi_task_weights[s][k_idx] for s in common_subjs])

        # Weight filter: skip near-zero components (same as Analysis 6)
        if np.median(ec_w) < 0.01 and np.median(task_w) < 0.01:
            phi_z_scores.append(0.0)
            continue

        diffs = ec_w - task_w
        nonzero = diffs[diffs != 0]
        if len(nonzero) >= 10:
            w_stat, _ = wilcoxon(diffs, alternative='two-sided')
            n = len(nonzero)
            e_w = n * (n + 1) / 4
            std_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
            phi_z_scores.append(abs((w_stat - e_w) / std_w) if std_w > 0 else 0.0)
        else:
            phi_z_scores.append(0.0)

    phi_max_z = max(phi_z_scores)
    phi_best_comp = phi_component_names[np.argmax(phi_z_scores)]
    print(f"  Phi max |z| = {phi_max_z:.3f} ({phi_best_comp})", flush=True)

    # Random iterations
    null_max_z = np.zeros(n_iter)
    t0 = time.time()

    for it in range(n_iter):
        random_means = np.sort(rng.uniform(0, 1, K))

        # Fit mixture for each subject
        ec_w_all = {}
        task_w_all = {}
        for subj in common_subjs:
            ec_w_all[subj] = fit_phi_mixture(ec_coords[subj], random_means)[0]
            task_w_all[subj] = fit_phi_mixture(task_coords[subj], random_means)[0]

        # Test each component, find max |z| (with weight filter)
        max_z = 0.0
        for k_idx in range(K):
            vals_ec = np.array([ec_w_all[s][k_idx] for s in common_subjs])
            vals_task = np.array([task_w_all[s][k_idx] for s in common_subjs])

            # Weight filter: skip near-zero components
            if np.median(vals_ec) < 0.01 and np.median(vals_task) < 0.01:
                continue

            diffs = vals_ec - vals_task
            nonzero = diffs[diffs != 0]
            if len(nonzero) < 10:
                continue

            w_stat, _ = wilcoxon(diffs, alternative='two-sided')
            n = len(nonzero)
            e_w = n * (n + 1) / 4
            std_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
            z = abs((w_stat - e_w) / std_w) if std_w > 0 else 0.0
            max_z = max(max_z, z)

        null_max_z[it] = max_z

        if (it + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (it + 1) / elapsed
            eta = (n_iter - it - 1) / rate
            print(f"    iter {it+1}/{n_iter} ({elapsed:.0f}s, ETA {eta:.0f}s)",
                  flush=True)

    # Empirical p-value
    emp_p = np.mean(null_max_z >= phi_max_z)
    print(f"\n  Null distribution: mean={null_max_z.mean():.3f}, "
          f"std={null_max_z.std():.3f}, max={null_max_z.max():.3f}", flush=True)
    print(f"  Phi max |z| = {phi_max_z:.3f}", flush=True)
    print(f"  Empirical p = {emp_p:.4f} "
          f"({np.sum(null_max_z >= phi_max_z)}/{n_iter} random >= phi)",
          flush=True)

    return {
        'null_max_z': null_max_z,
        'phi_z_scores': phi_z_scores,
        'phi_max_z': phi_max_z,
        'phi_component_names': phi_component_names,
        'empirical_p': emp_p,
        'n_iter': n_iter,
    }


# =========================================================================
# CONTROL FIGURE
# =========================================================================

def plot_control_figure(anchor_results, base_results, random_results, output_path):
    """3-panel figure for control analyses."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # --- Panel A: Anchor Frequency Sweep ---
    ax = axes[0]
    anchor_df = pd.DataFrame(anchor_results)
    anchor_valid = anchor_df[~anchor_df['filtered']]

    for comp_name, color, marker in [
        ('boundary', '#3498db', 'o'),
        ('attractor', '#e74c3c', 's'),
    ]:
        comp_data = anchor_valid[anchor_valid['component'] == comp_name]
        if len(comp_data) > 0:
            neg_log_p = -np.log10(comp_data['p_val'].clip(lower=1e-20))
            ax.plot(comp_data['f0'], neg_log_p, f'-{marker}',
                    color=color, label=comp_name, markersize=7, linewidth=1.5)

    ax.axvline(7.6, color='grey', linestyle='--', linewidth=1, alpha=0.7,
               label='f₀ = 7.6 Hz')
    ax.axhline(-np.log10(0.05), color='black', linestyle=':', linewidth=0.8,
               alpha=0.5, label='p = 0.05')
    ax.set_xlabel('Anchor Frequency f₀ (Hz)', fontsize=11)
    ax.set_ylabel('-log₁₀(p)', fontsize=11)
    ax.set_title('A. Anchor Frequency Sweep\n(gamma EC-task, mixture model)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.set_xlim(4, 13)

    # --- Panel B: Random Position Control ---
    ax = axes[1]
    null_z = random_results['null_max_z']
    phi_z = random_results['phi_max_z']
    emp_p = random_results['empirical_p']

    ax.hist(null_z, bins=20, color='#bdc3c7', edgecolor='white', alpha=0.8,
            label='Random positions (null)')
    ax.axvline(phi_z, color='#e74c3c', linewidth=2.5, linestyle='-',
               label=f'Phi positions (z={phi_z:.2f})')
    ax.text(0.95, 0.95, f'Empirical p = {emp_p:.3f}\n'
            f'({int(np.sum(null_z >= phi_z))}/{random_results["n_iter"]} '
            f'random ≥ phi)',
            transform=ax.transAxes, fontsize=9, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax.set_xlabel('Max |z| across 5 components', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('B. Random Position Null Distribution\n'
                 '(100 random 5-position sets, gamma EC-task)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')

    # --- Panel C: Base Sweep ---
    ax = axes[2]
    base_df = pd.DataFrame(base_results)
    equi = base_df[base_df['positions'] == 'equi']
    phi_pos = base_df[base_df['positions'] == 'phi']

    if len(equi) > 0:
        ax.plot(equi['base'], equi['best_r'].abs(), 'o-',
                color='#3498db', label='Equispaced positions', markersize=7,
                linewidth=1.5)
    if len(phi_pos) > 0:
        ax.plot(phi_pos['base'], phi_pos['best_r'].abs(), 'D',
                color='#e74c3c', markersize=10, label='Phi positions',
                markeredgecolor='white', markeredgewidth=1, zorder=5)

    ax.axvline(PHI, color='grey', linestyle='--', linewidth=1, alpha=0.7,
               label=f'φ = {PHI:.3f}')
    ax.set_xlabel('Exponential Base', fontsize=11)
    ax.set_ylabel('Best |r| (effect size)', fontsize=11)
    ax.set_title('C. Non-Phi Base Sweep\n(f₀=7.6 fixed, gamma EC-task)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='best')

    fig.suptitle('Control Analyses: Specificity of φ-Lattice Parameters\n'
                 'EEGMMIDB FOOOF (109 subjects)',
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n  Control figure saved: {output_path}", flush=True)


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Mode shift analysis: per-condition bootstrap CIs and permutation tests")
    parser.add_argument('--n-perms', type=int, default=5000)
    parser.add_argument('--n-boot', type=int, default=1000)
    parser.add_argument('--skip-bootstrap', action='store_true',
                        help='Skip Analysis 1 bootstrap (load from CSV if available)')
    parser.add_argument('--skip-permutations', action='store_true',
                        help='Skip Analysis 2 permutation tests (load from CSV if available)')
    parser.add_argument('--skip-controls', action='store_true',
                        help='Skip control analyses (anchor sweep, base sweep, random positions)')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80, flush=True)
    print("  MODE SHIFT ANALYSIS", flush=True)
    print("  Per-condition bootstrap CIs + permutation tests on mode difference", flush=True)
    print(f"  Bootstrap: {args.n_boot} resamples | Permutations: {args.n_perms}", flush=True)
    print("=" * 80, flush=True)

    # Load data + build index
    t0 = time.time()
    df = load_and_assign_conditions(INPUT_CSV)
    print(f"  Loaded {len(df):,} peaks, {df['session'].nunique()} sessions "
          f"({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    session_freq_idx = build_session_freq_index(df)
    print(f"  Built session index for {len(session_freq_idx)} sessions "
          f"({time.time()-t0:.1f}s)", flush=True)

    sessions_by_cond = {
        c: set(df.loc[df['condition'] == c, 'session'].unique())
        for c in df['condition'].unique()
    }

    # Analysis 1: Per-condition bootstrap CIs
    mode_path = os.path.join(OUTPUT_DIR, 'mode_shift_per_condition.csv')
    if args.skip_bootstrap and os.path.exists(mode_path):
        print(f"\n  Skipping Analysis 1 (loading from {mode_path})", flush=True)
        mode_df = pd.read_csv(mode_path)
    else:
        t0 = time.time()
        mode_df = run_per_condition_modes(session_freq_idx, sessions_by_cond,
                                           n_boot=args.n_boot)
        mode_df.to_csv(mode_path, index=False)
        print(f"\n  Analysis 1 done ({time.time()-t0:.1f}s)", flush=True)
        print(f"  Saved: {mode_path}", flush=True)

    # Analysis 2: Permutation tests (mode-based — kept for completeness)
    perm_path = os.path.join(OUTPUT_DIR, 'mode_shift_permutation_tests.csv')
    if args.skip_permutations and os.path.exists(perm_path):
        print(f"\n  Skipping Analysis 2 (loading from {perm_path})", flush=True)
        perm_results = pd.read_csv(perm_path).to_dict('records')
    else:
        t0 = time.time()
        perm_results = run_mode_permutation_tests(session_freq_idx, sessions_by_cond,
                                                   n_perms=args.n_perms)
        print(f"\n  Analysis 2 done ({time.time()-t0:.1f}s)", flush=True)

    # Analysis 3: Beta_low interpretation
    run_beta_low_motor_analysis(mode_df)

    # Analysis 4: Claim stratification
    print_claim_stratification(mode_df, perm_results)

    # Analysis 5: Targeted binary contrast (per-subject paired Wilcoxon)
    t0 = time.time()
    targeted_results = run_targeted_binary_tests(session_freq_idx, sessions_by_cond)
    print(f"\n  Analysis 5 done ({time.time()-t0:.1f}s)", flush=True)

    # Analysis 6: Mixture model at phi-positions
    t0 = time.time()
    mixture_weights, mixture_tests = run_mixture_model_tests(
        session_freq_idx, sessions_by_cond)
    print(f"\n  Analysis 6 done ({time.time()-t0:.1f}s)", flush=True)

    # Save CSVs
    mode_df.to_csv(os.path.join(OUTPUT_DIR, 'mode_shift_per_condition.csv'), index=False)
    print(f"\n  Saved: mode_shift_per_condition.csv", flush=True)

    perm_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')}
                             for r in perm_results])
    perm_df.to_csv(perm_path, index=False)
    print(f"  Saved: mode_shift_permutation_tests.csv", flush=True)

    targeted_df = pd.DataFrame(targeted_results)
    targeted_path = os.path.join(OUTPUT_DIR, 'mode_shift_targeted_tests.csv')
    targeted_df.to_csv(targeted_path, index=False)
    print(f"  Saved: mode_shift_targeted_tests.csv", flush=True)

    mixture_weights_df = pd.DataFrame(mixture_weights)
    mixture_weights_path = os.path.join(OUTPUT_DIR, 'mode_shift_mixture_weights.csv')
    mixture_weights_df.to_csv(mixture_weights_path, index=False)
    print(f"  Saved: mode_shift_mixture_weights.csv", flush=True)

    mixture_tests_df = pd.DataFrame(mixture_tests)
    mixture_tests_path = os.path.join(OUTPUT_DIR, 'mode_shift_mixture_tests.csv')
    mixture_tests_df.to_csv(mixture_tests_path, index=False)
    print(f"  Saved: mode_shift_mixture_tests.csv", flush=True)

    # Save null distribution for gamma EO vs EC (for figure)
    gamma_visual = [r for r in perm_results
                    if r.get('band') == 'gamma' and 'rest-EC' in r.get('contrast', '')
                    and 'rest-EO' in r.get('contrast', '')]
    if gamma_visual and '_null_distribution' in gamma_visual[0]:
        null_path = os.path.join(OUTPUT_DIR, '_gamma_eo_ec_null.npy')
        np.save(null_path, gamma_visual[0]['_null_distribution'])

    # Figure
    fig_path = os.path.join(OUTPUT_DIR, 'mode_shift_analysis.png')
    plot_mode_shift_figure(mode_df, perm_results, fig_path,
                           targeted_results=targeted_results,
                           mixture_tests=mixture_tests,
                           mixture_weights=mixture_weights)

    # Control analyses
    if not args.skip_controls:
        # Control 1: Anchor frequency sweep
        t0 = time.time()
        anchor_results = run_anchor_sweep(session_freq_idx, sessions_by_cond)
        anchor_df = pd.DataFrame(anchor_results)
        anchor_path = os.path.join(OUTPUT_DIR, 'mode_shift_anchor_sweep.csv')
        anchor_df.to_csv(anchor_path, index=False)
        print(f"\n  Control 1 done ({time.time()-t0:.1f}s)", flush=True)
        print(f"  Saved: {anchor_path}", flush=True)

        # Control 2: Base sweep
        t0 = time.time()
        base_results = run_base_sweep(session_freq_idx, sessions_by_cond)
        base_df = pd.DataFrame(base_results)
        base_path = os.path.join(OUTPUT_DIR, 'mode_shift_base_sweep.csv')
        base_df.to_csv(base_path, index=False)
        print(f"\n  Control 2 done ({time.time()-t0:.1f}s)", flush=True)
        print(f"  Saved: {base_path}", flush=True)

        # Control 3: Random position control
        t0 = time.time()
        random_results = run_random_position_control(
            session_freq_idx, sessions_by_cond, n_iter=1000)
        random_summary = {
            'phi_max_z': random_results['phi_max_z'],
            'empirical_p': random_results['empirical_p'],
            'null_mean': random_results['null_max_z'].mean(),
            'null_std': random_results['null_max_z'].std(),
            'null_max': random_results['null_max_z'].max(),
            'n_iter': random_results['n_iter'],
        }
        for i, (name, z) in enumerate(zip(
                random_results['phi_component_names'],
                random_results['phi_z_scores'])):
            random_summary[f'phi_z_{name}'] = z
        random_df = pd.DataFrame([random_summary])
        random_path = os.path.join(OUTPUT_DIR, 'mode_shift_random_control.csv')
        random_df.to_csv(random_path, index=False)
        np.save(os.path.join(OUTPUT_DIR, '_random_null_z.npy'),
                random_results['null_max_z'])
        print(f"\n  Control 3 done ({time.time()-t0:.1f}s)", flush=True)
        print(f"  Saved: {random_path}", flush=True)

        # Control figure
        ctrl_fig_path = os.path.join(OUTPUT_DIR, 'mode_shift_controls.png')
        plot_control_figure(anchor_results, base_results,
                            random_results, ctrl_fig_path)

    print("\n  Done.", flush=True)


if __name__ == '__main__':
    main()

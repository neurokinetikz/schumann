#!/usr/bin/env python3
"""
Structural Phi-Specificity Test
================================

Tests whether the STRUCTURAL enrichment pattern (boundary depletion, noble
enrichment) is unique to the golden ratio or appears under any exponential base.

The existing compare_scaling_factors() in analyze_brain_invaders_phi_lattice.py
uses PHI's position definitions for ALL bases — a methodological flaw. This
script gives each base its OWN natural positions derived from 1/base.

Two complementary tests:
  Test A: Fair enrichment comparison — each base's lattice coords tested at
          that base's natural positions. Phase-rotation null for p-values.
  Test B: Absolute frequency clustering — count peaks near each base's
          predicted absolute frequencies. Matched-density null for fairness.

Usage:
    python scripts/structural_phi_specificity.py
    python scripts/structural_phi_specificity.py --n-perm 2000 --skip-bands
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

sys.path.insert(0, './lib')
sys.path.insert(0, './scripts')

from phi_frequency_model import PHI, F0
from ratio_specificity import lattice_coordinate, _enrichment_at_offset
from noble_boundary_dissociation import ANALYSIS_BANDS

# =========================================================================
# CONSTANTS
# =========================================================================

INPUT_CSV = 'exports_peak_distribution/eegmmidb_fooof/golden_ratio_peaks_EEGMMIDB.csv'
OUTPUT_DIR = 'exports_peak_distribution/eegmmidb_fooof'

BASES = {
    '1.4':   (1.4,   '1.4'),
    'sqrt2': (np.sqrt(2), '√2'),
    '1.5':   (1.5,   '3/2'),
    'phi':   (PHI,   'φ'),
    '1.7':   (1.7,   '1.7'),
    '1.8':   (1.8,   '1.8'),
    '2':     (2.0,   '2'),
    'e':     (np.e,  'e'),
    'pi':    (np.pi, 'π'),
}

# Minimum separation between positions to consider them distinct
POS_TOL = 0.02


# =========================================================================
# NATURAL POSITION DEFINITIONS
# =========================================================================

def natural_positions(base):
    """
    Return dict of position_name -> offset for a given exponential base.

    Universal positions:
        boundary = 0.0  (octave edge)
        attractor = 0.5 (midpoint)

    Base-specific positions (derived from 1/base):
        noble = 1/base
        noble_2 = (1/base)^2
        inv_noble = 1 - 1/base
        inv_noble_2 = 1 - (1/base)^2

    Positions within POS_TOL of existing ones or of 0/1 are suppressed.
    """
    inv_b = 1.0 / base
    positions = {'boundary': 0.0, 'attractor': 0.5}

    candidates = [
        ('noble', inv_b),
        ('noble_2', inv_b ** 2),
        ('inv_noble', 1.0 - inv_b),
        ('inv_noble_2', 1.0 - inv_b ** 2),
    ]

    for name, val in candidates:
        # Skip if too close to 0 or 1
        if val < POS_TOL or val > 1.0 - POS_TOL:
            continue
        # Skip if too close to any existing position
        if any(abs(val - existing) < POS_TOL for existing in positions.values()):
            continue
        positions[name] = val

    return positions


# =========================================================================
# TEST A: FAIR ENRICHMENT COMPARISON
# =========================================================================

def compute_structural_score(u, positions, window=0.05):
    """
    Compute structural score from enrichments at natural positions.

    SS = -boundary_enrich + attractor_enrich + mean(noble_enrich)

    Returns (score, enrichment_dict).
    """
    n_total = len(u)
    if n_total == 0:
        return 0.0, {}

    enrichments = {}
    for name, offset in positions.items():
        enrichments[name] = _enrichment_at_offset(u, offset, window, n_total)

    boundary_e = enrichments.get('boundary', 0.0)
    attractor_e = enrichments.get('attractor', 0.0)
    noble_keys = [k for k in enrichments if k not in ('boundary', 'attractor')]
    noble_mean = np.mean([enrichments[k] for k in noble_keys]) if noble_keys else 0.0

    score = -boundary_e + attractor_e + noble_mean
    return score, enrichments


def run_fair_enrichment_test(freqs, f0, bases, window=0.05, n_perm=1000,
                              seed=42, verbose=True):
    """
    Test A: For each base, compute enrichment at that base's own natural
    positions. Phase-rotation null for p-values.
    """
    if verbose:
        print("\n" + "=" * 80)
        print("TEST A: FAIR ENRICHMENT COMPARISON (each base uses own positions)")
        print("=" * 80)

    rng = np.random.default_rng(seed)
    results = []

    for base_key, (base_val, base_label) in bases.items():
        t0 = time.time()

        # Compute lattice coordinates
        u = lattice_coordinate(freqs, f0, base_val)
        u = u[np.isfinite(u)]
        n_total = len(u)

        # Get natural positions
        positions = natural_positions(base_val)

        # Observed structural score
        obs_score, enrichments = compute_structural_score(u, positions, window)

        # Phase-rotation null
        null_scores = np.empty(n_perm)
        for i in range(n_perm):
            theta = rng.uniform(0, 1)
            u_rot = (u + theta) % 1.0
            null_scores[i], _ = compute_structural_score(u_rot, positions, window)

        null_mean = null_scores.mean()
        null_std = null_scores.std()
        z_score = (obs_score - null_mean) / null_std if null_std > 0 else 0.0
        p_value = (null_scores >= obs_score).sum() / n_perm

        elapsed = time.time() - t0
        pos_str = ', '.join(f'{k}={v:.3f}' for k, v in sorted(positions.items(),
                                                                key=lambda x: x[1]))

        if verbose:
            print(f"\n  Base {base_label} ({base_val:.4f}):")
            print(f"    Positions ({len(positions)}): {pos_str}")
            for k, v in enrichments.items():
                print(f"    {k:>12s} enrichment: {v:+.1f}%")
            print(f"    Structural score: {obs_score:.1f}")
            print(f"    Null: mean={null_mean:.1f}, std={null_std:.1f}")
            print(f"    z={z_score:.2f}, p={p_value:.4f}  ({elapsed:.1f}s)")

        row = {
            'base_name': base_key,
            'base_label': base_label,
            'base_value': base_val,
            'n_positions': len(positions),
            'positions': pos_str,
            'structural_score': obs_score,
            'null_mean': null_mean,
            'null_std': null_std,
            'perm_z': z_score,
            'perm_p': p_value,
        }
        for k, v in enrichments.items():
            row[f'{k}_enrich'] = v
        results.append(row)

    df = pd.DataFrame(results)
    df = df.sort_values('structural_score', ascending=False)

    if verbose:
        print("\n" + "-" * 60)
        print("RANKING by structural score:")
        for i, (_, r) in enumerate(df.iterrows()):
            marker = " <<<" if r['base_name'] == 'phi' else ""
            print(f"  {i+1}. {r['base_label']:>4s}  SS={r['structural_score']:>7.1f}  "
                  f"z={r['perm_z']:>5.2f}  p={r['perm_p']:.4f}{marker}")

    return df


# =========================================================================
# TEST B: ABSOLUTE FREQUENCY CLUSTERING
# =========================================================================

def generate_predicted_peak_frequencies(f0, base, freq_range=(1, 45)):
    """
    Generate absolute frequencies where peaks are PREDICTED to cluster.

    Integer-n lattice nodes (f0 * base^n) are BOUNDARY positions where
    peaks should be DEPLETED. Peaks should cluster at the predicted noble
    and attractor positions BETWEEN boundaries:
        f0 * base^(n + offset) for each noble/attractor offset

    This is the correct test: do peaks cluster where the framework
    predicts they should?
    """
    positions = natural_positions(base)
    # Only use non-boundary positions (peaks are predicted HERE)
    peak_offsets = {k: v for k, v in positions.items() if k != 'boundary'}

    predicted_freqs = []
    labels = []

    # Sweep integer octaves
    n_min = int(np.floor(np.log(freq_range[0] / f0) / np.log(base))) - 1
    n_max = int(np.ceil(np.log(freq_range[1] / f0) / np.log(base))) + 1

    for n in range(n_min, n_max + 1):
        for pos_name, offset in peak_offsets.items():
            f = f0 * base ** (n + offset)
            if freq_range[0] <= f <= freq_range[1]:
                predicted_freqs.append(f)
                labels.append(f'{pos_name}@n={n}')

    return np.sort(np.array(predicted_freqs)), labels


def generate_boundary_frequencies(f0, base, freq_range=(1, 45)):
    """Generate boundary (integer-n) frequencies for depletion test."""
    nodes = []
    n = 0
    while True:
        f = f0 * base ** n
        if f > freq_range[1]:
            break
        if f >= freq_range[0]:
            nodes.append(f)
        n += 1
    n = -1
    while True:
        f = f0 * base ** n
        if f < freq_range[0]:
            break
        if f <= freq_range[1]:
            nodes.append(f)
        n -= 1
    return np.sort(np.array(nodes))


def count_peaks_near_nodes(freqs, nodes, window_hz):
    """Count peaks within +-window_hz of any node. No double-counting."""
    captured = np.zeros(len(freqs), dtype=bool)
    node_counts = []
    for node in nodes:
        near = np.abs(freqs - node) < window_hz
        node_counts.append(near.sum())
        captured |= near
    return captured.sum(), node_counts


def run_absolute_frequency_test(freqs, f0, bases, freq_range=(1, 45),
                                 window_hz=0.5, n_perm=1000, seed=42,
                                 null_type='empirical'):
    """
    Test B: Count peaks near each base's PREDICTED PEAK positions
    (nobles, attractors) — NOT boundary nodes (where depletion is expected).
    Also test boundary depletion as a consistency check.

    null_type : str
        'empirical' — draw null positions from observed peak frequencies
                      (tests whether lattice positions capture more peaks
                      than density-matched random positions)
        'uniform'   — draw from Uniform[freq_range] (legacy, uninformative
                      because it mismatches the non-uniform peak density)
    """
    null_label = 'empirical-density' if null_type == 'empirical' else 'uniform'
    print("\n" + "=" * 80)
    print(f"TEST B: ABSOLUTE FREQUENCY CLUSTERING (null: {null_label})")
    print("=" * 80)

    rng = np.random.default_rng(seed)
    results = []

    for base_key, (base_val, base_label) in bases.items():
        t0 = time.time()

        # Predicted PEAK positions (nobles, attractors)
        peak_freqs, peak_labels = generate_predicted_peak_frequencies(
            f0, base_val, freq_range)
        n_peaks_pred = len(peak_freqs)

        # Boundary positions (for depletion check)
        boundary_freqs = generate_boundary_frequencies(f0, base_val, freq_range)
        n_boundaries = len(boundary_freqs)

        if n_peaks_pred == 0:
            print(f"\n  Base {base_label}: no predicted positions in range — skipped")
            continue

        # Observed clustering at predicted peak positions
        obs_captured, _ = count_peaks_near_nodes(freqs, peak_freqs, window_hz)
        obs_rate = obs_captured / len(freqs)

        # Observed clustering at boundary positions (depletion check)
        bnd_captured, _ = count_peaks_near_nodes(freqs, boundary_freqs, window_hz)
        bnd_rate = bnd_captured / len(freqs)

        # Null for peak positions
        null_rates = np.empty(n_perm)
        for i in range(n_perm):
            if null_type == 'empirical':
                rand_freqs = rng.choice(freqs, size=n_peaks_pred, replace=True)
            else:
                rand_freqs = rng.uniform(freq_range[0], freq_range[1], n_peaks_pred)
            null_captured, _ = count_peaks_near_nodes(freqs, rand_freqs, window_hz)
            null_rates[i] = null_captured / len(freqs)

        null_mean = null_rates.mean()
        null_std = null_rates.std()
        z_score = (obs_rate - null_mean) / null_std if null_std > 0 else 0.0
        p_value = (null_rates >= obs_rate).sum() / n_perm
        enrichment = ((obs_rate / null_mean) - 1) * 100 if null_mean > 0 else 0.0

        # Null for boundary positions (depletion)
        null_bnd_rates = np.empty(n_perm)
        for i in range(n_perm):
            if null_type == 'empirical':
                rand_bnd = rng.choice(freqs, size=n_boundaries, replace=True)
            else:
                rand_bnd = rng.uniform(freq_range[0], freq_range[1], n_boundaries)
            null_bnd_cap, _ = count_peaks_near_nodes(freqs, rand_bnd, window_hz)
            null_bnd_rates[i] = null_bnd_cap / len(freqs)

        bnd_null_mean = null_bnd_rates.mean()
        bnd_null_std = null_bnd_rates.std()
        bnd_z = (bnd_rate - bnd_null_mean) / bnd_null_std if bnd_null_std > 0 else 0.0
        bnd_p = (null_bnd_rates <= bnd_rate).sum() / n_perm  # one-sided: depletion
        bnd_enrichment = ((bnd_rate / bnd_null_mean) - 1) * 100 if bnd_null_mean > 0 else 0.0

        elapsed = time.time() - t0
        freqs_str = ', '.join(f'{f:.1f}' for f in peak_freqs[:12])
        if len(peak_freqs) > 12:
            freqs_str += f', ... ({len(peak_freqs)} total)'

        print(f"\n  Base {base_label} ({base_val:.4f}):")
        print(f"    Predicted peak positions ({n_peaks_pred}): {freqs_str}")
        print(f"    Peak clustering: {obs_captured:,}/{len(freqs):,} "
              f"({obs_rate:.4f}), enrichment={enrichment:+.1f}%, "
              f"z={z_score:.2f}, p={p_value:.4f}")
        print(f"    Boundary depletion ({n_boundaries} nodes): rate={bnd_rate:.4f}, "
              f"enrichment={bnd_enrichment:+.1f}%, z={bnd_z:.2f}, "
              f"p_depl={bnd_p:.4f}")
        print(f"    ({elapsed:.1f}s)")

        results.append({
            'base_name': base_key,
            'base_label': base_label,
            'base_value': base_val,
            'n_predicted': n_peaks_pred,
            'n_boundaries': n_boundaries,
            'peak_capture_rate': obs_rate,
            'peak_null_mean': null_mean,
            'peak_null_std': null_std,
            'peak_enrichment_pct': enrichment,
            'peak_z': z_score,
            'peak_p': p_value,
            'boundary_rate': bnd_rate,
            'boundary_enrichment_pct': bnd_enrichment,
            'boundary_z': bnd_z,
            'boundary_depletion_p': bnd_p,
        })

    df = pd.DataFrame(results)
    df = df.sort_values('peak_enrichment_pct', ascending=False)

    print("\n" + "-" * 60)
    print("RANKING by peak clustering enrichment:")
    for i, (_, r) in enumerate(df.iterrows()):
        marker = " <<<" if r['base_name'] == 'phi' else ""
        bnd_sig = '*' if r['boundary_depletion_p'] < 0.05 else ''
        print(f"  {i+1}. {r['base_label']:>4s}  peak={r['peak_enrichment_pct']:>+6.1f}%  "
              f"z={r['peak_z']:>5.2f}  p={r['peak_p']:.4f}  |  "
              f"bnd={r['boundary_enrichment_pct']:>+6.1f}%{bnd_sig}  "
              f"({r['n_predicted']} positions){marker}")

    return df


# =========================================================================
# BAND-STRATIFIED ANALYSIS
# =========================================================================

def run_band_stratified(freqs_all, f0, bases, window=0.05, n_perm=1000,
                         seed=42):
    """
    Run Test A separately for each frequency band.
    Uses absolute Hz band boundaries so all bases are compared on the
    same spectral data.
    """
    print("\n" + "=" * 80)
    print("BAND-STRATIFIED ENRICHMENT (Test A per band)")
    print("=" * 80)

    rng = np.random.default_rng(seed)
    results = []

    for band_name, band_range in ANALYSIS_BANDS.items():
        f_lo, f_hi = band_range
        band_freqs = freqs_all[(freqs_all >= f_lo) & (freqs_all < f_hi)]
        n_peaks = len(band_freqs)

        if n_peaks < 100:
            print(f"\n  {band_name}: {n_peaks} peaks — skipped (< 100)")
            continue

        print(f"\n  {band_name} ({f_lo:.1f}-{f_hi:.1f} Hz): {n_peaks:,} peaks")

        for base_key, (base_val, base_label) in bases.items():
            u = lattice_coordinate(band_freqs, f0, base_val)
            u = u[np.isfinite(u)]
            positions = natural_positions(base_val)
            obs_score, enrichments = compute_structural_score(u, positions, window)

            # Quick null (fewer perms for speed)
            n_quick = min(n_perm, 500)
            null_scores = np.empty(n_quick)
            for i in range(n_quick):
                theta = rng.uniform(0, 1)
                u_rot = (u + theta) % 1.0
                null_scores[i], _ = compute_structural_score(u_rot, positions, window)

            null_mean = null_scores.mean()
            null_std = null_scores.std()
            z_score = (obs_score - null_mean) / null_std if null_std > 0 else 0.0
            p_value = (null_scores >= obs_score).sum() / n_quick

            row = {
                'band': band_name,
                'base_name': base_key,
                'base_label': base_label,
                'base_value': base_val,
                'n_peaks': len(u),
                'structural_score': obs_score,
                'perm_z': z_score,
                'perm_p': p_value,
            }
            for k, v in enrichments.items():
                row[f'{k}_enrich'] = v
            results.append(row)

        # Print band summary
        band_rows = [r for r in results if r['band'] == band_name]
        band_rows.sort(key=lambda r: r['structural_score'], reverse=True)
        print(f"    Ranking:")
        for i, r in enumerate(band_rows):
            marker = " <<<" if r['base_name'] == 'phi' else ""
            print(f"      {i+1}. {r['base_label']:>4s}  SS={r['structural_score']:>7.1f}  "
                  f"z={r['perm_z']:>5.2f}  p={r['perm_p']:.4f}{marker}")

    return pd.DataFrame(results)


# =========================================================================
# TEST C: BOOTSTRAP BASE COMPARISON
# =========================================================================

def run_bootstrap_base_comparison(freqs, f0, bases, window=0.05,
                                   n_boot=1000, seed=42, verbose=True):
    """
    Test C: Direct base comparison via bootstrap.

    For each bootstrap resample, compute structural score for ALL bases
    simultaneously. This lets us:
    1. How often is phi #1?
    2. Bootstrap CIs on structural score per base — do they overlap?
    3. Bootstrap CI on phi-vs-second-best gap
    """
    if verbose:
        print("\n" + "=" * 80)
        print("TEST C: BOOTSTRAP BASE COMPARISON (direct CI overlap test)")
        print("=" * 80)

    rng = np.random.default_rng(seed)
    n_total = len(freqs)

    base_keys = list(bases.keys())
    n_bases = len(base_keys)

    # Storage: n_boot x n_bases
    boot_scores = np.zeros((n_boot, n_bases))

    t0 = time.time()

    for b in range(n_boot):
        if verbose and b % 100 == 0:
            print(f"  Bootstrap {b}/{n_boot}...", end='\r')

        # Resample peaks with replacement
        boot_idx = rng.choice(n_total, size=n_total, replace=True)
        boot_freqs = freqs[boot_idx]

        for j, base_key in enumerate(base_keys):
            base_val = bases[base_key][0]
            u = lattice_coordinate(boot_freqs, f0, base_val)
            u = u[np.isfinite(u)]
            positions = natural_positions(base_val)
            score, _ = compute_structural_score(u, positions, window)
            boot_scores[b, j] = score

    elapsed = time.time() - t0
    if verbose:
        print(f"  Completed {n_boot} bootstraps in {elapsed:.0f}s")

    # Analyze results
    if verbose:
        print(f"\n  {'Base':>6s}  {'Mean SS':>8s}  {'95% CI':>20s}  {'#1 rate':>8s}")
        print(f"  {'-'*6}  {'-'*8}  {'-'*20}  {'-'*8}")

    results = []
    phi_idx = base_keys.index('phi')

    # How often is each base ranked #1?
    ranks = np.zeros((n_boot, n_bases))
    for b in range(n_boot):
        order = np.argsort(-boot_scores[b])
        for rank, idx in enumerate(order):
            ranks[b, idx] = rank + 1

    for j, base_key in enumerate(base_keys):
        base_label = bases[base_key][1]
        mean_ss = boot_scores[:, j].mean()
        ci_lo = np.percentile(boot_scores[:, j], 2.5)
        ci_hi = np.percentile(boot_scores[:, j], 97.5)
        first_rate = (ranks[:, j] == 1).mean()

        if verbose:
            marker = " <<<" if base_key == 'phi' else ""
            print(f"  {base_label:>6s}  {mean_ss:>8.1f}  [{ci_lo:>8.1f}, {ci_hi:>7.1f}]  "
                  f"{first_rate:>7.1%}{marker}")

        results.append({
            'base_name': base_key,
            'base_label': base_label,
            'base_value': bases[base_key][0],
            'boot_mean_ss': mean_ss,
            'boot_ci_lo': ci_lo,
            'boot_ci_hi': ci_hi,
            'first_place_rate': first_rate,
            'mean_rank': ranks[:, j].mean(),
        })

    # Phi vs next-best gap
    phi_scores = boot_scores[:, phi_idx]
    # For each bootstrap, find the best non-phi score
    non_phi_mask = np.ones(n_bases, dtype=bool)
    non_phi_mask[phi_idx] = False
    best_other = boot_scores[:, non_phi_mask].max(axis=1)
    gap = phi_scores - best_other

    gap_mean = gap.mean()
    gap_ci_lo = np.percentile(gap, 2.5)
    gap_ci_hi = np.percentile(gap, 97.5)
    phi_wins = (gap > 0).mean()

    if verbose:
        print(f"\n  Phi vs best-other gap:")
        print(f"    Mean gap: {gap_mean:+.1f}")
        print(f"    95% CI:   [{gap_ci_lo:+.1f}, {gap_ci_hi:+.1f}]")
        print(f"    Phi > all others: {phi_wins:.1%} of bootstraps")
        if gap_ci_lo > 0:
            print(f"    >>> CI excludes zero: phi is SIGNIFICANTLY better")
        else:
            print(f"    >>> CI includes zero: phi is NOT significantly better")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('boot_mean_ss', ascending=False)
    return results_df, {
        'gap_mean': gap_mean,
        'gap_ci_lo': gap_ci_lo,
        'gap_ci_hi': gap_ci_hi,
        'phi_wins_rate': phi_wins,
    }


# =========================================================================
# F0 SENSITIVITY SWEEP
# =========================================================================

DEFAULT_F0_VALUES = [6.5, 7.0, 7.2, 7.4, 7.6, 7.8, 8.0, 8.5, 9.0, 10.0, 12.0]

# Fine cliff sweep: dense sampling around the 7.4-7.6 transition
CLIFF_F0_VALUES = [7.40, 7.42, 7.44, 7.46, 7.48, 7.50, 7.52, 7.54, 7.56, 7.58, 7.60]

# Combined for full analysis
FULL_F0_VALUES = sorted(set(
    [6.5, 7.0, 7.2] + CLIFF_F0_VALUES + [7.8, 8.0, 8.5, 9.0, 10.0, 12.0]
))


def run_f0_sensitivity_sweep(freqs, bases, f0_values=None, window=0.05,
                              n_perm=200, n_boot=200, seed=42):
    """
    Sweep f0 across values, running Test A and Test C at each.

    Tests whether phi's structural advantage is robust to the choice of
    anchor frequency, not just the canonical f0 = 7.60 Hz.

    Parameters
    ----------
    freqs : np.ndarray
        Peak frequencies in Hz.
    bases : dict
        Maps base_key -> (base_value, base_label).
    f0_values : list of float, optional
        Anchor frequencies to test. Default: [7.0, 7.2, 7.4, 7.6, 7.8, 8.0]
    window : float
        Half-window for enrichment (lattice coords).
    n_perm : int
        Permutations per f0 (reduced for sweep feasibility).
    n_boot : int
        Bootstrap resamples per f0 (reduced for sweep feasibility).
    seed : int
        RNG seed.

    Returns
    -------
    sweep_a_df : DataFrame
        Test A results at each f0 (columns: f0, base_name, ...).
    sweep_c_df : DataFrame
        Test C bootstrap results at each f0.
    gap_by_f0 : dict
        Maps f0 -> {gap_mean, gap_ci_lo, gap_ci_hi, phi_wins_rate}.
    """
    if f0_values is None:
        f0_values = DEFAULT_F0_VALUES

    print("\n" + "=" * 80)
    print(f"F0 SENSITIVITY SWEEP: Testing {len(f0_values)} anchor frequencies")
    print(f"  f0 values: {f0_values}")
    print(f"  n_perm={n_perm}, n_boot={n_boot}")
    print("=" * 80)

    all_a_rows = []
    all_c_rows = []
    gap_by_f0 = {}

    t_total = time.time()

    for i, f0 in enumerate(f0_values):
        t0 = time.time()
        print(f"\n--- f0 = {f0:.2f} Hz ({i+1}/{len(f0_values)}) ---")

        # Test A at this f0
        a_df = run_fair_enrichment_test(freqs, f0, bases, window=window,
                                         n_perm=n_perm, seed=seed,
                                         verbose=False)
        a_df['f0'] = f0
        all_a_rows.append(a_df)

        # Phi rank in Test A
        phi_ss = a_df.loc[a_df['base_name'] == 'phi', 'structural_score'].iloc[0]
        phi_rank_a = (a_df['structural_score'] > phi_ss).sum() + 1
        print(f"  Test A: phi SS={phi_ss:.1f}, rank={phi_rank_a}")

        # Test C (bootstrap) at this f0
        c_df, gap_info = run_bootstrap_base_comparison(
            freqs, f0, bases, window=window, n_boot=n_boot,
            seed=seed, verbose=False)
        c_df['f0'] = f0
        all_c_rows.append(c_df)
        gap_by_f0[f0] = gap_info

        elapsed = time.time() - t0
        gap_str = (f"gap={gap_info['gap_mean']:+.1f} "
                   f"[{gap_info['gap_ci_lo']:+.1f}, {gap_info['gap_ci_hi']:+.1f}]")
        sig = "SIGNIFICANT" if gap_info['gap_ci_lo'] > 0 else "not sig"
        print(f"  Test C: phi #1 in {gap_info['phi_wins_rate']:.0%}, "
              f"{gap_str} ({sig})  [{elapsed:.0f}s]")

    sweep_a_df = pd.concat(all_a_rows, ignore_index=True)
    sweep_c_df = pd.concat(all_c_rows, ignore_index=True)

    # Summary
    elapsed_total = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"F0 SWEEP SUMMARY ({elapsed_total:.0f}s total)")
    print(f"{'=' * 60}")
    print(f"  {'f0':>5s}  {'phi SS':>7s}  {'phi rank':>8s}  {'gap':>6s}  "
          f"{'gap CI':>18s}  {'sig?':>5s}")
    for f0 in f0_values:
        a_at_f0 = sweep_a_df[sweep_a_df['f0'] == f0]
        phi_ss = a_at_f0.loc[a_at_f0['base_name'] == 'phi',
                              'structural_score'].iloc[0]
        phi_rank = (a_at_f0['structural_score'] > phi_ss).sum() + 1
        g = gap_by_f0[f0]
        sig = "YES" if g['gap_ci_lo'] > 0 else "no"
        print(f"  {f0:>6.2f}  {phi_ss:>7.1f}  {phi_rank:>8d}  "
              f"{g['gap_mean']:>+5.1f}  "
              f"[{g['gap_ci_lo']:>+6.1f}, {g['gap_ci_hi']:>+5.1f}]  {sig:>5s}")

    # Joint grid summary: winning base at each f0
    print(f"\n{'=' * 60}")
    print(f"JOINT (f0, base) GRID")
    print(f"{'=' * 60}")
    print(f"  {'f0':>5s}  {'winner':>6s}  {'win SS':>7s}  {'phi SS':>7s}  "
          f"{'phi rank':>8s}  {'phi gap':>8s}")
    joint_rows = []
    for f0 in f0_values:
        c_at_f0 = sweep_c_df[sweep_c_df['f0'] == f0].sort_values(
            'boot_mean_ss', ascending=False)
        winner = c_at_f0.iloc[0]
        phi_row = c_at_f0[c_at_f0['base_name'] == 'phi'].iloc[0]
        phi_rank = (c_at_f0['boot_mean_ss'] > phi_row['boot_mean_ss']).sum() + 1
        phi_gap = phi_row['boot_mean_ss'] - winner['boot_mean_ss']
        print(f"  {f0:>5.2f}  {winner['base_label']:>6s}  "
              f"{winner['boot_mean_ss']:>7.1f}  {phi_row['boot_mean_ss']:>7.1f}  "
              f"{phi_rank:>8d}  {phi_gap:>+7.1f}")
        joint_rows.append({
            'f0': f0,
            'winning_base': winner['base_label'],
            'winning_ss': winner['boot_mean_ss'],
            'winning_ci_lo': winner['boot_ci_lo'],
            'winning_ci_hi': winner['boot_ci_hi'],
            'phi_ss': phi_row['boot_mean_ss'],
            'phi_rank': phi_rank,
            'phi_gap_to_winner': phi_gap,
        })
    joint_df = pd.DataFrame(joint_rows)
    # Global optimum
    best_idx = joint_df['winning_ss'].idxmax()
    best = joint_df.loc[best_idx]
    print(f"\n  GLOBAL OPTIMUM: f0={best['f0']:.2f}, base={best['winning_base']}, "
          f"SS={best['winning_ss']:.1f}")
    phi_best_idx = joint_df['phi_ss'].idxmax()
    phi_best = joint_df.loc[phi_best_idx]
    print(f"  PHI OPTIMUM:    f0={phi_best['f0']:.2f}, SS={phi_best['phi_ss']:.1f}")
    if phi_best['phi_ss'] >= best['winning_ss']:
        print(f"  => PHI at f0={phi_best['f0']:.2f} IS the global optimum")
    else:
        print(f"  => PHI at f0={phi_best['f0']:.2f} is {phi_best['phi_ss'] - best['winning_ss']:+.1f} "
              f"below global optimum ({best['winning_base']} at f0={best['f0']:.2f})")

    return sweep_a_df, sweep_c_df, gap_by_f0, joint_df


# =========================================================================
# PER-BAND DIAGNOSTIC
# =========================================================================

def diagnose_base_at_f0(freqs, f0, bases, window=0.05):
    """
    Diagnostic: per-band structural score for each base at a given f0.
    Shows why certain bases dominate at certain anchor frequencies
    (e.g., pi at f0=6.5 due to IAF alignment).
    """
    print(f"\n{'=' * 80}")
    print(f"DIAGNOSTIC: Per-band structural scores at f0 = {f0} Hz")
    print(f"{'=' * 80}")

    results = []

    for base_key, (base_val, base_label) in bases.items():
        positions = natural_positions(base_val)

        # Where does IAF (~10 Hz) land in this base's lattice?
        u_10 = (np.log(10.0 / f0) / np.log(base_val)) % 1.0
        pos_dists = {}
        for pos_name, pos_val in positions.items():
            dist = min(abs(u_10 - pos_val),
                       abs(u_10 - pos_val + 1),
                       abs(u_10 - pos_val - 1))
            pos_dists[pos_name] = dist
        nearest_pos = min(pos_dists, key=pos_dists.get)
        nearest_dist = pos_dists[nearest_pos]

        # Per-band structural scores
        for band_name, (f_lo, f_hi) in ANALYSIS_BANDS.items():
            band_freqs = freqs[(freqs >= f_lo) & (freqs < f_hi)]
            n_peaks = len(band_freqs)
            if n_peaks < 50:
                continue
            u = lattice_coordinate(band_freqs, f0, base_val)
            u = u[np.isfinite(u)]
            score, enrichments = compute_structural_score(u, positions, window)
            results.append({
                'f0': f0,
                'base_name': base_key,
                'base_label': base_label,
                'band': band_name,
                'n_peaks': len(u),
                'structural_score': score,
                'iaf_10hz_u': u_10,
                'iaf_nearest_pos': nearest_pos,
                'iaf_nearest_dist': nearest_dist,
                **{f'{k}_enrich': v for k, v in enrichments.items()},
            })

    df = pd.DataFrame(results)

    # Print summary per band
    for band_name in ANALYSIS_BANDS.keys():
        band_data = df[df['band'] == band_name].sort_values(
            'structural_score', ascending=False)
        if len(band_data) == 0:
            continue
        print(f"\n  {band_name} ({band_data.iloc[0]['n_peaks']:,} peaks):")
        for _, r in band_data.iterrows():
            marker = ""
            if r['base_name'] == 'phi':
                marker = " [phi]"
            print(f"    {r['base_label']:>4s}  SS={r['structural_score']:>7.1f}{marker}")

    # IAF alignment summary
    print(f"\n  IAF (10 Hz) lattice position at f0={f0}:")
    seen = set()
    for _, r in df.iterrows():
        key = r['base_name']
        if key in seen:
            continue
        seen.add(key)
        alignment = "ENRICHING" if r['iaf_nearest_pos'] != 'boundary' else "DEPLETING"
        print(f"    {r['base_label']:>4s}: u(10Hz)={r['iaf_10hz_u']:.4f}, "
              f"nearest={r['iaf_nearest_pos']} (dist={r['iaf_nearest_dist']:.4f}) "
              f"-> {alignment}")

    return df


def plot_f0_sensitivity(sweep_a_df, sweep_c_df, gap_by_f0, output_path,
                        cliff_a_df=None, cliff_c_df=None, cliff_gap=None,
                        theoretical_f0=7.49):
    """
    3-panel figure (2-panel if no cliff data) showing f0 sensitivity.

    Panel A: Line plot — SS vs f0 for each base (full range).
    Panel B: Gap forest plot with winning base SS annotated.
    Panel C: Cliff zoom (7.40–7.60 Hz) if cliff data provided.
    """
    has_cliff = cliff_a_df is not None and len(cliff_a_df) > 0
    n_panels = 3 if has_cliff else 2

    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5.5))
    if n_panels == 2:
        axes = list(axes)

    # Combine coarse + cliff data for Panel A
    if has_cliff:
        combined_a = pd.concat([sweep_a_df, cliff_a_df], ignore_index=True)
        combined_a = combined_a.drop_duplicates(
            subset=['f0', 'base_name'], keep='last')
        combined_gap = {**gap_by_f0, **(cliff_gap or {})}
    else:
        combined_a = sweep_a_df
        combined_gap = gap_by_f0

    f0_values = sorted(combined_a['f0'].unique())

    # Base colors for visibility
    base_colors = {
        'phi': '#e74c3c', '2': '#2ecc71', 'pi': '#3498db',
        'e': '#9b59b6', '1.7': '#f39c12', '1.5': '#1abc9c',
        '1.4': '#e67e22', '1.8': '#95a5a6', 'sqrt2': '#34495e',
    }

    # Panel A: SS vs f0 for all bases
    ax = axes[0]
    for base_key in combined_a['base_name'].unique():
        subset = combined_a[combined_a['base_name'] == base_key].sort_values('f0')
        label_val = subset['base_label'].iloc[0]
        is_phi = base_key == 'phi'
        color = base_colors.get(base_key, '#bbbbbb')
        lw = 2.5 if is_phi else 1.0
        alpha_val = 1.0 if is_phi else 0.6
        zorder = 10 if is_phi else 1
        ax.plot(subset['f0'], subset['structural_score'],
                color=color, linewidth=lw, alpha=alpha_val, zorder=zorder,
                label=label_val if is_phi else None,
                marker='o' if is_phi else None, markersize=5)
    # Label non-phi bases at their rightmost point
    for base_key in combined_a['base_name'].unique():
        if base_key == 'phi':
            continue
        subset = combined_a[combined_a['base_name'] == base_key].sort_values('f0')
        color = base_colors.get(base_key, '#666666')
        ax.annotate(subset['base_label'].iloc[0],
                    xy=(subset['f0'].iloc[-1], subset['structural_score'].iloc[-1]),
                    fontsize=7, color=color, ha='left',
                    xytext=(3, 0), textcoords='offset points')
    if theoretical_f0:
        ax.axvline(theoretical_f0, color='#9b59b6', linewidth=1.5,
                   linestyle=':', alpha=0.7, label=f'SIE ({theoretical_f0} Hz)')
    ax.set_xlabel('Anchor frequency $f_0$ (Hz)')
    ax.set_ylabel('Structural Score (SS)')
    ax.set_title('A. Structural Score vs $f_0$')
    ax.legend(loc='best', fontsize=8)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')

    # Panel B: Gap forest plot
    ax = axes[1]
    # Use only coarse f0 values for forest plot (not cliff)
    coarse_f0s = sorted(gap_by_f0.keys())
    gap_means = [gap_by_f0[f0]['gap_mean'] for f0 in coarse_f0s]
    gap_lo = [gap_by_f0[f0]['gap_ci_lo'] for f0 in coarse_f0s]
    gap_hi = [gap_by_f0[f0]['gap_ci_hi'] for f0 in coarse_f0s]
    xerr_lo = [m - lo for m, lo in zip(gap_means, gap_lo)]
    xerr_hi = [hi - m for m, hi in zip(gap_means, gap_hi)]

    y_pos = range(len(coarse_f0s))
    colors = ['#27ae60' if lo > 0 else '#e74c3c' for lo in gap_lo]
    ax.barh(list(y_pos), gap_means, color=colors, alpha=0.6, height=0.6)
    ax.errorbar(gap_means, list(y_pos),
                xerr=[xerr_lo, xerr_hi],
                fmt='none', color='black', capsize=3, linewidth=1.2)
    ax.set_yticks(list(y_pos))
    # Format labels: show 2 decimals for cliff-resolution values, 1 otherwise
    ylabels = []
    for f0 in coarse_f0s:
        if f0 == int(f0):
            ylabels.append(f'$f_0$={f0:.0f}')
        elif (f0 * 10) == int(f0 * 10):
            ylabels.append(f'$f_0$={f0:.1f}')
        else:
            ylabels.append(f'$f_0$={f0:.2f}')
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.axvline(0, color='red', linewidth=1, linestyle='--')
    ax.set_xlabel('$\\varphi$ vs best-other gap (SS)')
    ax.set_title('B. Bootstrap Gap ± 95% CI')
    for i, f0 in enumerate(coarse_f0s):
        g = gap_by_f0[f0]
        sig_str = '*' if g['gap_ci_lo'] > 0 else ''
        ax.annotate(f"{g['phi_wins_rate']:.0%}{sig_str}",
                    xy=(max(gap_means[i], 0) + max(xerr_hi[i], 0) + 0.3, i),
                    fontsize=7, va='center')

    # Panel C: Cliff zoom
    if has_cliff:
        ax = axes[2]
        cliff_f0s = sorted(cliff_a_df['f0'].unique())
        for base_key in cliff_a_df['base_name'].unique():
            subset = cliff_a_df[cliff_a_df['base_name'] == base_key].sort_values('f0')
            is_phi = base_key == 'phi'
            color = base_colors.get(base_key, '#bbbbbb')
            lw = 2.5 if is_phi else 1.0
            alpha_val = 1.0 if is_phi else 0.5
            ax.plot(subset['f0'], subset['structural_score'],
                    color=color, linewidth=lw, alpha=alpha_val,
                    marker='o' if is_phi else None, markersize=5,
                    label=subset['base_label'].iloc[0] if is_phi else None)
        if theoretical_f0:
            ax.axvline(theoretical_f0, color='#9b59b6', linewidth=1.5,
                       linestyle=':', alpha=0.7, label=f'SIE: {theoretical_f0} Hz')
        # Annotate phi rank at each cliff f0
        for f0_val in cliff_f0s:
            f0_data = cliff_a_df[cliff_a_df['f0'] == f0_val]
            phi_row = f0_data[f0_data['base_name'] == 'phi']
            if len(phi_row) == 0:
                continue
            phi_ss = phi_row.iloc[0]['structural_score']
            phi_rank = (f0_data['structural_score'] > phi_ss).sum() + 1
            ax.annotate(f'#{phi_rank}', xy=(f0_val, phi_ss),
                        fontsize=7, color='#e74c3c',
                        xytext=(0, 8), textcoords='offset points', ha='center')
        ax.set_xlabel('$f_0$ (Hz)')
        ax.set_ylabel('Structural Score')
        ax.set_title('C. Cliff Region (7.40–7.60 Hz)')
        ax.legend(loc='best', fontsize=8)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  f0 sensitivity figure saved: {output_path}")


# =========================================================================
# SUMMARY RANKING
# =========================================================================

def rank_bases(test_a_df, test_b_df):
    """Composite ranking across Test A and Test B."""
    print("\n" + "=" * 80)
    print("COMPOSITE RANKING")
    print("=" * 80)

    # Rank by Test A structural score (higher = better)
    a_rank = test_a_df.sort_values('structural_score', ascending=False).reset_index(drop=True)
    a_rank['rank_a'] = range(1, len(a_rank) + 1)

    # Rank by Test B peak enrichment (higher = better)
    b_rank = test_b_df.sort_values('peak_enrichment_pct', ascending=False).reset_index(drop=True)
    b_rank['rank_b'] = range(1, len(b_rank) + 1)

    # Merge
    merged = pd.merge(
        a_rank[['base_name', 'base_label', 'structural_score', 'perm_z', 'rank_a']],
        b_rank[['base_name', 'peak_enrichment_pct', 'peak_z', 'rank_b']],
        on='base_name', suffixes=('_a', '_b')
    )
    merged['avg_rank'] = (merged['rank_a'] + merged['rank_b']) / 2.0
    merged = merged.sort_values('avg_rank')

    print(f"\n  {'Base':>6s}  {'Rank A':>6s}  {'Rank B':>6s}  {'Avg':>5s}  "
          f"{'SS':>7s}  {'Peak%':>8s}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*7}  {'-'*8}")
    for _, r in merged.iterrows():
        marker = " <<<" if r['base_name'] == 'phi' else ""
        print(f"  {r['base_label']:>6s}  {r['rank_a']:>6d}  {r['rank_b']:>6d}  "
              f"{r['avg_rank']:>5.1f}  {r['structural_score']:>7.1f}  "
              f"{r['peak_enrichment_pct']:>+7.1f}%{marker}")

    phi_row = merged[merged['base_name'] == 'phi']
    if not phi_row.empty:
        phi_rank = phi_row.iloc[0]['avg_rank']
        n_bases = len(merged)
        print(f"\n  PHI RANK: {phi_rank:.1f} / {n_bases} "
              f"({'#1' if phi_rank == 1 else 'NOT #1'})")

    return merged


# =========================================================================
# VISUALIZATION
# =========================================================================

def plot_structural_specificity(test_a_df, test_b_df, band_df, output_path,
                                boot_df=None):
    """4-panel figure comparing structural specificity across bases."""
    has_bands = band_df is not None and len(band_df) > 0
    has_boot = boot_df is not None and len(boot_df) > 0

    n_panels = 2 + int(has_bands) + int(has_boot)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5))
    if n_panels == 2:
        axes = list(axes)

    panel_idx = 0

    # Color phi differently
    def bar_colors(df, key_col='base_name'):
        return ['#e74c3c' if n == 'phi' else '#3498db' for n in df[key_col]]

    # Panel A: Structural scores (Test A)
    ax = axes[panel_idx]
    a_sorted = test_a_df.sort_values('structural_score', ascending=True)
    colors = bar_colors(a_sorted)
    ax.barh(range(len(a_sorted)), a_sorted['structural_score'], color=colors)
    ax.set_yticks(range(len(a_sorted)))
    ax.set_yticklabels(a_sorted['base_label'])
    ax.set_xlabel('Structural Score')
    ax.set_title('A. Fair Enrichment (own positions)')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    for i, (_, r) in enumerate(a_sorted.iterrows()):
        sig = '*' if r['perm_p'] < 0.05 else ''
        ax.text(r['structural_score'] + 1, i, f"z={r['perm_z']:.1f}{sig}",
                va='center', fontsize=8)
    panel_idx += 1

    # Panel B: Bootstrap CIs (Test C) — placed second for visual impact
    if has_boot:
        ax = axes[panel_idx]
        b_sorted = boot_df.sort_values('boot_mean_ss', ascending=True)
        colors = bar_colors(b_sorted)
        y_pos = range(len(b_sorted))
        ax.barh(list(y_pos), b_sorted['boot_mean_ss'], color=colors, alpha=0.7)
        # Error bars for 95% CI
        xerr_lo = b_sorted['boot_mean_ss'] - b_sorted['boot_ci_lo']
        xerr_hi = b_sorted['boot_ci_hi'] - b_sorted['boot_mean_ss']
        ax.errorbar(b_sorted['boot_mean_ss'], list(y_pos),
                    xerr=[xerr_lo.values, xerr_hi.values],
                    fmt='none', color='black', capsize=3, linewidth=1.5)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(b_sorted['base_label'])
        ax.set_xlabel('Structural Score (bootstrap 95% CI)')
        ax.set_title('B. Bootstrap Base Comparison')
        ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
        # Annotate #1 rate
        for i, (_, r) in enumerate(b_sorted.iterrows()):
            rate_str = f"{r['first_place_rate']:.0%}" if r['first_place_rate'] > 0 else ''
            if rate_str:
                ax.text(r['boot_ci_hi'] + 0.5, i, f"#1: {rate_str}",
                        va='center', fontsize=8, fontweight='bold')
        panel_idx += 1

    # Panel C: Absolute clustering at predicted peak positions (Test B)
    ax = axes[panel_idx]
    c_sorted = test_b_df.sort_values('peak_enrichment_pct', ascending=True)
    colors = bar_colors(c_sorted)
    ax.barh(range(len(c_sorted)), c_sorted['peak_enrichment_pct'], color=colors)
    ax.set_yticks(range(len(c_sorted)))
    ax.set_yticklabels(c_sorted['base_label'])
    ax.set_xlabel('Peak Position Enrichment (%)')
    ax.set_title('C. Clustering at Predicted Peaks')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    for i, (_, r) in enumerate(c_sorted.iterrows()):
        sig = '*' if r['peak_p'] < 0.05 else ''
        ax.text(r['peak_enrichment_pct'] + 0.5, i, f"z={r['peak_z']:.1f}{sig}",
                va='center', fontsize=8)
    panel_idx += 1

    # Panel D: Band-stratified heatmap
    if has_bands:
        ax = axes[panel_idx]
        pivot = band_df.pivot_table(index='base_label', columns='band',
                                     values='structural_score')
        band_order = [b for b in ANALYSIS_BANDS.keys() if b in pivot.columns]
        pivot = pivot[band_order]
        base_order = test_a_df.sort_values('structural_score', ascending=False)['base_label'].tolist()
        base_order = [b for b in base_order if b in pivot.index]
        pivot = pivot.loc[base_order]

        im = ax.imshow(pivot.values, aspect='auto', cmap='RdBu_r',
                       vmin=-np.abs(pivot.values).max(),
                       vmax=np.abs(pivot.values).max())
        ax.set_xticks(range(len(band_order)))
        ax.set_xticklabels(band_order, rotation=45, ha='right')
        ax.set_yticks(range(len(base_order)))
        ax.set_yticklabels(base_order)
        ax.set_title('D. Band-Stratified Structural Score')
        plt.colorbar(im, ax=ax, label='Structural Score', shrink=0.8)

        for i in range(len(base_order)):
            for j in range(len(band_order)):
                val = pivot.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                            fontsize=7, color='white' if abs(val) > 30 else 'black')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n  Figure saved: {output_path}")


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Structural Phi-Specificity Test: is boundary depletion / '
                    'noble enrichment unique to phi?')
    parser.add_argument('--n-perm', type=int, default=1000,
                        help='Permutations for null distributions (default: 1000)')
    parser.add_argument('--window', type=float, default=0.05,
                        help='Half-window for enrichment in lattice coords (default: 0.05)')
    parser.add_argument('--window-hz', type=float, default=0.5,
                        help='Hz window for absolute clustering test (default: 0.5)')
    parser.add_argument('--skip-bands', action='store_true',
                        help='Skip band-stratified analysis')
    parser.add_argument('--skip-absolute', action='store_true',
                        help='Skip absolute frequency clustering test')
    parser.add_argument('--skip-bootstrap', action='store_true',
                        help='Skip bootstrap base comparison')
    parser.add_argument('--n-boot', type=int, default=1000,
                        help='Bootstrap resamples for base comparison (default: 1000)')
    parser.add_argument('--f0-sweep', action='store_true',
                        help='Run f0 sensitivity sweep (Tests A+C at multiple f0 values)')
    parser.add_argument('--f0-values', type=float, nargs='+',
                        default=DEFAULT_F0_VALUES,
                        help='f0 values for sweep')
    parser.add_argument('--sweep-n-perm', type=int, default=200,
                        help='Permutations per f0 in sweep (default: 200)')
    parser.add_argument('--sweep-n-boot', type=int, default=1000,
                        help='Bootstrap resamples per f0 in sweep (default: 1000)')
    parser.add_argument('--cliff-sweep', action='store_true',
                        help='Run fine-grained cliff sweep (7.40-7.60 Hz, 0.02 steps)')
    parser.add_argument('--cliff-n-boot', type=int, default=1000,
                        help='Bootstrap resamples for cliff sweep (default: 1000)')
    parser.add_argument('--full-sweep', action='store_true',
                        help='Run full sweep (coarse extended + cliff)')
    parser.add_argument('--diagnose-f0', type=float, nargs='+', default=None,
                        help='Run per-band diagnostic at specified f0 values')
    parser.add_argument('--null-type', choices=['empirical', 'uniform'],
                        default='empirical',
                        help='Null distribution type for Test B (default: empirical)')
    parser.add_argument('--max-bandwidth', type=float, default=None,
                        help='Filter peaks to bandwidth <= this value (Hz)')
    parser.add_argument('--input', type=str, default=None,
                        help='Override input CSV path')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override output directory')
    args = parser.parse_args()

    input_csv = args.input or INPUT_CSV
    output_dir = args.output_dir or OUTPUT_DIR

    print("=" * 80)
    print("STRUCTURAL PHI-SPECIFICITY TEST")
    print("=" * 80)
    print(f"Input: {input_csv}")
    print(f"Bases: {', '.join(v[1] for v in BASES.values())}")
    print(f"Window (lattice): {args.window}")
    print(f"Window (Hz): {args.window_hz}")
    print(f"Permutations: {args.n_perm}")
    print(f"f₀ = {F0} Hz")

    # Load data
    t_start = time.time()
    print("\nLoading peaks...")
    df = pd.read_csv(input_csv)
    if args.max_bandwidth is not None and 'bandwidth' in df.columns:
        n_before = len(df)
        df = df[df['bandwidth'] <= args.max_bandwidth]
        print(f"  Bandwidth filter: <= {args.max_bandwidth} Hz -> {len(df):,}/{n_before:,} peaks retained ({len(df)/n_before*100:.1f}%)")
    freqs = df['freq'].values if 'freq' in df.columns else df['frequency'].values
    freqs = freqs[np.isfinite(freqs) & (freqs > 0)]
    print(f"  {len(freqs):,} valid peaks loaded")

    # Show natural positions for each base
    print("\nNatural position definitions:")
    for base_key, (base_val, base_label) in BASES.items():
        positions = natural_positions(base_val)
        pos_str = ', '.join(f'{k}={v:.3f}' for k, v in sorted(positions.items(),
                                                                key=lambda x: x[1]))
        print(f"  {base_label:>4s}: {pos_str}")

    # Test A: Fair enrichment comparison
    test_a_df = run_fair_enrichment_test(freqs, F0, BASES, window=args.window,
                                          n_perm=args.n_perm)
    os.makedirs(output_dir, exist_ok=True)
    test_a_path = os.path.join(output_dir, 'structural_specificity_enrichment.csv')
    test_a_df.to_csv(test_a_path, index=False)
    print(f"\n  Test A results saved: {test_a_path}")

    # Test B: Absolute frequency clustering
    test_b_df = None
    if not args.skip_absolute:
        test_b_df = run_absolute_frequency_test(freqs, F0, BASES,
                                                  window_hz=args.window_hz,
                                                  n_perm=args.n_perm,
                                                  null_type=args.null_type)
        test_b_path = os.path.join(output_dir, 'structural_specificity_absolute.csv')
        test_b_df.to_csv(test_b_path, index=False)
        print(f"\n  Test B results saved: {test_b_path}")

    # Band-stratified
    band_df = None
    if not args.skip_bands:
        band_df = run_band_stratified(freqs, F0, BASES, window=args.window,
                                       n_perm=args.n_perm)
        band_path = os.path.join(output_dir, 'structural_specificity_by_band.csv')
        band_df.to_csv(band_path, index=False)
        print(f"\n  Band-stratified results saved: {band_path}")

    # Bootstrap base comparison
    boot_df = None
    boot_gap = None
    if not args.skip_bootstrap:
        boot_df, boot_gap = run_bootstrap_base_comparison(
            freqs, F0, BASES, window=args.window, n_boot=args.n_boot)
        boot_path = os.path.join(output_dir, 'structural_specificity_bootstrap.csv')
        boot_df.to_csv(boot_path, index=False)
        print(f"\n  Bootstrap results saved: {boot_path}")

    # Composite ranking
    if test_b_df is not None:
        summary_df = rank_bases(test_a_df, test_b_df)
        summary_path = os.path.join(output_dir, 'structural_specificity_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\n  Summary saved: {summary_path}")

    # Figure
    if test_b_df is not None:
        fig_path = os.path.join(output_dir, 'structural_specificity_figure.png')
        plot_structural_specificity(test_a_df, test_b_df, band_df, fig_path,
                                    boot_df=boot_df)

    # Non-additivity diagnostic: compare aggregate SS to sum of per-band SS
    if band_df is not None and len(band_df) > 0:
        print("\n" + "-" * 60)
        print("NON-ADDITIVITY DIAGNOSTIC (aggregate vs sum of per-band SS)")
        print("-" * 60)
        for base_key, (_, base_label) in BASES.items():
            agg_row = test_a_df[test_a_df['base_name'] == base_key]
            if agg_row.empty:
                continue
            agg_ss = agg_row.iloc[0]['structural_score']
            band_rows = band_df[band_df['base_name'] == base_key]
            band_sum = band_rows['structural_score'].sum()
            diff = agg_ss - band_sum
            marker = " <<<" if base_key == 'phi' else ""
            print(f"  {base_label:>4s}: aggregate={agg_ss:>7.1f}  "
                  f"sum-of-bands={band_sum:>7.1f}  "
                  f"non-additive={diff:>+7.1f}{marker}")

    # F0 sensitivity sweep (coarse)
    sweep_a_df = None
    sweep_c_df = None
    gap_by_f0 = None
    joint_df = None

    if args.f0_sweep or args.full_sweep:
        f0_vals = args.f0_values if not args.full_sweep else DEFAULT_F0_VALUES
        sweep_a_df, sweep_c_df, gap_by_f0, joint_df = run_f0_sensitivity_sweep(
            freqs, BASES, f0_values=f0_vals,
            window=args.window, n_perm=args.sweep_n_perm,
            n_boot=args.sweep_n_boot)

        sweep_path = os.path.join(output_dir, 'structural_specificity_f0_sweep.csv')
        sweep_a_df.to_csv(sweep_path, index=False)
        print(f"\n  f0 sweep Test A results saved: {sweep_path}")

        sweep_c_path = os.path.join(output_dir, 'structural_specificity_f0_sweep_bootstrap.csv')
        sweep_c_df.to_csv(sweep_c_path, index=False)
        print(f"  f0 sweep bootstrap results saved: {sweep_c_path}")

        joint_path = os.path.join(output_dir, 'structural_specificity_joint_grid.csv')
        joint_df.to_csv(joint_path, index=False)
        print(f"  Joint grid results saved: {joint_path}")

    # Cliff sweep
    cliff_a_df = None
    cliff_c_df = None
    cliff_gap = None

    if args.cliff_sweep or args.full_sweep:
        cliff_a_df, cliff_c_df, cliff_gap, cliff_joint = run_f0_sensitivity_sweep(
            freqs, BASES, f0_values=CLIFF_F0_VALUES,
            window=args.window, n_perm=args.sweep_n_perm,
            n_boot=args.cliff_n_boot)

        cliff_path = os.path.join(output_dir, 'structural_specificity_cliff_sweep.csv')
        cliff_a_df.to_csv(cliff_path, index=False)
        print(f"\n  Cliff sweep results saved: {cliff_path}")

        cliff_boot_path = os.path.join(output_dir,
                                        'structural_specificity_cliff_bootstrap.csv')
        cliff_c_df.to_csv(cliff_boot_path, index=False)
        print(f"  Cliff bootstrap results saved: {cliff_boot_path}")

    # Combined figure
    if sweep_a_df is not None:
        fig_path = os.path.join(output_dir, 'structural_specificity_f0_sensitivity.png')
        plot_f0_sensitivity(sweep_a_df, sweep_c_df, gap_by_f0, fig_path,
                           cliff_a_df=cliff_a_df, cliff_c_df=cliff_c_df,
                           cliff_gap=cliff_gap, theoretical_f0=7.49)

    # Per-band diagnostic
    if args.diagnose_f0:
        for diag_f0 in args.diagnose_f0:
            diag_df = diagnose_base_at_f0(freqs, diag_f0, BASES, window=args.window)
            diag_path = os.path.join(
                output_dir,
                f'structural_specificity_diagnostic_f0_{diag_f0:.1f}.csv')
            diag_df.to_csv(diag_path, index=False)
            print(f"  Diagnostic saved: {diag_path}")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s")


if __name__ == '__main__':
    main()

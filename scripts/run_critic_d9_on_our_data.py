#!/usr/bin/env python3
"""
Run Critic's D9 Analysis on Our Data
=====================================

Takes the critic's exact analysis functions (verbatim from eeg_phi.py)
and runs them on our pre-extracted FOOOF and GED peak datasets.

This isolates the question: is the difference between our results
(p < 0.05) and the critic's (p = 0.060) due to the data/extraction
method, or the analysis framework?

Usage:
  python scripts/run_critic_d9_on_our_data.py
  python scripts/run_critic_d9_on_our_data.py --dataset primary
  python scripts/run_critic_d9_on_our_data.py --all-datasets
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy import stats

# =========================================================================
# CRITIC'S FUNCTIONS — COPIED VERBATIM FROM eeg_phi.py
# (no modifications whatsoever)
# =========================================================================

PHI = (1 + np.sqrt(5)) / 2
F0_CLAIMED = 7.5          # critic's f0, not ours
N_PERM = 5000             # critic's permutation count
N_PERM_SWEEP = 1000

NAMED_RATIOS = {
    'φ': PHI, '2': 2.0, 'e': np.e, 'π': np.pi,
    '√2': np.sqrt(2), 'δ_S': 1 + np.sqrt(2),
    '2+√3': 2 + np.sqrt(3), '√3': np.sqrt(3),
    '3/2': 1.5, '5/3': 5/3, '7/4': 1.75, '√5': np.sqrt(5),
}


def lattice_phase(freqs, f0, ratio):
    """Map frequencies to lattice phase u in [0, 1).
    u = 0 -> boundary, u = 0.5 -> attractor."""
    return (np.log(freqs / f0) / np.log(ratio)) % 1.0


def enrichment_score(u_values, width=0.15):
    """(attractor density - boundary density) / expected.
    Positive supports hypothesis. Zero = uniform."""
    n = len(u_values)
    if n < 10:
        return 0.0
    near_bnd = np.sum((u_values < width) | (u_values > 1 - width))
    near_att = np.sum(np.abs(u_values - 0.5) < width)
    expected = n * 2 * width
    if expected < 1:
        return 0.0
    return (near_att - near_bnd) / expected


def enrichment_at(u_values, target, width):
    """Density enrichment at a specific phase position.
    Returns (observed - expected) / expected. Zero = uniform."""
    n = len(u_values)
    if n < 10:
        return 0.0
    d = np.abs(u_values - target)
    d = np.minimum(d, 1.0 - d)
    near = np.sum(d < width)
    expected = n * 2 * width
    return (near - expected) / expected if expected > 0 else 0.0


def phase_rotation_null(u_values, n_perm=N_PERM, rng=None):
    """Phase-rotation permutation null for enrichment score."""
    if rng is None:
        rng = np.random.default_rng(0)
    null = np.empty(n_perm)
    for i in range(n_perm):
        delta = rng.uniform()
        u_shifted = (u_values + delta) % 1.0
        null[i] = enrichment_score(u_shifted)
    return null


def phase_rotation_null_at(u_values, target, width, n_perm=N_PERM, rng=None):
    """Phase-rotation null for enrichment_at (position-specific)."""
    if rng is None:
        rng = np.random.default_rng(0)
    null = np.empty(n_perm)
    for i in range(n_perm):
        delta = rng.uniform()
        u_shifted = (u_values + delta) % 1.0
        null[i] = enrichment_at(u_shifted, target, width)
    return null


def kuiper_v(u_values):
    """Kuiper's V test for circular non-uniformity."""
    n = len(u_values)
    u_sorted = np.sort(u_values)
    i = np.arange(1, n + 1)
    D_plus = np.max(i / n - u_sorted)
    D_minus = np.max(u_sorted - (i - 1) / n)
    V = D_plus + D_minus
    Vstar = V * (np.sqrt(n) + 0.155 + 0.24 / np.sqrt(n))
    p = 0.0
    for j in range(1, 100):
        p += (4 * j**2 * Vstar**2 - 1) * np.exp(-2 * j**2 * Vstar**2)
    p = 2.0 * p
    return V, min(max(p, 0.0), 1.0)


# =========================================================================
# DATASET DEFINITIONS (from our run_ratio_specificity.py)
# =========================================================================

DATASETS = {
    'primary': {
        'path': 'papers/golden_ratio_peaks_ALL.csv',
        'freq_col': 'freq',
        'label': 'Primary (244K FOOOF peaks, 968 sessions)',
    },
    'emotions': {
        'path': 'golden_ratio_peaks_EMOTIONS copy.csv',
        'freq_col': 'freq',
        'label': 'EEGEmotions-27 (613K FOOOF peaks)',
    },
    'brain_invaders': {
        'path': 'golden_ratio_peaks_BRAIN_INVADERS_256Hz_clean.csv',
        'freq_col': 'freq',
        'label': 'Brain Invaders (828K FOOOF peaks)',
    },
    'physf_ged': {
        'path': 'exports_peak_distribution/physf_ged/truly_continuous/ged_peaks_truly_continuous.csv',
        'freq_col': 'frequency',
        'label': 'PhySF GED (407K spatial coherence peaks)',
    },
    'mpeng_ged': {
        'path': 'exports_peak_distribution/mpeng_ged/truly_continuous/ged_peaks_truly_continuous.csv',
        'freq_col': 'frequency',
        'label': 'MPENG GED peaks',
    },
    'emotions_ged': {
        'path': 'exports_peak_distribution/emotions_ged/truly_continuous/ged_peaks_truly_continuous.csv',
        'freq_col': 'frequency',
        'label': 'Emotions GED peaks',
    },
    # eegmmidb — same data as critic, three extraction methods
    'eegmmidb_medfilt': {
        'path': 'exports_peak_distribution/eegmmidb_peaks/eegmmidb_peaks_medfilt.csv',
        'freq_col': 'freq',
        'label': 'eegmmidb Medfilt (critic exact)',
    },
    'eegmmidb_fooof_critic': {
        'path': 'exports_peak_distribution/eegmmidb_peaks/eegmmidb_peaks_fooof_critic.csv',
        'freq_col': 'freq',
        'label': 'eegmmidb FOOOF (critic params)',
    },
    'eegmmidb_fooof_ours': {
        'path': 'exports_peak_distribution/eegmmidb_peaks/eegmmidb_peaks_fooof_ours.csv',
        'freq_col': 'freq',
        'label': 'eegmmidb FOOOF (our params)',
    },
}


def load_peaks(dataset_info):
    """Load peak frequencies from a dataset."""
    path = dataset_info['path']
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping")
        return None
    df = pd.read_csv(path)
    col = dataset_info['freq_col']
    if col not in df.columns:
        for alt in ['freq', 'frequency', 'peak_freq', 'cf']:
            if alt in df.columns:
                col = alt
                break
    freqs = df[col].values
    freqs = freqs[np.isfinite(freqs) & (freqs > 0)]
    return freqs


# =========================================================================
# RUN CRITIC'S D9 ON A SINGLE DATASET
# =========================================================================

def run_d9(freqs, dataset_name, output_dir):
    """Run the critic's D9 analysis on a peak frequency array."""
    os.makedirs(output_dir, exist_ok=True)
    inv_phi = 1.0 / PHI

    n = len(freqs)
    print(f"\n{'='*70}")
    print(f"CRITIC'S D9 ON: {dataset_name}")
    print(f"  {n:,} peaks, f0={F0_CLAIMED} (critic's value)")
    print(f"  Using critic's exact functions (verbatim from eeg_phi.py)")
    print(f"{'='*70}")

    u = lattice_phase(freqs, F0_CLAIMED, PHI)
    rng = np.random.default_rng(900)  # same seed as critic

    # ------------------------------------------------------------------
    # D1: Critic's original enrichment_score (attractor vs boundary)
    # ------------------------------------------------------------------
    print(f"\n[D1] Critic's original metric (attractor-boundary, w=0.15):")
    d1_obs = enrichment_score(u)
    d1_null = phase_rotation_null(u, N_PERM, np.random.default_rng(42))
    d1_p = np.mean(d1_null >= d1_obs)
    print(f"  Observed: {d1_obs:+.4f}")
    print(f"  Null: {d1_null.mean():+.4f} +/- {d1_null.std():.4f}")
    print(f"  p = {d1_p:.4f}  {'SIG' if d1_p < 0.05 else 'n.s.'}")

    # ------------------------------------------------------------------
    # D9 Part A: Targeted tests (exactly as critic does them)
    # ------------------------------------------------------------------
    print(f"\n[D9-A] Targeted tests:")
    tests = [
        ('u=0.500 w=0.15', 0.500, 0.15),
        ('u=0.618 w=0.15', inv_phi, 0.15),
        ('u=0.618 w=0.05', inv_phi, 0.05),
        ('u=0.500 w=0.05', 0.500, 0.05),
    ]

    targeted = {}
    for label, target, width in tests:
        obs = enrichment_at(u, target, width)
        null = phase_rotation_null_at(u, target, width, N_PERM, rng)
        p = np.mean(null >= obs)
        targeted[label] = (target, width, obs, null.mean(), null.std(), p)
        sig = " ***" if p < 0.01 else " *" if p < 0.05 else ""
        print(f"  {label}:  obs={obs:+.4f}  null={null.mean():+.4f}+/-{null.std():.4f}  p={p:.4f}{sig}")

    # ------------------------------------------------------------------
    # D9 Part B: Phase target sweep
    # ------------------------------------------------------------------
    print(f"\n[D9-B] Phase target sweep (w=0.05):")
    targets = np.linspace(0, 1, 200, endpoint=False)
    sweep_obs = np.array([enrichment_at(u, t, 0.05) for t in targets])
    sweep_excess = np.zeros_like(sweep_obs)

    null_idx = np.linspace(0, len(targets) - 1, 40, dtype=int)
    sweep_null_mean = np.full(len(targets), np.nan)
    for idx in null_idx:
        t = targets[idx]
        nl = phase_rotation_null_at(u, t, 0.05, N_PERM_SWEEP, rng)
        sweep_null_mean[idx] = nl.mean()
    valid = ~np.isnan(sweep_null_mean)
    sweep_null_mean = np.interp(np.arange(len(targets)),
                                np.where(valid)[0], sweep_null_mean[valid])
    sweep_excess = sweep_obs - sweep_null_mean

    best_target = targets[np.argmax(sweep_excess)]
    idx_05 = np.argmin(np.abs(targets - 0.5))
    idx_618 = np.argmin(np.abs(targets - inv_phi))
    print(f"  Best target: u={best_target:.3f} (excess={sweep_excess.max():+.4f})")
    print(f"  At u=0.500:  excess={sweep_excess[idx_05]:+.4f}")
    print(f"  At u=0.618:  excess={sweep_excess[idx_618]:+.4f}")

    # ------------------------------------------------------------------
    # D9 Part C: Width sensitivity at u=0.618
    # ------------------------------------------------------------------
    print(f"\n[D9-C] Width sensitivity at u=0.618:")
    widths = np.array([0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20])
    width_results = []
    for w in widths:
        obs_w = enrichment_at(u, inv_phi, w)
        nl = phase_rotation_null_at(u, inv_phi, w, N_PERM_SWEEP, rng)
        p_w = np.mean(nl >= obs_w)
        width_results.append((w, obs_w, nl.mean(), p_w))
        sig = " ***" if p_w < 0.01 else " *" if p_w < 0.05 else ""
        print(f"  w={w:.2f}: obs={obs_w:+.4f}  null={nl.mean():+.4f}  p={p_w:.4f}{sig}")

    # ------------------------------------------------------------------
    # D9 Part D: Kuiper V
    # ------------------------------------------------------------------
    V_raw, p_raw = kuiper_v(u)
    V_null = np.empty(N_PERM_SWEEP)
    for i in range(N_PERM_SWEEP):
        delta = rng.uniform()
        u_shifted = (u + delta) % 1.0
        V_null[i] = kuiper_v(u_shifted)[0]
    p_rot = np.mean(V_null >= V_raw)

    print(f"\n[D9-D] Kuiper V omnibus:")
    print(f"  V={V_raw:.4f}, p(asymptotic)={p_raw:.2e}")
    print(f"  p(phase-rotation)={p_rot:.4f}")
    if p_raw < 0.01 and p_rot > 0.05:
        print(f"  Non-uniform but NOT f0-specific")
    elif p_raw < 0.01 and p_rot < 0.05:
        print(f"  Non-uniform AND f0-specific")

    # ------------------------------------------------------------------
    # D9 Part E: D2 re-ranking at u=0.618
    # ------------------------------------------------------------------
    print(f"\n[D9-E] D2 re-ranking (u=0.618, w=0.05):")
    ratio_618 = {}
    for name, r in NAMED_RATIOS.items():
        u_r = lattice_phase(freqs, F0_CLAIMED, r)
        obs_r = enrichment_at(u_r, inv_phi, 0.05)
        nl = phase_rotation_null_at(u_r, inv_phi, 0.05, N_PERM_SWEEP, rng)
        p_r = np.mean(nl >= obs_r)
        excess_r = obs_r - nl.mean()
        ratio_618[name] = (r, obs_r, nl.mean(), excess_r, p_r)

    phi_excess = ratio_618['φ'][3]
    n_better = sum(1 for v in ratio_618.values() if v[3] > phi_excess)
    print(f"  φ rank: #{n_better + 1}/{len(ratio_618)}")

    for name in sorted(ratio_618, key=lambda n: -ratio_618[n][3]):
        v = ratio_618[name]
        sig = " ***" if v[4] < 0.01 else " *" if v[4] < 0.05 else ""
        marker = " <<<" if name == 'φ' else ""
        print(f"  {name:8s}: excess={v[3]:+.4f}  p={v[4]:.4f}{sig}{marker}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary_lines = [
        f"CRITIC'S D9 ANALYSIS — {dataset_name}",
        f"{'='*60}",
        f"Peaks: {n:,}  f0={F0_CLAIMED}  (critic's functions verbatim)",
        f"",
        f"D1 (attractor-boundary, w=0.15):",
        f"  obs={d1_obs:+.4f}  p={d1_p:.4f}  {'SIG' if d1_p < 0.05 else 'n.s.'}",
        f"",
        f"D9 targeted tests:",
    ]
    for label, (tgt, w, obs, nm, ns, p) in targeted.items():
        status = "SIG" if p < 0.05 else "n.s."
        summary_lines.append(f"  {label}: obs={obs:+.4f} p={p:.4f} [{status}]")

    summary_lines.extend([
        f"",
        f"Phase target sweep (w=0.05):",
        f"  Best: u={best_target:.3f}",
        f"  At u=0.618: excess={sweep_excess[idx_618]:+.4f}",
        f"  At u=0.500: excess={sweep_excess[idx_05]:+.4f}",
        f"",
        f"Kuiper V={V_raw:.4f}  p(asymp)={p_raw:.2e}  p(phase-rot)={p_rot:.4f}",
        f"",
        f"D2 re-ranking (u=0.618, w=0.05):",
        f"  φ rank: #{n_better + 1}/{len(ratio_618)}",
    ])
    for name in sorted(ratio_618, key=lambda n: -ratio_618[n][3])[:5]:
        v = ratio_618[name]
        marker = " <<<" if name == 'φ' else ""
        summary_lines.append(f"  {name:8s}: excess={v[3]:+.4f} p={v[4]:.4f}{marker}")

    summary = '\n'.join(summary_lines)
    print(f"\n{summary}")

    summary_path = os.path.join(output_dir, f'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary)

    return {
        'd1': (d1_obs, d1_p),
        'targeted': targeted,
        'best_target': best_target,
        'kuiper': (V_raw, p_raw, p_rot),
        'phi_rank': n_better + 1,
        'ratio_618': ratio_618,
    }


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Critic's D9 on our data")
    parser.add_argument('--dataset', type=str, default=None,
                        choices=list(DATASETS.keys()),
                        help='Named dataset')
    parser.add_argument('--all-datasets', action='store_true',
                        help='Run on all datasets')
    args = parser.parse_args()

    base_output = 'exports_peak_distribution/ratio_specificity/critic_d9'

    if args.all_datasets or args.dataset is None:
        run_datasets = DATASETS
    else:
        run_datasets = {args.dataset: DATASETS[args.dataset]}

    all_results = {}
    for key, info in run_datasets.items():
        freqs = load_peaks(info)
        if freqs is None:
            continue
        out_dir = os.path.join(base_output, key)
        result = run_d9(freqs, info['label'], out_dir)
        all_results[key] = result

    # ------------------------------------------------------------------
    # Cross-dataset summary
    # ------------------------------------------------------------------
    if len(all_results) > 1:
        print(f"\n\n{'='*70}")
        print("CROSS-DATASET COMPARISON (critic's D9 on our data)")
        print(f"{'='*70}")
        print(f"\n{'Dataset':<30s} {'D1 p':>8s} {'u=.618 p':>10s} {'φ rank':>8s} {'Kuiper p_rot':>12s}")
        print("-" * 70)

        for key, result in all_results.items():
            label = DATASETS[key]['label'][:28]
            d1_p = result['d1'][1]
            d9_p = result['targeted']['u=0.618 w=0.05'][5]
            phi_rank = result['phi_rank']
            k_p = result['kuiper'][2]
            d9_sig = "*" if d9_p < 0.05 else " "
            print(f"{label:<30s} {d1_p:>8.4f} {d9_p:>9.4f}{d9_sig} {phi_rank:>5d}/12 {k_p:>12.4f}")

        # Reference: critic's eegmmidb results
        print(f"\n{'Critic eegmmidb (reference)':<30s} {'---':>8s} {'0.0596':>10s} {'1':>5s}/12 {'0.6140':>12s}")

        summary_path = os.path.join(base_output, 'cross_dataset_summary.txt')
        os.makedirs(base_output, exist_ok=True)
        with open(summary_path, 'w') as f:
            f.write("CROSS-DATASET COMPARISON\n")
            f.write("Critic's D9 analysis (verbatim) on our pre-extracted peaks\n")
            f.write(f"f0={F0_CLAIMED}, N_PERM={N_PERM}\n\n")
            for key, result in all_results.items():
                d9_p = result['targeted']['u=0.618 w=0.05'][5]
                f.write(f"{DATASETS[key]['label']}: p(u=0.618)={d9_p:.4f}, "
                        f"phi_rank={result['phi_rank']}/12, "
                        f"kuiper_p_rot={result['kuiper'][2]:.4f}\n")
        print(f"\nSummary saved to: {summary_path}")


if __name__ == '__main__':
    main()

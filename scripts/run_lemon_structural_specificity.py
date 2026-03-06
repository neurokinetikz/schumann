#!/usr/bin/env python3
"""
Run EEGMMIDB Paper 2 Pipeline on LEMON Dataset
================================================================

Aggregates per-subject LEMON peak CSVs into a single flat CSV and runs
the full structural phi-specificity pipeline (Tests A-C, band-stratified,
f0 sweeps, shuffle/prominence) — identical to the EEGMMIDB analysis.

Variants:
  45hz  — per_subject/ extraction ([1,45] Hz FOOOF range)
  85hz  — per_subject_85hz/ extraction ([1,85] Hz FOOOF range)

Usage:
    python scripts/run_lemon_structural_specificity.py                # both variants
    python scripts/run_lemon_structural_specificity.py --variant 45hz # 45 Hz only
    python scripts/run_lemon_structural_specificity.py --aggregate-only
    python scripts/run_lemon_structural_specificity.py --skip-shuffle
"""

import os
import sys
import glob
import time
import argparse
import subprocess

import numpy as np
import pandas as pd


# =========================================================================
# CONSTANTS
# =========================================================================

VARIANTS = {
    '45hz': {
        'peak_dir': 'exports_lemon/per_subject',
        'out_dir': 'exports_lemon/structural_specificity',
        'glob': 'sub-*_peaks.csv',
        'label': 'LEMON [1,45] Hz',
    },
    '85hz': {
        'peak_dir': 'exports_lemon/per_subject_85hz',
        'out_dir': 'exports_lemon/structural_specificity_85hz',
        'glob': 'sub-*_peaks.csv',
        'label': 'LEMON [1,85] Hz',
    },
}

EEGMMIDB_RESULTS = 'exports_peak_distribution/eegmmidb_fooof'


# =========================================================================
# AGGREGATION
# =========================================================================

def aggregate_peaks(peak_dir, glob_pattern, out_path):
    """Concatenate per-subject peak CSVs into one flat file."""
    files = sorted(glob.glob(os.path.join(peak_dir, glob_pattern)))
    if not files:
        print(f"  ERROR: No files matching {peak_dir}/{glob_pattern}")
        return None

    dfs = []
    for f in files:
        subj = os.path.basename(f).replace('_peaks.csv', '')
        df = pd.read_csv(f)
        df['subject'] = subj
        df['dataset'] = 'lemon'
        dfs.append(df)

    all_peaks = pd.concat(dfs, ignore_index=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    all_peaks.to_csv(out_path, index=False)

    n_subj = len(files)
    n_peaks = len(all_peaks)
    freq_min = all_peaks['freq'].min()
    freq_max = all_peaks['freq'].max()
    print(f"  Aggregated {n_subj} subjects -> {n_peaks:,} peaks "
          f"[{freq_min:.1f}, {freq_max:.1f}] Hz")
    print(f"  Saved: {out_path}")
    return all_peaks


# =========================================================================
# PIPELINE RUNNER
# =========================================================================

def run_structural_tests(input_csv, output_dir, full_sweep=True):
    """Run structural_phi_specificity.py on aggregated peaks."""
    cmd = [
        sys.executable, 'scripts/structural_phi_specificity.py',
        '--input', input_csv,
        '--output-dir', output_dir,
    ]
    if full_sweep:
        cmd.append('--full-sweep')

    print(f"\n  Running structural tests...")
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def run_shuffle_prominence(input_csv, output_dir):
    """Run shuffle/prominence analysis on aggregated peaks."""
    cmd = [
        sys.executable, 'scripts/run_eegmmidb_shuffle_prominence.py',
        '--input', input_csv,
        '--output-dir', output_dir,
    ]

    print(f"\n  Running shuffle/prominence tests...")
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


# =========================================================================
# COMPARISON
# =========================================================================

def compare_results(lemon_dir, eegmmidb_dir, variant_label):
    """Print side-by-side comparison of LEMON vs EEGMMIDB results."""
    print(f"\n{'='*70}")
    print(f"COMPARISON: {variant_label} vs EEGMMIDB")
    print(f"{'='*70}")

    # Test A: Fair enrichment
    lemon_a = os.path.join(lemon_dir, 'structural_specificity_enrichment.csv')
    eegmmidb_a = os.path.join(eegmmidb_dir, 'structural_specificity_enrichment.csv')

    if os.path.exists(lemon_a) and os.path.exists(eegmmidb_a):
        df_l = pd.read_csv(lemon_a)
        df_e = pd.read_csv(eegmmidb_a)

        print(f"\n  TEST A: Fair Enrichment (Structural Score)")
        print(f"  {'Base':>6s} {'EEGMMIDB':>10s} {'LEMON':>10s} {'Diff':>10s}")
        print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10}")

        for _, row_l in df_l.sort_values('structural_score', ascending=False).iterrows():
            base = row_l['base_name']
            ss_l = row_l['structural_score']
            row_e = df_e[df_e['base_name'] == base]
            if len(row_e) > 0:
                ss_e = row_e.iloc[0]['structural_score']
                diff = ss_l - ss_e
                label = row_l.get('base_label', base)
                marker = " <<<" if base == 'phi' else ""
                print(f"  {label:>6s} {ss_e:+10.1f} {ss_l:+10.1f} {diff:+10.1f}{marker}")

    # Shuffle results
    lemon_sh = os.path.join(lemon_dir, 'shuffle_prominence', 'shuffle_test_results.csv')
    eegmmidb_sh = os.path.join(eegmmidb_dir, 'shuffle_prominence', 'shuffle_test_results.csv')

    if os.path.exists(lemon_sh) and os.path.exists(eegmmidb_sh):
        df_l = pd.read_csv(lemon_sh)
        df_e = pd.read_csv(eegmmidb_sh)

        print(f"\n  SHUFFLE TEST (phi only)")
        print(f"  {'f0':>5s} {'Dataset':>10s} {'SS':>8s} {'z_shuf':>8s} {'p_shuf':>8s}")
        print(f"  {'-'*5} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
        for _, row in df_e.iterrows():
            print(f"  {row['f0']:5.1f} {'EEGMMIDB':>10s} {row['SS_obs']:+8.1f} "
                  f"{row['z_shuffle']:+8.1f} {row['p_shuffle']:8.4f}")
        for _, row in df_l.iterrows():
            print(f"  {row['f0']:5.1f} {'LEMON':>10s} {row['SS_obs']:+8.1f} "
                  f"{row['z_shuffle']:+8.1f} {row['p_shuffle']:8.4f}")


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run EEGMMIDB paper 2 pipeline on LEMON peaks')
    parser.add_argument('--variant', choices=['45hz', '85hz', 'both'],
                        default='both', help='Which extraction variant to run')
    parser.add_argument('--aggregate-only', action='store_true',
                        help='Only aggregate peaks, skip analysis')
    parser.add_argument('--skip-shuffle', action='store_true',
                        help='Skip shuffle/prominence tests')
    parser.add_argument('--skip-sweep', action='store_true',
                        help='Skip f0 sensitivity sweeps')
    parser.add_argument('--compare', action='store_true',
                        help='Only print comparison (skip running)')
    args = parser.parse_args()

    variants = ['45hz', '85hz'] if args.variant == 'both' else [args.variant]
    t0 = time.time()

    for variant in variants:
        cfg = VARIANTS[variant]
        print(f"\n{'='*70}")
        print(f"  VARIANT: {cfg['label']}")
        print(f"{'='*70}")

        agg_path = os.path.join(cfg['out_dir'], f'golden_ratio_peaks_LEMON.csv')

        # Compare only mode
        if args.compare:
            compare_results(cfg['out_dir'], EEGMMIDB_RESULTS, cfg['label'])
            continue

        # Step 1: Aggregate
        print(f"\n--- Aggregating peaks from {cfg['peak_dir']} ---")
        agg_df = aggregate_peaks(cfg['peak_dir'], cfg['glob'], agg_path)
        if agg_df is None:
            print(f"  Skipping {variant} — no peaks found")
            continue

        if args.aggregate_only:
            continue

        # Step 2: Structural tests
        rc = run_structural_tests(agg_path, cfg['out_dir'],
                                  full_sweep=not args.skip_sweep)
        if rc != 0:
            print(f"  WARNING: structural tests exited with code {rc}")

        # Step 3: Shuffle/prominence
        if not args.skip_shuffle:
            shuffle_dir = os.path.join(cfg['out_dir'], 'shuffle_prominence')
            rc = run_shuffle_prominence(agg_path, shuffle_dir)
            if rc != 0:
                print(f"  WARNING: shuffle/prominence exited with code {rc}")

        # Step 4: Compare with EEGMMIDB
        compare_results(cfg['out_dir'], EEGMMIDB_RESULTS, cfg['label'])

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"COMPLETE — {elapsed:.0f} seconds")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

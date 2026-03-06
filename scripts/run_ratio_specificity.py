#!/usr/bin/env python3
"""
Ratio Specificity & Phase-Rotation Null — Full Analysis Runner
==============================================================

Runs all four analyses (D1–D4) on FOOOF and/or GED peak datasets.

Usage:
  python scripts/run_ratio_specificity.py
  python scripts/run_ratio_specificity.py --peaks papers/golden_ratio_peaks_ALL.csv
  python scripts/run_ratio_specificity.py --f0 7.6 --n-perm 2000
  python scripts/run_ratio_specificity.py --dataset emotions
  python scripts/run_ratio_specificity.py --all-datasets

Output: exports_peak_distribution/ratio_specificity/
"""

import sys
sys.path.insert(0, './lib')

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

from ratio_specificity import (
    PHI, PHI_INV, F0,
    phi_full_enrichment,
    phase_rotation_null, ratio_specificity_test,
    sweep_f0_with_null, sweep_f0_ratio_2d, null_threshold_2d,
    plot_d1_phase_rotation, plot_d2_ratio_specificity,
    plot_d3_f0_sweep, plot_d4_heatmap,
    generate_summary,
)

# ============================================================================
# DATASET DEFINITIONS
# ============================================================================

FOOOF_DATASETS = {
    'primary': {
        'path': 'papers/golden_ratio_peaks_ALL.csv',
        'freq_col': 'freq',
        'label': 'Primary (244K FOOOF peaks, 968 sessions)',
    },
    'emotions': {
        'path': 'golden_ratio_peaks_EMOTIONS copy.csv',
        'freq_col': 'freq',
        'label': 'EEGEmotions-27 (613K FOOOF peaks, 2342 sessions)',
    },
    'brain_invaders': {
        'path': 'golden_ratio_peaks_BRAIN_INVADERS_256Hz_clean.csv',
        'freq_col': 'freq',
        'label': 'Brain Invaders (828K FOOOF peaks)',
    },
}

GED_DATASETS = {
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
        # Try common alternatives
        for alt in ['freq', 'frequency', 'peak_freq', 'cf']:
            if alt in df.columns:
                col = alt
                break

    freqs = df[col].values
    freqs = freqs[np.isfinite(freqs) & (freqs > 0)]
    return freqs


def run_analysis(freqs, dataset_name, output_dir, f0=F0, n_perm=1000):
    """Run all four analyses on a single peak dataset."""
    os.makedirs(output_dir, exist_ok=True)
    n = len(freqs)
    print(f"\n{'='*60}")
    print(f"DATASET: {dataset_name}")
    print(f"  {n:,} peaks, range: {freqs.min():.2f}–{freqs.max():.2f} Hz")
    print(f"  f₀ = {f0:.2f} Hz, n_perm = {n_perm}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # D1: Phase-rotation null
    # ------------------------------------------------------------------
    print("\n[D1] Phase-rotation null for φ at noble_1 (0.618)...")
    null_dist, observed, p_val, z = phase_rotation_null(
        freqs, f0, PHI, n_perm=n_perm, window=0.05,
        metric='predicted', predicted_offset=PHI_INV
    )
    print(f"  Observed enrichment: {observed:+.1f}%")
    print(f"  Null 95th:           {np.percentile(null_dist, 95):+.1f}%")
    print(f"  p = {p_val:.4f}, z = {z:.2f}")
    print(f"  SURVIVES: {'YES' if p_val < 0.05 else 'NO'}")

    plot_d1_phase_rotation(
        observed, null_dist, 'φ', f0,
        output_path=os.path.join(output_dir, 'D1_phase_rotation_null.png')
    )
    pd.DataFrame({
        'metric': ['observed', 'null_mean', 'null_std', 'null_95th', 'p_value', 'z_score'],
        'value': [observed, null_dist.mean(), null_dist.std(),
                  np.percentile(null_dist, 95), p_val, z]
    }).to_csv(os.path.join(output_dir, 'D1_results.csv'), index=False)

    # ------------------------------------------------------------------
    # D2: Ratio specificity
    # ------------------------------------------------------------------
    print("\n[D2] Ratio specificity test (12 ratios)...")
    d2_df = ratio_specificity_test(freqs, f0=f0, n_perm=n_perm)
    print("  Ranking (φ@noble_1=0.618, others@lattice=0.0):")
    for _, row in d2_df.head(5).iterrows():
        marker = " <<<" if row['ratio_name'] == 'φ' else ""
        print(f"    {row['ratio_name']:>5s}: {row['predicted_enrichment']:+7.1f}% "
              f"@ {row['predicted_offset']:.3f}  "
              f"p={row['p_value']:.3f}{marker}")

    phi_row = d2_df[d2_df['ratio_name'] == 'φ']
    phi_enrich_pred = phi_row['predicted_enrichment'].values[0]
    phi_rank = (d2_df['predicted_enrichment'] >= phi_enrich_pred).sum()
    print(f"  φ rank: {phi_rank}/{len(d2_df)}")

    plot_d2_ratio_specificity(
        d2_df, output_path=os.path.join(output_dir, 'D2_ratio_specificity.png')
    )
    d2_df.to_csv(os.path.join(output_dir, 'D2_results.csv'), index=False)

    # ------------------------------------------------------------------
    # D3: f₀ sweep
    # ------------------------------------------------------------------
    print("\n[D3] f₀ sweep for φ (6.5–8.5 Hz)...")
    d3_df = sweep_f0_with_null(
        freqs, ratio=PHI, f0_range=(6.5, 8.5), step=0.01,
        n_perm=min(n_perm, 500), window=0.05
    )
    opt_idx = d3_df['enrichment'].idxmax()
    opt_f0 = d3_df.loc[opt_idx, 'f0']
    opt_enrich = d3_df.loc[opt_idx, 'enrichment']
    exceeds_null = (d3_df['enrichment'] > d3_df['null_95th']).any()
    print(f"  Optimal f₀: {opt_f0:.3f} Hz ({opt_enrich:+.1f}%)")
    print(f"  Exceeds null threshold: {'YES' if exceeds_null else 'NO'}")

    # Enrichment at paper f₀
    row_76 = d3_df.iloc[(d3_df['f0'] - 7.60).abs().argsort()[:1]]
    print(f"  At f₀=7.60: {row_76['enrichment'].values[0]:+.1f}%")

    plot_d3_f0_sweep(
        d3_df, 'φ', output_path=os.path.join(output_dir, 'D3_f0_sweep_phi.png')
    )
    d3_df.to_csv(os.path.join(output_dir, 'D3_f0_sweep.csv'), index=False)

    # ------------------------------------------------------------------
    # D4: 2D heatmap
    # ------------------------------------------------------------------
    print("\n[D4] 2D (f₀, ratio) heatmap...")
    f0_vals, ratio_vals, enrich_matrix = sweep_f0_ratio_2d(
        freqs,
        f0_range=(6.5, 8.5), f0_step=0.02,
        ratio_range=(1.1, 3.5), ratio_step=0.01,
        window=0.05
    )
    null_thresh = null_threshold_2d(freqs, n_perm=min(n_perm, 200), window=0.05)
    print(f"  Null threshold: {null_thresh:+.1f}%")

    max_idx = np.unravel_index(enrich_matrix.argmax(), enrich_matrix.shape)
    opt_r = ratio_vals[max_idx[0]]
    opt_f0_2d = f0_vals[max_idx[1]]
    opt_val = enrich_matrix[max_idx]
    print(f"  Unconstrained optimum: f₀={opt_f0_2d:.2f}, r={opt_r:.3f} ({opt_val:+.1f}%)")

    # Value at (7.6, φ)
    f0_idx = np.argmin(np.abs(f0_vals - 7.60))
    r_idx = np.argmin(np.abs(ratio_vals - PHI))
    phi_76 = enrich_matrix[r_idx, f0_idx]
    print(f"  At (7.6, φ):           {phi_76:+.1f}%")

    plot_d4_heatmap(
        f0_vals, ratio_vals, enrich_matrix, null_thresh,
        output_path=os.path.join(output_dir, 'D4_f0_ratio_heatmap.png')
    )

    # Save flattened heatmap
    heatmap_rows = []
    for i, r in enumerate(ratio_vals):
        for j, f0_v in enumerate(f0_vals):
            heatmap_rows.append({'f0': f0_v, 'ratio': r, 'enrichment': enrich_matrix[i, j]})
    pd.DataFrame(heatmap_rows).to_csv(
        os.path.join(output_dir, 'D4_heatmap.csv'), index=False
    )

    # ------------------------------------------------------------------
    # φ-specific full position enrichment
    # ------------------------------------------------------------------
    print("\n[Bonus] φ position-type enrichment at f₀=7.6:")
    phi_positions = phi_full_enrichment(freqs, f0, window=0.05)
    for ptype, enrich in sorted(phi_positions.items(),
                                  key=lambda x: x[1], reverse=True):
        print(f"    {ptype:<15s}: {enrich:+.1f}%")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    d1_results = (null_dist, observed, p_val, z)
    d4_info = (f0_vals, ratio_vals, enrich_matrix, null_thresh)
    summary = generate_summary(d1_results, d2_df, d3_df, d4_info, dataset_name)
    print(f"\n{summary}")

    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"\nAll outputs saved to: {output_dir}/")

    return {
        'd1': d1_results,
        'd2': d2_df,
        'd3': d3_df,
        'd4': d4_info,
        'phi_positions': phi_positions,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Ratio Specificity Analysis')
    parser.add_argument('--peaks', type=str, default=None,
                        help='Path to peaks CSV (freq column)')
    parser.add_argument('--freq-col', type=str, default='freq',
                        help='Column name for frequencies')
    parser.add_argument('--dataset', type=str, default='primary',
                        choices=list(FOOOF_DATASETS.keys()) + list(GED_DATASETS.keys()),
                        help='Named dataset to use')
    parser.add_argument('--all-datasets', action='store_true',
                        help='Run on all available datasets')
    parser.add_argument('--f0', type=float, default=F0,
                        help='Fundamental frequency (default 7.60)')
    parser.add_argument('--n-perm', type=int, default=1000,
                        help='Number of permutations (default 1000)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')
    args = parser.parse_args()

    base_output = args.output_dir or 'exports_peak_distribution/ratio_specificity'

    if args.peaks:
        # Custom peaks file
        print(f"Loading custom peaks: {args.peaks}")
        df = pd.read_csv(args.peaks)
        freqs = df[args.freq_col].values
        freqs = freqs[np.isfinite(freqs) & (freqs > 0)]
        name = os.path.basename(args.peaks).replace('.csv', '')
        out_dir = os.path.join(base_output, name)
        run_analysis(freqs, name, out_dir, f0=args.f0, n_perm=args.n_perm)

    elif args.all_datasets:
        # Run on all available datasets
        all_datasets = {**FOOOF_DATASETS, **GED_DATASETS}
        for key, info in all_datasets.items():
            freqs = load_peaks(info)
            if freqs is None:
                continue
            out_dir = os.path.join(base_output, key)
            run_analysis(freqs, info['label'], out_dir,
                         f0=args.f0, n_perm=args.n_perm)

    else:
        # Single named dataset
        all_datasets = {**FOOOF_DATASETS, **GED_DATASETS}
        if args.dataset not in all_datasets:
            print(f"Unknown dataset: {args.dataset}")
            print(f"Available: {list(all_datasets.keys())}")
            return

        info = all_datasets[args.dataset]
        freqs = load_peaks(info)
        if freqs is None:
            print("No data loaded!")
            return

        out_dir = os.path.join(base_output, args.dataset)
        run_analysis(freqs, info['label'], out_dir,
                     f0=args.f0, n_perm=args.n_perm)


if __name__ == '__main__':
    main()

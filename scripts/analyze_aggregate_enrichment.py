#!/usr/bin/env python3
"""
Comprehensive Statistical Validation of Combined GED Dataset
=============================================================

Answers three key questions about the truly continuous GED peak distribution:
1. Precise numerical enrichment at each position type
2. Session-level consistency (compare to paper's 87.4%)
3. Optimal f₀ validation (does combined data confirm 7.60 Hz?)

Input: Truly continuous GED peaks from PhySF, MPENG, Emotions
Output: Tables, figures, and console summary
"""

import sys
sys.path.insert(0, './lib')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

from phi_frequency_model import PHI, F0, POSITION_OFFSETS

# Extended position offsets (including noble_5, noble_6 and inverses)
PHI_INV = 1.0 / PHI
EXTENDED_OFFSETS = {
    'boundary':    0.000,
    'noble_6':     PHI_INV ** 6,  # ~0.0557
    'noble_5':     PHI_INV ** 5,  # ~0.0902
    'noble_4':     PHI_INV ** 4,  # ~0.146
    'noble_3':     PHI_INV ** 3,  # ~0.236
    'noble_2':     PHI_INV ** 2,  # ~0.382
    'attractor':   0.500,
    'noble_1':     PHI_INV ** 1,  # ~0.618
    'inv_noble_3': 1 - PHI_INV ** 3,  # ~0.764
    'inv_noble_4': 1 - PHI_INV ** 4,  # ~0.854
    'inv_noble_5': 1 - PHI_INV ** 5,  # ~0.9098
    'inv_noble_6': 1 - PHI_INV ** 6,  # ~0.9443
}

# Dataset paths
DATASETS = {
    'PhySF': 'exports_peak_distribution/physf_ged/truly_continuous/ged_peaks_truly_continuous.csv',
    'MPENG': 'exports_peak_distribution/mpeng_ged/truly_continuous/ged_peaks_truly_continuous.csv',
    'Emotions': 'exports_peak_distribution/emotions_ged/truly_continuous/ged_peaks_truly_continuous.csv',
}

OUTPUT_DIR = 'exports_peak_distribution/aggregate_truly_continuous'


# =============================================================================
# ENRICHMENT ANALYSIS
# =============================================================================

def compute_lattice_coordinate(freqs, f0=F0):
    """
    Map frequencies to lattice coordinate u = [log_φ(f/f0)] mod 1.

    This collapses all peaks onto the unit interval [0, 1) where
    position predictions are fixed offsets.
    """
    phi_exps = np.log(freqs / f0) / np.log(PHI)
    return phi_exps % 1.0


def compute_enrichment_at_positions(lattice_coords, position_offsets, window=0.05):
    """
    Compute enrichment at each position type.

    Enrichment = (observed_fraction / expected_fraction - 1) × 100%
    where expected_fraction = 2 × window (uniform distribution on [0,1))

    Parameters
    ----------
    lattice_coords : np.ndarray
        Peak positions in lattice coordinates [0, 1)
    position_offsets : dict
        Position type -> offset value
    window : float
        Half-window width for counting peaks

    Returns
    -------
    dict : Position type -> enrichment metrics
    """
    n_total = len(lattice_coords)
    expected_frac = 2 * window  # Expected under uniform distribution

    results = {}
    for ptype, offset in position_offsets.items():
        # Count peaks within window of this position
        # Handle wrap-around at boundary (0/1)
        if offset < window:
            # Window wraps from end
            in_window = ((lattice_coords >= 0) & (lattice_coords < offset + window)) | \
                        (lattice_coords >= 1 - (window - offset))
        elif offset > 1 - window:
            # Window wraps to beginning
            in_window = ((lattice_coords >= offset - window) & (lattice_coords < 1)) | \
                        (lattice_coords < window - (1 - offset))
        else:
            in_window = (lattice_coords >= offset - window) & (lattice_coords < offset + window)

        n_in_window = in_window.sum()
        observed_frac = n_in_window / n_total

        # Enrichment relative to uniform
        enrichment = (observed_frac / expected_frac - 1) * 100  # Percentage

        results[ptype] = {
            'offset': offset,
            'n_peaks': n_in_window,
            'observed_frac': observed_frac,
            'expected_frac': expected_frac,
            'enrichment_pct': enrichment,
        }

    return results


def bootstrap_enrichment_ci(lattice_coords, position_offsets, window=0.05,
                            n_bootstrap=1000, ci=0.95):
    """
    Compute bootstrap confidence intervals for enrichment at each position.
    """
    rng = np.random.default_rng(42)
    n = len(lattice_coords)

    # Storage for bootstrap samples
    boot_enrichments = {ptype: [] for ptype in position_offsets}

    for _ in range(n_bootstrap):
        # Resample with replacement
        boot_idx = rng.choice(n, size=n, replace=True)
        boot_coords = lattice_coords[boot_idx]

        # Compute enrichment for this bootstrap sample
        boot_results = compute_enrichment_at_positions(boot_coords, position_offsets, window)

        for ptype in position_offsets:
            boot_enrichments[ptype].append(boot_results[ptype]['enrichment_pct'])

    # Compute confidence intervals
    alpha = (1 - ci) / 2
    ci_results = {}
    for ptype in position_offsets:
        samples = np.array(boot_enrichments[ptype])
        ci_results[ptype] = {
            'mean': np.mean(samples),
            'std': np.std(samples),
            'ci_lower': np.percentile(samples, alpha * 100),
            'ci_upper': np.percentile(samples, (1 - alpha) * 100),
        }

    return ci_results


# =============================================================================
# SESSION-LEVEL CONSISTENCY
# =============================================================================

def compute_session_consistency(peaks_df, f0=F0, min_peaks=20, window=0.05):
    """
    Compute session-level consistency metrics.

    For each session with >= min_peaks:
    - Compute attractor enrichment vs boundary enrichment
    - Track if attractor > boundary (the key prediction)

    Returns
    -------
    dict with summary statistics and per-session results
    """
    # Get unique sessions
    if 'session' not in peaks_df.columns:
        print("Warning: No 'session' column found")
        return None

    sessions = peaks_df['session'].unique()

    results = []
    for sess in sessions:
        sess_peaks = peaks_df[peaks_df['session'] == sess]['frequency'].values

        if len(sess_peaks) < min_peaks:
            continue

        # Compute lattice coordinates
        lattice = compute_lattice_coordinate(sess_peaks, f0)

        # Compute enrichment at key positions
        enrichment = compute_enrichment_at_positions(lattice, EXTENDED_OFFSETS, window)

        attractor_enrich = enrichment['attractor']['enrichment_pct']
        boundary_enrich = enrichment['boundary']['enrichment_pct']
        noble1_enrich = enrichment['noble_1']['enrichment_pct']
        noble2_enrich = enrichment['noble_2']['enrichment_pct']

        results.append({
            'session': sess,
            'n_peaks': len(sess_peaks),
            'attractor_enrich': attractor_enrich,
            'boundary_enrich': boundary_enrich,
            'noble1_enrich': noble1_enrich,
            'noble2_enrich': noble2_enrich,
            'attractor_gt_boundary': attractor_enrich > boundary_enrich,
            'correct_ordering': noble1_enrich > attractor_enrich > noble2_enrich > boundary_enrich,
        })

    if not results:
        return None

    results_df = pd.DataFrame(results)

    # Compute summary statistics
    n_sessions = len(results_df)
    pct_attractor_gt_boundary = 100 * results_df['attractor_gt_boundary'].sum() / n_sessions
    pct_correct_ordering = 100 * results_df['correct_ordering'].sum() / n_sessions

    # Effect size (Cohen's d for attractor - boundary)
    diff = results_df['attractor_enrich'] - results_df['boundary_enrich']
    cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0

    return {
        'n_sessions': n_sessions,
        'pct_attractor_gt_boundary': pct_attractor_gt_boundary,
        'pct_correct_ordering': pct_correct_ordering,
        'cohens_d': cohens_d,
        'mean_attractor_enrich': results_df['attractor_enrich'].mean(),
        'mean_boundary_enrich': results_df['boundary_enrich'].mean(),
        'per_session_df': results_df,
    }


# =============================================================================
# F0 OPTIMIZATION
# =============================================================================

def compute_alignment_score(freqs, f0_candidate, position_offsets, kde_bw=0.02):
    """
    Compute alignment between peak distribution and φⁿ predictions for a given F0.

    Returns alignment score (sum of KDE density at each prediction offset).
    """
    phi_exps = np.log(freqs / f0_candidate) / np.log(PHI)
    fractional = phi_exps % 1.0

    kde = gaussian_kde(fractional, bw_method=kde_bw)

    alignment_score = 0
    details = {}

    for ptype, offset in position_offsets.items():
        if offset == 0.0:
            density = (kde(0.001)[0] + kde(0.999)[0]) / 2
        else:
            density = kde(offset)[0]

        alignment_score += density
        details[ptype] = {'offset': offset, 'density': density}

    return alignment_score, details


def sweep_f0_optimization(freqs, f0_range=(6.5, 8.5), step=0.005):
    """
    Sweep F0 values to find optimal alignment.
    """
    f0_values = np.arange(f0_range[0], f0_range[1] + step, step)

    scores = []
    for f0 in f0_values:
        score, details = compute_alignment_score(freqs, f0, EXTENDED_OFFSETS)
        row = {'f0': f0, 'alignment_score': score}
        for ptype, info in details.items():
            row[f'{ptype}_density'] = info['density']
        scores.append(row)

    results = pd.DataFrame(scores)

    # Find local maxima
    alignment_scores = results['alignment_score'].values
    peak_idx, _ = find_peaks(alignment_scores, distance=10, prominence=0.05)

    local_maxima = [(results.iloc[i]['f0'], results.iloc[i]['alignment_score'])
                    for i in peak_idx]
    local_maxima.sort(key=lambda x: x[1], reverse=True)

    # Choose best internal maximum (away from edges)
    edge_margin = 0.05 * (f0_range[1] - f0_range[0])
    interior_maxima = [(f0, score) for f0, score in local_maxima
                       if f0 > f0_range[0] + edge_margin and f0 < f0_range[1] - edge_margin]

    if interior_maxima:
        optimal_f0 = interior_maxima[0][0]
    else:
        optimal_f0 = results.loc[results['alignment_score'].idxmax(), 'f0']

    return results, optimal_f0, local_maxima


def plot_f0_optimization(results, freqs, optimal_f0, output_path):
    """Generate F0 optimization visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: F0 sweep curve
    ax = axes[0, 0]
    ax.plot(results['f0'], results['alignment_score'], 'b-', linewidth=2)
    ax.axvline(7.60, color='red', linestyle='--', linewidth=1.5, label='Current F0=7.60')
    ax.axvline(optimal_f0, color='green', linestyle='-', linewidth=2, label=f'Optimal F0={optimal_f0:.4f}')
    ax.set_xlabel('F0 (Hz)', fontsize=11)
    ax.set_ylabel('Alignment Score', fontsize=11)
    ax.set_title('F0 Optimization: Peak-Prediction Alignment', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 2: Fractional distribution comparison
    ax = axes[0, 1]
    frac_current = (np.log(freqs / 7.60) / np.log(PHI)) % 1.0
    frac_optimal = (np.log(freqs / optimal_f0) / np.log(PHI)) % 1.0

    ax.hist(frac_current, bins=100, range=(0, 1), alpha=0.5, label='F0=7.60', color='red', density=True)
    ax.hist(frac_optimal, bins=100, range=(0, 1), alpha=0.5, label=f'F0={optimal_f0:.4f}', color='green', density=True)

    for ptype, offset in EXTENDED_OFFSETS.items():
        ax.axvline(offset, color='gray', linestyle=':', alpha=0.5, linewidth=1)

    ax.set_xlabel('Fractional φ-exponent (mod 1)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Distribution Shift: Current vs Optimal F0', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)

    # Panel 3: Per-position density comparison
    ax = axes[1, 0]
    position_order = ['boundary', 'noble_6', 'noble_5', 'noble_4', 'noble_3', 'noble_2',
                      'attractor', 'noble_1', 'inv_noble_3', 'inv_noble_4', 'inv_noble_5', 'inv_noble_6']

    _, details_current = compute_alignment_score(freqs, 7.60, EXTENDED_OFFSETS)
    _, details_optimal = compute_alignment_score(freqs, optimal_f0, EXTENDED_OFFSETS)

    x = np.arange(len(position_order))
    width = 0.35

    densities_current = [details_current[p]['density'] for p in position_order]
    densities_optimal = [details_optimal[p]['density'] for p in position_order]

    ax.bar(x - width/2, densities_current, width, label='F0=7.60', color='red', alpha=0.7)
    ax.bar(x + width/2, densities_optimal, width, label=f'F0={optimal_f0:.4f}', color='green', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([p.replace('_', '\n') for p in position_order], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('KDE Density', fontsize=11)
    ax.set_title('Per-Position Alignment', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)

    # Panel 4: Summary text
    ax = axes[1, 1]
    ax.axis('off')

    current_score, _ = compute_alignment_score(freqs, 7.60, EXTENDED_OFFSETS)
    optimal_score, _ = compute_alignment_score(freqs, optimal_f0, EXTENDED_OFFSETS)
    improvement = (optimal_score - current_score) / current_score * 100

    summary = f"""
    F0 OPTIMIZATION RESULTS
    ══════════════════════════════════════

    Current F0:      7.6000 Hz
    Optimal F0:      {optimal_f0:.4f} Hz

    Alignment Scores:
      Current:       {current_score:.4f}
      Optimal:       {optimal_score:.4f}
      Improvement:   {improvement:+.2f}%

    F0 Shift:        {(optimal_f0 - 7.60)*1000:.1f} mHz
                     ({(optimal_f0/7.60 - 1)*100:+.3f}%)

    Total peaks:     {len(freqs):,}
    """

    ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Optimal F0 Analysis: Combined GED Dataset\n'
                 f'({len(freqs):,} peaks from PhySF + MPENG + Emotions)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("COMPREHENSIVE STATISTICAL VALIDATION OF COMBINED GED DATASET")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load all datasets
    all_data = []
    dataset_stats = {}

    for name, path in DATASETS.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['dataset'] = name
            all_data.append(df)
            n_sessions = df['session'].nunique() if 'session' in df.columns else 'N/A'
            dataset_stats[name] = {'peaks': len(df), 'sessions': n_sessions}
            print(f"Loaded {name}: {len(df):,} peaks from {n_sessions} sessions")
        else:
            print(f"Warning: {path} not found")

    if not all_data:
        print("No data loaded!")
        return

    combined = pd.concat(all_data, ignore_index=True)
    freqs = combined['frequency'].values
    print(f"\nTotal: {len(freqs):,} peaks")
    print(f"Frequency range: {freqs.min():.2f} - {freqs.max():.2f} Hz")

    # =========================================================================
    # 1. POSITION ENRICHMENT ANALYSIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("1. POSITION ENRICHMENT ANALYSIS")
    print("=" * 70)

    lattice_coords = compute_lattice_coordinate(freqs)

    # Compute enrichment
    enrichment = compute_enrichment_at_positions(lattice_coords, EXTENDED_OFFSETS, window=0.05)

    # Bootstrap confidence intervals
    print("\nComputing bootstrap 95% CIs (1000 iterations)...")
    ci_results = bootstrap_enrichment_ci(lattice_coords, EXTENDED_OFFSETS,
                                          window=0.05, n_bootstrap=1000)

    # Build results table
    enrichment_rows = []
    position_order = ['boundary', 'noble_6', 'noble_5', 'noble_4', 'noble_3', 'noble_2',
                      'attractor', 'noble_1', 'inv_noble_3', 'inv_noble_4', 'inv_noble_5', 'inv_noble_6']

    print("\nPosition Enrichment (relative to uniform distribution):")
    print("-" * 70)
    print(f"{'Position':<15} {'Offset':<8} {'N Peaks':<12} {'Enrichment':<12} {'95% CI':<20}")
    print("-" * 70)

    for ptype in position_order:
        e = enrichment[ptype]
        ci = ci_results[ptype]
        print(f"{ptype:<15} {e['offset']:.4f}   {e['n_peaks']:<12,} {e['enrichment_pct']:>+8.1f}%    "
              f"[{ci['ci_lower']:+.1f}%, {ci['ci_upper']:+.1f}%]")

        enrichment_rows.append({
            'position_type': ptype,
            'offset': e['offset'],
            'n_peaks': e['n_peaks'],
            'enrichment_pct': e['enrichment_pct'],
            'ci_lower': ci['ci_lower'],
            'ci_upper': ci['ci_upper'],
        })

    enrichment_df = pd.DataFrame(enrichment_rows)
    enrichment_df.to_csv(f'{OUTPUT_DIR}/aggregate_enrichment_table.csv', index=False)
    print(f"\nSaved: {OUTPUT_DIR}/aggregate_enrichment_table.csv")

    # =========================================================================
    # 2. SESSION-LEVEL CONSISTENCY
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. SESSION-LEVEL CONSISTENCY ANALYSIS")
    print("=" * 70)

    consistency_results = {}

    # Per-dataset consistency
    for name in DATASETS.keys():
        dataset_df = combined[combined['dataset'] == name]
        result = compute_session_consistency(dataset_df)
        if result:
            consistency_results[name] = result
            print(f"\n{name}:")
            print(f"  Sessions analyzed: {result['n_sessions']}")
            print(f"  Attractor > Boundary: {result['pct_attractor_gt_boundary']:.1f}%")
            print(f"  Correct ordering:     {result['pct_correct_ordering']:.1f}%")
            print(f"  Cohen's d:            {result['cohens_d']:.3f}")

    # Combined consistency
    combined_result = compute_session_consistency(combined)
    if combined_result:
        consistency_results['Combined'] = combined_result
        print(f"\nCOMBINED (all datasets):")
        print(f"  Sessions analyzed: {combined_result['n_sessions']}")
        print(f"  Attractor > Boundary: {combined_result['pct_attractor_gt_boundary']:.1f}%")
        print(f"  Correct ordering:     {combined_result['pct_correct_ordering']:.1f}%")
        print(f"  Cohen's d:            {combined_result['cohens_d']:.3f}")

        print(f"\n  (Paper reference: 87.4% attractor > boundary)")

    # Save consistency summary
    consistency_summary = []
    for name, result in consistency_results.items():
        consistency_summary.append({
            'dataset': name,
            'n_sessions': result['n_sessions'],
            'pct_attractor_gt_boundary': result['pct_attractor_gt_boundary'],
            'pct_correct_ordering': result['pct_correct_ordering'],
            'cohens_d': result['cohens_d'],
            'mean_attractor_enrich': result['mean_attractor_enrich'],
            'mean_boundary_enrich': result['mean_boundary_enrich'],
        })

    consistency_df = pd.DataFrame(consistency_summary)
    consistency_df.to_csv(f'{OUTPUT_DIR}/aggregate_session_consistency.csv', index=False)
    print(f"\nSaved: {OUTPUT_DIR}/aggregate_session_consistency.csv")

    # =========================================================================
    # 3. F0 OPTIMIZATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("3. F0 OPTIMIZATION SWEEP")
    print("=" * 70)

    print("\nSweeping F0 from 6.5 to 8.5 Hz in 5 mHz steps...")
    f0_results, optimal_f0, local_maxima = sweep_f0_optimization(freqs)

    current_score, _ = compute_alignment_score(freqs, 7.60, EXTENDED_OFFSETS)
    optimal_score, _ = compute_alignment_score(freqs, optimal_f0, EXTENDED_OFFSETS)
    improvement = (optimal_score - current_score) / current_score * 100

    print(f"\nResults:")
    print(f"  Current F0:     7.6000 Hz (score: {current_score:.4f})")
    print(f"  Optimal F0:     {optimal_f0:.4f} Hz (score: {optimal_score:.4f})")
    print(f"  Shift:          {(optimal_f0 - 7.60)*1000:+.1f} mHz ({(optimal_f0/7.60 - 1)*100:+.3f}%)")
    print(f"  Improvement:    {improvement:+.2f}%")

    print(f"\nTop 5 local maxima:")
    for i, (f0, score) in enumerate(local_maxima[:5], 1):
        marker = " <-- OPTIMAL" if f0 == optimal_f0 else ""
        marker = " <-- CURRENT" if abs(f0 - 7.60) < 0.01 else marker
        print(f"  {i}. F0 = {f0:.4f} Hz, score = {score:.4f}{marker}")

    # Generate visualization
    plot_f0_optimization(f0_results, freqs, optimal_f0,
                         f'{OUTPUT_DIR}/aggregate_f0_optimization.png')

    # Save F0 results
    f0_summary = {
        'current_f0': 7.60,
        'optimal_f0': optimal_f0,
        'current_score': current_score,
        'optimal_score': optimal_score,
        'improvement_pct': improvement,
        'shift_mhz': (optimal_f0 - 7.60) * 1000,
        'n_peaks': len(freqs),
    }
    pd.DataFrame([f0_summary]).to_csv(f'{OUTPUT_DIR}/aggregate_f0_summary.csv', index=False)
    print(f"\nSaved: {OUTPUT_DIR}/aggregate_f0_summary.csv")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"""
Dataset: Combined truly continuous GED peaks
  - PhySF:    {dataset_stats.get('PhySF', {}).get('peaks', 'N/A'):,} peaks, {dataset_stats.get('PhySF', {}).get('sessions', 'N/A')} sessions
  - MPENG:    {dataset_stats.get('MPENG', {}).get('peaks', 'N/A'):,} peaks, {dataset_stats.get('MPENG', {}).get('sessions', 'N/A')} sessions
  - Emotions: {dataset_stats.get('Emotions', {}).get('peaks', 'N/A'):,} peaks, {dataset_stats.get('Emotions', {}).get('sessions', 'N/A')} sessions
  - TOTAL:    {len(freqs):,} peaks

1. POSITION ENRICHMENT:
   - Strongest enrichment: {enrichment_df.iloc[enrichment_df['enrichment_pct'].idxmax()]['position_type']} ({enrichment_df['enrichment_pct'].max():+.1f}%)
   - Attractor enrichment: {enrichment['attractor']['enrichment_pct']:+.1f}%
   - Boundary enrichment:  {enrichment['boundary']['enrichment_pct']:+.1f}%

2. SESSION-LEVEL CONSISTENCY:
   - Attractor > Boundary: {combined_result['pct_attractor_gt_boundary']:.1f}% of sessions
   - (Paper reference: 87.4%)

3. F0 OPTIMIZATION:
   - Optimal F0: {optimal_f0:.4f} Hz
   - Current F0: 7.6000 Hz
   - Difference: {(optimal_f0 - 7.60)*1000:+.1f} mHz ({improvement:+.2f}% improvement)

Output files in: {OUTPUT_DIR}/
""")

    print("Done!")


if __name__ == '__main__':
    main()

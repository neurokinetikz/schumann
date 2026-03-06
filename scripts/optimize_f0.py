#!/usr/bin/env python3
"""
Find optimal F0 that maximizes alignment between GED peak distribution
and φⁿ position predictions in log-φ space.

Algorithm:
1. Convert all peaks to fractional φ-exponent space (n mod 1) for each candidate F0
2. Position predictions are fixed fractional offsets (0.0, 0.0557, 0.0902, etc.)
3. Sweep F0 from 7.0 to 8.2 Hz
4. Compute alignment score = sum of KDE density at each prediction offset
5. Find F0 that maximizes alignment
"""

import sys
sys.path.insert(0, './lib')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Constants
PHI = 1.618033988749895
PHI_INV = 1.0 / PHI

# Extended position offsets (fractional φ-exponent within each octave)
POSITION_OFFSETS = {
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

# Position colors for visualization
POSITION_COLORS = {
    'boundary': '#E74C3C',
    'noble_6': '#9B59B6',
    'noble_5': '#8E44AD',
    'noble_4': '#3498DB',
    'noble_3': '#1ABC9C',
    'noble_2': '#27AE60',
    'attractor': '#F39C12',
    'noble_1': '#2ECC71',
    'inv_noble_3': '#16A085',
    'inv_noble_4': '#2980B9',
    'inv_noble_5': '#7D3C98',
    'inv_noble_6': '#C0392B',
}


def compute_alignment_score(freqs, f0_candidate, position_offsets, kde_bw=0.02):
    """
    Compute alignment between peak distribution and φⁿ predictions for a given F0.

    Parameters
    ----------
    freqs : np.ndarray
        Peak frequencies (Hz)
    f0_candidate : float
        Candidate base frequency (Hz)
    position_offsets : dict
        Position type -> fractional offset (0-1)
    kde_bw : float
        KDE bandwidth

    Returns
    -------
    score : float
        Alignment score (higher = better alignment)
    details : dict
        Per-position alignment metrics
    """
    # Convert all peak frequencies to φ-exponent space with candidate F0
    phi_exps = np.log(freqs / f0_candidate) / np.log(PHI)

    # Extract fractional part (position within octave)
    fractional = phi_exps % 1.0

    # Create KDE for density estimation
    kde = gaussian_kde(fractional, bw_method=kde_bw)

    # Measure alignment: sum of KDE density at each prediction offset
    alignment_score = 0
    details = {}

    for ptype, offset in position_offsets.items():
        # Handle boundary at both 0 and 1
        if offset == 0.0:
            # Average density at 0 and 1 (they're the same position)
            density = (kde(0.001)[0] + kde(0.999)[0]) / 2
        else:
            density = kde(offset)[0]

        alignment_score += density
        details[ptype] = {
            'offset': offset,
            'density': density
        }

    return alignment_score, details, kde, fractional


def sweep_f0_optimization(freqs, f0_range=(6.5, 8.5), step=0.005):
    """
    Sweep F0 values to find optimal alignment.

    Returns
    -------
    results : pd.DataFrame
        F0 values with alignment scores
    optimal_f0 : float
        F0 with highest alignment score (excluding boundary effects)
    local_maxima : list
        List of (f0, score) for all local maxima
    """
    from scipy.signal import find_peaks

    f0_values = np.arange(f0_range[0], f0_range[1] + step, step)

    scores = []
    for f0 in f0_values:
        score, details, _, _ = compute_alignment_score(freqs, f0, POSITION_OFFSETS)
        row = {
            'f0': f0,
            'alignment_score': score,
        }
        for ptype, info in details.items():
            row[f'{ptype}_density'] = info['density']
        scores.append(row)

    results = pd.DataFrame(scores)

    # Find local maxima (excluding boundary effects)
    alignment_scores = results['alignment_score'].values
    peak_idx, props = find_peaks(alignment_scores, distance=10, prominence=0.05)

    local_maxima = [(results.iloc[i]['f0'], results.iloc[i]['alignment_score'])
                    for i in peak_idx]
    local_maxima.sort(key=lambda x: x[1], reverse=True)

    # Choose the best local maximum that's away from edges
    # Skip the very first/last 5% of the range
    edge_margin = 0.05 * (f0_range[1] - f0_range[0])
    interior_maxima = [(f0, score) for f0, score in local_maxima
                       if f0 > f0_range[0] + edge_margin and f0 < f0_range[1] - edge_margin]

    if interior_maxima:
        optimal_f0 = interior_maxima[0][0]
    else:
        # Fallback to global max
        optimal_f0 = results.loc[results['alignment_score'].idxmax(), 'f0']

    return results, optimal_f0, local_maxima


def plot_f0_optimization(results, freqs, optimal_f0, output_path):
    """
    Multi-panel visualization of F0 optimization.

    Panel 1: Alignment score vs F0 (find the peak)
    Panel 2: Fractional distribution at current F0=7.60 vs optimal
    Panel 3: Per-position density comparison
    Panel 4: Summary text
    """
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

    # Mark the optimal F0
    opt_score = results[np.abs(results['f0'] - optimal_f0) < 0.003]['alignment_score']
    if len(opt_score) > 0:
        opt_score = opt_score.values[0]
    else:
        opt_score, _, _, _ = compute_alignment_score(freqs, optimal_f0, POSITION_OFFSETS)
    ax.scatter([optimal_f0], [opt_score], color='green', s=150, zorder=5, marker='*')

    # Also mark F0=7.60
    current_score = results[np.abs(results['f0'] - 7.60) < 0.003]['alignment_score']
    if len(current_score) > 0:
        current_score = current_score.values[0]
        ax.scatter([7.60], [current_score], color='red', s=100, zorder=5, marker='o')

    # Panel 2: Fractional distribution comparison
    ax = axes[0, 1]

    # Current F0=7.60
    frac_current = (np.log(freqs / 7.60) / np.log(PHI)) % 1.0
    ax.hist(frac_current, bins=100, range=(0, 1), alpha=0.5, label='F0=7.60', color='red', density=True)

    # Optimal F0
    frac_optimal = (np.log(freqs / optimal_f0) / np.log(PHI)) % 1.0
    ax.hist(frac_optimal, bins=100, range=(0, 1), alpha=0.5, label=f'F0={optimal_f0:.4f}', color='green', density=True)

    # Prediction lines
    for ptype, offset in POSITION_OFFSETS.items():
        ax.axvline(offset, color='gray', linestyle=':', alpha=0.5, linewidth=1)

    ax.set_xlabel('Fractional φ-exponent (mod 1)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Distribution Shift: Current vs Optimal F0', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)

    # Panel 3: Per-position density at optimal
    ax = axes[1, 0]
    position_order = ['boundary', 'noble_6', 'noble_5', 'noble_4', 'noble_3', 'noble_2',
                      'attractor', 'noble_1', 'inv_noble_3', 'inv_noble_4', 'inv_noble_5', 'inv_noble_6']

    # Get densities at current and optimal F0
    _, details_current, _, _ = compute_alignment_score(freqs, 7.60, POSITION_OFFSETS)
    _, details_optimal, _, _ = compute_alignment_score(freqs, optimal_f0, POSITION_OFFSETS)

    x = np.arange(len(position_order))
    width = 0.35

    densities_current = [details_current[p]['density'] for p in position_order]
    densities_optimal = [details_optimal[p]['density'] for p in position_order]

    bars1 = ax.bar(x - width/2, densities_current, width, label='F0=7.60', color='red', alpha=0.7)
    bars2 = ax.bar(x + width/2, densities_optimal, width, label=f'F0={optimal_f0:.4f}', color='green', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([p.replace('_', '\n').replace('inv\n', 'inv_') for p in position_order],
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('KDE Density', fontsize=11)
    ax.set_title('Per-Position Alignment Comparison', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)

    # Panel 4: Summary text
    ax = axes[1, 1]
    ax.axis('off')

    current_score = results[np.abs(results['f0'] - 7.60) < 0.003]['alignment_score'].values
    if len(current_score) > 0:
        current_score = current_score[0]
    else:
        current_score = compute_alignment_score(freqs, 7.60, POSITION_OFFSETS)[0]

    optimal_score = results['alignment_score'].max()
    improvement = (optimal_score - current_score) / current_score * 100

    # Find secondary peaks in the optimization curve
    from scipy.signal import find_peaks
    peak_idx, _ = find_peaks(results['alignment_score'].values, distance=20)
    peak_f0s = results.iloc[peak_idx]['f0'].values
    peak_scores = results.iloc[peak_idx]['alignment_score'].values

    # Sort by score
    sort_idx = np.argsort(peak_scores)[::-1]
    top_peaks = [(peak_f0s[i], peak_scores[i]) for i in sort_idx[:5]]

    peaks_str = '\n'.join([f"    {f0:.4f} Hz (score: {score:.2f})" for f0, score in top_peaks])

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

    ──────────────────────────────────────
    Top 5 Local Maxima:
{peaks_str}

    ──────────────────────────────────────
    Position Offset Reference:
      boundary:    0.000 (φ⁰)
      noble_6:     {PHI_INV**6:.4f} (φ⁻⁶)
      noble_5:     {PHI_INV**5:.4f} (φ⁻⁵)
      noble_4:     {PHI_INV**4:.4f} (φ⁻⁴)
      noble_3:     {PHI_INV**3:.4f} (φ⁻³)
      noble_2:     {PHI_INV**2:.4f} (φ⁻²)
      attractor:   0.500
      noble_1:     {PHI_INV**1:.4f} (φ⁻¹)
    """

    ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Optimal F0 Fit: GED Peak Distribution vs φⁿ Predictions\n'
                 f'({len(freqs):,} peaks from continuous GED sweep)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_kde_comparison(freqs, optimal_f0, output_path):
    """
    Detailed KDE comparison showing how the distribution shifts with F0.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: KDE overlay with prediction lines
    ax = axes[0]

    # Compute KDEs
    frac_current = (np.log(freqs / 7.60) / np.log(PHI)) % 1.0
    frac_optimal = (np.log(freqs / optimal_f0) / np.log(PHI)) % 1.0

    kde_current = gaussian_kde(frac_current, bw_method=0.02)
    kde_optimal = gaussian_kde(frac_optimal, bw_method=0.02)

    x = np.linspace(0, 1, 500)
    y_current = kde_current(x)
    y_optimal = kde_optimal(x)

    ax.fill_between(x, y_current, alpha=0.3, color='red', label='F0=7.60')
    ax.plot(x, y_current, color='red', linewidth=2)
    ax.fill_between(x, y_optimal, alpha=0.3, color='green', label=f'F0={optimal_f0:.4f}')
    ax.plot(x, y_optimal, color='green', linewidth=2)

    # Prediction lines with labels
    for ptype, offset in POSITION_OFFSETS.items():
        color = POSITION_COLORS.get(ptype, 'gray')
        ax.axvline(offset, color=color, linestyle='--', alpha=0.7, linewidth=1.5)

    ax.set_xlabel('Fractional φ-exponent (mod 1)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('KDE Comparison: Current vs Optimal F0', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1)

    # Panel 2: Difference plot (optimal - current)
    ax = axes[1]
    y_diff = y_optimal - y_current
    ax.fill_between(x, y_diff, where=(y_diff >= 0), alpha=0.5, color='green', label='Optimal > Current')
    ax.fill_between(x, y_diff, where=(y_diff < 0), alpha=0.5, color='red', label='Current > Optimal')
    ax.axhline(0, color='black', linewidth=1)

    # Prediction lines
    for ptype, offset in POSITION_OFFSETS.items():
        color = POSITION_COLORS.get(ptype, 'gray')
        ax.axvline(offset, color=color, linestyle='--', alpha=0.5, linewidth=1)

    ax.set_xlabel('Fractional φ-exponent (mod 1)', fontsize=12)
    ax.set_ylabel('Density Difference', fontsize=12)
    ax.set_title(f'KDE Difference (F0={optimal_f0:.4f} - F0=7.60)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    # Parse command-line arguments
    # Usage: python optimize_f0.py [peaks_path] [output_dir]
    if len(sys.argv) >= 3:
        peaks_path = sys.argv[1]
        output_dir = sys.argv[2]
    elif len(sys.argv) == 2:
        peaks_path = sys.argv[1]
        # Derive output dir from peaks path
        base_dir = os.path.dirname(os.path.dirname(peaks_path))
        output_dir = os.path.join(base_dir, 'f0_optimization')
    else:
        # Default to PhySF paths
        peaks_path = 'exports_peak_distribution/physf_ged/continuous_v2/ged_peaks_continuous.csv'
        output_dir = 'exports_peak_distribution/physf_ged/f0_optimization'

    print(f"Peaks file: {peaks_path}")
    print(f"Output dir: {output_dir}")

    if not os.path.exists(peaks_path):
        print(f"ERROR: Peaks file not found: {peaks_path}")
        print("Run continuous GED detection first.")
        return

    peaks_df = pd.read_csv(peaks_path)
    freqs = peaks_df['frequency'].values
    print(f"Loaded {len(freqs):,} peaks")
    print(f"Frequency range: {freqs.min():.2f} - {freqs.max():.2f} Hz")

    # Run F0 sweep (extended range to find true optimal)
    print("\nSweeping F0 from 6.5 to 8.5 Hz in 5 mHz steps...")
    results, optimal_f0, local_maxima = sweep_f0_optimization(
        freqs, f0_range=(6.5, 8.5), step=0.005
    )

    print(f"\n{'='*50}")
    print("RESULTS")
    print('='*50)
    print(f"Best internal maximum F0: {optimal_f0:.4f} Hz")
    print(f"Current F0:               7.6000 Hz")
    print(f"Difference:               {(optimal_f0 - 7.60)*1000:.1f} mHz "
          f"({(optimal_f0/7.60 - 1)*100:+.3f}%)")

    print(f"\nTop 10 local maxima:")
    for i, (f0, score) in enumerate(local_maxima[:10], 1):
        marker = " <-- BEST INTERNAL" if f0 == optimal_f0 else ""
        marker = " <-- CURRENT" if abs(f0 - 7.60) < 0.01 else marker
        print(f"  {i:2d}. F0 = {f0:.4f} Hz, score = {score:.4f}{marker}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save sweep results
    results.to_csv(f'{output_dir}/f0_sweep_results.csv', index=False)
    print(f"\nSaved: {output_dir}/f0_sweep_results.csv")

    # Generate visualizations
    plot_f0_optimization(results, freqs, optimal_f0, f'{output_dir}/f0_optimization.png')
    plot_kde_comparison(freqs, optimal_f0, f'{output_dir}/f0_kde_comparison.png')

    # Save summary
    summary = {
        'current_f0': 7.60,
        'optimal_f0': optimal_f0,
        'shift_mhz': (optimal_f0 - 7.60) * 1000,
        'shift_percent': (optimal_f0 / 7.60 - 1) * 100,
        'n_peaks': len(freqs),
    }

    # Add alignment scores
    current_score, current_details, _, _ = compute_alignment_score(freqs, 7.60, POSITION_OFFSETS)
    optimal_score, optimal_details, _, _ = compute_alignment_score(freqs, optimal_f0, POSITION_OFFSETS)

    summary['current_alignment_score'] = current_score
    summary['optimal_alignment_score'] = optimal_score
    summary['improvement_percent'] = (optimal_score - current_score) / current_score * 100

    pd.DataFrame([summary]).to_csv(f'{output_dir}/f0_optimization_summary.csv', index=False)
    print(f"Saved: {output_dir}/f0_optimization_summary.csv")

    print(f"\nDone! Results saved to {output_dir}/")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Comprehensive φⁿ Position Comparison Report
============================================

Compares detected gedBounds boundaries against ALL φⁿ position types
(boundaries, nobles, attractors, inverse nobles) for each dataset
and the aggregate.

Output:
- comprehensive_phi_comparison.csv (all detected boundaries vs all positions)
- phi_comparison_summary.txt (human-readable report)
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy import signal
from pathlib import Path

# Add lib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))

# ============================================================================
# φⁿ Position Framework
# ============================================================================

PHI = 1.6180339887
PHI_INV = 1.0 / PHI
F0 = 7.60

# Position offsets within each φ-octave
POSITION_OFFSETS = {
    'boundary':    0.000,
    'noble_4':     PHI_INV ** 4,      # 0.1459
    'noble_3':     PHI_INV ** 3,      # 0.2361
    'noble_2':     1 - PHI_INV,       # 0.3820
    'attractor':   0.500,
    'noble_1':     PHI_INV,           # 0.6180
    'inv_noble_3': 1 - PHI_INV ** 3,  # 0.7639
    'inv_noble_4': 1 - PHI_INV ** 4,  # 0.8541
}

# Band definitions
BANDS = {
    'theta':     (-1, 0),
    'alpha':     (0, 1),
    'beta_low':  (1, 2),
    'beta_high': (2, 3),
    'gamma':     (3, 4),
}


def generate_all_phi_positions(freq_range=(4.5, 45.0)):
    """Generate all φⁿ positions within frequency range."""
    positions = []

    for band_name, (n_start, n_end) in BANDS.items():
        for pos_name, offset in POSITION_OFFSETS.items():
            n = n_start + offset
            freq = F0 * (PHI ** n)

            if freq_range[0] <= freq <= freq_range[1]:
                positions.append({
                    'n': n,
                    'frequency': freq,
                    'position_type': pos_name,
                    'band': band_name,
                    'label': f"{band_name}_{pos_name}"
                })

    return pd.DataFrame(positions).sort_values('frequency').reset_index(drop=True)


def find_boundaries_from_similarity(frequencies, similarities, prominence_pct=5,
                                     min_distance_hz=2.0, smooth_window=1):
    """Find boundaries as local minima in similarity curve."""
    # Smooth
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        sim_smooth = np.convolve(similarities, kernel, mode='same')
    else:
        sim_smooth = similarities

    # Prominence threshold
    sim_range = np.max(sim_smooth) - np.min(sim_smooth)
    min_prominence = sim_range * (prominence_pct / 100)

    # Convert to samples
    freq_step = frequencies[1] - frequencies[0] if len(frequencies) > 1 else 0.5
    min_distance_samples = max(1, int(min_distance_hz / freq_step))

    # Find minima (invert to find as peaks)
    inverted = -sim_smooth
    peaks, props = signal.find_peaks(inverted, prominence=min_prominence,
                                      distance=min_distance_samples)

    prominences = props.get('prominences', np.zeros(len(peaks)))

    return frequencies[peaks], prominences


def compare_to_phi_positions(detected_freqs, phi_positions_df):
    """Compare each detected frequency to all φⁿ positions."""
    results = []

    for det_freq in detected_freqs:
        # Find nearest φⁿ position
        phi_positions_df['distance'] = np.abs(phi_positions_df['frequency'] - det_freq)
        nearest_idx = phi_positions_df['distance'].idxmin()
        nearest = phi_positions_df.loc[nearest_idx]

        results.append({
            'detected_freq': det_freq,
            'nearest_phi_label': nearest['label'],
            'nearest_phi_freq': nearest['frequency'],
            'nearest_phi_n': nearest['n'],
            'distance_hz': nearest['distance'],
            'position_type': nearest['position_type'],
            'band': nearest['band'],
            'match_0.5hz': nearest['distance'] <= 0.5,
            'match_1.0hz': nearest['distance'] <= 1.0,
        })

    return pd.DataFrame(results)


def process_dataset(npz_path, dataset_name, phi_positions_df, sensitivities):
    """Process a single dataset's similarity curves."""
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name}")
    print(f"{'='*60}")

    # Load data
    data = np.load(npz_path, allow_pickle=True)
    frequencies = data['frequencies']
    similarities = data['similarities']
    n_sessions = similarities.shape[0]

    print(f"  Sessions: {n_sessions}")
    print(f"  Frequencies: {frequencies[0]:.2f} - {frequencies[-1]:.2f} Hz")

    # Compute mean similarity (use nanmean to ignore invalid values)
    mean_sim = np.nanmean(similarities, axis=0)

    all_results = []

    for sens_name, prom_pct in sensitivities.items():
        # Detect boundaries
        detected, prominences = find_boundaries_from_similarity(
            frequencies, mean_sim,
            prominence_pct=prom_pct,
            min_distance_hz=2.0,
            smooth_window=1
        )

        if len(detected) == 0:
            print(f"  {sens_name} (prom={prom_pct}%): No boundaries detected")
            continue

        print(f"  {sens_name} (prom={prom_pct}%): {len(detected)} boundaries")

        # Compare to φⁿ positions
        comparison = compare_to_phi_positions(detected, phi_positions_df)
        comparison['dataset'] = dataset_name
        comparison['sensitivity'] = sens_name
        comparison['prominence_pct'] = prom_pct
        comparison['n_sessions'] = n_sessions

        all_results.append(comparison)

        # Print matches
        matches = comparison[comparison['match_0.5hz']]
        if len(matches) > 0:
            for _, row in matches.iterrows():
                print(f"    ✓ {row['detected_freq']:.2f} Hz -> {row['nearest_phi_label']} "
                      f"({row['nearest_phi_freq']:.2f} Hz, Δ={row['distance_hz']:.2f} Hz)")

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()


def process_aggregate(csv_path, phi_positions_df, sensitivities):
    """Process the aggregate grand similarity curve."""
    print(f"\n{'='*60}")
    print("Processing AGGREGATE")
    print(f"{'='*60}")

    df = pd.read_csv(csv_path)
    frequencies = df['frequency'].values
    mean_sim = df['similarity_mean'].values

    print(f"  Frequencies: {frequencies[0]:.2f} - {frequencies[-1]:.2f} Hz")

    all_results = []

    for sens_name, prom_pct in sensitivities.items():
        # Detect boundaries
        detected, prominences = find_boundaries_from_similarity(
            frequencies, mean_sim,
            prominence_pct=prom_pct,
            min_distance_hz=2.0,
            smooth_window=1
        )

        if len(detected) == 0:
            print(f"  {sens_name} (prom={prom_pct}%): No boundaries detected")
            continue

        print(f"  {sens_name} (prom={prom_pct}%): {len(detected)} boundaries")

        # Compare to φⁿ positions
        comparison = compare_to_phi_positions(detected, phi_positions_df)
        comparison['dataset'] = 'AGGREGATE'
        comparison['sensitivity'] = sens_name
        comparison['prominence_pct'] = prom_pct
        comparison['n_sessions'] = 2950

        all_results.append(comparison)

        # Print matches
        matches = comparison[comparison['match_0.5hz']]
        if len(matches) > 0:
            for _, row in matches.iterrows():
                print(f"    ✓ {row['detected_freq']:.2f} Hz -> {row['nearest_phi_label']} "
                      f"({row['nearest_phi_freq']:.2f} Hz, Δ={row['distance_hz']:.2f} Hz)")

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()


def generate_summary_report(results_df, phi_positions_df, output_path):
    """Generate human-readable summary report."""
    lines = []
    lines.append("=" * 70)
    lines.append("COMPREHENSIVE φⁿ POSITION COMPARISON REPORT")
    lines.append("gedBounds Covariance-Based Boundary Detection")
    lines.append("=" * 70)
    lines.append("")

    # φⁿ positions reference
    lines.append("REFERENCE: All φⁿ Positions (4.5-45 Hz)")
    lines.append("-" * 50)
    for _, row in phi_positions_df.iterrows():
        lines.append(f"  {row['frequency']:6.2f} Hz | n={row['n']:5.3f} | {row['label']}")
    lines.append("")

    # Summary by dataset
    lines.append("RESULTS BY DATASET")
    lines.append("=" * 70)

    for dataset in results_df['dataset'].unique():
        ds_data = results_df[results_df['dataset'] == dataset]
        n_sessions = ds_data['n_sessions'].iloc[0] if len(ds_data) > 0 else 0

        lines.append(f"\n{dataset} ({n_sessions} sessions)")
        lines.append("-" * 40)

        for sens in ['HIGH', 'MEDIUM', 'LOW']:
            sens_data = ds_data[ds_data['sensitivity'] == sens]
            if len(sens_data) == 0:
                continue

            prom = sens_data['prominence_pct'].iloc[0]
            lines.append(f"\n  {sens} sensitivity (prominence={prom}%):")

            for _, row in sens_data.iterrows():
                match_marker = "✓" if row['match_0.5hz'] else " "
                lines.append(f"    {match_marker} {row['detected_freq']:6.2f} Hz -> "
                           f"{row['nearest_phi_label']:20s} ({row['nearest_phi_freq']:5.2f} Hz) "
                           f"Δ={row['distance_hz']:.2f} Hz")

    # Summary statistics
    lines.append("\n" + "=" * 70)
    lines.append("SUMMARY STATISTICS")
    lines.append("=" * 70)

    total_detected = len(results_df)
    matches_0_5 = len(results_df[results_df['match_0.5hz']])
    matches_1_0 = len(results_df[results_df['match_1.0hz']])

    lines.append(f"\nTotal boundaries detected: {total_detected}")
    lines.append(f"Matches within 0.5 Hz: {matches_0_5} ({100*matches_0_5/total_detected:.1f}%)")
    lines.append(f"Matches within 1.0 Hz: {matches_1_0} ({100*matches_1_0/total_detected:.1f}%)")

    # Best matches
    lines.append("\nBest matches (closest to φⁿ positions):")
    best = results_df.nsmallest(10, 'distance_hz')
    for _, row in best.iterrows():
        lines.append(f"  {row['detected_freq']:6.2f} Hz -> {row['nearest_phi_label']:20s} "
                   f"Δ={row['distance_hz']:.3f} Hz ({row['dataset']}, {row['sensitivity']})")

    # By position type
    lines.append("\nMatches by position type:")
    for pos_type in ['boundary', 'attractor', 'noble_1', 'noble_2', 'noble_3', 'noble_4',
                     'inv_noble_3', 'inv_noble_4']:
        pos_matches = results_df[(results_df['position_type'] == pos_type) &
                                  results_df['match_0.5hz']]
        if len(pos_matches) > 0:
            lines.append(f"  {pos_type:15s}: {len(pos_matches)} matches")

    report = "\n".join(lines)

    with open(output_path, 'w') as f:
        f.write(report)

    print(report)
    return report


def main():
    """Run comprehensive φⁿ comparison."""

    # Output directory
    output_dir = 'exports_peak_distribution/true_gedbounds'
    os.makedirs(output_dir, exist_ok=True)

    # Generate all φⁿ positions
    print("Generating φⁿ position reference table...")
    phi_positions = generate_all_phi_positions(freq_range=(4.5, 45.0))
    print(f"  {len(phi_positions)} positions in range")

    # Save φⁿ positions reference
    phi_positions.to_csv(os.path.join(output_dir, 'phi_positions_reference.csv'), index=False)

    # Sensitivity levels
    sensitivities = {
        'HIGH': 3,
        'MEDIUM': 5,
        'LOW': 10,
    }

    # Process each dataset
    all_results = []

    datasets = {
        'physf': 'similarity_curves_physf.npz',
        'mpeng': 'similarity_curves_mpeng.npz',
        'vep': 'similarity_curves_vep.npz',
        'emotions': 'similarity_curves_emotions.npz',
    }

    for dataset_name, filename in datasets.items():
        npz_path = os.path.join(output_dir, filename)
        if os.path.exists(npz_path):
            results = process_dataset(npz_path, dataset_name, phi_positions, sensitivities)
            if len(results) > 0:
                all_results.append(results)
        else:
            print(f"\nSkipping {dataset_name}: {filename} not found")

    # Process aggregate
    agg_path = os.path.join(output_dir, 'grand_similarity_curve.csv')
    if os.path.exists(agg_path):
        agg_results = process_aggregate(agg_path, phi_positions, sensitivities)
        if len(agg_results) > 0:
            all_results.append(agg_results)

    # Combine all results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)

        # Save comprehensive CSV
        csv_path = os.path.join(output_dir, 'comprehensive_phi_comparison.csv')
        combined.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")

        # Generate summary report
        report_path = os.path.join(output_dir, 'phi_comparison_summary.txt')
        generate_summary_report(combined, phi_positions, report_path)
        print(f"Saved: {report_path}")
    else:
        print("\nNo results to save!")


if __name__ == '__main__':
    main()

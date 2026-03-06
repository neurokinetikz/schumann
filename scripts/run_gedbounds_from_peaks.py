#!/usr/bin/env python3
"""
gedBounds-style Boundary Detection from Existing GED Peaks
==========================================================

Identifies empirical frequency band boundaries using the distribution
of ~1.58M GED peaks already detected across PhySF, MPENG, and EEGEmotions-27.

The key insight: band boundaries should have fewer peaks (depletion),
so density minima naturally indicate boundaries.

Input: Aggregated GED peaks from truly_continuous detection
Output: Empirical boundaries and phi^n validation

Usage:
    python scripts/run_gedbounds_from_peaks.py
"""

import sys
sys.path.insert(0, './lib')

import os
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
from typing import Tuple

# Import the new boundary detection module
from ged_bounds_clustering import (
    run_boundary_detection_pipeline,
    validate_boundaries_vs_phi,
    get_consensus_boundaries,
    generate_boundary_report,
    plot_boundary_detection,
    PHI_BOUNDARIES_DEFAULT
)

# =============================================================================
# Configuration
# =============================================================================

# Input paths for truly continuous GED peaks
PEAK_FILES = {
    'physf': 'exports_peak_distribution/physf_ged/truly_continuous/ged_peaks_truly_continuous.csv',
    'mpeng': 'exports_peak_distribution/mpeng_ged/truly_continuous/ged_peaks_truly_continuous.csv',
    'emotions': 'exports_peak_distribution/emotions_ged/truly_continuous/ged_peaks_truly_continuous.csv',
}

# Output directory
OUTPUT_DIR = 'exports_peak_distribution/gedbounds_analysis'

# Analysis parameters
FREQ_RANGE = (4.5, 45.0)  # Hz
KDE_BANDWIDTH = 0.3  # Hz
DENSITY_PROMINENCE = 0.05
CLUSTERING_MAX_K = 8


# =============================================================================
# Main Analysis
# =============================================================================

def load_frequencies_only() -> Tuple[np.ndarray, dict]:
    """Load only frequency arrays to minimize memory usage."""
    all_freqs = []
    counts = {}

    for dataset, path in PEAK_FILES.items():
        if os.path.exists(path):
            print(f"Loading {dataset}: {path}")
            # Only load frequency column
            df = pd.read_csv(path, usecols=['frequency'])
            freqs = df['frequency'].values
            all_freqs.append(freqs)
            counts[dataset] = len(freqs)
            print(f"  -> {len(freqs):,} peaks")
            del df  # Free memory
        else:
            print(f"WARNING: {path} not found, skipping {dataset}")

    if not all_freqs:
        raise FileNotFoundError("No peak files found!")

    combined = np.concatenate(all_freqs)
    print(f"\nTotal peaks loaded: {len(combined):,}")

    return combined, counts


def load_all_peaks() -> pd.DataFrame:
    """Load and combine all GED peaks from truly continuous detection."""
    all_dfs = []

    for dataset, path in PEAK_FILES.items():
        if os.path.exists(path):
            print(f"Loading {dataset}: {path}")
            df = pd.read_csv(path)
            df['dataset'] = dataset
            all_dfs.append(df)
            print(f"  -> {len(df):,} peaks")
        else:
            print(f"WARNING: {path} not found, skipping {dataset}")

    if not all_dfs:
        raise FileNotFoundError("No peak files found!")

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal peaks loaded: {len(combined):,}")

    return combined


def run_chunked_analysis(all_freqs: np.ndarray, dataset_counts: dict) -> dict:
    """Run memory-efficient chunked analysis.

    Strategy:
    1. KDE density on full data (memory efficient - just 1D array)
    2. Clustering on sampled subsets from each dataset
    3. Aggregate boundaries across methods
    """
    from ged_bounds_clustering import (
        compute_peak_density,
        find_density_minima,
        cluster_peaks_by_frequency,
        PHI_BOUNDARIES_DEFAULT
    )

    print("\n" + "=" * 60)
    print("Running Chunked Boundary Detection Pipeline")
    print("=" * 60)

    all_boundaries = []

    # -------------------------------------------------------------------------
    # Method 1: Histogram-based Density on FULL data (very memory efficient)
    # -------------------------------------------------------------------------
    print("\n[1/3] Computing histogram density on full dataset...")

    # Use histogram instead of KDE for memory efficiency with 1.5M points
    n_bins = 500
    mask = (all_freqs >= FREQ_RANGE[0]) & (all_freqs <= FREQ_RANGE[1])
    freqs_in_range = all_freqs[mask]
    print(f"  {len(freqs_in_range):,} peaks in frequency range")

    hist, bin_edges = np.histogram(freqs_in_range, bins=n_bins, range=FREQ_RANGE, density=True)
    freqs_eval = (bin_edges[:-1] + bin_edges[1:]) / 2
    density = hist / np.max(hist)  # Normalize

    # Smooth with Gaussian filter
    from scipy.ndimage import gaussian_filter1d
    density_smooth = gaussian_filter1d(density, sigma=2.0)

    density_bounds = find_density_minima(
        freqs_eval, density_smooth,
        prominence_threshold=DENSITY_PROMINENCE,
        min_distance_hz=2.0
    )
    print(f"  Density method found {len(density_bounds)} boundaries: {[f'{b:.1f}' for b in density_bounds]}")

    for b in density_bounds:
        all_boundaries.append({'frequency': b, 'method': 'density'})

    del freqs_in_range  # Free memory

    # -------------------------------------------------------------------------
    # Method 2 & 3: Clustering on SAMPLED data (per-dataset to save memory)
    # -------------------------------------------------------------------------
    SAMPLE_PER_DATASET = 10000  # Reduced for agglomerative memory efficiency

    for method in ['gmm', 'agglomerative']:
        print(f"\n[{'2' if method == 'gmm' else '3'}/3] Running {method.upper()} clustering per dataset...")

        method_boundaries = []

        # Process each dataset separately
        start_idx = 0
        for dataset, count in dataset_counts.items():
            end_idx = start_idx + count
            dataset_freqs = all_freqs[start_idx:end_idx]

            # Sample if needed
            if len(dataset_freqs) > SAMPLE_PER_DATASET:
                np.random.seed(42)
                sample_idx = np.random.choice(len(dataset_freqs), SAMPLE_PER_DATASET, replace=False)
                sample_freqs = dataset_freqs[sample_idx]
            else:
                sample_freqs = dataset_freqs

            # Create minimal DataFrame for clustering
            sample_df = pd.DataFrame({'frequency': sample_freqs})

            try:
                _, bounds = cluster_peaks_by_frequency(
                    sample_df,
                    freq_col='frequency',
                    method=method,
                    n_clusters='auto',
                    freq_range=FREQ_RANGE,
                    max_clusters=CLUSTERING_MAX_K
                )
                method_boundaries.extend(bounds)
                print(f"    {dataset}: {len(bounds)} boundaries")
            except Exception as e:
                print(f"    {dataset}: FAILED - {e}")

            del sample_df
            start_idx = end_idx

        # Aggregate boundaries (take median of close ones)
        if method_boundaries:
            aggregated = _aggregate_boundaries(method_boundaries, tolerance=1.5)
            print(f"  {method.upper()} aggregated: {[f'{b:.1f}' for b in aggregated]}")
            for b in aggregated:
                all_boundaries.append({'frequency': b, 'method': method})

    # -------------------------------------------------------------------------
    # Build results DataFrame
    # -------------------------------------------------------------------------
    results_df = pd.DataFrame(all_boundaries)

    if results_df.empty:
        results_df = pd.DataFrame(columns=['frequency', 'method', 'confidence', 'nearest_phi', 'distance_to_phi'])
    else:
        # Compute confidence (how many methods agree within 1 Hz)
        def count_agreements(freq, tolerance=1.0):
            return sum(1 for _, row in results_df.iterrows() if abs(row['frequency'] - freq) <= tolerance)

        results_df['confidence'] = results_df['frequency'].apply(count_agreements)

        # Find nearest phi boundary
        def find_nearest_phi(freq):
            distances = [(phi, abs(freq - phi)) for phi in PHI_BOUNDARIES_DEFAULT]
            nearest = min(distances, key=lambda x: x[1])
            return nearest[0], nearest[1]

        results_df['nearest_phi'] = results_df['frequency'].apply(lambda f: find_nearest_phi(f)[0])
        results_df['distance_to_phi'] = results_df['frequency'].apply(lambda f: find_nearest_phi(f)[1])
        results_df = results_df.sort_values('frequency').reset_index(drop=True)

    print(f"\nDetected {len(results_df)} boundary candidates across all methods")

    # Get consensus boundaries
    consensus = get_consensus_boundaries(results_df, min_confidence=2)
    if not consensus:
        print("No consensus boundaries (min_confidence=2), trying min_confidence=1")
        consensus = get_consensus_boundaries(results_df, min_confidence=1)

    print(f"\nConsensus boundaries ({len(consensus)}):")
    for b in consensus:
        nearest = min(PHI_BOUNDARIES_DEFAULT, key=lambda x: abs(x - b))
        dist = abs(b - nearest)
        print(f"  {b:.2f} Hz  (nearest phi^n: {nearest:.2f} Hz, distance: {dist:.2f} Hz)")

    # Validate against phi^n predictions
    print("\n" + "-" * 40)
    print("Validating against phi^n predictions...")
    validation = validate_boundaries_vs_phi(
        consensus,
        phi_boundaries=PHI_BOUNDARIES_DEFAULT,
        freq_range=FREQ_RANGE,
        n_permutations=10000
    )

    print(f"\nPhi^n Validation Results:")
    print(f"  Boundaries matched (within 0.5 Hz): {validation['n_matched']}/{validation['n_empirical']}")
    print(f"  Mean distance to phi^n: {validation['mean_distance_hz']:.3f} Hz")
    print(f"  Random expectation: {validation['random_mean_distance']:.3f} +/- {validation['random_std_distance']:.3f} Hz")
    print(f"  Effect size (Cohen's d): {validation['effect_size']:.2f}")
    print(f"  P-value: {validation['p_value']:.4f}")

    if validation['p_value'] < 0.05:
        print("\n  *** SIGNIFICANT: Boundaries align with phi^n better than chance ***")
    else:
        print("\n  Not significant at p < 0.05")

    # Create minimal peaks_df for saving (just frequencies)
    peaks_df_minimal = pd.DataFrame({'frequency': all_freqs})

    return {
        'peaks_df': peaks_df_minimal,
        'results_df': results_df,
        'consensus': consensus,
        'validation': validation,
        'all_freqs': all_freqs
    }


def _aggregate_boundaries(boundaries: list, tolerance: float = 1.5) -> list:
    """Aggregate boundaries by merging close ones."""
    if not boundaries:
        return []

    boundaries = sorted(boundaries)
    aggregated = []

    i = 0
    while i < len(boundaries):
        group = [boundaries[i]]
        j = i + 1
        while j < len(boundaries) and boundaries[j] - group[0] <= tolerance:
            group.append(boundaries[j])
            j += 1
        aggregated.append(np.median(group))
        i = j

    return aggregated


def run_analysis(peaks_df: pd.DataFrame) -> dict:
    """Run full gedBounds analysis pipeline."""

    # 1. Run boundary detection with multiple methods
    print("\n" + "=" * 60)
    print("Running Boundary Detection Pipeline")
    print("=" * 60)

    results_df = run_boundary_detection_pipeline(
        peaks_df,
        freq_col='frequency',
        methods=['density', 'gmm', 'agglomerative'],
        freq_range=FREQ_RANGE,
        density_bandwidth=KDE_BANDWIDTH,
        density_prominence=DENSITY_PROMINENCE,
        clustering_max_k=CLUSTERING_MAX_K
    )

    print(f"\nDetected {len(results_df)} boundary candidates across all methods")

    # 2. Get consensus boundaries
    consensus = get_consensus_boundaries(results_df, min_confidence=2)
    if not consensus:
        print("No consensus boundaries (min_confidence=2), trying min_confidence=1")
        consensus = get_consensus_boundaries(results_df, min_confidence=1)

    print(f"\nConsensus boundaries ({len(consensus)}):")
    for b in consensus:
        nearest = min(PHI_BOUNDARIES_DEFAULT, key=lambda x: abs(x - b))
        dist = abs(b - nearest)
        print(f"  {b:.2f} Hz  (nearest phi^n: {nearest:.2f} Hz, distance: {dist:.2f} Hz)")

    # 3. Validate against phi^n predictions
    print("\n" + "-" * 40)
    print("Validating against phi^n predictions...")
    validation = validate_boundaries_vs_phi(
        consensus,
        phi_boundaries=PHI_BOUNDARIES_DEFAULT,
        freq_range=FREQ_RANGE,
        n_permutations=10000
    )

    print(f"\nPhi^n Validation Results:")
    print(f"  Boundaries matched (within 0.5 Hz): {validation['n_matched']}/{validation['n_empirical']}")
    print(f"  Mean distance to phi^n: {validation['mean_distance_hz']:.3f} Hz")
    print(f"  Random expectation: {validation['random_mean_distance']:.3f} +/- {validation['random_std_distance']:.3f} Hz")
    print(f"  Effect size (Cohen's d): {validation['effect_size']:.2f}")
    print(f"  P-value: {validation['p_value']:.4f}")

    if validation['p_value'] < 0.05:
        print("\n  *** SIGNIFICANT: Boundaries align with phi^n better than chance ***")
    else:
        print("\n  Not significant at p < 0.05")

    return {
        'peaks_df': peaks_df,
        'results_df': results_df,
        'consensus': consensus,
        'validation': validation
    }


def save_results(results: dict, output_dir: str):
    """Save analysis results to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Save boundary detection results
    results['results_df'].to_csv(
        os.path.join(output_dir, 'detected_boundaries.csv'),
        index=False
    )
    print(f"\nSaved: {output_dir}/detected_boundaries.csv")

    # Save consensus boundaries
    consensus_df = pd.DataFrame({
        'frequency': results['consensus'],
        'nearest_phi': [min(PHI_BOUNDARIES_DEFAULT, key=lambda x: abs(x - b))
                       for b in results['consensus']],
        'distance_to_phi': [min(abs(b - phi) for phi in PHI_BOUNDARIES_DEFAULT)
                          for b in results['consensus']]
    })
    consensus_df.to_csv(
        os.path.join(output_dir, 'consensus_boundaries.csv'),
        index=False
    )
    print(f"Saved: {output_dir}/consensus_boundaries.csv")

    # Save validation results
    val_df = pd.DataFrame([{
        'n_empirical': results['validation']['n_empirical'],
        'n_phi': results['validation']['n_phi'],
        'n_matched': results['validation']['n_matched'],
        'mean_distance_hz': results['validation']['mean_distance_hz'],
        'random_mean_distance': results['validation']['random_mean_distance'],
        'random_std_distance': results['validation']['random_std_distance'],
        'effect_size': results['validation']['effect_size'],
        'p_value': results['validation']['p_value']
    }])
    val_df.to_csv(
        os.path.join(output_dir, 'phi_validation.csv'),
        index=False
    )
    print(f"Saved: {output_dir}/phi_validation.csv")

    # Generate comprehensive report
    report = generate_boundary_report(
        results['peaks_df'],
        freq_col='frequency',
        freq_range=FREQ_RANGE
    )
    with open(os.path.join(output_dir, 'boundary_report.txt'), 'w') as f:
        f.write(report['summary_text'])
    print(f"Saved: {output_dir}/boundary_report.txt")


def create_figures(results: dict, output_dir: str):
    """Create visualization figures."""
    os.makedirs(output_dir, exist_ok=True)

    # Main boundary detection figure
    fig = plot_boundary_detection(
        results['peaks_df'],
        results['results_df'],
        freq_col='frequency',
        freq_range=FREQ_RANGE,
        density_bandwidth=KDE_BANDWIDTH,
        save_path=os.path.join(output_dir, 'gedbounds_analysis.png'),
        show=False
    )
    print(f"Saved: {output_dir}/gedbounds_analysis.png")


def run_per_dataset_analysis(peaks_df: pd.DataFrame, output_dir: str):
    """Run analysis separately for each dataset to check consistency."""
    print("\n" + "=" * 60)
    print("Per-Dataset Boundary Detection")
    print("=" * 60)

    dataset_results = []

    for dataset in peaks_df['dataset'].unique():
        subset = peaks_df[peaks_df['dataset'] == dataset]
        print(f"\n{dataset}: {len(subset):,} peaks")

        results_df = run_boundary_detection_pipeline(
            subset,
            freq_col='frequency',
            methods=['density'],  # Just density for per-dataset
            freq_range=FREQ_RANGE
        )

        consensus = get_consensus_boundaries(results_df, min_confidence=1)
        validation = validate_boundaries_vs_phi(consensus, freq_range=FREQ_RANGE)

        print(f"  Boundaries: {[f'{b:.1f}' for b in consensus]}")
        print(f"  Mean distance to phi: {validation['mean_distance_hz']:.2f} Hz")
        print(f"  P-value: {validation['p_value']:.4f}")

        for b in consensus:
            nearest = min(PHI_BOUNDARIES_DEFAULT, key=lambda x: abs(x - b))
            dataset_results.append({
                'dataset': dataset,
                'boundary_hz': b,
                'nearest_phi_hz': nearest,
                'distance_hz': abs(b - nearest)
            })

    # Save per-dataset results
    if dataset_results:
        per_dataset_df = pd.DataFrame(dataset_results)
        per_dataset_df.to_csv(
            os.path.join(output_dir, 'per_dataset_boundaries.csv'),
            index=False
        )
        print(f"\nSaved: {output_dir}/per_dataset_boundaries.csv")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("gedBounds-style Boundary Detection from GED Peaks")
    print("=" * 60)

    # Use memory-efficient chunked analysis
    USE_CHUNKED = True

    if USE_CHUNKED:
        # Load frequencies only (memory efficient)
        all_freqs, dataset_counts = load_frequencies_only()

        # Run chunked analysis
        results = run_chunked_analysis(all_freqs, dataset_counts)

        # Save results
        save_results(results, OUTPUT_DIR)

        # Create figures (using minimal peaks_df)
        create_figures(results, OUTPUT_DIR)

        # Skip per-dataset analysis to save memory
        print("\n(Skipping per-dataset analysis to conserve memory)")

    else:
        # Original approach - loads all data into memory
        peaks_df = load_all_peaks()
        results = run_analysis(peaks_df)
        save_results(results, OUTPUT_DIR)
        create_figures(results, OUTPUT_DIR)
        run_per_dataset_analysis(peaks_df, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"Results saved to: {OUTPUT_DIR}/")

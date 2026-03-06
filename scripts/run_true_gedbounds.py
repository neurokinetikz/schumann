#!/usr/bin/env python3
"""
TRUE gedBounds Analysis: Covariance-Based Frequency Boundary Detection
=======================================================================

Implements Cohen (2021)'s gedBounds algorithm on raw EEG data.
Processes sessions one at a time to minimize memory usage.

Output:
- Per-dataset similarity curves
- Grand aggregated similarity curve
- Detected boundary frequencies
- φⁿ validation statistics
"""

import sys
import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add lib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))

from true_gedbounds import (
    compute_similarity_curve,
    find_boundaries_from_similarity,
    aggregate_similarity_curves,
    validate_boundaries_vs_phi
)
from utilities import load_eeg_csv

# ============================================================================
# Configuration
# ============================================================================

FREQ_RANGE = (4.5, 45.0)
FREQ_RESOLUTION = 0.1  # 405 frequency bins (5x finer resolution for precision)
BANDWIDTH = 0.1  # Very narrow filter bandwidth (Hz) - breaks 1 Hz artifact
SIMILARITY_METHOD = 'frobenius'  # More sensitive to covariance structure changes

# Output directory
OUTPUT_DIR = 'exports_peak_distribution/true_gedbounds'

# Electrode configurations
EPOC_ELECTRODES = ['AF3','AF4','F7','F8','F3','F4','FC5','FC6','P7','P8','T7','T8','O1','O2']
INSIGHT_ELECTRODES = ['AF3', 'AF4', 'T7', 'T8', 'Pz']
MUSE_ELECTRODES = ['AF7', 'AF8', 'TP9', 'TP10']

# Datasets to process
DATASETS = {
    'physf': {
        'pattern': 'data/PhySF/*.csv',
        'electrodes': EPOC_ELECTRODES,
        'device': 'emotiv',
        'fs': 128
    },
    'mpeng': {
        'pattern': 'data/mpeng/*.csv',
        'electrodes': EPOC_ELECTRODES,
        'device': 'emotiv',
        'fs': 128
    },
    'emotions': {
        'pattern': 'data/emotions/*.csv',
        'electrodes': EPOC_ELECTRODES,
        'device': 'emotiv',
        'fs': 128
    },
    'areeg': {
        'pattern': 'data/ArEEG/*.csv',
        'electrodes': EPOC_ELECTRODES,
        'device': 'emotiv',
        'fs': 128
    },
    'vep': {
        'pattern': 'data/vep/*.csv',
        'electrodes': EPOC_ELECTRODES,
        'device': 'emotiv',
        'fs': 128
    },
}


def get_eeg_data_array(filepath, electrodes, device='emotiv', fs=128):
    """Load EEG data and return as (n_channels, n_samples) array."""
    try:
        # First try direct pandas load (faster, handles EEG. prefix files)
        import pandas as pd
        df = pd.read_csv(filepath)

        # Find EEG columns
        eeg_cols = [c for c in df.columns if c.startswith('EEG.') and
                    not any(x in c for x in ['FILTERED', 'POW', 'Delta',
                                              'Theta', 'Alpha', 'Beta', 'Gamma'])]

        if len(eeg_cols) < 4:
            # Try using utilities loader
            records = load_eeg_csv(filepath, electrodes=electrodes,
                                   device=device, fs=fs)
            eeg_cols = [c for c in records.columns if c.startswith('EEG.') and
                        not any(x in c for x in ['FILTERED', 'POW', 'Delta',
                                                  'Theta', 'Alpha', 'Beta', 'Gamma'])]
            if len(eeg_cols) >= 4:
                df = records

        if len(eeg_cols) < 4:
            return None, 0

        data = df[eeg_cols].values.T  # (n_channels, n_samples)

        # Check minimum duration (at least 30 seconds)
        duration = data.shape[1] / fs
        if duration < 30:
            return None, 0

        return data, fs

    except Exception:
        return None, 0


def process_dataset(dataset_name, config, max_sessions=None, checkpoint_interval=50):
    """
    Process all sessions in a dataset.

    Returns:
        all_similarities: list of similarity curves
        frequencies: frequency array (same for all)
        session_ids: list of session identifiers
    """
    pattern = config['pattern']
    electrodes = config['electrodes']
    device = config['device']
    fs = config['fs']

    files = sorted(glob.glob(pattern, recursive=True))
    if max_sessions:
        files = files[:max_sessions]

    print(f"\n{'='*60}", flush=True)
    print(f"Processing {dataset_name}: {len(files)} files", flush=True)
    print(f"{'='*60}", flush=True)

    all_similarities = []
    all_frequencies = None
    session_ids = []
    failed = 0

    start_time = time.time()

    for i, filepath in enumerate(files):
        session_id = Path(filepath).stem

        # Progress update
        if i % 10 == 0:
            elapsed = time.time() - start_time
            if i > 0:
                rate = i / elapsed
                remaining = (len(files) - i) / rate
                print(f"  [{i}/{len(files)}] {session_id} "
                      f"(elapsed: {elapsed:.1f}s, remaining: ~{remaining:.1f}s)", flush=True)
            else:
                print(f"  [{i}/{len(files)}] {session_id}", flush=True)

        # Load EEG data
        data, actual_fs = get_eeg_data_array(filepath, electrodes, device, fs)

        if data is None:
            failed += 1
            continue

        try:
            # Compute similarity curve
            frequencies, similarities = compute_similarity_curve(
                data, actual_fs, FREQ_RANGE, FREQ_RESOLUTION,
                BANDWIDTH, SIMILARITY_METHOD, verbose=False
            )

            all_similarities.append(similarities)
            session_ids.append(session_id)

            if all_frequencies is None:
                all_frequencies = frequencies

        except Exception as e:
            failed += 1
            continue

        # Checkpoint
        if (i + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(OUTPUT_DIR, f'checkpoint_{dataset_name}.npz')
            np.savez(checkpoint_path,
                     frequencies=all_frequencies,
                     similarities=np.array(all_similarities),
                     session_ids=session_ids)
            print(f"  Checkpoint saved: {len(all_similarities)} sessions")

    elapsed = time.time() - start_time
    print(f"\n  Completed {dataset_name}: {len(all_similarities)}/{len(files)} sessions "
          f"in {elapsed:.1f}s ({failed} failed)")

    return all_similarities, all_frequencies, session_ids


def main():
    """Run TRUE gedBounds analysis."""

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'figures'), exist_ok=True)

    print("="*70, flush=True)
    print("TRUE gedBounds: Covariance-Based Frequency Boundary Detection", flush=True)
    print("="*70, flush=True)
    print(f"Frequency range: {FREQ_RANGE[0]}-{FREQ_RANGE[1]} Hz", flush=True)
    print(f"Resolution: {FREQ_RESOLUTION} Hz ({int((FREQ_RANGE[1]-FREQ_RANGE[0])/FREQ_RESOLUTION)} bins)", flush=True)
    print(f"Filter bandwidth: {BANDWIDTH} Hz", flush=True)
    print(f"Similarity method: {SIMILARITY_METHOD}", flush=True)
    print(flush=True)

    # Collect all similarity curves
    all_similarities = []
    all_frequencies = None
    dataset_results = {}

    # Process each dataset
    for dataset_name, config in DATASETS.items():
        # Check if files exist
        files = glob.glob(config['pattern'], recursive=True)
        if len(files) == 0:
            print(f"Skipping {dataset_name}: no files found")
            continue

        similarities, frequencies, session_ids = process_dataset(dataset_name, config)

        if len(similarities) > 0:
            # Save per-dataset results
            np.savez(os.path.join(OUTPUT_DIR, f'similarity_curves_{dataset_name}.npz'),
                     frequencies=frequencies,
                     similarities=np.array(similarities),
                     session_ids=session_ids)

            # Store for aggregation
            dataset_results[dataset_name] = {
                'similarities': similarities,
                'frequencies': frequencies,
                'n_sessions': len(similarities)
            }

            all_similarities.extend(similarities)
            if all_frequencies is None:
                all_frequencies = frequencies

    if len(all_similarities) == 0:
        print("ERROR: No sessions processed successfully!")
        return

    # ========================================================================
    # Aggregate similarity curves
    # ========================================================================
    print("\n" + "="*60)
    print(f"Aggregating {len(all_similarities)} similarity curves")
    print("="*60)

    grand_freqs, grand_sim = aggregate_similarity_curves(
        [all_frequencies] * len(all_similarities),
        all_similarities,
        method='mean'
    )

    # Also compute median for robustness
    _, grand_sim_median = aggregate_similarity_curves(
        [all_frequencies] * len(all_similarities),
        all_similarities,
        method='median'
    )

    # Find boundaries
    boundaries = find_boundaries_from_similarity(
        grand_freqs, grand_sim,
        prominence_percentile=20,
        min_distance_hz=2.0,
        smooth_window=5
    )

    print(f"\nDetected {len(boundaries)} boundaries:")
    for b in sorted(boundaries):
        print(f"  {b:.2f} Hz")

    # ========================================================================
    # Validate vs φⁿ positions
    # ========================================================================
    print("\n" + "="*60)
    print("Validating boundaries vs φⁿ positions")
    print("="*60)

    # Integer φⁿ boundaries
    phi_integer = [7.60, 12.30, 19.90, 32.19]

    # All noble positions (expanded)
    phi = 1.618033988749895
    f0 = 7.6
    all_nobles = []
    for n in range(-1, 4):
        base = f0 * (phi ** n)
        if FREQ_RANGE[0] <= base <= FREQ_RANGE[1]:
            all_nobles.append(base)
        # Add nobles between integers
        noble1 = f0 * (phi ** (n + 1/phi))
        noble2 = f0 * (phi ** (n + 1/(phi**2)))
        if FREQ_RANGE[0] <= noble1 <= FREQ_RANGE[1]:
            all_nobles.append(noble1)
        if FREQ_RANGE[0] <= noble2 <= FREQ_RANGE[1]:
            all_nobles.append(noble2)
    all_nobles = sorted(set([round(x, 2) for x in all_nobles]))

    # Validate vs integer boundaries
    validation_integer = validate_boundaries_vs_phi(boundaries, phi_integer)
    print(f"\nInteger φⁿ boundaries ({len(phi_integer)} positions):")
    print(f"  Matched (≤0.5 Hz): {validation_integer['n_matched']}/{len(boundaries)}")
    print(f"  Mean distance: {validation_integer['mean_distance_hz']:.3f} Hz")
    print(f"  P-value: {validation_integer['p_value']:.4f}")
    print(f"  Effect size: {validation_integer['effect_size']:.2f}")

    # Validate vs all nobles
    validation_nobles = validate_boundaries_vs_phi(boundaries, all_nobles)
    print(f"\nAll noble positions ({len(all_nobles)} positions):")
    print(f"  Matched (≤0.5 Hz): {validation_nobles['n_matched']}/{len(boundaries)}")
    print(f"  Mean distance: {validation_nobles['mean_distance_hz']:.3f} Hz")
    print(f"  P-value: {validation_nobles['p_value']:.4f}")
    print(f"  Effect size: {validation_nobles['effect_size']:.2f}")

    # ========================================================================
    # Save results
    # ========================================================================
    print("\n" + "="*60)
    print("Saving results")
    print("="*60)

    # Grand similarity curve
    pd.DataFrame({
        'frequency': grand_freqs,
        'similarity_mean': grand_sim,
        'similarity_median': grand_sim_median
    }).to_csv(os.path.join(OUTPUT_DIR, 'grand_similarity_curve.csv'), index=False)

    # Detected boundaries
    boundary_df = pd.DataFrame({'frequency': boundaries})
    boundary_df['nearest_phi_integer'] = boundary_df['frequency'].apply(
        lambda x: min(phi_integer, key=lambda p: abs(x - p))
    )
    boundary_df['distance_to_phi_integer'] = boundary_df.apply(
        lambda r: abs(r['frequency'] - r['nearest_phi_integer']), axis=1
    )
    boundary_df.to_csv(os.path.join(OUTPUT_DIR, 'detected_boundaries.csv'), index=False)

    # Validation results
    pd.DataFrame([{
        'validation_type': 'integer_phi',
        'n_boundaries': len(boundaries),
        'n_phi_positions': len(phi_integer),
        'n_matched': validation_integer['n_matched'],
        'mean_distance_hz': validation_integer['mean_distance_hz'],
        'random_mean_distance': validation_integer['random_mean_distance'],
        'effect_size': validation_integer['effect_size'],
        'p_value': validation_integer['p_value']
    }, {
        'validation_type': 'all_nobles',
        'n_boundaries': len(boundaries),
        'n_phi_positions': len(all_nobles),
        'n_matched': validation_nobles['n_matched'],
        'mean_distance_hz': validation_nobles['mean_distance_hz'],
        'random_mean_distance': validation_nobles['random_mean_distance'],
        'effect_size': validation_nobles['effect_size'],
        'p_value': validation_nobles['p_value']
    }]).to_csv(os.path.join(OUTPUT_DIR, 'phi_validation.csv'), index=False)

    # ========================================================================
    # Generate figures
    # ========================================================================
    print("Generating figures...")

    # Figure 1: Grand similarity curve with boundaries
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(grand_freqs, grand_sim, 'b-', linewidth=1.5, label='Mean similarity')
    ax.plot(grand_freqs, grand_sim_median, 'g--', linewidth=1, alpha=0.7, label='Median similarity')

    # Mark boundaries
    for b in boundaries:
        ax.axvline(b, color='red', linestyle='--', alpha=0.7, linewidth=1.5)

    # Mark φⁿ positions
    for p in phi_integer:
        ax.axvline(p, color='gold', linestyle=':', alpha=0.8, linewidth=2)

    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Covariance Similarity', fontsize=12)
    ax.set_title(f'TRUE gedBounds: Covariance Similarity Curve (N={len(all_similarities)} sessions)\n'
                 f'Red dashed = detected boundaries, Gold dotted = φⁿ integer positions', fontsize=12)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(FREQ_RANGE)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'grand_similarity_curve.png'), dpi=150)
    plt.close()

    # Figure 2: Boundary comparison
    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot empirical boundaries
    ax.scatter(boundaries, [1]*len(boundaries), s=100, c='red', marker='v',
               label=f'Empirical boundaries (n={len(boundaries)})', zorder=3)

    # Plot φⁿ positions
    ax.scatter(phi_integer, [0]*len(phi_integer), s=100, c='gold', marker='^',
               label=f'φⁿ integer positions (n={len(phi_integer)})', zorder=3)

    # Connect matched pairs
    for b in boundaries:
        nearest = min(phi_integer, key=lambda p: abs(b - p))
        if abs(b - nearest) < 0.5:
            ax.plot([b, nearest], [1, 0], 'g-', alpha=0.5, linewidth=2)

    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['φⁿ positions', 'Empirical'])
    ax.set_title(f'Boundary Alignment: Empirical vs φⁿ\n'
                 f'p={validation_integer["p_value"]:.4f}, effect size={validation_integer["effect_size"]:.2f}',
                 fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(FREQ_RANGE)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'boundaries_vs_phi.png'), dpi=150)
    plt.close()

    # Summary
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print(f"  - grand_similarity_curve.csv")
    print(f"  - detected_boundaries.csv")
    print(f"  - phi_validation.csv")
    print(f"  - figures/grand_similarity_curve.png")
    print(f"  - figures/boundaries_vs_phi.png")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total sessions processed: {len(all_similarities)}")
    print(f"Detected boundaries: {len(boundaries)}")
    print(f"  {', '.join([f'{b:.2f}' for b in sorted(boundaries)])} Hz")
    print(f"\nValidation vs integer φⁿ positions:")
    print(f"  Matched: {validation_integer['n_matched']}/{len(boundaries)}")
    print(f"  P-value: {validation_integer['p_value']:.4f}")
    print(f"  Effect size: {validation_integer['effect_size']:.2f}")


if __name__ == '__main__':
    main()

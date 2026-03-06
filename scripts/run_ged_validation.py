#!/usr/bin/env python3
"""
GED φⁿ Validation Batch Runner
==============================

Runs GED-based validation of the φⁿ architecture hypothesis on all SIE sessions.

This script:
1. Loads all session files with pre-detected ignition windows
2. Runs both blind and canonical GED sweeps
3. Generates publication-quality validation tables
4. Outputs summary statistics and figures

Usage:
    python scripts/run_ged_validation.py [--output_dir PATH] [--verbose]

    Or import and customize:
        from scripts.run_ged_validation import run_full_validation
        results = run_full_validation(output_dir='my_exports/')
"""

import sys
import os
import argparse
from pathlib import Path
from glob import glob

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from ged_validation_pipeline import (
    batch_ged_validation,
    quick_validate_session,
    run_ged_validation_session,
    plot_ged_validation_summary,
    generate_table1_frequency_validation,
    generate_table2_noble_ratios,
    generate_table3_boundary_attractor,
    generate_table4_independence_convergence,
    timeout_handler, TimeoutError
)
from utilities import load_eeg_csv, ELECTRODES
from detect_ignition import detect_ignitions_session
from session_metadata import parse_session_metadata


# ============================================================================
# DATASET CONFIGURATIONS
# ============================================================================

# Update these paths to match your data directory structure
DATA_ROOT = Path(__file__).parent.parent / 'data'

# Device-specific electrodes (with EEG. prefix as in CSV files)
INSIGHT_ELECTRODES = ['EEG.AF3', 'EEG.T7', 'EEG.Pz', 'EEG.T8', 'EEG.AF4']
MUSE_ELECTRODES = ['EEG.AF7', 'EEG.AF8', 'EEG.TP9', 'EEG.TP10']
EPOC_ELECTRODES = ['EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5', 'EEG.T7', 'EEG.P7',
                   'EEG.O1', 'EEG.O2', 'EEG.P8', 'EEG.T8', 'EEG.FC6', 'EEG.F4',
                   'EEG.F8', 'EEG.AF4']

# Default harmonic labels for detection
DEFAULT_LABELS = ['sr1', 'sr1.5', 'sr2', 'sr2o', 'sr2.5', 'sr3', 'sr4', 'sr5', 'sr6']
DEFAULT_CANON = [7.6, 10, 12, 13.75, 15.5, 20, 25, 32, 40]
DEFAULT_HALF_BW = [0.6, 0.618, 0.7, 0.75, 0.8, 1, 2, 2.5, 3]


# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def discover_sessions(data_root: Path = DATA_ROOT) -> dict:
    """
    Discover all EEG session files in the data directory.

    Returns dict mapping dataset names to file lists.
    """
    datasets = {}

    # Look for common data organization patterns
    patterns = [
        ('files', 'files/*.csv'),
        ('physf', 'PhySF/*.csv'),
        ('vep', 'vep/*.csv'),
        ('mpeng', 'mpeng/*.csv'),
        ('insight', 'insight/*.csv'),
        ('muse', 'muse/*.csv'),
        ('emotions', 'emotions/*.csv'),
        ('epoc', 'epoc/*.csv'),
    ]

    for name, pattern in patterns:
        files = list(data_root.glob(pattern))
        if files:
            datasets[name] = [str(f) for f in sorted(files)]

    # Also check for loose CSV files in data/
    loose_files = list(data_root.glob('*.csv'))
    if loose_files:
        datasets['root'] = [str(f) for f in sorted(loose_files)]

    return datasets


def detect_ignitions_for_session(
    filepath: str,
    dataset: str,
    device: str = None,
    z_thresh: float = 3.0,
    show: bool = False,
    min_duration_sec: float = 60.0
) -> tuple:
    """
    Run ignition detection on a single session.

    Parameters
    ----------
    min_duration_sec : float
        Minimum session duration in seconds. Files shorter than this are skipped
        because FOOOF harmonic fitting becomes unstable with too few frequency bins.

    Returns (records, ignition_windows, metadata)
    """
    # Parse metadata
    metadata = parse_session_metadata(filepath, dataset)

    # Determine device
    if device is None:
        device = metadata.get('device', 'epoc')

    # Select electrodes (with EEG. prefix for raw column access)
    if device == 'insight':
        electrodes = INSIGHT_ELECTRODES
    elif device == 'muse':
        electrodes = MUSE_ELECTRODES
    else:  # epoc or default
        electrodes = EPOC_ELECTRODES

    # PhySF, MPeng, VEP, Emotions files don't have metadata header row - use header=0
    # EPOC/Insight files have metadata row - use header=1
    header = 0 if dataset in ('physf', 'mpeng', 'vep', 'emotions') else 1

    # Load data
    records = load_eeg_csv(filepath, electrodes=electrodes, device=device, header=header)

    # Infer sampling rate
    if 'Timestamp' in records.columns:
        dt = np.diff(records['Timestamp'].values[:1000])
        dt = dt[np.isfinite(dt) & (dt > 0)]
        fs = 1.0 / np.median(dt) if len(dt) > 0 else 128.0
    else:
        fs = 128.0

    # Calculate session duration
    if 'Timestamp' in records.columns:
        duration = records['Timestamp'].iloc[-1] - records['Timestamp'].iloc[0]
    else:
        duration = len(records) / fs

    # Adapt FOOOF parameters for short files to prevent hanging
    # Short files need: smaller nperseg, fewer peaks, simpler fitting
    if duration < 45:
        # Very short - skip FOOOF entirely, use simple PSD
        harmonic_method = 'psd'
        nperseg_sec = 2.0
        fooof_max_n_peaks = 5
        print(f"  Short file ({duration:.0f}s) - using simple PSD method")
    elif duration < 90:
        # Medium-short - use conservative FOOOF settings
        harmonic_method = 'fooof_hybrid'
        nperseg_sec = 2.0
        fooof_max_n_peaks = 6
        print(f"  Medium file ({duration:.0f}s) - using conservative FOOOF")
    else:
        # Normal length - use default settings
        harmonic_method = 'fooof_hybrid'
        nperseg_sec = 4.0
        fooof_max_n_peaks = 10

    # Run ignition detection
    # electrodes already have EEG. prefix
    eeg_channels = electrodes
    sr_channel = eeg_channels[0] if eeg_channels else 'EEG.F4'

    try:
        out, ignition_windows = detect_ignitions_session(
            records,
            sr_channel=sr_channel,
            eeg_channels=eeg_channels,
            harmonic_method=harmonic_method,  # Adaptive based on duration
            harmonics_hz=DEFAULT_CANON,
            labels=DEFAULT_LABELS,
            half_bw_hz=DEFAULT_HALF_BW,
            z_thresh=z_thresh,
            show=show,
            nperseg_sec=nperseg_sec,  # Adaptive based on duration
            fooof_max_n_peaks=fooof_max_n_peaks  # Adaptive based on duration
        )

        # Get rounded integer windows
        if 'ignition_windows_rounded' in out:
            ignition_windows = out['ignition_windows_rounded']

    except Exception as e:
        print(f"  Warning: Ignition detection failed: {e}")
        ignition_windows = []

    return records, ignition_windows, metadata


def run_full_validation(
    output_dir: str = 'exports_ged_validation',
    max_sessions_per_dataset: int = None,
    datasets_filter: list = None,
    verbose: bool = True,
    z_thresh: float = 3.0,
    show: bool = False
) -> pd.DataFrame:
    """
    Run GED validation on all discovered sessions.

    Parameters
    ----------
    output_dir : str
        Output directory for results
    max_sessions_per_dataset : int, optional
        Limit sessions per dataset (for testing)
    datasets_filter : list, optional
        Only process these datasets (e.g., ['files', 'emotions'])
    verbose : bool
        Print progress

    Returns
    -------
    pd.DataFrame with combined results
    """
    if verbose:
        print("=" * 70)
        print("GED φⁿ VALIDATION PIPELINE")
        print("=" * 70)

    # Discover sessions
    datasets = discover_sessions()

    if datasets_filter:
        datasets = {k: v for k, v in datasets.items() if k in datasets_filter}

    if verbose:
        print(f"\nDiscovered datasets:")
        for name, files in datasets.items():
            print(f"  {name}: {len(files)} sessions")

    # Build session configs
    session_configs = []

    for dataset_name, file_list in datasets.items():
        if max_sessions_per_dataset:
            file_list = file_list[:max_sessions_per_dataset]

        # Determine device
        device = 'insight' if 'insight' in dataset_name.lower() else \
                 'muse' if 'muse' in dataset_name.lower() else 'epoc'

        for filepath in file_list:
            session_name = Path(filepath).name
            try:
                if verbose:
                    print(f"\nProcessing {session_name}...")

                # Wrap ignition detection in timeout (60s per session for detection phase)
                with timeout_handler(60, session_name):
                    # Detect ignitions (min_duration_sec=60 skips short files that cause FOOOF hangs)
                    records, ign_windows, metadata = detect_ignitions_for_session(
                        filepath, dataset_name, device, z_thresh=z_thresh, show=show,
                        min_duration_sec=60.0
                    )

                if len(ign_windows) == 0:
                    if verbose:
                        print(f"  No ignition windows found, skipping")
                    continue

                if verbose:
                    print(f"  Found {len(ign_windows)} ignition windows")

                session_configs.append({
                    'filepath': filepath,
                    'ignition_windows': ign_windows,
                    'dataset': dataset_name,
                    'device': device,
                    'session_id': f"{metadata.get('subject', 'unk')}_{metadata.get('context', 'unk')}"
                })

            except TimeoutError:
                if verbose:
                    print(f"  TIMEOUT: {session_name} - skipping detection")
                continue
            except Exception as e:
                if verbose:
                    print(f"  Error: {e}")
                continue

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Running GED validation on {len(session_configs)} sessions...")
        print(f"{'=' * 70}")

    # Run batch validation
    results_df = batch_ged_validation(
        session_configs,
        output_dir=output_dir,
        verbose=verbose
    )

    # Generate summary visualization
    if verbose:
        print("\nGenerating summary figures...")

    # Determine dataset name for figure title
    dataset_label = None
    if datasets_filter and len(datasets_filter) == 1:
        dataset_label = datasets_filter[0].upper()
    elif datasets_filter:
        dataset_label = " + ".join([d.upper() for d in datasets_filter])

    try:
        canonical_path = os.path.join(output_dir, 'aggregate', 'all_windows_canonical.csv')
        blind_path = os.path.join(output_dir, 'aggregate', 'all_blind_peaks.csv')

        if os.path.exists(canonical_path):
            canonical_df = pd.read_csv(canonical_path)
            blind_df = pd.read_csv(blind_path) if os.path.exists(blind_path) else None

            fig = plot_ged_validation_summary(
                canonical_df, blind_df,
                output_path=os.path.join(output_dir, 'figures', 'ged_validation_summary.png'),
                show=show,
                dataset_name=dataset_label
            )

            if verbose:
                print(f"  Saved: figures/ged_validation_summary.png")
    except Exception as e:
        if verbose:
            print(f"  Warning: Could not generate figure: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    if verbose and len(results_df) > 0:
        print(f"\n{'=' * 70}")
        print("VALIDATION SUMMARY")
        print(f"{'=' * 70}")
        print(f"Sessions processed: {len(results_df)}")

        if 'mean_phi_score' in results_df.columns:
            phi_scores = results_df['mean_phi_score'].dropna()
            if len(phi_scores) > 0:
                print(f"Mean φ-score: {phi_scores.mean():.3f} ± {phi_scores.std():.3f}")

        if 'blind_phi_alignment' in results_df.columns:
            alignment = results_df['blind_phi_alignment'].dropna()
            if len(alignment) > 0:
                print(f"Blind φⁿ alignment: {alignment.mean()*100:.1f}% ± {alignment.std()*100:.1f}%")

        print(f"\nOutput directory: {output_dir}")
        print(f"{'=' * 70}")

    return results_df


def run_quick_test(
    test_file: str = None,
    output_dir: str = 'exports_ged_validation_test',
    verbose: bool = True
) -> dict:
    """
    Run a quick test on a single session.

    If test_file is None, will find the first available session.
    """
    if test_file is None:
        # Find a test file
        datasets = discover_sessions()
        for name, files in datasets.items():
            if files:
                test_file = files[0]
                dataset = name
                break

        if test_file is None:
            print("No data files found!")
            return None
    else:
        dataset = 'test'

    if verbose:
        print(f"Quick test on: {test_file}")

    # Detect ignitions
    records, ign_windows, metadata = detect_ignitions_for_session(test_file, dataset)

    if verbose:
        print(f"Found {len(ign_windows)} ignition windows")

    # Run validation
    result = quick_validate_session(
        test_file,
        ign_windows,
        dataset=dataset,
        output_dir=output_dir,
        verbose=verbose
    )

    return result


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run GED φⁿ validation on EEG sessions'
    )
    parser.add_argument(
        '--output_dir', '-o',
        default='exports_ged_validation',
        help='Output directory for results'
    )
    parser.add_argument(
        '--datasets', '-d',
        nargs='+',
        help='Only process these datasets (e.g., files emotions)'
    )
    parser.add_argument(
        '--max_sessions', '-m',
        type=int,
        default=None,
        help='Max sessions per dataset (for testing)'
    )
    parser.add_argument(
        '--quick_test', '-t',
        action='store_true',
        help='Run quick test on single session'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        default=True,
        help='Print progress'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress output'
    )

    args = parser.parse_args()

    verbose = args.verbose and not args.quiet

    if args.quick_test:
        result = run_quick_test(verbose=verbose)
    else:
        results_df = run_full_validation(
            output_dir=args.output_dir,
            max_sessions_per_dataset=args.max_sessions,
            datasets_filter=args.datasets,
            verbose=verbose
        )


if __name__ == '__main__':
    main()

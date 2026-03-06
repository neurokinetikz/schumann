#!/usr/bin/env python
"""
Run Phi Validation Pipeline
============================

Command-line script to run GED-based validation of the φⁿ frequency model
on EPOC EEG sessions.

Usage:
    python scripts/run_phi_validation.py [--data-dir DATA_DIR] [--output-dir OUTPUT_DIR]

Example:
    python scripts/run_phi_validation.py --data-dir data/PhySF --output-dir exports/phi_validation
"""

from __future__ import annotations
import sys
import os
import glob
import argparse
from pathlib import Path

# Add lib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))

import numpy as np
import pandas as pd

# Import phi validation modules
from phi_frequency_model import get_default_phi_table, print_phi_table_summary
from phi_validation_pipeline import (
    run_phi_validation_session, batch_phi_validation,
    export_validation_results, plot_phi_validation_summary,
    EPOC_ELECTRODES
)

# Import existing utilities
try:
    from utilities import load_eeg_csv
    from detect_ignition import detect_ignitions_session
except ImportError:
    print("Warning: Could not import utilities/detect_ignition. Using minimal loader.")
    load_eeg_csv = None
    detect_ignitions_session = None


def minimal_load_eeg(filepath: str, fs: float = 128.0) -> pd.DataFrame:
    """Minimal EEG loader if utilities not available."""
    df = pd.read_csv(filepath)
    if 'Timestamp' not in df.columns:
        df['Timestamp'] = np.arange(len(df)) / fs
    return df


def find_epoc_sessions(data_dir: str) -> list:
    """Find all EPOC EEG sessions in directory."""
    patterns = [
        os.path.join(data_dir, '*.csv'),
        os.path.join(data_dir, '**', '*.csv'),
    ]

    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))

    # Filter to likely EEG files (has EEG columns)
    eeg_files = []
    for f in files:
        try:
            df = pd.read_csv(f, nrows=5)
            if any('EEG' in col for col in df.columns):
                eeg_files.append(f)
        except Exception:
            pass

    return sorted(set(eeg_files))


def process_session(
    filepath: str,
    phi_table,
    fs: float = 128.0,
    verbose: bool = True
) -> dict:
    """Process a single EEG session."""
    session_id = Path(filepath).stem

    if verbose:
        print(f"  Loading {session_id}...")

    # Load data
    if load_eeg_csv is not None:
        try:
            records = load_eeg_csv(filepath, electrodes=None, device='emotiv', fs=fs)
        except Exception as e:
            if verbose:
                print(f"    Error loading with utilities: {e}")
            records = minimal_load_eeg(filepath, fs)
    else:
        records = minimal_load_eeg(filepath, fs)

    # Find available EEG channels
    eeg_channels = [col for col in records.columns if 'EEG' in col]
    if not eeg_channels:
        return {'session_id': session_id, 'error': 'No EEG channels found'}

    # Detect ignition windows
    if detect_ignitions_session is not None:
        try:
            ign_out, ignition_windows = detect_ignitions_session(
                records,
                eeg_channels=eeg_channels,
                sr_channel=eeg_channels[0] if eeg_channels else 'EEG.F4',
                center_hz=7.6,
                z_thresh=2.0,
            )
            if not ignition_windows:
                # No ignitions detected - use full session as one window
                duration = len(records) / fs
                ignition_windows = [(0, int(duration))]
        except Exception as e:
            if verbose:
                print(f"    Ignition detection failed: {e}")
            duration = len(records) / fs
            ignition_windows = [(0, int(duration))]
    else:
        # No ignition detection - use full session
        duration = len(records) / fs
        ignition_windows = [(0, int(duration))]

    if verbose:
        print(f"    {len(eeg_channels)} channels, {len(ignition_windows)} windows")

    # Run phi validation
    result = run_phi_validation_session(
        records=records,
        ignition_windows=ignition_windows,
        eeg_channels=eeg_channels,
        session_id=session_id,
        phi_table=phi_table,
        fs=fs,
        run_blind_sweep=True,
        run_position_sweep=True,
        run_baseline_comparison=True,
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Run GED φⁿ validation on EPOC EEG sessions'
    )
    parser.add_argument(
        '--data-dir', '-d',
        type=str,
        default='data',
        help='Directory containing EEG CSV files'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='exports/phi_validation',
        help='Output directory for results'
    )
    parser.add_argument(
        '--single', '-s',
        type=str,
        default=None,
        help='Process single file instead of directory'
    )
    parser.add_argument(
        '--fs',
        type=float,
        default=128.0,
        help='Sampling rate (Hz)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=300,
        help='Timeout per session (seconds)'
    )
    parser.add_argument(
        '--show-table',
        action='store_true',
        help='Print phi frequency table and exit'
    )

    args = parser.parse_args()

    # Get phi table
    phi_table = get_default_phi_table()

    if args.show_table:
        print_phi_table_summary(phi_table)
        return

    print("=" * 60)
    print("GED φⁿ Validation Pipeline")
    print("=" * 60)
    print(f"\nPhi table: {len(phi_table)} predictions (all 8 position types)")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")

    # Find or specify sessions
    if args.single:
        files = [args.single]
    else:
        files = find_epoc_sessions(args.data_dir)

    if not files:
        print(f"\nNo EEG files found in {args.data_dir}")
        print("Use --single to specify a single file, or check the data directory.")
        return

    print(f"\nFound {len(files)} EEG sessions")

    # Process sessions
    results = []
    for i, filepath in enumerate(files):
        print(f"\n[{i+1}/{len(files)}] {Path(filepath).name}")
        try:
            result = process_session(filepath, phi_table, args.fs, verbose=True)
            results.append(result)

            # Print summary
            summary = result.get('summary', {})
            if 'error' not in summary:
                align = summary.get('blind_alignment_fraction', 0)
                pval = summary.get('blind_alignment_pvalue', 1)
                print(f"    -> Alignment: {align:.1%}, p={pval:.4f}")
        except Exception as e:
            print(f"    -> ERROR: {e}")
            results.append({'session_id': Path(filepath).stem, 'error': str(e)})

    # Create summary DataFrame
    summaries = [r.get('summary', {'session_id': r.get('session_id'), 'error': r.get('error')})
                 for r in results]
    summary_df = pd.DataFrame(summaries)

    # Export results
    print(f"\n{'=' * 60}")
    print("Exporting results...")

    os.makedirs(args.output_dir, exist_ok=True)
    paths = export_validation_results(
        summary_df=summary_df,
        full_results=results,
        output_dir=args.output_dir,
        prefix='phi_validation'
    )

    print(f"\nOutput files:")
    for name, path in paths.items():
        print(f"  {name}: {path}")

    # Print summary statistics
    print(f"\n{'=' * 60}")
    print("Summary Statistics")
    print("-" * 40)

    successful = summary_df[~summary_df.get('error', pd.Series()).notna() |
                           (summary_df.get('error', pd.Series()).isna())]

    if 'blind_alignment_fraction' in successful.columns:
        mean_align = successful['blind_alignment_fraction'].mean()
        print(f"Mean alignment fraction: {mean_align:.1%}")

    if 'blind_alignment_pvalue' in successful.columns:
        sig_sessions = (successful['blind_alignment_pvalue'] < 0.05).sum()
        print(f"Significant sessions (p<0.05): {sig_sessions}/{len(successful)}")

    if 'mean_freq_deviation_pct' in successful.columns:
        mean_dev = successful['mean_freq_deviation_pct'].mean()
        print(f"Mean frequency deviation: {mean_dev:.2f}%")

    print(f"\nDone! Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()

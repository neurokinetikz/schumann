#!/usr/bin/env python
"""
Run GED validation with ignition vs baseline comparison on PhySF dataset.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))

import glob
import numpy as np
import pandas as pd

from utilities import load_eeg_csv
from detect_ignition import detect_ignitions_session
from ged_validation_pipeline import (
    batch_ged_validation, plot_ged_validation_summary,
    plot_ignition_baseline_comparison, EPOC_ELECTRODES
)
from ged_bounds import ged_ignition_baseline_contrast

# Configuration
PHYSF_DIR = 'data/PhySF'
OUTPUT_DIR = 'exports_physf_baseline'
FS = 128

# PhySF files have EEG. prefixed electrodes - pass prefixed names to load_eeg_csv
PHYSF_ELECTRODES = ['EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5', 'EEG.T7', 'EEG.P7', 'EEG.O1', 'EEG.O2', 'EEG.P8', 'EEG.T8', 'EEG.FC6', 'EEG.F4', 'EEG.F8', 'EEG.AF4']

# Harmonic configuration
LABELS = ['sr1', 'sr1.5', 'sr2', 'sr2o', 'sr2.5', 'sr3', 'sr4', 'sr5', 'sr6']
CANON = [7.6, 10, 12, 13.75, 15.5, 20, 25, 32, 40]
HALF_BW = [0.6, 0.618, 0.7, 0.75, 0.8, 1, 2, 2.5, 3]


def detect_ignitions_for_session(filepath):
    """Load data and detect ignitions for a single session."""
    try:
        # Use utilities.load_eeg_csv - PhySF uses header=0
        records = load_eeg_csv(filepath, electrodes=PHYSF_ELECTRODES, device='epoc', header=0)

        # Require at least 30 seconds of data
        if len(records) < 3840:
            print(f"  Skipping: too short ({len(records)} samples)")
            return None, [], []

        # Find EEG channels
        eeg_channels = [c for c in records.columns if c.startswith('EEG.') and
                        not c.endswith('.Counter') and not c.endswith('.Interpolated')]

        if len(eeg_channels) < 2:
            print(f"  Skipping: insufficient channels")
            return None, [], []

        _, ign_windows = detect_ignitions_session(
            records,
            sr_channel=eeg_channels[0],
            eeg_channels=eeg_channels,
            center_hz=7.83,
            half_bw_hz=HALF_BW,
            z_thresh=2.0,
            harmonics_hz=CANON,
            labels=LABELS,
            harmonic_method='psd',
            nperseg_sec=4.0,
            window_sec=15.0,
            min_isi_sec=3.0,
            make_passport=False,
            show=False,
        )

        return records, ign_windows, eeg_channels

    except Exception as e:
        print(f"  Error: {e}")
        return None, [], []


def main():
    """Main entry point."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'per_session'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'aggregate'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'figures'), exist_ok=True)

    # Find all PhySF files
    physf_files = sorted(glob.glob(os.path.join(PHYSF_DIR, '*.csv')))
    print(f"Found {len(physf_files)} PhySF files")

    # Prepare session configs
    session_configs = []

    for filepath in physf_files:
        fname = os.path.basename(filepath)
        if fname == 'readme.txt':
            continue
        print(f"Processing {fname}...")

        result = detect_ignitions_for_session(filepath)
        records, ign_windows, eeg_channels = result

        if records is None or len(ign_windows) == 0:
            print("  No ignitions detected, skipping")
            continue

        print(f"  Found {len(ign_windows)} ignition windows")

        session_configs.append({
            'filepath': filepath,
            'ignition_windows': ign_windows,
            'dataset': 'physf',
            'device': 'epoc',
            'electrodes': eeg_channels,
        })

    print(f"\n{len(session_configs)} sessions with ignitions")

    if len(session_configs) == 0:
        print("No sessions to process!")
        return

    # Run batch validation
    print("\nRunning GED validation with baseline comparison...")
    batch_ged_validation(
        session_configs,
        output_dir=OUTPUT_DIR,
        verbose=True,
        session_timeout=180
    )

    # Generate figures
    print("\nGenerating summary figures...")
    canonical_path = os.path.join(OUTPUT_DIR, 'aggregate', 'all_windows_canonical.csv')
    baseline_path = os.path.join(OUTPUT_DIR, 'aggregate', 'all_windows_baseline.csv')
    blind_path = os.path.join(OUTPUT_DIR, 'aggregate', 'all_blind_peaks.csv')

    canonical_df = pd.read_csv(canonical_path) if os.path.exists(canonical_path) else pd.DataFrame()
    baseline_df = pd.read_csv(baseline_path) if os.path.exists(baseline_path) else pd.DataFrame()
    blind_df = pd.read_csv(blind_path) if os.path.exists(blind_path) else None

    if len(canonical_df) > 0:
        plot_ged_validation_summary(
            canonical_df,
            blind_df=blind_df,
            output_path=os.path.join(OUTPUT_DIR, 'figures', 'ged_validation_summary.png'),
            show=False,
            dataset_name='PhySF'
        )
        print("  Created ged_validation_summary.png")

    if len(canonical_df) > 0 and len(baseline_df) > 0:
        contrast = ged_ignition_baseline_contrast(
            canonical_df.to_dict('records'),
            baseline_df.to_dict('records')
        )

        plot_ignition_baseline_comparison(
            canonical_df, baseline_df, contrast,
            output_path=os.path.join(OUTPUT_DIR, 'figures', 'ignition_vs_baseline.png'),
            show=False,
            title='PhySF Dataset: Ignition vs Baseline GED Comparison'
        )
        print("  Created ignition_vs_baseline.png")

        # Print summary
        print("\n" + "=" * 60)
        print("IGNITION VS BASELINE SUMMARY")
        print("=" * 60)

        agg = contrast.get('aggregate', {})
        for metric in ['fwhm', 'q_factor', 'eigenvalue', 'lambda_ratio']:
            if metric in agg:
                stats = agg[metric]
                d = stats.get('cohens_d', np.nan)
                p = stats.get('p_value', np.nan)
                delta = stats.get('delta', np.nan)
                pred = stats.get('prediction_confirmed', False)

                sig = ''
                if p < 0.001:
                    sig = '***'
                elif p < 0.01:
                    sig = '**'
                elif p < 0.05:
                    sig = '*'

                print(f"{metric.upper():15} Delta={delta:+.3f}  d={d:+.2f}  p={p:.4f}{sig}  Pred={'YES' if pred else 'NO'}")

        print(f"\nIgnition windows: {len(canonical_df)}")
        print(f"Baseline windows: {len(baseline_df)}")

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()

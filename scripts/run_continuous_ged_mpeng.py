#!/usr/bin/env python3
"""
Run continuous GED peak detection on MPENG dataset.

This replicates the PhySF analysis pipeline for the MPENG dataset (900 files).
"""

import sys
sys.path.insert(0, './lib')

import os
from glob import glob
from peak_distribution_analysis import run_continuous_ged_detection

# MPENG files (900 total)
MPENG_FILES = sorted(glob('data/mpeng/*.csv'))

# Emotiv EPOC electrodes
ELECTRODES = ['EEG.AF3', 'EEG.AF4', 'EEG.F7', 'EEG.F8', 'EEG.F3', 'EEG.F4',
              'EEG.FC5', 'EEG.FC6', 'EEG.P7', 'EEG.P8', 'EEG.T7', 'EEG.T8',
              'EEG.O1', 'EEG.O2']

OUTPUT_DIR = 'exports_peak_distribution/mpeng_ged/continuous_v2'


def main():
    print(f"=" * 60)
    print("MPENG Continuous GED Peak Detection")
    print("=" * 60)
    print(f"Found {len(MPENG_FILES)} MPENG files")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Electrodes: {len(ELECTRODES)} channels")
    print(f"Parameters:")
    print(f"  freq_range: (4.5, 45.0) Hz")
    print(f"  window_sec: 10.0")
    print(f"  step_sec: 5.0")
    print(f"  sweep_step_hz: 0.1")
    print(f"  use_band_normalization: True")
    print("=" * 60)

    if len(MPENG_FILES) == 0:
        print("ERROR: No MPENG files found in data/mpeng/")
        return

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run continuous GED detection
    results = run_continuous_ged_detection(
        epoc_files=MPENG_FILES,
        electrodes=ELECTRODES,
        fs=128,
        output_dir=OUTPUT_DIR,
        freq_range=(4.5, 45.0),
        window_sec=10.0,
        step_sec=5.0,
        sweep_step_hz=0.1,
        use_band_normalization=True
    )

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print("=" * 60)
    print(f"Total peaks detected: {len(results):,}")
    print(f"Output saved to: {OUTPUT_DIR}/ged_peaks_continuous.csv")

    # Summary statistics
    if len(results) > 0:
        print(f"\nPer-band peak counts:")
        for band in ['theta', 'alpha', 'beta_low', 'beta_high', 'gamma']:
            count = len(results[results['band'] == band])
            print(f"  {band}: {count:,}")

        print(f"\nFrequency range: {results['frequency'].min():.2f} - {results['frequency'].max():.2f} Hz")


if __name__ == '__main__':
    main()

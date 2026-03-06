#!/usr/bin/env python3
"""
Run continuous GED peak detection on EMOTION dataset.

Dataset: 2,343 sessions from 88 subjects, 14-channel Emotiv EPOC X at 256 Hz.
"""

import sys
sys.path.insert(0, './lib')

import os
from glob import glob
from peak_distribution_analysis import run_continuous_ged_detection

# EMOTION files (2,343 sessions)
EMOTION_FILES = sorted(glob('data/emotions/*.csv'))

# 14-channel Emotiv EPOC X electrodes
ELECTRODES = ['EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5', 'EEG.T7', 'EEG.P7', 'EEG.O1',
              'EEG.O2', 'EEG.P8', 'EEG.T8', 'EEG.FC6', 'EEG.F4', 'EEG.F8', 'EEG.AF4']

OUTPUT_DIR = 'exports_peak_distribution/emotions_ged/continuous_v2'


def main():
    print(f"=" * 60)
    print("EMOTION Dataset Continuous GED Peak Detection")
    print("=" * 60)
    print(f"Found {len(EMOTION_FILES)} EMOTION files")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Electrodes: {len(ELECTRODES)} channels")
    print(f"Sampling Rate: 256 Hz")
    print(f"Parameters:")
    print(f"  freq_range: (4.5, 45.0) Hz")
    print(f"  window_sec: 10.0")
    print(f"  step_sec: 5.0")
    print(f"  sweep_step_hz: 0.1")
    print(f"  use_band_normalization: True")
    print("=" * 60)

    if len(EMOTION_FILES) == 0:
        print("ERROR: No EMOTION files found in data/emotions/")
        return

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run continuous GED detection
    results = run_continuous_ged_detection(
        epoc_files=EMOTION_FILES,
        electrodes=ELECTRODES,
        fs=256,  # EMOTION dataset uses 256 Hz
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

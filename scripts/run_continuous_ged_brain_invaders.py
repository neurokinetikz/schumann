#!/usr/bin/env python3
"""
Run continuous GED peak detection on Brain Invaders dataset.

Dataset: 64 subjects, 16-channel standard EEG at 512 Hz.
"""

import sys
sys.path.insert(0, './lib')

import os
from glob import glob
from peak_distribution_analysis import run_continuous_ged_detection

# Brain Invaders files (64 subjects, exclude Header.csv)
BI_FILES = sorted([f for f in glob('data/brain_invaders/*.csv')
                   if 'Header' not in f])

# 16-channel standard EEG electrodes
ELECTRODES = ['EEG.Fp1', 'EEG.Fp2', 'EEG.F5', 'EEG.AFZ', 'EEG.F6',
              'EEG.T7', 'EEG.Cz', 'EEG.T8',
              'EEG.P7', 'EEG.P3', 'EEG.PZ', 'EEG.P4', 'EEG.P8',
              'EEG.O1', 'EEG.Oz', 'EEG.O2']

OUTPUT_DIR = 'exports_peak_distribution/brain_invaders_ged/continuous_v2'


def main():
    print(f"=" * 60)
    print("Brain Invaders Continuous GED Peak Detection")
    print("=" * 60)
    print(f"Found {len(BI_FILES)} Brain Invaders files")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Electrodes: {len(ELECTRODES)} channels")
    print(f"Sampling Rate: 512 Hz")
    print(f"Parameters:")
    print(f"  freq_range: (4.5, 45.0) Hz")
    print(f"  window_sec: 10.0")
    print(f"  step_sec: 5.0")
    print(f"  sweep_step_hz: 0.1")
    print(f"  use_band_normalization: True")
    print("=" * 60)

    if len(BI_FILES) == 0:
        print("ERROR: No Brain Invaders files found in data/brain_invaders/")
        return

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run continuous GED detection
    results = run_continuous_ged_detection(
        epoc_files=BI_FILES,
        electrodes=ELECTRODES,
        fs=512,  # Brain Invaders uses 512 Hz
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

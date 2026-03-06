#!/usr/bin/env python3
"""
Run continuous GED peak detection on full PhySF dataset.

This script runs continuous GED sweep detection with per-band normalization,
which produces peak counts comparable to band-by-band detection (~500k peaks)
while eliminating the ~1.8 Hz gaps at band boundaries.

Key difference from band-by-band:
- Continuous sweep: No gaps at 7.60, 12.30, 19.90, 32.19 Hz boundaries
- Per-band normalization: Comparable peak counts to band-by-band detection

Expected output: ~400,000-500,000 peaks (comparable to band-based 447,057)
"""

import sys
sys.path.insert(0, './lib')

from glob import glob
import time
from peak_distribution_analysis import run_continuous_ged_detection

# PhySF dataset files
PHYSF_FILES = sorted(glob('data/PhySF/**/*.csv', recursive=True))

# EPOC electrodes
ELECTRODES = [
    'EEG.AF3', 'EEG.AF4', 'EEG.F7', 'EEG.F8', 'EEG.F3', 'EEG.F4',
    'EEG.FC5', 'EEG.FC6', 'EEG.P7', 'EEG.P8', 'EEG.T7', 'EEG.T8',
    'EEG.O1', 'EEG.O2'
]

# Output directory
OUTPUT_DIR = 'exports_peak_distribution/physf_ged/continuous_v2'


def main():
    print("=" * 60)
    print("CONTINUOUS GED PEAK DETECTION - FULL PhySF DATASET")
    print("=" * 60)
    print(f"\nFiles to process: {len(PHYSF_FILES)}")
    print(f"Electrodes: {len(ELECTRODES)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nMethod: Continuous sweep + per-band normalization")
    print("Expected peaks: ~400,000-500,000")
    print("\n" + "=" * 60)

    start_time = time.time()

    # Run continuous detection with per-band normalization
    # Using 0.1 Hz step (same as band-based) for comparable speed
    results = run_continuous_ged_detection(
        epoc_files=PHYSF_FILES,
        electrodes=ELECTRODES,
        fs=128,
        output_dir=OUTPUT_DIR,
        freq_range=(4.5, 45.0),
        window_sec=10.0,
        step_sec=5.0,
        sweep_step_hz=0.1,  # Match band-based detection resolution
        use_band_normalization=True  # Key: per-band normalization for comparable counts
    )

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nTotal peaks detected: {len(results):,}")
    print(f"Processing time: {elapsed/60:.1f} minutes")
    print(f"\nPer-band counts:")
    for band in ['theta', 'alpha', 'beta_low', 'beta_high', 'gamma']:
        count = len(results[results['band'] == band])
        print(f"  {band}: {count:,}")

    # Compare to band-based reference
    print("\n" + "-" * 40)
    print("Reference: Band-based detection = 447,057 peaks")
    ratio = len(results) / 447057
    print(f"Ratio to band-based: {ratio:.2f}x")

    if 0.8 <= ratio <= 1.2:
        print("SUCCESS: Peak count is comparable to band-based detection!")
    else:
        print("NOTE: Peak count differs significantly from band-based detection")


if __name__ == '__main__':
    main()

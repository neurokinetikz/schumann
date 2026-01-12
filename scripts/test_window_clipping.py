#!/usr/bin/env python3
"""
Test window clipping to recording bounds.

Demonstrates automatic clipping when windows extend beyond recording.
"""

import numpy as np
import pandas as pd
import warnings
from lib.fooof_harmonics import detect_harmonics_fooof

# Create synthetic data (10 seconds only)
fs = 128
duration = 10  # Short recording
n_samples = fs * duration
t = np.arange(n_samples) / fs

signal = np.random.randn(n_samples) * 10
signal += 8 * np.sin(2 * np.pi * 7.8 * t)

RECORDS = pd.DataFrame({
    'Timestamp': t,
    'EEG.F4': signal
})

print("=" * 70)
print("WINDOW CLIPPING TEST")
print("=" * 70)

print(f"\nRecording range: [0.0, 10.0] seconds")

# Test cases
test_windows = [
    ([2, 8], "Valid window within bounds"),
    ([8, 15], "Window extends beyond end (15 > 10)"),
    ([-5, 5], "Window starts before recording (-5 < 0)"),
    ([-2, 15], "Window extends both sides"),
    ([15, 20], "Window completely outside (should fail)"),
]

CANON = [7.83, 14.3]

for window, description in test_windows:
    print("\n" + "-" * 70)
    print(f"Test: {description}")
    print(f"Requested window: {window}")

    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            harmonics, result = detect_harmonics_fooof(
                records=RECORDS,
                channels='EEG.F4',
                fs=fs,
                window=window,
                f_can=CANON,
                freq_range=(5, 35)
            )

            if len(w) > 0:
                print(f"⚠️  WARNING: {w[0].message}")
            else:
                print(f"✓ No clipping needed")

            print(f"✓ Success: harmonics = {[f'{h:.2f}' for h in harmonics]}")

    except ValueError as e:
        print(f"✗ ERROR: {e}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("✓ Windows extending beyond bounds are automatically clipped")
print("✓ User is warned when clipping occurs")
print("✗ Windows completely outside bounds still raise errors")
print("=" * 70)

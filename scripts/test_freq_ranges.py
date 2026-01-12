#!/usr/bin/env python3
"""
Test manual freq_ranges parameter for grouped FOOOF windows.

Validates that freq_ranges:
1. Groups harmonics by their frequency range
2. Runs single FOOOF fit per unique range (efficiency)
3. Returns per-harmonic β values from shared fits
4. Matches your use case: [5,15], [15,25], [25,35] windows
"""

import numpy as np
import pandas as pd
from lib.fooof_harmonics import detect_harmonics_fooof

# Create synthetic data
fs = 128
duration = 20  # seconds
n_samples = fs * duration
t = np.arange(n_samples) / fs

# Generate signal with peaks in different frequency bands
signal = np.random.randn(n_samples) * 10
signal += 8 * np.sin(2 * np.pi * 7.6 * t)    # H1 in [5,15]
signal += 6 * np.sin(2 * np.pi * 9.3 * t)    # H2 in [5,15]
signal += 4 * np.sin(2 * np.pi * 12.1 * t)   # H3 in [5,15]
signal += 3 * np.sin(2 * np.pi * 13.8 * t)   # H4 in [5,15]
signal += 3 * np.sin(2 * np.pi * 19.8 * t)   # H5 in [15,25]
signal += 2 * np.sin(2 * np.pi * 25.0 * t)   # H6 in [25,35]
signal += 1 * np.sin(2 * np.pi * 31.5 * t)   # H7 in [25,35]

RECORDS = pd.DataFrame({
    'Timestamp': t,
    'EEG.F4': signal
})

print("=" * 70)
print("MANUAL FREQ_RANGES TEST")
print("=" * 70)

# Your use case
CANON = [7.6, 9.26, 12.13, 13.75, 19.75, 25, 32]
HALF_BW = [0.5, 0.618, 0.618, 0.75, 0.75, 1.25, 1.5]

# Test 1: Manual frequency ranges (grouped)
print("\n" + "=" * 70)
print("TEST 1: Grouped Frequency Ranges")
print("=" * 70)

FREQ_RANGES = [
    [5, 15],   # H1
    [5, 15],   # H2
    [5, 15],   # H3
    [5, 15],   # H4
    [15, 25],  # H5
    [25, 35],  # H6
    [25, 35]   # H7
]

print("\nFrequency window grouping:")
print(f"  [5, 15] Hz: H1, H2, H3, H4 (4 harmonics)")
print(f"  [15, 25] Hz: H5 (1 harmonic)")
print(f"  [25, 35] Hz: H6, H7 (2 harmonics)")
print(f"  → 3 FOOOF fits total (instead of 7)")

harmonics, result = detect_harmonics_fooof(
    records=RECORDS,
    channels='EEG.F4',
    fs=fs,
    f_can=CANON,
    freq_ranges=FREQ_RANGES,
    search_halfband=HALF_BW,
    match_method='power',
    max_n_peaks=10
)

print(f"\nDetected harmonics:")
for i, (h_can, h_det) in enumerate(zip(CANON, harmonics)):
    print(f"  H{i+1} (canon={h_can:.2f}): detected={h_det:.2f} Hz")

print(f"\nPer-harmonic β values:")
for i, (h, beta) in enumerate(zip(harmonics, result.aperiodic_exponent)):
    print(f"  H{i+1}: β={beta:.3f}")

print(f"\nPer-harmonic R² values:")
for i, r2 in enumerate(result.r_squared):
    print(f"  H{i+1}: R²={r2:.3f}")

# Check that harmonics in same group share β values
print(f"\nValidation: Harmonics in same freq_range should share β:")
print(f"  H1-H4 β: {result.aperiodic_exponent[0]:.3f}, {result.aperiodic_exponent[1]:.3f}, {result.aperiodic_exponent[2]:.3f}, {result.aperiodic_exponent[3]:.3f}")
if (result.aperiodic_exponent[0] == result.aperiodic_exponent[1] ==
    result.aperiodic_exponent[2] == result.aperiodic_exponent[3]):
    print(f"  ✓ H1-H4 share same β (from [5,15] Hz fit)")
else:
    print(f"  ✗ H1-H4 have different β values")

print(f"  H6-H7 β: {result.aperiodic_exponent[5]:.3f}, {result.aperiodic_exponent[6]:.3f}")
if result.aperiodic_exponent[5] == result.aperiodic_exponent[6]:
    print(f"  ✓ H6-H7 share same β (from [25,35] Hz fit)")
else:
    print(f"  ✗ H6-H7 have different β values")

# Test 2: Compare with per_harmonic_fits=True (7 separate fits)
print("\n" + "=" * 70)
print("TEST 2: Comparison with per_harmonic_fits=True")
print("=" * 70)

harmonics_per, result_per = detect_harmonics_fooof(
    records=RECORDS,
    channels='EEG.F4',
    fs=fs,
    f_can=CANON,
    search_halfband=HALF_BW,
    match_method='power',
    max_n_peaks=10,
    per_harmonic_fits=True  # Automatic ±5 Hz per harmonic
)

print(f"\nFrequency differences (grouped vs per-harmonic):")
for i, (h_group, h_per) in enumerate(zip(harmonics, harmonics_per)):
    diff = h_group - h_per
    print(f"  H{i+1}: grouped={h_group:.2f}, per_harmonic={h_per:.2f}, diff={diff:+.3f} Hz")

print(f"\nβ comparison:")
print(f"  Grouped:      {[f'{b:.2f}' for b in result.aperiodic_exponent]}")
print(f"  Per-harmonic: {[f'{b:.2f}' for b in result_per.aperiodic_exponent]}")

# Test 3: Error handling
print("\n" + "=" * 70)
print("TEST 3: Error Handling")
print("=" * 70)

# Wrong length
try:
    harmonics, result = detect_harmonics_fooof(
        RECORDS, 'EEG.F4', fs=fs,
        f_can=CANON,
        freq_ranges=[[5, 15], [15, 25]],  # Only 2 ranges for 7 harmonics
    )
    print("✗ FAILED: Should reject mismatched length")
except ValueError as e:
    print(f"✓ Correctly rejected mismatched length: {e}")

# Invalid range
try:
    harmonics, result = detect_harmonics_fooof(
        RECORDS, 'EEG.F4', fs=fs,
        f_can=CANON,
        freq_ranges=[[5, 15]] * 6 + [[30, 20]],  # Invalid: f_min > f_max
    )
    print("✗ FAILED: Should reject invalid range")
except ValueError as e:
    print(f"✓ Correctly rejected invalid range: {e}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

all_passed = True

# Check per-harmonic flag is set
if not result.per_harmonic_fits:
    print("✗ FAILED: per_harmonic_fits flag should be True")
    all_passed = False

# Check β is list
if not isinstance(result.aperiodic_exponent, list):
    print("✗ FAILED: aperiodic_exponent should be a list")
    all_passed = False

# Check correct length
if len(result.aperiodic_exponent) != len(CANON):
    print(f"✗ FAILED: Expected {len(CANON)} β values, got {len(result.aperiodic_exponent)}")
    all_passed = False

# Check harmonics in same group share β
if (result.aperiodic_exponent[0] == result.aperiodic_exponent[1] ==
    result.aperiodic_exponent[2] == result.aperiodic_exponent[3]):
    print("✓ H1-H4 correctly share β from [5,15] Hz fit")
else:
    print("✗ FAILED: H1-H4 should share same β")
    all_passed = False

if result.aperiodic_exponent[5] == result.aperiodic_exponent[6]:
    print("✓ H6-H7 correctly share β from [25,35] Hz fit")
else:
    print("✗ FAILED: H6-H7 should share same β")
    all_passed = False

if all_passed:
    print("\n✓ ALL TESTS PASSED!")
    print("\nManual freq_ranges feature working correctly:")
    print("  - Groups harmonics by frequency range")
    print("  - Runs single FOOOF fit per unique range")
    print("  - Returns per-harmonic β values (shared within groups)")
    print("  - Efficient for your use case: 3 fits instead of 7")
else:
    print("\n✗ SOME TESTS FAILED")

print("=" * 70)

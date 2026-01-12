#!/usr/bin/env python3
"""
Debug script to verify which halfband values are actually being used.
"""

import numpy as np
import pandas as pd
from lib import fooof_harmonics

# Create test data
fs = 128
duration = 60
n_samples = fs * duration
t = np.arange(n_samples) / fs

signal = np.random.randn(n_samples) * 10
signal += 8 * np.sin(2 * np.pi * 7.8 * t)
signal += 6 * np.sin(2 * np.pi * 19.5 * t)
signal += 4 * np.sin(2 * np.pi * 34.27 * t)  # Peak exactly at 34.27

RECORDS = pd.DataFrame({
    'Timestamp': t,
    'EEG.F4': signal,
})

# Your exact configuration
CANON = [7.6, 20, 32.0]
_half_bw = [0.6, 1, 2]  # PER-HARMONIC BANDWIDTHS
FREQS = (5, 40)
FREQ_RANGES = [
    [5, 15],   # H1
    [15, 25],  # H2
    [27, 37]   # H3
]

print("=" * 70)
print("HALFBAND DEBUG TEST")
print("=" * 70)
print(f"\nConfiguration:")
print(f"  CANON: {CANON}")
print(f"  _half_bw: {_half_bw}")
print(f"  FREQ_RANGES: {FREQ_RANGES}")

# Add debug instrumentation to match_peaks_to_canonical
# Patch the function to print actual values
original_match = fooof_harmonics.match_peaks_to_canonical

def debug_match(peak_params, f_can, search_halfband, method='distance'):
    print(f"\n--- match_peaks_to_canonical called ---")
    print(f"f_can: {f_can}")
    print(f"search_halfband (input): {search_halfband}")
    print(f"method: {method}")

    # Convert search_halfband to list
    if isinstance(search_halfband, (int, float)):
        halfbands = [float(search_halfband)] * len(f_can)
    else:
        halfbands = list(search_halfband)

    print(f"halfbands (converted): {halfbands}")

    # Show search windows
    for i, (f0, halfband) in enumerate(zip(f_can, halfbands)):
        lo, hi = f0 - halfband, f0 + halfband
        print(f"  H{i+1}: f_can={f0:.2f}, halfband={halfband}, window=[{lo:.2f}, {hi:.2f}]")

    # Show detected peaks
    if len(peak_params) > 0:
        print(f"\nFOOOF detected {len(peak_params)} peaks:")
        for i, peak in enumerate(peak_params):
            freq, power, bw = peak
            print(f"  Peak {i+1}: freq={freq:.4f} Hz, power={power:.4f}, bw={bw:.4f}")

    # Call original function
    result = original_match(peak_params, f_can, search_halfband, method)

    print(f"\nMatched harmonics: {[f'{h:.4f}' for h in result[0]]}")
    print("=" * 70)

    return result

# Monkey-patch for debugging
fooof_harmonics.match_peaks_to_canonical = debug_match

# Run FOOOF
print("\n\nRunning FOOOF with freq_ranges...")
_harmonics, result = fooof_harmonics.detect_harmonics_fooof(
    RECORDS, ['EEG.F4'], fs=fs,
    f_can=CANON,
    freq_range=FREQS,
    freq_ranges=FREQ_RANGES,
    search_halfband=_half_bw,  # ← Array
    max_n_peaks=15,
    peak_threshold=0.01,
    min_peak_height=0.01,
    peak_width_limits=(0.5, 12.0),
    match_method='power',
)

print("\n\n" + "=" * 70)
print("FINAL RESULT")
print("=" * 70)
print(f"Detected harmonics: {[f'{h:.4f}' for h in _harmonics]}")
print(f"\nExpected for H3 (CANON=32, half_bw=2): [30.0, 34.0]")
print(f"Actual H3: {_harmonics[2]:.4f}")

if _harmonics[2] > 34.0:
    print(f"\n❌ ERROR: H3 = {_harmonics[2]:.4f} is OUTSIDE expected range [30, 34]!")
    print(f"   Peak is {_harmonics[2] - 34.0:.4f} Hz beyond upper limit.")
else:
    print(f"\n✅ H3 is within expected range")

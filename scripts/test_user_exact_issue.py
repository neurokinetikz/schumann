#!/usr/bin/env python3
"""
Reproduce the user's exact issue: detecting 34.27 Hz when window is [30, 34].
"""

import numpy as np
import pandas as pd
from lib import fooof_harmonics

# Create signal with peak at EXACTLY 34.2673892490006 Hz (user's value)
fs = 128
duration = 60
n_samples = fs * duration
t = np.arange(n_samples) / fs

signal = np.random.randn(n_samples) * 10
signal += 15 * np.sin(2 * np.pi * 7.77 * t)      # H1
signal += 12 * np.sin(2 * np.pi * 19.18 * t)     # H2
signal += 20 * np.sin(2 * np.pi * 34.2673892490006 * t)  # H3 at exact user's value

RECORDS = pd.DataFrame({
    'Timestamp': t,
    'EEG.F4': signal,
    'EEG.O1': signal + np.random.randn(n_samples) * 2,
    'EEG.C3': signal + np.random.randn(n_samples) * 2,
})

_electrodes = ['EEG.F4', 'EEG.O1', 'EEG.C3']

# User's exact configuration
CANON = [7.6, 20, 32.0]
_half_bw = [0.6, 1, 2]
FREQS = (5, 40)
FREQ_RANGES = [
    [5, 15],
    [15, 25],
    [27, 37]
]

print("=" * 70)
print("USER'S EXACT SCENARIO TEST")
print("=" * 70)
print(f"\nConfiguration:")
print(f"  CANON: {CANON}")
print(f"  _half_bw: {_half_bw}")
print(f"  FREQ_RANGES: {FREQ_RANGES}")
print(f"\nSignal contains peak at: 34.2673892490006 Hz (user's exact value)")
print(f"Expected search window for H3: [32-2, 32+2] = [30, 34]")
print(f"34.267 > 34.0? {34.2673892490006 > 34.0}")

# Test Case 1: With freq_ranges (user's config)
print("\n" + "=" * 70)
print("TEST CASE 1: With freq_ranges (user's configuration)")
print("=" * 70)

_harmonics1, result1 = fooof_harmonics.detect_harmonics_fooof(
    RECORDS, _electrodes, fs=128,
    f_can=CANON,
    freq_range=FREQS,
    freq_ranges=FREQ_RANGES,
    search_halfband=_half_bw,
    max_n_peaks=15,
    peak_threshold=0.01,
    min_peak_height=0.01,
    peak_width_limits=(0.5, 12.0),
    match_method='power',
)

print(f"\nDetected harmonics: {_harmonics1}")
print(f"H3 detected: {_harmonics1[2]:.10f} Hz")

# Check if it's the problematic value
if abs(_harmonics1[2] - 34.2673892490006) < 0.01:
    print(f"\n❌ REPRODUCED THE ISSUE!")
    print(f"   Peak at 34.267 Hz was selected despite being outside [30, 34] window")
    print(f"   Distance from upper limit: {_harmonics1[2] - 34.0:.4f} Hz")
else:
    print(f"\n✅ Issue NOT reproduced")
    print(f"   Peak at 34.267 Hz was correctly rejected")
    print(f"   Selected peak within window: {_harmonics1[2]:.4f} Hz")

# Test Case 2: Pass scalar halfband to see if that's the issue
print("\n" + "=" * 70)
print("TEST CASE 2: What if halfband is scalar instead of array?")
print("=" * 70)

# Try with scalar halfband (might accidentally use larger value)
_harmonics2, result2 = fooof_harmonics.detect_harmonics_fooof(
    RECORDS, _electrodes, fs=128,
    f_can=CANON,
    freq_range=FREQS,
    freq_ranges=FREQ_RANGES,
    search_halfband=2.5,  # Scalar value >= 2.267 would allow 34.267
    max_n_peaks=15,
    peak_threshold=0.01,
    min_peak_height=0.01,
    peak_width_limits=(0.5, 12.0),
    match_method='power',
)

print(f"\nWith scalar halfband=2.5:")
print(f"  Windows: [7.6±2.5], [20±2.5], [32±2.5] = [29.5, 34.5]")
print(f"  H3 detected: {_harmonics2[2]:.10f} Hz")

if abs(_harmonics2[2] - 34.2673892490006) < 0.01:
    print(f"\n⚠️  BINGO! With halfband=2.5, the 34.267 peak IS selected!")
    print(f"   This suggests the user might be passing a scalar halfband instead of array")

# Test Case 3: Check what halfband would be needed to select 34.267
print("\n" + "=" * 70)
print("TEST CASE 3: What halfband value allows 34.267 to be selected?")
print("=" * 70)

required_halfband = 34.2673892490006 - 32.0
print(f"\nFor peak at 34.267 Hz to be within [32-hb, 32+hb]:")
print(f"  Required: hb >= {required_halfband:.4f} Hz")
print(f"  User's config: hb = 2.0 Hz")
print(f"  Difference: {required_halfband - 2.0:.4f} Hz")

# Show all peaks detected by FOOOF for H3 window
print("\n" + "=" * 70)
print("PEAKS DETECTED BY FOOOF IN [27, 37] Hz WINDOW")
print("=" * 70)

if isinstance(result1.model, list):
    model = result1.model[2]  # Third harmonic's model
else:
    model = result1.model

if model is not None:
    from lib.fooof_harmonics import _get_peak_params
    peaks = _get_peak_params(model)
    print(f"\nFOOOF detected {len(peaks)} peaks in [27, 37] Hz:")
    for i, peak in enumerate(peaks):
        freq, power, bw = peak
        in_window = (30.0 <= freq <= 34.0)
        marker = "✓" if in_window else "✗"
        print(f"  {marker} Peak {i+1}: {freq:.4f} Hz, power={power:.4f}, bw={bw:.4f}")
        if abs(freq - 34.2673892490006) < 0.01:
            print(f"      ^ This is the 34.267 Hz peak")

    print(f"\nPeaks within [30, 34] window:")
    valid_peaks = [(f, p) for f, p, bw in peaks if 30.0 <= f <= 34.0]
    if valid_peaks:
        for freq, power in valid_peaks:
            print(f"  • {freq:.4f} Hz (power={power:.4f})")
        highest = max(valid_peaks, key=lambda x: x[1])
        print(f"\nHighest power within window: {highest[0]:.4f} Hz")
    else:
        print("  (none)")

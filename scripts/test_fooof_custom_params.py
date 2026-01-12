#!/usr/bin/env python3
"""
Test FOOOF integration with custom sensitivity parameters.

Demonstrates using your specific FOOOF configuration:
- freq_range=(5, 40)
- freq_ranges=[[5,15], [5,15], [15,25], [25,35], [27,37]]
- max_n_peaks=15
- peak_threshold=0.01
- min_peak_height=0.01
- peak_width_limits=(0.5, 12.0)
"""

import numpy as np
import pandas as pd
from lib.detect_ignition import detect_ignitions_session

# Create synthetic test data
fs = 128
duration = 60
n_samples = fs * duration
t = np.arange(n_samples) / fs

# Generate signal with Schumann harmonics at specific frequencies
signal = np.random.randn(n_samples) * 10
signal += 8 * np.sin(2 * np.pi * 7.6 * t)    # H1
signal += 6 * np.sin(2 * np.pi * 9.3 * t)    # H2
signal += 5 * np.sin(2 * np.pi * 12.1 * t)   # H3
signal += 4 * np.sin(2 * np.pi * 13.8 * t)   # H4
signal += 3 * np.sin(2 * np.pi * 19.8 * t)   # H5
signal += 2 * np.sin(2 * np.pi * 25.0 * t)   # H6
signal += 2 * np.sin(2 * np.pi * 31.5 * t)   # H7

# Add a strong ignition event at t=30s
event_start = int(25 * fs)
event_end = int(35 * fs)
signal[event_start:event_end] += 20 * np.sin(2 * np.pi * 7.6 * t[event_start:event_end])

RECORDS = pd.DataFrame({
    'Timestamp': t,
    'EEG.F4': signal,
    'EEG.O1': signal + np.random.randn(n_samples) * 2,
    'EEG.C3': signal + np.random.randn(n_samples) * 2,
})

# Your custom FOOOF parameters
CANON = [7.6, 9.26, 12.13, 13.75, 19.75, 25, 32]
FREQS = (5, 40)
FREQ_RANGES = [
    [5, 15],   # H1
    [5, 15],   # H2
    [15, 25],  # H3
    [15, 25],  # H4
    [15, 25],  # H5
    [25, 35],  # H6
    [27, 37]   # H7
]

print("=" * 70)
print("FOOOF CUSTOM PARAMETERS TEST")
print("=" * 70)

# Test 1: Default FOOOF parameters
print("\n" + "=" * 70)
print("TEST 1: Default FOOOF Parameters")
print("=" * 70)

results_default, _ = detect_ignitions_session(
    RECORDS,
    sr_channel='EEG.F4',
    eeg_channels=['EEG.F4', 'EEG.O1', 'EEG.C3'],
    time_col='Timestamp',
    out_dir='test_output/fooof_default',
    harmonics_hz=CANON,
    harmonic_method='fooof_session',
    make_passport=False,
    show=False,
    verbose=True
)

if 'events' in results_default and not results_default['events'].empty:
    print(f"\n✓ Default FOOOF detected {len(results_default['events'])} events")
    if 'ignition_freqs' in results_default['events'].columns:
        first_event_freqs = results_default['events']['ignition_freqs'].iloc[0]
        print(f"  Harmonics: {[f'{f:.2f}' for f in first_event_freqs]}")

# Test 2: Custom high-sensitivity FOOOF parameters
print("\n" + "=" * 70)
print("TEST 2: Custom High-Sensitivity FOOOF Parameters")
print("=" * 70)
print(f"  freq_range: {FREQS}")
print(f"  max_n_peaks: 15")
print(f"  peak_threshold: 0.01 (vs default 2.0)")
print(f"  min_peak_height: 0.01 (vs default 0.05)")
print(f"  peak_width_limits: (0.5, 12.0) (vs default (1.0, 8.0))")

results_custom, _ = detect_ignitions_session(
    RECORDS,
    sr_channel='EEG.F4',
    eeg_channels=['EEG.F4', 'EEG.O1', 'EEG.C3'],
    time_col='Timestamp',
    out_dir='test_output/fooof_custom',
    harmonics_hz=CANON,
    harmonic_method='fooof_session',
    fooof_freq_range=FREQS,
    fooof_max_n_peaks=15,
    fooof_peak_threshold=0.01,
    fooof_min_peak_height=0.01,
    fooof_peak_width_limits=(0.5, 12.0),
    make_passport=False,
    show=False,
    verbose=True
)

if 'events' in results_custom and not results_custom['events'].empty:
    print(f"\n✓ Custom FOOOF detected {len(results_custom['events'])} events")
    if 'ignition_freqs' in results_custom['events'].columns:
        first_event_freqs = results_custom['events']['ignition_freqs'].iloc[0]
        print(f"  Harmonics: {[f'{f:.2f}' for f in first_event_freqs]}")

# Test 3: With freq_ranges (grouped FOOOF windows)
print("\n" + "=" * 70)
print("TEST 3: Custom FOOOF with freq_ranges (Grouped Windows)")
print("=" * 70)
print(f"  freq_ranges: {FREQ_RANGES}")
print("  Groups:")
print("    [5, 15] Hz: H1, H2")
print("    [15, 25] Hz: H3, H4, H5")
print("    [25, 35] Hz: H6")
print("    [27, 37] Hz: H7")

results_ranges, _ = detect_ignitions_session(
    RECORDS,
    sr_channel='EEG.F4',
    eeg_channels=['EEG.F4', 'EEG.O1', 'EEG.C3'],
    time_col='Timestamp',
    out_dir='test_output/fooof_ranges',
    harmonics_hz=CANON,
    harmonic_method='fooof_session',
    fooof_freq_range=FREQS,
    fooof_freq_ranges=FREQ_RANGES,
    fooof_max_n_peaks=15,
    fooof_peak_threshold=0.01,
    fooof_min_peak_height=0.01,
    fooof_peak_width_limits=(0.5, 12.0),
    make_passport=False,
    show=False,
    verbose=True
)

if 'events' in results_ranges and not results_ranges['events'].empty:
    print(f"\n✓ FOOOF with freq_ranges detected {len(results_ranges['events'])} events")
    if 'ignition_freqs' in results_ranges['events'].columns:
        first_event_freqs = results_ranges['events']['ignition_freqs'].iloc[0]
        print(f"  Harmonics: {[f'{f:.2f}' for f in first_event_freqs]}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

methods = [
    ('Default FOOOF', results_default),
    ('Custom High-Sensitivity', results_custom),
    ('Custom with freq_ranges', results_ranges)
]

for name, results in methods:
    if 'events' in results and not results['events'].empty:
        n_events = len(results['events'])
        first_freqs = results['events']['ignition_freqs'].iloc[0]
        if isinstance(first_freqs, str):
            first_freqs = eval(first_freqs)
        print(f"{name:30s}: {n_events} events, H1={first_freqs[0]:.2f} Hz")
    else:
        print(f"{name:30s}: No events detected")

print("\n✓ Custom parameters test complete!")
print("=" * 70)

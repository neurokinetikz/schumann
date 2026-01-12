#!/usr/bin/env python3
"""
Test fooof_match_method parameter in detect_ignitions_session.

Demonstrates all three match methods:
1. 'distance' - Pick peak closest to canonical frequency
2. 'power' - Pick highest power peak
3. 'average' - Power-weighted average of peaks
"""

import numpy as np
import pandas as pd
from lib.detect_ignition import detect_ignitions_session

# Create synthetic test data
fs = 128
duration = 60
n_samples = fs * duration
t = np.arange(n_samples) / fs

# Generate signal with Schumann harmonics
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

print("=" * 70)
print("FOOOF MATCH_METHOD TEST")
print("=" * 70)

# Test 1: distance (closest to canonical)
print("\n" + "=" * 70)
print("TEST 1: match_method='distance' (Closest to Canonical)")
print("=" * 70)

results_distance, _ = detect_ignitions_session(
    RECORDS,
    sr_channel='EEG.F4',
    eeg_channels=['EEG.F4', 'EEG.O1', 'EEG.C3'],
    time_col='Timestamp',
    out_dir='test_output/match_distance',
    harmonics_hz=CANON,
    harmonic_method='fooof_session',
    fooof_freq_range=FREQS,
    fooof_max_n_peaks=15,
    fooof_peak_threshold=0.01,
    fooof_min_peak_height=0.01,
    fooof_peak_width_limits=(0.5, 12.0),
    fooof_match_method='distance',
    make_passport=False,
    show=False,
    verbose=True
)

if 'events' in results_distance and not results_distance['events'].empty:
    print(f"\n✓ match_method='distance' detected {len(results_distance['events'])} events")
    if 'ignition_freqs' in results_distance['events'].columns:
        first_event_freqs = results_distance['events']['ignition_freqs'].iloc[0]
        print(f"  Harmonics: {[f'{f:.2f}' for f in first_event_freqs]}")

# Test 2: power (highest power)
print("\n" + "=" * 70)
print("TEST 2: match_method='power' (Highest Power)")
print("=" * 70)

results_power, _ = detect_ignitions_session(
    RECORDS,
    sr_channel='EEG.F4',
    eeg_channels=['EEG.F4', 'EEG.O1', 'EEG.C3'],
    time_col='Timestamp',
    out_dir='test_output/match_power',
    harmonics_hz=CANON,
    harmonic_method='fooof_session',
    fooof_freq_range=FREQS,
    fooof_max_n_peaks=15,
    fooof_peak_threshold=0.01,
    fooof_min_peak_height=0.01,
    fooof_peak_width_limits=(0.5, 12.0),
    fooof_match_method='power',
    make_passport=False,
    show=False,
    verbose=True
)

if 'events' in results_power and not results_power['events'].empty:
    print(f"\n✓ match_method='power' detected {len(results_power['events'])} events")
    if 'ignition_freqs' in results_power['events'].columns:
        first_event_freqs = results_power['events']['ignition_freqs'].iloc[0]
        print(f"  Harmonics: {[f'{f:.2f}' for f in first_event_freqs]}")

# Test 3: average (power-weighted average)
print("\n" + "=" * 70)
print("TEST 3: match_method='average' (Power-Weighted Average)")
print("=" * 70)

results_average, _ = detect_ignitions_session(
    RECORDS,
    sr_channel='EEG.F4',
    eeg_channels=['EEG.F4', 'EEG.O1', 'EEG.C3'],
    time_col='Timestamp',
    out_dir='test_output/match_average',
    harmonics_hz=CANON,
    harmonic_method='fooof_session',
    fooof_freq_range=FREQS,
    fooof_max_n_peaks=15,
    fooof_peak_threshold=0.01,
    fooof_min_peak_height=0.01,
    fooof_peak_width_limits=(0.5, 12.0),
    fooof_match_method='average',
    make_passport=False,
    show=False,
    verbose=True
)

if 'events' in results_average and not results_average['events'].empty:
    print(f"\n✓ match_method='average' detected {len(results_average['events'])} events")
    if 'ignition_freqs' in results_average['events'].columns:
        first_event_freqs = results_average['events']['ignition_freqs'].iloc[0]
        print(f"  Harmonics: {[f'{f:.2f}' for f in first_event_freqs]}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

methods = [
    ('distance', results_distance),
    ('power', results_power),
    ('average', results_average)
]

print("\nComparison of detected harmonics (H1-H7):\n")
print(f"{'Method':<12} {'H1':>6} {'H2':>6} {'H3':>6} {'H4':>6} {'H5':>6} {'H6':>6} {'H7':>6}")
print("-" * 70)

for name, results in methods:
    if 'events' in results and not results['events'].empty:
        first_freqs = results['events']['ignition_freqs'].iloc[0]
        if isinstance(first_freqs, str):
            first_freqs = eval(first_freqs)
        freq_str = ' '.join([f'{f:>6.2f}' for f in first_freqs[:7]])
        print(f"{name:<12} {freq_str}")

print("\n✓ match_method parameter test complete!")
print("=" * 70)

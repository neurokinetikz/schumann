"""
Test to verify that FOOOF max_n_peaks correctly displays NaN for undetected peaks
"""
import numpy as np
import pandas as pd
from lib.detect_ignition import detect_ignitions_session

# Create synthetic data with 5 harmonics, but we'll limit FOOOF to detect only 2-3
fs = 128
duration = 120  # 2 minutes
t = np.arange(0, duration, 1/fs)

# Create strong burst in middle with multiple harmonics
ignition_start = 40
ignition_end = 70
envelope = np.ones_like(t) * 0.5
envelope[(t >= ignition_start) & (t <= ignition_end)] = 10.0  # Very strong burst

signal = envelope * (
    1.0 * np.sin(2 * np.pi * 7.83 * t) +   # Strongest
    0.8 * np.sin(2 * np.pi * 14.3 * t) +   # Second strongest
    0.6 * np.sin(2 * np.pi * 20.8 * t) +   # Third strongest
    0.4 * np.sin(2 * np.pi * 27.3 * t) +   # Weaker
    0.2 * np.sin(2 * np.pi * 33.8 * t)     # Weakest
) + np.random.normal(0, 0.1, len(t))

records = pd.DataFrame({
    'Timestamp': t,
    'EEG.F4': signal + np.random.normal(0, 0.1, len(t)),
    'EEG.O1': signal + np.random.normal(0, 0.1, len(t)),
    'EEG.O2': signal + np.random.normal(0, 0.1, len(t)),
})

CANON = [7.83, 14.3, 20.8, 27.3, 33.8]

print("="*80)
print("Testing FOOOF max_n_peaks with NaN display")
print("="*80)
print(f"Canonical frequencies: {CANON}")
print(f"\nWith max_n_peaks=10, all 5 harmonics should be detected")
print(f"With max_n_peaks=2, only 2 strongest should be detected, rest show NaN")
print(f"With max_n_peaks=1, only 1 strongest should be detected, rest show NaN")
print("="*80)

# Test 1: max_n_peaks=10 (should detect all 5)
print("\n### Test 1: fooof_max_n_peaks=10 ###")
result_dict, _ = detect_ignitions_session(
    records,
    eeg_channels=['EEG.F4', 'EEG.O1', 'EEG.O2'],
    time_col='Timestamp',
    center_hz=7.83,
    harmonics_hz=CANON,
    harmonic_method='fooof_event',  # Use event-level FOOOF only
    fooof_max_n_peaks=10,
    z_thresh=2.0,  # Lower threshold
    verbose=False
)
events = result_dict['events']
if not events.empty:
    print(f"Events detected: {len(events)}")
    for idx, row in events.iterrows():
        sr_vals = [row[f'sr{i}'] for i in range(1, 6) if f'sr{i}' in row.index]
        print(f"  Event {idx} harmonics: {sr_vals}")
        nan_count = sum([1 for v in sr_vals if v == 'NaN'])
        print(f"    -> {len(sr_vals) - nan_count} detected, {nan_count} NaN")
else:
    print("No events detected")

# Test 2: max_n_peaks=2 (should detect 2 strongest, rest NaN)
print("\n### Test 2: fooof_max_n_peaks=2 ###")
result_dict, _ = detect_ignitions_session(
    records,
    eeg_channels=['EEG.F4', 'EEG.O1', 'EEG.O2'],
    time_col='Timestamp',
    center_hz=7.83,
    harmonics_hz=CANON,
    harmonic_method='fooof_event',
    fooof_max_n_peaks=2,
    z_thresh=2.0,
    verbose=False
)
events = result_dict['events']
if not events.empty:
    print(f"Events detected: {len(events)}")
    for idx, row in events.iterrows():
        sr_vals = [row[f'sr{i}'] for i in range(1, 6) if f'sr{i}' in row.index]
        print(f"  Event {idx} harmonics: {sr_vals}")
        nan_count = sum([1 for v in sr_vals if v == 'NaN'])
        print(f"    -> {len(sr_vals) - nan_count} detected, {nan_count} NaN")
else:
    print("No events detected")

# Test 3: max_n_peaks=1 (should detect 1 strongest, rest NaN)
print("\n### Test 3: fooof_max_n_peaks=1 ###")
result_dict, _ = detect_ignitions_session(
    records,
    eeg_channels=['EEG.F4', 'EEG.O1', 'EEG.O2'],
    time_col='Timestamp',
    center_hz=7.83,
    harmonics_hz=CANON,
    harmonic_method='fooof_event',
    fooof_max_n_peaks=1,
    z_thresh=2.0,
    verbose=True  # Enable verbose for this test to see what's happening
)
events = result_dict['events']
if not events.empty:
    print(f"Events detected: {len(events)}")
    for idx, row in events.iterrows():
        sr_vals = [row[f'sr{i}'] for i in range(1, 6) if f'sr{i}' in row.index]
        print(f"  Event {idx} harmonics: {sr_vals}")
        nan_count = sum([1 for v in sr_vals if v == 'NaN'])
        print(f"    -> {len(sr_vals) - nan_count} detected, {nan_count} NaN")
else:
    print("No events detected")

print("\n" + "="*80)
print("✓ Test complete! Check that NaN appears for undetected harmonics.")
print("="*80)

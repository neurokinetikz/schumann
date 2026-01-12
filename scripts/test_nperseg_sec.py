"""
Test to verify nperseg_sec parameter controls spectral resolution
"""
import numpy as np
import pandas as pd
from lib.detect_ignition import detect_ignitions_session

# Create synthetic data
fs = 128
duration = 120
t = np.arange(0, duration, 1/fs)

# Strong burst with multiple harmonics
ignition_start = 40
ignition_end = 70
envelope = np.ones_like(t) * 0.5
envelope[(t >= ignition_start) & (t <= ignition_end)] = 10.0

signal = envelope * (
    1.0 * np.sin(2 * np.pi * 7.83 * t) +
    0.8 * np.sin(2 * np.pi * 14.3 * t) +
    0.6 * np.sin(2 * np.pi * 20.8 * t) +
    0.4 * np.sin(2 * np.pi * 27.3 * t) +
    0.2 * np.sin(2 * np.pi * 33.8 * t)
) + np.random.normal(0, 0.1, len(t))

records = pd.DataFrame({
    'Timestamp': t,
    'EEG.F4': signal + np.random.normal(0, 0.1, len(t)),
    'EEG.O1': signal + np.random.normal(0, 0.1, len(t)),
    'EEG.O2': signal + np.random.normal(0, 0.1, len(t)),
})

CANON = [7.83, 14.3, 20.8, 27.3, 33.8]

print("="*80)
print("Testing nperseg_sec Parameter")
print("="*80)
print("Expected frequency resolution:")
print("  nperseg_sec=4.0  → 0.25 Hz bins (fs / (4.0 * fs) = 128 / 512)")
print("  nperseg_sec=10.0 → 0.10 Hz bins (fs / (10.0 * fs) = 128 / 1280)")
print("="*80)

# Test 1: Default resolution (nperseg_sec=4.0)
print("\n### Test 1: nperseg_sec=4.0 (default, 0.25 Hz resolution) ###")
result_dict, _ = detect_ignitions_session(
    records,
    eeg_channels=['EEG.F4', 'EEG.O1', 'EEG.O2'],
    time_col='Timestamp',
    center_hz=7.83,
    harmonics_hz=CANON,
    harmonic_method='fooof_event',
    fooof_max_n_peaks=10,
    nperseg_sec=4.0,  # Default
    z_thresh=2.0,
    verbose=False
)
events_4 = result_dict['events']
if not events_4.empty:
    print(f"Events detected: {len(events_4)}")
    for idx, row in events_4.iterrows():
        sr_vals = [row[f'sr{i}'] for i in range(1, 6) if f'sr{i}' in row.index]
        print(f"  Event {idx}: {sr_vals}")
else:
    print("No events detected")

# Test 2: Higher resolution (nperseg_sec=10.0)
print("\n### Test 2: nperseg_sec=10.0 (0.10 Hz resolution) ###")
result_dict, _ = detect_ignitions_session(
    records,
    eeg_channels=['EEG.F4', 'EEG.O1', 'EEG.O2'],
    time_col='Timestamp',
    center_hz=7.83,
    harmonics_hz=CANON,
    harmonic_method='fooof_event',
    fooof_max_n_peaks=10,
    nperseg_sec=10.0,  # Higher resolution
    z_thresh=2.0,
    verbose=False
)
events_10 = result_dict['events']
if not events_10.empty:
    print(f"Events detected: {len(events_10)}")
    for idx, row in events_10.iterrows():
        sr_vals = [row[f'sr{i}'] for i in range(1, 6) if f'sr{i}' in row.index]
        print(f"  Event {idx}: {sr_vals}")
else:
    print("No events detected")

print("\n" + "="*80)
print("✓ Parameter test complete!")
print("  Higher nperseg_sec should provide more precise frequency estimates")
print("  (differences may be subtle with synthetic data)")
print("="*80)

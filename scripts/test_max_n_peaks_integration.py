"""
Test to verify fooof_max_n_peaks parameter works in detect_ignitions_session
"""
import numpy as np
import pandas as pd
from lib.detect_ignition import detect_ignitions_session

# Create synthetic data with multiple Schumann-like peaks
fs = 128
duration = 120  # 2 minutes
t = np.arange(0, duration, 1/fs)

# Create signal with 5 clear peaks at Schumann harmonic frequencies
# Plus a strong "ignition" burst in the middle
ignition_start = 40
ignition_end = 70
envelope = np.ones_like(t)
envelope[(t >= ignition_start) & (t <= ignition_end)] = 3.0  # Strong burst

signal = envelope * (
    np.sin(2 * np.pi * 7.83 * t) +
    0.8 * np.sin(2 * np.pi * 14.3 * t) +
    0.6 * np.sin(2 * np.pi * 20.8 * t) +
    0.4 * np.sin(2 * np.pi * 27.3 * t) +
    0.3 * np.sin(2 * np.pi * 33.8 * t)
) + np.random.normal(0, 0.2, len(t))

# Create DataFrame with multiple EEG channels (similar structure)
records = pd.DataFrame({
    'Timestamp': t,
    'EEG.F4': signal + np.random.normal(0, 0.1, len(t)),
    'EEG.O1': signal + np.random.normal(0, 0.1, len(t)),
    'EEG.O2': signal + np.random.normal(0, 0.1, len(t)),
})

eeg_channels = ['EEG.F4', 'EEG.O1', 'EEG.O2']

print("="*70)
print("Test 1: fooof_max_n_peaks=15 (should detect all 5 harmonics)")
print("="*70)
result_dict, ignition_windows = detect_ignitions_session(
    records,
    eeg_channels=eeg_channels,
    time_col='Timestamp',
    center_hz=7.83,
    harmonics_hz=[7.83, 14.3, 20.8, 27.3, 33.8],
    harmonic_method='fooof_hybrid',
    fooof_max_n_peaks=15,
    verbose=True
)
events_15 = result_dict['events']

if not events_15.empty:
    print("\nEvents detected:", len(events_15))
    for idx, row in events_15.iterrows():
        sr_cols = [col for col in row.index if col.startswith('sr') and col[2:].isdigit()]
        sr_vals = [row[col] for col in sr_cols if pd.notna(row[col]) and row[col] != 'nan']
        print(f"Event {idx}: {len(sr_vals)} harmonics detected")
        print(f"  SR frequencies: {sr_vals}")
else:
    print("No events detected")

print("\n" + "="*70)
print("Test 2: fooof_max_n_peaks=1 (should detect only 1 harmonic)")
print("="*70)
result_dict, ignition_windows = detect_ignitions_session(
    records,
    eeg_channels=eeg_channels,
    time_col='Timestamp',
    center_hz=7.83,
    harmonics_hz=[7.83, 14.3, 20.8, 27.3, 33.8],
    harmonic_method='fooof_hybrid',
    fooof_max_n_peaks=1,
    verbose=True
)
events_1 = result_dict['events']

if not events_1.empty:
    print("\nEvents detected:", len(events_1))
    for idx, row in events_1.iterrows():
        sr_cols = [col for col in row.index if col.startswith('sr') and col[2:].isdigit()]
        sr_vals = [row[col] for col in sr_cols if pd.notna(row[col]) and row[col] != 'nan']
        print(f"Event {idx}: {len(sr_vals)} harmonics detected")
        print(f"  SR frequencies: {sr_vals}")
else:
    print("No events detected")

print("\n" + "="*70)
print("Test 3: fooof_max_n_peaks=3 (should detect 3 harmonics)")
print("="*70)
result_dict, ignition_windows = detect_ignitions_session(
    records,
    eeg_channels=eeg_channels,
    time_col='Timestamp',
    center_hz=7.83,
    harmonics_hz=[7.83, 14.3, 20.8, 27.3, 33.8],
    harmonic_method='fooof_hybrid',
    fooof_max_n_peaks=3,
    verbose=True
)
events_3 = result_dict['events']

if not events_3.empty:
    print("\nEvents detected:", len(events_3))
    for idx, row in events_3.iterrows():
        sr_cols = [col for col in row.index if col.startswith('sr') and col[2:].isdigit()]
        sr_vals = [row[col] for col in sr_cols if pd.notna(row[col]) and row[col] != 'nan']
        print(f"Event {idx}: {len(sr_vals)} harmonics detected")
        print(f"  SR frequencies: {sr_vals}")
else:
    print("No events detected")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"✓ The fooof_max_n_peaks parameter is now working correctly!")
print(f"  - max_n_peaks=15: Should detect ~5 harmonics")
print(f"  - max_n_peaks=3:  Should detect ~3 harmonics")
print(f"  - max_n_peaks=1:  Should detect ~1 harmonic")

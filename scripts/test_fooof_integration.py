#!/usr/bin/env python3
"""
Test FOOOF integration in detect_ignitions_session.

Demonstrates all three harmonic detection methods:
1. 'psd' - Original two-pass Welch (default)
2. 'fooof_session' - Session-level FOOOF (fast)
3. 'fooof_event' - Per-event FOOOF (accurate)
"""

import numpy as np
import pandas as pd
from lib.detect_ignition import detect_ignitions_session

# Create synthetic test data
fs = 128
duration = 60  # 60 seconds
n_samples = fs * duration
t = np.arange(n_samples) / fs

# Generate signal with Schumann harmonics
signal = np.random.randn(n_samples) * 10
signal += 8 * np.sin(2 * np.pi * 7.8 * t)    # H1
signal += 6 * np.sin(2 * np.pi * 14.5 * t)   # H2
signal += 4 * np.sin(2 * np.pi * 20.9 * t)   # H3

# Add a strong ignition event at t=30s
event_start = int(25 * fs)
event_end = int(35 * fs)
signal[event_start:event_end] += 20 * np.sin(2 * np.pi * 7.8 * t[event_start:event_end])

RECORDS = pd.DataFrame({
    'Timestamp': t,
    'EEG.F4': signal,
    'EEG.O1': signal + np.random.randn(n_samples) * 2,
    'EEG.C3': signal + np.random.randn(n_samples) * 2,
})

print("=" * 70)
print("FOOOF INTEGRATION TEST")
print("=" * 70)

# Test 1: Default PSD method
print("\n" + "=" * 70)
print("TEST 1: PSD Method (Original)")
print("=" * 70)

results_psd, windows_psd = detect_ignitions_session(
    RECORDS,
    sr_channel='EEG.F4',
    eeg_channels=['EEG.F4', 'EEG.O1', 'EEG.C3'],
    time_col='Timestamp',
    out_dir='test_output/psd',
    harmonic_method='psd',
    make_passport=False,
    show=False,
    verbose=True
)

if 'events' in results_psd and not results_psd['events'].empty:
    print(f"\n✓ PSD method detected {len(results_psd['events'])} events")
    if 'ignition_freqs' in results_psd['events'].columns:
        first_event_freqs = results_psd['events']['ignition_freqs'].iloc[0]
        print(f"  First event harmonics: {[f'{f:.2f}' for f in first_event_freqs[:3]]}")
    print(f"  Method: {results_psd['events']['harmonic_method'].iloc[0]}")

# Test 2: Session-level FOOOF
print("\n" + "=" * 70)
print("TEST 2: FOOOF Session Method (Fast)")
print("=" * 70)

results_fooof_session, windows_fooof_session = detect_ignitions_session(
    RECORDS,
    sr_channel='EEG.F4',
    eeg_channels=['EEG.F4', 'EEG.O1', 'EEG.C3'],
    time_col='Timestamp',
    out_dir='test_output/fooof_session',
    harmonic_method='fooof_session',
    make_passport=False,
    show=False,
    verbose=True
)

if 'events' in results_fooof_session and not results_fooof_session['events'].empty:
    print(f"\n✓ FOOOF session method detected {len(results_fooof_session['events'])} events")
    if 'ignition_freqs' in results_fooof_session['events'].columns:
        first_event_freqs = results_fooof_session['events']['ignition_freqs'].iloc[0]
        print(f"  First event harmonics: {[f'{f:.2f}' for f in first_event_freqs[:3]]}")
    print(f"  Method: {results_fooof_session['events']['harmonic_method'].iloc[0]}")
    if 'fooof_beta' in results_fooof_session['events'].columns:
        beta = results_fooof_session['events']['fooof_beta'].iloc[0]
        r2 = results_fooof_session['events']['fooof_r2'].iloc[0]
        print(f"  FOOOF β: {beta:.3f}, R²: {r2:.3f}")

# Test 3: Per-event FOOOF
print("\n" + "=" * 70)
print("TEST 3: FOOOF Event Method (Most Accurate)")
print("=" * 70)

results_fooof_event, windows_fooof_event = detect_ignitions_session(
    RECORDS,
    sr_channel='EEG.F4',
    eeg_channels=['EEG.F4', 'EEG.O1', 'EEG.C3'],
    time_col='Timestamp',
    out_dir='test_output/fooof_event',
    harmonic_method='fooof_event',
    make_passport=False,
    show=False,
    verbose=True
)

if 'events' in results_fooof_event and not results_fooof_event['events'].empty:
    print(f"\n✓ FOOOF event method detected {len(results_fooof_event['events'])} events")
    if 'ignition_freqs' in results_fooof_event['events'].columns:
        first_event_freqs = results_fooof_event['events']['ignition_freqs'].iloc[0]
        print(f"  First event harmonics: {[f'{f:.2f}' for f in first_event_freqs[:3]]}")
    print(f"  Method: {results_fooof_event['events']['harmonic_method'].iloc[0]}")
    if 'fooof_beta' in results_fooof_event['events'].columns:
        beta = results_fooof_event['events']['fooof_beta'].iloc[0]
        r2 = results_fooof_event['events']['fooof_r2'].iloc[0]
        print(f"  FOOOF β: {beta:.3f}, R²: {r2:.3f}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

methods = [
    ('PSD', results_psd),
    ('FOOOF Session', results_fooof_session),
    ('FOOOF Event', results_fooof_event)
]

for name, results in methods:
    if 'events' in results and not results['events'].empty:
        n_events = len(results['events'])
        print(f"{name:20s}: {n_events} events detected")
    else:
        print(f"{name:20s}: No events detected")

print("\n✓ Integration test complete!")
print("=" * 70)

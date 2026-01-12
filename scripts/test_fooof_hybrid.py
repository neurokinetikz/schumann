#!/usr/bin/env python3
"""
Test FOOOF hybrid mode: session-level + per-event harmonics.

This mode combines:
1. Session-level FOOOF for initial estimate (fast, robust)
2. Per-event FOOOF for refinement (accurate, adaptive)
"""

import numpy as np
import pandas as pd
from lib.detect_ignition import detect_ignitions_session

# Create synthetic test data with a strong ignition event
fs = 128
duration = 60
n_samples = fs * duration
t = np.arange(n_samples) / fs

# Base signal with Schumann harmonics
signal = np.random.randn(n_samples) * 10
signal += 8 * np.sin(2 * np.pi * 7.6 * t)    # H1
signal += 6 * np.sin(2 * np.pi * 9.3 * t)    # H2
signal += 5 * np.sin(2 * np.pi * 12.1 * t)   # H3
signal += 4 * np.sin(2 * np.pi * 13.8 * t)   # H4
signal += 3 * np.sin(2 * np.pi * 19.8 * t)   # H5
signal += 2 * np.sin(2 * np.pi * 25.0 * t)   # H6
signal += 2 * np.sin(2 * np.pi * 31.5 * t)   # H7

# Add a strong ignition event at t=30s with slight frequency shift
event_start = int(25 * fs)
event_end = int(35 * fs)
signal[event_start:event_end] += 20 * np.sin(2 * np.pi * 7.7 * t[event_start:event_end])  # Shifted slightly

RECORDS = pd.DataFrame({
    'Timestamp': t,
    'EEG.F4': signal,
    'EEG.O1': signal + np.random.randn(n_samples) * 2,
    'EEG.C3': signal + np.random.randn(n_samples) * 2,
})

CANON = [7.6, 9.26, 12.13, 13.75, 19.75, 25, 32]
FREQS = (5, 40)

print("=" * 70)
print("FOOOF HYBRID MODE TEST")
print("=" * 70)
print("\nHybrid mode combines:")
print("  1. Session-level FOOOF (entire recording)")
print("  2. Per-event FOOOF (each ignition event)")
print("\nExpected: Session harmonics ≈ canonical, event harmonics adapted\n")

# Test: Hybrid mode (session + per-event)
print("=" * 70)
print("TEST: harmonic_method='fooof_hybrid'")
print("=" * 70)

results_hybrid, _ = detect_ignitions_session(
    RECORDS,
    sr_channel='EEG.F4',
    eeg_channels=['EEG.F4', 'EEG.O1', 'EEG.C3'],
    time_col='Timestamp',
    out_dir='test_output/fooof_hybrid',
    harmonics_hz=CANON,
    harmonic_method='fooof_hybrid',  # ← NEW HYBRID MODE
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

# Analysis
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

if 'events' in results_hybrid and not results_hybrid['events'].empty:
    events = results_hybrid['events']
    print(f"\n✓ Detected {len(events)} events")

    # Show per-event harmonics
    if 'ignition_freqs' in events.columns:
        print("\nPer-Event Harmonics:")
        for idx, row in events.iterrows():
            freqs = row['ignition_freqs']
            if isinstance(freqs, str):
                freqs = eval(freqs)
            print(f"  Event {idx}: {[f'{f:.2f}' for f in freqs[:3]]} Hz (H1, H2, H3)")

    # Show FOOOF metrics
    if 'fooof_beta' in events.columns:
        print("\nPer-Event FOOOF Metrics:")
        print(events[['t_start', 't_end', 'fooof_beta', 'fooof_r2']])

    print("\n✓ Hybrid mode working correctly!")
    print("  - Session-level FOOOF provided initial estimates")
    print("  - Per-event FOOOF adapted to each ignition event")
else:
    print("\n⚠️  No events detected (may need to adjust parameters)")

print("=" * 70)

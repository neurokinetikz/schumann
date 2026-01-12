#!/usr/bin/env python3
"""
Test the debug wrapper with your exact configuration.
"""

# IMPORTANT: Import debug wrapper FIRST
import debug_fooof_wrapper

# Now import your modules
import numpy as np
import pandas as pd
from lib.detect_ignition import detect_ignitions_session

# Create test data with peak at 34.27 Hz
fs = 128
duration = 60
n_samples = fs * duration
t = np.arange(n_samples) / fs

signal = np.random.randn(n_samples) * 10
signal += 15 * np.sin(2 * np.pi * 7.77 * t)      # H1
signal += 12 * np.sin(2 * np.pi * 19.18 * t)     # H2
signal += 20 * np.sin(2 * np.pi * 34.2673892490006 * t)  # H3 at 34.27 Hz

# Add ignition event
event_start = int(25 * fs)
event_end = int(35 * fs)
signal[event_start:event_end] += 25 * np.sin(2 * np.pi * 7.77 * t[event_start:event_end])

RECORDS = pd.DataFrame({
    'Timestamp': t,
    'EEG.F4': signal,
    'EEG.O1': signal + np.random.randn(n_samples) * 2,
    'EEG.C3': signal + np.random.randn(n_samples) * 2,
})

_electrodes = ['EEG.F4', 'EEG.O1', 'EEG.C3']

# Your exact configuration
CANON = [7.6, 20, 32.0]
_half_bw = [0.6, 1, 2]
FREQS = (5, 40)
FREQ_RANGES = [
    [5, 15],
    [15, 25],
    [27, 37]
]

print("\n" + "=" * 70)
print("RUNNING DETECT_IGNITIONS_SESSION WITH DEBUG WRAPPER")
print("=" * 70)
print(f"Configuration:")
print(f"  CANON = {CANON}")
print(f"  _half_bw = {_half_bw}")
print(f"  FREQ_RANGES = {FREQ_RANGES}")
print(f"\nLook for debug output showing 34.27 Hz peak handling...")
print("=" * 70 + "\n")

# Run with hybrid mode
_ign_out, _ign_windows = detect_ignitions_session(
    RECORDS,
    harmonic_method='fooof_hybrid',
    eeg_channels=_electrodes,
    center_hz=CANON[0],
    harmonics_hz=CANON,
    half_bw_hz=_half_bw,
    smooth_sec=0.01,
    z_thresh=3.0,
    min_isi_sec=2.0,
    window_sec=20.0,
    merge_gap_sec=10.0,
    sr_reference='auto-SSD',
    seed_method='latency',
    pel_band=(25, 45),
    make_passport=False,
    show=False,
    verbose=True,
    session_name='test_debug_wrapper',

    # FOOOF parameters
    fooof_freq_range=FREQS,
    fooof_freq_ranges=FREQ_RANGES,
    fooof_max_n_peaks=15,
    fooof_peak_threshold=0.01,
    fooof_min_peak_height=0.01,
    fooof_peak_width_limits=(0.5, 12.0),
    fooof_match_method='power',
)

# Check results
print("\n" + "=" * 70)
print("FINAL RESULTS FROM DETECT_IGNITIONS_SESSION")
print("=" * 70)

if 'events' in _ign_out and not _ign_out['events'].empty:
    events = _ign_out['events']
    print(f"\nDetected {len(events)} events")

    for idx, row in events.head(3).iterrows():
        freqs = row['ignition_freqs']
        if isinstance(freqs, str):
            freqs = eval(freqs)

        print(f"\nEvent {idx} (t={row['t_start']:.1f}-{row['t_end']:.1f}s):")
        print(f"  ignition_freqs: {freqs}")

        for i, f in enumerate(freqs):
            table_col = f"sr{i+1}"
            expected_lo = CANON[i] - _half_bw[i]
            expected_hi = CANON[i] + _half_bw[i]
            in_range = expected_lo <= f <= expected_hi
            marker = "✓" if in_range else "❌"

            print(f"  {marker} {table_col} = {f:.4f} Hz (expected [{expected_lo:.2f}, {expected_hi:.2f}])")

            if abs(f - 34.27) < 0.01:
                print(f"      ^^^ THIS IS 34.27 Hz!")
                print(f"      Should be in range? {expected_lo:.2f} <= 34.27 <= {expected_hi:.2f}")
                print(f"      Actual: {34.27 <= expected_hi}")
                if not in_range:
                    print(f"      ❌ ERROR: Peak is OUTSIDE expected window!")
else:
    print("\nNo events detected")

print("=" * 70)

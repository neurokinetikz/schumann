#!/usr/bin/env python3
"""
Test that per-event FOOOF uses CANONICAL values for f_can, not session-detected values.

This was the bug: session FOOOF detected 34.27 Hz, then per-event FOOOF searched
around 34.27 ± 2 = [32.27, 36.27], which INCLUDES 34.27 Hz.

After fix: per-event FOOOF searches around canonical 32 ± 2 = [30, 34], which
EXCLUDES 34.27 Hz.
"""

import numpy as np
import pandas as pd
from lib.detect_ignition import detect_ignitions_session

# Create signal where session FOOOF will detect peak at 34.27 Hz
fs = 128
duration = 60
n_samples = fs * duration
t = np.arange(n_samples) / fs

signal = np.random.randn(n_samples) * 5
signal += 10 * np.sin(2 * np.pi * 7.77 * t)      # H1 near 7.6
signal += 8 * np.sin(2 * np.pi * 19.18 * t)      # H2 near 20
signal += 15 * np.sin(2 * np.pi * 34.27 * t)     # H3 at 34.27 (NOT 32!)

# Add ignition event (same frequencies, much stronger)
event_start = int(25 * fs)
event_end = int(35 * fs)
signal[event_start:event_end] += 40 * np.sin(2 * np.pi * 7.77 * t[event_start:event_end])
signal[event_start:event_end] += 30 * np.sin(2 * np.pi * 19.18 * t[event_start:event_end])
signal[event_start:event_end] += 25 * np.sin(2 * np.pi * 34.27 * t[event_start:event_end])

RECORDS = pd.DataFrame({
    'Timestamp': t,
    'EEG.F4': signal,
    'EEG.O1': signal + np.random.randn(n_samples) * 2,
    'EEG.C3': signal + np.random.randn(n_samples) * 2,
})

_electrodes = ['EEG.F4', 'EEG.O1', 'EEG.C3']

# User's configuration
CANON = [7.6, 20, 32.0]
_half_bw = [0.6, 1, 2]
FREQS = (5, 40)
FREQ_RANGES = [
    [5, 15],
    [15, 25],
    [27, 37]
]

print("=" * 70)
print("TEST: Per-Event FOOOF Should Use CANONICAL Values")
print("=" * 70)
print(f"\nConfiguration:")
print(f"  CANON = {CANON}")
print(f"  _half_bw = {_half_bw}")
print(f"\nSignal has strong peak at 34.27 Hz (NOT at canonical 32 Hz)")
print(f"\nExpected behavior:")
print(f"  Session FOOOF: Detects 34.27 Hz (follows the data)")
print(f"  Per-event FOOOF: Searches around CANON=32 ± 2 = [30, 34]")
print(f"                   → Should REJECT 34.27 Hz (outside window)")
print("=" * 70)

# Run hybrid mode
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
    session_name='test_canonical_fix',

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
print("VERIFICATION")
print("=" * 70)

if 'events' in _ign_out and not _ign_out['events'].empty:
    events = _ign_out['events']

    print(f"\nDetected {len(events)} events")

    for idx, row in events.head(1).iterrows():
        freqs = row['ignition_freqs']
        if isinstance(freqs, str):
            freqs = eval(freqs)

        print(f"\nEvent {idx} (t={row['t_start']:.1f}-{row['t_end']:.1f}s):")
        print(f"  Per-event harmonics: {[f'{f:.4f}' for f in freqs]}")

        # Check third harmonic specifically
        h3_detected = freqs[2]
        h3_expected_lo = CANON[2] - _half_bw[2]  # 32 - 2 = 30
        h3_expected_hi = CANON[2] + _half_bw[2]  # 32 + 2 = 34

        print(f"\n  Third harmonic (sr3):")
        print(f"    CANON = {CANON[2]} Hz")
        print(f"    half_bw = {_half_bw[2]}")
        print(f"    Expected window: [{h3_expected_lo:.2f}, {h3_expected_hi:.2f}]")
        print(f"    Detected: {h3_detected:.4f} Hz")

        # Test the fix
        if abs(h3_detected - 34.27) < 0.2:
            print(f"\n  ❌ FAIL: Per-event detected {h3_detected:.4f} Hz ≈ 34.27 Hz")
            print(f"     This means it searched around session-detected 34.27, not canonical 32")
            print(f"     Bug still exists!")
        elif h3_expected_lo <= h3_detected <= h3_expected_hi:
            print(f"\n  ✅ PASS: Detected {h3_detected:.4f} Hz is within [{h3_expected_lo:.2f}, {h3_expected_hi:.2f}]")
            print(f"     Per-event FOOOF correctly used canonical 32 Hz as seed")
            print(f"     34.27 Hz peak was correctly REJECTED")
            print(f"     Fix is working! 🎉")
        else:
            print(f"\n  ⚠️  Detected {h3_detected:.4f} Hz is outside expected window")
            print(f"     Unexpected result")

else:
    print("\n⚠️  No events detected")

print("\n" + "=" * 70)

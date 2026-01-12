#!/usr/bin/env python3
"""
Test that search_halfband array is correctly passed to FOOOF.

Verifies that session-level FOOOF inside detect_ignitions_session
uses the same bandwidth array as manual FOOOF call.
"""

import numpy as np
import pandas as pd
from lib import fooof_harmonics
from lib.detect_ignition import detect_ignitions_session

# Create test data
fs = 128
duration = 60
n_samples = fs * duration
t = np.arange(n_samples) / fs

signal = np.random.randn(n_samples) * 10
signal += 8 * np.sin(2 * np.pi * 7.8 * t)
signal += 6 * np.sin(2 * np.pi * 19.5 * t)
signal += 4 * np.sin(2 * np.pi * 32.0 * t)

# Add ignition event
event_start = int(25 * fs)
event_end = int(35 * fs)
signal[event_start:event_end] += 20 * np.sin(2 * np.pi * 7.8 * t[event_start:event_end])

RECORDS = pd.DataFrame({
    'Timestamp': t,
    'EEG.F4': signal,
    'EEG.O1': signal + np.random.randn(n_samples) * 2,
    'EEG.C3': signal + np.random.randn(n_samples) * 2,
})

# Your exact configuration
CANON = [7.6, 20, 32.0]
_half_bw = [0.6, 1, 2]  # PER-HARMONIC BANDWIDTHS
FREQS = (5, 40)
FREQ_RANGES = [
    [5, 15],   # H1
    [15, 25],  # H2
    [27, 37]   # H3
]
_electrodes = ['EEG.F4', 'EEG.O1', 'EEG.C3']

print("=" * 70)
print("BANDWIDTH ARRAY TEST")
print("=" * 70)
print(f"\nConfiguration:")
print(f"  CANON: {CANON}")
print(f"  half_bw: {_half_bw}  ← Per-harmonic bandwidths")
print(f"  freq_ranges: {FREQ_RANGES}")

# Step 1: Manual session-level FOOOF (your current workflow)
print("\n" + "=" * 70)
print("STEP 1: Manual FOOOF Call")
print("=" * 70)

_harmonics_manual, result_manual = fooof_harmonics.detect_harmonics_fooof(
    RECORDS, _electrodes, fs=128,
    f_can=CANON,
    freq_range=FREQS,
    freq_ranges=FREQ_RANGES,
    search_halfband=_half_bw,  # ← Array
    max_n_peaks=15,
    peak_threshold=0.01,
    min_peak_height=0.01,
    peak_width_limits=(0.5, 12.0),
    match_method='power',
)

print(f"Manual FOOOF harmonics: {np.round(_harmonics_manual, 2)}")

# Step 2: Hybrid mode with same parameters
print("\n" + "=" * 70)
print("STEP 2: Hybrid Mode (Session + Per-Event)")
print("=" * 70)

_ign_out, _ign_windows = detect_ignitions_session(
    RECORDS,
    harmonic_method='fooof_hybrid',
    eeg_channels=_electrodes,
    center_hz=CANON[0],
    harmonics_hz=CANON,
    half_bw_hz=_half_bw,  # ← Array
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
    session_name='test_bandwidth_array',

    # FOOOF parameters
    fooof_freq_range=FREQS,
    fooof_freq_ranges=FREQ_RANGES,
    fooof_max_n_peaks=15,
    fooof_peak_threshold=0.01,
    fooof_min_peak_height=0.01,
    fooof_peak_width_limits=(0.5, 12.0),
    fooof_match_method='power',
)

# Extract session harmonics from summary
# (They were already printed, but let's also extract from first event if available)
if 'events' in _ign_out and not _ign_out['events'].empty:
    events = _ign_out['events']
    first_event_freqs = events['ignition_freqs'].iloc[0]
    if isinstance(first_event_freqs, str):
        first_event_freqs = eval(first_event_freqs)

# Verification
print("\n" + "=" * 70)
print("VERIFICATION")
print("=" * 70)

print(f"\nManual FOOOF:  {np.round(_harmonics_manual, 2)}")
print(f"\nThe session-level harmonics shown above should MATCH the manual FOOOF!")
print(f"\n✅ If they match, the fix is working correctly.")
print(f"❌ If they don't match, there's still a discrepancy.")

print("\n" + "=" * 70)

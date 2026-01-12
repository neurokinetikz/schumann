#!/usr/bin/env python3
"""
Demo: Using freq_ranges with your actual parameters

This shows how to use the new freq_ranges parameter with your
CANON and HALF_BW values for grouped FOOOF windows.
"""

import numpy as np
import pandas as pd
from lib import fooof_harmonics

# Your parameters
CANON = [7.6, 9.26, 12.13, 13.75, 19.75, 25, 32]
HALF_BW = [0.5, 0.618, 0.618, 0.75, 0.75, 1.25, 1.5]

# Define grouped frequency ranges (your simpler approach)
FREQ_RANGES = [
    [5, 15],   # SR1 (H1: 7.6 Hz)
    [5, 15],   # SR2 (H2: 9.26 Hz)
    [5, 15],   # SR3 (H3: 12.13 Hz)
    [5, 15],   # SR4 (H4: 13.75 Hz)
    [15, 25],  # SR5 (H5: 19.75 Hz)
    [25, 35],  # SR6 (H6: 25 Hz) - adjusted to 35 instead of 32
    [25, 35]   # SR7 (H7: 32 Hz)
]

# Alternative grouping (what you originally described)
FREQ_RANGES_ALT = [
    [5, 15],   # SR1
    [5, 15],   # SR2
    [15, 25],  # SR3
    [22, 32],  # SR4
    [22, 32],  # SR5 (or [27,37]?)
    [22, 32],  # SR6
    [27, 37]   # SR7
]

print("=" * 70)
print("FREQ_RANGES USAGE DEMO")
print("=" * 70)

print("\nYour canonical frequencies:")
for i, f in enumerate(CANON):
    print(f"  H{i+1}: {f} Hz")

print("\nRECOMMENDED: Grouped frequency ranges (clean, non-overlapping)")
print("  [5, 15] Hz  → H1, H2, H3, H4 (1 FOOOF fit)")
print("  [15, 25] Hz → H5 (1 FOOOF fit)")
print("  [25, 35] Hz → H6, H7 (1 FOOOF fit)")
print("  Total: 3 FOOOF fits\n")

print("ALTERNATIVE: Your original grouping (overlapping)")
print("  [5, 15] Hz  → H1, H2")
print("  [15, 25] Hz → H3")
print("  [22, 32] Hz → H4, H5, H6")
print("  [27, 37] Hz → H7")
print("  Total: 4 FOOOF fits\n")

# Example usage code
print("=" * 70)
print("EXAMPLE CODE")
print("=" * 70)

code_example = '''
# Your event analysis with grouped freq_ranges
t0 = 567  # Event time

harmonics, result = fooof_harmonics.detect_harmonics_fooof(
    RECORDS, ELECTRODES, fs=128,
    window=[t0 - 10, t0 + 10],  # 10-second time window
    f_can=CANON,
    freq_ranges=FREQ_RANGES,    # NEW: Grouped 10 Hz windows
    search_halfband=HALF_BW,    # Per-harmonic peak search
    match_method='power',       # Pick strongest peak
    max_n_peaks=10
)

# Result has per-harmonic β values
print(f"Detected harmonics: {harmonics}")
print(f"β per harmonic: {result.aperiodic_exponent}")

# Check which peaks were found vs canonical fallback
for i, (h_can, h_det) in enumerate(zip(CANON, harmonics)):
    if h_det == h_can:
        print(f"⚠️  H{i+1}: No peak found, returned canonical {h_can}")
    else:
        print(f"✓ H{i+1}: {h_det:.2f} Hz (canon={h_can})")

# Diagnostic: Check what FOOOF detected
for i, (f0, fm) in enumerate(zip(CANON, result.model)):
    if fm is None:
        print(f"H{i+1}: FOOOF fit failed")
        continue

    peaks = fm.results.params.periodic.params  # specparam 2.0
    print(f"H{i+1} ({f0} Hz): FOOOF found {len(peaks)} peaks")
    for peak in peaks:
        freq, power, bw = peak
        print(f"    {freq:.2f} Hz, power={power:.3f}, bw={bw:.2f}")
'''

print(code_example)

print("=" * 70)
print("KEY DIFFERENCES vs per_harmonic_fits=True")
print("=" * 70)

comparison = """
per_harmonic_fits=True:
  - Automatic ±5 Hz window per harmonic
  - 7 separate FOOOF fits (one per harmonic)
  - Each harmonic gets independent β value
  - Example: H1 at [2.6, 12.6], H2 at [4.26, 14.26], ...

freq_ranges=[[5,15], [5,15], ...]:
  - Manual window specification
  - 3 FOOOF fits (grouped by range)
  - Harmonics in same range share β value
  - Example: H1-H4 all use [5,15] fit

When to use each:
  - Use freq_ranges when harmonics cluster in distinct bands
  - Use per_harmonic_fits for independent analysis per harmonic
  - freq_ranges is more efficient (fewer fits)
  - freq_ranges gives more stable β estimates (more data per fit)
"""

print(comparison)

print("=" * 70)
print("NEXT STEPS")
print("=" * 70)

next_steps = """
1. Try FREQ_RANGES with your actual data at t0=567
2. Check diagnostic output to see what peaks FOOOF finds
3. Adjust HALF_BW if peaks are still outside search windows
4. Compare results with per_harmonic_fits=True

Recommended HALF_BW for freq_ranges approach:
  HALF_BW = [1.0, 1.5, 2.0, 2.0, 2.5, 3.0, 3.5]

  Rationale: Since you're using wider 10 Hz FOOOF windows,
  you can afford wider search halfbands for peak matching.
"""

print(next_steps)
print("=" * 70)

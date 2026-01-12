#!/usr/bin/env python3
"""
Demo: Advanced FOOOF Peak Matching
===================================
Shows new features for handling multiple peaks in search windows:
1. Different matching methods (distance, power, average)
2. Per-harmonic search halfbands
"""

import numpy as np
import pandas as pd
from lib.fooof_harmonics import detect_harmonics_fooof, match_peaks_to_canonical

# Create synthetic data with complex peak structure
fs = 128
duration = 10
n_samples = fs * duration
t = np.arange(n_samples) / fs

# Signal with multiple close peaks around harmonics
signal = np.random.randn(n_samples) * 10
signal += 5 * np.sin(2 * np.pi * 7.8 * t)    # H1 - strong
signal += 2 * np.sin(2 * np.pi * 10.2 * t)   # Alpha - interference!
signal += 3 * np.sin(2 * np.pi * 14.1 * t)   # H2 - medium
signal += 1.5 * np.sin(2 * np.pi * 20.5 * t) # H3 variant 1 - weaker but closer
signal += 2.5 * np.sin(2 * np.pi * 21.1 * t) # H3 variant 2 - stronger but farther

RECORDS = pd.DataFrame({
    'Timestamp': t,
    'EEG.F4': signal
})

print("=" * 70)
print("ADVANCED FOOOF PEAK MATCHING DEMO")
print("=" * 70)
print("\nSimulated scenario:")
print("  - H3 region has TWO peaks: 20.5 Hz (weak) and 21.1 Hz (strong)")
print("  - Canonical H3 = 20.8 Hz")
print("  - Which peak should we choose?")

# ============================================================================
# Method 1: Distance-based (default) - picks closest
# ============================================================================
print("\n" + "=" * 70)
print("METHOD 1: Distance-based (default)")
print("=" * 70)

harmonics1, result1 = detect_harmonics_fooof(
    RECORDS, 'EEG.F4', fs=fs,
    f_can=(7.83, 14.3, 20.8, 27.3),
    freq_range=(5, 35),
    search_halfband=0.8,
    match_method='distance'  # Pick closest to canonical
)

print(f"H1: {harmonics1[0]:.2f} Hz")
print(f"H2: {harmonics1[1]:.2f} Hz")
print(f"H3: {harmonics1[2]:.2f} Hz  ← Picks CLOSEST to 20.8 Hz")
print(f"β: {result1.aperiodic_exponent:.3f}, R²: {result1.r_squared:.3f}")

# Show what peaks were actually detected
all_peaks1 = result1.model.results.params.periodic.params
print(f"\nAll {len(all_peaks1)} peaks detected:")
for peak in all_peaks1:
    print(f"  {peak[0]:6.2f} Hz | power={peak[1]:5.2f} | bw={peak[2]:4.2f} Hz")

# ============================================================================
# Method 2: Power-based - picks strongest
# ============================================================================
print("\n" + "=" * 70)
print("METHOD 2: Power-based")
print("=" * 70)

harmonics2, result2 = detect_harmonics_fooof(
    RECORDS, 'EEG.F4', fs=fs,
    f_can=(7.83, 14.3, 20.8, 27.3),
    freq_range=(5, 35),
    search_halfband=0.8,
    match_method='power'  # Pick strongest (highest power)
)

print(f"H1: {harmonics2[0]:.2f} Hz")
print(f"H2: {harmonics2[1]:.2f} Hz")
print(f"H3: {harmonics2[2]:.2f} Hz  ← Picks STRONGEST peak")
print(f"H3 power: {result2.harmonic_powers[2]:.2f}")

# ============================================================================
# Method 3: Average - weighted average of all peaks
# ============================================================================
print("\n" + "=" * 70)
print("METHOD 3: Average (power-weighted)")
print("=" * 70)

harmonics3, result3 = detect_harmonics_fooof(
    RECORDS, 'EEG.F4', fs=fs,
    f_can=(7.83, 14.3, 20.8, 27.3),
    freq_range=(5, 35),
    search_halfband=0.8,
    match_method='average'  # Average all peaks in window
)

print(f"H1: {harmonics3[0]:.2f} Hz")
print(f"H2: {harmonics3[1]:.2f} Hz")
print(f"H3: {harmonics3[2]:.2f} Hz  ← AVERAGE of peaks in window")
print(f"H3 power: {result3.harmonic_powers[2]:.2f} (averaged)")

# ============================================================================
# Method 4: Per-harmonic halfbands
# ============================================================================
print("\n" + "=" * 70)
print("METHOD 4: Per-Harmonic Search Windows")
print("=" * 70)
print("Use different search halfbands for each harmonic:")
print("  H1: ±0.4 Hz (tight - very stable)")
print("  H2: ±0.5 Hz")
print("  H3: ±0.6 Hz")
print("  H4: ±1.0 Hz (wider - more variable)")

harmonics4, result4 = detect_harmonics_fooof(
    RECORDS, 'EEG.F4', fs=fs,
    f_can=(7.83, 14.3, 20.8, 27.3),
    freq_range=(5, 35),
    search_halfband=[0.4, 0.5, 0.6, 1.0],  # Different per harmonic!
    match_method='power'
)

print(f"\nResults with per-harmonic windows:")
for i, h in enumerate(harmonics4):
    if not np.isnan(h):
        print(f"  H{i+1}: {h:.2f} Hz")

# ============================================================================
# Comparison Summary
# ============================================================================
print("\n" + "=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)

comparison = pd.DataFrame({
    'Method': ['Distance', 'Power', 'Average', 'Per-Halfband'],
    'H3': [harmonics1[2], harmonics2[2], harmonics3[2], harmonics4[2]],
    'H3_Power': [
        result1.harmonic_powers[2],
        result2.harmonic_powers[2],
        result3.harmonic_powers[2],
        result4.harmonic_powers[2]
    ]
})

print(comparison.to_string(index=False))

print("\n" + "=" * 70)
print("RECOMMENDATIONS FOR EVENT ANALYSIS")
print("=" * 70)
print("""
1. For stable Schumann detection:
   - Use method='power' (picks strongest peak)
   - Use narrower halfbands: [0.4, 0.5, 0.6, 0.8, 1.0]

2. For conservative matching:
   - Use method='distance' (default)
   - Use standard halfband: 0.8

3. For ambiguous cases:
   - Use method='average' (smooths uncertainty)
   - Or use wider halfbands to capture more variation

4. Event-by-event analysis:
   - Start with method='power' and halfband=[0.5, 0.6, 0.8, 1.0, 1.2]
   - Flag events where distance and power disagree
   - Review flagged events manually
""")

print("\n" + "=" * 70)
print("EXAMPLE: Event Analysis with Advanced Matching")
print("=" * 70)

print("""
# Recommended settings for event analysis:
event_results = []

for event in ignition_events:
    t0 = event['peak_time']

    harmonics, result = detect_harmonics_fooof(
        RECORDS, 'EEG.F4', fs=128,
        window=[t0 - 5, t0 + 5],  # 10s window
        f_can=(7.83, 14.3, 20.8, 27.3, 33.8),
        search_halfband=[0.4, 0.5, 0.6, 0.8, 1.0],  # Per-harmonic
        match_method='power',  # Pick strongest
        max_n_peaks=8  # Limit total peaks
    )

    event_results.append({
        'event_id': event['id'],
        'time': t0,
        'h1': harmonics[0],
        'h2': harmonics[1],
        'h3': harmonics[2],
        'beta': result.aperiodic_exponent,
        'r_squared': result.r_squared
    })

df = pd.DataFrame(event_results)
df.to_csv('event_harmonics_advanced.csv', index=False)
""")

print("=" * 70)
print("Demo complete!")
print("=" * 70)

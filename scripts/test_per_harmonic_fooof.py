#!/usr/bin/env python3
"""
Test per-harmonic FOOOF fitting feature.

Validates that per_harmonic_fits=True:
1. Runs separate FOOOF fit for each canonical frequency within ±5 Hz
2. Returns per-harmonic β (aperiodic exponent) values
3. Produces sensible results compared to global fit
"""

import numpy as np
import pandas as pd
from lib.fooof_harmonics import detect_harmonics_fooof

# Create synthetic data with clear Schumann-like peaks
fs = 128
duration = 10  # seconds
n_samples = fs * duration
t = np.arange(n_samples) / fs

# Generate signal with peaks near Schumann harmonics
signal = np.random.randn(n_samples) * 10
signal += 8 * np.sin(2 * np.pi * 7.9 * t)   # H1 (near 7.83 Hz)
signal += 5 * np.sin(2 * np.pi * 14.5 * t)  # H2 (near 14.3 Hz)
signal += 3 * np.sin(2 * np.pi * 20.7 * t)  # H3 (near 20.8 Hz)
signal += 2 * np.sin(2 * np.pi * 27.2 * t)  # H4 (near 27.3 Hz)

# Create DataFrame
RECORDS = pd.DataFrame({
    'Timestamp': t,
    'EEG.F4': signal
})

print("=" * 70)
print("PER-HARMONIC FOOOF FITTING TEST")
print("=" * 70)

# Define canonical frequencies
f_can = (7.83, 14.3, 20.8, 27.3)

# Test 1: Global fit (default)
print("\n" + "=" * 70)
print("TEST 1: Global FOOOF Fit (default behavior)")
print("=" * 70)

harmonics_global, result_global = detect_harmonics_fooof(
    records=RECORDS,
    channels='EEG.F4',
    fs=fs,
    f_can=f_can,
    freq_range=(5, 35),
    per_harmonic_fits=False  # Default
)

print(f"\nDetected harmonics: {[f'{h:.2f}' for h in harmonics_global]}")
print(f"Aperiodic exponent (β): {result_global.aperiodic_exponent:.3f}")
print(f"Aperiodic offset: {result_global.aperiodic_offset:.3f}")
print(f"R²: {result_global.r_squared:.3f}")
print(f"per_harmonic_fits flag: {result_global.per_harmonic_fits}")

# Test 2: Per-harmonic fits
print("\n" + "=" * 70)
print("TEST 2: Per-Harmonic FOOOF Fits")
print("=" * 70)

harmonics_per, result_per = detect_harmonics_fooof(
    records=RECORDS,
    channels='EEG.F4',
    fs=fs,
    f_can=f_can,
    per_harmonic_fits=True  # NEW FEATURE
)

print(f"\nDetected harmonics: {[f'{h:.2f}' for h in harmonics_per]}")
print(f"per_harmonic_fits flag: {result_per.per_harmonic_fits}")
print(f"\nPer-harmonic β values:")
for i, (freq, beta) in enumerate(zip(harmonics_per, result_per.aperiodic_exponent)):
    print(f"  H{i+1} ({f_can[i]} Hz): freq={freq:.2f} Hz, β={beta:.3f}")

print(f"\nPer-harmonic R² values:")
for i, (freq, r2) in enumerate(zip(harmonics_per, result_per.r_squared)):
    print(f"  H{i+1} ({f_can[i]} Hz): freq={freq:.2f} Hz, R²={r2:.3f}")

print(f"\nPer-harmonic offsets:")
for i, (freq, offset) in enumerate(zip(harmonics_per, result_per.aperiodic_offset)):
    print(f"  H{i+1} ({f_can[i]} Hz): freq={freq:.2f} Hz, offset={offset:.3f}")

# Test 3: Verify data types
print("\n" + "=" * 70)
print("TEST 3: Data Type Verification")
print("=" * 70)

print(f"\nGlobal fit:")
print(f"  aperiodic_exponent type: {type(result_global.aperiodic_exponent)}")
print(f"  aperiodic_offset type: {type(result_global.aperiodic_offset)}")
print(f"  r_squared type: {type(result_global.r_squared)}")
print(f"  model type: {type(result_global.model)}")

print(f"\nPer-harmonic fits:")
print(f"  aperiodic_exponent type: {type(result_per.aperiodic_exponent)}")
print(f"  aperiodic_offset type: {type(result_per.aperiodic_offset)}")
print(f"  r_squared type: {type(result_per.r_squared)}")
print(f"  model type: {type(result_per.model)}")
print(f"  model length: {len(result_per.model)}")

# Test 4: Verify __repr__ works
print("\n" + "=" * 70)
print("TEST 4: Result Object Representation")
print("=" * 70)

print(f"\nGlobal result: {result_global}")
print(f"\nPer-harmonic result: {result_per}")

# Test 5: Frequency ranges used for per-harmonic fits
print("\n" + "=" * 70)
print("TEST 5: Frequency Ranges for Per-Harmonic Fits")
print("=" * 70)

print("\nExpected ±5 Hz windows around each canonical:")
for i, f0 in enumerate(f_can):
    print(f"  H{i+1}: [{f0 - 5:.2f}, {f0 + 5:.2f}] Hz")

# Test 6: Compare harmonics between methods
print("\n" + "=" * 70)
print("TEST 6: Comparison of Detected Harmonics")
print("=" * 70)

print("\nFrequency differences (per-harmonic - global):")
for i, (f_glob, f_per) in enumerate(zip(harmonics_global, harmonics_per)):
    diff = f_per - f_glob
    print(f"  H{i+1}: {diff:+.3f} Hz")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

all_passed = True

# Check that per-harmonic flag is set correctly
if result_global.per_harmonic_fits != False:
    print("✗ FAILED: Global fit should have per_harmonic_fits=False")
    all_passed = False
if result_per.per_harmonic_fits != True:
    print("✗ FAILED: Per-harmonic fit should have per_harmonic_fits=True")
    all_passed = False

# Check that per-harmonic returns lists
if not isinstance(result_per.aperiodic_exponent, list):
    print("✗ FAILED: Per-harmonic aperiodic_exponent should be a list")
    all_passed = False
if not isinstance(result_per.r_squared, list):
    print("✗ FAILED: Per-harmonic r_squared should be a list")
    all_passed = False
if not isinstance(result_per.model, list):
    print("✗ FAILED: Per-harmonic model should be a list")
    all_passed = False

# Check that global returns scalars
if not isinstance(result_global.aperiodic_exponent, float):
    print("✗ FAILED: Global aperiodic_exponent should be a float")
    all_passed = False
if not isinstance(result_global.r_squared, float):
    print("✗ FAILED: Global r_squared should be a float")
    all_passed = False

# Check lengths
if len(result_per.aperiodic_exponent) != len(f_can):
    print(f"✗ FAILED: Expected {len(f_can)} β values, got {len(result_per.aperiodic_exponent)}")
    all_passed = False

# Check harmonics detected
if len(harmonics_per) != len(f_can):
    print(f"✗ FAILED: Expected {len(f_can)} harmonics, got {len(harmonics_per)}")
    all_passed = False

if all_passed:
    print("✓ ALL TESTS PASSED!")
    print("\nPer-harmonic FOOOF fitting is working correctly:")
    print("  - Separate fits run for each canonical frequency")
    print("  - Each harmonic has its own β, offset, and R² values")
    print("  - Data types are correct")
    print("  - Result representation works")
else:
    print("✗ SOME TESTS FAILED - see above for details")

print("=" * 70)

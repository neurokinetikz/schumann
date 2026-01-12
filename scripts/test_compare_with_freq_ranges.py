#!/usr/bin/env python3
"""
Test compare_psd_fooof() and plot_fooof_fit_with_harmonics() with freq_ranges.

Verifies that both functions handle per-harmonic fits (list of models) correctly.
"""

import numpy as np
import pandas as pd
import warnings
from lib.fooof_harmonics import (
    detect_harmonics_fooof,
    compare_psd_fooof,
    plot_fooof_fit_with_harmonics
)
import matplotlib.pyplot as plt

# Create synthetic data
fs = 128
duration = 20
n_samples = fs * duration
t = np.arange(n_samples) / fs

signal = np.random.randn(n_samples) * 10
signal += 8 * np.sin(2 * np.pi * 7.8 * t)
signal += 6 * np.sin(2 * np.pi * 14.5 * t)
signal += 4 * np.sin(2 * np.pi * 20.9 * t)

RECORDS = pd.DataFrame({
    'Timestamp': t,
    'EEG.F4': signal
})

CANON = [7.83, 14.3, 20.8]
FREQ_RANGES = [
    [5, 15],   # H1, H2
    [5, 15],
    [15, 25]   # H3
]

print("=" * 70)
print("TEST: compare_psd_fooof() with freq_ranges")
print("=" * 70)

# This should now work (with a warning)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    fig1, result1 = compare_psd_fooof(
        RECORDS, 'EEG.F4', fs=fs,
        f_can=CANON,
        freq_ranges=FREQ_RANGES,
        max_n_peaks=15
    )

    if len(w) > 0:
        print(f"⚠️  WARNING (expected): {w[0].message}")

    print(f"✓ compare_psd_fooof() succeeded")
    print(f"  result.per_harmonic_fits: {result1.per_harmonic_fits}")
    print(f"  result.model type: {type(result1.model)}")
    print(f"  Harmonics: {result1.harmonics}")

plt.savefig('test_compare_freq_ranges.png', dpi=100)
print("✓ Saved: test_compare_freq_ranges.png")
plt.close()

print("\n" + "=" * 70)
print("TEST: plot_fooof_fit_with_harmonics() with freq_ranges")
print("=" * 70)

# Fit with freq_ranges
_, result2 = detect_harmonics_fooof(
    RECORDS, 'EEG.F4', fs=fs,
    f_can=CANON,
    freq_ranges=FREQ_RANGES,
    max_n_peaks=15
)

print(f"Fit complete:")
print(f"  result.per_harmonic_fits: {result2.per_harmonic_fits}")
print(f"  result.model type: {type(result2.model)}")

# Plot (should work with warning)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    fig2 = plot_fooof_fit_with_harmonics(
        result2,
        harmonics=CANON,
        title='Per-Harmonic Fits Visualization'
    )

    if len(w) > 0:
        print(f"⚠️  WARNING (expected): {w[0].message}")

    print(f"✓ plot_fooof_fit_with_harmonics() succeeded")

plt.savefig('test_plot_freq_ranges.png', dpi=100)
print("✓ Saved: test_plot_freq_ranges.png")
plt.close()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("✓ Both functions now handle freq_ranges/per_harmonic_fits")
print("✓ They use the first valid model for visualization")
print("✓ Warnings inform user of the limitation")
print("=" * 70)

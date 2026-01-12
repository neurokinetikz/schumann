#!/usr/bin/env python3
"""
Demo: Adjusting FOOOF sensitivity in plotting functions

Shows how to increase max_n_peaks for compare_psd_fooof() and
plot_fooof_fit_with_harmonics().
"""

import numpy as np
import pandas as pd
from lib.fooof_harmonics import (
    detect_harmonics_fooof,
    plot_fooof_fit_with_harmonics,
    compare_psd_fooof
)
import matplotlib.pyplot as plt

# Create synthetic data with many peaks
fs = 128
duration = 20
n_samples = fs * duration
t = np.arange(n_samples) / fs

signal = np.random.randn(n_samples) * 10

# Add multiple Schumann harmonics + some extra peaks
signal += 8 * np.sin(2 * np.pi * 7.8 * t)    # H1
signal += 6 * np.sin(2 * np.pi * 14.5 * t)   # H2
signal += 4 * np.sin(2 * np.pi * 20.9 * t)   # H3
signal += 3 * np.sin(2 * np.pi * 27.2 * t)   # H4
signal += 2 * np.sin(2 * np.pi * 11.0 * t)   # Extra peak 1
signal += 2 * np.sin(2 * np.pi * 17.5 * t)   # Extra peak 2
signal += 1.5 * np.sin(2 * np.pi * 23.0 * t) # Extra peak 3
signal += 1.5 * np.sin(2 * np.pi * 31.0 * t) # Extra peak 4

RECORDS = pd.DataFrame({
    'Timestamp': t,
    'EEG.F4': signal
})

CANON = [7.83, 14.3, 20.8, 27.3]

print("=" * 70)
print("FOOOF SENSITIVITY DEMO")
print("=" * 70)

# =============================================================================
# Example 1: compare_psd_fooof with default settings
# =============================================================================
print("\n1. compare_psd_fooof() with DEFAULT sensitivity (max_n_peaks=10)")
print("-" * 70)

fig1, result1 = compare_psd_fooof(
    RECORDS, 'EEG.F4', fs=fs,
    f_can=CANON,
    title='Default Settings (max_n_peaks=10)'
)

# Check peaks detected
if hasattr(result1.model, 'results'):
    peaks1 = result1.model.results.params.periodic.params
else:
    peaks1 = result1.model.peak_params_

print(f"Peaks detected: {len(peaks1)}")
for i, peak in enumerate(peaks1):
    freq, power, bw = peak
    print(f"  Peak {i+1}: {freq:.2f} Hz, power={power:.3f}, bw={bw:.2f}")

plt.savefig('fooof_sensitivity_default.png', dpi=150, bbox_inches='tight')
print("✓ Saved: fooof_sensitivity_default.png")

# =============================================================================
# Example 2: compare_psd_fooof with HIGH sensitivity
# =============================================================================
print("\n2. compare_psd_fooof() with HIGH sensitivity (max_n_peaks=15)")
print("-" * 70)

fig2, result2 = compare_psd_fooof(
    RECORDS, 'EEG.F4', fs=fs,
    f_can=CANON,
    max_n_peaks=15,              # More peaks allowed
    peak_threshold=0.75,         # Lower threshold
    min_peak_height=0.01,        # Lower floor
    peak_width_limits=(0.5, 12.0),  # Wider range
    title='High Sensitivity (max_n_peaks=15, threshold=0.75)'
)

# Check peaks detected
if hasattr(result2.model, 'results'):
    peaks2 = result2.model.results.params.periodic.params
else:
    peaks2 = result2.model.peak_params_

print(f"Peaks detected: {len(peaks2)}")
for i, peak in enumerate(peaks2):
    freq, power, bw = peak
    print(f"  Peak {i+1}: {freq:.2f} Hz, power={power:.3f}, bw={bw:.2f}")

plt.savefig('fooof_sensitivity_high.png', dpi=150, bbox_inches='tight')
print("✓ Saved: fooof_sensitivity_high.png")

# =============================================================================
# Example 3: plot_fooof_fit_with_harmonics (need to fit first)
# =============================================================================
print("\n3. plot_fooof_fit_with_harmonics() with HIGH sensitivity")
print("-" * 70)
print("Note: Must set parameters in detect_harmonics_fooof() first!")

# Fit with high sensitivity
_, result_sensitive = detect_harmonics_fooof(
    RECORDS, 'EEG.F4', fs=fs,
    f_can=CANON,
    max_n_peaks=15,
    peak_threshold=0.75,
    min_peak_height=0.01,
    peak_width_limits=(0.5, 12.0)
)

# Now plot
fig3 = plot_fooof_fit_with_harmonics(
    result_sensitive,
    harmonics=CANON,
    title='High Sensitivity - plot_fooof_fit_with_harmonics'
)

plt.savefig('fooof_sensitivity_plot.png', dpi=150, bbox_inches='tight')
print("✓ Saved: fooof_sensitivity_plot.png")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Default settings:  {len(peaks1)} peaks detected")
print(f"High sensitivity:  {len(peaks2)} peaks detected")
print(f"Difference:        {len(peaks2) - len(peaks1)} more peaks")
print("\nParameters for high sensitivity:")
print("  max_n_peaks=15")
print("  peak_threshold=0.75")
print("  min_peak_height=0.01")
print("  peak_width_limits=(0.5, 12.0)")
print("=" * 70)

plt.show()

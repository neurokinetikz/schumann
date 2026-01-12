#!/usr/bin/env python3
"""
Demo: FOOOF Peak Labels
Shows the new frequency labels on FOOOF-detected peaks
"""

import numpy as np
import pandas as pd
from lib.fooof_harmonics import detect_harmonics_fooof, plot_fooof_fit_with_harmonics
import matplotlib.pyplot as plt

# Create synthetic data with clear peaks
fs = 128
duration = 10
n_samples = fs * duration
t = np.arange(n_samples) / fs

# Generate signal with peaks at 8, 16, 24 Hz
signal = np.random.randn(n_samples) * 10
signal += 8 * np.sin(2 * np.pi * 7.8 * t)   # H1
signal += 5 * np.sin(2 * np.pi * 15.7 * t)  # H2
signal += 3 * np.sin(2 * np.pi * 23.6 * t)  # H3

# Create DataFrame
RECORDS = pd.DataFrame({
    'Timestamp': t,
    'EEG.F4': signal,
    'EEG.O1': signal + np.random.randn(n_samples) * 2
})

print("=" * 70)
print("FOOOF Peak Labels Demo")
print("=" * 70)

# Detect harmonics
harmonics, result = detect_harmonics_fooof(
    records=RECORDS,
    channels=['EEG.F4', 'EEG.O1'],
    fs=fs,
    f_can=(7.83, 14.3, 20.8, 27.3),
    freq_range=(1, 40),
    combine='median'
)

print(f"\nDetected harmonics: {[f'{h:.2f}' for h in harmonics]}")
print(f"Aperiodic exponent (β): {result.aperiodic_exponent:.3f}")
print(f"R²: {result.r_squared:.3f}")
print(f"Number of peaks: {len([p for p in result.harmonic_powers if not np.isnan(p)])}")

# Create plot with labeled peaks
fig = plot_fooof_fit_with_harmonics(
    result,
    harmonics=(7.83, 14.3, 20.8, 27.3),
    title='FOOOF Analysis with Peak Frequency Labels',
    freq_range=(5, 35),
    log_power=True
)

print("\nPlot created! Look at Panel B (right) - the red dots now have frequency labels.")
print("Each FOOOF-detected peak shows its frequency value to 2 decimal places.")
print("\nSaving plot...")

fig.savefig('fooof_peak_labels_demo.png', dpi=300, bbox_inches='tight')
print("✓ Saved as: fooof_peak_labels_demo.png")

plt.show()

print("\n" + "=" * 70)
print("Demo complete!")
print("=" * 70)

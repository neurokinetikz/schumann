"""
Test to verify fooof_max_n_peaks parameter is working correctly
"""
import numpy as np
import pandas as pd
from lib.fooof_harmonics import detect_harmonics_fooof

# Create synthetic data with multiple peaks
fs = 128
duration = 60
t = np.arange(0, duration, 1/fs)

# Create signal with 5 clear peaks at different frequencies
signal = (
    np.sin(2 * np.pi * 7.83 * t) +   # H1
    0.8 * np.sin(2 * np.pi * 14.3 * t) +  # H2
    0.6 * np.sin(2 * np.pi * 20.8 * t) +  # H3
    0.4 * np.sin(2 * np.pi * 27.3 * t) +  # H4
    0.2 * np.sin(2 * np.pi * 33.8 * t) +  # H5
    np.random.normal(0, 0.1, len(t))
)

records = pd.DataFrame({
    'Timestamp': t,
    'EEG.F4': signal
})

# Test 1: max_n_peaks=15 (default)
print("Test 1: max_n_peaks=15 (default)")
harmonics_15, result_15 = detect_harmonics_fooof(
    records,
    channels='EEG.F4',
    fs=fs,
    f_can=(7.83, 14.3, 20.8, 27.3, 33.8),
    max_n_peaks=15,
    freq_range=(1, 50)
)
print(f"  Harmonics detected: {harmonics_15}")
print(f"  Non-NaN harmonics: {sum([not np.isnan(h) for h in harmonics_15])}")
if hasattr(result_15.model, 'peak_params_'):
    print(f"  Total peaks in FOOOF model: {len(result_15.model.peak_params_)}")
print()

# Test 2: max_n_peaks=1
print("Test 2: max_n_peaks=1")
harmonics_1, result_1 = detect_harmonics_fooof(
    records,
    channels='EEG.F4',
    fs=fs,
    f_can=(7.83, 14.3, 20.8, 27.3, 33.8),
    max_n_peaks=1,
    freq_range=(1, 50)
)
print(f"  Harmonics detected: {harmonics_1}")
print(f"  Non-NaN harmonics: {sum([not np.isnan(h) for h in harmonics_1])}")
if hasattr(result_1.model, 'peak_params_'):
    print(f"  Total peaks in FOOOF model: {len(result_1.model.peak_params_)}")
print()

# Test 3: max_n_peaks=3
print("Test 3: max_n_peaks=3")
harmonics_3, result_3 = detect_harmonics_fooof(
    records,
    channels='EEG.F4',
    fs=fs,
    f_can=(7.83, 14.3, 20.8, 27.3, 33.8),
    max_n_peaks=3,
    freq_range=(1, 50)
)
print(f"  Harmonics detected: {harmonics_3}")
print(f"  Non-NaN harmonics: {sum([not np.isnan(h) for h in harmonics_3])}")
if hasattr(result_3.model, 'peak_params_'):
    print(f"  Total peaks in FOOOF model: {len(result_3.model.peak_params_)}")
print()

# Compare results
print("COMPARISON:")
print(f"  Are results different between max_n_peaks=15 and max_n_peaks=1? {harmonics_15 != harmonics_1}")
print(f"  Are results different between max_n_peaks=15 and max_n_peaks=3? {harmonics_15 != harmonics_3}")

"""
Test to verify fooof_max_n_peaks parameter - detailed version
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

# Test with different max_n_peaks values
for max_peaks in [15, 5, 3, 1]:
    print(f"\n{'='*60}")
    print(f"max_n_peaks = {max_peaks}")
    print('='*60)

    harmonics, result = detect_harmonics_fooof(
        records,
        channels='EEG.F4',
        fs=fs,
        f_can=(7.83, 14.3, 20.8, 27.3, 33.8),
        max_n_peaks=max_peaks,
        freq_range=(1, 50)
    )

    print(f"\nHarmonics: {harmonics}")
    print(f"Powers:    {result.harmonic_powers}")
    print(f"BWs:       {result.harmonic_bandwidths}")

    if hasattr(result.model, 'peak_params_'):
        print(f"\nTotal peaks fitted by FOOOF: {len(result.model.peak_params_)}")
        print("FOOOF peak params:")
        for i, (freq, power, bw) in enumerate(result.model.peak_params_):
            print(f"  Peak {i+1}: freq={freq:.2f} Hz, power={power:.3f}, bw={bw:.2f} Hz")

    # Count actual detected peaks (non-NaN power)
    actual_detected = sum([not np.isnan(p) for p in result.harmonic_powers])
    print(f"\nActual harmonics detected (non-NaN power): {actual_detected}/5")

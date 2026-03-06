#!/usr/bin/env python3
"""
Investigate Sampling Rate Artifacts in gedBounds
================================================

Tests whether 128 Hz sampling rate creates integer Hz artifacts:

1. Random noise test: Run gedBounds on Gaussian white noise
   - If integer clustering persists → methodological artifact
   - If no clustering → EEG-specific phenomenon

2. DFT bin analysis: Check relationship between window length and frequency resolution

3. Resampled data test: Resample 128 Hz data to 125 Hz (non-harmonic)
   - If boundaries shift to non-integers → confirms 128 Hz artifact
"""

import numpy as np
import sys
sys.path.insert(0, './lib')

from true_gedbounds import (
    compute_similarity_curve,
    find_boundaries_from_similarity,
    bandpass_filter
)


def analyze_decimal_distribution(boundaries, label=""):
    """Analyze the decimal portion distribution of detected boundaries."""
    if len(boundaries) == 0:
        print(f"  {label}: No boundaries detected")
        return {}

    decimals = [b % 1 for b in boundaries]

    # Count integer proximity (within ±0.15 of .0)
    near_integer = sum(1 for d in decimals if d < 0.15 or d > 0.85)
    pct_integer = 100 * near_integer / len(boundaries)

    print(f"  {label}: {len(boundaries)} boundaries")
    print(f"    Near-integer (±0.15 Hz): {near_integer}/{len(boundaries)} = {pct_integer:.1f}%")
    print(f"    Decimal positions: {[f'{d:.2f}' for d in sorted(decimals)]}")

    return {
        'n_boundaries': len(boundaries),
        'pct_near_integer': pct_integer,
        'decimals': decimals
    }


def test_random_noise(n_channels=14, duration_sec=300, fs=128, n_trials=5):
    """
    Test 1: Run gedBounds on random Gaussian noise.

    If integer Hz clustering persists in noise → artifact is methodological.
    """
    print("\n" + "="*70)
    print("TEST 1: Random Gaussian Noise")
    print("="*70)
    print(f"  Channels: {n_channels}, Duration: {duration_sec}s, fs: {fs} Hz")
    print(f"  Running {n_trials} trials...")

    all_boundaries = []

    for trial in range(n_trials):
        # Generate random noise
        n_samples = duration_sec * fs
        noise = np.random.randn(n_channels, n_samples)

        # Run gedBounds
        frequencies, similarities = compute_similarity_curve(
            noise, fs,
            freq_range=(4.5, 45.0),
            freq_resolution=0.1,
            bandwidth=2.0,
            verbose=False
        )

        boundaries = find_boundaries_from_similarity(
            frequencies, similarities,
            prominence_percentile=25,
            min_distance_hz=2.0
        )

        all_boundaries.extend(boundaries)
        print(f"    Trial {trial+1}: {len(boundaries)} boundaries")

    result = analyze_decimal_distribution(all_boundaries, "All trials combined")

    return result


def test_pink_noise(n_channels=14, duration_sec=300, fs=128, n_trials=3):
    """
    Test 2: Run gedBounds on 1/f (pink) noise - more similar to EEG.
    """
    print("\n" + "="*70)
    print("TEST 2: Pink (1/f) Noise")
    print("="*70)

    all_boundaries = []

    for trial in range(n_trials):
        n_samples = duration_sec * fs

        # Generate pink noise via spectral shaping
        white = np.random.randn(n_channels, n_samples)

        # FFT, apply 1/f envelope, IFFT
        fft = np.fft.rfft(white, axis=1)
        freqs = np.fft.rfftfreq(n_samples, 1/fs)

        # 1/f envelope (avoid divide by zero)
        envelope = np.where(freqs > 0, 1.0 / np.sqrt(freqs), 0)
        envelope[0] = 0  # Remove DC

        pink = np.fft.irfft(fft * envelope, n=n_samples, axis=1)

        # Normalize
        pink = pink / np.std(pink, axis=1, keepdims=True)

        # Run gedBounds
        frequencies, similarities = compute_similarity_curve(
            pink, fs,
            freq_range=(4.5, 45.0),
            freq_resolution=0.1,
            bandwidth=2.0,
            verbose=False
        )

        boundaries = find_boundaries_from_similarity(
            frequencies, similarities,
            prominence_percentile=25,
            min_distance_hz=2.0
        )

        all_boundaries.extend(boundaries)
        print(f"    Trial {trial+1}: {len(boundaries)} boundaries")

    result = analyze_decimal_distribution(all_boundaries, "All trials combined")

    return result


def test_phase_shuffled_eeg(filepath=None, n_shuffles=5, fs=128):
    """
    Test 3: Run gedBounds on phase-shuffled EEG.

    Phase shuffling preserves power spectrum but destroys phase relationships.
    If integer clustering persists → it's in the spectral structure.
    """
    print("\n" + "="*70)
    print("TEST 3: Phase-Shuffled Real EEG")
    print("="*70)

    if filepath is None:
        # Find a sample EEG file
        import glob
        files = glob.glob('data/PhySF/*.csv')
        if not files:
            print("  No EEG files found in data/PhySF/")
            return {}
        filepath = files[0]

    print(f"  Using: {filepath}")

    # Load EEG directly
    import pandas as pd
    df = pd.read_csv(filepath)
    # Find EEG columns
    eeg_cols = [c for c in df.columns if c.startswith('EEG.') and
                not any(x in c for x in ['FILTERED', 'POW', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'])]
    if not eeg_cols:
        # Try columns starting with raw electrode names
        electrode_names = ['AF3','AF4','F7','F8','F3','F4','FC5','FC6','P7','P8','T7','T8','O1','O2']
        eeg_cols = [c for c in df.columns if any(c.startswith(e) for e in electrode_names)]

    if not eeg_cols:
        print(f"  No EEG columns found in {filepath}")
        return {}

    data = df[eeg_cols].values.T  # (n_channels, n_samples)
    print(f"  Shape: {data.shape}")

    all_boundaries = []

    for shuffle in range(n_shuffles):
        # Phase shuffle each channel
        shuffled = np.zeros_like(data)
        for ch in range(data.shape[0]):
            fft = np.fft.rfft(data[ch])
            # Randomize phases but keep magnitudes
            phases = np.angle(fft)
            random_phases = np.random.uniform(-np.pi, np.pi, len(phases))
            shuffled_fft = np.abs(fft) * np.exp(1j * random_phases)
            shuffled[ch] = np.fft.irfft(shuffled_fft, n=data.shape[1])

        # Run gedBounds
        frequencies, similarities = compute_similarity_curve(
            shuffled, fs,
            freq_range=(4.5, 45.0),
            freq_resolution=0.1,
            bandwidth=2.0,
            verbose=False
        )

        boundaries = find_boundaries_from_similarity(
            frequencies, similarities,
            prominence_percentile=25,
            min_distance_hz=2.0
        )

        all_boundaries.extend(boundaries)
        print(f"    Shuffle {shuffle+1}: {len(boundaries)} boundaries")

    result = analyze_decimal_distribution(all_boundaries, "All shuffles combined")

    return result


def test_different_sampling_rates():
    """
    Test 4: Compare 128 Hz vs 125 Hz vs 127 Hz sampling.

    If integer clustering only occurs at 128 Hz → fs-related artifact.
    """
    print("\n" + "="*70)
    print("TEST 4: Different Sampling Rates")
    print("="*70)

    n_channels = 14
    duration_sec = 300
    n_trials = 3

    results = {}

    for fs in [125, 127, 128, 129, 130, 256]:
        print(f"\n  fs = {fs} Hz:")
        all_boundaries = []

        for trial in range(n_trials):
            n_samples = duration_sec * fs
            noise = np.random.randn(n_channels, n_samples)

            frequencies, similarities = compute_similarity_curve(
                noise, fs,
                freq_range=(4.5, 45.0),
                freq_resolution=0.1,
                bandwidth=2.0,
                verbose=False
            )

            boundaries = find_boundaries_from_similarity(
                frequencies, similarities,
                prominence_percentile=25,
                min_distance_hz=2.0
            )

            all_boundaries.extend(boundaries)

        result = analyze_decimal_distribution(all_boundaries, f"fs={fs} Hz")
        results[fs] = result

    return results


def test_dft_spectral_analysis():
    """
    Test 5: Analyze how DFT bin spacing affects gedBounds.

    With fs=128 Hz and N=128 samples (1 sec window): bins at 0, 1, 2, ... Hz
    The covariance computed from bandpass-filtered data may have structure at these bins.
    """
    print("\n" + "="*70)
    print("TEST 5: DFT Spectral Bin Analysis")
    print("="*70)

    fs = 128

    # Check typical window lengths and their frequency resolutions
    print("\n  DFT frequency resolution for different window lengths at 128 Hz:")
    for win_sec in [0.25, 0.5, 1.0, 2.0, 4.0]:
        n_samples = int(win_sec * fs)
        freq_res = fs / n_samples
        print(f"    {win_sec:.2f}s ({n_samples} samples): Δf = {freq_res:.3f} Hz")

        # Show first few DFT bin frequencies
        bin_freqs = np.fft.rfftfreq(n_samples, 1/fs)[:10]
        print(f"      First bins: {[f'{f:.2f}' for f in bin_freqs]}")

    print("\n  The gedBounds covariance is computed on the ENTIRE signal,")
    print("  not windowed, so DFT binning should NOT directly cause this issue.")
    print("  However, the Butterworth filter might interact with spectral structure.")


def test_covariance_at_integer_vs_noninteger():
    """
    Test 6: Direct comparison of covariance similarity at integer vs non-integer Hz.

    Compute similarity between adjacent frequencies and compare
    when the midpoint is near an integer vs away from an integer.
    """
    print("\n" + "="*70)
    print("TEST 6: Covariance Similarity at Integer vs Non-Integer Hz")
    print("="*70)

    import glob
    files = glob.glob('data/PhySF/*.csv')[:5]

    if not files:
        print("  No files found")
        return {}

    import pandas as pd

    all_sims_at_integer = []
    all_sims_away = []

    for f in files:
        df = pd.read_csv(f)
        eeg_cols = [c for c in df.columns if c.startswith('EEG.') and
                    not any(x in c for x in ['FILTERED', 'POW', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'])]
        if not eeg_cols:
            continue
        data = df[eeg_cols].values.T
        fs = 128

        frequencies, similarities = compute_similarity_curve(
            data, fs,
            freq_range=(4.5, 45.0),
            freq_resolution=0.1,
            bandwidth=2.0
        )

        # Categorize similarities by proximity to integer
        for freq, sim in zip(frequencies, similarities):
            if np.isnan(sim):
                continue
            decimal = freq % 1
            if decimal < 0.15 or decimal > 0.85:  # Near integer
                all_sims_at_integer.append(sim)
            elif 0.35 < decimal < 0.65:  # Away from integer
                all_sims_away.append(sim)

    print(f"\n  Similarity at integer Hz positions (±0.15):")
    print(f"    N = {len(all_sims_at_integer)}")
    print(f"    Mean = {np.mean(all_sims_at_integer):.4f}")
    print(f"    Std = {np.std(all_sims_at_integer):.4f}")

    print(f"\n  Similarity away from integers (0.35-0.65):")
    print(f"    N = {len(all_sims_away)}")
    print(f"    Mean = {np.mean(all_sims_away):.4f}")
    print(f"    Std = {np.std(all_sims_away):.4f}")

    # T-test
    from scipy.stats import ttest_ind
    t, p = ttest_ind(all_sims_at_integer, all_sims_away)
    print(f"\n  T-test: t = {t:.3f}, p = {p:.2e}")

    if np.mean(all_sims_at_integer) < np.mean(all_sims_away):
        print("  → Similarity is LOWER at integer Hz (boundaries cluster there)")
    else:
        print("  → No clear integer Hz preference")

    return {
        'sim_at_integer': np.mean(all_sims_at_integer),
        'sim_away': np.mean(all_sims_away),
        'p_value': p
    }


def main():
    print("="*70)
    print("SAMPLING RATE ARTIFACT INVESTIGATION")
    print("="*70)
    print("\nGoal: Determine if integer Hz clustering is caused by 128 Hz sampling")

    # Test 1: Random noise
    noise_result = test_random_noise(n_trials=5)

    # Test 2: Pink noise
    pink_result = test_pink_noise(n_trials=3)

    # Test 3: Phase-shuffled EEG
    shuffled_result = test_phase_shuffled_eeg(n_shuffles=5)

    # Test 4: Different sampling rates
    fs_results = test_different_sampling_rates()

    # Test 5: DFT analysis
    test_dft_spectral_analysis()

    # Test 6: Covariance at integer vs non-integer
    cov_result = test_covariance_at_integer_vs_noninteger()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print("\n  Test Results:")
    if noise_result.get('pct_near_integer', 0) > 50:
        print("  1. Random noise: INTEGER CLUSTERING PRESENT → methodological artifact")
    else:
        print("  1. Random noise: No integer clustering → EEG-specific")

    if pink_result.get('pct_near_integer', 0) > 50:
        print("  2. Pink noise: INTEGER CLUSTERING PRESENT → 1/f doesn't cause it")
    else:
        print("  2. Pink noise: No integer clustering → spectral shape matters")

    if shuffled_result.get('pct_near_integer', 0) > 50:
        print("  3. Phase-shuffled: INTEGER CLUSTERING PRESENT → in spectral structure")
    else:
        print("  3. Phase-shuffled: No integer clustering → phase relationships matter")

    if cov_result.get('p_value', 1) < 0.05:
        print("  6. Covariance: SIGNIFICANT difference at integer Hz")
    else:
        print("  6. Covariance: No significant integer Hz preference")


if __name__ == '__main__':
    main()

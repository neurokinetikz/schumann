#!/usr/bin/env python3
"""
TRUE Continuous GED Peak Detection - No Band Boundary Gaps

The previous "continuous" implementations still had gaps at φ^n boundaries
(7.60, 12.30, 19.90, 32.19 Hz) because peak detection was done per-band.

This script fixes that by:
1. Computing eigenvalues continuously across 4.5-45 Hz
2. Detecting peaks on the FULL eigenvalue profile (not per-band)
3. Assigning bands to peaks AFTER detection (post-hoc)

This ensures peaks at band boundaries CAN be detected.
"""

import sys
sys.path.insert(0, './lib')

import numpy as np
import pandas as pd
from glob import glob
from scipy.signal import find_peaks
import os
import time

# Import GED functions
from peak_distribution_analysis import ged_continuous_sweep

# Band definitions for post-hoc assignment only
BANDS = {
    'theta':     (4.70, 7.60),
    'alpha':     (7.60, 12.30),
    'beta_low':  (12.30, 19.90),
    'beta_high': (19.90, 32.19),
    'gamma':     (32.19, 45.0),
}


def assign_band(freq: float) -> str:
    """Assign a frequency to a band (post-hoc, after peak detection)."""
    for band, (f_lo, f_hi) in BANDS.items():
        if f_lo <= freq < f_hi:
            return band
    if freq >= 45.0:
        return 'gamma'
    return 'sub_theta'


def find_ged_peaks_truly_continuous(
    records: pd.DataFrame,
    eeg_channels: list,
    fs: float = 128,
    window_sec: float = 10.0,
    step_sec: float = 5.0,
    freq_range: tuple = (4.5, 45.0),
    sweep_step_hz: float = 0.1,
    prominence_frac: float = 0.05,
    min_distance_hz: float = 0.3
) -> pd.DataFrame:
    """
    TRUE continuous GED peak detection - no band boundary artifacts.

    Key difference from previous "continuous" implementations:
    Peak detection is done on the FULL eigenvalue profile, NOT per-band.
    This allows peaks at band boundaries (7.60, 12.30, 19.90, 32.19 Hz)
    to be detected.

    Parameters
    ----------
    records : pd.DataFrame
        EEG data
    eeg_channels : list
        Channel names
    fs : float
        Sampling rate
    window_sec : float
        Window length in seconds
    step_sec : float
        Step between windows
    freq_range : tuple
        Full frequency range to sweep
    sweep_step_hz : float
        Frequency resolution for sweep
    prominence_frac : float
        Peak prominence as fraction of max eigenvalue
    min_distance_hz : float
        Minimum distance between peaks (Hz)

    Returns
    -------
    peaks_df : pd.DataFrame
        Detected peaks with columns: frequency, eigenvalue, band, window_idx
    """
    # Build channel matrix
    available = [ch for ch in eeg_channels if ch in records.columns]
    if len(available) < 3:
        # Try with EEG. prefix
        available = [f'EEG.{ch}' for ch in eeg_channels if f'EEG.{ch}' in records.columns]

    if len(available) < 3:
        return pd.DataFrame()

    X = np.vstack([records[ch].values for ch in available])

    n_samples = X.shape[1]
    window_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)

    all_peaks = []

    for win_idx, start in enumerate(range(0, n_samples - window_samples, step_samples)):
        X_win = X[:, start:start + window_samples]

        # Run continuous sweep - get eigenvalue at EVERY frequency
        freqs, eigenvalues, _, _ = ged_continuous_sweep(
            X_win, fs, freq_range, step_hz=sweep_step_hz
        )

        if len(freqs) == 0 or len(eigenvalues) == 0:
            continue

        # ============================================================
        # KEY FIX: Peak detection on FULL profile, NOT per-band
        # ============================================================
        min_dist_samples = max(1, int(min_distance_hz / sweep_step_hz))
        max_eig = np.max(eigenvalues)
        prominence = prominence_frac * max_eig if max_eig > 0 else 0.01

        # Find peaks on the ENTIRE eigenvalue profile
        peak_idx, properties = find_peaks(
            eigenvalues,
            distance=min_dist_samples,
            prominence=prominence
        )

        # Assign bands AFTER peak detection (post-hoc)
        for idx in peak_idx:
            freq = freqs[idx]
            band = assign_band(freq)
            all_peaks.append({
                'frequency': float(freq),
                'eigenvalue': float(eigenvalues[idx]),
                'band': band,
                'window_idx': win_idx,
                'prominence': float(properties['prominences'][list(peak_idx).index(idx)])
            })

    return pd.DataFrame(all_peaks)


def run_true_continuous_detection(
    files: list,
    electrodes: list,
    fs: float = 128,
    output_dir: str = None,
    **kwargs
) -> pd.DataFrame:
    """Run true continuous GED detection on multiple files."""
    all_peaks = []

    for file_path in files:
        session_name = os.path.basename(file_path)
        print(f"Processing: {session_name}")

        try:
            # Load data
            records = pd.read_csv(file_path, skiprows=0)

            eeg_cols = [c for c in records.columns
                       if c.startswith('EEG.') and
                       not any(x in c for x in ['FILTERED', 'POW', 'Alpha', 'Beta', 'Theta', 'Delta', 'Gamma'])]

            if len(eeg_cols) < 3:
                records = pd.read_csv(file_path, skiprows=1)
                eeg_cols = [c for c in records.columns
                           if c.startswith('EEG.') and
                           not any(x in c for x in ['FILTERED', 'POW', 'Alpha', 'Beta', 'Theta', 'Delta', 'Gamma'])]

            if len(eeg_cols) < 3:
                print(f"  Skipping: insufficient channels ({len(eeg_cols)})")
                continue

            # Run TRUE continuous detection
            peaks_df = find_ged_peaks_truly_continuous(
                records, eeg_cols[:14], fs=fs, **kwargs
            )

            if len(peaks_df) > 0:
                peaks_df['session'] = session_name
                all_peaks.append(peaks_df)
                print(f"  Found {len(peaks_df)} peaks")

        except Exception as e:
            print(f"  Error: {str(e)[:60]}")

    if not all_peaks:
        return pd.DataFrame()

    combined = pd.concat(all_peaks, ignore_index=True)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/ged_peaks_truly_continuous.csv"
        combined.to_csv(output_path, index=False)
        print(f"\nSaved: {output_path}")

    return combined


def verify_boundary_coverage(df: pd.DataFrame):
    """Verify that boundary frequencies are now captured."""
    print("\n" + "=" * 60)
    print("BOUNDARY COVERAGE CHECK")
    print("=" * 60)

    boundaries = [
        (7.60, "theta/alpha"),
        (12.30, "alpha/beta_low"),
        (19.90, "beta_low/beta_high"),
        (32.19, "beta_high/gamma"),
    ]

    window = 0.15  # ±0.15 Hz around boundary

    print(f"\n{'Boundary':>12}  {'Hz':>8}  {'Peaks':>8}  {'Status'}")
    print("-" * 50)

    for freq, name in boundaries:
        count = len(df[(df['frequency'] >= freq - window) & (df['frequency'] <= freq + window)])
        status = "OK" if count > 0 else "MISSING!"
        print(f"{name:>12}  {freq:>8.2f}  {count:>8}  {status}")

    # Compare to mid-band
    print("\nControl frequencies (mid-band):")
    controls = [(10.0, "mid-alpha"), (15.0, "mid-beta_low"), (25.0, "mid-beta_high")]
    for freq, name in controls:
        count = len(df[(df['frequency'] >= freq - window) & (df['frequency'] <= freq + window)])
        print(f"{name:>12}  {freq:>8.2f}  {count:>8}")


if __name__ == '__main__':
    # PhySF dataset
    PHYSF_FILES = sorted(glob('data/PhySF/**/*.csv', recursive=True))

    ELECTRODES = [
        'EEG.AF3', 'EEG.AF4', 'EEG.F7', 'EEG.F8', 'EEG.F3', 'EEG.F4',
        'EEG.FC5', 'EEG.FC6', 'EEG.P7', 'EEG.P8', 'EEG.T7', 'EEG.T8',
        'EEG.O1', 'EEG.O2'
    ]

    OUTPUT_DIR = 'exports_peak_distribution/physf_ged/truly_continuous'

    print("=" * 60)
    print("TRUE CONTINUOUS GED PEAK DETECTION")
    print("=" * 60)
    print("\nThis fixes the boundary gaps in previous 'continuous' implementations.")
    print("Peak detection is now done on the FULL eigenvalue profile, not per-band.")
    print(f"\nFiles to process: {len(PHYSF_FILES)}")
    print(f"Output: {OUTPUT_DIR}")

    start_time = time.time()

    results = run_true_continuous_detection(
        PHYSF_FILES,
        ELECTRODES,
        fs=128,
        output_dir=OUTPUT_DIR,
        window_sec=10.0,
        step_sec=5.0,
        sweep_step_hz=0.1,
        prominence_frac=0.05,
        min_distance_hz=0.3
    )

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nTotal peaks: {len(results):,}")
    print(f"Processing time: {elapsed/60:.1f} minutes")

    # Per-band counts
    print("\nPer-band counts:")
    for band in ['theta', 'alpha', 'beta_low', 'beta_high', 'gamma']:
        count = len(results[results['band'] == band])
        print(f"  {band}: {count:,}")

    # Verify boundary coverage
    verify_boundary_coverage(results)

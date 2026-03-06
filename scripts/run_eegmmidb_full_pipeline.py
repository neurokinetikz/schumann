#!/usr/bin/env python3
"""
Run our full FOOOF + GED pipeline on eegmmidb.

Produces the same CSV formats and charts as our published datasets
(PhySF, Emotions, MPENG, etc.) for direct comparison.

Usage:
  python scripts/run_eegmmidb_full_pipeline.py
  python scripts/run_eegmmidb_full_pipeline.py --mode fooof
  python scripts/run_eegmmidb_full_pipeline.py --mode ged
  python scripts/run_eegmmidb_full_pipeline.py --subjects 1 2 3
  python scripts/run_eegmmidb_full_pipeline.py --runs 2  # R02 only (eyes-closed)
"""

import os
import sys
import gc
import time
import argparse
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import mne

sys.path.insert(0, './lib')
from peak_distribution_analysis import ged_continuous_sweep
from fooof_harmonics import detect_harmonics_fooof

# =========================================================================
# CONSTANTS
# =========================================================================
N_SUBJECTS = 109
N_RUNS = 14
SFREQ = 160.0
FILTER_LO = 1.0
FILTER_HI = 55.0

FOOOF_OUTPUT_DIR = 'exports_peak_distribution/eegmmidb_fooof'
GED_OUTPUT_DIR = 'exports_peak_distribution/eegmmidb_ged/truly_continuous'

# Our standard FOOOF params (same as golden_ratio_analysis.py)
FOOOF_PARAMS = dict(
    freq_range=(1.0, 50.0),
    max_n_peaks=20,
    peak_threshold=0.001,
    min_peak_height=0.0001,
    peak_width_limits=(0.2, 20.0),
)

# GED params (same as run_all_datasets_true_continuous.py)
GED_PARAMS = dict(
    window_sec=10.0,
    step_sec=5.0,
    freq_range=(4.5, 45.0),
    sweep_step_hz=0.1,
    prominence_frac=0.05,
    min_distance_hz=0.3,
)

BANDS = {
    'theta':     (4.70, 7.60),
    'alpha':     (7.60, 12.30),
    'beta_low':  (12.30, 19.90),
    'beta_high': (19.90, 32.19),
    'gamma':     (32.19, 45.0),
}


# =========================================================================
# HELPERS
# =========================================================================

def assign_band(freq):
    """Assign a frequency to a band (post-hoc)."""
    for band, (f_lo, f_hi) in BANDS.items():
        if f_lo <= freq < f_hi:
            return band
    if freq >= 45.0:
        return 'gamma'
    return 'sub_theta'


def edf_to_records(edf_path):
    """Load an EDF file and convert to RECORDS DataFrame format.

    Returns (records_df, eeg_channels, fs) or (None, None, None) on failure.
    """
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        raw.filter(FILTER_LO, FILTER_HI, verbose=False)
        data = raw.get_data()  # (n_channels, n_samples)
        ch_names = raw.ch_names
        fs = raw.info['sfreq']

        # Build DataFrame with EEG. prefix
        records = pd.DataFrame()
        records['Timestamp'] = np.arange(data.shape[1]) / fs
        eeg_channels = []
        for i, ch in enumerate(ch_names):
            col = f'EEG.{ch}'
            records[col] = data[i]
            eeg_channels.append(col)

        return records, eeg_channels, fs
    except Exception as e:
        return None, None, None


# =========================================================================
# FOOOF EXTRACTION (per-channel, same as golden_ratio_analysis.py)
# =========================================================================

def extract_fooof_peaks_from_records(records, eeg_channels, fs, session_name,
                                     nperseg_sec=4.0):
    """Run per-channel FOOOF on a RECORDS DataFrame.

    Returns list of peak dicts with freq, power, bandwidth, channel, session, dataset.
    """
    peaks = []
    for ch in eeg_channels:
        try:
            harmonics, result = detect_harmonics_fooof(
                records=records,
                channels=[ch],
                fs=fs,
                nperseg_sec=nperseg_sec,
                **FOOOF_PARAMS,
            )
            for p in result.all_peaks:
                peaks.append({
                    'freq': p['freq'],
                    'frequency': p['freq'],  # alias for chart compatibility
                    'power': p['power'],
                    'bandwidth': p['bandwidth'],
                    'channel': ch,
                    'session': session_name,
                    'dataset': 'eegmmidb',
                })
        except Exception:
            pass
    return peaks


# =========================================================================
# GED EXTRACTION (same as run_all_datasets_true_continuous.py)
# =========================================================================

def find_ged_peaks_from_records(records, eeg_channels, fs, session_name):
    """Run true continuous GED peak detection on a RECORDS DataFrame.

    Returns list of peak dicts with frequency, eigenvalue, band, window_idx, session, dataset.
    """
    available = [ch for ch in eeg_channels if ch in records.columns]
    if len(available) < 3:
        return []

    X = np.vstack([pd.to_numeric(records[ch], errors='coerce').fillna(0).values
                   for ch in available])

    n_samples = X.shape[1]
    window_samples = int(GED_PARAMS['window_sec'] * fs)
    step_samples = int(GED_PARAMS['step_sec'] * fs)

    peaks = []

    for win_idx, start in enumerate(range(0, n_samples - window_samples, step_samples)):
        X_win = X[:, start:start + window_samples]

        freqs, eigenvalues, _, _ = ged_continuous_sweep(
            X_win, fs, GED_PARAMS['freq_range'],
            step_hz=GED_PARAMS['sweep_step_hz']
        )

        if len(freqs) == 0 or len(eigenvalues) == 0:
            continue

        min_dist_samples = max(1, int(GED_PARAMS['min_distance_hz'] / GED_PARAMS['sweep_step_hz']))
        max_eig = np.max(eigenvalues)
        prominence = GED_PARAMS['prominence_frac'] * max_eig if max_eig > 0 else 0.01

        peak_idx, _props = find_peaks(
            eigenvalues,
            distance=min_dist_samples,
            prominence=prominence
        )

        for idx in peak_idx:
            freq = freqs[idx]
            peaks.append({
                'frequency': float(freq),
                'eigenvalue': float(eigenvalues[idx]),
                'band': assign_band(freq),
                'window_idx': win_idx,
                'session': session_name,
                'dataset': 'eegmmidb',
            })

    return peaks


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run our FOOOF + GED pipeline on eegmmidb")
    parser.add_argument('--data-dir', type=str, default='data/eegmmidb')
    parser.add_argument('--mode', choices=['both', 'fooof', 'ged'], default='both')
    parser.add_argument('--subjects', type=int, nargs='+', default=None)
    parser.add_argument('--runs', type=int, nargs='+', default=None,
                        help='Runs to process (default: all 1-14)')
    parser.add_argument('--nperseg-sec', type=float, default=4.0,
                        help='Welch segment length in seconds (default: 4.0)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override FOOOF output directory')
    args = parser.parse_args()

    subjects = args.subjects or list(range(1, N_SUBJECTS + 1))
    runs = args.runs or list(range(1, N_RUNS + 1))
    do_fooof = args.mode in ('both', 'fooof')
    do_ged = args.mode in ('both', 'ged')

    fooof_output_dir = args.output_dir or FOOOF_OUTPUT_DIR
    if do_fooof:
        os.makedirs(fooof_output_dir, exist_ok=True)
    if do_ged:
        os.makedirs(GED_OUTPUT_DIR, exist_ok=True)

    print(f"Processing {len(subjects)} subjects x {len(runs)} runs")
    print(f"  Mode: {args.mode}")
    if do_fooof:
        nperseg = int(args.nperseg_sec * SFREQ)
        print(f"  FOOOF output: {fooof_output_dir}")
        print(f"  nperseg_sec: {args.nperseg_sec} -> nperseg={nperseg}, Δf={SFREQ/nperseg:.4f} Hz")
    if do_ged:
        print(f"  GED output: {GED_OUTPUT_DIR}")

    all_fooof_peaks = []
    all_ged_peaks = []
    fooof_count = 0
    ged_count = 0
    n_runs_ok = 0

    t_start = time.time()

    for si, subj in enumerate(subjects):
        for run in runs:
            edf_path = os.path.join(args.data_dir, f'S{subj:03d}',
                                    f'S{subj:03d}R{run:02d}.edf')
            if not os.path.exists(edf_path):
                continue

            session_name = f'S{subj:03d}R{run:02d}'
            records, eeg_channels, fs = edf_to_records(edf_path)
            if records is None:
                continue

            n_runs_ok += 1

            if do_fooof:
                fp = extract_fooof_peaks_from_records(
                    records, eeg_channels, fs, session_name,
                    nperseg_sec=args.nperseg_sec)
                all_fooof_peaks.extend(fp)
                fooof_count += len(fp)

            if do_ged:
                gp = find_ged_peaks_from_records(
                    records, eeg_channels, fs, session_name)
                all_ged_peaks.extend(gp)
                ged_count += len(gp)

            del records
            gc.collect()

        if (si + 1) % 10 == 0 or (si + 1) == len(subjects):
            elapsed = time.time() - t_start
            rate = (si + 1) / elapsed
            remaining = (len(subjects) - si - 1) / rate if rate > 0 else 0
            parts = []
            if do_fooof:
                parts.append(f'fooof:{fooof_count:,}')
            if do_ged:
                parts.append(f'ged:{ged_count:,}')
            print(f"  [{si+1}/{len(subjects)} subjects, {n_runs_ok} runs] "
                  f"{elapsed:.0f}s, ~{remaining:.0f}s left  "
                  f"({', '.join(parts)})")

    elapsed = time.time() - t_start

    # Save FOOOF
    if do_fooof and all_fooof_peaks:
        fooof_df = pd.DataFrame(all_fooof_peaks)
        fooof_path = os.path.join(fooof_output_dir,
                                  'golden_ratio_peaks_EEGMMIDB.csv')
        fooof_df.to_csv(fooof_path, index=False)
        print(f"\nFOOOF peaks: {len(fooof_df):,}")
        print(f"  Saved: {fooof_path}")

    # Save GED
    if do_ged and all_ged_peaks:
        ged_df = pd.DataFrame(all_ged_peaks)
        ged_path = os.path.join(GED_OUTPUT_DIR,
                                'ged_peaks_truly_continuous.csv')
        ged_df.to_csv(ged_path, index=False)
        print(f"\nGED peaks: {len(ged_df):,}")
        print(f"  Per-band:")
        for band in ['theta', 'alpha', 'beta_low', 'beta_high', 'gamma']:
            count = len(ged_df[ged_df['band'] == band])
            print(f"    {band}: {count:,}")
        print(f"  Saved: {ged_path}")

    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Generate charts
    print("\nGenerating charts...")
    import subprocess

    if do_fooof and all_fooof_peaks:
        fooof_path = os.path.join(fooof_output_dir,
                                  'golden_ratio_peaks_EEGMMIDB.csv')
        result = subprocess.run([
            'python', 'scripts/create_clean_modes_chart.py',
            fooof_path, fooof_output_dir, '7.60'
        ], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)

    if do_ged and all_ged_peaks:
        ged_path = os.path.join(GED_OUTPUT_DIR,
                                'ged_peaks_truly_continuous.csv')
        result = subprocess.run([
            'python', 'scripts/create_clean_modes_chart.py',
            ged_path, GED_OUTPUT_DIR, '7.60'
        ], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Extract EEG spectral peaks from PhysioNet eegmmidb using three methods.

Loads 109 subjects' eyes-closed resting baseline (R02) EDF files from
local data/eegmmidb/ directory, applies 1-55 Hz bandpass (matching
critic's pipeline exactly), then runs per-channel peak extraction via:
  (a) Median-filter (critic's exact code from eeg_phi.py)
  (b) FOOOF with critic's params (max_n=8, bw=[1,8], mph=0.05)
  (c) FOOOF with our params (max_n=20, bw=[0.2,20], mph=0.0001)

Outputs three CSV files compatible with run_critic_d9_on_our_data.py.

Usage:
  python scripts/extract_eegmmidb_peaks.py
  python scripts/extract_eegmmidb_peaks.py --methods medfilt fooof_critic
  python scripts/extract_eegmmidb_peaks.py --subjects 1 2 3
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from scipy.signal import welch, find_peaks, medfilt
import mne

# =========================================================================
# CONSTANTS — MATCHING CRITIC'S eeg_phi.py EXACTLY
# =========================================================================
N_SUBJECTS = 109
N_RUNS = 14          # R01-R14 per subject
SFREQ = 160.0
NPERSEG = 512
NOVERLAP = 256
FREQ_LO = 2.0
FREQ_HI = 50.0
FILTER_LO = 1.0
FILTER_HI = 55.0

OUTPUT_DIR = 'exports_peak_distribution/eegmmidb_peaks'

# =========================================================================
# SPECPARAM COMPATIBILITY
# =========================================================================
try:
    from specparam import SpectralModel
    _SPECPARAM = True
except ImportError:
    try:
        from fooof import FOOOF as SpectralModel
        _SPECPARAM = True
    except ImportError:
        _SPECPARAM = False


def _get_peak_cfs(sm):
    """Get peak center frequencies from SpectralModel, version-agnostic."""
    try:
        peaks = sm.get_params('peak')
    except Exception:
        try:
            peaks = sm.peak_params_
        except AttributeError:
            return np.array([])
    if peaks is None:
        return np.array([])
    peaks = np.asarray(peaks)
    if peaks.size == 0:
        return np.array([])
    if peaks.ndim == 1:
        return np.array([peaks[0]])
    return peaks[:, 0]


# =========================================================================
# DATA LOADING
# =========================================================================
def load_subject_data(subj, data_dir, runs=None):
    """Load a single subject's EDF files for specified runs.

    Parameters
    ----------
    subj : int
        Subject number (1-109).
    data_dir : str
        Path to eegmmidb directory.
    runs : list of int or None
        Run numbers to load (1-14). None = all 14 runs.

    Returns list of (subject_id, channel_name, signal_array) tuples.
    """
    if runs is None:
        runs = list(range(1, N_RUNS + 1))

    result = []
    for run in runs:
        edf_path = os.path.join(data_dir, f'S{subj:03d}',
                                f'S{subj:03d}R{run:02d}.edf')
        if not os.path.exists(edf_path):
            continue
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            raw.filter(FILTER_LO, FILTER_HI, verbose=False)
            data = raw.get_data()
            ch_names = raw.ch_names
            for ch in range(data.shape[0]):
                result.append((subj, ch_names[ch], data[ch]))
        except Exception as e:
            print(f"  Subject {subj} Run {run}: FAILED ({e})")
    return result


def load_all_subjects(data_dir, runs=None):
    """Load all 109 subjects' EDF data for specified runs.

    Parameters
    ----------
    runs : list of int or None
        Run numbers to load. None = all 14 runs.
    """
    run_desc = f"runs {runs}" if runs else f"all {N_RUNS} runs"
    print(f"Loading {run_desc} for {N_SUBJECTS} subjects...")
    print(f"  Data directory: {data_dir}")

    all_data = []
    failed = []

    for subj in range(1, N_SUBJECTS + 1):
        ch_data = load_subject_data(subj, data_dir, runs=runs)
        if not ch_data:
            failed.append(subj)
        else:
            all_data.extend(ch_data)
        if subj % 20 == 0:
            print(f"  {subj}/{N_SUBJECTS} ({len(all_data)} channels)")

    print(f"  Total: {len(all_data)} channel time series "
          f"from {N_SUBJECTS - len(failed)} subjects")
    if failed:
        print(f"  Failed subjects: {failed}")
    return all_data


# =========================================================================
# PEAK EXTRACTION — THREE METHODS
# =========================================================================

def extract_peaks_medfilt(signal_data):
    """Critic's exact median-filter extraction (verbatim from eeg_phi.py)."""
    freqs, psd = welch(signal_data, fs=SFREQ, nperseg=NPERSEG,
                       noverlap=NOVERLAP)
    mask = (freqs >= FREQ_LO) & (freqs <= FREQ_HI)
    freqs, psd = freqs[mask], psd[mask]

    log_psd = np.log10(psd + 1e-30)
    kernel = min(len(log_psd) // 3, 51)
    if kernel % 2 == 0:
        kernel += 1
    if kernel < 3:
        return np.array([])
    background = medfilt(log_psd, kernel_size=kernel)
    residual = log_psd - background

    mad = np.median(np.abs(residual - np.median(residual)))
    if mad < 1e-10:
        return np.array([])

    peak_idx, _props = find_peaks(residual, prominence=2 * mad, height=0)
    return freqs[peak_idx]


def extract_peaks_fooof_critic(signal_data):
    """FOOOF with critic's exact parameters."""
    freqs, psd = welch(signal_data, fs=SFREQ, nperseg=NPERSEG,
                       noverlap=NOVERLAP)
    mask = (freqs >= FREQ_LO) & (freqs <= FREQ_HI)
    freqs, psd = freqs[mask], psd[mask]
    if len(freqs) < 10 or np.all(psd < 1e-30):
        return np.array([])
    try:
        sm = SpectralModel(
            peak_width_limits=[1.0, 8.0],
            max_n_peaks=8,
            min_peak_height=0.05,
            verbose=False,
        )
        sm.fit(freqs, psd)
        return _get_peak_cfs(sm)
    except Exception:
        return np.array([])


def extract_peaks_fooof_ours(signal_data):
    """FOOOF with our sensitive parameters."""
    freqs, psd = welch(signal_data, fs=SFREQ, nperseg=NPERSEG,
                       noverlap=NOVERLAP)
    mask = (freqs >= FREQ_LO) & (freqs <= FREQ_HI)
    freqs, psd = freqs[mask], psd[mask]
    if len(freqs) < 10 or np.all(psd < 1e-30):
        return np.array([])
    try:
        sm = SpectralModel(
            peak_width_limits=[0.2, 20.0],
            max_n_peaks=20,
            min_peak_height=0.0001,
            verbose=False,
        )
        sm.fit(freqs, psd)
        return _get_peak_cfs(sm)
    except Exception:
        return np.array([])


METHODS = {
    'medfilt': {
        'func': extract_peaks_medfilt,
        'filename': 'eegmmidb_peaks_medfilt.csv',
        'label': 'Median-filter (critic exact)',
    },
    'fooof_critic': {
        'func': extract_peaks_fooof_critic,
        'filename': 'eegmmidb_peaks_fooof_critic.csv',
        'label': 'FOOOF (critic params: max8, mph=0.05, bw=[1,8])',
    },
    'fooof_ours': {
        'func': extract_peaks_fooof_ours,
        'filename': 'eegmmidb_peaks_fooof_ours.csv',
        'label': 'FOOOF (our params: max20, mph=0.0001, bw=[0.2,20])',
    },
}


# =========================================================================
# MAIN PROCESSING — STREAMING (one subject at a time to avoid OOM)
# =========================================================================

def run_extraction_streaming(data_dir, subjects, runs, methods, output_dir):
    """Process one subject at a time: load, extract peaks, free memory."""
    import gc
    os.makedirs(output_dir, exist_ok=True)

    results = {m: [] for m in methods}
    peak_counts = {m: 0 for m in methods}
    subj_peaks = {m: {} for m in methods}
    n_channels_total = 0
    n_subjects_ok = 0

    t_start = time.time()

    for si, subj in enumerate(subjects):
        ch_data = load_subject_data(subj, data_dir, runs=runs)
        if not ch_data:
            continue
        n_subjects_ok += 1

        for _subj, ch_name, signal in ch_data:
            n_channels_total += 1
            for method_name in methods:
                peaks = METHODS[method_name]['func'](signal)
                for freq in peaks:
                    results[method_name].append({
                        'freq': float(freq),
                        'channel': ch_name,
                        'subject': f'S{subj:03d}',
                    })
                peak_counts[method_name] += len(peaks)
                subj_peaks[method_name][subj] = (
                    subj_peaks[method_name].get(subj, 0) + len(peaks)
                )

        # Free raw EEG memory after each subject
        del ch_data
        gc.collect()

        if (si + 1) % 10 == 0 or (si + 1) == len(subjects):
            elapsed = time.time() - t_start
            rate = (si + 1) / elapsed
            remaining = (len(subjects) - si - 1) / rate if rate > 0 else 0
            counts = ', '.join(f'{m}:{peak_counts[m]:,}' for m in methods)
            print(f"  [{si+1}/{len(subjects)} subjects, "
                  f"{n_channels_total:,} channels] "
                  f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining  "
                  f"({counts})")

    # Save and report
    dataframes = {}
    for method_name in methods:
        info = METHODS[method_name]
        df = pd.DataFrame(results[method_name])
        csv_path = os.path.join(output_dir, info['filename'])
        df.to_csv(csv_path, index=False)
        dataframes[method_name] = df

        n_peaks = len(df)
        s_counts = np.array(list(subj_peaks[method_name].values()))
        print(f"\n  {info['label']}:")
        print(f"    Total peaks: {n_peaks:,}")
        print(f"    Subjects: {len(s_counts)}")
        if len(s_counts) > 0:
            print(f"    Peaks/subject: {s_counts.mean():.1f} +/- "
                  f"{s_counts.std():.1f} "
                  f"(range: {s_counts.min()}-{s_counts.max()})")
            print(f"    Peaks/channel: {n_peaks / max(n_channels_total,1):.2f}")
        print(f"    Saved: {csv_path}")

    return dataframes


def main():
    parser = argparse.ArgumentParser(
        description="Extract eegmmidb peaks via medfilt and FOOOF")
    parser.add_argument('--data-dir', type=str, default='data/eegmmidb',
                        help='Path to eegmmidb directory')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--methods', nargs='+',
                        default=['medfilt', 'fooof_critic', 'fooof_ours'],
                        choices=list(METHODS.keys()))
    parser.add_argument('--subjects', type=int, nargs='+', default=None,
                        help='Specific subjects (default: all 1-109)')
    parser.add_argument('--runs', type=int, nargs='+', default=None,
                        help='Specific runs (default: all 1-14)')
    args = parser.parse_args()

    # Check FOOOF availability
    fooof_methods = [m for m in args.methods if m.startswith('fooof')]
    if fooof_methods and not _SPECPARAM:
        print("ERROR: specparam (or fooof) required for FOOOF methods.")
        print("  pip install specparam")
        sys.exit(1)

    subjects = args.subjects or list(range(1, N_SUBJECTS + 1))
    run_desc = f"runs {args.runs}" if args.runs else f"all {N_RUNS} runs"
    print(f"Processing {len(subjects)} subjects x {run_desc}")
    print(f"  Methods: {args.methods}")
    print(f"  Output: {args.output_dir}")

    t0 = time.time()
    dataframes = run_extraction_streaming(
        args.data_dir, subjects, args.runs, args.methods, args.output_dir)
    elapsed = time.time() - t0
    print(f"\nTotal extraction time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    print(f"\n{'='*70}")
    print("NEXT STEP: Run D9 analysis on extracted peaks:")
    print("  python scripts/run_critic_d9_on_our_data.py --all-datasets")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

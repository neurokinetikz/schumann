#!/usr/bin/env python3
"""
Global FOOOF Peak Extraction for EEGMMIDB
==========================================

Base-agnostic extraction: single FOOOF fit on [1, freq_ceil] Hz per channel.
No phi-octave structure — peaks are extracted without any lattice bias.

For each of 109 subjects:
  - Concatenate all 14 runs (or specified subset)
  - Per channel: Welch PSD → FOOOF fit on [1, 75] Hz → extract peaks
  - Save per-subject CSV with (channel, freq, power, bandwidth)

Usage:
    python scripts/run_eegmmidb_global_fooof.py
    python scripts/run_eegmmidb_global_fooof.py --freq-ceil 50
    python scripts/run_eegmmidb_global_fooof.py --runs 1 2  # resting only
"""

import os
import sys
import time
import argparse
import logging
import warnings
import gc

import numpy as np
import pandas as pd
from scipy.signal import welch
import mne

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from lemon_utils import _get_peak_params, _get_aperiodic_params, _get_r_squared

try:
    from specparam import SpectralModel
except ImportError:
    from fooof import FOOOF as SpectralModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# =========================================================================
# CONSTANTS
# =========================================================================
N_SUBJECTS = 109
N_RUNS = 14
SFREQ = 160.0
NPERSEG = 1280          # 0.125 Hz resolution (matches overlap-trim)
FREQ_CEIL = 75.0
FREQ_FLOOR = 1.0
R2_MIN = 0.70
FILTER_LO = 1.0
NOTCH_FREQ = 60.0

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'eegmmidb')
OUTPUT_BASE = os.path.join(os.path.dirname(__file__), '..', 'exports_eegmmidb')

FOOOF_PARAMS = dict(
    peak_threshold=0.001,
    min_peak_height=0.0001,
    aperiodic_mode='fixed',
    max_n_peaks=20,
    peak_width_limits=[0.2, 12.0],
)


# =========================================================================
# DATA LOADING (reused from overlap-trim script)
# =========================================================================

def load_subject_raw(subj, data_dir, runs=None, freq_ceil=FREQ_CEIL):
    """Load and concatenate all runs for one EEGMMIDB subject."""
    if runs is None:
        runs = list(range(1, N_RUNS + 1))

    raws = []
    for run in runs:
        edf_path = os.path.join(data_dir, f'S{subj:03d}',
                                f'S{subj:03d}R{run:02d}.edf')
        if not os.path.exists(edf_path):
            continue
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            raws.append(raw)
        except Exception:
            pass

    if not raws:
        return None, {'status': 'no_data'}

    # Resample mismatched runs
    for i, raw in enumerate(raws):
        if abs(raw.info['sfreq'] - SFREQ) > 0.1:
            raws[i] = raw.resample(SFREQ, verbose=False)

    raw_concat = mne.concatenate_raws(raws) if len(raws) > 1 else raws[0]

    filter_hi = min(freq_ceil + 5, SFREQ / 2 - 1)
    raw_concat.filter(FILTER_LO, filter_hi, verbose=False)
    if filter_hi > NOTCH_FREQ - 2:
        try:
            raw_concat.notch_filter(NOTCH_FREQ, verbose=False)
        except Exception:
            pass

    return raw_concat, {
        'status': 'ok',
        'n_runs': len(raws),
        'n_channels': len(raw_concat.ch_names),
        'duration_sec': raw_concat.n_times / raw_concat.info['sfreq'],
    }


# =========================================================================
# GLOBAL FOOOF EXTRACTION
# =========================================================================

def extract_global_subject(raw_clean, fs=SFREQ, nperseg=NPERSEG,
                            overlap=0.5, freq_floor=FREQ_FLOOR,
                            freq_ceil=FREQ_CEIL, r2_min=R2_MIN):
    """Single global FOOOF fit per channel — no lattice structure."""
    ch_names = raw_clean.ch_names
    noverlap = int(nperseg * overlap)

    all_peaks = []
    n_fitted = 0
    n_passed = 0
    r2s = []

    for ch in ch_names:
        try:
            data = raw_clean.get_data(picks=[ch])[0]
        except Exception:
            continue
        if len(data) < nperseg:
            continue

        freqs, psd = welch(data, fs, nperseg=nperseg, noverlap=noverlap)

        fit_mask = (freqs >= freq_floor) & (freqs <= freq_ceil)
        if fit_mask.sum() < 20:
            continue

        n_fitted += 1
        sm = SpectralModel(**FOOOF_PARAMS)
        try:
            sm.fit(freqs, psd, [freq_floor, freq_ceil])
        except Exception:
            continue

        r2 = _get_r_squared(sm)
        if np.isnan(r2) or r2 < r2_min:
            continue

        n_passed += 1
        r2s.append(r2)

        peak_params = _get_peak_params(sm)
        for row in peak_params:
            all_peaks.append({
                'channel': ch,
                'freq': row[0],
                'power': row[1],
                'bandwidth': row[2],
            })

    peaks_df = pd.DataFrame(all_peaks) if all_peaks else pd.DataFrame(
        columns=['channel', 'freq', 'power', 'bandwidth'])

    info = {
        'n_channels_fitted': n_fitted,
        'n_channels_passed': n_passed,
        'mean_r_squared': np.mean(r2s) if r2s else np.nan,
    }
    return peaks_df, info


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Global FOOOF peak extraction for EEGMMIDB (base-agnostic)')
    parser.add_argument('--runs', type=int, nargs='+', default=None)
    parser.add_argument('--subjects', type=int, nargs='+', default=None)
    parser.add_argument('--resume-from', type=int, default=None)
    parser.add_argument('--freq-ceil', type=float, default=FREQ_CEIL)
    parser.add_argument('--data-dir', type=str, default=DATA_DIR)
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    freq_ceil = args.freq_ceil
    data_dir = os.path.abspath(args.data_dir)
    runs = args.runs

    runs_tag = f'_R{"".join(str(r) for r in runs)}' if runs else ''
    out_dir = args.output_dir or os.path.join(
        OUTPUT_BASE, f'per_subject_global_{int(freq_ceil)}hz{runs_tag}')
    os.makedirs(out_dir, exist_ok=True)

    log.info("Global FOOOF Peak Extraction — EEGMMIDB")
    log.info(f"  FOOOF range: [{FREQ_FLOOR}, {freq_ceil}] Hz (single fit per channel)")
    log.info(f"  fs = {SFREQ} Hz, nperseg = {NPERSEG}")
    log.info(f"  FOOOF params: max_n_peaks={FOOOF_PARAMS['max_n_peaks']}, "
             f"peak_threshold={FOOOF_PARAMS['peak_threshold']}")
    log.info(f"  R² threshold: {R2_MIN}")
    log.info(f"  runs: {runs or 'all 14'}")
    log.info(f"  output: {out_dir}")

    subjects = args.subjects or list(range(1, N_SUBJECTS + 1))
    if args.resume_from:
        subjects = [s for s in subjects if s >= args.resume_from]

    log.info(f"  {len(subjects)} subjects to process")

    summary_rows = []
    t_start = time.time()

    for i, subj in enumerate(subjects):
        sid = f'S{subj:03d}'
        out_path = os.path.join(out_dir, f'{sid}_peaks.csv')

        if os.path.exists(out_path):
            continue

        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            raw, load_info = load_subject_raw(subj, data_dir, runs=runs,
                                              freq_ceil=freq_ceil)
        if raw is None:
            log.warning(f"  [{i+1}/{len(subjects)}] {sid} — no data")
            summary_rows.append({'subject_id': sid, 'status': 'no_data', 'n_peaks': 0})
            continue

        try:
            peaks_df, ext_info = extract_global_subject(
                raw, fs=SFREQ, nperseg=NPERSEG, freq_ceil=freq_ceil)
        except Exception as e:
            log.error(f"  {sid}: {e}")
            summary_rows.append({'subject_id': sid, 'status': f'error: {e}', 'n_peaks': 0})
            del raw; gc.collect()
            continue

        peaks_df.to_csv(out_path, index=False)

        elapsed = time.time() - t0
        n_peaks = len(peaks_df)
        mean_r2 = ext_info['mean_r_squared']

        summary_rows.append({
            'subject_id': sid, 'status': 'ok', 'n_peaks': n_peaks,
            'n_channels_passed': ext_info['n_channels_passed'],
            'mean_r_squared': round(mean_r2, 4) if not np.isnan(mean_r2) else np.nan,
            'extraction_time_sec': round(elapsed, 1),
        })

        log.info(f"  [{i+1}/{len(subjects)}] {sid}: {n_peaks} peaks "
                 f"({ext_info['n_channels_passed']}/{ext_info['n_channels_fitted']} ch) "
                 f"R²={mean_r2:.3f} {elapsed:.1f}s")

        del raw; gc.collect()

        if (i + 1) % 10 == 0:
            pd.DataFrame(summary_rows).to_csv(
                os.path.join(out_dir, 'extraction_summary.csv'), index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, 'extraction_summary.csv'), index=False)

    total_time = time.time() - t_start
    n_ok = (summary_df['status'] == 'ok').sum() if len(summary_df) > 0 else 0
    log.info(f"\nDONE: {n_ok}/{len(subjects)} subjects in {total_time/60:.1f} min")
    if n_ok > 0:
        ok = summary_df[summary_df.status == 'ok']
        log.info(f"  Mean peaks/subject: {ok.n_peaks.mean():.0f}")
        log.info(f"  Total peaks: {ok.n_peaks.sum():,}")


if __name__ == '__main__':
    main()

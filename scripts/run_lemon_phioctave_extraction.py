#!/usr/bin/env python3
"""
Per-Phi-Octave FOOOF Extraction for LEMON
==========================================

Runs FOOOF separately within each phi-octave band [f₀×φⁿ, f₀×φⁿ⁺¹],
giving each band its own local 1/f fit. This eliminates the cross-band
redistribution artifact that plagued global [1,45] and [1,85] extractions.

Phi-octave bands at f₀=7.83 Hz:
  n  |  Range (Hz)    | Width (Hz) | Content
  -4 |  1.14 – 1.84   |  0.70      | Sub-delta (merged with n=-3)
  -3 |  1.84 – 2.98   |  1.14      | Delta (merged with n=-4)
  -2 |  2.98 – 4.82   |  1.84      | Low theta
  -1 |  4.82 – 7.83   |  3.00      | High theta
   0 |  7.83 – 12.67  |  4.83      | Alpha
   1 | 12.67 – 20.50  |  7.83      | Low beta
   2 | 20.50 – 33.16  | 12.67      | High beta
   3 | 33.16 – 53.66  | 20.50      | Low gamma
   4 | 53.66 – 86.80  | 33.14      | High gamma (capped at 85)

Usage:
    python scripts/run_lemon_phioctave_extraction.py
    python scripts/run_lemon_phioctave_extraction.py --condition EC
    python scripts/run_lemon_phioctave_extraction.py --resume-from sub-010002
    python scripts/run_lemon_phioctave_extraction.py --offset 0.5  # half-octave offset control
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))

from lemon_utils import (
    load_preprocessed_subject, LEMON_PREPROC_ROOT,
    SFREQ, WELCH_NPERSEG, FOOOF_CHANNEL_R2_MIN,
    _get_peak_params, _get_aperiodic_params, _get_r_squared,
)

try:
    from specparam import SpectralModel
    _SPECPARAM = True
except ImportError:
    from fooof import FOOOF as SpectralModel
    _SPECPARAM = True

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# Constants
PHI = (1 + np.sqrt(5)) / 2
F0_DEFAULT = 7.83  # Schumann fundamental
FREQ_CEIL = 85.0
R2_MIN = 0.70  # per-band R² threshold (can be lower than global since narrow bands are harder)

# FOOOF params for narrow bands
FOOOF_BASE_PARAMS = dict(
    peak_threshold=0.001,
    min_peak_height=0.0001,
    aperiodic_mode='fixed',
)


def make_phi_octave_bands(f0, freq_ceil=85.0, offset=0.0):
    """Generate phi-octave bands from f₀.

    Parameters
    ----------
    f0 : float
        Anchor frequency (Hz).
    freq_ceil : float
        Upper frequency limit.
    offset : float
        Fractional octave offset (0.0 = primary, 0.5 = half-octave control).

    Returns
    -------
    list of (name, n, lo, hi) tuples.
    """
    bands = []
    # Go from n=-4 up until we exceed freq_ceil
    for n in range(-4, 10):
        lo = f0 * PHI ** (n + offset)
        hi = f0 * PHI ** (n + 1 + offset)

        if lo >= freq_ceil:
            break
        if hi <= 1.0:  # below usable EEG range
            continue

        lo = max(lo, 1.0)  # clamp to 1 Hz minimum
        hi = min(hi, freq_ceil)

        width = hi - lo
        if width < 0.5:  # skip bands narrower than 0.5 Hz
            continue

        bands.append((f'n{n:+d}', n, lo, hi))

    return bands


def merge_narrow_bands(bands, min_width_hz=1.5, min_bins=12, freq_res=0.122):
    """Merge consecutive narrow bands that are too small for reliable FOOOF.

    Parameters
    ----------
    bands : list of (name, n, lo, hi)
    min_width_hz : float
        Minimum band width in Hz.
    min_bins : int
        Minimum number of frequency bins.
    freq_res : float
        Frequency resolution (Hz/bin).

    Returns
    -------
    list of (name, n_start, lo, hi) with narrow bands merged.
    """
    merged = []
    i = 0
    while i < len(bands):
        name, n, lo, hi = bands[i]
        width = hi - lo
        n_bins = int(width / freq_res)

        # Check if this band needs merging
        if width < min_width_hz or n_bins < min_bins:
            # Merge with next band(s) until wide enough
            while i + 1 < len(bands) and (hi - lo < min_width_hz or
                                           int((hi - lo) / freq_res) < min_bins):
                i += 1
                _, _, _, hi = bands[i]
            name = f'n{n:+d}_merged'

        merged.append((name, n, lo, hi))
        i += 1

    return merged


def extract_phioctave_peaks_subject(raw_clean, f0, fs=SFREQ,
                                     nperseg=WELCH_NPERSEG, overlap=0.5,
                                     freq_ceil=85.0, offset=0.0,
                                     r2_min=R2_MIN):
    """Per-channel, per-phi-octave FOOOF fitting for one subject.

    Returns (peaks_df, band_info_df).
    """
    freq_res = fs / nperseg

    # Build and merge bands
    raw_bands = make_phi_octave_bands(f0, freq_ceil, offset)
    bands = merge_narrow_bands(raw_bands, min_width_hz=1.5,
                                min_bins=12, freq_res=freq_res)

    log.debug(f"Phi-octave bands ({len(bands)}):")
    for name, n, lo, hi in bands:
        n_bins = int((hi - lo) / freq_res)
        log.debug(f"  {name}: [{lo:.2f}, {hi:.2f}] Hz  "
                  f"({hi-lo:.2f} Hz, {n_bins} bins)")

    ch_names = [ch for ch in raw_clean.ch_names if ch != 'FCz']
    noverlap = int(nperseg * overlap)

    all_peaks = []
    band_stats = []  # per-band summary

    for bname, bn, blo, bhi in bands:
        band_width = bhi - blo
        n_bins = int(band_width / freq_res)

        # Adapt max_n_peaks to band width
        # ~1 peak per 2 Hz of bandwidth, minimum 3, maximum 10
        max_n_peaks = max(3, min(10, int(band_width / 2)))

        # Adapt peak_width_limits to band width
        max_peak_width = min(band_width * 0.8, 12.0)
        peak_width_limits = [0.2, max_peak_width]

        fooof_params = {
            **FOOOF_BASE_PARAMS,
            'max_n_peaks': max_n_peaks,
            'peak_width_limits': peak_width_limits,
        }

        band_r2s = []
        band_n_peaks = []
        band_aperiodic_exps = []
        n_fitted = 0
        n_passed = 0

        for ch in ch_names:
            try:
                data = raw_clean.get_data(picks=[ch])[0]
            except Exception:
                continue

            if len(data) < nperseg:
                continue

            freqs, psd = welch(data, fs, nperseg=nperseg, noverlap=noverlap)

            # Check we have enough bins in this band
            band_mask = (freqs >= blo) & (freqs <= bhi)
            if band_mask.sum() < 8:
                continue

            n_fitted += 1

            sm = SpectralModel(**fooof_params)
            try:
                sm.fit(freqs, psd, [blo, bhi])
            except Exception as e:
                log.debug(f"FOOOF failed on {ch} band {bname}: {e}")
                continue

            r2 = _get_r_squared(sm)
            if np.isnan(r2) or r2 < r2_min:
                continue

            n_passed += 1
            band_r2s.append(r2)

            # Aperiodic
            offset_val, exponent = _get_aperiodic_params(sm)
            band_aperiodic_exps.append(exponent)

            # Peaks
            peak_params = _get_peak_params(sm)
            for row in peak_params:
                all_peaks.append({
                    'channel': ch,
                    'freq': row[0],
                    'power': row[1],
                    'bandwidth': row[2],
                    'phi_octave': bname,
                    'phi_octave_n': bn,
                    'band_lo': blo,
                    'band_hi': bhi,
                })
            band_n_peaks.append(len(peak_params))

        band_stats.append({
            'band_name': bname,
            'band_n': bn,
            'band_lo': round(blo, 3),
            'band_hi': round(bhi, 3),
            'band_width': round(band_width, 3),
            'max_n_peaks': max_n_peaks,
            'n_channels_fitted': n_fitted,
            'n_channels_passed': n_passed,
            'mean_r_squared': np.mean(band_r2s) if band_r2s else np.nan,
            'mean_aperiodic_exponent': (np.mean(band_aperiodic_exps)
                                        if band_aperiodic_exps else np.nan),
            'total_peaks': sum(band_n_peaks),
            'mean_peaks_per_channel': (np.mean(band_n_peaks)
                                       if band_n_peaks else 0),
        })

    peaks_df = pd.DataFrame(all_peaks) if all_peaks else pd.DataFrame(
        columns=['channel', 'freq', 'power', 'bandwidth',
                 'phi_octave', 'phi_octave_n', 'band_lo', 'band_hi'])
    band_info_df = pd.DataFrame(band_stats)

    return peaks_df, band_info_df


def main():
    parser = argparse.ArgumentParser(
        description='Per-phi-octave FOOOF extraction for LEMON')
    parser.add_argument('--f0', type=float, default=F0_DEFAULT,
                        help=f'Anchor frequency (default: {F0_DEFAULT})')
    parser.add_argument('--condition', type=str, default='EO',
                        choices=['EO', 'EC'],
                        help='EO or EC (default: EO)')
    parser.add_argument('--offset', type=float, default=0.0,
                        help='Fractional octave offset (0.0=primary, 0.5=control)')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume from subject ID')
    parser.add_argument('--freq-ceil', type=float, default=FREQ_CEIL,
                        help=f'Upper frequency limit (default: {FREQ_CEIL})')
    args = parser.parse_args()

    f0 = args.f0
    condition = args.condition
    offset = args.offset
    freq_ceil = args.freq_ceil

    # Output directory
    offset_tag = f'_offset{offset:.1f}' if offset != 0.0 else ''
    out_dir = os.path.join(
        os.path.dirname(__file__), '..', 'exports_lemon',
        f'per_subject_phioctave_f0{f0:.2f}{offset_tag}')
    os.makedirs(out_dir, exist_ok=True)

    log.info(f"Per-Phi-Octave FOOOF Extraction")
    log.info(f"  f₀ = {f0} Hz")
    log.info(f"  condition = {condition}")
    log.info(f"  offset = {offset}")
    log.info(f"  freq_ceil = {freq_ceil} Hz")
    log.info(f"  output: {out_dir}")

    # Show band structure
    raw_bands = make_phi_octave_bands(f0, freq_ceil, offset)
    freq_res = SFREQ / WELCH_NPERSEG
    merged_bands = merge_narrow_bands(raw_bands, min_width_hz=1.5,
                                       min_bins=12, freq_res=freq_res)
    log.info(f"\nPhi-octave bands ({len(merged_bands)}):")
    for name, n, lo, hi in merged_bands:
        n_bins = int((hi - lo) / freq_res)
        log.info(f"  {name:>12}: [{lo:6.2f}, {hi:6.2f}] Hz  "
                 f"({hi-lo:5.2f} Hz, {n_bins:3d} bins)")

    # List subjects
    preproc_dir = LEMON_PREPROC_ROOT
    subjects = sorted(set(
        f.replace(f'_{condition}.set', '')
        for f in os.listdir(preproc_dir)
        if f.endswith(f'_{condition}.set') and not f.startswith('.')
    ))
    log.info(f"\nFound {len(subjects)} subjects with {condition} data")

    # Resume support
    if args.resume_from:
        try:
            idx = subjects.index(args.resume_from)
            subjects = subjects[idx:]
            log.info(f"Resuming from {args.resume_from} ({len(subjects)} remaining)")
        except ValueError:
            log.warning(f"Subject {args.resume_from} not found, starting from beginning")

    # Suffix for condition
    suffix = '_peaks' if condition == 'EO' else f'_peaks_{condition.lower()}'

    summary_rows = []
    t_start = time.time()

    for i, sid in enumerate(subjects):
        out_path = os.path.join(out_dir, f'{sid}{suffix}.csv')
        band_path = os.path.join(out_dir, f'{sid}_band_info.csv')

        # Skip if already done
        if os.path.exists(out_path):
            log.debug(f"Skipping {sid} (exists)")
            continue

        t0 = time.time()

        # Load
        raw, info = load_preprocessed_subject(sid, preproc_dir, condition)
        if raw is None:
            log.warning(f"  {sid}: no data")
            summary_rows.append({
                'subject_id': sid, 'condition': condition,
                'status': 'no_data', 'n_peaks': 0,
            })
            continue

        # Extract
        try:
            peaks_df, band_info_df = extract_phioctave_peaks_subject(
                raw, f0=f0, fs=SFREQ, nperseg=WELCH_NPERSEG,
                overlap=0.5, freq_ceil=freq_ceil, offset=offset,
                r2_min=R2_MIN)
        except Exception as e:
            log.error(f"  {sid}: extraction failed: {e}")
            summary_rows.append({
                'subject_id': sid, 'condition': condition,
                'status': f'error: {e}', 'n_peaks': 0,
            })
            del raw
            gc.collect()
            continue

        # Save peaks
        peaks_df.to_csv(out_path, index=False)
        band_info_df.to_csv(band_path, index=False)

        elapsed = time.time() - t0
        n_peaks = len(peaks_df)
        n_bands_ok = (band_info_df['n_channels_passed'] > 0).sum()
        mean_r2 = band_info_df['mean_r_squared'].mean()
        total_bands = len(band_info_df)

        summary_rows.append({
            'subject_id': sid,
            'condition': condition,
            'status': 'ok',
            'n_peaks': n_peaks,
            'n_bands_total': total_bands,
            'n_bands_with_peaks': n_bands_ok,
            'mean_r_squared': round(mean_r2, 4) if not np.isnan(mean_r2) else np.nan,
            'duration_sec': round(info['duration_sec'], 1),
            'extraction_time_sec': round(elapsed, 1),
        })

        # Per-band peak counts for logging
        band_counts = band_info_df.set_index('band_name')['total_peaks'].to_dict()
        band_str = ' '.join(f'{k}:{v}' for k, v in band_counts.items())

        log.info(f"  [{i+1}/{len(subjects)}] {sid}: {n_peaks} peaks "
                 f"({n_bands_ok}/{total_bands} bands) R²={mean_r2:.3f} "
                 f"{elapsed:.1f}s  [{band_str}]")

        del raw
        gc.collect()

        # Save summary incrementally
        if (i + 1) % 10 == 0:
            summary_df = pd.DataFrame(summary_rows)
            summary_path = os.path.join(
                out_dir, f'extraction_summary_{condition.lower()}.csv')
            summary_df.to_csv(summary_path, index=False)

    # Final summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(
        out_dir, f'extraction_summary_{condition.lower()}.csv')
    summary_df.to_csv(summary_path, index=False)

    total_time = time.time() - t_start
    n_ok = (summary_df['status'] == 'ok').sum() if len(summary_df) > 0 else 0
    log.info(f"\n{'='*60}")
    log.info(f"DONE: {n_ok}/{len(subjects)} subjects in {total_time/60:.1f} min")

    if n_ok > 0:
        ok_df = summary_df[summary_df['status'] == 'ok']
        log.info(f"  Mean peaks/subject: {ok_df['n_peaks'].mean():.0f}")
        log.info(f"  Mean R²: {ok_df['mean_r_squared'].mean():.3f}")
        log.info(f"  Mean extraction time: {ok_df['extraction_time_sec'].mean():.1f}s")

    log.info(f"  Output: {out_dir}")
    log.info(f"  Summary: {summary_path}")


if __name__ == '__main__':
    main()

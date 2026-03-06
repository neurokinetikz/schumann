#!/usr/bin/env python3
"""
Overlap-Trim Per-Phi-Octave FOOOF Extraction for LEMON
=======================================================

Solves the fundamental tension: global fitting misspecifies the aperiodic
component, but per-band fitting creates edge artifacts at every boundary.

Solution: decouple the fitting window from the analysis window.

For each target phi-octave [f₀×φⁿ, f₀×φⁿ⁺¹]:
  1. Fit FOOOF on a WIDER window: [f₀×φ^(n-0.5), f₀×φ^(n+1.5)]
     (target band + half phi-octave padding on each side)
  2. The aperiodic fit is local to this region (better than global)
  3. Only KEEP peaks within the target band interior
  4. Discard padding-zone peaks (they only anchored the 1/f fit)

Result: local aperiodic fitting without edge artifacts at analysis boundaries.

Validation: run with --offset 0.5 to shift target bands by half an octave.
If enrichment pattern is stable regardless of target definition, the
structure is real. If it shifts with the targets, it's residual edge effect.

Usage:
    python scripts/run_lemon_phioctave_overlap_trim.py
    python scripts/run_lemon_phioctave_overlap_trim.py --condition EC
    python scripts/run_lemon_phioctave_overlap_trim.py --offset 0.5
    python scripts/run_lemon_phioctave_overlap_trim.py --resume-from sub-010002
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
    load_preprocessed_subject, LEMON_PREPROC_ROOT, LEMON_RAW_ROOT,
    SFREQ, WELCH_NPERSEG,
    _get_peak_params, _get_aperiodic_params, _get_r_squared,
)

# Raw loading imports (deferred to avoid mne import cost when not needed)
_raw_funcs_loaded = False
def _ensure_raw_funcs():
    global _raw_funcs_loaded, discover_raw_subjects, load_and_preprocess_raw
    if not _raw_funcs_loaded:
        sys.path.insert(0, os.path.dirname(__file__))
        from run_lemon_raw_extraction import (
            discover_raw_subjects, load_and_preprocess_raw,
        )
        _raw_funcs_loaded = True

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
F0_DEFAULT = 7.83
FREQ_CEIL = 85.0
FREQ_FLOOR = 1.0
R2_MIN = 0.70
PAD_OCTAVES = 0.5  # half phi-octave padding on each side

FOOOF_BASE_PARAMS = dict(
    peak_threshold=0.001,
    min_peak_height=0.0001,
    aperiodic_mode='fixed',
)


def build_target_bands(f0, freq_ceil=85.0, freq_floor=1.0, offset=0.0):
    """Build target phi-octave bands (the analysis windows).

    Parameters
    ----------
    f0 : float
        Anchor frequency.
    freq_ceil, freq_floor : float
        Frequency limits.
    offset : float
        Fractional octave offset (0.0=primary, 0.5=half-octave control).

    Returns
    -------
    list of dict with keys: name, n, target_lo, target_hi, fit_lo, fit_hi
    """
    bands = []
    for n in range(-4, 10):
        target_lo = f0 * PHI ** (n + offset)
        target_hi = f0 * PHI ** (n + 1 + offset)

        if target_lo >= freq_ceil:
            break
        if target_hi <= freq_floor:
            continue

        # Clamp target to usable range
        target_lo = max(target_lo, freq_floor)
        target_hi = min(target_hi, freq_ceil)

        target_width = target_hi - target_lo
        if target_width < 0.5:
            continue

        # Fit window: target + padding on each side
        fit_lo = f0 * PHI ** (n + offset - PAD_OCTAVES)
        fit_hi = f0 * PHI ** (n + 1 + offset + PAD_OCTAVES)

        # Clamp fit window to data range
        fit_lo = max(fit_lo, freq_floor)
        fit_hi = min(fit_hi, freq_ceil)

        bands.append({
            'name': f'n{n:+d}',
            'n': n,
            'target_lo': target_lo,
            'target_hi': target_hi,
            'fit_lo': fit_lo,
            'fit_hi': fit_hi,
        })

    return bands


def merge_narrow_targets(bands, min_width_hz=1.5, min_bins=12, freq_res=0.122):
    """Merge consecutive narrow target bands."""
    merged = []
    i = 0
    while i < len(bands):
        b = bands[i].copy()
        width = b['target_hi'] - b['target_lo']
        n_bins = int(width / freq_res)

        if width < min_width_hz or n_bins < min_bins:
            while (i + 1 < len(bands) and
                   (b['target_hi'] - b['target_lo'] < min_width_hz or
                    int((b['target_hi'] - b['target_lo']) / freq_res) < min_bins)):
                i += 1
                b['target_hi'] = bands[i]['target_hi']
                b['fit_hi'] = bands[i]['fit_hi']
            b['name'] = f'{b["name"]}_merged'

        # Recompute fit window for merged bands
        # Ensure padding extends beyond the (now wider) target
        if b['fit_lo'] > b['target_lo']:
            b['fit_lo'] = b['target_lo'] * 0.7  # generous lower padding
        if b['fit_hi'] < b['target_hi']:
            b['fit_hi'] = b['target_hi'] * 1.3

        merged.append(b)
        i += 1

    return merged


def extract_overlap_trim_subject(raw_clean, f0, fs=SFREQ,
                                  nperseg=WELCH_NPERSEG, overlap=0.5,
                                  freq_ceil=85.0, offset=0.0,
                                  r2_min=R2_MIN):
    """Overlap-trim per-phi-octave extraction for one subject.

    Returns (peaks_df, band_info_df).
    """
    freq_res = fs / nperseg

    # Build bands
    raw_bands = build_target_bands(f0, freq_ceil, FREQ_FLOOR, offset)
    bands = merge_narrow_targets(raw_bands, min_width_hz=1.5,
                                  min_bins=12, freq_res=freq_res)

    ch_names = [ch for ch in raw_clean.ch_names if ch != 'FCz']
    noverlap = int(nperseg * overlap)

    all_peaks = []
    band_stats = []

    for band in bands:
        bname = band['name']
        target_lo = band['target_lo']
        target_hi = band['target_hi']
        fit_lo = band['fit_lo']
        fit_hi = band['fit_hi']

        fit_width = fit_hi - fit_lo
        target_width = target_hi - target_lo

        # Adapt FOOOF params to fit window width
        max_n_peaks = max(3, min(15, int(fit_width / 1.5)))
        max_peak_width = min(fit_width * 0.6, 12.0)
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
        n_peaks_kept = 0
        n_peaks_trimmed = 0

        for ch in ch_names:
            try:
                data = raw_clean.get_data(picks=[ch])[0]
            except Exception:
                continue

            if len(data) < nperseg:
                continue

            freqs, psd = welch(data, fs, nperseg=nperseg, noverlap=noverlap)

            # Check fit window has enough bins
            fit_mask = (freqs >= fit_lo) & (freqs <= fit_hi)
            if fit_mask.sum() < 10:
                continue

            n_fitted += 1

            sm = SpectralModel(**fooof_params)
            try:
                sm.fit(freqs, psd, [fit_lo, fit_hi])
            except Exception:
                continue

            r2 = _get_r_squared(sm)
            if np.isnan(r2) or r2 < r2_min:
                continue

            n_passed += 1
            band_r2s.append(r2)

            offset_val, exponent = _get_aperiodic_params(sm)
            band_aperiodic_exps.append(exponent)

            # Extract ALL peaks from the fit window
            peak_params = _get_peak_params(sm)

            for row in peak_params:
                peak_freq = row[0]

                # TRIM: only keep peaks within the TARGET band
                if target_lo <= peak_freq < target_hi:
                    all_peaks.append({
                        'channel': ch,
                        'freq': peak_freq,
                        'power': row[1],
                        'bandwidth': row[2],
                        'phi_octave': bname,
                        'phi_octave_n': band['n'],
                        'target_lo': target_lo,
                        'target_hi': target_hi,
                        'fit_lo': fit_lo,
                        'fit_hi': fit_hi,
                    })
                    n_peaks_kept += 1
                else:
                    n_peaks_trimmed += 1

            band_n_peaks.append(n_peaks_kept)

        band_stats.append({
            'band_name': bname,
            'band_n': band['n'],
            'target_lo': round(target_lo, 3),
            'target_hi': round(target_hi, 3),
            'target_width': round(target_width, 3),
            'fit_lo': round(fit_lo, 3),
            'fit_hi': round(fit_hi, 3),
            'fit_width': round(fit_width, 3),
            'max_n_peaks': max_n_peaks,
            'n_channels_fitted': n_fitted,
            'n_channels_passed': n_passed,
            'mean_r_squared': np.mean(band_r2s) if band_r2s else np.nan,
            'mean_aperiodic_exponent': (np.mean(band_aperiodic_exps)
                                        if band_aperiodic_exps else np.nan),
            'peaks_kept': n_peaks_kept,
            'peaks_trimmed': n_peaks_trimmed,
        })

    peaks_df = pd.DataFrame(all_peaks) if all_peaks else pd.DataFrame(
        columns=['channel', 'freq', 'power', 'bandwidth',
                 'phi_octave', 'phi_octave_n',
                 'target_lo', 'target_hi', 'fit_lo', 'fit_hi'])
    band_info_df = pd.DataFrame(band_stats)

    return peaks_df, band_info_df


def main():
    parser = argparse.ArgumentParser(
        description='Overlap-trim per-phi-octave FOOOF extraction for LEMON')
    parser.add_argument('--f0', type=float, default=F0_DEFAULT,
                        help=f'Anchor frequency (default: {F0_DEFAULT})')
    parser.add_argument('--condition', type=str, default='EO',
                        choices=['EO', 'EC'])
    parser.add_argument('--offset', type=float, default=0.0,
                        help='Fractional octave offset (0.0=primary, 0.5=control)')
    parser.add_argument('--resume-from', type=str, default=None)
    parser.add_argument('--freq-ceil', type=float, default=FREQ_CEIL)
    parser.add_argument('--raw', action='store_true',
                        help='Use raw .vhdr files instead of preprocessed .set')
    args = parser.parse_args()

    f0 = args.f0
    condition = args.condition
    offset = args.offset
    freq_ceil = args.freq_ceil
    use_raw = args.raw

    offset_tag = f'_offset{offset:.1f}' if offset != 0.0 else ''
    raw_tag = '_raw' if use_raw else ''
    out_dir = os.path.join(
        os.path.dirname(__file__), '..', 'exports_lemon',
        f'per_subject_overlap_trim{raw_tag}_f0{f0:.2f}{offset_tag}')
    os.makedirs(out_dir, exist_ok=True)

    log.info(f"Overlap-Trim Per-Phi-Octave FOOOF Extraction")
    log.info(f"  f₀ = {f0} Hz, condition = {condition}, offset = {offset}")
    log.info(f"  data source = {'RAW (.vhdr)' if use_raw else 'PREPROCESSED (.set)'}")
    log.info(f"  padding = {PAD_OCTAVES} phi-octaves per side")
    log.info(f"  output: {out_dir}")

    # Show band structure
    freq_res = SFREQ / WELCH_NPERSEG
    raw_bands = build_target_bands(f0, freq_ceil, FREQ_FLOOR, offset)
    bands = merge_narrow_targets(raw_bands, min_width_hz=1.5,
                                  min_bins=12, freq_res=freq_res)

    log.info(f"\nBands ({len(bands)}):")
    log.info(f"  {'name':>12}  {'target':>18}  {'fit window':>18}  {'trim zone':>12}")
    for b in bands:
        tgt = f"[{b['target_lo']:5.2f}, {b['target_hi']:5.2f}]"
        fit = f"[{b['fit_lo']:5.2f}, {b['fit_hi']:5.2f}]"
        pad_lo = b['target_lo'] - b['fit_lo']
        pad_hi = b['fit_hi'] - b['target_hi']
        trim = f"-{pad_lo:.1f}/+{pad_hi:.1f} Hz"
        log.info(f"  {b['name']:>12}  {tgt:>18}  {fit:>18}  {trim:>12}")

    # List subjects
    if use_raw:
        _ensure_raw_funcs()
        raw_subject_list = discover_raw_subjects(LEMON_RAW_ROOT)
        subjects = [sid for sid, _ in raw_subject_list]
        vhdr_map = {sid: vhdr for sid, vhdr in raw_subject_list}
    else:
        preproc_dir = LEMON_PREPROC_ROOT
        subjects = sorted(set(
            f.replace(f'_{condition}.set', '')
            for f in os.listdir(preproc_dir)
            if f.endswith(f'_{condition}.set') and not f.startswith('.')
        ))
        vhdr_map = {}
    log.info(f"\nFound {len(subjects)} subjects")

    if args.resume_from:
        try:
            idx = subjects.index(args.resume_from)
            subjects = subjects[idx:]
            log.info(f"Resuming from {args.resume_from} ({len(subjects)} remaining)")
        except ValueError:
            log.warning(f"Subject {args.resume_from} not found")

    suffix = '_peaks' if condition == 'EO' else f'_peaks_{condition.lower()}'

    summary_rows = []
    t_start = time.time()

    for i, sid in enumerate(subjects):
        out_path = os.path.join(out_dir, f'{sid}{suffix}.csv')
        band_path = os.path.join(out_dir, f'{sid}_band_info.csv')

        if os.path.exists(out_path):
            continue

        t0 = time.time()
        if use_raw:
            raw, info = load_and_preprocess_raw(vhdr_map[sid], condition)
        else:
            raw, info = load_preprocessed_subject(sid, preproc_dir, condition)
        if raw is None:
            status = info.get('status', 'no_data') if isinstance(info, dict) else 'no_data'
            log.warning(f"  [{i+1}/{len(subjects)}] {sid} — {status}")
            summary_rows.append({
                'subject_id': sid, 'condition': condition,
                'status': status, 'n_peaks': 0,
            })
            continue

        try:
            peaks_df, band_info_df = extract_overlap_trim_subject(
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

        peaks_df.to_csv(out_path, index=False)
        band_info_df.to_csv(band_path, index=False)

        elapsed = time.time() - t0
        n_peaks = len(peaks_df)
        n_bands_ok = (band_info_df['n_channels_passed'] > 0).sum()
        mean_r2 = band_info_df['mean_r_squared'].mean()
        total_kept = band_info_df['peaks_kept'].sum()
        total_trimmed = band_info_df['peaks_trimmed'].sum()
        trim_pct = total_trimmed / (total_kept + total_trimmed) * 100 if (total_kept + total_trimmed) > 0 else 0

        summary_rows.append({
            'subject_id': sid,
            'condition': condition,
            'status': 'ok',
            'n_peaks': n_peaks,
            'n_bands_total': len(band_info_df),
            'n_bands_with_peaks': n_bands_ok,
            'mean_r_squared': round(mean_r2, 4) if not np.isnan(mean_r2) else np.nan,
            'peaks_trimmed_pct': round(trim_pct, 1),
            'duration_sec': round(info.get('duration_sec',
                                         info.get('duration_after_preproc_sec', 0)), 1),
            'extraction_time_sec': round(elapsed, 1),
        })

        band_counts = band_info_df.set_index('band_name')['peaks_kept'].to_dict()
        band_str = ' '.join(f'{k}:{v}' for k, v in band_counts.items())

        log.info(f"  [{i+1}/{len(subjects)}] {sid}: {n_peaks} peaks "
                 f"({n_bands_ok}/{len(band_info_df)} bands) "
                 f"R²={mean_r2:.3f} trim={trim_pct:.0f}% "
                 f"{elapsed:.1f}s  [{band_str}]")

        del raw
        gc.collect()

        if (i + 1) % 10 == 0:
            pd.DataFrame(summary_rows).to_csv(
                os.path.join(out_dir, f'extraction_summary_{condition.lower()}.csv'),
                index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, f'extraction_summary_{condition.lower()}.csv')
    summary_df.to_csv(summary_path, index=False)

    total_time = time.time() - t_start
    n_ok = (summary_df['status'] == 'ok').sum() if len(summary_df) > 0 else 0
    log.info(f"\n{'='*60}")
    log.info(f"DONE: {n_ok}/{len(subjects)} subjects in {total_time/60:.1f} min")
    if n_ok > 0:
        ok_df = summary_df[summary_df['status'] == 'ok']
        log.info(f"  Mean peaks/subject: {ok_df['n_peaks'].mean():.0f}")
        log.info(f"  Mean R²: {ok_df['mean_r_squared'].mean():.3f}")
        log.info(f"  Mean trim rate: {ok_df['peaks_trimmed_pct'].mean():.0f}%")
        log.info(f"  Mean time: {ok_df['extraction_time_sec'].mean():.1f}s")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Re-extract FOOOF peaks at [1, 85] Hz for LEMON preprocessed data.
=================================================================

Step 0 of the amplitude-weighted compliance plan.

Loads each preprocessed .set file, runs extract_fooof_peaks_subject()
with freq_range=[1, 85], saves per-subject peak CSVs and extraction summary.

Usage:
    python scripts/run_lemon_reextract_peaks.py --condition EO
    python scripts/run_lemon_reextract_peaks.py --condition EC
    python scripts/run_lemon_reextract_peaks.py --condition EO --max-n-peaks 40
"""

import os
import sys
import argparse
import time
import logging
import warnings
import gc

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from lemon_utils import (
    LEMON_PREPROC_ROOT, SFREQ, WELCH_NPERSEG,
    FOOOF_CHANNEL_R2_MIN,
    load_preprocessed_subject, extract_fooof_peaks_subject,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


def get_subject_ids(preproc_root, condition='EO'):
    """Discover subject IDs from preprocessed .set files."""
    suffix = f'_{condition}.set'
    sids = []
    for fname in sorted(os.listdir(preproc_root)):
        if fname.endswith(suffix) and not fname.startswith('.'):
            sid = fname.replace(suffix, '')
            sids.append(sid)
    return sids


def run_extraction(condition='EO', max_n_peaks=20, out_dir=None,
                   resume_from=None):
    """Extract FOOOF peaks at [1, 85] Hz for all subjects."""

    if out_dir is None:
        out_dir = os.path.join(
            os.path.dirname(__file__), '..', 'exports_lemon',
            'per_subject_85hz')
    os.makedirs(out_dir, exist_ok=True)

    # FOOOF params — extended to 85 Hz
    fooof_params = dict(
        freq_range=[1, 85],
        max_n_peaks=max_n_peaks,
        peak_threshold=0.001,
        min_peak_height=0.0001,
        peak_width_limits=[0.2, 20],
    )

    subject_ids = get_subject_ids(LEMON_PREPROC_ROOT, condition)
    log.info(f"Found {len(subject_ids)} subjects for condition={condition}")

    # Determine output suffix
    if condition == 'EC':
        suffix = '_peaks_ec'
    elif max_n_peaks != 20:
        suffix = f'_peaks_max{max_n_peaks}'
    else:
        suffix = '_peaks'

    # Resume support
    if resume_from:
        try:
            idx = subject_ids.index(resume_from)
            subject_ids = subject_ids[idx:]
            log.info(f"Resuming from {resume_from} ({len(subject_ids)} remaining)")
        except ValueError:
            log.warning(f"Resume ID {resume_from} not found, starting from beginning")

    summary_rows = []
    t_start = time.time()

    for i, sid in enumerate(subject_ids):
        t0 = time.time()

        # Check if already done
        out_path = os.path.join(out_dir, f'{sid}{suffix}.csv')
        if os.path.exists(out_path) and not resume_from:
            log.info(f"[{i+1}/{len(subject_ids)}] {sid} — already exists, skipping")
            # Still load summary if it exists
            continue

        # Load preprocessed data
        raw, info = load_preprocessed_subject(sid, condition=condition)
        if raw is None:
            log.warning(f"[{i+1}/{len(subject_ids)}] {sid} — failed to load")
            summary_rows.append({
                'subject_id': sid, 'condition': condition,
                'status': 'load_failed',
                'n_peaks': 0, 'n_peaks_below_45hz': 0,
                'mean_r_squared': np.nan,
                'n_channels_passed': 0,
                'mean_aperiodic_exponent': np.nan,
                'duration_sec': 0,
            })
            continue

        # Extract peaks
        try:
            peaks_df, ch_info = extract_fooof_peaks_subject(
                raw, fs=SFREQ, fooof_params=fooof_params,
                nperseg=WELCH_NPERSEG, channel_r2_min=FOOOF_CHANNEL_R2_MIN)
        except Exception as e:
            log.error(f"[{i+1}/{len(subject_ids)}] {sid} — FOOOF failed: {e}")
            summary_rows.append({
                'subject_id': sid, 'condition': condition,
                'status': 'fooof_failed',
                'n_peaks': 0, 'n_peaks_below_45hz': 0,
                'mean_r_squared': np.nan,
                'n_channels_passed': 0,
                'mean_aperiodic_exponent': np.nan,
                'duration_sec': info.get('duration_sec', 0),
            })
            del raw
            gc.collect()
            continue

        # Save peaks
        peaks_df.to_csv(out_path, index=False)

        # Summary stats
        n_peaks = len(peaks_df)
        n_below_45 = (peaks_df['freq'] < 45).sum() if n_peaks > 0 else 0
        elapsed = time.time() - t0

        summary_rows.append({
            'subject_id': sid,
            'condition': condition,
            'status': 'ok',
            'n_peaks': n_peaks,
            'n_peaks_below_45hz': n_below_45,
            'mean_r_squared': ch_info.get('mean_r_squared', np.nan),
            'n_channels_passed': ch_info.get('n_channels_passed', 0),
            'n_channels_fitted': ch_info.get('n_channels_fitted', 0),
            'mean_aperiodic_exponent': ch_info.get('mean_aperiodic_exponent', np.nan),
            'mean_n_peaks_per_channel': ch_info.get('mean_n_peaks', 0),
            'duration_sec': info.get('duration_sec', 0),
            'extraction_time_sec': elapsed,
        })

        log.info(
            f"[{i+1}/{len(subject_ids)}] {sid} — "
            f"{n_peaks} peaks ({n_below_45} < 45 Hz), "
            f"R²={ch_info.get('mean_r_squared', 0):.3f}, "
            f"{ch_info.get('n_channels_passed', 0)} ch passed, "
            f"aperiodic={ch_info.get('mean_aperiodic_exponent', 0):.2f}, "
            f"{elapsed:.1f}s"
        )

        del raw
        gc.collect()

        # Save incremental summary every 10 subjects
        if (i + 1) % 10 == 0:
            _save_summary(summary_rows, out_dir, condition, max_n_peaks)

    # Final summary
    _save_summary(summary_rows, out_dir, condition, max_n_peaks)

    total_time = time.time() - t_start
    log.info(f"Done. {len(summary_rows)} subjects in {total_time/60:.1f} min")


def _save_summary(rows, out_dir, condition, max_n_peaks):
    """Save extraction summary CSV."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    tag = f'_{condition.lower()}'
    if max_n_peaks != 20:
        tag += f'_max{max_n_peaks}'
    path = os.path.join(out_dir, f'extraction_summary{tag}.csv')
    df.to_csv(path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Re-extract FOOOF peaks at [1, 85] Hz')
    parser.add_argument('--condition', default='EO', choices=['EO', 'EC'],
                        help='EO or EC condition')
    parser.add_argument('--max-n-peaks', type=int, default=20,
                        help='FOOOF max_n_peaks parameter')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Subject ID to resume from')
    args = parser.parse_args()

    run_extraction(
        condition=args.condition,
        max_n_peaks=args.max_n_peaks,
        resume_from=args.resume_from,
    )

#!/usr/bin/env python3
"""
Raw-data FOOOF extraction for LEMON — minimal preprocessing parallel track.
============================================================================

Step 0a of the amplitude-weighted compliance plan.

Processes raw .vhdr files with minimal preprocessing (no ICA, no epoch
rejection) to test whether ICA preprocessing removes phi-structured activity.

Pipeline per subject:
1. Load raw .vhdr (2500 Hz, 62 ch)
2. Drop non-EEG channels (VEOG), set standard 10-20 montage
3. Bandpass filter: 0.5–85 Hz (FIR, zero-phase)
4. Downsample to 250 Hz
5. Average re-reference
6. Parse markers: S210=EO, S200=EC, crop condition segments
7. Run extract_fooof_peaks_subject() with freq_range=[1, 85], max_n_peaks=20
8. Save per-subject peak CSVs

Usage:
    python scripts/run_lemon_raw_extraction.py --condition EO
    python scripts/run_lemon_raw_extraction.py --condition EC
"""

import os
import sys
import re
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
    LEMON_RAW_ROOT, SFREQ, WELCH_NPERSEG,
    FOOOF_CHANNEL_R2_MIN, extract_fooof_peaks_subject,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# Marker codes
MARKER_EO = 'S210'   # Eyes-open condition
MARKER_EC = 'S200'   # Eyes-closed condition
MARKER_BLOCK = 'S  1'  # Block onset

# Non-EEG channels to drop
NON_EEG_CHANNELS = ['VEOG']

# Minimum condition duration to include subject (seconds)
MIN_CONDITION_DURATION_SEC = 90.0


def discover_raw_subjects(raw_root):
    """Find all subject directories with RSEEG .vhdr files."""
    subjects = []
    for sid in sorted(os.listdir(raw_root)):
        if not sid.startswith('sub-'):
            continue
        rseeg_dir = os.path.join(raw_root, sid, 'RSEEG')
        vhdr_path = os.path.join(rseeg_dir, f'{sid}.vhdr')
        if os.path.exists(vhdr_path):
            subjects.append((sid, vhdr_path))
    return subjects


def parse_vmrk_markers(vmrk_path):
    """Parse .vmrk file to extract marker positions and descriptions.

    Returns list of (description, position_in_samples) tuples.
    """
    markers = []
    in_marker_section = False

    with open(vmrk_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if line == '[Marker Infos]':
                in_marker_section = True
                continue
            if line.startswith('[') and in_marker_section:
                break  # next section
            if not in_marker_section:
                continue
            if line.startswith(';') or not line:
                continue

            # Parse: Mk<N>=<Type>,<Description>,<Position>,<Size>,<Channel>,...
            match = re.match(r'Mk\d+=\w+,(.+?),(\d+),', line)
            if match:
                desc = match.group(1).strip()
                pos = int(match.group(2))
                markers.append((desc, pos))

    return markers


def get_condition_segments(markers, condition='EO', sfreq_raw=2500):
    """Extract contiguous condition segments from markers.

    Returns list of (start_sample, end_sample) tuples at original sfreq.

    Strategy: identify runs of S210 (EO) or S200 (EC) markers.
    Each block starts at first condition marker after an S1,
    ends at last condition marker + inter-marker gap.
    """
    target_marker = MARKER_EO if condition == 'EO' else MARKER_EC

    # Find all positions of the target marker
    target_positions = [pos for desc, pos in markers if desc == target_marker]

    if not target_positions:
        return []

    # Group into contiguous blocks: consecutive markers <10s apart
    max_gap = 10 * sfreq_raw  # 10 seconds
    blocks = []
    block_start = target_positions[0]
    prev_pos = target_positions[0]

    for pos in target_positions[1:]:
        if pos - prev_pos > max_gap:
            # New block — end previous at prev_pos + typical gap
            typical_gap = 5000  # ~2 sec at 2500 Hz
            blocks.append((block_start, prev_pos + typical_gap))
            block_start = pos
        prev_pos = pos

    # Close final block
    typical_gap = 5000
    blocks.append((block_start, prev_pos + typical_gap))

    return blocks


def load_and_preprocess_raw(vhdr_path, condition='EO'):
    """Load raw .vhdr, apply minimal preprocessing, crop condition.

    Returns (mne.io.Raw | None, info_dict).
    """
    import mne

    # Parse markers from .vmrk file
    vmrk_path = vhdr_path.replace('.vhdr', '.vmrk')
    if not os.path.exists(vmrk_path):
        return None, {'status': 'no_vmrk'}

    markers = parse_vmrk_markers(vmrk_path)
    if not markers:
        return None, {'status': 'no_markers'}

    # Get condition segments (in raw sample indices at 2500 Hz)
    segments = get_condition_segments(markers, condition, sfreq_raw=2500)
    if not segments:
        return None, {'status': 'no_condition_segments'}

    # Load raw
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            raw = mne.io.read_raw_brainvision(vhdr_path, preload=True,
                                               verbose=False)
        except Exception as e:
            return None, {'status': f'load_failed: {e}'}

    orig_sfreq = raw.info['sfreq']

    # Drop non-EEG channels
    drop_chs = [ch for ch in NON_EEG_CHANNELS if ch in raw.ch_names]
    if drop_chs:
        raw.drop_channels(drop_chs)

    # Set montage
    try:
        montage = mne.channels.make_standard_montage('standard_1005')
        raw.set_montage(montage, on_missing='warn')
    except Exception:
        pass

    # Crop to condition segments and concatenate
    raws_condition = []
    total_samples = raw.n_times
    total_duration_sec = 0.0

    for start_samp, end_samp in segments:
        # Clamp to data bounds
        start_samp = max(0, start_samp)
        end_samp = min(total_samples, end_samp)
        if end_samp <= start_samp:
            continue

        tmin = start_samp / orig_sfreq
        tmax = end_samp / orig_sfreq

        try:
            segment = raw.copy().crop(tmin=tmin, tmax=min(tmax, raw.times[-1]))
            dur = segment.times[-1] - segment.times[0]
            total_duration_sec += dur
            raws_condition.append(segment)
        except Exception:
            continue

    if not raws_condition:
        del raw
        return None, {'status': 'no_valid_segments'}

    # Check minimum duration
    if total_duration_sec < MIN_CONDITION_DURATION_SEC:
        del raw
        for r in raws_condition:
            del r
        return None, {
            'status': 'insufficient_duration',
            'eo_duration_sec' if condition == 'EO' else 'ec_duration_sec': total_duration_sec,
        }

    # Concatenate condition segments
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        raw_cond = mne.concatenate_raws(raws_condition)

    del raw
    for r in raws_condition:
        del r

    # Bandpass filter: 0.5–85 Hz
    raw_cond.filter(l_freq=0.5, h_freq=85.0, method='fir',
                    phase='zero-double', verbose=False)

    # Downsample to 250 Hz
    raw_cond.resample(SFREQ, verbose=False)

    # Average re-reference
    raw_cond.set_eeg_reference('average', projection=False, verbose=False)

    info = {
        'status': 'ok',
        'n_channels': len(raw_cond.ch_names),
        'n_blocks': len(segments),
        f'{condition.lower()}_duration_sec': total_duration_sec,
        'duration_after_preproc_sec': raw_cond.times[-1],
        'orig_sfreq': orig_sfreq,
    }

    return raw_cond, info


def run_raw_extraction(condition='EO', out_dir=None, resume_from=None):
    """Extract FOOOF peaks from minimally preprocessed raw data."""

    if out_dir is None:
        out_dir = os.path.join(
            os.path.dirname(__file__), '..', 'exports_lemon',
            'per_subject_raw85hz')
    os.makedirs(out_dir, exist_ok=True)

    # FOOOF params
    fooof_params = dict(
        freq_range=[1, 85],
        max_n_peaks=20,
        peak_threshold=0.001,
        min_peak_height=0.0001,
        peak_width_limits=[0.2, 20],
    )

    subjects = discover_raw_subjects(LEMON_RAW_ROOT)
    log.info(f"Found {len(subjects)} subjects with raw RSEEG data")

    suffix = '_peaks' if condition == 'EO' else '_peaks_ec'

    # Resume support
    if resume_from:
        sids = [s[0] for s in subjects]
        try:
            idx = sids.index(resume_from)
            subjects = subjects[idx:]
            log.info(f"Resuming from {resume_from} ({len(subjects)} remaining)")
        except ValueError:
            log.warning(f"Resume ID {resume_from} not found")

    summary_rows = []
    t_start = time.time()

    for i, (sid, vhdr_path) in enumerate(subjects):
        t0 = time.time()

        out_path = os.path.join(out_dir, f'{sid}{suffix}.csv')
        if os.path.exists(out_path) and not resume_from:
            log.info(f"[{i+1}/{len(subjects)}] {sid} — exists, skipping")
            continue

        # Load and preprocess
        raw_cond, info = load_and_preprocess_raw(vhdr_path, condition)
        if raw_cond is None:
            log.warning(f"[{i+1}/{len(subjects)}] {sid} — {info.get('status', 'failed')}")
            summary_rows.append({
                'subject_id': sid, 'condition': condition,
                **info,
                'n_peaks': 0, 'mean_r_squared': np.nan,
                'n_channels_passed': 0, 'mean_aperiodic_exponent': np.nan,
                'mean_peak_power': np.nan,
            })
            continue

        # Extract peaks
        try:
            peaks_df, ch_info = extract_fooof_peaks_subject(
                raw_cond, fs=SFREQ, fooof_params=fooof_params,
                nperseg=WELCH_NPERSEG, channel_r2_min=FOOOF_CHANNEL_R2_MIN)
        except Exception as e:
            log.error(f"[{i+1}/{len(subjects)}] {sid} — FOOOF failed: {e}")
            summary_rows.append({
                'subject_id': sid, 'condition': condition,
                **info, 'status': f'fooof_failed: {e}',
                'n_peaks': 0, 'mean_r_squared': np.nan,
                'n_channels_passed': 0, 'mean_aperiodic_exponent': np.nan,
                'mean_peak_power': np.nan,
            })
            del raw_cond
            gc.collect()
            continue

        # Save
        peaks_df.to_csv(out_path, index=False)

        n_peaks = len(peaks_df)
        mean_power = peaks_df['power'].mean() if n_peaks > 0 else np.nan
        elapsed = time.time() - t0

        row = {
            'subject_id': sid,
            'condition': condition,
            **info,
            'n_peaks': n_peaks,
            'mean_r_squared': ch_info.get('mean_r_squared', np.nan),
            'n_channels_passed': ch_info.get('n_channels_passed', 0),
            'n_channels_fitted': ch_info.get('n_channels_fitted', 0),
            'mean_aperiodic_exponent': ch_info.get('mean_aperiodic_exponent', np.nan),
            'mean_n_peaks_per_channel': ch_info.get('mean_n_peaks', 0),
            'mean_peak_power': mean_power,
            'extraction_time_sec': elapsed,
        }
        summary_rows.append(row)

        log.info(
            f"[{i+1}/{len(subjects)}] {sid} — "
            f"{n_peaks} peaks, mean_power={mean_power:.3f}, "
            f"R²={ch_info.get('mean_r_squared', 0):.3f}, "
            f"{ch_info.get('n_channels_passed', 0)} ch, "
            f"aperiodic={ch_info.get('mean_aperiodic_exponent', 0):.2f}, "
            f"dur={info.get(f'{condition.lower()}_duration_sec', 0):.0f}s, "
            f"{elapsed:.1f}s"
        )

        del raw_cond
        gc.collect()

        # Incremental save
        if (i + 1) % 5 == 0:
            _save_summary(summary_rows, out_dir, condition)

    _save_summary(summary_rows, out_dir, condition)
    total_time = time.time() - t_start
    log.info(f"Done. {len(summary_rows)} subjects in {total_time/60:.1f} min")


def _save_summary(rows, out_dir, condition):
    if not rows:
        return
    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, f'extraction_summary_{condition.lower()}.csv')
    df.to_csv(path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Raw LEMON FOOOF extraction with minimal preprocessing')
    parser.add_argument('--condition', default='EO', choices=['EO', 'EC'])
    parser.add_argument('--resume-from', type=str, default=None)
    args = parser.parse_args()

    run_raw_extraction(condition=args.condition, resume_from=args.resume_from)

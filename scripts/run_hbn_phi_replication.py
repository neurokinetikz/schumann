#!/usr/bin/env python3
"""
HBN Phi-Lattice Replication — Unified Protocol
================================================

THE replication test. Pre-registered predictions scored by
generate_preregistration_report().

Data: /Volumes/T9/hbn_data/cmi_bids_R1/sub-NDAR*/eeg/*RestingState*
Format: ~136 subjects, 500 Hz, 128 channels (EGI HydroCel)
Task: RestingState — alternating 20s EO/EC blocks
  event_code 20 = instructed_toOpenEyes (EO block start)
  event_code 30 = instructed_toCloseEyes (EC block start)
Preprocessing: pick EEG → notch [60, 120] Hz → resample 500→250 Hz

CMI Healthy Brain Network, Release 1
"""

import sys
import os
import glob
import numpy as np
import pandas as pd
import mne
import warnings
warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from phi_replication import (process_subjects_from_eeg, run_full_protocol,
                             test_theta_ec_convergence,
                             generate_preregistration_report)

# ── Data paths ────────────────────────────────────────────────────────
DATA_DIR = '/Volumes/T9/hbn_data/cmi_bids_R1'
OUT_DIR = 'exports_hbn'

TARGET_FS = 250


def discover_subjects():
    """Find all subjects with RestingState .set files."""
    pattern = os.path.join(DATA_DIR, 'sub-*', 'eeg',
                           '*_task-RestingState_eeg.set')
    files = sorted(glob.glob(pattern))
    subjects = []
    for f in files:
        parts = f.split(os.sep)
        for p in parts:
            if p.startswith('sub-'):
                subjects.append(p)
                break
    return subjects


def parse_events(sub_id):
    """Parse events.tsv to get EO/EC segment boundaries.

    Returns dict with 'EO' and 'EC' lists of (onset, offset) in seconds.
    """
    events_path = os.path.join(DATA_DIR, sub_id, 'eeg',
                                f'{sub_id}_task-RestingState_events.tsv')
    if not os.path.isfile(events_path):
        return None

    events = pd.read_csv(events_path, sep='\t')

    # Extract EO (code 20) and EC (code 30) onsets
    eo_onsets = events[events['event_code'] == 20]['onset'].values
    ec_onsets = events[events['event_code'] == 30]['onset'].values

    # Build segments: each block runs until the next event or end
    all_onsets = sorted(list(eo_onsets) + list(ec_onsets))
    segments = {'EO': [], 'EC': []}

    for onset in eo_onsets:
        # Find next event after this one
        later = [t for t in all_onsets if t > onset]
        offset = later[0] if later else onset + 20.0  # default 20s blocks
        segments['EO'].append((onset, offset))

    for onset in ec_onsets:
        later = [t for t in all_onsets if t > onset]
        offset = later[0] if later else onset + 20.0
        segments['EC'].append((onset, offset))

    # Validation: log event counts and durations
    eo_dur = sum(off - on for on, off in segments['EO']) if segments['EO'] else 0
    ec_dur = sum(off - on for on, off in segments['EC']) if segments['EC'] else 0
    print(f"    {sub_id}: EO={len(segments['EO'])} events ({eo_dur:.1f}s), "
          f"EC={len(segments['EC'])} events ({ec_dur:.1f}s)")
    if len(segments['EO']) == 0 or len(segments['EC']) == 0:
        print(f"    WARNING: {sub_id} has zero events for "
              f"{'EO' if len(segments['EO'])==0 else 'EC'}")

    return segments


# Minimum usable duration per condition (seconds)
MIN_CONDITION_DURATION = 10.0


def load_hbn_subject(sub_id, condition='combined'):
    """Load HBN RestingState .set file.

    Preprocessing: pick EEG → notch [60, 120] Hz → resample 500→250 Hz

    Parameters
    ----------
    sub_id : str — e.g. 'sub-NDARAC904DMU'
    condition : str — 'combined', 'EO', or 'EC'

    Returns (data_array, ch_names, fs, metadata_dict)
    """
    set_path = os.path.join(DATA_DIR, sub_id, 'eeg',
                             f'{sub_id}_task-RestingState_eeg.set')
    if not os.path.isfile(set_path):
        raise FileNotFoundError(f"No RestingState file for {sub_id}")

    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
    raw.pick_types(eeg=True, exclude='bads')

    # Notch filter at 60 Hz + harmonics (US power line)
    raw.notch_filter(freqs=[60, 120], verbose=False)

    # Downsample to 250 Hz
    if raw.info['sfreq'] > TARGET_FS:
        raw.resample(TARGET_FS, verbose=False)

    # Extract condition-specific segments if needed
    if condition in ('EO', 'EC'):
        segments = parse_events(sub_id)
        if segments and condition in segments and len(segments[condition]) > 0:
            raws = []
            for onset, offset in segments[condition]:
                try:
                    segment = raw.copy().crop(tmin=onset, tmax=min(offset, raw.times[-1]))
                    if len(segment.times) > 0:
                        raws.append(segment)
                except Exception:
                    continue
            if len(raws) > 0:
                raw = mne.concatenate_raws(raws)
                total_dur = raw.times[-1] - raw.times[0]
                if total_dur < MIN_CONDITION_DURATION:
                    raise ValueError(
                        f"{sub_id}: {condition} duration {total_dur:.1f}s "
                        f"< minimum {MIN_CONDITION_DURATION}s")
            else:
                raise ValueError(f"No valid {condition} segments for {sub_id}")

    # Load metadata from participants.tsv
    metadata = {'subject': sub_id}
    participants_path = os.path.join(DATA_DIR, 'participants.tsv')
    if os.path.isfile(participants_path):
        try:
            ptable = pd.read_csv(participants_path, sep='\t')
            prow = ptable[ptable['participant_id'] == sub_id]
            if len(prow) > 0:
                metadata['age'] = prow.iloc[0].get('age', None)
                metadata['sex'] = prow.iloc[0].get('sex', None)
        except Exception:
            pass

    return (raw.get_data(), raw.ch_names, raw.info['sfreq'], metadata)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"HBN Phi-Lattice Replication")
    print(f"Data: {DATA_DIR}")
    print(f"Output: {OUT_DIR}")
    print(f"THIS IS THE PRE-REGISTERED REPLICATION TEST")

    subjects = discover_subjects()
    print(f"Found {len(subjects)} subjects with RestingState data")

    if len(subjects) < 10:
        print("ERROR: Too few subjects!")
        sys.exit(1)

    # Run 3 conditions: combined (primary), then EO and EC separately
    condition_dfs = {}
    for condition in ['combined', 'EO', 'EC']:
        print(f"\n\n{'#'*70}")
        print(f"  Condition: {condition}")
        print(f"{'#'*70}")

        loader = lambda sub, cond=condition: load_hbn_subject(sub, cond)
        dom_df = process_subjects_from_eeg(
            subjects, loader,
            out_dir=os.path.join(OUT_DIR, condition),
            label=f'HBN {condition}')

        condition_dfs[condition] = dom_df

    # Theta EC convergence test
    if 'EO' in condition_dfs and 'EC' in condition_dfs:
        convergence = test_theta_ec_convergence(
            condition_dfs['EO'], condition_dfs['EC'])

    # Pre-registration scoring (on combined condition)
    print(f"\n\n{'#'*70}")
    print(f"  PRE-REGISTRATION SCORING")
    print(f"{'#'*70}")

    dom_df = condition_dfs.get('combined', pd.DataFrame())
    if len(dom_df) > 0:
        from phi_replication import run_statistics, run_14position_enrichment

        stats_result = run_statistics(dom_df, 'HBN combined [OT]')
        enrichment_df = run_14position_enrichment(dom_df)

        # Add theta convergence p-value if available
        if 'EO' in condition_dfs and 'EC' in condition_dfs:
            conv = test_theta_ec_convergence(
                condition_dfs['EO'], condition_dfs['EC'])
            stats_result['theta_ec_p'] = conv.get('theta_ec_p', 1.0)

        report = generate_preregistration_report(
            dom_df, enrichment_df, stats_result, label='HBN')

        # Save report
        report_df = pd.DataFrame([report])
        report_df.to_csv(os.path.join(OUT_DIR, 'preregistration_report.csv'),
                         index=False)

    print(f"\n\nDone! Results saved to {OUT_DIR}")


if __name__ == '__main__':
    main()

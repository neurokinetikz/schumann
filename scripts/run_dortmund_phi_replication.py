#!/usr/bin/env python3
"""
Dortmund Phi-Lattice Replication — Unified Protocol
====================================================

Re-extracts from raw EDF files using the locked protocol.
Verification target: EC-pre Cohen's d ≈ 0.39, phi rank 1/9.

Data: /Volumes/T9/dortmund_data_dl/sub-*/ses-1/eeg/*.edf
Format: 608 subjects, 500 Hz native, 64 channels, 4 conditions
Preprocessing: pick EEG → notch [50, 100] Hz → resample 500→250 Hz

Reference: Wascher et al. doi:10.18112/openneuro.ds005385.v1.0.3
"""

import sys
import os
import pandas as pd
import mne
import warnings
warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from phi_replication import (process_subjects_from_eeg, run_full_protocol,
                             test_theta_ec_convergence)

# ── Data paths ────────────────────────────────────────────────────────
DATA_DIR = '/Volumes/T9/dortmund_data_dl'
OUT_DIR = '/Volumes/T9/dortmund_data/lattice_results_replication_v2'

TARGET_FS = 250

# Conditions to analyze (in priority order)
CONDITIONS = [
    ('EyesClosed', 'pre'),   # Primary replication target
    ('EyesOpen', 'pre'),     # Secondary
    ('EyesClosed', 'post'),  # Post-cognitive fatigue
    ('EyesOpen', 'post'),    # Post-cognitive fatigue
]


def discover_subjects():
    """Find all subject directories."""
    subjects = sorted([d for d in os.listdir(DATA_DIR)
                       if d.startswith('sub-')
                       and os.path.isdir(os.path.join(DATA_DIR, d))])
    return subjects


def find_edf(sub_id, task, acq):
    """Find EDF file for a given subject/condition."""
    path = os.path.join(DATA_DIR, sub_id, 'ses-1', 'eeg',
                        f'{sub_id}_ses-1_task-{task}_acq-{acq}_eeg.edf')
    if os.path.isfile(path):
        return path
    return None


def load_dortmund_subject(sub_id, task='EyesClosed', acq='pre',
                          participants_df=None):
    """Load Dortmund EDF file.

    Preprocessing: pick EEG → notch [50, 100] Hz → resample 500→250 Hz
    Returns (data_array, ch_names, fs, metadata_dict)
    """
    edf_path = find_edf(sub_id, task, acq)
    if edf_path is None:
        raise FileNotFoundError(f"No EDF for {sub_id} {task} {acq}")

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.pick_types(eeg=True, exclude='bads')
    raw.notch_filter(freqs=[50, 100], verbose=False)

    if raw.info['sfreq'] > TARGET_FS:
        raw.resample(TARGET_FS, verbose=False)

    metadata = {'subject': sub_id}
    if participants_df is not None:
        prow = participants_df[participants_df['participant_id'] == sub_id]
        if len(prow) > 0:
            metadata['age'] = prow.iloc[0].get('age', None)
            metadata['sex'] = prow.iloc[0].get('sex', None)

    return (raw.get_data(), raw.ch_names, raw.info['sfreq'], metadata)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Dortmund Phi-Lattice Replication")
    print(f"Data: {DATA_DIR}")
    print(f"Output: {OUT_DIR}")

    # Load participants
    participants_df = None
    for p in [os.path.join(DATA_DIR, 'participants.tsv'),
              '/Volumes/T9/dortmund_data/participants.tsv']:
        if os.path.isfile(p):
            participants_df = pd.read_csv(p, sep='\t')
            print(f"Participants: {len(participants_df)} subjects loaded")
            break

    subjects = discover_subjects()
    print(f"Subject folders: {len(subjects)}")

    # Process each condition
    condition_dfs = {}
    for task, acq in CONDITIONS:
        condition_label = f"{task}_{acq}"
        print(f"\n\n{'#'*70}")
        print(f"  Condition: {condition_label}")
        print(f"{'#'*70}")

        # Filter to subjects with data for this condition
        available = [s for s in subjects if find_edf(s, task, acq) is not None]
        print(f"  Available: {len(available)}/{len(subjects)} subjects")

        if len(available) < 10:
            print(f"  Skipping — too few subjects")
            continue

        loader = lambda sub, t=task, a=acq: load_dortmund_subject(
            sub, task=t, acq=a, participants_df=participants_df)

        dom_df = process_subjects_from_eeg(
            available, loader,
            out_dir=os.path.join(OUT_DIR, condition_label),
            label=f'Dortmund {condition_label}')

        condition_dfs[condition_label] = dom_df

    # Theta EC convergence tests (EC-pre vs EO-pre)
    if 'EyesClosed_pre' in condition_dfs and 'EyesOpen_pre' in condition_dfs:
        print(f"\n  EC vs EO convergence (pre-condition):")
        test_theta_ec_convergence(
            condition_dfs['EyesOpen_pre'], condition_dfs['EyesClosed_pre'])

    if 'EyesClosed_post' in condition_dfs and 'EyesOpen_post' in condition_dfs:
        print(f"\n  EC vs EO convergence (post-condition):")
        test_theta_ec_convergence(
            condition_dfs['EyesOpen_post'], condition_dfs['EyesClosed_post'])

    print(f"\n\nDone! Results saved to {OUT_DIR}")


if __name__ == '__main__':
    main()

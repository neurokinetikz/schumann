#!/usr/bin/env python3
"""
EEGMMIDB Phi-Lattice Replication — Unified Protocol
====================================================

Re-extracts from raw EDF files using the locked protocol.
Verification target: mean_d ≈ 0.069, Cohen's d ≈ 0.38, phi rank 1/9.

Data: /Volumes/T9/eegmmidb/S001/S001R01.edf (R01=EO, R02=EC resting)
Format: 109 subjects (S001-S109), 160 Hz, 64 channels
Preprocessing: pick EEG → notch 60 Hz → keep 160 Hz native
"""

import sys
import os
import mne
import warnings
warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from phi_replication import (process_subjects_from_eeg, run_full_protocol,
                             test_theta_ec_convergence)

# ── Data paths ────────────────────────────────────────────────────────
DATA_DIR = '/Volumes/T9/eegmmidb'
OUT_DIR = 'exports_eegmmidb/replication'


def discover_subjects():
    """Find all subject directories (S001-S109)."""
    subjects = sorted([d for d in os.listdir(DATA_DIR)
                       if d.startswith('S') and len(d) == 4
                       and os.path.isdir(os.path.join(DATA_DIR, d))])
    return subjects


def load_eegmmidb_subject(sub_id, runs=None):
    """Load EEGMMIDB EDF files, concatenate specified runs.

    Preprocessing: pick EEG → notch 60 Hz → keep 160 Hz native
    Returns (data_array, ch_names, fs, metadata_dict)
    """
    if runs is None:
        runs = ['R01', 'R02']  # R01=EO resting, R02=EC resting

    raws = []
    for run in runs:
        edf_path = os.path.join(DATA_DIR, sub_id, f'{sub_id}{run}.edf')
        if not os.path.isfile(edf_path):
            continue
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        raw.pick_types(eeg=True, exclude='bads')
        raws.append(raw)

    if len(raws) == 0:
        raise FileNotFoundError(f"No EDF files found for {sub_id}")

    if len(raws) > 1:
        combined = mne.concatenate_raws(raws)
    else:
        combined = raws[0]

    # Notch filter at 60 Hz (US power line)
    combined.notch_filter([60], verbose=False)

    return (combined.get_data(), combined.ch_names, combined.info['sfreq'],
            {'subject': sub_id})


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"EEGMMIDB Phi-Lattice Replication")
    print(f"Data: {DATA_DIR}")
    print(f"Output: {OUT_DIR}")

    subjects = discover_subjects()
    print(f"Found {len(subjects)} subjects")

    # Combined (R01+R02) — primary analysis
    print(f"\n\n{'#'*70}")
    print(f"  Condition: Combined (R01+R02)")
    print(f"{'#'*70}")

    loader_combined = lambda sub: load_eegmmidb_subject(sub, runs=['R01', 'R02'])
    dom_df_combined = process_subjects_from_eeg(
        subjects, loader_combined,
        out_dir=os.path.join(OUT_DIR, 'combined'),
        label='EEGMMIDB combined')

    # Separate EO (R01) and EC (R02) for convergence test
    condition_dfs = {}
    for run_id, condition in [('R01', 'EO'), ('R02', 'EC')]:
        print(f"\n\n{'#'*70}")
        print(f"  Condition: {condition} ({run_id})")
        print(f"{'#'*70}")

        loader = lambda sub, r=run_id: load_eegmmidb_subject(sub, runs=[r])
        dom_df = process_subjects_from_eeg(
            subjects, loader,
            out_dir=os.path.join(OUT_DIR, condition),
            label=f'EEGMMIDB {condition}')

        condition_dfs[condition] = dom_df

    # Theta EC convergence test
    if 'EO' in condition_dfs and 'EC' in condition_dfs:
        convergence = test_theta_ec_convergence(
            condition_dfs['EO'], condition_dfs['EC'])

    print(f"\n\nDone! Results saved to {OUT_DIR}")


if __name__ == '__main__':
    main()

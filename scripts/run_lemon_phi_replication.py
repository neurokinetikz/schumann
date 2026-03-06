#!/usr/bin/env python3
"""
LEMON Phi-Lattice Replication — Unified Protocol
=================================================

Re-extracts from preprocessed .set files using the locked protocol.
Primary verification target: mean_d ≈ 0.069, Cohen's d ≈ 0.40, phi rank 1/9.

Data: /Volumes/T9/lemon_data/eeg_preprocessed/.../EEG_Preprocessed/
Format: sub-XXXXXX_EC.set and sub-XXXXXX_EO.set, 250 Hz, ~60 channels
Preprocessing: Already ICA-cleaned by LEMON consortium. No additional needed.
"""

import sys
import os
import glob
import mne
import warnings
warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from phi_replication import (process_subjects_from_eeg, run_full_protocol,
                             test_theta_ec_convergence)

# ── Data paths ────────────────────────────────────────────────────────
DATA_DIR = '/Volumes/T9/lemon_data/eeg_preprocessed/EEG_MPILMBB_LEMON/EEG_Preprocessed_BIDS_ID/EEG_Preprocessed'
OUT_DIR = 'exports_lemon/replication'


def discover_subjects(condition='EC'):
    """Find all subjects with .set files for a given condition."""
    pattern = os.path.join(DATA_DIR, f'*_{condition}.set')
    files = sorted(glob.glob(pattern))
    subjects = []
    for f in files:
        basename = os.path.basename(f)
        sub_id = basename.replace(f'_{condition}.set', '')
        subjects.append(sub_id)
    return subjects


def load_lemon_subject(sub_id, condition='EC'):
    """Load LEMON preprocessed .set file.

    Preprocessing: none needed (already preprocessed by consortium).
    Returns (data_array, ch_names, fs, metadata_dict)
    """
    set_path = os.path.join(DATA_DIR, f'{sub_id}_{condition}.set')
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
    raw.pick_types(eeg=True, exclude='bads')

    return (raw.get_data(), raw.ch_names, raw.info['sfreq'],
            {'subject': sub_id})


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"LEMON Phi-Lattice Replication")
    print(f"Data: {DATA_DIR}")
    print(f"Output: {OUT_DIR}")

    # Run EC (primary) and EO
    condition_dfs = {}
    for condition in ['EC', 'EO']:
        print(f"\n\n{'#'*70}")
        print(f"  Condition: {condition}")
        print(f"{'#'*70}")

        subjects = discover_subjects(condition)
        print(f"  Found {len(subjects)} subjects with {condition} data")

        if len(subjects) < 10:
            print(f"  Skipping — too few subjects")
            continue

        loader = lambda sub, cond=condition: load_lemon_subject(sub, cond)
        dom_df = process_subjects_from_eeg(
            subjects, loader,
            out_dir=os.path.join(OUT_DIR, condition),
            label=f'LEMON {condition}')

        condition_dfs[condition] = dom_df

    # Theta EC convergence test (if both conditions available)
    if 'EO' in condition_dfs and 'EC' in condition_dfs:
        convergence = test_theta_ec_convergence(
            condition_dfs['EO'], condition_dfs['EC'])

    print(f"\n\nDone! Results saved to {OUT_DIR}")


if __name__ == '__main__':
    main()

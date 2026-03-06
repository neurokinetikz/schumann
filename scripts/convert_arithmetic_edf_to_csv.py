#!/usr/bin/env python3
"""Convert arithmetic EEG dataset EDF files to pipeline-compatible CSV.

Source: NeXus-32 (BioTrace+) EDF files, 256 Hz, 24 channels (19 EEG + refs + EOG + annotations)
Target: CSV with Timestamp + EEG.<electrode> columns, native 256 Hz, 19 EEG channels

Electrode mapping (old 10-20 → modern 10-20):
    T3→T7, T4→T8, T5→P7, T6→P8, FP1→Fp1, FP2→Fp2

Usage:
    python scripts/convert_arithmetic_edf_to_csv.py
"""

import os
import sys
from glob import glob
from pathlib import Path

import mne
import numpy as np
import pandas as pd

# Old 10-20 (EDF) → Modern 10-20 (pipeline)
EDF_TO_MODERN = {
    'FP1': 'Fp1', 'FP2': 'Fp2',
    'F7': 'F7', 'F3': 'F3', 'FZ': 'Fz', 'F4': 'F4', 'F8': 'F8',
    'T3': 'T7', 'C3': 'C3', 'CZ': 'Cz', 'C4': 'C4', 'T4': 'T8',
    'T5': 'P7', 'P3': 'P3', 'PZ': 'Pz', 'P4': 'P4', 'T6': 'P8',
    'O1': 'O1', 'O2': 'O2',
}

# EEG channel names to pick from EDF (uppercase for matching)
EEG_PICK_NAMES = set(EDF_TO_MODERN.keys())


def convert_edf_to_csv(edf_path, csv_path):
    """Convert a single EDF file to pipeline-compatible CSV.

    Parameters
    ----------
    edf_path : str
        Path to input EDF file.
    csv_path : str
        Path to output CSV file.

    Returns
    -------
    dict
        Conversion metadata.
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    native_fs = raw.info['sfreq']

    # Pick only the 19 EEG channels (case-insensitive match)
    available = raw.ch_names
    picks = []
    pick_order = []  # track which old name each pick corresponds to
    for ch in available:
        ch_upper = ch.strip().upper()
        if ch_upper in EEG_PICK_NAMES:
            picks.append(ch)
            pick_order.append(ch_upper)

    if not picks:
        raise ValueError(f"No EEG channels found in {edf_path}. Available: {available}")

    raw.pick(picks)

    # Extract data and convert to microvolts.
    # MNE converts EDF channels with dim="uV" to Volts (SI), but channels
    # with non-standard dims like "Numerica" (PZ, O2 in some BioTrace+ files)
    # are passed through as raw physical values (already µV).
    # Fix: per-channel scaling based on magnitude.
    data = raw.get_data()  # (n_channels, n_samples)
    for i in range(data.shape[0]):
        ch_max = np.max(np.abs(data[i]))
        if ch_max < 1.0:  # values in Volts → convert to µV
            data[i] *= 1e6

    # Build column names with modern 10-20 naming and EEG. prefix
    col_names = []
    for old_upper in pick_order:
        modern = EDF_TO_MODERN[old_upper]
        col_names.append(f"EEG.{modern}")

    n_samples = data.shape[1]
    df = pd.DataFrame(data.T, columns=col_names)
    df.insert(0, 'Timestamp', np.arange(n_samples) / native_fs)

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)

    return {
        'fs': native_fs,
        'n_samples': n_samples,
        'duration_sec': round(n_samples / native_fs, 1),
        'n_channels': len(col_names),
    }


def main():
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'arithmetic')
    base_dir = os.path.abspath(base_dir)
    out_base = os.path.join(base_dir, 'csv')

    experiments = {
        'Experiment 1': 'Experiment1',
        'Experiment 2': 'Experiment2',
    }

    total = 0
    converted = 0
    errors = []

    for exp_dir, out_subdir in experiments.items():
        edf_files = sorted(glob(os.path.join(base_dir, exp_dir, '*.edf')))
        print(f"\n{'=' * 60}")
        print(f"{exp_dir}: {len(edf_files)} EDF files")
        print(f"{'=' * 60}")

        for edf_path in edf_files:
            total += 1
            stem = Path(edf_path).stem
            csv_path = os.path.join(out_base, out_subdir, f"{stem}.csv")

            try:
                info = convert_edf_to_csv(edf_path, csv_path)
                converted += 1
                print(f"  {stem}: {info['fs']}Hz, "
                      f"{info['duration_sec']}s, {info['n_channels']}ch → {csv_path}")
            except Exception as e:
                errors.append((edf_path, str(e)))
                print(f"  {stem}: ERROR — {e}")

    print(f"\n{'=' * 60}")
    print(f"Conversion complete: {converted}/{total} files")
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for path, err in errors:
            print(f"  {Path(path).name}: {err}")


if __name__ == '__main__':
    main()

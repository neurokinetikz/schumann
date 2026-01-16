#!/usr/bin/env python3
"""
Convert emotions dataset txt files to pipeline-compatible CSV format.

Input: Tab-separated txt files with 14 EEG channels (Emotiv EPOC X format)
Output: CSV files with EEG. prefix headers, compatible with analysis pipeline

Usage:
    python scripts/convert_emotions_to_csv.py
"""
import pandas as pd
import glob
import os
from pathlib import Path

# Column order in txt files (Emotiv EPOC X standard order)
COLUMNS = [
    'EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5', 'EEG.T7', 'EEG.P7',
    'EEG.O1', 'EEG.O2', 'EEG.P8', 'EEG.T8', 'EEG.FC6', 'EEG.F4',
    'EEG.F8', 'EEG.AF4'
]


def convert_file(txt_path: str) -> str:
    """Convert single txt file to CSV.

    Args:
        txt_path: Path to input txt file

    Returns:
        Path to output CSV file
    """
    # Read tab-separated file with no header
    df = pd.read_csv(txt_path, sep='\t', header=None, names=COLUMNS)

    # Save as CSV with index column (matches PhySF format)
    csv_path = txt_path.replace('.txt', '.csv')
    df.to_csv(csv_path, index=True)

    return csv_path


def main():
    """Convert all txt files in data/emotions/ to CSV format."""
    # Find all txt files
    txt_files = sorted(glob.glob('data/emotions/*.txt'))

    if not txt_files:
        print("No txt files found in data/emotions/")
        return

    print(f"Converting {len(txt_files)} files...")

    converted = 0
    errors = []

    for i, txt_path in enumerate(txt_files):
        try:
            convert_file(txt_path)
            converted += 1
        except Exception as e:
            errors.append((txt_path, str(e)))

        # Progress update every 100 files
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(txt_files)}")

    print(f"\nDone! Converted {converted}/{len(txt_files)} files")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for path, err in errors[:10]:  # Show first 10 errors
            print(f"  {Path(path).name}: {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")


if __name__ == '__main__':
    main()

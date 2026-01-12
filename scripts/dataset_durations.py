"""Calculate session durations for all datasets.

Outputs a CSV with:
- Each session: filename, dataset, duration_seconds
- Summary by dataset: total duration, file count
- Grand total across all datasets
"""

import sys; sys.path.insert(0, './lib')
import os
import pandas as pd
import numpy as np
import utilities

# Constants
FS = 128  # Sampling rate

# Electrode configurations
EPOC_ELECTRODES = ['EEG.AF3','EEG.F7','EEG.F3','EEG.FC5','EEG.T7','EEG.P7','EEG.O1',
                   'EEG.O2','EEG.P8','EEG.T8','EEG.FC6','EEG.F4','EEG.F8','EEG.AF4']
INSIGHT_ELECTRODES = ['EEG.AF3','EEG.AF4','EEG.T7','EEG.T8','EEG.Pz']
MUSE_ELECTRODES = ['EEG.TP9','EEG.AF7','EEG.AF8','EEG.TP10']

def list_csv_files(directory):
    """List all CSV files in a directory."""
    if not os.path.exists(directory):
        return []
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]

# Dataset definitions
FILES = [
    ('data/test schumann_EPOCX_111270_2023.04.23T14.50.35.05.00.md.pm.bp.csv', 1),
    ('data/Test_06.11.20_14.28.18.md.pm.bp.csv', 1),
    ('data/20201229_29.12.20_11.27.57.md.pm.bp.csv', 1),
    ('data/med_EPOCX_111270_2021.06.12T09.50.52.04.00.md.bp.csv', 1),
    ('data/binaural_EPOCX_111270_2021.06.17T10.04.52.04.00.md.bp.csv', 1),
]

PHYSF = [(f, 0) for f in list_csv_files("data/PhySF")]
VEP = [(f, 0) for f in list_csv_files("data/vep")]
MPENG1 = [(f, 0) for f in list_csv_files("data/mpeng1")]
MPENG2 = [(f, 0) for f in list_csv_files("data/mpeng2")]
INSIGHT = [(f, 1) for f in list_csv_files("data/insight")]
MUSE = [(f, 0) for f in list_csv_files("data/muse")]
# ArEEG excluded per user request
# ArEEG = [(f, 1) for f in list_csv_files("data/ArEEG")]

# Dataset configurations: (name, files_list, electrodes, device, fs)
DATASET_CONFIGS = [
    ('FILES', FILES, EPOC_ELECTRODES, 'emotiv', 128),
    ('INSIGHT', INSIGHT, INSIGHT_ELECTRODES, 'emotiv', 128),
    ('MUSE', MUSE, MUSE_ELECTRODES, 'muse', 256),
    ('PHYSF', PHYSF, EPOC_ELECTRODES, 'emotiv', 128),
    ('VEP', VEP, EPOC_ELECTRODES, 'emotiv', 128),
    ('MPENG1', MPENG1, EPOC_ELECTRODES, 'emotiv', 128),
    ('MPENG2', MPENG2, EPOC_ELECTRODES, 'emotiv', 128),
]


def get_session_duration(filepath, header, electrodes, fs=128, device='emotiv'):
    """Get duration in seconds for a single session."""
    try:
        records = utilities.load_eeg_csv(filepath, electrodes=electrodes, fs=fs, header=header, device=device)
        n_samples = len(records)
        duration_sec = n_samples / fs
        return duration_sec
    except Exception as e:
        return None


def format_duration(seconds):
    """Format seconds as HH:MM:SS."""
    if seconds is None:
        return "N/A"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main():
    print("=" * 80)
    print("Dataset Duration Analysis")
    print("=" * 80)

    all_sessions = []
    dataset_summaries = []

    for dataset_name, files_list, electrodes, device, fs in DATASET_CONFIGS:
        print(f"\n{dataset_name}: {len(files_list)} files")

        dataset_durations = []

        for i, (filepath, header) in enumerate(files_list):
            filename = os.path.basename(filepath)

            if not os.path.exists(filepath):
                print(f"  [{i+1}/{len(files_list)}] {filename[:50]} - NOT FOUND")
                continue

            duration = get_session_duration(filepath, header, electrodes, fs=fs, device=device)

            if duration is not None:
                dataset_durations.append(duration)
                all_sessions.append({
                    'dataset': dataset_name,
                    'filename': filename,
                    'filepath': filepath,
                    'duration_sec': duration,
                    'duration_min': duration / 60,
                    'duration_formatted': format_duration(duration)
                })

                if (i + 1) % 50 == 0 or i == len(files_list) - 1:
                    print(f"  [{i+1}/{len(files_list)}] Processed...")
            else:
                print(f"  [{i+1}/{len(files_list)}] {filename[:50]} - FAILED")

        # Dataset summary
        if dataset_durations:
            total_sec = sum(dataset_durations)
            dataset_summaries.append({
                'dataset': dataset_name,
                'file_count': len(files_list),
                'successful_count': len(dataset_durations),
                'total_duration_sec': total_sec,
                'total_duration_min': total_sec / 60,
                'total_duration_hours': total_sec / 3600,
                'total_duration_formatted': format_duration(total_sec),
                'mean_duration_sec': np.mean(dataset_durations),
                'mean_duration_min': np.mean(dataset_durations) / 60,
                'min_duration_sec': min(dataset_durations),
                'max_duration_sec': max(dataset_durations),
            })
            print(f"  Total: {format_duration(total_sec)} ({len(dataset_durations)} sessions)")

    # Create DataFrames
    sessions_df = pd.DataFrame(all_sessions)
    summary_df = pd.DataFrame(dataset_summaries)

    # Grand totals
    grand_total_sec = sessions_df['duration_sec'].sum()
    grand_total_sessions = len(sessions_df)

    # Save to CSV
    sessions_df.to_csv('dataset_session_durations.csv', index=False)
    summary_df.to_csv('dataset_duration_summary.csv', index=False)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY BY DATASET")
    print("=" * 80)
    print(f"{'Dataset':<12} {'Files':>8} {'Success':>8} {'Total Duration':>15} {'Mean (min)':>12}")
    print("-" * 60)

    for _, row in summary_df.iterrows():
        print(f"{row['dataset']:<12} {row['file_count']:>8} {row['successful_count']:>8} "
              f"{row['total_duration_formatted']:>15} {row['mean_duration_min']:>12.1f}")

    print("-" * 60)
    print(f"{'TOTAL':<12} {summary_df['file_count'].sum():>8} {grand_total_sessions:>8} "
          f"{format_duration(grand_total_sec):>15} {(grand_total_sec/grand_total_sessions/60):>12.1f}")

    print(f"\nGrand Total: {format_duration(grand_total_sec)} = {grand_total_sec/3600:.2f} hours")
    print(f"Sessions: {grand_total_sessions}")

    print("\n" + "=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)
    print("  dataset_session_durations.csv - Individual session durations")
    print("  dataset_duration_summary.csv  - Summary by dataset")

    return sessions_df, summary_df


if __name__ == "__main__":
    sessions_df, summary_df = main()

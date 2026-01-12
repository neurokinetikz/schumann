#!/usr/bin/env python3
"""
Null Control Test 4: Event Quality vs Random Windows

Tests if detected SIE events have significantly better metrics compared to
random 20s windows from the same recordings.

Methodology:
1. Run detection pipeline to get ignition windows with their metrics
2. Generate random 20s windows OUTSIDE of ignition windows
3. For random windows, use the FULL detection results to extract metrics from those time periods
4. Compare observed vs random metrics

Pass Criteria:
- Observed sr_score significantly better: p < 0.01
- Effect size > 0.5
- Observed in top 10% of random distribution
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict
from scipy import stats

# Add lib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

from detect_ignition import detect_ignitions_session

# ============================================================================
# Configuration
# ============================================================================

# Use specific test files
FILES = [
    'data/test schumann_EPOCX_111270_2023.04.23T14.50.35.05.00.md.pm.bp.csv',
    'data/20201229_29.12.20_11.27.57.md.pm.bp.csv',
    'data/binaural_EPOCX_111270_2021.06.17T10.04.52.04.00.md.bp.csv',
]

DATASET_SELECTION = {
    'FILES': True,
    'PHYSF': True,  # Use PhySF which has known events
    'INSIGHT': True,
    'MUSE': True,
}

MAX_FILES = 2  # Just test with 2 files

# Random window parameters
WINDOW_DURATION_SEC = 20.0
COVERAGE_FRACTION = 0.25

# ============================================================================
# File Selection
# ============================================================================

def get_file_list():
    """Get list of EEG files based on dataset selection."""
    all_files = []

    if DATASET_SELECTION.get('FILES', False):
        # Use specific FILES list
        all_files.extend(FILES)

    if DATASET_SELECTION.get('PHYSF', False):
        physf_dir = Path('data/PhySF')
        if physf_dir.exists():
            physf_files = list(physf_dir.glob('*.csv'))
            all_files.extend([str(f) for f in physf_files])

    if DATASET_SELECTION.get('INSIGHT', False):
        insight_dir = Path('data/insight')
        if insight_dir.exists():
            insight_files = list(insight_dir.glob('*.csv'))
            all_files.extend([str(f) for f in insight_files])

    if DATASET_SELECTION.get('MUSE', False):
        muse_dir = Path('data/muse')
        if muse_dir.exists():
            muse_files = list(muse_dir.glob('*.csv'))
            all_files.extend([str(f) for f in muse_files])

    return all_files[:MAX_FILES] if MAX_FILES else all_files


def get_device_config(file_path: str):
    """Determine device configuration from file path."""
    file_lower = file_path.lower()

    ELECTRODES = ['EEG.AF3', 'EEG.AF4', 'EEG.F3', 'EEG.F4', 'EEG.F7', 'EEG.F8',
                  'EEG.FC5', 'EEG.FC6', 'EEG.P7', 'EEG.P8', 'EEG.O1', 'EEG.O2',
                  'EEG.T7', 'EEG.T8']
    INSIGHT_ELECTRODES = ['EEG.AF3', 'EEG.AF4', 'EEG.T7', 'EEG.T8', 'EEG.Pz']
    MUSE_ELECTRODES = ['EEG.AF7', 'EEG.AF8', 'EEG.TP9', 'EEG.TP10']

    if 'insight' in file_lower:
        return {'sr_channel': 'EEG.T7', 'channels': INSIGHT_ELECTRODES}
    elif 'muse' in file_lower:
        return {'sr_channel': 'EEG.TP9', 'channels': MUSE_ELECTRODES}
    else:
        return {'sr_channel': 'EEG.T7', 'channels': ELECTRODES}


# ============================================================================
# Detection and Metric Collection
# ============================================================================

def run_detection_and_collect_metrics(file_path: str):
    """
    Run detection pipeline and collect metrics from both detected events
    and random windows.

    Returns:
        observed_events: DataFrame with detected event details
        random_events: DataFrame with random window details
    """
    import tempfile

    device_config = get_device_config(file_path)

    # Load data
    first_line = pd.read_csv(file_path, nrows=1, header=None).iloc[0, 0]
    if str(first_line).startswith('title:'):
        df = pd.read_csv(file_path, skiprows=1, low_memory=False)
    elif 'm1' in file_path.lower():
        df = pd.read_csv(file_path, header=1, low_memory=False)
    else:
        df = pd.read_csv(file_path, header=0, low_memory=False)

    # Handle MUSE format
    if 'muse' in file_path.lower():
        df.rename(columns={
            'timestamps': 'Timestamp',
            'eeg_1': 'EEG.TP9',
            'eeg_2': 'EEG.AF7',
            'eeg_3': 'EEG.AF8',
            'eeg_4': 'EEG.TP10'
        }, inplace=True)

    # Create Timestamp if needed
    if 'Timestamp' not in df.columns:
        df['Timestamp'] = df.index / 128.0

    # Validate channels
    missing = [ch for ch in device_config['channels'] if ch not in df.columns]
    if missing:
        raise ValueError(f"Missing channels {missing}")

    recording_duration = df['Timestamp'].max() - df['Timestamp'].min()

    # Run detection with verbose=True to get sr_scores
    with tempfile.TemporaryDirectory() as temp_dir:
        results, intervals = detect_ignitions_session(
            RECORDZ=df,
            sr_channel=device_config['sr_channel'],
            eeg_channels=device_config['channels'],
            time_col='Timestamp',
            out_dir=temp_dir,
            center_hz=7.6,
            harmonics_hz=[7.6, 20.0, 32.0],
            half_bw_hz=[0.6, 1.0, 2.0],
            smooth_sec=0.01,
            z_thresh=3.0,
            min_isi_sec=2.0,
            window_sec=20.0,
            merge_gap_sec=10.0,
            sr_reference='auto-SSD',
            seed_method='latency',
            pel_band=(25, 45),
            harmonic_method='psd',  # Use PSD for speed
            make_passport=False,
            show=False,
            verbose=True,  # Required for sr_score
            session_name=os.path.basename(file_path)
        )

    events_df = results.get('events', pd.DataFrame())

    if len(events_df) == 0 or 'sr_score' not in events_df.columns:
        return pd.DataFrame(), pd.DataFrame()

    # Add session name to observed events
    events_df['session'] = os.path.basename(file_path)
    events_df['type'] = 'Observed'

    # Generate random windows outside of detected events
    event_intervals = [(row['t_start'], row['t_end']) for _, row in events_df.iterrows()]
    random_windows = generate_random_windows(recording_duration, event_intervals)

    if len(random_windows) == 0:
        return events_df, pd.DataFrame()

    # For random windows, we'll use a simpler approach:
    # Run detection again with VERY low threshold to get metrics everywhere,
    # then extract scores from random window time periods
    print(f"  Analyzing {len(random_windows)} random windows...")

    # Re-run with z_thresh=0.5 to detect even weak activity
    with tempfile.TemporaryDirectory() as temp_dir:
        results_low, intervals_low = detect_ignitions_session(
            RECORDZ=df,
            sr_channel=device_config['sr_channel'],
            eeg_channels=device_config['channels'],
            time_col='Timestamp',
            out_dir=temp_dir,
            center_hz=7.6,
            harmonics_hz=[7.6, 20.0, 32.0],
            half_bw_hz=[0.6, 1.0, 2.0],
            smooth_sec=0.01,
            z_thresh=0.5,  # Very low to get coverage
            min_isi_sec=0.5,
            window_sec=20.0,
            merge_gap_sec=1.0,  # Less merging
            sr_reference='auto-SSD',
            seed_method='latency',
            pel_band=(25, 45),
            harmonic_method='psd',
            make_passport=False,
            show=False,
            verbose=True,
            session_name=os.path.basename(file_path)
        )

    events_low_df = results_low.get('events', pd.DataFrame())

    if len(events_low_df) == 0 or 'sr_score' not in events_low_df.columns:
        return events_df, pd.DataFrame()

    # For each random window, find events that overlap and collect metrics
    random_event_rows = []
    for i, (t_start, t_end) in enumerate(random_windows):
        # Find events that overlap this random window
        overlaps = events_low_df[
            (events_low_df['t_start'] < t_end) & (events_low_df['t_end'] > t_start)
        ]

        if len(overlaps) > 0:
            # Take mean of key metrics from overlapping events
            random_event_rows.append({
                'session': os.path.basename(file_path),
                'type': 'Random',
                't_start': t_start,
                't_end': t_end,
                'duration_s': t_end - t_start,
                'sr_score': overlaps['sr_score'].mean(),
                'sr_z_max': overlaps['sr_z_max'].mean() if 'sr_z_max' in overlaps.columns else np.nan,
                'msc_7p83_v': overlaps['msc_7p83_v'].mean() if 'msc_7p83_v' in overlaps.columns else np.nan,
                'plv_mean_pm5': overlaps['plv_mean_pm5'].mean() if 'plv_mean_pm5' in overlaps.columns else np.nan,
                'HSI': overlaps['HSI'].mean() if 'HSI' in overlaps.columns else np.nan,
            })
        else:
            # No activity in this window
            random_event_rows.append({
                'session': os.path.basename(file_path),
                'type': 'Random',
                't_start': t_start,
                't_end': t_end,
                'duration_s': t_end - t_start,
                'sr_score': 0.0,
                'sr_z_max': 0.0,
                'msc_7p83_v': 0.0,
                'plv_mean_pm5': 0.0,
                'HSI': 0.0,
            })

    random_events_df = pd.DataFrame(random_event_rows)

    return events_df, random_events_df


def generate_random_windows(recording_duration: float,
                           event_intervals: List[Tuple[float, float]]):
    """Generate random windows outside of event intervals."""
    # Calculate total event duration
    event_duration = sum(t_end - t_start for t_start, t_end in event_intervals)
    non_event_duration = recording_duration - event_duration

    if non_event_duration < WINDOW_DURATION_SEC:
        return []

    # Number of windows to cover COVERAGE_FRACTION of non-event time
    n_windows = int(non_event_duration * COVERAGE_FRACTION / WINDOW_DURATION_SEC)
    n_windows = max(1, min(n_windows, 50))  # Limit to reasonable number

    random_windows = []
    max_attempts = n_windows * 10

    for _ in range(max_attempts):
        if len(random_windows) >= n_windows:
            break

        # Random start time
        t_start = np.random.uniform(0, recording_duration - WINDOW_DURATION_SEC)
        t_end = t_start + WINDOW_DURATION_SEC

        # Check if it overlaps with any event interval
        overlaps = any(
            (t_start < ev_end and t_end > ev_start)
            for ev_start, ev_end in event_intervals
        )

        # Check if it overlaps with existing random windows
        overlaps_random = any(
            (t_start < rw_end and t_end > rw_start)
            for rw_start, rw_end in random_windows
        )

        if not overlaps and not overlaps_random:
            random_windows.append((t_start, t_end))

    return random_windows


# ============================================================================
# Statistical Testing
# ============================================================================

def perform_statistical_test(observed_scores, random_scores):
    """Compare observed vs random scores."""
    # Mann-Whitney U test (non-parametric)
    u_stat, p_value = stats.mannwhitneyu(observed_scores, random_scores, alternative='greater')

    # Cohen's d effect size
    pooled_std = np.sqrt((np.std(observed_scores)**2 + np.std(random_scores)**2) / 2)
    cohens_d = (np.mean(observed_scores) - np.mean(random_scores)) / (pooled_std + 1e-10)

    # Percentile rank
    all_scores = np.concatenate([observed_scores, random_scores])
    obs_median = np.median(observed_scores)
    percentile = stats.percentileofscore(all_scores, obs_median)

    return {
        'p_value': p_value,
        'cohens_d': cohens_d,
        'percentile': percentile,
        'obs_mean': np.mean(observed_scores),
        'obs_median': np.median(observed_scores),
        'random_mean': np.mean(random_scores),
        'random_median': np.median(random_scores),
        'n_observed': len(observed_scores),
        'n_random': len(random_scores)
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("NULL CONTROL TEST 4: Event Quality vs Random Windows")
    print("=" * 80)
    print()

    # Get file list
    all_files = get_file_list()
    print(f"Processing {len(all_files)} files...")
    print()

    all_observed_events = []
    all_random_events = []

    for i, file_path in enumerate(all_files, 1):
        print(f"[{i}/{len(all_files)}] {os.path.basename(file_path)}")

        try:
            obs_df, rand_df = run_detection_and_collect_metrics(file_path)

            if len(obs_df) > 0 and len(rand_df) > 0:
                all_observed_events.append(obs_df)
                all_random_events.append(rand_df)
                print(f"  Events: {len(obs_df)}, mean sr_score: {obs_df['sr_score'].mean():.4f}")
                print(f"  Random: {len(rand_df)}, mean sr_score: {rand_df['sr_score'].mean():.4f}")
            else:
                print("  Skipped (no data)")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(all_observed_events) == 0 or len(all_random_events) == 0:
        print("\nERROR: Insufficient data")
        return

    # Concatenate all events
    observed_df = pd.concat(all_observed_events, ignore_index=True)
    random_df = pd.concat(all_random_events, ignore_index=True)

    print()
    print(f"Total observed events: {len(observed_df)}")
    print(f"Total random windows: {len(random_df)}")

    # Display columns of interest
    display_cols = ['session', 'type', 't_start', 't_end', 'duration_s', 'sr_score',
                   'sr_z_max', 'msc_7p83_v', 'plv_mean_pm5', 'HSI']

    # Filter to columns that exist
    obs_display_cols = [c for c in display_cols if c in observed_df.columns]
    rand_display_cols = [c for c in display_cols if c in random_df.columns]

    print()
    print("=" * 80)
    print("OBSERVED EVENTS (Detected with z_thresh=3.0)")
    print("=" * 80)
    obs_display = observed_df[obs_display_cols].copy()
    for col in ['t_start', 't_end', 'duration_s']:
        if col in obs_display.columns:
            obs_display[col] = obs_display[col].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "")
    for col in ['sr_score', 'sr_z_max', 'msc_7p83_v', 'plv_mean_pm5', 'HSI']:
        if col in obs_display.columns:
            obs_display[col] = obs_display[col].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "")
    print(obs_display.to_string(index=False))

    print()
    print("=" * 80)
    print("RANDOM WINDOWS (Sampled outside detected events, metrics from z_thresh=0.5)")
    print("=" * 80)
    rand_display = random_df[rand_display_cols].copy()
    for col in ['t_start', 't_end', 'duration_s']:
        if col in rand_display.columns:
            rand_display[col] = rand_display[col].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "")
    for col in ['sr_score', 'sr_z_max', 'msc_7p83_v', 'plv_mean_pm5', 'HSI']:
        if col in rand_display.columns:
            rand_display[col] = rand_display[col].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "")
    print(rand_display.to_string(index=False))

    # Statistical test
    results = perform_statistical_test(
        observed_df['sr_score'].values,
        random_df['sr_score'].values
    )

    print()
    print("=" * 80)
    print("STATISTICAL RESULTS")
    print("=" * 80)
    print(f"Observed events (n={results['n_observed']}): mean={results['obs_mean']:.4f}, median={results['obs_median']:.4f}")
    print(f"Random windows (n={results['n_random']}): mean={results['random_mean']:.4f}, median={results['random_median']:.4f}")
    print()
    print(f"Mann-Whitney U test: p = {results['p_value']:.6f}")
    print(f"Cohen's d: {results['cohens_d']:.3f}")
    print(f"Percentile rank: {results['percentile']:.1f}%")
    print()

    # Pass/Fail
    passes = results['p_value'] < 0.01 and results['cohens_d'] > 0.5 and results['percentile'] > 90
    print(f"Test: {'PASS' if passes else 'FAIL'}")
    print("=" * 80)


if __name__ == '__main__':
    main()

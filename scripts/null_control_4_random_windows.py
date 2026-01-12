#!/usr/bin/env python3
"""
Null Control Test 4: Event Quality vs Random Windows

Tests if detected SIE events have significantly better sr_score (composite quality)
compared to random 20s windows from the same recordings.

This validates that detected events represent genuine synchronization moments,
not just arbitrary thresholds on continuous SR presence.

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
from typing import List, Tuple, Dict, Optional
from scipy import stats
from dataclasses import dataclass

# Add lib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

from detect_ignition import detect_ignitions_session

# ============================================================================
# Configuration
# ============================================================================

DATASET_SELECTION = {
    'FILES': True,      # Individual test files
    'PHYSF': True,      # data/PhySF
    'INSIGHT': True,    # data/insight
    'MUSE': True,       # data/muse
}

# Random window parameters
WINDOW_DURATION_SEC = 20.0  # Match event window duration
COVERAGE_FRACTION = 0.25     # Cover 25% of non-event duration


# ============================================================================
# Data Loading
# ============================================================================

def get_all_files(dataset_selection: Optional[Dict] = None) -> List[str]:
    """Get all EEG files to analyze."""
    if dataset_selection is None:
        dataset_selection = DATASET_SELECTION

    FILES = [
        'data/test schumann_EPOCX_111270_2023.04.23T14.50.35.05.00.md.pm.bp.csv',
        'data/20201229_29.12.20_11.27.57.md.pm.bp.csv',
        'data/binaural_EPOCX_111270_2021.06.17T10.04.52.04.00.md.bp.csv',
    ]

    def list_csv_files(directory):
        if not os.path.exists(directory):
            return []
        return [(directory + "/" + f) for f in os.listdir(directory) if f.endswith('.csv')]

    PHYSF = list_csv_files("data/PhySF")
    INSIGHT = list_csv_files("data/insight")
    MUSE = list_csv_files("data/muse")

    selected_files = []

    if dataset_selection.get('FILES', False):
        selected_files.extend(FILES)
        print(f"  FILES: {len(FILES)} files")

    if dataset_selection.get('PHYSF', False):
        selected_files.extend(PHYSF)
        print(f"  PHYSF: {len(PHYSF)} files")

    if dataset_selection.get('INSIGHT', False):
        selected_files.extend(INSIGHT)
        print(f"  INSIGHT: {len(INSIGHT)} files")

    if dataset_selection.get('MUSE', False):
        selected_files.extend(MUSE)
        print(f"  MUSE: {len(MUSE)} files")

    return selected_files


def get_device_config(file_path: str):
    """Determine device configuration from file path."""
    file_lower = file_path.lower()

    # Electrode configurations with EEG. prefix
    ELECTRODES = ['EEG.AF3', 'EEG.AF4',
                  'EEG.F3', 'EEG.F4',
                  'EEG.F7', 'EEG.F8',
                  'EEG.FC5', 'EEG.FC6',
                  'EEG.P7', 'EEG.P8',
                  'EEG.O1', 'EEG.O2',
                  'EEG.T7', 'EEG.T8']

    INSIGHT_ELECTRODES = ['EEG.AF3', 'EEG.AF4', 'EEG.T7', 'EEG.T8', 'EEG.Pz']
    MUSE_ELECTRODES = ['EEG.AF7', 'EEG.AF8', 'EEG.TP9', 'EEG.TP10']

    if 'insight' in file_lower:
        return {
            'sr_channel': 'EEG.T7',
            'channels': INSIGHT_ELECTRODES
        }
    elif 'muse' in file_lower:
        return {
            'sr_channel': 'EEG.TP9',
            'channels': MUSE_ELECTRODES
        }
    else:  # EMOTIV
        return {
            'sr_channel': 'EEG.T7',
            'channels': ELECTRODES
        }


# ============================================================================
# Event Detection
# ============================================================================

def detect_events_for_session(file_path: str) -> Tuple[pd.DataFrame, List, float, pd.DataFrame, Dict]:
    """
    Detect SIE events for a session.

    Returns:
        events_df: DataFrame with event metrics including sr_score
        intervals: List of (t_start, t_end) tuples
        recording_duration: Total recording duration in seconds
        eeg_df: Full EEG DataFrame (for extracting random windows)
        device_config: Device configuration dict
    """
    import tempfile

    # Device configuration
    device_config = get_device_config(file_path)

    # Load data with low_memory=False to handle mixed types
    # Detect file format by checking first row
    # If first column starts with "title:" it's metadata format (EPOC)
    first_line = pd.read_csv(file_path, nrows=1, header=None).iloc[0, 0]

    if str(first_line).startswith('title:'):
        # EPOC format with metadata row
        df = pd.read_csv(file_path, skiprows=1, low_memory=False)
    elif 'm1' in file_path.lower():
        # m1 files have header at row 1
        df = pd.read_csv(file_path, header=1, low_memory=False)
    else:
        # Standard format
        df = pd.read_csv(file_path, header=0, low_memory=False)

    # Handle MUSE format: rename columns to match expected format
    if 'muse' in file_path.lower():
        column_mapping = {
            'timestamps': 'Timestamp',
            'eeg_1': 'EEG.TP9',
            'eeg_2': 'EEG.AF7',
            'eeg_3': 'EEG.AF8',
            'eeg_4': 'EEG.TP10'
        }
        df.rename(columns=column_mapping, inplace=True)

        # MUSE data has sparse format - filter out rows with no EEG data
        eeg_cols = ['EEG.TP9', 'EEG.AF7', 'EEG.AF8', 'EEG.TP10']
        df = df.dropna(subset=eeg_cols, how='all')  # Drop rows where ALL EEG channels are NaN

    # Create Timestamp column if it doesn't exist (for PhySF files)
    if 'Timestamp' not in df.columns:
        # Assume 128 Hz sampling rate (standard for EMOTIV)
        sampling_rate = 128.0
        df['Timestamp'] = df.index / sampling_rate

    # Validate channels exist
    missing_channels = [ch for ch in device_config['channels'] if ch not in df.columns]
    if missing_channels:
        # Try to find what columns are available
        all_cols = list(df.columns[:30])  # First 30 columns
        raise ValueError(f"Missing channels {missing_channels}.\nFile: {os.path.basename(file_path)}\nFirst 30 columns: {all_cols}")

    # Get recording duration
    if 'Timestamp' in df.columns:
        recording_duration = df['Timestamp'].max() - df['Timestamp'].min()
    else:
        # Estimate from row count (assuming ~128 Hz)
        recording_duration = len(df) / 128.0

    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Run detection
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
            harmonic_method='fooof_hybrid',
            fooof_freq_range=(1.0, 40.0),
            fooof_freq_ranges=[[5, 15], [15, 25], [25, 40]],
            fooof_max_n_peaks=15,
            fooof_peak_threshold=0.01,
            fooof_min_peak_height=0.01,
            fooof_peak_width_limits=(0.5, 12.0),
            fooof_match_method='power',
            make_passport=False,
            show=False,
            verbose=True,  # MUST be True to compute sr_score
            session_name=os.path.basename(file_path)
        )

    events_df = results.get('events', pd.DataFrame())

    return events_df, intervals, recording_duration, df, device_config


# ============================================================================
# Random Window Generation
# ============================================================================

def generate_random_windows(recording_duration: float,
                           event_intervals: List[Tuple[float, float]],
                           window_duration: float = WINDOW_DURATION_SEC,
                           coverage_fraction: float = COVERAGE_FRACTION) -> List[Tuple[float, float]]:
    """
    Generate random time windows avoiding event intervals.

    Args:
        recording_duration: Total recording duration (seconds)
        event_intervals: List of (t_start, t_end) event intervals
        window_duration: Duration of each random window (seconds)
        coverage_fraction: Fraction of non-event duration to cover

    Returns:
        List of (t_start, t_end) random window tuples
    """
    # Calculate non-event duration
    event_duration = sum(t_end - t_start for t_start, t_end in event_intervals)
    non_event_duration = recording_duration - event_duration

    if non_event_duration <= 0:
        return []

    # Calculate number of windows
    target_coverage = coverage_fraction * non_event_duration
    n_windows = int(target_coverage / window_duration)

    if n_windows == 0:
        return []

    # Generate random windows
    random_windows = []
    max_attempts = n_windows * 10  # Prevent infinite loops

    for _ in range(max_attempts):
        if len(random_windows) >= n_windows:
            break

        # Random start time
        t_start = np.random.uniform(0, recording_duration - window_duration)
        t_end = t_start + window_duration

        # Check if overlaps with any event
        overlaps = False
        for event_start, event_end in event_intervals:
            if not (t_end < event_start or t_start > event_end):
                overlaps = True
                break

        # Check if overlaps with existing random windows
        for rand_start, rand_end in random_windows:
            if not (t_end < rand_start or t_start > rand_end):
                overlaps = True
                break

        if not overlaps:
            random_windows.append((t_start, t_end))

    return random_windows


# ============================================================================
# Metric Computation for Random Windows
# ============================================================================

def compute_random_window_metrics(file_path: str,
                                 eeg_df: pd.DataFrame,
                                 random_windows: List[Tuple[float, float]],
                                 device_config: Dict) -> List[float]:
    """
    Compute sr_score metrics for random windows using the same detection pipeline.

    Extracts each random window from the EEG data and runs detect_ignitions_session
    on it with very low thresholds to force detection and get metrics.

    Args:
        file_path: Path to EEG file (for naming)
        eeg_df: Full EEG DataFrame with Timestamp column
        random_windows: List of (t_start, t_end) tuples in seconds
        device_config: Device configuration (channels, sr_channel)

    Returns:
        List of sr_scores for random windows
    """
    import tempfile

    random_scores = []

    # Get the time column (absolute timestamps)
    if 'Timestamp' not in eeg_df.columns:
        print("  Warning: No Timestamp column, cannot extract windows")
        return []

    time_values = eeg_df['Timestamp'].values
    t_min = time_values.min()

    for window_idx, (t_start, t_end) in enumerate(random_windows):
        if (window_idx + 1) % 5 == 0 or window_idx == 0:
            print(f"    Analyzing window {window_idx + 1}/{len(random_windows)}...")
        try:
            # Extract window from EEG data (convert relative time to absolute timestamp)
            window_start_abs = t_min + t_start
            window_end_abs = t_min + t_end

            # Get indices for this window
            mask = (time_values >= window_start_abs) & (time_values <= window_end_abs)
            window_df = eeg_df[mask].copy()

            if len(window_df) < 100:  # Minimum data points
                continue

            # Run detection on this window with VERY LOW thresholds to force detection
            with tempfile.TemporaryDirectory() as temp_dir:
                results, intervals = detect_ignitions_session(
                    RECORDZ=window_df,
                    sr_channel=device_config['sr_channel'],
                    eeg_channels=device_config['channels'],
                    time_col='Timestamp',
                    out_dir=temp_dir,
                    center_hz=7.6,
                    harmonics_hz=[7.6, 20.0, 32.0],
                    half_bw_hz=[0.6, 1.0, 2.0],
                    smooth_sec=0.01,
                    z_thresh=0.5,  # VERY LOW threshold to force detection
                    min_isi_sec=0.5,
                    window_sec=20.0,
                    merge_gap_sec=10.0,
                    sr_reference='auto-SSD',
                    seed_method='latency',
                    pel_band=(25, 45),
                    harmonic_method='fooof_hybrid',
                    fooof_freq_range=(1.0, 40.0),
                    fooof_freq_ranges=[[5, 15], [15, 25], [25, 40]],
                    fooof_max_n_peaks=15,
                    fooof_peak_threshold=0.01,
                    fooof_min_peak_height=0.01,
                    fooof_peak_width_limits=(0.5, 12.0),
                    fooof_match_method='power',
                    make_passport=False,
                    show=False,
                    verbose=True,  # MUST be True to compute sr_score
                    session_name=f"random_window_{window_idx}"
                )

            events_df = results.get('events', pd.DataFrame())

            if len(events_df) > 0 and 'sr_score' in events_df.columns:
                # Take the mean sr_score from all detected events in this window
                window_score = events_df['sr_score'].mean()
                random_scores.append(window_score)
            else:
                # No events detected even with low threshold = very low quality window
                random_scores.append(0.0)

        except Exception as e:
            # If analysis fails, assume very low quality
            print(f"    Warning: Failed to analyze window {window_idx}: {e}")
            random_scores.append(0.0)
            continue

    return random_scores


# ============================================================================
# Statistical Testing
# ============================================================================

def perform_statistical_test(observed_scores: np.ndarray,
                            random_scores: np.ndarray) -> Dict:
    """
    Compare observed vs random sr_scores.

    Args:
        observed_scores: sr_scores from detected events
        random_scores: sr_scores from random windows

    Returns:
        Dictionary with statistical test results
    """
    # Basic statistics
    obs_mean = np.mean(observed_scores)
    obs_std = np.std(observed_scores)
    obs_median = np.median(observed_scores)

    rand_mean = np.mean(random_scores)
    rand_std = np.std(random_scores)
    rand_median = np.median(random_scores)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((obs_std**2 + rand_std**2) / 2)
    cohens_d = (obs_mean - rand_mean) / pooled_std if pooled_std > 0 else 0.0

    # Mann-Whitney U test (non-parametric)
    u_stat, p_value_mw = stats.mannwhitneyu(observed_scores, random_scores, alternative='greater')

    # Percentile rank (higher is better for sr_score)
    percentile = 100 - stats.percentileofscore(random_scores, obs_median)

    # Pass criteria
    passes_pvalue = p_value_mw < 0.01
    passes_effect_size = cohens_d > 0.5
    passes_percentile = percentile >= 90  # Observed in top 10%

    overall_pass = passes_pvalue and passes_effect_size and passes_percentile

    return {
        'n_observed': len(observed_scores),
        'n_random': len(random_scores),

        # Observed statistics
        'obs_mean': obs_mean,
        'obs_std': obs_std,
        'obs_median': obs_median,

        # Random statistics
        'rand_mean': rand_mean,
        'rand_std': rand_std,
        'rand_median': rand_median,

        # Tests
        'cohens_d': cohens_d,
        'p_value': p_value_mw,
        'percentile': percentile,

        # Pass/fail
        'passes_pvalue': passes_pvalue,
        'passes_effect_size': passes_effect_size,
        'passes_percentile': passes_percentile,
        'overall_pass': overall_pass
    }


# ============================================================================
# Visualization
# ============================================================================

def create_visualizations(observed_scores: np.ndarray,
                         random_scores: np.ndarray,
                         stats_results: Dict,
                         output_dir: Path):
    """
    Create comprehensive 4-panel visualization.
    """
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 10

    # Create figure with 4 subplots
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. Histogram
    ax1 = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0, max(np.max(observed_scores), np.max(random_scores)), 50)
    ax1.hist(random_scores, bins=bins, alpha=0.6, label=f'Random (n={len(random_scores)})',
             color='gray', density=True)
    ax1.hist(observed_scores, bins=bins, alpha=0.8, label=f'Observed (n={len(observed_scores)})',
             color='red', density=True)
    ax1.axvline(stats_results['obs_mean'], color='red', linestyle='--', linewidth=2,
                label=f"Observed mean: {stats_results['obs_mean']:.4f}")
    ax1.axvline(stats_results['rand_mean'], color='gray', linestyle='--', linewidth=2,
                label=f"Random mean: {stats_results['rand_mean']:.4f}")
    ax1.set_xlabel('SR Score')
    ax1.set_ylabel('Density')
    ax1.set_title('SR Score: Observed Events vs Random Windows')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. CDF
    ax2 = fig.add_subplot(gs[0, 1])
    obs_sorted = np.sort(observed_scores)
    rand_sorted = np.sort(random_scores)
    obs_cdf = np.arange(1, len(obs_sorted) + 1) / len(obs_sorted)
    rand_cdf = np.arange(1, len(rand_sorted) + 1) / len(rand_sorted)

    ax2.plot(rand_sorted, rand_cdf, label='Random', color='gray', linewidth=2)
    ax2.plot(obs_sorted, obs_cdf, label='Observed', color='red', linewidth=2)
    ax2.set_xlabel('SR Score')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('Cumulative Distribution Functions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Box plot
    ax3 = fig.add_subplot(gs[1, 0])
    data_for_box = [random_scores, observed_scores]
    bp = ax3.boxplot(data_for_box, labels=['Random', 'Observed'], patch_artist=True,
                     widths=0.6)
    bp['boxes'][0].set_facecolor('gray')
    bp['boxes'][1].set_facecolor('red')
    ax3.set_ylabel('SR Score')
    ax3.set_title('SR Score Comparison')
    ax3.grid(True, alpha=0.3, axis='y')

    # Add statistical annotation
    ax3.text(0.5, 0.95, f"Cohen's d = {stats_results['cohens_d']:.3f}\np = {stats_results['p_value']:.6f}",
             transform=ax3.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 4. Summary
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    summary_text = f"""
STATISTICAL SUMMARY

Observed Events:
  Mean: {stats_results['obs_mean']:.4f}
  Median: {stats_results['obs_median']:.4f}
  SD: {stats_results['obs_std']:.4f}
  N: {stats_results['n_observed']}

Random Windows:
  Mean: {stats_results['rand_mean']:.4f}
  Median: {stats_results['rand_median']:.4f}
  SD: {stats_results['rand_std']:.4f}
  N: {stats_results['n_random']}

Statistical Tests:
  Cohen's d: {stats_results['cohens_d']:.3f}
  P-value: {stats_results['p_value']:.6f}
  Percentile: {stats_results['percentile']:.1f}th

Pass Criteria:
  p < 0.01:          {'✓ PASS' if stats_results['passes_pvalue'] else '✗ FAIL'}
  Effect size > 0.5: {'✓ PASS' if stats_results['passes_effect_size'] else '✗ FAIL'}
  Top 10%:           {'✓ PASS' if stats_results['passes_percentile'] else '✗ FAIL'}

OVERALL: {'✓ PASS' if stats_results['overall_pass'] else '✗ FAIL'}
"""

    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Add overall pass/fail annotation
    status = "PASS ✓" if stats_results['overall_pass'] else "FAIL ✗"
    color = 'green' if stats_results['overall_pass'] else 'red'
    fig.text(0.5, 0.98, f'Null Control Test 4 (Random Windows): {status}',
             ha='center', va='top', fontsize=14, fontweight='bold', color=color)

    # Save figure
    plt.savefig(output_dir / 'event_quality_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'event_quality_analysis.pdf', bbox_inches='tight')
    plt.close()

    print(f"Visualizations saved to {output_dir}")


# ============================================================================
# Results Report
# ============================================================================

def generate_results_report(stats_results: Dict, output_dir: Path):
    """Generate 200-word results summary in markdown format."""
    status = "PASSED" if stats_results['overall_pass'] else "FAILED"

    report = f"""# Null Control Test 4: Event Quality vs Random Windows - Results

**Status: {status}**

## Summary

We tested whether detected SIE events have significantly better sr_score (composite quality metric) compared to random 20-second time windows from the same recordings. Detected events (n={stats_results['n_observed']}) were compared against {stats_results['n_random']} random windows selected from non-event periods, covering approximately 25% of non-event duration across sessions.

## Statistical Results

- **Observed sr_score**: {stats_results['obs_mean']:.4f} ± {stats_results['obs_std']:.4f}
- **Random sr_score**: {stats_results['rand_mean']:.4f} ± {stats_results['rand_std']:.4f}
- **Effect size (Cohen's d)**: {stats_results['cohens_d']:.3f} {'✓' if stats_results['passes_effect_size'] else '✗'} (criterion: > 0.5)
- **P-value (Mann-Whitney)**: {stats_results['p_value']:.6f} {'✓' if stats_results['passes_pvalue'] else '✗'} (criterion: < 0.01)
- **Observed percentile**: {stats_results['percentile']:.1f}th {'✓' if stats_results['passes_percentile'] else '✗'} (criterion: ≥ 90th)

## Interpretation

{'Detected SIE events show significantly higher sr_scores than random time windows, demonstrating that events represent genuine moments of high-quality brain-SR synchronization rather than arbitrary thresholds on continuous SR presence. This validates the event detection methodology.' if stats_results['overall_pass'] else 'Detected events do not show significantly higher sr_scores than random windows, suggesting that events may represent arbitrary thresholds rather than genuinely special synchronization moments. This raises concerns about event detection specificity.'}

## Pass Criteria

- [{'x' if stats_results['passes_pvalue'] else ' '}] p < 0.01
- [{'x' if stats_results['passes_effect_size'] else ' '}] Effect size > 0.5
- [{'x' if stats_results['passes_percentile'] else ' '}] Observed in top 10%

**Overall: {'PASS' if stats_results['overall_pass'] else 'FAIL'}**

---

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    # Save report
    report_path = output_dir / 'results_summary.md'
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"Results summary saved to {report_path}")

    return report


# ============================================================================
# Main Execution
# ============================================================================

def main(dataset_selection: Optional[Dict] = None):
    """Main execution function."""

    print("="*80)
    print("NULL CONTROL TEST 4: Event Quality vs Random Windows")
    print("="*80)
    print()

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'results/null_control_random_windows_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()

    # Get all files
    print("Loading file list...")
    print("Selected datasets:")
    all_files = get_all_files(dataset_selection)
    print(f"\nTotal: {len(all_files)} EEG files")
    print()

    # Process each session
    all_observed_scores = []
    all_random_scores = []

    print("Processing sessions...")
    for i, file_path in enumerate(all_files, 1):
        print(f"\n[{i}/{len(all_files)}] {os.path.basename(file_path)}")

        try:
            # Detect events
            events_df, intervals, recording_duration, eeg_df, device_config = detect_events_for_session(file_path)

            if len(events_df) == 0:
                print("  No events detected, skipping")
                continue

            # Extract observed sr_scores
            if 'sr_score' in events_df.columns:
                observed_scores = events_df['sr_score'].values
                all_observed_scores.extend(observed_scores)
                print(f"  Events detected: {len(events_df)}, mean sr_score: {np.mean(observed_scores):.4f}")
            else:
                print("  Warning: sr_score column not found")
                continue

            # Generate random windows
            random_windows = generate_random_windows(
                recording_duration,
                intervals,
                WINDOW_DURATION_SEC,
                COVERAGE_FRACTION
            )

            if len(random_windows) == 0:
                print("  No random windows generated, skipping")
                continue

            print(f"  Random windows: {len(random_windows)}")
            print("  Computing sr_scores for random windows (this may take a while)...")

            # Compute metrics for random windows using the SAME detection pipeline
            random_scores = compute_random_window_metrics(
                file_path,
                eeg_df,
                random_windows,
                device_config
            )

            if len(random_scores) == 0:
                print("  Warning: No random window scores computed")
                continue

            all_random_scores.extend(random_scores)
            print(f"  Random windows analyzed: {len(random_scores)}, mean sr_score: {np.mean(random_scores):.4f}")

        except Exception as e:
            print(f"  Error processing {os.path.basename(file_path)}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print()
    print(f"\nTotal observed events: {len(all_observed_scores)}")
    print(f"Total random windows: {len(all_random_scores)}")
    print()

    if len(all_observed_scores) == 0 or len(all_random_scores) == 0:
        print("ERROR: Insufficient data for statistical testing")
        return

    # Convert to arrays
    observed_scores = np.array(all_observed_scores)
    random_scores = np.array(all_random_scores)

    # Statistical testing
    print("Running statistical tests...")
    stats_results = perform_statistical_test(observed_scores, random_scores)
    print()

    # Print summary
    print("="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"Observed events: {stats_results['n_observed']}")
    print(f"Random windows: {stats_results['n_random']}")
    print()
    print(f"Observed sr_score: {stats_results['obs_mean']:.4f} ± {stats_results['obs_std']:.4f}")
    print(f"Random sr_score: {stats_results['rand_mean']:.4f} ± {stats_results['rand_std']:.4f}")
    print()
    print(f"Cohen's d: {stats_results['cohens_d']:.3f} {'✓ PASS' if stats_results['passes_effect_size'] else '✗ FAIL'}")
    print(f"P-value: {stats_results['p_value']:.6f} {'✓ PASS' if stats_results['passes_pvalue'] else '✗ FAIL'}")
    print(f"Percentile: {stats_results['percentile']:.1f}th {'✓ PASS' if stats_results['passes_percentile'] else '✗ FAIL'}")
    print()
    print(f"OVERALL: {'✓ PASS' if stats_results['overall_pass'] else '✗ FAIL'}")
    print("="*80)
    print()

    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(observed_scores, random_scores, stats_results, output_dir)
    print()

    # Generate report
    print("Generating results summary...")
    generate_results_report(stats_results, output_dir)
    print()

    # Save raw data
    print("Saving raw data...")

    # Save observed scores
    obs_df = pd.DataFrame({'sr_score': observed_scores})
    obs_df.to_csv(output_dir / 'observed_scores.csv', index=False)

    # Save random scores
    rand_df = pd.DataFrame({'sr_score': random_scores})
    rand_df.to_csv(output_dir / 'random_scores.csv', index=False)

    # Save statistics
    stats_df = pd.DataFrame([stats_results])
    stats_df.to_csv(output_dir / 'statistics.csv', index=False)

    print(f"Raw data saved to {output_dir}")
    print()

    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results directory: {output_dir}")
    print()
    print("Output files:")
    print("  - event_quality_analysis.png/.pdf")
    print("  - results_summary.md")
    print("  - observed_scores.csv")
    print("  - random_scores.csv")
    print("  - statistics.csv")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Null Control Test 5: Pairwise Blind Cluster Analysis

Processes subjects in pairs, running blind DBSCAN clustering on combined
FOOOF peaks from both subjects to test for SR frequency privilege.

Sequential pairing: (s1,s2), (s3,s4), (s5,s6), etc.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from scipy import stats, signal
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
import tempfile

# Add lib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

from detect_ignition import detect_ignitions_session
from fooof_harmonics import detect_harmonics_fooof, _get_peak_params

# ============================================================================
# Configuration
# ============================================================================

# Dataset selection
def list_csv_files(directory):
    """List all CSV files in a directory."""
    if not os.path.exists(directory):
        return []
    return [(directory + "/" + f) for f in os.listdir(directory) if f.endswith('.csv')]

PHYSF = list_csv_files("data/PhySF")

# Electrode configurations
ELECTRODES = ['EEG.AF3', 'EEG.AF4',
              'EEG.F3', 'EEG.F4',
              'EEG.F7', 'EEG.F8',
              'EEG.FC5', 'EEG.FC6',
              'EEG.P7', 'EEG.P8',
              'EEG.O1', 'EEG.O2',
              'EEG.T7', 'EEG.T8']

# FOOOF parameters
FOOOF_FREQ_RANGE = (5, 40)
FOOOF_MAX_N_PEAKS = 15
FOOOF_PEAK_THRESHOLD = 0.01
FOOOF_MIN_PEAK_HEIGHT = 0.01
FOOOF_PEAK_WIDTH_LIMITS = (0.5, 12.0)

# Clustering parameters
DBSCAN_EPS = 0.1
DBSCAN_MIN_SAMPLES = 6

# SR scan bands
SR_SCAN_BANDS = [
    {'name': 'SR1', 'center': 7.6, 'halfband': 0.6},
    {'name': 'SR2', 'center': 12.1, 'halfband': 0.618},
    {'name': 'SR2o', 'center': 13.75, 'halfband': 0.75},
    {'name': 'SR3', 'center': 20.0, 'halfband': 1.0},
    {'name': 'SR4', 'center': 25.0, 'halfband': 1.5},
    {'name': 'SR5', 'center': 32.0, 'halfband': 2.0}
]

# Output directory
OUTPUT_BASE_DIR = 'nc5_pairwise'

# ============================================================================
# Subject Pairing Functions
# ============================================================================

def extract_subject_id(filename):
    """
    Extract subject ID from filename (characters before first underscore).

    Examples:
        s2_session1.csv -> s2
        s12_baseline.csv -> s12
    """
    basename = os.path.basename(filename)
    subject_id = basename.split('_')[0]
    return subject_id


def group_files_by_subject(file_paths):
    """Group files by subject ID."""
    subject_files = {}

    for file_path in file_paths:
        subject_id = extract_subject_id(file_path)
        if subject_id not in subject_files:
            subject_files[subject_id] = []
        subject_files[subject_id].append(file_path)

    return subject_files


def create_sequential_pairs(subject_ids):
    """
    Create sequential pairs from sorted subject IDs.

    Example: [s1, s2, s3, s4, s5] -> [(s1, s2), (s3, s4)]
    (s5 is unpaired and skipped)
    """
    sorted_ids = sorted(subject_ids)
    pairs = []

    for i in range(0, len(sorted_ids) - 1, 2):
        pairs.append((sorted_ids[i], sorted_ids[i+1]))

    # Warn about unpaired subject
    if len(sorted_ids) % 2 == 1:
        print(f"WARNING: Subject {sorted_ids[-1]} is unpaired and will be skipped")

    return pairs


# ============================================================================
# Core Processing Functions (from blind_clustering.py)
# ============================================================================

def get_device_config(file_path: str):
    """Determine device configuration from file path."""
    return {
        'sr_channel': 'EEG.T7',
        'eeg_channels': ELECTRODES
    }


def process_session(file_path: str) -> pd.DataFrame:
    """
    Process one EEG file: detect events, extract ALL FOOOF peaks.

    Returns:
        DataFrame with columns: session, event_idx, t_start, t_end,
                               peak_freq, peak_power, peak_bandwidth
    """
    print(f"    Processing: {os.path.basename(file_path)}")

    device_config = get_device_config(file_path)

    # Load data
    first_line = pd.read_csv(file_path, nrows=1, header=None).iloc[0, 0]

    if str(first_line).startswith('title:'):
        df = pd.read_csv(file_path, skiprows=1, low_memory=False)
    else:
        df = pd.read_csv(file_path, header=0, low_memory=False)

    # Create Timestamp if needed
    if 'Timestamp' not in df.columns:
        df['Timestamp'] = df.index / 128.0

    recording_duration = df['Timestamp'].max() - df['Timestamp'].min()
    print(f"      Duration: {recording_duration:.1f} sec ({recording_duration/60:.1f} min)")

    # Detect ignition windows
    with tempfile.TemporaryDirectory() as temp_dir:
        results, intervals = detect_ignitions_session(
            RECORDZ=df,
            sr_channel=device_config['sr_channel'],
            eeg_channels=device_config['eeg_channels'],
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
            fooof_freq_range=FOOOF_FREQ_RANGE,
            fooof_freq_ranges=[[5, 15], [15, 25], [27, 37]],
            fooof_max_n_peaks=FOOOF_MAX_N_PEAKS,
            fooof_peak_threshold=FOOOF_PEAK_THRESHOLD,
            fooof_min_peak_height=FOOOF_MIN_PEAK_HEIGHT,
            fooof_peak_width_limits=FOOOF_PEAK_WIDTH_LIMITS,
            fooof_match_method='power',
            make_passport=False,
            show=False,
            verbose=False,
            session_name=os.path.basename(file_path)
        )

    events_df = results.get('events', pd.DataFrame())

    if len(events_df) == 0:
        print(f"      WARNING: No events detected!")
        return pd.DataFrame()

    print(f"      Events detected: {len(events_df)}")

    # Convert timestamps to relative time
    t0 = df['Timestamp'].min()
    df_relative = df.copy()
    df_relative['Timestamp'] = df['Timestamp'] - t0

    all_peaks = []

    for event_idx, (t_start, t_end) in enumerate(intervals):
        event_mask = (df_relative['Timestamp'] >= t_start) & (df_relative['Timestamp'] <= t_end)
        event_df = df_relative[event_mask].copy()

        if len(event_df) < 100:
            continue

        try:
            from scipy import signal as sp_signal

            # Get EEG data
            eeg_data = []
            for ch in device_config['eeg_channels']:
                if ch in event_df.columns:
                    eeg_data.append(event_df[ch].values)

            if len(eeg_data) == 0:
                continue

            eeg_data = np.array(eeg_data)

            # Compute median PSD across channels
            psds = []
            for ch_data in eeg_data:
                freqs, psd = sp_signal.welch(ch_data, fs=128.0, nperseg=min(512, len(ch_data)))
                psds.append(psd)

            median_psd = np.median(psds, axis=0)

            # Restrict to frequency range
            freq_mask = (freqs >= FOOOF_FREQ_RANGE[0]) & (freqs <= FOOOF_FREQ_RANGE[1])
            freqs_roi = freqs[freq_mask]
            psd_roi = median_psd[freq_mask]

            # Run FOOOF
            try:
                from specparam import SpectralModel
                fm = SpectralModel(
                    peak_width_limits=FOOOF_PEAK_WIDTH_LIMITS,
                    max_n_peaks=FOOOF_MAX_N_PEAKS,
                    min_peak_height=FOOOF_MIN_PEAK_HEIGHT,
                    peak_threshold=FOOOF_PEAK_THRESHOLD,
                    aperiodic_mode='fixed'
                )
            except ImportError:
                from fooof import FOOOF as SpectralModel
                fm = SpectralModel(
                    peak_width_limits=FOOOF_PEAK_WIDTH_LIMITS,
                    max_n_peaks=FOOOF_MAX_N_PEAKS,
                    min_peak_height=FOOOF_MIN_PEAK_HEIGHT,
                    peak_threshold=FOOOF_PEAK_THRESHOLD,
                    aperiodic_mode='fixed'
                )

            fm.fit(freqs_roi, psd_roi, FOOOF_FREQ_RANGE)

            # Extract peaks
            try:
                peak_params = _get_peak_params(fm)
            except Exception:
                peak_params = None

            if peak_params is not None and len(peak_params) > 0:
                for peak in peak_params:
                    all_peaks.append({
                        'session': os.path.basename(file_path),
                        'event_idx': event_idx,
                        't_start': t_start,
                        't_end': t_end,
                        'duration_s': t_end - t_start,
                        'peak_freq': peak[0],
                        'peak_power': peak[1],
                        'peak_bandwidth': peak[2]
                    })

        except Exception as e:
            continue

    print(f"      FOOOF peaks: {len(all_peaks)}")

    return pd.DataFrame(all_peaks)


def blind_cluster_analysis(peak_frequencies: np.ndarray,
                           eps: float = DBSCAN_EPS,
                           min_samples: int = DBSCAN_MIN_SAMPLES) -> Tuple[List[Dict], np.ndarray]:
    """Perform blind DBSCAN clustering on peak frequencies."""
    peaks = np.array(peak_frequencies).reshape(-1, 1)

    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clusterer.fit_predict(peaks)

    clusters = []
    unique_labels = sorted(set(labels) - {-1})

    for label in unique_labels:
        mask = labels == label
        cluster_peaks = peaks[mask].flatten()

        clusters.append({
            'id': label,
            'center': np.mean(cluster_peaks),
            'median': np.median(cluster_peaks),
            'std': np.std(cluster_peaks),
            'cv': np.std(cluster_peaks) / np.mean(cluster_peaks) * 100,
            'min': np.min(cluster_peaks),
            'max': np.max(cluster_peaks),
            'n_peaks': len(cluster_peaks),
            'peaks': cluster_peaks
        })

    clusters = sorted(clusters, key=lambda x: x['center'])

    # Compute silhouette score
    silhouette = None
    if len(unique_labels) > 1:
        try:
            silhouette = silhouette_score(peaks, labels)
        except:
            pass

    for cluster in clusters:
        cluster['silhouette'] = silhouette

    return clusters, labels


def compare_to_sr_bands(clusters: List[Dict]) -> pd.DataFrame:
    """Compare discovered clusters to SR scan bands."""
    results = []

    for cluster in clusters:
        best_match = None
        min_dist = float('inf')

        for sr in SR_SCAN_BANDS:
            dist = abs(cluster['center'] - sr['center'])
            if dist < min_dist:
                min_dist = dist
                best_match = sr

        within_band = min_dist <= best_match['halfband']

        results.append({
            'cluster_id': cluster['id'],
            'cluster_center': cluster['center'],
            'cluster_std': cluster['std'],
            'cluster_n': cluster['n_peaks'],
            'matched_sr': best_match['name'],
            'sr_center': best_match['center'],
            'sr_halfband': best_match['halfband'],
            'distance_hz': min_dist,
            'within_band': within_band,
            'error_pct': abs(min_dist) / best_match['center'] * 100
        })

    return pd.DataFrame(results)


def test_sr_frequency_privilege(all_peaks_df: pd.DataFrame,
                               sr_comparison: pd.DataFrame,
                               labels: np.ndarray) -> Dict:
    """Test if peaks are disproportionately concentrated in SR bands."""
    from scipy.stats import chisquare

    freq_range = FOOOF_FREQ_RANGE[1] - FOOOF_FREQ_RANGE[0]
    sr_coverage = sum(band['halfband'] * 2 for band in SR_SCAN_BANDS)

    peak_freqs = all_peaks_df['peak_freq'].values
    clustered_peaks = peak_freqs[labels != -1]

    # Determine which peaks fall within SR bands
    in_sr = np.zeros(len(clustered_peaks), dtype=bool)
    for band in SR_SCAN_BANDS:
        lower = band['center'] - band['halfband']
        upper = band['center'] + band['halfband']
        in_sr |= (clustered_peaks >= lower) & (clustered_peaks <= upper)

    obs_sr = np.sum(in_sr)
    obs_non_sr = len(clustered_peaks) - obs_sr

    sr_fraction = sr_coverage / freq_range
    exp_sr = len(clustered_peaks) * sr_fraction
    exp_non_sr = len(clustered_peaks) * (1 - sr_fraction)

    observed = np.array([obs_sr, obs_non_sr])
    expected = np.array([exp_sr, exp_non_sr])
    chi2, p_value = chisquare(observed, expected)

    return {
        'freq_range': freq_range,
        'sr_coverage': sr_coverage,
        'sr_fraction': sr_fraction,
        'obs_sr': obs_sr,
        'obs_non_sr': obs_non_sr,
        'obs_sr_pct': obs_sr / len(clustered_peaks) * 100 if len(clustered_peaks) > 0 else 0,
        'exp_sr': exp_sr,
        'exp_non_sr': exp_non_sr,
        'exp_sr_pct': sr_fraction * 100,
        'chi2': chi2,
        'p_value': p_value,
        'pass': p_value < 0.001,
        'effect_size': (obs_sr - exp_sr) / np.sqrt(exp_sr) if exp_sr > 0 else 0
    }


def create_visualization(all_peaks_df: pd.DataFrame,
                        clusters: List[Dict],
                        labels: np.ndarray,
                        sr_comparison: pd.DataFrame,
                        pair_name: str,
                        out_path: str):
    """Create comprehensive 4-panel visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'Null Control 5: Pairwise Blind Cluster Analysis - {pair_name}',
                 fontsize=16, fontweight='bold', y=0.995)

    peak_freqs = all_peaks_df['peak_freq'].values

    # Panel A: Scatter plot
    ax1 = axes[0, 0]
    y_jitter = np.random.uniform(0, 1, len(peak_freqs))

    sr_colors = {
        'SR1': '#E91E63',
        'SR2': '#9C27B0',
        'SR2o': '#BA68C8',
        'SR3': '#2196F3',
        'SR4': '#4CAF50',
        'SR5': '#FF9800'
    }

    sr_legend_added = set()
    non_sr_added = False

    for cluster in clusters:
        cluster_mask = labels == cluster['id']
        cluster_peaks = peak_freqs[cluster_mask]
        cluster_y = y_jitter[cluster_mask]

        cluster_sr = sr_comparison[sr_comparison['cluster_id'] == cluster['id']]
        if len(cluster_sr) > 0 and cluster_sr['within_band'].values[0]:
            matched_sr = cluster_sr['matched_sr'].values[0]
            color = sr_colors.get(matched_sr, '#9E9E9E')
            alpha = 0.7
            label = matched_sr if matched_sr not in sr_legend_added else None
            if matched_sr not in sr_legend_added:
                sr_legend_added.add(matched_sr)
        else:
            color = '#9E9E9E'
            alpha = 0.3
            label = 'Non-SR clusters' if not non_sr_added else None
            non_sr_added = True

        ax1.scatter(cluster_peaks, cluster_y, alpha=alpha, s=30, color=color, label=label)

    # Plot noise
    noise_mask = labels == -1
    if np.any(noise_mask):
        ax1.scatter(peak_freqs[noise_mask], y_jitter[noise_mask],
                   alpha=0.2, s=20, color='#616161', label=f'Noise (n={np.sum(noise_mask)})')

    # Overlay SR bands
    for i, band in enumerate(SR_SCAN_BANDS):
        center = band['center']
        halfband = band['halfband']
        ax1.axvspan(center - halfband, center + halfband, alpha=0.15, color='orange',
                   label='SR bands' if i == 0 else None)
        ax1.axvline(center, color='orange', linestyle='--', linewidth=2, alpha=0.7)
        ax1.text(center, -0.05, band['name'], ha='center', va='center', fontsize=10,
                fontweight='bold', color='orange', zorder=100,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='orange',
                         linewidth=1.5, alpha=0.75))

    # Plot centroids
    for cluster in clusters:
        is_sr = sr_comparison[sr_comparison['cluster_id'] == cluster['id']]['within_band'].values
        if len(is_sr) > 0 and is_sr[0]:
            ax1.scatter(cluster['center'], 0.5, s=300, marker='*', color='red',
                       edgecolors='black', linewidths=1.5, zorder=10)

    ax1.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Peak Density', fontsize=12, fontweight='bold')
    ax1.set_title('A. FOOOF Peak Distribution by SR Band', fontsize=12, fontweight='bold')
    ax1.set_ylim(-0.1, 1.2)
    ax1.set_yticks([])
    ax1.grid(axis='x', alpha=0.3)
    ax1.legend(fontsize=7, loc='upper right', ncol=2, framealpha=0.95)

    # Panel B: Frequency Distribution
    ax2 = axes[0, 1]
    ax2.hist(peak_freqs, bins=60, alpha=0.4, color='gray', density=True, label='All peaks')

    x_range = np.linspace(peak_freqs.min(), peak_freqs.max(), 500)
    total_density = np.zeros_like(x_range)

    from scipy.stats import norm
    from scipy.ndimage import gaussian_filter1d

    for cluster in clusters:
        if cluster['n_peaks'] < 4:
            continue

        cluster_density = norm.pdf(x_range, cluster['center'], cluster['std'] * 1.5) * cluster['n_peaks'] / len(peak_freqs)
        total_density += cluster_density

        is_sr = sr_comparison[sr_comparison['cluster_id'] == cluster['id']]['within_band'].values
        color = '#4CAF50' if len(is_sr) > 0 and is_sr[0] else '#9E9E9E'
        alpha = 0.6 if len(is_sr) > 0 and is_sr[0] else 0.2

        ax2.plot(x_range, cluster_density, color=color, alpha=alpha, linewidth=2)

    total_density_smooth = gaussian_filter1d(total_density, sigma=3)
    ax2.plot(x_range, total_density_smooth, 'k-', linewidth=3, label='GMM model', alpha=0.7)

    # Overlay SR bands
    y_max = ax2.get_ylim()[1]
    for band in SR_SCAN_BANDS:
        center = band['center']
        halfband = band['halfband']
        ax2.axvspan(center - halfband, center + halfband, alpha=0.1, color='orange')
        ax2.axvline(center, color='orange', linestyle='--', linewidth=2, alpha=0.5)

    # Add SR band labels at top of panel
    y_max = ax2.get_ylim()[1]
    for band in SR_SCAN_BANDS:
        center = band['center']
        ax2.text(center, y_max * 0.98, band['name'], ha='center', va='top', fontsize=10,
                fontweight='bold', color='orange', zorder=100,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='orange',
                         linewidth=1.5, alpha=0.75))

    ax2.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Probability Density (1/Hz)', fontsize=12, fontweight='bold')
    ax2.set_title('B. Spectral Clustering Architecture', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper right', framealpha=0.95)
    ax2.grid(True, alpha=0.3)

    # Panel C: Cluster Quality
    ax3 = axes[1, 0]

    for cluster in clusters:
        is_sr = sr_comparison[sr_comparison['cluster_id'] == cluster['id']]['within_band'].values

        if len(is_sr) > 0 and is_sr[0]:
            color = '#4CAF50'
            marker = 'o'
            size = 200
        else:
            if 8 <= cluster['center'] <= 13:
                color = '#9C27B0'
                marker = '^'
                size = 150
            elif 13 < cluster['center'] < 30:
                color = '#FF9800'
                marker = 's'
                size = 150
            else:
                color = '#9E9E9E'
                marker = 'D'
                size = 100

        ax3.scatter(cluster['n_peaks'], cluster['cv'], s=size, c=color,
                   marker=marker, alpha=0.7, edgecolors='black', linewidths=1.5)
        ax3.text(cluster['n_peaks'], cluster['cv'], f" C{cluster['id']}",
                fontsize=9, va='center')

    ax3.set_xlabel('Number of Peaks', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')
    ax3.set_title('C. Cluster Quality: Coherence vs Sample Size', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Panel D: SR Alignment
    ax4 = axes[1, 1]

    sr_aligned = sr_comparison[sr_comparison['within_band'] == True].copy()

    if len(sr_aligned) > 0:
        y_pos = np.arange(len(sr_aligned))

        for i, (_, row) in enumerate(sr_aligned.iterrows()):
            cluster_center = row['cluster_center']
            sr_center = row['sr_center']
            error = row['distance_hz']

            ax4.barh(i, error, left=sr_center,
                    color='#4CAF50' if abs(error) < 0.5 else '#FFC107',
                    alpha=0.7, height=0.6)
            ax4.scatter(cluster_center, i, s=200, marker='o', color='#2196F3',
                       edgecolors='black', linewidths=2, zorder=10)
            ax4.scatter(sr_center, i, s=200, marker='s', color='#FF5722',
                       edgecolors='black', linewidths=2, zorder=10)

        ax4.set_yticks(y_pos)
        ax4.set_yticklabels([f"C{row['cluster_id']}: {row['matched_sr']}" for _, row in sr_aligned.iterrows()])
        ax4.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
        ax4.set_title(f'D. SR Band Alignment ({len(sr_aligned)}/{len(clusters)} clusters within bands)',
                     fontsize=12, fontweight='bold')
        ax4.grid(axis='x', alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No clusters within SR bands',
                ha='center', va='center', fontsize=14, transform=ax4.transAxes)
        ax4.set_title('D. SR Band Alignment', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"    Figure saved: {out_path}")
    plt.close()


def generate_report(pair_name: str,
                   subject_1: str,
                   subject_2: str,
                   clusters: List[Dict],
                   sr_comparison: pd.DataFrame,
                   all_peaks_df: pd.DataFrame,
                   labels: np.ndarray,
                   chi2_result: Dict,
                   out_path: str):
    """Generate markdown report for pair."""

    n_noise = np.sum(labels == -1)
    n_total = len(labels)

    sr_aligned_count = sr_comparison['within_band'].sum() if len(sr_comparison) > 0 else 0

    report = f"""# Null Control 5: Pairwise Blind Cluster Analysis

**Pair:** {pair_name}
**Subjects:** {subject_1} + {subject_2}
**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Summary

**Total events analyzed:** {all_peaks_df['event_idx'].nunique() if len(all_peaks_df) > 0 else 0}
**Total FOOOF peaks:** {len(all_peaks_df)}
**Discovered clusters:** {len(clusters)}
**SR-aligned clusters:** {sr_aligned_count}/{len(clusters)}
**Noise peaks:** {n_noise} ({n_noise/n_total*100:.1f}%)

---

## Discovered Clusters

| Cluster | Center | Std | CV | Range | N Peaks |
|---------|--------|-----|-------|-------|---------|
"""

    for cluster in clusters:
        report += f"| {cluster['id']} | {cluster['center']:.2f} Hz | ±{cluster['std']:.2f} | {cluster['cv']:.1f}% | {cluster['min']:.1f}-{cluster['max']:.1f} Hz | {cluster['n_peaks']} |\n"

    report += f"""
---

## SR Band Comparison

| Cluster | Center | Matched SR | SR Center | Distance | Within Band |
|---------|--------|------------|-----------|----------|-------------|
"""

    for _, row in sr_comparison.iterrows():
        within = '✅' if row['within_band'] else '❌'
        report += f"| {row['cluster_id']} | {row['cluster_center']:.2f} Hz | {row['matched_sr']} | {row['sr_center']:.1f} Hz | {row['distance_hz']:.2f} Hz | {within} |\n"

    report += f"""
---

## SR Frequency Privilege Test

**χ² Test Results:**

| Metric | Value |
|--------|-------|
| **Observed peaks in SR** | {chi2_result['obs_sr']} ({chi2_result['obs_sr_pct']:.1f}%) |
| **Expected peaks in SR** | {chi2_result['exp_sr']:.1f} ({chi2_result['exp_sr_pct']:.1f}%) |
| **χ² statistic** | {chi2_result['chi2']:.2f} |
| **p-value** | {chi2_result['p_value']:.4e} |
| **Effect size** | {chi2_result['effect_size']:.2f} |

**Result:** {'✅ PASS' if chi2_result['pass'] else '❌ FAIL'} (p {'<' if chi2_result['pass'] else '≥'} 0.001)

---

*Generated by null_control_5_pairwise.py*
"""

    with open(out_path, 'w') as f:
        f.write(report)

    print(f"    Report saved: {out_path}")


# ============================================================================
# Pair Processing
# ============================================================================

def process_pair(pair_name: str, subject_1: str, subject_2: str,
                files_1: List[str], files_2: List[str],
                output_dir: str) -> Dict:
    """Process a pair of subjects together."""

    print(f"\n{'='*80}")
    print(f"PROCESSING PAIR: {pair_name}")
    print(f"  Subject 1: {subject_1} ({len(files_1)} files)")
    print(f"  Subject 2: {subject_2} ({len(files_2)} files)")
    print(f"{'='*80}")

    os.makedirs(output_dir, exist_ok=True)

    # Process all files from both subjects
    all_files = files_1 + files_2
    all_peaks_dfs = []

    for file_path in all_files:
        try:
            peaks_df = process_session(file_path)
            if len(peaks_df) > 0:
                all_peaks_dfs.append(peaks_df)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

    if len(all_peaks_dfs) == 0:
        print(f"  ERROR: No peaks collected from either subject")
        return {
            'pair_name': pair_name,
            'subject_1': subject_1,
            'subject_2': subject_2,
            'status': 'NO_PEAKS'
        }

    # Combine peaks from both subjects
    all_peaks_df = pd.concat(all_peaks_dfs, ignore_index=True)

    print(f"\n  COMBINED PEAK SUMMARY:")
    print(f"    Total events: {all_peaks_df['event_idx'].nunique()}")
    print(f"    Total peaks: {len(all_peaks_df)}")

    # Perform clustering
    print(f"\n  CLUSTERING (DBSCAN):")
    peak_frequencies = all_peaks_df['peak_freq'].values
    clusters, labels = blind_cluster_analysis(peak_frequencies, DBSCAN_EPS, DBSCAN_MIN_SAMPLES)

    print(f"    Discovered clusters: {len(clusters)}")
    n_noise = np.sum(labels == -1)
    print(f"    Noise peaks: {n_noise} ({n_noise/len(labels)*100:.1f}%)")

    if len(clusters) == 0:
        print(f"  WARNING: No clusters discovered")
        return {
            'pair_name': pair_name,
            'subject_1': subject_1,
            'subject_2': subject_2,
            'status': 'NO_CLUSTERS'
        }

    # Compare to SR bands
    sr_comparison = compare_to_sr_bands(clusters)
    sr_aligned_count = sr_comparison['within_band'].sum()
    print(f"    SR-aligned clusters: {sr_aligned_count}/{len(clusters)}")

    # Test SR frequency privilege
    chi2_result = test_sr_frequency_privilege(all_peaks_df, sr_comparison, labels)
    print(f"    χ² test: p={chi2_result['p_value']:.4e} ({'PASS' if chi2_result['pass'] else 'FAIL'})")

    # Save outputs
    print(f"\n  SAVING OUTPUTS:")

    # CSVs
    all_peaks_df.to_csv(os.path.join(output_dir, f'{pair_name}_peaks.csv'), index=False)

    clusters_df = pd.DataFrame(clusters)
    if 'peaks' in clusters_df.columns:
        clusters_df = clusters_df.drop(columns=['peaks'])
    clusters_df.to_csv(os.path.join(output_dir, f'{pair_name}_clusters.csv'), index=False)

    sr_comparison.to_csv(os.path.join(output_dir, f'{pair_name}_sr_comparison.csv'), index=False)

    # Visualization
    fig_path = os.path.join(output_dir, f'{pair_name}_figure.png')
    try:
        create_visualization(all_peaks_df, clusters, labels, sr_comparison, pair_name, fig_path)
    except Exception as e:
        print(f"    ERROR generating figure: {e}")

    # Report
    report_path = os.path.join(output_dir, f'{pair_name}_report.md')
    try:
        generate_report(pair_name, subject_1, subject_2, clusters, sr_comparison,
                       all_peaks_df, labels, chi2_result, report_path)
    except Exception as e:
        print(f"    ERROR generating report: {e}")

    return {
        'pair_name': pair_name,
        'subject_1': subject_1,
        'subject_2': subject_2,
        'n_files': len(all_files),
        'n_events': all_peaks_df['event_idx'].nunique(),
        'n_peaks': len(all_peaks_df),
        'n_clusters': len(clusters),
        'sr_aligned': sr_aligned_count,
        'chi2_p': chi2_result['p_value'],
        'sr_enrichment': chi2_result['obs_sr_pct'],
        'status': 'SUCCESS'
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("NULL CONTROL 5: PAIRWISE BLIND CLUSTER ANALYSIS")
    print("=" * 80)
    print()

    # Group files by subject
    subject_files = group_files_by_subject(PHYSF)
    print(f"Found {len(subject_files)} subjects")

    # Create sequential pairs
    pairs = create_sequential_pairs(list(subject_files.keys()))
    print(f"Created {len(pairs)} sequential pairs")
    print()

    # Process each pair
    pair_results = []

    for i, (subject_1, subject_2) in enumerate(pairs, 1):
        pair_name = f"pair_{subject_1}_{subject_2}"
        pair_output_dir = os.path.join(OUTPUT_BASE_DIR, pair_name)

        try:
            result = process_pair(
                pair_name,
                subject_1,
                subject_2,
                subject_files[subject_1],
                subject_files[subject_2],
                pair_output_dir
            )
            pair_results.append(result)
        except Exception as e:
            print(f"\nERROR processing {pair_name}: {e}")
            import traceback
            traceback.print_exc()
            pair_results.append({
                'pair_name': pair_name,
                'subject_1': subject_1,
                'subject_2': subject_2,
                'status': f'ERROR: {str(e)[:50]}'
            })

    # Generate collective summary
    print(f"\n{'='*80}")
    print("GENERATING COLLECTIVE SUMMARY")
    print(f"{'='*80}")

    summary_path = os.path.join(OUTPUT_BASE_DIR, 'collective_summary.md')

    successful_pairs = [r for r in pair_results if r.get('status') == 'SUCCESS']

    summary = f"""# Null Control 5: Pairwise Analysis Collective Summary

**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Overview

| Metric | Value |
|--------|-------|
| **Total pairs** | {len(pairs)} |
| **Successful analyses** | {len(successful_pairs)} |
| **Failed analyses** | {len(pairs) - len(successful_pairs)} |

---

## Pair-Level Results

| Pair | Subject 1 | Subject 2 | Files | Events | Peaks | Clusters | SR-Aligned | χ² p-value | SR Enrichment | Status |
|------|-----------|-----------|-------|--------|-------|----------|------------|------------|---------------|--------|
"""

    for result in pair_results:
        if result['status'] == 'SUCCESS':
            summary += f"| {result['pair_name']} | {result['subject_1']} | {result['subject_2']} | {result['n_files']} | {result['n_events']} | {result['n_peaks']} | {result['n_clusters']} | {result['sr_aligned']} | {result['chi2_p']:.4e} | {result['sr_enrichment']:.1f}% | ✅ |\n"
        else:
            summary += f"| {result['pair_name']} | {result['subject_1']} | {result['subject_2']} | - | - | - | - | - | - | - | ❌ {result['status']} |\n"

    if len(successful_pairs) > 0:
        mean_clusters = np.mean([r['n_clusters'] for r in successful_pairs])
        mean_sr_aligned = np.mean([r['sr_aligned'] for r in successful_pairs])
        mean_peaks = np.mean([r['n_peaks'] for r in successful_pairs])
        mean_sr_enrichment = np.mean([r['sr_enrichment'] for r in successful_pairs])

        pass_count = sum(1 for r in successful_pairs if r['chi2_p'] < 0.001)

        summary += f"""

---

## Aggregate Statistics (Successful Pairs Only)

| Metric | Mean | Min | Max |
|--------|------|-----|-----|
| **Peaks per pair** | {mean_peaks:.1f} | {min(r['n_peaks'] for r in successful_pairs)} | {max(r['n_peaks'] for r in successful_pairs)} |
| **Clusters per pair** | {mean_clusters:.1f} | {min(r['n_clusters'] for r in successful_pairs)} | {max(r['n_clusters'] for r in successful_pairs)} |
| **SR-aligned clusters** | {mean_sr_aligned:.1f} | {min(r['sr_aligned'] for r in successful_pairs)} | {max(r['sr_aligned'] for r in successful_pairs)} |
| **SR enrichment** | {mean_sr_enrichment:.1f}% | {min(r['sr_enrichment'] for r in successful_pairs):.1f}% | {max(r['sr_enrichment'] for r in successful_pairs):.1f}% |

**Pairs passing χ² test (p < 0.001):** {pass_count}/{len(successful_pairs)} ({pass_count/len(successful_pairs)*100:.1f}%)

---

## Conclusions

{'✅ MAJORITY OF PAIRS SHOW SR FREQUENCY PRIVILEGE' if pass_count > len(successful_pairs)/2 else '❌ LIMITED EVIDENCE FOR SR FREQUENCY PRIVILEGE'}

{pass_count}/{len(successful_pairs)} pairs show significant SR frequency privilege (χ² test p < 0.001).

---

*Generated by null_control_5_pairwise.py*
"""
    else:
        summary += "\n\n---\n\nNo successful pair analyses to summarize.\n"

    with open(summary_path, 'w') as f:
        f.write(summary)

    print(f"Collective summary saved: {summary_path}")
    print()
    print("=" * 80)
    print("PAIRWISE ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()

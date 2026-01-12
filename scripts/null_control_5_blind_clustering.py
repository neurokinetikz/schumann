#!/usr/bin/env python3
"""
Null Control Test 5: Blind Cluster Analysis of FOOOF Peaks

Tests if FOOOF-detected spectral peaks naturally cluster around Schumann Resonance
frequencies WITHOUT pre-specification of SR harmonics.

Methodology:
1. Run detection pipeline to get ignition windows
2. For each event, extract ALL FOOOF peaks (up to 15)
3. Perform blind DBSCAN clustering on all collected peaks
4. Compare discovered clusters to:
   a) SR scan bands (7.6±0.6, 20±1, 32±2 Hz)
   b) φ-scaled predictions (f₀, f₀×φ, f₀×φ², ...)

Pass Criteria:
- Discovered clusters align with SR scan bands (within halfband windows)
- φ-prediction error < 1% (mean absolute error)
- Strong correlation (r > 0.99) between observed and predicted clusters
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

# Dataset definitions
FILES = [
    'data/test schumann_EPOCX_111270_2023.04.23T14.50.35.05.00.md.pm.bp.csv',
    'data/20201229_29.12.20_11.27.57.md.pm.bp.csv',
    'data/binaural_EPOCX_111270_2021.06.17T10.04.52.04.00.md.bp.csv',
]

def list_csv_files(directory):
    """List all CSV files in a directory."""
    if not os.path.exists(directory):
        return []
    return [(directory + "/" + f) for f in os.listdir(directory) if f.endswith('.csv')]

KAGGLE = list_csv_files("data/mainData")
MPENG = list_csv_files("data/mpeng")
MPENG1 = list_csv_files("data/mpeng1")
MPENG2 = list_csv_files("data/mpeng2")
VEP = list_csv_files("data/vep")
PHYSF = list_csv_files("data/PhySF")
INSIGHT = list_csv_files("data/insight")
MUSE = list_csv_files("data/muse")

# Electrode configurations
ELECTRODES = ['EEG.AF3', 'EEG.AF4',
              'EEG.F3', 'EEG.F4',
              'EEG.F7', 'EEG.F8',
              'EEG.FC5', 'EEG.FC6',
              'EEG.P7', 'EEG.P8',
              'EEG.O1', 'EEG.O2',
              'EEG.T7', 'EEG.T8']

INSIGHT_ELECTRODES = ['EEG.AF3', 'EEG.AF4', 'EEG.T7', 'EEG.T8', 'EEG.Pz']
MUSE_ELECTRODES = ['EEG.AF7', 'EEG.AF8', 'EEG.TP9', 'EEG.TP10']

# Dataset selection
DATASET_SELECTION = {
    'FILES': True,
    'KAGGLE': False,
    'MPENG': False,
    'MPENG1': False,
    'MPENG2': False,
    'VEP': False,
    'PHYSF': False,  # PhySF dataset (disabled - using FILES only)
    'INSIGHT': False,  # Insight headset
    'MUSE': False,  # Muse headset
}

MAX_FILES = 3  # FILES dataset only

# FOOOF parameters for comprehensive peak detection
FOOOF_FREQ_RANGE = (5, 40)
FOOOF_MAX_N_PEAKS = 15  # Capture all significant peaks
FOOOF_PEAK_THRESHOLD = 0.01  # Original permissive threshold
FOOOF_MIN_PEAK_HEIGHT = 0.01  # Original permissive minimum peak height
FOOOF_PEAK_WIDTH_LIMITS = (0.5, 12.0)

# Clustering parameters
DBSCAN_EPS = 0.08  # Hz tolerance for clustering (tight for SR peak resolution)
DBSCAN_MIN_SAMPLES = 4  # Minimum peaks to form cluster (optimized for PHYSF 2-session datasets)

# φ-Predictions
PHI = 1.618033988749895
FUNDAMENTAL_HZ = 7.49  # Refined fundamental (or test with 7.83)

# SR scan bands (from pipeline)
SR_SCAN_BANDS = [
    {'name': 'SR1', 'center': 7.6, 'halfband': 0.6},
    {'name': 'SR2', 'center': 12.1, 'halfband': 0.618},
    {'name': 'SR2o', 'center': 13.75, 'halfband': 0.75},  # SR2 overtone
    {'name': 'SR3', 'center': 20.0, 'halfband': 1.0},
    {'name': 'SR4', 'center': 25.0, 'halfband': 1.5},
    {'name': 'SR5', 'center': 32.0, 'halfband': 2.0}
]

# ============================================================================
# Helper Functions
# ============================================================================

def get_file_list():
    """Get list of EEG files based on dataset selection."""
    all_files = []

    if DATASET_SELECTION.get('FILES', False):
        all_files.extend(FILES)

    if DATASET_SELECTION.get('KAGGLE', False):
        all_files.extend(KAGGLE)

    if DATASET_SELECTION.get('MPENG', False):
        all_files.extend(MPENG)

    if DATASET_SELECTION.get('MPENG1', False):
        all_files.extend(MPENG1)

    if DATASET_SELECTION.get('MPENG2', False):
        all_files.extend(MPENG2)

    if DATASET_SELECTION.get('VEP', False):
        all_files.extend(VEP)

    if DATASET_SELECTION.get('PHYSF', False):
        all_files.extend(PHYSF)

    if DATASET_SELECTION.get('INSIGHT', False):
        all_files.extend(INSIGHT)

    if DATASET_SELECTION.get('MUSE', False):
        all_files.extend(MUSE)

    # Apply MAX_FILES limit
    if MAX_FILES is not None:
        all_files = all_files[:MAX_FILES]

    return all_files


def get_device_config(file_path: str):
    """Determine device configuration from file path."""
    file_lower = file_path.lower()

    if 'insight' in file_lower:
        return {
            'sr_channel': 'EEG.T7',
            'eeg_channels': INSIGHT_ELECTRODES
        }
    elif 'muse' in file_lower:
        return {
            'sr_channel': 'EEG.TP9',
            'eeg_channels': MUSE_ELECTRODES
        }
    else:  # EMOTIV EPOC or other 14-channel devices
        return {
            'sr_channel': 'EEG.T7',
            'eeg_channels': ELECTRODES
        }


def extract_all_fooof_peaks(fm) -> np.ndarray:
    """
    Extract ALL peaks from FOOOF model(s).

    Args:
        fm: FOOOF model (single or list for multi-channel)

    Returns:
        Nx3 array: [frequency, power, bandwidth]
    """
    # Handle multi-channel (list of models)
    if isinstance(fm, list):
        all_peaks = []
        for fm_ch in fm:
            if fm_ch is not None:
                peaks = _get_peak_params(fm_ch)
                if peaks is not None and len(peaks) > 0:
                    all_peaks.append(peaks)

        if len(all_peaks) > 0:
            return np.vstack(all_peaks)
        else:
            return np.array([]).reshape(0, 3)
    else:
        # Single model
        if fm is not None:
            peaks = _get_peak_params(fm)
            if peaks is not None and len(peaks) > 0:
                return peaks

    return np.array([]).reshape(0, 3)


# ============================================================================
# Core Functions
# ============================================================================

def process_session(file_path: str) -> pd.DataFrame:
    """
    Process one EEG file: detect events, extract ALL FOOOF peaks.

    Returns:
        DataFrame with columns: session, event_idx, t_start, t_end,
                               peak_freq, peak_power, peak_bandwidth
    """
    print(f"\n{'='*80}")
    print(f"Processing: {os.path.basename(file_path)}")
    print(f"{'='*80}")

    # Get device config
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
        column_mapping = {
            'timestamps': 'Timestamp',
            'eeg_1': 'EEG.TP9',
            'eeg_2': 'EEG.AF7',
            'eeg_3': 'EEG.AF8',
            'eeg_4': 'EEG.TP10'
        }
        df.rename(columns=column_mapping, inplace=True)
        eeg_cols = ['EEG.TP9', 'EEG.AF7', 'EEG.AF8', 'EEG.TP10']
        df = df.dropna(subset=eeg_cols, how='all')

    # Create Timestamp if needed
    if 'Timestamp' not in df.columns:
        df['Timestamp'] = df.index / 128.0

    recording_duration = df['Timestamp'].max() - df['Timestamp'].min()
    print(f"  Recording duration: {recording_duration:.1f} sec ({recording_duration/60:.1f} min)")

    # Step 1: Detect ignition windows
    print(f"  Step 1: Detecting ignition events...")
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
            verbose=True,
            session_name=os.path.basename(file_path)
        )

    events_df = results.get('events', pd.DataFrame())

    if len(events_df) == 0:
        print(f"  WARNING: No events detected!")
        return pd.DataFrame()

    print(f"  Events detected: {len(events_df)}")

    # Step 2: Extract ALL FOOOF peaks from each event
    print(f"  Step 2: Extracting ALL FOOOF peaks from each event...")

    # CRITICAL FIX: Convert timestamps to relative time (seconds from start)
    # intervals are in relative time (0, 19, 170, etc.), but df['Timestamp'] may be Unix timestamps
    t0 = df['Timestamp'].min()
    df_relative = df.copy()
    df_relative['Timestamp'] = df['Timestamp'] - t0

    all_peaks = []

    for event_idx, (t_start, t_end) in enumerate(intervals):
        # Extract event window (use df_relative with relative timestamps)
        event_mask = (df_relative['Timestamp'] >= t_start) & (df_relative['Timestamp'] <= t_end)
        event_df = df_relative[event_mask].copy()

        if len(event_df) < 100:  # Skip very short windows
            continue

        # Run FOOOF on this specific event window
        try:
            # Compute PSD for this event window
            from scipy import signal as sp_signal

            # Get EEG data for this window
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

            # Run FOOOF on this PSD
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

            # Extract ALL peaks from FOOOF model
            try:
                peak_params = _get_peak_params(fm)
            except Exception as e:
                peak_params = None

            if peak_params is not None and len(peak_params) > 0:
                # Store each peak
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
            print(f"    Warning: Failed to extract peaks from event {event_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"  Total FOOOF peaks extracted: {len(all_peaks)}")

    return pd.DataFrame(all_peaks)


def blind_cluster_analysis(peak_frequencies: np.ndarray,
                           eps: float = DBSCAN_EPS,
                           min_samples: int = DBSCAN_MIN_SAMPLES) -> Tuple[List[Dict], np.ndarray]:
    """
    Perform blind DBSCAN clustering on peak frequencies.

    Args:
        peak_frequencies: 1D array of frequencies
        eps: DBSCAN epsilon (Hz tolerance)
        min_samples: Minimum peaks to form cluster

    Returns:
        clusters: List of cluster dicts
        labels: Cluster labels for each peak (-1 = noise)
    """
    peaks = np.array(peak_frequencies).reshape(-1, 1)

    # Cluster
    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clusterer.fit_predict(peaks)

    # Analyze clusters (exclude noise label -1)
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

    # Sort by frequency
    clusters = sorted(clusters, key=lambda x: x['center'])

    # Compute silhouette score if we have clusters
    silhouette = None
    if len(unique_labels) > 1:
        try:
            silhouette = silhouette_score(peaks, labels)
        except:
            pass

    # Add silhouette to output
    for cluster in clusters:
        cluster['silhouette'] = silhouette

    return clusters, labels


def compare_to_sr_bands(clusters: List[Dict]) -> pd.DataFrame:
    """
    Compare discovered clusters to SR scan bands.

    Returns:
        DataFrame with comparison results
    """
    results = []

    for cluster in clusters:
        # Find closest SR band
        best_match = None
        min_dist = float('inf')

        for sr in SR_SCAN_BANDS:
            dist = abs(cluster['center'] - sr['center'])
            if dist < min_dist:
                min_dist = dist
                best_match = sr

        # Check if within band
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
    """
    Test if peaks are disproportionately concentrated in SR bands.

    H0: SR frequencies are not privileged (uniform distribution)
    H1: SR frequencies contain disproportionate peak density

    Uses chi-squared goodness-of-fit test to compare observed peak distribution
    to expected uniform distribution across frequency range.

    Args:
        all_peaks_df: DataFrame with all FOOOF peaks
        sr_comparison: DataFrame with SR band alignment results
        labels: DBSCAN cluster labels for each peak

    Returns:
        Dict with test statistics and results
    """
    from scipy.stats import chisquare

    # Calculate total frequency range
    freq_range = FOOOF_FREQ_RANGE[1] - FOOOF_FREQ_RANGE[0]  # 40 - 5 = 35 Hz

    # Calculate SR band coverage (total width of all SR bands)
    sr_coverage = sum(band['halfband'] * 2 for band in SR_SCAN_BANDS)

    # Get peak frequencies (exclude noise peaks with label == -1)
    peak_freqs = all_peaks_df['peak_freq'].values
    clustered_peaks = peak_freqs[labels != -1]

    # Determine which peaks fall within SR bands
    in_sr = np.zeros(len(clustered_peaks), dtype=bool)
    for band in SR_SCAN_BANDS:
        lower = band['center'] - band['halfband']
        upper = band['center'] + band['halfband']
        in_sr |= (clustered_peaks >= lower) & (clustered_peaks <= upper)

    # Observed counts
    obs_sr = np.sum(in_sr)
    obs_non_sr = len(clustered_peaks) - obs_sr

    # Expected counts under uniform distribution
    sr_fraction = sr_coverage / freq_range
    exp_sr = len(clustered_peaks) * sr_fraction
    exp_non_sr = len(clustered_peaks) * (1 - sr_fraction)

    # Chi-squared test
    observed = np.array([obs_sr, obs_non_sr])
    expected = np.array([exp_sr, exp_non_sr])
    chi2, p_value = chisquare(observed, expected)

    return {
        'freq_range': freq_range,
        'sr_coverage': sr_coverage,
        'sr_fraction': sr_fraction,
        'obs_sr': obs_sr,
        'obs_non_sr': obs_non_sr,
        'obs_sr_pct': obs_sr / len(clustered_peaks) * 100,
        'exp_sr': exp_sr,
        'exp_non_sr': exp_non_sr,
        'exp_sr_pct': sr_fraction * 100,
        'chi2': chi2,
        'p_value': p_value,
        'pass': p_value < 0.001,
        'effect_size': (obs_sr - exp_sr) / np.sqrt(exp_sr)  # Standardized residual
    }


def compare_to_phi_predictions(clusters: List[Dict],
                               fundamental: float = FUNDAMENTAL_HZ) -> pd.DataFrame:
    """
    Compare discovered clusters to φ-scaled predictions.

    Args:
        clusters: List of cluster dicts
        fundamental: Fundamental frequency (Hz)

    Returns:
        DataFrame with prediction comparison
    """
    # Generate φ-predictions
    phi = PHI
    predictions = [
        {'harmonic': 'SR1', 'freq': fundamental},
        {'harmonic': 'SR2', 'freq': fundamental * phi},
        {'harmonic': 'SR3', 'freq': fundamental * phi**2},
        {'harmonic': 'SR4', 'freq': fundamental * phi**3},
        {'harmonic': 'SR5', 'freq': fundamental * phi**4},
    ]

    results = []

    # Match clusters to predictions (take first N clusters)
    for i, (cluster, pred) in enumerate(zip(clusters, predictions)):
        obs = cluster['center']
        pred_freq = pred['freq']
        error_hz = obs - pred_freq
        error_pct = abs(error_hz) / pred_freq * 100

        results.append({
            'cluster_id': cluster['id'],
            'harmonic': pred['harmonic'],
            'observed': obs,
            'predicted': pred_freq,
            'error_hz': error_hz,
            'error_pct': error_pct,
            'n_peaks': cluster['n_peaks'],
            'cluster_std': cluster['std']
        })

    return pd.DataFrame(results)


def create_visualization(all_peaks_df: pd.DataFrame,
                        clusters: List[Dict],
                        labels: np.ndarray,
                        sr_comparison: pd.DataFrame,
                        phi_comparison: pd.DataFrame,
                        out_path: str = 'null_control_5_figure.png'):
    """Create comprehensive 4-panel visualization - redesigned for clarity."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Null Control 5: Blind Cluster Analysis of FOOOF Peaks',
                 fontsize=16, fontweight='bold', y=0.995)

    peak_freqs = all_peaks_df['peak_freq'].values

    # Panel A: Scatter plot of all peaks with SR bands and cluster centers
    ax1 = axes[0, 0]

    # Create jittered y-values for better visibility
    y_jitter = np.random.uniform(0, 1, len(peak_freqs))

    # Color palette for each SR band
    sr_colors = {
        'SR1': '#E91E63',  # Pink
        'SR2': '#9C27B0',  # Purple
        'SR2o': '#BA68C8', # Light Purple (SR2 overtone)
        'SR3': '#2196F3',  # Blue
        'SR4': '#4CAF50',  # Green
        'SR5': '#FF9800'   # Orange
    }

    # Track which SR bands we've added to legend
    sr_legend_added = set()
    non_sr_added = False

    # Plot individual peaks by cluster
    for cluster in clusters:
        cluster_mask = labels == cluster['id']
        cluster_peaks = peak_freqs[cluster_mask]
        cluster_y = y_jitter[cluster_mask]

        # Determine if cluster is within SR band and which one
        cluster_sr = sr_comparison[sr_comparison['cluster_id'] == cluster['id']]
        if len(cluster_sr) > 0 and cluster_sr['within_band'].values[0]:
            matched_sr = cluster_sr['matched_sr'].values[0]
            color = sr_colors.get(matched_sr, '#9E9E9E')
            alpha = 0.7
            # Add to legend only once per SR band
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

    # Overlay SR scan bands
    for i, band in enumerate(SR_SCAN_BANDS):
        center = band['center']
        halfband = band['halfband']
        ax1.axvspan(center - halfband, center + halfband, alpha=0.15, color='orange',
                   label='SR bands (scan windows)' if i == 0 else None)
        ax1.axvline(center, color='orange', linestyle='--', linewidth=2, alpha=0.7)
        # Add SR label with white background box and orange border
        ax1.text(center, -0.05, band['name'], ha='center', va='center', fontsize=10,
                fontweight='bold', color='orange', zorder=100,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='orange',
                         linewidth=1.5, alpha=0.75))

    # Plot cluster centroids as large markers
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

    # Create ordered legend manually to ensure SR1, SR2, SR3, SR4, SR5 order
    from matplotlib.lines import Line2D
    ordered_handles = []
    ordered_labels = []

    # Add SR bands in order (SR1 through SR5, including SR2o)
    for sr_name in ['SR1', 'SR2', 'SR2o', 'SR3', 'SR4', 'SR5']:
        if sr_name in sr_legend_added:
            ordered_handles.append(Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor=sr_colors[sr_name],
                                         markersize=8, markeredgecolor='none'))
            ordered_labels.append(sr_name)

    # Add Non-SR clusters if present
    if non_sr_added:
        ordered_handles.append(Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor='#9E9E9E',
                                     markersize=8, markeredgecolor='none', alpha=0.3))
        ordered_labels.append('Non-SR clusters')

    # Add Noise if present
    if np.any(labels == -1):
        ordered_handles.append(Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor='#616161',
                                     markersize=6, markeredgecolor='none', alpha=0.2))
        ordered_labels.append(f'Noise (n={np.sum(labels == -1)})')

    # Add SR scan bands
    ordered_handles.append(plt.Rectangle((0, 0), 1, 1, fc='orange', alpha=0.15))
    ordered_labels.append('SR bands (scan windows)')

    # Add SR-aligned centroids (red stars)
    ordered_handles.append(Line2D([0], [0], marker='*', color='w',
                                  markerfacecolor='red',
                                  markersize=15, markeredgecolor='black', markeredgewidth=1.5))
    ordered_labels.append('SR-aligned centroids')

    # Legend spanning width of chart at top, inside the plot area
    ax1.legend(ordered_handles, ordered_labels, fontsize=7, loc='upper center', ncol=4,
              bbox_to_anchor=(0.5, 0.98), framealpha=0.95, edgecolor='gray')

    # Panel B: Frequency Distribution Density (moved from Panel D)
    ax2 = axes[0, 1]

    # Create density histogram
    ax2.hist(peak_freqs, bins=60, alpha=0.4, color='gray', density=True, label='All peaks')

    # Overlay cluster centers with Gaussians (filter small clusters, increase bandwidth)
    x_range = np.linspace(peak_freqs.min(), peak_freqs.max(), 500)
    total_density = np.zeros_like(x_range)

    from scipy.stats import norm
    from scipy.ndimage import gaussian_filter1d

    for cluster in clusters:
        # FILTER: Only plot clusters with 4+ peaks (removes tiny clusters)
        if cluster['n_peaks'] < 4:
            continue

        # Gaussian for each cluster with 1.5x bandwidth for smoother peaks
        cluster_density = norm.pdf(x_range, cluster['center'], cluster['std'] * 1.5) * cluster['n_peaks'] / len(peak_freqs)
        total_density += cluster_density

        # Color by SR alignment
        is_sr = sr_comparison[sr_comparison['cluster_id'] == cluster['id']]['within_band'].values
        color = '#4CAF50' if len(is_sr) > 0 and is_sr[0] else '#9E9E9E'
        alpha = 0.6 if len(is_sr) > 0 and is_sr[0] else 0.2

        ax2.plot(x_range, cluster_density, color=color, alpha=alpha, linewidth=2)

    # Apply Gaussian smoothing to total density for smoother envelope
    total_density_smooth = gaussian_filter1d(total_density, sigma=3)

    # Plot smoothed total model
    ax2.plot(x_range, total_density_smooth, 'k-', linewidth=3, label='GMM model', alpha=0.7)

    # Overlay SR bands with labels at top
    y_max = ax2.get_ylim()[1]  # Get max y value for positioning
    for i, band in enumerate(SR_SCAN_BANDS):
        center = band['center']
        halfband = band['halfband']
        ax2.axvspan(center - halfband, center + halfband, alpha=0.1, color='orange')
        ax2.axvline(center, color='orange', linestyle='--', linewidth=2, alpha=0.5)
        # Add SR label at top inside chart
        ax2.text(center, y_max * 0.95, band['name'], ha='center', va='top', fontsize=10,
                fontweight='bold', color='orange', zorder=100,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='orange',
                         linewidth=1.5, alpha=0.75))

    ax2.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Probability Density (1/Hz)', fontsize=12, fontweight='bold')
    ax2.set_title('B. Spectral Clustering Architecture', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10, loc='center right', framealpha=0.95, edgecolor='gray')
    ax2.grid(True, alpha=0.3)

    # Panel C: Cluster Quality vs SR Alignment
    ax3 = axes[1, 0]

    # Scatter: CV vs N_peaks, colored by SR alignment
    for cluster in clusters:
        is_sr = sr_comparison[sr_comparison['cluster_id'] == cluster['id']]['within_band'].values

        if len(is_sr) > 0 and is_sr[0]:
            color = '#4CAF50'
            marker = 'o'
            size = 200
            label = 'Within SR band'
            zorder = 10
        else:
            # Check if it's alpha band (8-13 Hz)
            if 8 <= cluster['center'] <= 13:
                color = '#9C27B0'
                marker = '^'
                size = 150
                label = 'Alpha (8-13 Hz)'
                zorder = 5
            # Check if it's beta band (13-30 Hz)
            elif 13 < cluster['center'] < 30:
                color = '#FF9800'
                marker = 's'
                size = 150
                label = 'Beta (13-30 Hz)'
                zorder = 5
            else:
                color = '#9E9E9E'
                marker = 'D'
                size = 100
                label = 'Other'
                zorder = 3

        ax3.scatter(cluster['n_peaks'], cluster['cv'], s=size, c=color,
                   marker=marker, alpha=0.7, edgecolors='black', linewidths=1.5,
                   zorder=zorder)

        # Annotate cluster ID
        ax3.text(cluster['n_peaks'], cluster['cv'], f" C{cluster['id']}",
                fontsize=9, va='center')

    ax3.set_xlabel('Number of Peaks', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')
    ax3.set_title('C. Cluster Quality: Coherence vs Sample Size', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4CAF50',
               markersize=12, label='SR-aligned', markeredgecolor='black'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#9C27B0',
               markersize=12, label='Alpha (8-13 Hz)', markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#FF9800',
               markersize=12, label='Beta (13-30 Hz)', markeredgecolor='black'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#9E9E9E',
               markersize=10, label='Other', markeredgecolor='black')
    ]
    ax3.legend(handles=legend_elements, fontsize=10, loc='upper right')

    # Panel D: SR Band Alignment (moved from Panel B)
    ax4 = axes[1, 1]

    # Highlight SR-aligned clusters
    sr_aligned = sr_comparison[sr_comparison['within_band'] == True].copy()
    sr_near = sr_comparison[sr_comparison['distance_hz'] < 2.0].copy()

    if len(sr_aligned) > 0:
        # Plot alignment
        y_pos = np.arange(len(sr_aligned))

        for i, (_, row) in enumerate(sr_aligned.iterrows()):
            # Bar from SR center to cluster center
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
        ax4.legend([plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3',
                              markersize=10, markeredgecolor='black'),
                   plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#FF5722',
                              markersize=10, markeredgecolor='black')],
                  ['Cluster Center', 'SR Band Center'], fontsize=10)
    else:
        ax4.text(0.5, 0.5, 'No clusters within SR bands',
                ha='center', va='center', fontsize=14, transform=ax4.transAxes)
        ax4.set_title('D. SR Band Alignment', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved: {out_path}")
    plt.close()


def generate_report(clusters: List[Dict],
                   sr_comparison: pd.DataFrame,
                   phi_comparison: pd.DataFrame,
                   all_peaks_df: pd.DataFrame,
                   labels: np.ndarray,
                   test_passes: bool,
                   chi2_result: Dict,
                   out_path: str = 'null_control_5_report.md'):
    """Generate comprehensive markdown report."""

    n_noise = np.sum(labels == -1)
    n_total = len(labels)
    noise_pct = n_noise / n_total * 100 if n_total > 0 else 0

    # Compute overall statistics
    mean_error_pct = phi_comparison['error_pct'].mean() if len(phi_comparison) > 0 else np.nan
    max_error_pct = phi_comparison['error_pct'].max() if len(phi_comparison) > 0 else np.nan

    # Correlation between observed and predicted
    if len(phi_comparison) > 0:
        obs_vals = phi_comparison['observed'].values
        pred_vals = phi_comparison['predicted'].values
        r, p = stats.pearsonr(obs_vals, pred_vals)
    else:
        r, p = np.nan, np.nan

    report = f"""# Null Control Test 5: Blind Cluster Analysis of FOOOF Peaks

**Test Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This test validates whether FOOOF-detected spectral peaks naturally cluster around
Schumann Resonance frequencies WITHOUT pre-specification of SR harmonics.

**Test Result:** {'✅ PASS' if test_passes else '❌ FAIL'}

---

## Methodology

1. **Event Detection**: Run standard SIE detection pipeline (z_thresh=3.0)
2. **Peak Extraction**: Extract ALL FOOOF peaks (up to {FOOOF_MAX_N_PEAKS}) from each event
3. **Blind Clustering**: Apply DBSCAN (eps={DBSCAN_EPS} Hz, min_samples={DBSCAN_MIN_SAMPLES})
4. **Comparison**: Test alignment with SR scan bands and φ-predictions

---

## Results

### 1. Peak Collection

| Metric | Value |
|--------|-------|
| **Total events analyzed** | {all_peaks_df['event_idx'].nunique() if len(all_peaks_df) > 0 else 0} |
| **Total FOOOF peaks extracted** | {len(all_peaks_df)} |
| **Peaks per event (mean)** | {len(all_peaks_df) / all_peaks_df['event_idx'].nunique():.1f} |
| **Frequency range** | {all_peaks_df['peak_freq'].min():.1f} - {all_peaks_df['peak_freq'].max():.1f} Hz |

---

### 2. Discovered Clusters (DBSCAN)

**Clustering Parameters:**
- Epsilon (eps): {DBSCAN_EPS} Hz
- Minimum samples: {DBSCAN_MIN_SAMPLES} peaks
- Algorithm: DBSCAN (density-based spatial clustering)

**Discovered Clusters:**

| Cluster | Center | Median | Std | CV | Range | N Peaks |
|---------|--------|--------|-----|-------|-------|---------|
"""

    for cluster in clusters:
        report += f"| {cluster['id']} | {cluster['center']:.2f} Hz | {cluster['median']:.2f} Hz | ±{cluster['std']:.2f} | {cluster['cv']:.1f}% | {cluster['min']:.1f}-{cluster['max']:.1f} Hz | {cluster['n_peaks']} |\n"

    report += f"""
**Noise peaks:** {n_noise} ({noise_pct:.1f}% of total)

"""

    if clusters and clusters[0].get('silhouette') is not None:
        report += f"**Silhouette score:** {clusters[0]['silhouette']:.3f} (cluster quality metric)\n\n"

    report += """---

### 3. Comparison to SR Scan Bands

SR scan bands are the predefined search windows used in the detection pipeline:
- SR1: 7.6 ± 0.6 Hz
- SR2: 12.1 ± 0.618 Hz
- SR2o: 13.75 ± 0.75 Hz (overtone)
- SR3: 20.0 ± 1.0 Hz
- SR4: 25.0 ± 1.5 Hz
- SR5: 32.0 ± 2.0 Hz

| Cluster | Center | Matched SR | SR Center | Distance | Within Band | Error |
|---------|--------|------------|-----------|----------|-------------|-------|
"""

    for _, row in sr_comparison.iterrows():
        within = '✅' if row['within_band'] else '❌'
        report += f"| {row['cluster_id']} | {row['cluster_center']:.2f} Hz | {row['matched_sr']} | {row['sr_center']:.1f} Hz | {row['distance_hz']:.2f} Hz | {within} | {row['error_pct']:.2f}% |\n"

    all_within = sr_comparison['within_band'].all()
    report += f"""
**Summary:** {'✅ All clusters within SR scan bands' if all_within else '❌ Some clusters outside SR scan bands'}

---

### 4. Comparison to φ-Predictions

φ-scaled predictions using fundamental f₀ = {FUNDAMENTAL_HZ} Hz:

| Harmonic | Observed | Predicted | Error (Hz) | Error (%) | N Peaks |
|----------|----------|-----------|------------|-----------|---------|
"""

    for _, row in phi_comparison.iterrows():
        report += f"| {row['harmonic']} | {row['observed']:.2f} Hz | {row['predicted']:.2f} Hz | {row['error_hz']:+.2f} Hz | {row['error_pct']:.2f}% | {row['n_peaks']} |\n"

    report += f"""
**Statistical Analysis:**
- Mean absolute error: {mean_error_pct:.2f}%
- Max error: {max_error_pct:.2f}%
- Pearson correlation (obs vs pred): r = {r:.4f}, p = {p:.2e}

---

### 5. SR Frequency Privilege Test (χ² Goodness-of-Fit)

**Hypothesis:**
- H₀: SR frequencies are not privileged (uniform distribution across 5-40 Hz)
- H₁: SR frequencies contain disproportionate peak density

**Test Design:**
This chi-squared goodness-of-fit test determines whether FOOOF peaks are disproportionately concentrated in SR frequency bands compared to a uniform distribution.

| Metric | Value |
|--------|-------|
| **Frequency range** | {chi2_result['freq_range']:.1f} Hz (5-40 Hz) |
| **SR band coverage** | {chi2_result['sr_coverage']:.1f} Hz ({chi2_result['sr_fraction']*100:.1f}% of range) |
| **Expected peaks in SR (uniform)** | {chi2_result['exp_sr']:.1f} ({chi2_result['exp_sr_pct']:.1f}%) |
| **Observed peaks in SR** | {chi2_result['obs_sr']} ({chi2_result['obs_sr_pct']:.1f}%) |
| **χ² statistic** | {chi2_result['chi2']:.2f} |
| **p-value** | {chi2_result['p_value']:.4e} |
| **Effect size (std. residual)** | {chi2_result['effect_size']:.2f} |

**Result:** {'✅ PASS' if chi2_result['pass'] else '❌ FAIL'} (p {'< 0.001' if chi2_result['pass'] else '≥ 0.001'})

**Interpretation:**
{'SR frequencies are significantly privileged - FOOOF peaks concentrate in SR bands far beyond chance.' if chi2_result['pass'] else 'No significant privilege detected - peak distribution consistent with uniform distribution.'}

---

## Interpretation

### What These Results Mean

**1. Blind Clustering Discovers Natural Frequency Groups**

The DBSCAN algorithm, without any prior knowledge of Schumann Resonance frequencies,
discovered {len(clusters)} distinct frequency clusters from {len(all_peaks_df)} FOOOF-detected peaks.

**2. Clusters Align with SR Scan Bands**

{'✅ All discovered clusters fall within the predefined SR scan bands used in the detection pipeline.' if all_within else '⚠️ Some clusters fall outside the SR scan bands, suggesting additional spectral structure.'}

**3. φ-Scaling Validation**

Mean prediction error of {mean_error_pct:.2f}% indicates {'strong' if mean_error_pct < 1.0 else 'moderate'} alignment
with φ-scaled harmonic series. Correlation r={r:.4f} shows {'excellent' if r > 0.99 else 'good' if r > 0.95 else 'moderate'}
agreement between observed and predicted frequencies.

---

## Pass/Fail Criteria

| Criterion | Threshold | Actual | Status |
|-----------|-----------|--------|--------|
| **SR band alignment** | All clusters within bands | {sr_comparison['within_band'].sum()}/{len(sr_comparison)} | {'✅ Pass' if all_within else '❌ Fail'} |
| **φ-prediction error** | Mean < 1% | {mean_error_pct:.2f}% | {'✅ Pass' if mean_error_pct < 1.0 else '❌ Fail'} |
| **Correlation** | r > 0.99 | {r:.4f} | {'✅ Pass' if r > 0.99 else '❌ Fail'} |

**Overall:** {'✅ PASS' if test_passes else '❌ FAIL'}

---

## Conclusions

1. {'✅' if len(clusters) == 3 else '⚠️'} Blind clustering discovered {len(clusters)} frequency clusters (expected: 3 for SR1, SR3, SR5)
2. {'✅' if all_within else '❌'} Clusters {'align' if all_within else 'partially align'} with SR scan bands
3. {'✅' if mean_error_pct < 1.0 else '❌'} φ-prediction error: {mean_error_pct:.2f}% ({'within' if mean_error_pct < 1.0 else 'exceeds'} 1% threshold)
4. {'✅' if r > 0.99 else '❌'} Strong correlation with predictions: r={r:.4f}

**Interpretation:** {'This validates that FOOOF-detected peaks naturally cluster around SR frequencies without pre-specification, supporting the physical reality of SR detection.' if test_passes else 'Results suggest some deviation from expected SR frequency structure. Review clustering parameters and data quality.'}

---

*Generated by null_control_5_blind_clustering.py*
"""

    with open(out_path, 'w') as f:
        f.write(report)

    print(f"Report saved: {out_path}")
    return report


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("NULL CONTROL TEST 5: Blind Cluster Analysis of FOOOF Peaks")
    print("=" * 80)
    print()

    all_files = get_file_list()
    print(f"Processing {len(all_files)} files...")
    print()

    all_peaks_dfs = []

    for i, file_path in enumerate(all_files, 1):
        print(f"\n[{i}/{len(all_files)}] {os.path.basename(file_path)}")

        try:
            peaks_df = process_session(file_path)

            if len(peaks_df) > 0:
                all_peaks_dfs.append(peaks_df)
            else:
                print("  Skipped (no peaks)")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(all_peaks_dfs) == 0:
        print("\nERROR: No peaks collected from any file")
        return

    # Concatenate all peaks
    all_peaks_df = pd.concat(all_peaks_dfs, ignore_index=True)

    print()
    print("=" * 80)
    print("PEAK COLLECTION SUMMARY")
    print("=" * 80)
    print(f"Total events analyzed: {all_peaks_df['event_idx'].nunique()}")
    print(f"Total FOOOF peaks collected: {len(all_peaks_df)}")
    print(f"Peaks per event (mean): {len(all_peaks_df) / all_peaks_df['event_idx'].nunique():.1f}")
    print(f"Frequency range: {all_peaks_df['peak_freq'].min():.1f} - {all_peaks_df['peak_freq'].max():.1f} Hz")
    print()

    # Perform blind clustering
    print("=" * 80)
    print("BLIND CLUSTERING (DBSCAN)")
    print("=" * 80)
    print(f"Parameters: eps={DBSCAN_EPS} Hz, min_samples={DBSCAN_MIN_SAMPLES}")
    print()

    peak_frequencies = all_peaks_df['peak_freq'].values
    clusters, labels = blind_cluster_analysis(peak_frequencies, DBSCAN_EPS, DBSCAN_MIN_SAMPLES)

    print(f"Discovered clusters: {len(clusters)}")
    for cluster in clusters:
        print(f"  Cluster {cluster['id']}: {cluster['center']:.2f} ± {cluster['std']:.2f} Hz "
              f"(n={cluster['n_peaks']}, CV={cluster['cv']:.1f}%)")

    n_noise = np.sum(labels == -1)
    print(f"\nNoise peaks: {n_noise} ({n_noise/len(labels)*100:.1f}%)")

    if clusters and clusters[0].get('silhouette') is not None:
        print(f"Silhouette score: {clusters[0]['silhouette']:.3f}")

    # Compare to SR bands
    print()
    print("=" * 80)
    print("COMPARISON TO SR SCAN BANDS")
    print("=" * 80)

    if len(clusters) == 0:
        print("⚠️  NO CLUSTERS FOUND")
        print("\nPossible reasons:")
        print(f"  - min_samples={DBSCAN_MIN_SAMPLES} too high for {len(peak_frequencies)} total peaks")
        print(f"  - eps={DBSCAN_EPS} Hz window too narrow")
        print(f"  - Peaks too dispersed across frequency range")
        print("\n❌ TEST FAIL: No clusters discovered for comparison")

        # Still save outputs
        sr_comparison = pd.DataFrame()
        phi_comparison = pd.DataFrame()
        chi2_result = {
            'freq_range': np.nan, 'sr_coverage': np.nan, 'sr_fraction': np.nan,
            'obs_sr': 0, 'obs_non_sr': 0, 'obs_sr_pct': np.nan,
            'exp_sr': np.nan, 'exp_non_sr': np.nan, 'exp_sr_pct': np.nan,
            'chi2': np.nan, 'p_value': np.nan, 'pass': False, 'effect_size': np.nan
        }
        pass_sr_bands = False
        pass_phi_error = False
        pass_correlation = False
        mean_error = np.nan
        max_error = np.nan
        r = np.nan
        p = np.nan

    else:
        sr_comparison = compare_to_sr_bands(clusters)
        print(sr_comparison.to_string(index=False))

        all_within = sr_comparison['within_band'].all()
        print(f"\nResult: {'✅ All clusters within SR bands' if all_within else '❌ Some clusters outside SR bands'}")

        # Compare to φ-predictions
        print()
        print("=" * 80)
        print(f"COMPARISON TO φ-PREDICTIONS (f₀={FUNDAMENTAL_HZ} Hz)")
        print("=" * 80)

        phi_comparison = compare_to_phi_predictions(clusters, FUNDAMENTAL_HZ)
        print(phi_comparison.to_string(index=False))

        mean_error = phi_comparison['error_pct'].mean()
        max_error = phi_comparison['error_pct'].max()

        obs_vals = phi_comparison['observed'].values
        pred_vals = phi_comparison['predicted'].values

        if len(obs_vals) >= 2:
            r, p = stats.pearsonr(obs_vals, pred_vals)
            print(f"\nMean absolute error: {mean_error:.2f}%")
            print(f"Max error: {max_error:.2f}%")
            print(f"Pearson r: {r:.4f}, p = {p:.2e}")
        else:
            r, p = np.nan, np.nan
            print(f"\nMean absolute error: {mean_error:.2f}%")
            print(f"Max error: {max_error:.2f}%")
            print(f"Pearson r: N/A (insufficient data points: n={len(obs_vals)})")

        # Test SR frequency privilege
        print()
        print("=" * 80)
        print("SR FREQUENCY PRIVILEGE TEST (χ² goodness-of-fit)")
        print("=" * 80)
        print("H0: SR frequencies are not privileged (uniform distribution)")
        print("H1: SR frequencies contain disproportionate peak density")
        print()

        chi2_result = test_sr_frequency_privilege(all_peaks_df, sr_comparison, labels)

        print(f"Frequency range: {chi2_result['freq_range']:.1f} Hz (5-40 Hz)")
        print(f"SR band coverage: {chi2_result['sr_coverage']:.1f} Hz ({chi2_result['sr_fraction']*100:.1f}% of range)")
        print()
        print(f"Expected peaks in SR bands (uniform): {chi2_result['exp_sr']:.1f} ({chi2_result['exp_sr_pct']:.1f}%)")
        print(f"Observed peaks in SR bands: {chi2_result['obs_sr']} ({chi2_result['obs_sr_pct']:.1f}%)")
        print()
        print(f"χ² statistic: {chi2_result['chi2']:.2f}")
        print(f"p-value: {chi2_result['p_value']:.4e}")
        print(f"Effect size (standardized residual): {chi2_result['effect_size']:.2f}")
        print()
        print(f"Result: {'✅ PASS' if chi2_result['pass'] else '❌ FAIL'} (p {'<' if chi2_result['pass'] else '>'} 0.001)")

        # Pass/Fail determination
        print()
        print("=" * 80)
        print("PASS/FAIL CRITERIA")
        print("=" * 80)

        pass_sr_bands = all_within
        pass_phi_error = mean_error < 1.0
        pass_correlation = r > 0.99

        print(f"  SR band alignment: {'✅ PASS' if pass_sr_bands else '❌ FAIL'} ({sr_comparison['within_band'].sum()}/{len(sr_comparison)} clusters within bands)")
        print(f"  φ-prediction error: {'✅ PASS' if pass_phi_error else '❌ FAIL'} (mean: {mean_error:.2f}% vs threshold: 1.0%)")
    print(f"  Correlation: {'✅ PASS' if pass_correlation else '❌ FAIL'} (r={r:.4f} vs threshold: 0.99)")

    test_passes = pass_sr_bands and pass_phi_error and pass_correlation
    print(f"\nOverall: {'✅ PASS' if test_passes else '❌ FAIL'}")

    # Generate outputs
    print()
    print("=" * 80)
    print("GENERATING OUTPUTS")
    print("=" * 80)

    # Save CSVs
    all_peaks_df.to_csv('null_control_5_all_peaks.csv', index=False)
    print("Saved: null_control_5_all_peaks.csv")

    clusters_df = pd.DataFrame(clusters)
    if 'peaks' in clusters_df.columns:
        clusters_df = clusters_df.drop(columns=['peaks'])  # Don't save peak arrays
    clusters_df.to_csv('null_control_5_clusters.csv', index=False)
    print("Saved: null_control_5_clusters.csv")

    sr_comparison.to_csv('null_control_5_sr_comparison.csv', index=False)
    print("Saved: null_control_5_sr_comparison.csv")

    phi_comparison.to_csv('null_control_5_phi_comparison.csv', index=False)
    print("Saved: null_control_5_phi_comparison.csv")

    # Generate visualization
    try:
        create_visualization(all_peaks_df, clusters, labels, sr_comparison,
                           phi_comparison, 'null_control_5_figure.png')
    except Exception as e:
        print(f"Warning: Failed to generate figure: {e}")

    # Generate report
    try:
        generate_report(clusters, sr_comparison, phi_comparison, all_peaks_df,
                       labels, test_passes, chi2_result, 'null_control_5_report.md')
    except Exception as e:
        print(f"Warning: Failed to generate report: {e}")

    print("=" * 80)
    print()


if __name__ == '__main__':
    main()

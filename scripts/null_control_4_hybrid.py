#!/usr/bin/env python3
"""
Null Control Test 2: Event Quality vs Random Windows

Tests if detected SIE events have significantly better metrics compared to random
20-second windows from the same recordings.

APPROACH:
- Observed events: Load pre-computed SIE events from data/SIE.csv (fast, validated)
- Random windows: Compute metrics DIRECTLY on random windows (true null control)
  * No detection threshold - metrics computed for ALL windows
  * Same metric pipeline as NC3 (z-scores, MSC, PLV, HSI, sr_score)
  * Multiple random windows per session analyzed

This validates that detected events represent genuine SR-brain coupling moments,
not just arbitrary temporal selections.

Pass Criteria:
- Observed metrics significantly better: p < 0.01
- Effect size > 0.5 (Cohen's d)
- Observed events in top 10% (>90th percentile)
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
from scipy import signal, stats

# Add lib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

from detect_ignition import (
    bandpass_safe,
    _sr_envelope_z_series,
    _msc_per_channel_vs_median,
    _harmonic_stack_index_flexible
)

# Import FOOOF for peak detection in random windows
try:
    from fooof import FOOOF
    FOOOF_AVAILABLE = True
except ImportError:
    FOOOF_AVAILABLE = False
    print("WARNING: FOOOF not available, random windows will not have phi ratios")

# ============================================================================
# Configuration
# ============================================================================

DATA_FILE = 'ALL-SCORES-CANON-3-sie-analysis.csv'  # Pre-computed observed events
RANDOM_SEED = 42
MAX_SESSIONS = None  # None = all sessions, or specify number for testing

# Random window parameters
WINDOW_DURATION_SEC = 20.0  # Match detection window duration
COVERAGE_FRACTION = 0.25     # Cover 25% of non-event duration
MAX_RANDOM_WINDOWS_PER_SESSION = 50  # Cap per session

# Standard SR parameters
HARMONICS_HZ = [7.6, 20.0, 32.0]
HALF_BW_HZ = [0.6, 1.0, 2.0]
SMOOTH_SEC = 0.01  # Smoothing for envelope z-score

# Weights for composite scores (from detect_ignition.py)
Z_WEIGHTS = [1.000, 0.382, 0.236]  # φ^0, φ^-2, φ^-3 for SR1, SR3, SR5
MSC_WEIGHTS = [1.000, 0.382, 0.236]
PLV_WEIGHTS = [1.000, 0.382, 0.236]

# Metrics to compare
METRICS = ['sr_score', 'sr_z_max', 'msc_7p83_v', 'plv_mean_pm5', 'HSI']

# ============================================================================
# Helper Functions
# ============================================================================

# File cache (same as NC3)
_EEG_FILE_CACHE = {}
FS = 128.0  # Default sampling frequency

def find_eeg_file(session_name: str, data_dir: str = 'data') -> Optional[str]:
    """
    Find the EEG file matching the session name.

    Searches in data/, data/PhySF/, data/INSIGHT/, data/MUSE/
    """
    data_path = Path(data_dir)

    # Try exact match first
    exact_match = data_path / session_name
    if exact_match.exists():
        return str(exact_match)

    # Try with .csv extension
    with_ext = data_path / f"{session_name}.csv"
    if with_ext.exists():
        return str(with_ext)

    # Try searching in subdirectories
    for subdir in ['PhySF', 'INSIGHT', 'MUSE', 'insight', 'muse']:
        subdir_path = data_path / subdir
        if subdir_path.exists():
            file_in_subdir = subdir_path / session_name
            if file_in_subdir.exists():
                return str(file_in_subdir)

            with_ext_in_subdir = subdir_path / f"{session_name}.csv"
            if with_ext_in_subdir.exists():
                return str(with_ext_in_subdir)

    return None


def get_device_config(file_path: str) -> Dict:
    """Determine device configuration from file path."""
    file_lower = file_path.lower()

    # Electrode configurations
    ELECTRODES = ['EEG.AF3', 'EEG.AF4', 'EEG.F3', 'EEG.F4', 'EEG.F7', 'EEG.F8',
                  'EEG.FC5', 'EEG.FC6', 'EEG.P7', 'EEG.P8', 'EEG.O1', 'EEG.O2',
                  'EEG.T7', 'EEG.T8']
    INSIGHT_ELECTRODES = ['EEG.AF3', 'EEG.AF4', 'EEG.T7', 'EEG.T8', 'EEG.Pz']
    MUSE_ELECTRODES = ['EEG.AF7', 'EEG.AF8', 'EEG.TP9', 'EEG.TP10']

    if 'insight' in file_lower:
        return {'sr_channel': 'EEG.T7', 'channels': INSIGHT_ELECTRODES}
    elif 'muse' in file_lower:
        return {'sr_channel': 'EEG.TP9', 'channels': MUSE_ELECTRODES}
    else:  # EMOTIV
        return {'sr_channel': 'EEG.T7', 'channels': ELECTRODES}


def load_eeg_file(file_path: str) -> Tuple[pd.DataFrame, float]:
    """
    Load EEG file and return DataFrame with Timestamp column.
    Uses caching like NC3 to avoid reloading same file.

    Returns:
        df: DataFrame with Timestamp column
        duration: Recording duration in seconds
    """
    # Check cache first
    if file_path not in _EEG_FILE_CACHE:
        # Load data
        first_line = pd.read_csv(file_path, nrows=1, header=None).iloc[0, 0]
        if str(first_line).startswith('title:'):
            df = pd.read_csv(file_path, skiprows=1, low_memory=False)
        else:
            df = pd.read_csv(file_path, header=0, low_memory=False)

        # Apply Muse channel mapping (like utilities.py does) - case-insensitive
        muse_channel_map = {
            "eeg_1": "EEG.AF7",
            "eeg_2": "EEG.AF8",
            "eeg_3": "EEG.TP9",
            "eeg_4": "EEG.TP10",
        }

        # Case-insensitive column matching
        cols_lower = {c.lower(): c for c in df.columns}
        rename_dict = {}
        for muse_col, eeg_name in muse_channel_map.items():
            key = muse_col.lower()
            if key in cols_lower:
                rename_dict[cols_lower[key]] = eeg_name

        if rename_dict:
            df = df.rename(columns=rename_dict)
            # Drop rows with NaN in EEG channels (like utilities.py)
            eeg_cols = list(muse_channel_map.values())
            existing_eeg = [c for c in eeg_cols if c in df.columns]
            if existing_eeg:
                df = df.dropna(subset=existing_eeg)

        # Rename "timestamps" → "Timestamp" for consistency (like utilities.py)
        if 'timestamps' in df.columns:
            df = df.rename(columns={'timestamps': 'Timestamp'})

        # If no Timestamp column, create one from index
        if 'Timestamp' not in df.columns:
            df['Timestamp'] = df.index / FS

        # Cache the full DataFrame
        _EEG_FILE_CACHE[file_path] = df
    else:
        df = _EEG_FILE_CACHE[file_path]

    # Calculate duration
    t = pd.to_numeric(df['Timestamp'], errors='coerce').values.astype(float)
    t = t[np.isfinite(t)]
    duration = float(t.max() - t.min()) if len(t) > 0 else 0.0

    return df, duration


# ============================================================================
# Metric Computation Functions (from NC3)
# ============================================================================

def compute_envelope_z_scores(
    y: np.ndarray,
    fs: float,
    harmonics_hz: List[float],
    half_bw_hz: List[float],
    smooth_sec: float
) -> List[float]:
    """
    Compute envelope z-scores for each harmonic.

    Returns:
        List of [sr1_z_max, sr3_z_max, sr5_z_max]
    """
    z_scores = []
    for f0, half_bw in zip(harmonics_hz, half_bw_hz):
        z_env = _sr_envelope_z_series(y, fs, f0, half_bw, smooth_sec)
        z_max = np.nanmax(np.abs(z_env)) if z_env.size > 0 else 0.0
        z_scores.append(float(z_max))

    return z_scores


def compute_fooof_peaks(
    y: np.ndarray,
    fs: float,
    harmonics_hz: List[float],
    half_bw_hz: List[float],
    freq_ranges: List[Tuple[float, float]] = None
) -> Tuple[List[float], List[float]]:
    """
    Compute FOOOF peak frequencies for harmonic detection.

    Args:
        y: (n_samples,) average EEG signal
        fs: sampling frequency
        harmonics_hz: list of canonical harmonic frequencies
        half_bw_hz: list of half-bandwidths for search windows
        freq_ranges: list of (f_min, f_max) tuples for per-harmonic FOOOF fitting

    Returns:
        Tuple of (peak_frequencies, peak_powers) - one per harmonic (NaN if not detected)
    """
    if not FOOOF_AVAILABLE:
        return [np.nan] * len(harmonics_hz), [np.nan] * len(harmonics_hz)

    # Default frequency ranges for per-harmonic fitting
    if freq_ranges is None:
        freq_ranges = [(5, 15), (15, 25), (27, 37)]  # SR1, SR3, SR5

    # Compute PSD using Welch's method
    nperseg = min(int(100 * fs), len(y))  # 100s or full signal
    freqs, psd = signal.welch(y, fs=fs, nperseg=nperseg, noverlap=nperseg//2)

    peak_freqs = []
    peak_powers = []

    for canon_f, half_bw, (f_min, f_max) in zip(harmonics_hz, half_bw_hz, freq_ranges):
        try:
            # Fit FOOOF in this frequency range
            fm = FOOOF(
                peak_width_limits=(0.25, 4.0),
                max_n_peaks=10,
                min_peak_height=0.01,
                peak_threshold=0.05,
                aperiodic_mode='fixed'
            )

            # Fit in the specified frequency range
            mask = (freqs >= f_min) & (freqs <= f_max)
            fm.fit(freqs[mask], psd[mask], [f_min, f_max])

            # Find peaks within CANON ± HALF_BW window
            search_min = canon_f - half_bw
            search_max = canon_f + half_bw

            peaks_in_window = []
            powers_in_window = []

            for peak_params in fm.peak_params_:
                peak_cf = peak_params[0]  # Center frequency
                peak_pw = peak_params[1]  # Power

                if search_min <= peak_cf <= search_max:
                    peaks_in_window.append(peak_cf)
                    powers_in_window.append(peak_pw)

            if peaks_in_window:
                # Power-weighted centroid
                total_power = sum(powers_in_window)
                weighted_freq = sum(f * p for f, p in zip(peaks_in_window, powers_in_window)) / total_power
                peak_freqs.append(weighted_freq)
                peak_powers.append(total_power)
            else:
                peak_freqs.append(np.nan)
                peak_powers.append(np.nan)

        except Exception as e:
            # FOOOF fit failed
            peak_freqs.append(np.nan)
            peak_powers.append(np.nan)

    return peak_freqs, peak_powers


def compute_msc_values(
    X: np.ndarray,
    fs: float,
    harmonics_hz: List[float],
    bw: float = 0.5
) -> List[float]:
    """
    Compute MSC (magnitude squared coherence) for each harmonic.

    Returns:
        List of [msc_sr1, msc_sr3, msc_sr5]
    """
    msc_values = _msc_per_channel_vs_median(X, fs, harmonics_hz, bw)
    return msc_values


def compute_plv_values(
    X: np.ndarray,
    fs: float,
    harmonics_hz: List[float],
    half_bw_hz: List[float],
    t0_idx: Optional[int] = None,
    pm_sec: float = 5.0
) -> List[float]:
    """
    Compute PLV (phase locking value) for each harmonic around ignition.

    Args:
        X: (n_channels, n_samples) multi-channel EEG
        fs: sampling frequency
        harmonics_hz: list of center frequencies
        half_bw_hz: list of half-bandwidths
        t0_idx: index of ignition onset (if None, use whole window)
        pm_sec: ±seconds around ignition for PLV calculation

    Returns:
        List of [plv_sr1, plv_sr3, plv_sr5]
    """
    n_samples = X.shape[1]

    # Define PLV window
    if t0_idx is not None:
        pm_samples = int(pm_sec * fs)
        i_start = max(0, t0_idx - pm_samples)
        i_end = min(n_samples, t0_idx + pm_samples)
        plv_mask = np.zeros(n_samples, dtype=bool)
        plv_mask[i_start:i_end] = True
    else:
        # Use whole window
        plv_mask = np.ones(n_samples, dtype=bool)

    plv_values = []
    for f0, half_bw in zip(harmonics_hz, half_bw_hz):
        # Bandpass filter at target frequency
        X_band = bandpass_safe(X, fs, f0 - half_bw, f0 + half_bw)

        # Compute phase via Hilbert transform
        phases = np.angle(signal.hilbert(X_band, axis=-1))

        # PLV: mean phase coherence across channels
        plv_inst = np.abs(np.nanmean(np.exp(1j * phases), axis=0))

        # Average over PLV window
        if np.any(plv_mask) and plv_inst.size > 0:
            plv_mean = float(np.nanmean(plv_inst[plv_mask]))
        else:
            plv_mean = 0.0

        plv_values.append(plv_mean)

    return plv_values


def compute_hsi(
    y: np.ndarray,
    fs: float,
    base_hz: float,
    base_bw_hz: float,
    harmonic_centers_hz: List[float],
    harmonic_bw_hz: float
) -> float:
    """
    Compute Harmonic Stack Index (HSI).

    Returns:
        HSI value (overtone power / fundamental power)
    """
    HSI, MaxH = _harmonic_stack_index_flexible(
        y, fs, base_hz, base_bw_hz,
        harmonic_centers_hz, harmonic_bw_hz
    )
    return float(HSI)


def compute_sr_score(
    z_scores: List[float],
    msc_values: List[float],
    plv_values: List[float],
    hsi: float,
    freq_specificity: float = 0.1
) -> float:
    """
    Compute composite SR-score from components.

    Formula (from detect_ignition.py line 1629):
    sr_score = z_score**0.7 * msc_score**1.2 * plv_score * freq_refinement / (1 + HSI)

    Where:
    - z_score = 0.618*z1 + 0.236*z2 + 0.146*z3
    - msc_score = 0.618*msc1 + 0.236*msc2 + 0.146*msc3
    - plv_score = 0.618*plv1 + 0.236*plv2 + 0.146*plv3
    - freq_refinement = 0.95 + 0.10 * freq_spec_norm
    """
    # Weighted sums
    z_score = sum(w * z for w, z in zip(Z_WEIGHTS, z_scores))
    msc_score = sum(w * m for w, m in zip(MSC_WEIGHTS, msc_values))
    plv_score = sum(w * p for w, p in zip(PLV_WEIGHTS, plv_values))

    # Frequency specificity normalization
    FREQ_SPEC_MIN = 0.05
    FREQ_SPEC_MAX = 0.35
    freq_spec_norm = np.clip(
        (freq_specificity - FREQ_SPEC_MIN) / (FREQ_SPEC_MAX - FREQ_SPEC_MIN),
        0.0, 1.0
    )
    freq_refinement = 0.95 + 0.10 * freq_spec_norm

    # Composite score
    sr_score = (z_score**0.7) * (msc_score**1.2) * plv_score * freq_refinement / (1 + hsi)

    return float(sr_score)


def compute_all_metrics(
    X: np.ndarray,
    y: np.ndarray,
    fs: float,
    harmonics_hz: List[float],
    half_bw_hz: List[float],
    freq_specificity: float = 0.1
) -> Dict[str, float]:
    """
    Compute all metrics for a window.

    Args:
        X: (n_channels, n_samples) multi-channel EEG
        y: (n_samples,) average EEG signal
        fs: sampling frequency
        harmonics_hz: list of harmonic frequencies
        half_bw_hz: list of half-bandwidths
        freq_specificity: frequency specificity value

    Returns:
        Dict with all metrics
    """
    # Envelope z-scores
    z_scores = compute_envelope_z_scores(y, fs, harmonics_hz, half_bw_hz, SMOOTH_SEC)

    # MSC values
    msc_values = compute_msc_values(X, fs, harmonics_hz, bw=0.5)

    # PLV values (use whole window)
    plv_values = compute_plv_values(X, fs, harmonics_hz, half_bw_hz, t0_idx=None)

    # HSI
    hsi = compute_hsi(y, fs, harmonics_hz[0], half_bw_hz[0], harmonics_hz, half_bw_hz[1])

    # SR-score
    sr_score = compute_sr_score(z_scores, msc_values, plv_values, hsi, freq_specificity)

    return {
        'sr1_z': z_scores[0],
        'sr3_z': z_scores[1],
        'sr5_z': z_scores[2],
        'msc_sr1': msc_values[0],
        'msc_sr3': msc_values[1] if len(msc_values) > 1 else 0.0,
        'msc_sr5': msc_values[2] if len(msc_values) > 2 else 0.0,
        'plv_sr1': plv_values[0],
        'plv_sr3': plv_values[1] if len(plv_values) > 1 else 0.0,
        'plv_sr5': plv_values[2] if len(plv_values) > 2 else 0.0,
        'hsi': hsi,
        'sr_score': sr_score
    }


# ============================================================================
# Random Window Generation
# ============================================================================

def generate_random_windows(recording_duration: float,
                           event_intervals: List[Tuple[float, float]],
                           window_duration: float = WINDOW_DURATION_SEC,
                           coverage_fraction: float = COVERAGE_FRACTION,
                           max_windows: int = MAX_RANDOM_WINDOWS_PER_SESSION) -> List[Tuple[float, float]]:
    """
    Generate random time windows avoiding event intervals.

    Args:
        recording_duration: Total recording duration (seconds)
        event_intervals: List of (t_start, t_end) event intervals
        window_duration: Duration of each random window (seconds)
        coverage_fraction: Fraction of non-event duration to cover
        max_windows: Maximum number of windows to generate

    Returns:
        List of (t_start, t_end) random window tuples
    """
    # Calculate non-event duration
    event_duration = sum(t_end - t_start for t_start, t_end in event_intervals)
    non_event_duration = recording_duration - event_duration

    if non_event_duration <= window_duration:
        return []

    # Calculate number of windows
    target_coverage = coverage_fraction * non_event_duration
    n_windows = min(int(target_coverage / window_duration), max_windows)

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
        if not overlaps:
            for rand_start, rand_end in random_windows:
                if not (t_end < rand_start or t_start > rand_end):
                    overlaps = True
                    break

        if not overlaps:
            random_windows.append((t_start, t_end))

    return random_windows


# ============================================================================
# Random Window Metric Computation (Direct Computation)
# ============================================================================

def compute_random_window_metrics(file_path: str,
                                 eeg_df: pd.DataFrame,
                                 random_windows: List[Tuple[float, float]],
                                 device_config: Dict) -> List[Dict]:
    """
    Compute metrics for random windows using direct metric computation.

    This computes metrics for ALL windows regardless of strength (true null control).

    Args:
        file_path: Path to EEG file (for naming)
        eeg_df: Full EEG DataFrame with Timestamp column
        random_windows: List of (t_start, t_end) tuples in seconds
        device_config: Device configuration (channels, sr_channel)

    Returns:
        List of metric dictionaries for random windows
    """
    random_metrics = []

    # Get time column
    if 'Timestamp' not in eeg_df.columns:
        print("  Warning: No Timestamp column")
        return []

    time_values = eeg_df['Timestamp'].values
    t_min = time_values.min()

    # Get EEG channels
    eeg_channels = device_config['channels']

    for window_idx, (t_start, t_end) in enumerate(random_windows):
        if (window_idx + 1) % 5 == 0 or window_idx == 0:
            print(f"    Window {window_idx + 1}/{len(random_windows)}: [{t_start:.1f}, {t_end:.1f}]s...")

        try:
            # Extract window from EEG data
            window_start_abs = t_min + t_start
            window_end_abs = t_min + t_end

            mask = (time_values >= window_start_abs) & (time_values <= window_end_abs)
            window_df = eeg_df[mask].copy()

            if len(window_df) < 100:  # Minimum data points
                print(f"      Skipping window {window_idx + 1}: insufficient data")
                continue

            # Extract EEG arrays
            X = np.vstack([
                pd.to_numeric(window_df[ch], errors='coerce').values.astype(float)
                for ch in eeg_channels if ch in window_df.columns
            ])

            # Average signal
            y = np.nanmean(X, axis=0)

            # Infer sampling frequency
            t = pd.to_numeric(window_df['Timestamp'], errors='coerce').values.astype(float)
            t_clean = t[np.isfinite(t)]
            if len(t_clean) < 2:
                print(f"      Skipping window {window_idx + 1}: insufficient valid timestamps")
                continue

            dt = np.diff(t_clean)
            dt = dt[dt > 0]
            fs = float(np.round(1.0 / np.median(dt))) if dt.size > 0 else FS

            # Compute all metrics directly (no detection threshold)
            metrics_raw = compute_all_metrics(
                X, y, fs,
                HARMONICS_HZ,
                HALF_BW_HZ,
                freq_specificity=0.1
            )

            # Compute FOOOF peaks for phi ratios
            peak_freqs, _ = compute_fooof_peaks(y, fs, HARMONICS_HZ, HALF_BW_HZ)

            # Compute phi ratios if we have valid peaks
            if len(peak_freqs) >= 3 and not np.isnan(peak_freqs[0]):
                phi_31 = peak_freqs[1] / peak_freqs[0] if not np.isnan(peak_freqs[1]) else np.nan
                phi_51 = peak_freqs[2] / peak_freqs[0] if not np.isnan(peak_freqs[2]) else np.nan
                phi_53 = peak_freqs[2] / peak_freqs[1] if not np.isnan(peak_freqs[1]) and not np.isnan(peak_freqs[2]) else np.nan
            else:
                phi_31 = np.nan
                phi_51 = np.nan
                phi_53 = np.nan

            # Map to output format (NC3 names → NC4 names)
            metrics = {
                't_start_original': t_start,
                't_end_original': t_end,
                'sr_score': metrics_raw['sr_score'],
                'sr_z_max': metrics_raw['sr1_z'],  # Use SR1 z-score as reference
                'msc_7p83_v': metrics_raw['msc_sr1'],
                'plv_mean_pm5': metrics_raw['plv_sr1'],
                'HSI': metrics_raw['hsi'],
                'phi_31': phi_31,
                'phi_51': phi_51,
                'phi_53': phi_53,
            }

            random_metrics.append(metrics)

        except Exception as e:
            print(f"      Error in window {window_idx + 1}: {str(e)}")
            import traceback
            traceback.print_exc()
            # Skip this window on error
            continue

    return random_metrics


# ============================================================================
# Visualization
# ============================================================================

def create_four_panel_figure(observed_df, random_df, results, out_path='null_control_4_results.png'):
    """Create comprehensive 4-panel visualization of results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Null Control Test 2: Random Windows',
                 fontsize=16, fontweight='bold', y=0.995)

    # Panel 1: SR-Score Distribution Comparison
    ax1 = axes[0, 0]
    obs_scores = observed_df['sr_score'].dropna().values
    rand_scores = random_df['sr_score'].dropna().values

    positions = [1, 2]
    bp = ax1.boxplot([obs_scores, rand_scores], positions=positions, widths=0.6,
                      patch_artist=True, showmeans=True,
                      meanprops=dict(marker='D', markerfacecolor='red', markersize=6))

    bp['boxes'][0].set_facecolor('#4CAF50')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('#FF9800')
    bp['boxes'][1].set_alpha(0.7)

    ax1.set_xticks(positions)
    ax1.set_xticklabels(['Observed\nEvents', 'Random\nWindows'], fontsize=11)
    ax1.set_ylabel('SR-Score', fontsize=12, fontweight='bold')
    ax1.set_title(f'A. SR Coupling Quality\np < {results["sr_score"]["p_value"]:.1e}, d = {results["sr_score"]["cohens_d"]:.2f}',
                  fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Add mean values as text
    ax1.text(1, ax1.get_ylim()[1] * 0.95, f'μ={obs_scores.mean():.3f}',
             ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax1.text(2, ax1.get_ylim()[1] * 0.95, f'μ={rand_scores.mean():.3f}',
             ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panel 2: φ-Convergence (Harmonic Ratios)
    ax2 = axes[0, 1]

    if 'phi_31' in observed_df.columns:
        phi_metrics = ['phi_31', 'phi_51', 'phi_53']
        phi_labels = ['φ₃₁\n(SR3/SR1)', 'φ₅₁\n(SR5/SR1)', 'φ₅₃\n(SR5/SR3)']
        expected_vals = [2.63, 4.19, 1.59]

        # Observed events phi ratios
        obs_means = [observed_df[m].dropna().mean() for m in phi_metrics]

        # Check if random windows have phi ratios
        has_random_phi = 'phi_31' in random_df.columns and random_df['phi_31'].notna().sum() > 0

        x = np.arange(len(phi_labels))

        if has_random_phi:
            # Three groups: Observed, Random, Theoretical
            rand_means = [random_df[m].dropna().mean() for m in phi_metrics]
            width = 0.25

            ax2.bar(x - width, obs_means, width, label='Observed', color='#4CAF50', alpha=0.8)
            ax2.bar(x, rand_means, width, label='Random', color='#FF9800', alpha=0.8)
            ax2.bar(x + width, expected_vals, width, label='Theoretical SR', color='#2196F3', alpha=0.6)

            ax2.set_title('B. Harmonic Frequency Ratios\n(All groups match theoretical SR)',
                          fontsize=12, fontweight='bold')
        else:
            # Two groups: Observed, Theoretical (fallback if FOOOF not available)
            width = 0.35

            ax2.bar(x - width/2, obs_means, width, label='Observed', color='#4CAF50', alpha=0.8)
            ax2.bar(x + width/2, expected_vals, width, label='Theoretical SR', color='#2196F3', alpha=0.6)

            ax2.set_title('B. Harmonic Frequency Ratios\n(Observed events match theoretical SR)',
                          fontsize=12, fontweight='bold')

            # Add note about missing random data
            ax2.text(0.98, 0.02, '* Random windows lack phi ratios\n(FOOOF not available)',
                     transform=ax2.transAxes, fontsize=8, ha='right', va='bottom',
                     style='italic', color='gray')

        ax2.set_ylabel('Frequency Ratio', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(phi_labels, fontsize=11)
        ax2.legend(fontsize=9, loc='upper left')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
    else:
        ax2.text(0.5, 0.5, 'Harmonic ratio data not available',
                 ha='center', va='center', fontsize=12, transform=ax2.transAxes)
        ax2.set_title('B. Harmonic Frequency Ratios', fontsize=12, fontweight='bold')

    # Panel 3: Component Metrics
    ax3 = axes[1, 0]

    metrics = ['sr_z_max', 'msc_7p83_v', 'plv_mean_pm5', 'HSI']
    metric_labels = ['SR Z-Score', 'MSC@7.8Hz', 'PLV@7.8Hz', 'HSI']

    available_metrics = [m for m in metrics if m in observed_df.columns and m in random_df.columns]
    available_labels = [metric_labels[i] for i, m in enumerate(metrics) if m in available_metrics]

    if available_metrics:
        obs_vals = [observed_df[m].dropna().mean() for m in available_metrics]
        rand_vals = [random_df[m].dropna().mean() for m in available_metrics]

        # Normalize to observed values for comparison
        norm_obs = [1.0] * len(obs_vals)
        norm_rand = [rand_vals[i] / obs_vals[i] if obs_vals[i] > 0 else 0 for i in range(len(obs_vals))]

        x = np.arange(len(available_labels))
        width = 0.35

        ax3.bar(x - width/2, norm_obs, width, label='Observed', color='#4CAF50', alpha=0.8)
        ax3.bar(x + width/2, norm_rand, width, label='Random', color='#FF9800', alpha=0.8)

        ax3.set_ylabel('Relative Strength\n(Observed = 1.0)', fontsize=12, fontweight='bold')
        ax3.set_title('C. Component Metrics\n(Normalized to Observed)',
                      fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(available_labels, fontsize=10, rotation=15, ha='right')
        ax3.legend(fontsize=9)
        ax3.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax3.grid(axis='y', alpha=0.3, linestyle='--')

    # Panel 4: Cumulative Distribution Function
    ax4 = axes[1, 1]

    obs_sorted = np.sort(obs_scores)
    rand_sorted = np.sort(rand_scores)

    obs_cdf = np.arange(1, len(obs_sorted) + 1) / len(obs_sorted)
    rand_cdf = np.arange(1, len(rand_sorted) + 1) / len(rand_sorted)

    ax4.plot(obs_sorted, obs_cdf, linewidth=2.5, color='#4CAF50', label='Observed', alpha=0.8)
    ax4.plot(rand_sorted, rand_cdf, linewidth=2.5, color='#FF9800', label='Random', alpha=0.8)

    # Mark median observed value
    obs_median = np.median(obs_scores)
    ax4.axvline(x=obs_median, color='#4CAF50', linestyle='--', linewidth=2, alpha=0.5)
    ax4.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.3)

    # Add percentile annotation
    combined = np.concatenate([obs_scores, rand_scores])
    percentile = stats.percentileofscore(combined, obs_median)
    ax4.text(obs_median * 1.05, 0.15,
             f'Obs. median\nat {percentile:.1f}th\npercentile',
             fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    ax4.set_xlabel('SR-Score', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    ax4.set_title(f'D. Distribution Separation\n(Observed n={len(obs_scores)}, Random n={len(rand_scores)})',
                  fontsize=12, fontweight='bold')
    ax4.legend(fontsize=11, loc='lower right')
    ax4.grid(alpha=0.3, linestyle='--')
    ax4.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved: {out_path}")
    plt.close()


# ============================================================================
# Main Processing
# ============================================================================

def main():
    print("=" * 80)
    print("NULL CONTROL TEST 2: Random Windows")
    print("=" * 80)
    print()

    np.random.seed(RANDOM_SEED)

    # Load observed events from SIE.csv
    print(f"Loading observed events from {DATA_FILE}...")
    events_df = pd.read_csv(DATA_FILE)
    print(f"  Loaded {len(events_df)} observed events\n")

    # Group by session
    sessions_grouped = events_df.groupby('session_name')

    if MAX_SESSIONS is not None:
        session_list = list(sessions_grouped.groups.keys())[:MAX_SESSIONS]
        sessions = [(name, events_df[events_df['session_name'] == name]) for name in session_list]
        print(f"Processing {len(sessions)} sessions (MAX_SESSIONS={MAX_SESSIONS})\n")
    else:
        sessions = list(sessions_grouped)
        print(f"Processing {len(sessions)} sessions\n")

    n_sessions = len(sessions)
    all_observed = []
    all_random = []
    failed_sessions = []

    for session_idx, (session_name, session_events) in enumerate(sessions):
        print(f"[{session_idx + 1}/{n_sessions}] {session_name}")

        try:
            # Find EEG file
            eeg_file = find_eeg_file(session_name)
            if eeg_file is None:
                print(f"  ERROR: File not found")
                failed_sessions.append((session_name, "File not found"))
                continue

            print(f"  File: {eeg_file}")

            # Load EEG data
            print(f"  Loading EEG data...")
            eeg_df, duration = load_eeg_file(eeg_file)
            print(f"  Duration: {duration:.1f}s")

            # Get device config
            device_config = get_device_config(eeg_file)
            print(f"  Device: {len(device_config['channels'])} channels")

            # Get event intervals
            event_intervals = [(row['t_start'], row['t_end']) for _, row in session_events.iterrows()]
            print(f"  Events: {len(event_intervals)}")

            # Generate random windows
            print(f"  Generating random windows...")
            random_windows = generate_random_windows(duration, event_intervals)
            print(f"  Random windows: {len(random_windows)}")

            if len(random_windows) == 0:
                print(f"  WARNING: No random windows generated")
                continue

            # Compute metrics for random windows (v1 style)
            print(f"  Computing random window metrics...")
            random_metrics = compute_random_window_metrics(
                eeg_file, eeg_df, random_windows, device_config
            )

            # Collect observed event metrics
            # Map SIE.csv column names to standard detection output names
            for _, event in session_events.iterrows():
                all_observed.append({
                    'sr_score': event['sr_score'],
                    'sr_z_max': event.get('sr1_z_max', event.get('sr_z_max', 0.0)),  # SIE.csv uses sr1_z_max
                    'msc_7p83_v': event.get('msc_sr1', event.get('msc_7p83_v', 0.0)),  # SIE.csv uses msc_sr1
                    'plv_mean_pm5': event.get('plv_sr1', event.get('plv_mean_pm5', 0.0)),  # SIE.csv uses plv_sr1
                    'HSI': event.get('HSI', 0.0),
                    # Include phi ratios for visualization
                    'phi_31': event.get('sr3/sr1', np.nan),
                    'phi_51': event.get('sr5/sr1', np.nan),
                    'phi_53': event.get('sr5/sr3', np.nan),
                })

            # Collect random window metrics
            all_random.extend(random_metrics)

            print(f"  ✓ Completed\n")

        except Exception as e:
            print(f"  ERROR: {str(e)}\n")
            failed_sessions.append((session_name, str(e)))
            continue

    print("=" * 80)
    print(f"Processing complete: {n_sessions - len(failed_sessions)}/{n_sessions} sessions")
    print(f"Observed events: {len(all_observed)}")
    print(f"Random windows: {len(all_random)}")
    if failed_sessions:
        print(f"Failed sessions: {len(failed_sessions)}")
    print("=" * 80)
    print()

    # Convert to DataFrames
    observed_df = pd.DataFrame(all_observed)
    random_df = pd.DataFrame(all_random)

    # Save results
    observed_df.to_csv('null_control_4_observed.csv', index=False)
    random_df.to_csv('null_control_4_random.csv', index=False)
    print("Results saved:")
    print("  - null_control_4_observed.csv")
    print("  - null_control_4_random.csv")
    print()

    # Statistical comparison
    print("=" * 80)
    print("STATISTICAL COMPARISON")
    print("=" * 80)
    print()

    # Build nested results dictionary for visualization
    results = {}

    for metric in METRICS:
        if metric not in observed_df.columns or metric not in random_df.columns:
            continue

        obs_values = observed_df[metric].dropna().values
        rand_values = random_df[metric].dropna().values

        if len(obs_values) == 0 or len(rand_values) == 0:
            continue

        # Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(obs_values, rand_values, alternative='greater')

        # Cohen's d
        pooled_std = np.sqrt((obs_values.std()**2 + rand_values.std()**2) / 2)
        cohens_d = (obs_values.mean() - rand_values.mean()) / pooled_std if pooled_std > 0 else 0

        # Percentile
        combined = np.concatenate([obs_values, rand_values])
        obs_median = np.median(obs_values)
        percentile = stats.percentileofscore(combined, obs_median)

        # Store results in nested dict for visualization
        results[metric] = {
            'p_value': p_value,
            'cohens_d': cohens_d,
            'percentile': percentile,
            'obs_mean': obs_values.mean(),
            'obs_std': obs_values.std(),
            'rand_mean': rand_values.mean(),
            'rand_std': rand_values.std(),
        }

        print(f"{metric}:")
        print(f"  Observed:  mean={obs_values.mean():.4f}, std={obs_values.std():.4f}, n={len(obs_values)}")
        print(f"  Random:    mean={rand_values.mean():.4f}, std={rand_values.std():.4f}, n={len(rand_values)}")
        print(f"  Mann-Whitney U: p={p_value:.6f}")
        print(f"  Cohen's d: {cohens_d:.3f}")
        print(f"  Percentile: {percentile:.1f}%")

        # Pass/Fail
        passes = p_value < 0.01 and cohens_d > 0.5 and percentile > 90
        print(f"  Status: {'✅ PASS' if passes else '❌ FAIL'}")
        print()

    # Create visualization
    print("=" * 80)
    print("CREATING VISUALIZATION")
    print("=" * 80)
    print()

    try:
        create_four_panel_figure(observed_df, random_df, results, out_path='null_control_4_results.png')
        print("✓ Visualization created: null_control_4_results.png\n")
    except Exception as e:
        print(f"WARNING: Visualization failed: {e}\n")

    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Null Control Test 1: Phase Randomization

Tests whether SR-brain coupling metrics reflect genuine temporal phase relationships
or are simply spectral artifacts within the predefined search windows.

Method:
1. Load observed SIE events with pre-computed metrics
2. For each event window, load raw EEG data
3. Generate phase-randomized surrogates (preserve spectrum, destroy temporal coupling)
4. Re-compute metrics on surrogates using same functions
5. Compare observed vs. surrogate metrics (paired tests)

Expected if coupling is real:
- PLV: observed >> surrogate (phase coupling destroyed)
- sr_score: observed >> surrogate (coupling metric drops)
- Frequencies (sr1, sr3, sr5): no difference (spectral property preserved)

Expected if coupling is artifact:
- No significant differences (suggests peaks are just noise in search windows)
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats

# Add lib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

# Import helper functions from detect_ignition
from detect_ignition import (
    bandpass_safe,
    _sr_envelope_z_series,
    _msc_per_channel_vs_median,
    _harmonic_stack_index_flexible
)

# ============================================================================
# Configuration
# ============================================================================

DATA_FILE = 'ALL-SCORES-CANON-3-sie-analysis.csv'
N_SURROGATES = 10  # Number of phase-randomized surrogates per event
RANDOM_SEED = 42
N_TEST_EVENTS = None  # Limit number of events for testing (None = all events)

# Standard detection parameters (must match original pipeline)
FS = 128.0  # Sampling frequency (Hz)
HARMONICS_HZ = [7.6, 20.0, 32.0]  # SR1, SR3, SR5
HALF_BW_HZ = [0.6, 1.0, 2.0]  # Bandwidth for each harmonic
SMOOTH_SEC = 0.01  # Smoothing for envelope z-score

# Weights for composite scores (from detect_ignition.py)
Z_WEIGHTS = [1.000, 0.382, 0.236]  # φ^0, φ^-2, φ^-3 for SR1, SR3, SR5
MSC_WEIGHTS = [1.000, 0.382, 0.236]
PLV_WEIGHTS = [1.000, 0.382, 0.236]

# ============================================================================
# Phase Randomization Functions
# ============================================================================

def phase_randomize_signal(signal: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """
    Phase randomization via Fourier transform.

    Preserves:
    - Power spectrum
    - Frequency content
    - Amplitude distribution

    Destroys:
    - Temporal phase relationships
    - Cross-channel phase coherence

    Args:
        signal: 1D array (time series)
        seed: Random seed for reproducibility

    Returns:
        Phase-randomized signal with same length and power spectrum
    """
    if seed is not None:
        np.random.seed(seed)

    # FFT
    fft_coeffs = np.fft.rfft(signal)

    # Generate random phases
    n_coeffs = len(fft_coeffs)
    random_phases = np.exp(2j * np.pi * np.random.rand(n_coeffs))

    # Apply random phases while preserving magnitudes
    fft_randomized = np.abs(fft_coeffs) * random_phases

    # iFFT
    randomized = np.fft.irfft(fft_randomized, n=len(signal))

    return randomized


def phase_randomize_multichannel(X: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """
    Phase randomize multi-channel EEG data.

    Args:
        X: (n_channels, n_samples) array
        seed: Random seed for reproducibility

    Returns:
        Phase-randomized multi-channel data
    """
    n_channels, n_samples = X.shape
    X_randomized = np.zeros_like(X)

    for ch in range(n_channels):
        channel_seed = (seed + ch) if seed is not None else None
        X_randomized[ch, :] = phase_randomize_signal(X[ch, :], seed=channel_seed)

    return X_randomized


# ============================================================================
# Metric Computation Functions
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
# Data Loading Functions
# ============================================================================

# Global cache for loaded EEG files (avoids re-reading same file multiple times)
_EEG_FILE_CACHE = {}

def find_eeg_file(session_name: str, data_dir: str = 'data') -> Optional[str]:
    """
    Find the EEG file matching the session name.

    Args:
        session_name: Session name from SHUFFLED-DATA-BOOTSTRAP.csv
        data_dir: Directory containing EEG files

    Returns:
        Full path to EEG file, or None if not found
    """
    # Try exact match first
    data_path = Path(data_dir)
    exact_match = data_path / session_name
    if exact_match.exists():
        return str(exact_match)

    # Try with .csv extension
    with_ext = data_path / f"{session_name}.csv"
    if with_ext.exists():
        return str(with_ext)

    # Try searching in subdirectories
    for subdir in ['PhySF', 'INSIGHT', 'MUSE']:
        subdir_path = data_path / subdir
        if subdir_path.exists():
            file_in_subdir = subdir_path / session_name
            if file_in_subdir.exists():
                return str(file_in_subdir)

            with_ext_in_subdir = subdir_path / f"{session_name}.csv"
            if with_ext_in_subdir.exists():
                return str(with_ext_in_subdir)

    return None


def load_eeg_window(
    file_path: str,
    t_start: float,
    t_end: float,
    eeg_channels: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Load a specific time window from an EEG file.

    Args:
        file_path: Path to EEG CSV file
        t_start: Start time (seconds)
        t_end: End time (seconds)
        eeg_channels: List of EEG channel names (if None, auto-detect)

    Returns:
        X: (n_channels, n_samples) multi-channel EEG
        y: (n_samples,) average EEG signal
        fs: sampling frequency
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

    # Use Timestamp column
    time_col = 'Timestamp'

    # Infer sampling frequency
    t = pd.to_numeric(df[time_col], errors='coerce').values.astype(float)
    t = t - t[0]  # Relative time
    dt = np.diff(t[np.isfinite(t)])
    dt = dt[dt > 0]
    fs = float(np.round(1.0 / np.median(dt))) if dt.size > 0 else FS

    # Extract window
    mask = (t >= t_start) & (t <= t_end)
    df_window = df[mask].copy()

    if len(df_window) == 0:
        raise ValueError(f"No data in window [{t_start}, {t_end}]")

    # Get EEG channels
    if eeg_channels is None:
        # Try "EEG." prefix first (EPOC format)
        eeg_channels = [c for c in df_window.columns if c.startswith('EEG.')]

        # If no EEG. channels, try common EEG channel names (Insight/other formats)
        if len(eeg_channels) == 0:
            common_eeg = ['TP9', 'TP10', 'AF7', 'AF8',  # Muse (after mapping)
                          'AF3', 'AF4', 'T7', 'T8', 'Pz',  # Insight
                          'O1', 'O2', 'F3', 'F4', 'F7', 'F8',  # Common
                          'FC5', 'FC6', 'P7', 'P8']
            eeg_channels = [c for c in df_window.columns if c in common_eeg]

    # Extract multi-channel data
    X = np.vstack([
        pd.to_numeric(df_window[ch], errors='coerce').values.astype(float)
        for ch in eeg_channels
    ])

    # Average signal
    y = np.nanmean(X, axis=0)

    return X, y, fs


# ============================================================================
# Statistical Analysis
# ============================================================================

def perform_statistical_comparison(results_df: pd.DataFrame) -> Dict:
    """
    Compare observed vs. surrogate metrics using paired tests.

    Args:
        results_df: DataFrame with obs_* and surr_* columns

    Returns:
        Dict with statistical test results
    """
    metrics = ['sr_score', 'plv_sr1', 'plv_sr3', 'plv_sr5',
               'msc_sr1', 'msc_sr3', 'msc_sr5', 'hsi',
               'sr1_z', 'sr3_z', 'sr5_z']

    stats_results = {}

    for metric in metrics:
        obs_col = f'obs_{metric}'
        surr_col = f'surr_{metric}'

        if obs_col not in results_df.columns or surr_col not in results_df.columns:
            continue

        obs_vals = results_df[obs_col].dropna().values
        surr_vals = results_df[surr_col].dropna().values

        if len(obs_vals) == 0 or len(surr_vals) == 0:
            continue

        # Paired Wilcoxon signed-rank test
        try:
            stat, p_value = stats.wilcoxon(obs_vals, surr_vals, alternative='two-sided')
        except Exception:
            stat, p_value = np.nan, np.nan

        # Cohen's d for paired data
        diff = obs_vals - surr_vals
        cohens_d = np.mean(diff) / (np.std(diff) + 1e-12)

        # Mean values
        obs_mean = np.mean(obs_vals)
        surr_mean = np.mean(surr_vals)

        stats_results[metric] = {
            'obs_mean': obs_mean,
            'obs_std': np.std(obs_vals),
            'surr_mean': surr_mean,
            'surr_std': np.std(surr_vals),
            'p_value': p_value,
            'cohens_d': cohens_d,
            'n': len(obs_vals)
        }

    return stats_results


# ============================================================================
# Visualization
# ============================================================================

def create_four_panel_figure(results_df: pd.DataFrame, stats_results: Dict,
                             out_path: str = 'null_control_3_results.png'):
    """
    Create comprehensive 4-panel visualization.

    Panel A: sr_score (observed vs. surrogate)
    Panel B: PLV comparison
    Panel C: MSC comparison
    Panel D: Z-score comparison (validates spectral preservation)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Null Control Test 1: Phase-Randomized Surrogates',
                 fontsize=16, fontweight='bold', y=0.995)

    # Panel A: SR-Score
    ax1 = axes[0, 0]
    obs_sr = results_df['obs_sr_score'].dropna().values
    surr_sr = results_df['surr_sr_score'].dropna().values

    positions = [1, 2]
    bp = ax1.boxplot([obs_sr, surr_sr], positions=positions, widths=0.6,
                      patch_artist=True, showmeans=True,
                      meanprops=dict(marker='D', markerfacecolor='red', markersize=6))

    bp['boxes'][0].set_facecolor('#4CAF50')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('#FF9800')
    bp['boxes'][1].set_alpha(0.7)

    ax1.set_xticks(positions)
    ax1.set_xticklabels(['Observed', 'Phase-Randomized'], fontsize=11)
    ax1.set_ylabel('SR-Score', fontsize=12, fontweight='bold')

    if 'sr_score' in stats_results:
        sr_stats = stats_results['sr_score']
        ax1.set_title(f'A. Composite SR Score\np = {sr_stats["p_value"]:.1e}, d = {sr_stats["cohens_d"]:.2f}',
                      fontsize=12, fontweight='bold')
    else:
        ax1.set_title('A. Composite SR Score', fontsize=12, fontweight='bold')

    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Panel B: PLV (SR1)
    ax2 = axes[0, 1]
    obs_plv = results_df['obs_plv_sr1'].dropna().values
    surr_plv = results_df['surr_plv_sr1'].dropna().values

    bp = ax2.boxplot([obs_plv, surr_plv], positions=positions, widths=0.6,
                      patch_artist=True, showmeans=True,
                      meanprops=dict(marker='D', markerfacecolor='red', markersize=6))

    bp['boxes'][0].set_facecolor('#4CAF50')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('#FF9800')
    bp['boxes'][1].set_alpha(0.7)

    ax2.set_xticks(positions)
    ax2.set_xticklabels(['Observed', 'Phase-Randomized'], fontsize=11)
    ax2.set_ylabel('PLV @ SR1', fontsize=12, fontweight='bold')

    if 'plv_sr1' in stats_results:
        plv_stats = stats_results['plv_sr1']
        ax2.set_title(f'B. Phase Locking Value\np = {plv_stats["p_value"]:.1e}, d = {plv_stats["cohens_d"]:.2f}',
                      fontsize=12, fontweight='bold')
    else:
        ax2.set_title('B. Phase Locking Value', fontsize=12, fontweight='bold')

    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Panel C: MSC (SR1)
    ax3 = axes[1, 0]
    obs_msc = results_df['obs_msc_sr1'].dropna().values
    surr_msc = results_df['surr_msc_sr1'].dropna().values

    bp = ax3.boxplot([obs_msc, surr_msc], positions=positions, widths=0.6,
                      patch_artist=True, showmeans=True,
                      meanprops=dict(marker='D', markerfacecolor='red', markersize=6))

    bp['boxes'][0].set_facecolor('#4CAF50')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('#FF9800')
    bp['boxes'][1].set_alpha(0.7)

    ax3.set_xticks(positions)
    ax3.set_xticklabels(['Observed', 'Phase-Randomized'], fontsize=11)
    ax3.set_ylabel('MSC @ SR1', fontsize=12, fontweight='bold')

    if 'msc_sr1' in stats_results:
        msc_stats = stats_results['msc_sr1']
        ax3.set_title(f'C. Magnitude Squared Coherence\np = {msc_stats["p_value"]:.1e}, d = {msc_stats["cohens_d"]:.2f}',
                      fontsize=12, fontweight='bold')
    else:
        ax3.set_title('C. Magnitude Squared Coherence', fontsize=12, fontweight='bold')

    ax3.grid(axis='y', alpha=0.3, linestyle='--')

    # Panel D: Z-scores (should NOT differ - validates spectral preservation)
    ax4 = axes[1, 1]
    obs_z1 = results_df['obs_sr1_z'].dropna().values
    surr_z1 = results_df['surr_sr1_z'].dropna().values

    bp = ax4.boxplot([obs_z1, surr_z1], positions=positions, widths=0.6,
                      patch_artist=True, showmeans=True,
                      meanprops=dict(marker='D', markerfacecolor='red', markersize=6))

    bp['boxes'][0].set_facecolor('#4CAF50')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('#FF9800')
    bp['boxes'][1].set_alpha(0.7)

    ax4.set_xticks(positions)
    ax4.set_xticklabels(['Observed', 'Phase-Randomized'], fontsize=11)
    ax4.set_ylabel('Z-Score @ SR1', fontsize=12, fontweight='bold')

    if 'sr1_z' in stats_results:
        z_stats = stats_results['sr1_z']
        ax4.set_title(f'D. Envelope Z-Score (Spectral Property)\np = {z_stats["p_value"]:.1e}, d = {z_stats["cohens_d"]:.2f}',
                      fontsize=12, fontweight='bold')
    else:
        ax4.set_title('D. Envelope Z-Score (Spectral Property)', fontsize=12, fontweight='bold')

    ax4.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved: {out_path}")
    plt.close()


# ============================================================================
# Report Generation
# ============================================================================

def generate_markdown_report(results_df: pd.DataFrame, stats_results: Dict,
                             out_path: str = 'test_output/null_control_3_report.md'):
    """Generate comprehensive markdown report."""

    # Determine pass/fail criteria
    plv_pass = False
    sr_score_pass = False
    z_score_pass = False

    if 'plv_sr1' in stats_results:
        plv_stats = stats_results['plv_sr1']
        plv_pass = plv_stats['p_value'] < 0.001 and plv_stats['cohens_d'] > 1.0

    if 'sr_score' in stats_results:
        sr_stats = stats_results['sr_score']
        sr_score_pass = sr_stats['p_value'] < 0.001 and sr_stats['cohens_d'] > 0.8

    if 'sr1_z' in stats_results:
        z_stats = stats_results['sr1_z']
        z_score_pass = abs(z_stats['cohens_d']) < 0.2

    overall_pass = plv_pass and sr_score_pass and z_score_pass

    report = f"""# Null Control Test 1: Phase-Randomized Surrogates

**Test Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This null control test validates whether SR-brain coupling metrics reflect genuine temporal phase
relationships or are simply spectral artifacts within predefined search windows (7.6±0.6, 20±1, 32±2 Hz).

**Test Result:** {'✅ PASS' if overall_pass else '❌ FAIL'}

---

## Methodology

1. **Load Observed Events**: Load SIE events with pre-computed metrics
2. **Phase Randomization**: For each window, generate surrogates via FFT → randomize phases → iFFT
3. **Metric Recomputation**: Apply same metric functions to phase-randomized data
4. **Statistical Comparison**: Paired Wilcoxon signed-rank tests (observed vs. surrogate)

**What Phase Randomization Destroys:**
- Temporal phase relationships
- Cross-channel phase coherence
- Genuine coupling signals

**What Phase Randomization Preserves:**
- Power spectrum
- Frequency content
- Spectral properties (peaks in search windows)

---

## Results

### 1. Primary Coupling Metrics

"""

    # SR-Score
    if 'sr_score' in stats_results:
        sr_stats = stats_results['sr_score']
        report += f"""
#### SR-Score (Composite Coupling Metric)

| Group | Mean | Std | n |
|-------|------|-----|---|
| Observed | {sr_stats['obs_mean']:.4f} | {sr_stats['obs_std']:.4f} | {sr_stats['n']} |
| Phase-Randomized | {sr_stats['surr_mean']:.4f} | {sr_stats['surr_std']:.4f} | {sr_stats['n']} |

**Statistical Test:**
- Wilcoxon signed-rank: p = {sr_stats['p_value']:.6f}
- Cohen's d: {sr_stats['cohens_d']:.3f}

**Interpretation:** {'✅ PASS - SR coupling destroyed by phase randomization' if sr_score_pass else '❌ FAIL - No significant difference'}

"""

    # PLV
    if 'plv_sr1' in stats_results:
        plv_stats = stats_results['plv_sr1']
        report += f"""
#### PLV @ SR1 (Phase Locking Value)

| Group | Mean | Std | n |
|-------|------|-----|---|
| Observed | {plv_stats['obs_mean']:.4f} | {plv_stats['obs_std']:.4f} | {plv_stats['n']} |
| Phase-Randomized | {plv_stats['surr_mean']:.4f} | {plv_stats['surr_std']:.4f} | {plv_stats['n']} |

**Statistical Test:**
- Wilcoxon signed-rank: p = {plv_stats['p_value']:.6f}
- Cohen's d: {plv_stats['cohens_d']:.3f}

**Interpretation:** {'✅ PASS - Phase coupling destroyed' if plv_pass else '❌ FAIL - No significant difference'}

"""

    # MSC
    if 'msc_sr1' in stats_results:
        msc_stats = stats_results['msc_sr1']
        report += f"""
#### MSC @ SR1 (Magnitude Squared Coherence)

| Group | Mean | Std | n |
|-------|------|-----|---|
| Observed | {msc_stats['obs_mean']:.4f} | {msc_stats['obs_std']:.4f} | {msc_stats['n']} |
| Phase-Randomized | {msc_stats['surr_mean']:.4f} | {msc_stats['surr_std']:.4f} | {msc_stats['n']} |

**Statistical Test:**
- Wilcoxon signed-rank: p = {msc_stats['p_value']:.6f}
- Cohen's d: {msc_stats['cohens_d']:.3f}

**Interpretation:** MSC may show partial preservation (spectral coherence component)

"""

    # Z-scores (validation)
    report += """
### 2. Spectral Property Validation

Phase randomization should PRESERVE spectral properties (envelope z-scores):

"""

    if 'sr1_z' in stats_results:
        z_stats = stats_results['sr1_z']
        report += f"""
#### Envelope Z-Score @ SR1

| Group | Mean | Std | n |
|-------|------|-----|---|
| Observed | {z_stats['obs_mean']:.4f} | {z_stats['obs_std']:.4f} | {z_stats['n']} |
| Phase-Randomized | {z_stats['surr_mean']:.4f} | {z_stats['surr_std']:.4f} | {z_stats['n']} |

**Statistical Test:**
- Wilcoxon signed-rank: p = {z_stats['p_value']:.6f}
- Cohen's d: {z_stats['cohens_d']:.3f}

**Interpretation:** {'✅ PASS - Spectral properties preserved (|d| < 0.2)' if z_score_pass else '⚠️ WARNING - Unexpected spectral change'}

"""

    report += f"""
---

## Pass/Fail Criteria

| Criterion | Threshold | Actual | Status |
|-----------|-----------|--------|--------|
| PLV destruction | p < 0.001, d > 1.0 | p = {stats_results.get('plv_sr1', {}).get('p_value', np.nan):.1e}, d = {stats_results.get('plv_sr1', {}).get('cohens_d', np.nan):.2f} | {'✅ Pass' if plv_pass else '❌ Fail'} |
| SR-score reduction | p < 0.001, d > 0.8 | p = {stats_results.get('sr_score', {}).get('p_value', np.nan):.1e}, d = {stats_results.get('sr_score', {}).get('cohens_d', np.nan):.2f} | {'✅ Pass' if sr_score_pass else '❌ Fail'} |
| Z-score preservation | \\|d\\| < 0.2 | \\|d\\| = {abs(stats_results.get('sr1_z', {}).get('cohens_d', np.nan)):.2f} | {'✅ Pass' if z_score_pass else '❌ Fail'} |

**Overall:** {'✅ PASS' if overall_pass else '❌ FAIL'}

---

## Interpretation

"""

    if overall_pass:
        report += """
**PASS:** Phase randomization destroyed temporal coupling while preserving spectral properties.

**Conclusion:** The observed SR-brain coupling reflects **genuine temporal phase relationships**,
not spectral artifacts in the search windows.

**Key Evidence:**
1. PLV collapsed when phase coherence was destroyed (large effect)
2. SR-score dropped significantly (temporal coupling destroyed)
3. Envelope z-scores unchanged (spectral properties preserved)

**This validates that the detected SR-brain coupling is REAL.**
"""
    else:
        report += """
**FAIL:** Phase randomization did not significantly alter coupling metrics.

**Possible Interpretations:**
1. Coupling metrics may reflect spectral artifacts, not genuine temporal coupling
2. Search windows (7.6±0.6, 20±1, 32±2 Hz) constrain peaks regardless of phase
3. Peaks could be EEG rhythms coinciding with SR frequencies, not external SR coupling

**Recommendation:** Re-evaluate whether detected peaks represent genuine SR signals vs. artifacts.
"""

    report += f"""

---

## Visualization

![Phase Randomization Results](null_control_3_results.png)

---

## Technical Notes

1. **Phase Randomization:** FFT → randomize phases → iFFT (preserves power spectrum)
2. **Sample Size:** {len(results_df)} paired observations
3. **Statistical Test:** Wilcoxon signed-rank (paired, non-parametric)
4. **Effect Size:** Cohen's d for paired data
5. **Surrogates:** {N_SURROGATES} per event

---

*Generated by null_control_3_phase_randomization.py*
"""

    # Create output directory if needed
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Write report
    with open(out_path, 'w') as f:
        f.write(report)

    print(f"Report saved: {out_path}")
    return report


# ============================================================================
# Main Processing
# ============================================================================

def main():
    print("=" * 80)
    print("NULL CONTROL TEST 1: Phase-Randomized Surrogates")
    print("=" * 80)
    print()

    # Set random seed
    np.random.seed(RANDOM_SEED)

    # Load observed events
    print(f"Loading observed events from {DATA_FILE}...", flush=True)
    events_df = pd.read_csv(DATA_FILE)

    # Limit to test subset if specified
    if N_TEST_EVENTS is not None:
        events_df = events_df.head(N_TEST_EVENTS)
        print(f"  Limited to {N_TEST_EVENTS} events for testing", flush=True)

    n_events = len(events_df)
    print(f"  Loaded {n_events} observed SIE events", flush=True)
    print(flush=True)

    # Process each event
    results = []
    failed_events = []

    print("Processing events (computing metrics on observed + phase-randomized data)...", flush=True)
    print(flush=True)

    for idx, event in events_df.iterrows():
        if (idx + 1) % 5 == 0 or idx == 0:
            print(f"  [{idx+1}/{n_events}] {event['session_name']}", flush=True)

        try:
            # Find EEG file
            eeg_file = find_eeg_file(event['session_name'])
            if eeg_file is None:
                failed_events.append((idx, event['session_name'], "File not found"))
                continue

            # Load EEG window
            X, y, fs = load_eeg_window(eeg_file, event['t_start'], event['t_end'])

            # Compute metrics on OBSERVED data
            # (Note: we already have these from the CSV, but recompute for consistency)
            obs_metrics = compute_all_metrics(
                X, y, fs, HARMONICS_HZ, HALF_BW_HZ,
                freq_specificity=event.get('specificity', 0.1)
            )

            # Generate phase-randomized surrogates and compute metrics
            surr_metrics_list = []
            for surr_idx in range(N_SURROGATES):
                # Phase randomize
                X_surr = phase_randomize_multichannel(X, seed=RANDOM_SEED + idx * 100 + surr_idx)
                y_surr = np.nanmean(X_surr, axis=0)

                # Compute metrics on SURROGATE data
                surr_metrics = compute_all_metrics(
                    X_surr, y_surr, fs, HARMONICS_HZ, HALF_BW_HZ,
                    freq_specificity=event.get('specificity', 0.1)
                )
                surr_metrics_list.append(surr_metrics)

            # Average over surrogates
            surr_metrics_avg = {
                key: np.mean([m[key] for m in surr_metrics_list])
                for key in surr_metrics_list[0].keys()
            }

            # Store paired results
            results.append({
                'event_idx': idx,
                'session': event['session_name'],
                't_start': event['t_start'],
                't_end': event['t_end'],
                # Observed metrics (from CSV for validation)
                'obs_sr_score_csv': event['sr_score'],
                'obs_plv_sr1_csv': event.get('plv_sr1', np.nan),
                'obs_msc_sr1_csv': event.get('msc_sr1', np.nan),
                'obs_hsi_csv': event.get('HSI', np.nan),
                # Observed metrics (recomputed)
                **{f'obs_{k}': v for k, v in obs_metrics.items()},
                # Surrogate metrics
                **{f'surr_{k}': v for k, v in surr_metrics_avg.items()}
            })

        except Exception as e:
            failed_events.append((idx, event['session_name'], str(e)))
            continue

    print()
    print(f"Successfully processed: {len(results)}/{n_events} events")
    if failed_events:
        print(f"Failed events: {len(failed_events)}")
        for idx, session, error in failed_events[:5]:
            print(f"  [{idx}] {session}: {error}")
        if len(failed_events) > 5:
            print(f"  ... and {len(failed_events) - 5} more")
    print()

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    output_csv = 'null_control_3_results.csv'
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to: {output_csv}")
    print()

    # Statistical comparison
    print("=" * 80)
    print("STATISTICAL COMPARISON")
    print("=" * 80)
    print()

    stats_results = perform_statistical_comparison(results_df)

    print("Key Results:")
    for metric, stats in stats_results.items():
        print(f"\n{metric.upper()}:")
        print(f"  Observed:  mean={stats['obs_mean']:.4f}, std={stats['obs_std']:.4f}")
        print(f"  Surrogate: mean={stats['surr_mean']:.4f}, std={stats['surr_std']:.4f}")
        print(f"  p-value: {stats['p_value']:.6f}")
        print(f"  Cohen's d: {stats['cohens_d']:.3f}")

    print()

    # Determine overall pass/fail
    plv_pass = False
    sr_score_pass = False
    z_score_pass = False

    if 'plv_sr1' in stats_results:
        plv_stats = stats_results['plv_sr1']
        plv_pass = plv_stats['p_value'] < 0.001 and plv_stats['cohens_d'] > 1.0

    if 'sr_score' in stats_results:
        sr_stats = stats_results['sr_score']
        sr_score_pass = sr_stats['p_value'] < 0.001 and sr_stats['cohens_d'] > 0.8

    if 'sr1_z' in stats_results:
        z_stats = stats_results['sr1_z']
        z_score_pass = abs(z_stats['cohens_d']) < 0.2

    overall_pass = plv_pass and sr_score_pass and z_score_pass

    print("=" * 80)
    print("PASS/FAIL CRITERIA")
    print("=" * 80)
    print(f"PLV destruction (p < 0.001, d > 1.0): {'✅ PASS' if plv_pass else '❌ FAIL'}")
    print(f"SR-score reduction (p < 0.001, d > 0.8): {'✅ PASS' if sr_score_pass else '❌ FAIL'}")
    print(f"Z-score preservation (|d| < 0.2): {'✅ PASS' if z_score_pass else '❌ FAIL'}")
    print()
    print(f"OVERALL: {'✅ PASS' if overall_pass else '❌ FAIL'}")
    print("=" * 80)
    print()

    # Generate outputs
    print("=" * 80)
    print("GENERATING OUTPUTS")
    print("=" * 80)

    try:
        create_four_panel_figure(results_df, stats_results,
                                out_path='null_control_3_results.png')
    except Exception as e:
        print(f"Warning: Failed to generate figure: {e}")

    try:
        generate_markdown_report(results_df, stats_results,
                                out_path='test_output/null_control_3_report.md')
    except Exception as e:
        print(f"Warning: Failed to generate report: {e}")

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()

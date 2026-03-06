"""
Unbiased Peak Distribution Analysis for φⁿ Frequency Validation

This module finds ALL PSD peaks in each EEG band without targeting specific
frequencies, then compares the distribution against φⁿ predictions.

This avoids circular reasoning: instead of searching around predictions,
we find peaks blindly and check if they cluster near predictions.
"""

import numpy as np
import pandas as pd
from scipy.signal import welch, find_peaks
from scipy import stats
from scipy.stats import wilcoxon
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import phi model
from phi_frequency_model import (
    generate_phi_table, get_default_phi_table, PhiTable, PhiPrediction,
    BANDS, PHI, F0
)

# Import GED functions
from ged_band_analysis import ged_weights, bandpass_safe

# Band definitions with upper limit at 45 Hz
ANALYSIS_BANDS = {
    'theta':     {'freq_range': (4.70, 7.60)},
    'alpha':     {'freq_range': (7.60, 12.30)},
    'beta_low':  {'freq_range': (12.30, 19.90)},
    'beta_high': {'freq_range': (19.90, 32.19)},
    'gamma':     {'freq_range': (32.19, 45.0)},  # Capped at 45 Hz
}

# Position colors
POSITION_COLORS = {
    'boundary': '#e74c3c',
    'attractor': '#2ecc71',
    'noble_1': '#3498db',
    'noble_2': '#9b59b6',
    'noble_3': '#f39c12',
    'noble_4': '#1abc9c',
    'inv_noble_3': '#e67e22',
    'inv_noble_4': '#95a5a6'
}


@dataclass
class PeakResult:
    """Single peak detection result."""
    frequency: float
    power: float
    band: str
    session: str
    window_idx: int
    z_score: float = 0.0


def find_all_peaks_in_band(
    x: np.ndarray,
    fs: float,
    band_range: Tuple[float, float],
    nperseg_sec: float = 4.0,
    min_distance_hz: float = 0.3,
    prominence_pct: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find ALL PSD peaks in a frequency band without prior constraints.

    Parameters
    ----------
    x : np.ndarray
        1D signal array
    fs : float
        Sampling rate (Hz)
    band_range : tuple
        (f_low, f_high) in Hz
    nperseg_sec : float
        PSD segment length in seconds
    min_distance_hz : float
        Minimum distance between peaks (Hz)
    prominence_pct : float
        Minimum prominence as fraction of max power in band

    Returns
    -------
    peak_freqs : np.ndarray
        Frequencies of detected peaks
    peak_powers : np.ndarray
        Power values at detected peaks
    """
    # Compute PSD
    nperseg = int(nperseg_sec * fs)
    f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg//2)

    # Extract band
    mask = (f >= band_range[0]) & (f <= band_range[1])
    f_band = f[mask]
    P_band = Pxx[mask]

    if len(f_band) < 3:
        return np.array([]), np.array([])

    # Find peaks
    df = f_band[1] - f_band[0]
    min_dist = max(1, int(min_distance_hz / df))
    prominence = prominence_pct * np.max(P_band)

    peak_idx, props = find_peaks(P_band, distance=min_dist, prominence=prominence)

    if len(peak_idx) == 0:
        return np.array([]), np.array([])

    # Parabolic refinement for sub-bin precision
    refined_freqs = []
    for i in peak_idx:
        refined_freqs.append(_parabolic_refine(f_band, P_band, i))

    return np.array(refined_freqs), P_band[peak_idx]


def _parabolic_refine(f: np.ndarray, y: np.ndarray, i: int) -> float:
    """Quadratic interpolation for sub-bin peak frequency."""
    if i <= 0 or i >= len(y) - 1:
        return float(f[i])

    y0, y1, y2 = np.log(y[i-1] + 1e-18), np.log(y[i] + 1e-18), np.log(y[i+1] + 1e-18)
    denom = y0 - 2*y1 + y2

    if abs(denom) < 1e-18:
        return float(f[i])

    delta = 0.5 * (y0 - y2) / denom
    df = f[1] - f[0]

    return float(f[i] + delta * df)


def find_peaks_in_windows(
    records: pd.DataFrame,
    eeg_channels: List[str],
    fs: float = 128,
    window_sec: float = 10.0,
    step_sec: float = 5.0,
    time_col: str = 'Timestamp',
    combine: str = 'mean'
) -> pd.DataFrame:
    """
    Find peaks across sliding windows in a recording.

    Parameters
    ----------
    records : pd.DataFrame
        EEG data
    eeg_channels : list
        Channel names
    fs : float
        Sampling rate
    window_sec : float
        Window length in seconds
    step_sec : float
        Step between windows
    time_col : str
        Time column name
    combine : str
        How to combine channels: 'mean', 'median', or channel name

    Returns
    -------
    peaks_df : pd.DataFrame
        All detected peaks with columns: frequency, power, band, window_idx
    """
    # Get signal
    X = np.vstack([records[ch].values for ch in eeg_channels if ch in records.columns])

    if combine == 'mean':
        x = X.mean(axis=0)
    elif combine == 'median':
        x = np.median(X, axis=0)
    else:
        x = records[combine].values if combine in records.columns else X.mean(axis=0)

    # Sliding windows
    n_samples = len(x)
    window_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)

    all_peaks = []

    for win_idx, start in enumerate(range(0, n_samples - window_samples, step_samples)):
        segment = x[start:start + window_samples]

        for band, info in ANALYSIS_BANDS.items():
            peak_freqs, peak_powers = find_all_peaks_in_band(
                segment, fs, info['freq_range']
            )

            for freq, power in zip(peak_freqs, peak_powers):
                all_peaks.append({
                    'frequency': freq,
                    'power': power,
                    'band': band,
                    'window_idx': win_idx
                })

    return pd.DataFrame(all_peaks)


def run_peak_distribution_analysis(
    epoc_files: List[str],
    electrodes: List[str],
    fs: float = 128,
    output_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Run blind peak detection on multiple EPOC files.

    Parameters
    ----------
    epoc_files : list
        Paths to EPOC CSV files
    electrodes : list
        EEG channel names
    fs : float
        Sampling rate
    output_dir : str or None
        Output directory for results

    Returns
    -------
    all_peaks : pd.DataFrame
        All detected peaks across all sessions
    """
    all_peaks = []

    for file_path in epoc_files:
        session_name = os.path.basename(file_path)
        print(f"Processing: {session_name}")

        try:
            records = pd.read_csv(file_path, skiprows=1)

            # Find EEG columns
            eeg_cols = [c for c in records.columns
                       if c.startswith('EEG.') and
                       not any(x in c for x in ['FILTERED', 'POW', 'Alpha', 'Beta', 'Theta', 'Delta', 'Gamma'])]

            if len(eeg_cols) < 3:
                print(f"  Skipping: insufficient channels")
                continue

            # Add timestamp if missing
            if 'Timestamp' not in records.columns:
                records['Timestamp'] = np.arange(len(records)) / fs

            # Find peaks
            peaks_df = find_peaks_in_windows(records, eeg_cols[:14], fs=fs)
            peaks_df['session'] = session_name

            all_peaks.append(peaks_df)
            print(f"  Found {len(peaks_df)} peaks")

        except Exception as e:
            print(f"  Error: {str(e)[:50]}")

    if not all_peaks:
        return pd.DataFrame()

    combined = pd.concat(all_peaks, ignore_index=True)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        combined.to_csv(f"{output_dir}/all_peaks.csv", index=False)

    return combined


def get_predictions_for_band(band: str, max_freq: float = 45.0) -> List[Dict]:
    """Get φⁿ predictions for a band, filtered by max frequency."""
    phi_table = get_default_phi_table()
    predictions = phi_table.by_band(band)

    # Filter by max frequency
    filtered = [p for p in predictions if p.frequency <= max_freq]

    return [{'frequency': p.frequency, 'position_type': p.position_type, 'label': p.label}
            for p in filtered]


def compute_clustering_statistics(
    peaks: np.ndarray,
    predictions: List[Dict],
    band_range: Tuple[float, float]
) -> Dict:
    """
    Test if peaks cluster around predictions more than random.

    Parameters
    ----------
    peaks : np.ndarray
        Peak frequencies
    predictions : list
        List of prediction dicts with 'frequency' key
    band_range : tuple
        (f_low, f_high)

    Returns
    -------
    stats : dict
        Clustering statistics
    """
    if len(peaks) == 0 or len(predictions) == 0:
        return {'mean_min_distance': np.nan, 'p_value': np.nan}

    pred_freqs = np.array([p['frequency'] for p in predictions])

    # For each peak, find distance to nearest prediction
    min_distances = []
    nearest_types = []

    for peak in peaks:
        dists = np.abs(peak - pred_freqs)
        min_idx = np.argmin(dists)
        min_distances.append(dists[min_idx])
        nearest_types.append(predictions[min_idx]['position_type'])

    min_distances = np.array(min_distances)

    # Expected mean distance if peaks were uniformly distributed
    band_width = band_range[1] - band_range[0]
    n_pred = len(predictions)

    # Monte Carlo simulation for null distribution
    n_sim = 1000
    null_means = []
    for _ in range(n_sim):
        random_peaks = np.random.uniform(band_range[0], band_range[1], len(peaks))
        random_dists = np.array([np.min(np.abs(rp - pred_freqs)) for rp in random_peaks])
        null_means.append(np.mean(random_dists))

    null_means = np.array(null_means)
    observed_mean = np.mean(min_distances)

    # P-value: fraction of null means <= observed (lower is better clustering)
    p_value = np.mean(null_means <= observed_mean)

    # Clustering ratio: how much better than random
    clustering_ratio = np.mean(null_means) / observed_mean if observed_mean > 0 else np.nan

    return {
        'n_peaks': len(peaks),
        'n_predictions': n_pred,
        'mean_min_distance': observed_mean,
        'median_min_distance': np.median(min_distances),
        'expected_random': np.mean(null_means),
        'std_random': np.std(null_means),
        'p_value': p_value,
        'clustering_ratio': clustering_ratio
    }


def plot_peak_distribution(
    peaks_df: pd.DataFrame,
    output_dir: str,
    max_freq: float = 45.0
):
    """
    Create distribution plots with φⁿ prediction reference lines.

    Parameters
    ----------
    peaks_df : pd.DataFrame
        All detected peaks
    output_dir : str
        Output directory for figures
    max_freq : float
        Maximum frequency for analysis
    """
    os.makedirs(output_dir, exist_ok=True)
    bands = ['theta', 'alpha', 'beta_low', 'beta_high', 'gamma']

    # ============ Figure 1: Multi-band panel ============
    fig, axes = plt.subplots(len(bands), 1, figsize=(14, 3 * len(bands)))

    clustering_stats = {}

    for ax, band in zip(axes, bands):
        band_peaks = peaks_df[peaks_df['band'] == band]['frequency'].values
        band_range = ANALYSIS_BANDS[band]['freq_range']
        predictions = get_predictions_for_band(band, max_freq)

        if len(band_peaks) > 10:
            # Histogram
            bins = np.linspace(band_range[0], band_range[1], 40)
            ax.hist(band_peaks, bins=bins, density=True, alpha=0.5,
                   color='steelblue', edgecolor='black', label='Observed peaks')

            # KDE
            try:
                sns.kdeplot(band_peaks, ax=ax, color='navy', linewidth=2, label='KDE')
            except Exception:
                pass

        # Prediction lines
        added_labels = set()
        for pred in predictions:
            ptype = pred['position_type']
            color = POSITION_COLORS.get(ptype, 'gray')
            label = ptype if ptype not in added_labels else None
            ax.axvline(pred['frequency'], color=color, linestyle='--',
                      alpha=0.8, linewidth=2, label=label)
            added_labels.add(ptype)

        ax.set_xlim(band_range)
        ax.set_xlabel('Frequency (Hz)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f"{band.replace('_', ' ').title()} Band ({band_range[0]:.1f}-{band_range[1]:.1f} Hz) — {len(band_peaks)} peaks",
                    fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8, ncol=2)

        # Compute clustering stats
        stats = compute_clustering_statistics(band_peaks, predictions, band_range)
        clustering_stats[band] = stats

    plt.suptitle('Blind Peak Detection: Distribution vs φⁿ Predictions',
                fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig1_peak_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved fig1_peak_distributions.png")

    # ============ Figure 2: Clustering statistics ============
    fig, ax = plt.subplots(figsize=(12, 6))

    stats_df = pd.DataFrame(clustering_stats).T
    stats_df.index.name = 'band'
    stats_df = stats_df.reset_index()

    x = np.arange(len(bands))
    width = 0.35

    ax.bar(x - width/2, stats_df['mean_min_distance'], width,
           label='Observed mean distance', color='steelblue', alpha=0.7)
    ax.bar(x + width/2, stats_df['expected_random'], width,
           label='Expected if random', color='coral', alpha=0.7)

    # Add error bars for random expectation
    ax.errorbar(x + width/2, stats_df['expected_random'],
                yerr=stats_df['std_random'], fmt='none', color='black', capsize=3)

    # Add p-value annotations
    for i, (_, row) in enumerate(stats_df.iterrows()):
        p = row['p_value']
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax.annotate(f'{sig}\np={p:.3f}', (i, max(row['mean_min_distance'], row['expected_random']) + 0.05),
                   ha='center', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([b.replace('_', ' ').title() for b in bands])
    ax.set_ylabel('Mean Distance to Nearest φⁿ Prediction (Hz)', fontsize=11)
    ax.set_title('Clustering Analysis: Do Peaks Cluster Around φⁿ Predictions?',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig2_clustering_stats.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved fig2_clustering_stats.png")

    # Save stats to CSV
    stats_df.to_csv(f'{output_dir}/clustering_statistics.csv', index=False)
    print(f"Saved clustering_statistics.csv")

    # ============ Figure 3: Cumulative distribution ============
    fig, ax = plt.subplots(figsize=(10, 6))

    for band in bands:
        band_peaks = peaks_df[peaks_df['band'] == band]['frequency'].values
        predictions = get_predictions_for_band(band, max_freq)
        pred_freqs = np.array([p['frequency'] for p in predictions])

        if len(band_peaks) > 0:
            min_dists = np.array([np.min(np.abs(p - pred_freqs)) for p in band_peaks])
            sorted_dists = np.sort(min_dists)
            cdf = np.arange(1, len(sorted_dists) + 1) / len(sorted_dists)

            ax.plot(sorted_dists, cdf, label=band.replace('_', ' ').title(), linewidth=2)

    # Reference lines
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5, label='0.5 Hz threshold')

    ax.set_xlabel('Distance to Nearest φⁿ Prediction (Hz)', fontsize=11)
    ax.set_ylabel('Cumulative Fraction of Peaks', fontsize=11)
    ax.set_title('Cumulative Distribution: How Close Are Peaks to Predictions?',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(0, 2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig3_cumulative_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved fig3_cumulative_distribution.png")

    return clustering_stats


# ============================================================================
# GED-BASED BLIND PEAK DETECTION
# ============================================================================

def ged_blind_sweep(
    X: np.ndarray,
    fs: float,
    band_range: Tuple[float, float],
    step_hz: float = 0.1,
    bw: float = 0.5,
    flank_bw: float = 0.8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sweep GED across full frequency range to find peaks blindly.

    Parameters
    ----------
    X : np.ndarray
        EEG data matrix, shape (n_channels, n_samples)
    fs : float
        Sampling rate (Hz)
    band_range : tuple
        (f_lo, f_hi) frequency range
    step_hz : float
        Frequency step (Hz)
    bw : float
        Signal bandwidth half-width (Hz)
    flank_bw : float
        Flanking noise bandwidth (Hz)

    Returns
    -------
    freqs : np.ndarray
        Frequencies tested
    eigenvalues : np.ndarray
        Eigenvalue at each frequency
    peak_freqs : np.ndarray
        Detected peak frequencies
    peak_eigenvalues : np.ndarray
        Eigenvalues at peaks
    """
    f_lo, f_hi = band_range

    # Frequency range (must have room for signal + flanking bands)
    margin = bw + flank_bw + 0.5
    f_start = max(f_lo, margin)
    f_end = min(f_hi, fs/2 - margin)

    if f_start >= f_end:
        return np.array([]), np.array([]), np.array([]), np.array([])

    freqs = np.arange(f_start, f_end, step_hz)
    eigenvalues = []

    for f in freqs:
        try:
            _, lam, _, _ = ged_weights(X, fs, f, bw=bw, flank_bw=flank_bw)
            eigenvalues.append(lam)
        except Exception:
            eigenvalues.append(0)

    eigenvalues = np.array(eigenvalues)

    if len(eigenvalues) < 3:
        return freqs, eigenvalues, np.array([]), np.array([])

    # Find peaks in eigenvalue profile
    min_dist = max(1, int(0.3 / step_hz))
    prominence = 0.05 * np.max(eigenvalues) if np.max(eigenvalues) > 0 else 0.01

    peak_idx, _ = find_peaks(eigenvalues, distance=min_dist, prominence=prominence)

    if len(peak_idx) == 0:
        return freqs, eigenvalues, np.array([]), np.array([])

    return freqs, eigenvalues, freqs[peak_idx], eigenvalues[peak_idx]


# ============================================================================
# CONTINUOUS GED SWEEP (NO BAND BOUNDARIES)
# ============================================================================

def ged_continuous_sweep(
    X: np.ndarray,
    fs: float,
    freq_range: Tuple[float, float] = (4.5, 45.0),
    step_hz: float = 0.05,
    bw: float = 0.5,
    flank_bw: float = 0.8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Continuous GED sweep across full frequency range.

    No artificial band boundaries - detects peaks wherever they occur.
    This avoids the ~1.8 Hz gaps at band boundaries that occur with
    band-by-band detection.

    Parameters
    ----------
    X : np.ndarray
        EEG data matrix, shape (n_channels, n_samples)
    fs : float
        Sampling rate (Hz)
    freq_range : tuple
        (f_lo, f_hi) full frequency range to sweep
    step_hz : float
        Frequency step (Hz)
    bw : float
        Signal bandwidth half-width (Hz)
    flank_bw : float
        Flanking noise bandwidth (Hz)

    Returns
    -------
    freqs : np.ndarray
        Frequencies tested (continuous range)
    eigenvalues : np.ndarray
        Eigenvalue at each frequency
    peak_freqs : np.ndarray
        Detected peak frequencies
    peak_eigenvalues : np.ndarray
        Eigenvalues at peaks
    """
    f_lo, f_hi = freq_range

    # Only apply margin at absolute edges, not at internal band boundaries
    margin = bw + flank_bw + 0.5
    f_start = max(f_lo, margin)
    f_end = min(f_hi, fs/2 - margin)

    if f_start >= f_end:
        return np.array([]), np.array([]), np.array([]), np.array([])

    freqs = np.arange(f_start, f_end, step_hz)
    eigenvalues = []

    for f in freqs:
        try:
            _, lam, _, _ = ged_weights(X, fs, f, bw=bw, flank_bw=flank_bw)
            eigenvalues.append(lam)
        except Exception:
            eigenvalues.append(0)

    eigenvalues = np.array(eigenvalues)

    if len(eigenvalues) < 3:
        return freqs, eigenvalues, np.array([]), np.array([])

    # Find peaks in eigenvalue profile
    min_dist = max(1, int(0.3 / step_hz))
    prominence = 0.05 * np.max(eigenvalues) if np.max(eigenvalues) > 0 else 0.01

    peak_idx, _ = find_peaks(eigenvalues, distance=min_dist, prominence=prominence)

    if len(peak_idx) == 0:
        return freqs, eigenvalues, np.array([]), np.array([])

    return freqs, eigenvalues, freqs[peak_idx], eigenvalues[peak_idx]


def assign_band_to_frequency(freq: float) -> str:
    """
    Assign a detected peak frequency to nearest φ-octave band.

    Parameters
    ----------
    freq : float
        Peak frequency (Hz)

    Returns
    -------
    band : str
        Band name ('theta', 'alpha', 'beta_low', 'beta_high', 'gamma', or 'unknown')
    """
    for band, info in ANALYSIS_BANDS.items():
        f_lo, f_hi = info['freq_range']
        if f_lo <= freq < f_hi:
            return band
    return 'unknown'


def find_ged_peaks_continuous(
    records: pd.DataFrame,
    eeg_channels: List[str],
    fs: float = 128,
    window_sec: float = 10.0,
    step_sec: float = 5.0,
    freq_range: Tuple[float, float] = (4.5, 45.0),
    sweep_step_hz: float = 0.05,
    use_band_normalization: bool = True
) -> pd.DataFrame:
    """
    Find GED peaks using continuous sweep (no band boundaries).

    This eliminates the ~1.8 Hz gaps at each band boundary that occur
    with band-by-band detection.

    Parameters
    ----------
    records : pd.DataFrame
        EEG data
    eeg_channels : list
        Channel names
    fs : float
        Sampling rate
    window_sec : float
        Window length in seconds
    step_sec : float
        Step between windows
    freq_range : tuple
        Full frequency range to sweep
    sweep_step_hz : float
        Frequency resolution for sweep
    use_band_normalization : bool
        If True, apply per-band peak detection with local prominence thresholds.
        This produces peak counts comparable to band-by-band detection (~500k)
        while still having continuous frequency coverage (no boundary gaps).
        If False, use global prominence threshold across full range.

    Returns
    -------
    peaks_df : pd.DataFrame
        All detected GED peaks with columns:
        frequency, eigenvalue, band (assigned post-hoc), window_idx
    """
    # Get multi-channel signal
    X = np.vstack([records[ch].values for ch in eeg_channels if ch in records.columns])

    n_samples = X.shape[1]
    window_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)

    all_peaks = []

    for win_idx, start in enumerate(range(0, n_samples - window_samples, step_samples)):
        X_win = X[:, start:start + window_samples]

        # Run continuous sweep to get eigenvalue profile (no gaps)
        freqs, eigenvalues, _, _ = ged_continuous_sweep(
            X_win, fs, freq_range, step_hz=sweep_step_hz
        )

        if len(freqs) == 0 or len(eigenvalues) == 0:
            continue

        if use_band_normalization:
            # Apply per-band peak detection on continuous eigenvalue profile
            # This gives continuous coverage + comparable peak counts to band-based
            for band, info in ANALYSIS_BANDS.items():
                f_lo, f_hi = info['freq_range']
                mask = (freqs >= f_lo) & (freqs < f_hi)

                if not np.any(mask):
                    continue

                band_freqs = freqs[mask]
                band_eigenvalues = eigenvalues[mask]

                if len(band_eigenvalues) < 3:
                    continue

                # Per-band prominence normalization (matches band-based behavior)
                prominence = 0.05 * np.max(band_eigenvalues) if np.max(band_eigenvalues) > 0 else 0.01
                min_dist = max(1, int(0.3 / sweep_step_hz))

                peak_idx, _ = find_peaks(band_eigenvalues, distance=min_dist, prominence=prominence)

                for idx in peak_idx:
                    all_peaks.append({
                        'frequency': band_freqs[idx],
                        'eigenvalue': band_eigenvalues[idx],
                        'band': band,
                        'window_idx': win_idx
                    })
        else:
            # Global normalization (original behavior - fewer peaks detected)
            min_dist = max(1, int(0.3 / sweep_step_hz))
            prominence = 0.05 * np.max(eigenvalues) if np.max(eigenvalues) > 0 else 0.01

            peak_idx, _ = find_peaks(eigenvalues, distance=min_dist, prominence=prominence)

            for idx in peak_idx:
                freq = freqs[idx]
                band = assign_band_to_frequency(freq)
                all_peaks.append({
                    'frequency': freq,
                    'eigenvalue': eigenvalues[idx],
                    'band': band,
                    'window_idx': win_idx
                })

    return pd.DataFrame(all_peaks)


def run_continuous_ged_detection(
    epoc_files: List[str],
    electrodes: List[str],
    fs: float = 128,
    output_dir: Optional[str] = None,
    freq_range: Tuple[float, float] = (4.5, 45.0),
    window_sec: float = 10.0,
    step_sec: float = 5.0,
    sweep_step_hz: float = 0.1,
    use_band_normalization: bool = True
) -> pd.DataFrame:
    """
    Run continuous GED peak detection (no band boundaries) on multiple files.

    This is the recommended method for detecting peaks across the full
    frequency range without gaps at band boundaries.

    Parameters
    ----------
    epoc_files : list
        List of EEG file paths
    electrodes : list
        Channel names to use
    fs : float
        Sampling rate
    output_dir : str, optional
        Directory to save results
    freq_range : tuple
        Full frequency range to sweep (default 4.5-45 Hz)
    window_sec : float
        Window length for GED
    step_sec : float
        Step between windows
    sweep_step_hz : float
        Frequency resolution for GED sweep (default 0.1 Hz, matching band-based)
    use_band_normalization : bool
        If True (default), apply per-band peak detection with local prominence
        thresholds. This produces peak counts comparable to band-by-band
        detection (~500k for PhySF) while having continuous frequency coverage.

    Returns
    -------
    combined : pd.DataFrame
        All detected peaks across all sessions
    """
    all_peaks = []

    for file_path in epoc_files:
        session_name = os.path.basename(file_path)
        print(f"Continuous GED: {session_name}")

        try:
            # Try loading with skiprows=0 first (PhySF format)
            records = pd.read_csv(file_path, skiprows=0)

            eeg_cols = [c for c in records.columns
                       if c.startswith('EEG.') and
                       not any(x in c for x in ['FILTERED', 'POW', 'Alpha', 'Beta', 'Theta', 'Delta', 'Gamma'])]

            # If no EEG columns found, try skiprows=1 (other formats)
            if len(eeg_cols) < 3:
                records = pd.read_csv(file_path, skiprows=1)
                eeg_cols = [c for c in records.columns
                           if c.startswith('EEG.') and
                           not any(x in c for x in ['FILTERED', 'POW', 'Alpha', 'Beta', 'Theta', 'Delta', 'Gamma'])]

            if len(eeg_cols) < 3:
                print(f"  Skipping: insufficient channels ({len(eeg_cols)})")
                continue

            if 'Timestamp' not in records.columns:
                records['Timestamp'] = np.arange(len(records)) / fs

            # Use continuous sweep (no band boundaries)
            peaks_df = find_ged_peaks_continuous(
                records, eeg_cols[:14], fs=fs,
                window_sec=window_sec, step_sec=step_sec,
                freq_range=freq_range,
                sweep_step_hz=sweep_step_hz,
                use_band_normalization=use_band_normalization
            )
            peaks_df['session'] = session_name

            all_peaks.append(peaks_df)
            print(f"  Found {len(peaks_df)} peaks (continuous)")

        except Exception as e:
            print(f"  Error: {str(e)[:50]}")

    if not all_peaks:
        return pd.DataFrame()

    combined = pd.concat(all_peaks, ignore_index=True)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        combined.to_csv(f"{output_dir}/ged_peaks_continuous.csv", index=False)
        print(f"\nSaved: {output_dir}/ged_peaks_continuous.csv")
        print(f"Total peaks: {len(combined)}")

        # Print coverage check
        freqs = combined['frequency'].values
        print(f"\nFrequency coverage:")
        print(f"  Min: {freqs.min():.2f} Hz")
        print(f"  Max: {freqs.max():.2f} Hz")

        # Check for gaps at boundaries
        boundary_freqs = [7.60, 12.30, 19.90, 32.19]
        print(f"\nBoundary coverage (peaks within ±0.5 Hz):")
        for bf in boundary_freqs:
            near_boundary = np.sum((freqs >= bf - 0.5) & (freqs <= bf + 0.5))
            print(f"  {bf} Hz: {near_boundary} peaks")

    return combined


def find_ged_peaks_in_windows(
    records: pd.DataFrame,
    eeg_channels: List[str],
    fs: float = 128,
    window_sec: float = 10.0,
    step_sec: float = 5.0
) -> pd.DataFrame:
    """
    Find GED peaks across sliding windows in a recording.

    Parameters
    ----------
    records : pd.DataFrame
        EEG data
    eeg_channels : list
        Channel names
    fs : float
        Sampling rate
    window_sec : float
        Window length in seconds
    step_sec : float
        Step between windows

    Returns
    -------
    peaks_df : pd.DataFrame
        All detected GED peaks
    """
    # Get multi-channel signal
    X = np.vstack([records[ch].values for ch in eeg_channels if ch in records.columns])

    n_samples = X.shape[1]
    window_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)

    all_peaks = []

    for win_idx, start in enumerate(range(0, n_samples - window_samples, step_samples)):
        X_win = X[:, start:start + window_samples]

        for band, info in ANALYSIS_BANDS.items():
            freqs, eigenvalues, peak_freqs, peak_eigs = ged_blind_sweep(
                X_win, fs, info['freq_range']
            )

            for freq, eig in zip(peak_freqs, peak_eigs):
                all_peaks.append({
                    'frequency': freq,
                    'eigenvalue': eig,
                    'band': band,
                    'window_idx': win_idx
                })

    return pd.DataFrame(all_peaks)


def run_ged_peak_analysis(
    epoc_files: List[str],
    electrodes: List[str],
    fs: float = 128,
    output_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Run GED-based blind peak detection on multiple EPOC files.
    """
    all_peaks = []

    for file_path in epoc_files:
        session_name = os.path.basename(file_path)
        print(f"GED Processing: {session_name}")

        try:
            records = pd.read_csv(file_path, skiprows=1)

            eeg_cols = [c for c in records.columns
                       if c.startswith('EEG.') and
                       not any(x in c for x in ['FILTERED', 'POW', 'Alpha', 'Beta', 'Theta', 'Delta', 'Gamma'])]

            if len(eeg_cols) < 3:
                print(f"  Skipping: insufficient channels")
                continue

            if 'Timestamp' not in records.columns:
                records['Timestamp'] = np.arange(len(records)) / fs

            peaks_df = find_ged_peaks_in_windows(records, eeg_cols[:14], fs=fs)
            peaks_df['session'] = session_name

            all_peaks.append(peaks_df)
            print(f"  Found {len(peaks_df)} GED peaks")

        except Exception as e:
            print(f"  Error: {str(e)[:50]}")

    if not all_peaks:
        return pd.DataFrame()

    combined = pd.concat(all_peaks, ignore_index=True)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        combined.to_csv(f"{output_dir}/ged_peaks.csv", index=False)

    return combined


def plot_ged_peak_distribution(
    peaks_df: pd.DataFrame,
    output_dir: str,
    max_freq: float = 45.0
):
    """
    Create GED peak distribution plots with φⁿ prediction reference lines.
    """
    os.makedirs(output_dir, exist_ok=True)
    bands = ['theta', 'alpha', 'beta_low', 'beta_high', 'gamma']

    # ============ Figure: Multi-band panel ============
    fig, axes = plt.subplots(len(bands), 1, figsize=(14, 3 * len(bands)))

    clustering_stats = {}

    for ax, band in zip(axes, bands):
        band_peaks = peaks_df[peaks_df['band'] == band]['frequency'].values
        band_range = ANALYSIS_BANDS[band]['freq_range']
        predictions = get_predictions_for_band(band, max_freq)

        if len(band_peaks) > 10:
            bins = np.linspace(band_range[0], min(band_range[1], max_freq), 40)
            ax.hist(band_peaks, bins=bins, density=True, alpha=0.5,
                   color='darkgreen', edgecolor='black', label='GED peaks')

            try:
                sns.kdeplot(band_peaks, ax=ax, color='darkgreen', linewidth=2, label='KDE')
            except Exception:
                pass

        # Prediction lines
        added_labels = set()
        for pred in predictions:
            ptype = pred['position_type']
            color = POSITION_COLORS.get(ptype, 'gray')
            label = ptype if ptype not in added_labels else None
            ax.axvline(pred['frequency'], color=color, linestyle='--',
                      alpha=0.8, linewidth=2, label=label)
            added_labels.add(ptype)

        ax.set_xlim(band_range[0], min(band_range[1], max_freq))
        ax.set_xlabel('Frequency (Hz)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f"{band.replace('_', ' ').title()} Band — {len(band_peaks)} GED peaks",
                    fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8, ncol=2)

        stats = compute_clustering_statistics(band_peaks, predictions, band_range)
        clustering_stats[band] = stats

    plt.suptitle('GED Blind Peak Detection: Distribution vs φⁿ Predictions',
                fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ged_peak_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved ged_peak_distributions.png")

    # Save stats
    stats_df = pd.DataFrame(clustering_stats).T
    stats_df.index.name = 'band'
    stats_df.to_csv(f'{output_dir}/ged_clustering_statistics.csv')
    print(f"Saved ged_clustering_statistics.csv")

    return clustering_stats


def plot_eigenvalue_profiles(
    records: pd.DataFrame,
    eeg_channels: List[str],
    output_dir: str,
    fs: float = 128,
    session_name: str = "session",
    max_freq: float = 45.0
):
    """
    Generate eigenvalue profile plots showing λ(f) curves for each band.

    Parameters
    ----------
    records : pd.DataFrame
        EEG data
    eeg_channels : list
        Channel names
    output_dir : str
        Output directory for figures
    fs : float
        Sampling rate
    session_name : str
        Session identifier for figure titles
    max_freq : float
        Maximum frequency for analysis
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get multi-channel signal
    X = np.vstack([records[ch].values for ch in eeg_channels if ch in records.columns])

    bands = ['theta', 'alpha', 'beta_low', 'beta_high', 'gamma']

    fig, axes = plt.subplots(len(bands), 1, figsize=(14, 3 * len(bands)))

    for ax, band in zip(axes, bands):
        band_range = ANALYSIS_BANDS[band]['freq_range']
        predictions = get_predictions_for_band(band, max_freq)

        # Run GED sweep
        freqs, eigenvalues, peak_freqs, peak_eigs = ged_blind_sweep(
            X, fs, band_range, step_hz=0.05, bw=0.5, flank_bw=0.8
        )

        if len(freqs) > 0:
            # Plot eigenvalue profile
            ax.plot(freqs, eigenvalues, 'b-', linewidth=1.5, label='λ(f)', alpha=0.8)

            # Mark detected peaks
            if len(peak_freqs) > 0:
                ax.scatter(peak_freqs, peak_eigs, c='red', s=60, zorder=5,
                          edgecolor='black', label='GED peaks')

        # Prediction lines
        added_labels = set()
        for pred in predictions:
            ptype = pred['position_type']
            color = POSITION_COLORS.get(ptype, 'gray')
            label = ptype if ptype not in added_labels else None
            ax.axvline(pred['frequency'], color=color, linestyle='--',
                      alpha=0.7, linewidth=1.5, label=label)
            added_labels.add(ptype)

        ax.set_xlim(band_range[0], min(band_range[1], max_freq))
        ax.set_xlabel('Frequency (Hz)', fontsize=10)
        ax.set_ylabel('GED Eigenvalue λ', fontsize=10)
        ax.set_title(f"{band.replace('_', ' ').title()} Band — Eigenvalue Profile",
                    fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'GED Eigenvalue Profiles: {session_name}',
                fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/eigenvalue_profiles_{session_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved eigenvalue_profiles_{session_name}.png")


def compare_psd_vs_ged_clustering(
    psd_peaks_df: pd.DataFrame,
    ged_peaks_df: pd.DataFrame,
    output_dir: str,
    max_freq: float = 45.0
):
    """
    Compare clustering statistics between PSD-based and GED-based peak detection.

    Parameters
    ----------
    psd_peaks_df : pd.DataFrame
        PSD peaks with 'frequency' and 'band' columns
    ged_peaks_df : pd.DataFrame
        GED peaks with 'frequency' and 'band' columns
    output_dir : str
        Output directory for figures
    max_freq : float
        Maximum frequency for analysis
    """
    os.makedirs(output_dir, exist_ok=True)
    bands = ['theta', 'alpha', 'beta_low', 'beta_high', 'gamma']

    psd_stats = {}
    ged_stats = {}

    for band in bands:
        band_range = ANALYSIS_BANDS[band]['freq_range']
        predictions = get_predictions_for_band(band, max_freq)

        psd_peaks = psd_peaks_df[psd_peaks_df['band'] == band]['frequency'].values
        ged_peaks = ged_peaks_df[ged_peaks_df['band'] == band]['frequency'].values

        psd_stats[band] = compute_clustering_statistics(psd_peaks, predictions, band_range)
        ged_stats[band] = compute_clustering_statistics(ged_peaks, predictions, band_range)

    # Create comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(bands))
    width = 0.35

    # Clustering ratio comparison
    ax = axes[0]
    psd_ratios = [psd_stats[b].get('clustering_ratio', 0) for b in bands]
    ged_ratios = [ged_stats[b].get('clustering_ratio', 0) for b in bands]

    bars1 = ax.bar(x - width/2, psd_ratios, width, label='PSD peaks', color='steelblue', alpha=0.7)
    bars2 = ax.bar(x + width/2, ged_ratios, width, label='GED peaks', color='darkgreen', alpha=0.7)

    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace('_', ' ').title() for b in bands])
    ax.set_ylabel('Clustering Ratio (higher = better clustering)', fontsize=11)
    ax.set_title('Clustering Ratio: PSD vs GED', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0.8, max(max(psd_ratios), max(ged_ratios)) * 1.1)

    # P-value comparison
    ax = axes[1]
    psd_pvals = [psd_stats[b].get('p_value', 1) for b in bands]
    ged_pvals = [ged_stats[b].get('p_value', 1) for b in bands]

    # Convert to -log10 for visualization
    psd_log = [-np.log10(max(p, 1e-10)) for p in psd_pvals]
    ged_log = [-np.log10(max(p, 1e-10)) for p in ged_pvals]

    bars1 = ax.bar(x - width/2, psd_log, width, label='PSD peaks', color='steelblue', alpha=0.7)
    bars2 = ax.bar(x + width/2, ged_log, width, label='GED peaks', color='darkgreen', alpha=0.7)

    # Significance thresholds
    ax.axhline(-np.log10(0.05), color='orange', linestyle='--', alpha=0.7, label='p=0.05')
    ax.axhline(-np.log10(0.001), color='red', linestyle='--', alpha=0.7, label='p=0.001')

    ax.set_xticks(x)
    ax.set_xticklabels([b.replace('_', ' ').title() for b in bands])
    ax.set_ylabel('-log₁₀(p-value)', fontsize=11)
    ax.set_title('Statistical Significance: PSD vs GED', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)

    plt.suptitle('Comparison: PSD vs GED Peak Clustering Around φⁿ Predictions',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/psd_vs_ged_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved psd_vs_ged_comparison.png")

    # Create summary table
    summary_rows = []
    for band in bands:
        psd = psd_stats[band]
        ged = ged_stats[band]
        summary_rows.append({
            'band': band,
            'psd_n_peaks': psd.get('n_peaks', 0),
            'psd_ratio': psd.get('clustering_ratio', np.nan),
            'psd_pvalue': psd.get('p_value', np.nan),
            'ged_n_peaks': ged.get('n_peaks', 0),
            'ged_ratio': ged.get('clustering_ratio', np.nan),
            'ged_pvalue': ged.get('p_value', np.nan)
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f'{output_dir}/psd_vs_ged_summary.csv', index=False)
    print(f"Saved psd_vs_ged_summary.csv")

    return psd_stats, ged_stats


# ============================================================================
# POSITION-TYPE STRATIFIED CLUSTERING ANALYSIS
# ============================================================================

# Position type ordering for consistent visualization
POSITION_ORDER = ['boundary', 'noble_4', 'noble_3', 'noble_2',
                  'attractor', 'noble_1', 'inv_noble_3', 'inv_noble_4']


def compute_clustering_by_position_type(
    peaks: np.ndarray,
    predictions: List[Dict],
    band_range: Tuple[float, float],
    n_simulations: int = 1000
) -> Dict[str, Dict]:
    """
    Compute clustering statistics stratified by position type.

    Parameters
    ----------
    peaks : np.ndarray
        Peak frequencies
    predictions : list
        List of prediction dicts with 'frequency' and 'position_type' keys
    band_range : tuple
        (f_low, f_high)
    n_simulations : int
        Number of Monte Carlo simulations for null distribution

    Returns
    -------
    dict : position_type -> clustering statistics
    """
    if len(peaks) == 0 or len(predictions) == 0:
        return {}

    # Group predictions by position type
    preds_by_type = defaultdict(list)
    for pred in predictions:
        preds_by_type[pred['position_type']].append(pred['frequency'])

    # Assign each peak to nearest prediction, track which type
    peak_assignments = defaultdict(list)  # position_type -> [distances]

    for peak in peaks:
        min_dist = float('inf')
        nearest_type = None
        for ptype, freqs in preds_by_type.items():
            for f in freqs:
                d = abs(peak - f)
                if d < min_dist:
                    min_dist = d
                    nearest_type = ptype
        if nearest_type is not None:
            peak_assignments[nearest_type].append(min_dist)

    # Compute stats per position type
    results = {}
    for ptype in POSITION_ORDER:
        distances = peak_assignments.get(ptype, [])

        if len(distances) == 0:
            results[ptype] = {
                'n_peaks_nearest': 0,
                'mean_distance': np.nan,
                'expected_random': np.nan,
                'clustering_ratio': np.nan,
                'p_value': np.nan
            }
            continue

        # Get frequencies for this position type
        type_freqs = preds_by_type.get(ptype, [])
        if len(type_freqs) == 0:
            results[ptype] = {
                'n_peaks_nearest': len(distances),
                'mean_distance': np.mean(distances),
                'expected_random': np.nan,
                'clustering_ratio': np.nan,
                'p_value': np.nan
            }
            continue

        # Monte Carlo null: random peaks, same count
        null_means = []
        for _ in range(n_simulations):
            rand_peaks = np.random.uniform(band_range[0], band_range[1], len(distances))
            rand_dists = [min(abs(rp - f) for f in type_freqs) for rp in rand_peaks]
            null_means.append(np.mean(rand_dists))

        null_means = np.array(null_means)
        observed_mean = np.mean(distances)
        p_value = np.mean(null_means <= observed_mean)

        results[ptype] = {
            'n_peaks_nearest': len(distances),
            'mean_distance': observed_mean,
            'expected_random': np.mean(null_means),
            'std_random': np.std(null_means),
            'clustering_ratio': np.mean(null_means) / observed_mean if observed_mean > 0 else np.nan,
            'p_value': p_value
        }

    return results


def plot_position_type_clustering(
    stats_by_band: Dict[str, Dict[str, Dict]],
    output_dir: str
):
    """
    Create visualizations for position-type stratified clustering.

    Parameters
    ----------
    stats_by_band : dict
        band -> position_type -> stats dict
    output_dir : str
        Output directory for figures
    """
    os.makedirs(output_dir, exist_ok=True)

    # ============ Figure 1: Aggregated across bands ============
    # Aggregate stats across bands
    aggregated = {}
    for ptype in POSITION_ORDER:
        all_peaks = 0
        weighted_ratio = 0
        total_weight = 0
        all_pvals = []

        for band, band_stats in stats_by_band.items():
            if ptype in band_stats:
                s = band_stats[ptype]
                n = s.get('n_peaks_nearest', 0)
                all_peaks += n
                ratio = s.get('clustering_ratio', np.nan)
                if not np.isnan(ratio) and n > 0:
                    weighted_ratio += ratio * n
                    total_weight += n
                p = s.get('p_value', np.nan)
                if not np.isnan(p):
                    all_pvals.append(p)

        aggregated[ptype] = {
            'total_peaks': all_peaks,
            'mean_ratio': weighted_ratio / total_weight if total_weight > 0 else np.nan,
            'min_p': min(all_pvals) if all_pvals else np.nan
        }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Peak counts
    ax = axes[0]
    counts = [aggregated[p]['total_peaks'] for p in POSITION_ORDER]
    bars = ax.bar(range(len(POSITION_ORDER)), counts,
                  color=[POSITION_COLORS[p] for p in POSITION_ORDER])
    ax.set_xticks(range(len(POSITION_ORDER)))
    ax.set_xticklabels(POSITION_ORDER, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Peak Count', fontsize=11)
    ax.set_title('Peaks Assigned to Each Position Type', fontsize=12, fontweight='bold')

    # Panel 2: Clustering ratio
    ax = axes[1]
    ratios = [aggregated[p]['mean_ratio'] for p in POSITION_ORDER]
    bars = ax.bar(range(len(POSITION_ORDER)), ratios,
                  color=[POSITION_COLORS[p] for p in POSITION_ORDER])
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.7, label='Random baseline')
    ax.set_xticks(range(len(POSITION_ORDER)))
    ax.set_xticklabels(POSITION_ORDER, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Clustering Ratio', fontsize=11)
    ax.set_title('Clustering Strength (higher = better)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)

    # Panel 3: Best p-values
    ax = axes[2]
    pvals = [aggregated[p]['min_p'] for p in POSITION_ORDER]
    log_pvals = [-np.log10(max(p, 1e-10)) if not np.isnan(p) else 0 for p in pvals]
    bars = ax.bar(range(len(POSITION_ORDER)), log_pvals,
                  color=[POSITION_COLORS[p] for p in POSITION_ORDER])
    ax.axhline(-np.log10(0.05), color='orange', linestyle='--', alpha=0.7, label='p=0.05')
    ax.axhline(-np.log10(0.001), color='red', linestyle='--', alpha=0.7, label='p=0.001')
    ax.set_xticks(range(len(POSITION_ORDER)))
    ax.set_xticklabels(POSITION_ORDER, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('-log₁₀(p-value)', fontsize=11)
    ax.set_title('Statistical Significance (best band)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)

    plt.suptitle('Position-Type Stratified Clustering Analysis (All Bands)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/position_type_clustering_aggregated.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved position_type_clustering_aggregated.png")

    # ============ Figure 2: Per-band heatmap ============
    bands = list(stats_by_band.keys())

    # Build ratio matrix
    ratio_matrix = np.zeros((len(POSITION_ORDER), len(bands)))
    for i, ptype in enumerate(POSITION_ORDER):
        for j, band in enumerate(bands):
            if band in stats_by_band and ptype in stats_by_band[band]:
                ratio_matrix[i, j] = stats_by_band[band][ptype].get('clustering_ratio', np.nan)
            else:
                ratio_matrix[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(ratio_matrix, cmap='RdYlGn', aspect='auto', vmin=0.8, vmax=1.6)
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([b.replace('_', ' ').title() for b in bands], fontsize=10)
    ax.set_yticks(range(len(POSITION_ORDER)))
    ax.set_yticklabels(POSITION_ORDER, fontsize=10)
    ax.set_xlabel('Band', fontsize=11)
    ax.set_ylabel('Position Type', fontsize=11)
    ax.set_title('Clustering Ratio by Position Type and Band', fontsize=13, fontweight='bold')

    # Add text annotations
    for i in range(len(POSITION_ORDER)):
        for j in range(len(bands)):
            val = ratio_matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val < 1.0 or val > 1.4 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=9)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Clustering Ratio', fontsize=11)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/position_type_clustering_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved position_type_clustering_heatmap.png")

    # ============ Save to CSV ============
    rows = []
    for band, band_stats in stats_by_band.items():
        for ptype, s in band_stats.items():
            rows.append({
                'band': band,
                'position_type': ptype,
                'n_peaks': s.get('n_peaks_nearest', 0),
                'mean_distance': s.get('mean_distance', np.nan),
                'expected_random': s.get('expected_random', np.nan),
                'clustering_ratio': s.get('clustering_ratio', np.nan),
                'p_value': s.get('p_value', np.nan)
            })

    df = pd.DataFrame(rows)
    df.to_csv(f'{output_dir}/position_type_clustering.csv', index=False)
    print(f"Saved position_type_clustering.csv")

    return aggregated


# ============================================================================
# NOBLE_1 VS ATTRACTOR HYPOTHESIS TEST
# ============================================================================

def compare_attractor_vs_noble1(
    peaks_df: pd.DataFrame,
    bands: List[str] = None,
    max_freq: float = 45.0
) -> Dict:
    """
    Direct comparison of clustering between attractor and noble_1 positions.

    Tests H₀: attractor (0.500) and noble_1 (0.618) show equal clustering
    Tests H₁: noble_1 shows stronger clustering than attractor

    Parameters
    ----------
    peaks_df : pd.DataFrame
        Peak frequencies with 'band' and 'frequency' columns
    bands : list
        Bands to analyze (default: all)
    max_freq : float
        Maximum frequency for analysis

    Returns
    -------
    dict : comparison results including effect size and p-value
    """
    if bands is None:
        bands = list(ANALYSIS_BANDS.keys())

    attractor_distances = []
    noble1_distances = []
    by_band = {}

    for band in bands:
        band_peaks = peaks_df[peaks_df['band'] == band]['frequency'].values
        predictions = get_predictions_for_band(band, max_freq)

        # Separate predictions
        attractor_preds = [p['frequency'] for p in predictions if p['position_type'] == 'attractor']
        noble1_preds = [p['frequency'] for p in predictions if p['position_type'] == 'noble_1']

        if len(attractor_preds) == 0 or len(noble1_preds) == 0:
            continue

        band_att_dists = []
        band_n1_dists = []

        # For each peak, compute distance to nearest attractor AND nearest noble_1
        for peak in band_peaks:
            d_att = min(abs(peak - f) for f in attractor_preds)
            d_n1 = min(abs(peak - f) for f in noble1_preds)
            attractor_distances.append(d_att)
            noble1_distances.append(d_n1)
            band_att_dists.append(d_att)
            band_n1_dists.append(d_n1)

        by_band[band] = {
            'n_peaks': len(band_peaks),
            'attractor_mean': np.mean(band_att_dists) if band_att_dists else np.nan,
            'noble1_mean': np.mean(band_n1_dists) if band_n1_dists else np.nan,
            'attractor_better': np.mean(band_att_dists) < np.mean(band_n1_dists) if band_att_dists else None
        }

    if len(attractor_distances) == 0:
        return {'error': 'No peaks found'}

    attractor_distances = np.array(attractor_distances)
    noble1_distances = np.array(noble1_distances)

    # Statistical comparison - paired Wilcoxon test (same peaks, different references)
    try:
        stat, p_wilcoxon = wilcoxon(attractor_distances, noble1_distances)
    except Exception as e:
        stat, p_wilcoxon = np.nan, np.nan

    # Effect size (Cohen's d)
    mean_diff = np.mean(attractor_distances) - np.mean(noble1_distances)
    pooled_std = np.sqrt((np.std(attractor_distances)**2 + np.std(noble1_distances)**2) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

    return {
        'n_peaks': len(attractor_distances),
        'attractor_mean_dist': np.mean(attractor_distances),
        'attractor_std_dist': np.std(attractor_distances),
        'noble1_mean_dist': np.mean(noble1_distances),
        'noble1_std_dist': np.std(noble1_distances),
        'mean_difference': mean_diff,
        'cohens_d': cohens_d,
        'wilcoxon_stat': stat,
        'wilcoxon_p': p_wilcoxon,
        'noble1_better': np.mean(noble1_distances) < np.mean(attractor_distances),
        'by_band': by_band,
        'attractor_distances': attractor_distances,
        'noble1_distances': noble1_distances
    }


def plot_attractor_vs_noble1(results: Dict, output_dir: str):
    """
    Visualization comparing attractor and noble_1 clustering.

    Parameters
    ----------
    results : dict
        Output from compare_attractor_vs_noble1()
    output_dir : str
        Output directory for figures
    """
    os.makedirs(output_dir, exist_ok=True)

    if 'error' in results:
        print(f"Cannot plot: {results['error']}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Distance distributions
    ax = axes[0]
    bins = np.linspace(0, 1.5, 40)
    ax.hist(results['attractor_distances'], bins=bins, alpha=0.5,
            label=f"Attractor (μ={results['attractor_mean_dist']:.3f})", color='#2ecc71')
    ax.hist(results['noble1_distances'], bins=bins, alpha=0.5,
            label=f"Noble_1 (μ={results['noble1_mean_dist']:.3f})", color='#3498db')
    ax.set_xlabel('Distance to Nearest Prediction (Hz)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.legend(fontsize=10)
    ax.set_title('Peak-to-Prediction Distances', fontsize=12, fontweight='bold')

    # Panel 2: Per-band comparison
    ax = axes[1]
    bands = list(results['by_band'].keys())
    x = np.arange(len(bands))
    width = 0.35
    att_means = [results['by_band'][b]['attractor_mean'] for b in bands]
    n1_means = [results['by_band'][b]['noble1_mean'] for b in bands]
    ax.bar(x - width/2, att_means, width, label='Attractor', color='#2ecc71', alpha=0.7)
    ax.bar(x + width/2, n1_means, width, label='Noble_1', color='#3498db', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace('_', ' ').title() for b in bands], fontsize=9)
    ax.set_ylabel('Mean Distance (Hz)', fontsize=11)
    ax.legend(fontsize=10)
    ax.set_title('Mean Distance by Band', fontsize=12, fontweight='bold')

    # Panel 3: Summary statistics
    ax = axes[2]
    ax.axis('off')

    p = results['wilcoxon_p']
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    winner = 'Noble_1' if results['noble1_better'] else 'Attractor'

    summary_text = f"""
    Attractor vs Noble_1 Comparison
    ════════════════════════════════

    Total peaks analyzed: {results['n_peaks']:,}

    Attractor (0.500):
      Mean distance: {results['attractor_mean_dist']:.4f} Hz
      Std: {results['attractor_std_dist']:.4f} Hz

    Noble_1 (0.618):
      Mean distance: {results['noble1_mean_dist']:.4f} Hz
      Std: {results['noble1_std_dist']:.4f} Hz

    ────────────────────────────────
    Cohen's d: {results['cohens_d']:.3f}
    Wilcoxon p: {p:.2e} {sig}

    Winner: {winner}
    ────────────────────────────────

    Interpretation:
    {'Noble_1 shows stronger clustering!' if results['noble1_better'] else 'Attractor shows stronger clustering.'}
    """

    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Overall title with significance
    fig.suptitle(f"Attractor vs Noble_1: d = {results['cohens_d']:.2f}, p = {p:.2e} {sig}",
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/attractor_vs_noble1.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved attractor_vs_noble1.png")

    # Save results to CSV
    summary_rows = [{
        'metric': 'attractor_mean_dist',
        'value': results['attractor_mean_dist']
    }, {
        'metric': 'attractor_std_dist',
        'value': results['attractor_std_dist']
    }, {
        'metric': 'noble1_mean_dist',
        'value': results['noble1_mean_dist']
    }, {
        'metric': 'noble1_std_dist',
        'value': results['noble1_std_dist']
    }, {
        'metric': 'mean_difference',
        'value': results['mean_difference']
    }, {
        'metric': 'cohens_d',
        'value': results['cohens_d']
    }, {
        'metric': 'wilcoxon_p',
        'value': results['wilcoxon_p']
    }, {
        'metric': 'n_peaks',
        'value': results['n_peaks']
    }, {
        'metric': 'noble1_better',
        'value': int(results['noble1_better'])
    }]

    pd.DataFrame(summary_rows).to_csv(f'{output_dir}/attractor_vs_noble1.csv', index=False)
    print(f"Saved attractor_vs_noble1.csv")


def run_position_type_analysis(
    peaks_df: pd.DataFrame,
    output_dir: str,
    max_freq: float = 45.0
) -> Tuple[Dict, Dict]:
    """
    Run complete position-type stratified analysis.

    Parameters
    ----------
    peaks_df : pd.DataFrame
        Peak frequencies with 'band' and 'frequency' columns
    output_dir : str
        Output directory for results
    max_freq : float
        Maximum frequency for analysis

    Returns
    -------
    tuple : (position_type_stats_by_band, attractor_vs_noble1_results)
    """
    os.makedirs(output_dir, exist_ok=True)
    bands = list(ANALYSIS_BANDS.keys())

    # 1. Position-type stratified clustering
    print("Computing position-type stratified clustering...")
    stats_by_band = {}
    for band in bands:
        band_peaks = peaks_df[peaks_df['band'] == band]['frequency'].values
        band_range = ANALYSIS_BANDS[band]['freq_range']
        predictions = get_predictions_for_band(band, max_freq)

        stats = compute_clustering_by_position_type(band_peaks, predictions, band_range)
        stats_by_band[band] = stats

        print(f"  {band}: {len(band_peaks)} peaks")

    # Generate position-type plots
    aggregated = plot_position_type_clustering(stats_by_band, output_dir)

    # 2. Attractor vs Noble_1 comparison
    print("\nComparing Attractor vs Noble_1...")
    att_vs_n1 = compare_attractor_vs_noble1(peaks_df, bands, max_freq)
    plot_attractor_vs_noble1(att_vs_n1, output_dir)

    # Print summary
    print("\n" + "="*60)
    print("POSITION-TYPE CLUSTERING SUMMARY")
    print("="*60)
    print(f"\n{'Position Type':<15} {'Total Peaks':>12} {'Mean Ratio':>12} {'Best p':>12}")
    print("-"*55)
    for ptype in POSITION_ORDER:
        agg = aggregated.get(ptype, {})
        peaks = agg.get('total_peaks', 0)
        ratio = agg.get('mean_ratio', np.nan)
        p = agg.get('min_p', np.nan)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"{ptype:<15} {peaks:>12} {ratio:>11.2f} {p:>11.4f}{sig}")

    print("\n" + "="*60)
    print("ATTRACTOR VS NOBLE_1 TEST")
    print("="*60)
    if 'error' not in att_vs_n1:
        print(f"\nAttractor mean distance: {att_vs_n1['attractor_mean_dist']:.4f} Hz")
        print(f"Noble_1 mean distance:   {att_vs_n1['noble1_mean_dist']:.4f} Hz")
        print(f"Cohen's d:               {att_vs_n1['cohens_d']:.3f}")
        print(f"Wilcoxon p:              {att_vs_n1['wilcoxon_p']:.2e}")
        winner = 'NOBLE_1' if att_vs_n1['noble1_better'] else 'ATTRACTOR'
        print(f"\nWinner: {winner}")

    return stats_by_band, att_vs_n1


# ============================================================================
# φ^0.25 SUBDIVISION HYPOTHESIS TEST
# ============================================================================

def extract_histogram_modes(
    peaks_df: pd.DataFrame,
    band: str,
    kde_bw: float = 0.1,
    min_prominence: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract mode frequencies from histogram/KDE of GED peaks.

    Parameters
    ----------
    peaks_df : pd.DataFrame
        GED peaks with 'frequency' and 'band' columns
    band : str
        Band name
    kde_bw : float
        KDE bandwidth adjustment
    min_prominence : float
        Minimum prominence for peak detection (fraction of max)

    Returns
    -------
    mode_freqs : np.ndarray
        Sorted mode frequencies
    mode_heights : np.ndarray
        KDE density at each mode
    """
    from scipy.stats import gaussian_kde
    from scipy.signal import find_peaks

    band_freqs = peaks_df[peaks_df['band'] == band]['frequency'].values
    if len(band_freqs) < 10:
        return np.array([]), np.array([])

    band_range = ANALYSIS_BANDS[band]['freq_range']

    # Create KDE
    kde = gaussian_kde(band_freqs, bw_method=kde_bw)
    freq_grid = np.linspace(band_range[0], band_range[1], 1000)
    kde_vals = kde(freq_grid)

    # Find modes
    prominence = min_prominence * kde_vals.max()
    modes_idx, props = find_peaks(kde_vals, prominence=prominence, distance=10)

    mode_freqs = freq_grid[modes_idx]
    mode_heights = kde_vals[modes_idx]

    # Sort by frequency
    sort_idx = np.argsort(mode_freqs)
    return mode_freqs[sort_idx], mode_heights[sort_idx]


def test_phi_025_ratios(
    peaks_df: pd.DataFrame,
    bands: List[str] = None,
    kde_bw: float = 0.1
) -> Dict:
    """
    Test if consecutive histogram mode ratios equal φ^0.25 ≈ 1.1279.

    Returns
    -------
    dict with keys:
        by_band: per-band results
        all_ratios: aggregated consecutive ratios
        mean_ratio: overall mean
        std_ratio: overall std
        t_stat: one-sample t-test statistic
        p_value: significance
        phi_025_hypothesis: 'supported' or 'rejected'
    """
    PHI_025 = PHI ** 0.25  # 1.1278902705

    if bands is None:
        bands = list(ANALYSIS_BANDS.keys())

    all_ratios = []
    by_band = {}

    for band in bands:
        mode_freqs, mode_heights = extract_histogram_modes(peaks_df, band, kde_bw)

        if len(mode_freqs) < 2:
            by_band[band] = {'n_modes': len(mode_freqs), 'ratios': [], 'mean_ratio': np.nan}
            continue

        # Consecutive ratios
        ratios = mode_freqs[1:] / mode_freqs[:-1]
        all_ratios.extend(ratios)

        by_band[band] = {
            'n_modes': len(mode_freqs),
            'mode_freqs': mode_freqs.tolist(),
            'ratios': ratios.tolist(),
            'mean_ratio': np.mean(ratios),
            'std_ratio': np.std(ratios),
            'deviation_from_phi025': (np.mean(ratios) - PHI_025) / PHI_025
        }

    all_ratios = np.array(all_ratios)

    if len(all_ratios) < 3:
        return {'error': 'Insufficient modes detected'}

    # One-sample t-test against φ^0.25
    t_stat, p_value = stats.ttest_1samp(all_ratios, PHI_025)

    # Effect size
    cohens_d = (np.mean(all_ratios) - PHI_025) / np.std(all_ratios)

    # Decision
    mean_ratio = np.mean(all_ratios)
    rel_error = abs(mean_ratio - PHI_025) / PHI_025

    return {
        'by_band': by_band,
        'all_ratios': all_ratios.tolist(),
        'n_ratios': len(all_ratios),
        'mean_ratio': mean_ratio,
        'std_ratio': np.std(all_ratios),
        'expected_phi_025': PHI_025,
        'relative_error': rel_error,
        't_stat': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'phi_025_hypothesis': 'supported' if (p_value > 0.05 and rel_error < 0.05) else 'rejected'
    }


def plot_phi_025_analysis(results: Dict, peaks_df: pd.DataFrame, output_dir: str):
    """
    Create visualization for φ^0.25 hypothesis test.

    Generates:
    1. Per-band KDE with detected modes marked
    2. Histogram of all consecutive ratios with φ^0.25 reference
    3. Summary statistics panel
    """
    from scipy.stats import gaussian_kde

    os.makedirs(output_dir, exist_ok=True)

    if 'error' in results:
        print(f"Cannot plot: {results['error']}")
        return

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    bands = ['theta', 'alpha', 'beta_low', 'beta_high', 'gamma']
    PHI_025 = PHI ** 0.25

    # Top two rows: 5 bands with KDE and modes
    for i, band in enumerate(bands):
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])

        band_freqs = peaks_df[peaks_df['band'] == band]['frequency'].values
        if len(band_freqs) < 10:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            ax.set_title(f"{band.replace('_', ' ').title()}")
            continue

        band_range = ANALYSIS_BANDS[band]['freq_range']

        # Plot KDE
        kde = gaussian_kde(band_freqs, bw_method=0.3)
        freq_grid = np.linspace(band_range[0], band_range[1], 500)
        kde_vals = kde(freq_grid)
        ax.plot(freq_grid, kde_vals, 'b-', linewidth=2)
        ax.fill_between(freq_grid, kde_vals, alpha=0.3)

        # Mark detected modes (red)
        band_results = results['by_band'].get(band, {})
        mode_freqs = band_results.get('mode_freqs', [])
        for mf in mode_freqs:
            ax.axvline(mf, color='red', linestyle='-', alpha=0.8, linewidth=2)

        # φ^0.25 predictions from band start (green dotted)
        pred_freq = band_range[0]
        while pred_freq <= band_range[1]:
            ax.axvline(pred_freq, color='green', linestyle=':', alpha=0.5, linewidth=1)
            pred_freq *= PHI_025

        ax.set_xlim(band_range)
        ax.set_title(f"{band.replace('_', ' ').title()} — {len(mode_freqs)} modes", fontsize=11, fontweight='bold')
        ax.set_xlabel('Frequency (Hz)', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)

        # Add ratio annotation
        ratios = band_results.get('ratios', [])
        if len(ratios) > 0:
            ratio_str = ', '.join([f'{r:.3f}' for r in ratios])
            ax.text(0.02, 0.98, f'Ratios: {ratio_str}', transform=ax.transAxes,
                   fontsize=8, va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Bottom left: Histogram of all ratios
    ax = fig.add_subplot(gs[2, 0])
    all_ratios = results.get('all_ratios', [])
    if len(all_ratios) > 0:
        ax.hist(all_ratios, bins=15, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(PHI_025, color='red', linewidth=2.5, label=f'φ^0.25 = {PHI_025:.4f}')
        ax.axvline(np.mean(all_ratios), color='green', linewidth=2.5, linestyle='--',
                  label=f'Mean = {np.mean(all_ratios):.4f}')
        ax.set_xlabel('Consecutive Mode Ratio', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.legend(fontsize=9, loc='upper right')
        ax.set_title('Distribution of Consecutive Ratios', fontsize=11, fontweight='bold')

    # Bottom middle: Per-band mean ratios
    ax = fig.add_subplot(gs[2, 1])
    band_means = [results['by_band'].get(b, {}).get('mean_ratio', np.nan) for b in bands]
    x = np.arange(len(bands))
    colors = ['#ff7f7f' if not np.isnan(m) and abs(m - PHI_025) / PHI_025 < 0.05 else '#7f7fff'
              for m in band_means]
    ax.bar(x, band_means, color=colors, alpha=0.8, edgecolor='black')
    ax.axhline(PHI_025, color='red', linewidth=2, linestyle='--', label=f'φ^0.25 = {PHI_025:.4f}')
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace('_', '\n') for b in bands], fontsize=9)
    ax.set_ylabel('Mean Consecutive Ratio', fontsize=11)
    ax.set_title('Mean Ratio by Band', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0.9, 1.4)

    # Bottom right: Summary text
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')

    p = results.get('p_value', np.nan)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    hypothesis = results.get('phi_025_hypothesis', 'unknown')

    summary = f"""
    φ^0.25 SUBDIVISION HYPOTHESIS
    ════════════════════════════════

    Expected ratio:  {PHI_025:.4f} (φ^0.25)
    Observed mean:   {results.get('mean_ratio', np.nan):.4f}
    Observed std:    {results.get('std_ratio', np.nan):.4f}
    Relative error:  {results.get('relative_error', np.nan)*100:.2f}%

    N ratios:        {results.get('n_ratios', 0)}
    t-statistic:     {results.get('t_stat', np.nan):.3f}
    p-value:         {p:.2e} {sig}
    Cohen's d:       {results.get('cohens_d', np.nan):.3f}

    ────────────────────────────────
    Hypothesis:      {hypothesis.upper()}
    ────────────────────────────────
    """

    if hypothesis == 'supported':
        summary += "\n    Modes follow φ^0.25 spacing!"
    else:
        summary += "\n    Modes do NOT follow φ^0.25 spacing."

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('φ^0.25 Subdivision Hypothesis: Do Histogram Modes Follow Golden Ratio?',
                 fontsize=14, fontweight='bold')
    plt.savefig(f'{output_dir}/phi_025_hypothesis_test.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved phi_025_hypothesis_test.png")

    # Save mode frequencies to CSV
    mode_rows = []
    for band, band_data in results['by_band'].items():
        mode_freqs = band_data.get('mode_freqs', [])
        ratios = band_data.get('ratios', [])
        for i, mf in enumerate(mode_freqs):
            mode_rows.append({
                'band': band,
                'mode_idx': i,
                'frequency': mf,
                'ratio_to_next': ratios[i] if i < len(ratios) else np.nan
            })
    if mode_rows:
        pd.DataFrame(mode_rows).to_csv(f'{output_dir}/phi_025_modes.csv', index=False)
        print(f"Saved phi_025_modes.csv")

    # Save summary
    summary_rows = [{
        'metric': k,
        'value': v
    } for k, v in results.items() if k not in ['by_band', 'all_ratios']]
    pd.DataFrame(summary_rows).to_csv(f'{output_dir}/phi_025_summary.csv', index=False)
    print(f"Saved phi_025_summary.csv")


def run_phi_025_analysis(
    peaks_df: pd.DataFrame,
    output_dir: str,
    kde_bw: float = 0.1
) -> Dict:
    """
    Run complete φ^0.25 subdivision hypothesis test.

    Parameters
    ----------
    peaks_df : pd.DataFrame
        GED peaks with 'frequency' and 'band' columns
    output_dir : str
        Output directory for results
    kde_bw : float
        KDE bandwidth adjustment

    Returns
    -------
    dict : test results
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("φ^0.25 SUBDIVISION HYPOTHESIS TEST")
    print("="*60)

    # Run the test
    results = test_phi_025_ratios(peaks_df, kde_bw=kde_bw)

    if 'error' in results:
        print(f"Error: {results['error']}")
        return results

    # Print per-band results
    print(f"\nExpected ratio: φ^0.25 = {PHI**0.25:.4f}")
    print(f"\n{'Band':<12} {'Modes':>6} {'Mean Ratio':>12} {'Deviation':>12}")
    print("-" * 45)
    for band in ANALYSIS_BANDS.keys():
        bd = results['by_band'].get(band, {})
        n_modes = bd.get('n_modes', 0)
        mean_r = bd.get('mean_ratio', np.nan)
        dev = bd.get('deviation_from_phi025', np.nan)
        dev_str = f"{dev*100:+.1f}%" if not np.isnan(dev) else "N/A"
        print(f"{band:<12} {n_modes:>6} {mean_r:>12.4f} {dev_str:>12}")

    # Print overall results
    print("\n" + "-" * 45)
    print(f"Overall mean ratio:  {results['mean_ratio']:.4f}")
    print(f"Overall std:         {results['std_ratio']:.4f}")
    print(f"Relative error:      {results['relative_error']*100:.2f}%")
    print(f"t-statistic:         {results['t_stat']:.3f}")
    print(f"p-value:             {results['p_value']:.2e}")
    print(f"Cohen's d:           {results['cohens_d']:.3f}")
    print(f"\nHypothesis:          {results['phi_025_hypothesis'].upper()}")

    # Generate plots
    plot_phi_025_analysis(results, peaks_df, output_dir)

    return results


if __name__ == '__main__':
    from glob import glob

    epoc_files = sorted(glob('data/epoc/*.csv'))
    ELECTRODES = ['AF3','AF4','F7','F8','F3','F4','FC5','FC6','P7','P8','T7','T8','O1','O2']

    output_dir = 'exports_peak_distribution'

    # Run analysis
    peaks_df = run_peak_distribution_analysis(epoc_files, ELECTRODES, output_dir=output_dir)

    # Generate figures
    if len(peaks_df) > 0:
        stats = plot_peak_distribution(peaks_df, f'{output_dir}/figures')
        print("\nClustering Statistics:")
        for band, s in stats.items():
            print(f"  {band}: ratio={s.get('clustering_ratio', 0):.2f}, p={s.get('p_value', 1):.4f}")

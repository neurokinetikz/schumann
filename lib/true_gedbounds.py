"""
True gedBounds: Covariance-Based Frequency Boundary Detection
=============================================================

Implements Cohen (2021)'s gedBounds algorithm that identifies frequency
boundaries by detecting where spatial covariance structure changes.

Unlike the peak-density proxy, this computes actual covariance matrices
at each frequency bin and finds transitions in covariance similarity.

Memory-efficient: Processes one session at a time, aggregates only
the similarity curves (~400 floats per session).
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import spearmanr
from typing import List, Tuple, Dict, Optional, Union
import warnings


def bandpass_filter(data: np.ndarray, fs: float, center_freq: float,
                   bandwidth: float = 2.0) -> np.ndarray:
    """
    Bandpass filter data around a center frequency.

    Parameters
    ----------
    data : array (n_channels, n_samples)
    fs : sampling rate
    center_freq : center frequency in Hz
    bandwidth : total bandwidth in Hz (default 2.0)

    Returns
    -------
    filtered : array (n_channels, n_samples)
    """
    nyq = fs / 2
    low = max(0.5, center_freq - bandwidth/2) / nyq
    high = min(center_freq + bandwidth/2, nyq * 0.95) / nyq

    if low >= high or high >= 1.0:
        # Return NaN array for invalid frequency ranges
        result = np.empty_like(data)
        result.fill(np.nan)
        return result

    try:
        b, a = signal.butter(3, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, data, axis=1)
        return filtered
    except Exception:
        # Return NaN array on filter failure
        result = np.empty_like(data)
        result.fill(np.nan)
        return result


def compute_covariance_at_frequency(data: np.ndarray, fs: float,
                                    center_freq: float,
                                    bandwidth: float = 2.0) -> np.ndarray:
    """
    Compute spatial covariance matrix at a specific frequency.

    Parameters
    ----------
    data : array (n_channels, n_samples)
    fs : sampling rate
    center_freq : frequency of interest
    bandwidth : filter bandwidth

    Returns
    -------
    cov : array (n_channels, n_channels) - covariance matrix
    """
    filtered = bandpass_filter(data, fs, center_freq, bandwidth)

    # Compute covariance
    cov = np.cov(filtered)

    # Handle single channel case
    if cov.ndim == 0:
        cov = np.array([[cov]])

    return cov


def covariance_similarity(cov1: np.ndarray, cov2: np.ndarray,
                          method: str = 'correlation') -> float:
    """
    Compute similarity between two covariance matrices.

    Parameters
    ----------
    cov1, cov2 : covariance matrices
    method : 'correlation' (default), 'frobenius', 'rv_coefficient'

    Returns
    -------
    similarity : float in [0, 1] for correlation/rv, or distance for frobenius
    """
    # Check for NaN in covariance matrices
    if np.any(np.isnan(cov1)) or np.any(np.isnan(cov2)):
        return np.nan

    if method == 'correlation':
        # Flatten and correlate
        flat1 = cov1.flatten()
        flat2 = cov2.flatten()

        # Handle constant arrays (return NaN, not 0)
        if np.std(flat1) == 0 or np.std(flat2) == 0:
            return np.nan

        corr = np.corrcoef(flat1, flat2)[0, 1]
        return max(0, corr)  # Clip negative correlations to 0

    elif method == 'frobenius':
        # Normalized Frobenius distance (lower = more similar)
        diff = np.linalg.norm(cov1 - cov2, 'fro')
        norm = (np.linalg.norm(cov1, 'fro') + np.linalg.norm(cov2, 'fro')) / 2
        if norm == 0:
            return np.nan
        return 1.0 - min(1.0, diff / norm)

    elif method == 'rv_coefficient':
        # RV coefficient - multivariate extension of correlation
        trace_prod = np.trace(cov1 @ cov2.T)
        norm1 = np.sqrt(np.trace(cov1 @ cov1.T))
        norm2 = np.sqrt(np.trace(cov2 @ cov2.T))
        if norm1 == 0 or norm2 == 0:
            return np.nan
        return trace_prod / (norm1 * norm2)

    else:
        raise ValueError(f"Unknown method: {method}")


def compute_similarity_curve(data: np.ndarray, fs: float,
                            freq_range: Tuple[float, float] = (4.5, 45.0),
                            freq_resolution: float = 0.1,
                            bandwidth: float = 2.0,
                            similarity_method: str = 'correlation',
                            verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute covariance similarity curve across frequencies for one session.

    At each frequency, compute covariance matrix. Then compute similarity
    between adjacent frequency covariances. Low similarity = potential boundary.

    Parameters
    ----------
    data : array (n_channels, n_samples) - raw EEG data
    fs : sampling rate
    freq_range : (min_freq, max_freq) in Hz
    freq_resolution : frequency step in Hz
    bandwidth : filter bandwidth for covariance computation
    similarity_method : 'correlation', 'frobenius', 'rv_coefficient'
    verbose : print progress

    Returns
    -------
    frequencies : array of frequency bin centers
    similarities : array of similarity values (length = len(frequencies) - 1)
    """
    # Generate frequency bins
    frequencies = np.arange(freq_range[0], freq_range[1] + freq_resolution, freq_resolution)
    n_freqs = len(frequencies)

    # Compute covariance at each frequency
    covariances = []
    for i, freq in enumerate(frequencies):
        if verbose and i % 50 == 0:
            print(f"  Frequency {freq:.1f} Hz ({i+1}/{n_freqs})")
        cov = compute_covariance_at_frequency(data, fs, freq, bandwidth)
        covariances.append(cov)

    # Compute similarity between adjacent frequencies
    similarities = np.zeros(n_freqs - 1)
    for i in range(n_freqs - 1):
        similarities[i] = covariance_similarity(
            covariances[i], covariances[i+1], similarity_method
        )

    # Return frequency midpoints and similarities
    freq_midpoints = (frequencies[:-1] + frequencies[1:]) / 2

    return freq_midpoints, similarities


def find_boundaries_from_similarity(frequencies: np.ndarray,
                                    similarities: np.ndarray,
                                    prominence_percentile: float = 25,
                                    min_distance_hz: float = 2.0,
                                    smooth_window: int = 5) -> List[float]:
    """
    Find frequency boundaries as local minima in similarity curve.

    Parameters
    ----------
    frequencies : array of frequency midpoints
    similarities : array of similarity values
    prominence_percentile : minimum prominence as percentile of similarity range
    min_distance_hz : minimum distance between boundaries in Hz
    smooth_window : window size for smoothing (odd number)

    Returns
    -------
    boundaries : list of boundary frequencies
    """
    # Smooth the similarity curve
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        similarities_smooth = np.convolve(similarities, kernel, mode='same')
    else:
        similarities_smooth = similarities

    # Compute prominence threshold
    sim_range = np.max(similarities_smooth) - np.min(similarities_smooth)
    min_prominence = sim_range * (prominence_percentile / 100)

    # Convert min_distance_hz to samples
    freq_step = frequencies[1] - frequencies[0] if len(frequencies) > 1 else 0.1
    min_distance_samples = max(1, int(min_distance_hz / freq_step))

    # Find local minima (boundaries are where similarity drops)
    # Invert to find minima using peak finding
    inverted = -similarities_smooth

    peaks, properties = signal.find_peaks(
        inverted,
        prominence=min_prominence,
        distance=min_distance_samples
    )

    # Apply parabolic interpolation for sub-bin precision
    boundaries = []
    for peak_idx in peaks:
        interp_freq = parabolic_interpolation(frequencies, similarities_smooth, peak_idx)
        boundaries.append(round(interp_freq, 2))  # Round to 2 decimal places

    return boundaries


def parabolic_interpolation(frequencies: np.ndarray,
                            values: np.ndarray,
                            peak_idx: int) -> float:
    """
    Use parabolic interpolation to find sub-bin peak/valley location.

    Fits a parabola through 3 points: (peak_idx-1, peak_idx, peak_idx+1)
    and returns the interpolated frequency of the minimum/maximum.

    This provides sub-bin precision beyond the frequency grid resolution,
    solving the X.05/X.95 clustering issue from midpoint-based frequencies.

    Parameters
    ----------
    frequencies : array of frequency values
    values : array of similarity values (we're finding minima in these)
    peak_idx : index of the detected peak/valley

    Returns
    -------
    interp_freq : interpolated frequency at the true minimum
    """
    # Edge cases: can't interpolate at boundaries
    if peak_idx <= 0 or peak_idx >= len(values) - 1:
        return frequencies[peak_idx]

    # Get 3 points around the minimum
    y0, y1, y2 = values[peak_idx-1], values[peak_idx], values[peak_idx+1]
    x0, x1, x2 = frequencies[peak_idx-1], frequencies[peak_idx], frequencies[peak_idx+1]

    # Parabolic interpolation formula
    # Vertex of parabola through (x0,y0), (x1,y1), (x2,y2):
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if abs(denom) < 1e-10:
        return x1

    A = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    if abs(A) < 1e-10:
        return x1

    B = (x2*x2 * (y0 - y1) + x1*x1 * (y2 - y0) + x0*x0 * (y1 - y2)) / denom

    # Vertex x-coordinate: x = -B / (2A)
    x_vertex = -B / (2 * A)

    # Clamp to reasonable range (within one bin of detected peak)
    x_vertex = max(x0, min(x2, x_vertex))

    return x_vertex


def process_session_gedbounds(eeg_data: np.ndarray, fs: float,
                              freq_range: Tuple[float, float] = (4.5, 45.0),
                              freq_resolution: float = 0.1,
                              bandwidth: float = 2.0,
                              similarity_method: str = 'correlation',
                              verbose: bool = False) -> Dict:
    """
    Run gedBounds analysis on a single session.

    Parameters
    ----------
    eeg_data : array (n_channels, n_samples) - raw EEG
    fs : sampling rate
    freq_range : frequency range to analyze
    freq_resolution : frequency step
    bandwidth : filter bandwidth
    similarity_method : similarity metric
    verbose : print progress

    Returns
    -------
    dict with:
        - frequencies: frequency midpoints
        - similarities: similarity curve
        - boundaries: detected boundary frequencies
    """
    frequencies, similarities = compute_similarity_curve(
        eeg_data, fs, freq_range, freq_resolution,
        bandwidth, similarity_method, verbose
    )

    boundaries = find_boundaries_from_similarity(frequencies, similarities)

    return {
        'frequencies': frequencies,
        'similarities': similarities,
        'boundaries': boundaries
    }


def aggregate_similarity_curves(all_frequencies: List[np.ndarray],
                                all_similarities: List[np.ndarray],
                                method: str = 'mean') -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate similarity curves across sessions.

    Parameters
    ----------
    all_frequencies : list of frequency arrays (should be identical)
    all_similarities : list of similarity arrays
    method : 'mean', 'median'

    Returns
    -------
    frequencies : aggregated frequency array
    similarities : aggregated similarity array
    """
    # Stack similarities (assuming all have same frequencies)
    stacked = np.vstack(all_similarities)

    # Use nanmean/nanmedian to ignore NaN values from invalid computations
    if method == 'mean':
        agg_sim = np.nanmean(stacked, axis=0)
    elif method == 'median':
        agg_sim = np.nanmedian(stacked, axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")

    return all_frequencies[0], agg_sim


def validate_boundaries_vs_phi(boundaries: List[float],
                               phi_positions: List[float] = None,
                               n_permutations: int = 10000,
                               freq_range: Tuple[float, float] = (4.5, 45.0)) -> Dict:
    """
    Validate whether boundaries align with φⁿ positions better than chance.

    Parameters
    ----------
    boundaries : empirical boundary frequencies
    phi_positions : φⁿ positions to compare against
    n_permutations : number of random permutations
    freq_range : frequency range for random baseline

    Returns
    -------
    dict with validation statistics
    """
    if phi_positions is None:
        # Default to integer φⁿ boundaries
        phi_positions = [7.60, 12.30, 19.90, 32.19]

    phi_positions = [p for p in phi_positions if freq_range[0] <= p <= freq_range[1]]

    if len(boundaries) == 0:
        return {
            'n_boundaries': 0,
            'n_phi': len(phi_positions),
            'mean_distance_hz': np.nan,
            'p_value': np.nan,
            'effect_size': np.nan
        }

    # Compute mean distance to nearest φⁿ
    def mean_min_distance(bounds, targets):
        if len(bounds) == 0 or len(targets) == 0:
            return np.nan
        distances = []
        for b in bounds:
            min_dist = min(abs(b - t) for t in targets)
            distances.append(min_dist)
        return np.mean(distances)

    observed_distance = mean_min_distance(boundaries, phi_positions)

    # Permutation test: random boundaries
    n_boundaries = len(boundaries)
    random_distances = []

    for _ in range(n_permutations):
        random_bounds = np.random.uniform(freq_range[0], freq_range[1], n_boundaries)
        random_distances.append(mean_min_distance(random_bounds, phi_positions))

    random_distances = np.array(random_distances)

    # P-value: proportion of random distances <= observed
    p_value = np.mean(random_distances <= observed_distance)

    # Effect size: (random_mean - observed) / random_std
    random_mean = np.mean(random_distances)
    random_std = np.std(random_distances)
    effect_size = (random_mean - observed_distance) / random_std if random_std > 0 else 0

    # Count matches (within 0.5 Hz)
    n_matched = sum(1 for b in boundaries if any(abs(b - t) < 0.5 for t in phi_positions))

    return {
        'n_boundaries': n_boundaries,
        'n_phi': len(phi_positions),
        'n_matched': n_matched,
        'mean_distance_hz': observed_distance,
        'random_mean_distance': random_mean,
        'random_std_distance': random_std,
        'effect_size': effect_size,
        'p_value': p_value
    }


def load_eeg_for_gedbounds(filepath: str, electrodes: List[str] = None,
                           fs: float = 128) -> Tuple[np.ndarray, float]:
    """
    Load EEG data from CSV and prepare for gedBounds analysis.

    Parameters
    ----------
    filepath : path to EEG CSV file
    electrodes : list of electrode names (without 'EEG.' prefix)
    fs : expected sampling rate

    Returns
    -------
    data : array (n_channels, n_samples)
    fs : sampling rate
    """
    import sys
    sys.path.insert(0, './lib')
    from utilities import load_eeg_csv

    if electrodes is None:
        electrodes = ['AF3','AF4','F7','F8','F3','F4','FC5','FC6','P7','P8','T7','T8','O1','O2']

    # Load with utilities
    records = load_eeg_csv(filepath, electrodes=electrodes, fs=fs)

    # Extract EEG channels as array
    eeg_cols = [f'EEG.{e}' for e in electrodes if f'EEG.{e}' in records.columns]
    if not eeg_cols:
        eeg_cols = [c for c in records.columns if c.startswith('EEG.') and
                    not any(x in c for x in ['FILTERED', 'POW', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'])]

    data = records[eeg_cols].values.T  # (n_channels, n_samples)

    return data, fs

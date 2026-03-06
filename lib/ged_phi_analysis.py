"""
GED Phi Analysis: GED-based validation of φⁿ frequency predictions
===================================================================

Implements GED (Generalized Eigendecomposition) analysis functions
specifically designed to test the complete φⁿ frequency model with
all 8 position types per octave.

Key Functions:
- ged_sweep_phi(): Position-aware GED sweep with adaptive parameters
- ged_blind_with_positions(): Blind sweep with automatic position assignment
- compute_alignment_stats(): Permutation-based alignment probability
- position_contrast(): Statistical comparison between position types
- phi_vs_null(): Test φⁿ model against null hypotheses

Dependencies: numpy, pandas, scipy, phi_frequency_model
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from scipy import signal, stats
from scipy.signal import find_peaks

# Import phi model
try:
    from phi_frequency_model import (
        PHI, F0, POSITION_OFFSETS, POSITION_HIERARCHY, BANDS,
        PhiPrediction, PhiTable, generate_phi_table, get_default_phi_table,
        phi_distance, assign_position, batch_assign_positions
    )
except ImportError:
    from lib.phi_frequency_model import (
        PHI, F0, POSITION_OFFSETS, POSITION_HIERARCHY, BANDS,
        PhiPrediction, PhiTable, generate_phi_table, get_default_phi_table,
        phi_distance, assign_position, batch_assign_positions
    )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _safe_band(f_lo: float, f_hi: float, fs: float, pad_frac: float = 0.02) -> Tuple[float, float]:
    """Clamp frequency band to safe range for bandpass filter."""
    nyq = 0.5 * fs
    pad = pad_frac * nyq
    lo = max(pad, min(f_lo, nyq - 2 * pad))
    hi = max(lo + pad, min(f_hi, nyq - pad))
    return lo, hi


def bandpass_safe(x: np.ndarray, fs: float, f1: float, f2: float, order: int = 4) -> np.ndarray:
    """Apply bandpass filter with safe frequency clamping."""
    f1, f2 = _safe_band(f1, f2, fs)
    ny = 0.5 * fs
    b, a = signal.butter(order, [f1 / ny, f2 / ny], btype='band')
    return signal.filtfilt(b, a, x, axis=-1)


# ============================================================================
# CORE GED FUNCTIONS
# ============================================================================

def ged_weights(
    X: np.ndarray,
    fs: float,
    f0: float,
    bw: float = 0.5,
    flank_bw: float = 1.0,
    regularize: float = 1e-6
) -> Tuple[np.ndarray, float, float, np.ndarray]:
    """
    Generalized Eigenvalue Decomposition for narrowband signal extraction.

    Solves: Cs @ w = λ * Cn @ w
    Where Cs = covariance of signal band, Cn = covariance of flanking bands.

    Parameters
    ----------
    X : np.ndarray
        EEG data matrix, shape (n_channels, n_samples)
    fs : float
        Sampling rate (Hz)
    f0 : float
        Center frequency (Hz)
    bw : float
        Signal bandwidth half-width (Hz), default 0.5
    flank_bw : float
        Flanking noise bandwidth (Hz), default 1.0
    regularize : float
        Regularization for noise covariance matrix

    Returns
    -------
    w : np.ndarray
        Spatial filter weights (normalized)
    lambda_max : float
        Maximum eigenvalue (SNR-like metric)
    lambda_ratio : float
        Ratio of max to 2nd eigenvalue (specificity)
    eigenvalues : np.ndarray
        All eigenvalues (sorted descending)
    """
    # Filter signal band
    Bs = bandpass_safe(X, fs, f0 - bw, f0 + bw)

    # Filter flanking noise bands
    N1 = bandpass_safe(X, fs, max(0.1, f0 - bw - flank_bw), f0 - bw)
    N2 = bandpass_safe(X, fs, f0 + bw, f0 + bw + flank_bw)

    # Compute covariance matrices
    Cs = np.cov(Bs)
    Cn = np.cov(np.hstack([N1, N2]))

    # Handle 1D case
    if Cs.ndim == 0:
        Cs = np.array([[Cs]])
    if Cn.ndim == 0:
        Cn = np.array([[Cn]])

    # Regularize noise covariance
    Cn = Cn + regularize * np.trace(Cn) / Cn.shape[0] * np.eye(Cn.shape[0])

    # Solve GED: Cs @ w = λ * Cn @ w => inv(Cn) @ Cs @ w = λ * w
    try:
        eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(Cn) @ Cs)
    except np.linalg.LinAlgError:
        # Fallback if eigendecomposition fails
        return np.ones(X.shape[0]) / X.shape[0], 0.0, 1.0, np.array([0.0])

    # Sort by eigenvalue (descending)
    eigvals = np.real(eigvals)
    idx_sorted = np.argsort(eigvals)[::-1]
    eigvals_sorted = eigvals[idx_sorted]

    # Extract top filter
    lambda_max = eigvals_sorted[0]
    lambda_2nd = eigvals_sorted[1] if len(eigvals_sorted) > 1 else 1e-12
    lambda_ratio = lambda_max / (abs(lambda_2nd) + 1e-12)

    w = np.real(eigvecs[:, idx_sorted[0]])
    w = w / (np.linalg.norm(w) + 1e-12)

    return w, float(lambda_max), float(lambda_ratio), eigvals_sorted


# ============================================================================
# PHI-AWARE GED SWEEP FUNCTIONS
# ============================================================================

def ged_sweep_phi(
    X: np.ndarray,
    fs: float,
    prediction: PhiPrediction,
    search_range: Optional[float] = None,
    n_steps: int = 41,
    flank_bw: float = 1.0
) -> Dict[str, Any]:
    """
    GED sweep around a phi-predicted frequency with position-aware parameters.

    Adaptive search range based on position type:
    - Boundaries: Wider search (may have more variability)
    - Attractors: Tighter search (should be more stable)
    - Nobles: Medium search

    Parameters
    ----------
    X : np.ndarray
        EEG data matrix, shape (n_channels, n_samples)
    fs : float
        Sampling rate (Hz)
    prediction : PhiPrediction
        Phi prediction with frequency, position type, bandwidth
    search_range : float or None
        Override automatic search range (Hz)
    n_steps : int
        Number of frequencies to test
    flank_bw : float
        Flanking noise bandwidth (Hz)

    Returns
    -------
    dict with keys:
        optimal_freq, canonical_freq, freq_deviation, freq_deviation_pct,
        fwhm, q_factor, peak_eigenvalue, lambda_ratio,
        position_type, label, band, success
    """
    f_center = prediction.frequency
    bw = prediction.search_bw

    # Adaptive search range based on position type
    if search_range is None:
        if prediction.position_type == 'attractor':
            search_range = 1.2 * bw  # Tighter for attractors
        elif prediction.position_type == 'boundary':
            search_range = 2.0 * bw  # Wider for boundaries
        elif prediction.position_type.startswith('noble'):
            search_range = 1.5 * bw  # Medium for nobles
        else:
            search_range = 1.5 * bw

    # Generate frequency grid
    freqs = np.linspace(f_center - search_range, f_center + search_range, n_steps)

    # Filter out frequencies too close to boundaries
    min_freq = bw + flank_bw + 0.5
    max_freq = fs / 2 - bw - flank_bw - 0.5
    freqs = freqs[(freqs >= min_freq) & (freqs <= max_freq)]

    if len(freqs) == 0:
        return {
            'optimal_freq': f_center,
            'canonical_freq': f_center,
            'freq_deviation': 0.0,
            'freq_deviation_pct': 0.0,
            'fwhm': 2 * bw,
            'q_factor': f_center / (2 * bw),
            'peak_eigenvalue': 0.0,
            'lambda_ratio': 1.0,
            'position_type': prediction.position_type,
            'label': prediction.label,
            'band': prediction.band,
            'n': prediction.n,
            'success': False
        }

    # Sweep frequencies
    eigenvalues = []
    lambda_ratios = []

    for f in freqs:
        try:
            _, lam, lam_ratio, _ = ged_weights(X, fs, f, bw=bw, flank_bw=flank_bw)
            eigenvalues.append(lam)
            lambda_ratios.append(lam_ratio)
        except Exception:
            eigenvalues.append(np.nan)
            lambda_ratios.append(np.nan)

    eigenvalues = np.array(eigenvalues)
    lambda_ratios = np.array(lambda_ratios)
    valid = np.isfinite(eigenvalues) & (eigenvalues > 0)

    if not np.any(valid):
        return {
            'optimal_freq': f_center,
            'canonical_freq': f_center,
            'freq_deviation': 0.0,
            'freq_deviation_pct': 0.0,
            'fwhm': 2 * bw,
            'q_factor': f_center / (2 * bw),
            'peak_eigenvalue': 0.0,
            'lambda_ratio': 1.0,
            'position_type': prediction.position_type,
            'label': prediction.label,
            'band': prediction.band,
            'n': prediction.n,
            'eigenvalue_curve': eigenvalues,
            'frequencies': freqs,
            'success': False
        }

    # Find optimal (peak eigenvalue)
    idx_opt = np.nanargmax(eigenvalues)
    f_opt = freqs[idx_opt]
    lam_max = eigenvalues[idx_opt]
    lam_ratio_opt = lambda_ratios[idx_opt]

    # Compute FWHM
    fwhm, fwhm_low, fwhm_high = _compute_fwhm(freqs, eigenvalues, idx_opt, bw, search_range)
    q_factor = f_opt / (fwhm + 1e-6)

    # Deviation from prediction
    freq_deviation = f_opt - f_center
    freq_deviation_pct = 100 * freq_deviation / f_center

    return {
        'optimal_freq': float(f_opt),
        'canonical_freq': float(f_center),
        'freq_deviation': float(freq_deviation),
        'freq_deviation_pct': float(freq_deviation_pct),
        'fwhm': float(fwhm),
        'fwhm_bounds': (float(fwhm_low), float(fwhm_high)),
        'q_factor': float(q_factor),
        'peak_eigenvalue': float(lam_max),
        'lambda_ratio': float(lam_ratio_opt) if np.isfinite(lam_ratio_opt) else 1.0,
        'position_type': prediction.position_type,
        'label': prediction.label,
        'band': prediction.band,
        'n': prediction.n,
        'eigenvalue_curve': eigenvalues,
        'frequencies': freqs,
        'success': True
    }


def _compute_fwhm(
    freqs: np.ndarray,
    eigenvalues: np.ndarray,
    idx_opt: int,
    bw: float,
    search_range: float
) -> Tuple[float, float, float]:
    """Compute FWHM from eigenvalue curve."""
    lam_max = eigenvalues[idx_opt]
    eig_min = np.nanmin(eigenvalues)
    eig_range = lam_max - eig_min

    if eig_range < 1e-12:
        # Flat profile
        f_opt = freqs[idx_opt]
        return search_range * 2, f_opt - search_range, f_opt + search_range

    # Normalize eigenvalues
    eig_norm = (eigenvalues - eig_min) / eig_range
    half_level = 0.5
    above_half = eig_norm >= half_level

    fwhm_low = freqs[0]
    fwhm_high = freqs[-1]

    # Find left crossing
    for i in range(idx_opt, 0, -1):
        if not above_half[i - 1] and above_half[i]:
            if eig_norm[i] != eig_norm[i - 1]:
                frac = (half_level - eig_norm[i - 1]) / (eig_norm[i] - eig_norm[i - 1])
                fwhm_low = freqs[i - 1] + frac * (freqs[i] - freqs[i - 1])
            else:
                fwhm_low = freqs[i]
            break

    # Find right crossing
    for i in range(idx_opt, len(freqs) - 1):
        if above_half[i] and not above_half[i + 1]:
            if eig_norm[i] != eig_norm[i + 1]:
                frac = (half_level - eig_norm[i]) / (eig_norm[i + 1] - eig_norm[i])
                fwhm_high = freqs[i] + frac * (freqs[i + 1] - freqs[i])
            else:
                fwhm_high = freqs[i]
            break

    fwhm = fwhm_high - fwhm_low
    return fwhm, fwhm_low, fwhm_high


def ged_sweep_all_positions(
    X: np.ndarray,
    fs: float,
    phi_table: PhiTable,
    flank_bw: float = 1.0,
    n_steps: int = 41
) -> pd.DataFrame:
    """
    Run GED sweep for all predictions in a phi table.

    Parameters
    ----------
    X : np.ndarray
        EEG data matrix, shape (n_channels, n_samples)
    fs : float
        Sampling rate (Hz)
    phi_table : PhiTable
        Table of phi predictions
    flank_bw : float
        Flanking noise bandwidth
    n_steps : int
        Number of frequencies per sweep

    Returns
    -------
    DataFrame with one row per prediction, columns for GED metrics
    """
    rows = []

    for label, pred in phi_table.items():
        result = ged_sweep_phi(X, fs, pred, flank_bw=flank_bw, n_steps=n_steps)
        rows.append(result)

    return pd.DataFrame(rows)


# ============================================================================
# BLIND GED SWEEP WITH POSITION ASSIGNMENT
# ============================================================================

def ged_blind_with_positions(
    X: np.ndarray,
    fs: float,
    phi_table: Optional[PhiTable] = None,
    freq_range: Tuple[float, float] = (3.0, 45.0),
    step_hz: float = 0.2,
    bw: float = 0.5,
    flank_bw: float = 1.0,
    n_peaks: int = 12,
    min_peak_distance_hz: float = 1.5,
    alignment_tol: float = 0.05
) -> Dict[str, Any]:
    """
    Blind GED sweep with automatic position assignment.

    Discovers peaks without knowing φⁿ predictions, then assigns
    each peak to its nearest predicted position.

    Parameters
    ----------
    X : np.ndarray
        EEG data matrix, shape (n_channels, n_samples)
    fs : float
        Sampling rate (Hz)
    phi_table : PhiTable or None
        Table of predictions (default: get_default_phi_table())
    freq_range : Tuple[float, float]
        Frequency range to search (Hz)
    step_hz : float
        Frequency step size (Hz)
    bw : float
        Signal bandwidth half-width (Hz)
    flank_bw : float
        Flanking noise bandwidth (Hz)
    n_peaks : int
        Number of top peaks to return
    min_peak_distance_hz : float
        Minimum distance between detected peaks (Hz)
    alignment_tol : float
        Relative tolerance for "aligned" classification (default 5%)

    Returns
    -------
    dict with keys:
        frequencies, eigenvalues, peaks, position_histogram,
        alignment_fraction, mean_alignment_distance, success
    """
    if phi_table is None:
        phi_table = get_default_phi_table()

    f_min, f_max = freq_range

    # Clamp to safe range
    min_safe = bw + flank_bw + 0.5
    max_safe = fs / 2 - bw - flank_bw - 0.5
    f_min = max(f_min, min_safe)
    f_max = min(f_max, max_safe)

    if f_min >= f_max:
        return {
            'frequencies': np.array([]),
            'eigenvalues': np.array([]),
            'peaks': [],
            'position_histogram': {},
            'alignment_fraction': 0.0,
            'mean_alignment_distance': 1.0,
            'n_peaks_found': 0,
            'success': False
        }

    # Generate frequency grid
    freqs = np.arange(f_min, f_max + step_hz, step_hz)

    # Sweep all frequencies
    eigenvalues = []
    lambda_ratios = []

    for f in freqs:
        try:
            _, lam, lam_ratio, _ = ged_weights(X, fs, f, bw=bw, flank_bw=flank_bw)
            eigenvalues.append(lam)
            lambda_ratios.append(lam_ratio)
        except Exception:
            eigenvalues.append(np.nan)
            lambda_ratios.append(np.nan)

    eigenvalues = np.array(eigenvalues)
    lambda_ratios = np.array(lambda_ratios)

    # Find local maxima (peaks)
    valid = np.isfinite(eigenvalues) & (eigenvalues > 0)
    if not np.any(valid):
        return {
            'frequencies': freqs,
            'eigenvalues': eigenvalues,
            'peaks': [],
            'position_histogram': {},
            'alignment_fraction': 0.0,
            'mean_alignment_distance': 1.0,
            'n_peaks_found': 0,
            'success': False
        }

    # Detect peaks using scipy
    peak_indices, properties = find_peaks(
        eigenvalues,
        distance=int(min_peak_distance_hz / step_hz),
        prominence=0.1 * np.nanmax(eigenvalues)
    )

    if len(peak_indices) == 0:
        # Fallback: just find global maximum
        idx_max = np.nanargmax(eigenvalues)
        peak_indices = np.array([idx_max])

    # Sort peaks by eigenvalue (descending)
    sorted_idx = np.argsort(eigenvalues[peak_indices])[::-1]
    peak_indices = peak_indices[sorted_idx]

    # Extract top N peaks with position assignment
    peaks = []
    position_counts = {pos: 0 for pos in POSITION_OFFSETS.keys()}
    aligned_count = 0
    distances = []

    for i, idx in enumerate(peak_indices[:n_peaks]):
        f_peak = freqs[idx]
        eig_peak = eigenvalues[idx]
        lam_ratio = lambda_ratios[idx] if np.isfinite(lambda_ratios[idx]) else 1.0

        # Compute local FWHM
        fwhm = _compute_local_fwhm(freqs, eigenvalues, idx, step_hz, bw)
        q_factor = f_peak / (fwhm + 1e-6)

        # Assign to nearest phi position
        nearest_pred, rel_dist, is_aligned = assign_position(f_peak, phi_table, alignment_tol)

        if nearest_pred is not None:
            position_counts[nearest_pred.position_type] += 1
            if is_aligned:
                aligned_count += 1
            distances.append(rel_dist)

            peaks.append({
                'rank': i + 1,
                'frequency': float(f_peak),
                'eigenvalue': float(eig_peak),
                'lambda_ratio': float(lam_ratio),
                'fwhm': float(fwhm),
                'q_factor': float(q_factor),
                'nearest_label': nearest_pred.label,
                'nearest_freq': nearest_pred.frequency,
                'position_type': nearest_pred.position_type,
                'rel_distance': float(rel_dist),
                'is_aligned': is_aligned,
                'n': nearest_pred.n,
                'band': nearest_pred.band,
            })
        else:
            # No position within tolerance - find nearest anyway for info
            nearest = phi_table.nearest(f_peak)
            rel_dist = abs(f_peak - nearest.frequency) / nearest.frequency
            distances.append(rel_dist)

            peaks.append({
                'rank': i + 1,
                'frequency': float(f_peak),
                'eigenvalue': float(eig_peak),
                'lambda_ratio': float(lam_ratio),
                'fwhm': float(fwhm),
                'q_factor': float(q_factor),
                'nearest_label': nearest.label,
                'nearest_freq': nearest.frequency,
                'position_type': nearest.position_type,
                'rel_distance': float(rel_dist),
                'is_aligned': False,
                'n': nearest.n,
                'band': nearest.band,
            })

    alignment_fraction = aligned_count / len(peaks) if peaks else 0.0
    mean_distance = np.mean(distances) if distances else 1.0

    return {
        'frequencies': freqs,
        'eigenvalues': eigenvalues,
        'peaks': peaks,
        'position_histogram': position_counts,
        'alignment_fraction': float(alignment_fraction),
        'mean_alignment_distance': float(mean_distance),
        'n_peaks_found': len(peaks),
        'success': True
    }


def _compute_local_fwhm(
    freqs: np.ndarray,
    eigenvalues: np.ndarray,
    idx: int,
    step_hz: float,
    default_bw: float
) -> float:
    """Compute FWHM around a single peak."""
    eig_peak = eigenvalues[idx]
    half_max = eig_peak / 2

    fwhm_low = freqs[idx] - default_bw
    fwhm_high = freqs[idx] + default_bw

    # Search left for half-max crossing
    for j in range(idx, 0, -1):
        if eigenvalues[j] < half_max:
            if eigenvalues[j] != eigenvalues[j + 1]:
                frac = (half_max - eigenvalues[j]) / (eigenvalues[j + 1] - eigenvalues[j])
                fwhm_low = freqs[j] + frac * step_hz
            else:
                fwhm_low = freqs[j]
            break

    # Search right for half-max crossing
    for j in range(idx, len(eigenvalues) - 1):
        if eigenvalues[j] < half_max:
            if eigenvalues[j] != eigenvalues[j - 1]:
                frac = (half_max - eigenvalues[j]) / (eigenvalues[j - 1] - eigenvalues[j])
                fwhm_high = freqs[j] - frac * step_hz
            else:
                fwhm_high = freqs[j]
            break

    return max(0.1, fwhm_high - fwhm_low)


# ============================================================================
# ALIGNMENT STATISTICS AND PERMUTATION TESTS
# ============================================================================

def compute_alignment_stats(
    discovered_peaks: List[Dict],
    phi_table: PhiTable,
    n_permutations: int = 1000,
    alignment_tol: float = 0.05
) -> Dict[str, Any]:
    """
    Compute alignment statistics with permutation-based p-values.

    Tests whether discovered peaks align with phi positions more than
    expected by chance (uniform distribution in frequency space).

    Parameters
    ----------
    discovered_peaks : List[Dict]
        Peaks from ged_blind_with_positions()
    phi_table : PhiTable
        Table of predictions
    n_permutations : int
        Number of permutations for null distribution
    alignment_tol : float
        Relative tolerance for alignment

    Returns
    -------
    dict with:
        observed_alignment, p_value, position_type_pvalues,
        null_distribution, effect_size
    """
    if not discovered_peaks:
        return {
            'observed_alignment': 0.0,
            'p_value': 1.0,
            'position_type_pvalues': {},
            'null_distribution': [],
            'effect_size': 0.0
        }

    # Observed alignment
    observed_freqs = np.array([p['frequency'] for p in discovered_peaks])
    observed_aligned = sum(1 for p in discovered_peaks if p.get('is_aligned', False))
    observed_alignment = observed_aligned / len(discovered_peaks)

    # Generate null distribution via permutation
    freq_range = (observed_freqs.min(), observed_freqs.max())
    null_alignments = []

    for _ in range(n_permutations):
        # Generate random frequencies uniformly in log-space
        log_min, log_max = np.log(freq_range[0]), np.log(freq_range[1])
        random_log_freqs = np.random.uniform(log_min, log_max, len(observed_freqs))
        random_freqs = np.exp(random_log_freqs)

        # Count alignments
        aligned = 0
        for f in random_freqs:
            nearest = phi_table.nearest(f)
            rel_dist = abs(f - nearest.frequency) / nearest.frequency
            if rel_dist <= alignment_tol:
                aligned += 1
        null_alignments.append(aligned / len(random_freqs))

    null_alignments = np.array(null_alignments)

    # P-value: fraction of null >= observed
    p_value = (np.sum(null_alignments >= observed_alignment) + 1) / (n_permutations + 1)

    # Effect size: (observed - null_mean) / null_std
    null_mean = np.mean(null_alignments)
    null_std = np.std(null_alignments) + 1e-6
    effect_size = (observed_alignment - null_mean) / null_std

    # Per-position-type p-values
    position_type_pvalues = {}
    for pos_type in POSITION_OFFSETS.keys():
        observed_count = sum(1 for p in discovered_peaks
                            if p.get('position_type') == pos_type and p.get('is_aligned', False))
        if observed_count > 0:
            # Bootstrap test for this position type
            null_counts = []
            for _ in range(n_permutations):
                random_log_freqs = np.random.uniform(log_min, log_max, len(observed_freqs))
                random_freqs = np.exp(random_log_freqs)
                count = 0
                for f in random_freqs:
                    nearest = phi_table.nearest(f)
                    if nearest.position_type == pos_type:
                        rel_dist = abs(f - nearest.frequency) / nearest.frequency
                        if rel_dist <= alignment_tol:
                            count += 1
                null_counts.append(count)
            position_type_pvalues[pos_type] = (np.sum(np.array(null_counts) >= observed_count) + 1) / (n_permutations + 1)

    return {
        'observed_alignment': float(observed_alignment),
        'p_value': float(p_value),
        'position_type_pvalues': position_type_pvalues,
        'null_distribution': null_alignments.tolist(),
        'null_mean': float(null_mean),
        'null_std': float(null_std),
        'effect_size': float(effect_size)
    }


# ============================================================================
# POSITION TYPE CONTRAST ANALYSIS
# ============================================================================

def position_contrast(
    results_df: pd.DataFrame,
    group_a: List[str],
    group_b: List[str],
    metrics: List[str] = ['fwhm', 'q_factor', 'peak_eigenvalue']
) -> Dict[str, Any]:
    """
    Statistical comparison of GED metrics between position type groups.

    Parameters
    ----------
    results_df : DataFrame
        Results from ged_sweep_all_positions()
    group_a : List[str]
        Position types for group A (e.g., ['boundary'])
    group_b : List[str]
        Position types for group B (e.g., ['attractor'])
    metrics : List[str]
        Metrics to compare

    Returns
    -------
    dict with comparison statistics for each metric
    """
    # Filter by position type
    mask_a = results_df['position_type'].isin(group_a)
    mask_b = results_df['position_type'].isin(group_b)

    df_a = results_df[mask_a]
    df_b = results_df[mask_b]

    comparisons = {}

    for metric in metrics:
        if metric not in results_df.columns:
            continue

        vals_a = df_a[metric].dropna().values
        vals_b = df_b[metric].dropna().values

        if len(vals_a) < 2 or len(vals_b) < 2:
            continue

        # T-test
        t_stat, t_pval = stats.ttest_ind(vals_a, vals_b, equal_var=False)

        # Mann-Whitney U
        u_stat, u_pval = stats.mannwhitneyu(vals_a, vals_b, alternative='two-sided')

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(vals_a) + np.var(vals_b)) / 2)
        cohens_d = (np.mean(vals_a) - np.mean(vals_b)) / (pooled_std + 1e-6)

        comparisons[metric] = {
            'group_a_mean': float(np.mean(vals_a)),
            'group_a_std': float(np.std(vals_a)),
            'group_a_n': len(vals_a),
            'group_b_mean': float(np.mean(vals_b)),
            'group_b_std': float(np.std(vals_b)),
            'group_b_n': len(vals_b),
            't_statistic': float(t_stat),
            't_pvalue': float(t_pval),
            'u_statistic': float(u_stat),
            'u_pvalue': float(u_pval),
            'cohens_d': float(cohens_d),
            'direction': 'A > B' if np.mean(vals_a) > np.mean(vals_b) else 'B > A'
        }

    return {
        'group_a': group_a,
        'group_b': group_b,
        'comparisons': comparisons
    }


def noble_hierarchy_test(
    results_df: pd.DataFrame,
    metric: str = 'peak_eigenvalue'
) -> Dict[str, Any]:
    """
    Test if noble hierarchy holds (1° > 2° > 3° > 4°).

    Parameters
    ----------
    results_df : DataFrame
        Results from ged_sweep_all_positions()
    metric : str
        Metric to compare (higher = better)

    Returns
    -------
    dict with hierarchy test results
    """
    noble_order = ['noble_1', 'noble_2', 'noble_3', 'noble_4']
    means = {}
    ns = {}

    for noble in noble_order:
        mask = results_df['position_type'] == noble
        vals = results_df.loc[mask, metric].dropna()
        means[noble] = float(np.mean(vals)) if len(vals) > 0 else np.nan
        ns[noble] = len(vals)

    # Check if hierarchy holds
    hierarchy_holds = True
    for i in range(len(noble_order) - 1):
        if np.isnan(means[noble_order[i]]) or np.isnan(means[noble_order[i + 1]]):
            continue
        if means[noble_order[i]] <= means[noble_order[i + 1]]:
            hierarchy_holds = False
            break

    # Correlation with noble rank
    ranks = []
    values = []
    for i, noble in enumerate(noble_order):
        if not np.isnan(means[noble]):
            ranks.append(i + 1)
            values.append(means[noble])

    correlation = float(np.corrcoef(ranks, values)[0, 1]) if len(ranks) > 2 else np.nan

    return {
        'means': means,
        'n_per_type': ns,
        'hierarchy_holds': hierarchy_holds,
        'rank_correlation': correlation,
        'metric': metric
    }


# ============================================================================
# NULL HYPOTHESIS TESTS
# ============================================================================

def phi_vs_null(
    blind_peaks: List[Dict],
    phi_table: PhiTable,
    null_model: str = 'uniform',
    n_permutations: int = 1000
) -> Dict[str, Any]:
    """
    Test phi model against alternative null hypotheses.

    Null models:
    - 'uniform': Peaks uniformly distributed in log-frequency space
    - 'harmonic': Peaks follow integer harmonic series (n * f0)
    - 'octave': Peaks follow power-of-2 octaves

    Parameters
    ----------
    blind_peaks : List[Dict]
        Discovered peaks from blind sweep
    phi_table : PhiTable
        Phi predictions
    null_model : str
        Null model type
    n_permutations : int
        Number of permutations

    Returns
    -------
    dict with model comparison statistics
    """
    if not blind_peaks:
        return {'error': 'No peaks provided'}

    observed_freqs = np.array([p['frequency'] for p in blind_peaks])

    # Compute phi-model fit (mean relative distance)
    phi_distances = []
    for f in observed_freqs:
        nearest = phi_table.nearest(f)
        rel_dist = abs(f - nearest.frequency) / nearest.frequency
        phi_distances.append(rel_dist)
    phi_fit = np.mean(phi_distances)

    # Generate null model predictions and compute fit
    if null_model == 'uniform':
        # Uniform: no structure, compare to random distribution
        null_fits = []
        freq_range = (observed_freqs.min(), observed_freqs.max())
        log_min, log_max = np.log(freq_range[0]), np.log(freq_range[1])

        for _ in range(n_permutations):
            random_freqs = np.exp(np.random.uniform(log_min, log_max, len(observed_freqs)))
            dists = []
            for f in random_freqs:
                nearest = phi_table.nearest(f)
                rel_dist = abs(f - nearest.frequency) / nearest.frequency
                dists.append(rel_dist)
            null_fits.append(np.mean(dists))

    elif null_model == 'harmonic':
        # Harmonic: integer multiples of f0
        harmonic_freqs = np.array([F0 * n for n in range(1, 8)])  # 7.6, 15.2, 22.8, ...
        harmonic_distances = []
        for f in observed_freqs:
            min_dist = min(abs(f - hf) / hf for hf in harmonic_freqs if hf > 0)
            harmonic_distances.append(min_dist)
        harmonic_fit = np.mean(harmonic_distances)

        # Null: random comparison
        null_fits = []
        for _ in range(n_permutations):
            random_freqs = np.random.choice(harmonic_freqs, len(observed_freqs), replace=True)
            random_freqs *= np.random.uniform(0.9, 1.1, len(random_freqs))  # Add noise
            dists = []
            for f in random_freqs:
                min_dist = min(abs(f - hf) / hf for hf in harmonic_freqs if hf > 0)
                dists.append(min_dist)
            null_fits.append(np.mean(dists))

    elif null_model == 'octave':
        # Octave: power-of-2 structure
        octave_freqs = np.array([F0 * (2 ** n) for n in range(-2, 4)])  # 1.9, 3.8, 7.6, 15.2, ...
        octave_distances = []
        for f in observed_freqs:
            min_dist = min(abs(f - of) / of for of in octave_freqs if of > 0)
            octave_distances.append(min_dist)
        octave_fit = np.mean(octave_distances)

        null_fits = []
        for _ in range(n_permutations):
            random_freqs = np.random.choice(octave_freqs, len(observed_freqs), replace=True)
            random_freqs *= np.random.uniform(0.9, 1.1, len(random_freqs))
            dists = []
            for f in random_freqs:
                min_dist = min(abs(f - of) / of for of in octave_freqs if of > 0)
                dists.append(min_dist)
            null_fits.append(np.mean(dists))
    else:
        return {'error': f'Unknown null model: {null_model}'}

    null_fits = np.array(null_fits)

    # P-value: fraction of null <= observed (lower is better fit)
    p_value = (np.sum(null_fits <= phi_fit) + 1) / (n_permutations + 1)

    # Likelihood ratio approximation
    null_mean = np.mean(null_fits)
    likelihood_ratio = null_mean / (phi_fit + 1e-6)

    return {
        'phi_fit': float(phi_fit),
        'null_model': null_model,
        'null_mean': float(null_mean),
        'null_std': float(np.std(null_fits)),
        'p_value': float(p_value),
        'likelihood_ratio': float(likelihood_ratio),
        'phi_better': phi_fit < null_mean
    }


# ============================================================================
# HELPER FUNCTIONS FOR PIPELINE INTEGRATION
# ============================================================================

def extract_eeg_matrix(
    records: pd.DataFrame,
    eeg_channels: List[str],
    time_col: str = 'Timestamp'
) -> np.ndarray:
    """
    Extract EEG data matrix from records DataFrame.

    Parameters
    ----------
    records : DataFrame
        EEG records with Timestamp and EEG channels
    eeg_channels : List[str]
        Channel names (with or without 'EEG.' prefix)
    time_col : str
        Timestamp column name

    Returns
    -------
    X : np.ndarray
        EEG matrix, shape (n_channels, n_samples)
    """
    # Normalize channel names
    cols = []
    for ch in eeg_channels:
        candidates = [ch, f'EEG.{ch}', ch.replace('EEG.', '')]
        for c in candidates:
            if c in records.columns:
                cols.append(c)
                break

    X = records[cols].values.T  # (n_channels, n_samples)
    return X


def infer_fs_from_records(records: pd.DataFrame, time_col: str = 'Timestamp') -> float:
    """Infer sampling rate from timestamp column."""
    if time_col not in records.columns:
        return 128.0  # Default

    t = records[time_col].values
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return 128.0

    return 1.0 / np.median(dt)


if __name__ == "__main__":
    # Quick test with synthetic data
    print("Testing GED Phi Analysis module...")

    # Generate synthetic multi-channel data
    np.random.seed(42)
    fs = 128
    duration = 60  # seconds
    n_channels = 14
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs

    # Create data with embedded oscillations at phi positions
    X = np.random.randn(n_channels, n_samples) * 0.5

    # Add oscillations at predicted frequencies
    phi_table = get_default_phi_table()
    for label, pred in list(phi_table.items())[:5]:
        amp = 2.0 if pred.position_type == 'attractor' else 1.0
        for ch in range(n_channels):
            phase = np.random.uniform(0, 2 * np.pi)
            X[ch] += amp * np.sin(2 * np.pi * pred.frequency * t + phase)

    print(f"Synthetic data: {n_channels} channels, {n_samples} samples, {fs} Hz")

    # Test blind sweep
    print("\nRunning blind sweep...")
    blind_result = ged_blind_with_positions(X, fs, phi_table)
    print(f"Found {blind_result['n_peaks_found']} peaks")
    print(f"Alignment fraction: {blind_result['alignment_fraction']:.2%}")

    # Test position sweep
    print("\nRunning position sweep...")
    sweep_df = ged_sweep_all_positions(X, fs, phi_table)
    print(f"Swept {len(sweep_df)} positions")
    print(f"Mean deviation: {sweep_df['freq_deviation_pct'].abs().mean():.2f}%")

    # Test contrast
    print("\nTesting boundary vs attractor contrast...")
    contrast = position_contrast(sweep_df, ['boundary'], ['attractor'])
    if 'fwhm' in contrast['comparisons']:
        fwhm_stats = contrast['comparisons']['fwhm']
        print(f"FWHM: boundaries={fwhm_stats['group_a_mean']:.2f}, "
              f"attractors={fwhm_stats['group_b_mean']:.2f}, "
              f"p={fwhm_stats['t_pvalue']:.4f}")

    print("\nAll tests passed!")

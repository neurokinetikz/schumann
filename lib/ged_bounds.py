"""
GED Bounds Analysis: Attractor vs Boundary Framework
=====================================================

Implements GED (Generalized Eigendecomposition) frequency optimization to test
the attractor-boundary framework for Schumann Ignition Event frequencies.

Core Hypothesis:
- Boundaries (integer φⁿ positions: SR1, SR2, SR3, SR5) = band edges, broader GED peaks
- Attractors (half-integer φⁿ positions: SR1.5, SR2.5, SR4, SR6) = stable points, sharper GED peaks

Primary Analysis: GED-derived bandwidth differences between attractors vs boundaries
Secondary Analysis: Noble number validation of inter-harmonic ratios

Dependencies: numpy, pandas, scipy, matplotlib
"""

from __future__ import annotations
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from scipy import signal, stats
from pathlib import Path

# ============================================================================
# CONSTANTS - Now imported from phi_frequency_model
# ============================================================================

# Import the complete phi model
try:
    from phi_frequency_model import (
        PHI, F0, POSITION_OFFSETS, POSITION_OFFSETS_EXACT,
        POSITION_HIERARCHY, BANDS, BAND_SEARCH_BW,
        PhiPrediction, PhiTable, generate_phi_table, get_default_phi_table,
        phi_distance as _phi_distance_new, assign_position, batch_assign_positions,
        EXPECTED_RATIOS as PHI_EXPECTED_RATIOS
    )
except ImportError:
    from lib.phi_frequency_model import (
        PHI, F0, POSITION_OFFSETS, POSITION_OFFSETS_EXACT,
        POSITION_HIERARCHY, BANDS, BAND_SEARCH_BW,
        PhiPrediction, PhiTable, generate_phi_table, get_default_phi_table,
        phi_distance as _phi_distance_new, assign_position, batch_assign_positions,
        EXPECTED_RATIOS as PHI_EXPECTED_RATIOS
    )

# Derived constant
PHI_INV = 1.0 / PHI  # 0.6180339887

# Generate SR_HARMONIC_TABLE from new phi model for backward compatibility
# This table now includes ALL position types, not just boundaries and attractors
_phi_table = get_default_phi_table()

SR_HARMONIC_TABLE = {}
for label, pred in _phi_table.items():
    SR_HARMONIC_TABLE[label] = {
        'n': pred.n,
        'freq': pred.frequency,
        'search_center': pred.frequency,  # Now same as freq (no separate empirical value)
        'type': pred.position_type,
        'band': pred.band,
        'half_bw': pred.search_bw,
    }

# Legacy aliases for backward compatibility with old label scheme
_LEGACY_MAPPING = {
    'sr1': 'alpha_boundary',      # n=0, 7.60 Hz
    'sr1.5': 'alpha_attractor',   # n=0.5, 9.67 Hz
    'sr2': 'beta_low_boundary',   # n=1, 12.30 Hz
    'sr2.5': 'beta_low_attractor', # n=1.5, 15.64 Hz
    'sr3': 'beta_high_boundary',  # n=2, 19.90 Hz
    'sr4': 'beta_high_attractor', # n=2.5, 25.31 Hz
    'sr5': 'gamma_boundary',      # n=3, 32.19 Hz
    'sr6': 'gamma_attractor',     # n=3.5, 40.95 Hz
}

for old_label, new_label in _LEGACY_MAPPING.items():
    if new_label in SR_HARMONIC_TABLE and old_label not in SR_HARMONIC_TABLE:
        SR_HARMONIC_TABLE[old_label] = SR_HARMONIC_TABLE[new_label].copy()

# Expected noble ratios (powers of φ) - expanded from new model
EXPECTED_RATIOS = {
    # Adjacent octave boundaries (φ¹ ratios)
    ('beta_low_boundary', 'alpha_boundary'): PHI ** 1,
    ('beta_high_boundary', 'beta_low_boundary'): PHI ** 1,
    ('gamma_boundary', 'beta_high_boundary'): PHI ** 1,
    # Legacy aliases
    ('sr3', 'sr1'): PHI ** 2,    # 2.618
    ('sr5', 'sr1'): PHI ** 3,    # 4.236
    ('sr5', 'sr3'): PHI ** 1,    # 1.618
    ('sr2', 'sr1'): PHI ** 1,    # 1.618
    ('sr4', 'sr2'): PHI ** 1,    # 1.618
}

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
# CORE GED FUNCTIONS (PRIMARY)
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


def ged_frequency_sweep(
    X: np.ndarray,
    fs: float,
    f_center: float,
    search_range: float = 1.5,
    n_steps: int = 31,
    bw: float = 0.5,
    flank_bw: float = 1.0
) -> Dict[str, Any]:
    """
    Sweep frequencies around a canonical SR harmonic to find GED-optimal center.

    Parameters
    ----------
    X : np.ndarray
        EEG data matrix, shape (n_channels, n_samples)
    fs : float
        Sampling rate (Hz)
    f_center : float
        Initial center frequency estimate (Hz)
    search_range : float
        Range to search on each side of f_center (Hz)
    n_steps : int
        Number of frequencies to test
    bw : float
        Signal bandwidth half-width (Hz)
    flank_bw : float
        Flanking noise bandwidth (Hz)

    Returns
    -------
    dict with keys:
        optimal_freq : float
        fwhm_bounds : Tuple[float, float]
        fwhm : float
        eigenvalue_curve : np.ndarray
        frequencies : np.ndarray
        peak_eigenvalue : float
        q_factor : float
        lambda_ratio : float
        success : bool
    """
    # Generate frequency grid
    freqs = np.linspace(f_center - search_range, f_center + search_range, n_steps)

    # Filter out frequencies too close to boundaries
    min_freq = bw + flank_bw + 0.5
    max_freq = fs / 2 - bw - flank_bw - 0.5
    freqs = freqs[(freqs >= min_freq) & (freqs <= max_freq)]

    if len(freqs) == 0:
        return {
            'optimal_freq': f_center,
            'fwhm_bounds': (f_center - bw, f_center + bw),
            'fwhm': 2 * bw,
            'eigenvalue_curve': np.array([]),
            'frequencies': np.array([]),
            'peak_eigenvalue': 0.0,
            'q_factor': f_center / (2 * bw),
            'lambda_ratio': 1.0,
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
            'fwhm_bounds': (f_center - bw, f_center + bw),
            'fwhm': 2 * bw,
            'eigenvalue_curve': eigenvalues,
            'frequencies': freqs,
            'peak_eigenvalue': 0.0,
            'q_factor': f_center / (2 * bw),
            'lambda_ratio': 1.0,
            'success': False
        }

    # Find optimal (peak eigenvalue)
    idx_opt = np.nanargmax(eigenvalues)
    f_opt = freqs[idx_opt]
    lam_max = eigenvalues[idx_opt]
    lam_ratio_opt = lambda_ratios[idx_opt]

    # Compute FWHM using interpolation for better accuracy
    # First, normalize eigenvalues to [0, 1] range
    eig_min = np.nanmin(eigenvalues)
    eig_range = lam_max - eig_min
    if eig_range < 1e-12:
        # Flat profile - no meaningful peak
        fwhm = search_range * 2  # Return full search range as FWHM
        fwhm_low = f_opt - search_range
        fwhm_high = f_opt + search_range
    else:
        # Normalize eigenvalues
        eig_norm = (eigenvalues - eig_min) / eig_range

        # Find half-maximum crossing points (at 0.5 normalized)
        half_level = 0.5
        above_half = eig_norm >= half_level

        # Find FWHM bounds by interpolating crossing points
        fwhm_low = freqs[0]
        fwhm_high = freqs[-1]

        # Find left crossing
        for i in range(idx_opt, 0, -1):
            if not above_half[i - 1] and above_half[i]:
                # Interpolate
                if eig_norm[i] != eig_norm[i - 1]:
                    frac = (half_level - eig_norm[i - 1]) / (eig_norm[i] - eig_norm[i - 1])
                    fwhm_low = freqs[i - 1] + frac * (freqs[i] - freqs[i - 1])
                else:
                    fwhm_low = freqs[i]
                break

        # Find right crossing
        for i in range(idx_opt, len(freqs) - 1):
            if above_half[i] and not above_half[i + 1]:
                # Interpolate
                if eig_norm[i] != eig_norm[i + 1]:
                    frac = (half_level - eig_norm[i]) / (eig_norm[i + 1] - eig_norm[i])
                    fwhm_high = freqs[i] + frac * (freqs[i + 1] - freqs[i])
                else:
                    fwhm_high = freqs[i]
                break

    fwhm = fwhm_high - fwhm_low
    q_factor = f_opt / (fwhm + 1e-6)

    return {
        'optimal_freq': float(f_opt),
        'fwhm_bounds': (float(fwhm_low), float(fwhm_high)),
        'fwhm': float(fwhm),
        'eigenvalue_curve': eigenvalues,
        'frequencies': freqs,
        'peak_eigenvalue': float(lam_max),
        'q_factor': float(q_factor),
        'lambda_ratio': float(lam_ratio_opt) if np.isfinite(lam_ratio_opt) else 1.0,
        'success': True
    }


def ged_blind_sweep(
    X: np.ndarray,
    fs: float,
    freq_range: Tuple[float, float] = (3.0, 45.0),
    step_hz: float = 0.25,
    bw: float = 0.5,
    flank_bw: float = 1.0,
    n_peaks: int = 8,
    min_peak_distance_hz: float = 1.5
) -> Dict[str, Any]:
    """
    Blind frequency sweep - find GED-optimal peaks WITHOUT canonical seeding.

    This is the key "blind validation" function: GED discovers peaks independently
    without knowing the φⁿ predictions. If discovered peaks cluster at φⁿ positions,
    that's strong evidence the brain organizes around these frequencies intrinsically.

    Parameters
    ----------
    X : np.ndarray
        EEG data matrix, shape (n_channels, n_samples)
    fs : float
        Sampling rate (Hz)
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

    Returns
    -------
    dict with keys:
        frequencies : np.ndarray - all swept frequencies
        eigenvalues : np.ndarray - eigenvalue at each frequency
        peaks : List[Dict] - top N peaks with freq, eigenvalue, fwhm, q_factor
        n_peaks_found : int
        success : bool
    """
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
            'n_peaks_found': 0,
            'success': False
        }

    # Detect peaks using scipy
    from scipy.signal import find_peaks

    # Find peaks with minimum prominence
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

    # Extract top N peaks with FWHM
    peaks = []
    for i, idx in enumerate(peak_indices[:n_peaks]):
        f_peak = freqs[idx]
        eig_peak = eigenvalues[idx]
        lam_ratio = lambda_ratios[idx] if np.isfinite(lambda_ratios[idx]) else 1.0

        # Compute local FWHM around this peak
        half_max = eig_peak / 2
        fwhm_low = f_peak - bw
        fwhm_high = f_peak + bw

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

        fwhm = max(0.1, fwhm_high - fwhm_low)
        q_factor = f_peak / fwhm

        # Check distance to nearest φⁿ prediction
        phi_info = phi_distance(f_peak / F0)  # Normalize to f0

        peaks.append({
            'rank': i + 1,
            'frequency': float(f_peak),
            'eigenvalue': float(eig_peak),
            'lambda_ratio': float(lam_ratio),
            'fwhm': float(fwhm),
            'q_factor': float(q_factor),
            'nearest_phi_n': phi_info['nearest_n'],
            'phi_distance_rel': phi_info['relative_distance'],
            'is_phi_aligned': phi_info['is_noble']
        })

    return {
        'frequencies': freqs,
        'eigenvalues': eigenvalues,
        'peaks': peaks,
        'n_peaks_found': len(peaks),
        'success': True
    }


# ============================================================================
# ATTRACTOR-BOUNDARY ANALYSIS (PRIMARY)
# ============================================================================

def classify_harmonic_type(label: str) -> str:
    """
    Classify harmonic as 'attractor' or 'boundary' based on φⁿ exponent.

    Integer φⁿ (n=0,1,2,3,4) → boundary (SR1, SR2, SR3, SR5, SR7)
    Half-integer φⁿ (n=0.5,1.5,2.5,3.5) → attractor (SR1.5, SR2.5, SR4, SR6)
    """
    if label in SR_HARMONIC_TABLE:
        return SR_HARMONIC_TABLE[label]['type']

    # Fallback: parse label
    match = re.search(r'sr(\d+\.?\d*)', label.lower())
    if match:
        num = float(match.group(1))
        n = num - 1  # sr1→n=0, sr2→n=1, sr3→n=2, etc.
        # Half-integer check
        if abs(n - round(n)) > 0.1:  # Not close to integer
            return 'attractor'
        else:
            return 'boundary'

    return 'unknown'


def compute_bandwidth_metrics(sweep_result: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute bandwidth-related metrics from GED sweep result.

    Returns
    -------
    dict with:
        fwhm : Full width at half maximum (Hz)
        q_factor : center_freq / bandwidth (higher = sharper)
        peak_sharpness : peak_eigenvalue / fwhm
        lambda_ratio : λ₁/λ₂ specificity
    """
    fwhm = sweep_result.get('fwhm', np.nan)
    f_opt = sweep_result.get('optimal_freq', np.nan)
    lam_max = sweep_result.get('peak_eigenvalue', np.nan)
    lam_ratio = sweep_result.get('lambda_ratio', np.nan)

    q_factor = f_opt / (fwhm + 1e-6) if np.isfinite(fwhm) and fwhm > 0 else np.nan
    peak_sharpness = lam_max / (fwhm + 1e-6) if np.isfinite(fwhm) and fwhm > 0 else np.nan

    return {
        'fwhm': fwhm,
        'q_factor': q_factor,
        'peak_sharpness': peak_sharpness,
        'lambda_ratio': lam_ratio
    }


def attractor_boundary_contrast(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Statistical comparison of attractor vs boundary harmonics.

    Key prediction: Attractors should have narrower FWHM (sharper peaks).

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with columns like 'sr1_fwhm', 'sr1.5_fwhm', 'sr1_type', etc.

    Returns
    -------
    dict with statistical comparison results
    """
    # Collect FWHM values by type
    attractor_fwhm = []
    boundary_fwhm = []
    attractor_q = []
    boundary_q = []
    attractor_sharpness = []
    boundary_sharpness = []

    for label, info in SR_HARMONIC_TABLE.items():
        fwhm_col = f'{label}_fwhm'
        q_col = f'{label}_q_factor'
        sharpness_col = f'{label}_peak_sharpness'

        if fwhm_col in results_df.columns:
            vals = results_df[fwhm_col].dropna().values
            if info['type'] == 'attractor':
                attractor_fwhm.extend(vals)
            else:
                boundary_fwhm.extend(vals)

        if q_col in results_df.columns:
            vals = results_df[q_col].dropna().values
            if info['type'] == 'attractor':
                attractor_q.extend(vals)
            else:
                boundary_q.extend(vals)

        if sharpness_col in results_df.columns:
            vals = results_df[sharpness_col].dropna().values
            if info['type'] == 'attractor':
                attractor_sharpness.extend(vals)
            else:
                boundary_sharpness.extend(vals)

    results = {
        'attractor_fwhm_n': len(attractor_fwhm),
        'boundary_fwhm_n': len(boundary_fwhm),
    }

    # FWHM comparison
    if len(attractor_fwhm) > 1 and len(boundary_fwhm) > 1:
        results['attractor_fwhm_mean'] = np.mean(attractor_fwhm)
        results['attractor_fwhm_std'] = np.std(attractor_fwhm)
        results['boundary_fwhm_mean'] = np.mean(boundary_fwhm)
        results['boundary_fwhm_std'] = np.std(boundary_fwhm)

        # t-test: attractors < boundaries (one-tailed)
        t_stat, p_two = stats.ttest_ind(attractor_fwhm, boundary_fwhm)
        results['fwhm_tstat'] = t_stat
        results['fwhm_pvalue'] = p_two / 2 if t_stat < 0 else 1 - p_two / 2  # one-tailed

        # Mann-Whitney U (non-parametric)
        u_stat, p_mw = stats.mannwhitneyu(attractor_fwhm, boundary_fwhm, alternative='less')
        results['fwhm_ustat'] = u_stat
        results['fwhm_pvalue_mw'] = p_mw

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(attractor_fwhm) + np.var(boundary_fwhm)) / 2)
        results['fwhm_cohens_d'] = (np.mean(attractor_fwhm) - np.mean(boundary_fwhm)) / (pooled_std + 1e-12)

        # Prediction check
        results['prediction_confirmed'] = results['attractor_fwhm_mean'] < results['boundary_fwhm_mean']

    # Q-factor comparison
    if len(attractor_q) > 1 and len(boundary_q) > 1:
        results['attractor_q_mean'] = np.mean(attractor_q)
        results['boundary_q_mean'] = np.mean(boundary_q)
        t_stat_q, p_q = stats.ttest_ind(attractor_q, boundary_q)
        results['q_tstat'] = t_stat_q
        results['q_pvalue'] = p_q / 2 if t_stat_q > 0 else 1 - p_q / 2  # attractors > boundaries

    return results


def ged_ignition_baseline_contrast(
    ignition_results: List[Dict[str, Any]],
    baseline_results: List[Dict[str, Any]],
    metrics: List[str] = None,
    harmonics: List[str] = None
) -> Dict[str, Any]:
    """
    Statistical comparison of GED metrics between ignition and baseline states.

    Tests whether φⁿ predictions are specifically tied to ignition states by
    comparing GED-derived metrics (FWHM, Q-factor, eigenvalue, etc.) between
    ignition windows and non-ignition baseline segments.

    Parameters
    ----------
    ignition_results : List[Dict]
        Per-window GED results from process_ignition_windows()
    baseline_results : List[Dict]
        Per-window GED results from process_baseline_windows()
    metrics : List[str], optional
        Metrics to compare. Default: ['fwhm', 'q_factor', 'eigenvalue', 'lambda_ratio']
    harmonics : List[str], optional
        Harmonics to include. Default: all in SR_HARMONIC_TABLE

    Returns
    -------
    Dict with:
        per_harmonic : Dict[str, Dict] - per-harmonic statistical comparisons
        aggregate : Dict - aggregated statistics across all harmonics
        summary_table : pd.DataFrame - formatted comparison table
    """
    if metrics is None:
        metrics = ['fwhm', 'q_factor', 'eigenvalue', 'lambda_ratio']
    if harmonics is None:
        harmonics = list(SR_HARMONIC_TABLE.keys())

    # Convert to DataFrames for easier analysis
    ign_df = pd.DataFrame(ignition_results) if ignition_results else pd.DataFrame()
    base_df = pd.DataFrame(baseline_results) if baseline_results else pd.DataFrame()

    if ign_df.empty or base_df.empty:
        return {
            'per_harmonic': {},
            'aggregate': {'error': 'Insufficient data for comparison'},
            'summary_table': pd.DataFrame()
        }

    per_harmonic = {}
    all_ignition_vals = {m: [] for m in metrics}
    all_baseline_vals = {m: [] for m in metrics}

    # Per-harmonic analysis
    for label in harmonics:
        per_harmonic[label] = {}

        for metric in metrics:
            col = f'{label}_{metric}'
            if col not in ign_df.columns or col not in base_df.columns:
                continue

            ign_vals = ign_df[col].dropna().values
            base_vals = base_df[col].dropna().values

            if len(ign_vals) < 2 or len(base_vals) < 2:
                continue

            # Collect for aggregate analysis
            all_ignition_vals[metric].extend(ign_vals)
            all_baseline_vals[metric].extend(base_vals)

            # Compute statistics
            result = _compute_contrast_stats(ign_vals, base_vals, metric)
            per_harmonic[label][metric] = result

    # Aggregate analysis across all harmonics
    aggregate = {}
    for metric in metrics:
        ign_vals = np.array(all_ignition_vals[metric])
        base_vals = np.array(all_baseline_vals[metric])

        if len(ign_vals) >= 2 and len(base_vals) >= 2:
            aggregate[metric] = _compute_contrast_stats(ign_vals, base_vals, metric)

    # Build summary table
    rows = []
    for label in harmonics:
        if label not in per_harmonic:
            continue
        for metric in metrics:
            if metric not in per_harmonic[label]:
                continue
            r = per_harmonic[label][metric]
            rows.append({
                'harmonic': label,
                'metric': metric,
                'ignition_mean': r['ignition_mean'],
                'ignition_std': r['ignition_std'],
                'ignition_n': r['ignition_n'],
                'baseline_mean': r['baseline_mean'],
                'baseline_std': r['baseline_std'],
                'baseline_n': r['baseline_n'],
                'delta': r['delta'],
                'cohens_d': r['cohens_d'],
                't_stat': r['t_stat'],
                'p_value': r['p_value'],
                'p_value_mw': r['p_value_mw'],
            })

    summary_table = pd.DataFrame(rows)

    return {
        'per_harmonic': per_harmonic,
        'aggregate': aggregate,
        'summary_table': summary_table
    }


def _compute_contrast_stats(
    ignition_vals: np.ndarray,
    baseline_vals: np.ndarray,
    metric: str
) -> Dict[str, Any]:
    """
    Compute statistical comparison between ignition and baseline values.

    Parameters
    ----------
    ignition_vals : np.ndarray
        Values from ignition windows
    baseline_vals : np.ndarray
        Values from baseline windows
    metric : str
        Name of metric (for determining expected direction)

    Returns
    -------
    Dict with statistical comparison results
    """
    ign_mean = np.mean(ignition_vals)
    ign_std = np.std(ignition_vals, ddof=1)
    base_mean = np.mean(baseline_vals)
    base_std = np.std(baseline_vals, ddof=1)

    delta = ign_mean - base_mean

    # Cohen's d effect size
    pooled_std = np.sqrt((ign_std**2 + base_std**2) / 2)
    cohens_d = delta / (pooled_std + 1e-12)

    # Independent samples t-test (two-tailed)
    t_stat, p_value = stats.ttest_ind(ignition_vals, baseline_vals)

    # Mann-Whitney U (non-parametric, two-tailed)
    try:
        u_stat, p_value_mw = stats.mannwhitneyu(
            ignition_vals, baseline_vals, alternative='two-sided'
        )
    except ValueError:
        u_stat, p_value_mw = np.nan, np.nan

    # Expected direction based on metric
    # FWHM should be LOWER (sharper) during ignition
    # Q-factor, eigenvalue, lambda_ratio should be HIGHER during ignition
    if metric == 'fwhm':
        prediction_confirmed = delta < 0  # ignition < baseline
    else:
        prediction_confirmed = delta > 0  # ignition > baseline

    return {
        'ignition_mean': ign_mean,
        'ignition_std': ign_std,
        'ignition_n': len(ignition_vals),
        'baseline_mean': base_mean,
        'baseline_std': base_std,
        'baseline_n': len(baseline_vals),
        'delta': delta,
        'cohens_d': cohens_d,
        't_stat': t_stat,
        'p_value': p_value,
        'u_stat': u_stat,
        'p_value_mw': p_value_mw,
        'prediction_confirmed': prediction_confirmed
    }


# ============================================================================
# NOBLE NUMBER FUNCTIONS (SECONDARY)
# ============================================================================

def continued_fraction(x: float, max_terms: int = 15, tol: float = 1e-10) -> List[int]:
    """
    Compute continued fraction representation of a real number.

    x = a0 + 1/(a1 + 1/(a2 + 1/(a3 + ...)))

    Noble numbers have CF = [1, 1, 1, 1, ...] (all 1s).
    """
    cf = []
    for _ in range(max_terms):
        a = int(np.floor(x))
        cf.append(a)
        frac = x - a
        if frac < tol:
            break
        x = 1.0 / frac
        if x > 1e10:  # Overflow protection
            break
    return cf


def nobility_index(ratio: float, max_terms: int = 10) -> Dict[str, Any]:
    """
    Compute nobility index from continued fraction.

    Nobility measures how close the CF is to all-1s (golden ratio property).
    Higher nobility = more "noble" ratio (closer to φ-family).
    """
    if ratio <= 0 or not np.isfinite(ratio):
        return {'nobility': 0.0, 'cf_length': 0, 'ones_fraction': 0.0, 'cf_terms': []}

    cf = continued_fraction(ratio, max_terms=max_terms)

    if len(cf) <= 1:
        return {
            'nobility': 1.0 if len(cf) == 1 and cf[0] == 1 else 0.0,
            'cf_length': len(cf),
            'ones_fraction': 1.0 if len(cf) == 1 and cf[0] == 1 else 0.0,
            'cf_terms': cf
        }

    cf_arr = np.array(cf[1:])  # Exclude integer part
    ones_fraction = float(np.sum(cf_arr == 1) / len(cf_arr)) if len(cf_arr) > 0 else 0.0

    # Nobility: exponential decay based on deviation from all-1s
    nobility = float(np.exp(-np.mean(np.abs(cf_arr - 1))))

    return {
        'nobility': nobility,
        'cf_length': len(cf),
        'ones_fraction': ones_fraction,
        'cf_terms': cf
    }


def phi_distance(ratio: float, max_n: int = 6) -> Dict[str, Any]:
    """
    Compute distance from nearest φⁿ power.
    """
    if ratio <= 0 or not np.isfinite(ratio):
        return {
            'nearest_n': np.nan,
            'phi_power': np.nan,
            'absolute_distance': np.nan,
            'relative_distance': np.nan,
            'is_noble': False
        }

    # Generate φⁿ for n in range
    phi_powers = {n: PHI ** n for n in np.arange(-max_n, max_n + 1, 0.5)}

    distances = {n: abs(ratio - phi_powers[n]) for n in phi_powers}
    nearest_n = min(distances, key=distances.get)

    phi_n = phi_powers[nearest_n]
    abs_dist = distances[nearest_n]
    rel_dist = abs_dist / phi_n if phi_n > 0 else np.inf

    return {
        'nearest_n': nearest_n,
        'phi_power': phi_n,
        'absolute_distance': abs_dist,
        'relative_distance': rel_dist,
        'is_noble': rel_dist < 0.05  # 5% threshold
    }


def validate_phi_ratios(detected_freqs: Dict[str, float], tolerance: float = 0.05) -> Dict[str, Any]:
    """
    Validate that detected harmonics follow φⁿ scaling.
    """
    results = {}
    valid_count = 0
    total_count = 0

    for (num_lbl, den_lbl), expected in EXPECTED_RATIOS.items():
        f_num = detected_freqs.get(num_lbl)
        f_den = detected_freqs.get(den_lbl)

        if f_num is None or f_den is None or f_den < 1e-6:
            continue

        observed = f_num / f_den
        deviation = abs(observed - expected) / expected
        valid = deviation < tolerance

        results[f'{num_lbl}/{den_lbl}'] = {
            'observed': observed,
            'expected': expected,
            'deviation': deviation,
            'valid': valid
        }

        total_count += 1
        if valid:
            valid_count += 1

    return {
        'ratios': results,
        'valid_count': valid_count,
        'total_count': total_count,
        'phi_score': valid_count / max(1, total_count)
    }


# ============================================================================
# SESSION PROCESSING
# ============================================================================

@dataclass
class GEDBoundsResult:
    """Container for GED Bounds analysis results."""
    session_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # GED-optimized frequencies
    ged_freqs: Dict[str, float] = field(default_factory=dict)
    ged_fwhm: Dict[str, float] = field(default_factory=dict)
    ged_eigenvalues: Dict[str, float] = field(default_factory=dict)
    ged_q_factors: Dict[str, float] = field(default_factory=dict)
    ged_peak_sharpness: Dict[str, float] = field(default_factory=dict)
    ged_lambda_ratios: Dict[str, float] = field(default_factory=dict)

    # Sweep results (for visualization)
    sweep_results: Dict[str, Dict] = field(default_factory=dict)

    # Noble analysis (secondary)
    phi_validation: Optional[Dict[str, Any]] = None

    # Summary metrics
    n_harmonics_detected: int = 0
    phi_score: float = np.nan


def process_session(
    records: pd.DataFrame,
    eeg_channels: List[str],
    canonical_freqs: Optional[Dict[str, float]] = None,
    fs: float = 128.0,
    time_col: str = 'Timestamp',
    ged_search_range: float = 1.5,
    ged_bw: float = 0.5,
    ged_flank_bw: float = 1.0,
    session_id: str = 'S01',
    metadata: Optional[Dict[str, Any]] = None,
    max_freq: Optional[float] = None
) -> GEDBoundsResult:
    """
    Process a single EEG session for GED bounds analysis.

    Parameters
    ----------
    records : pd.DataFrame
        EEG data with columns for each channel
    eeg_channels : List[str]
        List of EEG channel column names (e.g., ['EEG.F4', 'EEG.O1'])
    canonical_freqs : Dict[str, float], optional
        Canonical harmonic frequencies. Defaults to SR_HARMONIC_TABLE.
    fs : float
        Sampling rate (Hz)
    ged_search_range : float
        Frequency search range for GED sweep (Hz)
    ged_bw : float
        Default signal bandwidth for GED (Hz). Per-harmonic bandwidths
        from SR_HARMONIC_TABLE['half_bw'] are used when available.
    session_id : str
        Session identifier
    metadata : dict, optional
        Additional metadata
    max_freq : float, optional
        Maximum frequency to analyze (defaults to fs/2 - 5)

    Returns
    -------
    GEDBoundsResult
    """
    result = GEDBoundsResult(session_id=session_id, metadata=metadata or {})

    if canonical_freqs is None:
        canonical_freqs = {k: v['freq'] for k, v in SR_HARMONIC_TABLE.items()}

    if max_freq is None:
        max_freq = fs / 2 - 5

    # Build multi-channel data matrix
    available_channels = [ch for ch in eeg_channels if ch in records.columns]

    if len(available_channels) == 0:
        return result

    X = np.vstack([
        pd.to_numeric(records[ch], errors='coerce').fillna(0).values
        for ch in available_channels
    ])

    # Check minimum data length
    min_samples = int(fs * 10)  # At least 10 seconds
    if X.shape[1] < min_samples:
        return result

    # Process each harmonic
    detected_freqs = {}

    for label, f_canon in canonical_freqs.items():
        # Get per-harmonic bandwidth from SR_HARMONIC_TABLE, fallback to ged_bw
        harmonic_bw = SR_HARMONIC_TABLE.get(label, {}).get('half_bw', ged_bw)

        # Skip if frequency too high
        if f_canon > max_freq:
            continue

        # Skip if frequency too low for reliable analysis
        if f_canon < harmonic_bw + ged_flank_bw + 1:
            continue

        # Run GED sweep
        sweep = ged_frequency_sweep(
            X, fs, f_canon,
            search_range=ged_search_range,
            bw=harmonic_bw,
            flank_bw=ged_flank_bw
        )

        if sweep['success']:
            result.sweep_results[label] = sweep
            result.ged_freqs[label] = sweep['optimal_freq']
            result.ged_fwhm[label] = sweep['fwhm']
            result.ged_eigenvalues[label] = sweep['peak_eigenvalue']
            result.ged_q_factors[label] = sweep['q_factor']
            result.ged_lambda_ratios[label] = sweep['lambda_ratio']

            # Peak sharpness
            metrics = compute_bandwidth_metrics(sweep)
            result.ged_peak_sharpness[label] = metrics['peak_sharpness']

            detected_freqs[label] = sweep['optimal_freq']

    result.n_harmonics_detected = len(detected_freqs)

    # Validate phi ratios (secondary)
    if len(detected_freqs) >= 2:
        result.phi_validation = validate_phi_ratios(detected_freqs)
        result.phi_score = result.phi_validation.get('phi_score', np.nan)

    return result


def process_ignition_windows(
    records: pd.DataFrame,
    ignition_windows: List[Tuple[float, float]],
    eeg_channels: List[str],
    canonical_freqs: Optional[Dict[str, float]] = None,
    fs: float = 128.0,
    time_col: str = 'Timestamp',
    ged_search_range: float = 1.5,
    ged_bw: float = 0.5,
    ged_flank_bw: float = 1.0,
    min_window_samples: int = 512,
    session_id: str = 'S01',
    metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Process GED bounds analysis on ignition event windows only.

    Parameters
    ----------
    records : pd.DataFrame
        Full session EEG data
    ignition_windows : List[Tuple[float, float]]
        List of (start_sec, end_sec) tuples defining ignition windows
    eeg_channels : List[str]
        EEG channel column names
    canonical_freqs : Dict[str, float], optional
        Harmonic frequencies to analyze
    fs : float
        Sampling rate (Hz)
    time_col : str
        Timestamp column name
    ged_search_range : float
        Frequency search range (Hz)
    ged_bw : float
        Default signal bandwidth (Hz). Per-harmonic bandwidths from
        SR_HARMONIC_TABLE['half_bw'] are used when available.
    min_window_samples : int
        Minimum samples required per window
    session_id : str
        Session identifier
    metadata : dict, optional
        Additional metadata

    Returns
    -------
    List[Dict] with per-window GED results
    """
    if canonical_freqs is None:
        canonical_freqs = {k: v['freq'] for k, v in SR_HARMONIC_TABLE.items()}

    max_freq = fs / 2 - 5

    # Get available channels
    available_channels = [ch for ch in eeg_channels if ch in records.columns]
    if len(available_channels) == 0:
        return []

    # Get timestamps
    if time_col not in records.columns:
        records = records.copy()
        records[time_col] = np.arange(len(records)) / fs

    timestamps = records[time_col].values

    # Convert to relative timestamps if needed (for Unix timestamps)
    t0 = timestamps[0]
    if t0 > 1e9:  # Unix timestamp detected
        rel_timestamps = timestamps - t0
    else:
        rel_timestamps = timestamps

    window_results = []

    for win_idx, (t_start, t_end) in enumerate(ignition_windows):
        # Extract window data
        mask = (rel_timestamps >= t_start) & (rel_timestamps <= t_end)
        window_df = records.loc[mask]

        if len(window_df) < min_window_samples:
            continue

        # Build data matrix for this window
        X = np.vstack([
            pd.to_numeric(window_df[ch], errors='coerce').fillna(0).values
            for ch in available_channels
        ])

        if X.shape[1] < min_window_samples:
            continue

        row = {
            'session_id': session_id,
            'window_idx': win_idx,
            'window_start': t_start,
            'window_end': t_end,
            'window_duration': t_end - t_start,
            'window_samples': X.shape[1],
            'analysis_type': 'ignition'
        }
        row.update(metadata or {})

        detected_freqs = {}

        # GED sweep for each harmonic
        for label, f_canon in canonical_freqs.items():
            # Get per-harmonic parameters from SR_HARMONIC_TABLE
            harmonic_info = SR_HARMONIC_TABLE.get(label, {})
            harmonic_bw = harmonic_info.get('half_bw', ged_bw)
            # Use search_center (CANON) for GED sweep, f_canon is φⁿ prediction
            search_center = harmonic_info.get('search_center', f_canon)

            if search_center > max_freq or search_center < harmonic_bw + ged_flank_bw + 1:
                continue

            sweep = ged_frequency_sweep(
                X, fs, search_center,  # Use CANON center for search
                search_range=harmonic_bw,  # Per-harmonic search range
                bw=harmonic_bw,
                flank_bw=ged_flank_bw,
                n_steps=31
            )

            if sweep['success']:
                htype = harmonic_info.get('type', 'unknown')
                row[f'{label}_type'] = htype
                row[f'{label}_canonical_freq'] = f_canon  # φⁿ prediction
                row[f'{label}_search_center'] = search_center  # CANON search center
                row[f'{label}_ged_freq'] = sweep['optimal_freq']
                row[f'{label}_half_bw'] = harmonic_bw
                row[f'{label}_fwhm'] = sweep['fwhm']
                row[f'{label}_eigenvalue'] = sweep['peak_eigenvalue']
                row[f'{label}_q_factor'] = sweep['q_factor']
                row[f'{label}_lambda_ratio'] = sweep['lambda_ratio']

                detected_freqs[label] = sweep['optimal_freq']

        row['n_harmonics'] = len(detected_freqs)

        # Phi validation
        if len(detected_freqs) >= 2:
            phi_val = validate_phi_ratios(detected_freqs)
            row['phi_score'] = phi_val.get('phi_score', np.nan)

        window_results.append(row)

    return window_results


def process_baseline_windows(
    records: pd.DataFrame,
    ignition_windows: List[Tuple[float, float]],
    eeg_channels: List[str],
    baseline_window_duration: Optional[float] = None,
    canonical_freqs: Optional[Dict[str, float]] = None,
    fs: float = 128.0,
    time_col: str = 'Timestamp',
    ged_search_range: float = 1.5,
    ged_bw: float = 0.5,
    ged_flank_bw: float = 1.0,
    min_window_samples: int = 512,
    session_id: str = 'S01',
    metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Process GED bounds analysis on baseline (non-ignition) windows.

    Extracts non-ignition segments and slices them into windows matching
    ignition window duration for fair comparison.

    Parameters
    ----------
    records : pd.DataFrame
        Full session EEG data
    ignition_windows : List[Tuple[float, float]]
        List of (start_sec, end_sec) tuples defining ignition windows to exclude
    eeg_channels : List[str]
        EEG channel column names
    baseline_window_duration : float, optional
        Duration for baseline windows (seconds). Defaults to mean ignition duration.
    canonical_freqs : Dict[str, float], optional
        Harmonic frequencies to analyze
    fs : float
        Sampling rate (Hz)
    time_col : str
        Timestamp column name
    ged_search_range : float
        Frequency search range (Hz)
    ged_bw : float
        Default signal bandwidth (Hz). Per-harmonic bandwidths from
        SR_HARMONIC_TABLE['half_bw'] are used when available.
    ged_flank_bw : float
        Bandwidth for flank (noise) estimation (Hz)
    min_window_samples : int
        Minimum samples required per window
    session_id : str
        Session identifier
    metadata : dict, optional
        Additional metadata

    Returns
    -------
    List[Dict] with per-window GED results, analysis_type='baseline'
    """
    if canonical_freqs is None:
        canonical_freqs = {k: v['freq'] for k, v in SR_HARMONIC_TABLE.items()}

    max_freq = fs / 2 - 5

    # Get available channels
    available_channels = [ch for ch in eeg_channels if ch in records.columns]
    if len(available_channels) == 0:
        return []

    # Get timestamps
    if time_col not in records.columns:
        records = records.copy()
        records[time_col] = np.arange(len(records)) / fs

    timestamps = records[time_col].values

    # Convert to relative timestamps if needed (for Unix timestamps)
    t0 = timestamps[0]
    if t0 > 1e9:
        rel_timestamps = timestamps - t0
    else:
        rel_timestamps = timestamps

    # Compute baseline window duration (default: mean ignition duration)
    if baseline_window_duration is None:
        if ignition_windows:
            baseline_window_duration = np.mean([t1 - t0 for t0, t1 in ignition_windows])
        else:
            baseline_window_duration = 10.0  # fallback

    # Create non-ignition mask
    n_samples = len(records)
    mask = np.ones(n_samples, dtype=bool)
    for (t_start, t_end) in ignition_windows:
        i_start = int(t_start * fs)
        i_end = int(t_end * fs)
        mask[max(0, i_start):min(n_samples, i_end)] = False

    # Find contiguous non-ignition segments
    baseline_segments = []
    in_segment = False
    seg_start = 0

    for i in range(n_samples):
        if mask[i] and not in_segment:
            seg_start = i
            in_segment = True
        elif not mask[i] and in_segment:
            seg_end = i
            baseline_segments.append((seg_start, seg_end))
            in_segment = False

    if in_segment:
        baseline_segments.append((seg_start, n_samples))

    # Slice baseline segments into windows of matching duration
    window_samples = int(baseline_window_duration * fs)
    baseline_windows = []

    for seg_start, seg_end in baseline_segments:
        seg_len = seg_end - seg_start
        n_windows = seg_len // window_samples

        for w in range(n_windows):
            w_start = seg_start + w * window_samples
            w_end = w_start + window_samples
            t_start_sec = rel_timestamps[w_start]
            t_end_sec = rel_timestamps[min(w_end - 1, n_samples - 1)]
            baseline_windows.append((t_start_sec, t_end_sec))

    # Process each baseline window
    window_results = []

    for win_idx, (t_start, t_end) in enumerate(baseline_windows):
        # Extract window data
        win_mask = (rel_timestamps >= t_start) & (rel_timestamps <= t_end)
        window_df = records.loc[win_mask]

        if len(window_df) < min_window_samples:
            continue

        # Build data matrix for this window
        X = np.vstack([
            pd.to_numeric(window_df[ch], errors='coerce').fillna(0).values
            for ch in available_channels
        ])

        if X.shape[1] < min_window_samples:
            continue

        row = {
            'session_id': session_id,
            'window_idx': win_idx,
            'window_start': t_start,
            'window_end': t_end,
            'window_duration': t_end - t_start,
            'window_samples': X.shape[1],
            'analysis_type': 'baseline'
        }
        row.update(metadata or {})

        detected_freqs = {}

        # GED sweep for each harmonic
        for label, f_canon in canonical_freqs.items():
            # Get per-harmonic parameters from SR_HARMONIC_TABLE
            harmonic_info = SR_HARMONIC_TABLE.get(label, {})
            harmonic_bw = harmonic_info.get('half_bw', ged_bw)
            # Use search_center (CANON) for GED sweep, f_canon is φⁿ prediction
            search_center = harmonic_info.get('search_center', f_canon)

            if search_center > max_freq or search_center < harmonic_bw + ged_flank_bw + 1:
                continue

            sweep = ged_frequency_sweep(
                X, fs, search_center,  # Use CANON center for search
                search_range=harmonic_bw,  # Per-harmonic search range
                bw=harmonic_bw,
                flank_bw=ged_flank_bw,
                n_steps=31
            )

            if sweep['success']:
                htype = harmonic_info.get('type', 'unknown')
                row[f'{label}_type'] = htype
                row[f'{label}_canonical_freq'] = f_canon  # φⁿ prediction
                row[f'{label}_search_center'] = search_center  # CANON search center
                row[f'{label}_ged_freq'] = sweep['optimal_freq']
                row[f'{label}_half_bw'] = harmonic_bw
                row[f'{label}_fwhm'] = sweep['fwhm']
                row[f'{label}_eigenvalue'] = sweep['peak_eigenvalue']
                row[f'{label}_q_factor'] = sweep['q_factor']
                row[f'{label}_lambda_ratio'] = sweep['lambda_ratio']

                detected_freqs[label] = sweep['optimal_freq']

        row['n_harmonics'] = len(detected_freqs)

        # Phi validation
        if len(detected_freqs) >= 2:
            phi_val = validate_phi_ratios(detected_freqs)
            row['phi_score'] = phi_val.get('phi_score', np.nan)

        window_results.append(row)

    return window_results


def result_to_row(result: GEDBoundsResult) -> Dict[str, Any]:
    """Convert GEDBoundsResult to a flat dictionary for DataFrame."""
    row = {
        'session_id': result.session_id,
        'n_harmonics': result.n_harmonics_detected,
        'phi_score': result.phi_score,
    }

    # Add metadata
    row.update(result.metadata)

    # Add per-harmonic metrics
    for label in SR_HARMONIC_TABLE.keys():
        harmonic_type = SR_HARMONIC_TABLE[label]['type']

        row[f'{label}_type'] = harmonic_type
        row[f'{label}_canonical_freq'] = SR_HARMONIC_TABLE[label]['freq']
        row[f'{label}_ged_freq'] = result.ged_freqs.get(label, np.nan)
        row[f'{label}_fwhm'] = result.ged_fwhm.get(label, np.nan)
        row[f'{label}_eigenvalue'] = result.ged_eigenvalues.get(label, np.nan)
        row[f'{label}_q_factor'] = result.ged_q_factors.get(label, np.nan)
        row[f'{label}_peak_sharpness'] = result.ged_peak_sharpness.get(label, np.nan)
        row[f'{label}_lambda_ratio'] = result.ged_lambda_ratios.get(label, np.nan)

        # Frequency deviation from canonical
        if label in result.ged_freqs:
            canonical = SR_HARMONIC_TABLE[label]['freq']
            row[f'{label}_freq_deviation'] = result.ged_freqs[label] - canonical
            row[f'{label}_freq_deviation_pct'] = (result.ged_freqs[label] - canonical) / canonical * 100

    return row


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def batch_process_datasets(
    dataset_paths: Dict[str, List[str]],
    canonical_freqs: Optional[Dict[str, float]] = None,
    output_dir: str = 'exports_ged_bounds',
    verbose: bool = True,
    max_files_per_dataset: Optional[int] = None
) -> pd.DataFrame:
    """
    Process multiple datasets with GED bounds analysis.

    Parameters
    ----------
    dataset_paths : dict
        Mapping of dataset names to lists of file paths
    canonical_freqs : dict, optional
        SR harmonic frequencies (defaults to SR_HARMONIC_TABLE)
    output_dir : str
        Output directory for results
    verbose : bool
        Print progress
    max_files_per_dataset : int, optional
        Limit files per dataset (for testing)

    Returns
    -------
    pd.DataFrame with per-session results
    """
    # Import here to avoid circular imports
    try:
        from utilities import load_eeg_csv, ELECTRODES
        from session_metadata import parse_session_metadata
    except ImportError:
        from lib.utilities import load_eeg_csv, ELECTRODES
        from lib.session_metadata import parse_session_metadata

    # Device-specific electrode configurations
    INSIGHT_ELECTRODES = ['AF3', 'AF4', 'T7', 'T8', 'Pz']
    MUSE_ELECTRODES = ['AF7', 'AF8', 'TP9', 'TP10']

    if canonical_freqs is None:
        canonical_freqs = {k: v['freq'] for k, v in SR_HARMONIC_TABLE.items()}

    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    total_processed = 0
    total_errors = 0

    for dataset_name, file_list in dataset_paths.items():
        if verbose:
            print(f"\nProcessing dataset: {dataset_name} ({len(file_list)} files)")

        # Limit files if specified
        if max_files_per_dataset:
            file_list = file_list[:max_files_per_dataset]

        # Determine device and electrodes
        dataset_lower = dataset_name.lower()
        if 'insight' in dataset_lower:
            electrodes = INSIGHT_ELECTRODES
            device = 'insight'
        elif 'muse' in dataset_lower:
            electrodes = MUSE_ELECTRODES
            device = 'muse'
        else:
            electrodes = ELECTRODES
            device = 'epoc'

        eeg_channels = [f'EEG.{e}' for e in electrodes]

        for i, filepath in enumerate(file_list):
            try:
                # Parse metadata
                metadata = parse_session_metadata(filepath, dataset_name)
                session_id = f"{metadata.get('subject', 'unknown')}_{metadata.get('context', 'unknown')}"

                if verbose and (i + 1) % 50 == 0:
                    print(f"  [{i+1}/{len(file_list)}] Processing {session_id}...")

                # Load data
                records = load_eeg_csv(filepath, electrodes=electrodes, device=device)

                # Infer sampling rate
                if 'Timestamp' in records.columns:
                    dt = np.diff(records['Timestamp'].values[:1000])
                    dt = dt[np.isfinite(dt) & (dt > 0)]
                    fs = 1.0 / np.median(dt) if len(dt) > 0 else 128.0
                else:
                    fs = 128.0

                # Process session
                result = process_session(
                    records, eeg_channels, canonical_freqs,
                    fs=fs,
                    session_id=session_id,
                    metadata=metadata
                )

                # Convert to row and append
                row = result_to_row(result)
                row['filepath'] = filepath
                row['dataset'] = dataset_name
                all_results.append(row)
                total_processed += 1

            except Exception as e:
                total_errors += 1
                if verbose:
                    print(f"  Error processing {filepath}: {e}")
                continue

    if verbose:
        print(f"\nBatch processing complete: {total_processed} sessions, {total_errors} errors")

    # Create DataFrame
    results_df = pd.DataFrame(all_results)

    # Save main results
    results_df.to_csv(os.path.join(output_dir, 'batch_results.csv'), index=False)

    # Compute and save attractor-boundary comparison
    if len(results_df) > 0:
        contrast = attractor_boundary_contrast(results_df)
        contrast_df = pd.DataFrame([contrast])
        contrast_df.to_csv(os.path.join(output_dir, 'attractor_boundary_comparison.csv'), index=False)

        if verbose:
            print("\n" + "=" * 60)
            print("ATTRACTOR vs BOUNDARY COMPARISON")
            print("=" * 60)
            if 'attractor_fwhm_mean' in contrast:
                print(f"Attractor mean FWHM: {contrast['attractor_fwhm_mean']:.3f} Hz (n={contrast['attractor_fwhm_n']})")
                print(f"Boundary mean FWHM:  {contrast['boundary_fwhm_mean']:.3f} Hz (n={contrast['boundary_fwhm_n']})")
                print(f"FWHM t-test p-value: {contrast.get('fwhm_pvalue', np.nan):.4f}")
                print(f"Cohen's d: {contrast.get('fwhm_cohens_d', np.nan):.3f}")
                print(f"Prediction (attractors < boundaries): {contrast.get('prediction_confirmed', 'N/A')}")

    return results_df


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_ged_sweep(
    sweep_result: Dict[str, Any],
    title: str = 'GED Frequency Optimization',
    canonical_freq: Optional[float] = None,
    show: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot GED eigenvalue curve with optimal frequency and FWHM."""
    fig, ax = plt.subplots(figsize=(8, 5))

    freqs = sweep_result.get('frequencies', np.array([]))
    eigs = sweep_result.get('eigenvalue_curve', np.array([]))
    f_opt = sweep_result.get('optimal_freq', np.nan)
    fwhm_bounds = sweep_result.get('fwhm_bounds', (np.nan, np.nan))

    if len(freqs) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return fig

    # Plot eigenvalue curve
    ax.plot(freqs, eigs, 'b-', lw=2, label='GED Eigenvalue')

    # Mark optimal frequency
    ax.axvline(f_opt, color='r', ls='--', lw=1.5, label=f'Optimal: {f_opt:.2f} Hz')

    # Shade FWHM region
    if np.isfinite(fwhm_bounds[0]) and np.isfinite(fwhm_bounds[1]):
        ax.axvspan(fwhm_bounds[0], fwhm_bounds[1], alpha=0.2, color='green',
                   label=f'FWHM: {fwhm_bounds[1] - fwhm_bounds[0]:.2f} Hz')

    # Mark canonical frequency
    if canonical_freq is not None:
        ax.axvline(canonical_freq, color='orange', ls=':', lw=1.5, label=f'Canonical: {canonical_freq:.2f} Hz')

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Max Eigenvalue (SNR)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()

    return fig


def plot_attractor_boundary_comparison(
    results_df: pd.DataFrame,
    metric: str = 'fwhm',
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot comparison of metric distributions for attractors vs boundaries."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Collect values by type
    attractor_vals = []
    boundary_vals = []
    attractor_labels = []
    boundary_labels = []

    for label, info in SR_HARMONIC_TABLE.items():
        col = f'{label}_{metric}'
        if col in results_df.columns:
            vals = results_df[col].dropna().values
            if len(vals) > 0:
                if info['type'] == 'attractor':
                    attractor_vals.append(vals)
                    attractor_labels.append(label.upper())
                else:
                    boundary_vals.append(vals)
                    boundary_labels.append(label.upper())

    # Box plot by harmonic
    ax1 = axes[0]
    all_data = attractor_vals + boundary_vals
    all_labels = attractor_labels + boundary_labels

    if all_data:
        positions = list(range(len(all_data)))
        bp = ax1.boxplot(all_data, positions=positions, patch_artist=True)

        # Color by type
        colors = ['#2ecc71'] * len(attractor_vals) + ['#e74c3c'] * len(boundary_vals)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax1.set_xticks(positions)
        ax1.set_xticklabels(all_labels, rotation=45)
        ax1.set_ylabel(metric.upper())
        ax1.set_title(f'{metric.upper()} by Harmonic')

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', alpha=0.6, label='Attractor'),
            Patch(facecolor='#e74c3c', alpha=0.6, label='Boundary')
        ]
        ax1.legend(handles=legend_elements)

    # Histogram comparison
    ax2 = axes[1]
    if attractor_vals and boundary_vals:
        all_attractor = np.concatenate(attractor_vals)
        all_boundary = np.concatenate(boundary_vals)

        bins = np.linspace(
            min(all_attractor.min(), all_boundary.min()),
            max(all_attractor.max(), all_boundary.max()),
            30
        )

        ax2.hist(all_attractor, bins=bins, alpha=0.6, color='#2ecc71', label='Attractors', density=True)
        ax2.hist(all_boundary, bins=bins, alpha=0.6, color='#e74c3c', label='Boundaries', density=True)

        ax2.axvline(np.mean(all_attractor), color='#2ecc71', ls='--', lw=2)
        ax2.axvline(np.mean(all_boundary), color='#e74c3c', ls='--', lw=2)

        ax2.set_xlabel(metric.upper())
        ax2.set_ylabel('Density')
        ax2.set_title(f'{metric.upper()} Distribution: Attractor vs Boundary')
        ax2.legend()

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()

    return fig


def print_harmonic_table():
    """Print formatted SR harmonic table."""
    print("\n" + "=" * 80)
    print("SR HARMONIC TABLE (Attractor-Boundary Framework)")
    print("=" * 80)
    print(f"{'Label':<10} {'n':<6} {'φⁿ':<10} {'Freq (Hz)':<12} {'Type':<12} {'Band':<15}")
    print("-" * 80)

    for label, info in SR_HARMONIC_TABLE.items():
        phi_n = PHI ** info['n']
        print(f"{label.upper():<10} {info['n']:<6.1f} {phi_n:<10.4f} "
              f"{info['freq']:<12.2f} {info['type']:<12} {info['band']:<15}")

    print("=" * 80)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print_harmonic_table()

    print("\nGED Bounds Analysis Module")
    print("Usage:")
    print("  from lib.ged_bounds import batch_process_datasets, attractor_boundary_contrast")
    print("  results = batch_process_datasets({'emotions': glob('data/emotions/*.csv')})")

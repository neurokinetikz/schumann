"""
Band-by-Band GED Analysis: Position-Type Contrasts within EEG Frequency Bands
==============================================================================

Implements band-constrained GED analysis with position-type contrasts
within each phi-octave (EEG band) and cross-band pattern comparisons.

Key Features:
- Band-isolated GED: bandpass filter signal to band range first, then GED
- Per-band position sweep: all 8 position types per octave
- Position-type contrasts: boundary vs attractor, noble hierarchy, etc.
- Cross-band aggregation and pattern comparison

Bands (skipping Delta per user request):
- Theta: 4.70-7.60 Hz (octave -1 to 0)
- Alpha: 7.60-12.30 Hz (octave 0 to 1)
- Beta Low: 12.30-19.90 Hz (octave 1 to 2)
- Beta High: 19.90-32.19 Hz (octave 2 to 3)
- Gamma: 32.19-52.09 Hz (octave 3 to 4)

Position Types (8 per octave):
- Boundary (n + 0.000)
- 4-Noble (n + 0.146)
- 3-Noble (n + 0.236)
- 2-Noble (n + 0.382)
- Attractor (n + 0.500)
- 1-Noble (n + 0.618)
- 3-Inverse (n + 0.764)
- 4-Inverse (n + 0.854)

Dependencies: numpy, pandas, scipy, phi_frequency_model, ged_phi_analysis
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from scipy import signal, stats

# Import from existing modules
try:
    from phi_frequency_model import (
        PHI, F0, POSITION_OFFSETS, POSITION_HIERARCHY, BANDS,
        PhiPrediction, PhiTable, generate_phi_table, get_default_phi_table
    )
except ImportError:
    from lib.phi_frequency_model import (
        PHI, F0, POSITION_OFFSETS, POSITION_HIERARCHY, BANDS,
        PhiPrediction, PhiTable, generate_phi_table, get_default_phi_table
    )

# ============================================================================
# CONSTANTS
# ============================================================================

# Target bands (excluding Delta)
TARGET_BANDS = {
    'theta':     {'freq_range': (4.70, 7.60),  'octave': -1},
    'alpha':     {'freq_range': (7.60, 12.30), 'octave': 0},
    'beta_low':  {'freq_range': (12.30, 19.90), 'octave': 1},
    'beta_high': {'freq_range': (19.90, 32.19), 'octave': 2},
    'gamma':     {'freq_range': (32.19, 52.09), 'octave': 3},
}

# Position types in octave order
POSITION_TYPES_ORDERED = [
    'boundary', 'noble_4', 'noble_3', 'noble_2',
    'attractor', 'noble_1', 'inv_noble_3', 'inv_noble_4'
]

# Contrast definitions
CONTRAST_PAIRS = {
    'boundary_vs_attractor': (['boundary'], ['attractor']),
    'nobles_vs_inverses': (['noble_1', 'noble_2', 'noble_3', 'noble_4'],
                           ['inv_noble_3', 'inv_noble_4']),
    'primary_noble_vs_others': (['noble_1'],
                                ['noble_2', 'noble_3', 'noble_4']),
    'edges_vs_centers': (['boundary', 'inv_noble_4'],
                         ['attractor', 'noble_1']),
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class BandGEDResult:
    """Results from GED analysis of a single band."""
    band: str
    freq_range: Tuple[float, float]
    octave: int

    # Per-position results
    position_results: pd.DataFrame  # Columns: position_type, freq_predicted,
                                    # freq_observed, deviation_pct, fwhm,
                                    # q_factor, eigenvalue, lambda_ratio

    # Aggregated metrics
    mean_eigenvalue: float = np.nan
    mean_q_factor: float = np.nan
    mean_fwhm: float = np.nan
    mean_deviation_pct: float = np.nan

    # Success metrics
    n_positions_found: int = 0
    success: bool = False


@dataclass
class PositionContrastResult:
    """Statistical contrast between position type groups."""
    contrast_name: str
    group_a: List[str]
    group_b: List[str]
    band: str  # 'all' for cross-band

    # Per-metric statistics
    metrics: Dict[str, Dict[str, Any]]  # metric -> {t_stat, p_value, cohens_d, ...}

    # Summary
    significant_metrics: List[str] = field(default_factory=list)
    effect_direction: str = ''  # 'A>B', 'B>A', 'none'


@dataclass
class CrossBandPattern:
    """Cross-band pattern analysis results."""
    metric: str  # e.g., 'fwhm', 'q_factor'
    position_type: str  # e.g., 'attractor' or 'all'

    # Per-band values
    band_values: Dict[str, float]  # band -> mean value
    band_stds: Dict[str, float]  # band -> std
    band_ns: Dict[str, int]  # band -> n observations

    # Trend statistics
    spearman_r: float = np.nan  # correlation with band frequency
    spearman_p: float = np.nan
    kruskal_h: float = np.nan  # Kruskal-Wallis across bands
    kruskal_p: float = np.nan

    # Pattern summary
    pattern: str = ''  # 'increasing', 'decreasing', 'u_shaped', 'flat', 'inconsistent'


@dataclass
class AllBandsGEDResult:
    """Complete band-by-band GED analysis results."""
    # Per-band results
    band_results: Dict[str, BandGEDResult]

    # Position contrasts per band
    per_band_contrasts: Dict[str, Dict[str, PositionContrastResult]]

    # Aggregated position contrasts (across all bands)
    aggregated_contrasts: Dict[str, PositionContrastResult]

    # Cross-band patterns
    crossband_patterns: Dict[str, CrossBandPattern]

    # Noble hierarchy test results per band
    noble_hierarchy_per_band: Dict[str, Dict[str, Any]]

    # Summary statistics
    summary: Dict[str, Any]

    # Combined DataFrame for export
    combined_df: pd.DataFrame = field(default_factory=pd.DataFrame)


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


def bandpass_isolate(
    X: np.ndarray,
    fs: float,
    band_range: Tuple[float, float],
    order: int = 4,
    pad_frac: float = 0.02
) -> np.ndarray:
    """
    Bandpass filter the signal to isolate a specific frequency band.

    Parameters
    ----------
    X : np.ndarray
        EEG data matrix, shape (n_channels, n_samples)
    fs : float
        Sampling rate (Hz)
    band_range : Tuple[float, float]
        (low_freq, high_freq) in Hz
    order : int
        Butterworth filter order
    pad_frac : float
        Fraction of Nyquist for frequency clamping padding

    Returns
    -------
    X_band : np.ndarray
        Band-limited signal, same shape as X
    """
    f_lo, f_hi = band_range
    nyq = 0.5 * fs
    pad = pad_frac * nyq

    # Clamp to safe range
    f_lo = max(pad, min(f_lo, nyq - 2 * pad))
    f_hi = max(f_lo + pad, min(f_hi, nyq - pad))

    b, a = signal.butter(order, [f_lo / nyq, f_hi / nyq], btype='band')
    return signal.filtfilt(b, a, X, axis=-1)


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

    Solves: Cs @ w = lambda * Cn @ w
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

    # Solve GED: Cs @ w = lambda * Cn @ w => inv(Cn) @ Cs @ w = lambda * w
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


def ged_sweep_position(
    X: np.ndarray,
    fs: float,
    prediction: PhiPrediction,
    search_range: Optional[float] = None,
    n_steps: int = 41,
    flank_bw: float = 0.8
) -> Dict[str, Any]:
    """
    GED sweep around a phi-predicted frequency with position-aware parameters.

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

    # Uniform search range for all position types
    # Avoids circular reasoning: position-specific ranges would bias deviation comparisons
    if search_range is None:
        search_range = 2.0 * bw  # Uniform ±2.0×bandwidth for all position types

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


# ============================================================================
# BAND-LEVEL GED ANALYSIS
# ============================================================================

def ged_analyze_band(
    X: np.ndarray,
    fs: float,
    band: str,
    phi_table: Optional[PhiTable] = None,
    flank_bw: float = 0.8,
    n_steps: int = 41,
    isolate_band: bool = True
) -> BandGEDResult:
    """
    Run GED analysis on a single EEG band with all 8 position types.

    Parameters
    ----------
    X : np.ndarray
        EEG data matrix, shape (n_channels, n_samples)
    fs : float
        Sampling rate (Hz)
    band : str
        Band name: 'theta', 'alpha', 'beta_low', 'beta_high', 'gamma'
    phi_table : PhiTable, optional
        Phi prediction table. If None, generates default.
    flank_bw : float
        Flanking noise bandwidth (Hz)
    n_steps : int
        Number of frequencies per sweep
    isolate_band : bool
        If True, bandpass filter signal to band range first (band-constrained GED)

    Returns
    -------
    BandGEDResult
        Results with per-position DataFrame and aggregated metrics
    """
    if band not in TARGET_BANDS:
        raise ValueError(f"Unknown band: {band}. Must be one of {list(TARGET_BANDS.keys())}")

    band_info = TARGET_BANDS[band]
    freq_range = band_info['freq_range']
    octave = band_info['octave']

    # Generate phi table if needed
    if phi_table is None:
        phi_table = generate_phi_table(
            f0=F0,
            octave_range=(octave, octave + 1),  # Single octave
            freq_limits=freq_range
        )

    # Get predictions for this band
    band_predictions = phi_table.by_band(band)

    if len(band_predictions) == 0:
        return BandGEDResult(
            band=band,
            freq_range=freq_range,
            octave=octave,
            position_results=pd.DataFrame(),
            success=False
        )

    # Optionally isolate the band first
    if isolate_band:
        X_band = bandpass_isolate(X, fs, freq_range)
    else:
        X_band = X

    # Run GED sweep for each position
    rows = []
    for pred in band_predictions:
        # Adaptive flank_bw for narrower bands
        adaptive_flank = min(flank_bw, (freq_range[1] - freq_range[0]) * 0.15)

        result = ged_sweep_position(
            X_band, fs, pred,
            flank_bw=adaptive_flank,
            n_steps=n_steps
        )

        if result['success']:
            rows.append({
                'band': band,
                'octave': octave,
                'position_type': pred.position_type,
                'label': pred.label,
                'n': pred.n,
                'freq_predicted': pred.frequency,
                'freq_observed': result['optimal_freq'],
                'deviation_hz': result['freq_deviation'],
                'deviation_pct': result['freq_deviation_pct'],
                'fwhm': result['fwhm'],
                'q_factor': result['q_factor'],
                'eigenvalue': result['peak_eigenvalue'],
                'lambda_ratio': result['lambda_ratio'],
            })

    if len(rows) == 0:
        return BandGEDResult(
            band=band,
            freq_range=freq_range,
            octave=octave,
            position_results=pd.DataFrame(),
            success=False
        )

    df = pd.DataFrame(rows)

    return BandGEDResult(
        band=band,
        freq_range=freq_range,
        octave=octave,
        position_results=df,
        mean_eigenvalue=df['eigenvalue'].mean(),
        mean_q_factor=df['q_factor'].mean(),
        mean_fwhm=df['fwhm'].mean(),
        mean_deviation_pct=df['deviation_pct'].abs().mean(),
        n_positions_found=len(df),
        success=True
    )


def ged_analyze_all_bands(
    X: np.ndarray,
    fs: float,
    bands: Optional[List[str]] = None,
    phi_table: Optional[PhiTable] = None,
    flank_bw: float = 0.8,
    n_steps: int = 41,
    isolate_band: bool = True,
    run_contrasts: bool = True,
    run_crossband: bool = True
) -> AllBandsGEDResult:
    """
    Run complete band-by-band GED analysis pipeline.

    Parameters
    ----------
    X : np.ndarray
        EEG data matrix, shape (n_channels, n_samples)
    fs : float
        Sampling rate (Hz)
    bands : List[str], optional
        Bands to analyze. Default: all except delta
    phi_table : PhiTable, optional
        Phi prediction table
    flank_bw : float
        Flanking noise bandwidth (Hz)
    n_steps : int
        Number of frequencies per sweep
    isolate_band : bool
        If True, bandpass filter signal to band range first
    run_contrasts : bool
        If True, compute position-type contrasts
    run_crossband : bool
        If True, compute cross-band patterns

    Returns
    -------
    AllBandsGEDResult
        Complete analysis results
    """
    if bands is None:
        bands = list(TARGET_BANDS.keys())

    # Generate full phi table
    if phi_table is None:
        phi_table = generate_phi_table(
            f0=F0,
            octave_range=(-1, 4),
            freq_limits=(3.0, 55.0)
        )

    # Analyze each band
    band_results = {}
    for band in bands:
        band_results[band] = ged_analyze_band(
            X, fs, band,
            phi_table=phi_table,
            flank_bw=flank_bw,
            n_steps=n_steps,
            isolate_band=isolate_band
        )

    # Combine all position results
    all_rows = []
    for band, result in band_results.items():
        if result.success and not result.position_results.empty:
            all_rows.append(result.position_results)

    combined_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

    # Position contrasts
    per_band_contrasts = {}
    aggregated_contrasts = {}

    if run_contrasts and not combined_df.empty:
        # Per-band contrasts
        for band in bands:
            band_df = combined_df[combined_df['band'] == band]
            if len(band_df) >= 4:
                per_band_contrasts[band] = compute_all_position_contrasts(band_df, band)

        # Aggregated contrasts
        aggregated_contrasts = compute_all_position_contrasts(combined_df, 'all')

    # Cross-band patterns
    crossband_patterns = {}
    if run_crossband and not combined_df.empty:
        crossband_patterns = compute_crossband_patterns(combined_df, band_results)

    # Noble hierarchy per band
    noble_hierarchy_per_band = {}
    for band in bands:
        band_df = combined_df[combined_df['band'] == band]
        if len(band_df) >= 4:
            noble_hierarchy_per_band[band] = test_noble_hierarchy(
                band_df, metric='eigenvalue'
            )

    # Summary
    summary = compute_summary_statistics(band_results, combined_df, aggregated_contrasts)

    return AllBandsGEDResult(
        band_results=band_results,
        per_band_contrasts=per_band_contrasts,
        aggregated_contrasts=aggregated_contrasts,
        crossband_patterns=crossband_patterns,
        noble_hierarchy_per_band=noble_hierarchy_per_band,
        summary=summary,
        combined_df=combined_df
    )


# ============================================================================
# POSITION CONTRAST FUNCTIONS
# ============================================================================

def compute_position_contrast(
    df: pd.DataFrame,
    group_a: List[str],
    group_b: List[str],
    contrast_name: str,
    band: str = 'all',
    metrics: Optional[List[str]] = None
) -> PositionContrastResult:
    """
    Statistical comparison of GED metrics between position type groups.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with position_type column and metric columns
    group_a : List[str]
        Position types for group A
    group_b : List[str]
        Position types for group B
    contrast_name : str
        Name for this contrast
    band : str
        Band name or 'all'
    metrics : List[str], optional
        Metrics to compare. Default: ['fwhm', 'q_factor', 'eigenvalue', 'lambda_ratio']

    Returns
    -------
    PositionContrastResult
        Statistical comparison results
    """
    if metrics is None:
        metrics = ['fwhm', 'q_factor', 'eigenvalue', 'lambda_ratio']

    # Filter by position type
    mask_a = df['position_type'].isin(group_a)
    mask_b = df['position_type'].isin(group_b)

    df_a = df[mask_a]
    df_b = df[mask_b]

    metric_results = {}
    significant_metrics = []
    overall_direction = None

    for metric in metrics:
        if metric not in df.columns:
            continue

        vals_a = df_a[metric].dropna().values
        vals_b = df_b[metric].dropna().values

        if len(vals_a) < 2 or len(vals_b) < 2:
            continue

        # T-test (Welch's)
        t_stat, t_pval = stats.ttest_ind(vals_a, vals_b, equal_var=False)

        # Mann-Whitney U
        u_stat, u_pval = stats.mannwhitneyu(vals_a, vals_b, alternative='two-sided')

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(vals_a, ddof=1) + np.var(vals_b, ddof=1)) / 2)
        cohens_d = (np.mean(vals_a) - np.mean(vals_b)) / (pooled_std + 1e-12)

        # Direction
        direction = 'A>B' if np.mean(vals_a) > np.mean(vals_b) else 'B>A'

        metric_results[metric] = {
            'group_a_mean': float(np.mean(vals_a)),
            'group_a_std': float(np.std(vals_a, ddof=1)),
            'group_a_n': len(vals_a),
            'group_b_mean': float(np.mean(vals_b)),
            'group_b_std': float(np.std(vals_b, ddof=1)),
            'group_b_n': len(vals_b),
            't_statistic': float(t_stat),
            't_pvalue': float(t_pval),
            'u_statistic': float(u_stat),
            'u_pvalue': float(u_pval),
            'cohens_d': float(cohens_d),
            'direction': direction
        }

        # Track significant metrics (p < 0.05)
        if t_pval < 0.05:
            significant_metrics.append(metric)
            if overall_direction is None:
                overall_direction = direction

    return PositionContrastResult(
        contrast_name=contrast_name,
        group_a=group_a,
        group_b=group_b,
        band=band,
        metrics=metric_results,
        significant_metrics=significant_metrics,
        effect_direction=overall_direction or 'none'
    )


def compute_all_position_contrasts(
    df: pd.DataFrame,
    band: str = 'all'
) -> Dict[str, PositionContrastResult]:
    """
    Compute all defined position contrasts.

    Returns dict mapping contrast_name -> PositionContrastResult
    """
    results = {}

    for contrast_name, (group_a, group_b) in CONTRAST_PAIRS.items():
        results[contrast_name] = compute_position_contrast(
            df, group_a, group_b, contrast_name, band
        )

    return results


def test_noble_hierarchy(
    df: pd.DataFrame,
    metric: str = 'eigenvalue'
) -> Dict[str, Any]:
    """
    Test if noble hierarchy holds: 1-Noble > 2-Noble > 3-Noble > 4-Noble.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with position_type and metric columns
    metric : str
        Metric to test (higher = more prominent)

    Returns
    -------
    Dict with hierarchy test results
    """
    noble_order = ['noble_1', 'noble_2', 'noble_3', 'noble_4']

    means = {}
    stds = {}
    ns = {}
    all_values = {}

    for noble in noble_order:
        mask = df['position_type'] == noble
        vals = df.loc[mask, metric].dropna().values
        means[noble] = float(np.mean(vals)) if len(vals) > 0 else np.nan
        stds[noble] = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
        ns[noble] = len(vals)
        all_values[noble] = vals

    # Check monotonic decrease
    hierarchy_holds = True
    for i in range(len(noble_order) - 1):
        if np.isnan(means[noble_order[i]]) or np.isnan(means[noble_order[i + 1]]):
            continue
        if means[noble_order[i]] <= means[noble_order[i + 1]]:
            hierarchy_holds = False
            break

    # Jonckheere-Terpstra-like trend test using Spearman correlation
    ranks = []
    values = []
    for i, noble in enumerate(noble_order):
        for v in all_values[noble]:
            ranks.append(i + 1)
            values.append(v)

    if len(values) >= 4:
        spearman_r, spearman_p = stats.spearmanr(ranks, values)
    else:
        spearman_r, spearman_p = np.nan, np.nan

    # Kruskal-Wallis test for differences
    valid_groups = [all_values[n] for n in noble_order if len(all_values[n]) > 0]
    if len(valid_groups) >= 2:
        try:
            kruskal_h, kruskal_p = stats.kruskal(*valid_groups)
        except ValueError:
            kruskal_h, kruskal_p = np.nan, np.nan
    else:
        kruskal_h, kruskal_p = np.nan, np.nan

    return {
        'means': means,
        'stds': stds,
        'ns': ns,
        'hierarchy_holds': hierarchy_holds,
        'spearman_r': float(spearman_r) if np.isfinite(spearman_r) else np.nan,
        'spearman_p': float(spearman_p) if np.isfinite(spearman_p) else np.nan,
        'kruskal_h': float(kruskal_h) if np.isfinite(kruskal_h) else np.nan,
        'kruskal_p': float(kruskal_p) if np.isfinite(kruskal_p) else np.nan,
        'expected_direction': 'decreasing',  # 1 > 2 > 3 > 4
        'metric': metric
    }


# ============================================================================
# CROSS-BAND PATTERN FUNCTIONS
# ============================================================================

def compute_crossband_patterns(
    df: pd.DataFrame,
    band_results: Dict[str, BandGEDResult],
    metrics: Optional[List[str]] = None,
    position_types: Optional[List[str]] = None
) -> Dict[str, CrossBandPattern]:
    """
    Analyze how metrics vary across bands for each position type.

    Parameters
    ----------
    df : pd.DataFrame
        Combined DataFrame with all band results
    band_results : Dict[str, BandGEDResult]
        Per-band results
    metrics : List[str], optional
        Metrics to analyze
    position_types : List[str], optional
        Position types to analyze (plus 'all')

    Returns
    -------
    Dict mapping pattern_key -> CrossBandPattern
    """
    if metrics is None:
        metrics = ['fwhm', 'q_factor', 'eigenvalue', 'lambda_ratio']

    if position_types is None:
        position_types = ['all', 'boundary', 'attractor', 'noble_1']

    patterns = {}

    # Band center frequencies for trend analysis
    band_centers = {
        band: (info['freq_range'][0] + info['freq_range'][1]) / 2
        for band, info in TARGET_BANDS.items()
    }

    for metric in metrics:
        for pos_type in position_types:
            pattern_key = f"{metric}_{pos_type}"

            # Filter data
            if pos_type == 'all':
                sub_df = df
            else:
                sub_df = df[df['position_type'] == pos_type]

            if sub_df.empty or metric not in sub_df.columns:
                continue

            # Compute per-band statistics
            band_values = {}
            band_stds = {}
            band_ns = {}
            trend_x = []
            trend_y = []

            for band in TARGET_BANDS.keys():
                band_df = sub_df[sub_df['band'] == band]
                vals = band_df[metric].dropna().values

                if len(vals) > 0:
                    band_values[band] = float(np.mean(vals))
                    band_stds[band] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                    band_ns[band] = len(vals)

                    # For trend analysis
                    for v in vals:
                        trend_x.append(band_centers[band])
                        trend_y.append(v)

            if len(band_values) < 2:
                continue

            # Trend analysis: Spearman correlation with band center frequency
            if len(trend_x) >= 4:
                spearman_r, spearman_p = stats.spearmanr(trend_x, trend_y)
            else:
                spearman_r, spearman_p = np.nan, np.nan

            # Kruskal-Wallis test across bands
            band_groups = []
            for band in TARGET_BANDS.keys():
                band_df = sub_df[sub_df['band'] == band]
                vals = band_df[metric].dropna().values
                if len(vals) > 0:
                    band_groups.append(vals)

            if len(band_groups) >= 2:
                try:
                    kruskal_h, kruskal_p = stats.kruskal(*band_groups)
                except ValueError:
                    kruskal_h, kruskal_p = np.nan, np.nan
            else:
                kruskal_h, kruskal_p = np.nan, np.nan

            # Classify pattern
            pattern = classify_trend_pattern(band_values, spearman_r, spearman_p)

            patterns[pattern_key] = CrossBandPattern(
                metric=metric,
                position_type=pos_type,
                band_values=band_values,
                band_stds=band_stds,
                band_ns=band_ns,
                spearman_r=float(spearman_r) if np.isfinite(spearman_r) else np.nan,
                spearman_p=float(spearman_p) if np.isfinite(spearman_p) else np.nan,
                kruskal_h=float(kruskal_h) if np.isfinite(kruskal_h) else np.nan,
                kruskal_p=float(kruskal_p) if np.isfinite(kruskal_p) else np.nan,
                pattern=pattern
            )

    return patterns


def classify_trend_pattern(
    band_values: Dict[str, float],
    spearman_r: float,
    spearman_p: float,
    sig_threshold: float = 0.05,
    r_threshold: float = 0.5
) -> str:
    """
    Classify the cross-band pattern based on correlation and shape.

    Returns one of: 'increasing', 'decreasing', 'u_shaped', 'inverted_u', 'flat', 'inconsistent'
    """
    if not np.isfinite(spearman_r) or len(band_values) < 3:
        return 'insufficient_data'

    # Get values in band order
    ordered_bands = ['theta', 'alpha', 'beta_low', 'beta_high', 'gamma']
    vals = [band_values.get(b, np.nan) for b in ordered_bands]
    vals = [v for v in vals if np.isfinite(v)]

    if len(vals) < 3:
        return 'insufficient_data'

    # Check for significant monotonic trend
    if spearman_p < sig_threshold:
        if spearman_r > r_threshold:
            return 'increasing'
        elif spearman_r < -r_threshold:
            return 'decreasing'

    # Check for U-shape or inverted U-shape
    mid_idx = len(vals) // 2
    early_mean = np.mean(vals[:mid_idx])
    mid_mean = vals[mid_idx]
    late_mean = np.mean(vals[mid_idx+1:]) if mid_idx + 1 < len(vals) else mid_mean

    # U-shape: low in middle
    if mid_mean < early_mean and mid_mean < late_mean:
        return 'u_shaped'

    # Inverted U: high in middle
    if mid_mean > early_mean and mid_mean > late_mean:
        return 'inverted_u'

    # Check for flat pattern (low variance)
    cv = np.std(vals) / (np.mean(vals) + 1e-12)
    if cv < 0.15:
        return 'flat'

    return 'inconsistent'


# ============================================================================
# SUMMARY AND EXPORT FUNCTIONS
# ============================================================================

def compute_summary_statistics(
    band_results: Dict[str, BandGEDResult],
    combined_df: pd.DataFrame,
    aggregated_contrasts: Dict[str, PositionContrastResult]
) -> Dict[str, Any]:
    """
    Compute overall summary statistics from analysis.
    """
    summary = {
        'n_bands_analyzed': sum(1 for r in band_results.values() if r.success),
        'n_total_positions': len(combined_df) if not combined_df.empty else 0,
        'bands_with_data': [b for b, r in band_results.items() if r.success],
    }

    if not combined_df.empty:
        # Overall metrics
        summary['overall_mean_eigenvalue'] = float(combined_df['eigenvalue'].mean())
        summary['overall_mean_q_factor'] = float(combined_df['q_factor'].mean())
        summary['overall_mean_fwhm'] = float(combined_df['fwhm'].mean())
        summary['overall_mean_deviation_pct'] = float(combined_df['deviation_pct'].abs().mean())

        # Position type breakdown
        position_counts = combined_df['position_type'].value_counts().to_dict()
        summary['positions_per_type'] = position_counts

        # Key contrast results
        if 'boundary_vs_attractor' in aggregated_contrasts:
            contrast = aggregated_contrasts['boundary_vs_attractor']
            if 'fwhm' in contrast.metrics:
                summary['boundary_vs_attractor_fwhm_cohens_d'] = contrast.metrics['fwhm']['cohens_d']
                summary['boundary_vs_attractor_fwhm_pvalue'] = contrast.metrics['fwhm']['t_pvalue']
            if 'q_factor' in contrast.metrics:
                summary['boundary_vs_attractor_q_cohens_d'] = contrast.metrics['q_factor']['cohens_d']
                summary['boundary_vs_attractor_q_pvalue'] = contrast.metrics['q_factor']['t_pvalue']

    return summary


def export_contrast_table(
    contrasts: Dict[str, PositionContrastResult],
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Export position contrasts to a formatted DataFrame.
    """
    rows = []

    for contrast_name, contrast in contrasts.items():
        for metric, stat in contrast.metrics.items():
            rows.append({
                'contrast': contrast_name,
                'band': contrast.band,
                'metric': metric,
                'group_a': ', '.join(contrast.group_a),
                'group_b': ', '.join(contrast.group_b),
                'mean_a': stat['group_a_mean'],
                'std_a': stat['group_a_std'],
                'n_a': stat['group_a_n'],
                'mean_b': stat['group_b_mean'],
                'std_b': stat['group_b_std'],
                'n_b': stat['group_b_n'],
                't_statistic': stat['t_statistic'],
                't_pvalue': stat['t_pvalue'],
                'u_pvalue': stat['u_pvalue'],
                'cohens_d': stat['cohens_d'],
                'direction': stat['direction'],
                'significant': stat['t_pvalue'] < 0.05,
            })

    df = pd.DataFrame(rows)

    if output_path:
        df.to_csv(output_path, index=False)

    return df


def export_crossband_table(
    patterns: Dict[str, CrossBandPattern],
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Export cross-band patterns to a formatted DataFrame.
    """
    rows = []

    for pattern_key, pattern in patterns.items():
        row = {
            'metric': pattern.metric,
            'position_type': pattern.position_type,
            'pattern': pattern.pattern,
            'spearman_r': pattern.spearman_r,
            'spearman_p': pattern.spearman_p,
            'kruskal_h': pattern.kruskal_h,
            'kruskal_p': pattern.kruskal_p,
        }

        # Add per-band values
        for band, val in pattern.band_values.items():
            row[f'{band}_mean'] = val
        for band, std in pattern.band_stds.items():
            row[f'{band}_std'] = std
        for band, n in pattern.band_ns.items():
            row[f'{band}_n'] = n

        rows.append(row)

    df = pd.DataFrame(rows)

    if output_path:
        df.to_csv(output_path, index=False)

    return df


# ============================================================================
# CONVENIENCE PIPELINE FUNCTIONS
# ============================================================================

def run_band_ged_pipeline(
    records: pd.DataFrame,
    eeg_channels: List[str],
    fs: float = 128.0,
    bands: Optional[List[str]] = None,
    isolate_band: bool = True,
    output_dir: Optional[str] = None,
    session_id: str = 'session',
    verbose: bool = True
) -> AllBandsGEDResult:
    """
    Run complete band-by-band GED analysis pipeline.

    Parameters
    ----------
    records : pd.DataFrame
        EEG data with channel columns
    eeg_channels : List[str]
        Channel names (e.g., ['EEG.F4', 'EEG.O1'])
    fs : float
        Sampling rate (Hz)
    bands : List[str], optional
        Bands to analyze. Default: all except delta
    isolate_band : bool
        If True, bandpass filter before GED (band-constrained)
    output_dir : str, optional
        Directory for exporting results
    session_id : str
        Session identifier
    verbose : bool
        Print progress

    Returns
    -------
    AllBandsGEDResult
        Complete analysis results
    """
    import os

    if bands is None:
        bands = list(TARGET_BANDS.keys())

    # Get available channels and build matrix
    available_channels = [ch for ch in eeg_channels if ch in records.columns]
    if len(available_channels) == 0:
        # Try with EEG. prefix
        available_channels = [f'EEG.{ch}' for ch in eeg_channels if f'EEG.{ch}' in records.columns]

    if len(available_channels) == 0:
        raise ValueError("No valid EEG channels found in records")

    X = np.vstack([
        pd.to_numeric(records[ch], errors='coerce').fillna(0).values
        for ch in available_channels
    ])

    if verbose:
        print(f"Running band-by-band GED analysis on {len(available_channels)} channels")
        print(f"Bands: {bands}")
        print(f"Band isolation: {'enabled' if isolate_band else 'disabled'}")

    # Run analysis
    result = ged_analyze_all_bands(
        X, fs,
        bands=bands,
        isolate_band=isolate_band,
        run_contrasts=True,
        run_crossband=True
    )

    if verbose:
        print(f"\nResults summary:")
        print(f"  Bands analyzed: {result.summary.get('n_bands_analyzed', 0)}")
        print(f"  Total positions found: {result.summary.get('n_total_positions', 0)}")

        if 'boundary_vs_attractor_fwhm_cohens_d' in result.summary:
            print(f"  Boundary vs Attractor FWHM Cohen's d: "
                  f"{result.summary['boundary_vs_attractor_fwhm_cohens_d']:.3f}")

    # Export if directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Export combined results
        result.combined_df.to_csv(
            os.path.join(output_dir, f'{session_id}_band_ged_positions.csv'),
            index=False
        )

        # Export contrasts
        if result.aggregated_contrasts:
            export_contrast_table(
                result.aggregated_contrasts,
                os.path.join(output_dir, f'{session_id}_position_contrasts.csv')
            )

        # Export cross-band patterns
        if result.crossband_patterns:
            export_crossband_table(
                result.crossband_patterns,
                os.path.join(output_dir, f'{session_id}_crossband_patterns.csv')
            )

        if verbose:
            print(f"\nResults exported to: {output_dir}")

    return result


def extract_eeg_matrix(
    records: pd.DataFrame,
    eeg_channels: List[str]
) -> np.ndarray:
    """
    Extract EEG data matrix from records DataFrame.

    Returns
    -------
    X : np.ndarray
        EEG matrix, shape (n_channels, n_samples)
    """
    available = [ch for ch in eeg_channels if ch in records.columns]
    if len(available) == 0:
        # Try with EEG. prefix
        available = [f'EEG.{ch}' for ch in eeg_channels if f'EEG.{ch}' in records.columns]

    if len(available) == 0:
        raise ValueError("No valid channels found")

    return np.vstack([
        pd.to_numeric(records[ch], errors='coerce').fillna(0).values
        for ch in available
    ])


# ============================================================================
# MAIN / TEST
# ============================================================================

if __name__ == "__main__":
    print("Band-by-Band GED Analysis Module")
    print("=" * 50)
    print("\nTarget bands:")
    for band, info in TARGET_BANDS.items():
        print(f"  {band}: {info['freq_range']} Hz (octave {info['octave']})")

    print("\nPosition types per band:")
    for pos in POSITION_TYPES_ORDERED:
        offset = POSITION_OFFSETS.get(pos, 0)
        print(f"  {pos}: offset = {offset:.3f}")

    print("\nContrast pairs defined:")
    for name, (a, b) in CONTRAST_PAIRS.items():
        print(f"  {name}: {a} vs {b}")

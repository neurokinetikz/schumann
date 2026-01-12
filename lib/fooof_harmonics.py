"""
FOOOF/SpecParam-Based Schumann Harmonic Detection
==================================================

Separates periodic (oscillatory) peaks from aperiodic (1/f) background to improve
Schumann resonance harmonic detection. Builds on top of existing PSD methods.

Key Features
------------
- Parameterizes neural power spectra into aperiodic + periodic components
- Robust peak detection for Schumann harmonics (7.83, 14.3, 20.8, 27.3, 33.8 Hz)
- Multi-channel support with aggregation options (median, mean, per-channel)
- Aperiodic exponent (1/f slope) extraction for state analysis
- Visualization of FOOOF fits overlaid on raw PSD
- Compatible with existing channel extraction utilities

Dependencies
------------
numpy, pandas, scipy, matplotlib, specparam (or legacy fooof)

Installation:
    pip install specparam  # Recommended (replaces deprecated fooof)
    # OR
    pip install fooof      # Legacy (deprecated, but still works)

Usage Examples
--------------

# Example 1: Basic harmonic detection
harmonics, result = detect_harmonics_fooof(
    records=RECORDS,
    channels=['EEG.F4', 'EEG.O1', 'EEG.O2'],
    fs=128,
    f_can=(7.83, 14.3, 20.8, 27.3, 33.8),
    combine='median'
)
print("Detected harmonics:", harmonics)
print("Aperiodic exponent:", result.aperiodic_exponent)

# Example 2: Multi-channel with separate fits
results = detect_harmonics_fooof_multichannel(
    records=RECORDS,
    channels=['EEG.F4', 'EEG.O1', 'EEG.O2'],
    fs=128,
    separate_fits=True
)
for ch, data in results.items():
    print(f"{ch}: {data['harmonics']}, β={data['exponent']:.3f}")

# Example 3: Visualize FOOOF fit
fig = plot_fooof_fit_with_harmonics(
    fm,
    harmonics=(7.83, 14.3, 20.8, 27.3),
    title='Schumann Harmonics — FOOOF Parameterization'
)

# Example 4: Compare raw PSD vs FOOOF-cleaned peaks
compare_psd_fooof(records, channels, fs, f_can=(7.83, 14.3, 20.8, 27.3, 33.8))

API Overview
------------
detect_harmonics_fooof()              — Main single/combined PSD fit
detect_harmonics_fooof_multichannel() — Per-channel or combined fits
extract_aperiodic_params()            — Get 1/f exponent & offset
match_peaks_to_canonical()            — Peak-to-harmonic matching
plot_fooof_fit_with_harmonics()       — Visualization
compare_psd_fooof()                   — Before/after comparison plot
"""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple, Union, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

try:
    # Try new specparam package first (recommended)
    from specparam import SpectralModel, SpectralGroupModel
    import specparam
    FOOOF = SpectralModel
    FOOOFGroup = SpectralGroupModel
    FOOOF_AVAILABLE = True
    PACKAGE_NAME = 'specparam'
    # Check version to determine API (2.0+ uses new nested object API)
    SPECPARAM_VERSION = tuple(int(x) for x in specparam.__version__.split('.')[:2] if x.isdigit())
    IS_SPECPARAM_V2 = SPECPARAM_VERSION >= (2, 0)
except ImportError:
    try:
        # Fall back to legacy fooof package
        from fooof import FOOOF, FOOOFGroup
        FOOOF_AVAILABLE = True
        PACKAGE_NAME = 'fooof'
        IS_SPECPARAM_V2 = False
    except ImportError:
        FOOOF_AVAILABLE = False
        FOOOF = None
        FOOOFGroup = None
        PACKAGE_NAME = None
        IS_SPECPARAM_V2 = False


# ============================================================================
# Compatibility helpers for specparam vs fooof
# ============================================================================

def _has_model(fm) -> bool:
    """Check if model has been fitted (works with both specparam 1.x/2.x and fooof)."""
    if PACKAGE_NAME == 'specparam' and IS_SPECPARAM_V2:
        # specparam 2.0+: uses fm.results.has_model
        return hasattr(fm, 'results') and hasattr(fm.results, 'has_model') and fm.results.has_model
    elif PACKAGE_NAME == 'specparam':
        # specparam 1.x: check aperiodic_params_
        if not hasattr(fm, 'aperiodic_params_'):
            return False
        ap = fm.aperiodic_params_
        return ap is not None and len(ap) > 0 and not all(v is None for v in ap)
    else:
        # fooof: uses has_model attribute
        return hasattr(fm, 'has_model') and fm.has_model


def _get_power_spectrum(fm):
    """Get power spectrum from fitted model (works with all versions)."""
    if PACKAGE_NAME == 'specparam' and IS_SPECPARAM_V2:
        # specparam 2.0+: stored in data object
        return fm.data.power_spectrum
    elif hasattr(fm, 'power_spectrum'):
        return fm.power_spectrum
    elif hasattr(fm, 'power_spectrum_'):
        return fm.power_spectrum_
    else:
        raise AttributeError("Cannot find power spectrum attribute")


def _get_aperiodic_params(fm) -> np.ndarray:
    """Get aperiodic parameters [offset, exponent(, knee)] as array."""
    if PACKAGE_NAME == 'specparam' and IS_SPECPARAM_V2:
        # specparam 2.0+: fm.results.params.aperiodic.params
        return fm.results.params.aperiodic.params
    elif hasattr(fm, 'aperiodic_params_'):
        return fm.aperiodic_params_
    else:
        raise AttributeError("Cannot find aperiodic parameters")


def _get_peak_params(fm) -> np.ndarray:
    """Get peak parameters as Nx3 array [center_freq, power, bandwidth]."""
    if PACKAGE_NAME == 'specparam' and IS_SPECPARAM_V2:
        # specparam 2.0+: fm.results.params.periodic.params
        return fm.results.params.periodic.params
    elif hasattr(fm, 'peak_params_'):
        return fm.peak_params_
    else:
        raise AttributeError("Cannot find peak parameters")


def _get_r_squared(fm) -> float:
    """Get R² goodness of fit metric."""
    if PACKAGE_NAME == 'specparam' and IS_SPECPARAM_V2:
        # specparam 2.0+: stored in metrics results dictionary
        return fm.results.metrics.results['gof_rsquared']
    elif hasattr(fm, 'r_squared_'):
        return fm.r_squared_
    else:
        raise AttributeError("Cannot find R² metric")


def _get_error(fm) -> float:
    """Get error metric (MAE)."""
    if PACKAGE_NAME == 'specparam' and IS_SPECPARAM_V2:
        # specparam 2.0+: stored in metrics results dictionary
        return fm.results.metrics.results['error_mae']
    elif hasattr(fm, 'error_'):
        return fm.error_
    else:
        raise AttributeError("Cannot find error metric")


def _get_freqs(fm) -> np.ndarray:
    """Get frequency array from fitted model."""
    if PACKAGE_NAME == 'specparam' and IS_SPECPARAM_V2:
        # specparam 2.0+: stored in data object
        return fm.data.freqs
    elif hasattr(fm, 'freqs'):
        return fm.freqs
    else:
        raise AttributeError("Cannot find frequency array")


def _get_fooofed_spectrum(fm) -> np.ndarray:
    """Get full model fit spectrum (aperiodic + periodic)."""
    if PACKAGE_NAME == 'specparam' and IS_SPECPARAM_V2:
        # specparam 2.0+: fm.results.model.modeled_spectrum
        return fm.results.model.modeled_spectrum
    elif hasattr(fm, 'fooofed_spectrum_'):
        return fm.fooofed_spectrum_
    elif hasattr(fm, 'modeled_spectrum_'):
        return fm.modeled_spectrum_
    else:
        raise AttributeError("Cannot find modeled spectrum")


# ============================================================================
# Utilities (reuse patterns from existing codebase)
# ============================================================================

def _infer_fs(df: pd.DataFrame, time_col: str) -> float:
    """Infer sampling rate from time column."""
    t = np.asarray(pd.to_numeric(df[time_col], errors='coerce').values, float)
    dt = np.diff(t)
    dt = dt[(dt > 0) & np.isfinite(dt)]
    if dt.size == 0:
        raise ValueError("Cannot infer fs from time column.")
    return float(1.0 / np.median(dt))


def _get_channel_vector(records, ch: str) -> np.ndarray:
    """Extract single channel vector (compatible with existing pattern)."""
    candidates = {ch, f'EEG.{ch}', ch.upper(), f'EEG.{ch.upper()}'}

    if isinstance(records, dict):
        data = records.get('data', records.get('eeg', records.get('EEG')))
        if hasattr(data, 'columns'):
            cols = list(map(str, data.columns))
            for nm in candidates:
                if nm in cols:
                    return np.asarray(data[nm]).astype(float)
        ch_names = records.get('channel_names') or records.get('channels')
        if ch_names is not None and data is not None:
            ch_names = list(map(str, ch_names))
            for nm in candidates:
                if nm in ch_names:
                    i = ch_names.index(nm)
                    return np.asarray(data[:, i]).astype(float)

    if hasattr(records, ch):
        return np.asarray(getattr(records, ch)).astype(float)

    if hasattr(records, 'data') and hasattr(records, 'channel_names'):
        ch_names = list(map(str, getattr(records, 'channel_names')))
        for nm in candidates:
            if nm in ch_names:
                i = ch_names.index(nm)
                return np.asarray(getattr(records, 'data')[:, i]).astype(float)

    # DataFrame fallback
    if hasattr(records, 'columns'):
        for nm in candidates:
            if nm in records.columns:
                return np.asarray(records[nm]).astype(float)

    raise KeyError(f"Channel {ch} not found. Tried {sorted(candidates)}")


def _get_channel_array(records, channels: Sequence[str]) -> np.ndarray:
    """Stack multiple channel vectors into (n_channels, n_samples) array."""
    arr = []
    for ch in channels:
        v = _get_channel_vector(records, ch)
        v = np.asarray(v, dtype=float)
        if v.ndim != 1:
            raise ValueError(f"Channel {ch!r} is not a 1-D vector")
        arr.append(v)

    lens = {len(v) for v in arr}
    if len(lens) != 1:
        raise ValueError(f"Channels have mismatched lengths: {sorted(lens)}")

    return np.vstack(arr)


def _compute_welch_psd(
    x: np.ndarray,
    fs: float,
    nperseg_sec: float = 4.0,
    overlap: float = 0.5,
    detrend: str = 'constant'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Welch PSD for single channel or multi-channel array.

    Parameters
    ----------
    x : ndarray
        Signal data. Shape (n_samples,) or (n_channels, n_samples).
    fs : float
        Sampling frequency (Hz).
    nperseg_sec : float
        Segment length in seconds for Welch method.
    overlap : float
        Overlap fraction [0, 1).
    detrend : str
        Detrending method ('constant', 'linear', False).

    Returns
    -------
    freqs : ndarray
        Frequency bins (Hz).
    psd : ndarray
        Power spectral density. Shape (n_freqs,) or (n_channels, n_freqs).
    """
    nper = int(round(nperseg_sec * fs))
    nper = max(8, min(nper, x.shape[-1]))
    nover = int(round(overlap * nper))
    nover = min(nover, nper - 1)

    f, P = signal.welch(
        x, fs=fs, nperseg=nper, noverlap=nover,
        window='hann', detrend=detrend, scaling='density', axis=-1
    )

    if P.ndim == 1:
        P = P[None, :]

    return f, P


# ============================================================================
# FOOOF Core Functions
# ============================================================================

@dataclass
class FOOOFHarmonicResult:
    """Container for FOOOF harmonic detection results."""
    harmonics: List[float]  # Refined harmonic frequencies
    harmonic_powers: List[float]  # Peak powers at harmonics
    harmonic_bandwidths: List[float]  # Peak bandwidths
    aperiodic_offset: Union[float, List[float]]  # y-intercept of 1/f fit (per-harmonic if list)
    aperiodic_exponent: Union[float, List[float]]  # 1/f slope (per-harmonic if list)
    r_squared: Union[float, List[float]]  # FOOOF fit quality (per-harmonic if list)
    model: Any  # Full FOOOF model object(s) - single or list
    freqs: np.ndarray  # Frequency axis
    psd: np.ndarray  # Original PSD
    per_harmonic_fits: bool = False  # Whether per-harmonic fitting was used
    # Non-SR peak fields (for cluster analysis)
    unmatched_peaks: List[Dict] = field(default_factory=list)  # Peaks not matching any SR harmonic
    all_peaks: List[Dict] = field(default_factory=list)  # All detected peaks (matched + unmatched)

    def __repr__(self):
        harms = ', '.join(f'{h:.2f}' for h in self.harmonics)
        n_unmatched = len(self.unmatched_peaks)
        if self.per_harmonic_fits:
            # Show range of β values for per-harmonic fits
            if isinstance(self.aperiodic_exponent, list):
                beta_str = f"β=[{min(self.aperiodic_exponent):.3f}..{max(self.aperiodic_exponent):.3f}]"
            else:
                beta_str = f"β={self.aperiodic_exponent:.3f}"
            if isinstance(self.r_squared, list):
                r2_str = f"R²=[{min(self.r_squared):.3f}..{max(self.r_squared):.3f}]"
            else:
                r2_str = f"R²={self.r_squared:.3f}"
            return (f"FOOOFHarmonicResult(harmonics=[{harms}], {beta_str}, {r2_str}, "
                    f"per_harmonic=True, non_sr_peaks={n_unmatched})")
        else:
            return (f"FOOOFHarmonicResult(harmonics=[{harms}], "
                    f"β={self.aperiodic_exponent:.3f}, R²={self.r_squared:.3f}, "
                    f"non_sr_peaks={n_unmatched})")


def detect_harmonics_fooof(
    records,
    channels: Union[str, Sequence[str]],
    fs: Optional[float] = None,
    time_col: str = 'Timestamp',
    window: Optional[Sequence[float]] = None,
    f_can: Sequence[float] = (7.83, 14.3, 20.8, 27.3, 33.8),
    freq_range: Tuple[float, float] = (1.0, 50.0),
    nperseg_sec: float = 4.0,
    overlap: float = 0.5,
    combine: str = 'median',
    peak_width_limits: Tuple[float, float] = (1.0, 8.0),
    max_n_peaks: int = 10,
    min_peak_height: float = 0.05,
    peak_threshold: float = 2.0,
    search_halfband: Union[float, Sequence[float]] = 0.8,
    match_method: str = 'distance',
    per_harmonic_fits: bool = False,
    freq_ranges: Optional[Union[Sequence[Sequence[float]], Sequence[float]]] = None,
) -> Tuple[List[float], FOOOFHarmonicResult]:
    """
    Detect Schumann harmonics using FOOOF spectral parameterization.

    Separates periodic peaks from aperiodic 1/f background for robust
    harmonic detection. Supports single or multi-channel input with
    aggregation.

    Parameters
    ----------
    records : DataFrame or dict
        EEG data records.
    channels : str or sequence of str
        Channel name(s) to analyze. If multiple, PSDs are combined.
    fs : float, optional
        Sampling frequency (Hz). If None, inferred from time_col.
    time_col : str
        Timestamp column name for fs inference.
    window : sequence of float, optional
        Time window [start, end] in seconds. If None, uses full recording.
        Example: window=[564, 593] analyzes from 564s to 593s.
    f_can : sequence of float
        Canonical Schumann harmonic frequencies to search near (Hz).
    freq_range : tuple of float
        (f_min, f_max) frequency range for FOOOF fitting (Hz).
    nperseg_sec : float
        Welch segment length (seconds).
    overlap : float
        Welch overlap fraction [0, 1).
    combine : str
        How to combine multi-channel PSDs: 'median' | 'mean' | 'first'.
    peak_width_limits : tuple of float
        (min_bw, max_bw) allowable peak bandwidth in Hz.
    max_n_peaks : int
        Maximum number of peaks for FOOOF to fit.
    min_peak_height : float
        Minimum peak height (relative to aperiodic fit).
    peak_threshold : float
        Minimum peak height in absolute log power units.
    search_halfband : float or sequence of float
        Half-width (Hz) around each canonical frequency to search.
        If float: same halfband for all frequencies.
        If sequence: one halfband per frequency in f_can.
        Example: [0.5, 0.6, 0.8, 1.0, 1.2] for narrow H1, wider H5.
    match_method : str
        Method for selecting peak when multiple in window:
        'distance' (default) - Pick closest to canonical frequency
        'power' - Pick strongest (highest power) peak
        'average' - Power-weighted average of all peaks in window
    per_harmonic_fits : bool
        If True, run separate FOOOF fit for each canonical frequency within
        ±5 Hz window (10 Hz total). Each harmonic gets its own aperiodic
        parameters. If False (default), single FOOOF fit across freq_range.
        Ignored if freq_ranges is provided.
    freq_ranges : sequence of [f_min, f_max], optional
        Manual frequency ranges for FOOOF fitting, one per canonical frequency.
        Allows grouping harmonics into shared windows for efficiency.
        If provided, overrides per_harmonic_fits and freq_range.
        Example: [[5,15], [5,15], [15,25], [25,35], [25,35]]
        - Harmonics with same range share a single FOOOF fit
        - Results in per-harmonic β values (like per_harmonic_fits=True)

    Returns
    -------
    harmonics : list of float
        Refined harmonic frequencies (Hz), one per f_can.
    result : FOOOFHarmonicResult
        Full result object with aperiodic params, model, etc.

    Raises
    ------
    ImportError
        If fooof package is not installed.
    RuntimeError
        If FOOOF fit fails.

    Examples
    --------
    >>> # Analyze full recording
    >>> harmonics, result = detect_harmonics_fooof(
    ...     RECORDS, ['EEG.F4', 'EEG.O1'], fs=128,
    ...     f_can=(7.83, 14.3, 20.8, 27.3)
    ... )
    >>> print(f"H1: {harmonics[0]:.2f} Hz, β: {result.aperiodic_exponent:.3f}")

    >>> # Analyze specific time window
    >>> harmonics, result = detect_harmonics_fooof(
    ...     RECORDS, 'EEG.F4', fs=128,
    ...     window=[564, 593],  # 29-second window
    ...     f_can=(7.83, 14.3, 20.8, 27.3)
    ... )

    >>> # Event-triggered analysis
    >>> t0 = 100.5  # ignition onset time
    >>> harmonics, result = detect_harmonics_fooof(
    ...     RECORDS, 'EEG.F4', fs=128,
    ...     window=[t0 - 2.0, t0 + 2.0],  # ±2s around event
    ...     f_can=(7.83, 14.3, 20.8, 27.3)
    ... )

    >>> # Per-harmonic search windows (narrower for H1, wider for H5)
    >>> harmonics, result = detect_harmonics_fooof(
    ...     RECORDS, 'EEG.F4', fs=128,
    ...     f_can=(7.83, 14.3, 20.8, 27.3, 33.8),
    ...     search_halfband=[0.5, 0.6, 0.8, 1.0, 1.2]  # Different per harmonic
    ... )

    >>> # Pick strongest peak instead of closest
    >>> harmonics, result = detect_harmonics_fooof(
    ...     RECORDS, 'EEG.F4', fs=128,
    ...     window=[564, 593],
    ...     f_can=(7.83, 14.3, 20.8, 27.3),
    ...     match_method='power'  # Pick highest power peak
    ... )

    >>> # Average multiple peaks in window
    >>> harmonics, result = detect_harmonics_fooof(
    ...     RECORDS, 'EEG.F4', fs=128,
    ...     f_can=(7.83, 14.3, 20.8, 27.3),
    ...     match_method='average'  # Power-weighted average
    ... )

    >>> # Per-harmonic FOOOF fits (±5 Hz window per canonical)
    >>> harmonics, result = detect_harmonics_fooof(
    ...     RECORDS, 'EEG.F4', fs=128,
    ...     f_can=(7.83, 14.3, 20.8, 27.3),
    ...     per_harmonic_fits=True  # Separate fit for each harmonic
    ... )
    >>> # result.aperiodic_exponent is now a list with one β per harmonic
    >>> print(f"β values: {result.aperiodic_exponent}")

    >>> # Manual frequency ranges (grouped windows for efficiency)
    >>> harmonics, result = detect_harmonics_fooof(
    ...     RECORDS, 'EEG.F4', fs=128,
    ...     f_can=(7.6, 9.26, 12.13, 19.75, 25, 32),
    ...     freq_ranges=[[5,15], [5,15], [5,15], [15,25], [25,35], [25,35]],
    ...     match_method='power'
    ... )
    >>> # First 3 harmonics share [5,15] Hz fit, last 2 share [25,35] Hz fit
    >>> print(f"β per harmonic: {result.aperiodic_exponent}")
    """
    if not FOOOF_AVAILABLE:
        raise ImportError(
            "Spectral parameterization package not installed.\n"
            "Install with: pip install specparam (recommended)\n"
            "Or legacy: pip install fooof"
        )

    # Infer fs if needed
    if fs is None:
        if hasattr(records, 'columns') and time_col in records.columns:
            fs = _infer_fs(records, time_col)
        else:
            raise ValueError("fs must be provided or time_col must exist in records")

    # Filter by time window if specified
    if window is not None:
        if not hasattr(records, 'columns'):
            raise ValueError("window filtering requires records to be a DataFrame")
        if time_col not in records.columns:
            raise ValueError(f"time_col '{time_col}' not found in records")

        # Validate window format
        if len(window) != 2:
            raise ValueError("window must be [start, end] in seconds")
        start_time, end_time = float(window[0]), float(window[1])
        if start_time >= end_time:
            raise ValueError(f"window start ({start_time}) must be < end ({end_time})")

        # Get recording time bounds (normalize to relative time)
        t = records[time_col]
        t_min_abs = float(t.min())
        t = t - t_min_abs  # Convert to relative time
        t_min = float(t.min())  # Should be 0.0
        t_max = float(t.max())

        # Clip window to recording bounds
        original_start, original_end = start_time, end_time
        start_time = max(start_time, t_min)
        end_time = min(end_time, t_max)

        # Warn if window was clipped
        if start_time != original_start or end_time != original_end:
            import warnings
            warnings.warn(
                f"Window [{original_start:.2f}, {original_end:.2f}]s clipped to "
                f"recording bounds [{start_time:.2f}, {end_time:.2f}]s "
                f"(recording range: [{t_min:.2f}, {t_max:.2f}]s)",
                UserWarning
            )

        # Check if clipped window is valid
        if start_time >= end_time:
            raise ValueError(
                f"Window [{original_start:.2f}, {original_end:.2f}]s is completely "
                f"outside recording bounds [{t_min:.2f}, {t_max:.2f}]s"
            )

        # Apply time filter using relative time
        mask = (t >= start_time) & (t <= end_time)
        records = records[mask].copy()

        # Update time column in filtered records to use relative time
        records[time_col] = t[mask].values

        if len(records) == 0:
            raise ValueError(
                f"Window [{start_time:.2f}, {end_time:.2f}]s contains no data. "
                f"Recording range: [{t_min:.2f}, {t_max:.2f}]s"
            )

    # Get channel data
    if isinstance(channels, str):
        channels = [channels]

    X = _get_channel_array(records, channels)  # (n_channels, n_samples)

    # Compute PSD per channel
    freqs, Pxx = _compute_welch_psd(X, fs, nperseg_sec, overlap)

    # Combine channels
    if Pxx.shape[0] == 1:
        psd = Pxx[0]
    elif combine == 'median':
        psd = np.nanmedian(Pxx, axis=0)
    elif combine == 'mean':
        psd = np.nanmean(Pxx, axis=0)
    elif combine == 'first':
        psd = Pxx[0]
    else:
        raise ValueError("combine must be 'median', 'mean', or 'first'")

    # Branch: Manual freq_ranges, per-harmonic fits, or single global fit
    if freq_ranges is not None:
        # Manual frequency ranges mode: user specifies FOOOF window per harmonic
        # Validate format
        try:
            freq_ranges_list = [list(fr) for fr in freq_ranges]
        except TypeError as e:
            raise TypeError(
                f"freq_ranges must be a list of [f_min, f_max] ranges, one per harmonic. "
                f"Got: {freq_ranges}. "
                f"Example: freq_ranges=[[5,15], [5,15], [15,25], [25,35]]. "
                f"Did you mean to use freq_range={freq_ranges} (single range) instead?"
            ) from e
        if len(freq_ranges_list) != len(f_can):
            raise ValueError(
                f"freq_ranges length ({len(freq_ranges_list)}) must match "
                f"f_can length ({len(f_can)})"
            )

        # Validate each range
        for i, fr in enumerate(freq_ranges_list):
            if len(fr) != 2:
                raise ValueError(f"freq_ranges[{i}] must be [f_min, f_max], got {fr}")
            if fr[0] >= fr[1]:
                raise ValueError(f"freq_ranges[{i}]: f_min must be < f_max, got {fr}")

        # Group harmonics by unique frequency ranges
        # Create mapping: freq_range_tuple -> list of (harmonic_index, f_can_value)
        range_groups = {}
        for i, (f0, fr) in enumerate(zip(f_can, freq_ranges_list)):
            fr_tuple = tuple(fr)
            if fr_tuple not in range_groups:
                range_groups[fr_tuple] = []
            range_groups[fr_tuple].append((i, f0))

        # Prepare result containers
        harmonics = [np.nan] * len(f_can)
        powers = [np.nan] * len(f_can)
        bandwidths = [np.nan] * len(f_can)
        ap_offsets = [np.nan] * len(f_can)
        ap_exps = [np.nan] * len(f_can)
        r_squareds = [np.nan] * len(f_can)
        models = [None] * len(f_can)
        all_peaks_combined = []  # All peaks from all frequency range fits
        unmatched_combined = []  # Unmatched peaks from all frequency range fits
        seen_peak_groups = set()  # Track which groups we've already collected peaks from

        # Convert search_halfband to list for indexing
        if isinstance(search_halfband, (int, float)):
            halfbands = [float(search_halfband)] * len(f_can)
        else:
            halfbands = list(search_halfband)
            if len(halfbands) != len(f_can):
                raise ValueError(
                    f"search_halfband list length ({len(halfbands)}) must match "
                    f"f_can length ({len(f_can)})"
                )

        # Run FOOOF for each unique frequency range
        for fr_tuple, harm_indices_and_freqs in range_groups.items():
            f_min, f_max = fr_tuple

            # Get canonical frequencies in this group
            group_f_can = [f0 for (idx, f0) in harm_indices_and_freqs]
            group_indices = [idx for (idx, f0) in harm_indices_and_freqs]

            # Initialize FOOOF for this frequency range
            fm_group = FOOOF(
                peak_width_limits=peak_width_limits,
                max_n_peaks=max_n_peaks,
                min_peak_height=min_peak_height,
                peak_threshold=peak_threshold,
                aperiodic_mode='fixed',
                verbose=False
            )

            # Fit FOOOF on this frequency range
            try:
                fm_group.fit(freqs, psd, (f_min, f_max))
            except Exception as e:
                # If fit fails, all harmonics in this group get NaN
                continue

            if not _has_model(fm_group):
                # No model fitted for this group
                continue

            # Extract peaks from this group's FOOOF fit
            peak_params_group = _get_peak_params(fm_group)

            # Collect all peaks from this group (only once per unique freq range)
            if fr_tuple not in seen_peak_groups and len(peak_params_group) > 0:
                seen_peak_groups.add(fr_tuple)
                for row in peak_params_group:
                    all_peaks_combined.append({
                        'freq': float(row[0]),
                        'power': float(row[1]),
                        'bandwidth': float(row[2]),
                        'freq_range_min': f_min,
                        'freq_range_max': f_max
                    })

            # Extract aperiodic parameters for this group
            ap_params_group = _get_aperiodic_params(fm_group)
            r_squared_group = _get_r_squared(fm_group)

            # Track matched peak indices for this group
            group_matched_mask = np.zeros(len(peak_params_group), dtype=bool) if len(peak_params_group) > 0 else np.array([], dtype=bool)

            # Match each harmonic in this group to the peaks
            for idx, f0 in harm_indices_and_freqs:
                # Match this single canonical frequency to the group's peaks
                harms_single, pows_single, bws_single, unmatched_single = match_peaks_to_canonical(
                    peak_params_group,
                    [f0],
                    search_halfband=halfbands[idx],
                    method=match_method,
                    return_unmatched=True
                )

                # Store results for this harmonic
                harmonics[idx] = harms_single[0]
                powers[idx] = pows_single[0]
                bandwidths[idx] = bws_single[0]
                ap_offsets[idx] = float(ap_params_group[0])
                ap_exps[idx] = float(ap_params_group[1])
                r_squareds[idx] = float(r_squared_group)
                models[idx] = fm_group  # Share the same model for harmonics in this group

            # After processing all harmonics in this group, find truly unmatched peaks
            # (peaks not matched by ANY harmonic in this freq range)
            if len(peak_params_group) > 0:
                # Run match for all harmonics in group at once to find unmatched
                group_f_can = [f0 for (idx, f0) in harm_indices_and_freqs]
                group_halfbands = [halfbands[idx] for (idx, f0) in harm_indices_and_freqs]
                _, _, _, group_unmatched = match_peaks_to_canonical(
                    peak_params_group,
                    group_f_can,
                    search_halfband=group_halfbands,
                    method=match_method,
                    return_unmatched=True
                )
                unmatched_combined.extend(group_unmatched)

        # Create result with per-harmonic β values
        result = FOOOFHarmonicResult(
            harmonics=harmonics,
            harmonic_powers=powers,
            harmonic_bandwidths=bandwidths,
            aperiodic_offset=ap_offsets,
            aperiodic_exponent=ap_exps,
            r_squared=r_squareds,
            model=models,
            freqs=freqs,
            psd=psd,
            per_harmonic_fits=True,  # Set flag since we have per-harmonic results
            unmatched_peaks=unmatched_combined,
            all_peaks=all_peaks_combined
        )

        return harmonics, result

    elif per_harmonic_fits:
        # Per-harmonic fitting: separate FOOOF fit for each canonical frequency
        harmonics = []
        powers = []
        bandwidths = []
        ap_offsets = []
        ap_exps = []
        r_squareds = []
        models = []
        all_peaks_combined = []  # All peaks from all per-harmonic fits
        unmatched_combined = []  # Unmatched peaks from all per-harmonic fits

        # Convert search_halfband to list for indexing
        if isinstance(search_halfband, (int, float)):
            halfbands = [float(search_halfband)] * len(f_can)
        else:
            halfbands = list(search_halfband)
            if len(halfbands) != len(f_can):
                raise ValueError(
                    f"search_halfband list length ({len(halfbands)}) must match "
                    f"f_can length ({len(f_can)})"
                )

        for i, f0 in enumerate(f_can):
            # Define ±5 Hz window around canonical frequency
            f_min = max(freqs[0], f0 - 5.0)  # Clip to available frequency range
            f_max = min(freqs[-1], f0 + 5.0)

            # Initialize FOOOF for this harmonic
            fm_h = FOOOF(
                peak_width_limits=peak_width_limits,
                max_n_peaks=max_n_peaks,
                min_peak_height=min_peak_height,
                peak_threshold=peak_threshold,
                aperiodic_mode='fixed',
                verbose=False
            )

            # Fit FOOOF on this frequency window
            try:
                fm_h.fit(freqs, psd, (f_min, f_max))
            except Exception as e:
                # If fit fails for this harmonic, use NaN values
                harmonics.append(np.nan)
                powers.append(np.nan)
                bandwidths.append(np.nan)
                ap_offsets.append(np.nan)
                ap_exps.append(np.nan)
                r_squareds.append(np.nan)
                models.append(None)
                continue

            if not _has_model(fm_h):
                # No model fitted
                harmonics.append(np.nan)
                powers.append(np.nan)
                bandwidths.append(np.nan)
                ap_offsets.append(np.nan)
                ap_exps.append(np.nan)
                r_squareds.append(np.nan)
                models.append(None)
                continue

            # Extract peaks and match to this single canonical frequency
            peak_params_h = _get_peak_params(fm_h)
            harms_h, pows_h, bws_h, unmatched_h = match_peaks_to_canonical(
                peak_params_h,
                [f0],  # Single canonical frequency
                search_halfband=halfbands[i],  # Use this harmonic's halfband
                method=match_method,
                return_unmatched=True
            )

            # Collect all peaks and unmatched peaks from this harmonic's fit
            if len(peak_params_h) > 0:
                for row in peak_params_h:
                    all_peaks_combined.append({
                        'freq': float(row[0]),
                        'power': float(row[1]),
                        'bandwidth': float(row[2]),
                        'harmonic_window_idx': i,
                        'harmonic_window_center': f0
                    })
            unmatched_combined.extend(unmatched_h)

            # Extract aperiodic parameters for this harmonic
            ap_params_h = _get_aperiodic_params(fm_h)
            r_squared_h = _get_r_squared(fm_h)

            # Store results
            harmonics.append(harms_h[0])
            powers.append(pows_h[0])
            bandwidths.append(bws_h[0])
            ap_offsets.append(float(ap_params_h[0]))
            ap_exps.append(float(ap_params_h[1]))
            r_squareds.append(float(r_squared_h))
            models.append(fm_h)

        # Create result with per-harmonic β values
        result = FOOOFHarmonicResult(
            harmonics=harmonics,
            harmonic_powers=powers,
            harmonic_bandwidths=bandwidths,
            aperiodic_offset=ap_offsets,
            aperiodic_exponent=ap_exps,
            r_squared=r_squareds,
            model=models,
            freqs=freqs,
            psd=psd,
            per_harmonic_fits=True,
            unmatched_peaks=unmatched_combined,
            all_peaks=all_peaks_combined
        )

        return harmonics, result

    else:
        # Single global FOOOF fit (default behavior)
        fm = FOOOF(
            peak_width_limits=peak_width_limits,
            max_n_peaks=max_n_peaks,
            min_peak_height=min_peak_height,
            peak_threshold=peak_threshold,
            aperiodic_mode='fixed',  # 'fixed' = offset + exp, 'knee' adds knee param
            verbose=False
        )

        # Fit FOOOF model
        try:
            fm.fit(freqs, psd, freq_range)
        except Exception as e:
            raise RuntimeError(f"FOOOF fit failed: {e}")

        if not _has_model(fm):
            raise RuntimeError("FOOOF fitting produced no model")

        # Extract peaks and match to canonical harmonics
        peak_params = _get_peak_params(fm)
        harmonics, powers, bandwidths, unmatched = match_peaks_to_canonical(
            peak_params,
            f_can,
            search_halfband=search_halfband,
            method=match_method,
            return_unmatched=True
        )

        # Build all_peaks list from peak_params
        all_peaks = []
        if len(peak_params) > 0:
            for row in peak_params:
                all_peaks.append({
                    'freq': float(row[0]),
                    'power': float(row[1]),
                    'bandwidth': float(row[2])
                })

        # Extract aperiodic parameters
        ap_params = _get_aperiodic_params(fm)
        ap_offset, ap_exp = ap_params[0], ap_params[1]
        r_squared = _get_r_squared(fm)

        result = FOOOFHarmonicResult(
            harmonics=harmonics,
            harmonic_powers=powers,
            harmonic_bandwidths=bandwidths,
            aperiodic_offset=float(ap_offset),
            aperiodic_exponent=float(ap_exp),
            r_squared=float(r_squared),
            model=fm,
            freqs=freqs,
            psd=psd,
            per_harmonic_fits=False,
            unmatched_peaks=unmatched,
            all_peaks=all_peaks
        )

        return harmonics, result


def detect_harmonics_fooof_multichannel(
    records,
    channels: Sequence[str],
    fs: Optional[float] = None,
    time_col: str = 'Timestamp',
    window: Optional[Sequence[float]] = None,
    f_can: Sequence[float] = (7.83, 14.3, 20.8, 27.3, 33.8),
    freq_range: Tuple[float, float] = (1.0, 50.0),
    nperseg_sec: float = 4.0,
    overlap: float = 0.5,
    separate_fits: bool = False,
    **fooof_kwargs
) -> Dict[str, FOOOFHarmonicResult]:
    """
    Multi-channel FOOOF harmonic detection with optional separate fits.

    Parameters
    ----------
    records : DataFrame or dict
        EEG data records.
    channels : sequence of str
        Channel names to analyze.
    fs : float, optional
        Sampling frequency (Hz).
    time_col : str
        Timestamp column for fs inference.
    window : sequence of float, optional
        Time window [start, end] in seconds. If None, uses full recording.
    f_can : sequence of float
        Canonical Schumann harmonics (Hz).
    freq_range : tuple of float
        Frequency range for FOOOF (Hz).
    nperseg_sec : float
        Welch segment length (seconds).
    overlap : float
        Welch overlap fraction.
    separate_fits : bool
        If True, fit FOOOF separately per channel. If False, use median PSD.
    **fooof_kwargs
        Additional kwargs passed to detect_harmonics_fooof.

    Returns
    -------
    results : dict
        Keys: channel names (or 'combined').
        Values: FOOOFHarmonicResult objects.

    Examples
    --------
    >>> results = detect_harmonics_fooof_multichannel(
    ...     RECORDS, ['EEG.F4', 'EEG.O1', 'EEG.O2'], fs=128,
    ...     separate_fits=True
    ... )
    >>> for ch, res in results.items():
    ...     print(f"{ch}: {res.harmonics}, β={res.aperiodic_exponent:.3f}")
    """
    if not FOOOF_AVAILABLE:
        raise ImportError(
            "Spectral parameterization package not installed.\n"
            "Install with: pip install specparam (recommended)\n"
            "Or legacy: pip install fooof"
        )

    if fs is None:
        if hasattr(records, 'columns') and time_col in records.columns:
            fs = _infer_fs(records, time_col)
        else:
            raise ValueError("fs must be provided")

    results = {}

    if separate_fits:
        # Fit FOOOF separately for each channel
        for ch in channels:
            try:
                harms, result = detect_harmonics_fooof(
                    records, ch, fs=fs, time_col=time_col,
                    window=window,
                    f_can=f_can, freq_range=freq_range,
                    nperseg_sec=nperseg_sec, overlap=overlap,
                    **fooof_kwargs
                )
                results[ch] = result
            except Exception as e:
                print(f"Warning: FOOOF fit failed for {ch}: {e}")
                continue
    else:
        # Fit FOOOF on combined (median) PSD
        try:
            harms, result = detect_harmonics_fooof(
                records, channels, fs=fs, time_col=time_col,
                window=window,
                f_can=f_can, freq_range=freq_range,
                nperseg_sec=nperseg_sec, overlap=overlap,
                combine='median',
                **fooof_kwargs
            )
            results['combined'] = result
        except Exception as e:
            raise RuntimeError(f"FOOOF fit failed on combined PSD: {e}")

    return results


# ============================================================================
# Peak Matching Utilities
# ============================================================================

def match_peaks_to_canonical(
    peak_params: np.ndarray,
    f_can: Sequence[float],
    search_halfband: Union[float, Sequence[float]] = 0.8,
    method: str = 'distance',
    return_unmatched: bool = False
) -> Union[Tuple[List[float], List[float], List[float]],
           Tuple[List[float], List[float], List[float], List[Dict]]]:
    """
    Match FOOOF-detected peaks to canonical Schumann harmonics.

    Parameters
    ----------
    peak_params : ndarray
        FOOOF peak parameters, shape (n_peaks, 3): [freq, power, bandwidth].
    f_can : sequence of float
        Canonical harmonic frequencies to match (Hz).
    search_halfband : float or sequence of float
        Half-width search window(s) around canonical frequencies (Hz).
        If float: same halfband for all frequencies.
        If sequence: one halfband per frequency in f_can.
        Example: [0.5, 0.6, 0.8, 1.0, 1.2] for 5 harmonics.
    method : str
        Matching method when multiple peaks in window:
        - 'distance': Pick closest to canonical frequency (default)
        - 'power': Pick strongest (highest power)
        - 'average': Power-weighted average of all peaks in window
    return_unmatched : bool
        If True, also return peaks that were not matched to any canonical
        frequency (non-SR peaks). Default False for backward compatibility.

    Returns
    -------
    harmonics : list of float
        Matched harmonic frequencies (or canonical if no match found).
    powers : list of float
        Peak powers at harmonics (or NaN if no match).
    bandwidths : list of float
        Peak bandwidths (or NaN if no match).
    unmatched_peaks : list of dict (only if return_unmatched=True)
        List of peaks not matched to any canonical frequency.
        Each dict has keys: 'freq', 'power', 'bandwidth'.

    Examples
    --------
    >>> # Single halfband for all
    >>> harms, pows, bws = match_peaks_to_canonical(
    ...     peak_params, (7.83, 14.3, 20.8), search_halfband=0.8
    ... )

    >>> # Different halfband per harmonic
    >>> harms, pows, bws = match_peaks_to_canonical(
    ...     peak_params, (7.83, 14.3, 20.8),
    ...     search_halfband=[0.5, 0.6, 0.8],  # Narrower for H1
    ...     method='power'  # Pick strongest
    ... )

    >>> # Get unmatched (non-SR) peaks too
    >>> harms, pows, bws, non_sr = match_peaks_to_canonical(
    ...     peak_params, (7.83, 14.3, 20.8),
    ...     search_halfband=0.8,
    ...     return_unmatched=True
    ... )
    """
    # Validate method
    if method not in ('distance', 'power', 'average'):
        raise ValueError(f"method must be 'distance', 'power', or 'average', got '{method}'")

    # Convert search_halfband to list if scalar
    if isinstance(search_halfband, (int, float)):
        halfbands = [float(search_halfband)] * len(f_can)
    else:
        halfbands = list(search_halfband)
        if len(halfbands) != len(f_can):
            raise ValueError(
                f"search_halfband length ({len(halfbands)}) must match "
                f"f_can length ({len(f_can)})"
            )

    harmonics = []
    powers = []
    bandwidths = []

    # Track which peaks are matched (for return_unmatched)
    n_peaks = len(peak_params) if len(peak_params) > 0 else 0
    matched_mask = np.zeros(n_peaks, dtype=bool) if n_peaks > 0 else np.array([], dtype=bool)

    for f0, halfband in zip(f_can, halfbands):
        lo, hi = f0 - halfband, f0 + halfband

        if len(peak_params) == 0:
            # No peaks detected by FOOOF
            harmonics.append(float(f0))
            powers.append(np.nan)
            bandwidths.append(np.nan)
            continue

        # Find peaks within search window
        in_range = (peak_params[:, 0] >= lo) & (peak_params[:, 0] <= hi)

        if not np.any(in_range):
            # No peak near this canonical frequency
            harmonics.append(float(f0))
            powers.append(np.nan)
            bandwidths.append(np.nan)
        else:
            candidates = peak_params[in_range]
            candidate_indices = np.where(in_range)[0]

            if method == 'distance':
                # Pick peak closest to canonical frequency
                idx = np.argmin(np.abs(candidates[:, 0] - f0))
                best = candidates[idx]
                harmonics.append(float(best[0]))
                powers.append(float(best[1]))
                bandwidths.append(float(best[2]))
                # Mark this peak as matched
                matched_mask[candidate_indices[idx]] = True

            elif method == 'power':
                # Pick strongest peak (highest power)
                idx = np.argmax(candidates[:, 1])
                best = candidates[idx]
                harmonics.append(float(best[0]))
                powers.append(float(best[1]))
                bandwidths.append(float(best[2]))
                # Mark this peak as matched
                matched_mask[candidate_indices[idx]] = True

            elif method == 'average':
                # Power-weighted average of all peaks in window
                # Mark ALL peaks in window as matched (they're averaged together)
                matched_mask[candidate_indices] = True

                if len(candidates) == 1:
                    # Single peak - just use it
                    harmonics.append(float(candidates[0, 0]))
                    powers.append(float(candidates[0, 1]))
                    bandwidths.append(float(candidates[0, 2]))
                else:
                    # Multiple peaks - weighted average
                    peak_powers = candidates[:, 1]
                    weights = peak_powers / peak_powers.sum()

                    avg_freq = np.sum(candidates[:, 0] * weights)
                    avg_power = np.sum(candidates[:, 1] * weights)
                    avg_bw = np.sum(candidates[:, 2] * weights)

                    harmonics.append(float(avg_freq))
                    powers.append(float(avg_power))
                    bandwidths.append(float(avg_bw))

    if return_unmatched:
        # Collect peaks that weren't matched to any canonical frequency
        unmatched_peaks = []
        if n_peaks > 0:
            unmatched_indices = np.where(~matched_mask)[0]
            for idx in unmatched_indices:
                unmatched_peaks.append({
                    'freq': float(peak_params[idx, 0]),
                    'power': float(peak_params[idx, 1]),
                    'bandwidth': float(peak_params[idx, 2])
                })
        return harmonics, powers, bandwidths, unmatched_peaks

    return harmonics, powers, bandwidths


def extract_aperiodic_params(
    fm: FOOOF
) -> Tuple[float, float]:
    """
    Extract aperiodic (1/f) parameters from fitted FOOOF model.

    Parameters
    ----------
    fm : FOOOF
        Fitted FOOOF model object.

    Returns
    -------
    offset : float
        Aperiodic offset (y-intercept in log-log space).
    exponent : float
        Aperiodic exponent (1/f slope, negative = flatter spectrum).

    Examples
    --------
    >>> _, result = detect_harmonics_fooof(RECORDS, 'EEG.F4', fs=128)
    >>> offset, exp = extract_aperiodic_params(result.model)
    >>> print(f"1/f slope: {exp:.3f}")
    """
    if not _has_model(fm):
        raise ValueError("FOOOF model has not been fitted")

    ap_params = _get_aperiodic_params(fm)
    offset, exponent = ap_params[0], ap_params[1]
    return float(offset), float(exponent)


# ============================================================================
# Visualization
# ============================================================================

def plot_fooof_fit_with_harmonics(
    result: Union[FOOOF, FOOOFHarmonicResult],
    harmonics: Optional[Sequence[float]] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 5),
    freq_range: Optional[Tuple[float, float]] = None,
    log_power: bool = True,
    show_legend: bool = True
) -> plt.Figure:
    """
    Plot FOOOF fit with Schumann harmonic markers overlay.

    Parameters
    ----------
    result : FOOOF or FOOOFHarmonicResult
        Fitted FOOOF model or result object.
    harmonics : sequence of float, optional
        Harmonic frequencies to mark with vertical lines.
    title : str, optional
        Plot title.
    figsize : tuple of float
        Figure size (width, height).
    freq_range : tuple of float, optional
        (f_min, f_max) to zoom x-axis.
    log_power : bool
        If True, y-axis is log10 power.
    show_legend : bool
        Show legend.

    Returns
    -------
    fig : Figure
        Matplotlib figure object.

    Examples
    --------
    >>> _, result = detect_harmonics_fooof(RECORDS, 'EEG.F4', fs=128)
    >>> fig = plot_fooof_fit_with_harmonics(
    ...     result, harmonics=(7.83, 14.3, 20.8, 27.3),
    ...     title='F4 Schumann Harmonics — FOOOF Fit'
    ... )
    >>> fig.savefig('fooof_harmonics.png', dpi=300, bbox_inches='tight')

    >>> # To detect more peaks, set parameters in detect_harmonics_fooof()
    >>> _, result = detect_harmonics_fooof(
    ...     RECORDS, 'EEG.F4', fs=128,
    ...     max_n_peaks=15,
    ...     peak_threshold=0.75,
    ...     min_peak_height=0.01
    ... )
    >>> fig = plot_fooof_fit_with_harmonics(result)
    >>> plt.show()

    Notes
    -----
    To increase the number of peaks detected by FOOOF, pass sensitivity
    parameters (max_n_peaks, peak_threshold, min_peak_height,
    peak_width_limits) to detect_harmonics_fooof() before plotting.
    """
    if isinstance(result, FOOOFHarmonicResult):
        fm = result.model
        freqs_full = result.freqs  # Full PSD frequency range
        psd_full = result.psd
        if harmonics is None:
            harmonics = result.harmonics

        # Check if per-harmonic fits (result.model is a list)
        is_per_harmonic = isinstance(fm, list)
        if is_per_harmonic:
            # Keep all models for plotting
            models_list = [m for m in fm if m is not None]
            if len(models_list) == 0:
                raise ValueError("All FOOOF fits failed (no valid models)")
            # Use first model for reference in Panel A
            fm = models_list[0]
        else:
            models_list = None

    elif isinstance(result, FOOOF):
        fm = result
        freqs_full = _get_freqs(fm)
        psd_full = _get_power_spectrum(fm)
        is_per_harmonic = False
        models_list = None
    else:
        raise TypeError("result must be FOOOF or FOOOFHarmonicResult")

    if not _has_model(fm):
        raise ValueError("FOOOF model has not been fitted")

    # Validate freq_range parameter
    if freq_range is not None:
        if not isinstance(freq_range, (tuple, list)):
            raise TypeError(
                f"freq_range must be a tuple/list of 2 values (f_min, f_max), "
                f"got {type(freq_range).__name__}"
            )
        if len(freq_range) != 2:
            raise ValueError(
                f"freq_range must have exactly 2 values (f_min, f_max), got {len(freq_range)}. "
                f"NOTE: This parameter is for x-axis limits, not FOOOF fitting windows. "
                f"If you want per-harmonic FOOOF windows, use freq_ranges parameter in "
                f"detect_harmonics_fooof() or compare_psd_fooof(), not here."
            )
        try:
            f_min, f_max = freq_range
            if not (isinstance(f_min, (int, float)) and isinstance(f_max, (int, float))):
                raise TypeError
        except (TypeError, ValueError):
            raise TypeError(
                f"freq_range must be a tuple/list of 2 numbers (f_min, f_max), "
                f"got {freq_range}"
            )

    # Get model frequency range (may be subset of full PSD range)
    freqs_model = _get_freqs(fm)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Panel A: FOOOF fit overview ---
    ax = axes[0]

    # Original PSD (use full range if available, otherwise model range)
    if log_power:
        ax.semilogy(freqs_full, psd_full, color='0.5', lw=1.2, label='Original PSD', alpha=0.7)
    else:
        ax.plot(freqs_full, psd_full, color='0.5', lw=1.2, label='Original PSD', alpha=0.7)

    if is_per_harmonic and models_list is not None:
        # Plot FOOOF fits from ALL models
        for model_i in models_list:
            model_freqs_i = _get_freqs(model_i)
            model_spectrum_i = _get_fooofed_spectrum(model_i)
            ap_params_i = _get_aperiodic_params(model_i)

            if log_power:
                ax.semilogy(model_freqs_i, 10**model_spectrum_i, color='r', lw=1.8, alpha=0.8)
            else:
                ax.plot(model_freqs_i, 10**model_spectrum_i, color='r', lw=1.8, alpha=0.8)

            # Aperiodic component
            if len(ap_params_i) == 2:
                ap_i = ap_params_i[0] - ap_params_i[1] * np.log10(model_freqs_i)
            else:
                ap_i = ap_params_i[0] - np.log10(ap_params_i[2] + model_freqs_i**ap_params_i[1])

            if log_power:
                ax.semilogy(model_freqs_i, 10**ap_i, color='b', ls='--', lw=1.5, alpha=0.8)
            else:
                ax.plot(model_freqs_i, 10**ap_i, color='b', ls='--', lw=1.5, alpha=0.8)

        # Add legend labels (only once)
        ax.plot([], [], color='r', lw=1.8, label='FOOOF Full Model')
        ax.plot([], [], color='b', ls='--', lw=1.5, label='Aperiodic (1/f)')
    else:
        # Single model: plot as before
        freqs_model = _get_freqs(fm)
        model = _get_fooofed_spectrum(fm)
        if log_power:
            ax.semilogy(freqs_model, 10**model, color='r', lw=1.8, label='FOOOF Full Model')
        else:
            ax.plot(freqs_model, 10**model, color='r', lw=1.8, label='FOOOF Full Model')

        # Aperiodic fit (use model frequency range)
        # Compute aperiodic component from parameters
        ap_params = _get_aperiodic_params(fm)
        if len(ap_params) == 2:
            # No knee: log_power = offset - exponent * log(freqs)
            ap = ap_params[0] - ap_params[1] * np.log10(freqs_model)
        else:
            # With knee: log_power = offset - np.log10(knee + freqs^exponent)
            ap = ap_params[0] - np.log10(ap_params[2] + freqs_model**ap_params[1])

        if log_power:
            ax.semilogy(freqs_model, 10**ap, color='b', ls='--', lw=1.5, label='Aperiodic (1/f)')
        else:
            ax.plot(freqs_model, 10**ap, color='b', ls='--', lw=1.5, label='Aperiodic (1/f)')

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (μV²/Hz)' if not log_power else 'log₁₀ Power (μV²/Hz)')
    ax.set_title(title or 'FOOOF Spectral Parameterization')
    if show_legend:
        ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    if freq_range is not None:
        ax.set_xlim(freq_range)

    # --- Panel B: Periodic (oscillatory) component with harmonic markers ---
    ax2 = axes[1]

    if is_per_harmonic and models_list is not None:
        # Plot periodic component from ALL models
        for model in models_list:
            model_freqs_i = _get_freqs(model)
            model_spectrum_i = _get_fooofed_spectrum(model)
            ap_params_i = _get_aperiodic_params(model)

            if len(ap_params_i) == 2:
                ap_i = ap_params_i[0] - ap_params_i[1] * np.log10(model_freqs_i)
            else:
                ap_i = ap_params_i[0] - np.log10(ap_params_i[2] + model_freqs_i**ap_params_i[1])

            periodic_i = model_spectrum_i - ap_i

            if log_power:
                ax2.plot(model_freqs_i, 10**periodic_i, color='purple', lw=1.8, alpha=0.8)
            else:
                ax2.plot(model_freqs_i, periodic_i, color='purple', lw=1.8, alpha=0.8)

        # Add legend label (only once)
        ax2.plot([], [], color='purple', lw=1.8, label='Periodic (Oscillatory)')

        # Plot ONLY the matched harmonics (not all FOOOF peaks)
        if harmonics is not None and len(harmonics) > 0:
            # Get harmonic powers from result if available
            if isinstance(result, FOOOFHarmonicResult) and hasattr(result, 'harmonic_powers'):
                valid = ~np.isnan(result.harmonic_powers)
                hf = [harmonics[i] for i, v in enumerate(valid) if v]
                hp = [result.harmonic_powers[i] for i, v in enumerate(valid) if v]
            else:
                hf = list(harmonics)
                hp = None

            if hf and hp is not None:
                if log_power:
                    peak_y_values = 10**np.array(hp)
                else:
                    peak_y_values = np.array(hp)

                ax2.scatter(hf, peak_y_values, s=80, c='red',
                           marker='o', edgecolor='k', linewidth=1.5,
                           label='Matched Harmonics', zorder=5)

                # Add frequency labels to each peak
                for freq, y_val in zip(hf, peak_y_values):
                    ax2.text(freq, y_val * 0.8, f'{freq:.2f}',
                            ha='center', va='bottom', fontsize=8,
                            color='red', fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                     edgecolor='red', alpha=0.8, linewidth=0.5))
    else:
        # Single model: plot as before
        periodic = model - ap

        if log_power:
            ax2.plot(freqs_model, 10**periodic, color='purple', lw=1.8, label='Periodic (Oscillatory)')
        else:
            ax2.plot(freqs_model, periodic, color='purple', lw=1.8, label='Periodic (Oscillatory)')

        # Mark detected peaks
        peak_params = _get_peak_params(fm)
        if len(peak_params) > 0:
            peak_freqs = peak_params[:, 0]
            peak_powers = peak_params[:, 1]
            if log_power:
                peak_y_values = 10**peak_powers
                ax2.scatter(peak_freqs, peak_y_values, s=80, c='red',
                           marker='o', edgecolor='k', linewidth=1.5,
                           label='FOOOF Peaks', zorder=5)
            else:
                peak_y_values = peak_powers
                ax2.scatter(peak_freqs, peak_y_values, s=80, c='red',
                           marker='o', edgecolor='k', linewidth=1.5,
                           label='FOOOF Peaks', zorder=5)

            # Add frequency labels to each peak
            for freq, y_val in zip(peak_freqs, peak_y_values):
                ax2.text(freq, y_val * 0.8, f'{freq:.2f}',
                        ha='center', va='bottom', fontsize=8,
                        color='red', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                 edgecolor='red', alpha=0.8, linewidth=0.5))

    # Schumann harmonic markers
    if harmonics is not None:
        ylim = ax2.get_ylim()
        # Show harmonics across full freq_range if specified, otherwise model range
        if freq_range is not None:
            f_min, f_max = freq_range
            for f0 in harmonics:
                if f_min <= f0 <= f_max:
                    ax2.axvline(f0, color='k', ls='--', lw=1.0, alpha=0.6)
        else:
            for f0 in harmonics:
                if freqs_model[0] <= f0 <= freqs_model[-1]:
                    ax2.axvline(f0, color='k', ls='--', lw=1.0, alpha=0.6)
        # Restore ylim (axvline can expand it)
        ax2.set_ylim(ylim)

    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Periodic Power')
    ax2.set_title('Periodic Component (Peaks Only)')
    if show_legend:
        ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    if freq_range is not None:
        ax2.set_xlim(freq_range)

    # Add text annotation with fit stats
    if is_per_harmonic and models_list is not None:
        # Show range of β and R² values across all models
        all_exps = [_get_aperiodic_params(m)[1] for m in models_list]
        all_r2s = [_get_r_squared(m) for m in models_list]

        # Count matched harmonics
        if isinstance(result, FOOOFHarmonicResult) and hasattr(result, 'harmonic_powers'):
            n_harmonics = sum(~np.isnan(result.harmonic_powers))
        else:
            n_harmonics = len(harmonics) if harmonics is not None else 0

        exp_min, exp_max = min(all_exps), max(all_exps)
        r2_min, r2_max = min(all_r2s), max(all_r2s)

        txt = f"β = [{exp_min:.3f}..{exp_max:.3f}]\nR² = [{r2_min:.3f}..{r2_max:.3f}]\nn_harmonics = {n_harmonics}"
    else:
        # Single model: show single values
        ap_params = _get_aperiodic_params(fm)
        offset, exp = ap_params[0], ap_params[1]
        r2 = _get_r_squared(fm)
        peak_params = _get_peak_params(fm)
        txt = f"β = {exp:.3f}\nR² = {r2:.3f}\nn_peaks = {len(peak_params)}"

    ax2.text(0.98, 0.02, txt, transform=ax2.transAxes,
            fontsize=9, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()
    return fig


def compare_psd_fooof(
    records,
    channels: Union[str, Sequence[str]],
    fs: float,
    f_can: Sequence[float] = (7.83, 14.3, 20.8, 27.3, 33.8),
    figsize: Tuple[float, float] = (14, 5),
    freq_range: Tuple[float, float] = (1.0, 50.0),
    title: Optional[str] = None,
    max_n_peaks: int = 10,
    peak_threshold: float = 2.0,
    min_peak_height: float = 0.05,
    peak_width_limits: Tuple[float, float] = (1.0, 8.0),
    **fooof_kwargs
) -> Tuple[plt.Figure, FOOOFHarmonicResult]:
    """
    Side-by-side comparison: raw PSD peak detection vs FOOOF-based detection.

    Useful for validating that FOOOF improves harmonic detection over
    naive peak-finding on raw PSD.

    Parameters
    ----------
    records : DataFrame or dict
        EEG data records.
    channels : str or sequence of str
        Channel(s) to analyze.
    fs : float
        Sampling frequency (Hz).
    f_can : sequence of float
        Canonical Schumann harmonics (Hz).
    figsize : tuple of float
        Figure size.
    freq_range : tuple of float
        Frequency range to display (Hz).
    title : str, optional
        Overall figure title.
    max_n_peaks : int
        Maximum number of peaks for FOOOF to fit (default: 10).
    peak_threshold : float
        Peak detection threshold relative to aperiodic fit (default: 2.0).
    min_peak_height : float
        Minimum absolute peak height (default: 0.05).
    peak_width_limits : tuple of float
        (min_bw, max_bw) allowable peak bandwidth in Hz (default: (1.0, 8.0)).
    **fooof_kwargs
        Additional keyword arguments passed to detect_harmonics_fooof().

    Returns
    -------
    fig : Figure
        Matplotlib figure with 3 panels.
    result : FOOOFHarmonicResult
        FOOOF detection result.

    Examples
    --------
    >>> fig, result = compare_psd_fooof(
    ...     RECORDS, ['EEG.F4', 'EEG.O1'], fs=128,
    ...     f_can=(7.83, 14.3, 20.8, 27.3)
    ... )
    >>> plt.show()

    >>> # Increase sensitivity to detect more peaks
    >>> fig, result = compare_psd_fooof(
    ...     RECORDS, 'EEG.F4', fs=128,
    ...     f_can=(7.83, 14.3, 20.8, 27.3),
    ...     max_n_peaks=15,
    ...     peak_threshold=0.75,
    ...     min_peak_height=0.01
    ... )
    >>> plt.show()

    Notes
    -----
    When using freq_ranges or per_harmonic_fits=True, only the first FOOOF
    model is visualized. For full per-harmonic visualization, use
    detect_harmonics_fooof() and plot_fooof_fit_with_harmonics() separately.
    """
    # Run FOOOF detection
    harmonics, result = detect_harmonics_fooof(
        records, channels, fs=fs,
        f_can=f_can, freq_range=freq_range,
        max_n_peaks=max_n_peaks,
        peak_threshold=peak_threshold,
        min_peak_height=min_peak_height,
        peak_width_limits=peak_width_limits,
        **fooof_kwargs
    )

    freqs = result.freqs
    psd = result.psd
    fm = result.model

    # Check if per-harmonic fits (result.model is a list)
    is_per_harmonic = isinstance(fm, list)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # --- Panel A: Raw PSD with naive peak detection ---
    ax = axes[0]
    ax.semilogy(freqs, psd, color='0.3', lw=1.5, label='Raw PSD')

    # Naive approach: just look for max near canonical freqs
    naive_peaks = []
    for f0 in f_can:
        idx = np.argmin(np.abs(freqs - f0))
        naive_peaks.append((freqs[idx], psd[idx]))

    naive_f, naive_p = zip(*naive_peaks)
    ax.scatter(naive_f, naive_p, s=100, c='orange', marker='x',
              linewidth=2.5, label='Naive Peak (argmax)', zorder=5)

    for f0 in f_can:
        if freqs[0] <= f0 <= freqs[-1]:
            ax.axvline(f0, color='k', ls='--', lw=0.8, alpha=0.5)

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (μV²/Hz)')
    ax.set_title('A) Raw PSD — Naive Detection')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(freq_range)

    # --- Panel B: FOOOF periodic component ---
    ax2 = axes[1]

    if is_per_harmonic:
        # Plot periodic component from ALL models
        for model in fm:
            if model is None:
                continue
            model_freqs = _get_freqs(model)
            model_spectrum = _get_fooofed_spectrum(model)
            ap_params = _get_aperiodic_params(model)
            if len(ap_params) == 2:
                ap_spectrum = ap_params[0] - ap_params[1] * np.log10(model_freqs)
            else:
                ap_spectrum = ap_params[0] - np.log10(ap_params[2] + model_freqs**ap_params[1])
            periodic = model_spectrum - ap_spectrum
            ax2.plot(model_freqs, 10**periodic, color='purple', lw=1.5, alpha=0.8)
        # Add label for legend (only once)
        ax2.plot([], [], color='purple', lw=1.5, label='Periodic (FOOOF)')
    else:
        # Single model: plot as before
        model_freqs = _get_freqs(fm)
        model_spectrum = _get_fooofed_spectrum(fm)
        ap_params = _get_aperiodic_params(fm)
        if len(ap_params) == 2:
            ap_spectrum = ap_params[0] - ap_params[1] * np.log10(model_freqs)
        else:
            ap_spectrum = ap_params[0] - np.log10(ap_params[2] + model_freqs**ap_params[1])
        periodic = model_spectrum - ap_spectrum
        ax2.plot(model_freqs, 10**periodic, color='purple', lw=1.5, label='Periodic (FOOOF)')

    # FOOOF peaks - plot all detected harmonics across full range
    if len(result.harmonics) > 0:
        valid = ~np.isnan(result.harmonic_powers)
        hf = [result.harmonics[i] for i, v in enumerate(valid) if v]
        hp = [result.harmonic_powers[i] for i, v in enumerate(valid) if v]
        if hf:
            hp_linear = 10**np.array(hp)
            ax2.scatter(hf, hp_linear, s=100, c='red', marker='o',
                       edgecolor='k', linewidth=1.5, label='FOOOF Peaks', zorder=5)

            # Add frequency labels slightly below each peak
            for freq, y_val in zip(hf, hp_linear):
                ax2.text(freq, y_val * 0.7, f'{freq:.2f}',
                        ha='center', va='top', fontsize=8,
                        color='red', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                 edgecolor='red', alpha=0.8, linewidth=0.5))

    # Show canonical harmonic lines across full requested range
    for f0 in f_can:
        if freq_range[0] <= f0 <= freq_range[1]:
            ax2.axvline(f0, color='k', ls='--', lw=0.8, alpha=0.5)

    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Periodic Power')
    ax2.set_title('B) FOOOF Periodic — Detected Harmonics')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(freq_range)

    # --- Panel C: Comparison table ---
    ax3 = axes[2]
    ax3.axis('off')

    table_data = [['Harmonic', 'Canonical', 'Naive', 'FOOOF', 'Δ (Hz)']]
    for i, fc in enumerate(f_can):
        nf = naive_f[i]
        ff = result.harmonics[i] if i < len(result.harmonics) else np.nan
        delta = ff - fc if not np.isnan(ff) else np.nan

        table_data.append([
            f'H{i+1}',
            f'{fc:.2f}',
            f'{nf:.2f}',
            f'{ff:.2f}' if not np.isnan(ff) else '—',
            f'{delta:+.3f}' if not np.isnan(delta) else '—'
        ])

    # Add aperiodic info
    table_data.append(['', '', '', '', ''])
    table_data.append(['Metric', '', 'Value', '', ''])

    # Handle per-harmonic values (lists) vs single values (floats)
    if isinstance(result.aperiodic_exponent, list):
        # Per-harmonic fits: show range
        beta_min = min(result.aperiodic_exponent)
        beta_max = max(result.aperiodic_exponent)
        table_data.append(['Exponent β', '', f'[{beta_min:.3f}..{beta_max:.3f}]', '', ''])

        offset_min = min(result.aperiodic_offset)
        offset_max = max(result.aperiodic_offset)
        table_data.append(['Offset', '', f'[{offset_min:.3f}..{offset_max:.3f}]', '', ''])

        r2_min = min(result.r_squared)
        r2_max = max(result.r_squared)
        table_data.append(['R²', '', f'[{r2_min:.3f}..{r2_max:.3f}]', '', ''])
    else:
        # Single fit: show single values
        table_data.append(['Exponent β', '', f'{result.aperiodic_exponent:.3f}', '', ''])
        table_data.append(['Offset', '', f'{result.aperiodic_offset:.3f}', '', ''])
        table_data.append(['R²', '', f'{result.r_squared:.3f}', '', ''])

    tbl = ax3.table(cellText=table_data, cellLoc='center', loc='center',
                   colWidths=[0.15, 0.2, 0.2, 0.2, 0.2])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2)

    # Style header row
    for i in range(5):
        tbl[(0, i)].set_facecolor('#40466e')
        tbl[(0, i)].set_text_props(weight='bold', color='w')

    ax3.set_title('C) Comparison Summary', pad=20)

    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)

    fig.tight_layout()
    return fig, result


def plot_fooof_periodic(
    result: FOOOFHarmonicResult,
    f_can: Sequence[float],
    freq_range: Tuple[float, float] = (1.0, 50.0),
    figsize: Tuple[float, float] = (10, 6),
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot FOOOF periodic component with detected harmonics (standalone, larger).

    This is Panel B from compare_psd_fooof() as a standalone chart.

    Parameters
    ----------
    result : FOOOFHarmonicResult
        FOOOF detection result from detect_harmonics_fooof().
    f_can : sequence of float
        Canonical Schumann harmonic frequencies (Hz).
    freq_range : tuple of float
        Frequency range to display (Hz).
    figsize : tuple of float
        Figure size (width, height).
    title : str, optional
        Plot title.

    Returns
    -------
    fig : Figure
        Matplotlib figure object.

    Examples
    --------
    >>> harmonics, result = detect_harmonics_fooof(
    ...     RECORDS, 'EEG.F4', fs=128, f_can=CANON, freq_ranges=FREQ_RANGES
    ... )
    >>> fig = plot_fooof_periodic(result, CANON, freq_range=(5, 40))
    >>> plt.show()
    """
    fm = result.model
    is_per_harmonic = isinstance(fm, list)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if is_per_harmonic:
        # Plot periodic component from ALL models
        for model in fm:
            if model is None:
                continue
            model_freqs = _get_freqs(model)
            model_spectrum = _get_fooofed_spectrum(model)
            ap_params = _get_aperiodic_params(model)

            if len(ap_params) == 2:
                ap_spectrum = ap_params[0] - ap_params[1] * np.log10(model_freqs)
            else:
                ap_spectrum = ap_params[0] - np.log10(ap_params[2] + model_freqs**ap_params[1])

            periodic = model_spectrum - ap_spectrum
            ax.plot(model_freqs, 10**periodic, color='purple', lw=2.0, alpha=0.8)

        # Add label for legend (only once)
        ax.plot([], [], color='purple', lw=2.0, label='Periodic (FOOOF)')
    else:
        # Single model
        model_freqs = _get_freqs(fm)
        model_spectrum = _get_fooofed_spectrum(fm)
        ap_params = _get_aperiodic_params(fm)
        if len(ap_params) == 2:
            ap_spectrum = ap_params[0] - ap_params[1] * np.log10(model_freqs)
        else:
            ap_spectrum = ap_params[0] - np.log10(ap_params[2] + model_freqs**ap_params[1])
        periodic = model_spectrum - ap_spectrum
        ax.plot(model_freqs, 10**periodic, color='purple', lw=2.0, label='Periodic (FOOOF)')

    # FOOOF peaks - plot all detected harmonics
    if len(result.harmonics) > 0:
        valid = ~np.isnan(result.harmonic_powers)
        hf = [result.harmonics[i] for i, v in enumerate(valid) if v]
        hp = [result.harmonic_powers[i] for i, v in enumerate(valid) if v]
        if hf:
            hp_linear = 10**np.array(hp)
            ax.scatter(hf, hp_linear, s=120, c='red', marker='o',
                       edgecolor='k', linewidth=2.0, label='FOOOF Peaks', zorder=5)

            # Add frequency labels slightly below each peak
            for freq, y_val in zip(hf, hp_linear):
                ax.text(freq, y_val * 0.7, f'{freq:.2f}',
                        ha='center', va='top', fontsize=10,
                        color='red', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                 edgecolor='red', alpha=0.9, linewidth=0.8))

    # Show canonical harmonic lines
    for f0 in f_can:
        if freq_range[0] <= f0 <= freq_range[1]:
            ax.axvline(f0, color='k', ls='--', lw=1.0, alpha=0.5)

    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Periodic Power', fontsize=12)
    ax.set_title(title or 'FOOOF Periodic — Detected Harmonics', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(freq_range)

    fig.tight_layout()
    return fig


# ============================================================================
# Integration with existing pipeline
# ============================================================================

def fooof_refine_existing_harmonics(
    harmonics_in: Sequence[float],
    records,
    channels: Union[str, Sequence[str]],
    fs: float,
    search_halfband: float = 0.5,
    **fooof_kwargs
) -> List[float]:
    """
    Refine existing harmonic estimates using FOOOF.

    Takes coarse harmonic estimates (e.g., from existing Welch-based
    detection) and refines them using FOOOF's peak detection.

    Parameters
    ----------
    harmonics_in : sequence of float
        Initial harmonic frequency estimates (Hz).
    records : DataFrame or dict
        EEG data.
    channels : str or sequence of str
        Channel(s) to use.
    fs : float
        Sampling frequency (Hz).
    search_halfband : float
        Search window half-width around each initial estimate (Hz).
    **fooof_kwargs
        Additional kwargs for detect_harmonics_fooof.

    Returns
    -------
    harmonics_refined : list of float
        FOOOF-refined harmonic frequencies (Hz).

    Examples
    --------
    >>> # Start with existing detection
    >>> coarse_harms = estimate_sr_harmonics(RECORDS, 'EEG.F4', fs=128)
    >>> # Refine with FOOOF
    >>> refined_harms = fooof_refine_existing_harmonics(
    ...     coarse_harms, RECORDS, 'EEG.F4', fs=128
    ... )
    """
    refined, result = detect_harmonics_fooof(
        records, channels, fs=fs,
        f_can=harmonics_in,
        search_halfband=search_halfband,
        **fooof_kwargs
    )

    return refined


# ============================================================================
# Convenience wrappers
# ============================================================================

def quick_fooof_summary(
    records,
    channels: Union[str, Sequence[str]],
    fs: float,
    window: Optional[Sequence[float]] = None,
    f_can: Sequence[float] = (7.83, 14.3, 20.8, 27.3, 33.8),
) -> Dict[str, Any]:
    """
    Quick one-liner to get FOOOF harmonic summary.

    Returns a simple dict with harmonics and key aperiodic params.

    Parameters
    ----------
    records : DataFrame or dict
        EEG data.
    channels : str or sequence of str
        Channel(s) to analyze.
    fs : float
        Sampling frequency (Hz).
    window : sequence of float, optional
        Time window [start, end] in seconds. If None, uses full recording.
    f_can : sequence of float
        Canonical Schumann harmonics (Hz).

    Returns
    -------
    summary : dict
        Keys: 'harmonics', 'exponent', 'offset', 'r_squared', 'result'.

    Examples
    --------
    >>> summary = quick_fooof_summary(RECORDS, 'EEG.F4', fs=128)
    >>> print(f"Harmonics: {summary['harmonics']}")
    >>> print(f"1/f slope: {summary['exponent']:.3f}")

    >>> # With time window
    >>> summary = quick_fooof_summary(RECORDS, 'EEG.F4', fs=128, window=[100, 200])
    >>> print(f"Harmonics in 100-200s: {summary['harmonics']}")
    """
    harmonics, result = detect_harmonics_fooof(
        records, channels, fs=fs, window=window, f_can=f_can
    )

    return {
        'harmonics': harmonics,
        'exponent': result.aperiodic_exponent,
        'offset': result.aperiodic_offset,
        'r_squared': result.r_squared,
        'result': result
    }


# ============================================================================
# Module-level check
# ============================================================================

def check_fooof_available() -> bool:
    """Check if FOOOF is installed and importable."""
    return FOOOF_AVAILABLE


if __name__ == '__main__':
    print("FOOOF/SpecParam Harmonics Module")
    print("=" * 50)
    if FOOOF_AVAILABLE:
        print(f"✓ Spectral parameterization package is installed ({PACKAGE_NAME})")
        if PACKAGE_NAME == 'fooof':
            print("  ⚠️  Note: fooof is deprecated. Consider upgrading to specparam:")
            print("     pip install specparam")
    else:
        print("✗ Spectral parameterization package not found.")
        print("  Install with: pip install specparam (recommended)")
        print("  Or legacy: pip install fooof")
    print("\nUsage:")
    print("  from lib.fooof_harmonics import detect_harmonics_fooof")
    print("  harmonics, result = detect_harmonics_fooof(")
    print("      RECORDS, ['EEG.F4', 'EEG.O1'], fs=128)")
    print("  print(f'Harmonics: {harmonics}')")
    print("  print(f'Exponent: {result.aperiodic_exponent:.3f}')")

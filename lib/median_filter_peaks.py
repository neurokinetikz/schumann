"""
Median-Filter Peak Extraction (Critic's Method)
================================================

Implements the Welch PSD + median-filter 1/f subtraction pipeline
used in the independent replication, for direct comparison against
FOOOF parametric spectral fitting.

Usage:
    from median_filter_peaks import extract_peaks_welch_median, compare_extraction_methods

    peaks_mf = extract_peaks_welch_median(signal, fs=128)
    peaks_fooof = extract_peaks_fooof_single(signal, fs=128)
    comparison = compare_extraction_methods(signal, fs=128)
"""

import numpy as np
from scipy.signal import welch, find_peaks
from scipy.ndimage import median_filter
from typing import Dict, Tuple, Optional, List


def extract_peaks_welch_median(signal, fs, nperseg=512, median_kernel_frac=0.33,
                                peak_prominence=0.5, freq_range=(1, 45)):
    """
    Critic's peak extraction pipeline.

    1. Welch PSD
    2. Log-transform
    3. Subtract median-filtered log-PSD (kernel = median_kernel_frac × n_freqs)
    4. Find peaks in residual above prominence threshold

    Parameters
    ----------
    signal : np.ndarray
        1D EEG signal
    fs : float
        Sampling rate (Hz)
    nperseg : int
        Welch segment length
    median_kernel_frac : float
        Fraction of spectrum length for median filter kernel
    peak_prominence : float
        Minimum prominence for peak detection (in log10 units)
    freq_range : tuple
        (min_freq, max_freq) to search for peaks

    Returns
    -------
    peak_freqs : np.ndarray
        Peak frequencies (Hz)
    peak_powers : np.ndarray
        Peak powers (corrected, log10 units)
    psd_freqs : np.ndarray
        Full frequency axis
    psd_log : np.ndarray
        Log10 PSD
    psd_corrected : np.ndarray
        Median-subtracted log PSD
    """
    signal = np.asarray(signal, dtype=float)
    signal = signal[np.isfinite(signal)]

    if len(signal) < nperseg:
        nperseg = len(signal) // 2

    f, pxx = welch(signal, fs=fs, nperseg=nperseg)

    # Log-transform
    psd_log = np.log10(np.maximum(pxx, 1e-30))

    # Median filter for aperiodic estimate
    kernel_size = max(3, int(len(f) * median_kernel_frac))
    if kernel_size % 2 == 0:
        kernel_size += 1
    aperiodic = median_filter(psd_log, size=kernel_size)

    # Corrected spectrum (residual)
    psd_corrected = psd_log - aperiodic

    # Find peaks in frequency range
    freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
    f_sub = f[freq_mask]
    corr_sub = psd_corrected[freq_mask]

    if len(corr_sub) < 3:
        return np.array([]), np.array([]), f, psd_log, psd_corrected

    peak_idx, props = find_peaks(corr_sub, prominence=peak_prominence)

    peak_freqs = f_sub[peak_idx]
    peak_powers = corr_sub[peak_idx]

    return peak_freqs, peak_powers, f, psd_log, psd_corrected


def extract_peaks_fooof_single(signal, fs, nperseg=512, freq_range=(1, 45),
                                max_n_peaks=15, peak_threshold=2.0,
                                min_peak_height=0.05,
                                peak_width_limits=(0.5, 8.0)):
    """
    FOOOF/specparam peak extraction for a single signal.

    Wrapper that handles both specparam (2.0+) and legacy fooof imports.

    Returns
    -------
    peak_freqs : np.ndarray
    peak_powers : np.ndarray
    psd_freqs : np.ndarray
    psd_log : np.ndarray
    """
    from scipy.signal import welch as _welch

    signal = np.asarray(signal, dtype=float)
    signal = signal[np.isfinite(signal)]

    if len(signal) < nperseg:
        nperseg = len(signal) // 2

    f, pxx = _welch(signal, fs=fs, nperseg=nperseg)

    freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
    f_fit = f[freq_mask]
    pxx_fit = pxx[freq_mask]

    # Try specparam first, fall back to fooof
    try:
        from specparam import SpectralModel
        fm = SpectralModel(
            peak_width_limits=list(peak_width_limits),
            max_n_peaks=max_n_peaks,
            min_peak_height=min_peak_height,
            peak_threshold=peak_threshold,
        )
        fm.fit(f_fit, pxx_fit)
        peaks = fm.peak_params_
    except ImportError:
        try:
            from fooof import FOOOF
            fm = FOOOF(
                peak_width_limits=list(peak_width_limits),
                max_n_peaks=max_n_peaks,
                min_peak_height=min_peak_height,
                peak_threshold=peak_threshold,
            )
            fm.fit(f_fit, pxx_fit)
            peaks = fm.peak_params_
        except ImportError:
            raise ImportError("Neither specparam nor fooof is installed.")

    if peaks is not None and len(peaks) > 0:
        peak_freqs = peaks[:, 0]
        peak_powers = peaks[:, 1]
    else:
        peak_freqs = np.array([])
        peak_powers = np.array([])

    psd_log = np.log10(np.maximum(pxx, 1e-30))
    return peak_freqs, peak_powers, f, psd_log


def compare_extraction_methods(signal, fs, nperseg=512, freq_range=(1, 45),
                                **kwargs):
    """
    Run both extraction methods on the same signal, return comparison.

    Returns
    -------
    dict with keys:
        'median_filter': {'freqs', 'powers', 'n_peaks'}
        'fooof': {'freqs', 'powers', 'n_peaks'}
        'n_peaks_ratio': fooof/median_filter
        'common_peaks': frequencies found by both (within 0.5 Hz)
    """
    mf_freqs, mf_powers, _, _, _ = extract_peaks_welch_median(
        signal, fs, nperseg=nperseg, freq_range=freq_range, **kwargs
    )

    try:
        ff_freqs, ff_powers, _, _ = extract_peaks_fooof_single(
            signal, fs, nperseg=nperseg, freq_range=freq_range
        )
    except ImportError:
        ff_freqs, ff_powers = np.array([]), np.array([])

    # Find common peaks (within 0.5 Hz)
    common = []
    for mf in mf_freqs:
        for ff in ff_freqs:
            if abs(mf - ff) < 0.5:
                common.append((mf, ff))
                break

    n_mf = len(mf_freqs)
    n_ff = len(ff_freqs)

    return {
        'median_filter': {'freqs': mf_freqs, 'powers': mf_powers, 'n_peaks': n_mf},
        'fooof': {'freqs': ff_freqs, 'powers': ff_powers, 'n_peaks': n_ff},
        'n_peaks_ratio': n_ff / n_mf if n_mf > 0 else float('inf'),
        'common_peaks': common,
        'n_common': len(common),
        'jaccard': len(common) / (n_mf + n_ff - len(common)) if (n_mf + n_ff) > 0 else 0,
    }

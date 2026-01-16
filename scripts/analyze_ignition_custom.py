#!/usr/bin/env python3
"""
Custom EEG Ignition Detection Analysis
======================================
Standalone script to detect "ignition" events in EEG data based on
Schumann Resonance (~7.83 Hz) band power fluctuations.

This implements the detection from scratch without using lib modules.
"""

import warnings
import sys
import os
import io
from contextlib import contextmanager
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import zscore, ttest_ind
from scipy.optimize import OptimizeWarning
from typing import List, Tuple, Dict


@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout (for hiding FOOOF print warnings)."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


class CaptureStdout:
    """Context manager to capture stdout to both console and buffer."""
    def __init__(self):
        self.buffer = io.StringIO()
        self.original_stdout = None

    def __enter__(self):
        self.original_stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, *args):
        sys.stdout = self.original_stdout

    def write(self, text):
        """Write to both original stdout and buffer."""
        self.original_stdout.write(text)
        self.buffer.write(text)

    def flush(self):
        """Flush both streams."""
        self.original_stdout.flush()

    def getvalue(self):
        """Get captured content."""
        return self.buffer.getvalue()

# FOOOF/SpecParam import
try:
    from specparam import SpectralModel
    FOOOF_AVAILABLE = True
    FOOOF_PACKAGE = 'specparam'
except ImportError:
    try:
        from fooof import FOOOF as SpectralModel
        FOOOF_AVAILABLE = True
        FOOOF_PACKAGE = 'fooof'
    except ImportError:
        FOOOF_AVAILABLE = False
        FOOOF_PACKAGE = None
        SpectralModel = None

# Non-SR peak clustering import
try:
    from lib.non_sr_clustering import NonSRPeakCollector, NonSRPeak
    NON_SR_CLUSTERING_AVAILABLE = True
except ImportError:
    NON_SR_CLUSTERING_AVAILABLE = False
    NonSRPeakCollector = None
    NonSRPeak = None

# Surrogate analysis parameters
N_SURROGATES = 200  # Number of phase-shuffled surrogates for null distribution

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_FILE = "data/test schumann_EPOCX_111270_2023.04.23T14.50.35.05.00.md.pm.bp.csv"
FS = 128  # Sampling rate (Hz)

# Schumann Resonance parameters
SR_CENTER = 7.6  # Hz (fundamental)
SR_BANDWIDTH = 0.6  # Hz (half-bandwidth for fundamental, used for detection)

# Per-harmonic frequencies and bandwidths
SR_LABELS =   ['sr1', 'sr1.5', 'sr2p', 'sr2o', 'sr2.5', 'sr3',  'sr4', 'sr5']
SR_HARMONICS = [7.6,  9.5,     12,     13.75,   15.5,    20.0,   25.0,  32.0]  # 6 harmonics
SR_HALF_BW =   [0.6,  0.7,     0.7,    0.75,    0.85,    1.0,    2.0,   2.0]  # Per-harmonic half-bandwidths (widened f1/f2)

# Create lookup dict for easy access
SR_BW_DICT = dict(zip(SR_HARMONICS, SR_HALF_BW))

# Per-harmonic FOOOF fitting windows (10 Hz minimum for reliable fitting)
# Windows shifted to reduce alpha interference for f1/f2
SR_FREQ_RANGES = [
    (2, 12),    # f0: 7.8 Hz region (Theta-Alpha)
    (5, 15),   # f3: 20.0 Hz region (Low Beta)
    (7, 17),   # f5: 32.0 Hz region (High Beta)
    (9,19),
    (10,20),
    (15,25),
    (20,30),
    (25,35) 
]

# Flag to enable per-harmonic FOOOF fitting
USE_PER_HARMONIC_FOOOF = True

# Non-SR Peak Collection and Clustering
COLLECT_NON_SR_PEAKS = True  # Enable collection of non-SR peaks during FOOOF analysis
NON_SR_CLUSTER_METHOD = 'auto'  # 'kmeans', 'dbscan', 'hierarchical', or 'auto'
NON_SR_MIN_PEAKS_FOR_CLUSTERING = 5  # Minimum peaks required for clustering

# FOOOF match method: 'distance' (closest), 'power' (strongest), 'average' (centroid)
FOOOF_MATCH_METHOD = 'average'

# FOOOF verbose output for debugging peak detection
FOOOF_VERBOSE = False

# Skip coherence null control (time-consuming permutation test)
SKIP_COHERENCE_NULL = True

# Also run global (1-50 Hz) FOOOF on events to capture non-SR peaks
RUN_GLOBAL_FOOOF_ON_EVENTS = True

# CSV export settings for bulk analysis
EXPORT_CSV = True
EXPORT_DIR = "exports"

# Detection parameters
Z_THRESHOLD = 3.0  # Z-score threshold for detection (increased from 2.5)
MIN_ISI_SEC = 2.0  # Minimum inter-stimulus interval (seconds)
WINDOW_SEC = 20.0  # Window size around onset (seconds)
MERGE_GAP_SEC = 5.0  # Merge windows closer than this

# EEG channels
EEG_CHANNELS = ['EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5',  'EEG.P7',
                'EEG.O1', 'EEG.O2', 'EEG.P8',  'EEG.FC6', 'EEG.F4',
                'EEG.F8', 'EEG.AF4'] # 'EEG.T7','EEG.T8',


# =============================================================================
# SIGNAL PROCESSING FUNCTIONS
# =============================================================================

def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
    """Design Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = max(0.001, min(lowcut / nyq, 0.99))
    high = max(low + 0.001, min(highcut / nyq, 0.999))
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float,
                    fs: float, order: int = 4) -> np.ndarray:
    """Apply zero-phase bandpass filter."""
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return signal.filtfilt(b, a, data, axis=-1)


def compute_envelope(data: np.ndarray) -> np.ndarray:
    """Compute amplitude envelope using Hilbert transform."""
    analytic = signal.hilbert(data, axis=-1)
    return np.abs(analytic)


def smooth_signal(data: np.ndarray, window_sec: float, fs: float) -> np.ndarray:
    """Apply Hanning window smoothing."""
    n_samples = max(1, int(window_sec * fs))
    if n_samples > 1:
        window = np.hanning(n_samples)
        window /= window.sum()
        return np.convolve(data, window, mode='same')
    return data


def compute_kuramoto_r(phases: np.ndarray) -> float:
    """Compute Kuramoto order parameter R (phase synchrony)."""
    # phases: shape (n_channels, n_samples)
    mean_phase = np.mean(np.exp(1j * phases), axis=0)
    return float(np.mean(np.abs(mean_phase)))


def compute_msc(eeg_data: np.ndarray, fs: float, freq_band: Tuple[float, float],
                nperseg: int = None) -> float:
    """
    Compute Mean Squared Coherence (MSC) across all channel pairs in a frequency band.

    MSC measures the linear correlation between signals at each frequency.
    Values range from 0 (no coherence) to 1 (perfect coherence).

    Args:
        eeg_data: (n_channels, n_samples) array
        fs: Sampling rate
        freq_band: (f_low, f_high) frequency range for averaging MSC
        nperseg: Segment length for coherence calculation (default: fs)

    Returns:
        Mean MSC value across all channel pairs in the specified frequency band
    """
    n_channels, n_samples = eeg_data.shape
    if n_channels < 2:
        return np.nan

    if nperseg is None:
        nperseg = min(int(fs), n_samples // 2)

    # Compute pairwise coherence
    msc_values = []
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            try:
                freqs, coh = signal.coherence(eeg_data[i], eeg_data[j],
                                               fs=fs, nperseg=nperseg)
                # Average coherence in the frequency band
                mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
                if np.any(mask):
                    band_msc = np.mean(coh[mask])
                    msc_values.append(band_msc)
            except Exception:
                continue

    if len(msc_values) == 0:
        return np.nan

    return float(np.mean(msc_values))


def compute_plv(eeg_data: np.ndarray, fs: float, freq_band: Tuple[float, float]) -> float:
    """
    Compute Phase Locking Value (PLV) across all channel pairs in a frequency band.

    PLV measures the consistency of phase differences between signals.
    Values range from 0 (random phase) to 1 (perfect phase locking).

    Unlike MSC which measures amplitude coherence, PLV is purely phase-based
    and is insensitive to amplitude fluctuations.

    Args:
        eeg_data: (n_channels, n_samples) array
        fs: Sampling rate
        freq_band: (f_low, f_high) frequency range for bandpass filtering

    Returns:
        Mean PLV value across all channel pairs
    """
    from scipy.signal import hilbert, butter, filtfilt

    n_channels, n_samples = eeg_data.shape
    if n_channels < 2:
        return np.nan

    # Bandpass filter to the frequency band of interest
    nyq = fs / 2
    f_low = max(freq_band[0], 0.5)  # Minimum 0.5 Hz
    f_high = min(freq_band[1], nyq * 0.95)

    if f_low >= f_high:
        return np.nan

    try:
        # Design bandpass filter
        b, a = butter(3, [f_low / nyq, f_high / nyq], btype='band')

        # Filter each channel and extract phase via Hilbert transform
        phases = np.zeros((n_channels, n_samples))
        for ch in range(n_channels):
            filtered = filtfilt(b, a, eeg_data[ch])
            analytic = hilbert(filtered)
            phases[ch] = np.angle(analytic)

        # Compute pairwise PLV
        plv_values = []
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                # Phase difference
                phase_diff = phases[i] - phases[j]
                # PLV = |mean(exp(i * phase_diff))|
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                plv_values.append(plv)

        if len(plv_values) == 0:
            return np.nan

        return float(np.mean(plv_values))

    except Exception:
        return np.nan


def compute_plv_multiband(eeg_data: np.ndarray, fs: float,
                          harmonics: List[float] = None,
                          half_bandwidths: List[float] = None) -> float:
    """
    Compute PLV across multiple SR harmonic bands and average.

    This addresses the limitation of single-band PLV by:
    1. Computing PLV at each SR harmonic band
    2. Using wider bandwidths (2x the half_bandwidth) for stable phase estimation
    3. Averaging across harmonics with valid PLV

    Args:
        eeg_data: (n_channels, n_samples) array
        fs: Sampling rate
        harmonics: List of center frequencies (default: SR_HARMONICS)
        half_bandwidths: Per-harmonic half-bandwidths (default: SR_HALF_BW)

    Returns:
        Mean PLV across all harmonic bands
    """
    if harmonics is None:
        harmonics = SR_HARMONICS
    if half_bandwidths is None:
        half_bandwidths = SR_HALF_BW

    # Ensure we have matching lengths
    n_harmonics = min(len(harmonics), len(half_bandwidths))

    plv_values = []
    for i in range(n_harmonics):
        f_center = harmonics[i]
        # Use 2x the half-bandwidth for wider band (more stable phase estimation)
        half_bw = half_bandwidths[i] * 1.5  # Widen band for Hilbert stability
        f_low = f_center - half_bw
        f_high = f_center + half_bw

        # Skip if band is too narrow or invalid
        if f_high - f_low < 1.0:  # Minimum 1 Hz bandwidth
            continue

        plv = compute_plv(eeg_data, fs, (f_low, f_high))
        if not np.isnan(plv):
            plv_values.append(plv)

    if len(plv_values) == 0:
        return np.nan

    return float(np.mean(plv_values))


def compute_msc_multiband(eeg_data: np.ndarray, fs: float,
                          harmonics: List[float] = None,
                          half_bandwidths: List[float] = None) -> float:
    """
    Compute MSC across multiple SR harmonic bands and average.

    Args:
        eeg_data: (n_channels, n_samples) array
        fs: Sampling rate
        harmonics: List of center frequencies (default: SR_HARMONICS)
        half_bandwidths: Per-harmonic half-bandwidths (default: SR_HALF_BW)

    Returns:
        Mean MSC across all harmonic bands
    """
    if harmonics is None:
        harmonics = SR_HARMONICS
    if half_bandwidths is None:
        half_bandwidths = SR_HALF_BW

    n_harmonics = min(len(harmonics), len(half_bandwidths))

    msc_values = []
    for i in range(n_harmonics):
        f_center = harmonics[i]
        half_bw = half_bandwidths[i]
        f_low = f_center - half_bw
        f_high = f_center + half_bw

        msc = compute_msc(eeg_data, fs, (f_low, f_high))
        if not np.isnan(msc):
            msc_values.append(msc)

    if len(msc_values) == 0:
        return np.nan

    return float(np.mean(msc_values))


# =============================================================================
# FOOOF-BASED HARMONIC ESTIMATION
# =============================================================================

def estimate_harmonics_fooof(eeg_data: np.ndarray, fs: float,
                              canonical_freqs: List[float],
                              search_bw: float = 1.0,
                              per_harmonic_bw: List[float] = None,
                              freq_range: Tuple[float, float] = (1, 50),
                              nperseg_sec: float = 4.0) -> Dict:
    """
    Use FOOOF/SpecParam to estimate actual harmonic frequencies from data.

    FOOOF separates periodic peaks from aperiodic 1/f background, providing
    more accurate peak detection than raw PSD.

    Args:
        eeg_data: (n_channels, n_samples) array
        fs: Sampling rate
        canonical_freqs: Expected SR frequencies [7.6, 12.0, 13.75, ...]
        search_bw: Default half-bandwidth for matching (used if per_harmonic_bw not provided)
        per_harmonic_bw: Per-harmonic half-bandwidths (same length as canonical_freqs)
        freq_range: Frequency range for FOOOF fitting
        nperseg_sec: Segment length for PSD estimation

    Returns:
        Dict with detected harmonics, aperiodic params, peak info, and
        all detected peaks for characterization
    """
    # Use per-harmonic bandwidths if provided, otherwise use default
    if per_harmonic_bw is None:
        per_harmonic_bw = [search_bw] * len(canonical_freqs)
    if not FOOOF_AVAILABLE:
        print("  WARNING: FOOOF/specparam not available, using canonical freqs")
        return {
            'detected_freqs': canonical_freqs,
            'canonical_freqs': canonical_freqs,
            'peaks': [],
            'aperiodic_exponent': np.nan,
            'aperiodic_offset': np.nan,
            'r_squared': np.nan,
            'fooof_available': False,
            'all_peaks': []
        }

    # Compute PSD (mean across channels)
    mean_eeg = np.mean(eeg_data, axis=0)
    nperseg = int(nperseg_sec * fs)
    freqs, psd = signal.welch(mean_eeg, fs=fs, nperseg=nperseg)

    # Limit to freq_range
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    freqs_fit = freqs[mask]
    psd_fit = psd[mask]

    # Fit FOOOF model with INCREASED SENSITIVITY
    fm = SpectralModel(
        peak_width_limits=(0.5, 12.0),  # Wider range (was 1.0-8.0)
        max_n_peaks=20,                  # More peaks (was 10)
        min_peak_height=0.01,           # More sensitive (was 0.1)
        peak_threshold=1.0,             # Lower threshold (was 2.0)
        aperiodic_mode='fixed'
    )

    try:
        with suppress_stdout():
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*peak width limit.*')
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                warnings.simplefilter('ignore', OptimizeWarning)
                fm.fit(freqs_fit, psd_fit, freq_range)
    except Exception as e:
        print(f"  WARNING: FOOOF fit failed: {e}")
        return {
            'detected_freqs': canonical_freqs,
            'canonical_freqs': canonical_freqs,
            'peaks': [],
            'aperiodic_exponent': np.nan,
            'aperiodic_offset': np.nan,
            'r_squared': np.nan,
            'fooof_available': True,
            'fit_failed': True,
            'all_peaks': []
        }

    # Extract peak parameters using robust multi-version API handling
    peak_params, ap_params, r_squared = _extract_fooof_params(fm)

    # Compute aperiodic fit for SNR calculation
    aperiodic_fit = None
    if ap_params is not None and len(ap_params) >= 2:
        # 1/f model: offset - exponent * log10(freq)
        offset, exponent = float(ap_params[0]), float(ap_params[1])
        aperiodic_fit = 10 ** (offset - exponent * np.log10(freqs_fit))

    # Match detected peaks to canonical frequencies (using per-harmonic bandwidths)
    detected_freqs = []
    peak_info = []

    for idx, canon_f in enumerate(canonical_freqs):
        # Get the bandwidth for this harmonic
        harmonic_bw = per_harmonic_bw[idx] if idx < len(per_harmonic_bw) else search_bw

        if canon_f > freq_range[1]:
            detected_freqs.append(canon_f)
            peak_info.append({'canonical': canon_f, 'detected': canon_f,
                            'power': np.nan, 'bandwidth': np.nan, 'matched': False,
                            'snr': np.nan, 'prominence': np.nan, 'rank': np.nan,
                            'search_bw': harmonic_bw})
            continue

        # Find closest peak within this harmonic's search bandwidth
        best_peak = None
        best_dist = harmonic_bw

        if len(peak_params) > 0:
            for peak in peak_params:
                peak_f = peak[0]
                dist = abs(peak_f - canon_f)
                if dist < best_dist:
                    best_dist = dist
                    best_peak = peak

        if best_peak is not None:
            # Compute SNR (peak power above aperiodic background)
            snr = np.nan
            if aperiodic_fit is not None:
                freq_idx = np.argmin(np.abs(freqs_fit - best_peak[0]))
                bg_power = aperiodic_fit[freq_idx]
                snr = 10 ** best_peak[1] / bg_power if bg_power > 0 else np.nan

            # Compute prominence (relative to median peak height)
            prominence = np.nan
            if len(peak_params) > 1:
                all_heights = peak_params[:, 1]
                median_h = np.median(all_heights)
                prominence = best_peak[1] / median_h if median_h > 0 else np.nan

            # Compute rank (sorted by power)
            rank = np.nan
            if len(peak_params) > 0:
                sorted_by_power = sorted(peak_params, key=lambda x: -x[1])
                for i, p in enumerate(sorted_by_power):
                    if np.allclose(p, best_peak):
                        rank = i + 1
                        break

            detected_freqs.append(float(best_peak[0]))
            peak_info.append({
                'canonical': canon_f,
                'detected': float(best_peak[0]),
                'power': float(best_peak[1]),
                'bandwidth': float(best_peak[2]),
                'matched': True,
                'snr': float(snr) if not np.isnan(snr) else np.nan,
                'prominence': float(prominence) if not np.isnan(prominence) else np.nan,
                'rank': int(rank) if not np.isnan(rank) else np.nan,
                'search_bw': harmonic_bw
            })
        else:
            detected_freqs.append(canon_f)
            peak_info.append({'canonical': canon_f, 'detected': canon_f,
                            'power': np.nan, 'bandwidth': np.nan, 'matched': False,
                            'snr': np.nan, 'prominence': np.nan, 'rank': np.nan,
                            'search_bw': harmonic_bw})

    # Characterize ALL peaks (not just SR-matched ones) using per-harmonic bandwidths
    all_peaks_characterized = characterize_all_peaks(
        peak_params, canonical_freqs, per_harmonic_bw, freqs_fit, aperiodic_fit
    )

    # Compute SR Specificity Index
    sr_specificity = compute_sr_specificity(all_peaks_characterized)

    return {
        'detected_freqs': detected_freqs,
        'canonical_freqs': canonical_freqs,
        'peaks': peak_info,
        'aperiodic_exponent': float(ap_params[1]) if len(ap_params) > 1 else np.nan,
        'aperiodic_offset': float(ap_params[0]) if len(ap_params) > 0 else np.nan,
        'r_squared': float(r_squared) if r_squared is not None else np.nan,
        'fooof_available': True,
        'all_peaks': peak_params.tolist() if len(peak_params) > 0 else [],
        'all_peaks_characterized': all_peaks_characterized,
        'sr_specificity': sr_specificity,
        'freqs': freqs_fit,
        'psd': psd_fit,
        'aperiodic_fit': aperiodic_fit
    }


def estimate_harmonics_fooof_per_harmonic(eeg_data: np.ndarray, fs: float,
                                           canonical_freqs: List[float],
                                           freq_ranges: List[Tuple[float, float]],
                                           per_harmonic_bw: List[float] = None,
                                           nperseg_sec: float = 4.0,
                                           match_method: str = 'distance',
                                           verbose: bool = False) -> Dict:
    """
    Fit separate FOOOF model for each SR harmonic region.

    Instead of a single global 1-50 Hz fit, this fits individual FOOOF models
    for each canonical frequency in a region-specific window. This provides:
    - Per-harmonic aperiodic parameters (β, offset)
    - Region-specific 1/f subtraction for better peak detection
    - Insight into spectral slope variation across frequency bands

    Args:
        eeg_data: (n_channels, n_samples) array
        fs: Sampling rate
        canonical_freqs: Expected SR frequencies [7.8, 12.0, 13.75, ...]
        freq_ranges: Per-harmonic FOOOF windows [(f_min, f_max), ...]
        per_harmonic_bw: Per-harmonic half-bandwidths for peak matching
        nperseg_sec: Segment length for PSD estimation
        match_method: Peak matching method:
            'distance' - closest peak to canonical frequency
            'power' - strongest peak in search window
            'average' - power-weighted centroid of all peaks in window
        verbose: Print detailed peak detection info

    Returns:
        Dict with per-harmonic detection results including:
        - detected_freqs: List of detected frequencies
        - per_harmonic_beta: List of 1/f exponents per harmonic
        - per_harmonic_r2: List of R² values per harmonic
        - peaks: List of peak info dicts
    """
    if per_harmonic_bw is None:
        per_harmonic_bw = [0.8] * len(canonical_freqs)

    if not FOOOF_AVAILABLE:
        print("  WARNING: FOOOF/specparam not available")
        return {
            'detected_freqs': canonical_freqs,
            'canonical_freqs': canonical_freqs,
            'peaks': [],
            'per_harmonic_beta': [np.nan] * len(canonical_freqs),
            'per_harmonic_offset': [np.nan] * len(canonical_freqs),
            'per_harmonic_r2': [np.nan] * len(canonical_freqs),
            'fooof_available': False,
            'per_harmonic': True
        }

    # Compute full PSD once (mean across channels)
    mean_eeg = np.mean(eeg_data, axis=0)
    nperseg = int(nperseg_sec * fs)
    freqs, psd = signal.welch(mean_eeg, fs=fs, nperseg=nperseg)

    # Results containers
    detected_freqs = []
    peaks_info = []
    per_harmonic_beta = []
    per_harmonic_offset = []
    per_harmonic_r2 = []
    all_models = []

    for idx, (canon_f, (f_min, f_max), harmonic_bw) in enumerate(
            zip(canonical_freqs, freq_ranges, per_harmonic_bw)):

        # Skip if canonical frequency is outside Nyquist
        if canon_f > fs / 2 - 1:
            detected_freqs.append(canon_f)
            peaks_info.append({
                'canonical': canon_f, 'detected': canon_f,
                'power': np.nan, 'bandwidth': np.nan, 'matched': False,
                'beta': np.nan, 'r2': np.nan
            })
            per_harmonic_beta.append(np.nan)
            per_harmonic_offset.append(np.nan)
            per_harmonic_r2.append(np.nan)
            all_models.append(None)
            continue

        # Limit PSD to this harmonic's frequency range
        mask = (freqs >= f_min) & (freqs <= f_max)
        freqs_fit = freqs[mask]
        psd_fit = psd[mask]

        if len(freqs_fit) < 10:  # Need enough frequency bins
            detected_freqs.append(canon_f)
            peaks_info.append({
                'canonical': canon_f, 'detected': canon_f,
                'power': np.nan, 'bandwidth': np.nan, 'matched': False,
                'beta': np.nan, 'r2': np.nan
            })
            per_harmonic_beta.append(np.nan)
            per_harmonic_offset.append(np.nan)
            per_harmonic_r2.append(np.nan)
            all_models.append(None)
            continue

        # Fit FOOOF model for this frequency region
        # Parameters tuned for better f1/f2 detection:
        # - peak_threshold: 0.3 (lower than 0.5 for weak session peaks)
        # - min_peak_height: 0.01 (captures peaks with log-power > 0.01)
        # - max_n_peaks: 7 (allow multiple peaks per window)
        fm = SpectralModel(
            peak_width_limits=(0.5, 8.0),
            max_n_peaks=7,
            min_peak_height=0.01,  # Lower threshold for session-level detection
            peak_threshold=0.3,     # More sensitive than event-level (0.5)
            aperiodic_mode='fixed'
        )

        try:
            with suppress_stdout():
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='.*peak width limit.*')
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    warnings.simplefilter('ignore', OptimizeWarning)
                    fm.fit(freqs_fit, psd_fit, (f_min, f_max))
        except Exception as e:
            detected_freqs.append(canon_f)
            peaks_info.append({
                'canonical': canon_f, 'detected': canon_f,
                'power': np.nan, 'bandwidth': np.nan, 'matched': False,
                'beta': np.nan, 'r2': np.nan, 'error': str(e)
            })
            per_harmonic_beta.append(np.nan)
            per_harmonic_offset.append(np.nan)
            per_harmonic_r2.append(np.nan)
            all_models.append(None)
            continue

        # Extract parameters
        peak_params, ap_params, r_squared = _extract_fooof_params(fm)

        # Store aperiodic params for this harmonic
        beta = float(ap_params[1]) if len(ap_params) > 1 else np.nan
        offset = float(ap_params[0]) if len(ap_params) > 0 else np.nan
        r2 = float(r_squared) if r_squared is not None else np.nan

        per_harmonic_beta.append(beta)
        per_harmonic_offset.append(offset)
        per_harmonic_r2.append(r2)
        all_models.append(fm)

        # Compute aperiodic fit for SNR calculation
        aperiodic_fit = None
        if len(ap_params) >= 2:
            aperiodic_fit = 10 ** (offset - beta * np.log10(freqs_fit))

        # Verbose output for debugging
        if verbose:
            print(f"  f{idx} [{f_min}-{f_max} Hz]: {len(peak_params)} peaks found")
            for i, p in enumerate(peak_params):
                print(f"    Peak {i}: {p[0]:.2f} Hz, power={p[1]:.3f}, bw={p[2]:.2f}")

        # Get all peaks within search bandwidth
        candidates = []
        if len(peak_params) > 0:
            for peak in peak_params:
                peak_f = peak[0]
                if abs(peak_f - canon_f) < harmonic_bw:
                    candidates.append(peak)

        if verbose and len(candidates) > 0:
            print(f"    Candidates for f{idx} ({canon_f} Hz ±{harmonic_bw}): {len(candidates)}")

        # Match peak using specified method
        best_peak = None
        if len(candidates) == 0:
            best_peak = None
        elif match_method == 'distance':
            # Pick closest peak to canonical frequency
            best_idx = np.argmin([abs(p[0] - canon_f) for p in candidates])
            best_peak = candidates[best_idx]
        elif match_method == 'power':
            # Pick strongest (highest power) peak in search window
            best_idx = np.argmax([p[1] for p in candidates])
            best_peak = candidates[best_idx]
        elif match_method == 'average':
            # Power-weighted centroid of all peaks in search window
            freqs_cand = np.array([p[0] for p in candidates])
            powers_cand = np.array([p[1] for p in candidates])
            bandwidths_cand = np.array([p[2] for p in candidates])

            # Convert log-power to linear for proper weighting
            linear_powers = 10 ** powers_cand
            weights = linear_powers / linear_powers.sum()

            # Compute power-weighted centroid
            avg_freq = np.sum(freqs_cand * weights)
            avg_power = np.log10(np.sum(linear_powers))  # Total power (log)
            avg_bw = np.sum(bandwidths_cand * weights)

            best_peak = (avg_freq, avg_power, avg_bw)
            if verbose:
                print(f"    Average method: centroid={avg_freq:.2f} Hz from {len(candidates)} peaks")
        else:
            # Default to distance
            best_idx = np.argmin([abs(p[0] - canon_f) for p in candidates])
            best_peak = candidates[best_idx]

        # Track peak counts for this harmonic window
        n_peaks_total = len(peak_params)
        n_peaks_in_sr = len(candidates)

        # Store all peaks and identify non-SR peaks
        all_peaks_in_window = []
        non_sr_peaks = []
        for peak in peak_params:
            peak_f, peak_pow, peak_bw = peak[0], peak[1], peak[2]
            is_in_sr = abs(peak_f - canon_f) < harmonic_bw
            peak_info = {
                'freq': float(peak_f),
                'power': float(peak_pow),
                'bandwidth': float(peak_bw),
                'in_sr_window': is_in_sr
            }
            all_peaks_in_window.append(peak_info)
            if not is_in_sr:
                non_sr_peaks.append(peak_info)

        if best_peak is not None:
            # Compute SNR (peak power above aperiodic background)
            snr = np.nan
            if aperiodic_fit is not None:
                freq_idx = np.argmin(np.abs(freqs_fit - best_peak[0]))
                bg_power = aperiodic_fit[freq_idx]
                snr = 10 ** best_peak[1] / bg_power if bg_power > 0 else np.nan

            detected_freqs.append(float(best_peak[0]))
            peaks_info.append({
                'canonical': canon_f,
                'detected': float(best_peak[0]),
                'power': float(best_peak[1]),
                'bandwidth': float(best_peak[2]),
                'matched': True,
                'beta': beta,
                'r2': r2,
                'snr': float(snr) if not np.isnan(snr) else np.nan,
                'freq_range': (f_min, f_max),
                'n_peaks_total': n_peaks_total,
                'n_peaks_in_sr': n_peaks_in_sr,
                'all_peaks': all_peaks_in_window,
                'non_sr_peaks': non_sr_peaks
            })
        else:
            detected_freqs.append(canon_f)
            peaks_info.append({
                'canonical': canon_f, 'detected': canon_f,
                'power': np.nan, 'bandwidth': np.nan, 'matched': False,
                'beta': beta, 'r2': r2, 'snr': np.nan,
                'freq_range': (f_min, f_max),
                'n_peaks_total': n_peaks_total,
                'n_peaks_in_sr': n_peaks_in_sr,
                'all_peaks': all_peaks_in_window,
                'non_sr_peaks': non_sr_peaks
            })

    # Compute summary statistics
    valid_betas = [b for b in per_harmonic_beta if not np.isnan(b)]
    valid_r2s = [r for r in per_harmonic_r2 if not np.isnan(r)]
    n_matched = sum(1 for p in peaks_info if p.get('matched', False))

    return {
        'detected_freqs': detected_freqs,
        'canonical_freqs': canonical_freqs,
        'peaks': peaks_info,
        'per_harmonic_beta': per_harmonic_beta,
        'per_harmonic_offset': per_harmonic_offset,
        'per_harmonic_r2': per_harmonic_r2,
        'mean_beta': np.mean(valid_betas) if valid_betas else np.nan,
        'beta_range': (min(valid_betas), max(valid_betas)) if valid_betas else (np.nan, np.nan),
        'mean_r2': np.mean(valid_r2s) if valid_r2s else np.nan,
        'n_matched': n_matched,
        'n_total': len(canonical_freqs),
        'fooof_available': True,
        'per_harmonic': True,
        'models': all_models,
        'freqs': freqs,
        'psd': psd
    }


def _extract_fooof_params(fm) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Extract peak params, aperiodic params, and r_squared from FOOOF model.
    Handles multiple specparam/fooof API versions robustly.

    Returns:
        peak_params: (N, 3) array of [center_freq, power, bandwidth]
        ap_params: 1D array [offset, exponent] or [offset, knee, exponent]
        r_squared: model fit quality (0-1)
    """
    peak_params = np.array([]).reshape(0, 3)
    ap_params = np.array([])
    r_squared = np.nan

    # =====================================================================
    # METHOD 1: specparam 2.0+ API (fm.results.get_params)
    # =====================================================================
    try:
        if hasattr(fm, 'results') and fm.results is not None:
            res = fm.results
            # Get peak params via results.get_params('peak')
            if hasattr(res, 'get_params'):
                try:
                    pp = res.get_params('peak')
                    if pp is not None and len(pp) > 0:
                        peak_params = np.atleast_2d(np.array(pp))
                except Exception:
                    pass

                try:
                    ap = res.get_params('aperiodic')
                    if ap is not None and len(ap) > 0:
                        ap_params = np.array(ap).flatten()
                except Exception:
                    pass

            # Get r_squared from metrics.results
            if hasattr(res, 'metrics') and res.metrics is not None:
                if hasattr(res.metrics, 'results') and res.metrics.results:
                    r2 = res.metrics.results.get('gof_rsquared')
                    if r2 is not None:
                        r_squared = float(r2)
    except Exception:
        pass

    # =====================================================================
    # METHOD 2: specparam 1.x / fooof legacy (underscore attributes)
    # =====================================================================
    if len(peak_params) == 0:
        try:
            if hasattr(fm, 'peak_params_') and fm.peak_params_ is not None:
                pp = np.array(fm.peak_params_)
                if pp.size > 0:
                    peak_params = np.atleast_2d(pp)
        except Exception:
            pass

    if len(ap_params) == 0:
        try:
            if hasattr(fm, 'aperiodic_params_') and fm.aperiodic_params_ is not None:
                ap_params = np.array(fm.aperiodic_params_).flatten()
        except Exception:
            pass

    if np.isnan(r_squared):
        try:
            if hasattr(fm, 'r_squared_') and fm.r_squared_ is not None:
                r_squared = float(fm.r_squared_)
        except Exception:
            pass

    # =====================================================================
    # METHOD 3: fm.get_params method (specparam 1.x style)
    # =====================================================================
    if len(peak_params) == 0:
        try:
            pp = fm.get_params('peak_params')
            if pp is not None and len(pp) > 0:
                peak_params = np.atleast_2d(np.array(pp))
        except Exception:
            pass

    if len(ap_params) == 0:
        try:
            ap = fm.get_params('aperiodic_params')
            if ap is not None and len(ap) > 0:
                ap_params = np.array(ap).flatten()
        except Exception:
            pass

    if np.isnan(r_squared):
        try:
            r = fm.get_params('r_squared')
            if r is not None:
                r_squared = float(r)
        except Exception:
            pass

    # =====================================================================
    # Ensure correct output shapes
    # =====================================================================
    # peak_params should be (N, 3) for [freq, height, width]
    if len(peak_params) > 0:
        peak_params = np.atleast_2d(peak_params)
        if peak_params.ndim != 2 or peak_params.shape[1] != 3:
            peak_params = np.array([]).reshape(0, 3)
    else:
        peak_params = np.array([]).reshape(0, 3)

    # ap_params should be 1D array
    if len(ap_params) == 0:
        ap_params = np.array([])
    else:
        ap_params = np.atleast_1d(ap_params).flatten()

    return peak_params, ap_params, r_squared


def characterize_all_peaks(peak_params: np.ndarray, canonical_freqs: List[float],
                            per_harmonic_bw: List[float], freqs_fit: np.ndarray,
                            aperiodic_fit: np.ndarray) -> List[Dict]:
    """
    Characterize ALL detected peaks with metrics for comparison.

    Args:
        peak_params: (N, 3) array of [center_freq, power, bandwidth]
        canonical_freqs: SR harmonic frequencies
        per_harmonic_bw: Per-harmonic half-bandwidths for matching
        freqs_fit: Frequency array from PSD
        aperiodic_fit: Aperiodic background power

    Returns list of dicts with:
        - freq: Peak frequency
        - power: Peak power (log)
        - bandwidth: Peak width
        - snr: Signal-to-noise ratio vs aperiodic background
        - prominence: Relative height vs median
        - rank: Power rank (1 = highest)
        - sr_match: Which SR harmonic it matches (or None)
        - is_sr: Boolean if within any SR window
    """
    if len(peak_params) == 0:
        return []

    # Sort by power (descending) for ranking
    sorted_indices = np.argsort(-peak_params[:, 1])

    all_heights = peak_params[:, 1]
    median_h = np.median(all_heights) if len(all_heights) > 0 else 1.0

    characterized = []
    for rank, idx in enumerate(sorted_indices):
        peak = peak_params[idx]
        freq, height, width = peak[0], peak[1], peak[2]

        # SNR calculation
        snr = np.nan
        if aperiodic_fit is not None and len(freqs_fit) > 0:
            freq_idx = np.argmin(np.abs(freqs_fit - freq))
            bg_power = aperiodic_fit[freq_idx]
            snr = 10 ** height / bg_power if bg_power > 0 else np.nan

        # Prominence (relative to median height)
        prominence = height / median_h if median_h > 0 else np.nan

        # Check SR match using per-harmonic bandwidths
        sr_match = None
        matched_bw = None
        for i, canon_f in enumerate(canonical_freqs):
            bw = per_harmonic_bw[i] if i < len(per_harmonic_bw) else 1.0
            if abs(freq - canon_f) <= bw:
                sr_match = canon_f
                matched_bw = bw
                break

        characterized.append({
            'freq': float(freq),
            'power': float(height),
            'bandwidth': float(width),
            'snr': float(snr) if not np.isnan(snr) else np.nan,
            'prominence': float(prominence),
            'rank': rank + 1,
            'sr_match': sr_match,
            'is_sr': sr_match is not None,
            'matched_bw': matched_bw
        })

    return characterized


def compute_sr_specificity(all_peaks_characterized: List[Dict]) -> float:
    """
    Compute SR Specificity Index: fraction of total peak power in SR bands.

    SR Specificity = sum(SR peak powers) / sum(all peak powers)

    Range: 0-1, higher = more power concentrated at SR frequencies
    """
    if not all_peaks_characterized:
        return np.nan

    # Use linear power (10^log_power)
    sr_power = 0.0
    total_power = 0.0

    for peak in all_peaks_characterized:
        linear_power = 10 ** peak['power']
        total_power += linear_power
        if peak['is_sr']:
            sr_power += linear_power

    if total_power > 0:
        return sr_power / total_power
    return np.nan


def estimate_event_harmonics_fooof(eeg_data: np.ndarray, fs: float,
                                    start_sec: float, end_sec: float,
                                    canonical_freqs: List[float],
                                    per_harmonic_bw: List[float] = None,
                                    search_bw: float = 1.0) -> Dict:
    """
    Estimate harmonics for a single event window using FOOOF.

    Args:
        eeg_data: (n_channels, n_samples) array
        fs: Sampling rate
        start_sec, end_sec: Event window boundaries
        canonical_freqs: SR harmonic frequencies
        per_harmonic_bw: Per-harmonic half-bandwidths
        search_bw: Default bandwidth (used if per_harmonic_bw not provided)
    """
    i0 = int(start_sec * fs)
    i1 = int(end_sec * fs)
    i1 = min(i1, eeg_data.shape[1])

    if i1 - i0 < fs * 2:  # Need at least 2 seconds
        return {
            'detected_freqs': canonical_freqs,
            'error': 'Window too short for FOOOF'
        }

    segment = eeg_data[:, i0:i1]

    # Use shorter nperseg for event windows
    duration = (i1 - i0) / fs
    nperseg_sec = min(2.0, duration / 2)

    return estimate_harmonics_fooof(
        segment, fs, canonical_freqs,
        search_bw=search_bw,
        per_harmonic_bw=per_harmonic_bw,
        nperseg_sec=nperseg_sec
    )


def estimate_event_harmonics_fooof_per_harmonic(eeg_data: np.ndarray, fs: float,
                                                  start_sec: float, end_sec: float,
                                                  canonical_freqs: List[float],
                                                  freq_ranges: List[Tuple[float, float]],
                                                  per_harmonic_bw: List[float] = None) -> Dict:
    """
    Estimate harmonics for a single event using per-harmonic FOOOF fitting.

    Args:
        eeg_data: (n_channels, n_samples) array
        fs: Sampling rate
        start_sec, end_sec: Event window boundaries
        canonical_freqs: SR harmonic frequencies
        freq_ranges: Per-harmonic FOOOF windows [(f_min, f_max), ...]
        per_harmonic_bw: Per-harmonic half-bandwidths for peak matching
    """
    i0 = int(start_sec * fs)
    i1 = int(end_sec * fs)
    i1 = min(i1, eeg_data.shape[1])

    if i1 - i0 < fs * 2:  # Need at least 2 seconds
        return {
            'detected_freqs': canonical_freqs,
            'peaks': [],
            'error': 'Window too short for FOOOF',
            'per_harmonic': True
        }

    segment = eeg_data[:, i0:i1]

    # Use shorter nperseg for event windows
    duration = (i1 - i0) / fs
    nperseg_sec = min(2.0, duration / 2)

    return estimate_harmonics_fooof_per_harmonic(
        segment, fs, canonical_freqs, freq_ranges,
        per_harmonic_bw=per_harmonic_bw,
        nperseg_sec=nperseg_sec,
        match_method=FOOOF_MATCH_METHOD,
        verbose=FOOOF_VERBOSE
    )


def print_fooof_results(result: Dict, label: str = "Session", verbose: bool = True):
    """Print FOOOF harmonic estimation results with peak comparison table."""
    print(f"\n  {label} FOOOF Harmonic Estimation:")

    if not result.get('fooof_available', False):
        print("    FOOOF not available - using canonical frequencies")
        return

    if result.get('fit_failed', False):
        print("    FOOOF fit failed - using canonical frequencies")
        return

    # Basic model info
    ap_exp = result.get('aperiodic_exponent', np.nan)
    r_sq = result.get('r_squared', np.nan)
    print(f"    Aperiodic exponent (1/f slope): {ap_exp:.3f}")
    print(f"    Model R²: {r_sq:.3f}")

    # SR Harmonics summary
    print(f"\n    SR Harmonic Detection:")
    sr_matched = 0
    for peak in result.get('peaks', []):
        if peak.get('matched', False):
            sr_matched += 1
            shift = peak['detected'] - peak['canonical']
            snr_str = f"SNR:{peak.get('snr', np.nan):.1f}" if not np.isnan(peak.get('snr', np.nan)) else "SNR:N/A"
            rank_str = f"Rank:{peak.get('rank', '?')}" if peak.get('rank') else "Rank:N/A"
            print(f"      {peak['canonical']:.2f} Hz -> {peak['detected']:.2f} Hz "
                  f"(shift: {shift:+.2f} Hz, power: {peak['power']:.2f}, {snr_str}, {rank_str})")
        else:
            print(f"      {peak['canonical']:.2f} Hz -> [not detected]")

    print(f"    SR harmonics matched: {sr_matched}/{len(result.get('peaks', []))}")

    # SR Specificity Index
    sr_spec = result.get('sr_specificity', np.nan)
    if not np.isnan(sr_spec):
        print(f"    SR Specificity Index: {sr_spec:.3f} ({100*sr_spec:.1f}% of peak power in SR bands)")

    # Full peak comparison table (if verbose and peaks exist)
    all_peaks = result.get('all_peaks_characterized', [])
    if verbose and len(all_peaks) > 0:
        print(f"\n    --- ALL Detected Peaks (sorted by power) ---")
        print(f"    {'#':>3}  {'Freq(Hz)':>8}  {'Power':>6}  {'SNR':>6}  {'Prom':>5}  {'BW(Hz)':>6}  {'SR_Match':>10}")
        print(f"    {'-'*3}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*6}  {'-'*10}")

        for peak in all_peaks:
            snr_str = f"{peak['snr']:.2f}" if not np.isnan(peak.get('snr', np.nan)) else "N/A"
            prom_str = f"{peak['prominence']:.2f}" if not np.isnan(peak.get('prominence', np.nan)) else "N/A"
            sr_str = f"f{SR_HARMONICS.index(peak['sr_match'])}" if peak.get('sr_match') else "-"
            marker = "*" if peak.get('is_sr', False) else " "
            print(f"    {peak['rank']:>3}{marker} {peak['freq']:>8.2f}  {peak['power']:>6.3f}  {snr_str:>6}  "
                  f"{prom_str:>5}  {peak['bandwidth']:>6.2f}  {sr_str:>10}")

        # Summary statistics
        n_peaks = len(all_peaks)
        n_sr = sum(1 for p in all_peaks if p.get('is_sr', False))
        n_non_sr = n_peaks - n_sr
        print(f"\n    Total peaks: {n_peaks} ({n_sr} SR, {n_non_sr} non-SR)")

        # Top 3 peaks summary
        if n_peaks >= 3:
            top3_sr = sum(1 for p in all_peaks[:3] if p.get('is_sr', False))
            print(f"    Top-3 by power: {top3_sr} are SR-related")


def print_fooof_results_per_harmonic(result: Dict, label: str = "Session"):
    """Print per-harmonic FOOOF results with regional β values."""
    print(f"\n  {label} PER-HARMONIC FOOOF Results:")

    if not result.get('fooof_available', False):
        print("    FOOOF not available")
        return

    if not result.get('per_harmonic', False):
        print("    Not a per-harmonic result - use print_fooof_results()")
        return

    # Summary stats
    n_matched = result.get('n_matched', 0)
    n_total = result.get('n_total', 0)
    mean_beta = result.get('mean_beta', np.nan)
    beta_range = result.get('beta_range', (np.nan, np.nan))
    mean_r2 = result.get('mean_r2', np.nan)

    print(f"    Harmonics detected: {n_matched}/{n_total}")
    if not np.isnan(mean_beta):
        print(f"    Mean β (1/f slope): {mean_beta:.3f} (range: {beta_range[0]:.3f} - {beta_range[1]:.3f})")
    if not np.isnan(mean_r2):
        print(f"    Mean R²: {mean_r2:.3f}")

    # Per-harmonic table
    print(f"\n    {'Harm':>5}  {'Canon':>7}  {'Detect':>7}  {'β':>6}  {'R²':>5}  {'Window':>12}  {'Match':>5}")
    print(f"    {'-'*5}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*5}  {'-'*12}  {'-'*5}")

    peaks = result.get('peaks', [])
    for i, peak in enumerate(peaks):
        canon = peak.get('canonical', np.nan)
        detect = peak.get('detected', np.nan)
        beta = peak.get('beta', np.nan)
        r2 = peak.get('r2', np.nan)
        freq_range = peak.get('freq_range', (np.nan, np.nan))
        matched = peak.get('matched', False)

        detect_str = f"{detect:.2f}" if not np.isnan(detect) and matched else "—"
        beta_str = f"{beta:.3f}" if not np.isnan(beta) else "N/A"
        r2_str = f"{r2:.3f}" if not np.isnan(r2) else "N/A"
        window_str = f"[{freq_range[0]:.0f}-{freq_range[1]:.0f}]" if freq_range[0] is not None else "N/A"
        match_str = "Y" if matched else "—"

        print(f"    f{i:<4}  {canon:>7.2f}  {detect_str:>7}  {beta_str:>6}  {r2_str:>5}  {window_str:>12}  {match_str:>5}")

    # Comparison with global fit note
    print(f"\n    Note: Per-harmonic fitting uses region-specific 1/f background")
    print(f"          for improved peak detection in each frequency band.")


# =============================================================================
# DETAILED EVENT FOOOF REPORT
# =============================================================================

def print_detailed_event_fooof(events: List[Dict], event_fooof_results: List[Dict]):
    """
    Print detailed FOOOF peak information for all events with aggregate statistics.
    """
    print("\n" + "=" * 80)
    print("DETAILED PER-EVENT FOOOF PEAK ANALYSIS")
    print("=" * 80)

    # Aggregate statistics collectors
    all_event_peaks = []  # List of (event_idx, peak_dict)
    all_event_non_sr_peaks = []  # List of (event_idx, non_sr_peak_dict)
    sr_matched_counts = []
    total_peak_counts = []
    sr_specificities = []
    aperiodic_exponents = []
    r_squareds = []

    for i, (event, fooof_result) in enumerate(zip(events, event_fooof_results)):
        print(f"\n{'─' * 80}")
        print(f"EVENT {i+1}: {event['start_sec']:.1f}s - {event['end_sec']:.1f}s "
              f"(duration: {event['duration_sec']:.1f}s)")
        print(f"{'─' * 80}")

        # Event metrics
        print(f"  Peak Z-score: {event['peak_z']:.2f} | "
              f"Mean Z: {event['mean_z']:.2f} | "
              f"MSC: {event['msc']:.3f} | PLV: {event['plv']:.3f}")

        if not fooof_result.get('fooof_available', False):
            print("  FOOOF: Not available")
            continue

        if fooof_result.get('fit_failed', False) or fooof_result.get('error'):
            print(f"  FOOOF: Fit failed - {fooof_result.get('error', 'unknown error')}")
            continue

        # FOOOF model quality - handle per-harmonic vs global
        if fooof_result.get('per_harmonic'):
            ap_exp = fooof_result.get('mean_beta', np.nan)
            r_sq = fooof_result.get('mean_r2', np.nan)
            n_matched = fooof_result.get('n_matched', 0)
            print(f"  FOOOF Model [per-harm]: mean β={ap_exp:.3f}, mean R²={r_sq:.3f}, matched={n_matched}/{len(SR_HARMONICS)}")
        else:
            ap_exp = fooof_result.get('aperiodic_exponent', np.nan)
            r_sq = fooof_result.get('r_squared', np.nan)
            print(f"  FOOOF Model: 1/f slope={ap_exp:.3f}, R²={r_sq:.3f}")

        sr_spec = fooof_result.get('sr_specificity', np.nan)

        if not np.isnan(ap_exp):
            aperiodic_exponents.append(ap_exp)
        if not np.isnan(r_sq):
            r_squareds.append(r_sq)
        if not np.isnan(sr_spec):
            sr_specificities.append(sr_spec)

        # Handle both per-harmonic and global FOOOF results
        if fooof_result.get('per_harmonic'):
            # Per-harmonic results: show matched peaks per harmonic
            peaks = fooof_result.get('peaks', [])
            n_matched = fooof_result.get('n_matched', 0)

            total_peak_counts.append(len(SR_HARMONICS))  # Expected harmonics
            sr_matched_counts.append(n_matched)

            print(f"\n  HARMONIC PEAKS (matched={n_matched}/{len(SR_HARMONICS)}):")
            print(f"  {'Harm':>5}  {'Canon':>7}  {'Detect':>7}  {'Power':>7}  {'SNR':>8}  {'β':>6}  {'R²':>5}  {'BW':>6}  {'Pks':>4}  {'InSR':>4}")
            print(f"  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*6}  {'-'*5}  {'-'*6}  {'-'*4}  {'-'*4}")

            for j, p in enumerate(peaks):
                canon = p.get('canonical', np.nan)
                detect = p.get('detected', np.nan)
                power = p.get('power', np.nan)
                snr = p.get('snr', np.nan)
                beta = p.get('beta', np.nan)
                r2 = p.get('r2', np.nan)
                bw = p.get('bandwidth', np.nan)
                matched = p.get('matched', False)
                n_pks = p.get('n_peaks_total', 0)
                n_sr = p.get('n_peaks_in_sr', 0)

                detect_str = f"{detect:.2f}" if matched else "—"
                power_str = f"{power:.3f}" if not np.isnan(power) else "N/A"
                snr_str = f"{snr:.1f}" if not np.isnan(snr) else "N/A"
                beta_str = f"{beta:.3f}" if not np.isnan(beta) else "N/A"
                r2_str = f"{r2:.3f}" if not np.isnan(r2) else "N/A"
                bw_str = f"{bw:.2f}" if not np.isnan(bw) else "N/A"
                marker = "*" if matched else " "

                print(f"  f{j:<4}{marker} {canon:>7.2f}  {detect_str:>7}  {power_str:>7}  "
                      f"{snr_str:>8}  {beta_str:>6}  {r2_str:>5}  {bw_str:>6}  {n_pks:>4}  {n_sr:>4}")

                # Store matched peaks for aggregate analysis
                if matched:
                    snr_val = p.get('snr', np.nan)
                    all_event_peaks.append((i+1, {
                        'freq': detect, 'power': power, 'snr': snr_val, 'is_sr': True,
                        'sr_match': canon, 'rank': j+1, 'bandwidth': bw
                    }))

            # Summary
            matched_freqs = [f"{p['detected']:.1f}" for p in peaks if p.get('matched', False)]
            if matched_freqs:
                print(f"\n  Matched harmonics: {', '.join(matched_freqs)} Hz")
            else:
                print(f"\n  Matched harmonics: None")

            # Non-SR peaks summary per harmonic window
            all_non_sr_peaks = []
            for j, p in enumerate(peaks):
                non_sr = p.get('non_sr_peaks', [])
                if non_sr:
                    for nsp in non_sr:
                        nsp['harmonic_window'] = j
                        nsp['freq_range'] = p.get('freq_range', (np.nan, np.nan))
                        all_non_sr_peaks.append(nsp)

            if all_non_sr_peaks:
                print(f"\n  NON-SR PEAKS ({len(all_non_sr_peaks)} peaks outside SR search windows):")
                print(f"  {'Window':>6}  {'Range':>10}  {'Freq':>7}  {'Power':>7}  {'BW':>6}")
                print(f"  {'-'*6}  {'-'*10}  {'-'*7}  {'-'*7}  {'-'*6}")
                for nsp in sorted(all_non_sr_peaks, key=lambda x: x['freq']):
                    win_idx = nsp['harmonic_window']
                    f_range = nsp['freq_range']
                    range_str = f"[{f_range[0]:.0f}-{f_range[1]:.0f}]"
                    print(f"  f{win_idx:<5}  {range_str:>10}  {nsp['freq']:>7.2f}  {nsp['power']:>7.3f}  {nsp['bandwidth']:>6.2f}")

                # Store for aggregate analysis
                for nsp in all_non_sr_peaks:
                    all_event_non_sr_peaks.append((i+1, nsp))

        else:
            # Global results: show all characterized peaks
            all_peaks = fooof_result.get('all_peaks_characterized', [])
            if not all_peaks:
                print("  No peaks detected")
                continue

            total_peak_counts.append(len(all_peaks))
            n_sr = sum(1 for p in all_peaks if p.get('is_sr', False))
            sr_matched_counts.append(n_sr)

            print(f"\n  ALL PEAKS (n={len(all_peaks)}, SR-matched={n_sr}):")
            print(f"  {'Rank':>4}  {'Freq(Hz)':>8}  {'Power':>7}  {'SNR':>8}  {'Prom':>6}  {'BW':>6}  {'SR?':>8}")
            print(f"  {'-'*4}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*8}")

            for peak in all_peaks:
                snr_str = f"{peak['snr']:.1f}" if not np.isnan(peak.get('snr', np.nan)) else "N/A"
                prom_str = f"{peak['prominence']:.2f}" if not np.isnan(peak.get('prominence', np.nan)) else "N/A"
                sr_str = f"f{SR_HARMONICS.index(peak['sr_match'])}" if peak.get('sr_match') else "-"
                marker = "*" if peak.get('is_sr', False) else " "

                print(f"  {peak['rank']:>3}{marker}  {peak['freq']:>8.2f}  {peak['power']:>7.3f}  "
                      f"{snr_str:>8}  {prom_str:>6}  {peak['bandwidth']:>6.2f}  {sr_str:>8}")

                # Store for aggregate analysis
                all_event_peaks.append((i+1, peak))

            # SR-specific summary for this event
            sr_peaks = [p for p in all_peaks if p.get('is_sr', False)]

            if sr_peaks:
                sr_freqs = [f"{p['freq']:.1f}" for p in sr_peaks]
                sr_ranks = [p['rank'] for p in sr_peaks]
                print(f"\n  SR Peaks: {', '.join(sr_freqs)} Hz (ranks: {sr_ranks})")
            else:
                print(f"\n  SR Peaks: None detected")

            if not np.isnan(sr_spec):
                print(f"  SR Specificity Index: {sr_spec:.3f} ({100*sr_spec:.1f}% of peak power)")

    # =========================================================================
    # AGGREGATE STATISTICS ACROSS ALL EVENTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("AGGREGATE STATISTICS ACROSS ALL EVENTS")
    print("=" * 80)

    n_events = len(events)
    print(f"\nTotal Events: {n_events}")

    # FOOOF model quality
    if aperiodic_exponents:
        print(f"\nFOOOF Model Quality:")
        print(f"  Aperiodic exponent: {np.mean(aperiodic_exponents):.3f} ± {np.std(aperiodic_exponents):.3f}")
        print(f"  R²: {np.mean(r_squareds):.3f} ± {np.std(r_squareds):.3f}")

    # Peak statistics
    if total_peak_counts:
        print(f"\nSR Harmonic Detection:")
        print(f"  SR harmonics detected per event: {np.mean(sr_matched_counts):.1f} ± {np.std(sr_matched_counts):.1f} / {len(SR_HARMONICS)}")
        pct_sr = 100 * np.mean(sr_matched_counts) / len(SR_HARMONICS) if len(SR_HARMONICS) > 0 else 0
        print(f"  Detection rate: {pct_sr:.0f}%")

    # SR Specificity
    if sr_specificities:
        print(f"\nSR Specificity Index:")
        print(f"  Mean: {np.mean(sr_specificities):.3f} ± {np.std(sr_specificities):.3f}")
        print(f"  Range: {np.min(sr_specificities):.3f} - {np.max(sr_specificities):.3f}")

    # Aggregate Non-SR Peaks Analysis
    if all_event_non_sr_peaks:
        print(f"\nNon-SR Peaks Analysis (peaks outside SR search windows):")
        print(f"  Total non-SR peaks across all events: {len(all_event_non_sr_peaks)}")
        non_sr_per_event = {}
        for evt_idx, peak in all_event_non_sr_peaks:
            if evt_idx not in non_sr_per_event:
                non_sr_per_event[evt_idx] = []
            non_sr_per_event[evt_idx].append(peak)
        non_sr_counts = [len(v) for v in non_sr_per_event.values()]
        print(f"  Non-SR peaks per event: {np.mean(non_sr_counts):.1f} ± {np.std(non_sr_counts):.1f}")

        # Group by harmonic window
        non_sr_by_window = {}
        for evt_idx, peak in all_event_non_sr_peaks:
            win_idx = peak.get('harmonic_window', -1)
            if win_idx not in non_sr_by_window:
                non_sr_by_window[win_idx] = []
            non_sr_by_window[win_idx].append(peak)

        print(f"\n  Non-SR peaks by harmonic window:")
        print(f"  {'Window':>6}  {'Range':>10}  {'Count':>5}  {'Freq Range':>14}  {'Mean Freq':>10}  {'Mean Pow':>9}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*5}  {'-'*14}  {'-'*10}  {'-'*9}")
        for win_idx in sorted(non_sr_by_window.keys()):
            peaks_in_win = non_sr_by_window[win_idx]
            freqs = [p['freq'] for p in peaks_in_win]
            powers = [p['power'] for p in peaks_in_win]
            freq_range = peaks_in_win[0].get('freq_range', (np.nan, np.nan)) if peaks_in_win else (np.nan, np.nan)
            range_str = f"[{freq_range[0]:.0f}-{freq_range[1]:.0f}]"
            freq_span_str = f"{min(freqs):.1f}-{max(freqs):.1f}"
            print(f"  f{win_idx:<5}  {range_str:>10}  {len(peaks_in_win):>5}  {freq_span_str:>14}  {np.mean(freqs):>10.2f}  {np.mean(powers):>9.3f}")

        # List all unique non-SR frequencies
        all_non_sr_freqs = sorted(set(p['freq'] for _, p in all_event_non_sr_peaks))
        if len(all_non_sr_freqs) <= 10:
            print(f"\n  All non-SR peak frequencies: {', '.join([f'{f:.1f}' for f in all_non_sr_freqs])} Hz")
        else:
            print(f"\n  Non-SR peak frequency range: {min(all_non_sr_freqs):.1f} - {max(all_non_sr_freqs):.1f} Hz")
    else:
        print(f"\nNon-SR Peaks Analysis: No non-SR peaks detected")

    # Summary table of events
    print(f"\n{'─' * 80}")
    print("EVENT SUMMARY TABLE")
    print(f"{'─' * 80}")
    print(f"{'Event':>5}  {'Time(s)':>12}  {'Dur':>5}  {'PeakZ':>6}  {'MSC':>5}  {'PLV':>5}  "
          f"{'1/f':>5}  {'R²':>5}  {'#Harm':>5}  {'SRdet':>5}")
    print(f"{'-'*5}  {'-'*12}  {'-'*5}  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*5}")

    for i, (event, fooof_result) in enumerate(zip(events, event_fooof_results)):
        time_str = f"{event['start_sec']:.0f}-{event['end_sec']:.0f}"

        # Handle both per-harmonic and global FOOOF result structures
        if fooof_result.get('per_harmonic', False):
            # Per-harmonic FOOOF structure
            ap_exp = fooof_result.get('mean_beta', np.nan)
            r_sq = fooof_result.get('mean_r2', np.nan)
            n_sr = fooof_result.get('n_matched', 0)
        else:
            # Global FOOOF structure
            ap_exp = fooof_result.get('aperiodic_exponent', np.nan)
            r_sq = fooof_result.get('r_squared', np.nan)
            all_peaks = fooof_result.get('all_peaks_characterized', [])
            n_sr = sum(1 for p in all_peaks if p.get('is_sr', False))

        ap_str = f"{ap_exp:.2f}" if not np.isnan(ap_exp) else "N/A"
        r_str = f"{r_sq:.2f}" if not np.isnan(r_sq) else "N/A"

        # Show SR detection as "detected/total" format
        sr_det_str = f"{n_sr}/{len(SR_HARMONICS)}"

        print(f"{i+1:>5}  {time_str:>12}  {event['duration_sec']:>5.1f}  {event['peak_z']:>6.2f}  "
              f"{event['msc']:>5.3f}  {event['plv']:>5.3f}  {ap_str:>5}  {r_str:>5}  {len(SR_HARMONICS):>5}  {sr_det_str:>5}")

    # =========================================================================
    # PER-EVENT HARMONIC FREQUENCY TABLE
    # =========================================================================
    print(f"\n{'─' * 80}")
    print("PER-EVENT HARMONIC FREQUENCY ESTIMATES")
    print(f"{'─' * 80}")

    # Build header with harmonic labels
    harmonic_labels = [f"f{i}" for i in range(len(SR_HARMONICS))]
    header = f"{'Evt':>3}  {'Time(s)':>10}  {'PkZ':>5}  "
    header += "  ".join([f"{lbl:>6}" for lbl in harmonic_labels])
    print(header)

    # Subheader with canonical frequencies
    subheader = f"{'':>3}  {'':>10}  {'':>5}  "
    subheader += "  ".join([f"{f:.1f}" for f in SR_HARMONICS])
    print(subheader)
    print("-" * len(header))

    for i, (event, fooof_result) in enumerate(zip(events, event_fooof_results)):
        time_str = f"{event['start_sec']:.0f}-{event['end_sec']:.0f}"
        peak_z = event['peak_z']

        # Get detected frequencies for each harmonic
        detected_freqs = []
        peaks = fooof_result.get('peaks', [])

        for j, sr_freq in enumerate(SR_HARMONICS):
            # Find matching peak for this harmonic
            found_freq = None
            for p in peaks:
                if abs(p.get('canonical', 0) - sr_freq) < 0.1 and p.get('matched', False):
                    found_freq = p.get('detected', None)
                    break

            if found_freq is not None:
                detected_freqs.append(f"{found_freq:>6.2f}")
            else:
                detected_freqs.append(f"{'—':>6}")

        row = f"{i+1:>3}  {time_str:>10}  {peak_z:>5.1f}  "
        row += "  ".join(detected_freqs)
        print(row)

    # Add summary row
    print("-" * len(header))

    # Count detections per harmonic
    detection_counts = []
    for j, sr_freq in enumerate(SR_HARMONICS):
        count = 0
        for fooof_result in event_fooof_results:
            peaks = fooof_result.get('peaks', [])
            for p in peaks:
                if abs(p.get('canonical', 0) - sr_freq) < 0.1 and p.get('matched', False):
                    count += 1
                    break
        detection_counts.append(count)

    summary_row = f"{'#':>3}  {'detected':>10}  {'':>5}  "
    summary_row += "  ".join([f"{c:>4}/{n_events}" for c in detection_counts])
    print(summary_row)

    # =========================================================================
    # PER-HARMONIC METRICS TABLE (z-score, MSC, PLV per harmonic per event)
    # =========================================================================
    print(f"\n{'─' * 100}")
    print("PER-HARMONIC METRICS (z-score | MSC | PLV for each harmonic)")
    print(f"{'─' * 100}")

    # Build header
    n_harm = len(SR_HARMONICS)
    ph_header = f"{'Evt':>3}  {'Time':>8}  "
    for j in range(n_harm):
        ph_header += f"f{j} z   MSC   PLV   "
    print(ph_header)

    # Subheader with frequencies
    ph_subheader = f"{'':>3}  {'':>8}  "
    for j, h in enumerate(SR_HARMONICS):
        ph_subheader += f"{h:.1f}Hz           "
    print(ph_subheader)
    print("-" * len(ph_header))

    for i, event in enumerate(events):
        time_str = f"{event['start_sec']:.0f}-{event['end_sec']:.0f}"
        row = f"{i+1:>3}  {time_str:>8}  "

        # Get per-harmonic metrics from event
        h_z = event.get('harmonic_z_scores', [])
        h_msc = event.get('harmonic_msc', [])
        h_plv = event.get('harmonic_plv', [])

        for j in range(n_harm):
            z_val = h_z[j] if j < len(h_z) else np.nan
            msc_val = h_msc[j] if j < len(h_msc) else np.nan
            plv_val = h_plv[j] if j < len(h_plv) else np.nan

            z_str = f"{z_val:>4.1f}" if not np.isnan(z_val) else " —  "
            msc_str = f"{msc_val:.2f}" if not np.isnan(msc_val) else " — "
            plv_str = f"{plv_val:.2f}" if not np.isnan(plv_val) else " — "

            row += f"{z_str}  {msc_str}  {plv_str}  "

        print(row)

    # Summary row with means
    print("-" * len(ph_header))
    mean_row = f"{'μ':>3}  {'mean':>8}  "

    for j in range(n_harm):
        # Compute mean across events for this harmonic
        z_vals = [e.get('harmonic_z_scores', [])[j] for e in events
                  if j < len(e.get('harmonic_z_scores', [])) and not np.isnan(e.get('harmonic_z_scores', [])[j])]
        msc_vals = [e.get('harmonic_msc', [])[j] for e in events
                    if j < len(e.get('harmonic_msc', [])) and not np.isnan(e.get('harmonic_msc', [])[j])]
        plv_vals = [e.get('harmonic_plv', [])[j] for e in events
                    if j < len(e.get('harmonic_plv', [])) and not np.isnan(e.get('harmonic_plv', [])[j])]

        mean_z = np.mean(z_vals) if z_vals else np.nan
        mean_msc = np.mean(msc_vals) if msc_vals else np.nan
        mean_plv = np.mean(plv_vals) if plv_vals else np.nan

        z_str = f"{mean_z:>4.1f}" if not np.isnan(mean_z) else " —  "
        msc_str = f"{mean_msc:.2f}" if not np.isnan(mean_msc) else " — "
        plv_str = f"{mean_plv:.2f}" if not np.isnan(mean_plv) else " — "

        mean_row += f"{z_str}  {msc_str}  {plv_str}  "

    print(mean_row)

    # =========================================================================
    # FREQUENCY-SPECIFIC AGGREGATION
    # =========================================================================
    print(f"\n{'─' * 80}")
    print("SR HARMONIC DETECTION RATE ACROSS EVENTS")
    print(f"{'─' * 80}")

    for j, sr_freq in enumerate(SR_HARMONICS):
        detected_in = []
        for i, fooof_result in enumerate(event_fooof_results):
            # Handle both per-harmonic and global FOOOF results
            if fooof_result.get('per_harmonic'):
                # Per-harmonic: check peaks list directly
                peaks = fooof_result.get('peaks', [])
                for p in peaks:
                    if abs(p.get('canonical', 0) - sr_freq) < 0.1 and p.get('matched', False):
                        detected_in.append(i + 1)
                        break
            else:
                # Global: check all_peaks_characterized for sr_match
                all_peaks = fooof_result.get('all_peaks_characterized', [])
                for p in all_peaks:
                    if p.get('sr_match') == sr_freq:
                        detected_in.append(i + 1)
                        break

        detection_rate = 100 * len(detected_in) / n_events if n_events > 0 else 0
        events_str = ', '.join(map(str, detected_in)) if detected_in else "none"
        print(f"  f{j} ({sr_freq:.2f} Hz): Detected in {len(detected_in)}/{n_events} events "
              f"({detection_rate:.0f}%) - Events: {events_str}")

    # =========================================================================
    # ALL PEAKS RANKED GLOBALLY
    # =========================================================================
    if all_event_peaks:
        print(f"\n{'─' * 80}")
        print("TOP 20 PEAKS ACROSS ALL EVENTS (by power)")
        print(f"{'─' * 80}")
        sorted_peaks = sorted(all_event_peaks, key=lambda x: -x[1]['power'])[:20]
        print(f"{'Rank':>4}  {'Event':>5}  {'Freq(Hz)':>8}  {'Power':>7}  {'SNR':>8}  {'SR?':>8}")
        print(f"{'-'*4}  {'-'*5}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*8}")

        for rank, (event_idx, peak) in enumerate(sorted_peaks, 1):
            snr_str = f"{peak['snr']:.1f}" if not np.isnan(peak.get('snr', np.nan)) else "N/A"
            sr_str = f"f{SR_HARMONICS.index(peak['sr_match'])}" if peak.get('sr_match') else "-"
            marker = "*" if peak.get('is_sr', False) else " "
            print(f"{rank:>3}{marker}  {event_idx:>5}  {peak['freq']:>8.2f}  {peak['power']:>7.3f}  "
                  f"{snr_str:>8}  {sr_str:>8}")


# =============================================================================
# SURROGATE NULL CONTROL
# =============================================================================

def phase_shuffle_surrogate(sig: np.ndarray) -> np.ndarray:
    """
    Generate phase-randomized surrogate preserving power spectrum.

    This destroys temporal structure while keeping frequency content,
    providing a null hypothesis for event detection.
    """
    n = len(sig)
    fft = np.fft.rfft(sig)

    # Randomize phases while preserving amplitude
    random_phases = np.random.uniform(-np.pi, np.pi, len(fft))
    # Keep DC and Nyquist real
    random_phases[0] = 0
    if n % 2 == 0:
        random_phases[-1] = 0

    surrogate_fft = np.abs(fft) * np.exp(1j * random_phases)
    return np.fft.irfft(surrogate_fft, n)


def run_surrogate_null_control(eeg_data: np.ndarray, fs: float, center_hz: float,
                                half_bw: float, z_thresh: float, min_isi_sec: float,
                                n_surrogates: int = 200,
                                smooth_sec: float = 0.25) -> Dict:
    """
    Run surrogate analysis to test if detected events exceed chance.

    Returns:
        dict with null distribution stats and p-value
    """
    print(f"\n--- Surrogate Null Control ({n_surrogates} iterations) ---")

    # Get real event count
    mean_eeg = np.mean(eeg_data, axis=0)
    _, _, real_onsets = detect_ignitions(eeg_data, fs, center_hz, half_bw,
                                          z_thresh, min_isi_sec, smooth_sec)
    n_real = len(real_onsets)

    # Generate surrogate distribution
    surrogate_counts = []
    for i in range(n_surrogates):
        if (i + 1) % 50 == 0:
            print(f"  Processing surrogate {i+1}/{n_surrogates}...")

        # Phase-shuffle each channel independently
        surrogate_data = np.array([phase_shuffle_surrogate(ch) for ch in eeg_data])

        # Detect events in surrogate
        _, _, surr_onsets = detect_ignitions(surrogate_data, fs, center_hz, half_bw,
                                              z_thresh, min_isi_sec, smooth_sec)
        surrogate_counts.append(len(surr_onsets))

    surrogate_counts = np.array(surrogate_counts)

    # Compute p-value (one-tailed: real >= surrogate)
    p_value = np.mean(surrogate_counts >= n_real)

    # Statistics
    null_mean = np.mean(surrogate_counts)
    null_std = np.std(surrogate_counts)
    null_95 = np.percentile(surrogate_counts, 95)

    print(f"  Real events detected: {n_real}")
    print(f"  Null distribution: {null_mean:.1f} +/- {null_std:.1f} (95th %ile: {null_95:.1f})")
    print(f"  P-value (real >= surrogate): {p_value:.4f}")

    if p_value < 0.05:
        print(f"  --> SIGNIFICANT: Events exceed chance level (p < 0.05)")
    else:
        print(f"  --> NOT SIGNIFICANT: Events consistent with chance")

    return {
        'n_real': n_real,
        'null_mean': null_mean,
        'null_std': null_std,
        'null_95': null_95,
        'p_value': p_value,
        'surrogate_counts': surrogate_counts
    }


def coherence_null_control(eeg_data: np.ndarray, fs: float,
                            windows: List[Tuple[float, float]],
                            max_time: float,
                            n_permutations: int = 1000) -> Dict:
    """
    Test if ignition windows have higher MSC/PLV than random windows.

    This is a more meaningful null test than event-count surrogates because
    it tests whether detected events have distinctive coherence properties.

    Args:
        eeg_data: (n_channels, n_samples) array
        fs: Sampling rate
        windows: List of (start_sec, end_sec) ignition windows
        max_time: Total duration in seconds
        n_permutations: Number of permutations for null distribution

    Returns:
        dict with observed values, null distributions, and p-values
    """
    print(f"\n--- Coherence Null Control (Multi-band MSC/PLV vs Random Windows) ---")

    if len(windows) == 0:
        print("  No ignition windows to test")
        return {}

    # Use multi-band coherence across all SR harmonics
    print(f"  Using {len(SR_HARMONICS)} SR harmonics: {[f'{f:.1f}' for f in SR_HARMONICS]}")

    # Compute MSC and PLV for each ignition window
    ign_msc = []
    ign_plv = []
    window_durations = []

    for start, end in windows:
        i0, i1 = int(start * fs), int(end * fs)
        i1 = min(i1, eeg_data.shape[1])
        if i1 - i0 < int(fs):  # Skip very short windows
            continue

        seg = eeg_data[:, i0:i1]
        msc = compute_msc_multiband(seg, fs)
        plv = compute_plv_multiband(seg, fs)

        if not np.isnan(msc):
            ign_msc.append(msc)
        if not np.isnan(plv):
            ign_plv.append(plv)
        window_durations.append(end - start)

    if len(ign_msc) == 0:
        print("  Could not compute coherence for ignition windows")
        return {}

    # Observed means
    obs_msc = np.mean(ign_msc)
    obs_plv = np.mean(ign_plv)
    mean_duration = np.mean(window_durations)

    print(f"  Ignition windows: {len(windows)} (mean duration: {mean_duration:.1f}s)")
    print(f"  Observed MSC: {obs_msc:.3f}")
    print(f"  Observed PLV: {obs_plv:.3f}")

    # Build exclusion mask (ignition periods)
    n_samples = eeg_data.shape[1]
    excluded = np.zeros(n_samples, dtype=bool)
    for start, end in windows:
        i0, i1 = int(start * fs), int(end * fs)
        excluded[i0:min(i1, n_samples)] = True

    # Sample random windows and compute null distribution
    print(f"  Running {n_permutations} permutations...")
    null_msc = []
    null_plv = []

    n_windows = len(windows)
    win_samples = int(mean_duration * fs)

    for perm in range(n_permutations):
        perm_msc = []
        perm_plv = []

        # Sample n_windows random windows
        attempts = 0
        sampled = 0
        while sampled < n_windows and attempts < n_windows * 10:
            # Random start point in non-excluded region
            start_idx = np.random.randint(0, n_samples - win_samples)
            end_idx = start_idx + win_samples

            # Check if overlaps with ignition windows
            if not np.any(excluded[start_idx:end_idx]):
                seg = eeg_data[:, start_idx:end_idx]
                msc = compute_msc_multiband(seg, fs)
                plv = compute_plv_multiband(seg, fs)

                if not np.isnan(msc):
                    perm_msc.append(msc)
                if not np.isnan(plv):
                    perm_plv.append(plv)
                sampled += 1
            attempts += 1

        if len(perm_msc) > 0:
            null_msc.append(np.mean(perm_msc))
        if len(perm_plv) > 0:
            null_plv.append(np.mean(perm_plv))

    null_msc = np.array(null_msc)
    null_plv = np.array(null_plv)

    # Compute p-values (one-tailed: observed > null)
    p_msc = np.mean(null_msc >= obs_msc) if len(null_msc) > 0 else 1.0
    p_plv = np.mean(null_plv >= obs_plv) if len(null_plv) > 0 else 1.0

    # Statistics
    print(f"\n  MSC Null Control:")
    print(f"    Ignition MSC: {obs_msc:.3f}")
    print(f"    Random MSC:   {np.mean(null_msc):.3f} +/- {np.std(null_msc):.3f}")
    print(f"    P-value: {p_msc:.4f}")
    if p_msc < 0.05:
        print(f"    --> SIGNIFICANT: Ignition windows have higher MSC (p < 0.05)")
    else:
        print(f"    --> Not significant")

    print(f"\n  PLV Null Control:")
    print(f"    Ignition PLV: {obs_plv:.3f}")
    print(f"    Random PLV:   {np.mean(null_plv):.3f} +/- {np.std(null_plv):.3f}")
    print(f"    P-value: {p_plv:.4f}")
    if p_plv < 0.05:
        print(f"    --> SIGNIFICANT: Ignition windows have higher PLV (p < 0.05)")
    else:
        print(f"    --> Not significant")

    return {
        'obs_msc': obs_msc,
        'obs_plv': obs_plv,
        'null_msc_mean': np.mean(null_msc),
        'null_msc_std': np.std(null_msc),
        'null_plv_mean': np.mean(null_plv),
        'null_plv_std': np.std(null_plv),
        'p_msc': p_msc,
        'p_plv': p_plv,
        'null_msc': null_msc,
        'null_plv': null_plv
    }


# =============================================================================
# HARMONIC COHERENCE ANALYSIS
# =============================================================================

def compute_harmonic_coherence(eeg_data: np.ndarray, fs: float,
                                sr_harmonics: List[float],
                                bw: float = 0.5) -> Dict:
    """
    Analyze co-activation of SR harmonics.

    If SR-mediated, multiple harmonics should show correlated power fluctuations.
    If artifact, expect broadband (uncorrelated) power changes.
    """
    n_samples = eeg_data.shape[1]
    mean_eeg = np.mean(eeg_data, axis=0)

    # Extract envelope at each harmonic
    harmonic_envelopes = []
    valid_harmonics = []

    for h in sr_harmonics:
        if h < fs / 2 - bw:  # Check Nyquist
            filtered = bandpass_filter(mean_eeg, h - bw, h + bw, fs)
            env = compute_envelope(filtered)
            harmonic_envelopes.append(env)
            valid_harmonics.append(h)

    if len(harmonic_envelopes) < 2:
        return {'error': 'Not enough harmonics below Nyquist'}

    envelopes = np.array(harmonic_envelopes)
    n_harmonics = len(valid_harmonics)

    # Compute correlation matrix between harmonic envelopes
    corr_matrix = np.corrcoef(envelopes)

    # Extract upper triangle (unique pairs)
    correlations = []
    pairs = []
    for i in range(n_harmonics):
        for j in range(i + 1, n_harmonics):
            correlations.append(corr_matrix[i, j])
            pairs.append((valid_harmonics[i], valid_harmonics[j]))

    # Mean coherence across harmonic pairs
    mean_coherence = np.mean(correlations)

    # Power ratios (normalized to fundamental)
    fundamental_power = np.mean(envelopes[0] ** 2)
    power_ratios = [np.mean(env ** 2) / fundamental_power for env in envelopes]

    return {
        'valid_harmonics': valid_harmonics,
        'correlation_matrix': corr_matrix,
        'pairwise_correlations': dict(zip([f"{p[0]:.1f}-{p[1]:.1f}" for p in pairs], correlations)),
        'mean_harmonic_coherence': mean_coherence,
        'power_ratios': dict(zip([f"{h:.1f}Hz" for h in valid_harmonics], power_ratios)),
        'envelopes': envelopes
    }


def compute_harmonic_coherence_windowed(eeg_data: np.ndarray, z_envelope: np.ndarray,
                                         windows: List[Tuple[float, float]],
                                         fs: float, sr_harmonics: List[float],
                                         max_time: float) -> Dict:
    """
    Compare harmonic coherence between ignition windows and baseline.
    """
    # Get baseline windows (complement of ignition windows)
    baseline_windows = get_baseline_windows(windows, max_time, min_gap=5.0)

    def extract_segments(wins):
        segments = []
        for start, end in wins:
            i0, i1 = int(start * fs), int(end * fs)
            i1 = min(i1, eeg_data.shape[1])
            if i1 > i0:
                segments.append(eeg_data[:, i0:i1])
        return segments

    ignition_segments = extract_segments(windows)
    baseline_segments = extract_segments(baseline_windows)

    # Compute coherence for each condition
    def mean_coherence_for_segments(segments):
        coherences = []
        for seg in segments:
            if seg.shape[1] > fs:  # At least 1 second
                result = compute_harmonic_coherence(seg, fs, sr_harmonics)
                if 'mean_harmonic_coherence' in result:
                    coherences.append(result['mean_harmonic_coherence'])
        return coherences

    ign_coherences = mean_coherence_for_segments(ignition_segments)
    base_coherences = mean_coherence_for_segments(baseline_segments)

    return {
        'ignition_coherences': ign_coherences,
        'baseline_coherences': base_coherences,
        'ignition_mean': np.mean(ign_coherences) if ign_coherences else np.nan,
        'baseline_mean': np.mean(base_coherences) if base_coherences else np.nan
    }


# =============================================================================
# EFFECT SIZE CALCULATIONS
# =============================================================================

def get_baseline_windows(ignition_windows: List[Tuple[float, float]],
                          max_time: float,
                          min_gap: float = 5.0,
                          target_pct: float = 0.25,
                          min_window_sec: float = 20.0,
                          seed: int = 42) -> List[Tuple[float, float]]:
    """
    Select baseline windows totaling ~target_pct of session duration.

    Baseline windows are randomly sampled from non-ignition time periods.
    If insufficient non-ignition time exists, uses all available.

    Parameters
    ----------
    ignition_windows : list of (start, end) tuples
        Detected ignition windows to exclude
    max_time : float
        Total session duration in seconds
    min_gap : float
        Buffer zone around ignition windows (default 5.0 sec)
    target_pct : float
        Target fraction of session for baseline (default 0.25 = 25%)
    min_window_sec : float
        Minimum window duration to include (default 20 sec)
    seed : int
        Random seed for reproducibility (default 42)

    Returns
    -------
    list of (start, end) tuples
        Selected baseline windows, sorted by start time
    """
    rng = np.random.default_rng(seed)
    target_duration = target_pct * max_time

    # Handle edge case: no ignition windows
    if not ignition_windows:
        # Return single window of target duration from session start
        return [(0, min(target_duration, max_time))]

    # 1. Collect all available non-ignition segments (with buffer)
    available = []
    prev_end = 0
    for start, end in sorted(ignition_windows):
        gap_start = prev_end + min_gap / 2
        gap_end = start - min_gap / 2
        if gap_end - gap_start >= min_window_sec:
            available.append((gap_start, gap_end))
        prev_end = end

    # Trailing segment after last ignition
    trailing_start = prev_end + min_gap / 2
    trailing_end = max_time - min_gap / 2
    if trailing_end - trailing_start >= min_window_sec:
        available.append((trailing_start, trailing_end))

    # 2. Calculate total available time
    total_available = sum(end - start for start, end in available)

    # If not enough non-ignition time, use all available
    if total_available <= target_duration:
        return available

    # 3. Randomly sample windows to reach ~25% target
    baseline = []
    accumulated = 0.0

    # Shuffle available segments randomly
    indices = list(range(len(available)))
    rng.shuffle(indices)

    for idx in indices:
        start, end = available[idx]
        segment_duration = end - start

        if accumulated + segment_duration <= target_duration:
            # Take entire segment
            baseline.append((start, end))
            accumulated += segment_duration
        else:
            # Take partial segment to approach target
            needed = target_duration - accumulated
            if needed >= min_window_sec:
                baseline.append((start, start + needed))
                accumulated += needed
            break

        # Stop if we've reached target
        if accumulated >= target_duration:
            break

    return sorted(baseline, key=lambda x: x[0])


def concatenate_windows(eeg_data: np.ndarray, windows: List[Tuple[float, float]],
                        fs: float) -> np.ndarray:
    """Concatenate EEG segments from multiple time windows.

    Args:
        eeg_data: (n_channels, n_samples) array or (n_samples,) 1D array
        windows: [(start_sec, end_sec), ...]
        fs: Sampling rate

    Returns:
        Concatenated signal - same dimensionality as input
    """
    segments = []
    is_1d = eeg_data.ndim == 1

    for start, end in windows:
        i_start = int(start * fs)
        i_end = int(end * fs)

        if is_1d:
            segments.append(eeg_data[i_start:i_end])
        else:
            segments.append(eeg_data[:, i_start:i_end])

    if is_1d:
        return np.concatenate(segments)
    else:
        return np.concatenate(segments, axis=1)


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size for independent samples.

    Interpretation:
        |d| < 0.2: negligible
        0.2 <= |d| < 0.5: small
        0.5 <= |d| < 0.8: medium
        |d| >= 0.8: large
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std < 1e-10:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def compute_effect_sizes(eeg_data: np.ndarray, z_envelope: np.ndarray,
                          windows: List[Tuple[float, float]],
                          fs: float, max_time: float,
                          sr_harmonics: List[float]) -> Dict:
    """
    Compute effect sizes comparing ignition vs baseline periods.
    """
    print(f"\n--- Effect Size Analysis (Ignition vs Baseline) ---")

    baseline_windows = get_baseline_windows(windows, max_time)

    # Calculate and report baseline duration
    ignition_duration = sum(e - s for s, e in windows)
    baseline_duration = sum(e - s for s, e in baseline_windows)
    ignition_pct = 100 * ignition_duration / max_time
    baseline_pct = 100 * baseline_duration / max_time

    print(f"  Ignition windows: {len(windows)} ({ignition_duration:.1f}s, {ignition_pct:.1f}%)")
    print(f"  Baseline windows: {len(baseline_windows)} ({baseline_duration:.1f}s, {baseline_pct:.1f}%)")

    def extract_metrics(wins):
        z_means = []
        msc_values = []
        plv_values = []
        harmonic_powers = {h: [] for h in sr_harmonics if h < fs/2 - 0.5}

        mean_eeg = np.mean(eeg_data, axis=0)

        for start, end in wins:
            i0, i1 = int(start * fs), int(end * fs)
            i1 = min(i1, len(z_envelope))
            if i1 <= i0:
                continue

            # Z-envelope
            z_means.append(np.mean(z_envelope[i0:i1]))

            # MSC and PLV across all SR harmonics (multiband)
            seg = eeg_data[:, i0:i1]
            if seg.shape[1] > 10:
                msc_val = compute_msc_multiband(seg, fs)
                if not np.isnan(msc_val):
                    msc_values.append(msc_val)
                plv_val = compute_plv_multiband(seg, fs)
                if not np.isnan(plv_val):
                    plv_values.append(plv_val)

            # Harmonic powers
            for h in harmonic_powers.keys():
                h_filt = bandpass_filter(mean_eeg[i0:i1], h - 0.5, h + 0.5, fs)
                harmonic_powers[h].append(np.mean(h_filt ** 2))

        return {
            'z_mean': np.array(z_means),
            'msc': np.array(msc_values),
            'plv': np.array(plv_values),
            'harmonic_powers': {k: np.array(v) for k, v in harmonic_powers.items()}
        }

    ign_metrics = extract_metrics(windows)
    base_metrics = extract_metrics(baseline_windows)

    results = {}

    # Z-envelope effect size
    if len(ign_metrics['z_mean']) > 0 and len(base_metrics['z_mean']) > 0:
        d_z = cohens_d(ign_metrics['z_mean'], base_metrics['z_mean'])
        t_z, p_z = ttest_ind(ign_metrics['z_mean'], base_metrics['z_mean'])
        results['z_envelope'] = {
            'ignition_mean': np.mean(ign_metrics['z_mean']),
            'baseline_mean': np.mean(base_metrics['z_mean']),
            'cohens_d': d_z,
            't_stat': t_z,
            'p_value': p_z
        }
        print(f"\n  SR-band Z-envelope:")
        print(f"    Ignition: {results['z_envelope']['ignition_mean']:.3f}")
        print(f"    Baseline: {results['z_envelope']['baseline_mean']:.3f}")
        print(f"    Cohen's d: {d_z:.3f} ({_interpret_d(d_z)})")
        print(f"    t = {t_z:.2f}, p = {p_z:.4f}")

    # MSC effect size
    if len(ign_metrics['msc']) > 0 and len(base_metrics['msc']) > 0:
        d_r = cohens_d(ign_metrics['msc'], base_metrics['msc'])
        t_r, p_r = ttest_ind(ign_metrics['msc'], base_metrics['msc'])
        results['msc'] = {
            'ignition_mean': np.mean(ign_metrics['msc']),
            'baseline_mean': np.mean(base_metrics['msc']),
            'cohens_d': d_r,
            't_stat': t_r,
            'p_value': p_r
        }
        print(f"\n  Mean Squared Coherence (MSC):")
        print(f"    Ignition: {results['msc']['ignition_mean']:.3f}")
        print(f"    Baseline: {results['msc']['baseline_mean']:.3f}")
        print(f"    Cohen's d: {d_r:.3f} ({_interpret_d(d_r)})")
        print(f"    t = {t_r:.2f}, p = {p_r:.4f}")

    # PLV effect size
    if len(ign_metrics['plv']) > 0 and len(base_metrics['plv']) > 0:
        d_plv = cohens_d(ign_metrics['plv'], base_metrics['plv'])
        t_plv, p_plv = ttest_ind(ign_metrics['plv'], base_metrics['plv'])
        results['plv'] = {
            'ignition_mean': np.mean(ign_metrics['plv']),
            'baseline_mean': np.mean(base_metrics['plv']),
            'cohens_d': d_plv,
            't_stat': t_plv,
            'p_value': p_plv
        }
        print(f"\n  Phase Locking Value (PLV):")
        print(f"    Ignition: {results['plv']['ignition_mean']:.3f}")
        print(f"    Baseline: {results['plv']['baseline_mean']:.3f}")
        print(f"    Cohen's d: {d_plv:.3f} ({_interpret_d(d_plv)})")
        print(f"    t = {t_plv:.2f}, p = {p_plv:.4f}")

    # Harmonic power effect sizes
    print(f"\n  Harmonic Power:")
    results['harmonics'] = {}
    for h in ign_metrics['harmonic_powers'].keys():
        ign_pow = ign_metrics['harmonic_powers'][h]
        base_pow = base_metrics['harmonic_powers'][h]
        if len(ign_pow) > 0 and len(base_pow) > 0:
            # Use log power for more normal distribution
            ign_log = np.log10(ign_pow + 1e-10)
            base_log = np.log10(base_pow + 1e-10)
            d_h = cohens_d(ign_log, base_log)
            t_h, p_h = ttest_ind(ign_log, base_log)
            results['harmonics'][h] = {
                'cohens_d': d_h,
                't_stat': t_h,
                'p_value': p_h
            }
            sig_marker = "*" if p_h < 0.05 else ""
            print(f"    {h:.1f} Hz: d = {d_h:.3f} ({_interpret_d(d_h)}), p = {p_h:.4f} {sig_marker}")

    return results


def _interpret_d(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def compute_harmonic_bicoherence(eeg_data: np.ndarray, fs: float,
                                 ignition_windows: List[Tuple[float, float]],
                                 baseline_windows: List[Tuple[float, float]],
                                 harmonics: List[float]) -> Dict:
    """Compute harmonic-specific bicoherence during ignition vs baseline.

    Tests quadratic phase coupling: φ(f1) + φ(f2) = φ(f3) for true Schumann harmonic triads.
    Now uses detected harmonics as targets (e.g., f0+f0→f1 at 14.3 Hz, not 15.6 Hz).

    Args:
        eeg_data: (n_channels, n_samples) EEG data
        fs: Sampling rate
        ignition_windows: List of (start, end) ignition periods
        baseline_windows: List of (start, end) baseline periods
        harmonics: List of harmonic frequencies (from FOOOF)

    Returns:
        Dict with:
            'harmonics': frequency list
            'B_ignition': Bicoherence matrix during ignition [n_harm × n_harm]
            'B_baseline': Bicoherence matrix during baseline
            'delta_B': Ignition - baseline difference
            'cohen_d': Effect sizes for each pair
            'significant_pairs': List of (i, j, d, f1, f2) tuples
    """
    # Import harmonic triad bicoherence
    sys.path.insert(0, './lib')
    from shape_vs_resonance import bicoherence_harmonic_triad

    # Average across channels
    mean_eeg = np.mean(eeg_data, axis=0)

    # Concatenate ignition windows
    ignition_signal = concatenate_windows(mean_eeg, ignition_windows, fs)

    # Concatenate baseline windows
    baseline_signal = concatenate_windows(mean_eeg, baseline_windows, fs)

    # Get fixed set of 4 triads (no tolerance parameter needed)
    triad_indices, _, triad_target_indices = identify_harmonic_triads(harmonics)

    # Initialize matrices with NaN
    n_harm = len(harmonics)
    B_ignition = np.full((n_harm, n_harm), np.nan, dtype=float)
    B_baseline = np.full((n_harm, n_harm), np.nan, dtype=float)

    # Compute bicoherence only for true triads using detected harmonics
    for idx_pair, target_idx in zip(triad_indices, triad_target_indices):
        i, j = idx_pair
        f1, f2 = harmonics[i], harmonics[j]
        f3_target = harmonics[target_idx]

        # Compute for ignition
        B_ignition[i, j] = bicoherence_harmonic_triad(
            ignition_signal, fs,
            f1=f1, f2=f2, f3_target=f3_target,
            nperseg=None
        )

        # Compute for baseline
        B_baseline[i, j] = bicoherence_harmonic_triad(
            baseline_signal, fs,
            f1=f1, f2=f2, f3_target=f3_target,
            nperseg=None
        )

    # Compute differences
    delta_B = B_ignition - B_baseline

    # Compute Cohen's d for each harmonic pair
    # Simple estimate using baseline std
    baseline_std = np.std(B_baseline)
    if baseline_std < 1e-10:
        baseline_std = 1e-10
    cohen_d = delta_B / baseline_std

    # Identify significant pairs (|d| > 0.5)
    sig_pairs = []
    for i in range(len(harmonics)):
        for j in range(i, len(harmonics)):
            d = cohen_d[i, j]
            if abs(d) > 0.5:
                sig_pairs.append((i, j, d, harmonics[i], harmonics[j]))

    # Sort by effect size magnitude
    sig_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    return {
        'harmonics': harmonics,
        'B_ignition': B_ignition,
        'B_baseline': B_baseline,
        'delta_B': delta_B,
        'cohen_d': cohen_d,
        'significant_pairs': sig_pairs
    }


def print_bicoherence_results(results: Dict):
    """Print harmonic bicoherence results."""
    print("\n--- Harmonic Bicoherence Analysis ---")
    harmonics = results['harmonics']
    print(f"  Harmonics tested: {[f'{h:.2f}' for h in harmonics]}")

    B_ign = results['B_ignition']
    B_base = results['B_baseline']

    print(f"\n  Diagonal (self-coupling):")
    for i, h in enumerate(harmonics):
        print(f"    {h:.2f} + {h:.2f} Hz: "
              f"Ignition={B_ign[i,i]:.3f}, Baseline={B_base[i,i]:.3f}, "
              f"Δ={B_ign[i,i]-B_base[i,i]:+.3f}")

    if len(harmonics) > 1:
        print(f"\n  Cross-harmonic coupling:")
        for i in range(len(harmonics)):
            for j in range(i+1, len(harmonics)):
                h1, h2 = harmonics[i], harmonics[j]
                print(f"    {h1:.2f} + {h2:.2f} Hz: "
                      f"Ignition={B_ign[i,j]:.3f}, Baseline={B_base[i,j]:.3f}, "
                      f"Δ={B_ign[i,j]-B_base[i,j]:+.3f}")

    sig_pairs = results['significant_pairs']
    if sig_pairs:
        print(f"\n  Significant pairs (|Cohen's d| > 0.5):")
        for i, j, d, f1, f2 in sig_pairs[:10]:  # Top 10
            print(f"    {f1:.2f} + {f2:.2f} Hz: d = {d:+.2f}")
    else:
        print(f"\n  No significant harmonic pairs found.")


def identify_harmonic_triads(harmonics: List[float]) -> Tuple[List[Tuple[int, int]], List[str], List[int]]:
    """Generate fixed set of 4 harmonic triads: f0+f0→f1, f0+f0→f2, f0+f1→f2, f1+f1→f2.

    Args:
        harmonics: List of detected harmonic frequencies [f0, f1, f2, ...]

    Returns:
        triad_indices: [(0,0), (0,0), (0,1), (1,1)]
        triad_labels: Descriptive labels with frequencies
        triad_target_indices: [1, 2, 2, 2] (which harmonic is the target)
    """
    if len(harmonics) < 3:
        # Fallback: use self-coupling for available harmonics
        return [(i, i) for i in range(len(harmonics))], \
               [f'f{i}+f{i} ({harmonics[i]:.2f}+{harmonics[i]:.2f})' for i in range(len(harmonics))], \
               [min(i+1, len(harmonics)-1) for i in range(len(harmonics))]

    f0, f1, f2 = harmonics[0], harmonics[1], harmonics[2]

    triad_indices = [
        (0, 0),  # f0+f0
        (0, 0),  # f0+f0
        (0, 1),  # f0+f1
        (1, 1)   # f1+f1
    ]

    triad_labels = [
        f'f0+f0→f1 ({f0:.2f}+{f0:.2f}→{f1:.2f})',
        f'f0+f0→f2 ({f0:.2f}+{f0:.2f}→{f2:.2f})',
        f'f0+f1→f2 ({f0:.2f}+{f1:.2f}→{f2:.2f})',
        f'f1+f1→f2 ({f1:.2f}+{f1:.2f}→{f2:.2f})'
    ]

    triad_target_indices = [1, 2, 2, 2]  # Target harmonics: f1, f2, f2, f2

    return triad_indices, triad_labels, triad_target_indices


def compute_bicoherence_timeseries(eeg_data: np.ndarray, fs: float, timestamps: np.ndarray,
                                   harmonics: List[float],
                                   win_sec: float = 4.0, step_sec: float = 1.0,
                                   smooth_window: int = 5) -> Dict:
    """Compute sliding-window bicoherence time series for key harmonic triads.

    Args:
        eeg_data: (n_channels, n_samples) EEG data
        fs: Sampling rate
        timestamps: Time array (seconds)
        harmonics: List of harmonic frequencies
        win_sec: Window duration in seconds (default 4s for stable bicoherence)
        step_sec: Step size in seconds
        smooth_window: Number of time points for moving average smoothing (default 5)
                      Set to 1 or None to disable smoothing

    Returns:
        Dict with:
            'times': Time points for each window center
            'triads': Dict mapping triad names to time series arrays
            'triad_labels': List of triad labels for legend
    """
    sys.path.insert(0, './lib')
    from shape_vs_resonance import bicoherence_harmonic_triad

    # Average across channels
    mean_eeg = np.mean(eeg_data, axis=0)

    win_samples = int(win_sec * fs)
    step_samples = int(step_sec * fs)

    # Initialize storage for time series
    times = []
    triads_ts = {}

    # Get fixed set of 4 harmonic triads (f0+f0→f1, f0+f0→f2, f0+f1→f2, f1+f1→f2)
    triad_indices, triad_labels, triad_target_indices = identify_harmonic_triads(harmonics)

    # Initialize time series arrays
    for label in triad_labels:
        triads_ts[label] = []

    # Compute bicoherence for each window
    for j in range(0, len(mean_eeg) - win_samples, step_samples):
        window_signal = mean_eeg[j:j+win_samples]
        window_center_time = timestamps[j + win_samples // 2]

        try:
            # Use smaller nperseg to allow multiple FFT segments for proper bicoherence
            # For 4-second window at 128 Hz = 512 samples, use nperseg=256 (2s) with 50% overlap
            # This gives ~3 overlapping segments for statistical averaging
            nperseg = min(int(2 * fs), len(window_signal) // 2)

            # Compute bicoherence for each triad using TARGET harmonics (not arithmetic sums)
            for idx_pair, label, target_idx in zip(triad_indices, triad_labels, triad_target_indices):
                i, j = idx_pair
                if i < len(harmonics) and j < len(harmonics) and target_idx < len(harmonics):
                    f1, f2 = harmonics[i], harmonics[j]
                    f3_target = harmonics[target_idx]  # Use detected harmonic, not f1+f2!

                    B_val = bicoherence_harmonic_triad(
                        window_signal, fs,
                        f1=f1, f2=f2, f3_target=f3_target,
                        nperseg=nperseg
                    )
                    triads_ts[label].append(B_val)
                else:
                    triads_ts[label].append(np.nan)

            times.append(window_center_time)

        except Exception as e:
            # If window too short or computation fails, skip
            continue

    # Convert lists to arrays
    for label in triad_labels:
        triads_ts[label] = np.array(triads_ts[label])

    # Apply smoothing if requested
    if smooth_window is not None and smooth_window > 1:
        from scipy.ndimage import uniform_filter1d
        for label in triad_labels:
            data = triads_ts[label]
            # Only smooth if we have enough data points
            if len(data) >= smooth_window:
                # Use uniform_filter1d for moving average smoothing
                # mode='nearest' handles edges by extending edge values
                triads_ts[label] = uniform_filter1d(data, size=smooth_window, mode='nearest')

    return {
        'times': np.array(times),
        'triads': triads_ts,
        'triad_labels': triad_labels
    }


# =============================================================================
# IGNITION DETECTION
# =============================================================================

def detect_ignitions(eeg_data: np.ndarray, fs: float, center_hz: float,
                     half_bw: float, z_thresh: float, min_isi_sec: float,
                     smooth_sec: float = 0.25) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect ignition events based on SR-band envelope threshold crossings.

    Returns:
        z_envelope: Z-scored envelope time series
        onset_indices: Sample indices of detected onsets
        onset_times: Times (seconds) of detected onsets
    """
    # Mean across channels
    mean_eeg = np.mean(eeg_data, axis=0)

    # Bandpass filter at SR frequency
    filtered = bandpass_filter(mean_eeg, center_hz - half_bw, center_hz + half_bw, fs)

    # Compute envelope
    envelope = compute_envelope(filtered)

    # Smooth envelope
    envelope = smooth_signal(envelope, smooth_sec, fs)

    # Z-score normalize
    z_envelope = zscore(envelope, nan_policy='omit')

    # Find threshold crossings (upward)
    above_thresh = z_envelope >= z_thresh
    crossings = np.diff(above_thresh.astype(int))
    onset_indices = np.where(crossings == 1)[0] + 1

    # Enforce minimum ISI
    min_isi_samples = int(min_isi_sec * fs)
    filtered_onsets = []
    last_onset = -np.inf
    for idx in onset_indices:
        if idx - last_onset >= min_isi_samples:
            filtered_onsets.append(idx)
            last_onset = idx

    onset_indices = np.array(filtered_onsets, dtype=int)
    onset_times = onset_indices / fs

    return z_envelope, onset_indices, onset_times


def create_windows(onset_times: np.ndarray, window_sec: float,
                   merge_gap_sec: float, max_time: float) -> List[Tuple[float, float]]:
    """Create and merge windows around detected onsets."""
    if len(onset_times) == 0:
        return []

    windows = []
    for t in onset_times:
        start = max(0, t - window_sec / 2)
        end = min(max_time, t + window_sec / 2)

        # Merge with previous window if close enough
        if windows and start <= windows[-1][1] + merge_gap_sec:
            windows[-1] = (windows[-1][0], end)
        else:
            windows.append((start, end))

    # Filter out very short windows
    windows = [(a, b) for a, b in windows if b - a > 1.0]
    return windows


def characterize_event(eeg_data: np.ndarray, z_envelope: np.ndarray,
                       start_sec: float, end_sec: float, fs: float,
                       sr_harmonics: List[float],
                       session_harmonic_stats: List[Dict] = None) -> Dict:
    """Compute metrics for a single ignition event, including per-harmonic z-score, MSC, PLV.

    Args:
        session_harmonic_stats: List of {'mean': float, 'std': float} for each harmonic band,
                                computed from session baseline for proper z-scoring.
    """
    start_idx = int(start_sec * fs)
    end_idx = int(end_sec * fs)

    # Clip to valid range
    start_idx = max(0, start_idx)
    end_idx = min(len(z_envelope), end_idx)

    if end_idx <= start_idx:
        return {}

    # Z-envelope metrics (overall - uses detection band)
    z_segment = z_envelope[start_idx:end_idx]
    peak_z = float(np.nanmax(z_segment))
    mean_z = float(np.nanmean(z_segment))

    # Duration above threshold
    above_thresh = z_segment >= Z_THRESHOLD
    duration_above = float(np.sum(above_thresh) / fs)

    # EEG segment for this window
    eeg_segment = eeg_data[:, start_idx:end_idx]
    mean_eeg_segment = np.mean(eeg_segment, axis=0)

    # Overall MSC and PLV (multiband average)
    msc = compute_msc_multiband(eeg_segment, fs)
    plv = compute_plv_multiband(eeg_segment, fs)

    # Per-harmonic metrics
    harmonic_powers = []
    harmonic_z_scores = []
    harmonic_msc = []
    harmonic_plv = []

    for i, h in enumerate(sr_harmonics):
        if h >= fs / 2 - 1:  # Check Nyquist
            harmonic_powers.append(np.nan)
            harmonic_z_scores.append(np.nan)
            harmonic_msc.append(np.nan)
            harmonic_plv.append(np.nan)
            continue

        # Get bandwidth for this harmonic
        half_bw = SR_HALF_BW[i] if i < len(SR_HALF_BW) else 1.0
        f_low = max(h - half_bw, 0.5)
        f_high = min(h + half_bw, fs / 2 - 1)

        # Power
        h_filtered = bandpass_filter(mean_eeg_segment, f_low, f_high, fs)
        power = float(np.mean(h_filtered ** 2))
        harmonic_powers.append(power)

        # Z-score relative to session baseline
        envelope = np.abs(signal.hilbert(h_filtered))
        peak_envelope = float(np.nanmax(envelope))

        if session_harmonic_stats and i < len(session_harmonic_stats):
            mean_env = session_harmonic_stats[i]['mean']
            std_env = session_harmonic_stats[i]['std']
            if std_env > 0:
                z_score = (peak_envelope - mean_env) / std_env
            else:
                z_score = 0.0
        else:
            # Fallback: use raw peak envelope (not a true z-score)
            z_score = peak_envelope
        harmonic_z_scores.append(float(z_score))

        # MSC for this harmonic band
        freq_band = (f_low, f_high)
        msc_h = compute_msc(eeg_segment, fs, freq_band)
        harmonic_msc.append(msc_h if not np.isnan(msc_h) else 0.0)

        # PLV for this harmonic band (use wider band for stable phase)
        plv_f_low = max(h - half_bw * 1.5, 0.5)
        plv_f_high = min(h + half_bw * 1.5, fs / 2 - 1)
        plv_band = (plv_f_low, plv_f_high)
        plv_h = compute_plv(eeg_segment, fs, plv_band)
        harmonic_plv.append(plv_h if not np.isnan(plv_h) else 0.0)

    return {
        'start_sec': start_sec,
        'end_sec': end_sec,
        'duration_sec': end_sec - start_sec,
        'peak_z': peak_z,
        'mean_z': mean_z,
        'duration_above_thresh': duration_above,
        'msc': msc,
        'plv': plv,
        'harmonic_powers': harmonic_powers,
        'harmonic_z_scores': harmonic_z_scores,
        'harmonic_msc': harmonic_msc,
        'harmonic_plv': harmonic_plv
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def load_data(filepath: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load EEG data from CSV (legacy function for backward compatibility)."""
    print(f"Loading: {filepath}")

    # Read CSV (skip first row which contains metadata)
    df = pd.read_csv(filepath, skiprows=1)

    print(f"  Loaded {len(df)} samples ({len(df)/FS:.1f} seconds)")
    print(f"  Columns: {len(df.columns)}")

    # Extract timestamp
    timestamps = df['Timestamp'].values
    # Make relative (start from 0)
    timestamps = timestamps - timestamps[0]

    # Extract EEG channels
    eeg_cols = [c for c in EEG_CHANNELS if c in df.columns]
    print(f"  EEG channels found: {len(eeg_cols)}")

    eeg_data = df[eeg_cols].values.T  # Shape: (n_channels, n_samples)

    return df, timestamps, eeg_data


def load_data_for_device(csv_path: str, device: str = 'emotiv',
                         electrodes: List[str] = None,
                         header: int = 1, fs: float = 128.0
                         ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, float]:
    """Load EEG data from CSV with device-specific handling.

    Args:
        csv_path: Path to CSV file
        device: 'emotiv' or 'muse'
        electrodes: List of electrode columns (default: device-specific)
        header: Header row to skip (for emotiv only)
        fs: Sampling rate in Hz (default, may be overridden by detected rate)

    Returns:
        (df, timestamps, eeg_data, actual_fs) tuple
    """
    print(f"Loading: {csv_path}")
    print(f"  Device: {device}, Header row: {header}")

    if device == 'emotiv':
        df = pd.read_csv(csv_path, low_memory=False, header=header)
        df['Timestamp'] = np.arange(len(df)) / fs
        df = df.sort_values(by=['Timestamp']).reset_index(drop=True)

        # Use provided electrodes or default EPOC channels
        if electrodes is None:
            electrodes = EEG_CHANNELS
        eeg_cols = [c for c in electrodes if c in df.columns]

    else:  # muse
        df = pd.read_csv(csv_path, low_memory=False, header=0)
        df = df.sort_values(by=['timestamps']).reset_index(drop=True)
        df.rename(columns={'timestamps': 'Timestamp'}, inplace=True)

        # Map Muse channels → EEG.<name>
        muse_channel_map = {
            "eeg_1": "EEG.AF7",
            "eeg_2": "EEG.AF8",
            "eeg_3": "EEG.TP9",
            "eeg_4": "EEG.TP10",
        }
        for old_col, new_col in muse_channel_map.items():
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)

        # Use Muse electrodes
        if electrodes is None:
            electrodes = ['EEG.AF7', 'EEG.AF8', 'EEG.TP9', 'EEG.TP10']
        eeg_cols = [c for c in electrodes if c in df.columns]

        # IMPORTANT: Filter to only rows with EEG data (Muse files are sparse)
        if eeg_cols:
            eeg_mask = df[eeg_cols[0]].notna()
            original_rows = len(df)
            df = df[eeg_mask].reset_index(drop=True)
            print(f"  Filtered sparse Muse data: {original_rows} → {len(df)} rows with EEG")

            # Detect actual sample rate from timestamps (Muse = 256 Hz)
            if len(df) > 10:
                t = df['Timestamp'].values
                dt = np.diff(t)
                dt_valid = dt[(dt > 0) & (dt < 1)]
                if len(dt_valid) > 0:
                    fs = 1.0 / np.median(dt_valid)
                    print(f"  Detected Muse sample rate: {fs:.1f} Hz")

    if not eeg_cols:
        raise ValueError(f"No valid EEG columns found. Available: {list(df.columns[:20])}")

    print(f"  Loaded {len(df)} samples ({len(df)/fs:.1f} seconds)")
    print(f"  EEG channels found: {len(eeg_cols)} of {len(electrodes)} requested")
    print(f"  Channels: {eeg_cols}")
    print(f"  Using sample rate: {fs:.1f} Hz")

    timestamps = df['Timestamp'].values
    # Make relative (start from 0)
    timestamps = timestamps - timestamps[0]

    eeg_data = df[eeg_cols].values.T  # Shape: (n_channels, n_samples)

    return df, timestamps, eeg_data, fs


def plot_results(timestamps: np.ndarray, z_envelope: np.ndarray,
                 eeg_data: np.ndarray, windows: List[Tuple[float, float]],
                 events: List[Dict],
                 bico_ts: Dict = None,
                 session_f0: float = None,
                 session_name: str = None,
                 output_dir: str = None,
                 show: bool = True):
    """Create separate visualization charts for detection results.

    Args:
        bico_ts: Optional bicoherence time series dict from compute_bicoherence_timeseries()
        session_f0: Session-estimated f0 frequency (from FOOOF). If None, uses canonical SR_HARMONICS[0]
        session_name: Name to include in chart titles and filenames
        output_dir: Directory to save charts (if None, uses current directory)
    """
    # Extract base session name and output directory
    if output_dir is None:
        output_dir = '.'
    if session_name is None:
        session_name = 'session'

    # Use only f0 (fundamental) - prefer session-estimated value
    f0_center = session_f0 if session_f0 is not None else SR_HARMONICS[0]
    f0_half_bw = SR_HALF_BW[0]   # 0.6 Hz
    f0_low = max(f0_center - f0_half_bw, 0.5)
    f0_high = min(f0_center + f0_half_bw, FS / 2 - 1)

    # Common data preparation
    mean_eeg = np.mean(eeg_data, axis=0)
    filtered = bandpass_filter(mean_eeg, f0_low, f0_high, FS)
    envelope = np.abs(signal.hilbert(filtered))
    z_env = (envelope - np.mean(envelope)) / (np.std(envelope) + 1e-10)

    # Compute MSC and PLV time series once
    win_samples = int(2.0 * FS)  # 2 second window
    step_samples = int(0.5 * FS)  # 0.5 second step
    freq_band = (f0_low, f0_high)

    msc_times, msc_values = [], []
    plv_times, plv_values = [], []
    plv_f_low = max(f0_center - f0_half_bw * 1.5, 0.5)
    plv_f_high = min(f0_center + f0_half_bw * 1.5, FS / 2 - 1)
    plv_freq_band = (plv_f_low, plv_f_high)

    for j in range(0, len(timestamps) - win_samples, step_samples):
        seg = eeg_data[:, j:j+win_samples]
        t_center = timestamps[j + win_samples // 2]

        msc_val = compute_msc(seg, FS, freq_band)
        msc_times.append(t_center)
        msc_values.append(msc_val if not np.isnan(msc_val) else 0)

        plv_val = compute_plv(seg, FS, plv_freq_band)
        plv_times.append(t_center)
        plv_values.append(plv_val if not np.isnan(plv_val) else 0)

    # Apply smoothing to MSC and PLV (similar to bicoherence time series)
    from scipy.ndimage import uniform_filter1d
    smooth_window = 5  # Moving average window
    if len(msc_values) >= smooth_window:
        msc_values = uniform_filter1d(np.array(msc_values), size=smooth_window, mode='nearest')
        plv_values = uniform_filter1d(np.array(plv_values), size=smooth_window, mode='nearest')

    # --- Chart 1: Raw EEG ---
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    ax.plot(timestamps, mean_eeg, 'k-', linewidth=0.5, alpha=0.7)
    ax.set_ylabel('Mean EEG (μV)')
    ax.set_xlabel('Time (seconds)')
    ax.set_title(f'Raw EEG (mean across channels)\n{session_name}')
    for start, end in windows:
        ax.axvspan(start, end, alpha=0.3, color='red', label='_')
    plt.tight_layout()
    path1 = os.path.join(output_dir, f"{session_name}_1_raw_eeg.png")
    plt.savefig(path1, dpi=150)
    print(f"Saved: {path1}")
    if show:
        plt.show()
    plt.close(fig)

    # --- Chart 2: f0 filtered signal ---
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    ax.plot(timestamps, filtered, 'b-', linewidth=0.5)
    ax.set_ylabel(f'f0 ({f0_center:.2f} Hz)')
    ax.set_xlabel('Time (seconds)')
    ax.set_title(f'Bandpass filtered: {f0_low:.1f} - {f0_high:.1f} Hz\n{session_name}')
    for start, end in windows:
        ax.axvspan(start, end, alpha=0.3, color='red')
    plt.tight_layout()
    path2 = os.path.join(output_dir, f"{session_name}_2_f0_filtered.png")
    plt.savefig(path2, dpi=150)
    print(f"Saved: {path2}")
    if show:
        plt.show()
    plt.close(fig)

    # --- Chart 3: f0 Z-envelope ---
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    ax.plot(timestamps, z_env, 'g-', linewidth=0.8)
    ax.axhline(Z_THRESHOLD, color='r', linestyle='--', linewidth=1.5, label=f'Threshold (z={Z_THRESHOLD})')
    ax.set_ylabel('Z-score')
    ax.set_xlabel('Time (seconds)')
    ax.set_title(f'f0 ({f0_center:.2f} Hz) Z-scored Envelope\n{session_name}')
    ax.legend(loc='upper right')
    for start, end in windows:
        ax.axvspan(start, end, alpha=0.3, color='red')
    plt.tight_layout()
    path3 = os.path.join(output_dir, f"{session_name}_3_z_envelope.png")
    plt.savefig(path3, dpi=150)
    print(f"Saved: {path3}")
    if show:
        plt.show()
    plt.close(fig)

    # --- Chart 4: f0 MSC time series ---
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    ax.plot(msc_times, msc_values, 'm-', linewidth=0.8)
    ax.set_ylabel('MSC')
    ax.set_xlabel('Time (seconds)')
    ax.set_title(f'f0 ({f0_center:.2f} Hz) Mean Squared Coherence\n{session_name}')
    msc_min, msc_max = np.min(msc_values), np.max(msc_values)
    msc_range = msc_max - msc_min
    ax.set_ylim(max(0, msc_min - 0.1 * msc_range), min(1, msc_max + 0.1 * msc_range))
    for start, end in windows:
        ax.axvspan(start, end, alpha=0.3, color='red')
    plt.tight_layout()
    path4 = os.path.join(output_dir, f"{session_name}_4_msc.png")
    plt.savefig(path4, dpi=150)
    print(f"Saved: {path4}")
    if show:
        plt.show()
    plt.close(fig)

    # --- Chart 5: f0 PLV time series ---
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    ax.plot(plv_times, plv_values, color='teal', linewidth=0.8)
    ax.set_ylabel('PLV')
    ax.set_xlabel('Time (seconds)')
    ax.set_title(f'f0 ({f0_center:.2f} Hz) Phase Locking Value\n{session_name}')
    plv_min, plv_max = np.min(plv_values), np.max(plv_values)
    plv_range = plv_max - plv_min
    ax.set_ylim(max(0, plv_min - 0.1 * plv_range), min(1, plv_max + 0.1 * plv_range))
    for start, end in windows:
        ax.axvspan(start, end, alpha=0.3, color='red')
    plt.tight_layout()
    path5 = os.path.join(output_dir, f"{session_name}_5_plv.png")
    plt.savefig(path5, dpi=150)
    print(f"Saved: {path5}")
    if show:
        plt.show()
    plt.close(fig)

    # --- Chart 6: Bicoherence time series (if available) ---
    if bico_ts is not None and len(bico_ts['times']) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(14, 4))
        times = bico_ts['times']
        triads = bico_ts['triads']
        labels = bico_ts['triad_labels']

        colors = ['purple', 'orange', 'green', 'blue', 'red', 'brown']
        linestyles = ['-', '-', '-', '--', '--', '--']

        n_plotted = 0
        all_bico_values = []
        for i, label in enumerate(labels):
            if label in triads and len(triads[label]) > 0:
                data = triads[label]
                valid_data = data[~np.isnan(data)]
                if len(valid_data) == 0:
                    continue

                all_bico_values.extend(valid_data)
                color = colors[i % len(colors)]
                style = linestyles[i % len(linestyles)]
                ax.plot(times, data, color=color, linestyle=style,
                       linewidth=1.8, label=label, alpha=0.75)
                n_plotted += 1

        ax.set_ylabel('Bicoherence')
        ax.set_xlabel('Time (seconds)')
        ax.set_title(f'True Harmonic Triads: f₁+f₂≈f₃ (Quadratic Phase Coupling)\n{session_name}')

        if len(all_bico_values) > 0:
            bico_min, bico_max = np.min(all_bico_values), np.max(all_bico_values)
            bico_range = bico_max - bico_min
            ax.set_ylim(max(0, bico_min - 0.1 * bico_range), min(1, bico_max + 0.1 * bico_range))
        else:
            ax.set_ylim(0, 1)

        if n_plotted > 0:
            ax.legend(loc='upper right', fontsize=8, ncol=2)
        else:
            ax.text(0.5, 0.5, 'No bicoherence data available',
                   ha='center', va='center', transform=ax.transAxes)

        ax.grid(True, alpha=0.3)
        for start, end in windows:
            ax.axvspan(start, end, alpha=0.3, color='red')

        plt.tight_layout()
        path6 = os.path.join(output_dir, f"{session_name}_6_bicoherence.png")
        plt.savefig(path6, dpi=150)
        print(f"Saved: {path6}")
        if show:
            plt.show()
        plt.close(fig)


def plot_effect_sizes(effect_results: Dict, session_name: str, output_dir: str, show: bool = True):
    """Create standalone effect size chart.

    Args:
        effect_results: Dict from compute_effect_sizes() containing Cohen's d values
        session_name: Session name for title and filename
        output_dir: Directory to save chart
        show: Whether to display plot interactively
    """
    if not effect_results:
        print("  No effect size data to plot")
        return

    # Collect all metrics and their effect sizes
    metrics = []
    d_values = []
    p_values = []
    colors = []

    if 'z_envelope' in effect_results:
        metrics.append('Z-envelope')
        d_values.append(effect_results['z_envelope']['cohens_d'])
        p_values.append(effect_results['z_envelope']['p_value'])
        colors.append('green')

    if 'msc' in effect_results:
        metrics.append('MSC')
        d_values.append(effect_results['msc']['cohens_d'])
        p_values.append(effect_results['msc']['p_value'])
        colors.append('purple')

    if 'plv' in effect_results:
        metrics.append('PLV')
        d_values.append(effect_results['plv']['cohens_d'])
        p_values.append(effect_results['plv']['p_value'])
        colors.append('cyan')

    if 'harmonics' in effect_results:
        for h, vals in effect_results['harmonics'].items():
            metrics.append(f"{h:.1f} Hz")
            d_values.append(vals['cohens_d'])
            p_values.append(vals.get('p_value', 1.0))
            colors.append('steelblue')

    if len(metrics) == 0:
        print("  No effect size metrics available")
        return

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, max(4, len(metrics) * 0.5 + 1)))

    # Create horizontal bar chart
    y_pos = np.arange(len(metrics))
    ax.barh(y_pos, d_values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

    # Add reference lines for effect size interpretation
    ax.axvline(0, color='black', linewidth=1)
    ax.axvline(0.2, color='gray', linestyle=':', alpha=0.6, label='Small (0.2)')
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.6, label='Medium (0.5)')
    ax.axvline(0.8, color='gray', linestyle='-', alpha=0.6, label='Large (0.8)')
    ax.axvline(-0.2, color='gray', linestyle=':', alpha=0.6)
    ax.axvline(-0.5, color='gray', linestyle='--', alpha=0.6)
    ax.axvline(-0.8, color='gray', linestyle='-', alpha=0.6)

    # Add significance markers and value labels
    for i, (_, d, p) in enumerate(zip(metrics, d_values, p_values)):
        # Add value label
        x_offset = 0.05 if d >= 0 else -0.05
        ha = 'left' if d >= 0 else 'right'
        sig_marker = ' *' if p < 0.05 else ''
        ax.text(d + x_offset, i, f'{d:.2f}{sig_marker}', va='center', ha=ha, fontsize=9)

    # Labels and title
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics)
    ax.set_xlabel("Cohen's d (Ignition vs Baseline)", fontsize=11)
    ax.set_title(f"Effect Sizes: Ignition vs Baseline\n{session_name}", fontsize=12)

    # Add legend for reference lines (smaller, upper right to avoid data overlap)
    ax.legend(loc='upper right', fontsize=6, title='Thresholds', title_fontsize=7,
              framealpha=0.9, handlelength=1.5)

    # Auto-scale x-axis based on data range (filter NaN/Inf values)
    valid_d = [d for d in d_values if np.isfinite(d)]
    d_min = min(valid_d) if valid_d else -1.0
    d_max = max(valid_d) if valid_d else 1.0
    padding = 0.4  # Space for labels
    ax.set_xlim(min(d_min - padding, -0.3), max(d_max + padding, 0.3))

    # Add note about significance
    ax.text(0.02, 0.02, '* p < 0.05', transform=ax.transAxes, fontsize=8,
            style='italic', color='gray')

    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"{session_name}_7_effect_sizes.png")
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)


# =============================================================================
# CSV EXPORT FOR BULK ANALYSIS
# =============================================================================

def export_to_csv(results: Dict, data_file: str, export_dir: str = "exports"):
    """Export session and event data to CSV files for bulk analysis.

    Creates/appends to:
    - exports/sessions.csv (one row per session)
    - exports/events.csv (one row per event)

    Both files share session_id for easy joining in pandas.
    """
    from pathlib import Path

    os.makedirs(export_dir, exist_ok=True)

    # Generate session_id from filename
    session_id = Path(data_file).stem

    # Extract data from results dict
    events = results['events']
    session_fooof = results.get('session_fooof_per_harmonic') or results.get('session_fooof', {})
    session_harmonics = results['session_harmonics']
    max_time = results.get('max_time', 0)

    # === SESSION ROW ===
    n_events = len(events)

    # Compute summary stats
    peak_zs = [e['peak_z'] for e in events] if events else []
    msc_vals = [e['msc'] for e in events] if events else []
    plv_vals = [e['plv'] for e in events] if events else []
    durations = [e['duration_sec'] for e in events] if events else []

    # Per-harmonic z-score means
    f_z_means = []
    for i in range(len(SR_HARMONICS)):
        z_vals = [e['harmonic_z_scores'][i] for e in events
                  if 'harmonic_z_scores' in e and i < len(e['harmonic_z_scores'])]
        f_z_means.append(np.nanmean(z_vals) if z_vals else np.nan)

    session_row = {
        'session_id': session_id,
        'filename': data_file,
        'duration_sec': max_time,
        'n_events': n_events,
        'event_rate_per_min': n_events / (max_time / 60) if max_time > 0 else 0,
        'time_in_ignition_sec': sum(durations) if durations else 0,
        'time_in_ignition_pct': 100 * sum(durations) / max_time if max_time > 0 and durations else 0,
        'mean_peak_z': np.nanmean(peak_zs) if peak_zs else np.nan,
        'std_peak_z': np.nanstd(peak_zs) if peak_zs else np.nan,
        'mean_msc': np.nanmean(msc_vals) if msc_vals else np.nan,
        'mean_plv': np.nanmean(plv_vals) if plv_vals else np.nan,
    }

    # Add session harmonics (using SR_LABELS for column names)
    for i, h in enumerate(session_harmonics):
        label = SR_LABELS[i] if i < len(SR_LABELS) else f'f{i}'
        session_row[f'session_{label}_hz'] = h

    # Add per-harmonic z-score means
    for i, z in enumerate(f_z_means):
        label = SR_LABELS[i] if i < len(SR_LABELS) else f'f{i}'
        session_row[f'mean_{label}_z'] = z

    # Add session FOOOF params
    if session_fooof and session_fooof.get('fooof_available'):
        session_row['session_beta_mean'] = session_fooof.get('mean_beta', np.nan)
        session_row['session_r2_mean'] = session_fooof.get('mean_r2', np.nan)
        session_row['session_n_matched'] = session_fooof.get('n_matched', 0)

    # Add session bicoherence metrics
    bico_results = results.get('bico_results')
    if bico_results is not None:
        B_ign = bico_results['B_ignition']
        B_base = bico_results['B_baseline']

        # Key triads (diagonal = self-coupling) - using SR_LABELS
        l0 = SR_LABELS[0] if len(SR_LABELS) > 0 else 'f0'
        l1 = SR_LABELS[1] if len(SR_LABELS) > 1 else 'f1'
        session_row[f'bico_{l0}_self'] = B_ign[0, 0] if len(B_ign) > 0 else np.nan
        session_row[f'bico_{l0}_{l1}'] = B_ign[0, 1] if len(B_ign) > 1 else np.nan
        session_row[f'bico_{l1}_self'] = B_ign[1, 1] if len(B_ign) > 1 else np.nan

        # Summary statistics
        session_row['bico_mean_ignition'] = np.mean(B_ign)
        session_row['bico_mean_baseline'] = np.mean(B_base)
        session_row['bico_delta_mean'] = np.mean(bico_results['delta_B'])
        session_row['bico_max_cohen_d'] = np.max(np.abs(bico_results['cohen_d']))
        session_row['bico_n_significant'] = len(bico_results['significant_pairs'])

        # Diagonal vs off-diagonal
        n_h = len(bico_results['harmonics'])
        diag_mask = np.eye(n_h, dtype=bool)
        session_row['bico_diag_mean'] = np.mean(B_ign[diag_mask])
        session_row['bico_offdiag_mean'] = np.mean(B_ign[~diag_mask])
    else:
        l0 = SR_LABELS[0] if len(SR_LABELS) > 0 else 'f0'
        l1 = SR_LABELS[1] if len(SR_LABELS) > 1 else 'f1'
        session_row[f'bico_{l0}_self'] = np.nan
        session_row[f'bico_{l0}_{l1}'] = np.nan
        session_row[f'bico_{l1}_self'] = np.nan
        session_row['bico_mean_ignition'] = np.nan
        session_row['bico_mean_baseline'] = np.nan
        session_row['bico_delta_mean'] = np.nan
        session_row['bico_max_cohen_d'] = np.nan
        session_row['bico_n_significant'] = 0
        session_row['bico_diag_mean'] = np.nan
        session_row['bico_offdiag_mean'] = np.nan

    # Write/append session row (remove existing rows for this session first)
    sessions_file = os.path.join(export_dir, 'sessions.csv')
    session_df = pd.DataFrame([session_row])
    if os.path.exists(sessions_file):
        existing_df = pd.read_csv(sessions_file)
        # Remove any existing rows for this session_id
        existing_df = existing_df[existing_df['session_id'] != session_id]
        combined_df = pd.concat([existing_df, session_df], ignore_index=True)
        combined_df.to_csv(sessions_file, index=False)
    else:
        session_df.to_csv(sessions_file, index=False)

    # === EVENT ROWS ===
    event_rows = []
    for i, event in enumerate(events):
        event_row = {
            'session_id': session_id,
            'event_num': i + 1,
            'start_sec': event['start_sec'],
            'end_sec': event['end_sec'],
            'duration_sec': event['duration_sec'],
            'peak_z': event['peak_z'],
            'mean_z': event['mean_z'],
            'duration_above_thresh': event['duration_above_thresh'],
            'msc': event['msc'],
            'plv': event['plv'],
            'harmonics_source': event.get('harmonics_source', 'unknown'),
        }

        # Add per-harmonic frequencies used (using SR_LABELS for column names)
        harmonics_used = event.get('harmonics_used', session_harmonics)
        for j, h in enumerate(harmonics_used):
            label = SR_LABELS[j] if j < len(SR_LABELS) else f'f{j}'
            event_row[f'{label}_hz'] = h

        # Add per-harmonic z-scores
        if 'harmonic_z_scores' in event:
            for j, z in enumerate(event['harmonic_z_scores']):
                label = SR_LABELS[j] if j < len(SR_LABELS) else f'f{j}'
                event_row[f'{label}_z'] = z

        # Add per-harmonic MSC
        if 'harmonic_msc' in event:
            for j, m in enumerate(event['harmonic_msc']):
                label = SR_LABELS[j] if j < len(SR_LABELS) else f'f{j}'
                event_row[f'{label}_msc'] = m

        # Add per-harmonic PLV
        if 'harmonic_plv' in event:
            for j, p in enumerate(event['harmonic_plv']):
                label = SR_LABELS[j] if j < len(SR_LABELS) else f'f{j}'
                event_row[f'{label}_plv'] = p

        # Add per-event FOOOF params
        event_fooof = event.get('fooof', {})
        if event_fooof and event_fooof.get('fooof_available'):
            event_row['fooof_n_matched'] = event_fooof.get('n_matched', 0)
            event_row['fooof_beta_mean'] = event_fooof.get('mean_beta', np.nan)
            event_row['fooof_r2_mean'] = event_fooof.get('mean_r2', np.nan)

        event_rows.append(event_row)

    # Write/append event rows (remove existing rows for this session first)
    events_file = os.path.join(export_dir, 'events.csv')
    if event_rows:
        events_df = pd.DataFrame(event_rows)
        if os.path.exists(events_file):
            existing_df = pd.read_csv(events_file)
            # Remove any existing rows for this session_id
            existing_df = existing_df[existing_df['session_id'] != session_id]
            combined_df = pd.concat([existing_df, events_df], ignore_index=True)
            combined_df.to_csv(events_file, index=False)
        else:
            events_df.to_csv(events_file, index=False)

    print(f"\n--- CSV Export ---")
    print(f"  Sessions: {sessions_file} (updated)")
    print(f"  Events: {events_file} ({len(event_rows)} rows for this session)")

    return sessions_file, events_file


def main(data_file: str = None, device: str = 'emotiv',
         electrodes: List[str] = None, header: int = 1,
         session_name: str = None, output_dir: str = "exports",
         show_plots: bool = True, fs: float = 128.0):
    """Run ignition analysis on a single file.

    Args:
        data_file: Path to EEG CSV file (default: DATA_FILE constant)
        device: 'emotiv' or 'muse' - determines loading logic
        electrodes: Electrode list (default: device-specific)
        header: Header row to skip (for emotiv only)
        session_name: Name for output files (default: derived from filename)
        output_dir: Directory for all outputs (default: "exports")
        show_plots: Whether to display plots interactively (default: True)
        fs: Sampling rate in Hz (default: 128.0)
    """
    from pathlib import Path

    # Use defaults if not provided
    if data_file is None:
        data_file = DATA_FILE
    if session_name is None:
        session_name = Path(data_file).stem

    # Create session output directory
    session_dir = os.path.join(output_dir, session_name)
    os.makedirs(session_dir, exist_ok=True)

    # Start capturing console output
    capture = CaptureStdout()
    capture.__enter__()

    print("=" * 60)
    print("CUSTOM EEG IGNITION DETECTION ANALYSIS")
    print("=" * 60)
    print(f"Session: {session_name}")
    print(f"Output directory: {session_dir}")

    # Load data with device-specific handling
    if device == 'emotiv' and electrodes is None and header == 1 and fs == 128.0:
        # Use legacy loader for backward compatibility with single-file mode
        df, timestamps, eeg_data = load_data(data_file)
        actual_fs = fs
    else:
        df, timestamps, eeg_data, actual_fs = load_data_for_device(
            data_file, device=device, electrodes=electrodes, header=header, fs=fs
        )
    max_time = timestamps[-1]
    print(f"  Actual sample rate: {actual_fs:.1f} Hz")

    print("\n--- Detection Parameters ---")
    print(f"  SR center frequency: {SR_CENTER} Hz")
    print(f"  Bandwidth: +/- {SR_BANDWIDTH} Hz")
    print(f"  Z-threshold: {Z_THRESHOLD}")
    print(f"  Min ISI: {MIN_ISI_SEC} sec")
    print(f"  Window size: {WINDOW_SEC} sec")

    # ==========================================================================
    # SESSION-LEVEL FOOOF HARMONIC ESTIMATION
    # ==========================================================================
    print("\n--- Session-Level FOOOF Harmonic Estimation ---")
    print(f"  SR Harmonics: {SR_HARMONICS}")
    print(f"  Per-harmonic bandwidths: {SR_HALF_BW}")
    print(f"  Per-harmonic FOOOF: {'ENABLED' if USE_PER_HARMONIC_FOOOF else 'DISABLED'}")

    # Global FOOOF (single 1-50 Hz fit)
    session_fooof = estimate_harmonics_fooof(
        eeg_data, actual_fs, SR_HARMONICS, per_harmonic_bw=SR_HALF_BW
    )
    print_fooof_results(session_fooof, "Session (Global 1-50 Hz)")

    # Per-harmonic FOOOF (if enabled)
    session_fooof_per_harmonic = None
    if USE_PER_HARMONIC_FOOOF:
        print(f"\n  Per-harmonic FOOOF windows: {SR_FREQ_RANGES}")
        print(f"  Match method: {FOOOF_MATCH_METHOD}")
        session_fooof_per_harmonic = estimate_harmonics_fooof_per_harmonic(
            eeg_data, actual_fs, SR_HARMONICS, SR_FREQ_RANGES, per_harmonic_bw=SR_HALF_BW,
            nperseg_sec=2.0,  # Match event-level windowing for consistent detection
            match_method=FOOOF_MATCH_METHOD, verbose=FOOOF_VERBOSE
        )
        print_fooof_results_per_harmonic(session_fooof_per_harmonic, "Session (Per-Harmonic)")

        # Compare global vs per-harmonic detection
        global_matched = sum(1 for p in session_fooof.get('peaks', []) if p.get('matched', False))
        per_harm_matched = session_fooof_per_harmonic.get('n_matched', 0)
        print(f"\n  Comparison: Global detected {global_matched}/{len(SR_HARMONICS)}, "
              f"Per-harmonic detected {per_harm_matched}/{len(SR_HARMONICS)}")

    # Use FOOOF-detected harmonics if available
    # Prefer per-harmonic results if enabled and successful
    if USE_PER_HARMONIC_FOOOF and session_fooof_per_harmonic and session_fooof_per_harmonic.get('fooof_available'):
        session_harmonics = session_fooof_per_harmonic['detected_freqs']
        print(f"\n  Using per-harmonic FOOOF harmonics: {[f'{h:.2f}' for h in session_harmonics]}")
    elif session_fooof.get('fooof_available') and not session_fooof.get('fit_failed'):
        session_harmonics = session_fooof['detected_freqs']
        print(f"\n  Using global FOOOF harmonics: {[f'{h:.2f}' for h in session_harmonics]}")
    else:
        session_harmonics = SR_HARMONICS
        print(f"\n  Using canonical harmonics: {session_harmonics}")

    # Pre-compute session-level envelope statistics for each harmonic band
    # This enables proper z-scoring of per-harmonic metrics relative to baseline
    print("\n  Computing session baseline stats for per-harmonic z-scores...")
    session_harmonic_stats = []

    # Average channels first (to match how events are computed in characterize_event)
    mean_eeg = np.mean(eeg_data, axis=0)

    for i, h in enumerate(session_harmonics):
        half_bw = SR_HALF_BW[i] if i < len(SR_HALF_BW) else 1.0
        f_low = max(h - half_bw, 0.5)
        f_high = min(h + half_bw, FS / 2 - 1)

        # Filter channel-averaged signal and compute envelope stats
        h_filtered_session = bandpass_filter(mean_eeg, f_low, f_high, actual_fs)
        envelope_session = np.abs(signal.hilbert(h_filtered_session))

        session_harmonic_stats.append({
            'mean': float(np.nanmean(envelope_session)),
            'std': float(np.nanstd(envelope_session)),
            'freq': h,
            'band': (f_low, f_high)
        })

    # Detect ignitions
    print("\n--- Running Detection ---")
    z_envelope, onset_indices, onset_times = detect_ignitions(
        eeg_data, actual_fs, SR_CENTER, SR_BANDWIDTH, Z_THRESHOLD, MIN_ISI_SEC
    )

    print(f"  Found {len(onset_times)} onset events")

    # Create windows
    windows = create_windows(onset_times, WINDOW_SEC, MERGE_GAP_SEC, max_time)
    print(f"  Created {len(windows)} ignition windows (after merging)")

    # Characterize each event with per-event FOOOF
    fooof_mode = "Per-Harmonic" if USE_PER_HARMONIC_FOOOF else "Global"
    print(f"\n--- Event Characterization (with {fooof_mode} FOOOF) ---")
    events = []
    event_fooof_results = []

    for i, (start, end) in enumerate(windows):
        # Run per-event FOOOF FIRST to get event-specific frequencies
        if USE_PER_HARMONIC_FOOOF:
            # Use per-harmonic FOOOF for better lower-frequency detection
            event_fooof = estimate_event_harmonics_fooof_per_harmonic(
                eeg_data, actual_fs, start, end, SR_HARMONICS, SR_FREQ_RANGES,
                per_harmonic_bw=SR_HALF_BW
            )
        else:
            # Use global FOOOF (original approach)
            event_fooof = estimate_event_harmonics_fooof(
                eeg_data, actual_fs, start, end, SR_HARMONICS, per_harmonic_bw=SR_HALF_BW
            )

        # Determine which harmonics to use for this event's metrics
        if event_fooof.get('fooof_available') and event_fooof.get('n_matched', 0) > 0:
            event_harmonics = event_fooof['detected_freqs']
            harmonics_source = 'per_event_fooof'
        else:
            event_harmonics = session_harmonics
            harmonics_source = 'session_fooof'

        # Characterize event using event-specific FOOOF frequencies
        event = characterize_event(eeg_data, z_envelope, start, end, actual_fs, event_harmonics,
                                   session_harmonic_stats=session_harmonic_stats)

        # Store FOOOF results and frequency source info
        event['fooof'] = event_fooof
        event['fooof_harmonics'] = event_fooof.get('detected_freqs', session_harmonics)
        event['harmonics_used'] = event_harmonics
        event['harmonics_source'] = harmonics_source
        event_fooof_results.append(event_fooof)

        events.append(event)

        # Print event summary
        print(f"\n  Event {i+1}: {start:.1f}s - {end:.1f}s (duration: {end-start:.1f}s)")
        print(f"    Peak z-score: {event['peak_z']:.2f}")
        print(f"    Mean z-score: {event['mean_z']:.2f}")
        print(f"    Duration above threshold: {event['duration_above_thresh']:.2f}s")
        print(f"    MSC: {event['msc']:.3f} | PLV: {event['plv']:.3f}")

        # Per-harmonic metrics (using per-event FOOOF frequencies)
        if 'harmonic_msc' in event and 'harmonic_plv' in event:
            print(f"    Per-harmonic metrics:")
            harmonics_used = event.get('harmonics_used', SR_HARMONICS)
            for j in range(len(event['harmonic_msc'])):
                h_used = harmonics_used[j] if j < len(harmonics_used) else SR_HARMONICS[j]
                h_msc = event['harmonic_msc'][j]
                h_plv = event['harmonic_plv'][j]
                h_z = event['harmonic_z_scores'][j] if 'harmonic_z_scores' in event else np.nan
                print(f"      f{j} ({h_used:.2f} Hz): z={h_z:.2f}, MSC={h_msc:.3f}, PLV={h_plv:.3f}")

        # Print FOOOF results for this event
        if event_fooof.get('fooof_available') and not event_fooof.get('fit_failed', False):
            # Handle both per-harmonic and global FOOOF results
            if event_fooof.get('per_harmonic'):
                mean_beta = event_fooof.get('mean_beta', np.nan)
                mean_r2 = event_fooof.get('mean_r2', np.nan)
                n_matched = event_fooof.get('n_matched', 0)
                print(f"    FOOOF [per-harm]: mean β={mean_beta:.3f}, mean R²={mean_r2:.3f}, matched={n_matched}/{len(SR_HARMONICS)}")
            else:
                print(f"    FOOOF 1/f slope: {event_fooof['aperiodic_exponent']:.3f}")
                print(f"    FOOOF R²: {event_fooof['r_squared']:.3f}")
            harmonics_str = []
            for p in event_fooof['peaks']:
                if p.get('matched', False):
                    harmonics_str.append(f"{p['detected']:.2f}")
                else:
                    harmonics_str.append(f"[{p['canonical']:.1f}]")
            print(f"    FOOOF harmonics: {harmonics_str}")

    # Summary statistics
    if events:
        print(f"\n--- Summary Statistics ---")
        peak_zs = [e['peak_z'] for e in events]
        msc_vals = [e['msc'] for e in events]
        plv_vals = [e['plv'] for e in events]
        durations = [e['duration_sec'] for e in events]

        print(f"  Total events: {len(events)}")
        print(f"  Event rate: {len(events) / (max_time / 60):.2f} per minute")
        print(f"  Mean peak z-score: {np.mean(peak_zs):.2f} +/- {np.std(peak_zs):.2f}")
        print(f"  Mean MSC: {np.mean(msc_vals):.3f} +/- {np.std(msc_vals):.3f}")
        print(f"  Mean PLV: {np.mean(plv_vals):.3f} +/- {np.std(plv_vals):.3f}")
        print(f"  Mean duration: {np.mean(durations):.1f}s +/- {np.std(durations):.1f}s")
        print(f"  Total time in ignition: {sum(durations):.1f}s ({100*sum(durations)/max_time:.1f}%)")

    # Print windows in format compatible with lib functions
    print(f"\n--- Detected Windows (for use with lib functions) ---")
    windows_int = [(int(a), int(np.ceil(b))) for a, b in windows]
    print(f"IGNITION_WINDOWS = {windows_int}")

    # ==========================================================================
    # DETAILED PER-EVENT FOOOF ANALYSIS
    # ==========================================================================
    print_detailed_event_fooof(events, event_fooof_results)

    # ==========================================================================
    # NON-SR PEAK COLLECTION AND CLUSTERING
    # ==========================================================================
    non_sr_peaks = []
    non_sr_cluster_results = None

    if COLLECT_NON_SR_PEAKS and NON_SR_CLUSTERING_AVAILABLE:
        print("\n--- Non-SR Peak Collection ---")
        non_sr_collector = NonSRPeakCollector(freq_range=(1.0, 50.0))

        # Collect from session-level FOOOF result (global fit has all_peaks_characterized)
        n_session_peaks = 0
        if session_fooof:
            all_peaks = session_fooof.get('all_peaks_characterized', [])
            for peak in all_peaks:
                if not peak.get('is_sr', True):  # Non-SR peaks
                    added = non_sr_collector.add_from_dict(
                        {'freq': peak['freq'], 'power': peak['power'], 'bandwidth': peak['bandwidth']},
                        session_id=session_name,
                        window_type='session'
                    )
                    if added:
                        n_session_peaks += 1
        print(f"  Session-level non-SR peaks: {n_session_peaks}")

        # Collect from per-event FOOOF results
        n_event_peaks = 0
        for i, (event, event_fooof) in enumerate(zip(events, event_fooof_results)):
            # Check for all_peaks_characterized (global FOOOF) or peaks (per-harmonic)
            all_peaks = event_fooof.get('all_peaks_characterized', [])
            if not all_peaks:
                # Per-harmonic mode: check 'peaks' list for unmatched peaks
                peaks = event_fooof.get('peaks', [])
                for peak in peaks:
                    if not peak.get('matched', True):  # Unmatched peaks
                        added = non_sr_collector.add_from_dict(
                            {'freq': peak.get('detected', peak.get('canonical', np.nan)),
                             'power': peak.get('power', np.nan),
                             'bandwidth': peak.get('bandwidth', np.nan)},
                            session_id=session_name,
                            window_type='ignition',
                            window_index=i,
                            window_start=event['start_sec'],
                            window_end=event['end_sec']
                        )
                        if added:
                            n_event_peaks += 1
            else:
                # Global mode: filter for non-SR peaks
                for peak in all_peaks:
                    if not peak.get('is_sr', True):
                        added = non_sr_collector.add_from_dict(
                            {'freq': peak['freq'], 'power': peak['power'], 'bandwidth': peak['bandwidth']},
                            session_id=session_name,
                            window_type='ignition',
                            window_index=i,
                            window_start=event['start_sec'],
                            window_end=event['end_sec']
                        )
                        if added:
                            n_event_peaks += 1

        print(f"  Event-level non-SR peaks: {n_event_peaks}")
        print(f"  Total non-SR peaks collected: {non_sr_collector.n_peaks}")

        # Store peaks for return
        non_sr_peaks = non_sr_collector.peaks

        # Run clustering if enough peaks
        if non_sr_collector.n_peaks >= NON_SR_MIN_PEAKS_FOR_CLUSTERING:
            print(f"\n--- Non-SR Peak Clustering ---")
            non_sr_cluster_results, _ = non_sr_collector.cluster_and_plot(
                method=NON_SR_CLUSTER_METHOD,
                title=f'Non-SR Peak Clusters: {session_name}',
                out_dir=session_dir,
                session_name=session_name,
                show=show_plots
            )
            print(f"  Clusters found: {non_sr_cluster_results['n_clusters']}")
            if non_sr_cluster_results['cluster_centers_hz']:
                centers_str = ', '.join(f"{c:.1f}" for c in non_sr_cluster_results['cluster_centers_hz'])
                print(f"  Cluster centers (Hz): {centers_str}")

            # Export peaks CSV
            output_files = non_sr_collector.export_results(
                non_sr_cluster_results, out_dir=session_dir, session_name=session_name
            )
            for _, path in output_files.items():
                print(f"  Saved: {path}")
        else:
            print(f"  Skipping clustering (need >= {NON_SR_MIN_PEAKS_FOR_CLUSTERING} peaks)")

    elif COLLECT_NON_SR_PEAKS and not NON_SR_CLUSTERING_AVAILABLE:
        print("\n--- Non-SR Peak Collection: SKIPPED (sklearn not available) ---")

    # ==========================================================================
    # NEW ANALYSES: Null Control, Harmonic Coherence, Effect Sizes
    # ==========================================================================

    # 1. Coherence null control (MSC/PLV in ignition vs random windows)
    if SKIP_COHERENCE_NULL:
        print("\n--- Coherence Null Control: SKIPPED (SKIP_COHERENCE_NULL=True) ---")
        coherence_null_results = {}
    else:
        coherence_null_results = coherence_null_control(
            eeg_data, actual_fs, windows, max_time, n_permutations=500
        )

    # 2. Harmonic coherence analysis (using session-tuned harmonics)
    print("\n--- Harmonic Coherence Analysis (FOOOF-tuned) ---")
    harmonic_coh = compute_harmonic_coherence(eeg_data, actual_fs, session_harmonics)
    if 'error' not in harmonic_coh:
        print(f"  Valid harmonics: {harmonic_coh['valid_harmonics']}")
        print(f"  Mean inter-harmonic correlation: {harmonic_coh['mean_harmonic_coherence']:.3f}")
        print(f"  Pairwise correlations:")
        for pair, corr in harmonic_coh['pairwise_correlations'].items():
            print(f"    {pair} Hz: r = {corr:.3f}")
        print(f"  Power ratios (relative to fundamental):")
        for freq, ratio in harmonic_coh['power_ratios'].items():
            print(f"    {freq}: {ratio:.3f}")

    # 3. Effect size analysis (ignition vs baseline, using FOOOF-tuned harmonics)
    effect_results = compute_effect_sizes(
        eeg_data, z_envelope, windows, actual_fs, max_time, session_harmonics
    )

    # 4. Harmonic bicoherence (quadratic phase coupling during ignition)
    if len(windows) > 0:
        baseline_windows = get_baseline_windows(windows, max_time)
        if len(baseline_windows) > 0:
            print("\n--- Computing Harmonic Bicoherence ---")
            bico_results = compute_harmonic_bicoherence(
                eeg_data, actual_fs, windows, baseline_windows, session_harmonics
            )
            print_bicoherence_results(bico_results)
        else:
            print("\n--- Harmonic Bicoherence: SKIPPED (no baseline windows) ---")
            bico_results = None
    else:
        print("\n--- Harmonic Bicoherence: SKIPPED (no ignition windows) ---")
        bico_results = None

    # ==========================================================================
    # Extended Visualization
    # ==========================================================================
    print(f"\n--- Generating Extended Visualization ---")

    # Compute bicoherence time series for detection plot
    bico_ts = None
    if len(session_harmonics) > 0:
        try:
            print("  Computing bicoherence time series for key harmonic triads...")
            bico_ts = compute_bicoherence_timeseries(
                eeg_data, actual_fs, timestamps, session_harmonics,
                win_sec=4.0, step_sec=1.0, smooth_window=12  # Moderate smoothing: 12 seconds
            )
            print(f"  Tracked {len(bico_ts['triad_labels'])} triads over {len(bico_ts['times'])} windows (smoothed)")
        except Exception as e:
            print(f"  Warning: Could not compute bicoherence time series: {e}")
            bico_ts = None

    plot_results(timestamps, z_envelope, eeg_data, windows, events,
                 bico_ts=bico_ts,
                 session_f0=session_harmonics[0] if len(session_harmonics) > 0 else None,
                 session_name=session_name,
                 output_dir=session_dir,
                 show=show_plots)

    # Separate effect sizes chart
    plot_effect_sizes(effect_results, session_name, session_dir, show=show_plots)

    # ==========================================================================
    # CSV EXPORT
    # ==========================================================================
    if EXPORT_CSV:
        export_to_csv(
            results={
                'events': events,
                'windows': windows,
                'session_fooof': session_fooof,
                'session_fooof_per_harmonic': session_fooof_per_harmonic,
                'session_harmonics': session_harmonics,
                'bico_results': bico_results if 'bico_results' in locals() else None,
                'max_time': max_time,
            },
            data_file=data_file,
            export_dir=output_dir
        )

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    # Save console output to markdown file
    capture.__exit__(None, None, None)
    console_output = capture.getvalue()
    console_md_path = os.path.join(session_dir, f"{session_name}.md")
    with open(console_md_path, 'w') as f:
        f.write(f"# EEG Ignition Detection Analysis\n\n")
        f.write(f"**Session:** {session_name}\n\n")
        f.write(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        f.write("```\n")
        f.write(console_output)
        f.write("\n```\n")
    print(f"\n  Console output saved to: {console_md_path}")

    return {
        'windows': windows,
        'events': events,
        'session_fooof': session_fooof,
        'session_fooof_per_harmonic': session_fooof_per_harmonic,
        'session_harmonics': session_harmonics,
        'event_fooof_results': event_fooof_results,
        'coherence_null': coherence_null_results,
        'harmonic_coherence': harmonic_coh,
        'effect_sizes': effect_results,
        'max_time': max_time,
        'non_sr_peaks': non_sr_peaks,
        'non_sr_cluster_results': non_sr_cluster_results
    }


def plot_bicoherence_triads(bico_results: Dict, output_path: str, show: bool = True):
    """Create detailed bicoherence triad visualization."""
    harmonics = bico_results['harmonics']
    labels = [f"{h:.1f}" for h in harmonics]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: Ignition bicoherence
    im0 = axes[0, 0].imshow(bico_results['B_ignition'], cmap='hot', vmin=0, vmax=1)
    axes[0, 0].set_title('Bicoherence (Ignition)\nQuadratic Phase Coupling', fontsize=12)
    axes[0, 0].set_xticks(range(len(harmonics)))
    axes[0, 0].set_yticks(range(len(harmonics)))
    axes[0, 0].set_xticklabels(labels, fontsize=9)
    axes[0, 0].set_yticklabels(labels, fontsize=9)
    axes[0, 0].set_xlabel('f2 (Hz)', fontsize=10)
    axes[0, 0].set_ylabel('f1 (Hz)', fontsize=10)
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # Panel 2: Baseline bicoherence
    im1 = axes[0, 1].imshow(bico_results['B_baseline'], cmap='hot', vmin=0, vmax=1)
    axes[0, 1].set_title('Bicoherence (Baseline)', fontsize=12)
    axes[0, 1].set_xticks(range(len(harmonics)))
    axes[0, 1].set_yticks(range(len(harmonics)))
    axes[0, 1].set_xticklabels(labels, fontsize=9)
    axes[0, 1].set_yticklabels(labels, fontsize=9)
    axes[0, 1].set_xlabel('f2 (Hz)', fontsize=10)
    axes[0, 1].set_ylabel('f1 (Hz)', fontsize=10)
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Panel 3: Delta bicoherence
    vmax_diff = np.max(np.abs(bico_results['delta_B']))
    if vmax_diff < 1e-10:
        vmax_diff = 0.1
    im2 = axes[1, 0].imshow(bico_results['delta_B'], cmap='RdBu_r',
                            vmin=-vmax_diff, vmax=vmax_diff)
    axes[1, 0].set_title('Δ Bicoherence (Ignition - Baseline)', fontsize=12)
    axes[1, 0].set_xticks(range(len(harmonics)))
    axes[1, 0].set_yticks(range(len(harmonics)))
    axes[1, 0].set_xticklabels(labels, fontsize=9)
    axes[1, 0].set_yticklabels(labels, fontsize=9)
    axes[1, 0].set_xlabel('f2 (Hz)', fontsize=10)
    axes[1, 0].set_ylabel('f1 (Hz)', fontsize=10)
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Panel 4: Cohen's d (effect sizes)
    vmax_d = np.max(np.abs(bico_results['cohen_d']))
    if vmax_d < 1e-10:
        vmax_d = 1.0
    im3 = axes[1, 1].imshow(bico_results['cohen_d'], cmap='RdBu_r',
                            vmin=-vmax_d, vmax=vmax_d)
    axes[1, 1].set_title("Effect Size (Cohen's d)", fontsize=12)
    axes[1, 1].set_xticks(range(len(harmonics)))
    axes[1, 1].set_yticks(range(len(harmonics)))
    axes[1, 1].set_xticklabels(labels, fontsize=9)
    axes[1, 1].set_yticklabels(labels, fontsize=9)
    axes[1, 1].set_xlabel('f2 (Hz)', fontsize=10)
    axes[1, 1].set_ylabel('f1 (Hz)', fontsize=10)

    # Annotate significant pairs
    for i, j, d, f1, f2 in bico_results['significant_pairs'][:5]:  # Top 5
        axes[1, 1].text(j, i, f"{d:.1f}", ha='center', va='center',
                       fontsize=8, color='white' if abs(d) > vmax_d*0.5 else 'black',
                       fontweight='bold')

    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    if show:
        plt.show()
    plt.close()


def plot_extended_results(coherence_null_results: Dict, harmonic_coh: Dict, effect_results: Dict,
                          bico_results: Dict = None,
                          output_path: str = 'ignition_extended_analysis.png',
                          show: bool = True):
    """Plot coherence null control, harmonic coherence matrix, and bicoherence."""
    # Expand to 2x2 layout to accommodate bicoherence
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Coherence null distribution (MSC)
    ax1 = axes[0, 0]
    if coherence_null_results and 'null_msc' in coherence_null_results:
        null_msc = coherence_null_results['null_msc']
        obs_msc = coherence_null_results['obs_msc']
        p_msc = coherence_null_results['p_msc']
        ax1.hist(null_msc, bins=25, color='steelblue', edgecolor='white', alpha=0.7,
                 label='Random windows')
        ax1.axvline(obs_msc, color='red', linewidth=2,
                    label=f"Ignition: {obs_msc:.3f}")
        ax1.axvline(np.percentile(null_msc, 95), color='orange', linestyle='--',
                    label=f"95th %ile: {np.percentile(null_msc, 95):.3f}")
        ax1.set_xlabel('MSC')
        ax1.set_ylabel('Count')
        sig_str = "SIGNIFICANT" if p_msc < 0.05 else "n.s."
        ax1.set_title(f"MSC Null Control\np = {p_msc:.4f} ({sig_str})")
        ax1.legend(fontsize=8)
    else:
        ax1.text(0.5, 0.5, 'No coherence data', ha='center', va='center')

    # 2. Harmonic correlation matrix
    ax2 = axes[0, 1]
    if 'correlation_matrix' in harmonic_coh:
        corr = harmonic_coh['correlation_matrix']
        freqs = harmonic_coh['valid_harmonics']
        im = ax2.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
        ax2.set_xticks(range(len(freqs)))
        ax2.set_yticks(range(len(freqs)))
        ax2.set_xticklabels([f"{f:.1f}" for f in freqs], rotation=45)
        ax2.set_yticklabels([f"{f:.1f}" for f in freqs])
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Frequency (Hz)')
        ax2.set_title(f"Harmonic Envelope Correlations\nMean r = {harmonic_coh['mean_harmonic_coherence']:.3f}")
        plt.colorbar(im, ax=ax2, shrink=0.8)
    else:
        ax2.text(0.5, 0.5, 'Insufficient harmonics', ha='center', va='center')

    # 3. Effect sizes bar chart
    ax3 = axes[1, 0]
    if effect_results:
        metrics = []
        d_values = []
        colors = []

        if 'z_envelope' in effect_results:
            metrics.append('Z-envelope')
            d_values.append(effect_results['z_envelope']['cohens_d'])
            colors.append('green')

        if 'msc' in effect_results:
            metrics.append('MSC')
            d_values.append(effect_results['msc']['cohens_d'])
            colors.append('purple')

        if 'plv' in effect_results:
            metrics.append('PLV')
            d_values.append(effect_results['plv']['cohens_d'])
            colors.append('cyan')

        if 'harmonics' in effect_results:
            for h, vals in effect_results['harmonics'].items():
                metrics.append(f"{h:.0f} Hz")
                d_values.append(vals['cohens_d'])
                colors.append('blue')

        bars = ax3.barh(metrics, d_values, color=colors, alpha=0.7)
        ax3.axvline(0, color='black', linewidth=0.5)
        ax3.axvline(0.2, color='gray', linestyle=':', alpha=0.5)
        ax3.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax3.axvline(0.8, color='gray', linestyle='-', alpha=0.5)
        ax3.set_xlabel("Cohen's d (Ignition vs Baseline)")
        ax3.set_title("Effect Sizes")

        # Add significance markers
        for i, (metric, d) in enumerate(zip(metrics, d_values)):
            if metric == 'Z-envelope' and 'z_envelope' in effect_results:
                p = effect_results['z_envelope']['p_value']
            elif metric == 'MSC' and 'msc' in effect_results:
                p = effect_results['msc']['p_value']
            elif metric == 'PLV' and 'plv' in effect_results:
                p = effect_results['plv']['p_value']
            elif 'harmonics' in effect_results and ' Hz' in metric:
                h = float(metric.replace(' Hz', ''))
                p = effect_results['harmonics'].get(h, {}).get('p_value', 1.0)
            else:
                p = 1.0
            if p < 0.05:
                ax3.text(d + 0.05, i, '*', fontsize=14, va='center')

    # 4. Bicoherence matrix (if available)
    ax4 = axes[1, 1]
    if bico_results is not None:
        harmonics = bico_results['harmonics']
        labels = [f"{h:.1f}" for h in harmonics]

        # Plot ignition bicoherence matrix
        im4 = ax4.imshow(bico_results['B_ignition'], cmap='hot',
                        vmin=0, vmax=1, aspect='auto')
        ax4.set_title('Bicoherence (Ignition)\nf1 + f2 coupling', fontsize=10)
        ax4.set_xticks(range(len(harmonics)))
        ax4.set_yticks(range(len(harmonics)))
        ax4.set_xticklabels(labels, fontsize=8)
        ax4.set_yticklabels(labels, fontsize=8)
        ax4.set_xlabel('f2 (Hz)', fontsize=9)
        ax4.set_ylabel('f1 (Hz)', fontsize=9)

        # Annotate significant triads
        for i, j, d, _, _ in bico_results['significant_pairs'][:5]:  # Top 5
            if i != j:  # Skip diagonal
                ax4.text(j, i, '*', ha='center', va='center',
                        color='cyan', fontsize=14, fontweight='bold')

        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    else:
        ax4.text(0.5, 0.5, 'No bicoherence data', ha='center', va='center')
        ax4.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def run_batch(output_dir: str = "exports"):
    """Run analysis on all configured datasets."""
    import traceback

    def list_csv_files(directory):
        if not os.path.exists(directory):
            return []
        return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]

    # Electrode configurations
    EPOC_ELECTRODES = ['EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5', 'EEG.P7',
                       'EEG.O1', 'EEG.O2', 'EEG.P8', 'EEG.FC6', 'EEG.F4',
                       'EEG.F8', 'EEG.AF4']
    INSIGHT_ELECTRODES = ['EEG.AF3', 'EEG.T7', 'EEG.Pz', 'EEG.T8', 'EEG.AF4']
    MUSE_ELECTRODES = ['EEG.AF7', 'EEG.AF8', 'EEG.TP9', 'EEG.TP10']

    # Dataset lists with device type
    # Format: (filepath, device, electrodes, header)
    datasets = []

    # EPOC files
    for filepath in [
        'data/test schumann_EPOCX_111270_2023.04.23T14.50.35.05.00.md.pm.bp.csv',
        'data/Test_06.11.20_14.28.18.md.pm.bp.csv',
        'data/20201229_29.12.20_11.27.57.md.pm.bp.csv',
        'data/med_EPOCX_111270_2021.06.12T09.50.52.04.00.md.bp.csv',
        'data/binaural_EPOCX_111270_2021.06.17T10.04.52.04.00.md.bp.csv',
    ]:
        datasets.append((filepath, 'emotiv', EPOC_ELECTRODES, 1))

    # PhySF files (EPOC format but header at row 0)
    for filepath in list_csv_files("data/PhySF"):
        datasets.append((filepath, 'emotiv', EPOC_ELECTRODES, 0))

    # Insight files (EPOC format with different electrodes)
    for filepath in list_csv_files("data/insight"):
        datasets.append((filepath, 'emotiv', INSIGHT_ELECTRODES, 1))

    # Muse files (different format entirely)
    for filepath in list_csv_files("data/muse"):
        datasets.append((filepath, 'muse', MUSE_ELECTRODES, 0))

    # Clear existing CSV files for fresh batch
    os.makedirs(output_dir, exist_ok=True)
    for csv_file in ['sessions.csv', 'events.csv']:
        path = os.path.join(output_dir, csv_file)
        if os.path.exists(path):
            os.remove(path)
            print(f"Cleared: {path}")

    results = []
    errors = []

    for filepath, device, electrodes, header in datasets:
        if not os.path.exists(filepath):
            print(f"SKIP (not found): {filepath}")
            continue

        session_name = os.path.splitext(os.path.basename(filepath))[0]
        print(f"\n{'='*60}")
        print(f"Processing: {session_name}")
        print(f"  Device: {device}, Electrodes: {len(electrodes)}, Header: {header}")
        print(f"{'='*60}")

        try:
            result = main(
                data_file=filepath,
                device=device,
                electrodes=electrodes,
                header=header,
                session_name=session_name,
                output_dir=output_dir,
                show_plots=False
            )
            results.append((session_name, 'success'))
        except Exception as e:
            print(f"ERROR processing {session_name}: {e}")
            traceback.print_exc()
            errors.append((session_name, str(e)))
            results.append((session_name, 'error'))

    # Print summary
    print(f"\n{'='*60}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"  Processed: {len(results)}")
    print(f"  Successful: {sum(1 for _, s in results if s == 'success')}")
    print(f"  Errors: {len(errors)}")
    if errors:
        print("\nErrors:")
        for name, err in errors:
            print(f"  - {name}: {err}")

    return results, errors


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='EEG Ignition Detection Analysis')
    parser.add_argument('--batch', action='store_true', help='Run batch processing on all datasets')
    parser.add_argument('--file', type=str, help='Single file to process')
    parser.add_argument('--device', type=str, default='emotiv', choices=['emotiv', 'muse'],
                        help='Device type (emotiv or muse)')
    parser.add_argument('--output', type=str, default='exports', help='Output directory')

    args = parser.parse_args()

    if args.batch:
        run_batch(output_dir=args.output)
    elif args.file:
        main(data_file=args.file, device=args.device, output_dir=args.output)
    else:
        main()  # Run with defaults (DATA_FILE)

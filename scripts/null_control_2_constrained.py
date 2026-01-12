#!/usr/bin/env python3
"""
Null Control Test 2: Random Frequency Triplets (CONSTRAINED)

Tests if Schumann Ignition Events (SIE) show significantly better φ-convergence
than random frequency triplets sampled from the same canonical frequency ranges.

Test Types:
- Constrained: Sample triplets from canonical SR ranges (f1: 7.6±0.6, f2: 20±1.0, f3: 32±2.0 Hz)
- Tests if SIE pairing is special vs random peaks in same bands

Pass Criteria:
- SIE events show significantly better φ-convergence (p < 0.01)
- Effect size > 0.5
- SIE errors in bottom 10% of random distribution
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
from scipy import stats
from dataclasses import dataclass

# Add lib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

from fooof_harmonics import detect_harmonics_fooof, _get_peak_params
from utilities import load_eeg_csv
from detect_ignition import detect_ignitions_session


# ============================================================================
# Constants
# ============================================================================

# Golden ratio and powers
PHI = (1 + np.sqrt(5)) / 2  # 1.618033988749895
PHI_SQ = PHI ** 2            # 2.618033988749895
PHI_CUBE = PHI ** 3          # 4.236067977499790

# Default canonical frequency windows (adjustable)
DEFAULT_F1_CENTER = 7.6
DEFAULT_F1_HALFBAND = 0.6

DEFAULT_F2_CENTER = 20.0
DEFAULT_F2_HALFBAND = 1.0

DEFAULT_F3_CENTER = 32.0
DEFAULT_F3_HALFBAND = 2.0

# Number of random triplets to generate
N_RANDOM_TRIPLETS = 10000

# Null control sampling method
# 'per_event': Sample random triplets within each event (conservative, controls for temporal context)
# 'global_pool': Pool all peaks across all events and sample globally (standard, more diverse)
SAMPLING_METHOD = 'global_pool'  # Recommended: 'global_pool'

# FOOOF parameters (match existing pipeline)
FOOOF_FREQ_RANGE = (1.0, 40.0)
FOOOF_FREQ_RANGES = [[5, 15], [15, 25], [25, 40]]  # Per-harmonic frequency ranges
FOOOF_MAX_N_PEAKS = 15  # Match pipeline
FOOOF_PEAK_THRESHOLD = 0.01  # Lower threshold to detect more peaks (match pipeline)
FOOOF_MIN_PEAK_HEIGHT = 0.01  # Lower floor (match pipeline)
FOOOF_PEAK_WIDTH_LIMITS = (0.5, 12.0)  # Wider range (match pipeline)

# EEG parameters
SAMPLING_RATE = 128  # Hz

# Device-specific channel configurations
EMOTIV_CHANNELS = ['EEG.AF3','EEG.AF4','EEG.F3','EEG.F4','EEG.F7','EEG.F8',
                   'EEG.FC5','EEG.FC6','EEG.P7','EEG.P8','EEG.O1','EEG.O2',
                   'EEG.T7','EEG.T8']
INSIGHT_CHANNELS = ['EEG.AF3','EEG.AF4','EEG.T7','EEG.T8','EEG.Pz']
MUSE_CHANNELS = ['EEG.AF7','EEG.AF8','EEG.TP9','EEG.TP10']


# ============================================================================
# Device Detection
# ============================================================================

def get_device_config(file_path: str):
    """
    Detect device type from file path and return appropriate configuration.

    Returns:
        dict with keys: device, channels, sr_channel, header
    """
    file_path_lower = file_path.lower()

    # Muse detection
    if 'muse' in file_path_lower or 'Muse' in file_path:
        return {
            'device': 'muse',
            'channels': MUSE_CHANNELS,
            'sr_channel': 'EEG.AF7',  # Frontal left
            'header': 0
        }

    # INSIGHT detection
    if 'insight' in file_path_lower or 'INSIGHT' in file_path:
        return {
            'device': 'emotiv',
            'channels': INSIGHT_CHANNELS,
            'sr_channel': 'EEG.AF3',  # Frontal left
            'header': 1
        }

    # PHYSF detection (special header)
    if 'PhySF' in file_path or 'physf' in file_path_lower:
        return {
            'device': 'emotiv',
            'channels': EMOTIV_CHANNELS,
            'sr_channel': 'EEG.F4',  # Right frontal (standard for SR detection)
            'header': 0
        }

    # Default: Emotiv
    return {
        'device': 'emotiv',
        'channels': EMOTIV_CHANNELS,
        'sr_channel': 'EEG.F4',
        'header': 1
    }


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class FrequencyTriplet:
    """Represents a frequency triplet (f1, f2, f3)"""
    f1: float  # Fundamental (~7.6 Hz)
    f2: float  # Third harmonic (~20 Hz)
    f3: float  # Fifth harmonic (~32 Hz)
    source: str  # 'SIE' or 'random'
    session: Optional[str] = None
    event_id: Optional[int] = None  # Which event this came from
    power1: Optional[float] = None
    power2: Optional[float] = None
    power3: Optional[float] = None


@dataclass
class EventPeaks:
    """All peaks detected in an SIE event window"""
    event_id: int
    session: str
    f1_peaks: np.ndarray  # Nx3 array [freq, power, bandwidth] for SR1 range
    f2_peaks: np.ndarray  # Nx3 array [freq, power, bandwidth] for SR3 range
    f3_peaks: np.ndarray  # Nx3 array [freq, power, bandwidth] for SR5 range
    sie_triplet: FrequencyTriplet  # The max-power SIE triplet from this event


@dataclass
class PhiAnalysis:
    """Results of φ-ratio analysis for a triplet"""
    triplet: FrequencyTriplet
    ratio_f2_f1: float  # Should be ≈ φ²
    ratio_f3_f1: float  # Should be ≈ φ³
    ratio_f3_f2: float  # Should be ≈ φ
    error_f2_f1: float  # |ratio - φ²|
    error_f3_f1: float  # |ratio - φ³|
    error_f3_f2: float  # |ratio - φ|
    mean_phi_error: float  # Average of all errors
    phi_convergence: float  # 1 / (1 + mean_phi_error)


# ============================================================================
# File Specifications
# ============================================================================

# Dataset selection - set to True/False to include/exclude each dataset
DATASET_SELECTION = {
    'FILES': True,      # Individual test files
    # 'KAGGLE': True,     # data/mainData
    # 'MPENG': True,      # data/mpeng
    # 'MPENG1': True,     # data/mpeng1
    # 'MPENG2': True,     # data/mpeng2
    # 'VEP': True,        # data/vep
    'PHYSF': True,      # data/PhySF
    'INSIGHT': True,    # data/insight
    'MUSE': True,       # data/muse
}


def get_all_files(dataset_selection: Optional[Dict] = None) -> List[str]:
    """
    Get all EEG files to analyze (same as used in pipeline).

    Args:
        dataset_selection: Dict with dataset names as keys and True/False as values.
                          If None, uses DATASET_SELECTION global.

    Returns:
        List of file paths
    """
    if dataset_selection is None:
        dataset_selection = DATASET_SELECTION

    FILES = [
        'data/test schumann_EPOCX_111270_2023.04.23T14.50.35.05.00.md.pm.bp.csv',
        'data/20201229_29.12.20_11.27.57.md.pm.bp.csv',
        'data/binaural_EPOCX_111270_2021.06.17T10.04.52.04.00.md.bp.csv',
    ]

    def list_csv_files(directory):
        """List all CSV files in directory"""
        if not os.path.exists(directory):
            return []
        return [(directory + "/" + f) for f in os.listdir(directory) if f.endswith('.csv')]

    # Build dataset dictionary
    datasets = {
        'FILES': FILES if dataset_selection.get('FILES', False) else [],
        'KAGGLE': list_csv_files("data/mainData") if dataset_selection.get('KAGGLE', False) else [],
        'MPENG': list_csv_files("data/mpeng") if dataset_selection.get('MPENG', False) else [],
        'MPENG1': list_csv_files("data/mpeng1") if dataset_selection.get('MPENG1', False) else [],
        'MPENG2': list_csv_files("data/mpeng2") if dataset_selection.get('MPENG2', False) else [],
        'VEP': list_csv_files("data/vep") if dataset_selection.get('VEP', False) else [],
        'PHYSF': list_csv_files("data/PhySF") if dataset_selection.get('PHYSF', False) else [],
        'INSIGHT': list_csv_files("data/insight") if dataset_selection.get('INSIGHT', False) else [],
        'MUSE': list_csv_files("data/muse") if dataset_selection.get('MUSE', False) else [],
    }

    # Combine selected datasets
    all_files = []
    for dataset_name, files in datasets.items():
        if dataset_selection.get(dataset_name, False):
            all_files.extend(files)
            print(f"  {dataset_name}: {len(files)} files")

    # Filter to existing files only
    existing_files = [f for f in all_files if os.path.exists(f)]

    return existing_files


# ============================================================================
# φ-Ratio Calculation
# ============================================================================

def calculate_phi_ratios(f1: float, f2: float, f3: float,
                        triplet: FrequencyTriplet) -> PhiAnalysis:
    """
    Calculate φ-ratio analysis for a frequency triplet.

    Expected ratios:
    - f2/f1 ≈ φ² (2.618) - SR3/SR1
    - f3/f1 ≈ φ³ (4.236) - SR5/SR1
    - f3/f2 ≈ φ (1.618) - SR5/SR3

    Args:
        f1, f2, f3: Frequency triplet
        triplet: FrequencyTriplet object

    Returns:
        PhiAnalysis with all ratios and errors
    """
    # Calculate ratios
    ratio_f2_f1 = f2 / f1
    ratio_f3_f1 = f3 / f1
    ratio_f3_f2 = f3 / f2

    # Calculate deviations from golden ratio relationships
    error_f2_f1 = abs(ratio_f2_f1 - PHI_SQ)
    error_f3_f1 = abs(ratio_f3_f1 - PHI_CUBE)
    error_f3_f2 = abs(ratio_f3_f2 - PHI)

    # Mean error across all three relationships
    mean_phi_error = np.mean([error_f2_f1, error_f3_f1, error_f3_f2])

    # Convergence score (higher = better)
    phi_convergence = 1.0 / (1.0 + mean_phi_error)

    return PhiAnalysis(
        triplet=triplet,
        ratio_f2_f1=ratio_f2_f1,
        ratio_f3_f1=ratio_f3_f1,
        ratio_f3_f2=ratio_f3_f2,
        error_f2_f1=error_f2_f1,
        error_f3_f1=error_f3_f1,
        error_f3_f2=error_f3_f2,
        mean_phi_error=mean_phi_error,
        phi_convergence=phi_convergence
    )


# ============================================================================
# SIE Event Detection with ALL Peaks
# ============================================================================

def detect_sie_events_with_peaks(file_path: str,
                                f1_center: float = DEFAULT_F1_CENTER,
                                f1_halfband: float = DEFAULT_F1_HALFBAND,
                                f2_center: float = DEFAULT_F2_CENTER,
                                f2_halfband: float = DEFAULT_F2_HALFBAND,
                                f3_center: float = DEFAULT_F3_CENTER,
                                f3_halfband: float = DEFAULT_F3_HALFBAND) -> Tuple[List[FrequencyTriplet], List[EventPeaks]]:
    """
    Detect SIE events AND extract ALL peaks in each event window for null control.

    For each detected event:
    1. Runs FOOOF to find max-power peak (SIE triplet)
    2. Also extracts ALL other peaks in SR windows for random sampling

    Args:
        file_path: Path to EEG CSV file
        f1_center, f1_halfband: SR1 frequency window
        f2_center, f2_halfband: SR3 frequency window
        f3_center, f3_halfband: SR5 frequency window

    Returns:
        Tuple of (sie_triplets, event_peaks_list)
        - sie_triplets: SIE events with max-power peaks
        - event_peaks_list: ALL peaks in each event for random sampling
    """
    try:
        # Detect device type and get appropriate configuration
        device_config = get_device_config(file_path)

        # Load EEG data with device-specific settings
        df = load_eeg_csv(
            file_path,
            electrodes=device_config['channels'],
            device=device_config['device'],
            fs=SAMPLING_RATE,
            header=device_config['header']
        )

        # Get timestamp column
        if 'Timestamp' not in df.columns:
            df['Timestamp'] = np.arange(len(df)) / SAMPLING_RATE

        # Create temporary output directory
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run full ignition detection pipeline (MATCH USER'S EXACT WORKFLOW)
            results, intervals = detect_ignitions_session(
                RECORDZ=df,
                sr_channel=device_config['sr_channel'],
                eeg_channels=device_config['channels'],
                time_col='Timestamp',
                out_dir=temp_dir,
                center_hz=f1_center,
                harmonics_hz=[f1_center, f2_center, f3_center],
                half_bw_hz=[f1_halfband, f2_halfband, f3_halfband],
                smooth_sec=0.01,
                z_thresh=3.0,
                min_isi_sec=2.0,
                window_sec=20.0,
                merge_gap_sec=10.0,
                sr_reference='auto-SSD',
                seed_method='latency',
                pel_band=(25, 45),
                harmonic_method='fooof_hybrid',
                fooof_freq_range=FOOOF_FREQ_RANGE,
                fooof_freq_ranges=FOOOF_FREQ_RANGES,
                fooof_max_n_peaks=FOOOF_MAX_N_PEAKS,
                fooof_peak_threshold=FOOOF_PEAK_THRESHOLD,
                fooof_min_peak_height=FOOOF_MIN_PEAK_HEIGHT,
                fooof_peak_width_limits=FOOOF_PEAK_WIDTH_LIMITS,
                fooof_match_method='power',
                make_passport=False,
                show=False,
                verbose=False,
                session_name=os.path.basename(file_path)
            )

        events_df = results.get('events', pd.DataFrame())

        if events_df.empty:
            return [], []

        sie_triplets = []
        event_peaks_list = []

        session_name = os.path.basename(file_path)
        timestamps = df['Timestamp'].values

        # Process each detected event
        for event_id, (row, (t_start, t_end)) in enumerate(zip(events_df.iterrows(), intervals)):
            _, row_data = row
            ignition_freqs = row_data.get('ignition_freqs', [])

            if isinstance(ignition_freqs, str):
                try:
                    ignition_freqs = eval(ignition_freqs)
                except:
                    continue

            if not ignition_freqs or len(ignition_freqs) < 3:
                continue

            # Create SIE triplet (max power peaks)
            sie_triplet = FrequencyTriplet(
                f1=ignition_freqs[0],
                f2=ignition_freqs[1],
                f3=ignition_freqs[2],
                source='SIE',
                session=session_name,
                event_id=event_id
            )
            sie_triplets.append(sie_triplet)

            # Now detect ALL peaks in this event window for random sampling
            # Run FOOOF on the SAME event window (same as detect_ignitions_session)
            # Use full dataframe with window parameter to match pipeline workflow
            try:
                _, fooof_result = detect_harmonics_fooof(
                    records=df,
                    channels=device_config['channels'],
                    fs=SAMPLING_RATE,
                    time_col='Timestamp',
                    window=[t_start, t_end],
                    f_can=[f1_center, f2_center, f3_center],
                    freq_range=FOOOF_FREQ_RANGE,
                    freq_ranges=FOOOF_FREQ_RANGES,
                    search_halfband=[f1_halfband, f2_halfband, f3_halfband],
                    per_harmonic_fits=True,
                    max_n_peaks=FOOOF_MAX_N_PEAKS,
                    peak_threshold=FOOOF_PEAK_THRESHOLD,
                    min_peak_height=FOOOF_MIN_PEAK_HEIGHT,
                    peak_width_limits=FOOOF_PEAK_WIDTH_LIMITS,
                    match_method='power',
                    combine='median'
                )

                # Extract ALL peaks from all models (per-harmonic fits return list of models)
                if isinstance(fooof_result.model, list):
                    # Per-harmonic fits: combine peaks from all models
                    all_peaks_list = []
                    for fm in fooof_result.model:
                        peaks = _get_peak_params(fm)
                        if len(peaks) > 0:
                            all_peaks_list.append(peaks)
                    if len(all_peaks_list) == 0:
                        continue
                    all_peaks = np.vstack(all_peaks_list)
                else:
                    # Single model
                    all_peaks = _get_peak_params(fooof_result.model)
                    if len(all_peaks) == 0:
                        continue

                # Filter peaks by SR windows - use specific halfband for each window
                f1_min, f1_max = f1_center - f1_halfband, f1_center + f1_halfband
                f2_min, f2_max = f2_center - f2_halfband, f2_center + f2_halfband
                f3_min, f3_max = f3_center - f3_halfband, f3_center + f3_halfband

                f1_peaks = all_peaks[(all_peaks[:, 0] >= f1_min) & (all_peaks[:, 0] <= f1_max)]
                f2_peaks = all_peaks[(all_peaks[:, 0] >= f2_min) & (all_peaks[:, 0] <= f2_max)]
                f3_peaks = all_peaks[(all_peaks[:, 0] >= f3_min) & (all_peaks[:, 0] <= f3_max)]

                # Only store if we have peaks in all windows
                if len(f1_peaks) > 0 and len(f2_peaks) > 0 and len(f3_peaks) > 0:
                    event_peaks = EventPeaks(
                        event_id=event_id,
                        session=session_name,
                        f1_peaks=f1_peaks,
                        f2_peaks=f2_peaks,
                        f3_peaks=f3_peaks,
                        sie_triplet=sie_triplet
                    )
                    event_peaks_list.append(event_peaks)

            except Exception as e:
                # Skip this event if FOOOF fails
                continue

        return sie_triplets, event_peaks_list

    except Exception as e:
        print(f"Warning: Failed to process {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return [], []


# ============================================================================
# Random Triplet Generation from Event Peaks
# ============================================================================

def generate_random_triplets_from_events(event_peaks_list: List[EventPeaks],
                                        n_triplets_per_event: int = 10) -> List[FrequencyTriplet]:
    """
    Generate random frequency triplets by sampling from ALL detected peaks
    within the SAME event windows as SIE events.

    PER-EVENT SAMPLING: Controls for temporal context, more conservative.

    Args:
        event_peaks_list: List of EventPeaks with ALL peaks from each SIE event
        n_triplets_per_event: Number of random triplets to generate per event

    Returns:
        List of random FrequencyTriplet objects
    """
    random_triplets = []

    for event_peaks in event_peaks_list:
        # Generate n random triplets from this event's peaks
        for _ in range(n_triplets_per_event):
            # Randomly sample one peak from each SR window
            f1_idx = np.random.randint(len(event_peaks.f1_peaks))
            f2_idx = np.random.randint(len(event_peaks.f2_peaks))
            f3_idx = np.random.randint(len(event_peaks.f3_peaks))

            triplet = FrequencyTriplet(
                f1=event_peaks.f1_peaks[f1_idx, 0],
                f2=event_peaks.f2_peaks[f2_idx, 0],
                f3=event_peaks.f3_peaks[f3_idx, 0],
                source='random',
                session=event_peaks.session,
                event_id=event_peaks.event_id,
                power1=event_peaks.f1_peaks[f1_idx, 1],
                power2=event_peaks.f2_peaks[f2_idx, 1],
                power3=event_peaks.f3_peaks[f3_idx, 1]
            )
            random_triplets.append(triplet)

    return random_triplets


def generate_random_triplets_global_pool(event_peaks_list: List[EventPeaks],
                                        n_triplets: int = 10000) -> List[FrequencyTriplet]:
    """
    Generate random frequency triplets by GLOBALLY POOLING all peaks across
    ALL events and sessions, then randomly sampling.

    GLOBAL POOLING: More diverse, tests if φ-ratios are special across entire dataset.
    This is the standard approach for null control tests.

    Args:
        event_peaks_list: List of EventPeaks with ALL peaks from each SIE event
        n_triplets: Total number of random triplets to generate

    Returns:
        List of random FrequencyTriplet objects
    """
    # Pool ALL peaks across ALL events
    all_f1_peaks = []
    all_f2_peaks = []
    all_f3_peaks = []

    for event_peaks in event_peaks_list:
        all_f1_peaks.append(event_peaks.f1_peaks)
        all_f2_peaks.append(event_peaks.f2_peaks)
        all_f3_peaks.append(event_peaks.f3_peaks)

    # Concatenate into global arrays
    f1_pool = np.vstack(all_f1_peaks) if all_f1_peaks else np.array([])
    f2_pool = np.vstack(all_f2_peaks) if all_f2_peaks else np.array([])
    f3_pool = np.vstack(all_f3_peaks) if all_f3_peaks else np.array([])

    if len(f1_pool) == 0 or len(f2_pool) == 0 or len(f3_pool) == 0:
        return []

    random_triplets = []
    for _ in range(n_triplets):
        # Randomly sample from global pools (independent sampling)
        f1_idx = np.random.randint(len(f1_pool))
        f2_idx = np.random.randint(len(f2_pool))
        f3_idx = np.random.randint(len(f3_pool))

        triplet = FrequencyTriplet(
            f1=f1_pool[f1_idx, 0],
            f2=f2_pool[f2_idx, 0],
            f3=f3_pool[f3_idx, 0],
            source='random_global',
            session='pooled',
            event_id=None,
            power1=f1_pool[f1_idx, 1],
            power2=f2_pool[f2_idx, 1],
            power3=f3_pool[f3_idx, 1]
        )
        random_triplets.append(triplet)

    return random_triplets


# ============================================================================
# Statistical Analysis
# ============================================================================

def perform_statistical_test(sie_analyses: List[PhiAnalysis],
                            random_analyses: List[PhiAnalysis]) -> Dict:
    """
    Perform statistical comparison between SIE and random triplets.

    Args:
        sie_analyses: List of PhiAnalysis for SIE events
        random_analyses: List of PhiAnalysis for random triplets

    Returns:
        Dictionary with statistical results
    """
    # Extract φ-errors
    sie_errors = np.array([a.mean_phi_error for a in sie_analyses])
    random_errors = np.array([a.mean_phi_error for a in random_analyses])

    # Basic statistics
    sie_mean = np.mean(sie_errors)
    sie_std = np.std(sie_errors)
    sie_median = np.median(sie_errors)

    random_mean = np.mean(random_errors)
    random_std = np.std(random_errors)
    random_median = np.median(random_errors)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((sie_std**2 + random_std**2) / 2)
    cohens_d = (random_mean - sie_mean) / pooled_std  # Positive = SIE better

    # Percentile ranking (what % of random are worse than SIE mean)
    percentile = 100 * np.sum(random_errors > sie_mean) / len(random_errors)

    # Mann-Whitney U test (non-parametric)
    u_statistic, p_value_mw = stats.mannwhitneyu(sie_errors, random_errors, alternative='less')

    # Permutation test
    observed_diff = sie_mean - random_mean
    combined = np.concatenate([sie_errors, random_errors])
    n_sie = len(sie_errors)

    n_permutations = 10000
    perm_diffs = []
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_sie = combined[:n_sie]
        perm_random = combined[n_sie:]
        perm_diffs.append(np.mean(perm_sie) - np.mean(perm_random))

    p_value_perm = np.sum(np.array(perm_diffs) <= observed_diff) / n_permutations

    # Check pass criteria
    passes_pvalue = p_value_perm < 0.01
    passes_effect_size = cohens_d > 0.5
    passes_percentile = percentile <= 10  # SIE errors in bottom 10% (GOOD)

    overall_pass = passes_pvalue and passes_effect_size and passes_percentile

    return {
        'sie_mean': sie_mean,
        'sie_std': sie_std,
        'sie_median': sie_median,
        'random_mean': random_mean,
        'random_std': random_std,
        'random_median': random_median,
        'cohens_d': cohens_d,
        'percentile': percentile,
        'p_value_mannwhitney': p_value_mw,
        'p_value_permutation': p_value_perm,
        'passes_pvalue': passes_pvalue,
        'passes_effect_size': passes_effect_size,
        'passes_percentile': passes_percentile,
        'overall_pass': overall_pass,
        'n_sie': len(sie_errors),
        'n_random': len(random_errors)
    }


# ============================================================================
# Visualization
# ============================================================================

def create_visualizations(sie_analyses: List[PhiAnalysis],
                         random_analyses: List[PhiAnalysis],
                         stats_results: Dict,
                         output_dir: Path):
    """
    Create comprehensive visualizations comparing SIE vs random triplets.

    Args:
        sie_analyses: PhiAnalysis results for SIE events
        random_analyses: PhiAnalysis results for random triplets
        stats_results: Statistical test results
        output_dir: Directory to save plots
    """
    sie_errors = np.array([a.mean_phi_error for a in sie_analyses])
    random_errors = np.array([a.mean_phi_error for a in random_analyses])

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 10

    # Create figure with 4 subplots
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. Histogram with distributions
    ax1 = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0, max(np.max(sie_errors), np.max(random_errors)), 50)
    ax1.hist(random_errors, bins=bins, alpha=0.6, label=f'Random (n={len(random_errors)})',
             color='gray', density=True)
    ax1.hist(sie_errors, bins=bins, alpha=0.8, label=f'SIE (n={len(sie_errors)})',
             color='red', density=True)
    ax1.axvline(stats_results['sie_mean'], color='red', linestyle='--', linewidth=2,
                label=f"SIE mean: {stats_results['sie_mean']:.4f}")
    ax1.axvline(stats_results['random_mean'], color='gray', linestyle='--', linewidth=2,
                label=f"Random mean: {stats_results['random_mean']:.4f}")
    ax1.set_xlabel('Mean φ-Error')
    ax1.set_ylabel('Density')
    ax1.set_title('φ-Error Distributions: SIE vs Random Triplets')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Box plot comparison
    ax2 = fig.add_subplot(gs[0, 1])
    data_for_box = [random_errors, sie_errors]
    bp = ax2.boxplot(data_for_box, labels=['Random', 'SIE'], patch_artist=True,
                     widths=0.6)
    bp['boxes'][0].set_facecolor('gray')
    bp['boxes'][1].set_facecolor('red')
    ax2.set_ylabel('Mean φ-Error')
    ax2.set_title('φ-Error Comparison')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add statistical annotation
    y_max = max(np.max(random_errors), np.max(sie_errors))
    ax2.text(0.5, 0.95, f"Cohen's d = {stats_results['cohens_d']:.3f}\np = {stats_results['p_value_permutation']:.4f}",
             transform=ax2.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 3. Cumulative distribution
    ax3 = fig.add_subplot(gs[1, 0])
    random_sorted = np.sort(random_errors)
    sie_sorted = np.sort(sie_errors)
    random_cdf = np.arange(1, len(random_sorted) + 1) / len(random_sorted)
    sie_cdf = np.arange(1, len(sie_sorted) + 1) / len(sie_sorted)

    ax3.plot(random_sorted, random_cdf, label='Random', color='gray', linewidth=2)
    ax3.plot(sie_sorted, sie_cdf, label='SIE', color='red', linewidth=2)
    ax3.axvline(stats_results['sie_mean'], color='red', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Mean φ-Error')
    ax3.set_ylabel('Cumulative Probability')
    ax3.set_title('Cumulative Distribution Functions')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Percentile plot
    ax4 = fig.add_subplot(gs[1, 1])
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    random_percentile_values = np.percentile(random_errors, percentiles)
    sie_percentile_values = np.percentile(sie_errors, percentiles)

    x = np.arange(len(percentiles))
    width = 0.35
    ax4.bar(x - width/2, random_percentile_values, width, label='Random', color='gray', alpha=0.6)
    ax4.bar(x + width/2, sie_percentile_values, width, label='SIE', color='red', alpha=0.8)
    ax4.set_xlabel('Percentile')
    ax4.set_ylabel('Mean φ-Error')
    ax4.set_title('Percentile Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{p}th' for p in percentiles])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # Add overall pass/fail annotation
    status = "PASS ✓" if stats_results['overall_pass'] else "FAIL ✗"
    color = 'green' if stats_results['overall_pass'] else 'red'
    fig.text(0.5, 0.98, f'Null Control Test: {status}',
             ha='center', va='top', fontsize=14, fontweight='bold', color=color)

    # Save figure
    plt.savefig(output_dir / 'phi_convergence_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'phi_convergence_analysis.pdf', bbox_inches='tight')
    plt.close()

    print(f"Visualizations saved to {output_dir}")


# ============================================================================
# Results Report
# ============================================================================

def generate_results_report(stats_results: Dict, output_dir: Path, sampling_method: str = 'global_pool'):
    """
    Generate 200-word results summary in markdown format.

    Args:
        stats_results: Statistical test results
        output_dir: Directory to save report
        sampling_method: 'global_pool' or 'per_event'
    """
    status = "PASSED" if stats_results['overall_pass'] else "FAILED"

    sampling_desc = {
        'global_pool': 'globally pooled across all events and sessions',
        'per_event': 'sampled within each event window'
    }.get(sampling_method, 'unknown')

    report = f"""# Null Control Test 2: Random Frequency Triplets (CONSTRAINED) - Results

**Status: {status}**

## Summary

We tested whether Schumann Ignition Events (SIE) exhibit significantly better φ-convergence than random frequency triplets sampled from the same canonical frequency ranges. SIE triplets (n={stats_results['n_sie']}) were compared against {stats_results['n_random']:,} random triplets {sampling_desc} from detected spectral peaks within windows centered at 7.6±0.6 Hz, 20±1.0 Hz, and 32±2.0 Hz.

## Statistical Results

- **Mean φ-error (SIE)**: {stats_results['sie_mean']:.4f} ± {stats_results['sie_std']:.4f}
- **Mean φ-error (Random)**: {stats_results['random_mean']:.4f} ± {stats_results['random_std']:.4f}
- **Effect size (Cohen's d)**: {stats_results['cohens_d']:.3f} {'✓' if stats_results['passes_effect_size'] else '✗'} (criterion: > 0.5)
- **P-value (permutation test)**: {stats_results['p_value_permutation']:.6f} {'✓' if stats_results['passes_pvalue'] else '✗'} (criterion: < 0.01)
- **SIE percentile rank**: {stats_results['percentile']:.1f}th {'✓' if stats_results['passes_percentile'] else '✗'} (criterion: ≤ 10th)

## Interpretation

SIE events show {'significantly' if stats_results['overall_pass'] else 'no significant'} improvement in φ-convergence compared to random spectral peaks. {'This supports the hypothesis that harmonic relationships in SIE events reflect genuine φ-ratio coupling rather than artifacts of common EEG spectral structure.' if stats_results['overall_pass'] else 'This suggests φ-ratios may be common in these frequency bands, or detection bias exists. Further investigation recommended.'}

## Pass Criteria

- [{'x' if stats_results['passes_pvalue'] else ' '}] p < 0.01
- [{'x' if stats_results['passes_effect_size'] else ' '}] Effect size > 0.5
- [{'x' if stats_results['passes_percentile'] else ' '}] SIE errors in bottom 10%

**Overall: {'PASS' if stats_results['overall_pass'] else 'FAIL'}**

---

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    # Save report
    report_path = output_dir / 'results_summary.md'
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"Results summary saved to {report_path}")

    return report


# ============================================================================
# Main Execution
# ============================================================================

def main(dataset_selection: Optional[Dict] = None):
    """
    Main execution function.

    Args:
        dataset_selection: Optional dict to control which datasets to include.
                          If None, uses DATASET_SELECTION global.
                          Example: {'FILES': True, 'KAGGLE': False, ...}
    """

    print("="*80)
    print("NULL CONTROL TEST 2: Random Frequency Triplets (CONSTRAINED)")
    print("="*80)
    print()

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'results/null_control_test_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()

    # Get all files
    print("Loading file list...")
    print("Selected datasets:")
    all_files = get_all_files(dataset_selection)
    print(f"\nTotal: {len(all_files)} EEG files")
    print()

    # Detect SIE events AND extract all peaks from each event window
    print("Detecting SIE events and extracting all peaks...")
    sie_triplets = []
    all_event_peaks = []

    for i, file_path in enumerate(all_files, 1):
        print(f"  [{i}/{len(all_files)}] {os.path.basename(file_path)}", end='')
        triplets, event_peaks = detect_sie_events_with_peaks(file_path)
        sie_triplets.extend(triplets)
        all_event_peaks.extend(event_peaks)
        print(f" -> {len(triplets)} SIE event(s), {len(event_peaks)} usable for null")

    print(f"\nTotal SIE events detected: {len(sie_triplets)}")
    print(f"Total events with peaks for null control: {len(all_event_peaks)}")
    print()

    # Generate random triplets using selected sampling method
    print(f"Generating random triplets (method: {SAMPLING_METHOD})...")

    if SAMPLING_METHOD == 'global_pool':
        # GLOBAL POOLING: Sample from all peaks across all events
        random_triplets = generate_random_triplets_global_pool(all_event_peaks, N_RANDOM_TRIPLETS)
        print(f"Total random triplets generated: {len(random_triplets)}")
        print(f"  (Global pooling: sampled from {len(all_event_peaks)} events)")
    else:
        # PER-EVENT SAMPLING: Sample within each event
        if len(all_event_peaks) > 0:
            triplets_per_event = max(1, N_RANDOM_TRIPLETS // len(all_event_peaks))
        else:
            triplets_per_event = 10
        random_triplets = generate_random_triplets_from_events(all_event_peaks, triplets_per_event)
        print(f"Total random triplets generated: {len(random_triplets)}")
        print(f"  (Per-event sampling: ~{triplets_per_event} per event from {len(all_event_peaks)} events)")
    print()

    # Check if we have enough data
    if len(sie_triplets) == 0:
        print("ERROR: No SIE events detected!")
        return

    if len(random_triplets) < 100:
        print("ERROR: Insufficient random triplets generated!")
        return

    # Calculate φ-ratios for all triplets
    print("Calculating φ-ratios...")
    sie_analyses = [calculate_phi_ratios(t.f1, t.f2, t.f3, t) for t in sie_triplets]
    random_analyses = [calculate_phi_ratios(t.f1, t.f2, t.f3, t) for t in random_triplets]
    print(f"  SIE: {len(sie_analyses)} analyses")
    print(f"  Random: {len(random_analyses)} analyses")
    print()

    # Perform statistical test
    print("Performing statistical analysis...")
    stats_results = perform_statistical_test(sie_analyses, random_analyses)
    print()

    # Print results
    print("="*80)
    print("RESULTS")
    print("="*80)
    print(f"SIE mean φ-error:    {stats_results['sie_mean']:.4f} ± {stats_results['sie_std']:.4f}")
    print(f"Random mean φ-error: {stats_results['random_mean']:.4f} ± {stats_results['random_std']:.4f}")
    print(f"Cohen's d:           {stats_results['cohens_d']:.3f} (criterion: > 0.5) {'✓' if stats_results['passes_effect_size'] else '✗'}")
    print(f"P-value:             {stats_results['p_value_permutation']:.6f} (criterion: < 0.01) {'✓' if stats_results['passes_pvalue'] else '✗'}")
    print(f"SIE percentile:      {stats_results['percentile']:.1f}th (criterion: ≤ 10th) {'✓' if stats_results['passes_percentile'] else '✗'}")
    print()
    print(f"Overall: {'PASS ✓' if stats_results['overall_pass'] else 'FAIL ✗'}")
    print("="*80)
    print()

    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(sie_analyses, random_analyses, stats_results, output_dir)
    print()

    # Generate report
    print("Generating results summary...")
    generate_results_report(stats_results, output_dir, sampling_method=SAMPLING_METHOD)
    print()

    # Save raw data
    print("Saving raw data...")

    # Save SIE data
    sie_df = pd.DataFrame([
        {
            'f1': a.triplet.f1,
            'f2': a.triplet.f2,
            'f3': a.triplet.f3,
            'session': a.triplet.session,
            'ratio_f2_f1': a.ratio_f2_f1,
            'ratio_f3_f1': a.ratio_f3_f1,
            'ratio_f3_f2': a.ratio_f3_f2,
            'error_f2_f1': a.error_f2_f1,
            'error_f3_f1': a.error_f3_f1,
            'error_f3_f2': a.error_f3_f2,
            'mean_phi_error': a.mean_phi_error,
            'phi_convergence': a.phi_convergence
        }
        for a in sie_analyses
    ])
    sie_df.to_csv(output_dir / 'sie_triplets.csv', index=False)

    # Save random data (sample of 1000 for file size)
    random_sample = random_analyses[:1000] if len(random_analyses) > 1000 else random_analyses
    random_df = pd.DataFrame([
        {
            'f1': a.triplet.f1,
            'f2': a.triplet.f2,
            'f3': a.triplet.f3,
            'session': a.triplet.session,
            'ratio_f2_f1': a.ratio_f2_f1,
            'ratio_f3_f1': a.ratio_f3_f1,
            'ratio_f3_f2': a.ratio_f3_f2,
            'error_f2_f1': a.error_f2_f1,
            'error_f3_f1': a.error_f3_f1,
            'error_f3_f2': a.error_f3_f2,
            'mean_phi_error': a.mean_phi_error,
            'phi_convergence': a.phi_convergence
        }
        for a in random_sample
    ])
    random_df.to_csv(output_dir / 'random_triplets_sample.csv', index=False)

    # Save statistics
    stats_df = pd.DataFrame([stats_results])
    stats_df.to_csv(output_dir / 'statistics.csv', index=False)

    print(f"Raw data saved to {output_dir}")
    print()

    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"All results saved to: {output_dir}")
    print()


if __name__ == "__main__":
    main()

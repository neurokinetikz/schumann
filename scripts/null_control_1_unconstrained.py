#!/usr/bin/env python3
"""
Null Control Test 3: Random Spectral Peaks

Tests if φ-ratios appear when randomly combining ANY spectral peaks from EEG,
without constraining to specific frequency windows.

This is the SIMPLEST, most direct null control (START HERE):
- Extract ALL spectral peaks from entire EEG recordings
- Randomly combine any 3 peaks into triplets
- Test if random triplets show φ-convergence similar to SIE events

If SIE events show significantly better φ-convergence than random peak triplets,
this demonstrates that φ-ratios are genuinely rare in EEG spectral structure.
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

# SIE detection parameters (to get SIE triplets)
DEFAULT_F1_CENTER = 7.6
DEFAULT_F1_HALFBAND = 0.6
DEFAULT_F2_CENTER = 20.0
DEFAULT_F2_HALFBAND = 1.0
DEFAULT_F3_CENTER = 32.0
DEFAULT_F3_HALFBAND = 2.0

# Number of random triplets to generate
N_RANDOM_TRIPLETS = 10000

# FOOOF parameters
FOOOF_FREQ_RANGE = (5.0, 45.0)
FOOOF_FREQ_RANGES = [[5, 15], [15, 25], [27, 37]]
FOOOF_MAX_N_PEAKS = 15
FOOOF_PEAK_THRESHOLD = 0.01
FOOOF_MIN_PEAK_HEIGHT = 0.01
FOOOF_PEAK_WIDTH_LIMITS = (0.5, 12.0)

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
    """Detect device type from file path and return appropriate configuration."""
    file_path_lower = file_path.lower()

    if 'muse' in file_path_lower or 'Muse' in file_path:
        return {
            'device': 'muse',
            'channels': MUSE_CHANNELS,
            'sr_channel': 'EEG.AF7',
            'header': 0
        }

    if 'insight' in file_path_lower or 'INSIGHT' in file_path:
        return {
            'device': 'emotiv',
            'channels': INSIGHT_CHANNELS,
            'sr_channel': 'EEG.AF3',
            'header': 1
        }

    if 'PhySF' in file_path or 'physf' in file_path_lower:
        return {
            'device': 'emotiv',
            'channels': EMOTIV_CHANNELS,
            'sr_channel': 'EEG.F4',
            'header': 0
        }

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
    f1: float
    f2: float
    f3: float
    source: str  # 'SIE' or 'random'
    session: Optional[str] = None
    power1: Optional[float] = None
    power2: Optional[float] = None
    power3: Optional[float] = None


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
# Dataset Selection
# ============================================================================

DATASET_SELECTION = {
    'FILES': True,      # Individual test files
    # 'KAGGLE': True,   # data/mainData
    # 'MPENG': True,    # data/mpeng
    # 'MPENG1': True,   # data/mpeng1
    # 'MPENG2': True,   # data/mpeng2
    # 'VEP': True,      # data/vep
    'PHYSF': True,      # data/PhySF
    'INSIGHT': True,    # data/insight
    'MUSE': True,       # data/muse
}


def get_all_files(dataset_selection: Optional[Dict] = None) -> List[str]:
    """Get all EEG files to analyze."""
    if dataset_selection is None:
        dataset_selection = DATASET_SELECTION

    FILES = [
        'data/test schumann_EPOCX_111270_2023.04.23T14.50.35.05.00.md.pm.bp.csv',
        'data/20201229_29.12.20_11.27.57.md.pm.bp.csv',
        'data/binaural_EPOCX_111270_2021.06.17T10.04.52.04.00.md.bp.csv',
    ]

    def list_csv_files(directory):
        if not os.path.exists(directory):
            return []
        return [(directory + "/" + f) for f in os.listdir(directory) if f.endswith('.csv')]

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

    all_files = []
    for dataset_name, files in datasets.items():
        if dataset_selection.get(dataset_name, False):
            all_files.extend(files)
            print(f"  {dataset_name}: {len(files)} files")

    existing_files = [f for f in all_files if os.path.exists(f)]
    return existing_files


# ============================================================================
# φ-Ratio Analysis
# ============================================================================

def analyze_phi_ratios(triplet: FrequencyTriplet) -> PhiAnalysis:
    """Calculate φ-ratio convergence for a frequency triplet."""
    # Sort frequencies
    f1, f2, f3 = sorted([triplet.f1, triplet.f2, triplet.f3])

    # Calculate ratios
    ratio_f2_f1 = f2 / f1
    ratio_f3_f1 = f3 / f1
    ratio_f3_f2 = f3 / f2

    # Calculate errors from golden ratio expectations
    error_f2_f1 = abs(ratio_f2_f1 - PHI_SQ)
    error_f3_f1 = abs(ratio_f3_f1 - PHI_CUBE)
    error_f3_f2 = abs(ratio_f3_f2 - PHI)

    # Mean error
    mean_phi_error = (error_f2_f1 + error_f3_f1 + error_f3_f2) / 3
    phi_convergence = 1 / (1 + mean_phi_error)

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
# SIE Detection
# ============================================================================

def detect_sie_triplets(file_path: str) -> List[FrequencyTriplet]:
    """Detect SIE events using the standard pipeline."""
    try:
        device_config = get_device_config(file_path)

        df = load_eeg_csv(
            file_path,
            electrodes=device_config['channels'],
            device=device_config['device'],
            fs=SAMPLING_RATE,
            header=device_config['header']
        )

        if 'Timestamp' not in df.columns:
            df['Timestamp'] = np.arange(len(df)) / SAMPLING_RATE

        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            results, intervals = detect_ignitions_session(
                RECORDZ=df,
                sr_channel=device_config['sr_channel'],
                eeg_channels=device_config['channels'],
                time_col='Timestamp',
                out_dir=temp_dir,
                center_hz=DEFAULT_F1_CENTER,
                harmonics_hz=[DEFAULT_F1_CENTER, DEFAULT_F2_CENTER, DEFAULT_F3_CENTER],
                half_bw_hz=[DEFAULT_F1_HALFBAND, DEFAULT_F2_HALFBAND, DEFAULT_F3_HALFBAND],
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
            return []

        sie_triplets = []
        session_name = os.path.basename(file_path)

        for _, row_data in events_df.iterrows():
            ignition_freqs = row_data.get('ignition_freqs', [])

            if isinstance(ignition_freqs, str):
                try:
                    ignition_freqs = eval(ignition_freqs)
                except:
                    continue

            if not ignition_freqs or len(ignition_freqs) < 3:
                continue

            sie_triplet = FrequencyTriplet(
                f1=ignition_freqs[0],
                f2=ignition_freqs[1],
                f3=ignition_freqs[2],
                source='SIE',
                session=session_name
            )
            sie_triplets.append(sie_triplet)

        return sie_triplets

    except Exception as e:
        print(f"Warning: Failed to process {file_path}: {e}")
        return []


# ============================================================================
# Extract ALL Peaks from Entire Recording
# ============================================================================

def extract_all_peaks(file_path: str) -> np.ndarray:
    """
    Extract ALL spectral peaks from entire EEG recording (UNCONSTRAINED).
    No filtering by frequency windows - just get all peaks.

    Returns:
        Nx3 array of [frequency, power, bandwidth] for all peaks
    """
    try:
        device_config = get_device_config(file_path)

        df = load_eeg_csv(
            file_path,
            electrodes=device_config['channels'],
            device=device_config['device'],
            fs=SAMPLING_RATE,
            header=device_config['header']
        )

        if 'Timestamp' not in df.columns:
            df['Timestamp'] = np.arange(len(df)) / SAMPLING_RATE

        # Run FOOOF on ENTIRE recording
        _, fooof_result = detect_harmonics_fooof(
            records=df,
            channels=device_config['channels'],
            fs=SAMPLING_RATE,
            time_col='Timestamp',
            window=None,  # Entire recording
            f_can=[DEFAULT_F1_CENTER, DEFAULT_F2_CENTER, DEFAULT_F3_CENTER],
            freq_range=FOOOF_FREQ_RANGE,
            freq_ranges=FOOOF_FREQ_RANGES,
            per_harmonic_fits=True,
            max_n_peaks=FOOOF_MAX_N_PEAKS,
            peak_threshold=FOOOF_PEAK_THRESHOLD,
            min_peak_height=FOOOF_MIN_PEAK_HEIGHT,
            peak_width_limits=FOOOF_PEAK_WIDTH_LIMITS,
            match_method='power',
            combine='median'
        )

        # Extract ALL peaks from all models
        if isinstance(fooof_result.model, list):
            all_peaks_list = []
            for fm in fooof_result.model:
                peaks = _get_peak_params(fm)
                if len(peaks) > 0:
                    all_peaks_list.append(peaks)
            if len(all_peaks_list) == 0:
                return np.array([])
            all_peaks = np.vstack(all_peaks_list)
        else:
            all_peaks = _get_peak_params(fooof_result.model)

        return all_peaks

    except Exception as e:
        print(f"Warning: Failed to extract peaks from {file_path}: {e}")
        return np.array([])


# ============================================================================
# Generate Random Triplets from ALL Peaks
# ============================================================================

def generate_random_triplets_unconstrained(all_peaks: np.ndarray,
                                          n_triplets: int = 10000) -> List[FrequencyTriplet]:
    """
    Generate random triplets by sampling ANY 3 peaks (UNCONSTRAINED).
    No filtering by frequency windows - truly random combinations.

    Args:
        all_peaks: Nx3 array of [frequency, power, bandwidth]
        n_triplets: Number of random triplets to generate

    Returns:
        List of random FrequencyTriplet objects
    """
    if len(all_peaks) < 3:
        return []

    random_triplets = []

    for _ in range(n_triplets):
        # Randomly sample 3 different peaks
        indices = np.random.choice(len(all_peaks), size=3, replace=False)

        triplet = FrequencyTriplet(
            f1=all_peaks[indices[0], 0],
            f2=all_peaks[indices[1], 0],
            f3=all_peaks[indices[2], 0],
            source='random_unconstrained',
            session='pooled',
            power1=all_peaks[indices[0], 1],
            power2=all_peaks[indices[1], 1],
            power3=all_peaks[indices[2], 1]
        )
        random_triplets.append(triplet)

    return random_triplets


# ============================================================================
# Statistical Testing (reuse from null_control_1)
# ============================================================================

def perform_statistical_test(sie_analyses: List[PhiAnalysis],
                            random_analyses: List[PhiAnalysis]) -> Dict:
    """Perform statistical comparison between SIE and random triplets."""

    sie_errors = np.array([a.mean_phi_error for a in sie_analyses])
    random_errors = np.array([a.mean_phi_error for a in random_analyses])

    # Basic statistics
    sie_mean = np.mean(sie_errors)
    sie_std = np.std(sie_errors)
    random_mean = np.mean(random_errors)
    random_std = np.std(random_errors)

    # Permutation test
    observed_diff = sie_mean - random_mean
    n_permutations = 10000
    null_distribution = []

    all_errors = np.concatenate([sie_errors, random_errors])
    for _ in range(n_permutations):
        shuffled = np.random.permutation(all_errors)
        group1 = shuffled[:len(sie_errors)]
        group2 = shuffled[len(sie_errors):]
        null_diff = np.mean(group1) - np.mean(group2)
        null_distribution.append(null_diff)

    p_value_permutation = np.mean(np.array(null_distribution) <= observed_diff)

    # Cohen's d effect size
    pooled_std = np.sqrt((sie_std**2 + random_std**2) / 2)
    cohens_d = (random_mean - sie_mean) / pooled_std if pooled_std > 0 else 0

    # Percentile ranking
    percentile = stats.percentileofscore(random_errors, sie_mean)

    # Pass criteria
    passes_pvalue = p_value_permutation < 0.01
    passes_effect_size = cohens_d > 0.5
    passes_percentile = percentile <= 10  # SIE errors in bottom 10% (GOOD)
    overall_pass = passes_pvalue and passes_effect_size and passes_percentile

    return {
        'n_sie': len(sie_errors),
        'n_random': len(random_errors),
        'sie_mean': sie_mean,
        'sie_std': sie_std,
        'random_mean': random_mean,
        'random_std': random_std,
        'p_value_permutation': p_value_permutation,
        'cohens_d': cohens_d,
        'percentile': percentile,
        'passes_pvalue': passes_pvalue,
        'passes_effect_size': passes_effect_size,
        'passes_percentile': passes_percentile,
        'overall_pass': overall_pass
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
    fig.text(0.5, 0.98, f'Null Control Test 3: Random Spectral Peaks -  {status}',
             ha='center', va='top', fontsize=14, fontweight='bold', color=color)

    # Save figure
    plt.savefig(output_dir / 'phi_convergence_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'phi_convergence_analysis.pdf', bbox_inches='tight')
    plt.close()

    print(f"Visualizations saved to {output_dir}")


# ============================================================================
# Results Report
# ============================================================================

def generate_results_report(stats_results: Dict, output_dir: Path):
    """
    Generate 200-word results summary in markdown format.

    Args:
        stats_results: Statistical test results
        output_dir: Directory to save report
    """
    status = "PASSED" if stats_results['overall_pass'] else "FAILED"

    report = f"""# Null Control Test 3: Random Spectral Peaks - Results

**Status: {status}**

## Summary

We tested whether Schumann Ignition Events (SIE) exhibit significantly better φ-convergence than random frequency triplets formed from ANY spectral peaks across the entire EEG recording (unconstrained). SIE triplets (n={stats_results['n_sie']}) were compared against {stats_results['n_random']:,} random triplets created by randomly selecting any 3 peaks from all detected spectral peaks (no frequency filtering).

## Statistical Results

- **Mean φ-error (SIE)**: {stats_results['sie_mean']:.4f} ± {stats_results['sie_std']:.4f}
- **Mean φ-error (Random)**: {stats_results['random_mean']:.4f} ± {stats_results['random_std']:.4f}
- **Effect size (Cohen's d)**: {stats_results['cohens_d']:.3f} {'✓' if stats_results['passes_effect_size'] else '✗'} (criterion: > 0.5)
- **P-value (permutation test)**: {stats_results['p_value_permutation']:.6f} {'✓' if stats_results['passes_pvalue'] else '✗'} (criterion: < 0.01)
- **SIE percentile rank**: {stats_results['percentile']:.1f}th {'✓' if stats_results['passes_percentile'] else '✗'} (criterion: ≤ 10th)

## Interpretation

SIE events show {'significantly' if stats_results['overall_pass'] else 'no significant'} improvement in φ-convergence compared to completely random spectral peak combinations. {'This provides strong evidence that harmonic relationships in SIE events reflect genuine φ-ratio coupling, not common EEG spectral structure artifacts.' if stats_results['overall_pass'] else 'This suggests φ-ratios may be common across all EEG frequencies, or detection bias exists. Further investigation recommended.'}

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
    """Main execution function."""

    print("="*80)
    print("NULL CONTROL TEST 3: Random Spectral Peaks")
    print("="*80)
    print()

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'results/null_control_unconstrained_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()

    # Get all files
    print("Loading file list...")
    print("Selected datasets:")
    all_files = get_all_files(dataset_selection)
    print(f"\nTotal: {len(all_files)} EEG files")
    print()

    # Load pre-detected SIE events from data/SIE.csv
    print("Loading pre-detected SIE events from data/SIE.csv...")
    sie_df = pd.read_csv('data/SIE.csv')

    # Filter to only events with all 3 harmonics detected
    sie_df_valid = sie_df.dropna(subset=['sr1', 'sr3', 'sr5'])
    print(f"  Loaded {len(sie_df_valid)} SIE events with complete triplets")

    # Convert to FrequencyTriplet objects
    sie_triplets = [
        FrequencyTriplet(
            f1=row['sr1'],
            f2=row['sr3'],
            f3=row['sr5'],
            source='SIE',
            session=row['session_name']
        )
        for _, row in sie_df_valid.iterrows()
    ]
    print(f"  Total SIE triplets: {len(sie_triplets)}\n")

    # Get unique session names for peak extraction
    sie_sessions = set(sie_df_valid['session_name'].unique())
    print(f"  SIE events found in {len(sie_sessions)} unique sessions\n")

    # Extract ALL peaks from recordings with SIE events (UNCONSTRAINED)
    print("Extracting ALL spectral peaks from recordings with SIE events...")
    all_peaks_list = []
    processed_files = []

    for i, file_path in enumerate(all_files, 1):
        file_basename = os.path.basename(file_path)

        # Check if this file matches any SIE session
        session_matches = [s for s in sie_sessions if s in file_basename or file_basename in s]

        if not session_matches:
            print(f"  [{i}/{len(all_files)}] {file_basename} - SKIP (no SIE events)")
            continue

        print(f"  [{i}/{len(all_files)}] {file_basename}", end='')
        peaks = extract_all_peaks(file_path)
        if len(peaks) > 0:
            all_peaks_list.append(peaks)
            processed_files.append(file_path)
        print(f" -> {len(peaks)} peaks")

    # Combine all peaks into global pool
    all_peaks = np.vstack([p for p in all_peaks_list if len(p) > 0])
    print(f"\nProcessed {len(processed_files)} files (skipped {len(all_files) - len(processed_files)} without SIE events)")
    print(f"Total peaks across all files: {len(all_peaks)}")
    print()

    # Generate random triplets (UNCONSTRAINED - any 3 peaks)
    print(f"Generating {N_RANDOM_TRIPLETS} random triplets (unconstrained)...")
    random_triplets = generate_random_triplets_unconstrained(all_peaks, N_RANDOM_TRIPLETS)
    print(f"Total random triplets generated: {len(random_triplets)}")
    print()

    # Check if we have enough data
    if len(sie_triplets) == 0:
        print("ERROR: No SIE events detected!")
        return

    if len(random_triplets) < 100:
        print("ERROR: Not enough random triplets generated!")
        return

    # Analyze φ-convergence
    print("Analyzing φ-convergence...")
    sie_analyses = [analyze_phi_ratios(t) for t in sie_triplets]
    random_analyses = [analyze_phi_ratios(t) for t in random_triplets]
    print()

    # Statistical testing
    print("Running statistical tests...")
    stats_results = perform_statistical_test(sie_analyses, random_analyses)
    print()

    # Print summary
    print("="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"SIE events: {stats_results['n_sie']}")
    print(f"Random triplets: {stats_results['n_random']}")
    print()
    print(f"Mean φ-error (SIE): {stats_results['sie_mean']:.4f} ± {stats_results['sie_std']:.4f}")
    print(f"Mean φ-error (Random): {stats_results['random_mean']:.4f} ± {stats_results['random_std']:.4f}")
    print()
    print(f"P-value (permutation): {stats_results['p_value_permutation']:.6f} {'✓ PASS' if stats_results['passes_pvalue'] else '✗ FAIL'}")
    print(f"Effect size (Cohen's d): {stats_results['cohens_d']:.3f} {'✓ PASS' if stats_results['passes_effect_size'] else '✗ FAIL'}")
    print(f"SIE percentile: {stats_results['percentile']:.1f}th {'✓ PASS' if stats_results['passes_percentile'] else '✗ FAIL'}")
    print()
    print(f"OVERALL: {'✓ PASS' if stats_results['overall_pass'] else '✗ FAIL'}")
    print("="*80)
    print()

    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(sie_analyses, random_analyses, stats_results, output_dir)
    print()

    # Generate report
    print("Generating results summary...")
    generate_results_report(stats_results, output_dir)
    print()

    # Save raw data
    print("Saving raw data...")

    # Save SIE data
    sie_df_export = pd.DataFrame([
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
    sie_df_export.to_csv(output_dir / 'sie_triplets.csv', index=False)

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
    print(f"Results directory: {output_dir}")
    print()
    print("Output files:")
    print("  - phi_convergence_analysis.png/.pdf")
    print("  - results_summary.md")
    print("  - sie_triplets.csv")
    print("  - random_triplets_sample.csv")
    print("  - statistics.csv")


if __name__ == '__main__':
    main()

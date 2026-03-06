"""
Phi Validation Pipeline: Batch processing and publication tables
================================================================

Integration layer for running GED-based validation of the complete φⁿ
frequency model with all 8 position types per octave.

Key Features:
- Session-level phi validation with all position types
- Batch processing with timeout handling
- Publication table generators (Tables 1-4)
- 8-panel summary visualization

Dependencies: numpy, pandas, scipy, matplotlib, phi_frequency_model, ged_phi_analysis
"""

from __future__ import annotations
import os
import signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from scipy import stats
from pathlib import Path
from contextlib import contextmanager

# Import phi modules
try:
    from phi_frequency_model import (
        PHI, F0, POSITION_OFFSETS, POSITION_HIERARCHY, BANDS,
        PhiPrediction, PhiTable, generate_phi_table, get_default_phi_table,
        phi_distance, batch_assign_positions
    )
    from ged_phi_analysis import (
        ged_weights, ged_sweep_phi, ged_sweep_all_positions,
        ged_blind_with_positions, compute_alignment_stats,
        position_contrast, noble_hierarchy_test, phi_vs_null,
        extract_eeg_matrix, infer_fs_from_records
    )
except ImportError:
    from lib.phi_frequency_model import (
        PHI, F0, POSITION_OFFSETS, POSITION_HIERARCHY, BANDS,
        PhiPrediction, PhiTable, generate_phi_table, get_default_phi_table,
        phi_distance, batch_assign_positions
    )
    from lib.ged_phi_analysis import (
        ged_weights, ged_sweep_phi, ged_sweep_all_positions,
        ged_blind_with_positions, compute_alignment_stats,
        position_contrast, noble_hierarchy_test, phi_vs_null,
        extract_eeg_matrix, infer_fs_from_records
    )


# ============================================================================
# TIMEOUT MECHANISM
# ============================================================================

class TimeoutError(Exception):
    """Raised when a session times out."""
    pass


@contextmanager
def timeout_handler(seconds: int, session_id: str = "unknown"):
    """Context manager to timeout stuck session processing."""
    def _timeout_handler(signum, frame):
        raise TimeoutError(f"Session {session_id} timed out after {seconds}s")

    # Store old handler and set new one
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# ============================================================================
# DEVICE ELECTRODE CONFIGURATIONS
# ============================================================================

INSIGHT_ELECTRODES = ['EEG.AF3', 'EEG.T7', 'EEG.Pz', 'EEG.T8', 'EEG.AF4']
MUSE_ELECTRODES = ['EEG.AF7', 'EEG.AF8', 'EEG.TP9', 'EEG.TP10']
EPOC_ELECTRODES = ['EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5', 'EEG.T7', 'EEG.P7',
                   'EEG.O1', 'EEG.O2', 'EEG.P8', 'EEG.T8', 'EEG.FC6', 'EEG.F4',
                   'EEG.F8', 'EEG.AF4']


# ============================================================================
# SESSION-LEVEL VALIDATION
# ============================================================================

def run_phi_validation_session(
    records: pd.DataFrame,
    ignition_windows: List[Tuple[int, int]],
    eeg_channels: List[str],
    session_id: str,
    phi_table: Optional[PhiTable] = None,
    fs: float = 128.0,
    time_col: str = 'Timestamp',
    run_blind_sweep: bool = True,
    run_position_sweep: bool = True,
    run_baseline_comparison: bool = True,
    blind_freq_range: Tuple[float, float] = (3.0, 45.0),
    blind_n_peaks: int = 12,
    alignment_tol: float = 0.05,
    n_permutations: int = 500,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run complete phi validation on a single session.

    Parameters
    ----------
    records : pd.DataFrame
        EEG data with electrode columns
    ignition_windows : List[Tuple[int, int]]
        List of (start_sec, end_sec) tuples for ignition windows
    eeg_channels : List[str]
        Channel names (e.g., ['EEG.F4', 'EEG.O1'])
    session_id : str
        Session identifier
    phi_table : PhiTable, optional
        Custom phi table (defaults to standard 3-55 Hz)
    fs : float
        Sampling rate (Hz)
    run_blind_sweep : bool
        Run blind frequency discovery
    run_position_sweep : bool
        Run sweeps for all phi positions
    run_baseline_comparison : bool
        Run baseline (non-ignition) comparison
    blind_freq_range : Tuple[float, float]
        Frequency range for blind sweep
    blind_n_peaks : int
        Number of peaks to extract from blind sweep
    alignment_tol : float
        Relative tolerance for alignment (default 5%)
    n_permutations : int
        Number of permutations for statistical tests
    metadata : Dict, optional
        Additional session metadata

    Returns
    -------
    Dict with validation results
    """
    if phi_table is None:
        phi_table = get_default_phi_table()

    result = {
        'session_id': session_id,
        'metadata': metadata or {},
        'n_ignition_windows': len(ignition_windows),
        'blind_results': None,
        'position_results': None,
        'baseline_results': None,
        'contrast_results': None,
        'alignment_stats': None,
        'summary': {}
    }

    # Get available channels
    available_channels = [ch for ch in eeg_channels if ch in records.columns]
    if len(available_channels) == 0:
        # Try without EEG. prefix
        available_channels = [f'EEG.{ch}' for ch in eeg_channels if f'EEG.{ch}' in records.columns]
    if len(available_channels) == 0:
        result['summary']['error'] = 'No valid EEG channels found'
        return result

    # Get timestamps and convert to relative
    if time_col not in records.columns:
        records = records.copy()
        records[time_col] = np.arange(len(records)) / fs
    timestamps = records[time_col].values

    t0 = timestamps[0]
    if t0 > 1e9:  # Unix timestamp detected
        rel_timestamps = timestamps - t0
    else:
        rel_timestamps = timestamps

    # -------------------------------------------------------------------------
    # BLIND SWEEP
    # -------------------------------------------------------------------------
    if run_blind_sweep and len(ignition_windows) > 0:
        combined_data = []
        for t_start, t_end in ignition_windows:
            mask = (rel_timestamps >= t_start) & (rel_timestamps <= t_end)
            if mask.sum() > 0:
                window_data = np.vstack([
                    pd.to_numeric(records.loc[mask, ch], errors='coerce').fillna(0).values
                    for ch in available_channels
                ])
                combined_data.append(window_data)

        if combined_data:
            X_combined = np.hstack(combined_data)
            if X_combined.shape[1] >= 512:  # Minimum samples
                blind = ged_blind_with_positions(
                    X_combined, fs, phi_table,
                    freq_range=blind_freq_range,
                    n_peaks=blind_n_peaks,
                    alignment_tol=alignment_tol
                )
                result['blind_results'] = blind

                # Compute alignment statistics
                if blind['success'] and blind['peaks']:
                    align_stats = compute_alignment_stats(
                        blind['peaks'], phi_table,
                        n_permutations=n_permutations,
                        alignment_tol=alignment_tol
                    )
                    result['alignment_stats'] = align_stats

                    # Summary
                    result['summary']['blind_alignment_fraction'] = blind['alignment_fraction']
                    result['summary']['blind_alignment_pvalue'] = align_stats['p_value']
                    result['summary']['blind_effect_size'] = align_stats['effect_size']
                    result['summary']['position_histogram'] = blind['position_histogram']

    # -------------------------------------------------------------------------
    # POSITION SWEEP (all phi positions)
    # -------------------------------------------------------------------------
    if run_position_sweep and len(ignition_windows) > 0:
        combined_data = []
        for t_start, t_end in ignition_windows:
            mask = (rel_timestamps >= t_start) & (rel_timestamps <= t_end)
            if mask.sum() > 0:
                window_data = np.vstack([
                    pd.to_numeric(records.loc[mask, ch], errors='coerce').fillna(0).values
                    for ch in available_channels
                ])
                combined_data.append(window_data)

        if combined_data:
            X_combined = np.hstack(combined_data)
            if X_combined.shape[1] >= 512:
                position_df = ged_sweep_all_positions(X_combined, fs, phi_table)
                result['position_results'] = position_df

                # Position type contrast
                boundary_vs_attractor = position_contrast(
                    position_df,
                    ['boundary'], ['attractor'],
                    metrics=['fwhm', 'q_factor', 'peak_eigenvalue']
                )
                result['contrast_results'] = {
                    'boundary_vs_attractor': boundary_vs_attractor
                }

                # Noble hierarchy
                noble_test = noble_hierarchy_test(position_df, 'peak_eigenvalue')
                result['noble_hierarchy'] = noble_test

                # Summary statistics
                result['summary']['mean_freq_deviation_pct'] = position_df['freq_deviation_pct'].abs().mean()
                result['summary']['mean_fwhm'] = position_df['fwhm'].mean()
                result['summary']['mean_q_factor'] = position_df['q_factor'].mean()
                result['summary']['mean_eigenvalue'] = position_df['peak_eigenvalue'].mean()

    # -------------------------------------------------------------------------
    # BASELINE COMPARISON
    # -------------------------------------------------------------------------
    if run_baseline_comparison and len(ignition_windows) > 0:
        # Find baseline windows (non-ignition periods)
        total_duration = rel_timestamps[-1] - rel_timestamps[0]
        baseline_windows = _extract_baseline_windows(
            ignition_windows, total_duration, min_duration=5.0
        )

        if baseline_windows:
            baseline_data = []
            for t_start, t_end in baseline_windows[:len(ignition_windows)]:  # Match count
                mask = (rel_timestamps >= t_start) & (rel_timestamps <= t_end)
                if mask.sum() > 0:
                    window_data = np.vstack([
                        pd.to_numeric(records.loc[mask, ch], errors='coerce').fillna(0).values
                        for ch in available_channels
                    ])
                    baseline_data.append(window_data)

            if baseline_data:
                X_baseline = np.hstack(baseline_data)
                if X_baseline.shape[1] >= 512:
                    baseline_df = ged_sweep_all_positions(X_baseline, fs, phi_table)
                    result['baseline_results'] = baseline_df

                    # Compare ignition vs baseline
                    if result['position_results'] is not None:
                        ign_df = result['position_results']
                        base_df = baseline_df

                        comparison = {}
                        for metric in ['fwhm', 'q_factor', 'peak_eigenvalue']:
                            ign_vals = ign_df[metric].dropna().values
                            base_vals = base_df[metric].dropna().values

                            if len(ign_vals) > 1 and len(base_vals) > 1:
                                t_stat, t_pval = stats.ttest_ind(ign_vals, base_vals)
                                cohens_d = (np.mean(ign_vals) - np.mean(base_vals)) / \
                                           (np.sqrt((np.var(ign_vals) + np.var(base_vals)) / 2) + 1e-6)

                                comparison[metric] = {
                                    'ignition_mean': float(np.mean(ign_vals)),
                                    'baseline_mean': float(np.mean(base_vals)),
                                    'delta': float(np.mean(ign_vals) - np.mean(base_vals)),
                                    't_pvalue': float(t_pval),
                                    'cohens_d': float(cohens_d)
                                }

                        result['ignition_baseline_comparison'] = comparison

    result['summary']['session_id'] = session_id
    result['summary']['n_ignition_windows'] = len(ignition_windows)
    result['summary']['n_channels'] = len(available_channels)

    return result


def _extract_baseline_windows(
    ignition_windows: List[Tuple[int, int]],
    total_duration: float,
    min_duration: float = 5.0
) -> List[Tuple[float, float]]:
    """Extract non-ignition baseline windows."""
    baseline_windows = []

    # Sort ignition windows
    sorted_windows = sorted(ignition_windows)

    # Before first ignition
    if sorted_windows and sorted_windows[0][0] > min_duration:
        baseline_windows.append((0, sorted_windows[0][0] - 1))

    # Between ignitions
    for i in range(len(sorted_windows) - 1):
        gap_start = sorted_windows[i][1] + 1
        gap_end = sorted_windows[i + 1][0] - 1
        if gap_end - gap_start >= min_duration:
            baseline_windows.append((gap_start, gap_end))

    # After last ignition
    if sorted_windows and sorted_windows[-1][1] < total_duration - min_duration:
        baseline_windows.append((sorted_windows[-1][1] + 1, total_duration))

    return baseline_windows


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def batch_phi_validation(
    sessions: List[Dict],
    phi_table: Optional[PhiTable] = None,
    timeout_sec: int = 300,
    verbose: bool = True
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Run phi validation on multiple sessions with timeout handling.

    Parameters
    ----------
    sessions : List[Dict]
        Each dict should have:
        - 'records': DataFrame
        - 'ignition_windows': List[Tuple]
        - 'eeg_channels': List[str]
        - 'session_id': str
        - Optional: 'fs', 'metadata'
    phi_table : PhiTable, optional
        Custom phi table
    timeout_sec : int
        Timeout per session (seconds)
    verbose : bool
        Print progress

    Returns
    -------
    summary_df : DataFrame
        Summary statistics per session
    full_results : List[Dict]
        Full results for each session
    """
    if phi_table is None:
        phi_table = get_default_phi_table()

    summaries = []
    full_results = []

    for i, sess in enumerate(sessions):
        session_id = sess.get('session_id', f'session_{i}')

        if verbose:
            print(f"Processing {session_id} ({i+1}/{len(sessions)})...")

        try:
            with timeout_handler(timeout_sec, session_id):
                result = run_phi_validation_session(
                    records=sess['records'],
                    ignition_windows=sess['ignition_windows'],
                    eeg_channels=sess['eeg_channels'],
                    session_id=session_id,
                    phi_table=phi_table,
                    fs=sess.get('fs', 128.0),
                    metadata=sess.get('metadata')
                )

                full_results.append(result)
                summaries.append(result['summary'])

                if verbose:
                    print(f"  -> {result['summary'].get('n_ignition_windows', 0)} windows, "
                          f"alignment={result['summary'].get('blind_alignment_fraction', 0):.1%}")

        except TimeoutError as e:
            if verbose:
                print(f"  -> TIMEOUT: {e}")
            summaries.append({
                'session_id': session_id,
                'error': str(e)
            })
            full_results.append({'session_id': session_id, 'error': str(e)})

        except Exception as e:
            if verbose:
                print(f"  -> ERROR: {e}")
            summaries.append({
                'session_id': session_id,
                'error': str(e)
            })
            full_results.append({'session_id': session_id, 'error': str(e)})

    summary_df = pd.DataFrame(summaries)
    return summary_df, full_results


# ============================================================================
# PUBLICATION TABLE GENERATORS
# ============================================================================

def generate_table1_frequency_validation(
    results: List[Dict],
    phi_table: Optional[PhiTable] = None
) -> pd.DataFrame:
    """
    Generate Table 1: Phi Frequency Predictions vs GED Observations.

    Columns: Position, n, Predicted (Hz), GED Mean, GED Std, Deviation %, N
    """
    if phi_table is None:
        phi_table = get_default_phi_table()

    # Aggregate position results across sessions
    all_position_data = []

    for result in results:
        if 'position_results' not in result or result.get('position_results') is None:
            continue
        pos_df = result['position_results']
        if isinstance(pos_df, pd.DataFrame):
            all_position_data.append(pos_df)

    if not all_position_data:
        return pd.DataFrame()

    combined = pd.concat(all_position_data, ignore_index=True)

    # Group by label
    table_rows = []
    for label in combined['label'].unique():
        mask = combined['label'] == label
        subset = combined[mask]

        pred = phi_table.get(label)
        if pred is None:
            continue

        ged_mean = subset['optimal_freq'].mean()
        ged_std = subset['optimal_freq'].std()
        dev_pct = 100 * (ged_mean - pred.frequency) / pred.frequency

        table_rows.append({
            'Position': label,
            'Type': pred.position_type,
            'Band': pred.band,
            'n': pred.n,
            'Predicted (Hz)': pred.frequency,
            'GED Mean (Hz)': round(ged_mean, 2),
            'GED Std (Hz)': round(ged_std, 3),
            'Deviation %': round(dev_pct, 2),
            'N': len(subset)
        })

    df = pd.DataFrame(table_rows)
    return df.sort_values('n').reset_index(drop=True)


def generate_table2_position_properties(
    results: List[Dict]
) -> pd.DataFrame:
    """
    Generate Table 2: Position Type Characteristics.

    Columns: Type, N Positions, Mean FWHM, Mean Q, Mean Eigenvalue, Alignment %
    """
    all_position_data = []
    for result in results:
        if 'position_results' not in result or result.get('position_results') is None:
            continue
        pos_df = result['position_results']
        if isinstance(pos_df, pd.DataFrame):
            all_position_data.append(pos_df)

    if not all_position_data:
        return pd.DataFrame()

    combined = pd.concat(all_position_data, ignore_index=True)

    # Group by position type
    table_rows = []
    for pos_type in POSITION_OFFSETS.keys():
        mask = combined['position_type'] == pos_type
        subset = combined[mask]

        if len(subset) == 0:
            continue

        # Alignment: deviation < 5%
        aligned = (subset['freq_deviation_pct'].abs() < 5).sum()

        table_rows.append({
            'Position Type': pos_type,
            'N Positions': len(subset),
            'Mean FWHM (Hz)': round(subset['fwhm'].mean(), 3),
            'Std FWHM (Hz)': round(subset['fwhm'].std(), 3),
            'Mean Q-factor': round(subset['q_factor'].mean(), 2),
            'Mean Eigenvalue': round(subset['peak_eigenvalue'].mean(), 3),
            'Alignment %': round(100 * aligned / len(subset), 1)
        })

    return pd.DataFrame(table_rows)


def generate_table3_alignment_probability(
    results: List[Dict]
) -> pd.DataFrame:
    """
    Generate Table 3: Alignment Probability by Position Type.

    Shows permutation-based p-values for alignment to each position type.
    """
    table_rows = []

    for result in results:
        session_id = result.get('session_id', 'unknown')
        align_stats = result.get('alignment_stats')

        if align_stats is None:
            continue

        row = {
            'Session': session_id,
            'Overall Alignment': align_stats.get('observed_alignment', np.nan),
            'Overall p-value': align_stats.get('p_value', np.nan),
            'Effect Size': align_stats.get('effect_size', np.nan)
        }

        # Add per-position-type p-values
        type_pvals = align_stats.get('position_type_pvalues', {})
        for pos_type, pval in type_pvals.items():
            row[f'{pos_type}_pvalue'] = pval

        table_rows.append(row)

    return pd.DataFrame(table_rows)


def generate_table4_ignition_baseline(
    results: List[Dict]
) -> pd.DataFrame:
    """
    Generate Table 4: Ignition vs Baseline Contrast.

    Compares GED metrics between ignition and baseline windows.
    """
    table_rows = []

    for result in results:
        session_id = result.get('session_id', 'unknown')
        comparison = result.get('ignition_baseline_comparison')

        if comparison is None:
            continue

        row = {'Session': session_id}

        for metric, stats in comparison.items():
            if isinstance(stats, dict):
                row[f'{metric}_ignition'] = stats.get('ignition_mean', np.nan)
                row[f'{metric}_baseline'] = stats.get('baseline_mean', np.nan)
                row[f'{metric}_delta'] = stats.get('delta', np.nan)
                row[f'{metric}_cohens_d'] = stats.get('cohens_d', np.nan)
                row[f'{metric}_pvalue'] = stats.get('t_pvalue', np.nan)

        table_rows.append(row)

    return pd.DataFrame(table_rows)


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_phi_validation_summary(
    results: List[Dict],
    phi_table: Optional[PhiTable] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Generate 8-panel summary figure.

    Panels:
    A. GED freq vs phi predictions (scatter + identity line)
    B. Position histogram (blind peaks per position type)
    C. FWHM by position type (boxplot)
    D. Q-factor by position type (boxplot)
    E. Alignment heatmap (band x position)
    F. Ratio validation scatter
    G. Ignition vs baseline comparison
    H. Summary statistics text
    """
    if phi_table is None:
        phi_table = get_default_phi_table()

    fig, axes = plt.subplots(2, 4, figsize=figsize)
    axes = axes.flatten()

    # Collect data
    all_position_data = []
    all_blind_peaks = []
    position_histograms = []

    for result in results:
        if 'position_results' in result and result['position_results'] is not None:
            all_position_data.append(result['position_results'])
        if 'blind_results' in result and result['blind_results'] is not None:
            blind = result['blind_results']
            if blind.get('peaks'):
                all_blind_peaks.extend(blind['peaks'])
            if blind.get('position_histogram'):
                position_histograms.append(blind['position_histogram'])

    # Panel A: GED vs Predictions
    ax = axes[0]
    if all_position_data:
        combined = pd.concat(all_position_data, ignore_index=True)
        ax.scatter(combined['canonical_freq'], combined['optimal_freq'], alpha=0.5, s=20)
        lims = [combined['canonical_freq'].min(), combined['canonical_freq'].max()]
        ax.plot(lims, lims, 'r--', label='Identity')
        ax.set_xlabel('Predicted (Hz)')
        ax.set_ylabel('GED Optimal (Hz)')
        ax.set_title('A. GED vs Phi Predictions')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('A. GED vs Phi Predictions')

    # Panel B: Position histogram
    ax = axes[1]
    if position_histograms:
        combined_hist = {k: 0 for k in POSITION_OFFSETS.keys()}
        for hist in position_histograms:
            for k, v in hist.items():
                combined_hist[k] = combined_hist.get(k, 0) + v
        ax.bar(combined_hist.keys(), combined_hist.values())
        ax.set_xlabel('Position Type')
        ax.set_ylabel('Count')
        ax.set_title('B. Blind Peak Distribution')
        ax.tick_params(axis='x', rotation=45)
    else:
        ax.text(0.5, 0.5, 'No blind data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('B. Blind Peak Distribution')

    # Panel C: FWHM by position type
    ax = axes[2]
    if all_position_data:
        combined = pd.concat(all_position_data, ignore_index=True)
        pos_types = [pt for pt in POSITION_OFFSETS.keys() if pt in combined['position_type'].values]
        data_for_box = [combined[combined['position_type'] == pt]['fwhm'].dropna() for pt in pos_types]
        if any(len(d) > 0 for d in data_for_box):
            ax.boxplot(data_for_box, labels=pos_types)
            ax.set_ylabel('FWHM (Hz)')
            ax.set_title('C. FWHM by Position Type')
            ax.tick_params(axis='x', rotation=45)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('C. FWHM by Position Type')

    # Panel D: Q-factor by position type
    ax = axes[3]
    if all_position_data:
        combined = pd.concat(all_position_data, ignore_index=True)
        pos_types = [pt for pt in POSITION_OFFSETS.keys() if pt in combined['position_type'].values]
        data_for_box = [combined[combined['position_type'] == pt]['q_factor'].dropna() for pt in pos_types]
        if any(len(d) > 0 for d in data_for_box):
            ax.boxplot(data_for_box, labels=pos_types)
            ax.set_ylabel('Q-factor')
            ax.set_title('D. Q-factor by Position Type')
            ax.tick_params(axis='x', rotation=45)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('D. Q-factor by Position Type')

    # Panel E: Alignment heatmap
    ax = axes[4]
    if all_position_data:
        combined = pd.concat(all_position_data, ignore_index=True)
        # Create heatmap: bands x position types
        bands = list(BANDS.keys())[:5]  # theta, alpha, beta_low, beta_high, gamma
        pos_types = list(POSITION_OFFSETS.keys())
        heatmap_data = np.zeros((len(bands), len(pos_types)))

        for i, band in enumerate(bands):
            for j, pt in enumerate(pos_types):
                mask = (combined['band'] == band) & (combined['position_type'] == pt)
                subset = combined[mask]
                if len(subset) > 0:
                    aligned = (subset['freq_deviation_pct'].abs() < 5).mean()
                    heatmap_data[i, j] = aligned * 100

        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        ax.set_xticks(range(len(pos_types)))
        ax.set_xticklabels([p[:6] for p in pos_types], rotation=45, ha='right')
        ax.set_yticks(range(len(bands)))
        ax.set_yticklabels(bands)
        ax.set_title('E. Alignment % (Band x Position)')
        plt.colorbar(im, ax=ax, label='Alignment %')
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('E. Alignment Heatmap')

    # Panel F: Frequency deviation distribution
    ax = axes[5]
    if all_position_data:
        combined = pd.concat(all_position_data, ignore_index=True)
        ax.hist(combined['freq_deviation_pct'], bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='r', linestyle='--', label='Zero deviation')
        ax.axvline(-5, color='orange', linestyle=':', label='5% threshold')
        ax.axvline(5, color='orange', linestyle=':')
        ax.set_xlabel('Frequency Deviation (%)')
        ax.set_ylabel('Count')
        ax.set_title('F. Deviation Distribution')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('F. Deviation Distribution')

    # Panel G: Eigenvalue by position type
    ax = axes[6]
    if all_position_data:
        combined = pd.concat(all_position_data, ignore_index=True)
        pos_types = [pt for pt in POSITION_OFFSETS.keys() if pt in combined['position_type'].values]
        data_for_box = [combined[combined['position_type'] == pt]['peak_eigenvalue'].dropna() for pt in pos_types]
        if any(len(d) > 0 for d in data_for_box):
            ax.boxplot(data_for_box, labels=pos_types)
            ax.set_ylabel('Peak Eigenvalue')
            ax.set_title('G. Eigenvalue by Position Type')
            ax.tick_params(axis='x', rotation=45)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('G. Eigenvalue by Position Type')

    # Panel H: Summary statistics
    ax = axes[7]
    ax.axis('off')

    # Compute summary stats
    n_sessions = len(results)
    n_successful = sum(1 for r in results if 'error' not in r.get('summary', {}))

    summary_text = f"Summary Statistics\n{'='*30}\n\n"
    summary_text += f"Sessions: {n_successful}/{n_sessions}\n"

    if all_position_data:
        combined = pd.concat(all_position_data, ignore_index=True)
        summary_text += f"\nPosition Sweeps:\n"
        summary_text += f"  Mean deviation: {combined['freq_deviation_pct'].abs().mean():.2f}%\n"
        summary_text += f"  Mean FWHM: {combined['fwhm'].mean():.2f} Hz\n"
        summary_text += f"  Mean Q-factor: {combined['q_factor'].mean():.1f}\n"

    if all_blind_peaks:
        aligned = sum(1 for p in all_blind_peaks if p.get('is_aligned', False))
        summary_text += f"\nBlind Validation:\n"
        summary_text += f"  Total peaks: {len(all_blind_peaks)}\n"
        summary_text += f"  Aligned: {aligned} ({100*aligned/len(all_blind_peaks):.1f}%)\n"

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontsize=10, family='monospace', va='top')
    ax.set_title('H. Summary')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {output_path}")

    return fig


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_validation_results(
    summary_df: pd.DataFrame,
    full_results: List[Dict],
    output_dir: str,
    prefix: str = 'phi_validation'
) -> Dict[str, str]:
    """
    Export all validation results to files.

    Returns dict of output file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = {}

    # Summary CSV
    summary_path = os.path.join(output_dir, f'{prefix}_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    paths['summary'] = summary_path

    # Publication tables
    table1 = generate_table1_frequency_validation(full_results)
    if len(table1) > 0:
        table1_path = os.path.join(output_dir, f'{prefix}_table1_predictions.csv')
        table1.to_csv(table1_path, index=False)
        paths['table1'] = table1_path

    table2 = generate_table2_position_properties(full_results)
    if len(table2) > 0:
        table2_path = os.path.join(output_dir, f'{prefix}_table2_properties.csv')
        table2.to_csv(table2_path, index=False)
        paths['table2'] = table2_path

    table3 = generate_table3_alignment_probability(full_results)
    if len(table3) > 0:
        table3_path = os.path.join(output_dir, f'{prefix}_table3_alignment.csv')
        table3.to_csv(table3_path, index=False)
        paths['table3'] = table3_path

    table4 = generate_table4_ignition_baseline(full_results)
    if len(table4) > 0:
        table4_path = os.path.join(output_dir, f'{prefix}_table4_contrast.csv')
        table4.to_csv(table4_path, index=False)
        paths['table4'] = table4_path

    # Summary figure
    fig_path = os.path.join(output_dir, f'{prefix}_summary.png')
    plot_phi_validation_summary(full_results, output_path=fig_path)
    paths['figure'] = fig_path
    plt.close()

    return paths


if __name__ == "__main__":
    print("Phi Validation Pipeline")
    print("=" * 50)
    print("\nThis module provides:")
    print("  - run_phi_validation_session(): Single session validation")
    print("  - batch_phi_validation(): Multi-session batch processing")
    print("  - generate_table*(): Publication table generators")
    print("  - plot_phi_validation_summary(): 8-panel summary figure")
    print("\nSee scripts/run_phi_validation.py for usage example.")

"""
GED φⁿ Validation Pipeline
===========================

Integration layer for running GED-based validation of the φⁿ (golden ratio)
architecture hypothesis on Schumann Ignition Event data.

Key Features:
- Blind frequency sweep: GED discovers peaks without knowing φⁿ predictions
- Canonical sweep: GED optimizes around expected φⁿ frequencies
- Noble number validation: Tests inter-harmonic ratios against φⁿ expectations
- Boundary-attractor contrast: Statistical comparison of FWHM/Q-factor by type
- Independence-convergence test: Validates paradox (independent absolutes, converged ratios)

Output:
- Table 1: GED vs φⁿ frequency predictions
- Table 2: Noble number ratio validation
- Table 3: Boundary vs attractor properties
- Table 4: Independence-convergence test

Dependencies: numpy, pandas, scipy, matplotlib
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


# ============================================================================
# TIMEOUT MECHANISM FOR STUCK SESSIONS
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

# Import from existing modules
try:
    from ged_bounds import (
        PHI, F0, SR_HARMONIC_TABLE, EXPECTED_RATIOS,
        ged_weights, ged_frequency_sweep, ged_blind_sweep,
        process_ignition_windows, process_baseline_windows,
        validate_phi_ratios, attractor_boundary_contrast,
        ged_ignition_baseline_contrast, phi_distance, nobility_index
    )
    from utilities import load_eeg_csv, ELECTRODES
    from session_metadata import parse_session_metadata
except ImportError:
    from lib.ged_bounds import (
        PHI, F0, SR_HARMONIC_TABLE, EXPECTED_RATIOS,
        ged_weights, ged_frequency_sweep, ged_blind_sweep,
        process_ignition_windows, process_baseline_windows,
        validate_phi_ratios, attractor_boundary_contrast,
        ged_ignition_baseline_contrast, phi_distance, nobility_index
    )
    from lib.utilities import load_eeg_csv, ELECTRODES
    from lib.session_metadata import parse_session_metadata


# Device electrode configurations (with EEG. prefix as in CSV files)
INSIGHT_ELECTRODES = ['EEG.AF3', 'EEG.T7', 'EEG.Pz', 'EEG.T8', 'EEG.AF4']
MUSE_ELECTRODES = ['EEG.AF7', 'EEG.AF8', 'EEG.TP9', 'EEG.TP10']
EPOC_ELECTRODES = ['EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5', 'EEG.T7', 'EEG.P7',
                   'EEG.O1', 'EEG.O2', 'EEG.P8', 'EEG.T8', 'EEG.FC6', 'EEG.F4',
                   'EEG.F8', 'EEG.AF4']


# ============================================================================
# CORE VALIDATION FUNCTIONS
# ============================================================================

def run_ged_validation_session(
    records: pd.DataFrame,
    ignition_windows: List[Tuple[int, int]],
    eeg_channels: List[str],
    session_id: str,
    fs: float = 128.0,
    time_col: str = 'Timestamp',
    run_blind_sweep: bool = True,
    run_canonical_sweep: bool = True,
    run_baseline_comparison: bool = True,
    blind_freq_range: Tuple[float, float] = (3.0, 45.0),
    blind_n_peaks: int = 8,
    canonical_freqs: Optional[Dict[str, float]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run complete GED φⁿ validation on a single session.

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
    fs : float
        Sampling rate (Hz)
    run_blind_sweep : bool
        Run blind frequency discovery
    run_canonical_sweep : bool
        Run sweeps centered on canonical φⁿ frequencies
    run_baseline_comparison : bool
        Run baseline (non-ignition) comparison for ignition vs baseline contrast
    blind_freq_range : Tuple[float, float]
        Frequency range for blind sweep
    blind_n_peaks : int
        Number of peaks to extract from blind sweep
    canonical_freqs : Dict[str, float], optional
        Custom canonical frequencies (defaults to SR_HARMONIC_TABLE)
    metadata : Dict, optional
        Additional session metadata

    Returns
    -------
    Dict with:
        session_id : str
        metadata : Dict
        blind_results : Dict (if run_blind_sweep)
        canonical_results : List[Dict] (per-window results from canonical sweep)
        baseline_results : List[Dict] (per-window results from baseline, if run_baseline_comparison)
        ignition_baseline_contrast : Dict (statistical comparison, if run_baseline_comparison)
        phi_validation : Dict (ratio validation results)
        summary : Dict (session-level summary)
    """
    result = {
        'session_id': session_id,
        'metadata': metadata or {},
        'n_ignition_windows': len(ignition_windows),
        'blind_results': None,
        'canonical_results': [],
        'baseline_results': [],
        'ignition_baseline_contrast': None,
        'phi_validation': None,
        'summary': {}
    }

    if canonical_freqs is None:
        canonical_freqs = {k: v['freq'] for k, v in SR_HARMONIC_TABLE.items()}

    # Get available channels
    available_channels = [ch for ch in eeg_channels if ch in records.columns]
    if len(available_channels) == 0:
        result['summary']['error'] = 'No valid EEG channels found'
        return result

    # Get timestamps and convert to relative (seconds from start)
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

    # -------------------------------------------------------------------------
    # BLIND SWEEP (on full session or combined ignition windows)
    # -------------------------------------------------------------------------
    if run_blind_sweep and len(ignition_windows) > 0:
        # Combine all ignition window data for blind sweep
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
                blind = ged_blind_sweep(
                    X_combined, fs,
                    freq_range=blind_freq_range,
                    n_peaks=blind_n_peaks
                )
                result['blind_results'] = blind

                # Analyze blind peaks vs φⁿ predictions
                if blind['success'] and blind['peaks']:
                    phi_analysis = validate_blind_against_phi(
                        [p['frequency'] for p in blind['peaks']]
                    )
                    result['blind_phi_analysis'] = phi_analysis

    # -------------------------------------------------------------------------
    # CANONICAL SWEEP (per ignition window)
    # -------------------------------------------------------------------------
    if run_canonical_sweep and len(ignition_windows) > 0:
        canonical_results = process_ignition_windows(
            records, ignition_windows, available_channels,
            canonical_freqs=canonical_freqs,
            fs=fs, time_col=time_col,
            session_id=session_id, metadata=metadata
        )
        result['canonical_results'] = canonical_results

        # Aggregate phi validation across all windows
        if canonical_results:
            all_phi_scores = [r.get('phi_score', np.nan) for r in canonical_results]
            valid_scores = [s for s in all_phi_scores if np.isfinite(s)]
            if valid_scores:
                result['summary']['mean_phi_score'] = np.mean(valid_scores)
                result['summary']['std_phi_score'] = np.std(valid_scores)
                result['summary']['n_windows_with_phi'] = len(valid_scores)

    # -------------------------------------------------------------------------
    # BASELINE COMPARISON (non-ignition segments)
    # -------------------------------------------------------------------------
    if run_baseline_comparison and len(ignition_windows) > 0:
        baseline_results = process_baseline_windows(
            records, ignition_windows, available_channels,
            canonical_freqs=canonical_freqs,
            fs=fs, time_col=time_col,
            session_id=session_id, metadata=metadata
        )
        result['baseline_results'] = baseline_results

        # Compute ignition vs baseline contrast
        if result['canonical_results'] and baseline_results:
            contrast = ged_ignition_baseline_contrast(
                result['canonical_results'],
                baseline_results
            )
            result['ignition_baseline_contrast'] = contrast

            # Add summary statistics
            if 'aggregate' in contrast and contrast['aggregate']:
                for metric, stats in contrast['aggregate'].items():
                    if isinstance(stats, dict):
                        result['summary'][f'baseline_{metric}_delta'] = stats.get('delta', np.nan)
                        result['summary'][f'baseline_{metric}_cohens_d'] = stats.get('cohens_d', np.nan)
                        result['summary'][f'baseline_{metric}_pvalue'] = stats.get('p_value', np.nan)

        result['summary']['n_baseline_windows'] = len(baseline_results)

    # -------------------------------------------------------------------------
    # SESSION SUMMARY
    # -------------------------------------------------------------------------
    result['summary']['session_id'] = session_id
    result['summary']['n_ignition_windows'] = len(ignition_windows)
    result['summary']['n_canonical_results'] = len(result['canonical_results'])

    if result['blind_results'] and result['blind_results']['success']:
        result['summary']['n_blind_peaks'] = result['blind_results']['n_peaks_found']
        if 'blind_phi_analysis' in result:
            result['summary']['blind_phi_alignment'] = result['blind_phi_analysis']['alignment_fraction']

    return result


def validate_blind_against_phi(blind_peak_freqs: List[float], tolerance: float = 0.05) -> Dict[str, Any]:
    """
    Validate that blind-discovered peaks align with φⁿ predictions.

    This is the KEY validation: if GED discovers peaks at φⁿ positions
    without being told where to look, that's strong evidence the brain
    truly organizes around these frequencies.

    Parameters
    ----------
    blind_peak_freqs : List[float]
        Frequencies discovered by blind GED sweep
    tolerance : float
        Relative tolerance for "alignment" (default 5%)

    Returns
    -------
    Dict with:
        peak_analysis : List[Dict] - per-peak φⁿ alignment info
        alignment_fraction : float - fraction of peaks within tolerance of φⁿ
        mean_distance : float - mean relative distance to nearest φⁿ
        phi_n_coverage : List[float] - which φⁿ values were covered
    """
    # Generate all φⁿ predictions in the frequency range
    phi_n_values = {}
    for n in np.arange(-2, 5, 0.5):
        f_pred = F0 * (PHI ** n)
        if 2 <= f_pred <= 50:
            phi_n_values[n] = f_pred

    peak_analysis = []
    aligned_count = 0
    total_distance = 0
    covered_phi_n = set()

    for f in blind_peak_freqs:
        # Find nearest φⁿ prediction
        nearest_n = None
        nearest_dist = np.inf
        nearest_f_pred = None

        for n, f_pred in phi_n_values.items():
            dist = abs(f - f_pred) / f_pred
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_n = n
                nearest_f_pred = f_pred

        is_aligned = nearest_dist < tolerance

        if is_aligned:
            aligned_count += 1
            covered_phi_n.add(nearest_n)

        total_distance += nearest_dist

        peak_analysis.append({
            'frequency': f,
            'nearest_phi_n': nearest_n,
            'nearest_phi_freq': nearest_f_pred,
            'relative_distance': nearest_dist,
            'is_aligned': is_aligned
        })

    n_peaks = len(blind_peak_freqs)
    return {
        'peak_analysis': peak_analysis,
        'alignment_fraction': aligned_count / max(1, n_peaks),
        'mean_distance': total_distance / max(1, n_peaks),
        'phi_n_coverage': sorted(list(covered_phi_n)),
        'n_peaks': n_peaks,
        'n_aligned': aligned_count
    }


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def batch_ged_validation(
    session_configs: List[Dict[str, Any]],
    output_dir: str = 'exports_ged_validation',
    verbose: bool = True,
    session_timeout: int = 120  # seconds per session
) -> pd.DataFrame:
    """
    Run GED validation on multiple sessions.

    Parameters
    ----------
    session_configs : List[Dict]
        Each dict should contain:
            - 'filepath': path to EEG CSV
            - 'ignition_windows': List[Tuple[int, int]]
            - 'dataset': dataset name (for metadata)
            - Optional: 'electrodes', 'device', 'session_id'
    output_dir : str
        Output directory for results
    verbose : bool
        Print progress

    Returns
    -------
    pd.DataFrame with combined results
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'per_session'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'aggregate'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)

    all_results = []
    all_canonical = []
    all_baseline = []
    all_blind_peaks = []

    n_skipped = 0
    n_timeout = 0

    for i, config in enumerate(session_configs):
        try:
            filepath = config['filepath']
            ign_windows = config.get('ignition_windows', [])
            dataset = config.get('dataset', 'unknown')

            # Parse metadata
            metadata = parse_session_metadata(filepath, dataset)
            session_id = config.get('session_id', f"{metadata.get('subject', 'unk')}_{i:03d}")

            if verbose and (i + 1) % 10 == 0:
                print(f"[{i+1}/{len(session_configs)}] Processing {session_id}...")

            # Determine device and electrodes (electrodes already have EEG. prefix)
            device = config.get('device', metadata.get('device', 'epoc'))
            if device == 'insight':
                electrodes = config.get('electrodes', INSIGHT_ELECTRODES)
            elif device == 'muse':
                electrodes = config.get('electrodes', MUSE_ELECTRODES)
            else:
                electrodes = config.get('electrodes', EPOC_ELECTRODES)

            eeg_channels = electrodes  # Already have EEG. prefix

            # PhySF, MPeng, VEP, Emotions files don't have metadata header row - use header=0
            # EPOC/Insight files have metadata row - use header=1
            header = 0 if dataset in ('physf', 'mpeng', 'vep', 'emotions') else 1

            # Wrap the entire session processing in a timeout handler
            with timeout_handler(session_timeout, session_id):
                # Load data
                records = load_eeg_csv(filepath, electrodes=electrodes, device=device, header=header)

                # Infer fs
                if 'Timestamp' in records.columns:
                    dt = np.diff(records['Timestamp'].values[:1000])
                    dt = dt[np.isfinite(dt) & (dt > 0)]
                    fs = 1.0 / np.median(dt) if len(dt) > 0 else 128.0
                else:
                    fs = 128.0

                # Run validation
                result = run_ged_validation_session(
                    records, ign_windows, eeg_channels,
                    session_id=session_id,
                    fs=fs,
                    metadata=metadata
                )

            # Store summary
            summary = result['summary'].copy()
            summary['session_id'] = session_id
            summary['dataset'] = dataset
            summary['filepath'] = filepath
            summary.update(metadata)
            all_results.append(summary)

            # Store per-window canonical results
            for win_result in result.get('canonical_results', []):
                win_result['session_id'] = session_id
                win_result['dataset'] = dataset
                all_canonical.append(win_result)

            # Store per-window baseline results
            for win_result in result.get('baseline_results', []):
                win_result['session_id'] = session_id
                win_result['dataset'] = dataset
                all_baseline.append(win_result)

            # Store blind peaks
            if result.get('blind_results') and result['blind_results'].get('peaks'):
                for peak in result['blind_results']['peaks']:
                    peak['session_id'] = session_id
                    peak['dataset'] = dataset
                    all_blind_peaks.append(peak)

            # Save per-session results
            session_df = pd.DataFrame(result.get('canonical_results', []))
            session_df.to_csv(
                os.path.join(output_dir, 'per_session', f'{session_id}_ged_results.csv'),
                index=False
            )

        except TimeoutError as e:
            n_timeout += 1
            if verbose:
                print(f"TIMEOUT: {session_id} - skipping")
            continue
        except Exception as e:
            n_skipped += 1
            if verbose:
                print(f"Error processing {config.get('filepath', 'unknown')}: {e}")
            continue

    # Report skipped sessions
    if verbose and (n_timeout > 0 or n_skipped > 0):
        print(f"\nSkipped: {n_skipped} errors, {n_timeout} timeouts")

    # Create aggregate DataFrames
    results_df = pd.DataFrame(all_results)
    canonical_df = pd.DataFrame(all_canonical)
    baseline_df = pd.DataFrame(all_baseline)
    blind_df = pd.DataFrame(all_blind_peaks)

    # Save aggregate results
    results_df.to_csv(os.path.join(output_dir, 'aggregate', 'all_sessions_summary.csv'), index=False)
    canonical_df.to_csv(os.path.join(output_dir, 'aggregate', 'all_windows_canonical.csv'), index=False)
    baseline_df.to_csv(os.path.join(output_dir, 'aggregate', 'all_windows_baseline.csv'), index=False)
    blind_df.to_csv(os.path.join(output_dir, 'aggregate', 'all_blind_peaks.csv'), index=False)

    if verbose:
        print(f"\nProcessed {len(results_df)} sessions")
        print(f"Total canonical (ignition) windows: {len(canonical_df)}")
        print(f"Total baseline windows: {len(baseline_df)}")
        print(f"Total blind peaks: {len(blind_df)}")

    # Generate publication tables
    if len(canonical_df) > 0:
        table1 = generate_table1_frequency_validation(canonical_df)
        table1.to_csv(os.path.join(output_dir, 'aggregate', 'table1_frequency_validation.csv'), index=False)

        table1_summary = generate_table1_summary(canonical_df)
        table1_summary.to_csv(os.path.join(output_dir, 'aggregate', 'table1_summary.csv'), index=False)

        table2 = generate_table2_noble_ratios(canonical_df)
        table2.to_csv(os.path.join(output_dir, 'aggregate', 'table2_noble_ratios.csv'), index=False)

        table3 = generate_table3_boundary_attractor(canonical_df)
        table3.to_csv(os.path.join(output_dir, 'aggregate', 'table3_boundary_attractor.csv'), index=False)

        table4 = generate_table4_independence_convergence(canonical_df)
        table4.to_csv(os.path.join(output_dir, 'aggregate', 'table4_independence_convergence.csv'), index=False)

        if verbose:
            print("\nGenerated publication tables:")
            print("  - table1_frequency_validation.csv")
            print("  - table1_summary.csv (GED vs Predictions per harmonic)")
            print("  - table2_noble_ratios.csv")
            print("  - table3_boundary_attractor.csv")
            print("  - table4_independence_convergence.csv")

    # Generate blind validation table
    if len(blind_df) > 0:
        table_blind = generate_table_blind_validation(blind_df)
        table_blind.to_csv(os.path.join(output_dir, 'aggregate', 'table_blind_phi_alignment.csv'), index=False)
        if verbose:
            print("  - table_blind_phi_alignment.csv")

    # Generate Table 5: Ignition vs Baseline comparison
    if len(canonical_df) > 0 and len(baseline_df) > 0:
        # Compute contrast
        contrast = ged_ignition_baseline_contrast(
            canonical_df.to_dict('records'),
            baseline_df.to_dict('records')
        )

        table5 = generate_table_ignition_baseline(canonical_df, baseline_df, contrast)
        table5.to_csv(os.path.join(output_dir, 'aggregate', 'table5_ignition_baseline.csv'), index=False)

        # Generate comparison visualization
        fig = plot_ignition_baseline_comparison(
            canonical_df, baseline_df, contrast,
            output_path=os.path.join(output_dir, 'figures', 'ignition_vs_baseline_comparison.png'),
            show=False
        )

        if verbose:
            print("  - table5_ignition_baseline.csv")
            print("  - figures/ignition_vs_baseline_comparison.png")

    return results_df


# ============================================================================
# PUBLICATION TABLE GENERATORS
# ============================================================================

def generate_table1_frequency_validation(canonical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Table 1: GED vs φⁿ Predictions.

    Shows per-session comparison of GED-optimized frequencies vs canonical φⁿ predictions,
    including deviation percentages for each harmonic.
    """
    rows = []

    # Harmonics to include
    harmonics = ['sr1', 'sr1.5', 'sr2', 'sr2o', 'sr2.5', 'sr3', 'sr4', 'sr5', 'sr6']

    # Get unique sessions
    sessions = canonical_df['session_id'].unique()

    for sess in sessions:
        sess_df = canonical_df[canonical_df['session_id'] == sess]
        row = {'session_id': sess}

        for h in harmonics:
            ged_col = f'{h}_ged_freq'
            canon_col = f'{h}_canonical_freq'

            ged_mean = None
            canon_val = None

            if ged_col in sess_df.columns:
                ged_vals = sess_df[ged_col].dropna().values
                if len(ged_vals) > 0:
                    ged_mean = np.mean(ged_vals)
                    row[f'{h}_ged_mean'] = ged_mean
                    row[f'{h}_ged_std'] = np.std(ged_vals)

            if canon_col in sess_df.columns:
                canon_vals = sess_df[canon_col].dropna().values
                if len(canon_vals) > 0:
                    canon_val = canon_vals[0]
                    row[f'{h}_canonical'] = canon_val

            # Compute per-harmonic deviation percentage
            if ged_mean is not None and canon_val is not None and canon_val > 0:
                deviation_pct = (ged_mean - canon_val) / canon_val * 100
                row[f'{h}_deviation_pct'] = deviation_pct

        # Compute ratios
        if f'sr3_ged_mean' in row and f'sr1_ged_mean' in row:
            if row.get('sr1_ged_mean', 0) > 0:
                row['sr3_sr1_ratio'] = row['sr3_ged_mean'] / row['sr1_ged_mean']
                row['sr3_sr1_expected'] = PHI ** 2
                row['sr3_sr1_deviation_pct'] = (row['sr3_sr1_ratio'] - PHI**2) / (PHI**2) * 100

        if f'sr5_ged_mean' in row and f'sr3_ged_mean' in row:
            if row.get('sr3_ged_mean', 0) > 0:
                row['sr5_sr3_ratio'] = row['sr5_ged_mean'] / row['sr3_ged_mean']
                row['sr5_sr3_expected'] = PHI ** 1
                row['sr5_sr3_deviation_pct'] = (row['sr5_sr3_ratio'] - PHI) / PHI * 100

        rows.append(row)

    # Add aggregate row
    df = pd.DataFrame(rows)
    if len(df) > 1:
        agg_row = {'session_id': 'MEAN'}
        for col in df.columns:
            if col != 'session_id' and df[col].dtype in [np.float64, np.int64]:
                agg_row[col] = df[col].mean()
        rows.append(agg_row)

    return pd.DataFrame(rows)


def generate_table1_summary(canonical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary Table 1 with GED vs Predictions per harmonic.

    Shows: Harmonic | Predicted (Hz) | GED Mean (Hz) | GED Std | Deviation % | N
    """
    harmonics = ['sr1', 'sr1.5', 'sr2', 'sr2o', 'sr2.5', 'sr3', 'sr4', 'sr5', 'sr6']
    rows = []

    for h in harmonics:
        ged_col = f'{h}_ged_freq'
        canon_col = f'{h}_canonical_freq'

        row = {'harmonic': h.upper()}

        # Get canonical value (should be same across all rows)
        if canon_col in canonical_df.columns:
            canon_vals = canonical_df[canon_col].dropna().values
            if len(canon_vals) > 0:
                row['predicted_hz'] = canon_vals[0]

        # Get GED values across all windows
        if ged_col in canonical_df.columns:
            ged_vals = canonical_df[ged_col].dropna().values
            if len(ged_vals) > 0:
                row['ged_mean_hz'] = np.mean(ged_vals)
                row['ged_std_hz'] = np.std(ged_vals)
                row['n_observations'] = len(ged_vals)

                # Compute deviation
                if 'predicted_hz' in row and row['predicted_hz'] > 0:
                    row['deviation_pct'] = (row['ged_mean_hz'] - row['predicted_hz']) / row['predicted_hz'] * 100

        if len(row) > 1:  # Has data beyond just harmonic name
            rows.append(row)

    return pd.DataFrame(rows)


def generate_table2_noble_ratios(canonical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Table 2: Noble Number Validation.

    Tests whether inter-harmonic ratios equal powers of φ.
    """
    # Define ratio tests
    ratio_tests = [
        ('sr2', 'sr1', PHI ** 1, 'φ¹'),
        ('sr3', 'sr1', PHI ** 2, 'φ²'),
        ('sr5', 'sr1', PHI ** 3, 'φ³'),
        ('sr5', 'sr3', PHI ** 1, 'φ¹'),
        ('sr4', 'sr2', PHI ** 1, 'φ¹'),
        ('sr1.5', 'sr1', PHI ** 0.5, 'φ⁰·⁵'),
        ('sr2.5', 'sr1.5', PHI ** 1, 'φ¹'),
    ]

    rows = []

    for num_h, den_h, expected, expected_label in ratio_tests:
        num_col = f'{num_h}_ged_freq'
        den_col = f'{den_h}_ged_freq'

        if num_col not in canonical_df.columns or den_col not in canonical_df.columns:
            continue

        # Compute ratios for all windows where both exist
        num_vals = canonical_df[num_col].dropna()
        den_vals = canonical_df[den_col].dropna()

        # Align by index
        valid_idx = num_vals.index.intersection(den_vals.index)
        if len(valid_idx) < 3:
            continue

        ratios = num_vals.loc[valid_idx].values / den_vals.loc[valid_idx].values
        ratios = ratios[np.isfinite(ratios) & (ratios > 0)]

        if len(ratios) < 3:
            continue

        observed_mean = np.mean(ratios)
        observed_std = np.std(ratios)
        deviation = (observed_mean - expected) / expected
        deviation_pct = deviation * 100

        # One-sample t-test: is observed ratio different from expected?
        t_stat, p_value = stats.ttest_1samp(ratios, expected)

        # Compute nobility index for the mean ratio
        nob = nobility_index(observed_mean)

        rows.append({
            'ratio': f'{num_h.upper()}/{den_h.upper()}',
            'expected_noble': expected_label,
            'expected_value': expected,
            'observed_mean': observed_mean,
            'observed_std': observed_std,
            'deviation_pct': deviation_pct,
            'nobility_index': nob['nobility'],
            't_statistic': t_stat,
            'p_value': p_value,
            'n_observations': len(ratios),
            'is_valid': abs(deviation) < 0.05  # Within 5% of expected
        })

    return pd.DataFrame(rows)


def generate_table3_boundary_attractor(canonical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Table 3: Boundary vs Attractor Properties.

    Tests the key prediction: attractors have narrower FWHM than boundaries.
    """
    # Use existing attractor_boundary_contrast function
    contrast = attractor_boundary_contrast(canonical_df)

    # Format for publication
    rows = []

    # Boundary row
    rows.append({
        'type': 'Boundary',
        'n_values': '0, 1, 2, 3',
        'harmonics': 'SR1, SR2, SR3, SR5',
        'mean_fwhm': contrast.get('boundary_fwhm_mean', np.nan),
        'std_fwhm': contrast.get('boundary_fwhm_std', np.nan),
        'mean_q_factor': contrast.get('boundary_q_mean', np.nan),
        'n_observations': contrast.get('boundary_fwhm_n', 0)
    })

    # Attractor row
    rows.append({
        'type': 'Attractor',
        'n_values': '0.5, 1.5, 2.5, 3.5',
        'harmonics': 'SR1.5, SR2.5, SR4, SR6',
        'mean_fwhm': contrast.get('attractor_fwhm_mean', np.nan),
        'std_fwhm': contrast.get('attractor_fwhm_std', np.nan),
        'mean_q_factor': contrast.get('attractor_q_mean', np.nan),
        'n_observations': contrast.get('attractor_fwhm_n', 0)
    })

    # Difference row
    att_fwhm = contrast.get('attractor_fwhm_mean', np.nan)
    bnd_fwhm = contrast.get('boundary_fwhm_mean', np.nan)
    if np.isfinite(att_fwhm) and np.isfinite(bnd_fwhm) and bnd_fwhm > 0:
        fwhm_diff_pct = (att_fwhm - bnd_fwhm) / bnd_fwhm * 100
    else:
        fwhm_diff_pct = np.nan

    rows.append({
        'type': 'Δ (Att-Bnd)',
        'n_values': '',
        'harmonics': '',
        'mean_fwhm': att_fwhm - bnd_fwhm if np.isfinite(att_fwhm) and np.isfinite(bnd_fwhm) else np.nan,
        'std_fwhm': np.nan,
        'mean_q_factor': np.nan,
        'n_observations': 0,
        'fwhm_diff_pct': fwhm_diff_pct,
        'cohens_d': contrast.get('fwhm_cohens_d', np.nan),
        'p_value_ttest': contrast.get('fwhm_pvalue', np.nan),
        'p_value_mw': contrast.get('fwhm_pvalue_mw', np.nan),
        'prediction_confirmed': contrast.get('prediction_confirmed', False)
    })

    return pd.DataFrame(rows)


def generate_table4_independence_convergence(canonical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Table 4: Independence-Convergence Test.

    Tests the paradox: absolute frequencies are independent (r ≈ 0)
    but ratios converge tightly around φⁿ values.
    """
    rows = []

    # Test 1: Correlation between absolute frequencies
    freq_pairs = [
        ('sr1_ged_freq', 'sr3_ged_freq', 'SR1 vs SR3'),
        ('sr1_ged_freq', 'sr5_ged_freq', 'SR1 vs SR5'),
        ('sr3_ged_freq', 'sr5_ged_freq', 'SR3 vs SR5'),
    ]

    for col1, col2, label in freq_pairs:
        if col1 not in canonical_df.columns or col2 not in canonical_df.columns:
            continue

        v1 = canonical_df[col1].dropna()
        v2 = canonical_df[col2].dropna()
        idx = v1.index.intersection(v2.index)

        if len(idx) < 5:
            continue

        r, p = stats.pearsonr(v1.loc[idx], v2.loc[idx])

        rows.append({
            'test': f'corr({label})',
            'result': f'r = {r:.3f}',
            'p_value': p,
            'interpretation': 'Independent' if p > 0.05 else 'Correlated',
            'expected': 'r ≈ 0'
        })

    # Test 2: Ratio statistics
    ratio_cols = [
        ('sr3_ged_freq', 'sr1_ged_freq', 'SR3/SR1', PHI**2),
        ('sr5_ged_freq', 'sr3_ged_freq', 'SR5/SR3', PHI**1),
        ('sr5_ged_freq', 'sr1_ged_freq', 'SR5/SR1', PHI**3),
    ]

    for num_col, den_col, label, expected in ratio_cols:
        if num_col not in canonical_df.columns or den_col not in canonical_df.columns:
            continue

        num = canonical_df[num_col].dropna()
        den = canonical_df[den_col].dropna()
        idx = num.index.intersection(den.index)

        if len(idx) < 5:
            continue

        ratios = num.loc[idx].values / den.loc[idx].values
        ratios = ratios[np.isfinite(ratios)]

        if len(ratios) < 3:
            continue

        mean_r = np.mean(ratios)
        std_r = np.std(ratios)
        cv = std_r / mean_r * 100  # Coefficient of variation

        rows.append({
            'test': f'mean({label})',
            'result': f'{mean_r:.4f}',
            'p_value': np.nan,
            'interpretation': f'≈ φⁿ ({(mean_r - expected)/expected * 100:.2f}% error)',
            'expected': f'{expected:.4f}'
        })

        rows.append({
            'test': f'std({label})',
            'result': f'{std_r:.4f}',
            'p_value': np.nan,
            'interpretation': 'Tight' if cv < 2 else 'Loose',
            'expected': '< 2% CV'
        })

    return pd.DataFrame(rows)


def generate_table_ignition_baseline(
    canonical_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    contrast: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Generate Table 5: Ignition vs Baseline GED Comparison.

    Tests whether φⁿ predictions are specifically tied to ignition states
    by comparing GED-derived metrics between ignition and baseline segments.

    Parameters
    ----------
    canonical_df : pd.DataFrame
        Per-window GED results from ignition windows
    baseline_df : pd.DataFrame
        Per-window GED results from baseline windows
    contrast : Dict, optional
        Pre-computed contrast from ged_ignition_baseline_contrast()

    Returns
    -------
    pd.DataFrame with comparison statistics
    """
    if canonical_df.empty or baseline_df.empty:
        return pd.DataFrame()

    # Compute contrast if not provided
    if contrast is None:
        contrast = ged_ignition_baseline_contrast(
            canonical_df.to_dict('records'),
            baseline_df.to_dict('records')
        )

    rows = []

    # Use summary_table if available
    if 'summary_table' in contrast and not contrast['summary_table'].empty:
        st = contrast['summary_table']
        for _, row in st.iterrows():
            sig = ''
            p = row.get('p_value', 1.0)
            if p < 0.001:
                sig = '***'
            elif p < 0.01:
                sig = '**'
            elif p < 0.05:
                sig = '*'

            rows.append({
                'harmonic': row.get('harmonic', ''),
                'metric': row.get('metric', ''),
                'ignition_mean': row.get('ignition_mean', np.nan),
                'ignition_std': row.get('ignition_std', np.nan),
                'ignition_n': row.get('ignition_n', 0),
                'baseline_mean': row.get('baseline_mean', np.nan),
                'baseline_std': row.get('baseline_std', np.nan),
                'baseline_n': row.get('baseline_n', 0),
                'delta': row.get('delta', np.nan),
                'cohens_d': row.get('cohens_d', np.nan),
                'p_value': p,
                'significance': sig
            })
    else:
        # Build from aggregate if summary_table not available
        if 'aggregate' in contrast:
            for metric, stats in contrast['aggregate'].items():
                if not isinstance(stats, dict):
                    continue

                p = stats.get('p_value', 1.0)
                sig = ''
                if p < 0.001:
                    sig = '***'
                elif p < 0.01:
                    sig = '**'
                elif p < 0.05:
                    sig = '*'

                rows.append({
                    'harmonic': 'ALL',
                    'metric': metric,
                    'ignition_mean': stats.get('ignition_mean', np.nan),
                    'ignition_std': stats.get('ignition_std', np.nan),
                    'ignition_n': stats.get('ignition_n', 0),
                    'baseline_mean': stats.get('baseline_mean', np.nan),
                    'baseline_std': stats.get('baseline_std', np.nan),
                    'baseline_n': stats.get('baseline_n', 0),
                    'delta': stats.get('delta', np.nan),
                    'cohens_d': stats.get('cohens_d', np.nan),
                    'p_value': p,
                    'significance': sig
                })

    return pd.DataFrame(rows)


def generate_table_blind_validation(blind_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate validation table for blind-discovered peaks vs φⁿ predictions.
    """
    if len(blind_df) == 0:
        return pd.DataFrame()

    # Analyze φⁿ alignment
    n_aligned = (blind_df['is_phi_aligned'] == True).sum()
    n_total = len(blind_df)
    alignment_fraction = n_aligned / max(1, n_total)

    # Distribution of nearest φⁿ values
    phi_n_counts = blind_df['nearest_phi_n'].value_counts().to_dict()

    # Mean distance to φⁿ
    mean_distance = blind_df['phi_distance_rel'].mean()
    std_distance = blind_df['phi_distance_rel'].std()

    rows = [
        {
            'metric': 'Total blind peaks',
            'value': n_total,
            'interpretation': ''
        },
        {
            'metric': 'φⁿ-aligned peaks (< 5%)',
            'value': n_aligned,
            'interpretation': f'{alignment_fraction*100:.1f}% of peaks'
        },
        {
            'metric': 'Mean distance to φⁿ',
            'value': f'{mean_distance*100:.2f}%',
            'interpretation': 'Strong' if mean_distance < 0.03 else 'Moderate' if mean_distance < 0.05 else 'Weak'
        },
        {
            'metric': 'Std distance to φⁿ',
            'value': f'{std_distance*100:.2f}%',
            'interpretation': ''
        }
    ]

    # Add φⁿ coverage
    for n, count in sorted(phi_n_counts.items()):
        f_pred = F0 * (PHI ** n)
        rows.append({
            'metric': f'Peaks near φⁿ={n} ({f_pred:.1f} Hz)',
            'value': count,
            'interpretation': f'{count/n_total*100:.1f}%'
        })

    return pd.DataFrame(rows)


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_ged_validation_summary(
    canonical_df: pd.DataFrame,
    blind_df: pd.DataFrame = None,
    output_path: str = None,
    show: bool = True,
    title: str = None,
    dataset_name: str = None
) -> plt.Figure:
    """
    Generate comprehensive 8-panel summary visualization of GED validation results.

    Panels:
    A. GED frequencies vs φⁿ predictions (with deviation %)
    B. Inter-harmonic ratio validation (with deviation %)
    C. Blind peak frequency distribution (histogram)
    D. Blind peaks by nearest φⁿ harmonic (bar chart)
    E. Independence test: SR1 vs SR3 scatter
    F. Ratio convergence: SR5/SR3 histogram
    G. Boundary vs attractor FWHM comparison
    H. Summary statistics panel
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Title
    n_windows = len(canonical_df)
    n_blind = len(blind_df) if blind_df is not None else 0
    n_sessions = canonical_df['session_id'].nunique() if 'session_id' in canonical_df.columns else n_windows

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    else:
        dataset_str = f" ({dataset_name})" if dataset_name else ""
        fig.suptitle(f"GED φⁿ Validation{dataset_str}\nN = {n_windows} canonical windows | {n_blind} blind peaks | {n_sessions} sessions",
                     fontsize=14, fontweight='bold', y=1.02)

    harmonics = ['sr1', 'sr1.5', 'sr2', 'sr2o', 'sr2.5', 'sr3', 'sr4', 'sr5', 'sr6']

    # =========================================================================
    # Panel A: GED vs φⁿ Frequency Predictions
    # =========================================================================
    ax1 = axes[0, 0]
    x_pos = np.arange(len(harmonics))
    canonical_freqs = []
    ged_means = []
    ged_stds = []
    deviations = []

    for h in harmonics:
        f_canon = SR_HARMONIC_TABLE.get(h, {}).get('freq', np.nan)
        canonical_freqs.append(f_canon)

        ged_col = f'{h}_ged_freq'
        if ged_col in canonical_df.columns:
            vals = canonical_df[ged_col].dropna().values
            if len(vals) > 0:
                mean_val = np.mean(vals)
                ged_means.append(mean_val)
                ged_stds.append(np.std(vals))
                dev_pct = (mean_val - f_canon) / f_canon * 100 if f_canon > 0 else 0
                deviations.append(dev_pct)
            else:
                ged_means.append(np.nan)
                ged_stds.append(np.nan)
                deviations.append(np.nan)
        else:
            ged_means.append(np.nan)
            ged_stds.append(np.nan)
            deviations.append(np.nan)

    # Y positions for harmonics (reversed so SR1 is at top)
    y_pos = np.arange(len(harmonics))[::-1]

    # Horizontal bars
    bars1 = ax1.barh(y_pos + 0.18, canonical_freqs, 0.3, label='Predicted (φⁿ)', color='steelblue', alpha=0.8)
    bars2 = ax1.barh(y_pos - 0.18, ged_means, 0.3, xerr=ged_stds, label='GED Observed', color='coral', alpha=0.8, capsize=2)

    # Right-side Hz labels
    x_max = max(max(canonical_freqs), max([g + s for g, s in zip(ged_means, ged_stds) if np.isfinite(g)]))
    for i, (canon, ged, std, dev) in enumerate(zip(canonical_freqs, ged_means, ged_stds, deviations)):
        # Predicted label (right of bar)
        ax1.text(canon + 0.8, y_pos[i] + 0.18, f'{canon:.2f}', va='center', ha='left', fontsize=7, color='steelblue')
        # GED label with std and deviation
        if np.isfinite(ged):
            dev_color = 'green' if abs(dev) < 1 else 'orange' if abs(dev) < 5 else 'red'
            ax1.text(ged + std + 0.8, y_pos[i] - 0.18,
                     f'{ged:.2f} ±{std:.1f}  ({dev:+.1f}%)',
                     va='center', ha='left', fontsize=7, color=dev_color, fontweight='bold')

    ax1.set_yticks(y_pos)
    # Labels with φⁿ exponent using Unicode superscripts
    superscript_map = {'0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
                       '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', '.': '·'}
    phi_labels = []
    for h in harmonics:
        n = SR_HARMONIC_TABLE.get(h, {}).get('n', 0)
        if n == int(n):
            exp_str = str(int(n))
        else:
            exp_str = f'{n:.1f}'
        exp_super = ''.join(superscript_map.get(c, c) for c in exp_str)
        phi_labels.append(f'{h.upper()} (φ{exp_super})')
    ax1.set_yticklabels(phi_labels)
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_xlim(0, x_max + 18)  # Room for labels
    ax1.set_title('A. GED vs φⁿ Frequency Predictions')
    ax1.legend(loc='upper right', fontsize=7)
    ax1.grid(True, alpha=0.3, axis='x')

    # =========================================================================
    # Panel B: Inter-Harmonic Ratio Validation
    # =========================================================================
    ax2 = axes[0, 1]
    ratio_info = []

    # Define ratios to show
    ratio_pairs = [
        ('sr2', 'sr1', 'φ¹'),
        ('sr3', 'sr1', 'φ²'),
        ('sr5', 'sr1', 'φ³'),
        ('sr5', 'sr3', 'φ¹'),
        ('sr2.5', 'sr1.5', 'φ¹'),
        ('sr4', 'sr2', 'φ¹'),
    ]

    for num_h, den_h, phi_label in ratio_pairs:
        num_col = f'{num_h}_ged_freq'
        den_col = f'{den_h}_ged_freq'
        if num_col in canonical_df.columns and den_col in canonical_df.columns:
            num = canonical_df[num_col].dropna()
            den = canonical_df[den_col].dropna()
            idx = num.index.intersection(den.index)
            if len(idx) > 0:
                ratios = num.loc[idx].values / den.loc[idx].values
                ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
                if len(ratios) > 0:
                    expected = EXPECTED_RATIOS.get((num_h, den_h), PHI)
                    ratio_info.append({
                        'label': f'{num_h.upper()}/{den_h.upper()}',
                        'mean': np.mean(ratios),
                        'std': np.std(ratios),
                        'expected': expected,
                        'phi_label': phi_label
                    })

    if ratio_info:
        x_pos = np.arange(len(ratio_info))
        expected_vals = [r['expected'] for r in ratio_info]
        observed_vals = [r['mean'] for r in ratio_info]
        observed_stds = [r['std'] for r in ratio_info]

        ax2.bar(x_pos - 0.2, expected_vals, 0.35, label='Expected', color='steelblue', alpha=0.8)
        bars = ax2.bar(x_pos + 0.2, observed_vals, 0.35, yerr=observed_stds, label='Observed', color='coral', alpha=0.8, capsize=3)

        # Add deviation labels
        for i, (bar, r) in enumerate(zip(bars, ratio_info)):
            dev_pct = (r['mean'] - r['expected']) / r['expected'] * 100
            color = 'green' if abs(dev_pct) < 1 else 'orange' if abs(dev_pct) < 5 else 'red'
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + r['std'] + 0.05,
                     f'{dev_pct:+.1f}%', ha='center', va='bottom', fontsize=8, color=color, fontweight='bold')

        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([r['label'] for r in ratio_info], rotation=45, ha='right')
        ax2.set_ylabel('Ratio Value')
        ax2.set_title('B. Inter-Harmonic Ratio Validation')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)

    # =========================================================================
    # Panel C: Blind Peak Frequency Distribution
    # =========================================================================
    ax3 = axes[0, 2]
    if blind_df is not None and len(blind_df) > 0 and 'frequency' in blind_df.columns:
        freqs = blind_df['frequency'].values
        aligned = blind_df['is_phi_aligned'].values if 'is_phi_aligned' in blind_df.columns else np.zeros(len(freqs), dtype=bool)

        bins = np.arange(3, 46, 1)
        ax3.hist(freqs, bins=bins, color='steelblue', alpha=0.7, edgecolor='white', label='All peaks')

        # Overlay aligned peaks
        aligned_freqs = freqs[aligned]
        if len(aligned_freqs) > 0:
            ax3.hist(aligned_freqs, bins=bins, color='coral', alpha=0.7, edgecolor='white', label='φⁿ-aligned')

        # Mark φⁿ frequencies
        for n in np.arange(-0.5, 4, 0.5):
            f_pred = F0 * (PHI ** n)
            if 3 <= f_pred <= 45:
                ax3.axvline(f_pred, color='red', linestyle='--', alpha=0.5, linewidth=1)

        alignment_pct = aligned.mean() * 100 if len(aligned) > 0 else 0
        ax3.text(0.95, 0.95, f'φⁿ-aligned (<5%): {alignment_pct:.1f}%',
                 transform=ax3.transAxes, ha='right', va='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Count')
        ax3.set_title(f'C. Blind Peak Frequency Distribution (n={len(freqs)})')
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No blind peak data', ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('C. Blind Peak Frequency Distribution')

    # =========================================================================
    # Panel D: Blind Peaks by Nearest φⁿ Harmonic
    # =========================================================================
    ax4 = axes[0, 3]
    if blind_df is not None and len(blind_df) > 0 and 'nearest_phi_n' in blind_df.columns:
        phi_n_counts = blind_df['nearest_phi_n'].value_counts().sort_index()

        # Create labels
        x_labels = [f'φⁿ={n:.1f}\n({F0 * PHI**n:.1f} Hz)' for n in phi_n_counts.index]

        # Color by type (boundary vs attractor)
        colors = ['steelblue' if n == int(n) else 'coral' for n in phi_n_counts.index]

        bars = ax4.bar(range(len(phi_n_counts)), phi_n_counts.values, color=colors, alpha=0.8, edgecolor='white')

        # Add percentage labels
        total = phi_n_counts.sum()
        for i, (bar, count) in enumerate(zip(bars, phi_n_counts.values)):
            pct = count / total * 100
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)

        ax4.set_xticks(range(len(phi_n_counts)))
        ax4.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
        ax4.set_ylabel('Peak Count')
        ax4.set_title('D. Blind Peaks by Nearest φⁿ Harmonic')

        # Legend for boundary/attractor
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='steelblue', label='Boundary (integer n)'),
                          Patch(facecolor='coral', label='Attractor (half-integer n)')]
        ax4.legend(handles=legend_elements, loc='upper left', fontsize=8)
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        ax4.text(0.5, 0.5, 'No φⁿ assignment data', ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('D. Blind Peaks by Nearest φⁿ Harmonic')

    # =========================================================================
    # Panel E: Independence Test (SR1 vs SR3)
    # =========================================================================
    ax5 = axes[1, 0]
    sr1_col = 'sr1_ged_freq'
    sr3_col = 'sr3_ged_freq'

    if sr1_col in canonical_df.columns and sr3_col in canonical_df.columns:
        sr1 = canonical_df[sr1_col].dropna()
        sr3 = canonical_df[sr3_col].dropna()
        idx = sr1.index.intersection(sr3.index)

        if len(idx) > 10:
            sr1_vals = sr1.loc[idx].values
            sr3_vals = sr3.loc[idx].values

            ax5.scatter(sr1_vals, sr3_vals, alpha=0.5, s=30, c='steelblue')

            # Add correlation
            r, p = stats.pearsonr(sr1_vals, sr3_vals)
            ax5.text(0.05, 0.95, f'r = {r:.3f} (p = {p:.3f})',
                     transform=ax5.transAxes, ha='left', va='top', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Add trend line
            z = np.polyfit(sr1_vals, sr3_vals, 1)
            p_line = np.poly1d(z)
            x_line = np.linspace(sr1_vals.min(), sr1_vals.max(), 100)
            ax5.plot(x_line, p_line(x_line), 'r--', alpha=0.5, linewidth=2)

            ax5.set_xlabel('SR1 Frequency (Hz)')
            ax5.set_ylabel('SR3 Frequency (Hz)')
            ax5.set_title('E. Independence: SR1 vs SR3')
            ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax5.transAxes, fontsize=12)
        ax5.set_title('E. Independence: SR1 vs SR3')

    # =========================================================================
    # Panel F: Ratio Convergence (SR5/SR3 → φ)
    # =========================================================================
    ax6 = axes[1, 1]
    sr5_col = 'sr5_ged_freq'
    sr3_col = 'sr3_ged_freq'

    if sr5_col in canonical_df.columns and sr3_col in canonical_df.columns:
        sr5 = canonical_df[sr5_col].dropna()
        sr3 = canonical_df[sr3_col].dropna()
        idx = sr5.index.intersection(sr3.index)

        if len(idx) > 5:
            ratios = sr5.loc[idx].values / sr3.loc[idx].values
            ratios = ratios[np.isfinite(ratios) & (ratios > 0) & (ratios < 3)]

            if len(ratios) > 0:
                ax6.hist(ratios, bins=30, color='steelblue', alpha=0.7, edgecolor='white')
                ax6.axvline(PHI, color='red', linestyle='--', linewidth=2, label=f'φ = {PHI:.3f}')
                ax6.axvline(np.mean(ratios), color='green', linestyle='-', linewidth=2, label=f'Mean = {np.mean(ratios):.3f}')

                error_pct = (np.mean(ratios) - PHI) / PHI * 100
                ax6.text(0.95, 0.95, f'Error: {error_pct:+.2f}%',
                         transform=ax6.transAxes, ha='right', va='top', fontsize=10,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                ax6.set_xlabel('SR5/SR3 Ratio')
                ax6.set_ylabel('Count')
                ax6.set_title('F. Ratio Convergence: SR5/SR3 → φ')
                ax6.legend(loc='upper left', fontsize=8)
                ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        ax6.set_title('F. Ratio Convergence: SR5/SR3 → φ')

    # =========================================================================
    # Panel G: Boundary vs Attractor FWHM
    # =========================================================================
    ax7 = axes[1, 2]
    attractor_fwhm = []
    boundary_fwhm = []

    for h, info in SR_HARMONIC_TABLE.items():
        fwhm_col = f'{h}_fwhm'
        if fwhm_col in canonical_df.columns:
            vals = canonical_df[fwhm_col].dropna().values
            vals = vals[np.isfinite(vals) & (vals > 0) & (vals < 10)]
            if info['type'] == 'attractor':
                attractor_fwhm.extend(vals)
            else:
                boundary_fwhm.extend(vals)

    if attractor_fwhm and boundary_fwhm:
        boundary_mean = np.mean(boundary_fwhm)
        attractor_mean = np.mean(attractor_fwhm)
        boundary_std = np.std(boundary_fwhm)
        attractor_std = np.std(attractor_fwhm)

        x_pos = [0, 1]
        bars = ax7.bar(x_pos, [boundary_mean, attractor_mean],
                       yerr=[boundary_std, attractor_std],
                       color=['steelblue', 'coral'], alpha=0.8, capsize=5, edgecolor='white')

        # Significance test
        _, p = stats.mannwhitneyu(attractor_fwhm, boundary_fwhm, alternative='two-sided')
        diff_pct = (attractor_mean - boundary_mean) / boundary_mean * 100
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

        ax7.text(0.5, max(boundary_mean, attractor_mean) + max(boundary_std, attractor_std) + 0.1,
                 f'Δ = {diff_pct:+.1f}%, p = {p:.2e} {sig}', ha='center', fontsize=10, fontweight='bold')

        ax7.set_xticks(x_pos)
        ax7.set_xticklabels(['Boundary\n(integer n)', 'Attractor\n(half-integer n)'])
        ax7.set_ylabel('Mean FWHM (Hz)')
        ax7.set_title('G. Boundary vs Attractor FWHM')
        ax7.grid(True, alpha=0.3, axis='y')
    else:
        ax7.text(0.5, 0.5, 'Insufficient FWHM data', ha='center', va='center', transform=ax7.transAxes, fontsize=12)
        ax7.set_title('G. Boundary vs Attractor FWHM')

    # =========================================================================
    # Panel H: Summary Statistics
    # =========================================================================
    ax8 = axes[1, 3]
    ax8.axis('off')

    # Compute summary statistics
    summary_text = "Summary Statistics\n" + "="*30 + "\n\n"

    # Mean deviation across harmonics
    valid_devs = [d for d in deviations if np.isfinite(d)]
    if valid_devs:
        mean_abs_dev = np.mean(np.abs(valid_devs))
        summary_text += f"Mean |Deviation|: {mean_abs_dev:.2f}%\n"

    # Phi score (fraction of valid ratios)
    if ratio_info:
        valid_ratios = sum(1 for r in ratio_info if abs((r['mean'] - r['expected']) / r['expected']) < 0.05)
        summary_text += f"Ratios within 5%: {valid_ratios}/{len(ratio_info)}\n"

    # Blind alignment
    if blind_df is not None and 'is_phi_aligned' in blind_df.columns:
        align_pct = blind_df['is_phi_aligned'].mean() * 100
        summary_text += f"Blind φⁿ alignment: {align_pct:.1f}%\n"

    # FWHM comparison
    if attractor_fwhm and boundary_fwhm:
        summary_text += f"\nFWHM Comparison:\n"
        summary_text += f"  Boundary: {np.mean(boundary_fwhm):.3f} ± {np.std(boundary_fwhm):.3f} Hz\n"
        summary_text += f"  Attractor: {np.mean(attractor_fwhm):.3f} ± {np.std(attractor_fwhm):.3f} Hz\n"

    # Independence
    if sr1_col in canonical_df.columns and sr3_col in canonical_df.columns:
        sr1 = canonical_df[sr1_col].dropna()
        sr3 = canonical_df[sr3_col].dropna()
        idx = sr1.index.intersection(sr3.index)
        if len(idx) > 10:
            r, _ = stats.pearsonr(sr1.loc[idx].values, sr3.loc[idx].values)
            summary_text += f"\nIndependence (SR1-SR3): r={r:.3f}\n"

    # Dataset info
    summary_text += f"\n" + "="*30 + "\n"
    summary_text += f"Windows: {n_windows}\n"
    summary_text += f"Sessions: {n_sessions}\n"
    summary_text += f"Blind peaks: {n_blind}\n"

    ax8.text(0.1, 0.9, summary_text, transform=ax8.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax8.set_title('H. Summary Statistics')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        # Also save as PDF
        pdf_path = output_path.rsplit('.', 1)[0] + '.pdf'
        fig.savefig(pdf_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_ignition_baseline_comparison(
    ignition_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    contrast: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
    show: bool = True,
    title: str = None
) -> plt.Figure:
    """
    Generate 4-panel comparison visualization: Ignition vs Baseline GED metrics.

    Panels:
    A. Bar plot: Ignition vs Baseline FWHM per harmonic
    B. Bar plot: Ignition vs Baseline Q-factor per harmonic
    C. Effect size heatmap (Cohen's d per metric × harmonic)
    D. Summary statistics panel

    Parameters
    ----------
    ignition_df : pd.DataFrame
        Per-window GED results from ignition windows
    baseline_df : pd.DataFrame
        Per-window GED results from baseline windows
    contrast : Dict, optional
        Pre-computed contrast from ged_ignition_baseline_contrast()
    output_path : str, optional
        Path to save figure
    show : bool
        Whether to display the figure
    title : str, optional
        Custom title

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Compute contrast if not provided
    if contrast is None:
        contrast = ged_ignition_baseline_contrast(
            ignition_df.to_dict('records'),
            baseline_df.to_dict('records')
        )

    harmonics = list(SR_HARMONIC_TABLE.keys())
    metrics = ['fwhm', 'q_factor', 'eigenvalue', 'lambda_ratio']

    n_ign = len(ignition_df)
    n_base = len(baseline_df)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    else:
        fig.suptitle(f"GED Ignition vs Baseline Comparison\n"
                     f"Ignition: {n_ign} windows | Baseline: {n_base} windows",
                     fontsize=14, fontweight='bold', y=1.02)

    # =========================================================================
    # Panel A: FWHM Comparison
    # =========================================================================
    ax1 = axes[0, 0]
    x_pos = np.arange(len(harmonics))
    ign_fwhm = []
    ign_fwhm_err = []
    base_fwhm = []
    base_fwhm_err = []

    for h in harmonics:
        col = f'{h}_fwhm'
        if col in ignition_df.columns:
            vals = ignition_df[col].dropna()
            ign_fwhm.append(vals.mean() if len(vals) > 0 else np.nan)
            ign_fwhm_err.append(vals.std() if len(vals) > 1 else 0)
        else:
            ign_fwhm.append(np.nan)
            ign_fwhm_err.append(0)

        if col in baseline_df.columns:
            vals = baseline_df[col].dropna()
            base_fwhm.append(vals.mean() if len(vals) > 0 else np.nan)
            base_fwhm_err.append(vals.std() if len(vals) > 1 else 0)
        else:
            base_fwhm.append(np.nan)
            base_fwhm_err.append(0)

    ax1.bar(x_pos - 0.2, ign_fwhm, 0.35, yerr=ign_fwhm_err, label='Ignition',
            color='coral', alpha=0.8, capsize=3)
    ax1.bar(x_pos + 0.2, base_fwhm, 0.35, yerr=base_fwhm_err, label='Baseline',
            color='steelblue', alpha=0.8, capsize=3)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([h.upper() for h in harmonics], rotation=45)
    ax1.set_ylabel('FWHM (Hz)')
    ax1.set_title('A. FWHM: Ignition vs Baseline\n(Lower = Sharper peaks)')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # =========================================================================
    # Panel B: Q-factor Comparison
    # =========================================================================
    ax2 = axes[0, 1]
    ign_q = []
    ign_q_err = []
    base_q = []
    base_q_err = []

    for h in harmonics:
        col = f'{h}_q_factor'
        if col in ignition_df.columns:
            vals = ignition_df[col].dropna()
            ign_q.append(vals.mean() if len(vals) > 0 else np.nan)
            ign_q_err.append(vals.std() if len(vals) > 1 else 0)
        else:
            ign_q.append(np.nan)
            ign_q_err.append(0)

        if col in baseline_df.columns:
            vals = baseline_df[col].dropna()
            base_q.append(vals.mean() if len(vals) > 0 else np.nan)
            base_q_err.append(vals.std() if len(vals) > 1 else 0)
        else:
            base_q.append(np.nan)
            base_q_err.append(0)

    ax2.bar(x_pos - 0.2, ign_q, 0.35, yerr=ign_q_err, label='Ignition',
            color='coral', alpha=0.8, capsize=3)
    ax2.bar(x_pos + 0.2, base_q, 0.35, yerr=base_q_err, label='Baseline',
            color='steelblue', alpha=0.8, capsize=3)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([h.upper() for h in harmonics], rotation=45)
    ax2.set_ylabel('Q-factor')
    ax2.set_title('B. Q-factor: Ignition vs Baseline\n(Higher = More selective)')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # =========================================================================
    # Panel C: Effect Size Heatmap
    # =========================================================================
    ax3 = axes[1, 0]

    # Build effect size matrix
    effect_matrix = np.full((len(metrics), len(harmonics)), np.nan)

    per_harmonic = contrast.get('per_harmonic', {})
    for j, h in enumerate(harmonics):
        if h in per_harmonic:
            for i, m in enumerate(metrics):
                if m in per_harmonic[h]:
                    effect_matrix[i, j] = per_harmonic[h][m].get('cohens_d', np.nan)

    # Plot heatmap
    im = ax3.imshow(effect_matrix, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
    plt.colorbar(im, ax=ax3, label="Cohen's d")

    ax3.set_xticks(np.arange(len(harmonics)))
    ax3.set_xticklabels([h.upper() for h in harmonics], rotation=45)
    ax3.set_yticks(np.arange(len(metrics)))
    ax3.set_yticklabels([m.upper() for m in metrics])
    ax3.set_title("C. Effect Size: Cohen's d (Ignition - Baseline)\n"
                  "(Red = Ignition higher, Blue = Baseline higher)")

    # Add text annotations
    for i in range(len(metrics)):
        for j in range(len(harmonics)):
            val = effect_matrix[i, j]
            if np.isfinite(val):
                color = 'white' if abs(val) > 1 else 'black'
                ax3.text(j, i, f'{val:.2f}', ha='center', va='center',
                         fontsize=8, color=color)

    # =========================================================================
    # Panel D: Summary Statistics
    # =========================================================================
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = "IGNITION vs BASELINE SUMMARY\n"
    summary_text += "=" * 35 + "\n\n"

    # Aggregate statistics
    agg = contrast.get('aggregate', {})
    for metric in metrics:
        if metric in agg:
            stats = agg[metric]
            d = stats.get('cohens_d', np.nan)
            p = stats.get('p_value', np.nan)
            delta = stats.get('delta', np.nan)
            pred = stats.get('prediction_confirmed', False)

            sig = ''
            if p < 0.001:
                sig = '***'
            elif p < 0.01:
                sig = '**'
            elif p < 0.05:
                sig = '*'

            pred_str = 'YES' if pred else 'NO'
            summary_text += f"{metric.upper()}:\n"
            summary_text += f"  Delta: {delta:+.3f}\n"
            summary_text += f"  Cohen's d: {d:+.2f}\n"
            summary_text += f"  p-value: {p:.4f} {sig}\n"
            summary_text += f"  Prediction: {pred_str}\n\n"

    # Add sample sizes
    summary_text += "-" * 35 + "\n"
    summary_text += f"Ignition windows: {n_ign}\n"
    summary_text += f"Baseline windows: {n_base}\n"

    # Expected predictions
    summary_text += "\n" + "-" * 35 + "\n"
    summary_text += "EXPECTED:\n"
    summary_text += "  FWHM: Ignition < Baseline (sharper)\n"
    summary_text += "  Q-factor: Ignition > Baseline\n"
    summary_text += "  Eigenvalue: Ignition > Baseline\n"
    summary_text += "  Lambda ratio: Ignition > Baseline\n"

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax4.set_title('D. Summary Statistics')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        # Also save as PDF
        pdf_path = output_path.rsplit('.', 1)[0] + '.pdf'
        fig.savefig(pdf_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_validate_session(
    filepath: str,
    ignition_windows: List[Tuple[int, int]],
    dataset: str = 'unknown',
    device: str = None,
    output_dir: str = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Quick one-liner validation for a single session.

    Example:
        result = quick_validate_session(
            'data/my_session.csv',
            [(100, 120), (250, 270)],
            dataset='my_study'
        )
    """
    # Parse metadata and determine device
    metadata = parse_session_metadata(filepath, dataset)
    if device is None:
        device = metadata.get('device', 'epoc')

    # Select electrodes
    if device == 'insight':
        electrodes = INSIGHT_ELECTRODES
    elif device == 'muse':
        electrodes = MUSE_ELECTRODES
    else:
        electrodes = ELECTRODES

    eeg_channels = [f'EEG.{e}' for e in electrodes]

    # Load data
    records = load_eeg_csv(filepath, electrodes=electrodes, device=device)

    # Infer fs
    if 'Timestamp' in records.columns:
        dt = np.diff(records['Timestamp'].values[:1000])
        dt = dt[np.isfinite(dt) & (dt > 0)]
        fs = 1.0 / np.median(dt) if len(dt) > 0 else 128.0
    else:
        fs = 128.0

    # Run validation
    session_id = metadata.get('subject', Path(filepath).stem)
    result = run_ged_validation_session(
        records, ignition_windows, eeg_channels,
        session_id=session_id,
        fs=fs,
        metadata=metadata
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"GED Validation: {session_id}")
        print(f"{'='*60}")
        print(f"Ignition windows: {len(ignition_windows)}")

        if result.get('blind_results') and result['blind_results'].get('success'):
            print(f"\nBlind Sweep Results:")
            print(f"  Peaks found: {result['blind_results']['n_peaks_found']}")
            if 'blind_phi_analysis' in result:
                ba = result['blind_phi_analysis']
                print(f"  φⁿ-aligned: {ba['alignment_fraction']*100:.1f}%")
                print(f"  Mean distance to φⁿ: {ba['mean_distance']*100:.2f}%")

        if result.get('summary', {}).get('mean_phi_score') is not None:
            print(f"\nCanonical Sweep Results:")
            print(f"  Mean φ-score: {result['summary']['mean_phi_score']:.3f}")

    # Save if output_dir specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(result.get('canonical_results', [])).to_csv(
            os.path.join(output_dir, f'{session_id}_ged_results.csv'), index=False
        )

    return result


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("GED φⁿ Validation Pipeline")
    print("=" * 60)
    print("""
Usage:
    from lib.ged_validation_pipeline import (
        quick_validate_session,
        batch_ged_validation,
        run_ged_validation_session
    )

    # Quick single-session validation:
    result = quick_validate_session(
        'data/session.csv',
        ignition_windows=[(100, 120), (200, 220)],
        dataset='my_study'
    )

    # Batch validation:
    configs = [
        {'filepath': 'data/s1.csv', 'ignition_windows': [...], 'dataset': 'study1'},
        {'filepath': 'data/s2.csv', 'ignition_windows': [...], 'dataset': 'study1'},
    ]
    results_df = batch_ged_validation(configs, output_dir='exports_ged_validation/')
    """)

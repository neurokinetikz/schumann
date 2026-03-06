#!/usr/bin/env python3
"""
Run TRUE Continuous GED Peak Detection on Multiple Datasets

Processes PhySF, Emotions, and MPENG datasets with true continuous
GED peak detection (no band boundary gaps), then creates individual
and aggregate charts.

Usage:
    python scripts/run_all_datasets_true_continuous.py [dataset]

    dataset: 'physf', 'emotions', 'mpeng', 'all', or 'charts'
"""

import sys
sys.path.insert(0, './lib')

import numpy as np
import pandas as pd
from glob import glob
from scipy.signal import find_peaks
import os
import time
import argparse

from peak_distribution_analysis import ged_continuous_sweep

# ============================================================================
# CONSTANTS
# ============================================================================

BANDS = {
    'theta':     (4.70, 7.60),
    'alpha':     (7.60, 12.30),
    'beta_low':  (12.30, 19.90),
    'beta_high': (19.90, 32.19),
    'gamma':     (32.19, 45.0),
}

EPOC_ELECTRODES = [
    'EEG.AF3', 'EEG.AF4', 'EEG.F7', 'EEG.F8', 'EEG.F3', 'EEG.F4',
    'EEG.FC5', 'EEG.FC6', 'EEG.P7', 'EEG.P8', 'EEG.T7', 'EEG.T8',
    'EEG.O1', 'EEG.O2'
]

NEXUS32_ELECTRODES = [
    'EEG.Fp1', 'EEG.Fp2', 'EEG.F7', 'EEG.F3', 'EEG.Fz', 'EEG.F4', 'EEG.F8',
    'EEG.T7', 'EEG.C3', 'EEG.Cz', 'EEG.C4', 'EEG.T8',
    'EEG.P7', 'EEG.P3', 'EEG.Pz', 'EEG.P4', 'EEG.P8',
    'EEG.O1', 'EEG.O2'
]

DATASET_CONFIGS = {
    'physf': {
        'pattern': 'data/PhySF/**/*.csv',
        'output_dir': 'exports_peak_distribution/physf_ged/truly_continuous',
        'fs': 128,
        'electrodes': EPOC_ELECTRODES,
    },
    'emotions': {
        'pattern': 'data/emotions/*.csv',
        'output_dir': 'exports_peak_distribution/emotions_ged/truly_continuous',
        'fs': 128,
        'electrodes': EPOC_ELECTRODES,
    },
    'mpeng': {
        'pattern': 'data/mpeng/*.csv',
        'output_dir': 'exports_peak_distribution/mpeng_ged/truly_continuous',
        'fs': 128,
        'electrodes': EPOC_ELECTRODES,
    },
    'arithmetic_A': {
        'pattern': 'data/arithmetic/csv/**/*A.csv',
        'output_dir': 'exports_peak_distribution/arithmetic_ged/arithmetic',
        'fs': 256,
        'electrodes': NEXUS32_ELECTRODES,
        'header': 0,
    },
    'arithmetic_M': {
        'pattern': 'data/arithmetic/csv/Experiment1/*M.csv',
        'output_dir': 'exports_peak_distribution/arithmetic_ged/meditation',
        'fs': 256,
        'electrodes': NEXUS32_ELECTRODES,
        'header': 0,
    },
    'arithmetic_B': {
        'pattern': 'data/arithmetic/csv/Experiment2/*B.csv',
        'output_dir': 'exports_peak_distribution/arithmetic_ged/baseline',
        'fs': 256,
        'electrodes': NEXUS32_ELECTRODES,
        'header': 0,
    },
    'arithmetic_R': {
        'pattern': 'data/arithmetic/csv/**/*R.csv',
        'output_dir': 'exports_peak_distribution/arithmetic_ged/rest',
        'fs': 256,
        'electrodes': NEXUS32_ELECTRODES,
        'header': 0,
    },
}

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def assign_band(freq: float) -> str:
    """Assign a frequency to a band (post-hoc, after peak detection)."""
    for band, (f_lo, f_hi) in BANDS.items():
        if f_lo <= freq < f_hi:
            return band
    if freq >= 45.0:
        return 'gamma'
    return 'sub_theta'


def find_ged_peaks_truly_continuous(
    records: pd.DataFrame,
    eeg_channels: list,
    fs: float = 128,
    window_sec: float = 10.0,
    step_sec: float = 5.0,
    freq_range: tuple = (4.5, 45.0),
    sweep_step_hz: float = 0.1,
    prominence_frac: float = 0.05,
    min_distance_hz: float = 0.3
) -> pd.DataFrame:
    """
    TRUE continuous GED peak detection - no band boundary artifacts.
    """
    # Build channel matrix
    available = [ch for ch in eeg_channels if ch in records.columns]
    if len(available) < 3:
        # Try without EEG. prefix
        bare_channels = [ch.replace('EEG.', '') for ch in eeg_channels]
        available = [ch for ch in bare_channels if ch in records.columns]
        if len(available) < 3:
            return pd.DataFrame()

    X = np.vstack([pd.to_numeric(records[ch], errors='coerce').fillna(0).values
                   for ch in available])

    n_samples = X.shape[1]
    window_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)

    all_peaks = []

    for win_idx, start in enumerate(range(0, n_samples - window_samples, step_samples)):
        X_win = X[:, start:start + window_samples]

        # Run continuous sweep
        freqs, eigenvalues, _, _ = ged_continuous_sweep(
            X_win, fs, freq_range, step_hz=sweep_step_hz
        )

        if len(freqs) == 0 or len(eigenvalues) == 0:
            continue

        # Peak detection on FULL profile (not per-band!)
        min_dist_samples = max(1, int(min_distance_hz / sweep_step_hz))
        max_eig = np.max(eigenvalues)
        prominence = prominence_frac * max_eig if max_eig > 0 else 0.01

        peak_idx, properties = find_peaks(
            eigenvalues,
            distance=min_dist_samples,
            prominence=prominence
        )

        # Assign bands AFTER peak detection
        for idx in peak_idx:
            freq = freqs[idx]
            band = assign_band(freq)
            all_peaks.append({
                'frequency': float(freq),
                'eigenvalue': float(eigenvalues[idx]),
                'band': band,
                'window_idx': win_idx,
            })

    return pd.DataFrame(all_peaks)


def process_dataset(dataset_name: str, max_files: int = None) -> pd.DataFrame:
    """Process a single dataset with true continuous GED detection."""

    config = DATASET_CONFIGS[dataset_name]
    files = sorted(glob(config['pattern'], recursive=True))

    if max_files:
        files = files[:max_files]

    print(f"\n{'='*60}")
    print(f"PROCESSING: {dataset_name.upper()}")
    print(f"{'='*60}")
    print(f"Files: {len(files)}")
    print(f"Output: {config['output_dir']}")

    os.makedirs(config['output_dir'], exist_ok=True)

    all_peaks = []
    start_time = time.time()

    for i, file_path in enumerate(files):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"[{i+1}/{len(files)}] Processing: {os.path.basename(file_path)}", flush=True)

        try:
            # Try different loading strategies
            records = None
            header_rows = [config.get('header', 0), 1, 0]  # try config header first
            seen = set()
            for skiprows in header_rows:
                if skiprows in seen:
                    continue
                seen.add(skiprows)
                try:
                    records = pd.read_csv(file_path, skiprows=skiprows)
                    eeg_cols = [c for c in records.columns
                               if (c.startswith('EEG.') or c in ['AF3','AF4','F7','F8','F3','F4','FC5','FC6','P7','P8','T7','T8','O1','O2']) and
                               not any(x in c for x in ['FILTERED', 'POW', 'Alpha', 'Beta', 'Theta', 'Delta', 'Gamma'])]
                    if len(eeg_cols) >= 3:
                        break
                except:
                    continue

            if records is None or len(eeg_cols) < 3:
                continue

            # Run true continuous detection (use all available EEG channels)
            peaks_df = find_ged_peaks_truly_continuous(
                records, eeg_cols, fs=config['fs']
            )

            if len(peaks_df) > 0:
                peaks_df['session'] = os.path.basename(file_path)
                peaks_df['dataset'] = dataset_name
                all_peaks.append(peaks_df)

        except Exception as e:
            if i < 5:  # Only print first few errors
                print(f"  Error: {str(e)[:50]}")

    if not all_peaks:
        print(f"No peaks found for {dataset_name}")
        return pd.DataFrame()

    combined = pd.concat(all_peaks, ignore_index=True)
    elapsed = time.time() - start_time

    # Save
    output_path = f"{config['output_dir']}/ged_peaks_truly_continuous.csv"
    combined.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"{dataset_name.upper()} SUMMARY")
    print(f"{'='*60}")
    print(f"Total peaks: {len(combined):,}")
    print(f"Processing time: {elapsed/60:.1f} minutes")
    print(f"Saved: {output_path}")

    # Per-band counts
    print("\nPer-band counts:")
    for band in ['theta', 'alpha', 'beta_low', 'beta_high', 'gamma']:
        count = len(combined[combined['band'] == band])
        print(f"  {band}: {count:,}")

    # Boundary check
    verify_boundary_coverage(combined, dataset_name)

    return combined


def verify_boundary_coverage(df: pd.DataFrame, dataset_name: str):
    """Verify that boundary frequencies are captured."""
    print(f"\nBOUNDARY COVERAGE ({dataset_name}):")

    boundaries = [
        (7.60, "theta/alpha"),
        (12.30, "alpha/beta_low"),
        (19.90, "beta_low/beta_high"),
        (32.19, "beta_high/gamma"),
    ]

    window = 0.15

    for freq, name in boundaries:
        count = len(df[(df['frequency'] >= freq - window) & (df['frequency'] <= freq + window)])
        status = "OK" if count > 0 else "MISSING!"
        print(f"  {name:>18}: {freq:>6.2f} Hz -> {count:>6} peaks  {status}")


def create_individual_chart(dataset_name: str, f0: float = 7.60):
    """Create chart for a single dataset."""
    config = DATASET_CONFIGS[dataset_name]
    peaks_file = f"{config['output_dir']}/ged_peaks_truly_continuous.csv"

    if not os.path.exists(peaks_file):
        print(f"Peaks file not found: {peaks_file}")
        return

    print(f"\nCreating chart for {dataset_name}...")

    import subprocess
    result = subprocess.run([
        'python', 'scripts/create_clean_modes_chart.py',
        peaks_file,
        config['output_dir'],
        str(f0)
    ], capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print(result.stderr)


def create_aggregate_chart(f0: float = 7.60, datasets: list = None,
                           output_dir: str = None):
    """Create aggregate chart combining datasets.

    Parameters
    ----------
    f0 : float
        Base frequency for phi-scale.
    datasets : list, optional
        List of dataset keys to include. None = all datasets.
    output_dir : str, optional
        Override output directory. None = default aggregate_ged.
    """
    import matplotlib.pyplot as plt

    PHI = 1.618033988749895

    print("\n" + "="*60)
    print("CREATING AGGREGATE CHART")
    print("="*60)

    # Load datasets
    all_dfs = []
    dataset_counts = {}

    configs_to_load = {k: v for k, v in DATASET_CONFIGS.items()
                       if datasets is None or k in datasets}

    for dataset_name, config in configs_to_load.items():
        peaks_file = f"{config['output_dir']}/ged_peaks_truly_continuous.csv"
        if os.path.exists(peaks_file):
            df = pd.read_csv(peaks_file)
            df['dataset'] = dataset_name
            all_dfs.append(df)
            dataset_counts[dataset_name] = len(df)
            print(f"  {dataset_name}: {len(df):,} peaks")

    if not all_dfs:
        print("No data found!")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal combined: {len(combined):,} peaks")

    # Filter to max_freq
    max_freq = 45.0
    combined = combined[combined['frequency'] <= max_freq].copy()
    freqs = combined['frequency'].values

    # Convert to φ-exponent
    def freq_to_phi_exp(f):
        return np.log(f / f0) / np.log(PHI)

    def phi_exp_to_freq(n):
        return f0 * (PHI ** n)

    phi_exps = freq_to_phi_exp(freqs)

    # Create figure
    fig, ax = plt.subplots(figsize=(18, 7))

    n_min = -1.0
    n_max = freq_to_phi_exp(max_freq)

    # Band shading
    ANALYSIS_BANDS = {
        'theta': {'octave': -1, 'color': '#ffcccc'},
        'alpha': {'octave': 0, 'color': '#ccffcc'},
        'beta_low': {'octave': 1, 'color': '#ccccff'},
        'beta_high': {'octave': 2, 'color': '#ffffcc'},
        'gamma': {'octave': 3, 'color': '#ffccff'},
    }

    for band, info in ANALYSIS_BANDS.items():
        n_lo = info['octave']
        n_hi = info['octave'] + 1
        if n_lo < n_max:
            ax.axvspan(n_lo, min(n_hi, n_max), alpha=0.08, color=info['color'], zorder=0)

    # Histogram
    bins_per_octave = 30
    phi_bins = np.linspace(n_min, n_max, int((n_max - n_min) * bins_per_octave) + 1)
    counts, bin_edges, patches = ax.hist(phi_exps, bins=phi_bins, alpha=0.6,
                                          color='#b0b0b0', edgecolor='none', zorder=1)

    # φⁿ prediction lines
    POSITION_OFFSETS = {
        'boundary': 0.0,
        'noble_6': 0.0557,
        'noble_5': 0.0902,
        'noble_4': 0.146,
        'noble_3': 0.236,
        'noble_2': 0.382,
        'attractor': 0.500,
        'noble_1': 0.618,
        'inv_noble_3': 0.764,
        'inv_noble_4': 0.854,
        'inv_noble_5': 0.9098,
        'inv_noble_6': 0.9443,
    }

    POSITION_STYLES = {
        'boundary': {'color': '#e74c3c', 'linestyle': '-', 'linewidth': 2.0, 'alpha': 0.8},
        'attractor': {'color': '#27ae60', 'linestyle': '--', 'linewidth': 1.5, 'alpha': 0.6},
        'noble_1': {'color': '#3498db', 'linestyle': '--', 'linewidth': 1.2, 'alpha': 0.5},
        'noble_2': {'color': '#9b59b6', 'linestyle': ':', 'linewidth': 1.0, 'alpha': 0.4},
        'noble_3': {'color': '#f39c12', 'linestyle': ':', 'linewidth': 1.0, 'alpha': 0.4},
        'noble_4': {'color': '#1abc9c', 'linestyle': ':', 'linewidth': 1.0, 'alpha': 0.4},
        'noble_5': {'color': '#16a085', 'linestyle': ':', 'linewidth': 0.8, 'alpha': 0.3},
        'noble_6': {'color': '#2c3e50', 'linestyle': ':', 'linewidth': 0.8, 'alpha': 0.3},
        'inv_noble_3': {'color': '#e67e22', 'linestyle': ':', 'linewidth': 1.0, 'alpha': 0.35},
        'inv_noble_4': {'color': '#95a5a6', 'linestyle': ':', 'linewidth': 1.0, 'alpha': 0.35},
        'inv_noble_5': {'color': '#7f8c8d', 'linestyle': ':', 'linewidth': 0.8, 'alpha': 0.3},
        'inv_noble_6': {'color': '#bdc3c7', 'linestyle': ':', 'linewidth': 0.8, 'alpha': 0.3},
    }

    ymax = counts.max() * 1.15
    ax.set_ylim(0, ymax)

    bbox_style = dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.85, edgecolor='none')

    # Add prediction lines
    for octave in range(-1, 4):
        for pos_type, offset in POSITION_OFFSETS.items():
            n = octave + offset
            freq = phi_exp_to_freq(n)

            if freq < 4.5 or freq > max_freq:
                continue

            style = POSITION_STYLES.get(pos_type, POSITION_STYLES['boundary'])

            ax.axvline(n, color=style['color'], linestyle=style['linestyle'],
                      linewidth=style['linewidth'], alpha=style['alpha'], zorder=5)

            # Label at top
            ax.text(n, ymax * 0.98, f"{freq:.1f}",
                   ha='center', va='top', fontsize=6, fontweight='bold',
                   color=style['color'], alpha=0.9, rotation=90, bbox=bbox_style)

    # Position type legend
    position_labels = {
        'boundary': 'Boundary φ⁰',
        'noble_6': 'Noble₆ φ⁻⁶',
        'noble_5': 'Noble₅ φ⁻⁵',
        'noble_4': 'Noble₄ φ⁻⁴',
        'noble_3': 'Noble₃ φ⁻³',
        'noble_2': 'Noble₂ φ⁻²',
        'attractor': 'Attractor φ⁰·⁵',
        'noble_1': 'Noble₁ φ⁻¹',
        'inv_noble_3': 'Inv₃ 1-φ⁻³',
        'inv_noble_4': 'Inv₄ 1-φ⁻⁴',
        'inv_noble_5': 'Inv₅ 1-φ⁻⁵',
        'inv_noble_6': 'Inv₆ 1-φ⁻⁶',
    }

    unique_types = ['boundary', 'noble_6', 'noble_5', 'noble_4', 'noble_3', 'noble_2',
                    'attractor', 'noble_1', 'inv_noble_3', 'inv_noble_4', 'inv_noble_5', 'inv_noble_6']

    for i, ptype in enumerate(unique_types):
        if ptype in POSITION_STYLES:
            style = POSITION_STYLES[ptype]
            x_pos = 0.0 + i * 0.075
            ax.annotate(position_labels.get(ptype, ptype),
                       xy=(x_pos, 1.08), xycoords='axes fraction',
                       fontsize=7, color=style['color'], ha='left', fontweight='bold')

    # Band labels
    band_positions = {
        'theta': (-0.5, 'Theta'),
        'alpha': (0.5, 'Alpha'),
        'beta_low': (1.5, 'Beta Low'),
        'beta_high': (2.5, 'Beta High'),
        'gamma': (3.25, 'Gamma')
    }
    for band, (mid_n, label) in band_positions.items():
        if mid_n < n_max:
            ax.text(mid_n, ymax * 0.03, label,
                   ha='center', va='bottom', fontsize=11, fontweight='bold', color='#555555')

    # Frequency labels at integer φ-exponents
    for n in range(-1, 4):
        freq = phi_exp_to_freq(n)
        if freq <= max_freq:
            ax.text(n, ymax * 1.02, f'{freq:.1f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333333')

    # Formatting
    ax.set_xlim(n_min, n_max)
    ax.set_xlabel(f'φ-exponent (n)        f = {f0:.2f} × φⁿ Hz', fontsize=12, fontweight='bold')
    ax.set_ylabel('Peak Count', fontsize=12, fontweight='bold')
    ax.set_xticks(range(-1, 4))
    ax.tick_params(axis='x', which='both', length=0)
    ax.tick_params(axis='y', which='both', length=0)

    # Title matching individual chart style
    DISPLAY_NAMES = {
        'physf': 'PhySF', 'emotions': 'Emotions', 'mpeng': 'MPENG',
        'arithmetic_A': 'Arithmetic', 'arithmetic_M': 'Meditation',
        'arithmetic_B': 'Baseline', 'arithmetic_R': 'Rest',
    }
    ARITHMETIC_KEYS = {'arithmetic_A', 'arithmetic_M', 'arithmetic_B', 'arithmetic_R'}
    display_names = []
    arith_present = ARITHMETIC_KEYS & set(dataset_counts.keys())
    if arith_present == ARITHMETIC_KEYS:
        # All 4 arithmetic conditions → collapse to "MATH"
        for k in dataset_counts:
            if k not in ARITHMETIC_KEYS:
                display_names.append(DISPLAY_NAMES.get(k, k))
        display_names.append('MATH')
    else:
        display_names = [DISPLAY_NAMES.get(k, k) for k in dataset_counts]
    dataset_str = ' + '.join(display_names)
    ax.set_title(
        f'GED Peak Distribution vs φⁿ Predictions — Log-φ Scale (F0={f0:.2f} Hz)\n'
        f'{len(combined):,} peaks | Equal-width φ-octaves\n'
        f'({dataset_str})',
        fontsize=13, fontweight='bold', pad=40
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save
    if output_dir is None:
        output_dir = 'exports_peak_distribution/aggregate_ged'
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/aggregate_modes_vs_predictions_logphi_f0_{f0:.2f}.png".replace('.', '')
    output_path = output_path.replace('png', '.png')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nSaved aggregate chart: {output_path}")

    # Save combined CSV
    csv_path = f"{output_dir}/ged_peaks_all_datasets.csv"
    combined.to_csv(csv_path, index=False)
    print(f"Saved combined CSV: {csv_path}")

    # Verify boundary coverage for aggregate
    verify_boundary_coverage(combined, "AGGREGATE")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run true continuous GED detection')
    all_datasets = list(DATASET_CONFIGS.keys())
    parser.add_argument('dataset', nargs='?', default='all',
                       choices=all_datasets + ['all', 'charts', 'aggregate',
                                               'arithmetic', 'arithmetic_aggregate'],
                       help='Dataset to process or "all" for all datasets')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Limit number of files per dataset (for testing)')
    parser.add_argument('--f0', type=float, default=7.60,
                       help='F0 value for charts')

    args = parser.parse_args()

    ARITHMETIC_CONDITIONS = ['arithmetic_A', 'arithmetic_M', 'arithmetic_B', 'arithmetic_R']
    LEGACY_DATASETS = ['physf', 'emotions', 'mpeng']

    if args.dataset == 'all':
        # Process all datasets
        for dataset in LEGACY_DATASETS:
            process_dataset(dataset, args.max_files)

        # Create individual charts
        for dataset in LEGACY_DATASETS:
            create_individual_chart(dataset, args.f0)

        # Create aggregate chart
        create_aggregate_chart(args.f0)

    elif args.dataset == 'arithmetic':
        # Process all 4 arithmetic conditions
        for dataset in ARITHMETIC_CONDITIONS:
            process_dataset(dataset, args.max_files)

        # Create individual charts
        for dataset in ARITHMETIC_CONDITIONS:
            create_individual_chart(dataset, args.f0)

    elif args.dataset == 'arithmetic_aggregate':
        # Aggregate chart for arithmetic conditions only
        create_aggregate_chart(args.f0, datasets=ARITHMETIC_CONDITIONS,
                               output_dir='exports_peak_distribution/arithmetic_ged/aggregate')

    elif args.dataset == 'charts':
        # Just create charts (assumes data already processed)
        for dataset in ['physf', 'emotions', 'mpeng']:
            create_individual_chart(dataset, args.f0)
        create_aggregate_chart(args.f0)

    elif args.dataset == 'aggregate':
        # Just create aggregate chart
        create_aggregate_chart(args.f0)

    else:
        # Process single dataset
        process_dataset(args.dataset, args.max_files)
        create_individual_chart(args.dataset, args.f0)

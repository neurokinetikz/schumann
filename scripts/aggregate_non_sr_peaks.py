#!/usr/bin/env python3
"""
Aggregate Non-SR Peaks from Batch Analysis
==========================================

Reads per-session *_non_sr_peaks.csv files from batch analysis,
aggregates into a single dataset, and runs cross-session clustering analysis.

Usage:
    python aggregate_non_sr_peaks.py [--input-dir exports] [--output-dir exports/aggregated]
    python aggregate_non_sr_peaks.py --method kmeans --min-peaks 20

Outputs:
    - aggregated_non_sr_peaks.csv: Combined peaks from all sessions
    - aggregated_non_sr_cluster_centers.csv: Cluster summary stats
    - aggregated_non_sr_clusters.png: Cluster visualization
    - aggregated_summary.txt: Text summary
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from pathlib import Path

# Add lib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

from non_sr_clustering import NonSRPeakCollector, NonSRPeak


def find_non_sr_peak_files(input_dir: str,
                           exclude_patterns: List[str] = None) -> List[str]:
    """Find all *_non_sr_peaks.csv files recursively.

    Parameters
    ----------
    input_dir : str
        Root directory to search.
    exclude_patterns : list of str or None
        Filename patterns to exclude (e.g., ['aggregated', 'all_']).

    Returns
    -------
    list of str
        Paths to all found CSV files.
    """
    if exclude_patterns is None:
        exclude_patterns = ['aggregated', 'all_non_sr']

    pattern = os.path.join(input_dir, '**', '*_non_sr_peaks.csv')
    files = glob.glob(pattern, recursive=True)

    # Filter out excluded patterns
    filtered = []
    for f in files:
        basename = os.path.basename(f)
        if not any(excl in basename for excl in exclude_patterns):
            filtered.append(f)

    return sorted(filtered)


def load_and_aggregate(file_paths: List[str]) -> Tuple[pd.DataFrame, dict]:
    """Load and concatenate all CSV files.

    Parameters
    ----------
    file_paths : list of str
        Paths to CSV files.

    Returns
    -------
    tuple of (DataFrame, dict)
        Combined DataFrame and stats dict with per-file counts.
    """
    dfs = []
    stats = {'files': [], 'peaks_per_file': []}

    for fpath in file_paths:
        try:
            df = pd.read_csv(fpath)
            if len(df) > 0:
                # Add source file for tracking
                df['source_file'] = os.path.basename(fpath)
                dfs.append(df)
                stats['files'].append(fpath)
                stats['peaks_per_file'].append(len(df))
                print(f"  Loaded {len(df):>4} peaks from {os.path.basename(fpath)}")
        except Exception as e:
            print(f"  WARNING: Failed to load {fpath}: {e}")

    if not dfs:
        return pd.DataFrame(), stats

    combined = pd.concat(dfs, ignore_index=True)
    return combined, stats


def row_to_peak(row: pd.Series) -> NonSRPeak:
    """Convert a DataFrame row to NonSRPeak object.

    Parameters
    ----------
    row : pd.Series
        Row from aggregated DataFrame.

    Returns
    -------
    NonSRPeak
        Reconstructed peak object.
    """
    # Handle potential NaN values
    def safe_float(val, default=np.nan):
        if pd.isna(val):
            return default
        return float(val)

    def safe_int(val, default=0):
        if pd.isna(val):
            return default
        return int(val)

    def safe_str(val, default=''):
        if pd.isna(val):
            return default
        return str(val)

    return NonSRPeak(
        freq_hz=safe_float(row['freq_hz']),
        power_log10=safe_float(row['power_log10']),
        bandwidth_hz=safe_float(row['bandwidth_hz']),
        session_id=safe_str(row['session_id'], 'unknown'),
        window_type=safe_str(row.get('window_type', 'session'), 'session'),
        window_index=safe_int(row.get('window_index', 0)),
        window_start_sec=safe_float(row.get('window_start_sec', np.nan)),
        window_end_sec=safe_float(row.get('window_end_sec', np.nan)),
        cluster_label=-1  # Reset for re-clustering
    )


def dataframe_to_peaks(df: pd.DataFrame) -> List[NonSRPeak]:
    """Convert DataFrame to list of NonSRPeak objects.

    Parameters
    ----------
    df : pd.DataFrame
        Aggregated peaks DataFrame.

    Returns
    -------
    list of NonSRPeak
        Peak objects ready for clustering.
    """
    peaks = []
    for _, row in df.iterrows():
        try:
            peak = row_to_peak(row)
            # Skip invalid peaks
            if np.isnan(peak.freq_hz) or np.isnan(peak.power_log10):
                continue
            peaks.append(peak)
        except Exception as e:
            print(f"  WARNING: Failed to convert row: {e}")
    return peaks


def cluster_by_frequency_only(peaks: List[NonSRPeak],
                               method: str = 'meanshift',
                               bandwidth: float = None) -> Tuple[np.ndarray, dict]:
    """Cluster peaks by frequency only (1D clustering).

    Parameters
    ----------
    peaks : list of NonSRPeak
        Peaks to cluster.
    method : str
        'meanshift' (auto bandwidth) or 'kmeans'.
    bandwidth : float or None
        Bandwidth for MeanShift. If None, estimated automatically.

    Returns
    -------
    tuple of (labels, info)
        Cluster labels and clustering info dict.
    """
    from sklearn.cluster import MeanShift, KMeans, estimate_bandwidth
    from sklearn.preprocessing import StandardScaler

    # Extract frequencies only (1D)
    freqs = np.array([p.freq_hz for p in peaks]).reshape(-1, 1)

    if method == 'meanshift':
        # Estimate bandwidth if not provided
        if bandwidth is None:
            bandwidth = estimate_bandwidth(freqs, quantile=0.1)
            if bandwidth <= 0:
                bandwidth = 2.0  # Default 2 Hz bandwidth
        print(f"  MeanShift bandwidth: {bandwidth:.2f} Hz")

        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        labels = ms.fit_predict(freqs)
        info = {'bandwidth': bandwidth, 'cluster_centers': ms.cluster_centers_.flatten()}

    elif method == 'kmeans':
        # Auto-select k using silhouette score
        from sklearn.metrics import silhouette_score

        best_k, best_score = 2, -1
        max_k = min(15, len(freqs) - 1)

        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels_k = km.fit_predict(freqs)
            if len(set(labels_k)) > 1:
                score = silhouette_score(freqs, labels_k)
                if score > best_score:
                    best_score = score
                    best_k = k

        print(f"  KMeans optimal k: {best_k} (silhouette: {best_score:.3f})")
        km = KMeans(n_clusters=best_k, n_init=20, random_state=42)
        labels = km.fit_predict(freqs)
        info = {'n_clusters': best_k, 'silhouette': best_score,
                'cluster_centers': km.cluster_centers_.flatten()}

    else:
        raise ValueError(f"Unknown method: {method}")

    return labels, info


def compute_cluster_stats_freq_only(peaks: List[NonSRPeak],
                                     labels: np.ndarray) -> List[dict]:
    """Compute cluster statistics for frequency-only clustering.

    Parameters
    ----------
    peaks : list of NonSRPeak
        All peaks.
    labels : ndarray
        Cluster labels.

    Returns
    -------
    list of dict
        Statistics for each cluster.
    """
    unique_labels = sorted(set(labels))
    if -1 in unique_labels:
        unique_labels.remove(-1)  # Remove noise label

    cluster_stats = []
    for label in unique_labels:
        mask = labels == label
        cluster_peaks = [p for p, m in zip(peaks, mask) if m]

        freqs = np.array([p.freq_hz for p in cluster_peaks])
        powers = np.array([p.power_log10 for p in cluster_peaks])
        bws = np.array([p.bandwidth_hz for p in cluster_peaks])
        sessions = set(p.session_id for p in cluster_peaks)

        stats = {
            'cluster_id': int(label),
            'n_peaks': int(mask.sum()),
            'center_freq_hz': float(np.mean(freqs)),
            'std_freq_hz': float(np.std(freqs)),
            'min_freq_hz': float(np.min(freqs)),
            'max_freq_hz': float(np.max(freqs)),
            'mean_power': float(np.mean(powers)),
            'std_power': float(np.std(powers)),
            'mean_bandwidth': float(np.mean(bws)),
            'n_sessions': len(sessions),
            'session_prevalence': len(sessions) / len(set(p.session_id for p in peaks))
        }
        cluster_stats.append(stats)

    # Sort by center frequency
    cluster_stats.sort(key=lambda x: x['center_freq_hz'])

    return cluster_stats


def run_aggregated_clustering(peaks: List[NonSRPeak],
                               output_dir: str,
                               method: str = 'auto',
                               freq_only: bool = True,
                               bandwidth: float = None,
                               show: bool = False) -> dict:
    """Run clustering on aggregated peaks.

    Parameters
    ----------
    peaks : list of NonSRPeak
        All peaks to cluster.
    output_dir : str
        Output directory for results.
    method : str
        Clustering method ('auto', 'kmeans', 'dbscan', 'hierarchical', 'meanshift').
    freq_only : bool
        If True, cluster by frequency only (1D). Default True.
    bandwidth : float or None
        Bandwidth for frequency-only MeanShift clustering.
    show : bool
        Whether to display plots.

    Returns
    -------
    dict
        Clustering results.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create collector and add peaks
    collector = NonSRPeakCollector(freq_range=(1.0, 50.0))
    n_added = collector.add_precomputed(peaks)
    print(f"\n  Added {n_added} peaks to collector")

    if freq_only:
        # Use frequency-only 1D clustering
        cluster_method = 'meanshift' if method == 'auto' else method
        print(f"\n--- Running frequency-only {cluster_method} clustering ---")

        labels, cluster_info = cluster_by_frequency_only(
            collector.peaks, method=cluster_method, bandwidth=bandwidth
        )

        # Compute cluster stats
        cluster_stats = compute_cluster_stats_freq_only(collector.peaks, labels)

        # Build results dict matching NonSRPeakCollector format
        cluster_centers_hz = [s['center_freq_hz'] for s in cluster_stats]
        results = {
            'labels': labels,
            'n_clusters': len(cluster_stats),
            'cluster_centers_hz': cluster_centers_hz,
            'cluster_stats': cluster_stats,
            'method': f'freq_only_{cluster_method}',
            'cluster_info': cluster_info
        }

        # Assign labels back to peaks
        for peak, label in zip(collector.peaks, labels):
            peak.cluster_label = int(label)

        # Generate plot
        fig = plot_freq_only_clusters(collector.peaks, labels, cluster_stats,
                                       output_dir, show=show)

        # Export results
        output_files = collector.export_results(
            results, out_dir=output_dir, session_name='aggregated'
        )

    else:
        # Use original 3D clustering
        print(f"\n--- Running {method} clustering (3D: freq, power, bw) ---")
        results, fig = collector.cluster_and_plot(
            method=method,
            title=f'Aggregated Non-SR Peak Clusters ({collector.n_peaks} peaks, {collector.n_sessions} sessions)',
            out_dir=output_dir,
            session_name='aggregated',
            show=show
        )

        output_files = collector.export_results(
            results, out_dir=output_dir, session_name='aggregated'
        )

    return results, output_files, collector


def plot_freq_only_clusters(peaks: List[NonSRPeak],
                             labels: np.ndarray,
                             cluster_stats: List[dict],
                             output_dir: str,
                             show: bool = False):
    """Plot frequency-only clustering results.

    Parameters
    ----------
    peaks : list of NonSRPeak
        All peaks.
    labels : ndarray
        Cluster labels.
    cluster_stats : list of dict
        Cluster statistics.
    output_dir : str
        Output directory.
    show : bool
        Whether to display plot.
    """
    import matplotlib.pyplot as plt

    freqs = np.array([p.freq_hz for p in peaks])
    powers = np.array([p.power_log10 for p in peaks])

    # Color map
    unique_labels = sorted(set(labels))
    n_clusters = len([l for l in unique_labels if l >= 0])
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, n_clusters)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Frequency histogram with cluster coloring
    ax1 = axes[0, 0]
    for i, label in enumerate(unique_labels):
        if label < 0:
            continue
        mask = labels == label
        ax1.hist(freqs[mask], bins=30, alpha=0.6, label=f'C{label}', color=colors[i % len(colors)])
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Count')
    ax1.set_title('Frequency Distribution by Cluster')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. Scatter: Frequency vs Power (colored by cluster)
    ax2 = axes[0, 1]
    for i, label in enumerate(unique_labels):
        if label < 0:
            continue
        mask = labels == label
        ax2.scatter(freqs[mask], powers[mask], alpha=0.5, s=20,
                   color=colors[i % len(colors)], label=f'C{label}')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power (log10)')
    ax2.set_title('Frequency vs Power (colored by cluster)')
    ax2.grid(True, alpha=0.3)

    # Add vertical lines for cluster centers
    for stats in cluster_stats:
        ax2.axvline(stats['center_freq_hz'], color='red', alpha=0.3, linestyle='--')

    # 3. Cluster centers bar chart
    ax3 = axes[1, 0]
    centers = [s['center_freq_hz'] for s in cluster_stats]
    stds = [s['std_freq_hz'] for s in cluster_stats]
    n_peaks = [s['n_peaks'] for s in cluster_stats]
    x = range(len(centers))

    bars = ax3.bar(x, centers, yerr=stds, capsize=3, alpha=0.7,
                   color=[colors[i % len(colors)] for i in range(len(centers))])
    ax3.set_xlabel('Cluster')
    ax3.set_ylabel('Center Frequency (Hz)')
    ax3.set_title('Cluster Centers with Std Dev')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'C{i}\n(n={n})' for i, n in enumerate(n_peaks)], fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Session prevalence
    ax4 = axes[1, 1]
    prevalence = [s['session_prevalence'] * 100 for s in cluster_stats]
    n_sessions = [s['n_sessions'] for s in cluster_stats]

    bars = ax4.bar(x, prevalence, alpha=0.7,
                   color=[colors[i % len(colors)] for i in range(len(prevalence))])
    ax4.set_xlabel('Cluster')
    ax4.set_ylabel('Session Prevalence (%)')
    ax4.set_title('Cross-Session Prevalence')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{c:.1f} Hz\n({n} sess)' for c, n in zip(centers, n_sessions)],
                        fontsize=7, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save
    out_path = os.path.join(output_dir, 'aggregated_non_sr_clusters.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out_path}")

    if show:
        plt.show()
    plt.close(fig)

    return fig


def write_summary(output_dir: str,
                  load_stats: dict,
                  cluster_results: dict,
                  collector: NonSRPeakCollector):
    """Write text summary file.

    Parameters
    ----------
    output_dir : str
        Output directory.
    load_stats : dict
        Stats from loading phase.
    cluster_results : dict
        Results from clustering.
    collector : NonSRPeakCollector
        Collector with peaks.
    """
    summary_path = os.path.join(output_dir, 'aggregated_summary.txt')

    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("AGGREGATED NON-SR PEAKS ANALYSIS\n")
        f.write("=" * 60 + "\n\n")

        f.write("INPUT FILES\n")
        f.write("-" * 40 + "\n")
        f.write(f"Files loaded: {len(load_stats['files'])}\n")
        f.write(f"Total peaks:  {sum(load_stats['peaks_per_file'])}\n\n")

        f.write("Per-file breakdown:\n")
        for fpath, count in zip(load_stats['files'], load_stats['peaks_per_file']):
            fname = os.path.basename(fpath)
            f.write(f"  {count:>4} peaks: {fname}\n")

        f.write("\n\nCLUSTERING RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Method:       {cluster_results.get('method', 'N/A')}\n")
        f.write(f"Sessions:     {collector.n_sessions}\n")
        f.write(f"Total peaks:  {collector.n_peaks}\n")
        f.write(f"Clusters:     {cluster_results.get('n_clusters', 0)}\n")

        if cluster_results.get('cluster_stats'):
            f.write("\nCluster Centers:\n")
            f.write(f"  {'Cluster':>7}  {'Center (Hz)':>11}  {'Std (Hz)':>8}  {'N peaks':>7}  {'Sessions':>8}\n")
            f.write(f"  {'-'*7}  {'-'*11}  {'-'*8}  {'-'*7}  {'-'*8}\n")

            for i, stats in enumerate(cluster_results['cluster_stats']):
                f.write(f"  {i:>7}  {stats['center_freq_hz']:>11.2f}  "
                       f"{stats['std_freq_hz']:>8.2f}  {stats['n_peaks']:>7}  "
                       f"{stats['n_sessions']:>8}\n")

    print(f"  Saved: {summary_path}")
    return summary_path


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate non-SR peaks from batch analysis and run clustering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python aggregate_non_sr_peaks.py
  python aggregate_non_sr_peaks.py --input-dir exports --output-dir exports/aggregated
  python aggregate_non_sr_peaks.py --method kmeans --min-peaks 20
        """
    )

    parser.add_argument('--input-dir', type=str, default='exports',
                        help='Directory to search for *_non_sr_peaks.csv files (default: exports)')
    parser.add_argument('--output-dir', type=str, default='exports/aggregated',
                        help='Output directory for results (default: exports/aggregated)')
    parser.add_argument('--method', type=str, default='auto',
                        choices=['auto', 'kmeans', 'meanshift'],
                        help='Clustering method (default: auto = meanshift)')
    parser.add_argument('--min-peaks', type=int, default=10,
                        help='Minimum peaks required for clustering (default: 10)')
    parser.add_argument('--bandwidth', type=float, default=None,
                        help='MeanShift bandwidth in Hz (default: auto-estimated)')
    parser.add_argument('--no-freq-only', action='store_true',
                        help='Use 3D clustering (freq+power+bw) instead of frequency-only')
    parser.add_argument('--show', action='store_true',
                        help='Display plots interactively')

    args = parser.parse_args()

    freq_only = not args.no_freq_only

    print("=" * 60)
    print("AGGREGATE NON-SR PEAKS")
    print("=" * 60)
    print(f"\nInput directory:  {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Clustering: {'frequency-only (1D)' if freq_only else '3D (freq+power+bw)'}")
    print(f"Method: {args.method}")
    if args.bandwidth:
        print(f"Bandwidth: {args.bandwidth} Hz")
    print(f"Min peaks: {args.min_peaks}")

    # Find files
    print("\n--- Finding CSV files ---")
    files = find_non_sr_peak_files(args.input_dir)
    print(f"Found {len(files)} *_non_sr_peaks.csv files")

    if not files:
        print("ERROR: No files found. Check --input-dir path.")
        sys.exit(1)

    # Load and aggregate
    print("\n--- Loading and aggregating ---")
    df, load_stats = load_and_aggregate(files)

    if df.empty:
        print("ERROR: No peaks loaded from any file.")
        sys.exit(1)

    print(f"\nTotal: {len(df)} peaks from {len(load_stats['files'])} files")

    # Convert to peak objects
    print("\n--- Converting to peak objects ---")
    peaks = dataframe_to_peaks(df)
    print(f"Converted {len(peaks)} valid peaks")

    if len(peaks) < args.min_peaks:
        print(f"ERROR: Only {len(peaks)} peaks found, need at least {args.min_peaks} for clustering.")
        sys.exit(1)

    # Run clustering
    results, output_files, collector = run_aggregated_clustering(
        peaks, args.output_dir,
        method=args.method,
        freq_only=freq_only,
        bandwidth=args.bandwidth,
        show=args.show
    )

    print(f"\n  Clusters found: {results.get('n_clusters', 0)}")
    if results.get('cluster_centers_hz'):
        centers_str = ', '.join(f"{c:.1f}" for c in results['cluster_centers_hz'])
        print(f"  Cluster centers (Hz): {centers_str}")

    # Print output files
    print("\n--- Output files ---")
    for name, path in output_files.items():
        print(f"  {name}: {path}")

    # Write summary
    summary_path = write_summary(args.output_dir, load_stats, results, collector)

    # Also save the aggregated raw DataFrame
    agg_csv_path = os.path.join(args.output_dir, 'aggregated_all_peaks_raw.csv')
    df.to_csv(agg_csv_path, index=False)
    print(f"  raw_csv: {agg_csv_path}")

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)

    return results


if __name__ == '__main__':
    main()

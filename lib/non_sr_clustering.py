"""
Non-SR Peak Collection and Clustering
=====================================

Collects non-Schumann Resonance peaks from FOOOF harmonic refinement
and clusters them to identify common oscillation frequencies.

Two collection modes:
1. Per-session: Collect from session-level and per-event FOOOF fits
2. Cross-session: Aggregate peaks across multiple sessions for grand clustering

Usage Example
-------------
# Per-session collection
collector = NonSRPeakCollector()
collector.add_from_fooof_result(session_fooof_result, session_id='S01')
collector.add_from_fooof_result(event_fooof_result, session_id='S01',
                                 window_type='ignition', window_index=0)

# Clustering and visualization
clusters, fig = collector.cluster_and_plot(n_clusters='auto', out_dir='exports/S01')

# Cross-session aggregation
grand_collector = NonSRPeakCollector()
for session_collector in session_collectors:
    grand_collector.add_precomputed(session_collector.peaks)
grand_clusters, fig = grand_collector.cluster_and_plot(n_clusters='auto')
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Union, Set, Any
import warnings


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class NonSRPeak:
    """Container for a single non-SR peak with metadata."""
    freq_hz: float              # Peak center frequency
    power_log10: float          # Peak power (log10 scale from FOOOF)
    bandwidth_hz: float         # Peak bandwidth
    session_id: str             # Session identifier
    window_type: str = 'session'  # 'session' | 'ignition' | 'baseline'
    window_index: int = 0       # Index within window type (0 for session, 0-N for ignition)
    window_start_sec: float = np.nan  # Window start time (NaN for session-level)
    window_end_sec: float = np.nan    # Window end time (NaN for session-level)
    cluster_label: int = -1     # Assigned cluster (-1 = unassigned)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


# =============================================================================
# Peak Collector
# =============================================================================

class NonSRPeakCollector:
    """Collector for non-SR peaks with clustering capabilities.

    This class collects non-Schumann Resonance peaks from FOOOF harmonic
    detection and provides methods for clustering and visualization.

    Parameters
    ----------
    freq_range : tuple of float
        Only collect peaks within this frequency range (Hz).
        Default (1.0, 50.0).
    min_power : float or None
        Minimum peak power (log10) to include. Default None (no filter).

    Examples
    --------
    >>> collector = NonSRPeakCollector()
    >>> collector.add_from_fooof_result(fooof_result, session_id='session1')
    >>> df = collector.to_dataframe()
    >>> results, fig = collector.cluster_and_plot(method='auto')
    """

    def __init__(self,
                 freq_range: Tuple[float, float] = (1.0, 50.0),
                 min_power: Optional[float] = None):
        self.freq_range = freq_range
        self.min_power = min_power
        self._peaks: List[NonSRPeak] = []
        self._session_ids: Set[str] = set()

    @property
    def peaks(self) -> List[NonSRPeak]:
        """Get all collected peaks."""
        return self._peaks

    @property
    def n_peaks(self) -> int:
        """Number of collected peaks."""
        return len(self._peaks)

    @property
    def n_sessions(self) -> int:
        """Number of unique sessions."""
        return len(self._session_ids)

    @property
    def session_ids(self) -> List[str]:
        """List of unique session IDs."""
        return sorted(self._session_ids)

    def add_from_fooof_result(self,
                               fooof_result: Any,
                               session_id: str,
                               window_type: str = 'session',
                               window_index: int = 0,
                               window_start: float = np.nan,
                               window_end: float = np.nan) -> int:
        """Add non-SR peaks from a FOOOFHarmonicResult.

        Parameters
        ----------
        fooof_result : FOOOFHarmonicResult
            Result from detect_harmonics_fooof() with unmatched_peaks populated.
        session_id : str
            Session identifier.
        window_type : str
            Type of window: 'session', 'ignition', or 'baseline'.
        window_index : int
            Index of the window (0 for session-level).
        window_start : float
            Window start time in seconds (NaN for session-level).
        window_end : float
            Window end time in seconds (NaN for session-level).

        Returns
        -------
        int
            Number of peaks added.
        """
        if not hasattr(fooof_result, 'unmatched_peaks'):
            return 0

        count = 0
        for peak_dict in fooof_result.unmatched_peaks:
            freq = peak_dict.get('freq', np.nan)
            power = peak_dict.get('power', np.nan)
            bandwidth = peak_dict.get('bandwidth', np.nan)

            # Apply filters
            if not np.isfinite(freq):
                continue
            if freq < self.freq_range[0] or freq > self.freq_range[1]:
                continue
            if self.min_power is not None and power < self.min_power:
                continue

            peak = NonSRPeak(
                freq_hz=freq,
                power_log10=power,
                bandwidth_hz=bandwidth,
                session_id=session_id,
                window_type=window_type,
                window_index=window_index,
                window_start_sec=window_start,
                window_end_sec=window_end
            )
            self._peaks.append(peak)
            count += 1

        if count > 0:
            self._session_ids.add(session_id)

        return count

    def add_from_dict(self,
                      peak_dict: Dict,
                      session_id: str,
                      window_type: str = 'session',
                      window_index: int = 0,
                      window_start: float = np.nan,
                      window_end: float = np.nan) -> bool:
        """Add a single peak from a dictionary.

        Parameters
        ----------
        peak_dict : dict
            Dictionary with keys 'freq', 'power', 'bandwidth'.
        session_id : str
            Session identifier.
        window_type : str
            Type of window.
        window_index : int
            Index of the window.
        window_start : float
            Window start time.
        window_end : float
            Window end time.

        Returns
        -------
        bool
            True if peak was added, False if filtered out.
        """
        freq = peak_dict.get('freq', np.nan)
        power = peak_dict.get('power', np.nan)
        bandwidth = peak_dict.get('bandwidth', np.nan)

        # Apply filters
        if not np.isfinite(freq):
            return False
        if freq < self.freq_range[0] or freq > self.freq_range[1]:
            return False
        if self.min_power is not None and power < self.min_power:
            return False

        peak = NonSRPeak(
            freq_hz=freq,
            power_log10=power,
            bandwidth_hz=bandwidth,
            session_id=session_id,
            window_type=window_type,
            window_index=window_index,
            window_start_sec=window_start,
            window_end_sec=window_end
        )
        self._peaks.append(peak)
        self._session_ids.add(session_id)
        return True

    def add_precomputed(self, peaks: List[NonSRPeak]) -> int:
        """Add precomputed NonSRPeak objects (for cross-session aggregation).

        Parameters
        ----------
        peaks : list of NonSRPeak
            Pre-existing peak objects to add.

        Returns
        -------
        int
            Number of peaks added.
        """
        count = 0
        for peak in peaks:
            # Apply filters
            if peak.freq_hz < self.freq_range[0] or peak.freq_hz > self.freq_range[1]:
                continue
            if self.min_power is not None and peak.power_log10 < self.min_power:
                continue

            self._peaks.append(peak)
            self._session_ids.add(peak.session_id)
            count += 1

        return count

    def to_dataframe(self) -> pd.DataFrame:
        """Export all peaks to a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns for all peak attributes.
        """
        if not self._peaks:
            return pd.DataFrame(columns=[
                'session_id', 'window_type', 'window_index',
                'window_start_sec', 'window_end_sec',
                'freq_hz', 'power_log10', 'bandwidth_hz', 'cluster_label'
            ])

        records = [peak.to_dict() for peak in self._peaks]
        return pd.DataFrame(records)

    def get_feature_matrix(self) -> np.ndarray:
        """Get feature matrix for clustering.

        Returns
        -------
        np.ndarray
            Feature matrix with shape (n_peaks, 3) containing
            [frequency, power, bandwidth].
        """
        if not self._peaks:
            return np.empty((0, 3))

        features = np.array([
            [p.freq_hz, p.power_log10, p.bandwidth_hz]
            for p in self._peaks
        ])
        return features

    def cluster(self,
                method: str = 'auto',
                n_clusters: Union[int, str] = 'auto',
                **kwargs) -> Dict:
        """Cluster peaks by frequency, power, and bandwidth.

        Parameters
        ----------
        method : str
            Clustering method: 'kmeans', 'dbscan', 'hierarchical', or 'auto'.
            'auto' selects based on data characteristics.
        n_clusters : int or 'auto'
            Number of clusters. Ignored for DBSCAN.
            'auto' uses elbow method for hierarchical or silhouette for kmeans.
        **kwargs
            Additional parameters passed to clustering functions.

        Returns
        -------
        dict
            Clustering results with keys:
            - 'labels': cluster labels for each peak
            - 'n_clusters': number of clusters found
            - 'cluster_centers_hz': center frequencies of each cluster
            - 'cluster_stats': statistics for each cluster
            - 'method': clustering method used
            - 'linkage': linkage matrix (only for hierarchical)
        """
        if len(self._peaks) < 2:
            return {
                'labels': np.array([-1] * len(self._peaks)),
                'n_clusters': 0,
                'cluster_centers_hz': [],
                'cluster_stats': [],
                'method': method
            }

        features = self.get_feature_matrix()

        # Filter out rows with NaN or Inf values
        valid_mask = np.all(np.isfinite(features), axis=1)
        n_invalid = (~valid_mask).sum()
        if n_invalid > 0:
            warnings.warn(f"Excluding {n_invalid} peaks with NaN/Inf values from clustering")

        # If too few valid peaks for clustering, return all as noise
        if valid_mask.sum() < 2:
            return {
                'labels': np.array([-1] * len(self._peaks)),
                'n_clusters': 0,
                'cluster_centers_hz': [],
                'cluster_stats': [],
                'method': method,
                'n_excluded': n_invalid
            }

        features_valid = features[valid_mask]
        valid_indices = np.where(valid_mask)[0]

        # Auto-select method
        if method == 'auto':
            method = self._select_clustering_method()

        # Run clustering on valid features only
        if method == 'kmeans':
            labels_valid, info = self._cluster_kmeans(features_valid, n_clusters, **kwargs)
        elif method == 'dbscan':
            labels_valid, info = self._cluster_dbscan(features_valid, **kwargs)
        elif method == 'hierarchical':
            labels_valid, info = self._cluster_hierarchical(features_valid, n_clusters, **kwargs)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        # Map labels back to full array (invalid peaks get -1)
        labels = np.full(len(self._peaks), -1, dtype=int)
        labels[valid_indices] = labels_valid

        # Update peak cluster labels
        for i, label in enumerate(labels):
            self._peaks[i].cluster_label = int(label)

        # Compute cluster statistics (using valid features only)
        unique_labels = sorted(set(labels_valid) - {-1})
        cluster_stats = []
        cluster_centers_hz = []

        for label in unique_labels:
            mask = labels_valid == label
            cluster_freqs = features_valid[mask, 0]
            cluster_powers = features_valid[mask, 1]
            cluster_bws = features_valid[mask, 2]

            # Find sessions in this cluster (map back through valid_indices)
            cluster_peak_indices = valid_indices[np.where(mask)[0]]
            cluster_sessions = set(self._peaks[i].session_id for i in cluster_peak_indices)

            stats = {
                'cluster_id': label,
                'n_peaks': int(mask.sum()),
                'center_freq_hz': float(np.mean(cluster_freqs)),
                'std_freq_hz': float(np.std(cluster_freqs)),
                'min_freq_hz': float(np.min(cluster_freqs)),
                'max_freq_hz': float(np.max(cluster_freqs)),
                'mean_power': float(np.mean(cluster_powers)),
                'std_power': float(np.std(cluster_powers)),
                'mean_bandwidth': float(np.mean(cluster_bws)),
                'n_sessions': len(cluster_sessions),
                'session_prevalence': len(cluster_sessions) / max(1, self.n_sessions)
            }
            cluster_stats.append(stats)
            cluster_centers_hz.append(stats['center_freq_hz'])

        results = {
            'labels': labels,
            'n_clusters': len(unique_labels),
            'cluster_centers_hz': cluster_centers_hz,
            'cluster_stats': cluster_stats,
            'method': method,
            'n_excluded': n_invalid
        }
        results.update(info)

        return results

    def _select_clustering_method(self) -> str:
        """Auto-select clustering method based on data characteristics."""
        n_peaks = len(self._peaks)
        n_sessions = self.n_sessions

        if n_peaks < 10:
            return 'hierarchical'  # Better for small datasets
        elif n_sessions == 1:
            return 'dbscan'  # Good for single-session, unknown n_clusters
        else:
            return 'kmeans'  # Good for cross-session with expected structure

    def _cluster_kmeans(self,
                        features: np.ndarray,
                        n_clusters: Union[int, str] = 'auto',
                        seed: int = 42,
                        **kwargs) -> Tuple[np.ndarray, Dict]:
        """K-means clustering on features."""
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import silhouette_score
        except ImportError:
            raise ImportError("sklearn required for kmeans clustering")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)

        # Auto-select n_clusters using silhouette score
        if n_clusters == 'auto':
            best_k = 2
            best_score = -1
            max_k = min(10, len(features) - 1)

            for k in range(2, max_k + 1):
                km = KMeans(n_clusters=k, n_init=10, random_state=seed)
                labels_k = km.fit_predict(X_scaled)
                if len(set(labels_k)) > 1:
                    score = silhouette_score(X_scaled, labels_k)
                    if score > best_score:
                        best_score = score
                        best_k = k

            n_clusters = best_k

        km = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
        labels = km.fit_predict(X_scaled)

        return labels, {'silhouette_score': best_score if n_clusters == 'auto' else None}

    def _cluster_dbscan(self,
                        features: np.ndarray,
                        eps: float = 0.5,
                        min_samples: int = 3,
                        **kwargs) -> Tuple[np.ndarray, Dict]:
        """DBSCAN clustering for density-based grouping."""
        try:
            from sklearn.cluster import DBSCAN
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise ImportError("sklearn required for DBSCAN clustering")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)

        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X_scaled)

        n_noise = (labels == -1).sum()

        return labels, {'eps': eps, 'min_samples': min_samples, 'n_noise': n_noise}

    def _cluster_hierarchical(self,
                               features: np.ndarray,
                               n_clusters: Union[int, str] = 'auto',
                               **kwargs) -> Tuple[np.ndarray, Dict]:
        """Hierarchical clustering with dendrogram support."""
        try:
            from scipy.cluster.hierarchy import linkage, fcluster
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise ImportError("scipy and sklearn required for hierarchical clustering")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)

        Z = linkage(X_scaled, method='ward')

        if n_clusters == 'auto':
            n_clusters = self._find_optimal_clusters(Z)

        labels = fcluster(Z, n_clusters, criterion='maxclust') - 1  # Convert to 0-indexed

        return labels, {'linkage': Z, 'optimal_k': n_clusters}

    def _find_optimal_clusters(self, Z: np.ndarray, max_clusters: int = 10) -> int:
        """Find optimal number of clusters using elbow method on linkage matrix."""
        distances = Z[:, 2]

        if len(distances) < 2:
            return 2

        acceleration = np.diff(np.diff(distances))

        if len(acceleration) == 0:
            return 2

        # Find elbow point
        elbow_idx = np.argmax(acceleration) + 2
        return min(max(2, elbow_idx), max_clusters)

    def plot_clusters(self,
                      labels: Optional[np.ndarray] = None,
                      cluster_stats: Optional[List[Dict]] = None,
                      title: str = 'Non-SR Peak Clusters',
                      figsize: Tuple[float, float] = (14, 10),
                      out_path: Optional[str] = None,
                      show: bool = True) -> plt.Figure:
        """Generate cluster visualization.

        Parameters
        ----------
        labels : ndarray or None
            Cluster labels. If None, uses labels from peaks.
        cluster_stats : list of dict or None
            Cluster statistics for annotation.
        title : str
            Plot title.
        figsize : tuple
            Figure size.
        out_path : str or None
            Path to save figure.
        show : bool
            Whether to display the figure.

        Returns
        -------
        plt.Figure
            The generated figure.
        """
        if not self._peaks:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'No peaks collected', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title(title)
            if out_path:
                fig.savefig(out_path, dpi=150, bbox_inches='tight')
            if show:
                plt.show()
            return fig

        features = self.get_feature_matrix()

        if labels is None:
            labels = np.array([p.cluster_label for p in self._peaks])

        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(title, fontsize=14, fontweight='bold')

        # Get unique labels (excluding noise label -1)
        unique_labels = sorted(set(labels))
        n_clusters = len([l for l in unique_labels if l >= 0])

        # Color map
        if n_clusters > 0:
            cmap = plt.cm.get_cmap('tab10', max(n_clusters, 1))
            colors = [cmap(l) if l >= 0 else (0.5, 0.5, 0.5, 0.5) for l in labels]
        else:
            colors = [(0.5, 0.5, 0.5, 0.5)] * len(labels)

        # 1. Scatter: Frequency vs Power (size = bandwidth)
        ax1 = axes[0, 0]
        sizes = 20 + 100 * (features[:, 2] / (features[:, 2].max() + 1e-6))
        ax1.scatter(features[:, 0], features[:, 1], c=colors, s=sizes, alpha=0.7, edgecolors='k', linewidths=0.5)
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Power (log10)')
        ax1.set_title('Frequency vs Power (size = bandwidth)')
        ax1.grid(True, alpha=0.3)

        # Add cluster center annotations
        if cluster_stats:
            for stats in cluster_stats:
                ax1.axvline(stats['center_freq_hz'], color='red', linestyle='--', alpha=0.5, linewidth=0.8)

        # 2. Histogram: Frequency distribution
        ax2 = axes[0, 1]
        freq_bins = np.linspace(self.freq_range[0], self.freq_range[1], 50)

        for label in unique_labels:
            if label < 0:
                continue
            mask = labels == label
            ax2.hist(features[mask, 0], bins=freq_bins, alpha=0.6,
                     label=f'Cluster {label}', color=cmap(label))

        # Show noise points if any
        noise_mask = labels == -1
        if noise_mask.any():
            ax2.hist(features[noise_mask, 0], bins=freq_bins, alpha=0.4,
                     label='Noise', color='gray')

        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Count')
        ax2.set_title('Frequency Distribution by Cluster')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)

        # 3. Scatter: Frequency vs Bandwidth
        ax3 = axes[1, 0]
        ax3.scatter(features[:, 0], features[:, 2], c=colors, s=50, alpha=0.7, edgecolors='k', linewidths=0.5)
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Bandwidth (Hz)')
        ax3.set_title('Frequency vs Bandwidth')
        ax3.grid(True, alpha=0.3)

        # 4. Cluster summary table
        ax4 = axes[1, 1]
        ax4.axis('off')

        if cluster_stats:
            table_data = []
            for stats in cluster_stats:
                table_data.append([
                    f"C{stats['cluster_id']}",
                    f"{stats['center_freq_hz']:.1f}",
                    f"{stats['std_freq_hz']:.2f}",
                    f"{stats['n_peaks']}",
                    f"{stats['n_sessions']}",
                    f"{stats['mean_power']:.2f}"
                ])

            table = ax4.table(
                cellText=table_data,
                colLabels=['Cluster', 'Center\n(Hz)', 'Std\n(Hz)', 'N Peaks', 'N Sess', 'Mean\nPower'],
                loc='center',
                cellLoc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            ax4.set_title('Cluster Summary', fontsize=11, fontweight='bold', pad=20)
        else:
            ax4.text(0.5, 0.5, f'Total peaks: {len(self._peaks)}\n'
                              f'Sessions: {self.n_sessions}\n'
                              f'Clusters: {n_clusters}',
                     ha='center', va='center', fontsize=12,
                     transform=ax4.transAxes)

        plt.tight_layout()

        if out_path:
            fig.savefig(out_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def cluster_and_plot(self,
                         method: str = 'auto',
                         n_clusters: Union[int, str] = 'auto',
                         title: str = None,
                         out_dir: str = None,
                         session_name: str = None,
                         show: bool = True,
                         **kwargs) -> Tuple[Dict, plt.Figure]:
        """Convenience method: cluster and visualize in one call.

        Parameters
        ----------
        method : str
            Clustering method.
        n_clusters : int or 'auto'
            Number of clusters.
        title : str or None
            Plot title. Auto-generated if None.
        out_dir : str or None
            Output directory for saving files.
        session_name : str or None
            Session name for file naming.
        show : bool
            Whether to display figures.
        **kwargs
            Additional parameters for clustering.

        Returns
        -------
        tuple of (dict, Figure)
            Clustering results and visualization figure.
        """
        import os

        # Run clustering
        results = self.cluster(method=method, n_clusters=n_clusters, **kwargs)

        # Generate title
        if title is None:
            if session_name:
                title = f'Non-SR Peak Clusters: {session_name}'
            else:
                title = f'Non-SR Peak Clusters ({self.n_peaks} peaks, {results["n_clusters"]} clusters)'

        # Determine output path
        out_path = None
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            prefix = f'{session_name}_' if session_name else ''
            out_path = os.path.join(out_dir, f'{prefix}non_sr_clusters.png')

        # Generate plot
        fig = self.plot_clusters(
            labels=results['labels'],
            cluster_stats=results['cluster_stats'],
            title=title,
            out_path=out_path,
            show=show
        )

        return results, fig

    def export_results(self,
                       cluster_results: Dict,
                       out_dir: str,
                       session_name: str = None) -> Dict[str, str]:
        """Export clustering results to files.

        Parameters
        ----------
        cluster_results : dict
            Results from cluster() method.
        out_dir : str
            Output directory.
        session_name : str or None
            Session name for file naming.

        Returns
        -------
        dict
            Mapping of output type to file path.
        """
        import os
        os.makedirs(out_dir, exist_ok=True)

        prefix = f'{session_name}_' if session_name else ''
        output_files = {}

        # Export peaks CSV
        peaks_df = self.to_dataframe()
        peaks_path = os.path.join(out_dir, f'{prefix}non_sr_peaks.csv')
        peaks_df.to_csv(peaks_path, index=False)
        output_files['peaks_csv'] = peaks_path

        # Export cluster centers CSV
        if cluster_results['cluster_stats']:
            centers_df = pd.DataFrame(cluster_results['cluster_stats'])
            centers_path = os.path.join(out_dir, f'{prefix}non_sr_cluster_centers.csv')
            centers_df.to_csv(centers_path, index=False)
            output_files['centers_csv'] = centers_path

        return output_files


# =============================================================================
# Cross-Session Collector
# =============================================================================

class CrossSessionNonSRCollector:
    """Aggregates non-SR peaks across all sessions for grand clustering.

    This class is designed for use in batch processing scripts to aggregate
    peaks from multiple sessions and perform cross-session clustering.

    Examples
    --------
    >>> grand_collector = CrossSessionNonSRCollector()
    >>> for session_name, result in session_results.items():
    ...     grand_collector.add_session_results(session_name, result)
    >>> grand_results = grand_collector.run_grand_clustering()
    >>> grand_collector.export_summary(out_dir='exports')
    """

    def __init__(self,
                 freq_range: Tuple[float, float] = (1.0, 50.0),
                 min_power: Optional[float] = None):
        self._collector = NonSRPeakCollector(freq_range=freq_range, min_power=min_power)
        self._session_results: Dict[str, Dict] = {}

    @property
    def n_peaks(self) -> int:
        """Total number of peaks across all sessions."""
        return self._collector.n_peaks

    @property
    def n_sessions(self) -> int:
        """Number of sessions."""
        return len(self._session_results)

    @property
    def session_ids(self) -> List[str]:
        """List of session IDs."""
        return list(self._session_results.keys())

    def add_session_results(self, session_name: str, result: Dict) -> int:
        """Add results from a single session's analysis.

        Parameters
        ----------
        session_name : str
            Session identifier.
        result : dict
            Result dictionary from analyze_ignition_custom.main(),
            expected to contain 'non_sr_peaks' key.

        Returns
        -------
        int
            Number of peaks added from this session.
        """
        if 'non_sr_peaks' not in result or not result['non_sr_peaks']:
            self._session_results[session_name] = {'n_peaks': 0}
            return 0

        peaks = result['non_sr_peaks']
        count = self._collector.add_precomputed(peaks)

        self._session_results[session_name] = {
            'n_peaks': count,
            'cluster_results': result.get('non_sr_cluster_results')
        }

        return count

    def add_peaks_from_collector(self, session_collector: NonSRPeakCollector) -> int:
        """Add peaks from a per-session collector.

        Parameters
        ----------
        session_collector : NonSRPeakCollector
            Collector from a single session.

        Returns
        -------
        int
            Number of peaks added.
        """
        return self._collector.add_precomputed(session_collector.peaks)

    def run_grand_clustering(self,
                             method: str = 'kmeans',
                             n_clusters: Union[int, str] = 'auto',
                             **kwargs) -> Dict:
        """Run clustering on aggregated peaks from all sessions.

        Parameters
        ----------
        method : str
            Clustering method.
        n_clusters : int or 'auto'
            Number of clusters.
        **kwargs
            Additional clustering parameters.

        Returns
        -------
        dict
            Clustering results.
        """
        return self._collector.cluster(method=method, n_clusters=n_clusters, **kwargs)

    def plot_grand_clusters(self,
                            cluster_results: Dict,
                            title: str = 'Cross-Session Non-SR Peak Clusters',
                            out_path: str = None,
                            show: bool = True) -> plt.Figure:
        """Plot grand clustering results.

        Parameters
        ----------
        cluster_results : dict
            Results from run_grand_clustering().
        title : str
            Plot title.
        out_path : str or None
            Path to save figure.
        show : bool
            Whether to display figure.

        Returns
        -------
        plt.Figure
            The generated figure.
        """
        return self._collector.plot_clusters(
            labels=cluster_results['labels'],
            cluster_stats=cluster_results['cluster_stats'],
            title=title,
            out_path=out_path,
            show=show
        )

    def export_summary(self,
                       cluster_results: Dict,
                       out_dir: str) -> Dict[str, str]:
        """Export cross-session summary to files.

        Parameters
        ----------
        cluster_results : dict
            Results from run_grand_clustering().
        out_dir : str
            Output directory.

        Returns
        -------
        dict
            Mapping of output type to file path.
        """
        import os
        os.makedirs(out_dir, exist_ok=True)

        output_files = {}

        # Export all peaks CSV
        peaks_df = self._collector.to_dataframe()
        peaks_path = os.path.join(out_dir, 'all_non_sr_peaks.csv')
        peaks_df.to_csv(peaks_path, index=False)
        output_files['all_peaks_csv'] = peaks_path

        # Export cluster centers CSV
        if cluster_results['cluster_stats']:
            centers_df = pd.DataFrame(cluster_results['cluster_stats'])
            centers_path = os.path.join(out_dir, 'non_sr_cluster_centers.csv')
            centers_df.to_csv(centers_path, index=False)
            output_files['centers_csv'] = centers_path

        # Export markdown summary
        summary_lines = [
            '# Cross-Session Non-SR Peak Clustering Summary\n',
            f'**Total Sessions:** {self.n_sessions}\n',
            f'**Total Peaks:** {self.n_peaks}\n',
            f'**Clusters Found:** {cluster_results["n_clusters"]}\n',
            '\n## Cluster Centers\n',
        ]

        if cluster_results['cluster_stats']:
            summary_lines.append('| Cluster | Center (Hz) | Std (Hz) | N Peaks | Sessions | Prevalence |')
            summary_lines.append('|---------|------------|----------|---------|----------|------------|')
            for stats in cluster_results['cluster_stats']:
                summary_lines.append(
                    f"| {stats['cluster_id']} | {stats['center_freq_hz']:.2f} | "
                    f"{stats['std_freq_hz']:.2f} | {stats['n_peaks']} | "
                    f"{stats['n_sessions']} | {stats['session_prevalence']:.1%} |"
                )

        summary_lines.append('\n## Per-Session Statistics\n')
        summary_lines.append('| Session | N Peaks |')
        summary_lines.append('|---------|---------|')
        for session_id, info in sorted(self._session_results.items()):
            summary_lines.append(f"| {session_id} | {info['n_peaks']} |")

        summary_path = os.path.join(out_dir, 'non_sr_cross_session_summary.md')
        with open(summary_path, 'w') as f:
            f.write('\n'.join(summary_lines))
        output_files['summary_md'] = summary_path

        return output_files


# =============================================================================
# Utility Functions
# =============================================================================

def compute_cluster_distances_to_sr(cluster_stats: List[Dict],
                                    sr_harmonics: Tuple[float, ...] = (7.83, 14.3, 20.8, 27.3, 33.8)) -> List[Dict]:
    """Compute distance from each cluster center to nearest SR harmonic.

    Parameters
    ----------
    cluster_stats : list of dict
        Cluster statistics from clustering results.
    sr_harmonics : tuple of float
        Schumann Resonance harmonic frequencies.

    Returns
    -------
    list of dict
        Updated cluster stats with 'nearest_sr_hz' and 'distance_from_sr_hz'.
    """
    for stats in cluster_stats:
        center = stats['center_freq_hz']
        distances = [abs(center - h) for h in sr_harmonics]
        min_idx = np.argmin(distances)
        stats['nearest_sr_hz'] = sr_harmonics[min_idx]
        stats['distance_from_sr_hz'] = distances[min_idx]

    return cluster_stats

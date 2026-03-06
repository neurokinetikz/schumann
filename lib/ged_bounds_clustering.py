"""
gedBounds-Inspired Boundary Detection from GED Peaks
====================================================

Identifies empirical frequency band boundaries using the distribution
of GED peaks. Following the gedBounds philosophy (Cohen 2021) of finding
boundaries where spatial structure changes.

Approach:
1. Compute peak density across frequency (KDE)
2. Find local minima (depleted regions = boundaries)
3. Cluster peaks by frequency to find natural groupings
4. Compare discovered boundaries with phi^n predictions

The key insight: band boundaries should have fewer peaks (depletion),
so density minima naturally indicate boundaries without needing raw EEG.

Author: Generated for schumann project
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats, signal
from scipy.ndimage import gaussian_filter1d
import warnings

# Attempt sklearn imports
try:
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("sklearn not available - some clustering methods disabled")

# Import phi constants from existing module
try:
    from phi_frequency_model import PHI, F0
except ImportError:
    try:
        from lib.phi_frequency_model import PHI, F0
    except ImportError:
        PHI = 1.6180339887
        F0 = 7.60

# Default phi^n boundary frequencies (integer n positions)
PHI_BOUNDARIES_DEFAULT = [
    F0 * (PHI ** 0),   # 7.60 Hz (theta/alpha)
    F0 * (PHI ** 1),   # 12.30 Hz (alpha/beta_low)
    F0 * (PHI ** 2),   # 19.90 Hz (beta_low/beta_high)
    F0 * (PHI ** 3),   # 32.19 Hz (beta_high/gamma)
]


# =============================================================================
# Peak Density Analysis
# =============================================================================

def compute_peak_density(
    peaks_df: pd.DataFrame,
    freq_col: str = 'frequency',
    freq_range: Tuple[float, float] = (4.5, 45.0),
    bandwidth: float = 0.3,
    n_points: int = 500
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute kernel density estimate of peak frequencies.

    Parameters
    ----------
    peaks_df : pd.DataFrame
        DataFrame with peak frequencies
    freq_col : str
        Column name for frequency values
    freq_range : tuple
        (min, max) frequency range for density estimation
    bandwidth : float
        KDE bandwidth in Hz (Scott's rule used if None)
    n_points : int
        Number of evaluation points

    Returns
    -------
    freqs : np.ndarray
        Array of evaluation frequencies
    density : np.ndarray
        Array of density values (normalized)
    """
    if freq_col not in peaks_df.columns:
        raise ValueError(f"Column '{freq_col}' not found in DataFrame")

    # Extract frequencies within range
    freqs_data = peaks_df[freq_col].values
    mask = (freqs_data >= freq_range[0]) & (freqs_data <= freq_range[1])
    freqs_data = freqs_data[mask]

    if len(freqs_data) < 10:
        warnings.warn(f"Only {len(freqs_data)} peaks in range - density may be unreliable")

    # Create evaluation grid
    freqs_eval = np.linspace(freq_range[0], freq_range[1], n_points)

    # Compute KDE using scipy
    try:
        kde = stats.gaussian_kde(freqs_data, bw_method=bandwidth / np.std(freqs_data))
        density = kde(freqs_eval)
    except (np.linalg.LinAlgError, ValueError) as e:
        # Fallback to histogram-based density
        warnings.warn(f"KDE failed ({e}), using histogram fallback")
        hist, bin_edges = np.histogram(freqs_data, bins=n_points, range=freq_range, density=True)
        freqs_eval = (bin_edges[:-1] + bin_edges[1:]) / 2
        density = hist

    # Normalize
    density = density / np.max(density)

    return freqs_eval, density


def find_density_minima(
    freqs: np.ndarray,
    density: np.ndarray,
    prominence_threshold: float = 0.05,
    min_distance_hz: float = 2.0,
    smooth_sigma: float = 1.0
) -> List[float]:
    """
    Find local minima in peak density (candidate boundaries).

    Minima in density indicate frequencies where peaks are depleted,
    consistent with band boundary positions.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency evaluation points
    density : np.ndarray
        Density values at each frequency
    prominence_threshold : float
        Minimum prominence for a minimum to be detected (as fraction of range)
    min_distance_hz : float
        Minimum distance between detected minima (Hz)
    smooth_sigma : float
        Gaussian smoothing sigma (in index units) before finding minima

    Returns
    -------
    boundaries : List[float]
        List of boundary frequencies (Hz) at density minima
    """
    # Smooth density to reduce noise
    if smooth_sigma > 0:
        density_smooth = gaussian_filter1d(density, sigma=smooth_sigma)
    else:
        density_smooth = density

    # Find minima by finding maxima of negative density
    freq_step = freqs[1] - freqs[0] if len(freqs) > 1 else 0.1
    min_distance_samples = max(1, int(min_distance_hz / freq_step))

    # Invert to find minima as peaks
    neg_density = -density_smooth

    # Find peaks (minima in original)
    prominence = prominence_threshold * (np.max(density_smooth) - np.min(density_smooth))
    peak_indices, properties = signal.find_peaks(
        neg_density,
        distance=min_distance_samples,
        prominence=prominence
    )

    # Extract boundary frequencies
    boundaries = [float(freqs[idx]) for idx in peak_indices]

    return sorted(boundaries)


# =============================================================================
# Frequency Clustering
# =============================================================================

def cluster_peaks_by_frequency(
    peaks_df: pd.DataFrame,
    freq_col: str = 'frequency',
    method: str = 'gmm',
    n_clusters: Union[int, str] = 'auto',
    freq_range: Tuple[float, float] = (4.5, 45.0),
    max_clusters: int = 10
) -> Tuple[np.ndarray, List[float]]:
    """
    Cluster peaks by frequency to find natural band groupings.

    Parameters
    ----------
    peaks_df : pd.DataFrame
        DataFrame with peak frequencies
    freq_col : str
        Column name for frequency values
    method : str
        Clustering method: 'gmm', 'kmeans', 'agglomerative'
    n_clusters : int or 'auto'
        Number of clusters. If 'auto', determined by BIC (GMM) or silhouette.
    freq_range : tuple
        Frequency range to include
    max_clusters : int
        Maximum clusters to test when n_clusters='auto'

    Returns
    -------
    labels : np.ndarray
        Cluster assignment for each peak
    boundaries : List[float]
        Frequencies at cluster transitions (midpoints between adjacent cluster means)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn required for clustering methods")

    # Extract frequencies
    freqs = peaks_df[freq_col].values
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    freqs_filtered = freqs[mask].reshape(-1, 1)

    if len(freqs_filtered) < 50:
        warnings.warn(f"Only {len(freqs_filtered)} peaks - clustering may be unreliable")

    # Determine optimal number of clusters if auto
    if n_clusters == 'auto':
        n_clusters = _find_optimal_clusters(freqs_filtered, method, max_clusters)

    # Run clustering
    if method == 'gmm':
        model = GaussianMixture(n_components=n_clusters, random_state=42)
        labels = model.fit_predict(freqs_filtered)
        centers = model.means_.flatten()
    elif method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(freqs_filtered)
        centers = model.cluster_centers_.flatten()
    elif method == 'agglomerative':
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(freqs_filtered)
        # Compute centers manually
        centers = np.array([freqs_filtered[labels == i].mean() for i in range(n_clusters)])
    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute boundaries as midpoints between sorted cluster centers
    centers_sorted = np.sort(centers)
    boundaries = []
    for i in range(len(centers_sorted) - 1):
        boundary = (centers_sorted[i] + centers_sorted[i + 1]) / 2
        boundaries.append(float(boundary))

    # Pad labels back to original length (NaN for out-of-range)
    full_labels = np.full(len(freqs), -1)
    full_labels[mask] = labels

    return full_labels, boundaries


def _find_optimal_clusters(
    X: np.ndarray,
    method: str,
    max_clusters: int
) -> int:
    """Find optimal number of clusters using BIC or silhouette."""
    if len(X) < max_clusters * 5:
        max_clusters = max(2, len(X) // 5)

    if method == 'gmm':
        # Use BIC for GMM
        bics = []
        for k in range(2, max_clusters + 1):
            try:
                gmm = GaussianMixture(n_components=k, random_state=42)
                gmm.fit(X)
                bics.append(gmm.bic(X))
            except Exception:
                bics.append(np.inf)
        optimal = np.argmin(bics) + 2
    else:
        # Use silhouette score
        scores = []
        for k in range(2, max_clusters + 1):
            try:
                if method == 'kmeans':
                    model = KMeans(n_clusters=k, random_state=42, n_init=10)
                else:
                    model = AgglomerativeClustering(n_clusters=k)
                labels = model.fit_predict(X)
                score = silhouette_score(X, labels)
                scores.append(score)
            except Exception:
                scores.append(-1)
        optimal = np.argmax(scores) + 2

    return int(optimal)


# =============================================================================
# Phi Validation
# =============================================================================

def validate_boundaries_vs_phi(
    empirical_boundaries: List[float],
    phi_boundaries: Optional[List[float]] = None,
    freq_range: Tuple[float, float] = (4.5, 45.0),
    n_permutations: int = 10000,
    tolerance_hz: float = 0.5
) -> Dict:
    """
    Compare empirically detected boundaries with phi^n predictions.

    Statistical test: Are empirical boundaries closer to phi^n positions
    than random boundaries in the same frequency range?

    Parameters
    ----------
    empirical_boundaries : List[float]
        Detected boundary frequencies (Hz)
    phi_boundaries : List[float] or None
        phi^n boundary predictions. Default: [7.60, 12.30, 19.90, 32.19]
    freq_range : tuple
        Frequency range for random boundary generation
    n_permutations : int
        Number of permutations for significance test
    tolerance_hz : float
        Maximum distance to count as "matched"

    Returns
    -------
    dict with:
    - n_empirical: number of empirical boundaries
    - n_phi: number of phi boundaries in range
    - n_matched: boundaries within tolerance of phi^n
    - mean_distance_hz: average distance to nearest phi^n
    - min_distances: list of distances for each empirical boundary
    - p_value: permutation test p-value
    - effect_size: Cohen's d comparing to random
    - nearest_phi: list of nearest phi boundary for each empirical
    """
    if phi_boundaries is None:
        phi_boundaries = PHI_BOUNDARIES_DEFAULT

    # Filter to frequency range
    phi_in_range = [f for f in phi_boundaries if freq_range[0] <= f <= freq_range[1]]
    emp_in_range = [f for f in empirical_boundaries if freq_range[0] <= f <= freq_range[1]]

    if len(emp_in_range) == 0:
        return {
            'n_empirical': 0,
            'n_phi': len(phi_in_range),
            'n_matched': 0,
            'mean_distance_hz': np.nan,
            'min_distances': [],
            'p_value': 1.0,
            'effect_size': 0.0,
            'nearest_phi': []
        }

    # Compute distances to nearest phi boundary
    min_distances = []
    nearest_phi = []
    for emp in emp_in_range:
        distances = [abs(emp - phi) for phi in phi_in_range]
        min_dist = min(distances) if distances else np.inf
        min_distances.append(min_dist)
        if distances:
            nearest_phi.append(phi_in_range[np.argmin(distances)])
        else:
            nearest_phi.append(np.nan)

    mean_distance = np.mean(min_distances)
    n_matched = sum(1 for d in min_distances if d <= tolerance_hz)

    # Permutation test: generate random boundaries and compute their distances
    random_distances = []
    for _ in range(n_permutations):
        # Generate same number of random boundaries
        random_bounds = np.random.uniform(freq_range[0], freq_range[1], len(emp_in_range))
        rand_min_dists = []
        for rb in random_bounds:
            distances = [abs(rb - phi) for phi in phi_in_range]
            rand_min_dists.append(min(distances) if distances else np.inf)
        random_distances.append(np.mean(rand_min_dists))

    random_distances = np.array(random_distances)

    # P-value: proportion of random samples with smaller mean distance
    p_value = np.mean(random_distances <= mean_distance)

    # Effect size: Cohen's d
    if np.std(random_distances) > 0:
        effect_size = (np.mean(random_distances) - mean_distance) / np.std(random_distances)
    else:
        effect_size = 0.0

    return {
        'n_empirical': len(emp_in_range),
        'n_phi': len(phi_in_range),
        'n_matched': n_matched,
        'mean_distance_hz': float(mean_distance),
        'min_distances': min_distances,
        'p_value': float(p_value),
        'effect_size': float(effect_size),
        'nearest_phi': nearest_phi,
        'random_mean_distance': float(np.mean(random_distances)),
        'random_std_distance': float(np.std(random_distances))
    }


# =============================================================================
# Pipeline Wrapper
# =============================================================================

def run_boundary_detection_pipeline(
    peaks_df: pd.DataFrame,
    freq_col: str = 'frequency',
    methods: List[str] = None,
    freq_range: Tuple[float, float] = (4.5, 45.0),
    density_bandwidth: float = 0.3,
    density_prominence: float = 0.05,
    clustering_max_k: int = 8
) -> pd.DataFrame:
    """
    Run multiple boundary detection methods and combine results.

    Parameters
    ----------
    peaks_df : pd.DataFrame
        DataFrame with peak frequencies
    freq_col : str
        Column name for frequency
    methods : List[str] or None
        Methods to run. Default: ['density', 'gmm', 'agglomerative']
    freq_range : tuple
        Frequency range for analysis
    density_bandwidth : float
        KDE bandwidth for density method
    density_prominence : float
        Prominence threshold for density minima
    clustering_max_k : int
        Maximum clusters for auto selection

    Returns
    -------
    pd.DataFrame with columns:
    - frequency: boundary frequency
    - method: detection method
    - confidence: how many methods agree (within 1 Hz)
    - nearest_phi: nearest phi^n boundary
    - distance_to_phi: distance in Hz
    """
    if methods is None:
        methods = ['density']
        if SKLEARN_AVAILABLE:
            methods.extend(['gmm', 'agglomerative'])

    all_boundaries = []

    # Run each method
    for method in methods:
        try:
            if method == 'density':
                freqs, density = compute_peak_density(
                    peaks_df, freq_col, freq_range, bandwidth=density_bandwidth
                )
                boundaries = find_density_minima(
                    freqs, density, prominence_threshold=density_prominence
                )
            elif method in ['gmm', 'kmeans', 'agglomerative']:
                _, boundaries = cluster_peaks_by_frequency(
                    peaks_df, freq_col, method=method,
                    n_clusters='auto', freq_range=freq_range,
                    max_clusters=clustering_max_k
                )
            else:
                warnings.warn(f"Unknown method: {method}")
                continue

            for b in boundaries:
                all_boundaries.append({'frequency': b, 'method': method})

        except Exception as e:
            warnings.warn(f"Method {method} failed: {e}")

    if not all_boundaries:
        return pd.DataFrame(columns=['frequency', 'method', 'confidence', 'nearest_phi', 'distance_to_phi'])

    df = pd.DataFrame(all_boundaries)

    # Compute confidence (how many methods agree within 1 Hz)
    def count_agreements(freq, tolerance=1.0):
        return sum(1 for _, row in df.iterrows() if abs(row['frequency'] - freq) <= tolerance)

    df['confidence'] = df['frequency'].apply(count_agreements)

    # Find nearest phi boundary
    phi_boundaries = PHI_BOUNDARIES_DEFAULT
    def find_nearest_phi(freq):
        distances = [(phi, abs(freq - phi)) for phi in phi_boundaries]
        nearest = min(distances, key=lambda x: x[1])
        return nearest[0], nearest[1]

    df['nearest_phi'] = df['frequency'].apply(lambda f: find_nearest_phi(f)[0])
    df['distance_to_phi'] = df['frequency'].apply(lambda f: find_nearest_phi(f)[1])

    # Sort by frequency
    df = df.sort_values('frequency').reset_index(drop=True)

    return df


def get_consensus_boundaries(
    results_df: pd.DataFrame,
    min_confidence: int = 2,
    merge_tolerance_hz: float = 1.0
) -> List[float]:
    """
    Extract consensus boundaries that multiple methods agree on.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output from run_boundary_detection_pipeline
    min_confidence : int
        Minimum number of methods that must agree
    merge_tolerance_hz : float
        Merge boundaries within this distance

    Returns
    -------
    List[float] : Consensus boundary frequencies
    """
    # Filter by confidence
    confident = results_df[results_df['confidence'] >= min_confidence]

    if confident.empty:
        return []

    # Get unique boundaries by merging close ones
    boundaries = sorted(confident['frequency'].unique())
    merged = []

    i = 0
    while i < len(boundaries):
        group = [boundaries[i]]
        j = i + 1
        while j < len(boundaries) and boundaries[j] - group[0] <= merge_tolerance_hz:
            group.append(boundaries[j])
            j += 1
        merged.append(np.mean(group))
        i = j

    return merged


# =============================================================================
# Visualization
# =============================================================================

def plot_boundary_detection(
    peaks_df: pd.DataFrame,
    results_df: pd.DataFrame,
    freq_col: str = 'frequency',
    freq_range: Tuple[float, float] = (4.5, 45.0),
    density_bandwidth: float = 0.3,
    figsize: Tuple[float, float] = (14, 8),
    save_path: Optional[str] = None,
    show: bool = True
) -> 'plt.Figure':
    """
    Create visualization of boundary detection results.

    Parameters
    ----------
    peaks_df : pd.DataFrame
        Original peaks DataFrame
    results_df : pd.DataFrame
        Output from run_boundary_detection_pipeline
    freq_col : str
        Column name for frequency
    freq_range : tuple
        Frequency range for plotting
    density_bandwidth : float
        KDE bandwidth
    figsize : tuple
        Figure size
    save_path : str or None
        Path to save figure
    show : bool
        Whether to display figure

    Returns
    -------
    plt.Figure
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Panel A: Peak histogram with phi boundaries
    ax = axes[0, 0]
    freqs = peaks_df[freq_col].values
    freqs_in_range = freqs[(freqs >= freq_range[0]) & (freqs <= freq_range[1])]
    ax.hist(freqs_in_range, bins=80, alpha=0.7, color='steelblue', edgecolor='white')
    for phi in PHI_BOUNDARIES_DEFAULT:
        if freq_range[0] <= phi <= freq_range[1]:
            ax.axvline(phi, color='red', linestyle='--', linewidth=2, alpha=0.7,
                      label=f'$\\phi^n$ = {phi:.2f} Hz' if phi == PHI_BOUNDARIES_DEFAULT[0] else '')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Peak Count')
    ax.set_title('A) Peak Distribution with $\\phi^n$ Boundaries')
    ax.legend(loc='upper right')

    # Panel B: KDE density with detected minima
    ax = axes[0, 1]
    f_eval, density = compute_peak_density(peaks_df, freq_col, freq_range, density_bandwidth)
    ax.plot(f_eval, density, 'b-', linewidth=2, label='KDE Density')

    # Mark density minima
    density_bounds = results_df[results_df['method'] == 'density']['frequency'].values
    for b in density_bounds:
        idx = np.argmin(np.abs(f_eval - b))
        ax.plot(b, density[idx], 'go', markersize=10, markeredgecolor='darkgreen',
               markeredgewidth=2, label='Detected Boundary' if b == density_bounds[0] else '')

    # Mark phi boundaries
    for phi in PHI_BOUNDARIES_DEFAULT:
        if freq_range[0] <= phi <= freq_range[1]:
            ax.axvline(phi, color='red', linestyle='--', alpha=0.5)

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Normalized Density')
    ax.set_title('B) KDE Density with Detected Minima')
    ax.legend(loc='upper right')

    # Panel C: Method comparison
    ax = axes[1, 0]
    methods_present = results_df['method'].unique()
    y_positions = {m: i for i, m in enumerate(methods_present)}

    for _, row in results_df.iterrows():
        y = y_positions[row['method']]
        ax.scatter(row['frequency'], y, s=100, c='steelblue', alpha=0.7)

    # Phi boundaries
    for phi in PHI_BOUNDARIES_DEFAULT:
        if freq_range[0] <= phi <= freq_range[1]:
            ax.axvline(phi, color='red', linestyle='--', alpha=0.5)

    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(list(y_positions.keys()))
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Method')
    ax.set_title('C) Boundaries by Method')
    ax.set_xlim(freq_range)

    # Panel D: Distance to phi comparison
    ax = axes[1, 1]

    # Get consensus boundaries
    consensus = get_consensus_boundaries(results_df, min_confidence=1)

    if consensus:
        # Compute distances
        distances = []
        for b in consensus:
            min_dist = min(abs(b - phi) for phi in PHI_BOUNDARIES_DEFAULT)
            distances.append(min_dist)

        ax.bar(range(len(consensus)), distances, color='steelblue', alpha=0.7)
        ax.axhline(0.5, color='red', linestyle='--', label='0.5 Hz tolerance')
        ax.set_xticks(range(len(consensus)))
        ax.set_xticklabels([f'{b:.1f}' for b in consensus], rotation=45)
        ax.set_xlabel('Detected Boundary (Hz)')
        ax.set_ylabel('Distance to Nearest $\\phi^n$ (Hz)')
        ax.set_title('D) Alignment with $\\phi^n$ Predictions')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No boundaries detected', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('D) Alignment with $\\phi^n$ Predictions')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# =============================================================================
# Summary Report
# =============================================================================

def generate_boundary_report(
    peaks_df: pd.DataFrame,
    freq_col: str = 'frequency',
    freq_range: Tuple[float, float] = (4.5, 45.0),
    methods: List[str] = None
) -> Dict:
    """
    Generate comprehensive boundary detection report.

    Returns
    -------
    dict with:
    - n_peaks: total peaks analyzed
    - detected_boundaries: DataFrame of all boundaries
    - consensus_boundaries: list of high-confidence boundaries
    - phi_validation: validation statistics
    - summary_text: human-readable summary
    """
    # Run pipeline
    results = run_boundary_detection_pipeline(
        peaks_df, freq_col, methods, freq_range
    )

    # Get consensus
    consensus = get_consensus_boundaries(results, min_confidence=2)
    if not consensus:
        consensus = get_consensus_boundaries(results, min_confidence=1)

    # Validate vs phi
    validation = validate_boundaries_vs_phi(consensus, freq_range=freq_range)

    # Generate summary text
    summary_lines = [
        "=" * 60,
        "gedBounds Empirical Boundary Detection Report",
        "=" * 60,
        f"Total peaks analyzed: {len(peaks_df):,}",
        f"Frequency range: {freq_range[0]:.1f} - {freq_range[1]:.1f} Hz",
        f"Methods used: {', '.join(results['method'].unique())}",
        "",
        "Detected Boundaries:",
        "-" * 40,
    ]

    for b in consensus:
        nearest = min(PHI_BOUNDARIES_DEFAULT, key=lambda x: abs(x - b))
        dist = abs(b - nearest)
        summary_lines.append(f"  {b:.2f} Hz (nearest phi^n: {nearest:.2f} Hz, distance: {dist:.2f} Hz)")

    summary_lines.extend([
        "",
        "Phi^n Validation:",
        "-" * 40,
        f"  Boundaries matched (within 0.5 Hz): {validation['n_matched']}/{validation['n_empirical']}",
        f"  Mean distance to phi^n: {validation['mean_distance_hz']:.3f} Hz",
        f"  Random expectation: {validation['random_mean_distance']:.3f} +/- {validation['random_std_distance']:.3f} Hz",
        f"  Effect size (Cohen's d): {validation['effect_size']:.2f}",
        f"  P-value: {validation['p_value']:.4f}",
        "",
        "Conclusion:",
        "-" * 40,
    ])

    if validation['p_value'] < 0.05 and validation['effect_size'] > 0.5:
        summary_lines.append("  Empirical boundaries show SIGNIFICANT alignment with phi^n predictions.")
    elif validation['p_value'] < 0.05:
        summary_lines.append("  Empirical boundaries show moderate alignment with phi^n predictions.")
    else:
        summary_lines.append("  No significant alignment with phi^n predictions detected.")

    summary_lines.append("=" * 60)

    return {
        'n_peaks': len(peaks_df),
        'detected_boundaries': results,
        'consensus_boundaries': consensus,
        'phi_validation': validation,
        'summary_text': '\n'.join(summary_lines)
    }

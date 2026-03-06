"""
GED Poster Figure: 4-Panel Boundary vs Attractor Comparison
============================================================

Conference poster-ready visualization showing the dual-property framework:
- Boundaries excel at spatial precision (clustering ratio)
- Attractors excel at temporal stability (Q-factor)

Usage:
    from lib.ged_poster_figure import generate_boundary_attractor_figure
    fig = generate_boundary_attractor_figure(show=True)
"""

from __future__ import annotations
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from pathlib import Path


# =============================================================================
# VISUAL CONSTANTS
# =============================================================================

COLORS = {
    'boundary': '#e74c3c',       # Red (warm)
    'attractor': '#2ecc71',      # Green (cool)
    'boundary_light': '#f5b7b1', # Light red for error bars
    'attractor_light': '#abebc6',# Light green for error bars
    'winner_star': '#f39c12',    # Gold for winner indicator
    'equal': '#7f8c8d',          # Gray for equal bars
    'text': '#2c3e50',           # Dark text
    'grid': '#ecf0f1',           # Light grid
    'background': '#fafafa'      # Off-white background
}

FONTS = {
    'suptitle': 20,       # Overall figure title
    'panel_title': 16,    # Panel titles (A, B, C, D)
    'question': 13,       # Panel subtitles ("Where do peaks snap to?")
    'metric_value': 18,   # Big numbers on bars
    'bar_label': 12,      # "Boundary", "Attractor"
    'interpretation': 11, # "Hard frequency walls"
    'stats': 10,          # p-values, effect sizes
    'footer': 11          # Summary footer
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class BoundaryAttractorMetrics:
    """Aggregated metrics for boundary vs attractor comparison."""

    # Clustering ratio (from peak_distribution_analysis)
    clustering_boundary: float = 0.0
    clustering_attractor: float = 0.0
    clustering_boundary_std: float = 0.0
    clustering_attractor_std: float = 0.0
    clustering_n_boundary: int = 0
    clustering_n_attractor: int = 0
    clustering_pvalue: float = 1.0
    clustering_cohens_d: float = 0.0

    # Q-factor (from GED results)
    q_boundary: float = 0.0
    q_attractor: float = 0.0
    q_boundary_std: float = 0.0
    q_attractor_std: float = 0.0
    q_n_boundary: int = 0
    q_n_attractor: int = 0
    q_pvalue: float = 1.0
    q_cohens_d: float = 0.0

    # FWHM (from GED results)
    fwhm_boundary: float = 0.0
    fwhm_attractor: float = 0.0
    fwhm_boundary_std: float = 0.0
    fwhm_attractor_std: float = 0.0
    fwhm_n_boundary: int = 0
    fwhm_n_attractor: int = 0
    fwhm_pvalue: float = 1.0
    fwhm_cohens_d: float = 0.0

    # Eigenvalue (from GED results)
    eigenvalue_boundary: float = 0.0
    eigenvalue_attractor: float = 0.0
    eigenvalue_boundary_std: float = 0.0
    eigenvalue_attractor_std: float = 0.0
    eigenvalue_n_boundary: int = 0
    eigenvalue_n_attractor: int = 0
    eigenvalue_pvalue: float = 1.0
    eigenvalue_cohens_d: float = 0.0


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_clustering_data(csv_path: str) -> pd.DataFrame:
    """
    Load pre-computed clustering ratios from peak_distribution_analysis output.

    Parameters
    ----------
    csv_path : str
        Path to position_type_clustering.csv

    Returns
    -------
    pd.DataFrame with columns: band, position_type, n_peaks, mean_distance,
                               expected_random, clustering_ratio, p_value
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Clustering data not found: {csv_path}")
    return pd.read_csv(csv_path)


def load_ged_results(csv_dir: str) -> pd.DataFrame:
    """
    Load and concatenate all GED session results, melting to long format.

    Parameters
    ----------
    csv_dir : str
        Directory containing *_ged_results.csv files

    Returns
    -------
    pd.DataFrame with columns: session_id, harmonic, position_type,
                               freq, fwhm, eigenvalue, q_factor, lambda_ratio
    """
    csv_files = glob.glob(os.path.join(csv_dir, '*_ged_results.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No GED result files found in: {csv_dir}")

    all_rows = []
    harmonics = ['sr1', 'sr1.5', 'sr2', 'sr2.5', 'sr3', 'sr4', 'sr5', 'sr6']

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            session_id = row.get('session_id', os.path.basename(csv_file))
            for h in harmonics:
                type_col = f'{h}_type'
                if type_col not in df.columns:
                    continue
                pos_type = row.get(type_col)
                if pd.isna(pos_type):
                    continue

                all_rows.append({
                    'session_id': session_id,
                    'harmonic': h,
                    'position_type': pos_type,
                    'canonical_freq': row.get(f'{h}_canonical_freq', np.nan),
                    'ged_freq': row.get(f'{h}_ged_freq', np.nan),
                    'fwhm': row.get(f'{h}_fwhm', np.nan),
                    'eigenvalue': row.get(f'{h}_eigenvalue', np.nan),
                    'q_factor': row.get(f'{h}_q_factor', np.nan),
                    'lambda_ratio': row.get(f'{h}_lambda_ratio', np.nan)
                })

    return pd.DataFrame(all_rows)


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def compute_aggregate_metrics(
    clustering_csv: str,
    ged_results_dir: str
) -> BoundaryAttractorMetrics:
    """
    Compute all metrics for the 4-panel figure.

    Parameters
    ----------
    clustering_csv : str
        Path to position_type_clustering.csv
    ged_results_dir : str
        Directory containing GED result CSV files

    Returns
    -------
    BoundaryAttractorMetrics with all fields populated
    """
    metrics = BoundaryAttractorMetrics()

    # ----- CLUSTERING RATIO -----
    try:
        clust_df = load_clustering_data(clustering_csv)
        # Aggregate across bands
        boundary_clust = clust_df[clust_df['position_type'] == 'boundary']['clustering_ratio'].dropna()
        attractor_clust = clust_df[clust_df['position_type'] == 'attractor']['clustering_ratio'].dropna()

        if len(boundary_clust) > 0 and len(attractor_clust) > 0:
            metrics.clustering_boundary = boundary_clust.mean()
            metrics.clustering_attractor = attractor_clust.mean()
            metrics.clustering_boundary_std = boundary_clust.std()
            metrics.clustering_attractor_std = attractor_clust.std()
            metrics.clustering_n_boundary = len(boundary_clust)
            metrics.clustering_n_attractor = len(attractor_clust)

            # Statistical test
            t_stat, p_val = stats.ttest_ind(boundary_clust, attractor_clust)
            metrics.clustering_pvalue = p_val
            metrics.clustering_cohens_d = compute_cohens_d(
                boundary_clust.values, attractor_clust.values
            )
    except Exception as e:
        print(f"Warning: Could not load clustering data: {e}")

    # ----- GED METRICS (Q-factor, FWHM, Eigenvalue) -----
    try:
        ged_df = load_ged_results(ged_results_dir)

        # Filter to boundary and attractor only
        boundary_df = ged_df[ged_df['position_type'] == 'boundary']
        attractor_df = ged_df[ged_df['position_type'] == 'attractor']

        # Q-factor
        b_q = boundary_df['q_factor'].dropna()
        a_q = attractor_df['q_factor'].dropna()
        if len(b_q) > 0 and len(a_q) > 0:
            metrics.q_boundary = b_q.mean()
            metrics.q_attractor = a_q.mean()
            metrics.q_boundary_std = b_q.std()
            metrics.q_attractor_std = a_q.std()
            metrics.q_n_boundary = len(b_q)
            metrics.q_n_attractor = len(a_q)
            _, metrics.q_pvalue = stats.ttest_ind(b_q, a_q)
            metrics.q_cohens_d = compute_cohens_d(b_q.values, a_q.values)

        # FWHM
        b_fwhm = boundary_df['fwhm'].dropna()
        a_fwhm = attractor_df['fwhm'].dropna()
        if len(b_fwhm) > 0 and len(a_fwhm) > 0:
            metrics.fwhm_boundary = b_fwhm.mean()
            metrics.fwhm_attractor = a_fwhm.mean()
            metrics.fwhm_boundary_std = b_fwhm.std()
            metrics.fwhm_attractor_std = a_fwhm.std()
            metrics.fwhm_n_boundary = len(b_fwhm)
            metrics.fwhm_n_attractor = len(a_fwhm)
            _, metrics.fwhm_pvalue = stats.ttest_ind(b_fwhm, a_fwhm)
            metrics.fwhm_cohens_d = compute_cohens_d(b_fwhm.values, a_fwhm.values)

        # Eigenvalue
        b_eig = boundary_df['eigenvalue'].dropna()
        a_eig = attractor_df['eigenvalue'].dropna()
        if len(b_eig) > 0 and len(a_eig) > 0:
            metrics.eigenvalue_boundary = b_eig.mean()
            metrics.eigenvalue_attractor = a_eig.mean()
            metrics.eigenvalue_boundary_std = b_eig.std()
            metrics.eigenvalue_attractor_std = a_eig.std()
            metrics.eigenvalue_n_boundary = len(b_eig)
            metrics.eigenvalue_n_attractor = len(a_eig)
            _, metrics.eigenvalue_pvalue = stats.ttest_ind(b_eig, a_eig)
            metrics.eigenvalue_cohens_d = compute_cohens_d(b_eig.values, a_eig.values)

    except Exception as e:
        print(f"Warning: Could not load GED results: {e}")

    return metrics


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def _format_pvalue(p: float) -> str:
    """Format p-value with significance stars."""
    if p < 0.001:
        return "p < 0.001 ***"
    elif p < 0.01:
        return f"p = {p:.3f} **"
    elif p < 0.05:
        return f"p = {p:.3f} *"
    else:
        return f"p = {p:.3f} n.s."


def _format_cohens_d(d: float) -> str:
    """Format Cohen's d with effect size label."""
    d_abs = abs(d)
    if d_abs >= 0.8:
        label = "large"
    elif d_abs >= 0.5:
        label = "medium"
    elif d_abs >= 0.2:
        label = "small"
    else:
        label = "negligible"
    return f"d = {d:.2f} ({label})"


def _plot_comparison_panel(
    ax: plt.Axes,
    boundary_val: float,
    attractor_val: float,
    boundary_std: float,
    attractor_std: float,
    panel_label: str,
    metric_name: str,
    question: str,
    unit: str,
    winner: str,  # 'boundary' | 'attractor' | 'equal'
    interpretation: str,
    pvalue: float,
    cohens_d: float,
    n_boundary: int = 0,
    n_attractor: int = 0
) -> None:
    """
    Render a single comparison panel with horizontal bars.

    Parameters
    ----------
    ax : matplotlib Axes
    boundary_val, attractor_val : float
        Mean values for each group
    boundary_std, attractor_std : float
        Standard deviations
    panel_label : str
        e.g., "A", "B", "C", "D"
    metric_name : str
        e.g., "CLUSTERING RATIO"
    question : str
        e.g., "Where do peaks snap to?"
    unit : str
        e.g., "", "Hz", "λ"
    winner : str
        'boundary', 'attractor', or 'equal'
    interpretation : str
        e.g., "Hard frequency walls"
    pvalue, cohens_d : float
        Statistical values
    n_boundary, n_attractor : int
        Sample sizes (optional)
    """
    ax.set_facecolor(COLORS['background'])

    # Determine colors based on winner
    if winner == 'equal':
        b_color = COLORS['equal']
        a_color = COLORS['equal']
    else:
        b_color = COLORS['boundary']
        a_color = COLORS['attractor']

    # Bar positions and values
    y_pos = [0.6, 0.2]  # Boundary on top, Attractor below
    values = [boundary_val, attractor_val]
    errors = [boundary_std, attractor_std]
    colors = [b_color, a_color]
    labels = ['Boundary', 'Attractor']

    # Draw horizontal bars
    bars = ax.barh(y_pos, values, height=0.25, color=colors, edgecolor='white', linewidth=1.5)

    # Add error bars
    ax.errorbar(values, y_pos, xerr=errors, fmt='none', ecolor='#555555',
                capsize=5, capthick=1.5, elinewidth=1.5)

    # Add value labels on bars
    for i, (bar, val, std) in enumerate(zip(bars, values, errors)):
        x_pos = val + std + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02
        # Check if value fits inside bar
        if val > (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.3:
            # Inside bar
            ax.text(val * 0.5, y_pos[i], f'{val:.1f}',
                    ha='center', va='center', fontsize=FONTS['metric_value'],
                    fontweight='bold', color='white')
        else:
            # Outside bar
            ax.text(val + std + 0.5, y_pos[i], f'{val:.1f}',
                    ha='left', va='center', fontsize=FONTS['metric_value'],
                    fontweight='bold', color=COLORS['text'])

    # Add winner star
    if winner != 'equal':
        winner_idx = 0 if winner == 'boundary' else 1
        star_x = values[winner_idx] + errors[winner_idx] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.08
        ax.text(star_x, y_pos[winner_idx], '\u2605',  # Unicode star
                fontsize=24, color=COLORS['winner_star'], ha='center', va='center')

    # Panel title with label
    ax.set_title(f'{panel_label}. {metric_name}', fontsize=FONTS['panel_title'],
                 fontweight='bold', color=COLORS['text'], loc='left', pad=10)

    # Question subtitle
    ax.text(0.0, 1.02, f'"{question}"', transform=ax.transAxes,
            fontsize=FONTS['question'], style='italic', color='#666666')

    # Y-axis labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=FONTS['bar_label'])

    # X-axis
    ax.set_xlabel(f'{metric_name} {unit}' if unit else metric_name, fontsize=FONTS['bar_label'])
    ax.set_xlim(0, None)

    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add interpretation and stats at bottom
    winner_text = f"Winner: {'BOUNDARY' if winner == 'boundary' else 'ATTRACTOR' if winner == 'attractor' else 'EQUAL'}"
    if winner == 'boundary' and 'FWHM' in metric_name:
        winner_text = "Winner: BOUNDARY (wider)"

    stats_text = f"{_format_pvalue(pvalue)}  |  {_format_cohens_d(cohens_d)}"
    interp_text = f"\u2192 {interpretation}"

    # Position text below the plot
    ax.text(0.0, -0.18, winner_text, transform=ax.transAxes,
            fontsize=FONTS['interpretation'], fontweight='bold',
            color=COLORS['winner_star'] if winner != 'equal' else COLORS['equal'])
    ax.text(0.0, -0.30, interp_text, transform=ax.transAxes,
            fontsize=FONTS['interpretation'], color=COLORS['text'])
    ax.text(0.0, -0.42, stats_text, transform=ax.transAxes,
            fontsize=FONTS['stats'], color='#888888')


def plot_boundary_attractor_quartet(
    metrics: BoundaryAttractorMetrics,
    title: str = "Dual-Property Framework: Boundaries vs Attractors",
    save_path: Optional[str] = None,
    show: bool = True,
    style: str = 'poster'
) -> plt.Figure:
    """
    Generate the 4-panel comparison figure.

    Parameters
    ----------
    metrics : BoundaryAttractorMetrics
        All metrics for the comparison
    title : str
        Overall figure title
    save_path : str, optional
        Path to save figure (will save both PNG and PDF)
    show : bool
        Whether to display the figure
    style : str
        'poster' for large fonts, 'paper' for smaller

    Returns
    -------
    matplotlib Figure
    """
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('white')

    # Adjust subplot spacing
    plt.subplots_adjust(wspace=0.35, hspace=0.55, top=0.88, bottom=0.12)

    # ===== Panel A: Clustering Ratio =====
    _plot_comparison_panel(
        axes[0, 0],
        boundary_val=metrics.clustering_boundary,
        attractor_val=metrics.clustering_attractor,
        boundary_std=metrics.clustering_boundary_std,
        attractor_std=metrics.clustering_attractor_std,
        panel_label='A',
        metric_name='CLUSTERING RATIO',
        question='Where do peaks snap to?',
        unit='',
        winner='boundary' if metrics.clustering_boundary > metrics.clustering_attractor else 'attractor',
        interpretation='Hard frequency walls',
        pvalue=metrics.clustering_pvalue,
        cohens_d=metrics.clustering_cohens_d,
        n_boundary=metrics.clustering_n_boundary,
        n_attractor=metrics.clustering_n_attractor
    )

    # ===== Panel B: Q-Factor =====
    _plot_comparison_panel(
        axes[0, 1],
        boundary_val=metrics.q_boundary,
        attractor_val=metrics.q_attractor,
        boundary_std=metrics.q_boundary_std,
        attractor_std=metrics.q_attractor_std,
        panel_label='B',
        metric_name='Q-FACTOR',
        question='How stable when there?',
        unit='',
        winner='attractor' if metrics.q_attractor > metrics.q_boundary else 'boundary',
        interpretation='Energy basins',
        pvalue=metrics.q_pvalue,
        cohens_d=metrics.q_cohens_d,
        n_boundary=metrics.q_n_boundary,
        n_attractor=metrics.q_n_attractor
    )

    # ===== Panel C: FWHM =====
    _plot_comparison_panel(
        axes[1, 0],
        boundary_val=metrics.fwhm_boundary,
        attractor_val=metrics.fwhm_attractor,
        boundary_std=metrics.fwhm_boundary_std,
        attractor_std=metrics.fwhm_attractor_std,
        panel_label='C',
        metric_name='FWHM',
        question='How broad is activity?',
        unit='(Hz)',
        winner='boundary' if metrics.fwhm_boundary > metrics.fwhm_attractor else 'attractor',
        interpretation='Transition zones',
        pvalue=metrics.fwhm_pvalue,
        cohens_d=metrics.fwhm_cohens_d,
        n_boundary=metrics.fwhm_n_boundary,
        n_attractor=metrics.fwhm_n_attractor
    )

    # ===== Panel D: Eigenvalue =====
    # Determine if equal (p > 0.05)
    eig_winner = 'equal' if metrics.eigenvalue_pvalue > 0.05 else (
        'boundary' if metrics.eigenvalue_boundary > metrics.eigenvalue_attractor else 'attractor'
    )
    _plot_comparison_panel(
        axes[1, 1],
        boundary_val=metrics.eigenvalue_boundary,
        attractor_val=metrics.eigenvalue_attractor,
        boundary_std=metrics.eigenvalue_boundary_std,
        attractor_std=metrics.eigenvalue_attractor_std,
        panel_label='D',
        metric_name='EIGENVALUE',
        question='How much power?',
        unit='(\u03bb)',  # lambda symbol
        winner=eig_winner,
        interpretation='Same total energy',
        pvalue=metrics.eigenvalue_pvalue,
        cohens_d=metrics.eigenvalue_cohens_d,
        n_boundary=metrics.eigenvalue_n_boundary,
        n_attractor=metrics.eigenvalue_n_attractor
    )

    # Overall title
    fig.suptitle(title, fontsize=FONTS['suptitle'], fontweight='bold',
                 color=COLORS['text'], y=0.96)

    # Summary footer
    footer_text = (
        "\u03c6\u207f architecture defines BOTH where frequency bands separate (boundaries) "
        "AND where sustained oscillations preferentially dwell (attractors)"
    )
    fig.text(0.5, 0.02, footer_text, ha='center', fontsize=FONTS['footer'],
             style='italic', color='#666666', wrap=True)

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['boundary'], edgecolor='white',
                       label='Boundary (integer \u03c6\u207f)'),
        mpatches.Patch(facecolor=COLORS['attractor'], edgecolor='white',
                       label='Attractor (half-integer \u03c6\u207f)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=COLORS['winner_star'],
               markersize=15, label='Winner')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.92),
               fontsize=FONTS['bar_label'], frameon=True, fancybox=True)

    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        # Save PNG
        png_path = save_path if save_path.endswith('.png') else f"{save_path}.png"
        fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {png_path}")
        # Save PDF
        pdf_path = png_path.replace('.png', '.pdf')
        fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        print(f"Saved: {pdf_path}")

    if show:
        plt.show()

    return fig


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def generate_boundary_attractor_figure(
    clustering_csv: str = None,
    ged_results_dir: str = None,
    output_dir: str = None,
    show: bool = True
) -> plt.Figure:
    """
    Generate the boundary vs attractor 4-panel figure using default paths.

    Parameters
    ----------
    clustering_csv : str, optional
        Path to clustering data. Defaults to standard location.
    ged_results_dir : str, optional
        Path to GED results. Defaults to standard location.
    output_dir : str, optional
        Output directory. Defaults to 'exports_poster/'.
    show : bool
        Whether to display the figure.

    Returns
    -------
    matplotlib Figure
    """
    # Default paths relative to project root
    project_root = Path(__file__).parent.parent

    if clustering_csv is None:
        clustering_csv = project_root / 'exports_peak_distribution' / 'position_type_ged' / 'position_type_clustering.csv'

    if ged_results_dir is None:
        # Try multiple possible locations
        possible_dirs = [
            project_root / 'exports_ged_validation_physf' / 'per_session',
            project_root / 'exports_ged_validation_all' / 'per_session',
            project_root / 'exports_ged_validation_mpeng' / 'per_session',
        ]
        for d in possible_dirs:
            if d.exists() and list(d.glob('*_ged_results.csv')):
                ged_results_dir = d
                break
        if ged_results_dir is None:
            ged_results_dir = possible_dirs[0]  # Use first as default

    if output_dir is None:
        output_dir = project_root / 'exports_poster'

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading clustering data from: {clustering_csv}")
    print(f"Loading GED results from: {ged_results_dir}")

    # Compute metrics
    metrics = compute_aggregate_metrics(
        clustering_csv=str(clustering_csv),
        ged_results_dir=str(ged_results_dir)
    )

    # Print summary
    print("\n" + "=" * 60)
    print("BOUNDARY vs ATTRACTOR METRICS SUMMARY")
    print("=" * 60)
    print(f"\nClustering Ratio:")
    print(f"  Boundary:  {metrics.clustering_boundary:.2f} \u00b1 {metrics.clustering_boundary_std:.2f}")
    print(f"  Attractor: {metrics.clustering_attractor:.2f} \u00b1 {metrics.clustering_attractor_std:.2f}")
    print(f"  {_format_pvalue(metrics.clustering_pvalue)}, {_format_cohens_d(metrics.clustering_cohens_d)}")

    print(f"\nQ-Factor:")
    print(f"  Boundary:  {metrics.q_boundary:.1f} \u00b1 {metrics.q_boundary_std:.1f}")
    print(f"  Attractor: {metrics.q_attractor:.1f} \u00b1 {metrics.q_attractor_std:.1f}")
    print(f"  {_format_pvalue(metrics.q_pvalue)}, {_format_cohens_d(metrics.q_cohens_d)}")

    print(f"\nFWHM (Hz):")
    print(f"  Boundary:  {metrics.fwhm_boundary:.2f} \u00b1 {metrics.fwhm_boundary_std:.2f}")
    print(f"  Attractor: {metrics.fwhm_attractor:.2f} \u00b1 {metrics.fwhm_attractor_std:.2f}")
    print(f"  {_format_pvalue(metrics.fwhm_pvalue)}, {_format_cohens_d(metrics.fwhm_cohens_d)}")

    print(f"\nEigenvalue (\u03bb):")
    print(f"  Boundary:  {metrics.eigenvalue_boundary:.2f} \u00b1 {metrics.eigenvalue_boundary_std:.2f}")
    print(f"  Attractor: {metrics.eigenvalue_attractor:.2f} \u00b1 {metrics.eigenvalue_attractor_std:.2f}")
    print(f"  {_format_pvalue(metrics.eigenvalue_pvalue)}, {_format_cohens_d(metrics.eigenvalue_cohens_d)}")
    print("=" * 60 + "\n")

    # Generate figure
    save_path = output_dir / 'boundary_attractor_quartet'
    fig = plot_boundary_attractor_quartet(
        metrics=metrics,
        save_path=str(save_path),
        show=show
    )

    return fig


if __name__ == '__main__':
    generate_boundary_attractor_figure(show=True)

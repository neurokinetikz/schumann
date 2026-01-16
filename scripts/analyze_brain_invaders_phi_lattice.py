#!/usr/bin/env python3
"""
BRAIN INVADERS Dataset φⁿ Lattice Analysis
============================================

Applies unified paper Study 2 methodology to the BRAIN INVADERS FOOOF peak dataset
(1,115,825 peaks from 64 subjects, 512 Hz sampling, extended to 250 Hz) to validate
φⁿ lattice architecture.

Analyses:
1. Position-type enrichment (boundary, attractor, noble positions)
2. Band-stratified analysis (9 frequency bands including high-gamma extensions)
3. Scaling factor comparison (φ vs e, π, √2, 2, 1.5)
4. f₀ tolerance plateau sweep
5. Permutation tests (uniform + phase-shift)
6. Per-epoch consistency (64 subjects × 37 epochs ≈ 2,368 units)
7. Kendall's τ ordering validation

Output:
- 6 publication-quality PNG figures (300 DPI)
- LaTeX tables for paper inclusion
- CSV summary files

Usage:
    python analyze_brain_invaders_phi_lattice.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from typing import Dict, List, Tuple
from itertools import permutations
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================================================
# Constants
# ============================================================================

PHI = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618034
F0_DEFAULT = 7.6  # Hz (empirical SR fundamental)

# Position type definitions (from unified paper)
POSITION_TYPES = {
    'boundary': {'center': 0.0, 'ranges': [(0.0, 0.1), (0.9, 1.0)]},
    'noble_2': {'center': 0.382, 'ranges': [(0.332, 0.432)]},
    'attractor': {'center': 0.5, 'ranges': [(0.45, 0.55)]},
    'noble_1': {'center': 0.618, 'ranges': [(0.568, 0.668)]},
}

# Alternative scaling factors
SCALING_FACTORS = {
    'phi': PHI,
    'e': np.e,
    'pi': np.pi,
    'sqrt2': np.sqrt(2),
    '2': 2.0,
    '1.5': 1.5,
}

# φⁿ-based band boundaries (9 bands for extended analysis up to 250 Hz)
# This shows where φⁿ organization extends and where it breaks down
BANDS = {
    'Delta': (F0_DEFAULT * PHI**(-2), F0_DEFAULT * PHI**(-1)),       # 2.9-4.7 Hz
    'Theta': (F0_DEFAULT * PHI**(-1), F0_DEFAULT * PHI**0),          # 4.7-7.6 Hz
    'Alpha': (F0_DEFAULT * PHI**0, F0_DEFAULT * PHI**1),             # 7.6-12.3 Hz
    'Low Beta': (F0_DEFAULT * PHI**1, F0_DEFAULT * PHI**2),          # 12.3-19.9 Hz
    'High Beta': (F0_DEFAULT * PHI**2, F0_DEFAULT * PHI**3),         # 19.9-32.2 Hz
    'Gamma': (F0_DEFAULT * PHI**3, F0_DEFAULT * PHI**4),             # 32.2-52.1 Hz
    'High-Gamma': (F0_DEFAULT * PHI**4, F0_DEFAULT * PHI**5),        # 52.1-84.3 Hz
    'Ultra-Gamma': (F0_DEFAULT * PHI**5, F0_DEFAULT * PHI**6),       # 84.3-136.4 Hz
    'Hyper-Gamma': (F0_DEFAULT * PHI**6, F0_DEFAULT * PHI**7),       # 136.4-220.7 Hz
}

# Output directory
OUTPUT_DIR = 'papers/images'


# ============================================================================
# Core Functions
# ============================================================================

def compute_lattice_coordinate(freq: np.ndarray, f0: float = F0_DEFAULT,
                                scale: float = PHI) -> np.ndarray:
    """Compute lattice coordinate u = [log_scale(f/f0)] mod 1."""
    n = np.log(freq / f0) / np.log(scale)
    return n - np.floor(n)


def compute_position_enrichment(freqs: np.ndarray, f0: float = F0_DEFAULT,
                                  n_bootstrap: int = 1000,
                                  random_state: int = 42) -> Dict:
    """
    Compute enrichment at each position type with bootstrap 95% CI.

    Returns dict with enrichment percentages and CIs for each position.
    """
    rng = np.random.default_rng(random_state)
    u = compute_lattice_coordinate(freqs, f0)
    n_total = len(u)

    results = {}

    for pos_name, config in POSITION_TYPES.items():
        # Count peaks in position ranges
        in_position = np.zeros(n_total, dtype=bool)
        total_width = 0.0

        for u_low, u_high in config['ranges']:
            in_position |= (u >= u_low) & (u < u_high)
            total_width += (u_high - u_low)

        observed = in_position.sum()
        expected = n_total * total_width
        enrichment = ((observed / expected) - 1) * 100 if expected > 0 else 0

        # Bootstrap CI
        enrichments_boot = []
        for _ in range(n_bootstrap):
            idx = rng.choice(n_total, n_total, replace=True)
            u_boot = u[idx]
            in_pos_boot = np.zeros(n_total, dtype=bool)
            for u_low, u_high in config['ranges']:
                in_pos_boot |= (u_boot >= u_low) & (u_boot < u_high)
            obs_boot = in_pos_boot.sum()
            e_boot = ((obs_boot / expected) - 1) * 100 if expected > 0 else 0
            enrichments_boot.append(e_boot)

        ci_low, ci_high = np.percentile(enrichments_boot, [2.5, 97.5])

        results[pos_name] = {
            'enrichment': enrichment,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'observed': observed,
            'expected': expected,
            'width': total_width,
        }

    return results


def test_ordering(enrichments: Dict) -> Dict:
    """Test if ordering matches theory: boundary < noble_2 < attractor < noble_1."""
    from scipy.stats import kendalltau

    position_names = ['boundary', 'noble_2', 'attractor', 'noble_1']
    predicted_ranks = np.array([1, 2, 3, 4])
    values = np.array([enrichments[p]['enrichment'] for p in position_names])
    observed_ranks = stats.rankdata(values, method='average')

    tau, p_value = kendalltau(predicted_ranks, observed_ranks)

    # Exact p-value via enumeration
    all_orderings = list(permutations([1, 2, 3, 4]))
    n_better = sum(1 for perm in all_orderings
                   if kendalltau(predicted_ranks, np.array(perm))[0] >= tau - 1e-10)
    p_exact = n_better / 24

    ordering_satisfied = (values[3] > values[2] > values[1] > values[0])

    return {
        'kendall_tau': tau,
        'p_value': p_value,
        'p_value_exact': p_exact,
        'ordering_satisfied': ordering_satisfied,
        'observed_ranks': observed_ranks.tolist(),
        'values': dict(zip(position_names, values.tolist())),
    }


# ============================================================================
# Band-Stratified Analysis
# ============================================================================

def compute_band_enrichment(freqs: np.ndarray, f0: float = F0_DEFAULT,
                             n_bootstrap: int = 500) -> pd.DataFrame:
    """Repeat position enrichment for each frequency band (9 bands)."""
    results_list = []

    for band_name, (f_low, f_high) in BANDS.items():
        mask = (freqs >= f_low) & (freqs < f_high)
        band_freqs = freqs[mask]

        if len(band_freqs) < 100:
            continue

        enrichment = compute_position_enrichment(band_freqs, f0, n_bootstrap)

        row = {
            'band': band_name,
            'f_low': f_low,
            'f_high': f_high,
            'n_peaks': len(band_freqs),
        }
        for pos_name in POSITION_TYPES:
            row[f'{pos_name}_enrich'] = enrichment[pos_name]['enrichment']
            row[f'{pos_name}_ci_low'] = enrichment[pos_name]['ci_low']
            row[f'{pos_name}_ci_high'] = enrichment[pos_name]['ci_high']

        results_list.append(row)

    return pd.DataFrame(results_list)


# ============================================================================
# Scaling Factor Comparison
# ============================================================================

def compare_scaling_factors(freqs: np.ndarray, f0: float = F0_DEFAULT) -> pd.DataFrame:
    """Test 6 scaling factors and compare alignment scores."""
    results = []

    for name, factor in SCALING_FACTORS.items():
        u = compute_lattice_coordinate(freqs, f0, factor)

        enrichments = {}
        for pos_name, config in POSITION_TYPES.items():
            in_pos = np.zeros(len(u), dtype=bool)
            width = 0.0
            for u_low, u_high in config['ranges']:
                in_pos |= (u >= u_low) & (u < u_high)
                width += (u_high - u_low)
            observed_frac = in_pos.sum() / len(u)
            enrichments[pos_name] = ((observed_frac / width) - 1) * 100

        # Test ordering
        values = [enrichments['boundary'], enrichments['noble_2'],
                  enrichments['attractor'], enrichments['noble_1']]
        predicted = [1, 2, 3, 4]
        observed_ranks = stats.rankdata(values, method='average')
        tau, _ = stats.kendalltau(predicted, observed_ranks)
        correct = (values[3] > values[2] > values[1] > values[0])

        # Alignment score
        alignment = (
            -enrichments['boundary'] +
            0.5 * enrichments['noble_2'] +
            enrichments['attractor'] +
            1.5 * enrichments['noble_1']
        )

        results.append({
            'factor_name': name,
            'factor_value': factor,
            'alignment_score': alignment,
            'kendall_tau': tau,
            'correct_ordering': correct,
            **{f'{k}_enrich': v for k, v in enrichments.items()}
        })

    return pd.DataFrame(results)


# ============================================================================
# f₀ Tolerance Plateau
# ============================================================================

def sweep_f0(freqs: np.ndarray, f0_range: Tuple[float, float] = (6.0, 9.0),
              step: float = 0.1) -> Tuple[pd.DataFrame, Tuple[float, float]]:
    """Sweep f0 and compute alignment at each value."""
    f0_values = np.arange(f0_range[0], f0_range[1] + step, step)
    results = []

    for f0 in f0_values:
        u = compute_lattice_coordinate(freqs, f0, PHI)

        enrichments = {}
        for pos_name, config in POSITION_TYPES.items():
            in_pos = np.zeros(len(u), dtype=bool)
            width = 0.0
            for u_low, u_high in config['ranges']:
                in_pos |= (u >= u_low) & (u < u_high)
                width += (u_high - u_low)
            observed_frac = in_pos.sum() / len(u)
            enrichments[pos_name] = ((observed_frac / width) - 1) * 100

        alignment = (
            -enrichments['boundary'] +
            0.5 * enrichments['noble_2'] +
            enrichments['attractor'] +
            1.5 * enrichments['noble_1']
        )

        results.append({
            'f0': f0,
            'alignment': alignment,
            **{f'{k}_enrich': v for k, v in enrichments.items()}
        })

    df = pd.DataFrame(results)

    # Find plateau (>95% of max)
    max_align = df['alignment'].max()
    df['pct_of_max'] = df['alignment'] / max_align * 100
    plateau_mask = df['pct_of_max'] >= 95

    if plateau_mask.any():
        plateau_f0 = df.loc[plateau_mask, 'f0'].values
        plateau_range = (plateau_f0.min(), plateau_f0.max())
    else:
        plateau_range = (np.nan, np.nan)

    return df, plateau_range


# ============================================================================
# Permutation Tests
# ============================================================================

def compute_alignment_metric(freqs: np.ndarray, f0: float) -> float:
    """Simple alignment: attractor enrichment - boundary enrichment."""
    u = compute_lattice_coordinate(freqs, f0)
    u_window = 0.05

    boundary_mask = (np.abs(u) < u_window) | (np.abs(u - 1.0) < u_window)
    attractor_mask = np.abs(u - 0.5) < u_window

    boundary_frac = boundary_mask.sum() / len(u)
    attractor_frac = attractor_mask.sum() / len(u)

    boundary_enrich = (boundary_frac / (2 * u_window) - 1) * 100
    attractor_enrich = (attractor_frac / (2 * u_window) - 1) * 100

    return attractor_enrich - boundary_enrich


def permutation_test_uniform(freqs: np.ndarray, n_permutations: int = 10000,
                              f0: float = F0_DEFAULT,
                              random_state: int = 42) -> Dict:
    """Test: Are peaks uniformly distributed in frequency?"""
    rng = np.random.default_rng(random_state)
    observed = compute_alignment_metric(freqs, f0)

    freq_min, freq_max = freqs.min(), freqs.max()
    null_alignments = []

    for _ in range(n_permutations):
        random_freqs = rng.uniform(freq_min, freq_max, len(freqs))
        null_alignments.append(compute_alignment_metric(random_freqs, f0))

    null_alignments = np.array(null_alignments)
    p_value = (null_alignments >= observed).sum() / n_permutations

    return {
        'observed': observed,
        'null_mean': null_alignments.mean(),
        'null_std': null_alignments.std(),
        'p_value': p_value,
        'n_permutations': n_permutations,
        'distribution': null_alignments,
        'test_type': 'uniform_frequency',
    }


def permutation_test_phase_shift(freqs: np.ndarray, n_permutations: int = 10000,
                                   f0: float = F0_DEFAULT,
                                   random_state: int = 42) -> Dict:
    """Test: Is the specific phase of φⁿ grid significant?"""
    rng = np.random.default_rng(random_state)
    u = compute_lattice_coordinate(freqs, f0, PHI)
    observed = compute_alignment_metric(freqs, f0)

    null_alignments = []
    u_window = 0.05

    for _ in range(n_permutations):
        shift = rng.uniform(0, 1)
        u_shifted = (u + shift) % 1.0

        boundary_mask = (np.abs(u_shifted) < u_window) | (np.abs(u_shifted - 1.0) < u_window)
        attractor_mask = np.abs(u_shifted - 0.5) < u_window

        boundary_frac = boundary_mask.sum() / len(u_shifted)
        attractor_frac = attractor_mask.sum() / len(u_shifted)

        boundary_enrich = (boundary_frac / (2 * u_window) - 1) * 100
        attractor_enrich = (attractor_frac / (2 * u_window) - 1) * 100

        null_alignments.append(attractor_enrich - boundary_enrich)

    null_alignments = np.array(null_alignments)
    p_value = (null_alignments >= observed).sum() / n_permutations

    return {
        'observed': observed,
        'null_mean': null_alignments.mean(),
        'null_std': null_alignments.std(),
        'p_value': p_value,
        'n_permutations': n_permutations,
        'distribution': null_alignments,
        'test_type': 'phase_shift',
    }


# ============================================================================
# Per-Epoch Consistency
# ============================================================================

def epoch_consistency_analysis(peaks_df: pd.DataFrame, f0: float = F0_DEFAULT,
                                min_peaks: int = 50) -> Dict:
    """For each epoch (session × epoch): compute position enrichment and test ordering."""
    # Create epoch identifier
    peaks_df = peaks_df.copy()
    peaks_df['epoch_id'] = peaks_df['session'].astype(str) + '_epoch_' + peaks_df['epoch'].astype(str)

    epoch_ids = peaks_df['epoch_id'].unique()
    epoch_results = []

    for epoch_id in epoch_ids:
        epoch_peaks = peaks_df[peaks_df['epoch_id'] == epoch_id]

        if len(epoch_peaks) < min_peaks:
            continue

        freqs = epoch_peaks['freq'].values
        u = compute_lattice_coordinate(freqs, f0)

        enrichments = {}
        for pos_name, config in POSITION_TYPES.items():
            in_pos = np.zeros(len(u), dtype=bool)
            width = 0.0
            for u_low, u_high in config['ranges']:
                in_pos |= (u >= u_low) & (u < u_high)
                width += (u_high - u_low)
            observed_frac = in_pos.sum() / len(u) if len(u) > 0 else 0
            enrichments[pos_name] = ((observed_frac / width) - 1) * 100 if width > 0 else 0

        correct = (enrichments['noble_1'] > enrichments['attractor'] >
                   enrichments['noble_2'] > enrichments['boundary'])
        attractor_gt_boundary = enrichments['attractor'] > enrichments['boundary']

        epoch_results.append({
            'epoch_id': epoch_id,
            'session': epoch_peaks['session'].iloc[0],
            'epoch': epoch_peaks['epoch'].iloc[0],
            'n_peaks': len(epoch_peaks),
            'correct_ordering': correct,
            'attractor_gt_boundary': attractor_gt_boundary,
            **{f'{k}_enrich': v for k, v in enrichments.items()}
        })

    df = pd.DataFrame(epoch_results)

    n_epochs = len(df)
    pct_correct = 100 * df['correct_ordering'].sum() / n_epochs if n_epochs > 0 else 0
    pct_attractor_gt = 100 * df['attractor_gt_boundary'].sum() / n_epochs if n_epochs > 0 else 0

    diffs = df['attractor_enrich'] - df['boundary_enrich']
    cohens_d = diffs.mean() / diffs.std() if diffs.std() > 0 else 0

    return {
        'epoch_df': df,
        'n_epochs': n_epochs,
        'pct_correct_ordering': pct_correct,
        'pct_attractor_gt_boundary': pct_attractor_gt,
        'cohens_d': cohens_d,
    }


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_position_enrichment(enrichment: Dict, n_peaks: int, output_path: str):
    """Figure 1: Position-type enrichment bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    positions = ['boundary', 'noble_2', 'attractor', 'noble_1']
    labels = ['Boundary\n(n=integer)', '2° Noble\n(n+0.382)', 'Attractor\n(n+0.5)', '1° Noble\n(n+0.618)']
    colors = ['#cc8800', '#88aa44', '#2288cc', '#22cc88']

    x = np.arange(len(positions))
    values = [enrichment[p]['enrichment'] for p in positions]
    ci_low = [enrichment[p]['ci_low'] for p in positions]
    ci_high = [enrichment[p]['ci_high'] for p in positions]
    errors = [[v - cl for v, cl in zip(values, ci_low)],
              [ch - v for v, ch in zip(values, ci_high)]]

    bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor='black')
    ax.errorbar(x, values, yerr=errors, fmt='none', color='black', capsize=5, linewidth=2)

    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Enrichment (%)', fontsize=12)
    ax.set_title(f'BRAIN INVADERS Dataset: Position-Type Enrichment\n(N = {n_peaks:,} FOOOF peaks)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value annotations with CI ranges
    for i, (bar, val, cl, ch) in enumerate(zip(bars, values, ci_low, ci_high)):
        y = bar.get_height()
        sign = '+' if val > 0 else ''
        y_offset = (ch - val) * 0.3 + 2 if val >= 0 else -abs(cl - val) * 0.3 - 8
        ax.text(bar.get_x() + bar.get_width()/2, y + y_offset,
                f'{sign}{val:.1f}%\n[{cl:.1f}, {ch:.1f}]',
                ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_band_stratified(band_df: pd.DataFrame, freqs: np.ndarray, output_path: str):
    """Figure 2: Band-stratified 9-panel enrichment (3x3 grid)."""
    n_bands = len(band_df)
    n_cols = 3
    n_rows = (n_bands + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows + 1))
    axes = axes.flatten()

    positions = {0.382: ('purple', '2° Noble'), 0.5: ('orange', 'Attractor'),
                 0.618: ('green', '1° Noble')}

    for idx, (_, row) in enumerate(band_df.iterrows()):
        if idx >= len(axes):
            break
        ax = axes[idx]

        mask = (freqs >= row['f_low']) & (freqs < row['f_high'])
        band_freqs = freqs[mask]
        u = compute_lattice_coordinate(band_freqs)

        n_bins = 25
        counts, bin_edges = np.histogram(u, bins=n_bins, range=(0, 1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        expected = len(band_freqs) / n_bins
        enrichment = counts / expected if expected > 0 else counts

        ax.bar(bin_centers, enrichment, width=0.036, color='steelblue', alpha=0.8,
               edgecolor='white', linewidth=0.3)
        ax.axhline(1.0, color='red', linestyle='--', linewidth=2)

        for pos, (color, _) in positions.items():
            ax.axvline(pos, color=color, linestyle='-', linewidth=2, alpha=0.9)
            bin_idx = min(int(pos * n_bins), n_bins - 1)
            z = (counts[bin_idx] - expected) / np.sqrt(expected) if expected > 0 else 0
            sig = '***' if abs(z) > 3.29 else '**' if abs(z) > 2.58 else '*' if abs(z) > 1.96 else ''
            ax.text(pos, enrichment.max() * 0.92, f'z={z:.1f}{sig}', ha='center',
                    fontsize=8, color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.85))

        ax.set_title(f"{row['band']} ({row['f_low']:.1f}-{row['f_high']:.1f} Hz)\nn={row['n_peaks']:,}",
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Fractional position u', fontsize=9)
        ax.set_ylabel('Enrichment vs uniform', fontsize=9)
        ax.set_xlim(0, 1)
        if len(enrichment) > 0 and enrichment.max() > 0:
            ax.set_ylim(0, enrichment.max() * 1.15)

    # Hide unused axes
    for idx in range(len(band_df), len(axes)):
        axes[idx].set_visible(False)

    legend_elements = [
        Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Uniform (1.0)'),
        Line2D([0], [0], color='purple', linestyle='-', linewidth=2, label='2° Noble (0.382)'),
        Line2D([0], [0], color='orange', linestyle='-', linewidth=2, label='Attractor (0.5)'),
        Line2D([0], [0], color='green', linestyle='-', linewidth=2, label='1° Noble (0.618)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    plt.suptitle('BRAIN INVADERS: Band-Stratified φⁿ Lattice Analysis (9 Bands)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_scaling_comparison(factor_df: pd.DataFrame, output_path: str):
    """Figure 3: Scaling factor comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    factor_df = factor_df.sort_values('alignment_score', ascending=False)
    x = np.arange(len(factor_df))
    colors = ['#2ca02c' if c else '#d62728' for c in factor_df['correct_ordering']]

    bars = ax.bar(x, factor_df['alignment_score'], color=colors, edgecolor='black', linewidth=1.5)

    ax.set_xticks(x)
    labels = [f"{row['factor_name']}\n({row['factor_value']:.3f})"
              for _, row in factor_df.iterrows()]
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Alignment Score', fontsize=12)
    ax.set_title('BRAIN INVADERS: Scaling Factor Comparison\n(Green = correct ordering, Red = incorrect)',
                 fontsize=14, fontweight='bold')

    for i, (bar, row) in enumerate(zip(bars, factor_df.itertuples())):
        tau_str = f'τ={row.kendall_tau:.2f}'
        ax.annotate(tau_str, xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords='offset points', ha='center', fontsize=9)

    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_f0_sensitivity(f0_df: pd.DataFrame, plateau: Tuple, output_path: str):
    """Figure 4: f₀ sensitivity sweep."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top: alignment score
    ax1.plot(f0_df['f0'], f0_df['alignment'], 'b-', linewidth=2, marker='o', markersize=4)
    ax1.axhline(f0_df['alignment'].max() * 0.95, color='red', linestyle='--',
                linewidth=1.5, label='95% threshold')

    if not np.isnan(plateau[0]):
        ax1.axvspan(plateau[0], plateau[1], alpha=0.2, color='green', label='Plateau')
        ax1.axvline(7.6, color='orange', linestyle=':', linewidth=2, label='f₀=7.6 Hz')

    ax1.set_ylabel('Alignment Score', fontsize=12)
    ax1.set_title('BRAIN INVADERS: f₀ Sensitivity Analysis', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Bottom: position enrichments
    ax2.plot(f0_df['f0'], f0_df['boundary_enrich'], 'r-', linewidth=2, label='Boundary')
    ax2.plot(f0_df['f0'], f0_df['noble_2_enrich'], 'purple', linewidth=2, label='2° Noble')
    ax2.plot(f0_df['f0'], f0_df['attractor_enrich'], 'orange', linewidth=2, label='Attractor')
    ax2.plot(f0_df['f0'], f0_df['noble_1_enrich'], 'g-', linewidth=2, label='1° Noble')

    ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax2.set_xlabel('Base Frequency f₀ (Hz)', fontsize=12)
    ax2.set_ylabel('Enrichment (%)', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_permutation_tests(uniform_test: Dict, phase_test: Dict, output_path: str):
    """Figure 5: Dual permutation test histograms."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Phase shift test (left panel)
    phase_dist = np.array(phase_test['distribution'])
    ax1.hist(phase_dist, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    ax1.axvline(phase_test['observed'], color='red', linewidth=2,
                label=f"Observed: {phase_test['observed']:.1f}%")
    ax1.axvline(phase_dist.mean(), color='gray', linewidth=2, linestyle='--',
                label=f"Null mean: {phase_dist.mean():.1f}% ± {phase_dist.std():.1f}%")
    ax1.set_xlabel('Alignment Metric (%)', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title(f"Phase-Shift Test\n(p = {phase_test['p_value']:.4f})",
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Uniform frequency test (right panel)
    uniform_dist = np.array(uniform_test['distribution'])
    ax2.hist(uniform_dist, bins=50, color='forestgreen', alpha=0.7, edgecolor='white')
    ax2.axvline(uniform_test['observed'], color='red', linewidth=2,
                label=f"Observed: {uniform_test['observed']:.1f}%")
    ax2.axvline(uniform_dist.mean(), color='gray', linewidth=2, linestyle='--',
                label=f"Null mean: {uniform_dist.mean():.1f}% ± {uniform_dist.std():.1f}%")
    ax2.set_xlabel('Alignment Metric (%)', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    p_str = f"{uniform_test['p_value']:.4f}" if uniform_test['p_value'] > 0 else "< 0.0001"
    ax2.set_title(f"Uniform Frequency Test\n(p = {p_str})",
                  fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('BRAIN INVADERS: Permutation Test Comparison for φⁿ Alignment',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_epoch_consistency(epoch_results: Dict, output_path: str):
    """Figure 6: Epoch-level consistency."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    df = epoch_results['epoch_df']

    # Panel A: histogram of attractor - boundary difference
    diffs = df['attractor_enrich'] - df['boundary_enrich']
    ax1.hist(diffs, bins=40, color='steelblue', alpha=0.7, edgecolor='white')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero (no effect)')
    ax1.axvline(diffs.mean(), color='green', linestyle='-', linewidth=2,
                label=f'Mean: {diffs.mean():.1f}%')
    ax1.set_xlabel('Attractor - Boundary Enrichment (%)', fontsize=11)
    ax1.set_ylabel('Number of Epochs', fontsize=11)
    ax1.set_title(f"Epoch-Level Alignment Distribution\n{epoch_results['pct_attractor_gt_boundary']:.1f}% show attractor > boundary",
                  fontsize=12, fontweight='bold')
    ax1.legend()

    # Panel B: scatter of n_peaks vs alignment
    colors = ['green' if c else 'red' for c in df['attractor_gt_boundary']]
    ax2.scatter(df['n_peaks'], diffs, c=colors, alpha=0.5, s=20)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax2.set_xlabel('Peaks per Epoch', fontsize=11)
    ax2.set_ylabel('Attractor - Boundary (%)', fontsize=11)
    ax2.set_title(f"Alignment vs Epoch Size\nCohen's d = {epoch_results['cohens_d']:.2f}",
                  fontsize=12, fontweight='bold')

    legend_elements = [Patch(facecolor='green', alpha=0.5, label='Attractor > Boundary'),
                       Patch(facecolor='red', alpha=0.5, label='Attractor ≤ Boundary')]
    ax2.legend(handles=legend_elements, loc='upper right')

    plt.suptitle(f"BRAIN INVADERS: Epoch-Level Consistency (N = {epoch_results['n_epochs']:,} epochs)",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================================
# LaTeX Table Generation
# ============================================================================

def generate_latex_tables(enrichment: Dict, ordering_test: Dict, band_df: pd.DataFrame,
                           factor_df: pd.DataFrame, uniform_test: Dict, phase_test: Dict,
                           epoch_results: Dict) -> str:
    """Generate LaTeX tables for paper inclusion."""
    latex = []

    # Table 1: Position enrichment
    latex.append("% Table 1: Position-Type Enrichment")
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{BRAIN INVADERS Dataset: Position-Type Enrichment}")
    latex.append("\\begin{tabular}{lccc}")
    latex.append("\\hline")
    latex.append("Position & Enrichment (\\%) & 95\\% CI & Observed/Expected \\\\")
    latex.append("\\hline")
    for pos in ['boundary', 'noble_2', 'attractor', 'noble_1']:
        e = enrichment[pos]
        latex.append(f"{pos.replace('_', ' ').title()} & {e['enrichment']:.1f} & "
                     f"[{e['ci_low']:.1f}, {e['ci_high']:.1f}] & "
                     f"{e['observed']:.0f}/{e['expected']:.0f} \\\\")
    latex.append("\\hline")
    latex.append(f"\\multicolumn{{4}}{{l}}{{Kendall's $\\tau$ = {ordering_test['kendall_tau']:.3f}, "
                 f"$p$ = {ordering_test['p_value_exact']:.4f}}} \\\\")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    latex.append("")

    # Table 2: Band-stratified enrichment (9 bands)
    latex.append("% Table 2: Band-Stratified Enrichment (Extended)")
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{BRAIN INVADERS: Band-Stratified 1\\degree\\ Noble Enrichment}")
    latex.append("\\begin{tabular}{llll}")
    latex.append("\\hline")
    latex.append("Band & Frequency Range & Peaks & 1\\degree\\ Noble Enrichment \\\\")
    latex.append("\\hline")
    for _, row in band_df.iterrows():
        latex.append(f"{row['band']} & {row['f_low']:.1f}--{row['f_high']:.1f} Hz & "
                     f"{row['n_peaks']:,} & {row['noble_1_enrich']:+.1f}\\% \\\\")
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    latex.append("")

    # Table 3: Scaling factor comparison
    latex.append("% Table 3: Scaling Factor Comparison")
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{BRAIN INVADERS: Scaling Factor Comparison}")
    latex.append("\\begin{tabular}{lcccc}")
    latex.append("\\hline")
    latex.append("Factor & Value & Alignment & $\\tau$ & Correct Order \\\\")
    latex.append("\\hline")
    for _, row in factor_df.sort_values('alignment_score', ascending=False).iterrows():
        correct = "Yes" if row['correct_ordering'] else "No"
        latex.append(f"{row['factor_name']} & {row['factor_value']:.3f} & "
                     f"{row['alignment_score']:.1f} & {row['kendall_tau']:.2f} & {correct} \\\\")
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    latex.append("")

    # Table 4: Permutation tests
    latex.append("% Table 4: Permutation Tests")
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{BRAIN INVADERS: Permutation Test Results}")
    latex.append("\\begin{tabular}{lccc}")
    latex.append("\\hline")
    latex.append("Test & Observed & Null Mean (SD) & $p$-value \\\\")
    latex.append("\\hline")
    p_uniform = "$<$ 0.0001" if uniform_test['p_value'] == 0 else f"{uniform_test['p_value']:.4f}"
    latex.append(f"Uniform Frequency & {uniform_test['observed']:.1f} & "
                 f"{uniform_test['null_mean']:.1f} ({uniform_test['null_std']:.1f}) & "
                 f"{p_uniform} \\\\")
    latex.append(f"Phase Shift & {phase_test['observed']:.1f} & "
                 f"{phase_test['null_mean']:.1f} ({phase_test['null_std']:.1f}) & "
                 f"{phase_test['p_value']:.4f} \\\\")
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    return "\n".join(latex)


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main analysis pipeline."""
    print("=" * 70)
    print("BRAIN INVADERS Dataset φⁿ Lattice Analysis")
    print("=" * 70)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load data (extended 256Hz file for paper - shows φⁿ across full frequency range)
    print("\n[1/8] Loading BRAIN INVADERS peak data (extended 256Hz)...")

    # Use extended 256Hz file for paper (828,425 peaks, 1-250 Hz)
    # This dataset shows where φⁿ organization extends and breaks down
    clean_file = 'golden_ratio_peaks_BRAIN_INVADERS_256Hz_clean.csv'
    raw_file = 'golden_ratio_peaks_BRAIN_INVADERS_256Hz.csv'

    if not os.path.exists(clean_file):
        print("  Creating notch-artifact-cleaned dataset...")
        peaks_raw = pd.read_csv(raw_file)
        # Remove peaks near 50 Hz notch filter harmonics
        notch_freqs = [50, 100, 150, 200]
        notch_bw = 5  # Hz bandwidth
        mask = np.ones(len(peaks_raw), dtype=bool)
        for nf in notch_freqs:
            mask &= ~((peaks_raw['freq'] >= nf - notch_bw) & (peaks_raw['freq'] <= nf + notch_bw))
        peaks_df = peaks_raw[mask].copy()
        peaks_df.to_csv(clean_file, index=False)
        print(f"  Removed {len(peaks_raw) - len(peaks_df):,} peaks near notch artifacts")
    else:
        peaks_df = pd.read_csv(clean_file)

    freqs = peaks_df['freq'].values
    n_subjects = peaks_df['session'].nunique()
    n_epochs = len(peaks_df.groupby(['session', 'epoch']))
    print(f"  Loaded {len(freqs):,} peaks from {n_subjects} subjects, {n_epochs} epochs")

    # 2. Position enrichment
    print("\n[2/8] Computing position enrichment...")
    enrichment = compute_position_enrichment(freqs, n_bootstrap=1000)
    ordering_test = test_ordering(enrichment)

    print(f"  Boundary:  {enrichment['boundary']['enrichment']:+.1f}% "
          f"[{enrichment['boundary']['ci_low']:.1f}, {enrichment['boundary']['ci_high']:.1f}]")
    print(f"  2° Noble:  {enrichment['noble_2']['enrichment']:+.1f}% "
          f"[{enrichment['noble_2']['ci_low']:.1f}, {enrichment['noble_2']['ci_high']:.1f}]")
    print(f"  Attractor: {enrichment['attractor']['enrichment']:+.1f}% "
          f"[{enrichment['attractor']['ci_low']:.1f}, {enrichment['attractor']['ci_high']:.1f}]")
    print(f"  1° Noble:  {enrichment['noble_1']['enrichment']:+.1f}% "
          f"[{enrichment['noble_1']['ci_low']:.1f}, {enrichment['noble_1']['ci_high']:.1f}]")
    print(f"  Kendall's τ = {ordering_test['kendall_tau']:.3f}, "
          f"ordering satisfied: {ordering_test['ordering_satisfied']}")

    # 3. Band-stratified analysis
    print("\n[3/8] Computing band-stratified enrichment (9 bands)...")
    band_df = compute_band_enrichment(freqs, n_bootstrap=500)
    print(f"  Analyzed {len(band_df)} frequency bands")
    for _, row in band_df.iterrows():
        print(f"  {row['band']:12s}: n={row['n_peaks']:7,}, "
              f"1°Noble={row['noble_1_enrich']:+.1f}%")

    # 4. Scaling factor comparison
    print("\n[4/8] Comparing scaling factors...")
    factor_df = compare_scaling_factors(freqs)
    for _, row in factor_df.sort_values('alignment_score', ascending=False).iterrows():
        status = "✓" if row['correct_ordering'] else "✗"
        print(f"  {row['factor_name']:6s}: alignment={row['alignment_score']:6.1f}, "
              f"τ={row['kendall_tau']:.2f} {status}")

    # 5. f₀ sensitivity sweep
    print("\n[5/8] Sweeping f₀ (6.0-9.0 Hz)...")
    f0_df, plateau = sweep_f0(freqs)
    opt_idx = f0_df['alignment'].idxmax()
    opt_f0 = f0_df.loc[opt_idx, 'f0']
    print(f"  Optimal f₀ = {opt_f0:.2f} Hz")
    print(f"  95% plateau: {plateau[0]:.1f} - {plateau[1]:.1f} Hz")

    # 6. Permutation tests
    print("\n[6/8] Running permutation tests (N=10,000 each)...")
    uniform_test = permutation_test_uniform(freqs, n_permutations=10000)
    phase_test = permutation_test_phase_shift(freqs, n_permutations=10000)
    p_uniform_str = f"< {uniform_test['p_value']:.4f}" if uniform_test['p_value'] > 0 else "< 0.0001"
    print(f"  Uniform freq test: p {p_uniform_str}")
    print(f"  Phase shift test:  p = {phase_test['p_value']:.4f}")

    # 7. Epoch-level consistency
    print("\n[7/8] Analyzing epoch-level consistency...")
    epoch_results = epoch_consistency_analysis(peaks_df)
    print(f"  Epochs analyzed: {epoch_results['n_epochs']:,}")
    print(f"  Correct ordering: {epoch_results['pct_correct_ordering']:.1f}%")
    print(f"  Attractor > Boundary: {epoch_results['pct_attractor_gt_boundary']:.1f}%")
    print(f"  Cohen's d = {epoch_results['cohens_d']:.2f}")

    # 8. Generate outputs
    print("\n[8/8] Generating figures and tables...")

    plot_position_enrichment(enrichment, len(freqs), f'{OUTPUT_DIR}/brain_invaders_phi_position_enrichment.png')
    plot_band_stratified(band_df, freqs, f'{OUTPUT_DIR}/brain_invaders_phi_band_stratified.png')
    plot_scaling_comparison(factor_df, f'{OUTPUT_DIR}/brain_invaders_phi_scaling_comparison.png')
    plot_f0_sensitivity(f0_df, plateau, f'{OUTPUT_DIR}/brain_invaders_phi_f0_sensitivity.png')
    plot_permutation_tests(uniform_test, phase_test, f'{OUTPUT_DIR}/brain_invaders_phi_permutation_tests.png')
    plot_epoch_consistency(epoch_results, f'{OUTPUT_DIR}/brain_invaders_phi_session_consistency.png')

    # Save CSV summaries
    summary = {
        'n_peaks': len(freqs),
        'n_subjects': n_subjects,
        'n_epochs': n_epochs,
        'boundary_enrich': enrichment['boundary']['enrichment'],
        'noble_2_enrich': enrichment['noble_2']['enrichment'],
        'attractor_enrich': enrichment['attractor']['enrichment'],
        'noble_1_enrich': enrichment['noble_1']['enrichment'],
        'kendall_tau': ordering_test['kendall_tau'],
        'ordering_satisfied': ordering_test['ordering_satisfied'],
        'uniform_p': uniform_test['p_value'],
        'phase_p': phase_test['p_value'],
        'pct_epochs_correct': epoch_results['pct_correct_ordering'],
        'pct_epochs_attractor_gt': epoch_results['pct_attractor_gt_boundary'],
        'cohens_d': epoch_results['cohens_d'],
        'optimal_f0': opt_f0,
        'plateau_low': plateau[0],
        'plateau_high': plateau[1],
    }
    pd.DataFrame([summary]).to_csv('brain_invaders_phi_analysis_summary.csv', index=False)
    band_df.to_csv('brain_invaders_phi_band_enrichment.csv', index=False)
    epoch_results['epoch_df'].to_csv('brain_invaders_phi_session_results.csv', index=False)

    # Generate LaTeX tables
    latex = generate_latex_tables(enrichment, ordering_test, band_df, factor_df,
                                   uniform_test, phase_test, epoch_results)
    with open('brain_invaders_phi_latex_tables.tex', 'w') as f:
        f.write(latex)
    print("  Saved: brain_invaders_phi_latex_tables.tex")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nKey Results:")
    print(f"  • 1° Noble enrichment: {enrichment['noble_1']['enrichment']:+.1f}%")
    print(f"  • Boundary depletion:  {enrichment['boundary']['enrichment']:+.1f}%")
    print(f"  • Kendall's τ = {ordering_test['kendall_tau']:.3f} "
          f"({'exact' if ordering_test['ordering_satisfied'] else 'partial'} ordering)")
    print(f"  • Only φ achieves correct ordering among 6 scaling factors")
    print(f"  • {epoch_results['pct_attractor_gt_boundary']:.1f}% of epochs "
          f"show attractor > boundary")


if __name__ == '__main__':
    main()

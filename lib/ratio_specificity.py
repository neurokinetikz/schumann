"""
Ratio Specificity & Phase-Rotation Null Analysis
=================================================

Tests whether φ (golden ratio) produces uniquely high spectral enrichment
compared to other mathematical constants, with proper null controls.

Implements four analyses responding to independent replication challenges:
  D1: Phase-rotation null — does enrichment survive circular shuffling?
  D2: Ratio specificity — does φ outperform 11 other ratios?
  D3: f₀ sweep with null threshold
  D4: 2D (f₀, ratio) heatmap with null contour

Dependencies: numpy, pandas, scipy, matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from typing import Dict, Tuple, Optional

# ============================================================================
# CONSTANTS
# ============================================================================

PHI = (1 + np.sqrt(5)) / 2  # 1.6180339887...
F0 = 7.60

# Critic's 12 candidate ratios
NAMED_RATIOS = {
    'φ':     PHI,
    '2':     2.0,
    'e':     np.e,
    'π':     np.pi,
    '√2':    np.sqrt(2),
    'δ_S':   1 + np.sqrt(2),
    '2+√3':  2 + np.sqrt(3),
    '√3':    np.sqrt(3),
    '3/2':   1.5,
    '5/3':   5 / 3,
    '7/4':   1.75,
    '√5':    np.sqrt(5),
}

# φ-specific position offsets (for enrichment at all noble positions)
PHI_INV = 1.0 / PHI
PHI_POSITION_OFFSETS = {
    'boundary':    0.000,
    'noble_6':     PHI_INV ** 6,
    'noble_5':     PHI_INV ** 5,
    'noble_4':     PHI_INV ** 4,
    'noble_3':     PHI_INV ** 3,
    'noble_2':     PHI_INV ** 2,
    'attractor':   0.500,
    'noble_1':     PHI_INV ** 1,
    'inv_noble_3': 1 - PHI_INV ** 3,
    'inv_noble_4': 1 - PHI_INV ** 4,
    'inv_noble_5': 1 - PHI_INV ** 5,
    'inv_noble_6': 1 - PHI_INV ** 6,
}


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def lattice_coordinate(freqs, f0, ratio):
    """
    Map frequencies to lattice coordinate u = [log_r(f/f₀)] mod 1.

    Generalizes phi_frequency_model's mapping to any scaling ratio.
    Under perfect lattice alignment, peaks cluster near u = 0.
    """
    freqs = np.asarray(freqs, dtype=float)
    mask = freqs > 0
    result = np.full_like(freqs, np.nan)
    result[mask] = (np.log(freqs[mask] / f0) / np.log(ratio)) % 1.0
    return result


def _enrichment_at_offset(u, offset, window, n_total):
    """Compute enrichment at a single offset with wrap-around."""
    expected_frac = 2 * window
    if offset < window:
        in_window = (u < offset + window) | (u > 1 - (window - offset))
    elif offset > 1 - window:
        in_window = (u > offset - window) | (u < window - (1 - offset))
    else:
        in_window = (u >= offset - window) & (u < offset + window)
    observed_frac = in_window.sum() / n_total
    return (observed_frac / expected_frac - 1) * 100


def lattice_enrichment(freqs, f0, ratio, window=0.05):
    """
    Enrichment at lattice points (u ≈ 0) for arbitrary ratio.

    Enrichment = (observed_frac / expected_frac - 1) × 100%
    where expected_frac = 2 × window (uniform on [0,1)).

    Handles wrap-around: peaks near u=0 and u=1 are both "at the lattice point".
    """
    u = lattice_coordinate(freqs, f0, ratio)
    u = u[np.isfinite(u)]
    n_total = len(u)
    if n_total == 0:
        return 0.0

    return _enrichment_at_offset(u, 0.0, window, n_total)


def max_enrichment(freqs, f0, ratio, window=0.05, n_scan=100):
    """
    Maximum enrichment at ANY position in the lattice coordinate space.

    Scans n_scan evenly-spaced offsets in [0, 1) and returns the maximum.
    This gives each ratio its "best shot" — a fair comparison since φ's
    strength is at noble positions (u≈0.618), not at boundaries (u≈0).
    """
    u = lattice_coordinate(freqs, f0, ratio)
    u = u[np.isfinite(u)]
    n_total = len(u)
    if n_total == 0:
        return 0.0, 0.0

    offsets = np.linspace(0, 1 - 1.0 / n_scan, n_scan)
    best_enrich = -np.inf
    best_offset = 0.0

    for offset in offsets:
        enrich = _enrichment_at_offset(u, offset, window, n_total)
        if enrich > best_enrich:
            best_enrich = enrich
            best_offset = offset

    return best_enrich, best_offset


def phi_full_enrichment(freqs, f0, window=0.05):
    """
    Enrichment at all φ-specific positions (boundary, attractor, nobles).

    Returns dict of position_type -> enrichment_pct.
    This tests the richer φ prediction (8+ positions, not just lattice points).
    """
    u = lattice_coordinate(freqs, f0, PHI)
    u = u[np.isfinite(u)]
    n_total = len(u)
    if n_total == 0:
        return {}

    expected_frac = 2 * window
    results = {}

    for ptype, offset in PHI_POSITION_OFFSETS.items():
        if offset < window:
            in_window = (u < offset + window) | (u > 1 - (window - offset))
        elif offset > 1 - window:
            in_window = (u > offset - window) | (u < window - (1 - offset))
        else:
            in_window = (u >= offset - window) & (u < offset + window)

        n_in = in_window.sum()
        observed_frac = n_in / n_total
        results[ptype] = (observed_frac / expected_frac - 1) * 100

    return results


# ============================================================================
# D1: PHASE-ROTATION NULL
# ============================================================================

def phase_rotation_null(freqs, f0, ratio, n_perm=1000, window=0.05, seed=42,
                         metric='predicted', predicted_offset=None):
    """
    Generate null distribution by circular rotation of lattice coordinates.

    For each permutation:
    1. Compute u = lattice_coordinate(freqs, f0, ratio)
    2. Add uniform random offset θ ~ U(0,1), take mod 1
    3. Compute enrichment on rotated coordinates

    This preserves the peak distribution shape but destroys alignment
    with specific lattice positions.

    Parameters
    ----------
    metric : str
        'predicted' — enrichment at predicted_offset (default: noble_1 for φ,
                      0 for other ratios). This is the principled test: does
                      enrichment at the a priori predicted position exceed null?
        'lattice' — enrichment at u=0 only (lattice-point alignment)
        'max' — maximum enrichment at any position (generous, exploratory)
    predicted_offset : float or None
        For 'predicted' metric, the a priori offset to test.
        If None, defaults to noble_1 (0.618) for φ, or 0.0 otherwise.

    Returns
    -------
    null_enrichments : np.ndarray of shape (n_perm,)
    observed : float
    p_value : float
    z_score : float
    """
    u = lattice_coordinate(freqs, f0, ratio)
    u = u[np.isfinite(u)]
    n_total = len(u)

    # Determine offset for 'predicted' metric
    if predicted_offset is None:
        if metric == 'predicted':
            predicted_offset = PHI_INV if abs(ratio - PHI) < 0.01 else 0.0
        else:
            predicted_offset = 0.0

    # Observed enrichment
    if metric == 'max':
        observed, _ = _max_enrichment_from_u(u, window, n_total)
    elif metric == 'predicted':
        observed = _enrichment_at_offset(u, predicted_offset, window, n_total)
    else:
        observed = _enrichment_at_offset(u, 0.0, window, n_total)

    # Null distribution
    rng = np.random.default_rng(seed)
    null_enrichments = np.empty(n_perm)

    for i in range(n_perm):
        theta = rng.uniform(0, 1)
        u_rotated = (u + theta) % 1.0
        if metric == 'max':
            null_enrichments[i], _ = _max_enrichment_from_u(
                u_rotated, window, n_total
            )
        elif metric == 'predicted':
            null_enrichments[i] = _enrichment_at_offset(
                u_rotated, predicted_offset, window, n_total
            )
        else:
            null_enrichments[i] = _enrichment_at_offset(
                u_rotated, 0.0, window, n_total
            )

    null_mean = null_enrichments.mean()
    null_std = null_enrichments.std()
    z_score = (observed - null_mean) / null_std if null_std > 0 else 0.0
    p_value = (null_enrichments >= observed).sum() / n_perm

    return null_enrichments, observed, p_value, z_score


def _max_enrichment_from_u(u, window, n_total, n_scan=100):
    """Find max enrichment across scanned offsets for pre-computed u values."""
    offsets = np.linspace(0, 1 - 1.0 / n_scan, n_scan)
    best_enrich = -np.inf
    best_offset = 0.0
    for offset in offsets:
        enrich = _enrichment_at_offset(u, offset, window, n_total)
        if enrich > best_enrich:
            best_enrich = enrich
            best_offset = offset
    return best_enrich, best_offset


# ============================================================================
# D2: RATIO SPECIFICITY TEST
# ============================================================================

def ratio_specificity_test(freqs, f0=F0, ratios=None, n_perm=1000,
                            window=0.05, seed=42):
    """
    Test enrichment for each candidate ratio with phase-rotation null.

    Uses the 'predicted' metric: each ratio is tested at its best a priori
    position. For φ, that's the noble_1 position (0.618); for other ratios
    (which lack internal structure predictions), we use lattice alignment
    (offset=0.0). This gives φ credit for its theoretical prediction while
    giving alternatives credit for their natural prediction (lattice alignment).

    Also reports max_enrichment (best position found by scanning) for
    exploratory comparison.

    Returns DataFrame sorted by predicted_enrichment (descending).
    """
    if ratios is None:
        ratios = NAMED_RATIOS

    freqs = np.asarray(freqs, dtype=float)
    freqs = freqs[freqs > 0]

    rows = []
    for name, value in ratios.items():
        if value <= 1.0:
            continue

        # Predicted-position null test
        # φ: test at noble_1 (0.618), others: test at lattice point (0.0)
        is_phi = abs(value - PHI) < 0.01
        pred_offset = PHI_INV if is_phi else 0.0

        null_dist, obs_pred, p_pred, z_pred = phase_rotation_null(
            freqs, f0, value, n_perm=n_perm, window=window, seed=seed,
            metric='predicted', predicted_offset=pred_offset
        )

        # Also get max enrichment (exploratory)
        obs_max, best_offset = max_enrichment(freqs, f0, value, window=window)

        rows.append({
            'ratio_name': name,
            'ratio_value': round(value, 6),
            'predicted_enrichment': round(obs_pred, 2),
            'predicted_offset': round(pred_offset, 3),
            'max_enrichment': round(obs_max, 2),
            'best_offset': round(best_offset, 3),
            'null_mean': round(null_dist.mean(), 2),
            'null_std': round(null_dist.std(), 2),
            'null_95th': round(np.percentile(null_dist, 95), 2),
            'p_value': round(p_pred, 4),
            'z_score': round(z_pred, 2),
        })

    df = pd.DataFrame(rows).sort_values('predicted_enrichment', ascending=False)
    return df.reset_index(drop=True)


# ============================================================================
# D3: F₀ SWEEP WITH NULL
# ============================================================================

def sweep_f0_with_null(freqs, ratio=PHI, f0_range=(6.5, 8.5), step=0.01,
                        n_perm=500, window=0.05, seed=42, metric='max'):
    """
    f₀ sweep for given ratio, with phase-rotation null threshold.

    Returns DataFrame with columns: f0, enrichment, null_95th
    """
    freqs = np.asarray(freqs, dtype=float)
    freqs = freqs[freqs > 0]

    f0_values = np.arange(f0_range[0], f0_range[1] + step, step)

    # Compute global null threshold at midpoint
    mid_f0 = (f0_range[0] + f0_range[1]) / 2
    null_dist, _, _, _ = phase_rotation_null(
        freqs, mid_f0, ratio, n_perm=n_perm, window=window, seed=seed,
        metric=metric
    )
    null_95th = np.percentile(null_dist, 95)

    rows = []
    for f0 in f0_values:
        if metric == 'max':
            enrich, _ = max_enrichment(freqs, f0, ratio, window=window)
        else:
            enrich = lattice_enrichment(freqs, f0, ratio, window=window)
        rows.append({
            'f0': round(f0, 4),
            'enrichment': round(enrich, 2),
            'null_95th': round(null_95th, 2),
        })

    return pd.DataFrame(rows)


# ============================================================================
# D4: 2D (F₀, RATIO) HEATMAP
# ============================================================================

def sweep_f0_ratio_2d(freqs, f0_range=(6.5, 8.5), f0_step=0.02,
                       ratio_range=(1.1, 3.5), ratio_step=0.01,
                       window=0.05, metric='max'):
    """
    Compute enrichment over 2D (f₀ × ratio) grid.

    Parameters
    ----------
    metric : str
        'lattice' — enrichment at u=0 only
        'max' — maximum enrichment at any position (default, fair comparison)

    Returns
    -------
    f0_vals : np.ndarray
    ratio_vals : np.ndarray
    enrichment_matrix : np.ndarray of shape (len(ratio_vals), len(f0_vals))
    """
    freqs = np.asarray(freqs, dtype=float)
    freqs = freqs[freqs > 0]

    f0_vals = np.arange(f0_range[0], f0_range[1] + f0_step, f0_step)
    ratio_vals = np.arange(ratio_range[0], ratio_range[1] + ratio_step, ratio_step)

    enrichment_matrix = np.empty((len(ratio_vals), len(f0_vals)))

    for i, r in enumerate(ratio_vals):
        for j, f0 in enumerate(f0_vals):
            if metric == 'max':
                enrichment_matrix[i, j], _ = max_enrichment(freqs, f0, r, window=window)
            else:
                enrichment_matrix[i, j] = lattice_enrichment(freqs, f0, r, window=window)

    return f0_vals, ratio_vals, enrichment_matrix


def null_threshold_2d(freqs, n_perm=200, window=0.05, percentile=95, seed=42):
    """
    Global null threshold via phase rotation.

    The phase-rotation null is approximately invariant to (f₀, r) because
    rotation destroys all alignment equally. We compute it at a reference
    point and return the scalar threshold.
    """
    null_dist, _, _, _ = phase_rotation_null(
        freqs, F0, PHI, n_perm=n_perm, window=window, seed=seed
    )
    return np.percentile(null_dist, percentile)


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_d1_phase_rotation(observed, null_dist, ratio_name='φ', f0=F0,
                            output_path=None):
    """D1: Histogram of null distribution with observed value marked."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(null_dist, bins=50, color='#95a5a6', alpha=0.7, edgecolor='white',
            label=f'Phase-rotation null (n={len(null_dist)})')

    null_95 = np.percentile(null_dist, 95)
    ax.axvline(null_95, color='#e74c3c', linestyle='--', linewidth=2,
               label=f'95th percentile: {null_95:+.1f}%')
    ax.axvline(observed, color='#2ecc71', linestyle='-', linewidth=3,
               label=f'Observed ({ratio_name}): {observed:+.1f}%')

    p_val = (null_dist >= observed).sum() / len(null_dist)
    z = (observed - null_dist.mean()) / null_dist.std() if null_dist.std() > 0 else 0

    ax.set_xlabel('Lattice Enrichment (%)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'D1: Phase-Rotation Null for {ratio_name} at f₀={f0:.2f} Hz\n'
                 f'p = {p_val:.4f}, z = {z:.2f}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    return fig


def plot_d2_ratio_specificity(results_df, output_path=None):
    """D2: Bar chart of enrichment per ratio with null threshold."""
    fig, ax = plt.subplots(figsize=(12, 6))

    n = len(results_df)
    x = np.arange(n)
    colors = ['#f39c12' if name == 'φ' else '#3498db'
              for name in results_df['ratio_name']]

    bars = ax.bar(x, results_df['predicted_enrichment'], color=colors, alpha=0.8,
                  edgecolor='white', linewidth=0.5)

    # Null 95th percentile line
    null_95 = results_df['null_95th'].median()
    ax.axhline(null_95, color='#e74c3c', linestyle='--', linewidth=2,
               label=f'Null 95th percentile: {null_95:+.1f}%')
    ax.axhline(0, color='black', linewidth=0.5)

    # Add p-value annotations
    for i, row in results_df.iterrows():
        p_str = f'p={row["p_value"]:.3f}' if row['p_value'] >= 0.001 else 'p<0.001'
        y_pos = row['predicted_enrichment']
        va = 'bottom' if y_pos >= 0 else 'top'
        offset = 0.5 if y_pos >= 0 else -0.5
        ax.text(i, y_pos + offset, p_str, ha='center', va=va,
                fontsize=7, rotation=45)

    ax.set_xticks(x)
    labels = [f"{row['ratio_name']}\n({row['ratio_value']:.3f})\n@{row['predicted_offset']:.2f}"
              for _, row in results_df.iterrows()]
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Lattice Enrichment (%)', fontsize=12)
    ax.set_title('D2: Ratio Specificity — Enrichment at Lattice Points\n'
                 '(gold = φ, blue = alternatives)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    return fig


def plot_d3_f0_sweep(sweep_df, ratio_name='φ', output_path=None):
    """D3: f₀ sweep curve with null threshold band."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(sweep_df['f0'], sweep_df['enrichment'], 'b-', linewidth=2,
            label=f'{ratio_name} enrichment')
    ax.fill_between(sweep_df['f0'], sweep_df['null_95th'],
                     alpha=0.2, color='red', label='95th percentile null')
    ax.axhline(sweep_df['null_95th'].iloc[0], color='#e74c3c',
               linestyle='--', linewidth=1, alpha=0.5)

    # Mark key f₀ values
    ax.axvline(7.60, color='green', linestyle='-', linewidth=2, alpha=0.7,
               label='f₀ = 7.60 Hz (paper)')
    ax.axvline(7.83, color='orange', linestyle=':', linewidth=2, alpha=0.7,
               label='f₀ = 7.83 Hz (SR canonical)')

    # Find and mark optimum
    idx_max = sweep_df['enrichment'].idxmax()
    opt_f0 = sweep_df.loc[idx_max, 'f0']
    opt_enrich = sweep_df.loc[idx_max, 'enrichment']
    ax.scatter([opt_f0], [opt_enrich], color='red', s=100, zorder=5, marker='*')
    ax.annotate(f'Max: f₀={opt_f0:.3f}', (opt_f0, opt_enrich),
                textcoords="offset points", xytext=(10, 10), fontsize=9)

    ax.set_xlabel('f₀ (Hz)', fontsize=12)
    ax.set_ylabel('Lattice Enrichment (%)', fontsize=12)
    ax.set_title(f'D3: f₀ Sweep for {ratio_name} with Phase-Rotation Null',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    return fig


def plot_d4_heatmap(f0_vals, ratio_vals, enrichment_matrix,
                     null_threshold=None, output_path=None):
    """D4: 2D heatmap with marked points and optional null contour."""
    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(enrichment_matrix, aspect='auto', origin='lower',
                    extent=[f0_vals[0], f0_vals[-1],
                            ratio_vals[0], ratio_vals[-1]],
                    cmap='RdYlBu_r', interpolation='bilinear')

    cbar = plt.colorbar(im, ax=ax, label='Lattice Enrichment (%)')

    # Null threshold contour
    if null_threshold is not None:
        ax.contour(f0_vals, ratio_vals, enrichment_matrix,
                    levels=[null_threshold], colors='black',
                    linewidths=2, linestyles='--')
        # Add label
        ax.text(f0_vals[0] + 0.05, ratio_vals[-1] - 0.05,
                f'null 95th = {null_threshold:+.1f}%',
                fontsize=9, color='black', fontweight='bold',
                verticalalignment='top')

    # Mark key points
    # (7.6, φ)
    ax.plot(7.60, PHI, 'w*', markersize=15, markeredgecolor='black',
            markeredgewidth=1.5, label=f'(7.60, φ={PHI:.3f})')
    # (7.83, φ)
    ax.plot(7.83, PHI, 'ws', markersize=10, markeredgecolor='black',
            markeredgewidth=1.5, label='(7.83, φ)')

    # Find and mark unconstrained optimum
    max_idx = np.unravel_index(enrichment_matrix.argmax(), enrichment_matrix.shape)
    opt_r = ratio_vals[max_idx[0]]
    opt_f0 = f0_vals[max_idx[1]]
    opt_val = enrichment_matrix[max_idx]
    ax.plot(opt_f0, opt_r, 'r^', markersize=12, markeredgecolor='black',
            markeredgewidth=1.5,
            label=f'Optimum: ({opt_f0:.2f}, {opt_r:.3f}) = {opt_val:+.1f}%')

    # Mark the 12 named ratios on the y-axis
    for name, val in NAMED_RATIOS.items():
        if ratio_vals[0] <= val <= ratio_vals[-1]:
            ax.plot(f0_vals[0] - 0.02, val, 'k<', markersize=5, clip_on=False)
            ax.text(f0_vals[0] - 0.08, val, name, fontsize=7,
                    ha='right', va='center', clip_on=False)

    ax.set_xlabel('f₀ (Hz)', fontsize=12)
    ax.set_ylabel('Scaling Ratio (r)', fontsize=12)
    ax.set_title('D4: 2D Enrichment Landscape — (f₀, r) Parameter Space\n'
                 'Does (7.6, φ) outperform the unconstrained optimum?',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    return fig


# ============================================================================
# SUMMARY REPORT
# ============================================================================

def generate_summary(d1_results, d2_df, d3_df, d4_info, dataset_name=''):
    """Generate human-readable summary of all four analyses."""
    null_dist, observed, p_val, z = d1_results
    f0_vals, ratio_vals, enrich_matrix, null_thresh = d4_info

    # D4 optimum
    max_idx = np.unravel_index(enrich_matrix.argmax(), enrich_matrix.shape)
    opt_r = ratio_vals[max_idx[0]]
    opt_f0 = f0_vals[max_idx[1]]
    opt_val = enrich_matrix[max_idx]

    # φ value at f0=7.6
    phi_row = d2_df[d2_df['ratio_name'] == 'φ']
    phi_enrich = phi_row['predicted_enrichment'].values[0] if len(phi_row) > 0 else 0
    phi_rank = (d2_df['predicted_enrichment'] >= phi_enrich).sum()

    # D3 optimal f0
    d3_opt_idx = d3_df['enrichment'].idxmax()
    d3_opt_f0 = d3_df.loc[d3_opt_idx, 'f0']
    d3_above_null = (d3_df['enrichment'] > d3_df['null_95th']).any()

    lines = [
        f"RATIO SPECIFICITY ANALYSIS — {dataset_name}",
        "=" * 60,
        "",
        "D1: PHASE-ROTATION NULL",
        f"  Observed φ enrichment:  {observed:+.1f}%",
        f"  Null 95th percentile:   {np.percentile(null_dist, 95):+.1f}%",
        f"  p-value:                {p_val:.4f}",
        f"  z-score:                {z:.2f}",
        f"  SURVIVES NULL:          {'YES' if p_val < 0.05 else 'NO'}",
        "",
        "D2: RATIO SPECIFICITY (predicted-position enrichment)",
        f"  φ enrichment @ 0.618:   {phi_enrich:+.1f}%",
        f"  φ rank:                 {phi_rank}/{len(d2_df)}",
        f"  Top ratio:              {d2_df.iloc[0]['ratio_name']} "
        f"({d2_df.iloc[0]['predicted_enrichment']:+.1f}%)",
        "",
        "  Ranking (predicted position: φ@noble_1, others@lattice):",
    ]
    for _, row in d2_df.iterrows():
        marker = " <<<" if row['ratio_name'] == 'φ' else ""
        sig = "*" if row['p_value'] < 0.05 else " "
        lines.append(
            f"    {sig} {row['ratio_name']:>5s} ({row['ratio_value']:.3f}): "
            f"{row['predicted_enrichment']:+7.1f}% @ {row['predicted_offset']:.3f}  "
            f"p={row['p_value']:.3f}  z={row['z_score']:+.1f}{marker}"
        )

    lines += [
        "",
        "D3: f₀ SWEEP (φ)",
        f"  Optimal f₀:            {d3_opt_f0:.3f} Hz",
        f"  Paper f₀:              7.600 Hz",
        f"  Exceeds null:           {'YES' if d3_above_null else 'NO'}",
        "",
        "D4: 2D HEATMAP",
        f"  Unconstrained optimum:  f₀={opt_f0:.2f}, r={opt_r:.3f} ({opt_val:+.1f}%)",
        f"  φ at f₀=7.6:           {phi_enrich:+.1f}%",
        f"  Null threshold:         {null_thresh:+.1f}%",
        "",
        "=" * 60,
    ]

    return "\n".join(lines)

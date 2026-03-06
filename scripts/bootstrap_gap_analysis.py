#!/usr/bin/env python3
"""
Bootstrap Gap Analysis: Irrational vs Rational Structural Score Divergence
==========================================================================

Tests whether the gap G(f₀) = mean(SS_irrationals) - mean(SS_rationals)
is statistically significant using peak-level bootstrap resampling.

The exact permutation test (C(9,4) = 126) gives a finest p ≈ 0.008.
By bootstrapping the underlying peaks, we get:
  1. Bootstrap distribution of G at each f₀ with arbitrary resolution
  2. Bootstrap CIs on G (not limited to 126 permutations)
  3. p-value = fraction of bootstrap resamples where G ≤ 0

Also implements:
  - Continuous regression: correlate SS with Diophantine approximation quality
  - Combined bootstrap + exact permutation for nested significance

Usage:
    python scripts/bootstrap_gap_analysis.py \
        --input exports_peak_distribution/eegmmidb_fooof_nperseg256/golden_ratio_peaks_EEGMMIDB.csv \
        --output-dir exports_peak_distribution/eegmmidb_fooof_nperseg256/ \
        --n-boot 1000 \
        --f0-range 7.0 10.0 0.10
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

sys.path.insert(0, './lib')
sys.path.insert(0, './scripts')

from phi_frequency_model import PHI, F0
from ratio_specificity import lattice_coordinate, _enrichment_at_offset

# =========================================================================
# CONSTANTS
# =========================================================================

BASES = {
    '1.4':   (1.4,   '1.4'),
    'sqrt2': (np.sqrt(2), '√2'),
    '1.5':   (1.5,   '3/2'),
    'phi':   (PHI,   'φ'),
    '1.7':   (1.7,   '1.7'),
    '1.8':   (1.8,   '1.8'),
    '2':     (2.0,   '2'),
    'e':     (np.e,  'e'),
    'pi':    (np.pi, 'π'),
}

# Classification
IRRATIONAL_BASES = {'phi', 'sqrt2', 'e', 'pi'}
RATIONAL_BASES = {'1.4', '1.5', '1.7', '1.8', '2'}

# Diophantine approximation quality (irrationality measure μ)
# Higher = harder to approximate by rationals = "more irrational"
# Rationals have μ = 1, algebraic irrationals μ = 2, most transcendentals μ = 2
# but practical approximation difficulty varies
IRRATIONALITY_MEASURE = {
    '1.4':   1.0,   # rational
    '1.5':   1.0,   # rational
    '1.7':   1.0,   # rational
    '1.8':   1.0,   # rational
    '2':     1.0,   # rational (integer)
    'sqrt2': 2.0,   # algebraic irrational (Roth's theorem)
    'phi':   2.0,   # algebraic irrational (worst approximable by rationals)
    'e':     2.0,   # transcendental (proven μ=2)
    'pi':    2.0,   # transcendental (believed μ=2, unproven)
}

# More useful: practical approximation resistance
# φ is the "most irrational" number — its continued fraction [1;1,1,1,...]
# converges slowest. We use 1/q_n convergence rate as a proxy.
# Lower value = harder to approximate = more irrational
APPROX_QUALITY = {
    '1.4':   0.0,   # exact rational 7/5
    '1.5':   0.0,   # exact rational 3/2
    '1.7':   0.0,   # exact rational 17/10
    '1.8':   0.0,   # exact rational 9/5
    '2':     0.0,   # exact integer
    'sqrt2': 0.7,   # [1;2,2,2,...] — fast convergents
    'e':     0.6,   # [2;1,2,1,1,4,...] — fast convergents
    'pi':    0.5,   # [3;7,15,1,...] — fast convergents (355/113 is very good)
    'phi':   1.0,   # [1;1,1,1,...] — slowest convergents, hardest to approximate
}

POS_TOL = 0.02


# =========================================================================
# CORE FUNCTIONS (from structural_phi_specificity.py)
# =========================================================================

def natural_positions(base):
    """Return dict of position_name -> offset for a given exponential base."""
    inv_b = 1.0 / base
    positions = {'boundary': 0.0, 'attractor': 0.5}

    candidates = [
        ('noble', inv_b),
        ('noble_2', inv_b ** 2),
        ('inv_noble', 1.0 - inv_b),
        ('inv_noble_2', 1.0 - inv_b ** 2),
    ]

    for name, val in candidates:
        if val < POS_TOL or val > 1.0 - POS_TOL:
            continue
        if any(abs(val - existing) < POS_TOL for existing in positions.values()):
            continue
        positions[name] = val

    return positions


def compute_structural_score(u, positions, window=0.05):
    """
    Compute structural score = -boundary + attractor + mean(nobles).
    """
    n_total = len(u)
    if n_total == 0:
        return 0.0

    enrichments = {}
    for name, offset in positions.items():
        enrichments[name] = _enrichment_at_offset(u, offset, window, n_total)

    boundary_e = enrichments.get('boundary', 0.0)
    attractor_e = enrichments.get('attractor', 0.0)
    noble_keys = [k for k in enrichments if k not in ('boundary', 'attractor')]
    noble_mean = np.mean([enrichments[k] for k in noble_keys]) if noble_keys else 0.0

    return -boundary_e + attractor_e + noble_mean


def compute_all_ss(freqs, f0, bases, window=0.05):
    """Compute structural score for all bases at a given f0."""
    scores = {}
    for base_key, (base_val, _) in bases.items():
        u = lattice_coordinate(freqs, f0, base_val)
        u = u[np.isfinite(u)]
        positions = natural_positions(base_val)
        scores[base_key] = compute_structural_score(u, positions, window)
    return scores


def compute_gap(scores, irrational_keys, rational_keys):
    """G = mean(SS_irrationals) - mean(SS_rationals)."""
    irr_vals = [scores[k] for k in irrational_keys if k in scores]
    rat_vals = [scores[k] for k in rational_keys if k in scores]
    return np.mean(irr_vals) - np.mean(rat_vals)


# =========================================================================
# BOOTSTRAP GAP TEST
# =========================================================================

def bootstrap_gap_test(freqs, f0_values, bases, window=0.05,
                       n_boot=1000, seed=42, verbose=True):
    """
    Peak-level bootstrap test for irrational vs rational gap.

    For each bootstrap resample of the peaks:
      1. Recompute all 9 SS values at each f0
      2. Compute G = mean(SS_irr) - mean(SS_rat)

    Returns DataFrame with columns:
      f0, G_observed, G_boot_mean, G_boot_std, G_ci_lo, G_ci_hi,
      p_boot (fraction of G_boot <= 0), n_boot
    """
    rng = np.random.default_rng(seed)
    n_peaks = len(freqs)
    n_f0 = len(f0_values)

    irr_keys = sorted(IRRATIONAL_BASES & set(bases.keys()))
    rat_keys = sorted(RATIONAL_BASES & set(bases.keys()))

    if verbose:
        print(f"\n{'='*80}")
        print(f"PEAK-LEVEL BOOTSTRAP GAP TEST")
        print(f"{'='*80}")
        print(f"  Peaks: {n_peaks:,}")
        print(f"  f0 values: {n_f0} ({f0_values[0]:.2f} to {f0_values[-1]:.2f} Hz)")
        print(f"  Bootstraps: {n_boot}")
        print(f"  Irrational bases ({len(irr_keys)}): {irr_keys}")
        print(f"  Rational bases ({len(rat_keys)}): {rat_keys}")

    # Precompute observed G at each f0
    observed_G = np.empty(n_f0)
    observed_scores = []  # Store for display
    for i, f0 in enumerate(f0_values):
        scores = compute_all_ss(freqs, f0, bases, window)
        observed_G[i] = compute_gap(scores, irr_keys, rat_keys)
        observed_scores.append(scores)

    # Precompute base positions (these don't change with bootstrap)
    base_positions = {}
    for base_key, (base_val, _) in bases.items():
        base_positions[base_key] = natural_positions(base_val)

    # Precompute log(base) for each base
    log_bases = {}
    for base_key, (base_val, _) in bases.items():
        log_bases[base_key] = np.log(base_val)

    # Bootstrap
    boot_G = np.empty((n_boot, n_f0))
    t0 = time.time()

    for b in range(n_boot):
        if verbose and (b % 50 == 0 or b == n_boot - 1):
            elapsed = time.time() - t0
            rate = (b + 1) / elapsed if elapsed > 0 else 0
            eta = (n_boot - b - 1) / rate if rate > 0 else 0
            print(f"  Bootstrap {b+1}/{n_boot} "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)", end='\r')

        # Resample peaks
        boot_idx = rng.choice(n_peaks, size=n_peaks, replace=True)
        boot_freqs = freqs[boot_idx]

        # Precompute log(boot_freqs) once per resample
        log_boot = np.log(boot_freqs)

        for i, f0 in enumerate(f0_values):
            log_f0 = np.log(f0)

            irr_ss = []
            rat_ss = []

            for base_key, (base_val, _) in bases.items():
                # Fast lattice coordinate: (log(f) - log(f0)) / log(base) mod 1
                u = ((log_boot - log_f0) / log_bases[base_key]) % 1.0
                positions = base_positions[base_key]
                ss = compute_structural_score(u, positions, window)

                if base_key in IRRATIONAL_BASES:
                    irr_ss.append(ss)
                else:
                    rat_ss.append(ss)

            boot_G[b, i] = np.mean(irr_ss) - np.mean(rat_ss)

    elapsed = time.time() - t0
    if verbose:
        print(f"\n  Completed {n_boot} bootstraps in {elapsed:.0f}s "
              f"({elapsed/n_boot:.1f}s per bootstrap)")

    # Compile results
    results = []
    for i, f0 in enumerate(f0_values):
        G_obs = observed_G[i]
        G_boots = boot_G[:, i]
        G_mean = G_boots.mean()
        G_std = G_boots.std()
        ci_lo = np.percentile(G_boots, 2.5)
        ci_hi = np.percentile(G_boots, 97.5)
        # p-value: fraction of bootstrap G ≤ 0 (one-sided test for positive gap)
        p_boot = (G_boots <= 0).sum() / n_boot

        results.append({
            'f0': f0,
            'G_observed': G_obs,
            'G_boot_mean': G_mean,
            'G_boot_std': G_std,
            'G_ci_lo': ci_lo,
            'G_ci_hi': ci_hi,
            'p_boot': p_boot,
            'ci_excludes_zero': ci_lo > 0,
            'n_boot': n_boot,
        })

    df = pd.DataFrame(results)

    if verbose:
        print(f"\n{'='*70}")
        print(f"  {'f0':>5s}  {'G_obs':>7s}  {'G_boot':>7s}  "
              f"{'95% CI':>20s}  {'p_boot':>7s}  {'sig':>5s}")
        print(f"  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*20}  {'-'*7}  {'-'*5}")
        for _, r in df.iterrows():
            sig = "***" if r['p_boot'] < 0.001 else \
                  "**" if r['p_boot'] < 0.01 else \
                  "*" if r['p_boot'] < 0.05 else ""
            print(f"  {r['f0']:>5.2f}  {r['G_observed']:>+7.1f}  "
                  f"{r['G_boot_mean']:>+7.1f}  "
                  f"[{r['G_ci_lo']:>+7.1f}, {r['G_ci_hi']:>+6.1f}]  "
                  f"{r['p_boot']:>7.3f}  {sig:>5s}")

    return df, boot_G


# =========================================================================
# EXACT PERMUTATION (for comparison)
# =========================================================================

def exact_permutation_gap(scores_dict, irrational_keys, rational_keys):
    """
    Exact permutation test: all C(9,4) = 126 ways to assign 4 of 9 bases
    to the "irrational" group.

    Returns (G_observed, p_value, null_distribution).
    """
    from itertools import combinations

    all_keys = sorted(set(irrational_keys) | set(rational_keys))
    all_scores = np.array([scores_dict[k] for k in all_keys])
    n_total = len(all_keys)
    n_irr = len(irrational_keys)

    # Observed G
    irr_idx = [all_keys.index(k) for k in irrational_keys]
    rat_idx = [all_keys.index(k) for k in rational_keys]
    G_obs = all_scores[irr_idx].mean() - all_scores[rat_idx].mean()

    # All permutations
    null_G = []
    for combo in combinations(range(n_total), n_irr):
        combo_set = set(combo)
        other = [j for j in range(n_total) if j not in combo_set]
        g = all_scores[list(combo)].mean() - all_scores[other].mean()
        null_G.append(g)

    null_G = np.array(null_G)
    p_value = (null_G >= G_obs).sum() / len(null_G)

    return G_obs, p_value, null_G


# =========================================================================
# CONTINUOUS REGRESSION TEST
# =========================================================================

def continuous_irrationality_test(scores_dict, f0_values_scores, bases,
                                  verbose=True):
    """
    Correlate SS with continuous approximation quality measure.

    Uses Spearman rank correlation (robust to outliers) and
    permutation-based p-value.
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"CONTINUOUS IRRATIONALITY REGRESSION TEST")
        print(f"{'='*80}")
        print(f"  Approximation quality metric: φ=1.0 (hardest), rationals=0.0")

    base_keys = sorted(bases.keys())
    approx_vals = np.array([APPROX_QUALITY[k] for k in base_keys])

    results = []
    for f0, scores in zip(f0_values_scores, [f0_scores for f0_scores in f0_values_scores]):
        pass  # placeholder, will be filled by caller

    return results


def run_regression_at_f0(scores_dict, verbose=False):
    """
    Run Spearman correlation between SS and approximation quality
    at a single f0.
    """
    base_keys = sorted(scores_dict.keys())
    ss_vals = np.array([scores_dict[k] for k in base_keys])
    approx_vals = np.array([APPROX_QUALITY[k] for k in base_keys])

    # Spearman correlation
    rho, p_spearman = sp_stats.spearmanr(approx_vals, ss_vals)

    # Pearson correlation
    r, p_pearson = sp_stats.pearsonr(approx_vals, ss_vals)

    return {
        'spearman_rho': rho,
        'spearman_p': p_spearman,
        'pearson_r': r,
        'pearson_p': p_pearson,
    }


# =========================================================================
# VISUALIZATION
# =========================================================================

def plot_bootstrap_gap(results_df, output_path, exact_p_df=None):
    """
    Plot G(f₀) with bootstrap CIs and significance markers.

    Two panels:
      Top: G with 95% CI band, significance highlighted
      Bottom: Bootstrap p-value across f₀
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1],
                              sharex=True)
    fig.suptitle('Irrational vs Rational Gap: Peak-Level Bootstrap Test',
                 fontsize=14, fontweight='bold')

    f0 = results_df['f0'].values
    G = results_df['G_observed'].values
    ci_lo = results_df['G_ci_lo'].values
    ci_hi = results_df['G_ci_hi'].values
    p_boot = results_df['p_boot'].values

    # --- Top panel: G with CI ---
    ax = axes[0]

    # CI band
    ax.fill_between(f0, ci_lo, ci_hi, alpha=0.25, color='steelblue',
                    label='95% bootstrap CI')

    # Observed G
    ax.plot(f0, G, 'o-', color='steelblue', linewidth=2, markersize=4,
            label='G(f₀) observed')

    # Zero line
    ax.axhline(0, color='black', linewidth=1, linestyle='-', alpha=0.5)

    # Highlight significant regions (CI excludes zero)
    sig_mask = ci_lo > 0
    if sig_mask.any():
        # Find contiguous significant regions
        changes = np.diff(sig_mask.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1
        if sig_mask[0]:
            starts = np.r_[0, starts]
        if sig_mask[-1]:
            ends = np.r_[ends, len(sig_mask)]
        for s, e in zip(starts, ends):
            ax.axvspan(f0[s], f0[min(e, len(f0)-1)], alpha=0.15,
                       color='green', zorder=0)
        # Legend-only proxy (avoids axvspan(0,0) which extends x-axis to 0)
        from matplotlib.patches import Rectangle
        ax.legend_proxy_ci = Rectangle((0, 0), 0, 0, fc='green', alpha=0.15,
                                        label='CI excludes 0')
        ax.add_patch(ax.legend_proxy_ci)
        ax.legend_proxy_ci.set_visible(False)

    # Add exact permutation p-values if provided
    if exact_p_df is not None:
        ax2 = ax.twinx()
        ax2.plot(exact_p_df['f0'], exact_p_df['exact_p'], 's--',
                 color='red', alpha=0.5, markersize=3, linewidth=1,
                 label='Exact perm p (C(9,4)=126)')
        ax2.set_ylabel('Exact permutation p-value', color='red', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(0, 0.5)
        ax2.legend(loc='upper right', fontsize=9)

    ax.set_ylabel('Gap G = mean(SS_irr) − mean(SS_rat)', fontsize=11)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Constrain x-axis to data range (avoids dead space)
    margin = (f0[-1] - f0[0]) * 0.03
    ax.set_xlim(f0[0] - margin, f0[-1] + margin)

    # Mark peak gap
    peak_idx = np.argmax(G)
    ax.annotate(f'Peak: G={G[peak_idx]:+.1f}\nf₀={f0[peak_idx]:.2f} Hz',
                xy=(f0[peak_idx], G[peak_idx]),
                xytext=(f0[peak_idx] + 0.3, G[peak_idx] + 5),
                fontsize=9, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='steelblue'),
                bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow'))

    # --- Bottom panel: p-value ---
    ax = axes[1]
    ax.plot(f0, p_boot, 'o-', color='darkred', linewidth=1.5, markersize=3)
    ax.axhline(0.05, color='red', linestyle='--', linewidth=1, alpha=0.7,
               label='α = 0.05')
    ax.axhline(0.01, color='red', linestyle=':', linewidth=1, alpha=0.5,
               label='α = 0.01')
    ax.set_ylabel('Bootstrap p-value', fontsize=11)
    ax.set_xlabel('Anchor frequency f₀ (Hz)', fontsize=11)
    ax.set_ylim(-0.02, max(0.5, p_boot.max() * 1.1))
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Highlight significant region
    sig_mask_p = p_boot < 0.05
    if sig_mask_p.any():
        sig_f0 = f0[sig_mask_p]
        ax.fill_between(f0, 0, p_boot, where=sig_mask_p,
                        alpha=0.2, color='green')

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Figure saved: {output_path}")


def plot_per_base_trajectories(sweep_df, output_path, bases):
    """
    Plot SS trajectory for each base across f₀, colored by class.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle('Structural Score Trajectories by Base Class',
                 fontsize=14, fontweight='bold')

    f0_values = sorted(sweep_df['f0'].unique())

    # Plot each base
    for base_key, (_, base_label) in bases.items():
        sub = sweep_df[sweep_df['base_name'] == base_key].sort_values('f0')
        is_irr = base_key in IRRATIONAL_BASES
        color = 'steelblue' if is_irr else 'firebrick'
        linestyle = '-' if is_irr else '--'
        linewidth = 2.5 if base_key == 'phi' else 1.5
        alpha = 1.0 if is_irr else 0.6
        marker = 'o' if base_key == 'phi' else ('s' if is_irr else '^')

        ax.plot(sub['f0'], sub['structural_score'],
                marker=marker, markersize=3 if base_key != 'phi' else 5,
                linewidth=linewidth, linestyle=linestyle,
                color=color, alpha=alpha, label=f'{base_label} ({"irr" if is_irr else "rat"})')

    # Add class means
    for f0 in f0_values:
        sub = sweep_df[sweep_df['f0'] == f0]
        irr_mean = sub[sub['base_name'].isin(IRRATIONAL_BASES)]['structural_score'].mean()
        rat_mean = sub[sub['base_name'].isin(RATIONAL_BASES)]['structural_score'].mean()
        if f0 == f0_values[0]:
            ax.plot(f0, irr_mean, 'D', color='navy', markersize=6,
                    label='Irrational mean')
            ax.plot(f0, rat_mean, 'D', color='darkred', markersize=6,
                    label='Rational mean')
        else:
            ax.plot(f0, irr_mean, 'D', color='navy', markersize=6)
            ax.plot(f0, rat_mean, 'D', color='darkred', markersize=6)

    ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Anchor frequency f₀ (Hz)', fontsize=11)
    ax.set_ylabel('Structural Score (SS)', fontsize=11)
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Trajectory figure saved: {output_path}")


# =========================================================================
# LINEAR GRID CONTROL (non-exponential baseline)
# =========================================================================

# Linear step sizes (Hz) to test
LINEAR_STEPS = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0]


def linear_lattice_coordinate(freqs, f0, step_k):
    """
    Linear lattice coordinate: u = ((f - f₀) / k) mod 1.0

    For exponential bases, u = (log(f/f₀) / log(b)) mod 1.0
    For linear grids, u = ((f - f₀) / k) mod 1.0

    Both give uniform [0,1) under null hypothesis.
    """
    u = ((freqs - f0) / step_k) % 1.0
    return u


def compute_linear_grid_scores(freqs, f0, k_values, window=0.05):
    """
    Compute structural scores for linear grids f₀ + n×k.

    Uses the same enrichment framework as exponential bases:
      - boundary (u=0): grid positions — enrichment here means peaks cluster on-grid
      - attractor (u=0.5): midpoints between grid positions
      - SS_linear = -boundary + attractor (same formula, no noble positions)

    For fair comparison, also computes SS_minimal for exponential bases
    (boundary + attractor only, without nobles).
    """
    results = []
    for k in k_values:
        u = linear_lattice_coordinate(freqs, f0, k)
        u = u[np.isfinite(u)]
        n_total = len(u)

        boundary_e = _enrichment_at_offset(u, 0.0, window, n_total)
        attractor_e = _enrichment_at_offset(u, 0.5, window, n_total)

        # Same formula as exponential: SS = -boundary + attractor
        # (no noble positions for linear grids)
        ss_linear = -boundary_e + attractor_e

        # Count predicted positions in analysis range for reference
        f_min, f_max = freqs.min(), freqs.max()
        n_min = max(0, int(np.ceil((f_min - f0) / k)))
        n_max = int(np.floor((f_max - f0) / k))
        n_predicted = max(0, n_max - n_min + 1)

        results.append({
            'model': 'linear',
            'label': f'k={k:.0f}Hz',
            'step_k': k,
            'grid_enrichment': boundary_e,
            'midpoint_enrichment': attractor_e,
            'structural_score': ss_linear,
            'n_peaks': n_total,
            'n_predicted_positions': n_predicted,
        })

    return pd.DataFrame(results)


def compute_exponential_minimal_scores(freqs, f0, bases, window=0.05):
    """
    Compute boundary-only + attractor-only SS for exponential bases,
    for direct comparison with linear grids (no noble advantage).
    """
    results = []
    for base_key, (base_val, base_label) in bases.items():
        u = lattice_coordinate(freqs, f0, base_val)
        u = u[np.isfinite(u)]
        n_total = len(u)

        boundary_e = _enrichment_at_offset(u, 0.0, window, n_total)
        attractor_e = _enrichment_at_offset(u, 0.5, window, n_total)

        # Full structural score (with nobles)
        positions = natural_positions(base_val)
        ss_full = compute_structural_score(u, positions, window)

        # Minimal score (boundary + attractor only, same as linear)
        ss_minimal = -boundary_e + attractor_e

        results.append({
            'model': 'exponential',
            'label': base_label,
            'base_key': base_key,
            'base_value': base_val,
            'grid_enrichment': boundary_e,
            'midpoint_enrichment': attractor_e,
            'structural_score_full': ss_full,
            'structural_score_minimal': ss_minimal,
            'n_peaks': n_total,
            'is_irrational': base_key in IRRATIONAL_BASES,
        })

    return pd.DataFrame(results)


def linear_grid_f0_sweep(freqs, f0_values, k_values, window=0.05):
    """
    Sweep f₀ for linear grids — compute SS at each (f₀, k) pair.
    """
    results = []
    for f0 in f0_values:
        for k in k_values:
            u = linear_lattice_coordinate(freqs, f0, k)
            u = u[np.isfinite(u)]
            n_total = len(u)

            boundary_e = _enrichment_at_offset(u, 0.0, window, n_total)
            attractor_e = _enrichment_at_offset(u, 0.5, window, n_total)
            ss = -boundary_e + attractor_e

            results.append({
                'f0': f0,
                'step_k': k,
                'label': f'k={k:.0f}Hz',
                'grid_enrichment': boundary_e,
                'midpoint_enrichment': attractor_e,
                'structural_score': ss,
            })

    return pd.DataFrame(results)


def plot_linear_vs_exponential(linear_df, exp_df, output_path,
                                linear_sweep_df=None, f0_highlight=None):
    """
    Comparative plot: linear grid SS vs exponential base SS.

    Panel layout:
      Top: Bar chart of SS at peak f₀ — linear grids vs exponential bases
      Bottom: f₀ sweep showing linear grid SS range vs exponential SS range
    """
    has_sweep = linear_sweep_df is not None

    n_panels = 2 if has_sweep else 1
    fig, axes = plt.subplots(n_panels, 1,
                              figsize=(14, 5 * n_panels),
                              squeeze=False)

    # --- Panel A: Bar comparison at single f₀ ---
    ax = axes[0, 0]

    # Combine for plotting
    linear_labels = linear_df['label'].values
    linear_ss = linear_df['structural_score'].values

    exp_labels = exp_df['label'].values
    exp_ss_minimal = exp_df['structural_score_minimal'].values
    exp_ss_full = exp_df['structural_score_full'].values
    exp_irr = exp_df['is_irrational'].values

    n_lin = len(linear_labels)
    n_exp = len(exp_labels)

    x_lin = np.arange(n_lin)
    x_exp = np.arange(n_lin + 1, n_lin + 1 + n_exp)

    # Linear bars (gray)
    ax.bar(x_lin, linear_ss, color='gray', alpha=0.7, edgecolor='black',
           linewidth=0.5, label='Linear grid (SS = -bnd + att)')

    # Exponential bars — minimal SS (for fair comparison)
    colors_exp = ['steelblue' if irr else 'firebrick' for irr in exp_irr]
    ax.bar(x_exp, exp_ss_minimal, color=colors_exp, alpha=0.5, edgecolor='black',
           linewidth=0.5, label='Exponential (minimal: -bnd + att)')

    # Exponential full SS as diamonds
    ax.scatter(x_exp, exp_ss_full, marker='D', s=40, color=colors_exp,
               edgecolors='black', linewidths=0.5, zorder=5,
               label='Exponential (full SS with nobles)')

    ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)

    all_labels = list(linear_labels) + [''] + list(exp_labels)
    all_x = list(x_lin) + [n_lin] + list(x_exp)
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Structural Score', fontsize=11)
    ax.set_title('Linear Grid Control: Non-Exponential vs Exponential Models',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Add mean lines
    lin_mean = linear_ss.mean()
    exp_min_mean = exp_ss_minimal.mean()
    ax.axhline(lin_mean, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(exp_min_mean, color='purple', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(0.02, 0.95, f'Linear mean: {lin_mean:.1f}',
            transform=ax.transAxes, fontsize=9, color='gray',
            verticalalignment='top')
    ax.text(0.02, 0.88, f'Exponential minimal mean: {exp_min_mean:.1f}',
            transform=ax.transAxes, fontsize=9, color='purple',
            verticalalignment='top')

    # --- Panel B: f₀ sweep (if available) ---
    if has_sweep:
        ax = axes[1, 0]

        # Linear grid envelope
        sweep_pivot = linear_sweep_df.pivot_table(
            index='f0', columns='label', values='structural_score')
        f0s = sweep_pivot.index.values
        lin_max = sweep_pivot.max(axis=1).values
        lin_min = sweep_pivot.min(axis=1).values
        lin_mean_sweep = sweep_pivot.mean(axis=1).values

        ax.fill_between(f0s, lin_min, lin_max, alpha=0.2, color='gray',
                        label='Linear grid range')
        ax.plot(f0s, lin_mean_sweep, '-', color='gray', linewidth=2,
                label='Linear grid mean')

        ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)

        if f0_highlight is not None:
            ax.axvline(f0_highlight, color='orange', linestyle='--',
                       linewidth=1, alpha=0.7, label=f'Peak f₀={f0_highlight:.1f}')

        ax.set_xlabel('Anchor frequency f₀ (Hz)', fontsize=11)
        ax.set_ylabel('Structural Score', fontsize=11)
        ax.set_title('Linear Grid SS Across f₀ Sweep', fontsize=12)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Linear control figure saved: {output_path}")


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Bootstrap gap analysis: irrational vs rational divergence')
    parser.add_argument('--input', type=str, required=True,
                        help='Input peaks CSV')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--n-boot', type=int, default=1000,
                        help='Number of bootstrap resamples (default: 1000)')
    parser.add_argument('--window', type=float, default=0.05,
                        help='Half-window for enrichment (default: 0.05)')
    parser.add_argument('--f0-range', nargs=3, type=float, default=None,
                        metavar=('START', 'END', 'STEP'),
                        help='f0 sweep range: start end step (default: use existing sweep CSV)')
    parser.add_argument('--sweep-csv', type=str, default=None,
                        help='Existing f0 sweep CSV (to extract f0 values and exact perm results)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--linear-control', action='store_true',
                        help='Run linear grid control analysis (non-exponential baseline)')
    parser.add_argument('--linear-only', action='store_true',
                        help='Run ONLY the linear grid control (skip bootstrap/permutation)')
    parser.add_argument('--linear-steps', nargs='+', type=float, default=None,
                        help='Linear step sizes in Hz (default: 3 4 5 6 7 8 9 10 12)')
    args = parser.parse_args()

    # --linear-only implies --linear-control
    if args.linear_only:
        args.linear_control = True

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("BOOTSTRAP GAP ANALYSIS: Irrational vs Rational Divergence")
    print("=" * 80)

    # Load peaks
    print(f"\nLoading peaks from: {args.input}")
    peaks_df = pd.read_csv(args.input)
    freqs = peaks_df['freq'].values if 'freq' in peaks_df.columns else peaks_df['frequency'].values
    freqs = freqs[np.isfinite(freqs) & (freqs > 0)]
    print(f"  {len(freqs):,} valid peaks")

    # Determine f0 values
    if args.f0_range:
        start, end, step = args.f0_range
        f0_values = np.arange(start, end + step/2, step)
        f0_values = np.round(f0_values, 4)
    elif args.sweep_csv:
        sweep = pd.read_csv(args.sweep_csv)
        f0_values = np.sort(sweep['f0'].unique())
    else:
        # Default: 7.0 to 10.0 at 0.10 Hz
        f0_values = np.arange(7.0, 10.05, 0.10)
        f0_values = np.round(f0_values, 2)

    print(f"  f0 sweep: {len(f0_values)} values from {f0_values[0]:.2f} to {f0_values[-1]:.2f} Hz")

    k_values = args.linear_steps or LINEAR_STEPS

    # --- If linear-only mode, skip to linear control ---
    if args.linear_only:
        print(f"\n{'='*70}")
        print("LINEAR GRID CONTROL ANALYSIS (non-exponential baseline)")
        print(f"{'='*70}")
        print(f"  Linear step sizes: {k_values} Hz")

        # Find peak f₀ from existing bootstrap results or use midpoint
        peak_f0 = np.median(f0_values)
        existing_boot = os.path.join(args.output_dir, 'bootstrap_gap_analysis.csv')
        if os.path.exists(existing_boot):
            prev = pd.read_csv(existing_boot)
            peak_f0 = prev.loc[prev['G_observed'].idxmax(), 'f0']
            print(f"  Using peak f₀ = {peak_f0:.2f} Hz from existing bootstrap results")
        else:
            print(f"  Using median f₀ = {peak_f0:.2f} Hz (no existing bootstrap results)")

        # Compute at peak f₀
        print(f"\n  Computing linear grid scores at f₀ = {peak_f0:.2f} Hz...")
        linear_df = compute_linear_grid_scores(freqs, peak_f0, k_values, args.window)
        exp_df = compute_exponential_minimal_scores(freqs, peak_f0, BASES, args.window)

        # Display results
        print(f"\n  {'Model':>12s}  {'Label':>8s}  {'Grid Enr':>9s}  "
              f"{'Mid Enr':>9s}  {'SS':>8s}")
        print(f"  {'-'*12}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*8}")
        for _, r in linear_df.iterrows():
            print(f"  {'linear':>12s}  {r['label']:>8s}  {r['grid_enrichment']:>+9.1f}  "
                  f"{r['midpoint_enrichment']:>+9.1f}  {r['structural_score']:>+8.1f}")
        print()
        for _, r in exp_df.iterrows():
            irr = "irr" if r['is_irrational'] else "rat"
            print(f"  {'exponential':>12s}  {r['label']:>8s}  {r['grid_enrichment']:>+9.1f}  "
                  f"{r['midpoint_enrichment']:>+9.1f}  {r['structural_score_minimal']:>+8.1f}  "
                  f"(full: {r['structural_score_full']:>+8.1f}) [{irr}]")

        # Summary statistics
        lin_ss = linear_df['structural_score'].values
        exp_ss_min = exp_df['structural_score_minimal'].values
        exp_ss_full = exp_df['structural_score_full'].values
        print(f"\n  Linear grid SS:  mean={lin_ss.mean():+.1f}, "
              f"range=[{lin_ss.min():+.1f}, {lin_ss.max():+.1f}]")
        print(f"  Exponential SS (minimal): mean={exp_ss_min.mean():+.1f}, "
              f"range=[{exp_ss_min.min():+.1f}, {exp_ss_min.max():+.1f}]")
        print(f"  Exponential SS (full):    mean={exp_ss_full.mean():+.1f}, "
              f"range=[{exp_ss_full.min():+.1f}, {exp_ss_full.max():+.1f}]")

        # Mann-Whitney U test: linear vs exponential (minimal)
        from scipy.stats import mannwhitneyu, rankdata
        if len(lin_ss) >= 3 and len(exp_ss_min) >= 3:
            u_stat, u_p = mannwhitneyu(exp_ss_min, lin_ss, alternative='greater')
            print(f"\n  Mann-Whitney U (exp_minimal > linear): U={u_stat:.0f}, p={u_p:.4f}")

        # f₀ sweep for linear grids
        print(f"\n  Computing linear grid f₀ sweep ({len(f0_values)} values)...")
        linear_sweep_df = linear_grid_f0_sweep(freqs, f0_values, k_values, args.window)

        # Save results
        linear_path = os.path.join(args.output_dir, 'linear_grid_control.csv')
        linear_df.to_csv(linear_path, index=False)
        print(f"  Linear grid results saved: {linear_path}")

        exp_path = os.path.join(args.output_dir, 'exponential_minimal_scores.csv')
        exp_df.to_csv(exp_path, index=False)
        print(f"  Exponential minimal scores saved: {exp_path}")

        sweep_path = os.path.join(args.output_dir, 'linear_grid_f0_sweep.csv')
        linear_sweep_df.to_csv(sweep_path, index=False)
        print(f"  Linear grid sweep saved: {sweep_path}")

        # Plot
        fig_path = os.path.join(args.output_dir, 'linear_grid_control.png')
        plot_linear_vs_exponential(linear_df, exp_df, fig_path,
                                    linear_sweep_df=linear_sweep_df,
                                    f0_highlight=peak_f0)

        print(f"\nDone (linear control only).")
        return

    # --- Analysis 1: Exact permutation at each f0 (fast, for comparison) ---
    print(f"\n{'='*70}")
    print("ANALYSIS 1: Exact permutation test (C(9,4) = 126)")
    print(f"{'='*70}")

    irr_keys = sorted(IRRATIONAL_BASES & set(BASES.keys()))
    rat_keys = sorted(RATIONAL_BASES & set(BASES.keys()))

    exact_results = []
    for f0 in f0_values:
        scores = compute_all_ss(freqs, f0, BASES, args.window)
        G_obs, p_exact, _ = exact_permutation_gap(scores, irr_keys, rat_keys)
        exact_results.append({
            'f0': f0,
            'G_observed': G_obs,
            'exact_p': p_exact,
        })

    exact_df = pd.DataFrame(exact_results)
    sig_exact = exact_df[exact_df['exact_p'] < 0.05]
    print(f"\n  Significant at α=0.05: {len(sig_exact)}/{len(exact_df)} f0 values")
    if len(sig_exact) > 0:
        print(f"  Range: f0 = {sig_exact['f0'].min():.2f} to {sig_exact['f0'].max():.2f} Hz")
        peak_row = exact_df.loc[exact_df['G_observed'].idxmax()]
        print(f"  Peak gap: G={peak_row['G_observed']:+.1f} at f0={peak_row['f0']:.2f} Hz "
              f"(p={peak_row['exact_p']:.4f})")

    # --- Analysis 2: Peak-level bootstrap (main result) ---
    boot_df, boot_G_matrix = bootstrap_gap_test(
        freqs, f0_values, BASES, window=args.window,
        n_boot=args.n_boot, seed=args.seed)

    # --- Analysis 3: Continuous regression at each f0 ---
    print(f"\n{'='*70}")
    print("ANALYSIS 3: Continuous irrationality correlation")
    print(f"{'='*70}")

    regression_results = []
    for f0 in f0_values:
        scores = compute_all_ss(freqs, f0, BASES, args.window)
        reg = run_regression_at_f0(scores)
        reg['f0'] = f0
        regression_results.append(reg)

    reg_df = pd.DataFrame(regression_results)
    sig_reg = reg_df[reg_df['spearman_p'] < 0.05]
    print(f"\n  Spearman correlation (SS vs approximation quality):")
    print(f"  Significant at α=0.05: {len(sig_reg)}/{len(reg_df)} f0 values")
    if len(sig_reg) > 0:
        best = reg_df.loc[reg_df['spearman_rho'].idxmax()]
        print(f"  Peak correlation: ρ={best['spearman_rho']:.3f} at f0={best['f0']:.2f} Hz "
              f"(p={best['spearman_p']:.4f})")

    # --- Merge results ---
    merged = boot_df.merge(exact_df[['f0', 'exact_p']], on='f0', how='left')
    merged = merged.merge(reg_df[['f0', 'spearman_rho', 'spearman_p']], on='f0', how='left')

    # --- Summary ---
    print(f"\n{'='*70}")
    print("COMPARISON: Exact Permutation vs Bootstrap p-values")
    print(f"{'='*70}")
    print(f"  {'f0':>5s}  {'G':>7s}  {'exact_p':>8s}  {'boot_p':>8s}  "
          f"{'CI':>20s}  {'ρ':>5s}")
    for _, r in merged.iterrows():
        sig_e = "*" if r['exact_p'] < 0.05 else " "
        sig_b = "*" if r['p_boot'] < 0.05 else " "
        sig_r = "*" if r['spearman_p'] < 0.05 else " "
        print(f"  {r['f0']:>5.2f}  {r['G_observed']:>+7.1f}  "
              f"{r['exact_p']:>7.3f}{sig_e}  {r['p_boot']:>7.3f}{sig_b}  "
              f"[{r['G_ci_lo']:>+7.1f}, {r['G_ci_hi']:>+6.1f}]  "
              f"{r['spearman_rho']:>+5.2f}{sig_r}")

    # --- Key finding: where bootstrap gives finer p than exact ---
    print(f"\n{'='*70}")
    print("KEY FINDING: Bootstrap resolution advantage")
    print(f"{'='*70}")

    # Count where boot gives p < 0.008 (finer than exact can resolve)
    ultra_sig = merged[merged['p_boot'] < 0.008]
    print(f"  f0 values with p_boot < 0.008 (beyond exact resolution): "
          f"{len(ultra_sig)}/{len(merged)}")
    if len(ultra_sig) > 0:
        for _, r in ultra_sig.iterrows():
            print(f"    f0={r['f0']:.2f}  G={r['G_observed']:+.1f}  "
                  f"p_boot={r['p_boot']:.4f}  exact_p={r['exact_p']:.4f}")

    # Window where CI excludes zero
    ci_sig = merged[merged['G_ci_lo'] > 0]
    print(f"\n  f0 values where 95% CI excludes zero: {len(ci_sig)}/{len(merged)}")
    if len(ci_sig) > 0:
        print(f"    Range: f0 = {ci_sig['f0'].min():.2f} to {ci_sig['f0'].max():.2f} Hz")

    # --- Save ---
    merged_path = os.path.join(args.output_dir, 'bootstrap_gap_analysis.csv')
    merged.to_csv(merged_path, index=False)
    print(f"\n  Results saved: {merged_path}")

    # Save regression details
    reg_path = os.path.join(args.output_dir, 'bootstrap_gap_regression.csv')
    reg_df.to_csv(reg_path, index=False)

    # --- Plots ---
    fig_path = os.path.join(args.output_dir, 'bootstrap_gap_analysis.png')
    plot_bootstrap_gap(merged, fig_path, exact_p_df=exact_df)

    # Trajectory plot using existing sweep data if available
    if args.sweep_csv:
        sweep = pd.read_csv(args.sweep_csv)
        traj_path = os.path.join(args.output_dir, 'ss_trajectories_by_class.png')
        plot_per_base_trajectories(sweep, traj_path, BASES)

    # --- Analysis 4: Linear grid control (if requested) ---
    if args.linear_control:
        print(f"\n{'='*70}")
        print("ANALYSIS 4: Linear grid control (non-exponential baseline)")
        print(f"{'='*70}")
        print(f"  Linear step sizes: {k_values} Hz")

        # Use peak f₀ from bootstrap
        peak_idx = np.argmax(merged['G_observed'].values)
        peak_f0 = merged.iloc[peak_idx]['f0']
        print(f"  Using peak f₀ = {peak_f0:.2f} Hz from bootstrap results")

        linear_df = compute_linear_grid_scores(freqs, peak_f0, k_values, args.window)
        exp_df = compute_exponential_minimal_scores(freqs, peak_f0, BASES, args.window)

        print(f"\n  {'Model':>12s}  {'Label':>8s}  {'Grid Enr':>9s}  "
              f"{'Mid Enr':>9s}  {'SS':>8s}")
        print(f"  {'-'*12}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*8}")
        for _, r in linear_df.iterrows():
            print(f"  {'linear':>12s}  {r['label']:>8s}  {r['grid_enrichment']:>+9.1f}  "
                  f"{r['midpoint_enrichment']:>+9.1f}  {r['structural_score']:>+8.1f}")
        print()
        for _, r in exp_df.iterrows():
            irr = "irr" if r['is_irrational'] else "rat"
            print(f"  {'exponential':>12s}  {r['label']:>8s}  {r['grid_enrichment']:>+9.1f}  "
                  f"{r['midpoint_enrichment']:>+9.1f}  {r['structural_score_minimal']:>+8.1f}  "
                  f"(full: {r['structural_score_full']:>+8.1f}) [{irr}]")

        lin_ss = linear_df['structural_score'].values
        exp_ss_min = exp_df['structural_score_minimal'].values
        print(f"\n  Linear mean SS: {lin_ss.mean():+.1f}  "
              f"Exp minimal mean SS: {exp_ss_min.mean():+.1f}")

        # f₀ sweep for linear grids
        linear_sweep_df = linear_grid_f0_sweep(freqs, f0_values, k_values, args.window)

        # Save
        linear_df.to_csv(os.path.join(args.output_dir, 'linear_grid_control.csv'), index=False)
        exp_df.to_csv(os.path.join(args.output_dir, 'exponential_minimal_scores.csv'), index=False)
        linear_sweep_df.to_csv(os.path.join(args.output_dir, 'linear_grid_f0_sweep.csv'), index=False)

        fig_path = os.path.join(args.output_dir, 'linear_grid_control.png')
        plot_linear_vs_exponential(linear_df, exp_df, fig_path,
                                    linear_sweep_df=linear_sweep_df,
                                    f0_highlight=peak_f0)

    print(f"\nDone.")


if __name__ == '__main__':
    main()

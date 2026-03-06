#!/usr/bin/env python3
"""
Continuous Phi-Lattice Reanalysis — LEMON Dataset
====================================================

Recomputes compliance using Gaussian kernel density instead of binary windows.
Uses saved peak CSVs — no EEG file loading needed. Runtime: ~5 minutes.

Usage:
  python scripts/run_continuous_reanalysis.py
  python scripts/run_continuous_reanalysis.py --sigma 0.025
  python scripts/run_continuous_reanalysis.py --sigma-sweep
"""

import os
import sys
import json
import argparse
import logging
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from continuous_compliance import (
    continuous_compliance_score, continuous_structural_score,
    continuous_ratio_specificity, mean_min_distance,
    sigma_sweep, kernel_enrichment, null_kernel_density,
    circular_distance, SIGMA_DEFAULT, SIGMA_SWEEP_VALUES,
    SPECIFICITY_BASES, PHI,
)
from ratio_specificity import lattice_coordinate, ratio_specificity_test
from structural_phi_specificity import natural_positions, compute_structural_score
from lemon_utils import (
    F0_PRIMARY, COMPLIANCE_WINDOW, COG_TESTS,
    hierarchical_regression, run_group_comparison,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('continuous_reanalysis')

# Paths
EXPORTS_DIR = 'exports_lemon'
OUTPUT_DIR = os.path.join(EXPORTS_DIR, 'continuous')
FIG_DIR = os.path.join(OUTPUT_DIR, 'figures')
SUBJ_DIR = os.path.join(EXPORTS_DIR, 'per_subject')

BASE_PREDICTORS = ['age_midpoint', 'sex', 'education_years',
                   'iaf', 'alpha_power_eo', 'mean_aperiodic_exponent',
                   'n_peaks', 'n_channels_fooof_passed']


def load_saved_data():
    """Load saved pipeline outputs."""
    features = pd.read_csv(os.path.join(EXPORTS_DIR, 'subject_features.csv'))
    master = pd.read_csv(os.path.join(EXPORTS_DIR, 'master_behavioral.csv'))
    with open(os.path.join(EXPORTS_DIR, 'held_out_ids.json')) as f:
        ids = json.load(f)

    included = features[~features['excluded']]
    analysis_ids = set(ids['analysis'])
    included = included[included['subject_id'].isin(analysis_ids)]
    merged = included.merge(master, on='subject_id', how='inner')

    log.info(f"Loaded: {len(included)} analysis subjects, "
             f"{len(merged)} with behavioral data")
    return features, master, included, merged


def load_subject_peaks(subject_id, suffix=''):
    """Load peak CSV for one subject."""
    fname = f'{subject_id}_peaks{suffix}.csv'
    path = os.path.join(SUBJ_DIR, fname)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ============================================================================
# PHASE 1: Per-subject continuous compliance
# ============================================================================

def phase1_subject_compliance(included, sigma):
    """Compute continuous compliance for each subject from saved peaks."""
    log.info("=" * 60)
    log.info(f"PHASE 1: Per-subject continuous compliance (sigma={sigma})")
    log.info("=" * 60)

    rows = []
    all_peaks = []

    for si, sid in enumerate(included['subject_id']):
        peaks_df = load_subject_peaks(sid)
        if peaks_df is None or peaks_df.empty:
            continue

        freqs = peaks_df['freq'].values
        comp = continuous_compliance_score(freqs, f0=F0_PRIMARY, sigma=sigma)

        rows.append({
            'subject_id': sid,
            'compliance_kernel': comp['compliance'],
            'E_boundary_kernel': comp['E_boundary'],
            'E_noble_2_kernel': comp['E_noble_2'],
            'E_attractor_kernel': comp['E_attractor'],
            'E_noble_1_kernel': comp['E_noble_1'],
            'mmd': comp['mmd'],
        })

        # Collect peaks for aggregate analysis
        peaks_df['subject_id'] = sid
        all_peaks.append(peaks_df[['subject_id', 'freq']])

    cont_df = pd.DataFrame(rows)
    cont_df.to_csv(os.path.join(OUTPUT_DIR, 'continuous_features.csv'),
                   index=False)
    log.info(f"  Computed for {len(cont_df)} subjects")

    # Aggregate peaks
    if all_peaks:
        peaks_agg = pd.concat(all_peaks, ignore_index=True)
    else:
        peaks_agg = pd.DataFrame(columns=['subject_id', 'freq'])

    return cont_df, peaks_agg


# ============================================================================
# PHASE 2: Diagnostics
# ============================================================================

def phase2_diagnostics(cont_df, peaks_agg, merged, sigma):
    """Sigma sweep and window-vs-kernel comparison."""
    log.info("=" * 60)
    log.info("PHASE 2: Diagnostics")
    log.info("=" * 60)

    positions = natural_positions(PHI)
    freqs_all = peaks_agg['freq'].values
    u_all = lattice_coordinate(freqs_all, F0_PRIMARY, PHI)
    u_all = u_all[np.isfinite(u_all)]

    # --- Sigma sweep ---
    log.info("  Sigma sweep...")
    sweep_df = sigma_sweep(u_all, positions)

    # Add window-based reference
    window_score, _ = compute_structural_score(u_all, positions, COMPLIANCE_WINDOW)
    sweep_df['window_reference'] = window_score
    sweep_df.to_csv(os.path.join(OUTPUT_DIR, 'sigma_sweep.csv'), index=False)

    log.info(f"  Window SS (w=0.05): {window_score:.1f}")
    for _, row in sweep_df.iterrows():
        log.info(f"  sigma={row['sigma']:.3f}: SS={row['score']:.1f}")

    # Figure: SS vs sigma
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sweep_df['sigma'], sweep_df['score'], 'o-', color='steelblue',
            linewidth=2, markersize=6, label='Kernel SS')
    ax.axhline(window_score, color='coral', linestyle='--', linewidth=1.5,
               label=f'Window SS (w=0.05) = {window_score:.1f}')
    ax.axvline(sigma, color='grey', linestyle=':', linewidth=1,
               label=f'Primary sigma = {sigma}')
    ax.set_xlabel('Kernel bandwidth (sigma)', fontsize=12)
    ax.set_ylabel('Structural Score', fontsize=12)
    ax.set_title(f'Structural Score vs Kernel Bandwidth\n'
                 f'N = {len(u_all):,} aggregate peaks', fontsize=13,
                 fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig_sigma_sweep.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"  Saved sigma sweep figure")

    # --- Window vs kernel correlation ---
    log.info("  Window vs kernel correlation...")
    merged_c = merged.merge(cont_df, on='subject_id', how='inner')

    if 'compliance' in merged_c.columns and 'compliance_kernel' in merged_c.columns:
        valid = merged_c[['compliance', 'compliance_kernel', 'age_group']].dropna()
        r_p, p_p = stats.pearsonr(valid['compliance'], valid['compliance_kernel'])
        rho, p_rho = stats.spearmanr(valid['compliance'], valid['compliance_kernel'])
        log.info(f"  Pearson r = {r_p:.3f} (p={p_p:.4f})")
        log.info(f"  Spearman rho = {rho:.3f} (p={p_rho:.4f})")

        fig, ax = plt.subplots(figsize=(7, 6))
        for grp, color, marker in [('young', 'steelblue', 'o'),
                                    ('elderly', 'coral', 's')]:
            mask = valid['age_group'] == grp
            ax.scatter(valid.loc[mask, 'compliance'],
                       valid.loc[mask, 'compliance_kernel'],
                       alpha=0.5, s=30, color=color, marker=marker,
                       edgecolors='white', linewidths=0.3, label=grp.title())
        ax.set_xlabel('Window Compliance (w=0.05)', fontsize=12)
        ax.set_ylabel(f'Kernel Compliance (sigma={sigma})', fontsize=12)
        ax.set_title(f'Window vs Kernel Compliance\n'
                     f'r={r_p:.3f}, rho={rho:.3f}, N={len(valid)}',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)

        # Identity line
        lims = [min(valid['compliance'].min(), valid['compliance_kernel'].min()) - 5,
                max(valid['compliance'].max(), valid['compliance_kernel'].max()) + 5]
        ax.plot(lims, lims, '--', color='grey', linewidth=0.8, alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'fig_window_vs_kernel.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        log.info(f"  Saved window-vs-kernel figure")

    return merged_c


# ============================================================================
# PHASE 3: Ratio specificity
# ============================================================================

def phase3_ratio_specificity(peaks_agg, sigma):
    """Ratio specificity with continuous kernel vs window."""
    log.info("=" * 60)
    log.info("PHASE 3: Ratio specificity (continuous)")
    log.info("=" * 60)

    freqs_all = peaks_agg['freq'].values

    # Continuous kernel specificity
    log.info("  Running kernel specificity (1000 perms)...")
    kernel_df = continuous_ratio_specificity(freqs_all, f0=F0_PRIMARY,
                                              sigma=sigma, n_perm=1000)
    kernel_df.to_csv(os.path.join(OUTPUT_DIR, 'ratio_specificity_kernel.csv'),
                     index=False)

    log.info("  Kernel ranking:")
    for _, row in kernel_df.iterrows():
        log.info(f"    #{int(row['rank_SS'])} {row['base_name']}: "
                 f"SS={row['SS_kernel']:.1f} (z={row['z_score']:.2f}, "
                 f"p={row['p_value']:.3f}), "
                 f"MMD rank={int(row['rank_MMD'])}")

    # Window-based for comparison
    log.info("  Running window specificity for comparison...")
    window_df = ratio_specificity_test(freqs_all, f0=F0_PRIMARY, n_perm=1000)
    window_df.to_csv(os.path.join(OUTPUT_DIR, 'ratio_specificity_window.csv'),
                     index=False)

    # Figure: side-by-side bar chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: Kernel SS by base
    ax = axes[0]
    kernel_sorted = kernel_df.sort_values('SS_kernel', ascending=True)
    colors = ['#e74c3c' if n == 'φ' else 'steelblue'
              for n in kernel_sorted['base_name']]
    ax.barh(range(len(kernel_sorted)), kernel_sorted['SS_kernel'],
            color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(kernel_sorted)))
    ax.set_yticklabels(kernel_sorted['base_name'], fontsize=10)
    ax.set_xlabel('Structural Score (kernel)', fontsize=11)
    ax.set_title(f'A. Kernel SS (sigma={sigma})', fontsize=12, fontweight='bold')
    ax.axvline(0, color='grey', linewidth=0.5)

    # Panel B: MMD by base (lower = better)
    ax = axes[1]
    mmd_sorted = kernel_df.sort_values('MMD', ascending=False)
    colors_mmd = ['#e74c3c' if n == 'φ' else 'steelblue'
                  for n in mmd_sorted['base_name']]
    ax.barh(range(len(mmd_sorted)), mmd_sorted['MMD'],
            color=colors_mmd, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(mmd_sorted)))
    ax.set_yticklabels(mmd_sorted['base_name'], fontsize=10)
    ax.set_xlabel('Mean Min Distance (lower = better)', fontsize=11)
    ax.set_title('B. MMD (parameter-free)', fontsize=12, fontweight='bold')

    # Panel C: Kernel z-scores
    ax = axes[2]
    z_sorted = kernel_df.sort_values('z_score', ascending=True)
    colors_z = ['#e74c3c' if n == 'φ' else 'steelblue'
                for n in z_sorted['base_name']]
    ax.barh(range(len(z_sorted)), z_sorted['z_score'],
            color=colors_z, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(z_sorted)))
    ax.set_yticklabels(z_sorted['base_name'], fontsize=10)
    ax.set_xlabel('z-score (vs phase-rotation null)', fontsize=11)
    ax.set_title('C. Kernel z-scores', fontsize=12, fontweight='bold')
    ax.axvline(0, color='grey', linewidth=0.5)
    ax.axvline(1.96, color='coral', linewidth=0.8, linestyle='--', alpha=0.5)

    plt.suptitle(f'Ratio Specificity — Continuous Kernel Metrics\n'
                 f'N = {len(freqs_all):,} peaks, f₀ = {F0_PRIMARY} Hz',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig_ratio_specificity_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    log.info("  Saved ratio specificity figure")

    return kernel_df


# ============================================================================
# PHASE 4: LEMON reanalysis
# ============================================================================

def phase4_lemon_reanalysis(merged_c, sigma):
    """Rerun LEMON analyses with continuous compliance."""
    log.info("=" * 60)
    log.info("PHASE 4: LEMON reanalysis (continuous compliance)")
    log.info("=" * 60)

    if 'compliance_kernel' not in merged_c.columns:
        log.error("  No kernel compliance — skipping")
        return

    # --- Step 1 check: position enrichments ---
    log.info("  --- Step 1: Position enrichments (kernel) ---")
    for col in ['E_boundary_kernel', 'E_noble_2_kernel',
                'E_attractor_kernel', 'E_noble_1_kernel']:
        if col in merged_c.columns:
            m = merged_c[col].mean()
            log.info(f"    Mean {col}: {m:+.1f}%")

    # --- Step 4: Zero-order correlations ---
    log.info("  --- Step 4: Zero-order correlations ---")
    corr_results = []
    for test_name, spec in COG_TESTS.items():
        col = f'log_{test_name}' if spec.get('log_transform') else test_name
        if col not in merged_c.columns:
            continue
        valid = merged_c[['compliance_kernel', col]].dropna()
        if len(valid) < 10:
            continue
        r, p = stats.pearsonr(valid['compliance_kernel'], valid[col])
        corr_results.append({
            'test': test_name, 'r': r, 'p': p, 'n': len(valid),
        })
        log.info(f"    {test_name}: r={r:.3f}, p={p:.4f}")
    pd.DataFrame(corr_results).to_csv(
        os.path.join(OUTPUT_DIR, 'correlations_kernel.csv'), index=False)

    # --- Step 5: Hierarchical regression ---
    log.info("  --- Step 5: Hierarchical regression ---")
    reg_results = []
    for test_name, spec in COG_TESTS.items():
        col = f'log_{test_name}' if spec.get('log_transform') else test_name
        if col not in merged_c.columns or 'compliance_kernel' not in merged_c.columns:
            continue
        needed = [col, 'compliance_kernel'] + BASE_PREDICTORS
        valid = merged_c.dropna(subset=needed)
        if len(valid) < 20:
            continue
        y = valid[col].values
        X_base = valid[BASE_PREDICTORS].values
        X_full = np.column_stack([X_base, valid['compliance_kernel'].values])
        try:
            reg = hierarchical_regression(
                y, X_base, X_full, BASE_PREDICTORS,
                BASE_PREDICTORS + ['compliance_kernel'])
            reg_results.append({
                'test': test_name, 'delta_R2': reg['delta_R2'],
                'lrt_p': reg['lrt_p'], 'n': reg['n'],
                'compliance_beta': reg['compliance_beta'],
            })
            log.info(f"    {test_name}: dR2={reg['delta_R2']:.4f}, "
                     f"p={reg['lrt_p']:.4f}")
        except Exception as e:
            log.warning(f"    {test_name}: {e}")
    pd.DataFrame(reg_results).to_csv(
        os.path.join(OUTPUT_DIR, 'regression_kernel.csv'), index=False)

    # --- Step 6: Age group comparison ---
    log.info("  --- Step 6: Age group (kernel) ---")
    if 'age_group' in merged_c.columns:
        young = merged_c.loc[merged_c['age_group'] == 'young',
                             'compliance_kernel'].dropna().values
        elderly = merged_c.loc[merged_c['age_group'] == 'elderly',
                               'compliance_kernel'].dropna().values
        res = run_group_comparison(young, elderly)
        log.info(f"    Young: M={res['mean_young']:.2f} (n={res['n_young']})")
        log.info(f"    Elderly: M={res['mean_elderly']:.2f} (n={res['n_elderly']})")
        log.info(f"    d={res['cohens_d']:.3f}, p={res['p_ttest']:.4f}")
        pd.DataFrame([res]).to_csv(
            os.path.join(OUTPUT_DIR, 'age_comparison_kernel.csv'), index=False)

    # --- Step 6b: Position-specific age trends ---
    log.info("  --- Step 6b: Position-specific age trends (kernel) ---")
    age_col = 'age_midpoint'
    if age_col in merged_c.columns:
        pos_cols = {
            'E_boundary_kernel': ('Boundary', '#e74c3c'),
            'E_noble_2_kernel': ('Noble_2', '#9b59b6'),
            'E_attractor_kernel': ('Attractor', '#3498db'),
            'E_noble_1_kernel': ('Noble_1', '#f39c12'),
        }

        fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True)
        axes_flat = axes.flatten()
        trend_results = []

        for idx, (col, (label, color)) in enumerate(pos_cols.items()):
            ax = axes_flat[idx]
            if col not in merged_c.columns:
                continue
            valid = merged_c.dropna(subset=[age_col, col])
            if len(valid) < 10:
                continue
            ages = valid[age_col].values
            vals = valid[col].values

            # Jittered scatter
            jitter = np.random.default_rng(42).uniform(-0.8, 0.8, len(ages))
            ax.scatter(ages + jitter, vals, alpha=0.35, s=18, color=color,
                       edgecolors='white', linewidths=0.3)

            # Bin means
            bin_stats = valid.groupby(age_col)[col].agg(['mean', 'sem', 'count'])
            bin_stats = bin_stats[bin_stats['count'] >= 3]
            ax.errorbar(bin_stats.index, bin_stats['mean'], yerr=bin_stats['sem'],
                        fmt='o-', color=color, markersize=7, linewidth=2,
                        capsize=4, markeredgecolor='white', markeredgewidth=0.8,
                        zorder=5)

            # Regression line
            slope, intercept, r, p, se = stats.linregress(ages, vals)
            x_line = np.array([ages.min(), ages.max()])
            ax.plot(x_line, intercept + slope * x_line, '--', color='grey',
                    linewidth=1.5, alpha=0.7)

            ax.axhline(0, color='grey', linewidth=0.6, linestyle=':')
            ax.set_ylabel('Enrichment (%)', fontsize=10)
            ax.set_title(f'{label}\nr={r:.3f}, p={p:.3f}', fontsize=11,
                         fontweight='bold')

            trend_results.append({
                'position': label, 'column': col,
                'r': r, 'p': p, 'slope': slope, 'se': se,
                'metric': 'kernel',
            })
            log.info(f"    {label}: r={r:.3f}, p={p:.4f}")

            # Also test categorical
            if 'age_group' in merged_c.columns:
                y_grp = merged_c.loc[merged_c['age_group'] == 'young', col].dropna().values
                e_grp = merged_c.loc[merged_c['age_group'] == 'elderly', col].dropna().values
                if len(y_grp) > 5 and len(e_grp) > 5:
                    grp_res = run_group_comparison(y_grp, e_grp)
                    log.info(f"      categorical: d={grp_res['cohens_d']:.3f}, "
                             f"p={grp_res['p_ttest']:.4f}")

        axes[1, 0].set_xlabel('Age (midpoint)', fontsize=11)
        axes[1, 1].set_xlabel('Age (midpoint)', fontsize=11)
        fig.suptitle(f'Position-Specific Enrichment vs Age (Kernel, sigma={sigma})',
                     fontsize=13, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'fig_attractor_age_kernel.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        log.info("  Saved age trends figure")

        pd.DataFrame(trend_results).to_csv(
            os.path.join(OUTPUT_DIR, 'age_trends_kernel.csv'), index=False)

    return merged_c


# ============================================================================
# PHASE 5: Max-peaks robustness
# ============================================================================

def phase5_max_peaks(merged_c, sigma):
    """Test if kernel compliance is more robust to max_n_peaks=40."""
    log.info("=" * 60)
    log.info("PHASE 5: Max-peaks robustness (kernel)")
    log.info("=" * 60)

    # Load max40 peaks
    rows = []
    for sid in merged_c['subject_id']:
        peaks_df = load_subject_peaks(sid, suffix='_max40')
        if peaks_df is None or peaks_df.empty:
            continue
        freqs = peaks_df['freq'].values
        comp = continuous_compliance_score(freqs, f0=F0_PRIMARY, sigma=sigma)
        rows.append({
            'subject_id': sid,
            'compliance_kernel_max40': comp['compliance'],
            'E_attractor_kernel_max40': comp['E_attractor'],
            'E_boundary_kernel_max40': comp['E_boundary'],
            'mmd_max40': comp['mmd'],
        })

    if not rows:
        log.warning("  No max40 peak files found — skipping")
        log.info("  (Run sensitivity (l) first to generate max40 peaks)")
        return

    max40_df = pd.DataFrame(rows)
    merged_max = merged_c.merge(max40_df, on='subject_id', how='inner')
    log.info(f"  {len(merged_max)} subjects with both max20 and max40")

    # Rank-order preservation
    both = merged_max[['compliance_kernel', 'compliance_kernel_max40']].dropna()
    if len(both) > 5:
        rho_k, p_k = stats.spearmanr(both['compliance_kernel'],
                                       both['compliance_kernel_max40'])
        log.info(f"  Kernel: rho(max20, max40) = {rho_k:.3f} (p={p_k:.4f})")
        log.info(f"  Window: rho(max20, max40) = 0.838 (from sensitivity (l))")
        log.info(f"  Improvement: {'YES' if rho_k > 0.838 else 'NO'} "
                 f"(delta = {rho_k - 0.838:+.3f})")

    # MMD comparison
    both_mmd = merged_max[['mmd', 'mmd_max40']].dropna()
    if len(both_mmd) > 5:
        rho_mmd, _ = stats.spearmanr(both_mmd['mmd'], both_mmd['mmd_max40'])
        log.info(f"  MMD: rho(max20, max40) = {rho_mmd:.3f}")

    # Attractor age trend with max40 kernel
    if 'age_midpoint' in merged_max.columns and 'E_attractor_kernel_max40' in merged_max.columns:
        valid = merged_max.dropna(subset=['age_midpoint', 'E_attractor_kernel_max40'])
        r, p = stats.pearsonr(valid['age_midpoint'], valid['E_attractor_kernel_max40'])
        log.info(f"  Attractor age trend (kernel max40): r={r:.3f}, p={p:.4f}")
        log.info(f"  Compare: kernel max20 = see Phase 4, window max40 = r=-0.110 p=.138")

    # Age group with max40 kernel
    if 'age_group' in merged_max.columns:
        y = merged_max.loc[merged_max['age_group'] == 'young',
                           'compliance_kernel_max40'].dropna().values
        e = merged_max.loc[merged_max['age_group'] == 'elderly',
                           'compliance_kernel_max40'].dropna().values
        if len(y) > 5 and len(e) > 5:
            res = run_group_comparison(y, e)
            log.info(f"  Age group (kernel max40): d={res['cohens_d']:.3f}, "
                     f"p={res['p_ttest']:.4f}")

    # Figure: kernel max20 vs max40 scatter
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    if len(both) > 5:
        ax.scatter(both['compliance_kernel'], both['compliance_kernel_max40'],
                   alpha=0.4, s=25, color='steelblue', edgecolors='white',
                   linewidths=0.3)
        lims = [min(both['compliance_kernel'].min(),
                    both['compliance_kernel_max40'].min()) - 5,
                max(both['compliance_kernel'].max(),
                    both['compliance_kernel_max40'].max()) + 5]
        ax.plot(lims, lims, '--', color='grey', linewidth=0.8, alpha=0.5)
        ax.set_xlabel(f'Kernel Compliance (max20)', fontsize=11)
        ax.set_ylabel(f'Kernel Compliance (max40)', fontsize=11)
        ax.set_title(f'A. Kernel: max20 vs max40\n'
                     f'rho={rho_k:.3f} (window rho=0.838)',
                     fontsize=12, fontweight='bold')

    ax = axes[1]
    if 'age_midpoint' in merged_max.columns and 'E_attractor_kernel_max40' in merged_max.columns:
        valid = merged_max.dropna(subset=['age_midpoint', 'E_attractor_kernel_max40',
                                           'E_attractor_kernel'])
        ages = valid['age_midpoint'].values

        for col, label, color, marker in [
            ('E_attractor_kernel', 'Kernel max20', 'steelblue', 'o'),
            ('E_attractor_kernel_max40', 'Kernel max40', '#e74c3c', 's'),
        ]:
            vals = valid[col].values
            bin_stats = valid.groupby('age_midpoint')[col].agg(['mean', 'sem', 'count'])
            bin_stats = bin_stats[bin_stats['count'] >= 3]
            ax.errorbar(bin_stats.index, bin_stats['mean'], yerr=bin_stats['sem'],
                        fmt=f'{marker}-', color=color, markersize=6,
                        linewidth=1.5, capsize=3, label=label, zorder=5)

            slope, intercept, r, p, _ = stats.linregress(ages, vals)
            x_line = np.array([ages.min(), ages.max()])
            ax.plot(x_line, intercept + slope * x_line, '--', color=color,
                    linewidth=1, alpha=0.5)

        ax.axhline(0, color='grey', linewidth=0.6, linestyle=':')
        ax.set_xlabel('Age (midpoint)', fontsize=11)
        ax.set_ylabel('Attractor Enrichment (%)', fontsize=11)
        ax.set_title('B. Attractor Age Trend: max20 vs max40',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig_max40_kernel_robustness.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    log.info("  Saved max40 robustness figure")

    max40_df.to_csv(os.path.join(OUTPUT_DIR, 'max40_kernel_comparison.csv'),
                    index=False)


# ============================================================================
# EC reanalysis
# ============================================================================

def phase_ec_reanalysis(merged_c, sigma):
    """Recompute kernel compliance on EC peaks if available."""
    log.info("=" * 60)
    log.info("PHASE EC: Eyes-closed kernel compliance")
    log.info("=" * 60)

    rows = []
    for sid in merged_c['subject_id']:
        peaks_df = load_subject_peaks(sid, suffix='_ec')
        if peaks_df is None or peaks_df.empty:
            continue
        freqs = peaks_df['freq'].values
        comp = continuous_compliance_score(freqs, f0=F0_PRIMARY, sigma=sigma)
        rows.append({
            'subject_id': sid,
            'compliance_kernel_ec': comp['compliance'],
            'E_attractor_kernel_ec': comp['E_attractor'],
            'E_boundary_kernel_ec': comp['E_boundary'],
        })

    if not rows:
        log.warning("  No EC peak files — skipping")
        return

    ec_df = pd.DataFrame(rows)
    merged_ec = merged_c.merge(ec_df, on='subject_id', how='inner')
    log.info(f"  {len(merged_ec)} subjects with EC kernel compliance")

    # EO vs EC correlation
    both = merged_ec[['compliance_kernel', 'compliance_kernel_ec']].dropna()
    if len(both) > 5:
        r_p, _ = stats.pearsonr(both['compliance_kernel'],
                                 both['compliance_kernel_ec'])
        rho, _ = stats.spearmanr(both['compliance_kernel'],
                                  both['compliance_kernel_ec'])
        log.info(f"  EO-EC kernel: Pearson r={r_p:.3f}, Spearman rho={rho:.3f}")

    # Attractor age trend (EC kernel)
    if 'age_midpoint' in merged_ec.columns and 'E_attractor_kernel_ec' in merged_ec.columns:
        valid = merged_ec.dropna(subset=['age_midpoint', 'E_attractor_kernel_ec'])
        r, p = stats.pearsonr(valid['age_midpoint'], valid['E_attractor_kernel_ec'])
        log.info(f"  Attractor age trend (EC kernel): r={r:.3f}, p={p:.4f}")

    # Age group (EC kernel)
    if 'age_group' in merged_ec.columns:
        y = merged_ec.loc[merged_ec['age_group'] == 'young',
                          'compliance_kernel_ec'].dropna().values
        e = merged_ec.loc[merged_ec['age_group'] == 'elderly',
                          'compliance_kernel_ec'].dropna().values
        if len(y) > 5 and len(e) > 5:
            res = run_group_comparison(y, e)
            log.info(f"  Age group (EC kernel): d={res['cohens_d']:.3f}, "
                     f"p={res['p_ttest']:.4f}")

    ec_df.to_csv(os.path.join(OUTPUT_DIR, 'ec_kernel_features.csv'), index=False)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sigma', type=float, default=SIGMA_DEFAULT)
    parser.add_argument('--sigma-sweep', action='store_true')
    args = parser.parse_args()

    sigma = args.sigma
    t_start = time.time()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    features, master, included, merged = load_saved_data()

    cont_df, peaks_agg = phase1_subject_compliance(included, sigma)
    merged_c = phase2_diagnostics(cont_df, peaks_agg, merged, sigma)
    phase3_ratio_specificity(peaks_agg, sigma)
    phase4_lemon_reanalysis(merged_c, sigma)
    phase5_max_peaks(merged_c, sigma)
    phase_ec_reanalysis(merged_c, sigma)

    elapsed = time.time() - t_start
    log.info(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

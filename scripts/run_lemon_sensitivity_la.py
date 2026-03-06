#!/usr/bin/env python3
"""
LEMON Sensitivity Analyses (l) and (a)
========================================

Standalone script that loads saved features/master tables and runs:
  (l) max_n_peaks=40 — FOOOF re-extraction + Step 1 check + regression
  (a) Eyes-closed    — FOOOF on EC blocks + compliance + attractor age trend

Usage:
  python scripts/run_lemon_sensitivity_la.py
"""

import os
import sys
import json
import gc
import time
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from lemon_utils import (
    LEMON_PREPROC_ROOT, F0_PRIMARY, PHI, COMPLIANCE_WINDOW, SFREQ,
    FOOOF_PARAMS, COG_TESTS, WELCH_NPERSEG,
    load_preprocessed_subject, extract_fooof_peaks_subject,
    compute_compliance_score,
    hierarchical_regression, run_group_comparison,
)
from ratio_specificity import lattice_coordinate, _enrichment_at_offset
from structural_phi_specificity import natural_positions

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('sensitivity_la')

OUTPUT_DIR = 'exports_lemon'
FIG_DIR = os.path.join(OUTPUT_DIR, 'figures')
TABLE_DIR = os.path.join(OUTPUT_DIR, 'tables')
SUBJ_DIR = os.path.join(OUTPUT_DIR, 'per_subject')

PREPROC_ROOT = LEMON_PREPROC_ROOT

# Base predictors for regression (same as main pipeline)
BASE_PREDICTORS = ['age_midpoint', 'sex', 'education_years',
                   'iaf', 'alpha_power_eo', 'mean_aperiodic_exponent',
                   'n_peaks', 'n_channels_fooof_passed']


def load_saved_data():
    """Load saved features and master tables from N=202 run."""
    features = pd.read_csv(os.path.join(OUTPUT_DIR, 'subject_features.csv'))
    master = pd.read_csv(os.path.join(OUTPUT_DIR, 'master_behavioral.csv'))
    with open(os.path.join(OUTPUT_DIR, 'held_out_ids.json')) as f:
        ids = json.load(f)

    included = features[~features['excluded']]
    # Filter to analysis set (not held-out)
    analysis_ids = set(ids['analysis'])
    included = included[included['subject_id'].isin(analysis_ids)]

    merged = included.merge(master, on='subject_id', how='inner')
    log.info(f"Loaded: {len(included)} analysis subjects, "
             f"{len(merged)} with behavioral data")
    return features, master, included, merged


def run_regression_variant(label, merged_var, compliance_col='compliance'):
    """Run hierarchical regression for one sensitivity variant."""
    results = []
    for test_name, spec in COG_TESTS.items():
        col = f'log_{test_name}' if spec.get('log_transform') else test_name
        if col not in merged_var.columns or compliance_col not in merged_var.columns:
            continue
        needed = [col, compliance_col] + BASE_PREDICTORS
        valid = merged_var.dropna(subset=needed)
        if len(valid) < 20:
            continue
        y = valid[col].values
        X_base = valid[BASE_PREDICTORS].values
        X_full = np.column_stack([X_base, valid[compliance_col].values])
        try:
            reg = hierarchical_regression(
                y, X_base, X_full, BASE_PREDICTORS,
                BASE_PREDICTORS + [compliance_col])
            results.append({
                'test': test_name, 'delta_R2': reg['delta_R2'],
                'lrt_p': reg['lrt_p'], 'n': reg['n'],
                'compliance_beta': reg['compliance_beta'],
            })
        except Exception as e:
            log.warning(f"  {label}/{test_name}: {e}")
    return pd.DataFrame(results)


# ============================================================================
# SENSITIVITY (l): max_n_peaks=40
# ============================================================================

def sensitivity_l(included, merged):
    """Re-run FOOOF with max_n_peaks=40 for all subjects.

    Tests:
    1. Step 1 replication with max40 peaks (boundary depletion, noble enrichment)
    2. Compliance rank-order preservation (Spearman rho vs max20)
    3. Hierarchical regression with max40 compliance
    """
    log.info("=" * 60)
    log.info("SENSITIVITY (l): max_n_peaks=40")
    log.info("=" * 60)

    params_40 = FOOOF_PARAMS.copy()
    params_40['max_n_peaks'] = 40

    l_features = []
    all_peaks_max40 = []
    n_total = len(included)
    t0 = time.time()

    for si, sid in enumerate(included['subject_id']):
        raw_eo, info_eo = load_preprocessed_subject(sid, PREPROC_ROOT, condition='EO')
        if raw_eo is None:
            continue

        peaks_df, ch_info = extract_fooof_peaks_subject(
            raw_eo, fs=SFREQ, fooof_params=params_40)

        comp = compute_compliance_score(
            peaks_df['freq'].values if not peaks_df.empty else np.array([]))

        l_features.append({
            'subject_id': sid,
            'compliance_max40': comp['compliance'],
            'n_peaks_max40': ch_info['total_peak_count'],
            'E_boundary_max40': comp['E_boundary'],
            'E_noble_2_max40': comp['E_noble_2'],
            'E_attractor_max40': comp['E_attractor'],
            'E_noble_1_max40': comp['E_noble_1'],
        })

        # Save peaks for Step 1 aggregate check
        if not peaks_df.empty:
            peaks_df['subject_id'] = sid
            all_peaks_max40.append(peaks_df)

        # Save max40 peaks to per_subject dir
        if not peaks_df.empty:
            peaks_path = os.path.join(SUBJ_DIR, f'{sid}_peaks_max40.csv')
            peaks_df.to_csv(peaks_path, index=False)

        del raw_eo
        gc.collect()

        if (si + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (si + 1) / elapsed
            eta = (n_total - si - 1) / rate
            log.info(f"  [{si+1}/{n_total}] {elapsed:.0f}s elapsed, "
                     f"ETA {eta:.0f}s")

    elapsed = time.time() - t0
    log.info(f"  FOOOF re-extraction done: {len(l_features)} subjects "
             f"in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    if not l_features:
        log.error("  No features extracted!")
        return

    l_df = pd.DataFrame(l_features)

    # --- Step 1 check with max40 peaks ---
    log.info("  --- Step 1 replication with max40 peaks ---")
    if all_peaks_max40:
        peaks_agg = pd.concat(all_peaks_max40, ignore_index=True)
        freqs = peaks_agg['freq'].values
        u = lattice_coordinate(freqs, F0_PRIMARY, PHI)
        u = u[np.isfinite(u)]
        log.info(f"  Aggregate max40 peaks: {len(freqs):,} "
                 f"(vs ~{included['n_peaks'].sum():.0f} at max20)")

        # Position enrichments
        positions = natural_positions(PHI)
        for name, offset in positions.items():
            e = _enrichment_at_offset(u, offset, COMPLIANCE_WINDOW, len(u))
            log.info(f"    {name} ({offset:.3f}): {e:+.1f}%")

        # Permutation test for boundary depletion
        boundary_enrich = _enrichment_at_offset(u, 0.0, COMPLIANCE_WINDOW, len(u))
        noble_1_enrich = _enrichment_at_offset(u, 1.0 / PHI, COMPLIANCE_WINDOW, len(u))

        null_boundary = []
        null_noble = []
        rng = np.random.default_rng(42)
        for _ in range(1000):
            theta = rng.uniform(0, 1)
            u_rot = (u + theta) % 1.0
            null_boundary.append(
                _enrichment_at_offset(u_rot, 0.0, COMPLIANCE_WINDOW, len(u_rot)))
            null_noble.append(
                _enrichment_at_offset(u_rot, 1.0 / PHI, COMPLIANCE_WINDOW, len(u_rot)))

        null_boundary = np.array(null_boundary)
        null_noble = np.array(null_noble)
        p_boundary = (null_boundary <= boundary_enrich).sum() / 1000
        p_noble = (null_noble >= noble_1_enrich).sum() / 1000

        log.info(f"  Boundary: {boundary_enrich:+.1f}% (p={p_boundary:.4f})")
        log.info(f"  Noble₁: {noble_1_enrich:+.1f}% (p={p_noble:.4f})")

        step1_max40 = {
            'n_peaks': len(freqs),
            'boundary_enrichment': boundary_enrich,
            'noble_1_enrichment': noble_1_enrich,
            'p_boundary': p_boundary,
            'p_noble': p_noble,
        }
        pd.DataFrame([step1_max40]).to_csv(
            os.path.join(TABLE_DIR, 'sensitivity_l_step1_max40.csv'), index=False)

    # --- Rank-order preservation ---
    log.info("  --- Rank-order preservation ---")
    merged_l = merged.merge(l_df, on='subject_id', how='inner')
    from scipy.stats import spearmanr
    both = merged_l[['compliance', 'compliance_max40']].dropna()
    if len(both) > 5:
        rho, p_rho = spearmanr(both['compliance'], both['compliance_max40'])
        log.info(f"  Spearman rho(max20, max40) = {rho:.3f} (p={p_rho:.4f}), "
                 f"N={len(both)}")
        log.info(f"  Mean peaks: max20={merged_l['n_peaks'].mean():.0f}, "
                 f"max40={merged_l['n_peaks_max40'].mean():.0f}")

    # --- Regression ---
    log.info("  --- Hierarchical regression with max40 compliance ---")
    sens_l = run_regression_variant('max_n_peaks=40', merged_l,
                                    compliance_col='compliance_max40')
    sens_l.to_csv(os.path.join(TABLE_DIR, 'sensitivity_l_max_peaks_40.csv'),
                  index=False)
    if not sens_l.empty:
        for _, row in sens_l.iterrows():
            log.info(f"    {row['test']}: ΔR²={row['delta_R2']:.4f}, "
                     f"p={row['lrt_p']:.4f}")

    # --- Age group comparison with max40 ---
    log.info("  --- Age group comparison with max40 compliance ---")
    if 'age_group' in merged_l.columns:
        young = merged_l.loc[merged_l['age_group'] == 'young', 'compliance_max40'].dropna().values
        elderly = merged_l.loc[merged_l['age_group'] == 'elderly', 'compliance_max40'].dropna().values
        res = run_group_comparison(young, elderly)
        log.info(f"  Young: M={res['mean_young']:.2f} (n={res['n_young']})")
        log.info(f"  Elderly: M={res['mean_elderly']:.2f} (n={res['n_elderly']})")
        log.info(f"  d={res['cohens_d']:.3f}, p={res['p_ttest']:.4f}")

        # Position-specific with max40
        for pos_col in ['E_attractor_max40', 'E_boundary_max40',
                        'E_noble_1_max40', 'E_noble_2_max40']:
            if pos_col in merged_l.columns:
                y = merged_l.loc[merged_l['age_group'] == 'young', pos_col].dropna().values
                e = merged_l.loc[merged_l['age_group'] == 'elderly', pos_col].dropna().values
                if len(y) > 5 and len(e) > 5:
                    r = run_group_comparison(y, e)
                    log.info(f"    {pos_col}: d={r['cohens_d']:.3f}, "
                             f"p={r['p_ttest']:.4f}")

        # Attractor continuous age trend with max40
        if 'age_midpoint' in merged_l.columns and 'E_attractor_max40' in merged_l.columns:
            valid = merged_l.dropna(subset=['age_midpoint', 'E_attractor_max40'])
            r, p = stats.pearsonr(valid['age_midpoint'], valid['E_attractor_max40'])
            log.info(f"  Attractor age trend (max40): r={r:.3f}, p={p:.4f}")

    return merged_l


# ============================================================================
# SENSITIVITY (a): Eyes-Closed
# ============================================================================

def sensitivity_a(included, merged):
    """Compute compliance from eyes-closed EEG blocks.

    Tests whether the attractor erosion with age holds under EC condition.
    """
    log.info("=" * 60)
    log.info("SENSITIVITY (a): Eyes-Closed compliance")
    log.info("=" * 60)

    a_features = []
    n_total = len(included)
    n_missing = 0
    t0 = time.time()

    for si, sid in enumerate(included['subject_id']):
        raw_ec, info_ec = load_preprocessed_subject(sid, PREPROC_ROOT, condition='EC')
        if raw_ec is None:
            n_missing += 1
            continue

        peaks_df, ch_info = extract_fooof_peaks_subject(
            raw_ec, fs=SFREQ, fooof_params=FOOOF_PARAMS)

        comp = compute_compliance_score(
            peaks_df['freq'].values if not peaks_df.empty else np.array([]))

        a_features.append({
            'subject_id': sid,
            'compliance_ec': comp['compliance'],
            'n_peaks_ec': ch_info['total_peak_count'],
            'mean_r_squared_ec': ch_info['mean_r_squared'],
            'E_boundary_ec': comp['E_boundary'],
            'E_noble_2_ec': comp['E_noble_2'],
            'E_attractor_ec': comp['E_attractor'],
            'E_noble_1_ec': comp['E_noble_1'],
        })

        # Save EC peaks
        if not peaks_df.empty:
            peaks_path = os.path.join(SUBJ_DIR, f'{sid}_peaks_ec.csv')
            peaks_df.to_csv(peaks_path, index=False)

        del raw_ec
        gc.collect()

        if (si + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (si + 1) / elapsed
            eta = (n_total - si - 1) / rate
            log.info(f"  [{si+1}/{n_total}] {elapsed:.0f}s elapsed, "
                     f"ETA {eta:.0f}s")

    elapsed = time.time() - t0
    log.info(f"  EC extraction done: {len(a_features)} subjects "
             f"in {elapsed:.0f}s ({elapsed/60:.1f} min), "
             f"{n_missing} missing EC files")

    if not a_features:
        log.error("  No EC features extracted!")
        return

    a_df = pd.DataFrame(a_features)
    a_df.to_csv(os.path.join(TABLE_DIR, 'sensitivity_a_ec_features.csv'),
                index=False)

    merged_a = merged.merge(a_df, on='subject_id', how='inner')
    log.info(f"  Merged with behavioral: N={len(merged_a)}")

    # --- EO vs EC compliance correlation ---
    log.info("  --- EO vs EC compliance correlation ---")
    both = merged_a[['compliance', 'compliance_ec']].dropna()
    if len(both) > 5:
        from scipy.stats import spearmanr, pearsonr
        rho, p_rho = spearmanr(both['compliance'], both['compliance_ec'])
        r_p, p_p = pearsonr(both['compliance'], both['compliance_ec'])
        log.info(f"  EO-EC: Pearson r={r_p:.3f} (p={p_p:.4f}), "
                 f"Spearman rho={rho:.3f} (p={p_rho:.4f}), N={len(both)}")

    # --- Age group comparison with EC compliance ---
    log.info("  --- Age group comparison (EC) ---")
    if 'age_group' in merged_a.columns:
        young = merged_a.loc[merged_a['age_group'] == 'young', 'compliance_ec'].dropna().values
        elderly = merged_a.loc[merged_a['age_group'] == 'elderly', 'compliance_ec'].dropna().values
        res = run_group_comparison(young, elderly)
        log.info(f"  Young: M={res['mean_young']:.2f} (n={res['n_young']})")
        log.info(f"  Elderly: M={res['mean_elderly']:.2f} (n={res['n_elderly']})")
        log.info(f"  d={res['cohens_d']:.3f}, p={res['p_ttest']:.4f}")

        # Position-specific EC
        for pos_col, label in [('E_attractor_ec', 'Attractor'),
                                ('E_boundary_ec', 'Boundary'),
                                ('E_noble_1_ec', 'Noble₁'),
                                ('E_noble_2_ec', 'Noble₂')]:
            if pos_col in merged_a.columns:
                y = merged_a.loc[merged_a['age_group'] == 'young', pos_col].dropna().values
                e = merged_a.loc[merged_a['age_group'] == 'elderly', pos_col].dropna().values
                if len(y) > 5 and len(e) > 5:
                    r = run_group_comparison(y, e)
                    log.info(f"    {label} (EC): d={r['cohens_d']:.3f}, "
                             f"p={r['p_ttest']:.4f}")

    # --- Continuous age trends (EC) ---
    log.info("  --- Continuous age trends (EC) ---")
    age_col = 'age_midpoint'
    if age_col in merged_a.columns:
        pos_cols = {
            'E_boundary_ec': 'Boundary',
            'E_noble_2_ec': 'Noble₂',
            'E_attractor_ec': 'Attractor',
            'E_noble_1_ec': 'Noble₁',
        }
        trend_results = []
        for col, label in pos_cols.items():
            if col not in merged_a.columns:
                continue
            valid = merged_a.dropna(subset=[age_col, col])
            if len(valid) < 10:
                continue
            slope, intercept, r, p, se = stats.linregress(
                valid[age_col].values, valid[col].values)
            log.info(f"    {label} (EC): r={r:.3f}, p={p:.4f}, "
                     f"slope={slope:.2f}/yr")
            trend_results.append({
                'position': label, 'column': col,
                'r': r, 'p': p, 'slope': slope, 'se': se,
                'condition': 'EC',
            })
        pd.DataFrame(trend_results).to_csv(
            os.path.join(TABLE_DIR, 'sensitivity_a_age_trends_ec.csv'),
            index=False)

    # --- Regression with EC compliance ---
    log.info("  --- Hierarchical regression (EC compliance) ---")
    sens_a = run_regression_variant('eyes_closed', merged_a,
                                    compliance_col='compliance_ec')
    sens_a.to_csv(os.path.join(TABLE_DIR, 'sensitivity_a_eyes_closed.csv'),
                  index=False)
    if not sens_a.empty:
        for _, row in sens_a.iterrows():
            log.info(f"    {row['test']}: ΔR²={row['delta_R2']:.4f}, "
                     f"p={row['lrt_p']:.4f}")

    # --- Figure: EO vs EC comparison ---
    log.info("  --- Generating EO vs EC comparison figure ---")
    if age_col in merged_a.columns and 'E_attractor_ec' in merged_a.columns:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Panel A: EO vs EC compliance scatter
        ax = axes[0]
        valid = merged_a.dropna(subset=['compliance', 'compliance_ec'])
        ax.scatter(valid['compliance'], valid['compliance_ec'],
                   alpha=0.4, s=25, color='steelblue', edgecolors='white',
                   linewidths=0.3)
        lims = [min(valid['compliance'].min(), valid['compliance_ec'].min()) - 5,
                max(valid['compliance'].max(), valid['compliance_ec'].max()) + 5]
        ax.plot(lims, lims, '--', color='grey', linewidth=1, alpha=0.5)
        r_p, _ = stats.pearsonr(valid['compliance'], valid['compliance_ec'])
        ax.set_xlabel('Compliance (EO)', fontsize=11)
        ax.set_ylabel('Compliance (EC)', fontsize=11)
        ax.set_title(f'A. EO vs EC Compliance\nr={r_p:.3f}, N={len(valid)}',
                     fontsize=12, fontweight='bold')

        # Panel B: Attractor age trend — EO vs EC overlay
        ax = axes[1]
        for cond, col, color, marker in [
            ('EO', 'E_attractor', 'steelblue', 'o'),
            ('EC', 'E_attractor_ec', '#e74c3c', 's'),
        ]:
            if col not in merged_a.columns:
                continue
            valid = merged_a.dropna(subset=[age_col, col])
            ages = valid[age_col].values
            vals = valid[col].values

            # Bin means
            bin_stats = valid.groupby(age_col)[col].agg(['mean', 'sem', 'count'])
            bin_stats = bin_stats[bin_stats['count'] >= 3]
            ax.errorbar(bin_stats.index, bin_stats['mean'], yerr=bin_stats['sem'],
                        fmt=f'{marker}-', color=color, markersize=6, linewidth=1.5,
                        capsize=3, label=f'{cond}', zorder=5)

            # Regression line
            slope, intercept, r, p, _ = stats.linregress(ages, vals)
            x_line = np.array([ages.min(), ages.max()])
            ax.plot(x_line, intercept + slope * x_line, '--', color=color,
                    linewidth=1, alpha=0.5)

        ax.axhline(0, color='grey', linewidth=0.6, linestyle=':')
        ax.set_xlabel('Age (midpoint)', fontsize=11)
        ax.set_ylabel('Attractor Enrichment (%)', fontsize=11)
        ax.set_title('B. Attractor Age Trend: EO vs EC', fontsize=12,
                     fontweight='bold')
        ax.legend(fontsize=10)

        # Panel C: Age group bars — EO vs EC for attractor
        ax = axes[2]
        if 'age_group' in merged_a.columns:
            groups = ['young', 'elderly']
            conditions = [('EO', 'E_attractor', 'steelblue'),
                          ('EC', 'E_attractor_ec', '#e74c3c')]
            x = np.arange(len(groups))
            bar_w = 0.35
            for ci, (cond, col, color) in enumerate(conditions):
                if col not in merged_a.columns:
                    continue
                means = [merged_a.loc[merged_a['age_group'] == g, col].mean()
                         for g in groups]
                sems = [merged_a.loc[merged_a['age_group'] == g, col].sem()
                        for g in groups]
                ax.bar(x + ci * bar_w - bar_w / 2, means, bar_w, yerr=sems,
                       color=color, alpha=0.8, label=cond, capsize=4,
                       edgecolor='white', linewidth=0.5)
            ax.axhline(0, color='grey', linewidth=0.6, linestyle=':')
            ax.set_xticks(x)
            ax.set_xticklabels(['Young', 'Elderly'], fontsize=11)
            ax.set_ylabel('Attractor Enrichment (%)', fontsize=11)
            ax.set_title('C. Attractor by Age Group: EO vs EC', fontsize=12,
                         fontweight='bold')
            ax.legend(fontsize=10)

        plt.tight_layout()
        fig_path = os.path.join(FIG_DIR, 'fig_sensitivity_a_eo_ec.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        log.info(f"  Saved {fig_path}")

    return merged_a


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    t_start = time.time()

    features, master, included, merged = load_saved_data()

    # (l) max_n_peaks=40
    merged_l = sensitivity_l(included, merged)

    # (a) Eyes-closed
    merged_a = sensitivity_a(included, merged)

    elapsed = time.time() - t_start
    log.info(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

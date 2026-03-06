#!/usr/bin/env python3
"""
LEMON Phi-Lattice Cognition Analysis — Main Pipeline
======================================================

Paper 3: "Golden Ratio Lattice Precision Predicts Cognitive Performance
Across the Adult Lifespan."

Usage:
  python scripts/run_lemon_phi_cognition.py                                    # held-out only (debug)
  python scripts/run_lemon_phi_cognition.py --full-run                         # all subjects
  python scripts/run_lemon_phi_cognition.py --full-run --skip-preprocessing    # resume from features CSV
  python scripts/run_lemon_phi_cognition.py --full-run --sensitivity           # + sensitivity analyses
  python scripts/run_lemon_phi_cognition.py --full-run --no-sie               # skip SIE detection
"""

import os
import sys
import json
import gc
import time
import argparse
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from lemon_utils import (
    # Constants
    LEMON_PREPROC_ROOT, LEMON_META_PATH, LEMON_COG_ROOT,
    F0_PRIMARY, F0_SENSITIVITY, PHI, COMPLIANCE_WINDOW, SFREQ,
    COG_TESTS, CHANNELS_1020, FOOOF_PARAMS,
    # Functions
    discover_subjects, select_held_out,
    load_demographics, load_cognitive_data, build_master_table,
    load_preprocessed_subject, extract_fooof_peaks_subject,
    process_single_subject,
    compute_compliance_score, make_phi_bands,
    fdr_correct, compute_icc_2way, hierarchical_regression,
    bootstrap_delta_r2_ci, bootstrap_mediation,
    run_group_comparison,
)
from ratio_specificity import (
    lattice_coordinate, phase_rotation_null, ratio_specificity_test,
)
from structural_phi_specificity import natural_positions, compute_structural_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('lemon_pipeline')

OUTPUT_DIR = 'exports_lemon'
FIG_DIR = os.path.join(OUTPUT_DIR, 'figures')
TABLE_DIR = os.path.join(OUTPUT_DIR, 'tables')
SUBJ_DIR = os.path.join(OUTPUT_DIR, 'per_subject')


# ============================================================================
# PHASE 0: SETUP
# ============================================================================

def phase0_setup(args):
    """Discover subjects, split held-out, load behavioral data."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(SUBJ_DIR, exist_ok=True)

    # Discover subjects
    all_subjects = discover_subjects(args.preproc_root)
    log.info(f"Discovered {len(all_subjects)} subjects with preprocessed EO data")
    if not all_subjects:
        log.error("No subjects found. Is preprocessed data downloaded?")
        sys.exit(1)

    # Held-out split
    held_out_path = os.path.join(OUTPUT_DIR, 'held_out_ids.json')
    held_out, analysis_ids = select_held_out(all_subjects)
    with open(held_out_path, 'w') as f:
        json.dump({'held_out': held_out, 'analysis': analysis_ids}, f, indent=2)
    log.info(f"Held-out: {len(held_out)}, Analysis: {len(analysis_ids)}")

    # Select target set
    if args.full_run:
        target_ids = analysis_ids
    else:
        target_ids = held_out
        log.info("DEBUG MODE: processing held-out subjects only")

    # Behavioral data
    demographics = load_demographics(args.meta_path)
    cognitive = load_cognitive_data(args.cog_root)
    master = build_master_table(demographics, cognitive)
    master.to_csv(os.path.join(OUTPUT_DIR, 'master_behavioral.csv'), index=False)
    log.info(f"Master behavioral table: {len(master)} rows")

    # Gamma loss quantification (from EEGMMIDB peaks)
    gamma_loss_quantification()

    return target_ids, master, held_out, analysis_ids


def gamma_loss_quantification():
    """Quantify fraction of EEGMMIDB peaks in 45-50 Hz lost to preprocessing."""
    eegmmidb_path = ('exports_peak_distribution/eegmmidb_fooof/'
                     'golden_ratio_peaks_EEGMMIDB.csv')
    report_path = os.path.join(OUTPUT_DIR, 'gamma_loss_report.txt')

    if not os.path.exists(eegmmidb_path):
        log.warning(f"EEGMMIDB peaks CSV not found at {eegmmidb_path}")
        with open(report_path, 'w') as f:
            f.write("EEGMMIDB peaks CSV not found — cannot quantify gamma loss.\n")
        return

    peaks = pd.read_csv(eegmmidb_path)
    freq_col = [c for c in peaks.columns if 'freq' in c.lower() or 'cf' in c.lower()]
    if not freq_col:
        freq_col = [peaks.columns[1]]  # fallback to second column
    freqs = pd.to_numeric(peaks[freq_col[0]], errors='coerce').dropna().values

    total = len(freqs)
    in_45_50 = np.sum((freqs >= 45) & (freqs <= 50))
    in_gamma = np.sum(freqs >= 36)
    pct_total = 100 * in_45_50 / total if total > 0 else 0
    pct_gamma = 100 * in_45_50 / in_gamma if in_gamma > 0 else 0

    report = (
        f"Gamma Loss Quantification (EEGMMIDB reference)\n"
        f"{'=' * 50}\n"
        f"Total peaks: {total:,}\n"
        f"Peaks in 45-50 Hz: {in_45_50:,} ({pct_total:.1f}% of all peaks)\n"
        f"Peaks >= 36 Hz (gamma): {in_gamma:,}\n"
        f"Fraction of gamma peaks in 45-50 Hz: {pct_gamma:.1f}%\n"
        f"\nPre-processing ceiling at 45 Hz loses {pct_total:.1f}% of total peaks "
        f"and {pct_gamma:.1f}% of gamma-band peaks.\n"
    )
    with open(report_path, 'w') as f:
        f.write(report)
    log.info(f"Gamma loss: {pct_total:.1f}% total, {pct_gamma:.1f}% of gamma band")


# ============================================================================
# PHASE 1: PER-SUBJECT EEG PROCESSING
# ============================================================================

def phase1_preprocessing(target_ids, args):
    """Process each subject: FOOOF + compliance + SIE."""
    features_path = os.path.join(OUTPUT_DIR, 'subject_features.csv')

    if args.skip_preprocessing and os.path.exists(features_path):
        log.info(f"Loading existing features from {features_path}")
        return pd.read_csv(features_path)

    all_features = []
    exclusions = []
    n_total = len(target_ids)

    for i, sid in enumerate(target_ids):
        t0 = time.time()
        log.info(f"[{i + 1}/{n_total}] Processing {sid}...")

        feat = process_single_subject(
            sid,
            preproc_root=args.preproc_root,
            f0=F0_PRIMARY,
            detect_sie=not args.no_sie,
            output_dir=SUBJ_DIR,
        )
        if feat is None:
            exclusions.append({'subject_id': sid, 'reason': 'EO file missing'})
            continue

        all_features.append(feat)
        elapsed = time.time() - t0

        status = 'EXCLUDED' if feat['excluded'] else 'OK'
        log.info(f"  {status} | ch={feat['n_channels_loaded']} "
                 f"fooof={feat['n_channels_fooof_passed']} "
                 f"peaks={feat['n_peaks']} "
                 f"compliance={feat['compliance']:.2f} "
                 f"| {elapsed:.1f}s"
                 + (f" [{feat['exclusion_reason']}]" if feat['excluded'] else ''))

        if feat['excluded']:
            exclusions.append({
                'subject_id': sid,
                'reason': feat['exclusion_reason'],
            })

    features_df = pd.DataFrame(all_features)
    features_df.to_csv(features_path, index=False)

    excl_df = pd.DataFrame(exclusions)
    excl_df.to_csv(os.path.join(OUTPUT_DIR, 'exclusions.csv'), index=False)

    n_included = (~features_df['excluded']).sum()
    n_excluded = features_df['excluded'].sum()
    log.info(f"Phase 1 complete: {n_included} included, "
             f"{n_excluded} excluded, {len(exclusions)} total issues")

    return features_df


# ============================================================================
# PHASE 2: ANALYSIS STEPS 1-9
# ============================================================================

def step1_replication(features_df, master):
    """Step 1: Replication — aggregate lattice histogram + significance."""
    log.info("=" * 60)
    log.info("STEP 1: Replication (pre-registered gate)")
    log.info("=" * 60)

    included = features_df[~features_df['excluded']]

    # Aggregate all peaks
    all_peaks = []
    for sid in included['subject_id']:
        peaks_path = os.path.join(SUBJ_DIR, f'{sid}_peaks.csv')
        if os.path.exists(peaks_path):
            df = pd.read_csv(peaks_path)
            all_peaks.append(df)
    if not all_peaks:
        log.error("No peak files found!")
        return False

    peaks_agg = pd.concat(all_peaks, ignore_index=True)
    freqs = peaks_agg['freq'].values
    log.info(f"Aggregate peaks: {len(freqs):,} from {len(included)} subjects")

    # Lattice coordinates
    u = lattice_coordinate(freqs, F0_PRIMARY, PHI)
    u = u[np.isfinite(u)]

    # Phase-rotation permutation null (1000 iterations)
    positions = natural_positions(PHI)

    # Test boundary depletion
    from ratio_specificity import _enrichment_at_offset
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

    # Boundary depletion: fraction of null more depleted than observed
    p_boundary = (null_boundary <= boundary_enrich).sum() / 1000
    p_noble = (null_noble >= noble_1_enrich).sum() / 1000

    log.info(f"Boundary enrichment: {boundary_enrich:.1f}% (p={p_boundary:.4f})")
    log.info(f"Noble-1 enrichment: {noble_1_enrich:.1f}% (p={p_noble:.4f})")

    # Kendall's tau: ordering test
    position_enrichments = {}
    for name, offset in positions.items():
        position_enrichments[name] = _enrichment_at_offset(
            u, offset, COMPLIANCE_WINDOW, len(u))

    # Expected order: boundary < noble_2 < attractor < noble_1
    expected_order = ['boundary', 'noble_2', 'attractor', 'noble']
    available = [k for k in expected_order if k in position_enrichments]
    observed_values = [position_enrichments[k] for k in available]
    expected_ranks = list(range(len(available)))
    tau, tau_p = stats.kendalltau(expected_ranks, stats.rankdata(observed_values))
    log.info(f"Kendall's tau (ordering): tau={tau:.3f}, p={tau_p:.4f}")

    # Ratio specificity test
    spec_df = ratio_specificity_test(freqs, f0=F0_PRIMARY, n_perm=1000)
    spec_df.to_csv(os.path.join(TABLE_DIR, 'step1_ratio_specificity.csv'),
                   index=False)

    # Save results
    results = {
        'n_peaks': len(freqs),
        'n_subjects': len(included),
        'boundary_enrichment': boundary_enrich,
        'noble_1_enrichment': noble_1_enrich,
        'p_boundary': p_boundary,
        'p_noble': p_noble,
        'kendall_tau': tau,
        'kendall_p': tau_p,
        'position_enrichments': position_enrichments,
    }
    pd.DataFrame([results]).to_csv(
        os.path.join(TABLE_DIR, 'step1_replication.csv'), index=False)

    # Figure 1: Lattice coordinate histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(u, bins=100, density=True, color='steelblue', alpha=0.7,
            edgecolor='white', linewidth=0.3)
    colors = {'boundary': 'red', 'noble_2': 'orange',
              'attractor': 'green', 'noble': 'gold'}
    for name, offset in positions.items():
        c = colors.get(name, 'gray')
        ax.axvline(offset, color=c, linestyle='--', linewidth=2,
                   label=f'{name} ({offset:.3f}): {position_enrichments.get(name, 0):.1f}%')
    ax.set_xlabel('Lattice Coordinate u', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Step 1: Aggregate Lattice Histogram '
                 f'(N={len(freqs):,} peaks, {len(included)} subjects)\n'
                 f'Boundary p={p_boundary:.4f}, Noble-1 p={p_noble:.4f}, '
                 f'Kendall τ={tau:.3f}', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig1_lattice_histogram.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Figure S1: Ratio specificity
    fig, ax = plt.subplots(figsize=(12, 5))
    n_ratios = len(spec_df)
    x = np.arange(n_ratios)
    colors_bar = ['#f39c12' if name == 'φ' else '#3498db'
                  for name in spec_df['ratio_name']]
    ax.bar(x, spec_df['predicted_enrichment'], color=colors_bar, alpha=0.8)
    null_95 = spec_df['null_95th'].median()
    ax.axhline(null_95, color='red', linestyle='--', linewidth=1.5,
               label=f'Null 95th: {null_95:.1f}%')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}\n({v:.3f})" for r, v in
                        zip(spec_df['ratio_name'], spec_df['ratio_value'])],
                       fontsize=8)
    ax.set_ylabel('Predicted-Position Enrichment (%)', fontsize=11)
    ax.set_title('Step 1 (S1): Ratio Specificity — φ vs 8 Alternatives', fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'figS1_ratio_specificity.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # GATE: halt if boundary depletion fails
    if p_boundary > 0.05:
        log.error("GATE FAILED: boundary depletion not significant "
                  f"(p={p_boundary:.4f} > 0.05). Halting.")
        return False

    log.info("GATE PASSED: boundary depletion significant.")
    return True


def step2_reliability(features_df):
    """Step 2: Split-half and spatial ICC reliability."""
    log.info("=" * 60)
    log.info("STEP 2: Reliability (ICC)")
    log.info("=" * 60)

    included = features_df[~features_df['excluded']].dropna(
        subset=['compliance_odd', 'compliance_even'])

    # Split-half ICC
    icc_split, ci_lo_s, ci_hi_s = compute_icc_2way(
        included['compliance_odd'].values,
        included['compliance_even'].values)

    # Spatial ICC
    spatial = included.dropna(
        subset=['compliance_anterior', 'compliance_posterior'])
    icc_spatial, ci_lo_sp, ci_hi_sp = compute_icc_2way(
        spatial['compliance_anterior'].values,
        spatial['compliance_posterior'].values)

    log.info(f"Split-half ICC: {icc_split:.3f} [{ci_lo_s:.3f}, {ci_hi_s:.3f}] "
             f"(N={len(included)})")
    log.info(f"Spatial ICC:    {icc_spatial:.3f} [{ci_lo_sp:.3f}, {ci_hi_sp:.3f}] "
             f"(N={len(spatial)})")
    log.info("NOTE: PCA preprocessing inflates these — interpret as upper bounds")

    results = pd.DataFrame([{
        'measure': 'split_half', 'icc': icc_split,
        'ci_lo': ci_lo_s, 'ci_hi': ci_hi_s, 'n': len(included),
    }, {
        'measure': 'spatial', 'icc': icc_spatial,
        'ci_lo': ci_lo_sp, 'ci_hi': ci_hi_sp, 'n': len(spatial),
    }])
    results.to_csv(os.path.join(TABLE_DIR, 'step2_reliability.csv'), index=False)

    # Figure S2: ICC bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    measures = ['Split-Half\n(odd/even)', 'Spatial\n(ant/post)']
    iccs = [icc_split, icc_spatial]
    ci_lo_arr = [ci_lo_s, ci_lo_sp]
    ci_hi_arr = [ci_hi_s, ci_hi_sp]
    yerr = [[i - lo for i, lo in zip(iccs, ci_lo_arr)],
            [hi - i for i, hi in zip(iccs, ci_hi_arr)]]
    ax.bar([0, 1], iccs, yerr=yerr, color=['steelblue', 'coral'],
           alpha=0.8, capsize=8, edgecolor='black', linewidth=0.5)
    ax.axhline(0.75, color='green', linestyle='--', alpha=0.5, label='Good (0.75)')
    ax.axhline(0.50, color='orange', linestyle='--', alpha=0.5, label='Moderate (0.50)')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(measures)
    ax.set_ylabel('ICC(2,1)', fontsize=11)
    ax.set_title('Step 2: Compliance Reliability\n'
                 '(PCA-inflated — upper bounds)', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'figS2_reliability.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    return results


def step3_descriptives(features_df, master):
    """Step 3: Descriptive statistics and distributions."""
    log.info("=" * 60)
    log.info("STEP 3: Descriptives")
    log.info("=" * 60)

    included = features_df[~features_df['excluded']]
    merged = included.merge(master, on='subject_id', how='inner')

    # Compliance descriptives
    c = merged['compliance']
    desc = {
        'mean': c.mean(), 'sd': c.std(), 'median': c.median(),
        'min': c.min(), 'max': c.max(),
        'skewness': c.skew(), 'kurtosis': c.kurtosis(),
        'shapiro_stat': stats.shapiro(c.dropna())[0] if len(c.dropna()) >= 3 else np.nan,
        'shapiro_p': stats.shapiro(c.dropna())[1] if len(c.dropna()) >= 3 else np.nan,
        'n': len(c.dropna()),
    }
    pd.DataFrame([desc]).to_csv(
        os.path.join(TABLE_DIR, 'step3_compliance_descriptives.csv'), index=False)
    log.info(f"Compliance: M={desc['mean']:.2f}, SD={desc['sd']:.2f}, "
             f"range=[{desc['min']:.2f}, {desc['max']:.2f}]")

    # Channel count distribution
    ch_desc = included['n_channels_loaded'].describe()
    log.info(f"Channels loaded: M={ch_desc['mean']:.1f}, "
             f"range=[{ch_desc['min']:.0f}, {ch_desc['max']:.0f}]")

    # Cognitive test inter-correlations
    test_cols = [t for t in COG_TESTS if t in merged.columns]
    # Use log-transformed where applicable
    corr_cols = []
    for t in test_cols:
        if COG_TESTS[t].get('log_transform') and f'log_{t}' in merged.columns:
            corr_cols.append(f'log_{t}')
        else:
            corr_cols.append(t)

    if corr_cols:
        corr_matrix = merged[corr_cols].corr()
        corr_matrix.to_csv(
            os.path.join(TABLE_DIR, 'step3_cognitive_intercorrelations.csv'))

    # Figure 2: Violin plots by age group
    if 'age_group' in merged.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        groups = merged.dropna(subset=['compliance', 'age_group'])
        if len(groups) > 0:
            sns.violinplot(data=groups, x='age_group', y='compliance',
                           palette=['steelblue', 'coral'], ax=ax, inner='box')
            ax.set_xlabel('Age Group', fontsize=12)
            ax.set_ylabel('Phi-Lattice Compliance', fontsize=12)
            n_young = (groups['age_group'] == 'young').sum()
            n_elder = (groups['age_group'] == 'elderly').sum()
            ax.set_title(f'Step 3: Compliance by Age Group\n'
                         f'Young (n={n_young}) vs Elderly (n={n_elder})',
                         fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'fig2_compliance_violin.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()


def step4_zero_order(features_df, master):
    """Step 4: H1 — Zero-order correlations (compliance × 9 tests)."""
    log.info("=" * 60)
    log.info("STEP 4: H1 — Zero-order correlations")
    log.info("=" * 60)

    included = features_df[~features_df['excluded']]
    # NOTE: subjects with missing IAF ARE included here (IAF not needed)
    merged = included.merge(master, on='subject_id', how='inner')

    results = []
    for test_name, spec in COG_TESTS.items():
        col = f'log_{test_name}' if spec.get('log_transform') else test_name
        if col not in merged.columns:
            continue

        valid = merged.dropna(subset=['compliance', col])
        if len(valid) < 10:
            continue

        # Shapiro-Wilk to decide Pearson vs Spearman
        _, sw_p = stats.shapiro(valid['compliance']) if len(valid) < 5000 else (0, 0)
        if sw_p < 0.05:
            r, p = stats.spearmanr(valid['compliance'], valid[col])
            method = 'Spearman'
        else:
            r, p = stats.pearsonr(valid['compliance'], valid[col])
            method = 'Pearson'

        results.append({
            'test': test_name, 'col_used': col, 'method': method,
            'r': r, 'p': p, 'n': len(valid),
        })
        log.info(f"  {test_name}: {method} r={r:.3f}, p={p:.4f} (n={len(valid)})")

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        adj_p, reject = fdr_correct(results_df['p'].values)
        results_df['p_fdr'] = adj_p
        results_df['significant_fdr'] = reject

    results_df.to_csv(
        os.path.join(TABLE_DIR, 'step4_zero_order_correlations.csv'), index=False)

    # Figure 3: Correlation heatmap
    if not results_df.empty:
        fig, ax = plt.subplots(figsize=(10, 3))
        r_vals = results_df.set_index('test')['r']
        p_fdr_vals = results_df.set_index('test')['p_fdr']

        data_matrix = r_vals.values.reshape(1, -1)
        sns.heatmap(data_matrix, annot=True, fmt='.3f', cmap='RdBu_r',
                    center=0, vmin=-0.4, vmax=0.4,
                    xticklabels=r_vals.index, yticklabels=['Compliance'],
                    ax=ax, linewidths=0.5)
        # Add FDR stars
        for i, (test, p) in enumerate(p_fdr_vals.items()):
            if pd.notna(p) and p < 0.05:
                ax.text(i + 0.5, 0.8, '*', ha='center', va='center',
                        fontsize=16, fontweight='bold', color='black')
        ax.set_title('Step 4: Compliance × Cognition Correlations '
                     '(* = FDR p < 0.05)', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'fig3_correlation_heatmap.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    return results_df


def step5_hierarchical_regression(features_df, master):
    """Step 5: H2 — Hierarchical regression (PRIMARY TEST)."""
    log.info("=" * 60)
    log.info("STEP 5: H2 — Hierarchical regression (PRIMARY)")
    log.info("=" * 60)

    included = features_df[~features_df['excluded']]
    merged = included.merge(master, on='subject_id', how='inner')

    base_predictors = ['age_midpoint', 'sex', 'education_years',
                       'iaf', 'alpha_power_eo', 'mean_aperiodic_exponent',
                       'n_peaks', 'n_channels_fooof_passed']

    results = []
    for test_name, spec in COG_TESTS.items():
        col = f'log_{test_name}' if spec.get('log_transform') else test_name
        if col not in merged.columns:
            continue

        # Listwise deletion per-test
        needed_cols = [col, 'compliance'] + base_predictors
        valid = merged.dropna(subset=needed_cols)
        if len(valid) < 20:
            log.warning(f"  {test_name}: only {len(valid)} valid rows, skipping")
            continue

        y = valid[col].values
        X_base = valid[base_predictors].values
        X_full = np.column_stack([X_base, valid['compliance'].values])
        names_full = base_predictors + ['compliance']

        reg = hierarchical_regression(y, X_base, X_full,
                                      base_predictors, names_full)

        # Bootstrap CI for delta-R²
        ci_lo, ci_hi = bootstrap_delta_r2_ci(y, X_base, X_full)

        row = {
            'test': test_name,
            'col_used': col,
            'n': reg['n'],
            'R2_m1': reg['R2_m1'],
            'R2_m2': reg['R2_m2'],
            'delta_R2': reg['delta_R2'],
            'delta_R2_ci_lo': ci_lo,
            'delta_R2_ci_hi': ci_hi,
            'lrt_chi2': reg['lrt_chi2'],
            'lrt_p': reg['lrt_p'],
            'compliance_beta': reg['compliance_beta'],
            'compliance_se': reg['compliance_se'],
            'compliance_t': reg['compliance_t'],
            'compliance_p': reg['compliance_p'],
            'resid_shapiro_p': reg['resid_shapiro_p'],
            'used_robust_se': reg['used_robust_se'],
        }

        # VIF check
        max_vif_name = max(reg['vifs'], key=reg['vifs'].get) if reg['vifs'] else ''
        max_vif = max(reg['vifs'].values()) if reg['vifs'] else 0
        row['max_vif'] = max_vif
        row['max_vif_predictor'] = max_vif_name

        if max_vif > 5:
            log.warning(f"  {test_name}: VIF > 5 for {max_vif_name} "
                        f"({max_vif:.1f}). Running reduced model.")
            # Re-run without the collinear predictor
            reduced_preds = [p for p in base_predictors if p != max_vif_name]
            X_base_r = valid[reduced_preds].values
            X_full_r = np.column_stack([X_base_r, valid['compliance'].values])
            reg_r = hierarchical_regression(
                y, X_base_r, X_full_r, reduced_preds, reduced_preds + ['compliance'])
            row['delta_R2_reduced'] = reg_r['delta_R2']
            row['compliance_p_reduced'] = reg_r['compliance_p']

        results.append(row)
        log.info(f"  {test_name}: ΔR²={reg['delta_R2']:.4f} "
                 f"[{ci_lo:.4f}, {ci_hi:.4f}], "
                 f"LRT p={reg['lrt_p']:.4f}, "
                 f"β={reg['compliance_beta']:.4f} (n={reg['n']})"
                 + (" [HC3]" if reg['used_robust_se'] else ""))

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        adj_p, reject = fdr_correct(results_df['lrt_p'].values)
        results_df['lrt_p_fdr'] = adj_p
        results_df['significant_fdr'] = reject

        # Success criterion
        any_sig = (results_df['significant_fdr']).any()
        any_large = (results_df['delta_R2'] > 0.02).any()
        log.info(f"SUCCESS criterion: FDR sig={any_sig}, ΔR²>0.02={any_large}")

    results_df.to_csv(
        os.path.join(TABLE_DIR, 'step5_hierarchical_regression.csv'), index=False)

    # Figure 4: Forest plot of ΔR²
    if not results_df.empty and len(results_df) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        y_pos = np.arange(len(results_df))
        colors = ['#2ecc71' if sig else '#95a5a6'
                  for sig in results_df.get('significant_fdr', [False] * len(results_df))]
        xerr = np.array([
            results_df['delta_R2'] - results_df['delta_R2_ci_lo'],
            results_df['delta_R2_ci_hi'] - results_df['delta_R2'],
        ])
        xerr = np.clip(xerr, 0, None)
        ax.barh(y_pos, results_df['delta_R2'], xerr=xerr,
                color=colors, alpha=0.8, capsize=4, edgecolor='black',
                linewidth=0.5, height=0.6)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.axvline(0.02, color='red', linestyle='--', alpha=0.5,
                   label='ΔR² = 0.02 threshold')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(results_df['test'])
        ax.set_xlabel('ΔR² (Model 2 − Model 1)', fontsize=11)
        ax.set_title('Step 5: Incremental Variance Explained by '
                     'Phi-Compliance\n(green = FDR significant)', fontsize=12)
        ax.legend(fontsize=9)
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'fig4_delta_r2_forest.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    return results_df


def step6_age_group(features_df, master):
    """Step 6: H3 — Age group comparison."""
    log.info("=" * 60)
    log.info("STEP 6: H3 — Age group comparison")
    log.info("=" * 60)

    included = features_df[~features_df['excluded']]
    merged = included.merge(master, on='subject_id', how='inner')

    young = merged.loc[merged['age_group'] == 'young', 'compliance'].values
    elderly = merged.loc[merged['age_group'] == 'elderly', 'compliance'].values

    result = run_group_comparison(young, elderly)
    log.info(f"Young: M={result['mean_young']:.2f} (n={result['n_young']})")
    log.info(f"Elderly: M={result['mean_elderly']:.2f} (n={result['n_elderly']})")
    log.info(f"Cohen's d={result['cohens_d']:.3f}, "
             f"t={result['t_stat']:.3f}, p={result['p_ttest']:.4f}")

    pd.DataFrame([result]).to_csv(
        os.path.join(TABLE_DIR, 'step6_age_group_comparison.csv'), index=False)
    return result


def step6b_position_age_decomposition(features_df, master):
    """Step 6b: Position-specific lattice enrichment by age group.

    Decomposes the composite compliance age effect into its four
    lattice-position components (boundary, noble₂, attractor, noble₁)
    and tests each for age-group differences. Also computes aggregate
    lattice histograms per age group from per-subject peak files.
    """
    log.info("=" * 60)
    log.info("STEP 6b: Position-specific age decomposition")
    log.info("=" * 60)

    included = features_df[~features_df['excluded']]
    merged = included.merge(master, on='subject_id', how='inner')

    if 'age_group' not in merged.columns:
        log.warning("  No age_group column — skipping Step 6b")
        return None

    young_df = merged[merged['age_group'] == 'young']
    elderly_df = merged[merged['age_group'] == 'elderly']
    log.info(f"  Young: n={len(young_df)}, Elderly: n={len(elderly_df)}")

    # --- Part A: Per-subject enrichment comparison ---
    positions = {
        'E_boundary': 'Boundary (0.0)',
        'E_noble_2': 'Noble₂ (0.382)',
        'E_attractor': 'Attractor (0.5)',
        'E_noble_1': 'Noble₁ (0.618)',
        'compliance': 'Compliance (composite)',
    }

    results = []
    for col, label in positions.items():
        if col not in merged.columns:
            continue
        res = run_group_comparison(
            young_df[col].dropna().values,
            elderly_df[col].dropna().values,
        )
        res['position'] = label
        res['column'] = col
        log.info(f"  {label}: Young={res['mean_young']:+.2f}, "
                 f"Elderly={res['mean_elderly']:+.2f}, "
                 f"d={res['cohens_d']:.3f}, p={res['p_ttest']:.4f}")
        results.append(res)

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        os.path.join(TABLE_DIR, 'step6b_position_age_decomposition.csv'),
        index=False)

    # --- Part B: Aggregate lattice histograms per age group ---
    per_subj_dir = os.path.join(OUTPUT_DIR, 'per_subject')
    agg_stats = {}
    for grp_name, grp_df in [('young', young_df), ('elderly', elderly_df)]:
        all_freqs = []
        for sid in grp_df['subject_id']:
            pf = os.path.join(per_subj_dir, f'{sid}_peaks.csv')
            if os.path.exists(pf):
                peaks = pd.read_csv(pf)
                if 'freq' in peaks.columns:
                    all_freqs.extend(peaks['freq'].dropna().values.tolist())
        freqs = np.array(all_freqs)
        u = lattice_coordinate(freqs, F0_PRIMARY, PHI)
        u = u[np.isfinite(u)]
        n_total = len(u)
        w = COMPLIANCE_WINDOW
        pos_stats = {'age_group': grp_name, 'n_peaks': n_total}
        for pos_name, offset in [('boundary', 0.0), ('noble_2', 0.382),
                                  ('attractor', 0.5), ('noble_1', 0.618)]:
            in_window = np.sum(np.minimum(np.abs(u - offset),
                                           1 - np.abs(u - offset)) <= w)
            expected = n_total * 2 * w
            enrichment = (in_window - expected) / expected * 100 if expected > 0 else np.nan
            pos_stats[f'{pos_name}_observed'] = int(in_window)
            pos_stats[f'{pos_name}_expected'] = expected
            pos_stats[f'{pos_name}_enrichment_pct'] = enrichment
        agg_stats[grp_name] = (u, pos_stats)
        log.info(f"  Aggregate {grp_name}: {n_total} peaks — "
                 f"boundary={pos_stats['boundary_enrichment_pct']:+.1f}%, "
                 f"attractor={pos_stats['attractor_enrichment_pct']:+.1f}%, "
                 f"noble₁={pos_stats['noble_1_enrichment_pct']:+.1f}%")

    agg_df = pd.DataFrame([s for _, (_, s) in agg_stats.items()])
    agg_df.to_csv(
        os.path.join(TABLE_DIR, 'step6b_aggregate_lattice_by_age.csv'),
        index=False)

    # --- Part C: Figure — 2-panel decomposition ---
    pos_order = ['E_boundary', 'E_noble_2', 'E_attractor', 'E_noble_1']
    pos_labels = ['Boundary\n(0.0)', 'Noble₂\n(0.382)',
                  'Attractor\n(0.5)', 'Noble₁\n(0.618)']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Per-subject position enrichments by age group (grouped bar)
    ax = axes[0]
    x = np.arange(len(pos_order))
    bar_w = 0.35
    young_means = [young_df[c].mean() for c in pos_order]
    young_sems = [young_df[c].sem() for c in pos_order]
    elderly_means = [elderly_df[c].mean() for c in pos_order]
    elderly_sems = [elderly_df[c].sem() for c in pos_order]

    bars_y = ax.bar(x - bar_w / 2, young_means, bar_w, yerr=young_sems,
                    color='steelblue', alpha=0.85, label=f'Young (n={len(young_df)})',
                    capsize=3, edgecolor='white', linewidth=0.5)
    bars_e = ax.bar(x + bar_w / 2, elderly_means, bar_w, yerr=elderly_sems,
                    color='coral', alpha=0.85, label=f'Elderly (n={len(elderly_df)})',
                    capsize=3, edgecolor='white', linewidth=0.5)

    ax.axhline(0, color='grey', linewidth=0.8, linestyle='-')
    ax.set_xticks(x)
    ax.set_xticklabels(pos_labels, fontsize=10)
    ax.set_ylabel('Mean Enrichment (%)', fontsize=11)
    ax.set_title('A. Position-Specific Enrichment by Age Group', fontsize=12,
                 fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')

    # Add significance stars
    for i, col in enumerate(pos_order):
        row = results_df[results_df['column'] == col]
        if len(row) > 0:
            p = row.iloc[0]['p_ttest']
            d = row.iloc[0]['cohens_d']
            star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            y_max = max(abs(young_means[i]) + young_sems[i],
                        abs(elderly_means[i]) + elderly_sems[i])
            y_pos = max(young_means[i] + young_sems[i],
                        elderly_means[i] + elderly_sems[i]) + 3
            ax.text(i, y_pos, f'{star}\nd={d:.2f}',
                    ha='center', va='bottom', fontsize=8)

    # Panel B: Aggregate lattice coordinate histograms overlaid
    ax2 = axes[1]
    bins = np.linspace(0, 1, 101)
    if 'young' in agg_stats and 'elderly' in agg_stats:
        u_young = agg_stats['young'][0]
        u_elderly = agg_stats['elderly'][0]
        ax2.hist(u_young, bins=bins, density=True, alpha=0.5,
                 color='steelblue', label=f'Young ({len(u_young):,} peaks)')
        ax2.hist(u_elderly, bins=bins, density=True, alpha=0.5,
                 color='coral', label=f'Elderly ({len(u_elderly):,} peaks)')

    # Mark lattice positions
    pos_markers = {'Boundary': 0.0, 'Noble₂': 0.382,
                   'Attractor': 0.5, 'Noble₁': 0.618}
    colors = {'Boundary': '#e74c3c', 'Noble₂': '#9b59b6',
              'Attractor': '#3498db', 'Noble₁': '#f39c12'}
    for name, offset in pos_markers.items():
        ax2.axvline(offset, color=colors[name], linewidth=1.5,
                    linestyle='--', alpha=0.8, label=name)
        # Shade the ±window region
        ax2.axvspan(offset - COMPLIANCE_WINDOW, offset + COMPLIANCE_WINDOW,
                    color=colors[name], alpha=0.07)

    ax2.set_xlabel('Lattice Coordinate (u)', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('B. Aggregate Lattice Distribution by Age Group', fontsize=12,
                  fontweight='bold')
    ax2.legend(fontsize=7.5, loc='upper right', ncol=2)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig6b_position_age_decomposition.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"  Saved {fig_path}")

    # --- Part D: Continuous age trends per position ---
    # Plot each enrichment component vs age midpoint to reveal whether
    # degradation is gradual (linear erosion) or threshold-like (cliff at ~60).
    age_col = 'age_midpoint'
    if age_col in merged.columns:
        pos_cols = {
            'E_boundary': ('Boundary (0.0)', '#e74c3c'),
            'E_noble_2': ('Noble₂ (0.382)', '#9b59b6'),
            'E_attractor': ('Attractor (0.5)', '#3498db'),
            'E_noble_1': ('Noble₁ (0.618)', '#f39c12'),
        }

        fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True)
        axes_flat = axes.flatten()

        trend_results = []
        for idx, (col, (label, color)) in enumerate(pos_cols.items()):
            ax = axes_flat[idx]
            valid = merged.dropna(subset=[age_col, col])
            ages = valid[age_col].values
            vals = valid[col].values

            # Scatter (jitter age slightly for visibility at discrete bins)
            jitter = np.random.default_rng(42).uniform(-0.8, 0.8, len(ages))
            ax.scatter(ages + jitter, vals, alpha=0.35, s=18, color=color,
                       edgecolors='white', linewidths=0.3)

            # Per-bin means with SEM error bars
            bin_stats = valid.groupby(age_col)[col].agg(['mean', 'sem', 'count'])
            bin_stats = bin_stats[bin_stats['count'] >= 3]  # require ≥3 per bin
            ax.errorbar(bin_stats.index, bin_stats['mean'], yerr=bin_stats['sem'],
                        fmt='o-', color=color, markersize=7, linewidth=2,
                        capsize=4, markeredgecolor='white', markeredgewidth=0.8,
                        zorder=5, label='Bin mean ± SEM')

            # Linear regression line
            slope, intercept, r, p, se = stats.linregress(ages, vals)
            x_line = np.array([ages.min(), ages.max()])
            ax.plot(x_line, intercept + slope * x_line, '--', color='grey',
                    linewidth=1.5, alpha=0.7)

            ax.axhline(0, color='grey', linewidth=0.6, linestyle=':')
            ax.set_ylabel('Enrichment (%)', fontsize=10)
            ax.set_title(f'{label}\nr={r:.3f}, p={p:.3f}, slope={slope:.2f}/yr',
                         fontsize=11, fontweight='bold')
            ax.legend(fontsize=8, loc='best')

            trend_results.append({
                'position': label, 'column': col,
                'r': r, 'p': p, 'slope': slope, 'intercept': intercept, 'se': se,
            })
            log.info(f"  Age trend {label}: r={r:.3f}, p={p:.3f}, "
                     f"slope={slope:.2f}/yr")

        axes[1, 0].set_xlabel('Age (midpoint)', fontsize=11)
        axes[1, 1].set_xlabel('Age (midpoint)', fontsize=11)
        fig.suptitle('Position-Specific Enrichment vs Age (Continuous)',
                     fontsize=13, fontweight='bold', y=1.01)
        plt.tight_layout()
        fig_path2 = os.path.join(FIG_DIR, 'fig6c_position_age_trends.png')
        plt.savefig(fig_path2, dpi=300, bbox_inches='tight')
        plt.close()
        log.info(f"  Saved {fig_path2}")

        pd.DataFrame(trend_results).to_csv(
            os.path.join(TABLE_DIR, 'step6b_position_age_trends.csv'),
            index=False)

    return results_df


def step7_mediation(features_df, master):
    """Step 7: H4 — Mediation (Age → Compliance → Cognition)."""
    log.info("=" * 60)
    log.info("STEP 7: H4 — Mediation")
    log.info("=" * 60)

    included = features_df[~features_df['excluded']]
    merged = included.merge(master, on='subject_id', how='inner')

    results = []
    for test_name, spec in COG_TESTS.items():
        col = f'log_{test_name}' if spec.get('log_transform') else test_name
        if col not in merged.columns:
            continue

        needed = ['age_midpoint', 'compliance', col, 'sex', 'education_years']
        valid = merged.dropna(subset=needed)
        if len(valid) < 30:
            continue

        # Check age effect first
        r_age, p_age = stats.pearsonr(valid['age_midpoint'], valid[col])
        if p_age > 0.10:
            log.info(f"  {test_name}: no age effect (r={r_age:.3f}, p={p_age:.3f}), skipping")
            continue

        med = bootstrap_mediation(
            X=valid['age_midpoint'].values,
            M=valid['compliance'].values,
            Y=valid[col].values,
            covariates=valid[['sex', 'education_years']].values,
            n_boot=5000, seed=42,
        )
        med['test'] = test_name
        med['age_r'] = r_age
        med['age_p'] = p_age
        med['n'] = len(valid)
        results.append(med)

        sig = 'SIG' if (med['indirect_ci_lo'] > 0 or med['indirect_ci_hi'] < 0) else 'NS'
        log.info(f"  {test_name}: indirect={med['indirect_effect']:.4f} "
                 f"[{med['indirect_ci_lo']:.4f}, {med['indirect_ci_hi']:.4f}] "
                 f"{sig} (n={len(valid)})")

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        os.path.join(TABLE_DIR, 'step7_mediation.csv'), index=False)

    # Figure 5: Path diagram for strongest mediation
    if not results_df.empty:
        # Find strongest (largest |indirect|)
        best = results_df.loc[results_df['indirect_effect'].abs().idxmax()]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis('off')

        # Boxes
        for (x, y, label) in [(1, 3, 'Age'), (5, 5.5, 'Compliance'),
                               (9, 3, best['test'])]:
            ax.add_patch(plt.Rectangle((x - 0.8, y - 0.4), 1.6, 0.8,
                                       facecolor='lightblue', edgecolor='black'))
            ax.text(x, y, label, ha='center', va='center', fontsize=11,
                    fontweight='bold')

        # Arrows with coefficients
        ax.annotate('', xy=(4.2, 5.5), xytext=(1.8, 3.4),
                    arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
        ax.text(2.5, 4.7, f'a = {best["a_coef"]:.4f}', fontsize=10, color='blue')

        ax.annotate('', xy=(8.2, 3.4), xytext=(5.8, 5.1),
                    arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
        ax.text(7.3, 4.5, f'b = {best["b_coef"]:.4f}', fontsize=10, color='blue')

        ax.annotate('', xy=(8.2, 3), xytext=(1.8, 3),
                    arrowprops=dict(arrowstyle='->', lw=2, color='gray',
                                    linestyle='dashed'))
        ax.text(5, 2.3, f"c' = {best['direct_effect']:.4f}", fontsize=10,
                color='gray')

        ax.text(5, 1.2, f"Indirect: {best['indirect_effect']:.4f} "
                f"[{best['indirect_ci_lo']:.4f}, {best['indirect_ci_hi']:.4f}]\n"
                f"Proportion mediated: {best['proportion_mediated']:.1%}",
                ha='center', fontsize=10)
        ax.set_title(f'Step 7: Mediation — Age → Compliance → {best["test"]}',
                     fontsize=12, fontweight='bold')
        plt.savefig(os.path.join(FIG_DIR, 'fig5_mediation_path.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    return results_df


def step8_band_specific(features_df, master):
    """Step 8: H5/H6 — Band-specific compliance × cognition."""
    log.info("=" * 60)
    log.info("STEP 8: H5/H6 — Band-specific (exploratory)")
    log.info("=" * 60)

    included = features_df[~features_df['excluded']]
    merged = included.merge(master, on='subject_id', how='inner')

    bands = ['theta', 'alpha', 'beta_low', 'beta_high', 'gamma']
    band_cols = [f'compliance_{b}' for b in bands]

    results = []
    for test_name, spec in COG_TESTS.items():
        col = f'log_{test_name}' if spec.get('log_transform') else test_name
        if col not in merged.columns:
            continue

        for band, band_col in zip(bands, band_cols):
            if band_col not in merged.columns:
                continue
            valid = merged.dropna(subset=[band_col, col])
            if len(valid) < 10:
                continue
            r, p = stats.pearsonr(valid[band_col], valid[col])
            results.append({
                'test': test_name, 'band': band, 'r': r, 'p': p,
                'n': len(valid),
            })

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        # FDR across all 45 tests
        adj_p, reject = fdr_correct(results_df['p'].values)
        results_df['p_fdr'] = adj_p
        results_df['significant_fdr'] = reject

    results_df.to_csv(
        os.path.join(TABLE_DIR, 'step8_band_specific.csv'), index=False)

    # Figure 6: Band × test heatmap
    if not results_df.empty:
        pivot_r = results_df.pivot(index='band', columns='test', values='r')
        pivot_p = results_df.pivot(index='band', columns='test', values='p_fdr')

        # Reorder bands
        band_order = [b for b in bands if b in pivot_r.index]
        pivot_r = pivot_r.reindex(band_order)
        pivot_p = pivot_p.reindex(band_order)

        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(pivot_r, annot=True, fmt='.3f', cmap='RdBu_r',
                    center=0, vmin=-0.3, vmax=0.3, ax=ax,
                    linewidths=0.5, linecolor='white')

        # Add stars for significant
        if pivot_p is not None:
            for i, band in enumerate(pivot_r.index):
                for j, test in enumerate(pivot_r.columns):
                    try:
                        p = pivot_p.loc[band, test]
                        if pd.notna(p) and p < 0.05:
                            ax.text(j + 0.5, i + 0.2, '*', ha='center',
                                    fontsize=14, fontweight='bold')
                    except KeyError:
                        pass

        ax.set_title('Step 8: Band-Specific Compliance × Cognition\n'
                     '(* = FDR p < 0.05; gamma truncated at 45 Hz)',
                     fontsize=12)
        ax.set_ylabel('Phi-Octave Band', fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'fig6_band_heatmap.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    return results_df


def step9_exploratory(features_df, master):
    """Step 9: Exploratory analyses (position-specific + SIE)."""
    log.info("=" * 60)
    log.info("STEP 9: Exploratory (Tier 3)")
    log.info("=" * 60)

    included = features_df[~features_df['excluded']]
    merged = included.merge(master, on='subject_id', how='inner')

    # --- 9a: Position-specific enrichments ---
    log.info("  9a: Position-specific enrichments as predictors")
    enrichment_cols = ['E_boundary', 'E_noble_2', 'E_attractor', 'E_noble_1']
    results_9a = []

    for test_name, spec in COG_TESTS.items():
        col = f'log_{test_name}' if spec.get('log_transform') else test_name
        if col not in merged.columns:
            continue
        for e_col in enrichment_cols:
            if e_col not in merged.columns:
                continue
            valid = merged.dropna(subset=[e_col, col])
            if len(valid) < 10:
                continue
            r, p = stats.pearsonr(valid[e_col], valid[col])
            results_9a.append({
                'test': test_name, 'enrichment': e_col, 'r': r, 'p': p,
                'n': len(valid),
            })

    results_9a_df = pd.DataFrame(results_9a)
    if not results_9a_df.empty:
        adj_p, reject = fdr_correct(results_9a_df['p'].values)
        results_9a_df['p_fdr'] = adj_p
        results_9a_df['significant_fdr'] = reject
    results_9a_df.to_csv(
        os.path.join(TABLE_DIR, 'step9a_position_specific.csv'), index=False)

    # --- 9b: SIE Rate Analysis ---
    log.info("  9b: SIE Rate Analysis")
    results_9b = {}

    if 'sie_rate' in merged.columns and merged['sie_rate'].notna().sum() > 10:
        sie_valid = merged.dropna(subset=['sie_rate'])
        sie_rate = sie_valid['sie_rate']

        # Descriptives
        sie_desc = {
            'n_total': len(sie_valid),
            'n_zero_events': (sie_valid['n_ignitions'] == 0).sum(),
            'median_rate': sie_rate.median(),
            'iqr_lo': sie_rate.quantile(0.25),
            'iqr_hi': sie_rate.quantile(0.75),
            'max_rate': sie_rate.max(),
            'mean_rate': sie_rate.mean(),
        }
        pd.DataFrame([sie_desc]).to_csv(
            os.path.join(TABLE_DIR, 'step9b_sie_descriptives.csv'), index=False)
        log.info(f"  SIE: {sie_desc['n_total']} subjects, "
                 f"{sie_desc['n_zero_events']} with 0 events, "
                 f"median rate={sie_desc['median_rate']:.2f}/min")

        # Zero-order: Spearman sie_rate × 9 tests
        sie_corr = []
        for test_name, spec in COG_TESTS.items():
            col = f'log_{test_name}' if spec.get('log_transform') else test_name
            if col not in merged.columns:
                continue
            valid = merged.dropna(subset=['sie_rate', col])
            if len(valid) < 10:
                continue
            rho, p = stats.spearmanr(valid['sie_rate'], valid[col])
            sie_corr.append({
                'test': test_name, 'rho': rho, 'p': p, 'n': len(valid),
            })
        sie_corr_df = pd.DataFrame(sie_corr)
        if not sie_corr_df.empty:
            adj_p, reject = fdr_correct(sie_corr_df['p'].values)
            sie_corr_df['p_fdr'] = adj_p
            sie_corr_df['significant_fdr'] = reject
        sie_corr_df.to_csv(
            os.path.join(TABLE_DIR, 'step9b_sie_correlations.csv'), index=False)

        # SIE-compliance relationship
        comp_sie = merged.dropna(subset=['compliance', 'sie_rate'])
        if len(comp_sie) > 10:
            rho, p = stats.spearmanr(comp_sie['compliance'], comp_sie['sie_rate'])
            log.info(f"  Compliance-SIE: Spearman ρ={rho:.3f}, p={p:.4f}")
            results_9b['compliance_sie_rho'] = rho
            results_9b['compliance_sie_p'] = p

        # Age-group comparison for SIE rate
        if 'age_group' in merged.columns:
            y_sie = merged.loc[merged['age_group'] == 'young', 'sie_rate'].dropna()
            e_sie = merged.loc[merged['age_group'] == 'elderly', 'sie_rate'].dropna()
            if len(y_sie) > 2 and len(e_sie) > 2:
                U, p_u = stats.mannwhitneyu(y_sie, e_sie, alternative='two-sided')
                log.info(f"  SIE age group: U={U:.0f}, p={p_u:.4f}")

        # Pre-registered test: Model 2a (SIE replacing compliance, 9 predictors)
        base_predictors = ['age_midpoint', 'sex', 'education_years',
                           'iaf', 'alpha_power_eo', 'mean_aperiodic_exponent',
                           'n_peaks', 'n_channels_fooof_passed']
        sie_reg_results = []
        for test_name, spec in COG_TESTS.items():
            col = f'log_{test_name}' if spec.get('log_transform') else test_name
            if col not in merged.columns:
                continue
            needed = [col, 'sie_rate'] + base_predictors
            valid = merged.dropna(subset=needed)
            if len(valid) < 20:
                continue
            y = valid[col].values
            X_base = valid[base_predictors].values
            X_full = np.column_stack([X_base, valid['sie_rate'].values])
            names_full = base_predictors + ['sie_rate']
            reg = hierarchical_regression(y, X_base, X_full,
                                          base_predictors, names_full)
            sie_reg_results.append({
                'test': test_name, 'model': '2a_sie_replacing',
                'delta_R2': reg['delta_R2'], 'lrt_p': reg['lrt_p'],
                'sie_beta': reg['compliance_beta'],
                'sie_p': reg['compliance_p'], 'n': reg['n'],
            })

        # Post-hoc exploratory: Model 2b (both compliance + SIE, 10 predictors)
        for test_name, spec in COG_TESTS.items():
            col = f'log_{test_name}' if spec.get('log_transform') else test_name
            if col not in merged.columns:
                continue
            needed = [col, 'compliance', 'sie_rate'] + base_predictors
            valid = merged.dropna(subset=needed)
            if len(valid) < 20:
                continue
            y = valid[col].values
            X_base_comp = np.column_stack([
                valid[base_predictors].values,
                valid['compliance'].values])
            X_full_both = np.column_stack([X_base_comp, valid['sie_rate'].values])
            names_base_comp = base_predictors + ['compliance']
            names_full_both = names_base_comp + ['sie_rate']
            reg = hierarchical_regression(
                y, X_base_comp, X_full_both,
                names_base_comp, names_full_both)
            sie_reg_results.append({
                'test': test_name, 'model': '2b_both_posthoc',
                'delta_R2': reg['delta_R2'], 'lrt_p': reg['lrt_p'],
                'sie_beta': reg['compliance_beta'],
                'sie_p': reg['compliance_p'], 'n': reg['n'],
            })

        pd.DataFrame(sie_reg_results).to_csv(
            os.path.join(TABLE_DIR, 'step9b_sie_regression.csv'), index=False)

        # Figure 7: SIE rate distribution + compliance-SIE scatter
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Left: SIE rate distribution
        rate_data = merged['sie_rate'].dropna()
        ax1.hist(rate_data, bins=30, color='steelblue', alpha=0.7,
                 edgecolor='white')
        ax1.axvline(rate_data.median(), color='red', linestyle='--',
                    label=f'Median: {rate_data.median():.2f}')
        ax1.set_xlabel('SIE Rate (events/min)', fontsize=11)
        ax1.set_ylabel('Count', fontsize=11)
        ax1.set_title(f'SIE Rate Distribution (N={len(rate_data)})', fontsize=11)
        ax1.legend()

        # Right: Compliance-SIE scatter
        if len(comp_sie) > 5:
            ax2.scatter(comp_sie['compliance'], comp_sie['sie_rate'],
                        alpha=0.5, s=20, color='steelblue')
            ax2.set_xlabel('Phi-Lattice Compliance', fontsize=11)
            ax2.set_ylabel('SIE Rate (events/min)', fontsize=11)
            rho_str = f'ρ={results_9b.get("compliance_sie_rho", 0):.3f}'
            ax2.set_title(f'Compliance vs SIE Rate ({rho_str})', fontsize=11)

        plt.suptitle('Step 9b: Schumann Ignition Events (exploratory)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'fig7_sie_analysis.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
    else:
        log.info("  SIE data not available or insufficient")

    return results_9a_df


# ============================================================================
# PHASE 3: SENSITIVITY ANALYSES
# ============================================================================

def phase3_sensitivity(features_df, master, args):
    """Run pre-registered sensitivity analyses (a-k)."""
    log.info("=" * 60)
    log.info("PHASE 3: Sensitivity Analyses")
    log.info("=" * 60)

    included = features_df[~features_df['excluded']]
    merged = included.merge(master, on='subject_id', how='inner')

    sensitivity_results = {}

    base_predictors = ['age_midpoint', 'sex', 'education_years',
                       'iaf', 'alpha_power_eo', 'mean_aperiodic_exponent',
                       'n_peaks', 'n_channels_fooof_passed']

    # Helper: run regression for one variant
    def _run_variant(label, merged_var, compliance_col='compliance',
                     extra_predictors=None):
        results = []
        preds = base_predictors.copy()
        if extra_predictors:
            preds += extra_predictors
        for test_name, spec in COG_TESTS.items():
            col = f'log_{test_name}' if spec.get('log_transform') else test_name
            if col not in merged_var.columns or compliance_col not in merged_var.columns:
                continue
            needed = [col, compliance_col] + preds
            valid = merged_var.dropna(subset=needed)
            if len(valid) < 20:
                continue
            y = valid[col].values
            X_base = valid[preds].values
            X_full = np.column_stack([X_base, valid[compliance_col].values])
            try:
                reg = hierarchical_regression(
                    y, X_base, X_full, preds, preds + [compliance_col])
                results.append({
                    'test': test_name, 'delta_R2': reg['delta_R2'],
                    'lrt_p': reg['lrt_p'], 'n': reg['n'],
                    'compliance_beta': reg['compliance_beta'],
                })
            except Exception as e:
                log.warning(f"  {label}/{test_name}: {e}")
        return pd.DataFrame(results)

    # (b) f0=7.6 Hz sensitivity
    log.info("  (b) f0=7.6 Hz sensitivity")
    # Recompute compliance at f0=7.6 from saved peaks
    f0_alt_features = []
    for sid in included['subject_id']:
        peaks_path = os.path.join(SUBJ_DIR, f'{sid}_peaks.csv')
        if os.path.exists(peaks_path):
            pf = pd.read_csv(peaks_path)
            comp = compute_compliance_score(pf['freq'].values, f0=F0_SENSITIVITY)
            f0_alt_features.append({
                'subject_id': sid,
                'compliance_f076': comp['compliance'],
            })
    if f0_alt_features:
        f0_df = pd.DataFrame(f0_alt_features)
        merged_b = merged.merge(f0_df, on='subject_id', how='inner')
        sens_b = _run_variant('f0=7.6', merged_b, compliance_col='compliance_f076')
        sens_b.to_csv(os.path.join(TABLE_DIR, 'sensitivity_b_f076.csv'), index=False)
        sensitivity_results['b_f076'] = sens_b

    # (c) Strict FOOOF: mean R² < 0.90 excluded
    log.info("  (c) Strict FOOOF R² >= 0.90")
    strict = merged[merged['mean_r_squared'] >= 0.90] if 'mean_r_squared' in merged.columns else merged
    sens_c = _run_variant('strict_fooof', strict)
    sens_c.to_csv(os.path.join(TABLE_DIR, 'sensitivity_c_strict_fooof.csv'), index=False)
    sensitivity_results['c_strict_fooof'] = sens_c
    log.info(f"    N={len(strict)} (of {len(merged)})")

    # (d) Recording length covariate
    log.info("  (d) Recording length covariate")
    if 'duration_sec' in merged.columns:
        sens_d = _run_variant('duration_cov', merged,
                              extra_predictors=['duration_sec'])
        sens_d.to_csv(os.path.join(TABLE_DIR, 'sensitivity_d_duration.csv'),
                      index=False)
        sensitivity_results['d_duration'] = sens_d

    # (h) Peak count × compliance interaction
    log.info("  (h) Peak count × compliance interaction")
    merged_h = merged.copy()
    if 'n_peaks' in merged_h.columns and 'compliance' in merged_h.columns:
        merged_h['peak_x_compliance'] = merged_h['n_peaks'] * merged_h['compliance']
        sens_h = _run_variant('interaction', merged_h,
                              extra_predictors=['peak_x_compliance'])
        sens_h.to_csv(os.path.join(TABLE_DIR, 'sensitivity_h_interaction.csv'),
                      index=False)
        sensitivity_results['h_interaction'] = sens_h

    # (i) Categorical age-group replacing continuous midpoints
    log.info("  (i) Categorical age group")
    if 'age_group' in merged.columns:
        merged_i = merged.copy()
        merged_i['age_categorical'] = (merged_i['age_group'] == 'elderly').astype(int)
        preds_i = ['age_categorical'] + base_predictors[1:]  # replace age_midpoint
        sens_i_results = []
        for test_name, spec in COG_TESTS.items():
            col = f'log_{test_name}' if spec.get('log_transform') else test_name
            if col not in merged_i.columns:
                continue
            needed = [col, 'compliance'] + preds_i
            valid = merged_i.dropna(subset=needed)
            if len(valid) < 20:
                continue
            y = valid[col].values
            X_base = valid[preds_i].values
            X_full = np.column_stack([X_base, valid['compliance'].values])
            try:
                reg = hierarchical_regression(y, X_base, X_full,
                                              preds_i, preds_i + ['compliance'])
                sens_i_results.append({
                    'test': test_name, 'delta_R2': reg['delta_R2'],
                    'lrt_p': reg['lrt_p'], 'n': reg['n'],
                })
            except Exception:
                pass
        sens_i = pd.DataFrame(sens_i_results)
        sens_i.to_csv(os.path.join(TABLE_DIR, 'sensitivity_i_categorical_age.csv'),
                      index=False)
        sensitivity_results['i_categorical_age'] = sens_i

    # (k) Channel count restriction: ≥55 surviving channels
    log.info("  (k) Channel count restriction (≥55 channels)")
    if 'n_channels_loaded' in merged.columns:
        strict_ch = merged[merged['n_channels_loaded'] >= 55]
        sens_k = _run_variant('ch_restrict', strict_ch)
        sens_k.to_csv(os.path.join(TABLE_DIR, 'sensitivity_k_channel_restrict.csv'),
                      index=False)
        sensitivity_results['k_channel_restrict'] = sens_k
        log.info(f"    N={len(strict_ch)} (of {len(merged)})")

    # (l) max_n_peaks=40 (peak count ceiling sensitivity)
    # At LEMON's 0.122 Hz resolution, max_n_peaks=20 is a binding ceiling
    # on ~54% of channels. This re-runs FOOOF with max=40 (where <5% hit ceiling)
    # to test robustness of compliance-cognition relationships.
    log.info("  (l) max_n_peaks=40 (peak count ceiling)")
    try:
        l_features = []
        preproc_root = args.preproc_root if hasattr(args, 'preproc_root') else LEMON_PREPROC_ROOT
        for si, sid in enumerate(included['subject_id']):
            raw_eo, info_eo = load_preprocessed_subject(sid, preproc_root, condition='EO')
            if raw_eo is None:
                continue
            params_40 = FOOOF_PARAMS.copy()
            params_40['max_n_peaks'] = 40
            peaks_df, ch_info = extract_fooof_peaks_subject(
                raw_eo, fs=SFREQ, fooof_params=params_40)
            comp = compute_compliance_score(
                peaks_df['freq'].values if not peaks_df.empty else np.array([]))
            l_features.append({
                'subject_id': sid,
                'compliance_max40': comp['compliance'],
                'n_peaks_max40': ch_info['total_peak_count'],
            })
            del raw_eo
            gc.collect()
            if (si + 1) % 20 == 0:
                log.info(f"    [{si+1}/{len(included)}] re-extracted")

        if l_features:
            l_df = pd.DataFrame(l_features)
            merged_l = merged.merge(l_df, on='subject_id', how='inner')
            sens_l = _run_variant('max_n_peaks=40', merged_l,
                                  compliance_col='compliance_max40')
            sens_l.to_csv(os.path.join(TABLE_DIR, 'sensitivity_l_max_peaks_40.csv'),
                          index=False)
            sensitivity_results['l_max_peaks_40'] = sens_l

            # Report rank-order preservation
            from scipy.stats import spearmanr
            both = merged_l[['compliance', 'compliance_max40']].dropna()
            if len(both) > 5:
                rho, p_rho = spearmanr(both['compliance'], both['compliance_max40'])
                log.info(f"    N={len(both)}, Spearman rho(max20,max40)={rho:.3f} (p={p_rho:.4f})")
                log.info(f"    Mean peaks: max20={merged_l['n_peaks'].mean():.0f}, "
                         f"max40={merged_l['n_peaks_max40'].mean():.0f}")
    except Exception as e:
        log.error(f"  Sensitivity (l) failed: {e}")

    # Sensitivity analyses requiring re-processing (a, e, f, g, j) are logged
    # as requiring separate runs
    for label in ['a_eyes_closed', 'e_nperseg', 'f_bin_width',
                  'g_spatial', 'j_raw_pipeline']:
        log.info(f"  ({label[0]}): Requires re-processing — deferred")

    return sensitivity_results


# ============================================================================
# PHASE 4: REPORT GENERATION
# ============================================================================

def phase4_report(features_df, master):
    """Generate summary report."""
    log.info("=" * 60)
    log.info("PHASE 4: Report Generation")
    log.info("=" * 60)

    included = features_df[~features_df['excluded']]

    lines = [
        "# LEMON Phi-Lattice Cognition Analysis — Summary Report",
        "",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}",
        f"**Total subjects processed**: {len(features_df)}",
        f"**Included**: {len(included)}",
        f"**Excluded**: {features_df['excluded'].sum()}",
        "",
        "## CONSORT Flow",
        "",
    ]

    # Exclusion reasons
    excluded = features_df[features_df['excluded']]
    if not excluded.empty:
        reasons = excluded['exclusion_reason'].value_counts()
        for reason, count in reasons.items():
            lines.append(f"- {reason}: {count}")
    lines.append("")

    # Key metrics
    lines.append("## Key Metrics")
    lines.append("")
    lines.append(f"- Mean compliance: {included['compliance'].mean():.2f} "
                 f"(SD={included['compliance'].std():.2f})")
    lines.append(f"- Mean channels loaded: {included['n_channels_loaded'].mean():.1f}")
    lines.append(f"- Mean FOOOF R²: {included['mean_r_squared'].mean():.3f}")
    lines.append(f"- Mean IAF: {included['iaf'].mean():.2f} Hz "
                 f"(N with IAF: {included['iaf'].notna().sum()})")
    if 'sie_rate' in included.columns:
        sie_valid = included['sie_rate'].dropna()
        lines.append(f"- SIE rate: median={sie_valid.median():.2f}/min "
                     f"(N with SIE data: {len(sie_valid)})")
    lines.append("")

    # Figure list
    lines.append("## Figures")
    lines.append("")
    for fn in sorted(os.listdir(FIG_DIR)):
        if fn.endswith('.png'):
            lines.append(f"- ![{fn}](figures/{fn})")
    lines.append("")

    # Table list
    lines.append("## Tables")
    lines.append("")
    for fn in sorted(os.listdir(TABLE_DIR)):
        if fn.endswith('.csv'):
            lines.append(f"- [{fn}](tables/{fn})")

    report_path = os.path.join(OUTPUT_DIR, 'summary_report.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    log.info(f"Report saved to {report_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='LEMON Phi-Lattice Cognition Analysis')
    parser.add_argument('--full-run', action='store_true',
                        help='Process all analysis subjects (not just held-out)')
    parser.add_argument('--skip-preprocessing', action='store_true',
                        help='Skip EEG processing, load from existing CSV')
    parser.add_argument('--sensitivity', action='store_true',
                        help='Run sensitivity analyses')
    parser.add_argument('--no-sie', action='store_true',
                        help='Skip SIE detection')
    parser.add_argument('--preproc-root', default=LEMON_PREPROC_ROOT,
                        help='Preprocessed EEG root directory')
    parser.add_argument('--meta-path', default=LEMON_META_PATH,
                        help='META CSV path')
    parser.add_argument('--cog-root', default=LEMON_COG_ROOT,
                        help='Cognitive test data root')
    args = parser.parse_args()

    t_start = time.time()
    log.info("LEMON Phi-Lattice Cognition Analysis")
    log.info(f"Mode: {'FULL' if args.full_run else 'DEBUG (held-out only)'}")
    log.info(f"SIE: {'enabled' if not args.no_sie else 'disabled'}")

    # PHASE 0
    target_ids, master, held_out, analysis_ids = phase0_setup(args)

    # PHASE 1
    features_df = phase1_preprocessing(target_ids, args)

    # PHASE 2
    gate_passed = step1_replication(features_df, master)
    if not gate_passed:
        log.error("Step 1 gate FAILED. Proceeding with remaining steps "
                  "but results are exploratory only.")

    step2_reliability(features_df)
    step3_descriptives(features_df, master)
    step4_zero_order(features_df, master)
    step5_hierarchical_regression(features_df, master)
    step6_age_group(features_df, master)
    step6b_position_age_decomposition(features_df, master)
    step7_mediation(features_df, master)
    step8_band_specific(features_df, master)
    step9_exploratory(features_df, master)

    # PHASE 3
    if args.sensitivity:
        phase3_sensitivity(features_df, master, args)

    # PHASE 4
    phase4_report(features_df, master)

    elapsed = time.time() - t_start
    log.info(f"Pipeline complete in {elapsed / 60:.1f} minutes")


if __name__ == '__main__':
    main()

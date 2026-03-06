#!/usr/bin/env python3
"""
Amplitude-Weighted Compliance Analysis for LEMON
==================================================

Step 2: Loads saved peak CSVs (from 85 Hz re-extraction), computes weighted
compliance metrics, runs age analyses. No EEG loading.

Phases:
  0. Master peak table + f₀ correction
  1. Per-subject weighted compliance
  2. Attractor age trend — weighted vs unweighted
  3. max_n_peaks=40 robustness (critical test)
  4. Prominence mediation
  5. Per-subject within-band shuffle z-scores
  6. Summary figures

Requires: exports_lemon/per_subject_85hz/ from run_lemon_reextract_peaks.py

Usage:
    python scripts/run_lemon_amplitude_weighted.py
    python scripts/run_lemon_amplitude_weighted.py --phases 0 1 2
    python scripts/run_lemon_amplitude_weighted.py --phases 0b  # redistribution only
"""

import os
import sys
import argparse
import time
import logging
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from ratio_specificity import lattice_coordinate, PHI
from structural_phi_specificity import natural_positions, compute_structural_score
from continuous_compliance import (
    SIGMA_DEFAULT,
    continuous_structural_score,
    continuous_compliance_score,
    kernel_enrichment,
    weighted_compliance_score,
    weighted_structural_score,
    weighted_within_band_shuffle,
    _apply_weight_transform,
)
from lemon_utils import (
    F0_PRIMARY, COMPLIANCE_WINDOW, make_phi_bands,
    compute_compliance_score, run_group_comparison,
    fdr_correct, bootstrap_mediation,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# Paths
EXPORTS = os.path.join(os.path.dirname(__file__), '..', 'exports_lemon')
PEAKS_85HZ = os.path.join(EXPORTS, 'per_subject_85hz')
PEAKS_45HZ = os.path.join(EXPORTS, 'per_subject')
PEAKS_RAW = os.path.join(EXPORTS, 'per_subject_raw85hz')
OUT_DIR = os.path.join(EXPORTS, 'amplitude_weighted')
FIG_DIR = os.path.join(OUT_DIR, 'figures')
PEAKS_ENRICHED_DIR = os.path.join(OUT_DIR, 'peaks')

# Constants
SIGMA = SIGMA_DEFAULT  # 0.035
WEIGHT_TRANSFORMS = ['rank', 'zscore', 'linear', 'raw']
F0_SWEEP = np.arange(6.5, 9.55, 0.05)  # 61 candidates
F0_OPT_MIN_PEAKS = 50  # guard against noise-fitting


def load_subject_features():
    """Load subject_features + master_behavioral merged."""
    sf = pd.read_csv(os.path.join(EXPORTS, 'subject_features.csv'))
    mb = pd.read_csv(os.path.join(EXPORTS, 'master_behavioral.csv'))
    df = sf.merge(mb, on='subject_id', how='inner')
    df = df[df['excluded'] != True].copy()
    return df


def load_peaks_85hz(sid, suffix='_peaks'):
    """Load per-subject peak CSV from 85 Hz extraction."""
    path = os.path.join(PEAKS_85HZ, f'{sid}{suffix}.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty or 'freq' not in df.columns:
        return None
    return df


def load_peaks_45hz(sid, suffix='_peaks'):
    """Load original [1, 45] Hz peaks."""
    path = os.path.join(PEAKS_45HZ, f'{sid}{suffix}.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty or 'freq' not in df.columns:
        return None
    return df


def available_subjects_85hz(suffix='_peaks'):
    """List subject IDs with 85 Hz peak CSVs."""
    sids = []
    if not os.path.exists(PEAKS_85HZ):
        return sids
    for f in sorted(os.listdir(PEAKS_85HZ)):
        if f.endswith(f'{suffix}.csv') and f.startswith('sub-'):
            sid = f.replace(f'{suffix}.csv', '')
            sids.append(sid)
    return sids


# ============================================================================
# PHASE 0b: REDISTRIBUTION DIAGNOSTIC (go/no-go gate)
# ============================================================================

def phase_0b_redistribution(features_df):
    """Compare [1,85] Hz vs [1,45] Hz peak distributions."""
    log.info("=== Phase 0b: Redistribution Diagnostic ===")

    sids_85 = available_subjects_85hz()
    if not sids_85:
        log.error("No 85 Hz peak CSVs found. Run run_lemon_reextract_peaks.py first.")
        return None

    positions = natural_positions(PHI)
    rows = []

    for sid in sids_85:
        peaks_85 = load_peaks_85hz(sid)
        peaks_45 = load_peaks_45hz(sid)

        if peaks_85 is None:
            continue

        n85 = len(peaks_85)
        n85_below45 = (peaks_85['freq'] < 45).sum()

        # (a) Peak counts
        n45 = len(peaks_45) if peaks_45 is not None else 0

        # (c) Peak overlap
        overlap_count = 0
        if peaks_45 is not None and n45 > 0 and n85 > 0:
            for _, p45 in peaks_45.iterrows():
                match = peaks_85[
                    (peaks_85['channel'] == p45['channel']) &
                    (np.abs(peaks_85['freq'] - p45['freq']) < 0.25)
                ]
                if len(match) > 0:
                    overlap_count += 1

        # (d) Unweighted compliance at 85 Hz
        freqs_85 = peaks_85['freq'].values
        res_85 = continuous_compliance_score(freqs_85, f0=F0_PRIMARY,
                                              sigma=SIGMA, freq_ceil=85.0)

        # Original compliance at 45 Hz
        freqs_45 = peaks_45['freq'].values if peaks_45 is not None else np.array([])
        res_45 = continuous_compliance_score(freqs_45, f0=F0_PRIMARY,
                                              sigma=SIGMA, freq_ceil=45.0)

        # (e) Gamma peaks (>30 Hz)
        n_gamma_85 = (peaks_85['freq'] > 30).sum()

        rows.append({
            'subject_id': sid,
            'n_peaks_45hz': n45,
            'n_peaks_85hz': n85,
            'n_peaks_85hz_below45': n85_below45,
            'n_peaks_lost': n45 - overlap_count,
            'n_peaks_gained': n85 - overlap_count,
            'overlap_count': overlap_count,
            'overlap_frac_45': overlap_count / n45 if n45 > 0 else np.nan,
            'SS_45hz': res_45.get('compliance', np.nan),
            'SS_85hz': res_85.get('compliance', np.nan),
            'E_attractor_45hz': res_45.get('E_attractor', np.nan),
            'E_attractor_85hz': res_85.get('E_attractor', np.nan),
            'E_boundary_45hz': res_45.get('E_boundary', np.nan),
            'E_boundary_85hz': res_85.get('E_boundary', np.nan),
            'n_gamma_peaks': n_gamma_85,
        })

    diag_df = pd.DataFrame(rows)
    if diag_df.empty:
        log.error("No overlapping subjects between 85 Hz and 45 Hz extractions")
        return None

    # Merge with demographics
    diag_df = diag_df.merge(
        features_df[['subject_id', 'age_midpoint', 'age_group']],
        on='subject_id', how='left')

    # Save
    diag_path = os.path.join(PEAKS_85HZ, 'redistribution_diagnostic.csv')
    diag_df.to_csv(diag_path, index=False)

    # Report
    log.info(f"Subjects: {len(diag_df)}")
    log.info(f"Mean peaks [1,45]: {diag_df['n_peaks_45hz'].mean():.0f}, "
             f"[1,85]: {diag_df['n_peaks_85hz'].mean():.0f}")
    log.info(f"Mean overlap: {diag_df['overlap_frac_45'].mean():.1%}")
    log.info(f"Mean SS [1,45]: {diag_df['SS_45hz'].mean():.1f}, "
             f"[1,85]: {diag_df['SS_85hz'].mean():.1f}")

    # (d) Attractor age test at 85 Hz
    valid = diag_df.dropna(subset=['age_midpoint', 'E_attractor_85hz'])
    r_att_85, p_att_85 = stats.pearsonr(valid['age_midpoint'],
                                          valid['E_attractor_85hz'])
    r_att_45, p_att_45 = stats.pearsonr(valid['age_midpoint'],
                                          valid['E_attractor_45hz'])

    log.info(f"Attractor age trend [1,45]: r={r_att_45:.3f}, p={p_att_45:.4f}")
    log.info(f"Attractor age trend [1,85]: r={r_att_85:.3f}, p={p_att_85:.4f}")

    # GO/NO-GO GATE
    if r_att_85 > 0:
        log.error("!!! GO/NO-GO GATE FAILED: Attractor age correlation "
                   f"REVERSED at [1,85] (r={r_att_85:.3f}). STOP.")
        return diag_df
    elif r_att_85 > -0.05:
        log.error("!!! GO/NO-GO GATE FAILED: Attractor age correlation "
                   f"effectively vanished at [1,85] (r={r_att_85:.3f}). STOP.")
        return diag_df
    else:
        log.info(f"GO/NO-GO GATE PASSED: r={r_att_85:.3f} (attenuated but present)")

    # (b) Aperiodic exponent comparison — need extraction summaries
    summary_85_path = os.path.join(PEAKS_85HZ, 'extraction_summary_eo.csv')
    if os.path.exists(summary_85_path):
        summ = pd.read_csv(summary_85_path)
        summ = summ.merge(
            features_df[['subject_id', 'age_midpoint', 'age_group',
                          'mean_aperiodic_exponent']],
            on='subject_id', how='left',
            suffixes=('_85hz', '_45hz'))
        valid_ap = summ.dropna(subset=['mean_aperiodic_exponent_85hz',
                                        'mean_aperiodic_exponent_45hz',
                                        'age_midpoint'])
        if len(valid_ap) > 10:
            delta_ap = (valid_ap['mean_aperiodic_exponent_85hz'] -
                        valid_ap['mean_aperiodic_exponent_45hz'])
            r_delta_age, p_delta_age = stats.pearsonr(
                valid_ap['age_midpoint'], delta_ap)
            log.info(f"Aperiodic exponent shift ~ age: r={r_delta_age:.3f}, "
                     f"p={p_delta_age:.4f}")

    # (e) Gamma artifact check — EO peaks only for now
    mean_gamma = diag_df['n_gamma_peaks'].mean()
    log.info(f"Mean gamma peaks (>30 Hz) per subject: {mean_gamma:.1f}")

    return diag_df


# ============================================================================
# PHASE 0: MASTER PEAK TABLE + f₀ CORRECTION
# ============================================================================

def phase_0_f0_correction(features_df):
    """Build enriched peak tables and test f₀ correction."""
    log.info("=== Phase 0: f₀ Correction ===")

    os.makedirs(PEAKS_ENRICHED_DIR, exist_ok=True)
    positions = natural_positions(PHI)

    sids = available_subjects_85hz()
    log.info(f"Processing {len(sids)} subjects")

    f0_rows = []

    for sid in sids:
        peaks = load_peaks_85hz(sid)
        if peaks is None:
            continue

        freqs = peaks['freq'].values
        powers = peaks['power'].values
        n_peaks = len(freqs)

        # Get IAF from features
        feat = features_df[features_df['subject_id'] == sid]
        if feat.empty:
            continue
        iaf = feat['iaf'].values[0]
        f0_iaf = iaf / (PHI ** 0.618) if np.isfinite(iaf) else F0_PRIMARY

        # f₀ optimization — sweep
        f0_star = f0_iaf  # default
        ss_star = -np.inf
        optimization_used = False

        if n_peaks >= F0_OPT_MIN_PEAKS:
            for f0_cand in F0_SWEEP:
                u_cand = lattice_coordinate(freqs, f0_cand, PHI)
                u_valid = u_cand[np.isfinite(u_cand)]
                if len(u_valid) < 10:
                    continue
                ss_cand, _ = continuous_structural_score(
                    u_valid, positions, SIGMA)
                if ss_cand > ss_star:
                    ss_star = ss_cand
                    f0_star = f0_cand
            optimization_used = True

        # Compute SS at three anchors
        anchors = {
            'f0_85': F0_PRIMARY,
            'f0_iaf': f0_iaf,
            'f0_star': f0_star,
        }
        anchor_results = {}
        for aname, f0_val in anchors.items():
            u = lattice_coordinate(freqs, f0_val, PHI)
            u_valid = u[np.isfinite(u)]
            if len(u_valid) > 0:
                ss, enrich = continuous_structural_score(
                    u_valid, positions, SIGMA)
                anchor_results[f'SS_{aname}'] = ss
                anchor_results[f'E_attractor_{aname}'] = enrich.get('attractor', np.nan)
                anchor_results[f'E_boundary_{aname}'] = enrich.get('boundary', np.nan)
            else:
                anchor_results[f'SS_{aname}'] = np.nan
                anchor_results[f'E_attractor_{aname}'] = np.nan
                anchor_results[f'E_boundary_{aname}'] = np.nan

        # Build enriched peak table
        u_85 = lattice_coordinate(freqs, F0_PRIMARY, PHI)
        u_iaf = lattice_coordinate(freqs, f0_iaf, PHI)
        u_star = lattice_coordinate(freqs, f0_star, PHI)

        # Assign phi-octave bands (using f0=8.5 as reference)
        bands = make_phi_bands(F0_PRIMARY, 85.0)
        band_labels = []
        for f in freqs:
            assigned = 'other'
            for bname, (lo, hi) in bands.items():
                if lo <= f < hi:
                    assigned = bname
                    break
            band_labels.append(assigned)

        enriched = peaks.copy()
        enriched['u_f0_85'] = u_85
        enriched['u_f0_iaf'] = u_iaf
        enriched['u_f0_star'] = u_star
        enriched['phi_band'] = band_labels
        enriched.to_csv(os.path.join(PEAKS_ENRICHED_DIR,
                                      f'{sid}_peaks_enriched.csv'), index=False)

        f0_rows.append({
            'subject_id': sid,
            'iaf': iaf,
            'f0_iaf': f0_iaf,
            'f0_star': f0_star,
            'optimization_used': optimization_used,
            'n_peaks': n_peaks,
            **anchor_results,
        })

    f0_df = pd.DataFrame(f0_rows)
    f0_df = f0_df.merge(features_df[['subject_id', 'age_midpoint', 'age_group']],
                         on='subject_id', how='left')

    f0_path = os.path.join(OUT_DIR, 'f0_correction.csv')
    f0_df.to_csv(f0_path, index=False)

    # Report
    log.info(f"f₀* distribution: mean={f0_df['f0_star'].mean():.2f}, "
             f"SD={f0_df['f0_star'].std():.2f}")
    log.info(f"f₀_IAF distribution: mean={f0_df['f0_iaf'].mean():.2f}, "
             f"SD={f0_df['f0_iaf'].std():.2f}")

    # Circularity check: r(n_peaks, f₀*)
    valid = f0_df.dropna(subset=['n_peaks', 'f0_star'])
    if len(valid) > 10:
        r_np, p_np = stats.pearsonr(valid['n_peaks'], valid['f0_star'])
        log.info(f"r(n_peaks, f₀*) = {r_np:.3f}, p = {p_np:.4f} "
                 f"{'⚠ STRONG' if abs(r_np) > 0.3 else '✓ weak'}")

    # Attractor age at each anchor
    valid = f0_df.dropna(subset=['age_midpoint'])
    for aname in ['f0_85', 'f0_iaf', 'f0_star']:
        col = f'E_attractor_{aname}'
        v = valid.dropna(subset=[col])
        if len(v) > 10:
            r, p = stats.pearsonr(v['age_midpoint'], v[col])
            log.info(f"E_attractor ~ age at {aname}: r={r:.3f}, p={p:.4f}")

    # f₀* ~ age
    valid_opt = f0_df[f0_df['optimization_used']].dropna(subset=['age_midpoint'])
    if len(valid_opt) > 10:
        r_fa, p_fa = stats.pearsonr(valid_opt['age_midpoint'],
                                      valid_opt['f0_star'])
        log.info(f"f₀* ~ age: r={r_fa:.3f}, p={p_fa:.4f}")

        # Schumann window clustering
        in_window = ((valid_opt['f0_star'] >= 7.0) &
                     (valid_opt['f0_star'] <= 8.5)).mean()
        log.info(f"f₀* in Schumann window [7.0, 8.5]: {in_window:.1%}")

    return f0_df


# ============================================================================
# PHASE 1: PER-SUBJECT WEIGHTED COMPLIANCE
# ============================================================================

def phase_1_weighted_compliance(features_df, f0_df):
    """Compute weighted compliance for all subjects and transforms."""
    log.info("=== Phase 1: Weighted Compliance ===")

    sids = available_subjects_85hz()
    rows = []

    for sid in sids:
        peaks = load_peaks_85hz(sid)
        if peaks is None:
            continue

        freqs = peaks['freq'].values
        powers = peaks['power'].values

        # Get f₀ values for this subject
        f0_row = f0_df[f0_df['subject_id'] == sid]
        f0_iaf = f0_row['f0_iaf'].values[0] if len(f0_row) > 0 else F0_PRIMARY

        row = {'subject_id': sid}

        # Unweighted baseline at 85 Hz (new baseline)
        res_uw = continuous_compliance_score(freqs, f0=F0_PRIMARY,
                                              sigma=SIGMA, freq_ceil=85.0)
        row['SS_85hz_unweighted'] = res_uw['compliance']
        row['E_attractor_85hz_uw'] = res_uw['E_attractor']
        row['E_boundary_85hz_uw'] = res_uw['E_boundary']
        row['E_noble_1_85hz_uw'] = res_uw['E_noble_1']
        row['E_noble_2_85hz_uw'] = res_uw['E_noble_2']
        row['n_peaks_85hz'] = res_uw['n_peaks']

        # Weighted at each transform × each anchor
        for wt in WEIGHT_TRANSFORMS:
            for anchor_name, f0_val in [('f085', F0_PRIMARY),
                                         ('iaf', f0_iaf)]:
                res = weighted_compliance_score(
                    freqs, powers, f0=f0_val, sigma=SIGMA,
                    freq_ceil=85.0, weight_transform=wt)

                prefix = f'{wt}_{anchor_name}'
                row[f'SS_{prefix}'] = res['compliance_weighted']
                row[f'E_attractor_{prefix}'] = res['E_attractor_w']
                row[f'E_boundary_{prefix}'] = res['E_boundary_w']
                row[f'E_noble_1_{prefix}'] = res['E_noble_1_w']
                row[f'E_noble_2_{prefix}'] = res['E_noble_2_w']

        # Mean peak power for mediation
        row['mean_peak_power'] = powers.mean() if len(powers) > 0 else np.nan

        rows.append(row)

    wc_df = pd.DataFrame(rows)

    # Merge demographics
    wc_df = wc_df.merge(
        features_df[['subject_id', 'age_midpoint', 'age_group', 'iaf',
                      'compliance', 'E_attractor', 'E_boundary']],
        on='subject_id', how='left',
        suffixes=('', '_orig45'))
    # Also merge sex for mediation
    mb = pd.read_csv(os.path.join(EXPORTS, 'master_behavioral.csv'))
    wc_df = wc_df.merge(mb[['subject_id', 'sex']], on='subject_id', how='left')

    out_path = os.path.join(OUT_DIR, 'subject_features_weighted.csv')
    wc_df.to_csv(out_path, index=False)
    log.info(f"Saved {len(wc_df)} subjects to {out_path}")

    return wc_df


# ============================================================================
# PHASE 2: AGE TREND COMPARISON
# ============================================================================

def phase_2_age_trends(wc_df):
    """Compare attractor age trend across weighting methods."""
    log.info("=== Phase 2: Age Trend Comparison ===")

    valid = wc_df.dropna(subset=['age_midpoint']).copy()
    young = valid[valid['age_group'] == 'young']
    elderly = valid[valid['age_group'] == 'elderly']

    trend_rows = []

    # Positions to test
    position_names = ['attractor', 'boundary', 'noble_1', 'noble_2']

    # Test configurations
    configs = [
        ('unweighted_45hz', 'E_{pos}', 'compliance'),
        ('unweighted_85hz', 'E_{pos}_85hz_uw', 'SS_85hz_unweighted'),
    ]
    for wt in WEIGHT_TRANSFORMS:
        configs.append((f'{wt}_f085', f'E_{{pos}}_{wt}_f085', f'SS_{wt}_f085'))
        configs.append((f'{wt}_iaf', f'E_{{pos}}_{wt}_iaf', f'SS_{wt}_iaf'))

    for config_name, col_template, ss_col in configs:
        for pos in position_names:
            col = col_template.format(pos=pos)
            if col not in valid.columns:
                # Try original naming for 45 Hz
                if config_name == 'unweighted_45hz':
                    col = f'E_{pos}'
                    if pos == 'noble_1':
                        col = 'E_noble_1'
                    elif pos == 'noble_2':
                        col = 'E_noble_2'
                if col not in valid.columns:
                    continue

            v = valid.dropna(subset=[col])
            if len(v) < 20:
                continue

            r, p = stats.pearsonr(v['age_midpoint'], v[col])

            # Group comparison
            y_vals = young[col].dropna()
            e_vals = elderly[col].dropna()
            gc = run_group_comparison(y_vals.values, e_vals.values)

            trend_rows.append({
                'config': config_name,
                'position': pos,
                'column': col,
                'r_age': r,
                'p_age': p,
                'cohens_d': gc['cohens_d'],
                'p_ttest': gc['p_ttest'],
                'mean_young': gc['mean_young'],
                'mean_elderly': gc['mean_elderly'],
                'n': len(v),
            })

    trend_df = pd.DataFrame(trend_rows)
    out_path = os.path.join(OUT_DIR, 'age_trend_comparison.csv')
    trend_df.to_csv(out_path, index=False)

    # Report key results
    att_rows = trend_df[trend_df['position'] == 'attractor'].copy()
    att_rows = att_rows.sort_values('r_age')
    log.info("Attractor age trends:")
    for _, row in att_rows.iterrows():
        sig = '*' if row['p_age'] < 0.05 else ' '
        log.info(f"  {row['config']:25s} r={row['r_age']:+.3f} p={row['p_age']:.4f}{sig} "
                 f"d={row['cohens_d']:+.3f}")

    return trend_df


# ============================================================================
# PHASE 3: MAX_N_PEAKS=40 ROBUSTNESS
# ============================================================================

def phase_3_max40_robustness(features_df):
    """Critical test: does weighting rescue max_n_peaks=40?"""
    log.info("=== Phase 3: max_n_peaks=40 Robustness ===")

    sids = available_subjects_85hz(suffix='_peaks_max40')
    if not sids:
        log.warning("No max40 peak CSVs found. Skipping Phase 3.")
        return None

    log.info(f"Processing {len(sids)} subjects with max40 peaks")

    rows = []
    for sid in sids:
        peaks = load_peaks_85hz(sid, suffix='_peaks_max40')
        if peaks is None:
            continue

        freqs = peaks['freq'].values
        powers = peaks['power'].values

        row = {'subject_id': sid, 'n_peaks_max40': len(freqs)}

        # Unweighted
        res_uw = continuous_compliance_score(freqs, f0=F0_PRIMARY,
                                              sigma=SIGMA, freq_ceil=85.0)
        row['SS_max40_unweighted'] = res_uw['compliance']
        row['E_attractor_max40_uw'] = res_uw['E_attractor']

        # Weighted (rank only — primary)
        res_w = weighted_compliance_score(
            freqs, powers, f0=F0_PRIMARY, sigma=SIGMA,
            freq_ceil=85.0, weight_transform='rank')
        row['SS_max40_rank'] = res_w['compliance_weighted']
        row['E_attractor_max40_rank'] = res_w['E_attractor_w']

        rows.append(row)

    max40_df = pd.DataFrame(rows)
    max40_df = max40_df.merge(
        features_df[['subject_id', 'age_midpoint', 'age_group']],
        on='subject_id', how='left')

    out_path = os.path.join(OUT_DIR, 'max40_robustness.csv')
    max40_df.to_csv(out_path, index=False)

    # Test
    valid = max40_df.dropna(subset=['age_midpoint'])
    young = valid[valid['age_group'] == 'young']
    elderly = valid[valid['age_group'] == 'elderly']

    for col_name, label in [('E_attractor_max40_uw', 'max40 unweighted'),
                             ('E_attractor_max40_rank', 'max40 rank-weighted')]:
        v = valid.dropna(subset=[col_name])
        r, p = stats.pearsonr(v['age_midpoint'], v[col_name])
        gc = run_group_comparison(young[col_name].dropna().values,
                                   elderly[col_name].dropna().values)
        log.info(f"  {label}: r={r:.3f}, p={p:.4f}, d={gc['cohens_d']:.3f}")

    # Three-tier assessment
    r_rank, p_rank = stats.pearsonr(
        valid.dropna(subset=['E_attractor_max40_rank'])['age_midpoint'],
        valid.dropna(subset=['E_attractor_max40_rank'])['E_attractor_max40_rank'])

    if abs(r_rank) >= 0.15 and p_rank < 0.01:
        tier = 'Tier 1 (strong rescue)'
    elif p_rank < 0.05:
        tier = 'Tier 2 (partial rescue)'
    else:
        tier = 'Tier 3 (failure)'
    log.info(f"  Three-tier assessment: {tier}")

    return max40_df


# ============================================================================
# PHASE 4: PROMINENCE MEDIATION
# ============================================================================

def phase_4_mediation(wc_df):
    """Test prominence mediation: Age → Peak Power → Compliance."""
    log.info("=== Phase 4: Prominence Mediation ===")

    valid = wc_df.dropna(subset=['age_midpoint', 'mean_peak_power',
                                   'SS_85hz_unweighted', 'sex']).copy()
    log.info(f"Mediation N={len(valid)}")

    results = {}

    # Primary: Y = unweighted compliance
    log.info("Model 1: Age → Mean Peak Power → Unweighted Compliance")
    sex_numeric = (valid['sex'] == 'male').astype(float).values
    med1 = bootstrap_mediation(
        X=valid['age_midpoint'].values,
        M=valid['mean_peak_power'].values,
        Y=valid['SS_85hz_unweighted'].values,
        covariates=sex_numeric,
        n_boot=5000, seed=42)

    ci_excludes_zero = (med1['indirect_ci_lo'] > 0 or med1['indirect_ci_hi'] < 0)
    log.info(f"  a (age→power): {med1['a_coef']:.4f}")
    log.info(f"  b (power→compliance|age): {med1['b_coef']:.4f}")
    log.info(f"  indirect: {med1['indirect_effect']:.4f} "
             f"CI [{med1['indirect_ci_lo']:.4f}, {med1['indirect_ci_hi']:.4f}] "
             f"{'*' if ci_excludes_zero else 'ns'}")
    log.info(f"  direct: {med1['direct_effect']:.4f}")
    log.info(f"  total: {med1['total_effect']:.4f}")
    log.info(f"  proportion mediated: {med1['proportion_mediated']:.1%}")
    results['unweighted'] = med1

    # Complementary: Y = rank-weighted compliance
    rank_col = 'SS_rank_f085'
    if rank_col in valid.columns:
        valid_w = valid.dropna(subset=[rank_col])
        log.info("Model 2: Age → Mean Peak Power → Rank-Weighted Compliance")
        sex_w = (valid_w['sex'] == 'male').astype(float).values
        med2 = bootstrap_mediation(
            X=valid_w['age_midpoint'].values,
            M=valid_w['mean_peak_power'].values,
            Y=valid_w[rank_col].values,
            covariates=sex_w,
            n_boot=5000, seed=42)

        ci2_excludes = (med2['indirect_ci_lo'] > 0 or med2['indirect_ci_hi'] < 0)
        log.info(f"  indirect: {med2['indirect_effect']:.4f} "
                 f"CI [{med2['indirect_ci_lo']:.4f}, {med2['indirect_ci_hi']:.4f}] "
                 f"{'*' if ci2_excludes else 'ns'}")
        log.info(f"  proportion mediated: {med2['proportion_mediated']:.1%}")
        results['rank_weighted'] = med2

    # Save
    med_rows = []
    for model_name, med in results.items():
        med_rows.append({'model': model_name, **med})
    med_df = pd.DataFrame(med_rows)
    out_path = os.path.join(OUT_DIR, 'mediation_results.csv')
    med_df.to_csv(out_path, index=False)

    return results


# ============================================================================
# PHASE 5: PER-SUBJECT WITHIN-BAND SHUFFLE
# ============================================================================

def phase_5_shuffle(features_df, n_perm=1000):
    """Per-subject within-band shuffle z-scores."""
    log.info(f"=== Phase 5: Within-Band Shuffle (n_perm={n_perm}) ===")

    sids = available_subjects_85hz()

    # Benchmark on first subject
    peaks = load_peaks_85hz(sids[0]) if sids else None
    if peaks is not None:
        t0 = time.time()
        _ = weighted_within_band_shuffle(
            peaks['freq'].values, peaks['power'].values,
            f0=F0_PRIMARY, sigma=SIGMA, n_perm=100,
            freq_ceil=85.0, weight_transform='rank')
        bench_time = time.time() - t0
        est_total = bench_time * (n_perm / 100) * len(sids)
        log.info(f"Benchmark: {bench_time:.1f}s for 100 perms on 1 subject")
        log.info(f"Estimated total: {est_total/60:.0f} min for {len(sids)} subjects × {n_perm} perms")

        # Auto-reduce if too slow
        if est_total > 7200:  # > 2 hours
            n_perm = 500
            log.info(f"Reducing to n_perm={n_perm} to keep runtime reasonable")

    rows = []
    t_start = time.time()

    for i, sid in enumerate(sids):
        peaks = load_peaks_85hz(sid)
        if peaks is None:
            continue

        res = weighted_within_band_shuffle(
            peaks['freq'].values, peaks['power'].values,
            f0=F0_PRIMARY, sigma=SIGMA, n_perm=n_perm,
            freq_ceil=85.0, weight_transform='rank')

        rows.append({
            'subject_id': sid,
            'SS_obs': res['SS_obs'],
            'null_mean': res['null_mean'],
            'null_std': res['null_std'],
            'z_score': res['z_score'],
            'p_value': res['p_value'],
        })

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t_start
            log.info(f"  [{i+1}/{len(sids)}] {elapsed/60:.1f} min elapsed")

    shuf_df = pd.DataFrame(rows)
    shuf_df = shuf_df.merge(
        features_df[['subject_id', 'age_midpoint', 'age_group']],
        on='subject_id', how='left')

    out_path = os.path.join(OUT_DIR, 'subject_shuffle_zscores.csv')
    shuf_df.to_csv(out_path, index=False)

    # Report
    n_sig = (shuf_df['p_value'] < 0.05).sum()
    log.info(f"Individually significant (p<0.05): {n_sig}/{len(shuf_df)} "
             f"({n_sig/len(shuf_df):.1%})")

    valid = shuf_df.dropna(subset=['age_midpoint', 'z_score'])
    if len(valid) > 10:
        r, p = stats.pearsonr(valid['age_midpoint'], valid['z_score'])
        log.info(f"z_score ~ age: r={r:.3f}, p={p:.4f}")

    return shuf_df


# ============================================================================
# PHASE 0b(f,g): RAW vs PREPROCESSED COMPARISON
# ============================================================================

def phase_0bf_raw_comparison(features_df):
    """Compare compliance between raw-minimal and preprocessed extractions."""
    log.info("=== Phase 0b(f,g): Raw vs Preprocessed Comparison ===")

    raw_summary_path = os.path.join(PEAKS_RAW, 'extraction_summary_eo.csv')
    if not os.path.exists(raw_summary_path):
        log.warning("Raw extraction summary not found. Skipping.")
        return None

    raw_summ = pd.read_csv(raw_summary_path)
    raw_ok = raw_summ[raw_summ['status'] == 'ok'].copy()

    positions = natural_positions(PHI)
    rows = []

    for _, rrow in raw_ok.iterrows():
        sid = rrow['subject_id']

        # Raw peaks
        raw_path = os.path.join(PEAKS_RAW, f'{sid}_peaks.csv')
        if not os.path.exists(raw_path):
            continue
        raw_peaks = pd.read_csv(raw_path)

        # Preprocessed 85 Hz peaks
        pre_peaks = load_peaks_85hz(sid)
        if pre_peaks is None:
            continue

        # Compliance: raw vs preprocessed
        res_raw = continuous_compliance_score(
            raw_peaks['freq'].values, f0=F0_PRIMARY, sigma=SIGMA, freq_ceil=85.0)
        res_pre = continuous_compliance_score(
            pre_peaks['freq'].values, f0=F0_PRIMARY, sigma=SIGMA, freq_ceil=85.0)

        rows.append({
            'subject_id': sid,
            'SS_raw': res_raw['compliance'],
            'SS_preprocessed': res_pre['compliance'],
            'E_attractor_raw': res_raw['E_attractor'],
            'E_attractor_preprocessed': res_pre['E_attractor'],
            'n_peaks_raw': len(raw_peaks),
            'n_peaks_preprocessed': len(pre_peaks),
            'aperiodic_raw': rrow.get('mean_aperiodic_exponent', np.nan),
        })

    if not rows:
        log.warning("No overlapping subjects between raw and preprocessed")
        return None

    comp_df = pd.DataFrame(rows)

    # Merge preprocessed aperiodic from extraction summary
    pre_summ_path = os.path.join(PEAKS_85HZ, 'extraction_summary_eo.csv')
    if os.path.exists(pre_summ_path):
        pre_summ = pd.read_csv(pre_summ_path)
        comp_df = comp_df.merge(
            pre_summ[['subject_id', 'mean_aperiodic_exponent']].rename(
                columns={'mean_aperiodic_exponent': 'aperiodic_preprocessed'}),
            on='subject_id', how='left')

    comp_df = comp_df.merge(
        features_df[['subject_id', 'age_midpoint', 'age_group']],
        on='subject_id', how='left')

    out_path = os.path.join(OUT_DIR, 'raw_vs_preprocessed.csv')
    comp_df.to_csv(out_path, index=False)

    # (f) Compliance comparison
    log.info(f"N overlapping subjects: {len(comp_df)}")
    log.info(f"Mean SS raw: {comp_df['SS_raw'].mean():.1f}, "
             f"preprocessed: {comp_df['SS_preprocessed'].mean():.1f}")

    delta_ss = comp_df['SS_raw'] - comp_df['SS_preprocessed']
    log.info(f"Mean ΔSS (raw - preproc): {delta_ss.mean():.1f}")

    valid = comp_df.dropna(subset=['age_midpoint'])
    if len(valid) > 10:
        r_gap, p_gap = stats.pearsonr(valid['age_midpoint'],
                                        delta_ss[valid.index])
        log.info(f"ΔSS ~ age: r={r_gap:.3f}, p={p_gap:.4f}")

    # (g) Aperiodic comparison
    if 'aperiodic_preprocessed' in comp_df.columns:
        delta_ap = (comp_df['aperiodic_raw'] - comp_df['aperiodic_preprocessed'])
        log.info(f"Mean Δaperiodic (raw - preproc): {delta_ap.mean():.3f}")

        valid_ap = comp_df.dropna(subset=['age_midpoint', 'aperiodic_raw',
                                           'aperiodic_preprocessed'])
        if len(valid_ap) > 10:
            r_ap, p_ap = stats.pearsonr(
                valid_ap['age_midpoint'],
                valid_ap['aperiodic_raw'] - valid_ap['aperiodic_preprocessed'])
            log.info(f"Δaperiodic ~ age: r={r_ap:.3f}, p={p_ap:.4f}")

    return comp_df


# ============================================================================
# PHASE 6: SUMMARY FIGURES
# ============================================================================

def phase_6_figures(wc_df, trend_df, diag_df=None, shuf_df=None,
                     med_results=None, max40_df=None, f0_df=None):
    """Generate summary figures."""
    log.info("=== Phase 6: Summary Figures ===")
    os.makedirs(FIG_DIR, exist_ok=True)

    # --- Fig 1: Redistribution diagnostic ---
    if diag_df is not None and len(diag_df) > 10:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        ax = axes[0, 0]
        ax.scatter(diag_df['n_peaks_45hz'], diag_df['n_peaks_85hz'],
                   alpha=0.5, s=15)
        ax.plot([0, diag_df['n_peaks_45hz'].max()],
                [0, diag_df['n_peaks_45hz'].max()], 'k--', alpha=0.3)
        ax.set_xlabel('Peaks [1, 45 Hz]')
        ax.set_ylabel('Peaks [1, 85 Hz]')
        ax.set_title('Peak count comparison')

        ax = axes[0, 1]
        ax.scatter(diag_df['E_attractor_45hz'], diag_df['E_attractor_85hz'],
                   alpha=0.5, s=15)
        ax.axhline(0, color='gray', ls='--', alpha=0.3)
        ax.axvline(0, color='gray', ls='--', alpha=0.3)
        ax.set_xlabel('E_attractor [1, 45 Hz]')
        ax.set_ylabel('E_attractor [1, 85 Hz]')
        ax.set_title('Attractor enrichment comparison')

        ax = axes[1, 0]
        valid = diag_df.dropna(subset=['age_midpoint'])
        colors = ['tab:blue' if g == 'young' else 'tab:red'
                  for g in valid['age_group']]
        ax.scatter(valid['age_midpoint'], valid['E_attractor_85hz'],
                   c=colors, alpha=0.5, s=15)
        r, p = stats.pearsonr(valid['age_midpoint'], valid['E_attractor_85hz'])
        ax.set_xlabel('Age')
        ax.set_ylabel('E_attractor [1, 85 Hz]')
        ax.set_title(f'Attractor vs age (85 Hz) r={r:.3f}, p={p:.4f}')

        ax = axes[1, 1]
        ax.hist(diag_df['overlap_frac_45'].dropna(), bins=30, edgecolor='k',
                alpha=0.7)
        ax.set_xlabel('Overlap fraction')
        ax.set_ylabel('Count')
        ax.set_title(f'Peak overlap [45→85 Hz], mean={diag_df["overlap_frac_45"].mean():.2f}')

        plt.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, 'fig_redistribution.png'), dpi=150)
        plt.close(fig)
        log.info("  Saved fig_redistribution.png")

    # --- Fig 2: f₀ correction ---
    if f0_df is not None and len(f0_df) > 10:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        ax = axes[0]
        ax.hist(f0_df['f0_star'].dropna(), bins=30, edgecolor='k', alpha=0.7)
        ax.axvline(8.5, color='red', ls='--', label='f₀=8.5')
        ax.axvline(f0_df['f0_iaf'].mean(), color='blue', ls='--', label='mean f₀_IAF')
        ax.set_xlabel('f₀* (Hz)')
        ax.set_title('Optimized f₀* distribution')
        ax.legend()

        ax = axes[1]
        valid = f0_df.dropna(subset=['age_midpoint', 'f0_star'])
        ax.scatter(valid['age_midpoint'], valid['f0_star'], alpha=0.5, s=15)
        r, p = stats.pearsonr(valid['age_midpoint'], valid['f0_star'])
        ax.set_xlabel('Age')
        ax.set_ylabel('f₀* (Hz)')
        ax.set_title(f'f₀* vs age r={r:.3f}, p={p:.4f}')

        ax = axes[2]
        att_trend = trend_df[trend_df['position'] == 'attractor'].copy()
        configs = ['unweighted_85hz', 'rank_f085', 'rank_iaf']
        d_vals = []
        labels = []
        for c in configs:
            row = att_trend[att_trend['config'] == c]
            if len(row) > 0:
                d_vals.append(row['cohens_d'].values[0])
                labels.append(c.replace('_', '\n'))
        if d_vals:
            bars = ax.bar(labels, d_vals, color=['gray', 'steelblue', 'darkorange'])
            ax.set_ylabel("Cohen's d")
            ax.set_title('Attractor age effect by f₀')
            ax.axhline(0, color='k', lw=0.5)

        plt.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, 'fig_f0_correction.png'), dpi=150)
        plt.close(fig)
        log.info("  Saved fig_f0_correction.png")

    # --- Fig 3: Weight transform comparison ---
    if wc_df is not None and 'age_midpoint' in wc_df.columns:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        valid = wc_df.dropna(subset=['age_midpoint']).copy()

        for idx, wt in enumerate(WEIGHT_TRANSFORMS):
            ax = axes[idx // 2, idx % 2]
            col = f'E_attractor_{wt}_f085'
            if col not in valid.columns:
                ax.set_title(f'{wt}: no data')
                continue

            v = valid.dropna(subset=[col])
            colors = ['tab:blue' if g == 'young' else 'tab:red'
                      for g in v['age_group']]
            ax.scatter(v['age_midpoint'], v[col], c=colors, alpha=0.5, s=15)
            r, p = stats.pearsonr(v['age_midpoint'], v[col])
            ax.set_xlabel('Age')
            ax.set_ylabel('E_attractor')
            ax.set_title(f'{wt}: r={r:.3f}, p={p:.4f}')
            # Trend line
            z = np.polyfit(v['age_midpoint'], v[col], 1)
            x_line = np.linspace(v['age_midpoint'].min(), v['age_midpoint'].max(), 100)
            ax.plot(x_line, np.polyval(z, x_line), 'k--', alpha=0.5)

        plt.suptitle('Attractor Enrichment vs Age by Weight Transform', y=1.01)
        plt.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, 'fig_weight_transform_comparison.png'),
                    dpi=150)
        plt.close(fig)
        log.info("  Saved fig_weight_transform_comparison.png")

    # --- Fig 4: max40 rescue ---
    if max40_df is not None and trend_df is not None:
        fig, ax = plt.subplots(figsize=(8, 5))

        # Collect d values
        labels = []
        d_vals = []
        colors_bar = []

        # max20 unweighted (original baseline from features)
        att_45 = trend_df[(trend_df['config'] == 'unweighted_45hz') &
                          (trend_df['position'] == 'attractor')]
        if len(att_45) > 0:
            labels.append('max20\n[1,45]\nunweighted')
            d_vals.append(att_45['cohens_d'].values[0])
            colors_bar.append('lightgray')

        att_85 = trend_df[(trend_df['config'] == 'unweighted_85hz') &
                          (trend_df['position'] == 'attractor')]
        if len(att_85) > 0:
            labels.append('max20\n[1,85]\nunweighted')
            d_vals.append(att_85['cohens_d'].values[0])
            colors_bar.append('gray')

        att_rank = trend_df[(trend_df['config'] == 'rank_f085') &
                            (trend_df['position'] == 'attractor')]
        if len(att_rank) > 0:
            labels.append('max20\n[1,85]\nrank')
            d_vals.append(att_rank['cohens_d'].values[0])
            colors_bar.append('steelblue')

        # max40 values
        valid40 = max40_df.dropna(subset=['age_midpoint'])
        young40 = valid40[valid40['age_group'] == 'young']
        elderly40 = valid40[valid40['age_group'] == 'elderly']

        for col, label, clr in [
            ('E_attractor_max40_uw', 'max40\n[1,85]\nunweighted', 'lightsalmon'),
            ('E_attractor_max40_rank', 'max40\n[1,85]\nrank', 'indianred'),
        ]:
            gc = run_group_comparison(young40[col].dropna().values,
                                       elderly40[col].dropna().values)
            labels.append(label)
            d_vals.append(gc['cohens_d'])
            colors_bar.append(clr)

        bars = ax.bar(labels, d_vals, color=colors_bar, edgecolor='k', linewidth=0.5)
        ax.set_ylabel("Cohen's d (attractor age effect)")
        ax.set_title('max_n_peaks Robustness: Weighted vs Unweighted')
        ax.axhline(0, color='k', lw=0.5)

        plt.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, 'fig_max40_rescue.png'), dpi=150)
        plt.close(fig)
        log.info("  Saved fig_max40_rescue.png")

    # --- Fig 5: Mediation path ---
    if med_results is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-0.5, 2.5)
        ax.axis('off')

        # Boxes
        for x, y, label in [(0, 1, 'Age'), (1.5, 2, 'Peak\nProminence'),
                             (3, 1, 'Compliance')]:
            ax.add_patch(plt.Rectangle((x - 0.4, y - 0.3), 0.8, 0.6,
                                        fill=False, edgecolor='k', lw=2))
            ax.text(x, y, label, ha='center', va='center', fontsize=11)

        med_uw = med_results.get('unweighted', {})
        med_w = med_results.get('rank_weighted', {})

        # Path a
        a = med_uw.get('a_coef', np.nan)
        ax.annotate('', xy=(1.1, 1.85), xytext=(0.4, 1.25),
                    arrowprops=dict(arrowstyle='->', lw=2))
        ax.text(0.6, 1.7, f'a={a:.4f}', fontsize=9)

        # Path b (unweighted)
        b_uw = med_uw.get('b_coef', np.nan)
        ax.annotate('', xy=(2.6, 1.25), xytext=(1.9, 1.85),
                    arrowprops=dict(arrowstyle='->', lw=2, color='steelblue'))
        ax.text(2.3, 1.7, f'b={b_uw:.4f}', fontsize=9, color='steelblue')

        # Direct
        direct = med_uw.get('direct_effect', np.nan)
        ax.annotate('', xy=(2.6, 1.0), xytext=(0.4, 1.0),
                    arrowprops=dict(arrowstyle='->', lw=2, ls='--', color='gray'))
        ax.text(1.5, 0.75, f"c'={direct:.4f}", fontsize=9, color='gray')

        # Indirect
        ind_uw = med_uw.get('indirect_effect', np.nan)
        ci_lo = med_uw.get('indirect_ci_lo', np.nan)
        ci_hi = med_uw.get('indirect_ci_hi', np.nan)
        ax.text(1.5, 0.3, f'Unweighted Y: indirect={ind_uw:.4f}\n'
                f'CI [{ci_lo:.4f}, {ci_hi:.4f}]',
                ha='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow'))

        if med_w:
            ind_w = med_w.get('indirect_effect', np.nan)
            ci_lo_w = med_w.get('indirect_ci_lo', np.nan)
            ci_hi_w = med_w.get('indirect_ci_hi', np.nan)
            ax.text(1.5, -0.1, f'Weighted Y: indirect={ind_w:.4f}\n'
                    f'CI [{ci_lo_w:.4f}, {ci_hi_w:.4f}]',
                    ha='center', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightblue'))

        ax.set_title('Prominence Mediation: Age → Peak Power → Compliance')
        fig.savefig(os.path.join(FIG_DIR, 'fig_mediation_path.png'), dpi=150)
        plt.close(fig)
        log.info("  Saved fig_mediation_path.png")

    # --- Fig 6: Shuffle z-scores vs age ---
    if shuf_df is not None and len(shuf_df) > 10:
        fig, ax = plt.subplots(figsize=(8, 5))
        valid = shuf_df.dropna(subset=['age_midpoint', 'z_score'])
        colors = ['tab:blue' if g == 'young' else 'tab:red'
                  for g in valid['age_group']]
        ax.scatter(valid['age_midpoint'], valid['z_score'],
                   c=colors, alpha=0.5, s=15)
        ax.axhline(1.96, color='green', ls='--', alpha=0.5, label='z=1.96')
        ax.axhline(0, color='gray', ls='--', alpha=0.3)
        r, p = stats.pearsonr(valid['age_midpoint'], valid['z_score'])
        ax.set_xlabel('Age')
        ax.set_ylabel('Within-band shuffle z-score')
        ax.set_title(f'Individual lattice significance vs age (r={r:.3f}, p={p:.4f})')
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, 'fig_shuffle_zscores_vs_age.png'), dpi=150)
        plt.close(fig)
        log.info("  Saved fig_shuffle_zscores_vs_age.png")


# ============================================================================
# MAIN
# ============================================================================

def main(phases=None):
    """Run all phases (or selected ones)."""
    t_start = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load base data
    features_df = load_subject_features()
    log.info(f"Loaded {len(features_df)} subjects from features + behavioral")

    all_phases = phases or ['0b', '0', '1', '2', '4', '5', '3', '6']

    diag_df = None
    f0_df = None
    wc_df = None
    trend_df = None
    max40_df = None
    shuf_df = None
    med_results = None

    for phase in all_phases:
        if phase == '0b':
            diag_df = phase_0b_redistribution(features_df)
            # Check if gate failed
            if diag_df is not None:
                valid = diag_df.dropna(subset=['age_midpoint', 'E_attractor_85hz'])
                if len(valid) > 10:
                    r, _ = stats.pearsonr(valid['age_midpoint'],
                                           valid['E_attractor_85hz'])
                    if r > 0 or r > -0.05:
                        log.error("GO/NO-GO GATE FAILED. Stopping.")
                        break

        elif phase == '0':
            f0_df = phase_0_f0_correction(features_df)

        elif phase == '1':
            if f0_df is None:
                f0_path = os.path.join(OUT_DIR, 'f0_correction.csv')
                if os.path.exists(f0_path):
                    f0_df = pd.read_csv(f0_path)
                else:
                    log.error("Phase 0 must run first")
                    continue
            wc_df = phase_1_weighted_compliance(features_df, f0_df)

        elif phase == '2':
            if wc_df is None:
                wc_path = os.path.join(OUT_DIR, 'subject_features_weighted.csv')
                if os.path.exists(wc_path):
                    wc_df = pd.read_csv(wc_path)
                else:
                    log.error("Phase 1 must run first")
                    continue
            trend_df = phase_2_age_trends(wc_df)

        elif phase == '3':
            max40_df = phase_3_max40_robustness(features_df)

        elif phase == '4':
            if wc_df is None:
                wc_path = os.path.join(OUT_DIR, 'subject_features_weighted.csv')
                if os.path.exists(wc_path):
                    wc_df = pd.read_csv(wc_path)
                else:
                    log.error("Phase 1 must run first")
                    continue
            med_results = phase_4_mediation(wc_df)

        elif phase == '5':
            shuf_df = phase_5_shuffle(features_df)

        elif phase == '6':
            # Load any saved results we don't have yet
            if wc_df is None:
                wc_path = os.path.join(OUT_DIR, 'subject_features_weighted.csv')
                if os.path.exists(wc_path):
                    wc_df = pd.read_csv(wc_path)
            if trend_df is None:
                trend_path = os.path.join(OUT_DIR, 'age_trend_comparison.csv')
                if os.path.exists(trend_path):
                    trend_df = pd.read_csv(trend_path)
            if diag_df is None:
                diag_path = os.path.join(PEAKS_85HZ, 'redistribution_diagnostic.csv')
                if os.path.exists(diag_path):
                    diag_df = pd.read_csv(diag_path)
            if shuf_df is None:
                shuf_path = os.path.join(OUT_DIR, 'subject_shuffle_zscores.csv')
                if os.path.exists(shuf_path):
                    shuf_df = pd.read_csv(shuf_path)
            if max40_df is None:
                max40_path = os.path.join(OUT_DIR, 'max40_robustness.csv')
                if os.path.exists(max40_path):
                    max40_df = pd.read_csv(max40_path)
            if f0_df is None:
                f0_path = os.path.join(OUT_DIR, 'f0_correction.csv')
                if os.path.exists(f0_path):
                    f0_df = pd.read_csv(f0_path)

            phase_6_figures(wc_df, trend_df, diag_df, shuf_df,
                            med_results, max40_df, f0_df)

        elif phase == 'raw':
            phase_0bf_raw_comparison(features_df)

    total = time.time() - t_start
    log.info(f"Total runtime: {total/60:.1f} min")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Amplitude-weighted compliance analysis for LEMON')
    parser.add_argument('--phases', nargs='+', default=None,
                        help='Phases to run (0b, 0, 1, 2, 3, 4, 5, 6, raw)')
    args = parser.parse_args()
    main(phases=args.phases)

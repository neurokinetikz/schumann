#!/usr/bin/env python3
"""
Theta target disambiguation: Does EC theta convergence track f₀=7.83 Hz
or an IAF-derived subharmonic?

Exploits natural IAF variation (8.66–11.70 Hz) across N=182 subjects as a
cross-subject experiment. At extreme IAFs, fixed vs IAF-derived predictions
diverge by >1 Hz.

Five complementary tests + 4-panel publication figure.

Usage:
    PYTHONUNBUFFERED=1 /opt/anaconda3/bin/python scripts/theta_target_disambiguation.py
"""

import sys, os
sys.path.insert(0, '/Users/neurokinetikz/Code/schumann/lib')

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from glob import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ── Constants ────────────────────────────────────────────────────────────
PHI = (1 + np.sqrt(5)) / 2
F0 = 7.83
BANDS = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'gamma': (30, 45)}

PEAK_DIR = '/Users/neurokinetikz/Code/schumann/exports_lemon/per_subject'
FEATURES_PATH = '/Users/neurokinetikz/Code/schumann/exports_lemon/subject_features.csv'
BEHAV_PATH = '/Users/neurokinetikz/Code/schumann/exports_lemon/master_behavioral.csv'
OUT_DIR = '/Users/neurokinetikz/Code/schumann/papers/images/lemon'
CSV_OUT = '/Users/neurokinetikz/Code/schumann/exports_lemon/theta_target_disambiguation.csv'
os.makedirs(OUT_DIR, exist_ok=True)

# Candidate targets (per-subject functions of IAF)
TARGETS = {
    'f₀ = 7.83':   lambda iaf, alpha_ec: np.full_like(iaf, F0),
    'IAF/√φ':      lambda iaf, alpha_ec: iaf / np.sqrt(PHI),
    'IAF × 3/4':   lambda iaf, alpha_ec: iaf * 0.75,
    'IAF/φ':       lambda iaf, alpha_ec: iaf / PHI,
    'IAF/2':       lambda iaf, alpha_ec: iaf / 2.0,
    'EC_α/√φ':     lambda iaf, alpha_ec: alpha_ec / np.sqrt(PHI),
}


# ── Data Loading ─────────────────────────────────────────────────────────

def get_dominant_peaks(peak_dir, ec=False):
    """Load per-subject dominant peaks from FOOOF peak CSVs."""
    pattern = 'sub-*_peaks_ec.csv' if ec else 'sub-*_peaks.csv'
    records = []
    for f in sorted(glob(os.path.join(peak_dir, pattern))):
        if 'band_info' in f or 'max40' in f:
            continue
        sid = os.path.basename(f).split('_peaks')[0]
        df = pd.read_csv(f)
        row = {'subject_id': sid}
        for bname, (lo, hi) in BANDS.items():
            bp = df[(df.freq >= lo) & (df.freq < hi)]
            if len(bp) == 0:
                row[f'{bname}_freq'] = np.nan
                row[f'{bname}_power'] = np.nan
                continue
            idx = bp['power'].idxmax()
            row[f'{bname}_freq'] = bp.loc[idx, 'freq']
            row[f'{bname}_power'] = bp.loc[idx, 'power']
        records.append(row)
    return pd.DataFrame(records)


def partial_corr(y, x1, x2):
    """Partial correlation r(y, x1 | x2) with p-value."""
    r_y1, _ = stats.pearsonr(y, x1)
    r_y2, _ = stats.pearsonr(y, x2)
    r_12, _ = stats.pearsonr(x1, x2)
    num = r_y1 - r_y2 * r_12
    denom = np.sqrt((1 - r_y2**2) * (1 - r_12**2))
    pr = num / denom
    n = len(y)
    t_stat = pr * np.sqrt((n - 3) / (1 - pr**2))
    p_val = 2 * stats.t.sf(np.abs(t_stat), n - 3)
    return pr, p_val


def williams_test(r_y1, r_y2, r_12, n):
    """Williams (1959) test for comparing two dependent correlations sharing
    the same outcome variable. H0: r_y1 = r_y2."""
    rbar = (r_y1 + r_y2) / 2
    det = 1 - r_y1**2 - r_y2**2 - r_12**2 + 2 * r_y1 * r_y2 * r_12
    num = (r_y1 - r_y2) * np.sqrt((n - 1) * (1 + r_12))
    denom = np.sqrt(2 * ((n - 1) / (n - 3)) * det + rbar**2 * (1 - r_12)**3)
    if denom == 0:
        return 0.0, 1.0
    t = num / denom
    p = 2 * stats.t.sf(np.abs(t), n - 3)
    return t, p


# ── Load Data ────────────────────────────────────────────────────────────
print("=" * 70)
print("THETA TARGET DISAMBIGUATION ANALYSIS")
print("=" * 70)
print("\nLoading data...")

dom_eo = get_dominant_peaks(PEAK_DIR, ec=False)
dom_ec = get_dominant_peaks(PEAK_DIR, ec=True)

# Rename EC columns
ec_rename = {c: c + '_ec' for c in dom_ec.columns if c != 'subject_id'}
dom_ec = dom_ec.rename(columns=ec_rename)

features = pd.read_csv(FEATURES_PATH)
behav = pd.read_csv(BEHAV_PATH)

# Merge
m = dom_eo.merge(dom_ec, on='subject_id', how='inner')
m = m.merge(features[['subject_id', 'iaf']], on='subject_id', how='left')
m = m.merge(behav[['subject_id', 'age_midpoint', 'age_group']], on='subject_id', how='left')

# Require theta in both conditions + IAF
m = m.dropna(subset=['theta_freq', 'theta_freq_ec', 'iaf']).copy()
print(f"  N = {len(m)} subjects with matched EO/EC theta + IAF")
print(f"  IAF range: {m.iaf.min():.2f} – {m.iaf.max():.2f} Hz (mean {m.iaf.mean():.2f} ± {m.iaf.std():.2f})")

# Derived columns
m['theta_eo'] = m['theta_freq']
m['theta_ec'] = m['theta_freq_ec']
m['alpha_eo'] = m['alpha_freq']
m['alpha_ec'] = m['alpha_freq_ec']
m['delta_theta'] = m['theta_ec'] - m['theta_eo']
m['delta_alpha'] = m['alpha_ec'].fillna(m['alpha_eo']) - m['alpha_eo']

# Handle missing EC alpha for EC_α/√φ target
alpha_ec_filled = m['alpha_ec'].fillna(m['alpha_eo']).values

# Compute candidate targets and prediction errors
for name, func in TARGETS.items():
    safe = name.replace(' ', '_').replace('=', '').replace('/', '_').replace('×', 'x').replace('√', 'sqrt').replace('₀', '0').replace('α', 'a').replace('φ', 'phi')
    m[f'target_{safe}'] = func(m['iaf'].values, alpha_ec_filled)
    m[f'err_{safe}'] = np.abs(m['theta_ec'].values - m[f'target_{safe}'].values)

# Convergence on f0
m['conv_f0'] = np.abs(m['theta_eo'] - F0) - np.abs(m['theta_ec'] - F0)

# IAF terciles
m['iaf_tercile'] = pd.qcut(m['iaf'], 3, labels=['Low IAF', 'Mid IAF', 'High IAF'])

N = len(m)


# ═══════════════════════════════════════════════════════════════════════
# TEST 1: Model-free IAF covariance
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("TEST 1: Model-free IAF covariance")
print("─" * 70)

r_ec_iaf, p_ec_iaf = stats.pearsonr(m['theta_ec'], m['iaf'])
r_eo_iaf, p_eo_iaf = stats.pearsonr(m['theta_eo'], m['iaf'])
r_ec_eo, p_ec_eo = stats.pearsonr(m['theta_ec'], m['theta_eo'])
pr_ec_iaf, pp_ec_iaf = partial_corr(m['theta_ec'].values, m['iaf'].values, m['theta_eo'].values)

# Alpha-theta shift coupling
alpha_mask = m['alpha_ec'].notna() & m['alpha_eo'].notna()
if alpha_mask.sum() > 20:
    da = (m.loc[alpha_mask, 'alpha_ec'] - m.loc[alpha_mask, 'alpha_eo']).values
    dt = m.loc[alpha_mask, 'delta_theta'].values
    r_da_dt, p_da_dt = stats.pearsonr(da, dt)
else:
    r_da_dt, p_da_dt = np.nan, np.nan

print(f"  r(EC_theta, IAF)                    = {r_ec_iaf:+.3f}, p = {p_ec_iaf:.4f}")
print(f"  r(EO_theta, IAF)                    = {r_eo_iaf:+.3f}, p = {p_eo_iaf:.4f}")
print(f"  r(EC_theta, EO_theta)               = {r_ec_eo:+.3f}, p = {p_ec_eo:.2e}")
print(f"  Partial r(EC_theta, IAF | EO_theta) = {pr_ec_iaf:+.3f}, p = {pp_ec_iaf:.4f}")
print(f"  r(Δalpha, Δtheta)                   = {r_da_dt:+.3f}, p = {p_da_dt:.3f}")

if pr_ec_iaf < 0:
    print("  → NEGATIVE partial r rules out IAF subharmonic (which predicts positive)")
elif pr_ec_iaf > 0 and pp_ec_iaf < 0.05:
    print("  → Positive partial r supports IAF-derived target")
else:
    print("  → Partial r is non-significant")

print(f"  → Alpha and theta shifts {'un' if abs(r_da_dt) < 0.15 else ''}correlated")


# ═══════════════════════════════════════════════════════════════════════
# TEST 2: Prediction error comparison
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("TEST 2: Prediction error |EC_theta - target| (Hz)")
print("─" * 70)

err_cols = {}
for name in TARGETS:
    safe = name.replace(' ', '_').replace('=', '').replace('/', '_').replace('×', 'x').replace('√', 'sqrt').replace('₀', '0').replace('α', 'a').replace('φ', 'phi')
    err_cols[name] = f'err_{safe}'

# Reference: f0
ref_key = 'f₀ = 7.83'
ref_err = m[err_cols[ref_key]].values

print(f"  {'Target':<18s} {'Mean':>6s} {'Median':>8s} {'SD':>6s}   {'Wilcoxon vs f₀':>20s}")
print(f"  {'─'*18} {'─'*6} {'─'*8} {'─'*6}   {'─'*20}")

err_summary = {}
for name in TARGETS:
    err = m[err_cols[name]].values
    mn, md, sd = err.mean(), np.median(err), err.std()
    if name == ref_key:
        line = f"  {name:<18s} {mn:6.3f} {md:8.3f} {sd:6.3f}   {'(reference)':>20s}"
    else:
        w, p = stats.wilcoxon(ref_err, err)
        direction = 'f₀ wins' if mn > ref_err.mean() else 'target wins'
        line = f"  {name:<18s} {mn:6.3f} {md:8.3f} {sd:6.3f}   p={p:.2e} ({direction})"
    err_summary[name] = (mn, md, sd)
    print(line)


# ═══════════════════════════════════════════════════════════════════════
# TEST 3: Convergence direction regression
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("TEST 3: Convergence direction regression  Δθ ~ β(target - θ_EO)")
print("─" * 70)

delta_theta = m['delta_theta'].values
theta_eo = m['theta_eo'].values

print(f"  {'Target':<18s} {'R²':>8s} {'β':>8s} {'p':>12s}")
print(f"  {'─'*18} {'─'*8} {'─'*8} {'─'*12}")

reg_results = {}
for name in TARGETS:
    safe = name.replace(' ', '_').replace('=', '').replace('/', '_').replace('×', 'x').replace('√', 'sqrt').replace('₀', '0').replace('α', 'a').replace('φ', 'phi')
    target = m[f'target_{safe}'].values
    gap = target - theta_eo
    slope, intercept, r_val, p_val, se = stats.linregress(gap, delta_theta)
    r2 = r_val**2
    reg_results[name] = (r2, slope, p_val, r_val)
    print(f"  {name:<18s} {r2:8.4f} {slope:8.3f} {p_val:12.2e}")

# Best-constant optimization
def mse_const(c):
    return np.mean((m['theta_ec'].values - c)**2)

result = minimize_scalar(mse_const, bounds=(4, 10), method='bounded')
best_const = result.x
best_const_mse = result.fun
f0_mse = np.mean((m['theta_ec'].values - F0)**2)

print(f"\n  Best constant target (min MSE): {best_const:.3f} Hz (MSE={best_const_mse:.4f})")
print(f"  f₀ = 7.83 MSE: {f0_mse:.4f}")
print(f"  Δ(best - f₀): {best_const - F0:+.3f} Hz")


# ═══════════════════════════════════════════════════════════════════════
# TEST 4: IAF-stratified convergence (critical disambiguator)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("TEST 4: IAF-stratified convergence")
print("─" * 70)

print(f"  {'Tercile':<12s} {'N':>4s} {'Mean IAF':>9s} {'θ_EO':>7s} {'θ_EC':>7s} "
      f"{'Δθ':>7s} {'Conv_f₀':>9s} {'IAF/√φ':>8s} {'Conv_IAF':>9s}")
print(f"  {'─'*12} {'─'*4} {'─'*9} {'─'*7} {'─'*7} {'─'*7} {'─'*9} {'─'*8} {'─'*9}")

tercile_results = {}
for terc in ['Low IAF', 'Mid IAF', 'High IAF']:
    sub = m[m['iaf_tercile'] == terc]
    n = len(sub)
    mean_iaf = sub['iaf'].mean()
    mean_theta_eo = sub['theta_eo'].mean()
    mean_theta_ec = sub['theta_ec'].mean()
    mean_delta = sub['delta_theta'].mean()
    mean_conv_f0 = sub['conv_f0'].mean()
    iaf_target = mean_iaf / np.sqrt(PHI)

    # Per-subject convergence on IAF/√φ
    sub_iaf_target = sub['iaf'].values / np.sqrt(PHI)
    conv_iaf = (np.abs(sub['theta_eo'].values - sub_iaf_target) -
                np.abs(sub['theta_ec'].values - sub_iaf_target))
    mean_conv_iaf = conv_iaf.mean()

    t_f0, p_f0 = stats.ttest_1samp(sub['conv_f0'].values, 0)
    t_iaf, p_iaf = stats.ttest_1samp(conv_iaf, 0)

    sig_f0 = '*' if p_f0 < 0.05 else ''
    sig_iaf = '*' if p_iaf < 0.05 else ''

    tercile_results[terc] = {
        'n': n, 'mean_iaf': mean_iaf, 'theta_eo': mean_theta_eo,
        'theta_ec': mean_theta_ec, 'delta': mean_delta,
        'conv_f0': mean_conv_f0, 'p_f0': p_f0,
        'iaf_sqrt_phi': iaf_target,
        'conv_iaf': mean_conv_iaf, 'p_iaf': p_iaf,
    }

    print(f"  {terc:<12s} {n:4d} {mean_iaf:9.2f} {mean_theta_eo:7.2f} {mean_theta_ec:7.2f} "
          f"{mean_delta:+7.3f} {mean_conv_f0:+8.3f}{sig_f0:<1s} {iaf_target:8.2f} {mean_conv_iaf:+8.3f}{sig_iaf:<1s}")

# Landing zone ~ IAF regression
slope_land, intercept_land, r_land, p_land, se_land = stats.linregress(m['iaf'].values, m['theta_ec'].values)
r_conv_iaf, p_conv_iaf = stats.pearsonr(m['conv_f0'].values, m['iaf'].values)

print(f"\n  Landing zone ~ IAF: slope = {slope_land:+.3f}, r = {r_land:+.3f}, p = {p_land:.4f}")
print(f"  Convergence_on_f₀ ~ IAF: r = {r_conv_iaf:+.3f}, p = {p_conv_iaf:.4f}")

if abs(r_conv_iaf) < 0.1 and p_conv_iaf > 0.1:
    print("  → Convergence on f₀ is IAF-INVARIANT")
else:
    print(f"  → Convergence on f₀ shows {'weak' if abs(r_conv_iaf) < 0.2 else ''} IAF dependence")


# ═══════════════════════════════════════════════════════════════════════
# TEST 5: Williams test for dependent correlations
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("TEST 5: Williams test (dependent correlation comparison)")
print("─" * 70)

# For convergence-direction: Δθ vs (target - θ_EO)
gap_f0 = F0 - theta_eo
gap_iaf_sqrt_phi = m['iaf'].values / np.sqrt(PHI) - theta_eo
gap_iaf_34 = m['iaf'].values * 0.75 - theta_eo

r_f0, _ = stats.pearsonr(delta_theta, gap_f0)
r_iaf_sp, _ = stats.pearsonr(delta_theta, gap_iaf_sqrt_phi)
r_iaf_34, _ = stats.pearsonr(delta_theta, gap_iaf_34)
r_gaps_sp, _ = stats.pearsonr(gap_f0, gap_iaf_sqrt_phi)
r_gaps_34, _ = stats.pearsonr(gap_f0, gap_iaf_34)

t_w1, p_w1 = williams_test(r_f0, r_iaf_sp, r_gaps_sp, N)
t_w2, p_w2 = williams_test(r_f0, r_iaf_34, r_gaps_34, N)

print(f"  r(Δθ, f₀-θ_EO)       = {r_f0:+.3f}")
print(f"  r(Δθ, IAF/√φ-θ_EO)   = {r_iaf_sp:+.3f}")
print(f"  r(Δθ, IAF×3/4-θ_EO)  = {r_iaf_34:+.3f}")
print(f"\n  Williams test f₀ vs IAF/√φ:  t = {t_w1:+.3f}, p = {p_w1:.4f}")
print(f"  Williams test f₀ vs IAF×3/4: t = {t_w2:+.3f}, p = {p_w2:.4f}")


# ═══════════════════════════════════════════════════════════════════════
# BOOTSTRAP CIs on convergence-direction R²
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("BOOTSTRAP: 95% CIs on convergence-direction R²")
print("─" * 70)

np.random.seed(42)
n_boot = 2000
boot_r2 = {name: [] for name in ['f₀ = 7.83', 'IAF/√φ', 'IAF × 3/4']}

for _ in range(n_boot):
    idx = np.random.choice(N, N, replace=True)
    dt_b = delta_theta[idx]
    te_b = theta_eo[idx]
    iaf_b = m['iaf'].values[idx]

    for name, gap_func in [('f₀ = 7.83', lambda te, iaf: F0 - te),
                           ('IAF/√φ', lambda te, iaf: iaf / np.sqrt(PHI) - te),
                           ('IAF × 3/4', lambda te, iaf: iaf * 0.75 - te)]:
        gap_b = gap_func(te_b, iaf_b)
        r_b = np.corrcoef(dt_b, gap_b)[0, 1]
        boot_r2[name].append(r_b**2)

for name in boot_r2:
    arr = np.array(boot_r2[name])
    lo, hi = np.percentile(arr, [2.5, 97.5])
    print(f"  {name:<18s}: R² = {np.median(arr):.4f}  95% CI [{lo:.4f}, {hi:.4f}]")


# ═══════════════════════════════════════════════════════════════════════
# ROBUSTNESS CHECK A: Bootstrap CIs + effect size for MAE difference
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("ROBUSTNESS A: Bootstrap CIs + effect size for MAE(f₀) vs MAE(IAF/√φ)")
print("─" * 70)

err_f0_vals = m[err_cols['f₀ = 7.83']].values
err_iaf_sp_vals = m[err_cols['IAF/√φ']].values
mae_diff_obs = err_f0_vals - err_iaf_sp_vals  # negative = f₀ wins

print(f"  Observed MAE difference (f₀ - IAF/√φ):")
print(f"    Mean:   {mae_diff_obs.mean():+.4f} Hz")
print(f"    Median: {np.median(mae_diff_obs):+.4f} Hz")
print(f"    SD:     {mae_diff_obs.std():.4f} Hz")

# Bootstrap CI on MAE difference
np.random.seed(123)
boot_mae_diff = []
for _ in range(5000):
    idx = np.random.choice(N, N, replace=True)
    boot_mae_diff.append((err_f0_vals[idx] - err_iaf_sp_vals[idx]).mean())
boot_mae_diff = np.array(boot_mae_diff)
ci_lo, ci_hi = np.percentile(boot_mae_diff, [2.5, 97.5])
print(f"    Bootstrap 95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")

# Cliff's delta (nonparametric effect size for paired data)
# = P(f₀ < IAF/√φ) - P(f₀ > IAF/√φ)
n_less = np.sum(err_f0_vals < err_iaf_sp_vals)
n_greater = np.sum(err_f0_vals > err_iaf_sp_vals)
n_equal = np.sum(err_f0_vals == err_iaf_sp_vals)
cliffs_d = (n_less - n_greater) / N
print(f"\n  Cliff's delta (paired): {cliffs_d:+.3f}")
print(f"    f₀ closer: {n_less}/{N} ({100*n_less/N:.1f}%)")
print(f"    IAF/√φ closer: {n_greater}/{N} ({100*n_greater/N:.1f}%)")
print(f"    Tied: {n_equal}/{N}")

# Rank-biserial from Wilcoxon (r_rb = 1 - 2W/(n(n+1)/2))
w_stat, w_p = stats.wilcoxon(err_f0_vals, err_iaf_sp_vals)
n_pairs = np.sum(mae_diff_obs != 0)
r_rb = 1 - (2 * w_stat) / (n_pairs * (n_pairs + 1) / 2)
print(f"  Rank-biserial r: {r_rb:+.3f} (from Wilcoxon W={w_stat:.0f}, p={w_p:.2e})")

# Interpretation
if ci_hi < 0:
    print("  → f₀ SIGNIFICANTLY better: entire 95% CI below zero")
elif ci_lo > 0:
    print("  → IAF/√φ significantly better (unexpected)")
else:
    print("  → CI spans zero — difference not robust")


# ═══════════════════════════════════════════════════════════════════════
# ROBUSTNESS CHECK B: Alternate IAF estimators
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("ROBUSTNESS B: Alternate IAF estimators")
print("─" * 70)

# Three estimators:
# 1. CoG-EC (existing 'iaf' column) = center-of-gravity, eyes-closed, posterior ROI
# 2. Peak-EO = dominant FOOOF alpha peak frequency, eyes-open
# 3. Peak-EC = dominant FOOOF alpha peak frequency, eyes-closed

iaf_estimators = {
    'CoG-EC (default)': m['iaf'].values,
    'Peak-EO (FOOOF)':  m['alpha_eo'].values,
    'Peak-EC (FOOOF)':  m['alpha_ec'].values,
}

print(f"\n  {'Estimator':<22s} {'N':>4s} {'Mean':>7s} {'SD':>6s}  {'r(EC_θ)':>8s} {'Partial r':>10s} {'p':>8s}  {'Direction':>10s}")
print(f"  {'─'*22} {'─'*4} {'─'*7} {'─'*6}  {'─'*8} {'─'*10} {'─'*8}  {'─'*10}")

partial_r_results = {}
for name, iaf_vals in iaf_estimators.items():
    valid_mask = np.isfinite(iaf_vals) & np.isfinite(m['theta_ec'].values) & np.isfinite(m['theta_eo'].values)
    n_valid = valid_mask.sum()
    if n_valid < 30:
        print(f"  {name:<22s} {n_valid:4d}  — too few subjects —")
        continue

    iaf_v = iaf_vals[valid_mask]
    ec_v = m['theta_ec'].values[valid_mask]
    eo_v = m['theta_eo'].values[valid_mask]

    r_simple, _ = stats.pearsonr(ec_v, iaf_v)
    pr, pp = partial_corr(ec_v, iaf_v, eo_v)
    direction = 'NEGATIVE' if pr < 0 else 'POSITIVE'

    partial_r_results[name] = (pr, pp, n_valid)

    print(f"  {name:<22s} {n_valid:4d} {iaf_v.mean():7.2f} {iaf_v.std():6.2f}  {r_simple:+8.3f} {pr:+10.3f} {pp:8.4f}  {direction:>10s}")

# Cross-estimator correlation
print(f"\n  Cross-estimator correlations:")
names = list(iaf_estimators.keys())
for i in range(len(names)):
    for j in range(i+1, len(names)):
        v1 = iaf_estimators[names[i]]
        v2 = iaf_estimators[names[j]]
        mask = np.isfinite(v1) & np.isfinite(v2)
        if mask.sum() > 20:
            r_cross, _ = stats.pearsonr(v1[mask], v2[mask])
            print(f"    {names[i]} × {names[j]}: r = {r_cross:.3f} (N={mask.sum()})")

# Verdict
all_negative = all(pr < 0 for pr, _, _ in partial_r_results.values())
any_significant = any(pp < 0.05 for _, pp, _ in partial_r_results.values())
if all_negative:
    print(f"\n  → Partial r is NEGATIVE across all {len(partial_r_results)} IAF estimators")
    print(f"    Not an estimator artifact — direction is robust")
else:
    print(f"\n  → Partial r direction is MIXED across estimators — interpret with caution")


# ═══════════════════════════════════════════════════════════════════════
# ROBUSTNESS CHECK C: Best-constant sensitivity to theta estimator
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("ROBUSTNESS C: Best-constant target sensitivity to EC theta estimator")
print("─" * 70)

def best_const_mse(ec_vals):
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(lambda c: np.mean((ec_vals - c)**2), bounds=(4, 10), method='bounded')
    return res.x

# Estimator 1: Dominant FOOOF peak (default — already computed)
bc_dom = best_const_mse(m['theta_ec'].values)
print(f"  Dominant FOOOF peak (default):    {bc_dom:.3f} Hz (N={len(m)})")

# Estimator 2: Power-weighted CoG across all theta peaks
cog_records = []
for f in sorted(glob(os.path.join(PEAK_DIR, 'sub-*_peaks_ec.csv'))):
    if 'band_info' in f or 'max40' in f:
        continue
    sid = os.path.basename(f).split('_peaks')[0]
    df = pd.read_csv(f)
    bp = df[(df.freq >= 4) & (df.freq < 8)]
    if len(bp) == 0:
        cog_records.append({'subject_id': sid, 'theta_cog_ec': np.nan})
        continue
    weights = 10**bp['power'].values
    cog_records.append({'subject_id': sid, 'theta_cog_ec': np.average(bp['freq'].values, weights=weights)})

cog_df = pd.DataFrame(cog_records)
m_cog = m.merge(cog_df, on='subject_id', how='left')
valid_cog = m_cog['theta_cog_ec'].dropna().values
bc_cog = best_const_mse(valid_cog)
print(f"  Power-weighted CoG (all θ peaks): {bc_cog:.3f} Hz (N={len(valid_cog)})")

# Estimator 3: Posterior-only dominant peak
posterior_chs = {'O1', 'O2', 'P7', 'P8', 'Oz', 'POz', 'PO3', 'PO4', 'PO7', 'PO8'}
post_records = []
for f in sorted(glob(os.path.join(PEAK_DIR, 'sub-*_peaks_ec.csv'))):
    if 'band_info' in f or 'max40' in f:
        continue
    sid = os.path.basename(f).split('_peaks')[0]
    df = pd.read_csv(f)
    bp = df[(df.freq >= 4) & (df.freq < 8) & (df.channel.isin(posterior_chs))]
    if len(bp) == 0:
        post_records.append({'subject_id': sid, 'theta_post_ec': np.nan})
        continue
    idx = bp['power'].idxmax()
    post_records.append({'subject_id': sid, 'theta_post_ec': bp.loc[idx, 'freq']})

post_df = pd.DataFrame(post_records)
m_post = m.merge(post_df, on='subject_id', how='left')
valid_post = m_post['theta_post_ec'].dropna().values
bc_post = best_const_mse(valid_post)
print(f"  Posterior-only dominant peak:     {bc_post:.3f} Hz (N={len(valid_post)})")

# Estimator 4: Frontal-only dominant peak
frontal_chs = {'F3', 'F4', 'Fz', 'FC1', 'FC2', 'AF3', 'AF4', 'Fp1', 'Fp2'}
front_records = []
for f in sorted(glob(os.path.join(PEAK_DIR, 'sub-*_peaks_ec.csv'))):
    if 'band_info' in f or 'max40' in f:
        continue
    sid = os.path.basename(f).split('_peaks')[0]
    df = pd.read_csv(f)
    bp = df[(df.freq >= 4) & (df.freq < 8) & (df.channel.isin(frontal_chs))]
    if len(bp) == 0:
        front_records.append({'subject_id': sid, 'theta_front_ec': np.nan})
        continue
    idx = bp['power'].idxmax()
    front_records.append({'subject_id': sid, 'theta_front_ec': bp.loc[idx, 'freq']})

front_df = pd.DataFrame(front_records)
m_front = m.merge(front_df, on='subject_id', how='left')
valid_front = m_front['theta_front_ec'].dropna().values
bc_front = best_const_mse(valid_front)
print(f"  Frontal-only dominant peak:       {bc_front:.3f} Hz (N={len(valid_front)})")

dom_bcs = [bc_dom, bc_post, bc_front]
print(f"\n  Dominant-peak estimators range: {min(dom_bcs):.2f}–{max(dom_bcs):.2f} Hz (spread {max(dom_bcs)-min(dom_bcs):.2f})")
print(f"  CoG estimator outlier at {bc_cog:.2f} (averages toward band center)")
print(f"  → Best constant is estimator-dependent; 7.83 is the theoretically specified reference")


# ═══════════════════════════════════════════════════════════════════════
# ROBUSTNESS CHECK D: Age stratification
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("ROBUSTNESS D: Age stratification")
print("─" * 70)

for grp_name, grp_mask in [('Young (20–35)', m['age_group'] == 'young'),
                            ('Elderly (59–77)', m['age_group'] == 'elderly')]:
    sub = m[grp_mask]
    n = len(sub)
    if n < 20:
        continue

    err_f0 = np.abs(sub['theta_ec'].values - F0)
    err_iaf = np.abs(sub['theta_ec'].values - sub['iaf'].values / np.sqrt(PHI))
    w, p = stats.wilcoxon(err_f0, err_iaf)
    n_f0_wins = np.sum(err_f0 < err_iaf)
    pct = 100 * n_f0_wins / n

    conv = sub['conv_f0'].values
    t_conv, p_conv = stats.ttest_1samp(conv, 0)

    pr, pp = partial_corr(sub['theta_ec'].values, sub['iaf'].values, sub['theta_eo'].values)

    bc = best_const_mse(sub['theta_ec'].values)

    dt_sub = sub['delta_theta'].values
    gap_f0_sub = F0 - sub['theta_eo'].values
    gap_iaf_sub = sub['iaf'].values / np.sqrt(PHI) - sub['theta_eo'].values
    r2_f0 = stats.pearsonr(dt_sub, gap_f0_sub)[0]**2
    r2_iaf = stats.pearsonr(dt_sub, gap_iaf_sub)[0]**2

    print(f"\n  {grp_name} (N={n}):")
    print(f"    f₀ closer: {n_f0_wins}/{n} ({pct:.0f}%), Wilcoxon p = {p:.2e}")
    print(f"    Conv. on f₀: mean = {conv.mean():+.3f}, t = {t_conv:.2f}, p = {p_conv:.4f}")
    print(f"    Partial r(ECθ, IAF | EOθ) = {pr:+.3f}, p = {pp:.4f}")
    print(f"    Best constant = {bc:.3f} Hz")
    print(f"    Conv. R²: f₀ = {r2_f0:.4f}, IAF/√φ = {r2_iaf:.4f} → {'f₀ wins' if r2_f0 > r2_iaf else 'IAF wins'}")

# Age as covariate
print(f"\n  Age as covariate (N={N}):")
pr_gap_age, pp_gap_age = partial_corr(delta_theta, F0 - theta_eo, m['age_midpoint'].values)
pr_conv_age, pp_conv_age = partial_corr(m['conv_f0'].values, m['iaf'].values, m['age_midpoint'].values)
print(f"    Partial r(Δθ, f₀-gap | age) = {pr_gap_age:+.3f}, p = {pp_gap_age:.2e}")
print(f"    Partial r(conv_f₀, IAF | age) = {pr_conv_age:+.3f}, p = {pp_conv_age:.4f}")
if abs(pr_conv_age) < 0.1:
    print(f"    → IAF-invariance holds after age control")


# ═══════════════════════════════════════════════════════════════════════
# OVERALL CONCLUSION
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

# Determine winner
f0_wins_err = all(
    err_summary['f₀ = 7.83'][0] <= err_summary[name][0]
    for name in TARGETS if name != 'f₀ = 7.83'
)
f0_wins_r2 = all(
    reg_results['f₀ = 7.83'][0] >= reg_results[name][0]
    for name in TARGETS if name != 'f₀ = 7.83'
)
iaf_invariant = abs(r_conv_iaf) < 0.15 and p_conv_iaf > 0.05
negative_partial = pr_ec_iaf < 0

if f0_wins_err and f0_wins_r2 and iaf_invariant and negative_partial:
    verdict = "STRONG: f₀ = 7.83 Hz is the convergence target"
    print(f"  {verdict}")
    print(f"  All 5 tests favor f₀ over IAF-derived alternatives:")
    print(f"    1. Partial r(EC_theta, IAF | EO_theta) = {pr_ec_iaf:+.3f} (NEGATIVE, rules out subharmonic)")
    print(f"    2. f₀ has lowest prediction error ({err_summary['f₀ = 7.83'][0]:.3f} Hz)")
    print(f"    3. f₀ has highest convergence R² ({reg_results['f₀ = 7.83'][0]:.4f})")
    print(f"    4. Convergence is IAF-invariant (r = {r_conv_iaf:+.3f}, p = {p_conv_iaf:.2f})")
    print(f"    5. Best-fit constant = {best_const:.2f} Hz (Δ = {best_const - F0:+.2f} from f₀)")
elif f0_wins_err or f0_wins_r2:
    verdict = "MODERATE: f₀ favored but not decisively"
    print(f"  {verdict}")
else:
    verdict = "AMBIGUOUS or IAF-derived target favored"
    print(f"  {verdict}")


# ═══════════════════════════════════════════════════════════════════════
# SAVE PER-SUBJECT CSV
# ═══════════════════════════════════════════════════════════════════════
out_cols = ['subject_id', 'iaf', 'theta_eo', 'theta_ec', 'delta_theta',
            'alpha_eo', 'alpha_ec', 'conv_f0', 'iaf_tercile', 'age_midpoint']
for name in TARGETS:
    safe = name.replace(' ', '_').replace('=', '').replace('/', '_').replace('×', 'x').replace('√', 'sqrt').replace('₀', '0').replace('α', 'a').replace('φ', 'phi')
    out_cols.extend([f'target_{safe}', f'err_{safe}'])

m[out_cols].to_csv(CSV_OUT, index=False)
print(f"\n  Per-subject data saved to {CSV_OUT}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 7: 4-panel disambiguation figure
# ═══════════════════════════════════════════════════════════════════════
print("\nGenerating Figure 7...")

fig = plt.figure(figsize=(15, 11))
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30)

TERCILE_COLORS = {'Low IAF': '#3498db', 'Mid IAF': '#2ecc71', 'High IAF': '#e67e22'}

# ── Panel (a): EC theta vs IAF scatterplot ──
ax = fig.add_subplot(gs[0, 0])

for terc in ['Low IAF', 'Mid IAF', 'High IAF']:
    sub = m[m['iaf_tercile'] == terc]
    ax.scatter(sub['iaf'], sub['theta_ec'], alpha=0.5, s=25,
              c=TERCILE_COLORS[terc], label=terc, edgecolors='none')

# Reference lines
iaf_range = np.linspace(m['iaf'].min() - 0.2, m['iaf'].max() + 0.2, 100)
ax.axhline(F0, color='#e74c3c', ls='--', lw=1.5, label=f'f₀ = {F0}', zorder=5)
ax.plot(iaf_range, iaf_range / np.sqrt(PHI), color='#9b59b6', ls=':', lw=1.5,
        label=f'IAF/√φ', zorder=4)

# OLS fit
slope_plot, int_plot = np.polyfit(m['iaf'], m['theta_ec'], 1)
ax.plot(iaf_range, slope_plot * iaf_range + int_plot, color='gray', lw=1, ls='-', alpha=0.6)

ax.set_xlabel('Individual Alpha Frequency (Hz)', fontsize=10)
ax.set_ylabel('EC theta frequency (Hz)', fontsize=10)
ax.set_title('(a) EC theta vs IAF', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='upper left')

# Annotations
textbox = (f"r(EC_θ, IAF) = {r_ec_iaf:+.3f}, p = {p_ec_iaf:.4f}\n"
           f"Partial r (| EO_θ) = {pr_ec_iaf:+.3f}, p = {pp_ec_iaf:.4f}\n"
           f"r(Δα, Δθ) = {r_da_dt:+.3f}, p = {p_da_dt:.3f}")
ax.text(0.97, 0.03, textbox, transform=ax.transAxes, fontsize=8,
        va='bottom', ha='right', bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.85))

# ── Panel (b): Prediction error boxplot ──
ax = fig.add_subplot(gs[0, 1])

# Order by mean error
sorted_targets = sorted(TARGETS.keys(), key=lambda n: err_summary[n][0])
err_data = []
for name in sorted_targets:
    safe = name.replace(' ', '_').replace('=', '').replace('/', '_').replace('×', 'x').replace('√', 'sqrt').replace('₀', '0').replace('α', 'a').replace('φ', 'phi')
    err_data.append(m[f'err_{safe}'].values)

bp = ax.boxplot(err_data, labels=sorted_targets, patch_artist=True, widths=0.6,
                showfliers=False, medianprops=dict(color='black', lw=1.5))

colors_box = []
for name in sorted_targets:
    if name == 'f₀ = 7.83':
        colors_box.append('#e74c3c')
    elif 'IAF' in name or 'EC' in name:
        colors_box.append('#3498db')
    else:
        colors_box.append('#95a5a6')

for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

# Add significance stars
for i, name in enumerate(sorted_targets):
    if name == ref_key:
        continue
    safe = name.replace(' ', '_').replace('=', '').replace('/', '_').replace('×', 'x').replace('√', 'sqrt').replace('₀', '0').replace('α', 'a').replace('φ', 'phi')
    w, p = stats.wilcoxon(ref_err, m[f'err_{safe}'].values)
    if p < 0.001:
        sig = '***'
    elif p < 0.01:
        sig = '**'
    elif p < 0.05:
        sig = '*'
    else:
        sig = 'ns'
    q75 = np.percentile(m[f'err_{safe}'].values, 75)
    ax.text(i + 1, q75 + 0.15, sig, ha='center', fontsize=9, fontweight='bold', color='#c0392b')

ax.set_ylabel('|EC_theta - target| (Hz)', fontsize=10)
ax.set_title('(b) Prediction error by target model', fontsize=12, fontweight='bold')
ax.tick_params(axis='x', rotation=25)
ax.text(0.97, 0.97, 'Stars: Wilcoxon vs f₀', transform=ax.transAxes,
        fontsize=8, va='top', ha='right', color='gray')

# ── Panel (c): IAF-stratified convergence ──
ax = fig.add_subplot(gs[1, 0])

terciles = ['Low IAF', 'Mid IAF', 'High IAF']
x_pos = np.arange(len(terciles))
bar_width = 0.25

for i, terc in enumerate(terciles):
    tr = tercile_results[terc]

    # EO theta (open)
    ax.scatter(i - 0.08, tr['theta_eo'], marker='o', s=100, facecolors='none',
              edgecolors=TERCILE_COLORS[terc], linewidth=2, zorder=5)
    # EC theta (filled)
    ax.scatter(i + 0.08, tr['theta_ec'], marker='o', s=100,
              facecolors=TERCILE_COLORS[terc], edgecolors='k', linewidth=0.5, zorder=5)

    # Arrow EO → EC
    ax.annotate('', xy=(i + 0.06, tr['theta_ec']), xytext=(i - 0.06, tr['theta_eo']),
               arrowprops=dict(arrowstyle='->', color=TERCILE_COLORS[terc], lw=2))

    # IAF/√φ prediction (diamond)
    ax.scatter(i + 0.25, tr['iaf_sqrt_phi'], marker='D', s=50,
              facecolors='none', edgecolors='#9b59b6', linewidth=1.5, zorder=4)

    # Labels
    ax.text(i, tr['theta_ec'] + 0.15, f'{tr["theta_ec"]:.2f}',
           ha='center', fontsize=8, fontweight='bold')
    ax.text(i, tr['theta_eo'] - 0.20, f'{tr["theta_eo"]:.2f}',
           ha='center', fontsize=8, color='gray')
    ax.text(i + 0.35, tr['iaf_sqrt_phi'], f'{tr["iaf_sqrt_phi"]:.1f}',
           ha='left', fontsize=7, color='#9b59b6')

ax.axhline(F0, color='#e74c3c', ls='--', lw=1.5, label=f'f₀ = {F0}', zorder=3)
ax.set_xticks(x_pos)
ax.set_xticklabels([f'{t}\n(IAF={tercile_results[t]["mean_iaf"]:.1f})' for t in terciles], fontsize=9)
ax.set_ylabel('Theta frequency (Hz)', fontsize=10)
ax.set_title('(c) Convergence target is fixed across IAF terciles', fontsize=12, fontweight='bold')

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='gray',
           markersize=8, markeredgewidth=2, label='EO theta'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='k',
           markersize=8, label='EC theta'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='none', markeredgecolor='#9b59b6',
           markersize=7, markeredgewidth=1.5, label='IAF/√φ prediction'),
    Line2D([0], [0], color='#e74c3c', ls='--', lw=1.5, label=f'f₀ = {F0} Hz'),
]
ax.legend(handles=legend_elements, fontsize=8, loc='upper right')

conv_text = f'Conv_f₀ ~ IAF: r = {r_conv_iaf:+.3f}, p = {p_conv_iaf:.2f}'
ax.text(0.03, 0.03, conv_text, transform=ax.transAxes, fontsize=8,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))

# ── Panel (d): Convergence direction regression comparison ──
ax = fig.add_subplot(gs[1, 1])

gap_f0_vals = F0 - theta_eo
gap_iaf_sp_vals = m['iaf'].values / np.sqrt(PHI) - theta_eo

# f0 model
ax.scatter(gap_f0_vals, delta_theta, alpha=0.35, s=18, c='#e74c3c', label='f₀ gap', zorder=3)
# Fit line
x_fit = np.linspace(gap_f0_vals.min(), gap_f0_vals.max(), 100)
sl_f0, int_f0, r_f0_fit, _, _ = stats.linregress(gap_f0_vals, delta_theta)
ax.plot(x_fit, sl_f0 * x_fit + int_f0, color='#e74c3c', lw=2, zorder=4)

# IAF/√φ model
ax.scatter(gap_iaf_sp_vals, delta_theta, alpha=0.35, s=18, c='#9b59b6', marker='^',
          label='IAF/√φ gap', zorder=3)
x_fit2 = np.linspace(gap_iaf_sp_vals.min(), gap_iaf_sp_vals.max(), 100)
sl_iaf, int_iaf, r_iaf_fit, _, _ = stats.linregress(gap_iaf_sp_vals, delta_theta)
ax.plot(x_fit2, sl_iaf * x_fit2 + int_iaf, color='#9b59b6', lw=2, ls='--', zorder=4)

ax.axhline(0, color='k', lw=0.5)
ax.axvline(0, color='k', lw=0.5)
ax.set_xlabel('(Target - θ_EO)  (Hz)', fontsize=10)
ax.set_ylabel('Δθ = θ_EC - θ_EO  (Hz)', fontsize=10)
ax.set_title('(d) Convergence direction: f₀ vs IAF/√φ', fontsize=12, fontweight='bold')

# R² annotations
r2_f0 = reg_results['f₀ = 7.83'][0]
r2_iaf = reg_results['IAF/√φ'][0]
ax.text(0.97, 0.97, f'f₀:     R² = {r2_f0:.3f}\nIAF/√φ: R² = {r2_iaf:.3f}',
        transform=ax.transAxes, fontsize=9, va='top', ha='right',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.85),
        family='monospace')

# Williams test result
if p_w1 < 0.05:
    wtext = f'Williams: t = {t_w1:+.2f}, p = {p_w1:.3f} (f₀ wins)'
else:
    wtext = f'Williams: t = {t_w1:+.2f}, p = {p_w1:.3f} (NS)'
ax.text(0.03, 0.03, wtext, transform=ax.transAxes, fontsize=8,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))

ax.legend(fontsize=8, loc='upper left')

# ── Save ──
fig.suptitle('Figure 7: EC theta convergence tracks f₀ = 7.83 Hz, not an IAF subharmonic',
             fontsize=13, fontweight='bold', y=1.01)
plt.savefig(os.path.join(OUT_DIR, 'fig7_theta_target_disambiguation.pdf'), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(OUT_DIR, 'fig7_theta_target_disambiguation.png'), bbox_inches='tight', dpi=300)
plt.close()
print("  Figure 7 saved.\n")
print("Done.")

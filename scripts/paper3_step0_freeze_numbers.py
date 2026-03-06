#!/usr/bin/env python3
"""
Paper 3 Step 0: Freeze numbers and validate measurement-quality gradient.

This script:
1. Computes measurement-quality gradient (d vs spectral SNR proxy) — GO/NO-GO GATE
2. Computes Dortmund full 9-base degree-3 cross-base CSV for all 4 conditions
3. Verifies all key numbers against source CSVs
4. Saves everything to exports_dortmund/paper3_frozen/

The measurement-quality hypothesis: Cohen's d (or d̄/d̄_null) tracks spectral SNR
via aperiodic fit quality. All conditions/datasets should fall on a single curve
when plotted against alpha peak prominence.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys

# ============================================================
# CONFIGURATION
# ============================================================

PHI = (1 + np.sqrt(5)) / 2
F0 = 7.83

# Degree-3 position generation (from generate_lemon_paper_figures.py)
def natural_positions_degree3(base):
    inv_b = 1.0 / base
    positions = {'boundary': 0.0, 'attractor': 0.5}
    candidates = [
        ('noble_1', inv_b),
        ('noble_2', inv_b**2),
        ('inv_noble_1', 1.0 - inv_b),
        ('inv_noble_2', 1.0 - inv_b**2),
        ('noble_3', inv_b**3),
        ('inv_noble_3', 1.0 - inv_b**3),
    ]
    MIN_SEP = 0.02
    for name, val in candidates:
        if val < MIN_SEP or val > (1.0 - MIN_SEP):
            continue
        too_close = any(min(abs(val - p), 1 - abs(val - p)) < MIN_SEP
                        for p in positions.values())
        if not too_close:
            positions[name] = val
    return positions

def lattice_coordinate(f, f0, base):
    return (np.log(f / f0) / np.log(base)) % 1.0

def min_distance_to_positions(u, positions):
    pos_vals = np.array(list(positions.values()))
    dists = np.abs(u[:, None] - pos_vals[None, :])
    dists = np.minimum(dists, 1.0 - dists)
    return np.min(dists, axis=1)

def null_expected_d(positions):
    """Expected mean distance for uniform distribution given position layout."""
    pos_vals = sorted(positions.values())
    pos_vals_ext = pos_vals + [p + 1.0 for p in pos_vals]
    total = 0.0
    for i in range(len(pos_vals)):
        left = pos_vals_ext[i]
        right = pos_vals_ext[i + 1] if i + 1 < len(pos_vals_ext) else pos_vals[0] + 1.0
        # For each Voronoi cell, the expected distance is (right - left) / 4
        cell_width = right - left
        total += cell_width * (cell_width / 4.0)
    return total

# 9 bases
BASES = {
    'phi': PHI,
    'sqrt2': np.sqrt(2),
    '3/2': 1.5,
    '2': 2.0,
    '1.7': 1.7,
    '1.8': 1.8,
    'pi': np.pi,
    '1.4': 1.4,
    'e': np.e,
}

# Degree-2 (4 positions for phi)
POSITIONS_DEG2 = {
    'boundary': 0.0,
    'noble_2': 1.0 / PHI**2,
    'attractor': 0.5,
    'noble_1': 1.0 / PHI,
}

# Band definitions
BANDS = ['delta', 'theta', 'alpha', 'gamma']

# ============================================================
# DATA LOADING
# ============================================================

def load_dataset(path, label):
    df = pd.read_csv(path)
    df['dataset_label'] = label
    return df

print("=" * 70)
print("STEP 0: FREEZE NUMBERS AND VALIDATE MEASUREMENT-QUALITY GRADIENT")
print("=" * 70)

# Load all per-subject CSVs
datasets = {}

# Dortmund OT — 4 conditions
dort_base = '/Volumes/T9/dortmund_data/lattice_results_ot'
for cond in ['EyesClosed_pre', 'EyesClosed_post', 'EyesOpen_pre', 'EyesOpen_post']:
    path = f'{dort_base}/dortmund_ot_dominant_peaks_{cond}.csv'
    if os.path.exists(path):
        datasets[f'Dortmund_{cond}'] = load_dataset(path, f'Dort_{cond}')
        print(f"  Loaded Dortmund {cond}: N={len(datasets[f'Dortmund_{cond}'])}")

# LEMON OT
lemon_path = 'exports_lemon/per_subject_overlap_trim_f07.83/dominant_peak/per_subject_dominant_peaks.csv'
if os.path.exists(lemon_path):
    datasets['LEMON_EO'] = load_dataset(lemon_path, 'LEMON_EO')
    print(f"  Loaded LEMON EO: N={len(datasets['LEMON_EO'])}")

# LEMON EC (check if exists)
lemon_ec_path = 'exports_lemon/per_subject_overlap_trim_f07.83/dominant_peak/per_subject_dominant_peaks_EC.csv'
if os.path.exists(lemon_ec_path):
    datasets['LEMON_EC'] = load_dataset(lemon_ec_path, 'LEMON_EC')
    print(f"  Loaded LEMON EC: N={len(datasets['LEMON_EC'])}")

# EEGMMIDB OT
eegmmidb_path = 'exports_eegmmidb/per_subject_overlap_trim_f07.83/dominant_peak/per_subject_dominant_peaks.csv'
if os.path.exists(eegmmidb_path):
    datasets['EEGMMIDB'] = load_dataset(eegmmidb_path, 'EEGMMIDB')
    print(f"  Loaded EEGMMIDB: N={len(datasets['EEGMMIDB'])}")

# ============================================================
# PART 1: DEGREE-3 CROSS-BASE COMPARISON FOR ALL CONDITIONS
# ============================================================

print("\n" + "=" * 70)
print("PART 1: DEGREE-3 CROSS-BASE COMPARISON")
print("=" * 70)

def compute_crossbase_degree3(df, label):
    """Compute degree-3 cross-base comparison for a dataset."""
    results = []

    for base_name, base_val in sorted(BASES.items()):
        positions = natural_positions_degree3(base_val)
        n_pos = len(positions)
        null_d = null_expected_d(positions)

        # Compute per-subject mean_d
        subject_ds = []
        for _, row in df.iterrows():
            ds = []
            for band in BANDS:
                freq_col = f'{band}_freq'
                if freq_col in row.index and pd.notna(row[freq_col]):
                    u = lattice_coordinate(row[freq_col], F0, base_val)
                    pos_vals = np.array(list(positions.values()))
                    dists = np.abs(u - pos_vals)
                    dists = np.minimum(dists, 1.0 - dists)
                    ds.append(np.min(dists))
            if len(ds) >= 3:  # require at least 3 bands
                subject_ds.append(np.mean(ds))

        subject_ds = np.array(subject_ds)
        raw_mean_d = np.mean(subject_ds)
        d_over_null = raw_mean_d / null_d
        cohen_d = (raw_mean_d - null_d) / np.std(subject_ds, ddof=1)
        t_stat, p_val = stats.ttest_1samp(subject_ds, null_d)
        # One-sided: mean < null
        p_one = p_val / 2 if t_stat < 0 else 1.0 - p_val / 2

        results.append({
            'base': base_name,
            'base_value': base_val,
            'n_positions': n_pos,
            'raw_mean_d': raw_mean_d,
            'null_expected_d': null_d,
            'd_over_null': d_over_null,
            'd_times_n': raw_mean_d * n_pos,
            'cohen_d': cohen_d,
            't_stat': t_stat,
            'p_value': p_one,
            'n_subjects': len(subject_ds),
        })

    results_df = pd.DataFrame(results).sort_values('d_over_null')
    return results_df

# Compute for all datasets
crossbase_results = {}
for name, df in datasets.items():
    print(f"\n  --- {name} ---")
    cb = compute_crossbase_degree3(df, name)
    crossbase_results[name] = cb

    # Print top 3
    for i, row in cb.head(3).iterrows():
        print(f"    {row['base']:6s}: d/null={row['d_over_null']:.4f}, "
              f"Cohen d={row['cohen_d']:.4f}, p={row['p_value']:.4e}, "
              f"n_pos={row['n_positions']}")

    # Print phi specifically
    phi_row = cb[cb['base'] == 'phi'].iloc[0]
    phi_rank = (cb['d_over_null'] < phi_row['d_over_null']).sum() + 1
    print(f"    PHI rank: {phi_rank}/9, d/null={phi_row['d_over_null']:.4f}")

# Save Dortmund cross-base CSVs
out_dir = '/Volumes/T9/dortmund_data/lattice_results_ot/degree3_crossbase'
os.makedirs(out_dir, exist_ok=True)
for name, cb in crossbase_results.items():
    if name.startswith('Dortmund'):
        cond = name.replace('Dortmund_', '')
        out_path = f'{out_dir}/degree3_crossbase_{cond}.csv'
        cb.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")

# ============================================================
# PART 2: DEGREE-2 (4-POSITION) COHEN'S d FOR ALL CONDITIONS
# ============================================================

print("\n" + "=" * 70)
print("PART 2: DEGREE-2 (4-POSITION) COHEN'S d")
print("=" * 70)

def compute_cohens_d_deg2(df, label):
    """Compute degree-2 (4-position) per-subject mean_d and Cohen's d."""
    positions = POSITIONS_DEG2
    null_d = null_expected_d(positions)

    subject_ds = []
    for _, row in df.iterrows():
        ds = []
        for band in BANDS:
            freq_col = f'{band}_freq'
            if freq_col in row.index and pd.notna(row[freq_col]):
                u = lattice_coordinate(row[freq_col], F0, PHI)
                pos_vals = np.array(list(positions.values()))
                dists = np.abs(u - pos_vals)
                dists = np.minimum(dists, 1.0 - dists)
                ds.append(np.min(dists))
        if len(ds) >= 3:
            subject_ds.append(np.mean(ds))

    subject_ds = np.array(subject_ds)
    mean_d = np.mean(subject_ds)
    std_d = np.std(subject_ds, ddof=1)
    cohen_d = (null_d - mean_d) / std_d  # positive = better than null
    t_stat, p_val = stats.ttest_1samp(subject_ds, null_d)
    p_one = p_val / 2 if t_stat < 0 else 1.0 - p_val / 2

    return {
        'label': label,
        'n': len(subject_ds),
        'mean_d': mean_d,
        'std_d': std_d,
        'null_d': null_d,
        'cohen_d': cohen_d,
        't_stat': t_stat,
        'p_value': p_one,
    }

deg2_results = {}
for name, df in datasets.items():
    r = compute_cohens_d_deg2(df, name)
    deg2_results[name] = r
    print(f"  {name:25s}: mean_d={r['mean_d']:.4f}, null={r['null_d']:.4f}, "
          f"Cohen d={r['cohen_d']:.3f}, p={r['p_value']:.2e}, N={r['n']}")

# ============================================================
# PART 3: MEASUREMENT-QUALITY GRADIENT
# ============================================================

print("\n" + "=" * 70)
print("PART 3: MEASUREMENT-QUALITY GRADIENT (GO/NO-GO GATE)")
print("=" * 70)

# For each condition/dataset, compute:
# - Spectral quality proxy: median alpha peak power
# - Also: median n_peaks_total, mean aperiodic_exponent (where available)
# - d metric: Cohen's d (degree-2) and d/null (degree-3)

gradient_data = []
for name, df in datasets.items():
    # Spectral quality proxies
    alpha_power = df['alpha_power'].median() if 'alpha_power' in df.columns else np.nan
    n_peaks = df['n_peaks_total'].median() if 'n_peaks_total' in df.columns else np.nan
    aperiodic_exp = df['aperiodic_exponent'].mean() if 'aperiodic_exponent' in df.columns else np.nan

    # Also compute: mean alpha power (not median)
    alpha_power_mean = df['alpha_power'].mean() if 'alpha_power' in df.columns else np.nan

    # d metrics
    d2 = deg2_results[name]
    cb = crossbase_results[name]
    phi_row = cb[cb['base'] == 'phi'].iloc[0]

    gradient_data.append({
        'condition': name,
        'dataset': name.split('_')[0],
        'N': d2['n'],
        'alpha_power_median': alpha_power,
        'alpha_power_mean': alpha_power_mean,
        'n_peaks_median': n_peaks,
        'aperiodic_exponent_mean': aperiodic_exp,
        'cohen_d_deg2': d2['cohen_d'],
        'mean_d_deg2': d2['mean_d'],
        'p_value_deg2': d2['p_value'],
        'd_over_null_deg3': phi_row['d_over_null'],
        'cohen_d_deg3': phi_row['cohen_d'],
        'p_value_deg3': phi_row['p_value'],
        'phi_rank': (cb['d_over_null'] < phi_row['d_over_null']).sum() + 1,
    })

gradient_df = pd.DataFrame(gradient_data)
print("\n  Measurement-Quality Gradient Data:")
print(gradient_df[['condition', 'N', 'alpha_power_median', 'cohen_d_deg2',
                    'd_over_null_deg3', 'phi_rank']].to_string(index=False))

# Compute correlation: alpha_power vs d metrics
valid = gradient_df.dropna(subset=['alpha_power_median', 'cohen_d_deg2'])
if len(valid) >= 3:
    r_alpha_d2, p_alpha_d2 = stats.pearsonr(valid['alpha_power_median'], valid['cohen_d_deg2'])
    r_alpha_d3, p_alpha_d3 = stats.pearsonr(valid['alpha_power_median'],
                                              1.0 - valid['d_over_null_deg3'])
    rho_alpha_d2, p_rho_d2 = stats.spearmanr(valid['alpha_power_median'], valid['cohen_d_deg2'])

    print(f"\n  Correlation (alpha_power_median vs Cohen d deg2):")
    print(f"    Pearson r = {r_alpha_d2:.3f}, p = {p_alpha_d2:.4f}")
    print(f"    Spearman rho = {rho_alpha_d2:.3f}, p = {p_rho_d2:.4f}")
    print(f"  Correlation (alpha_power_median vs 1-d/null deg3):")
    print(f"    Pearson r = {r_alpha_d3:.3f}, p = {p_alpha_d3:.4f}")

    # Also try n_peaks as proxy
    if 'n_peaks_median' in valid.columns and valid['n_peaks_median'].notna().sum() >= 3:
        valid_np = valid.dropna(subset=['n_peaks_median'])
        r_np, p_np = stats.pearsonr(valid_np['n_peaks_median'], valid_np['cohen_d_deg2'])
        print(f"  Correlation (n_peaks_median vs Cohen d deg2):")
        print(f"    Pearson r = {r_np:.3f}, p = {p_np:.4f}")

# ============================================================
# PART 4: GENERATE GRADIENT FIGURE
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Color map by dataset
colors = {
    'Dortmund': '#e74c3c',  # red
    'LEMON': '#2ecc71',     # green
    'EEGMMIDB': '#3498db',  # blue
}
markers = {
    'Dortmund': 'o',
    'LEMON': 's',
    'EEGMMIDB': '^',
}

# Panel A: alpha_power vs Cohen's d (degree-2)
ax = axes[0]
for _, row in gradient_df.iterrows():
    ds = row['dataset']
    ax.scatter(row['alpha_power_median'], row['cohen_d_deg2'],
               c=colors.get(ds, 'gray'), marker=markers.get(ds, 'o'),
               s=100, zorder=5, edgecolors='black', linewidths=0.5)
    # Label
    label = row['condition'].replace('Dortmund_', 'D:').replace('Eyes', '').replace('_', '-')
    label = label.replace('LEMON_', 'L:').replace('EEGMMIDB', 'MMI')
    ax.annotate(label, (row['alpha_power_median'], row['cohen_d_deg2']),
                textcoords="offset points", xytext=(5, 5), fontsize=7)

# Fit line if enough points
valid = gradient_df.dropna(subset=['alpha_power_median', 'cohen_d_deg2'])
if len(valid) >= 3:
    slope, intercept, r, p, se = stats.linregress(valid['alpha_power_median'],
                                                     valid['cohen_d_deg2'])
    x_fit = np.linspace(valid['alpha_power_median'].min() * 0.9,
                         valid['alpha_power_median'].max() * 1.1, 100)
    ax.plot(x_fit, slope * x_fit + intercept, '--', color='gray', alpha=0.5)
    ax.text(0.05, 0.95, f'r = {r:.3f}, p = {p:.3f}', transform=ax.transAxes,
            fontsize=9, va='top')

ax.set_xlabel('Median Alpha Peak Power (a.u.)')
ax.set_ylabel("Cohen's d (degree-2, 4 positions)")
ax.set_title('A. Measurement Quality vs Alignment (degree-2)')
ax.axhline(0, color='gray', linestyle=':', alpha=0.3)

# Panel B: alpha_power vs d/null (degree-3)
ax = axes[1]
for _, row in gradient_df.iterrows():
    ds = row['dataset']
    ax.scatter(row['alpha_power_median'], row['d_over_null_deg3'],
               c=colors.get(ds, 'gray'), marker=markers.get(ds, 'o'),
               s=100, zorder=5, edgecolors='black', linewidths=0.5)
    label = row['condition'].replace('Dortmund_', 'D:').replace('Eyes', '').replace('_', '-')
    label = label.replace('LEMON_', 'L:').replace('EEGMMIDB', 'MMI')
    ax.annotate(label, (row['alpha_power_median'], row['d_over_null_deg3']),
                textcoords="offset points", xytext=(5, 5), fontsize=7)

valid = gradient_df.dropna(subset=['alpha_power_median', 'd_over_null_deg3'])
if len(valid) >= 3:
    slope, intercept, r, p, se = stats.linregress(valid['alpha_power_median'],
                                                     valid['d_over_null_deg3'])
    x_fit = np.linspace(valid['alpha_power_median'].min() * 0.9,
                         valid['alpha_power_median'].max() * 1.1, 100)
    ax.plot(x_fit, slope * x_fit + intercept, '--', color='gray', alpha=0.5)
    ax.text(0.05, 0.95, f'r = {r:.3f}, p = {p:.3f}', transform=ax.transAxes,
            fontsize=9, va='top')

ax.set_xlabel('Median Alpha Peak Power (a.u.)')
ax.set_ylabel('d̄ / d̄_null (degree-3, 6 positions)')
ax.set_title('B. Measurement Quality vs Alignment (degree-3)')
ax.axhline(1.0, color='gray', linestyle=':', alpha=0.3, label='null (d/null = 1)')

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
           markersize=10, label='Dortmund (N=608)'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#2ecc71',
           markersize=10, label='LEMON (N=202)'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='#3498db',
           markersize=10, label='EEGMMIDB (N=109)'),
]
axes[1].legend(handles=legend_elements, loc='lower right', fontsize=8)

plt.tight_layout()
out_fig = 'papers/images/lemon/measurement_quality_gradient.png'
os.makedirs(os.path.dirname(out_fig), exist_ok=True)
plt.savefig(out_fig, dpi=200, bbox_inches='tight')
plt.savefig(out_fig.replace('.png', '.pdf'), bbox_inches='tight')
print(f"\n  Saved: {out_fig}")

# ============================================================
# PART 5: THREE-DATASET CONVERGENCE TABLE
# ============================================================

print("\n" + "=" * 70)
print("PART 5: THREE-DATASET CONVERGENCE VERIFICATION")
print("=" * 70)

# Verify existing degree-3 CSVs match our computation
for ds_name, ds_path in [
    ('LEMON', 'exports_lemon/per_subject_overlap_trim_f07.83/dominant_peak/degree3_crossbase.csv'),
    ('EEGMMIDB', 'exports_eegmmidb/per_subject_overlap_trim_f07.83/dominant_peak/degree3_crossbase.csv'),
]:
    if os.path.exists(ds_path):
        existing = pd.read_csv(ds_path)
        phi_existing = existing[existing['base'] == 'phi'].iloc[0]
        our_key = f'{ds_name}_EO' if ds_name == 'LEMON' else ds_name
        if our_key in crossbase_results:
            our_phi = crossbase_results[our_key][crossbase_results[our_key]['base'] == 'phi'].iloc[0]
            match = abs(phi_existing['d_over_null'] - our_phi['d_over_null']) < 0.001
            print(f"  {ds_name:10s}: existing d/null={phi_existing['d_over_null']:.6f}, "
                  f"recomputed={our_phi['d_over_null']:.6f}, "
                  f"{'MATCH' if match else 'MISMATCH!'}")

# Three-dataset convergence table
print("\n  === THREE-DATASET DEGREE-3 CONVERGENCE ===")
key_conditions = ['EEGMMIDB', 'Dortmund_EyesClosed_pre', 'LEMON_EO']
for name in key_conditions:
    if name in crossbase_results:
        cb = crossbase_results[name]
        phi = cb[cb['base'] == 'phi'].iloc[0]
        d2 = deg2_results[name]
        print(f"  {name:30s}: d/null_deg3={phi['d_over_null']:.4f}, "
              f"mean_d_deg2={d2['mean_d']:.4f}, Cohen_d_deg2={d2['cohen_d']:.3f}, "
              f"N={d2['n']}")

# ============================================================
# PART 6: SAVE ALL FROZEN DATA
# ============================================================

out_dir2 = 'exports_dortmund_frozen'
os.makedirs(out_dir2, exist_ok=True)

# Save gradient data
gradient_df.to_csv(f'{out_dir2}/measurement_quality_gradient.csv', index=False)
print(f"\n  Saved: {out_dir2}/measurement_quality_gradient.csv")

# Save all degree-3 cross-base results
for name, cb in crossbase_results.items():
    safe_name = name.replace(' ', '_')
    cb.to_csv(f'{out_dir2}/degree3_crossbase_{safe_name}.csv', index=False)

# Save degree-2 results
deg2_df = pd.DataFrame(deg2_results.values())
deg2_df.to_csv(f'{out_dir2}/degree2_cohens_d_all.csv', index=False)
print(f"  Saved: {out_dir2}/degree2_cohens_d_all.csv")

# Print summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("\n  Key numbers for paper:")
for name in key_conditions:
    if name in crossbase_results:
        cb = crossbase_results[name]
        phi = cb[cb['base'] == 'phi'].iloc[0]
        d2 = deg2_results[name]
        short = name.replace('Dortmund_EyesClosed_pre', 'Dort-EC').replace('LEMON_EO', 'LEMON-EO')
        print(f"    {short:12s}: mean_d(4p)={d2['mean_d']:.4f}, "
              f"d/null(6p)={phi['d_over_null']:.4f}, "
              f"rank={int((cb['d_over_null'] < phi['d_over_null']).sum() + 1)}/9")

print("\n  Measurement-quality gradient:")
for _, row in gradient_df.sort_values('alpha_power_median').iterrows():
    short = row['condition'].replace('Dortmund_', 'D:').replace('Eyes', '').replace('_', '-')
    short = short.replace('LEMON_', 'L:').replace('EEGMMIDB', 'MMI')
    print(f"    {short:20s}: alpha_pow={row['alpha_power_median']:.3f}, "
          f"d(deg2)={row['cohen_d_deg2']:.3f}, "
          f"d/null(deg3)={row['d_over_null_deg3']:.4f}")

print("\n  GO/NO-GO: Check if points fall on a single curve in the figure.")
print(f"  Figure saved to: {out_fig}")
print("  Done.")

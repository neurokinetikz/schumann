#!/usr/bin/env python3
"""
Generate new figures for Paper 3 revision incorporating Dortmund replication.

Figures generated:
1. fig8_three_dataset_replication — d̄/d̄_null for all 9 bases across 3 datasets
2. fig9_measurement_sensitivity — (a) within-Dortmund degree-2 gradient, (b) degree-3 invariance
3. fig10_disambiguation_signflip — f₀ convergence vs IAF/2 divergence in Dortmund
4. fig11_crossdataset_theta — OT EO theta convergence from opposite directions
5. fig12_position_count — d vs number of positions (4, 6, 8)
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ============================================================
# CONFIGURATION
# ============================================================

PHI = (1 + np.sqrt(5)) / 2
F0 = 7.83

BASES = {
    'phi': PHI, 'sqrt2': np.sqrt(2), '3/2': 1.5, '2': 2.0,
    '1.7': 1.7, '1.8': 1.8, 'pi': np.pi, '1.4': 1.4, 'e': np.e,
}

BANDS = ['delta', 'theta', 'alpha', 'gamma']

# Nice base labels for plots
BASE_LABELS = {
    'phi': r'$\varphi$', 'sqrt2': r'$\sqrt{2}$', '3/2': '3/2',
    '2': '2', '1.7': '1.7', '1.8': '1.8',
    'pi': r'$\pi$', '1.4': '1.4', 'e': r'$e$',
}

OUT_DIR = 'papers/images/lemon'
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

# Load pre-computed degree-3 cross-base results from Step 0
frozen_dir = 'exports_dortmund_frozen'

# Load gradient data
gradient = pd.read_csv(f'{frozen_dir}/measurement_quality_gradient.csv')

# Load degree-3 cross-base for each dataset/condition
crossbase = {}
for f in os.listdir(frozen_dir):
    if f.startswith('degree3_crossbase_') and f.endswith('.csv'):
        name = f.replace('degree3_crossbase_', '').replace('.csv', '')
        crossbase[name] = pd.read_csv(os.path.join(frozen_dir, f))

print("Loaded crossbase results:")
for k in sorted(crossbase.keys()):
    print(f"  {k}: {len(crossbase[k])} bases")

print("\nGradient data:")
print(gradient[['condition', 'N', 'cohen_d_deg2', 'd_over_null_deg3', 'phi_rank']].to_string(index=False))

# ============================================================
# FIGURE 8: THREE-DATASET REPLICATION
# ============================================================

print("\n--- Figure 8: Three-dataset replication ---")

fig, ax = plt.subplots(figsize=(10, 5))

# Three key datasets
datasets_3 = {
    'EEGMMIDB': ('EEGMMIDB', '#3498db', 'N=109'),
    'Dortmund EC-pre': ('Dortmund_EyesClosed_pre', '#e74c3c', 'N=608'),
    'LEMON EO': ('LEMON_EO', '#2ecc71', 'N=202'),
}

# Get all bases sorted by mean d/null across datasets
base_order = []
for base_name in BASES:
    vals = []
    for ds_label, (ds_key, _, _) in datasets_3.items():
        if ds_key in crossbase:
            row = crossbase[ds_key][crossbase[ds_key]['base'] == base_name]
            if len(row) > 0:
                vals.append(row.iloc[0]['d_over_null'])
    if vals:
        base_order.append((base_name, np.mean(vals)))
base_order.sort(key=lambda x: x[1])

x = np.arange(len(base_order))
width = 0.25
offsets = [-width, 0, width]

for i, (ds_label, (ds_key, color, n_label)) in enumerate(datasets_3.items()):
    if ds_key in crossbase:
        cb = crossbase[ds_key]
        vals = []
        for base_name, _ in base_order:
            row = cb[cb['base'] == base_name]
            vals.append(row.iloc[0]['d_over_null'] if len(row) > 0 else np.nan)
        bars = ax.bar(x + offsets[i], vals, width, label=f'{ds_label} ({n_label})',
                      color=color, alpha=0.8, edgecolor='black', linewidth=0.5)

ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5, label='null (d/null = 1)')
ax.set_xlabel('Base', fontsize=12)
ax.set_ylabel(r'$\bar{d} / \bar{d}_{\mathrm{null}}$ (degree-3)', fontsize=12)
ax.set_title('Cross-Base Dominant-Peak Alignment: Three-Dataset Replication', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels([BASE_LABELS.get(b, b) for b, _ in base_order], fontsize=11)
ax.legend(fontsize=9, loc='upper left')
ax.set_ylim(0.85, 1.10)

# Highlight phi
phi_idx = [i for i, (b, _) in enumerate(base_order) if b == 'phi']
if phi_idx:
    ax.axvspan(phi_idx[0] - 0.45, phi_idx[0] + 0.45, alpha=0.1, color='gold')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig8_three_dataset_replication.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig8_three_dataset_replication.pdf', bbox_inches='tight')
plt.close()
print("  Saved fig8_three_dataset_replication")

# ============================================================
# FIGURE 9: MEASUREMENT SENSITIVITY vs ARCHITECTURAL INVARIANCE
# ============================================================

print("\n--- Figure 9: Measurement sensitivity ---")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Within-Dortmund degree-2 condition gradient
dort_conditions = [
    ('Dortmund_EyesOpen_pre', 'EO-pre', '#fee08b'),
    ('Dortmund_EyesOpen_post', 'EO-post', '#fdae61'),
    ('Dortmund_EyesClosed_post', 'EC-post', '#f46d43'),
    ('Dortmund_EyesClosed_pre', 'EC-pre', '#d73027'),
]

x_dort = np.arange(len(dort_conditions))
d_vals = []
for cond_key, label, color in dort_conditions:
    row = gradient[gradient['condition'] == cond_key]
    if len(row) > 0:
        d = row.iloc[0]['cohen_d_deg2']
        d_vals.append(d)
        ax1.bar(len(d_vals) - 1, d, color=color, edgecolor='black', linewidth=0.8,
                width=0.6)
    else:
        d_vals.append(0)

ax1.set_xticks(range(len(dort_conditions)))
ax1.set_xticklabels([l for _, l, _ in dort_conditions], fontsize=10)
ax1.set_ylabel("Cohen's d (degree-2, 4 positions)", fontsize=11)
ax1.set_title('A. Measurement Sensitivity\n(Within-Dortmund, N=608)', fontsize=12)
ax1.axhline(0, color='gray', linestyle=':', alpha=0.5)

# Add significance stars
for i, d in enumerate(d_vals):
    row = gradient[gradient['condition'] == dort_conditions[i][0]]
    if len(row) > 0:
        p = row.iloc[0]['p_value_deg2']
        if p < 0.001:
            ax1.text(i, d + 0.01, '***', ha='center', fontsize=10)
        elif p < 0.01:
            ax1.text(i, d + 0.01, '**', ha='center', fontsize=10)
        elif p < 0.05:
            ax1.text(i, d + 0.01, '*', ha='center', fontsize=10)
        else:
            ax1.text(i, d + 0.01, 'ns', ha='center', fontsize=9, color='gray')

# Panel B: Three-dataset degree-3 invariance
ds_bars = [
    ('EEGMMIDB', 'EEGMMIDB\n(N=109)', '#3498db'),
    ('Dortmund_EyesClosed_pre', 'Dortmund\n(N=608)', '#e74c3c'),
    ('LEMON_EO', 'LEMON\n(N=202)', '#2ecc71'),
]

x_ds = np.arange(len(ds_bars))
for i, (cond_key, label, color) in enumerate(ds_bars):
    row = gradient[gradient['condition'] == cond_key]
    if len(row) > 0:
        d_null = row.iloc[0]['d_over_null_deg3']
        ax2.bar(i, d_null, color=color, edgecolor='black', linewidth=0.8, width=0.5)
        ax2.text(i, d_null + 0.003, f'{d_null:.3f}', ha='center', fontsize=10, fontweight='bold')

ax2.axhline(1.0, color='gray', linestyle=':', alpha=0.5, label='null')
ax2.set_xticks(x_ds)
ax2.set_xticklabels([l for _, l, _ in ds_bars], fontsize=10)
ax2.set_ylabel(r'$\bar{d} / \bar{d}_{\mathrm{null}}$ (degree-3, $\varphi$)', fontsize=11)
ax2.set_title('B. Architectural Invariance\n(Three Datasets, degree-3)', fontsize=12)
ax2.set_ylim(0.85, 1.05)

# Add all 4 Dortmund conditions as small markers
dort_d3 = []
for cond_key, label, _ in dort_conditions:
    row = gradient[gradient['condition'] == cond_key]
    if len(row) > 0:
        dort_d3.append((label, row.iloc[0]['d_over_null_deg3']))

# Plot Dortmund conditions as small dots on the Dortmund bar
for label, val in dort_d3:
    ax2.plot(1, val, 'o', color='white', markersize=5, markeredgecolor='black',
             markeredgewidth=0.8, zorder=5)
    short = label.replace('-pre', 'p').replace('-post', 'P')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig9_measurement_sensitivity.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig9_measurement_sensitivity.pdf', bbox_inches='tight')
plt.close()
print("  Saved fig9_measurement_sensitivity")

# ============================================================
# FIGURE 10: DISAMBIGUATION SIGN-FLIP
# ============================================================

print("\n--- Figure 10: Disambiguation sign-flip ---")

# Load Dortmund OT data for convergence computation
dort_eo = pd.read_csv('/Volumes/T9/dortmund_data/lattice_results_ot/dortmund_ot_dominant_peaks_EyesOpen_pre.csv')
dort_ec = pd.read_csv('/Volumes/T9/dortmund_data/lattice_results_ot/dortmund_ot_dominant_peaks_EyesClosed_pre.csv')

# Merge on subject
merged = dort_eo.merge(dort_ec, on='subject', suffixes=('_eo', '_ec'))
merged = merged.dropna(subset=['theta_freq_eo', 'theta_freq_ec', 'alpha_freq_eo'])

# Compute distances
merged['d_f0_eo'] = np.abs(merged['theta_freq_eo'] - F0)
merged['d_f0_ec'] = np.abs(merged['theta_freq_ec'] - F0)
merged['d_iaf2_eo'] = np.abs(merged['theta_freq_eo'] - merged['alpha_freq_eo'] / 2)
merged['d_iaf2_ec'] = np.abs(merged['theta_freq_ec'] - merged['alpha_freq_ec'] / 2)

# Convergence: negative delta = converging
merged['delta_f0'] = merged['d_f0_ec'] - merged['d_f0_eo']  # negative = converging on f0
merged['delta_iaf2'] = merged['d_iaf2_ec'] - merged['d_iaf2_eo']  # negative = converging on IAF/2

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel A: f₀ convergence
ax = axes[0]
ax.hist(merged['delta_f0'], bins=50, color='#2166ac', alpha=0.7, edgecolor='black', linewidth=0.3)
mean_df0 = merged['delta_f0'].mean()
t_f0, p_f0 = stats.ttest_1samp(merged['delta_f0'], 0)
d_f0 = mean_df0 / merged['delta_f0'].std()
ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
ax.axvline(mean_df0, color='red', linewidth=2, label=f'mean={mean_df0:.3f}\nd={d_f0:.3f}')
ax.set_xlabel(r'$|\theta_{EC} - f_0| - |\theta_{EO} - f_0|$ (Hz)', fontsize=10)
ax.set_ylabel('Count', fontsize=10)
ax.set_title(r'A. Convergence on $f_0$ = 7.83 Hz', fontsize=12)
ax.legend(fontsize=9)

# Panel B: IAF/2 divergence
ax = axes[1]
ax.hist(merged['delta_iaf2'], bins=50, color='#b2182b', alpha=0.7, edgecolor='black', linewidth=0.3)
mean_diaf2 = merged['delta_iaf2'].mean()
t_iaf2, p_iaf2 = stats.ttest_1samp(merged['delta_iaf2'], 0)
d_iaf2 = mean_diaf2 / merged['delta_iaf2'].std()
ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
ax.axvline(mean_diaf2, color='red', linewidth=2, label=f'mean={mean_diaf2:+.3f}\nd={d_iaf2:+.3f}')
ax.set_xlabel(r'$|\theta_{EC} - \mathrm{IAF}/2| - |\theta_{EO} - \mathrm{IAF}/2|$ (Hz)', fontsize=10)
ax.set_ylabel('Count', fontsize=10)
ax.set_title('B. Divergence from IAF/2', fontsize=12)
ax.legend(fontsize=9)

# Panel C: Sign-flip comparison
ax = axes[2]
targets = [r'$f_0$', 'IAF/2']
effects = [d_f0, d_iaf2]
colors_bar = ['#2166ac', '#b2182b']
bars = ax.bar(targets, effects, color=colors_bar, edgecolor='black', linewidth=0.8, width=0.5)
ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax.set_ylabel("Cohen's d (EC - EO distance change)", fontsize=10)
ax.set_title('C. Sign Flip: Disambiguation', fontsize=12)
for bar, val in zip(bars, effects):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02 * np.sign(val),
            f'd = {val:+.3f}', ha='center', fontsize=11, fontweight='bold')

# Add annotation
ax.annotate('Converges', xy=(0, d_f0), xytext=(0.3, d_f0 - 0.15),
            fontsize=9, color='#2166ac', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#2166ac'))
ax.annotate('Diverges', xy=(1, d_iaf2), xytext=(0.7, d_iaf2 + 0.15),
            fontsize=9, color='#b2182b', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#b2182b'))

plt.suptitle('Dortmund (N=608): EC Theta Converges on $f_0$, Diverges from IAF/2',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig10_disambiguation_signflip.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig10_disambiguation_signflip.pdf', bbox_inches='tight')
plt.close()

print(f"  f₀ convergence: mean={mean_df0:.4f}, d={d_f0:.4f}, p={p_f0:.2e}")
print(f"  IAF/2 divergence: mean={mean_diaf2:.4f}, d={d_iaf2:.4f}, p={p_iaf2:.2e}")
print("  Saved fig10_disambiguation_signflip")

# ============================================================
# FIGURE 11: CROSS-DATASET THETA CONVERGENCE
# ============================================================

print("\n--- Figure 11: Cross-dataset theta convergence ---")

# Data from analysis
theta_data = {
    'LEMON STD EO': 7.6443,
    'Dort STD EO': 5.0000,
    'LEMON OT EO': 6.5685,
    'Dort OT EO': 6.5594,
}

fig, ax = plt.subplots(figsize=(8, 5))

# Plot arrows showing convergence
# LEMON: STD 7.64 → OT 6.57 (downward)
ax.annotate('', xy=(1.5, 6.5685), xytext=(0.5, 7.6443),
            arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=2.5))
ax.scatter([0.5], [7.6443], color='#2ecc71', s=120, zorder=5, edgecolors='black',
           linewidths=0.8, marker='s')
ax.scatter([1.5], [6.5685], color='#2ecc71', s=120, zorder=5, edgecolors='black',
           linewidths=0.8, marker='s')
ax.text(0.3, 7.6443, 'LEMON STD\n7.64 Hz', fontsize=9, ha='right', va='center', color='#2ecc71')

# Dortmund: STD 5.00 → OT 6.56 (upward)
ax.annotate('', xy=(1.5, 6.5594), xytext=(0.5, 5.0000),
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2.5))
ax.scatter([0.5], [5.0000], color='#e74c3c', s=120, zorder=5, edgecolors='black',
           linewidths=0.8, marker='o')
ax.scatter([1.5], [6.5594], color='#e74c3c', s=120, zorder=5, edgecolors='black',
           linewidths=0.8, marker='o')
ax.text(0.3, 5.0000, 'Dortmund STD\n5.00 Hz', fontsize=9, ha='right', va='center', color='#e74c3c')

# Convergence zone
ax.axhspan(6.55, 6.58, alpha=0.2, color='gold')
ax.text(1.8, 6.57, r'$\Delta$ = 0.009 Hz', fontsize=11, fontweight='bold',
        va='center', color='#8B6914')

# f₀ reference
ax.axhline(F0, color='gray', linestyle='--', alpha=0.5)
ax.text(2.0, F0, r'$f_0$ = 7.83 Hz', fontsize=9, va='bottom', color='gray')

# Noble_1 reference
noble1_freq = F0 * PHI**(1/PHI - 1)  # approximate noble_1 frequency
ax.axhline(6.56, color='orange', linestyle=':', alpha=0.3)

ax.set_xlim(-0.1, 2.5)
ax.set_ylim(4.0, 8.5)
ax.set_xticks([0.5, 1.5])
ax.set_xticklabels(['Standard\nExtraction', 'Overlap-Trim\nExtraction'], fontsize=11)
ax.set_ylabel('Population Median Theta Frequency (Hz)', fontsize=11)
ax.set_title('Cross-Dataset EO Theta Convergence Under Overlap-Trim', fontsize=13)

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#2ecc71',
           markersize=10, label='LEMON (N=202)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
           markersize=10, label='Dortmund (N=608)'),
]
ax.legend(handles=legend_elements, fontsize=9, loc='upper right')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig11_crossdataset_theta.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig11_crossdataset_theta.pdf', bbox_inches='tight')
plt.close()
print("  Saved fig11_crossdataset_theta")

# ============================================================
# FIGURE 12: POSITION-COUNT DECOMPOSITION
# ============================================================

print("\n--- Figure 12: Position-count decomposition ---")

# Need to compute degree-2 (4-pos) and degree-4 (8-pos) for comparison
# Degree-2 already computed. For degree-4, use 8 positions from phi_frequency_model.py
POSITIONS_DEG2 = {
    'boundary': 0.0, 'noble_2': 1.0/PHI**2, 'attractor': 0.5, 'noble_1': 1.0/PHI
}
POSITIONS_DEG3 = {
    'boundary': 0.0, 'noble_3': 1.0/PHI**3, 'noble_2': 1.0/PHI**2,
    'attractor': 0.5, 'noble_1': 1.0/PHI, 'inv_noble_3': 1.0 - 1.0/PHI**3
}
POSITIONS_DEG4 = {
    'boundary': 0.0, 'noble_4': 1.0/PHI**4, 'noble_3': 1.0/PHI**3,
    'noble_2': 1.0/PHI**2, 'attractor': 0.5, 'noble_1': 1.0/PHI,
    'inv_noble_3': 1.0 - 1.0/PHI**3, 'inv_noble_4': 1.0 - 1.0/PHI**4
}

def lattice_coordinate(f, f0, base):
    return (np.log(f / f0) / np.log(base)) % 1.0

def compute_cohens_d(df, positions):
    """Compute Cohen's d for a given position set."""
    pos_vals = sorted(positions.values())
    # Compute null via gaps
    gaps = [pos_vals[i+1] - pos_vals[i] for i in range(len(pos_vals)-1)]
    gaps.append(1.0 - pos_vals[-1] + pos_vals[0])  # wrap-around
    null_d = sum(g**2 for g in gaps) / 4.0

    subject_ds = []
    for _, row in df.iterrows():
        ds = []
        for band in BANDS:
            freq_col = f'{band}_freq'
            if freq_col in row.index and pd.notna(row[freq_col]):
                u = lattice_coordinate(row[freq_col], F0, PHI)
                pv = np.array(list(positions.values()))
                dists = np.abs(u - pv)
                dists = np.minimum(dists, 1.0 - dists)
                ds.append(np.min(dists))
        if len(ds) >= 3:
            subject_ds.append(np.mean(ds))
    subject_ds = np.array(subject_ds)
    mean_d = np.mean(subject_ds)
    std_d = np.std(subject_ds, ddof=1)
    cohen_d = (null_d - mean_d) / std_d
    return cohen_d, mean_d, null_d, len(subject_ds)

# Compute for 3 key datasets × 3 position sets
pos_sets = [
    ('Degree-2\n(4 pos)', POSITIONS_DEG2),
    ('Degree-3\n(6 pos)', POSITIONS_DEG3),
    ('Degree-4\n(8 pos)', POSITIONS_DEG4),
]

datasets_key = {
    'EEGMMIDB': pd.read_csv('exports_eegmmidb/per_subject_overlap_trim_f07.83/dominant_peak/per_subject_dominant_peaks.csv'),
    'Dortmund': pd.read_csv('/Volumes/T9/dortmund_data/lattice_results_ot/dortmund_ot_dominant_peaks_EyesClosed_pre.csv'),
    'LEMON': pd.read_csv('exports_lemon/per_subject_overlap_trim_f07.83/dominant_peak/per_subject_dominant_peaks.csv'),
}

fig, ax = plt.subplots(figsize=(8, 5))
colors_ds = {'EEGMMIDB': '#3498db', 'Dortmund': '#e74c3c', 'LEMON': '#2ecc71'}
markers_ds = {'EEGMMIDB': '^', 'Dortmund': 'o', 'LEMON': 's'}

for ds_name, df in datasets_key.items():
    n_pos_list = []
    d_list = []
    for pos_label, positions in pos_sets:
        d, mean_d, null_d, n = compute_cohens_d(df, positions)
        n_pos_list.append(len(positions))
        d_list.append(d)
        print(f"  {ds_name} {pos_label.split(chr(10))[0]}: d={d:.3f}, mean_d={mean_d:.4f}, null={null_d:.4f}")

    ax.plot(n_pos_list, d_list, '-', color=colors_ds[ds_name], linewidth=2, alpha=0.8)
    ax.scatter(n_pos_list, d_list, color=colors_ds[ds_name], s=100, zorder=5,
              marker=markers_ds[ds_name], edgecolors='black', linewidths=0.8,
              label=ds_name)

ax.set_xlabel('Number of Lattice Positions', fontsize=12)
ax.set_ylabel("Cohen's d (alignment vs null)", fontsize=12)
ax.set_title(r'Position-Count Decomposition: $\varphi$-Lattice Alignment', fontsize=13)
ax.set_xticks([4, 6, 8])
ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig12_position_count.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig12_position_count.pdf', bbox_inches='tight')
plt.close()
print("  Saved fig12_position_count")

print("\n=== All figures generated ===")

#!/usr/bin/env python3
"""
Figure 6: Structural Specificity of φⁿ Architecture
Nature Neuroscience composite figure

Panels:
  a) Bootstrap gap G(f₀) with 95% CI, SR range shaded
  b) SS trajectories by class — irrational (blue) vs rational (red)
  c) Cross-resolution robustness — G(f₀) at nperseg=256 and 640
  d) Band decomposition — per-band SS heatmap showing which bands drive the effect
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 7,
    'axes.linewidth': 0.5,
    'axes.labelsize': 7,
    'axes.titlesize': 8,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 2,
    'ytick.major.size': 2,
    'legend.fontsize': 6,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

PHI = (1 + np.sqrt(5)) / 2

ROOT = Path('/Users/neurokinetikz/Code/schumann')
OUT = ROOT / 'papers' / 'images'
OUT.mkdir(parents=True, exist_ok=True)

DATA_256 = ROOT / 'exports_peak_distribution' / 'eegmmidb_fooof_nperseg256'
DATA_640 = ROOT / 'exports_peak_distribution' / 'eegmmidb_fooof'

# ── Classification ───────────────────────────────────────────────────────────
IRRATIONAL_BASES = {'phi', 'sqrt2', 'e', 'pi'}
RATIONAL_BASES = {'1.4', '1.5', '1.7', '1.8', '2'}

# SR fundamental variation range
SR_LO, SR_HI = 7.0, 8.5

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")

# Bootstrap gap analysis
boot_256 = pd.read_csv(DATA_256 / 'bootstrap_gap_analysis.csv')
boot_640 = pd.read_csv(DATA_640 / 'bootstrap_gap_analysis.csv')

# f₀ sweep (per-base structural scores)
sweep_256 = pd.read_csv(DATA_256 / 'structural_specificity_f0_sweep.csv')
sweep_640 = pd.read_csv(DATA_640 / 'structural_specificity_f0_sweep.csv')

# Band decomposition
band_256 = pd.read_csv(DATA_256 / 'structural_specificity_by_band.csv')
band_640 = pd.read_csv(DATA_640 / 'structural_specificity_by_band.csv')

# Linear grid control
linear_256 = pd.read_csv(DATA_256 / 'linear_grid_control.csv')
linear_640 = pd.read_csv(DATA_640 / 'linear_grid_control.csv')

print(f"  nperseg=256: {len(boot_256)} f₀ values, {len(sweep_256)} sweep rows")
print(f"  nperseg=640: {len(boot_640)} f₀ values, {len(sweep_640)} sweep rows")

# ── Create figure ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.2, 7.5))
gs = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.35,
                       left=0.09, right=0.97, top=0.96, bottom=0.06)

# ── Panel a: Bootstrap gap G(f₀) with CI ────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])

f0 = boot_256['f0'].values
G = boot_256['G_observed'].values
ci_lo = boot_256['G_ci_lo'].values
ci_hi = boot_256['G_ci_hi'].values

# SR range shading
ax_a.axvspan(SR_LO, SR_HI, alpha=0.12, color='#F39C12', zorder=0,
             label='SR variation (7.0–8.5 Hz)')

# CI band
ax_a.fill_between(f0, ci_lo, ci_hi, alpha=0.25, color='#2980B9')

# Observed G
ax_a.plot(f0, G, 'o-', color='#2980B9', linewidth=1.2, markersize=2.5,
          label=r'G(f$_0$) = $\overline{SS}_{irr}$ − $\overline{SS}_{rat}$')

# Significance shading (CI excludes zero)
sig_mask = ci_lo > 0
if sig_mask.any():
    changes = np.diff(sig_mask.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1
    if sig_mask[0]:
        starts = np.r_[0, starts]
    if sig_mask[-1]:
        ends = np.r_[ends, len(sig_mask)]
    for s, e in zip(starts, ends):
        ax_a.axvspan(f0[s], f0[min(e, len(f0)-1)], alpha=0.08,
                     color='#27AE60', zorder=0)

# Zero line
ax_a.axhline(0, color='black', linewidth=0.4, alpha=0.4)

# Peak annotation
peak_idx = np.argmax(G)
ax_a.annotate(f'G = {G[peak_idx]:+.1f}\n'
              f'f$_0$ = {f0[peak_idx]:.2f} Hz',
              xy=(f0[peak_idx], G[peak_idx]),
              xytext=(f0[peak_idx] + 0.6, G[peak_idx] - 5),
              fontsize=5.5, fontweight='bold',
              arrowprops=dict(arrowstyle='->', color='#2980B9',
                              linewidth=0.6),
              bbox=dict(boxstyle='round,pad=0.2', fc='#EBF5FB',
                        ec='#2980B9', linewidth=0.4))

# Exact permutation p on twin axis
if 'exact_p' in boot_256.columns:
    ax_a2 = ax_a.twinx()
    ax_a2.plot(f0, boot_256['exact_p'].values, 's', color='#E74C3C',
               alpha=0.4, markersize=1.5)
    ax_a2.set_ylabel('Exact perm. p', color='#E74C3C', fontsize=6)
    ax_a2.tick_params(axis='y', labelcolor='#E74C3C', labelsize=5)
    ax_a2.set_ylim(0, 0.6)
    ax_a2.axhline(0.05, color='#E74C3C', linestyle=':', linewidth=0.4,
                  alpha=0.5)

ax_a.set_ylabel('Gap G = $\\overline{SS}_{irr}$ − $\\overline{SS}_{rat}$')
ax_a.set_xlabel('Anchor frequency f$_0$ (Hz)')
ax_a.legend(loc='upper left', frameon=True, fancybox=False,
            edgecolor='#CCCCCC', fontsize=5)
ax_a.grid(True, alpha=0.15, linewidth=0.3)
ax_a.set_title('Bootstrap gap: irrational vs rational', fontsize=7)
ax_a.text(-0.15, 1.05, 'a', transform=ax_a.transAxes,
          fontsize=10, fontweight='bold', va='top')

# ── Panel b: SS trajectories by class ────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])

# SR range
ax_b.axvspan(SR_LO, SR_HI, alpha=0.12, color='#F39C12', zorder=0)

# Plot each base trajectory
f0_vals = sorted(sweep_256['f0'].unique())

for _, row in sweep_256.groupby('base_name'):
    base_key = row.iloc[0]['base_name']
    base_label = row.iloc[0]['base_label']
    sub = row.sort_values('f0')

    is_irr = base_key in IRRATIONAL_BASES
    color = '#2980B9' if is_irr else '#E74C3C'
    lw = 1.8 if base_key == 'phi' else (0.8 if is_irr else 0.6)
    alpha = 1.0 if base_key == 'phi' else (0.7 if is_irr else 0.4)
    ls = '-' if is_irr else '--'
    zorder = 5 if base_key == 'phi' else (3 if is_irr else 2)

    ax_b.plot(sub['f0'], sub['structural_score'],
              linewidth=lw, linestyle=ls, color=color,
              alpha=alpha, zorder=zorder)

    # Label at end of line
    last = sub.iloc[-1]
    if base_key in ('phi', 'e', 'pi', '1.5', '2'):
        ax_b.text(last['f0'] + 0.05, last['structural_score'],
                  base_label, fontsize=4.5, color=color, alpha=0.8,
                  va='center')

# Class means as diamonds
for f0v in f0_vals[::3]:  # every 3rd point
    sub = sweep_256[sweep_256['f0'] == f0v]
    irr_mean = sub[sub['base_name'].isin(IRRATIONAL_BASES)]['structural_score'].mean()
    rat_mean = sub[sub['base_name'].isin(RATIONAL_BASES)]['structural_score'].mean()
    ax_b.plot(f0v, irr_mean, 'D', color='#1A5276', markersize=3, zorder=6)
    ax_b.plot(f0v, rat_mean, 'D', color='#922B21', markersize=3, zorder=6)

ax_b.axhline(0, color='black', linewidth=0.4, alpha=0.4)
ax_b.set_xlabel('Anchor frequency f$_0$ (Hz)')
ax_b.set_ylabel('Structural Score (SS)')
ax_b.set_title('SS trajectories: 9 exponential bases', fontsize=7)
ax_b.grid(True, alpha=0.15, linewidth=0.3)

# Custom legend
from matplotlib.lines import Line2D
leg = [
    Line2D([0], [0], color='#2980B9', linewidth=1.8, label='φ (irrational)'),
    Line2D([0], [0], color='#2980B9', linewidth=0.8, alpha=0.7,
           label='Other irrationals'),
    Line2D([0], [0], color='#E74C3C', linewidth=0.6, linestyle='--',
           alpha=0.5, label='Rationals'),
    Line2D([0], [0], marker='D', color='#1A5276', linewidth=0,
           markersize=3, label='Irrational mean'),
    Line2D([0], [0], marker='D', color='#922B21', linewidth=0,
           markersize=3, label='Rational mean'),
]
ax_b.legend(handles=leg, loc='best', frameon=True, fancybox=False,
            edgecolor='#CCCCCC', fontsize=5)
ax_b.text(-0.15, 1.05, 'b', transform=ax_b.transAxes,
          fontsize=10, fontweight='bold', va='top')

# ── Panel c: Cross-resolution overlay ────────────────────────────────────────
ax_c = fig.add_subplot(gs[1, 0])

# SR range
ax_c.axvspan(SR_LO, SR_HI, alpha=0.12, color='#F39C12', zorder=0,
             label='SR variation')

# nperseg=256 G(f₀) with CI
f0_256 = boot_256['f0'].values
G_256 = boot_256['G_observed'].values
ci_lo_256 = boot_256['G_ci_lo'].values
ci_hi_256 = boot_256['G_ci_hi'].values

ax_c.fill_between(f0_256, ci_lo_256, ci_hi_256, alpha=0.15, color='#2980B9')
ax_c.plot(f0_256, G_256, 'o-', color='#2980B9', linewidth=1.0,
          markersize=2, label='nperseg = 256 (1.42M peaks)')

# nperseg=640 G(f₀) with CI
f0_640 = boot_640['f0'].values
G_640 = boot_640['G_observed'].values
ci_lo_640 = boot_640['G_ci_lo'].values
ci_hi_640 = boot_640['G_ci_hi'].values

ax_c.fill_between(f0_640, ci_lo_640, ci_hi_640, alpha=0.15, color='#E67E22')
ax_c.plot(f0_640, G_640, 's-', color='#E67E22', linewidth=1.0,
          markersize=2, label='nperseg = 640 (1.86M peaks)')

# Linear grid range (for reference)
lin_sweep_256 = pd.read_csv(DATA_256 / 'linear_grid_f0_sweep.csv')
lin_pivot = lin_sweep_256.pivot_table(index='f0', columns='label',
                                       values='structural_score')
lin_max = lin_pivot.max(axis=1).values
lin_min = lin_pivot.min(axis=1).values
lin_f0 = lin_pivot.index.values
ax_c.fill_between(lin_f0, lin_min, lin_max, alpha=0.15, color='gray',
                  label='Linear grid range')

ax_c.axhline(0, color='black', linewidth=0.4, alpha=0.4)
ax_c.set_xlabel('Anchor frequency f$_0$ (Hz)')
ax_c.set_ylabel('Gap G')
ax_c.set_title('Cross-resolution robustness', fontsize=7)
ax_c.legend(loc='upper left', frameon=True, fancybox=False,
            edgecolor='#CCCCCC', fontsize=5)
ax_c.grid(True, alpha=0.15, linewidth=0.3)
ax_c.text(-0.15, 1.05, 'c', transform=ax_c.transAxes,
          fontsize=10, fontweight='bold', va='top')

# ── Panel d: Band decomposition heatmap ──────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 1])

# Use nperseg=256 data (more peaks across more bands)
band_data = band_256.copy()

# Pivot: base × band
pivot = band_data.pivot_table(index='base_label', columns='band',
                               values='structural_score')

# Band order (frequency)
band_order = ['theta', 'alpha', 'beta_low', 'beta_high', 'gamma']
band_order = [b for b in band_order if b in pivot.columns]
pivot = pivot[band_order]

# Base order: sort by aggregate SS (highest on top)
# Get aggregate SS from the enrichment file
agg_256 = pd.read_csv(DATA_256 / 'structural_specificity_enrichment.csv')
base_order = agg_256.sort_values('structural_score', ascending=False)['base_label'].tolist()
base_order = [b for b in base_order if b in pivot.index]
pivot = pivot.loc[base_order]

# Plot heatmap
vmax = np.nanpercentile(np.abs(pivot.values), 95)
im = ax_d.imshow(pivot.values, aspect='auto', cmap='RdBu_r',
                 vmin=-vmax, vmax=vmax, interpolation='nearest')

# Labels
band_display = {
    'theta': 'θ', 'alpha': 'α', 'beta_low': 'βL',
    'beta_high': 'βH', 'gamma': 'γ'
}
ax_d.set_xticks(range(len(band_order)))
ax_d.set_xticklabels([band_display.get(b, b) for b in band_order], fontsize=6)
ax_d.set_yticks(range(len(base_order)))
ax_d.set_yticklabels(base_order, fontsize=6)

# Color values in cells
for i in range(len(base_order)):
    for j in range(len(band_order)):
        val = pivot.values[i, j]
        if np.isfinite(val):
            color = 'white' if abs(val) > vmax * 0.6 else 'black'
            ax_d.text(j, i, f'{val:.0f}', ha='center', va='center',
                      fontsize=4.5, color=color, fontweight='bold')

# Mark irrational bases
for i, label in enumerate(base_order):
    if label in ('φ', '√2', 'e', 'π'):
        ax_d.text(-0.7, i, '●', ha='center', va='center',
                  fontsize=5, color='#2980B9')

# Colorbar
cbar = fig.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04)
cbar.set_label('SS', fontsize=6)
cbar.ax.tick_params(labelsize=5)

ax_d.set_xlabel('Frequency band')
ax_d.set_title('Band decomposition of structural score', fontsize=7)
ax_d.text(-0.20, 1.05, 'd', transform=ax_d.transAxes,
          fontsize=10, fontweight='bold', va='top')

# ── Save ─────────────────────────────────────────────────────────────────────
fig.savefig(OUT / 'nature_fig6.png', dpi=300, facecolor='white')
fig.savefig(OUT / 'nature_fig6.pdf', facecolor='white')
plt.close()
print(f"Figure 6 saved to {OUT / 'nature_fig6.png'}")
print(f"Figure 6 saved to {OUT / 'nature_fig6.pdf'}")

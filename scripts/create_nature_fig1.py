#!/usr/bin/env python3
"""
Figure 1: SIE Discovery and Characterization
Nature Neuroscience composite figure

Panels:
  a) Exemplar SIE spectrogram + piano-roll (from source image)
  b) Harmonic ratio precision: measured vs phi^n predictions
  c) Independence-convergence: SR1 vs SR3 scatter + ratio histogram
  d) Peak-based null control (from source image)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.image import imread
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
IMG = ROOT / 'papers' / 'images'
OUT = ROOT / 'papers' / 'images'

# ── Load SIE event data ─────────────────────────────────────────────────────
df = pd.read_csv(ROOT / 'csv' / 'SIE-PAPER-FINAL.csv')
# Clean column names (some have leading spaces)
df.columns = [c.strip() for c in df.columns]

# Filter to valid triplets (sr1, sr3, sr5 all present)
mask = df['sr1'].notna() & df['sr3'].notna() & df['sr5'].notna()
df_valid = df[mask].copy()

# ── Create figure ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.2, 6.5))  # ~183mm wide, tall enough for 4 panels

# Use gridspec for layout: 2 rows x 2 cols
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3,
                       left=0.08, right=0.97, top=0.95, bottom=0.06)

# ── Panel a: Exemplar SIE ───────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
img_sie = imread(str(IMG / 'download (5).png'))
# Crop to top half (spectrogram + piano-roll)
h = img_sie.shape[0]
img_top = img_sie[:int(h * 0.37), :]
ax_a.imshow(img_top)
ax_a.set_xticks([])
ax_a.set_yticks([])
for spine in ax_a.spines.values():
    spine.set_visible(False)
ax_a.text(-0.05, 1.05, 'a', transform=ax_a.transAxes,
          fontsize=10, fontweight='bold', va='top')

# ── Panel b: Harmonic ratio precision ────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])

# Compute ratios from data
sr3_sr1 = df_valid['sr3'] / df_valid['sr1']
sr5_sr1 = df_valid['sr5'] / df_valid['sr1']
sr5_sr3 = df_valid['sr5'] / df_valid['sr3']

ratios_measured = [sr3_sr1.mean(), sr5_sr1.mean(), sr5_sr3.mean()]
ratios_predicted = [PHI**2, PHI**3, PHI]
ratio_labels = ['SR3/SR1\n(vs φ²)', 'SR5/SR1\n(vs φ³)', 'SR5/SR3\n(vs φ)']
errors_pct = [abs(m - p) / p * 100 for m, p in zip(ratios_measured, ratios_predicted)]

x = np.arange(len(ratio_labels))
w = 0.35

bars_m = ax_b.bar(x - w/2, ratios_measured, w, color='#E74C3C', alpha=0.85,
                  label='Measured', edgecolor='white', linewidth=0.5)
bars_p = ax_b.bar(x + w/2, ratios_predicted, w, color='#3498DB', alpha=0.85,
                  label='Predicted', edgecolor='white', linewidth=0.5)

# Add error annotations
for i, (m, p, e) in enumerate(zip(ratios_measured, ratios_predicted, errors_pct)):
    ax_b.annotate(f'{e:.2f}%', xy=(x[i], max(m, p) + 0.05),
                  ha='center', va='bottom', fontsize=5.5, color='#555555')

ax_b.set_xticks(x)
ax_b.set_xticklabels(ratio_labels)
ax_b.set_ylabel('Ratio value')
ax_b.set_title('Harmonic ratio precision', fontsize=8)
ax_b.legend(loc='upper left', frameon=False, fontsize=6)
ax_b.set_ylim(0, max(ratios_measured + ratios_predicted) * 1.15)
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)
ax_b.text(-0.15, 1.05, 'b', transform=ax_b.transAxes,
          fontsize=10, fontweight='bold', va='top')

# ── Panel c: Independence-convergence ────────────────────────────────────────
# Split into two sub-panels: scatter (left) and ratio histogram (right)
gs_c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 0],
                                        wspace=0.4)

# c-left: SR1 vs SR3 scatter
ax_c1 = fig.add_subplot(gs_c[0, 0])
ax_c1.scatter(df_valid['sr1'], df_valid['sr3'], s=3, alpha=0.3,
              color='#7F8C8D', edgecolors='none', rasterized=True)
# Compute correlation
from scipy import stats
r, p = stats.pearsonr(df_valid['sr1'], df_valid['sr3'])
ax_c1.set_xlabel('SR1 frequency (Hz)')
ax_c1.set_ylabel('SR3 frequency (Hz)')
ax_c1.set_title(f'r = {r:+.3f}, p = {p:.2f}', fontsize=6.5)
ax_c1.spines['top'].set_visible(False)
ax_c1.spines['right'].set_visible(False)
ax_c1.text(-0.2, 1.12, 'c', transform=ax_c1.transAxes,
           fontsize=10, fontweight='bold', va='top')

# c-right: SR3/SR1 ratio histogram
ax_c2 = fig.add_subplot(gs_c[0, 1])
ax_c2.hist(sr3_sr1, bins=40, color='#E74C3C', alpha=0.7, edgecolor='white',
           linewidth=0.3, density=True)
ax_c2.axvline(PHI**2, color='#3498DB', linestyle='--', linewidth=1.2,
              label=f'φ² = {PHI**2:.3f}')
ax_c2.set_xlabel('SR3 / SR1 ratio')
ax_c2.set_ylabel('Density')
ax_c2.set_title(f'Mean = {sr3_sr1.mean():.3f}', fontsize=6.5)
ax_c2.legend(frameon=False, fontsize=5.5)
ax_c2.spines['top'].set_visible(False)
ax_c2.spines['right'].set_visible(False)

# ── Panel d: Peak-based null control ─────────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 1])
img_null = imread(str(IMG / 'nc7_peak_based_results.png'))
# Crop to top-right quadrant (box comparison)
h, w_img = img_null.shape[:2]
img_box = img_null[:int(h * 0.52), int(w_img * 0.5):]
ax_d.imshow(img_box)
ax_d.set_xticks([])
ax_d.set_yticks([])
for spine in ax_d.spines.values():
    spine.set_visible(False)
ax_d.text(-0.05, 1.05, 'd', transform=ax_d.transAxes,
          fontsize=10, fontweight='bold', va='top')

# ── Save ─────────────────────────────────────────────────────────────────────
fig.savefig(OUT / 'nature_fig1.png', dpi=300, facecolor='white')
fig.savefig(OUT / 'nature_fig1.pdf', facecolor='white')
plt.close()
print(f"Figure 1 saved to {OUT / 'nature_fig1.png'}")

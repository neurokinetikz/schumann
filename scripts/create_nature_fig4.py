#!/usr/bin/env python3
"""
Figure 4: Multi-Channel GED Convergence
Nature Neuroscience composite figure

Panels:
  a) GED peak distribution on log-phi scale (from source image)
  b) GED position-type enrichment bar chart
  c) Methodological comparison: FOOOF vs GED
"""

import numpy as np
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

ROOT = Path('/Users/neurokinetikz/Code/schumann')
IMG = ROOT / 'papers' / 'images'
OUT = ROOT / 'papers' / 'images'

# ── Create figure ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.2, 5.5))

gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30,
                       left=0.07, right=0.97, top=0.96, bottom=0.08,
                       height_ratios=[1.2, 1])

# ── Panel a: GED peak distribution (source image, full width) ────────────────
ax_a = fig.add_subplot(gs[0, :])
img_ged = imread(str(IMG / 'aggregate_modes_logphi_f0_760.png'))
ax_a.imshow(img_ged)
ax_a.set_xticks([])
ax_a.set_yticks([])
for spine in ax_a.spines.values():
    spine.set_visible(False)
ax_a.text(-0.02, 1.03, 'a', transform=ax_a.transAxes,
          fontsize=10, fontweight='bold', va='top')

# ── Panel b: GED enrichment bar chart ────────────────────────────────────────
ax_b = fig.add_subplot(gs[1, 0])

positions = ['Boundary\n(n=integer)', '2° Noble\n(n+0.382)', 'Attractor\n(n+0.5)', '1° Noble\n(n+0.618)']
enrichment_ged = [-9.2, 2.0, 14.8, 27.5]
ci_low_ged = [-9.7, 1.2, 14.3, 27.0]
ci_high_ged = [-8.8, 2.8, 15.3, 28.0]
colors = ['#E8963E', '#B5B843', '#5B9BD5', '#70C270']

yerr_low = [e - cl for e, cl in zip(enrichment_ged, ci_low_ged)]
yerr_high = [ch - e for e, ch in zip(enrichment_ged, ci_high_ged)]

bars = ax_b.bar(range(4), enrichment_ged, color=colors, edgecolor='white',
                linewidth=0.5, alpha=0.9,
                yerr=[yerr_low, yerr_high], capsize=2,
                error_kw={'linewidth': 0.8, 'color': '#333333'})

ax_b.axhline(0, color='#AAAAAA', linewidth=0.5, linestyle='--')
for i, (e, cl, ch) in enumerate(zip(enrichment_ged, ci_low_ged, ci_high_ged)):
    y_pos = e + 1.8 if e > 0 else e - 3.0
    ax_b.text(i, y_pos, f'{e:+.1f}%\n[{cl:.1f}, {ch:.1f}]',
              ha='center', va='bottom' if e > 0 else 'top', fontsize=5)

ax_b.set_xticks(range(4))
ax_b.set_xticklabels(positions, fontsize=5.5)
ax_b.set_ylabel('Enrichment (%)')
ax_b.set_title('GED: position-type enrichment\n(1,584,561 peaks, 3,261 sessions)', fontsize=7)
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)
ax_b.text(-0.15, 1.05, 'b', transform=ax_b.transAxes,
          fontsize=10, fontweight='bold', va='top')

# ── Panel c: Methodological comparison ───────────────────────────────────────
ax_c = fig.add_subplot(gs[1, 1])

# Three methods x four position types
methods = ['FOOOF\n(primary)', 'FOOOF\n(replication)', 'GED']
boundary_vals = [-18.0, -20.5, -9.2]
attractor_vals = [21.0, 22.1, 14.8]
noble_vals = [39.0, 26.3, 27.5]

x = np.arange(len(methods))
w = 0.25

bars1 = ax_c.bar(x - w, boundary_vals, w, color='#E8963E', alpha=0.9,
                 label='Boundary', edgecolor='white', linewidth=0.3)
bars2 = ax_c.bar(x, attractor_vals, w, color='#5B9BD5', alpha=0.9,
                 label='Attractor', edgecolor='white', linewidth=0.3)
bars3 = ax_c.bar(x + w, noble_vals, w, color='#70C270', alpha=0.9,
                 label='1° Noble', edgecolor='white', linewidth=0.3)

ax_c.axhline(0, color='#AAAAAA', linewidth=0.5, linestyle='--')
ax_c.set_xticks(x)
ax_c.set_xticklabels(methods, fontsize=6)
ax_c.set_ylabel('Enrichment (%)')
ax_c.set_title('Methodological convergence', fontsize=7)
ax_c.legend(frameon=False, fontsize=5.5, ncol=3, loc='upper center',
            bbox_to_anchor=(0.5, 0.98))
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)

# Add tau annotations
for i in range(3):
    ax_c.text(x[i], max(noble_vals[i], attractor_vals[i]) + 3,
              'τ = 1.0', ha='center', fontsize=5, color='#555555',
              fontstyle='italic')

ax_c.text(-0.15, 1.05, 'c', transform=ax_c.transAxes,
          fontsize=10, fontweight='bold', va='top')

# ── Save ─────────────────────────────────────────────────────────────────────
fig.savefig(OUT / 'nature_fig4.png', dpi=300, facecolor='white')
fig.savefig(OUT / 'nature_fig4.pdf', facecolor='white')
plt.close()
print(f"Figure 4 saved to {OUT / 'nature_fig4.png'}")

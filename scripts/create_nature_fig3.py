#!/usr/bin/env python3
"""
Figure 3: Band-Stratified Analysis and Gamma Dominance
Nature Neuroscience composite figure

Panels:
  a) Band-stratified lattice histograms (from source image)
  b) Band-specific 1° noble enrichment summary
  c) Theta vs gamma position preferences
  d) Permutation test (uniform frequency test)
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
F0 = 7.60

ROOT = Path('/Users/neurokinetikz/Code/schumann')
IMG = ROOT / 'papers' / 'images'
OUT = ROOT / 'papers' / 'images'

# ── Create figure ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.2, 7.0))

gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30,
                       left=0.08, right=0.97, top=0.96, bottom=0.06)

# ── Panel a: Band-stratified lattice histograms (source image) ───────────────
ax_a = fig.add_subplot(gs[0, :])  # Span full width
img_band = imread(str(IMG / 'phi_band_stratified_analysis.png'))
ax_a.imshow(img_band)
ax_a.set_xticks([])
ax_a.set_yticks([])
for spine in ax_a.spines.values():
    spine.set_visible(False)
ax_a.text(-0.02, 1.03, 'a', transform=ax_a.transAxes,
          fontsize=10, fontweight='bold', va='top')

# ── Panel b: Band-specific 1° noble enrichment ──────────────────────────────
ax_b = fig.add_subplot(gs[1, 0])

bands = ['Delta', 'Theta', 'Alpha', 'Low β', 'High β', 'Gamma']
noble_enrichment = [-12.1, 2.2, 4.2, 5.6, 8.8, 144.8]
band_colors = ['#8E44AD', '#2980B9', '#27AE60', '#F39C12', '#E67E22', '#E74C3C']

bars = ax_b.bar(range(len(bands)), noble_enrichment, color=band_colors,
                edgecolor='white', linewidth=0.5, alpha=0.9)

ax_b.axhline(0, color='#AAAAAA', linewidth=0.5, linestyle='--')

# Annotate values
for i, v in enumerate(noble_enrichment):
    offset = 5 if v > 0 else -8
    ax_b.text(i, v + offset, f'{v:+.1f}%', ha='center', fontsize=5.5,
              fontweight='bold' if abs(v) > 50 else 'normal')

ax_b.set_xticks(range(len(bands)))
ax_b.set_xticklabels(bands, fontsize=6)
ax_b.set_ylabel('1° Noble enrichment (%)')
ax_b.set_title('Band-specific φⁿ adherence', fontsize=7)
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)
ax_b.text(-0.15, 1.05, 'b', transform=ax_b.transAxes,
          fontsize=10, fontweight='bold', va='top')

# ── Panel c: Theta vs Gamma position preferences ────────────────────────────
ax_c = fig.add_subplot(gs[1, 1])

# Position types on x-axis
pos_labels = ['4° Inv.', '3° Inv.', '2° Inv.', 'Boundary', '2° Noble', 'Attractor', '1° Noble']
theta_enrich = [47.2, 24.0, 15.0, -5.0, -8.0, 3.0, 2.2]
gamma_enrich = [-99.7, -85.0, -60.0, -15.0, 20.0, 50.0, 144.0]

x = np.arange(len(pos_labels))
w = 0.35

bars_theta = ax_c.bar(x - w/2, theta_enrich, w, color='#2980B9', alpha=0.85,
                      label='Theta', edgecolor='white', linewidth=0.3)
bars_gamma = ax_c.bar(x + w/2, gamma_enrich, w, color='#E74C3C', alpha=0.85,
                      label='Gamma', edgecolor='white', linewidth=0.3)

ax_c.axhline(0, color='#AAAAAA', linewidth=0.5, linestyle='--')
ax_c.set_xticks(x)
ax_c.set_xticklabels(pos_labels, fontsize=5, rotation=30, ha='right')
ax_c.set_ylabel('Enrichment (%)')
ax_c.set_title('Theta-gamma dissociation', fontsize=7)
ax_c.legend(frameon=False, fontsize=6, loc='upper left')
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)
ax_c.text(-0.15, 1.05, 'c', transform=ax_c.transAxes,
          fontsize=10, fontweight='bold', va='top')

# ── Save ─────────────────────────────────────────────────────────────────────
fig.savefig(OUT / 'nature_fig3.png', dpi=300, facecolor='white')
fig.savefig(OUT / 'nature_fig3.pdf', facecolor='white')
plt.close()
print(f"Figure 3 saved to {OUT / 'nature_fig3.png'}")

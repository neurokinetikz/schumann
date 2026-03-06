#!/usr/bin/env python3
"""
Figure 2: Continuous Spectral Architecture
Nature Neuroscience composite figure

Panels:
  a) Peak distribution, primary dataset (244,955 peaks)
  b) Position-type enrichment, primary
  c) Peak distribution, EEGEmotions-27 replication (612,990 peaks)
  d) Position-type enrichment, replication
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

PHI = (1 + np.sqrt(5)) / 2
F0 = 7.60

ROOT = Path('/Users/neurokinetikz/Code/schumann')
IMG = ROOT / 'papers' / 'images'
OUT = ROOT / 'papers' / 'images'

# ── Create figure ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.2, 7.0))

gs = gridspec.GridSpec(2, 2, hspace=0.30, wspace=0.30,
                       left=0.06, right=0.97, top=0.96, bottom=0.05)

# ── Panel a: Primary peak distribution ───────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
try:
    import pandas as pd
    peaks = pd.read_csv(ROOT / 'csv' / 'golden_ratio_peaks_ALL.csv')
    freqs = peaks['freq'].values

    # Plot histogram
    bins = np.arange(1, 50.5, 0.5)
    ax_a.hist(freqs, bins=bins, color='#5B9BD5', edgecolor='white',
              linewidth=0.2, alpha=0.85)

    # Add phi^n vertical lines
    for n in range(-1, 5):
        f = F0 * PHI**n
        if 1 < f < 50:
            ax_a.axvline(f, color='#E74C3C', linewidth=0.6, alpha=0.7,
                        linestyle='-')
    # Add attractor lines (half-integer)
    for n in range(0, 4):
        f = F0 * PHI**(n + 0.5)
        if 1 < f < 50:
            ax_a.axvline(f, color='#F4D03F', linewidth=0.6, alpha=0.7,
                        linestyle='--')
    # Noble positions
    for n in range(0, 4):
        f = F0 * PHI**(n + 0.618)
        if 1 < f < 50:
            ax_a.axvline(f, color='#27AE60', linewidth=0.6, alpha=0.7,
                        linestyle='-.')

    ax_a.set_xlabel('Frequency (Hz)')
    ax_a.set_ylabel('Peak count')
    ax_a.set_title(f'Primary dataset ({len(freqs):,} peaks, 968 sessions)', fontsize=7)
    ax_a.set_xlim(1, 50)

except Exception as e:
    # Fallback to source image
    img = imread(str(IMG / 'golden_ratio_peaks_ALL.png'))
    ax_a.imshow(img)
    ax_a.set_xticks([]); ax_a.set_yticks([])
    for s in ax_a.spines.values(): s.set_visible(False)

ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)
ax_a.text(-0.12, 1.05, 'a', transform=ax_a.transAxes,
          fontsize=10, fontweight='bold', va='top')

# ── Panel b: Primary enrichment bar chart ────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])

positions = ['Boundary\n(n=integer)', '2° Noble\n(n+0.382)', 'Attractor\n(n+0.5)', '1° Noble\n(n+0.618)']
enrichment_primary = [-18.0, 1.9, 21.0, 39.0]
ci_low_primary = [-19.1, 0.8, 19.7, 37.5]
ci_high_primary = [-16.9, 3.2, 22.2, 40.3]
colors = ['#E8963E', '#B5B843', '#5B9BD5', '#70C270']

yerr_low = [e - cl for e, cl in zip(enrichment_primary, ci_low_primary)]
yerr_high = [ch - e for e, ch in zip(enrichment_primary, ci_high_primary)]

bars = ax_b.bar(range(4), enrichment_primary, color=colors, edgecolor='white',
                linewidth=0.5, alpha=0.9,
                yerr=[yerr_low, yerr_high], capsize=2,
                error_kw={'linewidth': 0.8, 'color': '#333333'})

ax_b.axhline(0, color='#AAAAAA', linewidth=0.5, linestyle='--')
for i, (e, cl, ch) in enumerate(zip(enrichment_primary, ci_low_primary, ci_high_primary)):
    y_pos = e + 2.5 if e > 0 else e - 4.5
    ax_b.text(i, y_pos, f'{e:+.1f}%\n[{cl:.1f}, {ch:.1f}]',
              ha='center', va='bottom' if e > 0 else 'top', fontsize=5)

ax_b.set_xticks(range(4))
ax_b.set_xticklabels(positions, fontsize=5.5)
ax_b.set_ylabel('Enrichment (%)')
ax_b.set_title('Primary: position-type enrichment', fontsize=7)
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)
ax_b.text(-0.15, 1.05, 'b', transform=ax_b.transAxes,
          fontsize=10, fontweight='bold', va='top')

# ── Panel c: Replication peak distribution ───────────────────────────────────
ax_c = fig.add_subplot(gs[1, 0])
try:
    peaks_emo = pd.read_csv(ROOT / 'golden_ratio_peaks_EMOTIONS.csv')
    freqs_emo = peaks_emo['freq'].values

    bins = np.arange(1, 50.5, 0.5)
    ax_c.hist(freqs_emo, bins=bins, color='#5B9BD5', edgecolor='white',
              linewidth=0.2, alpha=0.85)

    for n in range(-1, 5):
        f = F0 * PHI**n
        if 1 < f < 50:
            ax_c.axvline(f, color='#E74C3C', linewidth=0.6, alpha=0.7)
    for n in range(0, 4):
        f = F0 * PHI**(n + 0.5)
        if 1 < f < 50:
            ax_c.axvline(f, color='#F4D03F', linewidth=0.6, alpha=0.7, linestyle='--')
    for n in range(0, 4):
        f = F0 * PHI**(n + 0.618)
        if 1 < f < 50:
            ax_c.axvline(f, color='#27AE60', linewidth=0.6, alpha=0.7, linestyle='-.')

    ax_c.set_xlabel('Frequency (Hz)')
    ax_c.set_ylabel('Peak count')
    ax_c.set_title(f'EEGEmotions-27 replication ({len(freqs_emo):,} peaks, 2,342 sessions)', fontsize=7)
    ax_c.set_xlim(1, 50)

except Exception as e:
    img = imread(str(IMG / 'golden_ratio_peaks_EMOTIONS.png'))
    ax_c.imshow(img)
    ax_c.set_xticks([]); ax_c.set_yticks([])
    for s in ax_c.spines.values(): s.set_visible(False)

ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)
ax_c.text(-0.12, 1.05, 'c', transform=ax_c.transAxes,
          fontsize=10, fontweight='bold', va='top')

# ── Panel d: Replication enrichment bar chart ────────────────────────────────
ax_d = fig.add_subplot(gs[1, 1])

enrichment_repl = [-20.5, 6.6, 22.1, 26.3]
ci_low_repl = [-20.9, 5.9, 21.2, 25.4]
ci_high_repl = [-20.0, 7.4, 22.9, 27.1]

yerr_low_r = [e - cl for e, cl in zip(enrichment_repl, ci_low_repl)]
yerr_high_r = [ch - e for e, ch in zip(enrichment_repl, ci_high_repl)]

bars_r = ax_d.bar(range(4), enrichment_repl, color=colors, edgecolor='white',
                  linewidth=0.5, alpha=0.9,
                  yerr=[yerr_low_r, yerr_high_r], capsize=2,
                  error_kw={'linewidth': 0.8, 'color': '#333333'})

ax_d.axhline(0, color='#AAAAAA', linewidth=0.5, linestyle='--')
for i, (e, cl, ch) in enumerate(zip(enrichment_repl, ci_low_repl, ci_high_repl)):
    y_pos = e + 2 if e > 0 else e - 4
    ax_d.text(i, y_pos, f'{e:+.1f}%\n[{cl:.1f}, {ch:.1f}]',
              ha='center', va='bottom' if e > 0 else 'top', fontsize=5)

ax_d.set_xticks(range(4))
ax_d.set_xticklabels(positions, fontsize=5.5)
ax_d.set_ylabel('Enrichment (%)')
ax_d.set_title('Replication: position-type enrichment', fontsize=7)
ax_d.spines['top'].set_visible(False)
ax_d.spines['right'].set_visible(False)
ax_d.text(-0.15, 1.05, 'd', transform=ax_d.transAxes,
          fontsize=10, fontweight='bold', va='top')

# ── Save ─────────────────────────────────────────────────────────────────────
fig.savefig(OUT / 'nature_fig2.png', dpi=300, facecolor='white')
fig.savefig(OUT / 'nature_fig2.pdf', facecolor='white')
plt.close()
print(f"Figure 2 saved to {OUT / 'nature_fig2.png'}")

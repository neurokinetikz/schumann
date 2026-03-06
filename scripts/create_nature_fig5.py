#!/usr/bin/env python3
"""
Figure 5: Substrate-Ignition Model
Nature Neuroscience composite figure

Panels:
  a) φⁿ lattice schematic on frequency axis with EEG bands
  b) Substrate-ignition conceptual model (precision gradient)
  c) f₀ convergence from three independent sources
  d) Proposed φⁿ-based EEG band definitions
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
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
OUT = ROOT / 'papers' / 'images'

# ── Create figure ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.2, 7.5))

gs = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.30,
                       left=0.08, right=0.97, top=0.96, bottom=0.06)

# ── Panel a: φⁿ lattice schematic ───────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, :])  # Full width

# Frequency axis (log scale)
freq_range = (4, 52)
ax_a.set_xlim(np.log10(freq_range[0]), np.log10(freq_range[1]))
ax_a.set_ylim(-0.5, 3.5)

# Traditional EEG bands (background shading)
bands_traditional = [
    ('Theta', 4, 8, '#D4E6F1', 0),
    ('Alpha', 8, 13, '#D5F5E3', 0),
    ('Beta', 13, 30, '#FDEBD0', 0),
    ('Gamma', 30, 52, '#FADBD8', 0),
]
for name, f1, f2, color, _ in bands_traditional:
    ax_a.axvspan(np.log10(f1), np.log10(f2), alpha=0.3, color=color, zorder=0)
    ax_a.text(np.log10(np.sqrt(f1 * f2)), 3.3, name, ha='center', fontsize=6,
              color='#666666', fontstyle='italic')

# φⁿ positions
for n in range(-1, 5):
    f_boundary = F0 * PHI**n
    f_attractor = F0 * PHI**(n + 0.5)
    f_noble = F0 * PHI**(n + 0.618)

    if freq_range[0] <= f_boundary <= freq_range[1]:
        ax_a.axvline(np.log10(f_boundary), color='#E74C3C', linewidth=1.5,
                     alpha=0.8, zorder=2)
        ax_a.text(np.log10(f_boundary), -0.3, f'{f_boundary:.1f}',
                  ha='center', fontsize=5, color='#E74C3C')
        ax_a.text(np.log10(f_boundary), 2.8, f'n={n}',
                  ha='center', fontsize=5, color='#E74C3C', fontweight='bold')

    if freq_range[0] <= f_attractor <= freq_range[1]:
        ax_a.axvline(np.log10(f_attractor), color='#F4D03F', linewidth=1.0,
                     linestyle='--', alpha=0.8, zorder=2)
        ax_a.text(np.log10(f_attractor), -0.3, f'{f_attractor:.1f}',
                  ha='center', fontsize=5, color='#B7950B')

    if freq_range[0] <= f_noble <= freq_range[1]:
        ax_a.axvline(np.log10(f_noble), color='#27AE60', linewidth=1.0,
                     linestyle='-.', alpha=0.8, zorder=2)
        ax_a.text(np.log10(f_noble), -0.3, f'{f_noble:.1f}',
                  ha='center', fontsize=5, color='#27AE60')

# Add enrichment indicators at y=1.5
for n in range(0, 4):
    f_b = F0 * PHI**n
    f_a = F0 * PHI**(n + 0.5)
    f_n = F0 * PHI**(n + 0.618)
    if freq_range[0] <= f_b <= freq_range[1]:
        ax_a.plot(np.log10(f_b), 1.5, 'v', color='#E74C3C', markersize=6, zorder=3)
    if freq_range[0] <= f_a <= freq_range[1]:
        ax_a.plot(np.log10(f_a), 1.5, '^', color='#F4D03F', markersize=5, zorder=3)
    if freq_range[0] <= f_n <= freq_range[1]:
        ax_a.plot(np.log10(f_n), 1.5, 'D', color='#27AE60', markersize=5, zorder=3)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#E74C3C', linewidth=1.5, label='Boundary (depletion)'),
    Line2D([0], [0], color='#F4D03F', linewidth=1.0, linestyle='--', label='Attractor (enrichment)'),
    Line2D([0], [0], color='#27AE60', linewidth=1.0, linestyle='-.', label='1° Noble (max enrichment)'),
]
ax_a.legend(handles=legend_elements, loc='upper right', frameon=True,
            fancybox=False, edgecolor='#CCCCCC', fontsize=5.5)

# Format
ax_a.set_xlabel('Frequency (Hz)')
freqs_ticks = [5, 7.6, 10, 15, 20, 30, 40, 50]
ax_a.set_xticks([np.log10(f) for f in freqs_ticks])
ax_a.set_xticklabels([str(f) for f in freqs_ticks])
ax_a.set_yticks([])
ax_a.set_title(r'$\varphi^n$ lattice: frequency architecture with EEG bands', fontsize=8)
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)
ax_a.spines['left'].set_visible(False)
ax_a.text(-0.03, 1.05, 'a', transform=ax_a.transAxes,
          fontsize=10, fontweight='bold', va='top')

# ── Panel b: Substrate-ignition conceptual model ─────────────────────────────
ax_b = fig.add_subplot(gs[1, 0])

# Create conceptual diagram
t = np.linspace(0, 10, 500)

# Substrate: continuous low-amplitude oscillation with phi^n peaks
substrate = 0.3 * np.sin(2 * np.pi * 0.5 * t)
for n in range(0, 4):
    f = F0 * PHI**(n * 0.3)
    substrate += 0.15 * np.sin(2 * np.pi * f * 0.1 * t + n)

# SIE event: transient amplification between t=3 and t=7
sie_envelope = np.zeros_like(t)
mask_sie = (t > 3) & (t < 7)
sie_envelope[mask_sie] = np.exp(-((t[mask_sie] - 5)**2) / 1.5)

ignition = substrate + 1.5 * sie_envelope * np.sin(2 * np.pi * 2 * t)

ax_b.fill_between(t, -1.8, 1.8, alpha=0.08, color='#3498DB', label='Continuous substrate')
ax_b.fill_between(t[mask_sie], -1.8, 1.8, alpha=0.15, color='#E74C3C')
ax_b.plot(t, substrate, color='#3498DB', linewidth=0.8, alpha=0.6)
ax_b.plot(t, ignition, color='#2C3E50', linewidth=0.6)

# Annotations
ax_b.annotate('SIE\n(transient\namplification)',
              xy=(5, 1.5), fontsize=6, ha='center', color='#E74C3C',
              fontweight='bold')
ax_b.annotate(r'Continuous $\varphi^n$ substrate',
              xy=(1.5, -1.5), fontsize=5.5, ha='center', color='#3498DB')

# Precision labels
ax_b.annotate('±0.2 Hz\ntolerance', xy=(1.5, 1.2), fontsize=5,
              ha='center', color='#7F8C8D')
ax_b.annotate('<1% ratio\nerror', xy=(5, -1.4), fontsize=5,
              ha='center', color='#C0392B', fontweight='bold')

ax_b.set_xlim(0, 10)
ax_b.set_ylim(-2, 2.2)
ax_b.set_xlabel('Time (conceptual)')
ax_b.set_ylabel('Amplitude')
ax_b.set_title('Substrate-ignition model', fontsize=7)
ax_b.set_xticks([])
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)
ax_b.text(-0.12, 1.05, 'b', transform=ax_b.transAxes,
          fontsize=10, fontweight='bold', va='top')

# ── Panel c: f₀ convergence ─────────────────────────────────────────────────
ax_c = fig.add_subplot(gs[1, 1])

sources = ['Geophysical\n(Tomsk)', 'Neural SIE\nmean', 'Spectral\noptimization']
f0_values = [7.60, 7.63, 7.60]
f0_errors = [0.20, 0.33, 0.0]

# Plot with error bars
y_pos = np.arange(len(sources))
colors_f0 = ['#2980B9', '#E74C3C', '#27AE60']

for i, (val, err, color) in enumerate(zip(f0_values, f0_errors, colors_f0)):
    ax_c.errorbar(val, i, xerr=err, fmt='o', color=color, markersize=6,
                  capsize=4, capthick=1.2, linewidth=1.2, zorder=3)

# Weighted mean line
ax_c.axvline(7.60, color='#2C3E50', linewidth=0.8, linestyle='--', alpha=0.5,
             label='Consensus: 7.60 Hz')

# Shade convergence zone
ax_c.axvspan(7.3, 7.9, alpha=0.1, color='#F39C12', label='0.6 Hz plateau')

ax_c.set_yticks(y_pos)
ax_c.set_yticklabels(sources, fontsize=6)
ax_c.set_xlabel(r'$f_0$ (Hz)')
ax_c.set_xlim(6.8, 8.2)
ax_c.set_title(r'Three independent $f_0$ estimates', fontsize=7)
ax_c.legend(frameon=False, fontsize=5.5, loc='upper right')
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)

# Agreement annotation
ax_c.text(7.85, 1, 'Agreement\nwithin 0.4%', fontsize=5.5,
          ha='center', color='#555555', fontstyle='italic')

ax_c.text(-0.15, 1.05, 'c', transform=ax_c.transAxes,
          fontsize=10, fontweight='bold', va='top')

# ── Save ─────────────────────────────────────────────────────────────────────
fig.savefig(OUT / 'nature_fig5.png', dpi=300, facecolor='white')
fig.savefig(OUT / 'nature_fig5.pdf', facecolor='white')
plt.close()
print(f"Figure 5 saved to {OUT / 'nature_fig5.png'}")

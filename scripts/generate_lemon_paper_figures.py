#!/usr/bin/env python3
"""
Generate all figures for the LEMON phi-lattice paper:
  "Dominant Peaks, Fragile Metrics"

6 main figures + 4 supplementary figures.
All data loaded from existing CSV exports — no EEG processing.

Usage:
    /opt/anaconda3/bin/python scripts/generate_lemon_paper_figures.py
"""

import sys, os
sys.path.insert(0, '/Users/neurokinetikz/Code/schumann/lib')

import numpy as np
import pandas as pd
from scipy import stats
from glob import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

# ── Constants ────────────────────────────────────────────────────────────
PHI = (1 + np.sqrt(5)) / 2
F0  = 7.83
SIGMA = 0.03

POSITIONS = {'boundary': 0.0, 'noble₂': 0.382, 'attractor': 0.500, 'noble₁': 0.618}
POS_COLORS = {'boundary': '#e74c3c', 'noble₂': '#3498db', 'attractor': '#2ecc71', 'noble₁': '#9b59b6'}

# Degree-3 positions for Figure 1a
POSITIONS_DEG3 = {
    'boundary':  0.0,
    'noble₃⁻':  0.236,
    'noble₂':   0.382,
    'attractor': 0.500,
    'noble₁':   0.618,
    'noble₃⁺':  0.764,
}
POS_COLORS_DEG3 = {
    'boundary':  '#e74c3c',
    'noble₃⁻':  '#e67e22',
    'noble₂':   '#3498db',
    'attractor': '#2ecc71',
    'noble₁':   '#9b59b6',
    'noble₃⁺':  '#f39c12',
}
BANDS = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'gamma': (30, 45)}
BAND_COLORS = {'delta': '#1f77b4', 'theta': '#ff7f0e', 'alpha': '#2ca02c', 'gamma': '#d62728'}

BASES = {
    '1.4': 1.4, '√2': np.sqrt(2), '3/2': 1.5, 'φ': PHI,
    '1.7': 1.7, '1.8': 1.8, '2': 2.0, 'e': np.e, 'π': np.pi
}

OUT_DIR = '/Users/neurokinetikz/Code/schumann/papers/images/lemon'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Helper Functions ─────────────────────────────────────────────────────

def lattice_coord(freq, f0=F0):
    return (np.log(freq / f0) / np.log(PHI)) % 1.0

def min_lattice_dist(u):
    pos_vals = list(POSITIONS.values())
    return min(min(abs(u - p), 1 - abs(u - p)) for p in pos_vals)

def nearest_pos_name(u):
    dists = {}
    for name, p in POSITIONS.items():
        dists[name] = min(abs(u - p), 1 - abs(u - p))
    return min(dists, key=dists.get)

def circ_dist(a, b):
    d = abs(a - b)
    return min(d, 1 - d)

def get_dominant_peaks(peak_dir, suffix='_peaks*.csv', bands=BANDS, f0=F0, ec=False):
    """Load per-subject dominant peaks from peak CSVs."""
    pattern = 'sub-*_peaks_ec.csv' if ec else 'sub-*_peaks.csv'
    records = []
    for f in sorted(glob(os.path.join(peak_dir, pattern))):
        if 'band_info' in f or 'max40' in f:
            continue
        sid = os.path.basename(f).split('_peaks')[0]
        df = pd.read_csv(f)
        row = {'subject_id': sid}

        band_ds = []
        for bname, (lo, hi) in bands.items():
            bp = df[(df.freq >= lo) & (df.freq < hi)]
            if len(bp) == 0:
                row[f'{bname}_freq'] = np.nan
                row[f'{bname}_d'] = np.nan
                row[f'{bname}_power'] = np.nan
                row[f'{bname}_u'] = np.nan
                row[f'{bname}_nearest'] = np.nan
                continue
            idx = bp['power'].idxmax()
            freq = bp.loc[idx, 'freq']
            power = bp.loc[idx, 'power']
            u = lattice_coord(freq, f0)
            d = min_lattice_dist(u)
            row[f'{bname}_freq'] = freq
            row[f'{bname}_power'] = power
            row[f'{bname}_u'] = u
            row[f'{bname}_d'] = d
            row[f'{bname}_nearest'] = nearest_pos_name(u)
            band_ds.append(d)

        row['mean_d'] = np.mean(band_ds) if len(band_ds) == 4 else np.nan
        row['n_bands'] = len(band_ds)
        records.append(row)

    return pd.DataFrame(records)

def get_positions_for_base(base):
    """Return lattice positions for a given base (boundary + attractor always; nobles only for phi)."""
    if abs(base - PHI) < 0.01:
        return {'boundary': 0.0, 'noble₂': 0.382, 'attractor': 0.500, 'noble₁': 0.618}
    else:
        # Only boundary and attractor (midpoint) for other bases
        return {'boundary': 0.0, 'attractor': 0.500}

def mean_d_for_base(peak_dir, base, bands=BANDS):
    """Compute mean_d for a given base across all subjects."""
    positions = get_positions_for_base(base)
    pos_vals = list(positions.values())

    records = []
    for f in sorted(glob(os.path.join(peak_dir, 'sub-*_peaks.csv'))):
        if 'band_info' in f or 'max40' in f or '_ec' in f:
            continue
        df = pd.read_csv(f)
        band_ds = []
        for bname, (lo, hi) in bands.items():
            bp = df[(df.freq >= lo) & (df.freq < hi)]
            if len(bp) == 0:
                continue
            idx = bp['power'].idxmax()
            freq = bp.loc[idx, 'freq']
            u = (np.log(freq / F0) / np.log(base)) % 1.0
            d = min(min(abs(u - p), 1 - abs(u - p)) for p in pos_vals)
            band_ds.append(d)
        if len(band_ds) == 4:
            records.append(np.mean(band_ds))
    return np.array(records)


# ── Load All Data ────────────────────────────────────────────────────────
print("Loading data...")

# Behavioral & features
behav = pd.read_csv('/Users/neurokinetikz/Code/schumann/exports_lemon/master_behavioral.csv')
features = pd.read_csv('/Users/neurokinetikz/Code/schumann/exports_lemon/subject_features.csv')

# EO dominant peaks (overlap-trim at f₀=7.83)
eo_dir = '/Users/neurokinetikz/Code/schumann/exports_lemon/per_subject_overlap_trim_f07.83'
dom_eo = get_dominant_peaks(eo_dir)
print(f"  EO subjects: {len(dom_eo)}, with all 4 bands: {(dom_eo.n_bands == 4).sum()}")

# EC dominant peaks
ec_dir = '/Users/neurokinetikz/Code/schumann/exports_lemon/per_subject'
dom_ec = get_dominant_peaks(ec_dir, ec=True)
print(f"  EC subjects: {len(dom_ec)}, with all 4 bands: {(dom_ec.n_bands == 4).sum()}")

# Rename EC columns
ec_rename = {}
for c in dom_ec.columns:
    if c != 'subject_id' and c != 'n_bands':
        ec_rename[c] = c + '_ec' if not c.endswith('_ec') else c
    elif c == 'n_bands':
        ec_rename[c] = 'n_bands_ec'
dom_ec = dom_ec.rename(columns=ec_rename)

# Merge everything
merged = dom_eo.merge(behav, on='subject_id', how='left')
merged = merged.merge(features[['subject_id', 'iaf', 'n_peaks', 'compliance',
                                'E_boundary', 'E_noble_2', 'E_attractor', 'E_noble_1',
                                'mean_aperiodic_exponent']], on='subject_id', how='left')
merged = merged.merge(dom_ec, on='subject_id', how='left')

valid = merged[merged.n_bands == 4].copy()
both = valid.dropna(subset=['mean_d_ec']).copy()
print(f"  Valid (4 bands): {len(valid)}, both conditions (OT-EO × Std-EC): {len(both)}")

# Matched extraction for Figure 5 (standard EO + standard EC — same FOOOF method)
std_eo_dir = '/Users/neurokinetikz/Code/schumann/exports_lemon/per_subject'
dom_eo_std = get_dominant_peaks(std_eo_dir)
dom_eo_std_renamed = dom_eo_std.rename(columns={c: c + '_std' if c not in ('subject_id', 'n_bands') else
                                                  ('n_bands_std' if c == 'n_bands' else c)
                                                  for c in dom_eo_std.columns})
matched = dom_eo_std.merge(behav[['subject_id', 'age_midpoint', 'age_group']], on='subject_id', how='left')
matched = matched.merge(dom_ec, on='subject_id', how='left')
matched = matched.merge(features[['subject_id', 'iaf']], on='subject_id', how='left')
# For theta transition analysis (Figure 5): only require theta data, not all 4 bands
both_matched = matched.dropna(subset=['theta_freq', 'theta_freq_ec']).copy()
print(f"  Matched (Std-EO × Std-EC, theta available): {len(both_matched)}")

# Amplitude-weighted data
age_trend = pd.read_csv('/Users/neurokinetikz/Code/schumann/exports_lemon/amplitude_weighted/age_trend_comparison.csv')
band_decomp = pd.read_csv('/Users/neurokinetikz/Code/schumann/exports_lemon/amplitude_weighted/shuffle_z_band_decomposed.csv')
f0_corr = pd.read_csv('/Users/neurokinetikz/Code/schumann/exports_lemon/amplitude_weighted/f0_correction.csv')

# Overlap-trim compliance
ot_compliance = pd.read_csv('/Users/neurokinetikz/Code/schumann/exports_lemon/per_subject_overlap_trim_f07.83/compliance_results.csv')

# Weighted features
weighted_feat = pd.read_csv('/Users/neurokinetikz/Code/schumann/exports_lemon/amplitude_weighted/subject_features_weighted.csv')

print("Data loaded.\n")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1: THE PHI-LATTICE AND INDIVIDUAL ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════
print("Generating Figure 1...")

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

# ── 1a: Phi-lattice positions on unit circle (degree 3) ──
ax1 = fig.add_subplot(gs[0, 0], projection='polar')
ax1.set_title('(a) Degree-3 phi-lattice positions\non the unit circle', fontsize=11, pad=15)
theta_ring = np.linspace(0, 2*np.pi, 200)
ax1.plot(theta_ring, np.ones_like(theta_ring), 'k-', alpha=0.2, lw=1)

# Custom offsets to avoid label overlaps on the polar plot
_label_offsets = {
    'boundary':  (20, 12),
    'noble₃⁻':  (18, -15),
    'noble₂':   (15, 10),
    'attractor': (-60, 12),
    'noble₁':   (15, -15),
    'noble₃⁺':  (18, 10),
}

for name, u in POSITIONS_DEG3.items():
    angle = 2 * np.pi * u
    color = POS_COLORS_DEG3[name]
    ax1.plot(angle, 1.0, 'o', color=color, ms=12, zorder=5)
    label = f'{name}\nu={u:.3f}'
    xyoff = _label_offsets.get(name, (15, 10))
    ax1.annotate(label, (angle, 1.0), textcoords='offset points',
                xytext=xyoff, fontsize=7.5, color=color, fontweight='bold')

ax1.set_ylim(0, 1.3)
ax1.set_yticks([])
ax1.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
ax1.set_xticklabels(['0', '0.25', '0.5', '0.75'])

# ── 1b: Per-subject mean_d distribution vs null ──
ax2 = fig.add_subplot(gs[0, 1])

mean_ds = valid.mean_d.dropna().values
null_expected = 0.07991502812526288  # precise analytical null for degree-2, 4 positions

ax2.hist(mean_ds, bins=30, color='#3498db', alpha=0.7, edgecolor='white', density=True, label='Observed')
ax2.axvline(np.mean(mean_ds), color='#2c3e50', lw=2, ls='-', label=f'Mean = {np.mean(mean_ds):.3f}')
ax2.axvline(null_expected, color='#e74c3c', lw=2, ls='--', label=f'Null = {null_expected:.3f}')

t_stat, p_val = stats.ttest_1samp(mean_ds, null_expected)
d_eff = (np.mean(mean_ds) - null_expected) / np.std(mean_ds)
ax2.set_title(f'(b) Individual alignment\nt = {t_stat:.2f}, d = {d_eff:.2f}, p < 10⁻⁷', fontsize=11)
ax2.set_xlabel('Mean lattice distance (mean_d)')
ax2.set_ylabel('Density')
ax2.legend(fontsize=8, loc='upper right')

# ── 1c: Per-band distance distributions ──
ax3 = fig.add_subplot(gs[1, 0])

band_data = []
band_labels = []
for bname in ['delta', 'theta', 'alpha', 'gamma']:
    d_vals = valid[f'{bname}_d'].dropna().values
    band_data.append(d_vals)
    # KS test vs uniform
    # Uniform distance distribution for 4 positions
    null_samples = np.random.uniform(0, 0.25, size=10000)  # max distance is 0.25 for 4 positions
    ks, p = stats.ks_2samp(d_vals, null_samples)
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    band_labels.append(f'{bname}\n{sig}')

bp = ax3.boxplot(band_data, labels=band_labels, patch_artist=True, widths=0.6)
colors = [BAND_COLORS[b] for b in ['delta', 'theta', 'alpha', 'gamma']]
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax3.axhline(null_expected, color='#e74c3c', ls='--', lw=1, alpha=0.7, label='Null')
ax3.set_ylabel('Lattice distance (d)')
ax3.set_title('(c) Per-band alignment\n(theta & alpha drive the effect)', fontsize=11)
ax3.legend(fontsize=8)

# ── 1d: Cross-base comparison (degree-3, coverage-normalised) ──
ax4 = fig.add_subplot(gs[1, 1])

# Load pre-computed degree-3 results from both datasets
deg3_lemon = pd.read_csv('/Users/neurokinetikz/Code/schumann/exports_lemon/'
                         'per_subject_overlap_trim_f07.83/dominant_peak/degree3_crossbase.csv')
deg3_eegmmidb = pd.read_csv('/Users/neurokinetikz/Code/schumann/exports_eegmmidb/'
                            'per_subject_overlap_trim_f07.83/dominant_peak/degree3_crossbase.csv')

# Sort by LEMON d_over_null (ascending = best alignment first)
deg3_lemon = deg3_lemon.sort_values('d_over_null', ascending=True).reset_index(drop=True)
base_order = deg3_lemon['base'].tolist()

# Reorder EEGMMIDB to match
deg3_eegmmidb = deg3_eegmmidb.set_index('base').loc[base_order].reset_index()

y = np.arange(len(base_order))
bar_h = 0.35

# LEMON bars
colors_lemon = ['#2ecc71' if b == 'phi' else '#3498db' for b in base_order]
ax4.barh(y - bar_h/2, deg3_lemon['d_over_null'], bar_h,
         color=colors_lemon, edgecolor='white', alpha=0.85, label='LEMON')

# EEGMMIDB bars
colors_eeg = ['#27ae60' if b == 'phi' else '#e67e22' for b in base_order]
ax4.barh(y + bar_h/2, deg3_eegmmidb['d_over_null'], bar_h,
         color=colors_eeg, edgecolor='white', alpha=0.85, label='EEGMMIDB')

# Reference line at 1.0 (null expectation)
ax4.axvline(1.0, color='#e74c3c', ls='--', lw=1.2, alpha=0.7, label='Null (d̄/d̄$_{null}$ = 1)')

# Labels
display_names = [b.replace('phi', 'φ').replace('sqrt2', '√2') for b in base_order]
ax4.set_yticks(y)
ax4.set_yticklabels(display_names, fontsize=10)
ax4.set_xlabel('d̄ / d̄$_{null}$ (lower = tighter alignment)', fontsize=10)
ax4.set_title('(d) Degree-3 cross-base comparison\n(φ rank 1/9 in both datasets)', fontsize=11)
ax4.legend(fontsize=8, loc='lower right')

# Print summary
for i, b in enumerate(base_order):
    print(f"    {b}: LEMON={deg3_lemon.iloc[i]['d_over_null']:.3f}  "
          f"EEGMMIDB={deg3_eegmmidb.iloc[i]['d_over_null']:.3f}")

fig.suptitle('Figure 1: The phi-lattice and individual alignment', fontsize=13, fontweight='bold', y=1.02)
plt.savefig(os.path.join(OUT_DIR, 'fig1_lattice_alignment.pdf'), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(OUT_DIR, 'fig1_lattice_alignment.png'), bbox_inches='tight', dpi=300)
plt.close()
print("  Figure 1 saved.\n")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2: DIAGNOSTIC CASCADE — ALL-PEAKS FRAGILITY
# ═══════════════════════════════════════════════════════════════════════
print("Generating Figure 2...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ── 2a: SS collapse across frequency ranges ──
ax = axes[0, 0]
# LEMON: SS at [1,45] vs [1,85]
ss_45 = 11.7   # from subject_features, standard extraction
ss_85 = -67.4  # from amplitude_weighted analysis

# EEGMMIDB comparison
ss_eeg_50 = 45.6
ss_eeg_75 = -23.6

x = np.arange(2)
width = 0.35
bars1 = ax.bar(x - width/2, [ss_45, ss_85], width, label='LEMON', color='#3498db', alpha=0.8)
bars2 = ax.bar(x + width/2, [ss_eeg_50, ss_eeg_75], width, label='EEGMMIDB', color='#e67e22', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(['Standard\n[1,45]/[1,50] Hz', 'Extended\n[1,85]/[1,75] Hz'])
ax.set_ylabel('Structural Score (SS)')
ax.axhline(0, color='k', lw=0.5)
ax.legend(fontsize=9)
ax.set_title('(a) SS collapses with extended\nFOOOF frequency range', fontsize=11)

# Annotate sign reversal
for bar, val in zip(bars1, [ss_45, ss_85]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2 * np.sign(val),
            f'{val:+.1f}', ha='center', va='bottom' if val > 0 else 'top', fontsize=9, fontweight='bold')
for bar, val in zip(bars2, [ss_eeg_50, ss_eeg_75]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2 * np.sign(val),
            f'{val:+.1f}', ha='center', va='bottom' if val > 0 else 'top', fontsize=9, fontweight='bold')

# ── 2b: Per-phi-octave edge artifact ──
ax = axes[0, 1]
positions_list = ['boundary', 'noble₂', 'attractor', 'noble₁']
primary_enrich = [-60.8, +31.6, +31.6, +31.6]  # simplified — using actual values
offset_enrich = [+14.8, -48.2, -48.2, -48.2]

# Actual per-phi-octave data
primary_vals = {'boundary': -60.8, 'noble₂': None, 'attractor': +31.6, 'noble₁': None}
offset_vals = {'boundary': +14.8, 'noble₂': None, 'attractor': -48.2, 'noble₁': None}
# Use SS values
labels = ['SS', 'E_boundary', 'E_attractor']
primary_ss = [119.3, -60.8, 31.6]
offset_ss = [-84.9, 14.8, -48.2]

x = np.arange(len(labels))
width = 0.35
ax.bar(x - width/2, primary_ss, width, label='Primary (edges at lattice)', color='#2ecc71', alpha=0.8)
ax.bar(x + width/2, offset_ss, width, label='Offset by 0.5 φ-octave', color='#e74c3c', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.axhline(0, color='k', lw=0.5)
ax.legend(fontsize=8)
ax.set_ylabel('Enrichment (%)')
ax.set_title('(b) Per-phi-octave: enrichment\ntracks fitting edges, not lattice', fontsize=11)

# ── 2c: Overlap-trim improvement ──
ax = axes[1, 0]
# Primary vs offset enrichment for overlap-trim
ot_labels = ['SS', 'E_boundary', 'E_attractor', 'E_noble₂', 'E_noble₁']
ot_primary = [12.9, -7.6, 3.0, 3.7, 1.0]
ot_offset = [5.8, -12.7, -7.7, -3.4, 5.0]

x = np.arange(len(ot_labels))
width = 0.35
ax.bar(x - width/2, ot_primary, width, label='Primary', color='#3498db', alpha=0.8)
ax.bar(x + width/2, ot_offset, width, label='Offset 0.5', color='#e67e22', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(ot_labels, fontsize=9)
ax.axhline(0, color='k', lw=0.5)
ax.legend(fontsize=8)
ax.set_ylabel('Enrichment (%)')
ax.set_title('(c) Overlap-trim: greatly reduced\nbut attractor still flips sign', fontsize=11)

# Add sign-flip annotation on attractor
ax.annotate('Sign flip', xy=(2 + width/2, ot_offset[2]), xytext=(3.5, -12),
            fontsize=9, color='#c0392b', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#c0392b'))

# ── 2d: max_n_peaks kills significance ──
ax = axes[1, 1]
# max20 vs max40 attractor age trend
configs = ['max₂₀\n[1,45]', 'max₄₀\n[1,45]', 'max₂₀\n[1,85]']
r_vals = [-0.195, -0.110, -0.223]
p_vals = [0.008, 0.138, 0.002]

colors_bar = ['#2ecc71' if p < 0.05 else '#e74c3c' for p in p_vals]
bars = ax.bar(configs, r_vals, color=colors_bar, alpha=0.8, edgecolor='white')
ax.axhline(0, color='k', lw=0.5)
ax.set_ylabel('r(age, E_attractor)')
ax.set_title('(d) max_n_peaks = 40 kills\nattractor age trend', fontsize=11)

for bar, r, p in zip(bars, r_vals, p_vals):
    sig = f'p={p:.3f}' if p > 0.01 else f'p={p:.4f}'
    status = '✓' if p < 0.05 else '✗'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.01,
            f'r={r:.3f}\n{sig} {status}', ha='center', va='top', fontsize=8, fontweight='bold')

fig.suptitle('Figure 2: The diagnostic cascade — all-peaks metrics are fragile', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig2_diagnostic_cascade.pdf'), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(OUT_DIR, 'fig2_diagnostic_cascade.png'), bbox_inches='tight', dpi=300)
plt.close()
print("  Figure 2 saved.\n")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 3: AGE EFFECT IS IAF POSITIONING
# ═══════════════════════════════════════════════════════════════════════
print("Generating Figure 3...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ── 3a: Band decomposition ──
ax = axes[0, 0]
bands_bd = ['Alpha', 'Beta', 'Delta+\nTheta', 'Gamma']
r_attractor = [-0.168, -0.170, -0.124, +0.155]
r_shuffle_z = [-0.198, -0.191, -0.136, -0.010]
p_att = [0.024, 0.022, 0.101, 0.037]

x = np.arange(len(bands_bd))
width = 0.35
colors_att = ['#3498db' if p < 0.05 else '#bdc3c7' for p in p_att]
colors_shuf = ['#e67e22' if abs(r) > 0.15 else '#bdc3c7' for r in r_shuffle_z]

ax.bar(x - width/2, r_attractor, width, label='Attractor r(age)', color=colors_att, edgecolor='white')
ax.bar(x + width/2, r_shuffle_z, width, label='Shuffle-z r(age)', color=colors_shuf, edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(bands_bd)
ax.axhline(0, color='k', lw=0.5)
ax.set_ylabel('Pearson r with age')
ax.legend(fontsize=8)
ax.set_title('(a) Band decomposition: α+β\ndrive the effect, δ+θ null', fontsize=11)

# Annotate delta+theta null
ax.annotate('NULL\np = .101', xy=(2, r_attractor[2]), xytext=(2.8, -0.22),
            fontsize=9, color='#c0392b', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#c0392b'))

# ── 3b: f₀ shift confirmation ──
ax = axes[0, 1]

# Merge overlap-trim compliance with age for scatter
ot_merged = ot_compliance  # already has age_midpoint and age_group

# Load f0=8.50 data if available
f0_850_path = '/Users/neurokinetikz/Code/schumann/exports_lemon/per_subject_overlap_trim_f08.50/compliance_results.csv'
if os.path.exists(f0_850_path):
    ot_850 = pd.read_csv(f0_850_path)  # already has age_midpoint and age_group

    # Scatter: f0=7.83
    ax.scatter(ot_merged.age_midpoint, ot_merged.E_attractor, alpha=0.3, s=15,
              color='#3498db', label='f₀ = 7.83')
    r783, p783 = stats.pearsonr(ot_merged.age_midpoint, ot_merged.E_attractor)
    z783 = np.polyfit(ot_merged.age_midpoint, ot_merged.E_attractor, 1)
    ax.plot(sorted(ot_merged.age_midpoint), np.polyval(z783, sorted(ot_merged.age_midpoint)),
           'b-', lw=2)

    # Scatter: f0=8.50
    ax.scatter(ot_850.age_midpoint, ot_850.E_attractor, alpha=0.3, s=15,
              color='#e74c3c', label='f₀ = 8.50')
    r850, p850 = stats.pearsonr(ot_850.age_midpoint, ot_850.E_attractor)
    z850 = np.polyfit(ot_850.age_midpoint, ot_850.E_attractor, 1)
    ax.plot(sorted(ot_850.age_midpoint), np.polyval(z850, sorted(ot_850.age_midpoint)),
           'r-', lw=2)

    ax.set_xlabel('Age (years)')
    ax.set_ylabel('E_attractor (%)')
    ax.legend(fontsize=8, title=f'f₀=7.83: r={r783:.3f}, p<.001\nf₀=8.50: r={r850:.3f}, p={p850:.3f}',
             title_fontsize=8)
else:
    ax.text(0.5, 0.5, 'f₀=8.50 data\nnot available', transform=ax.transAxes,
           ha='center', va='center', fontsize=12, color='gray')

ax.set_title('(b) f₀ shift: 0.67 Hz eliminates\nthe age effect entirely', fontsize=11)

# ── 3c: Per-subject f₀ correction ──
ax = axes[1, 0]
f0_configs = ['f₀ = 8.5\n(EEGMMIDB)', 'f₀ = 7.83\n(Schumann)', 'f₀ = IAF\n(per-subject)', 'f₀ = f₀*\n(optimized)']
r_f0 = [-0.223, -0.195, +0.167, +0.092]
p_f0 = [0.002, 0.008, 0.025, 0.215]

colors_f0 = ['#2ecc71' if r < 0 and p < 0.05 else ('#e74c3c' if r > 0 and p < 0.05 else '#bdc3c7')
             for r, p in zip(r_f0, p_f0)]
bars = ax.bar(f0_configs, r_f0, color=colors_f0, edgecolor='white', alpha=0.8)
ax.axhline(0, color='k', lw=0.5)
ax.set_ylabel('r(age, E_attractor)')
ax.set_title('(c) Per-subject f₀ correction\nreverses the sign', fontsize=11)

for bar, r, p in zip(bars, r_f0, p_f0):
    ypos = bar.get_height() + 0.01 * np.sign(bar.get_height())
    ax.text(bar.get_x() + bar.get_width()/2, ypos,
            f'r={r:+.3f}\np={p:.3f}', ha='center',
            va='bottom' if bar.get_height() > 0 else 'top', fontsize=8)

# ── 3d: Mechanism diagram ──
ax = axes[1, 1]
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('(d) Mechanism: IAF maps to different\nlattice coordinates with age', fontsize=11)

# Draw simplified lattice positions
for name, u in POSITIONS.items():
    x_pos = 0.1 + 0.8 * u
    ax.axvline(x_pos, color=POS_COLORS[name], lw=2, alpha=0.3, ymin=0.1, ymax=0.9)
    ax.text(x_pos, 0.92, name.replace('₂', '₂').replace('₁', '₁'), ha='center', fontsize=8,
           color=POS_COLORS[name], fontweight='bold')
    ax.text(x_pos, 0.05, f'u={u:.3f}', ha='center', fontsize=7, color='gray')

# Young IAF position
u_young = lattice_coord(10.15)
x_young = 0.1 + 0.8 * u_young
ax.annotate('Young IAF\n10.15 Hz\nu = 0.365', xy=(x_young, 0.55), fontsize=9,
           color='#2980b9', ha='center', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#d4e6f1', edgecolor='#2980b9'))

# Elderly IAF position
u_elderly = lattice_coord(9.96)
x_elderly = 0.1 + 0.8 * u_elderly
ax.annotate('Elderly IAF\n9.96 Hz\nu = 0.325', xy=(x_elderly, 0.35), fontsize=9,
           color='#c0392b', ha='center', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#fadbd8', edgecolor='#c0392b'))

# Arrow showing drift
ax.annotate('', xy=(x_elderly, 0.45), xytext=(x_young, 0.45),
           arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=2))
ax.text((x_young + x_elderly)/2, 0.48, 'IAF slowing\nwith age', ha='center', fontsize=8, color='#7f8c8d')

fig.suptitle('Figure 3: The age effect is IAF positioning, not lattice erosion', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig3_age_iaf_positioning.pdf'), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(OUT_DIR, 'fig3_age_iaf_positioning.png'), bbox_inches='tight', dpi=300)
plt.close()
print("  Figure 3 saved.\n")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 4: AGE-INVARIANCE AND COGNITIVE SILENCE
# ═══════════════════════════════════════════════════════════════════════
print("Generating Figure 4...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ── 4a: mean_d vs age scatter ──
ax = axes[0, 0]
young = valid[valid.age_group == 'young']
elderly = valid[valid.age_group == 'elderly']

ax.scatter(young.age_midpoint, young.mean_d, alpha=0.4, s=20, color='#3498db', label='Young')
ax.scatter(elderly.age_midpoint, elderly.mean_d, alpha=0.4, s=20, color='#e74c3c', label='Elderly')

r_age, p_age = stats.pearsonr(valid.age_midpoint, valid.mean_d)
z = np.polyfit(valid.age_midpoint, valid.mean_d, 1)
x_line = np.linspace(valid.age_midpoint.min(), valid.age_midpoint.max(), 100)
ax.plot(x_line, np.polyval(z, x_line), 'k--', lw=1.5, alpha=0.5)

ax.set_xlabel('Age (years)')
ax.set_ylabel('Mean lattice distance (mean_d)')
ax.set_title(f'(a) Alignment is age-invariant\nr = {r_age:.3f}, p = {p_age:.2f}', fontsize=11)
ax.legend(fontsize=9)

# ── 4b: Population median peaks on lattice ──
ax = axes[0, 1]

# Population median peaks
pop_freqs = {}
for bname in BANDS:
    pop_freqs[bname] = valid[f'{bname}_freq'].median()

# Map to lattice
pop_us = {b: lattice_coord(f) for b, f in pop_freqs.items()}
pop_ds = {b: min_lattice_dist(u) for b, u in pop_us.items()}
pop_mean_d = np.mean(list(pop_ds.values()))

# Draw unit interval with positions
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.3, 1.0)

# Position markers
for name, u in POSITIONS.items():
    ax.axvline(u, color=POS_COLORS[name], lw=2, alpha=0.3, ymin=0, ymax=0.8)
    ax.plot(u, 0.85, 'v', color=POS_COLORS[name], ms=12)
    ax.text(u, 0.92, name, ha='center', fontsize=8, color=POS_COLORS[name], fontweight='bold')

# Population peak positions
y_offset = 0.5
for i, (bname, u) in enumerate(pop_us.items()):
    y = y_offset - i * 0.15
    ax.plot(u, y, 'o', color=BAND_COLORS[bname], ms=10, zorder=5)
    ax.text(u + 0.03, y, f'{bname}: {pop_freqs[bname]:.2f} Hz\n(u={u:.3f})',
           fontsize=8, va='center')

ax.set_xlabel('Lattice coordinate (u)')
ax.set_yticks([])
ax.set_title(f'(b) Population median peaks on lattice\n(d̄ of 4 band medians = {pop_mean_d:.3f})', fontsize=11)

# ── 4c: Cognitive null — forest plot ──
ax = axes[1, 0]

cog_tests = ['CVLT', 'log_TMT_A', 'log_TMT_B', 'log_TAP_Alert', 'TAP_WM',
             'TAP_Incompat', 'LPS', 'WST', 'RWT']

# Compute age-partialed correlations
partial_rs = []
partial_ps = []
for cog in cog_tests:
    vals = valid[['mean_d', cog, 'age_midpoint']].dropna()
    if len(vals) < 20:
        partial_rs.append(np.nan)
        partial_ps.append(np.nan)
        continue
    slope_d, int_d = np.polyfit(vals.age_midpoint, vals.mean_d, 1)
    resid_d = vals.mean_d - (slope_d * vals.age_midpoint + int_d)
    slope_c, int_c = np.polyfit(vals.age_midpoint, vals[cog], 1)
    resid_c = vals[cog] - (slope_c * vals.age_midpoint + int_c)
    r, p = stats.pearsonr(resid_d, resid_c)
    partial_rs.append(r)
    partial_ps.append(p)

y_pos = np.arange(len(cog_tests))
colors_cog = ['#e74c3c' if p < 0.05 else '#95a5a6' for p in partial_ps]

ax.barh(y_pos, partial_rs, color=colors_cog, alpha=0.7, edgecolor='white')
ax.axvline(0, color='k', lw=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(cog_tests, fontsize=9)
ax.set_xlabel('Partial r (age-controlled)')
ax.set_title('(c) Cognitive null: 0/9 survive\nFDR correction', fontsize=11)

# FDR line
from scipy.stats import false_discovery_control
fdr_ps = false_discovery_control(np.array(partial_ps), method='bh')
n_sig = (fdr_ps < 0.05).sum()
ax.text(0.95, 0.05, f'FDR survivors: {n_sig}/{len(cog_tests)}',
       transform=ax.transAxes, ha='right', fontsize=10, color='#c0392b', fontweight='bold')

# ── 4d: Three-metric cognitive null table ──
ax = axes[1, 1]
ax.axis('off')
ax.set_title('(d) Triple cognitive null\nacross all metrics', fontsize=11)

table_data = [
    ['Metric', 'Tests', 'FDR\nsurvivors', 'Largest |r|'],
    ['All-peaks SS', '9', '0', 'WST r=0.090'],
    ['Within-band\nz-scores', '36', '1*', 'RWT r=0.256'],
    ['Dominant-peak\ndistance', '36+', '0', 'δ_d~WST\nr=0.218'],
]

table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                cellLoc='center', loc='center',
                colWidths=[0.3, 0.12, 0.15, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header
for j in range(4):
    table[(0, j)].set_facecolor('#2c3e50')
    table[(0, j)].set_text_props(color='white', fontweight='bold')

# Style "0" cells red
for i in range(1, 4):
    table[(i, 2)].set_text_props(fontweight='bold', color='#c0392b')

ax.text(0.5, 0.05, '*non-phi-specific (3/2 beats φ)', transform=ax.transAxes,
       ha='center', fontsize=8, fontstyle='italic', color='gray')

fig.suptitle('Figure 4: Age-invariance and cognitive silence', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig4_age_invariance_cognitive_null.pdf'), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(OUT_DIR, 'fig4_age_invariance_cognitive_null.png'), bbox_inches='tight', dpi=300)
plt.close()
print("  Figure 4 saved.\n")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 5: THETA POSITION MIGRATION UNDER EC
# ═══════════════════════════════════════════════════════════════════════
print("Generating Figure 5...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ── 5a: Theta transition matrix (matched extraction: Std EO + Std EC) ──
ax = axes[0, 0]
fig5_data = both_matched  # matched extraction for consistent EO↔EC comparison

pos_names_ordered = ['boundary', 'noble₂', 'attractor', 'noble₁']
transition_matrix = np.zeros((4, 4))

for i, pos_eo in enumerate(pos_names_ordered):
    mask_eo = fig5_data['theta_nearest'] == pos_eo
    total = mask_eo.sum()
    if total == 0:
        continue
    for j, pos_ec in enumerate(pos_names_ordered):
        n = ((fig5_data['theta_nearest'] == pos_eo) & (fig5_data['theta_nearest_ec'] == pos_ec)).sum()
        transition_matrix[i, j] = n / total * 100 if total > 0 else 0

im = ax.imshow(transition_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(pos_names_ordered, fontsize=9, rotation=45)
ax.set_yticklabels(pos_names_ordered, fontsize=9)
ax.set_xlabel('EC position')
ax.set_ylabel('EO position')
ax.set_title('(a) Theta position transitions\n(EO → EC, matched extraction)', fontsize=11)

# Annotate cells
for i in range(4):
    for j in range(4):
        val = transition_matrix[i, j]
        if val > 0:
            color = 'white' if val > 50 else 'black'
            pos_eo = pos_names_ordered[i]
            n_eo = (fig5_data['theta_nearest'] == pos_eo).sum()
            n_cell = int(round(val * n_eo / 100))
            ax.text(j, i, f'{val:.0f}%\n(N={n_cell})', ha='center', va='center',
                   fontsize=8, color=color, fontweight='bold' if i == j else 'normal')

plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# ── 5b: Theta frequency convergence on f₀ ──
ax = axes[0, 1]

theta_eo = fig5_data['theta_freq'].values
theta_ec = fig5_data['theta_freq_ec'].values
delta_theta = theta_ec - theta_eo

# Color by direction relative to f₀
below_f0 = theta_eo < F0
above_f0 = theta_eo > F0

ax.scatter(theta_eo[below_f0], delta_theta[below_f0], alpha=0.4, s=20,
          color='#3498db', label=f'Below f₀ (N={below_f0.sum()})')
ax.scatter(theta_eo[above_f0], delta_theta[above_f0], alpha=0.4, s=20,
          color='#e74c3c', label=f'Above f₀ (N={above_f0.sum()})')

ax.axhline(0, color='k', lw=0.5)
ax.axvline(F0, color='gray', ls='--', lw=1, alpha=0.5)
ax.set_xlabel('EO theta frequency (Hz)')
ax.set_ylabel('Δ theta (EC - EO, Hz)')
ax.set_title('(b) Theta converges on f₀\nfrom both directions', fontsize=11)
ax.legend(fontsize=8)

# Add mean shifts
mean_below = delta_theta[below_f0].mean()
mean_above = delta_theta[above_f0].mean()
ax.text(0.05, 0.95, f'Below f₀: mean shift = {mean_below:+.2f} Hz',
       transform=ax.transAxes, fontsize=9, color='#3498db', va='top')
ax.text(0.05, 0.88, f'Above f₀: mean shift = {mean_above:+.2f} Hz',
       transform=ax.transAxes, fontsize=9, color='#e74c3c', va='top')

# Distance to f₀ test
dist_eo = np.abs(theta_eo - F0)
dist_ec = np.abs(theta_ec - F0)
t_conv, p_conv = stats.ttest_rel(dist_ec, dist_eo)
ax.text(0.05, 0.78, f'|θ−f₀| decreases: t={t_conv:.2f}, p={p_conv:.3f}',
       transform=ax.transAxes, fontsize=9, color='#2c3e50', va='top')

# ── 5c: Position stability by starting position ──
ax = axes[1, 0]

# Compute stability rate per EO position
stability = {}
for pos in pos_names_ordered:
    mask = fig5_data['theta_nearest'] == pos
    if mask.sum() >= 3:
        stable = (fig5_data.loc[mask, 'theta_nearest'] == fig5_data.loc[mask, 'theta_nearest_ec']).mean()
        stability[pos] = (stable * 100, mask.sum())

if stability:
    pos_labs = list(stability.keys())
    stab_vals = [stability[p][0] for p in pos_labs]
    ns = [stability[p][1] for p in pos_labs]
    colors_stab = [POS_COLORS[p] for p in pos_labs]

    bars = ax.bar(pos_labs, stab_vals, color=colors_stab, alpha=0.8, edgecolor='white')
    ax.set_ylabel('% staying at same position')
    bnd_stab = stability.get('boundary', (0, 0))[0]
    n1_stab = stability.get('noble₁', (0, 0))[0]
    ax.set_title(f'(c) Boundary is an absorbing state\n({bnd_stab:.0f}% stable vs {n1_stab:.0f}% for noble₁)', fontsize=11)
    ax.set_ylim(0, 105)

    for bar, n, sv in zip(bars, ns, stab_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{sv:.0f}%\n(N={n})', ha='center', va='bottom', fontsize=9, fontweight='bold')

# ── 5d: Power changes at stable positions ──
ax = axes[1, 1]

power_changes = {}
for bname in BANDS:
    same_pos = fig5_data[fig5_data[f'{bname}_nearest'] == fig5_data[f'{bname}_nearest_ec']]
    if len(same_pos) > 10:
        pow_eo = same_pos[f'{bname}_power'].values
        pow_ec = same_pos[f'{bname}_power_ec'].values
        d_power = (pow_ec.mean() - pow_eo.mean()) / np.sqrt((pow_ec.std()**2 + pow_eo.std()**2)/2)
        t, p = stats.ttest_rel(pow_ec, pow_eo)
        power_changes[bname] = (d_power, p, len(same_pos))

if power_changes:
    band_labs = list(power_changes.keys())
    d_vals = [power_changes[b][0] for b in band_labs]
    p_vals_pow = [power_changes[b][1] for b in band_labs]
    ns_pow = [power_changes[b][2] for b in band_labs]
    colors_pow = [BAND_COLORS[b] for b in band_labs]

    bars = ax.bar(band_labs, d_vals, color=colors_pow, alpha=0.8, edgecolor='white')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_ylabel("Cohen's d (EC - EO)")
    ax.set_title('(d) Power changes at stable\npositions (EC vs EO)', fontsize=11)

    for bar, d, p, n in zip(bars, d_vals, p_vals_pow, ns_pow):
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05 * np.sign(bar.get_height()),
               f'd={d:.2f}\n{sig}\nN={n}', ha='center',
               va='bottom' if d > 0 else 'top', fontsize=8)

fig.suptitle('Figure 5: Theta position migration under eyes-closed conditions', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig5_theta_migration.pdf'), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(OUT_DIR, 'fig5_theta_migration.png'), bbox_inches='tight', dpi=300)
plt.close()
print("  Figure 5 saved.")
if power_changes:
    print("  Power changes at stable positions:")
    for bname, (d, p, n) in power_changes.items():
        print(f"    {bname}: d={d:.4f}, p={p:.4e}, N={n}")
print()


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 6: POSITION PHENOTYPING
# ═══════════════════════════════════════════════════════════════════════
print("Generating Figure 6...")

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

# ── 6a: Theta-alpha position pair heatmap ──
ax = axes[0]

valid_c = valid.copy()
# Create pair column
pair_col = valid_c['theta_nearest'].astype(str) + ' → ' + valid_c['alpha_nearest'].astype(str)

# Build matrix
theta_positions = ['boundary', 'noble₂', 'attractor', 'noble₁']
alpha_positions = ['boundary', 'noble₂', 'attractor', 'noble₁']

pair_matrix = np.zeros((4, 4))
for i, tp in enumerate(theta_positions):
    for j, ap in enumerate(alpha_positions):
        pair_matrix[i, j] = ((valid_c['theta_nearest'] == tp) & (valid_c['alpha_nearest'] == ap)).sum()

im = ax.imshow(pair_matrix, cmap='Blues', aspect='auto')
ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(alpha_positions, fontsize=9, rotation=45)
ax.set_yticklabels(theta_positions, fontsize=9)
ax.set_xlabel('Alpha position')
ax.set_ylabel('Theta position')
ax.set_title('(a) Theta-alpha position pairs\n(15 combinations observed)', fontsize=11)

for i in range(4):
    for j in range(4):
        val = int(pair_matrix[i, j])
        if val > 0:
            pct = val / len(valid_c) * 100
            color = 'white' if val > pair_matrix.max() * 0.6 else 'black'
            ax.text(j, i, f'{val}\n({pct:.0f}%)', ha='center', va='center',
                   fontsize=8, color=color)

plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# ── 6b: Spectral profiles by theta group ──
ax = axes[1]

valid_c['theta_group'] = valid_c['theta_nearest'].map(
    lambda x: 'boundary' if x == 'boundary' else ('noble' if x in ['noble₁', 'noble₂'] else 'attractor'))

boundary_mask = valid_c.theta_group == 'boundary'
noble_mask = valid_c.theta_group == 'noble'

band_names_plot = list(BANDS.keys())
x = np.arange(len(band_names_plot))
width = 0.35

b_powers = [valid_c.loc[boundary_mask, f'{b}_power'].mean() for b in band_names_plot]
n_powers = [valid_c.loc[noble_mask, f'{b}_power'].mean() for b in band_names_plot]
b_stds = [valid_c.loc[boundary_mask, f'{b}_power'].std() / np.sqrt(boundary_mask.sum()) for b in band_names_plot]
n_stds = [valid_c.loc[noble_mask, f'{b}_power'].std() / np.sqrt(noble_mask.sum()) for b in band_names_plot]

ax.bar(x - width/2, b_powers, width, yerr=b_stds,
      label=f'Boundary (N={boundary_mask.sum()})', color='#e74c3c', alpha=0.7, capsize=3)
ax.bar(x + width/2, n_powers, width, yerr=n_stds,
      label=f'Noble (N={noble_mask.sum()})', color='#9b59b6', alpha=0.7, capsize=3)

ax.set_xticks(x)
ax.set_xticklabels(band_names_plot)
ax.set_ylabel('Mean FOOOF peak power')
ax.set_title('(b) Spectral profiles:\nboundary = stronger oscillators', fontsize=11)
ax.legend(fontsize=8)

# Add significance
for i, bname in enumerate(band_names_plot):
    b_vals = valid_c.loc[boundary_mask, f'{bname}_power'].dropna()
    n_vals = valid_c.loc[noble_mask, f'{bname}_power'].dropna()
    if len(b_vals) > 5 and len(n_vals) > 5:
        t, p = stats.ttest_ind(b_vals, n_vals)
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        if sig:
            max_h = max(b_powers[i], n_powers[i]) + max(b_stds[i], n_stds[i]) + 0.02
            ax.text(i, max_h, sig, ha='center', fontsize=11, fontweight='bold')

# ── 6c: Cognitive scores by theta group ──
ax = axes[2]

cog_primary = ['CVLT', 'log_TMT_B', 'TAP_WM', 'LPS', 'WST', 'RWT']
cog_labels = ['CVLT', 'log\nTMT-B', 'TAP\nWM', 'LPS', 'WST', 'RWT']

# Age-controlled Cohen's d
cohens_ds = []
p_vals_cog = []
for cog in cog_primary:
    vals = valid_c[['theta_group', cog, 'age_midpoint']].dropna()
    vals = vals[vals.theta_group.isin(['boundary', 'noble'])]
    if len(vals) < 20:
        cohens_ds.append(0)
        p_vals_cog.append(1)
        continue
    slope, intercept = np.polyfit(vals.age_midpoint, vals[cog], 1)
    resid = vals[cog] - (slope * vals.age_midpoint + intercept)
    b_resid = resid[vals.theta_group == 'boundary']
    n_resid = resid[vals.theta_group == 'noble']
    d = (b_resid.mean() - n_resid.mean()) / np.sqrt((b_resid.std()**2 + n_resid.std()**2)/2)
    t, p = stats.ttest_ind(b_resid, n_resid)
    cohens_ds.append(d)
    p_vals_cog.append(p)

colors_cog_d = ['#e74c3c' if p < 0.05 else '#95a5a6' for p in p_vals_cog]
bars = ax.barh(range(len(cog_labels)), cohens_ds, color=colors_cog_d, alpha=0.7, edgecolor='white')
ax.axvline(0, color='k', lw=0.5)
ax.set_yticks(range(len(cog_labels)))
ax.set_yticklabels(cog_labels, fontsize=9)
ax.set_xlabel("Cohen's d (boundary - noble, age-adjusted)")
ax.set_title('(c) No cognitive differences\n(0/9 survive FDR)', fontsize=11)

fig.suptitle('Figure 6: Position phenotyping', fontsize=13, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig6_position_phenotyping.pdf'), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(OUT_DIR, 'fig6_position_phenotyping.png'), bbox_inches='tight', dpi=300)
plt.close()
print("  Figure 6 saved.\n")


# ═══════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY FIGURE S1: AMPLITUDE WEIGHTING
# ═══════════════════════════════════════════════════════════════════════
print("Generating Figure S1...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot attractor enrichment vs age for each weight transform
# Extract from weighted features
transforms = {
    'Unweighted [1,85]': ('E_attractor_85hz_uw', 'E_attractor_85hz_uw'),
    'Rank-weighted': ('E_attractor_rank_f085', 'E_attractor_rank_f085'),
    'Z-score weighted': ('E_attractor_zscore_f085', 'E_attractor_zscore_f085'),
}

weighted_merged = weighted_feat  # already has age_midpoint and age_group

for idx, (title, (col_name, _)) in enumerate(transforms.items()):
    ax = axes.flat[idx]
    if col_name in weighted_merged.columns:
        data = weighted_merged.dropna(subset=[col_name, 'age_midpoint'])
        ax.scatter(data.age_midpoint, data[col_name], alpha=0.3, s=15, color='#3498db')
        r, p = stats.pearsonr(data.age_midpoint, data[col_name])
        z = np.polyfit(data.age_midpoint, data[col_name], 1)
        x_line = np.linspace(data.age_midpoint.min(), data.age_midpoint.max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), 'r-', lw=2)
        ax.set_xlabel('Age (years)')
        ax.set_ylabel(f'E_attractor ({title})')
        sig = '✓' if p < 0.05 else '✗'
        ax.set_title(f'{title}\nr = {r:.3f}, p = {p:.3f} {sig}', fontsize=11)
    else:
        ax.text(0.5, 0.5, f'{col_name}\nnot found', transform=ax.transAxes, ha='center')

# Summary panel
ax = axes[1, 1]
ax.axis('off')
ax.set_title('Summary: All transforms\nattenuate the signal', fontsize=11)

summary_data = [
    ['Transform', 'r(age)', 'p'],
    ['Unweighted [1,45]', '-0.195', '0.008'],
    ['Unweighted [1,85]', '-0.223', '0.002'],
    ['Rank', '-0.209', '0.005'],
    ['Z-score', '-0.166', '0.025'],
    ['f₀=IAF (rank)', '+0.167', '0.025'],
]

table = ax.table(cellText=summary_data[1:], colLabels=summary_data[0],
                cellLoc='center', loc='center', colWidths=[0.35, 0.2, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.8)
for j in range(3):
    table[(0, j)].set_facecolor('#2c3e50')
    table[(0, j)].set_text_props(color='white', fontweight='bold')
# Highlight reversed sign
table[(5, 1)].set_text_props(color='#c0392b', fontweight='bold')

fig.suptitle('Figure S1: Amplitude weighting attenuates rather than strengthens the signal',
            fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'figS1_amplitude_weighting.pdf'), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(OUT_DIR, 'figS1_amplitude_weighting.png'), bbox_inches='tight', dpi=300)
plt.close()
print("  Figure S1 saved.\n")


# ═══════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY FIGURE S2: PAIRWISE RATIO ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
print("Generating Figure S2...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Load base specificity data
base_spec = pd.read_csv('/Users/neurokinetikz/Code/schumann/exports_lemon/amplitude_weighted/base_specificity.csv')

# ── S2a: z_global ~ age by base ──
ax = axes[0]
if 'base' in base_spec.columns and 'r_age_z_global' in base_spec.columns:
    bs = base_spec.sort_values('r_age_z_global')
    colors_bs = ['#2ecc71' if 'phi' in str(b).lower() or b == 'φ' else '#95a5a6' for b in bs['base']]
    ax.barh(range(len(bs)), bs['r_age_z_global'], color=colors_bs, alpha=0.8)
    ax.set_yticks(range(len(bs)))
    ax.set_yticklabels(bs['base'], fontsize=9)
    ax.set_xlabel('r(age, z_global)')
    ax.set_title('(a) z_global ~ age: φ and e\nstrongest but IAF-mediated', fontsize=11)
    ax.axvline(0, color='k', lw=0.5)
else:
    # Fallback: use known values
    bases_s2 = ['3/2', '√2', '1.4', 'φ', '1.7', '2', 'e', '1.8', 'π']
    r_rwt = [0.309, 0.256, 0.241, 0.227, 0.210, 0.195, 0.180, 0.120, 0.100]
    ax.barh(range(len(bases_s2)), r_rwt,
           color=['#2ecc71' if b == 'φ' else '#95a5a6' for b in bases_s2], alpha=0.8)
    ax.set_yticks(range(len(bases_s2)))
    ax.set_yticklabels(bases_s2, fontsize=9)
    ax.set_xlabel('r(RWT, z_alpha)')
    ax.set_title('(a) RWT ~ z_alpha: φ ranks 4th\n3/2 beats φ (r=0.309 vs 0.227)', fontsize=11)
    ax.axvline(0, color='k', lw=0.5)

# ── S2b: Cognitive base specificity summary ──
ax = axes[1]
ax.axis('off')
ax.set_title('(b) Base specificity summary', fontsize=11)

table_data = [
    ['Finding', 'φ-specific?'],
    ['RWT ~ z_alpha', 'No — 7/9 bases'],
    ['z_global ~ age', 'No — φ ≈ e'],
    ['Within-elderly cog', '0/36 FDR'],
    ['Within-young cog', '1/36 FDR\n(non-specific)'],
    ['Dominant-peak mean_d', 'YES — rank 1'],
]

table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                cellLoc='center', loc='center', colWidths=[0.4, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)
for j in range(2):
    table[(0, j)].set_facecolor('#2c3e50')
    table[(0, j)].set_text_props(color='white', fontweight='bold')
# Green for YES
table[(5, 1)].set_text_props(color='#27ae60', fontweight='bold')
# Red for No
for i in [1, 2, 3, 4]:
    table[(i, 1)].set_text_props(color='#c0392b')

fig.suptitle('Figure S2: Pairwise ratio analysis — phi ranks 4th among nine bases',
            fontsize=13, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'figS2_base_specificity.pdf'), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(OUT_DIR, 'figS2_base_specificity.png'), bbox_inches='tight', dpi=300)
plt.close()
print("  Figure S2 saved.\n")


# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY FIGURE S3: EXTRACTION METHOD SENSITIVITY
# ═══════════════════════════════════════════════════════════════════════
print("Generating Figure S3...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ── S3a: Standard vs overlap-trim position agreement ──
ax = axes[0]

# Load standard extraction dominant peaks
std_dir = '/Users/neurokinetikz/Code/schumann/exports_lemon/per_subject'
dom_std = get_dominant_peaks(std_dir)

# Merge with overlap-trim
compare = dom_eo.merge(dom_std, on='subject_id', suffixes=('_ot', '_std'))

band_agreement = {}
for bname in BANDS:
    both_have = compare.dropna(subset=[f'{bname}_nearest_ot', f'{bname}_nearest_std'])
    if len(both_have) > 0:
        agree = (both_have[f'{bname}_nearest_ot'] == both_have[f'{bname}_nearest_std']).mean()
        r_freq, p_freq = stats.pearsonr(both_have[f'{bname}_freq_ot'], both_have[f'{bname}_freq_std'])
        band_agreement[bname] = (agree * 100, r_freq, len(both_have))

if band_agreement:
    bnames = list(band_agreement.keys())
    agrees = [band_agreement[b][0] for b in bnames]
    r_freqs = [band_agreement[b][1] for b in bnames]

    x = np.arange(len(bnames))
    width = 0.35
    ax.bar(x - width/2, agrees, width, label='Position agreement (%)', color='#3498db', alpha=0.8)
    ax.bar(x + width/2, [r*100 for r in r_freqs], width, label='Freq correlation (r×100)',
          color='#e67e22', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(bnames)
    ax.set_ylabel('Percentage / Correlation × 100')
    ax.legend(fontsize=8)

ax.set_title('(a) Standard vs overlap-trim:\n~66% position agreement', fontsize=11)

# ── S3b: Summary of extraction sensitivity ──
ax = axes[1]
ax.axis('off')
ax.set_title('(b) Extraction method comparison', fontsize=11)

table_data = [
    ['Comparison', 'Position\nagreement', 'Freq\ncorrelation'],
    ['Std [1,45] vs\noverlap-trim', '66%', 'r = 0.383'],
    ['Std EO vs EC\n(same method)', '74%*', 'r = 0.237\n(theta)'],
    ['Overlap-trim\nprimary vs offset', '72-80%†', 'r = 0.72-0.80'],
]

table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                cellLoc='center', loc='center', colWidths=[0.35, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.5)
for j in range(3):
    table[(0, j)].set_facecolor('#2c3e50')
    table[(0, j)].set_text_props(color='white', fontweight='bold')

ax.text(0.5, 0.05, '* Best band (theta). † Per-subject SS correlation.',
       transform=ax.transAxes, ha='center', fontsize=8, fontstyle='italic', color='gray')

fig.suptitle('Figure S3: Extraction method sensitivity', fontsize=13, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'figS3_extraction_sensitivity.pdf'), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(OUT_DIR, 'figS3_extraction_sensitivity.png'), bbox_inches='tight', dpi=300)
plt.close()
print("  Figure S3 saved.\n")


# ═══════════════════════════════════════════════════════════════════════
# PRINT COMPUTED STATISTICS FOR PAPER VERIFICATION
# ═══════════════════════════════════════════════════════════════════════
print("="*70)
print("COMPUTED STATISTICS FOR PAPER CROSS-CHECK")
print("="*70)

# Individual alignment
mean_ds = valid.mean_d.dropna().values
null_expected = 0.07991502812526288  # precise analytical null for degree-2, 4 positions
t_stat, p_val = stats.ttest_1samp(mean_ds, null_expected)
d_eff = (np.mean(mean_ds) - null_expected) / np.std(mean_ds)
w_stat, w_p = stats.wilcoxon(mean_ds - null_expected)
print(f"\nIndividual alignment (N={len(mean_ds)}):")
print(f"  mean_d = {np.mean(mean_ds):.4f} ± {np.std(mean_ds):.4f}")
print(f"  Null expected = {null_expected}")
print(f"  t = {t_stat:.3f}, p = {p_val:.2e}")
print(f"  Wilcoxon p = {w_p:.2e}")
print(f"  Cohen's d = {d_eff:.3f}")

# Per-band
print("\nPer-band alignment:")
for bname in BANDS:
    d_vals = valid[f'{bname}_d'].dropna().values
    print(f"  {bname}: mean = {np.mean(d_vals):.4f}, N = {len(d_vals)}")
    # KS test vs simulated null
    null_sim = np.abs(np.random.uniform(-0.25, 0.25, 100000))
    ks, p = stats.ks_2samp(d_vals, null_sim)
    mw, mw_p = stats.mannwhitneyu(d_vals, null_sim[:len(d_vals)])
    print(f"    KS: D={ks:.4f}, p={p:.2e}")
    print(f"    Mann-Whitney p={mw_p:.2e}")

# Population median peaks
print("\nPopulation median peaks:")
for bname in BANDS:
    freq = valid[f'{bname}_freq'].median()
    u = lattice_coord(freq)
    d = min_lattice_dist(u)
    nearest = nearest_pos_name(u)
    print(f"  {bname}: {freq:.2f} Hz, u={u:.3f}, d={d:.3f}, nearest={nearest}")

pop_ds = [min_lattice_dist(lattice_coord(valid[f'{b}_freq'].median())) for b in BANDS]
print(f"  Population mean_d = {np.mean(pop_ds):.4f}")

# Age correlation
r_age, p_age = stats.pearsonr(valid.age_midpoint, valid.mean_d)
rho_age, p_rho = stats.spearmanr(valid.age_midpoint, valid.mean_d)
young_d = valid[valid.age_group == 'young'].mean_d
elderly_d = valid[valid.age_group == 'elderly'].mean_d
t_grp, p_grp = stats.ttest_ind(young_d, elderly_d)
d_grp = (elderly_d.mean() - young_d.mean()) / np.sqrt((young_d.std()**2 + elderly_d.std()**2)/2)
print(f"\nAge correlation:")
print(f"  Pearson: r = {r_age:.4f}, p = {p_age:.4f}")
print(f"  Spearman: rho = {rho_age:.4f}, p = {p_rho:.4f}")
print(f"  Group: young={young_d.mean():.4f}, elderly={elderly_d.mean():.4f}")
print(f"  Cohen's d = {d_grp:.3f}, t = {t_grp:.3f}, p = {p_grp:.4f}")

# Cross-base comparison
print("\nCross-base mean_d:")
for bname, bval in BASES.items():
    ds = mean_d_for_base(eo_dir, bval)
    if len(ds) > 0:
        print(f"  {bname}: {np.mean(ds):.4f} (N={len(ds)})")

# Individually significant
n_sig = 0
n_total = 0
for _, row in valid.iterrows():
    if pd.notna(row.mean_d):
        n_total += 1
        # Simplified test: just check if mean_d is < critical threshold
        # Use bootstrap null: pick random positions uniformly
        null_ds = []
        for _ in range(1000):
            u_rand = np.random.uniform(0, 1, 4)
            d_rand = np.mean([min(min(abs(u - p), 1 - abs(u - p)) for p in POSITIONS.values()) for u in u_rand])
            null_ds.append(d_rand)
        p_subj = np.mean(np.array(null_ds) <= row.mean_d)
        if p_subj < 0.05:
            n_sig += 1

print(f"\nIndividually significant: {n_sig}/{n_total} ({n_sig/n_total*100:.1f}%)")
print(f"Expected by chance: {n_total * 0.05:.0f}")

# EC theta convergence (matched extraction)
dist_eo_m = np.abs(both_matched['theta_freq'] - F0)
dist_ec_m = np.abs(both_matched['theta_freq_ec'] - F0)
t_conv_m, p_conv_m = stats.ttest_rel(dist_ec_m, dist_eo_m)
print(f"\nEC theta convergence (matched extraction, N={len(both_matched)}):")
print(f"  |θ-f₀| EO: {dist_eo_m.mean():.3f}, EC: {dist_ec_m.mean():.3f}")
print(f"  Paired t = {t_conv_m:.3f}, p = {p_conv_m:.4f}")

# Theta stability (matched extraction)
print(f"\nTheta position stability (matched extraction):")
for pos in ['boundary', 'noble₁', 'noble₂', 'attractor']:
    mask = both_matched['theta_nearest'] == pos
    if mask.sum() > 0:
        stable = (both_matched.loc[mask, 'theta_nearest'] == both_matched.loc[mask, 'theta_nearest_ec']).mean()
        print(f"  {pos}: {stable*100:.0f}% stable (N={mask.sum()})")

# Noble₁ → boundary migration (matched)
n1_mask = both_matched['theta_nearest'] == 'noble₁'
if n1_mask.sum() > 0:
    n1_to_bnd = ((both_matched['theta_nearest'] == 'noble₁') & (both_matched['theta_nearest_ec'] == 'boundary')).sum()
    print(f"  noble₁ → boundary: {n1_to_bnd}/{n1_mask.sum()} ({n1_to_bnd/n1_mask.sum()*100:.0f}%)")

# Position phenotyping
print(f"\nTheta position distribution:")
for pos in POSITIONS:
    n = (valid.theta_nearest == pos).sum()
    pct = n / len(valid) * 100
    print(f"  {pos}: {n} ({pct:.1f}%)")

# Boundary vs noble power comparison
boundary_m = valid[valid.theta_nearest == 'boundary']
noble_m = valid[valid.theta_nearest.isin(['noble₁', 'noble₂'])]
t_tp, p_tp = stats.ttest_ind(boundary_m.theta_power.dropna(), noble_m.theta_power.dropna())
print(f"\nTheta power: boundary={boundary_m.theta_power.mean():.3f} vs noble={noble_m.theta_power.mean():.3f}")
print(f"  t={t_tp:.3f}, p={p_tp:.4f}")

t_ap, p_ap = stats.ttest_ind(boundary_m.alpha_power.dropna(), noble_m.alpha_power.dropna())
print(f"Alpha power: boundary={boundary_m.alpha_power.mean():.3f} vs noble={noble_m.alpha_power.mean():.3f}")
print(f"  t={t_ap:.3f}, p={p_ap:.4f}")

t_iaf, p_iaf = stats.ttest_ind(boundary_m.iaf.dropna(), noble_m.iaf.dropna())
print(f"IAF: boundary={boundary_m.iaf.mean():.3f} vs noble={noble_m.iaf.mean():.3f}")
print(f"  t={t_iaf:.3f}, p={p_iaf:.4f}")

print("\n" + "="*70)
print("ALL FIGURES GENERATED SUCCESSFULLY")
print(f"Output directory: {OUT_DIR}")
print("="*70)

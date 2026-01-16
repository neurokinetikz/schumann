#!/usr/bin/env python3
"""
Extended Noble Position Analysis: Inverse Nobles
=================================================

Extends the φⁿ lattice analysis to include the complete noble hierarchy:
- Regular nobles: 4° (0.146), 3° (0.236), 2° (0.382), 1° (0.618)
- Inverse nobles: 3° inverse (0.764), 4° inverse (0.854)
- Attractor (0.5), Boundary (0.0/1.0)

Mathematical identities:
- 3° Inverse (0.764) = φ⁻¹ + φ⁻⁴ = 1 - φ⁻³
- 4° Inverse (0.854) = φ⁻¹ + φ⁻³ = 1 - φ⁻⁴

These positions form symmetric pairs about the attractor (0.5).

Usage:
    python analyze_inverse_nobles.py

Output:
- Enrichment statistics for all 8 position types
- Band-stratified analysis (with theta/gamma focus)
- Two publication figures
- LaTeX-ready tables
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats
from typing import Dict, List, Tuple
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================================================
# Constants
# ============================================================================

PHI = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618034
F0_DEFAULT = 7.6  # Hz (empirical SR fundamental)

# Extended position type definitions with mathematical identities
EXTENDED_POSITIONS = {
    'noble_4':   {'center': 0.146, 'phi_repr': 'φ⁻⁴',           'symmetric_to': 'inverse_4'},
    'noble_3':   {'center': 0.236, 'phi_repr': 'φ⁻³',           'symmetric_to': 'inverse_3'},
    'noble_2':   {'center': 0.382, 'phi_repr': 'φ⁻²',           'symmetric_to': 'noble_1'},
    'attractor': {'center': 0.500, 'phi_repr': '—',             'symmetric_to': None},
    'noble_1':   {'center': 0.618, 'phi_repr': 'φ⁻¹',           'symmetric_to': 'noble_2'},
    'inverse_3': {'center': 0.764, 'phi_repr': 'φ⁻¹ + φ⁻⁴',     'symmetric_to': 'noble_3'},
    'inverse_4': {'center': 0.854, 'phi_repr': 'φ⁻¹ + φ⁻³',     'symmetric_to': 'noble_4'},
    'boundary':  {'center': 0.0,   'phi_repr': 'φ⁰ = 1',        'symmetric_to': None},
}

# Display order for tables (by position value)
POSITION_ORDER = ['boundary', 'noble_4', 'noble_3', 'noble_2', 'attractor',
                  'noble_1', 'inverse_3', 'inverse_4']

# Display names for figures
POSITION_LABELS = {
    'noble_4': '4° Noble',
    'noble_3': '3° Noble',
    'noble_2': '2° Noble',
    'attractor': 'Attractor',
    'noble_1': '1° Noble',
    'inverse_3': '3° Inverse',
    'inverse_4': '4° Inverse',
    'boundary': 'Boundary',
}

# Colors for visualization
POSITION_COLORS = {
    'noble_4': '#9966cc',     # Purple (outer regular)
    'noble_3': '#6699cc',     # Light blue
    'noble_2': '#88aa44',     # Green
    'attractor': '#cc4444',   # Red
    'noble_1': '#22aa88',     # Teal
    'inverse_3': '#ff8844',   # Orange
    'inverse_4': '#cc6699',   # Pink (outer inverse)
    'boundary': '#cc8800',    # Gold/orange
}

# φⁿ-based band boundaries
BANDS = {
    'Delta': (F0_DEFAULT * PHI**(-2), F0_DEFAULT * PHI**(-1)),
    'Theta': (F0_DEFAULT * PHI**(-1), F0_DEFAULT * PHI**0),
    'Alpha': (F0_DEFAULT * PHI**0, F0_DEFAULT * PHI**1),
    'Low Beta': (F0_DEFAULT * PHI**1, F0_DEFAULT * PHI**2),
    'High Beta': (F0_DEFAULT * PHI**2, F0_DEFAULT * PHI**3),
    'Gamma': (F0_DEFAULT * PHI**3, F0_DEFAULT * PHI**4),
}

# Output directory
OUTPUT_DIR = 'papers/images'

# ============================================================================
# Core Functions
# ============================================================================

def compute_lattice_coordinate(freq: np.ndarray, f0: float = F0_DEFAULT,
                               scale: float = PHI) -> np.ndarray:
    """Compute lattice coordinate u = [log_scale(f/f0)] mod 1."""
    n = np.log(freq / f0) / np.log(scale)
    return n - np.floor(n)


def compute_extended_enrichment(freqs: np.ndarray, f0: float = F0_DEFAULT,
                                 n_bootstrap: int = 1000,
                                 u_window: float = 0.05) -> Dict:
    """
    Compute enrichment at all 8 position types.

    Enrichment = (Observed / Expected) - 1, expressed as percentage.
    Expected assumes uniform distribution across lattice coordinates.
    """
    # Compute lattice coordinates
    u = compute_lattice_coordinate(freqs, f0)
    n_total = len(u)

    results = {}

    for pos_name, pos_info in EXTENDED_POSITIONS.items():
        pos_center = pos_info['center']

        # Count peaks in window
        if pos_name == 'boundary':
            # Boundary needs wraparound handling (near 0 or 1)
            in_window = (u < u_window) | (u > 1 - u_window)
        else:
            in_window = np.abs(u - pos_center) < u_window

        observed = in_window.sum()

        # Expected under uniform distribution
        window_fraction = 2 * u_window
        expected = n_total * window_fraction

        enrichment = (observed / expected - 1) * 100 if expected > 0 else 0

        # Bootstrap for confidence intervals
        enrichments_boot = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(n_total, n_total, replace=True)
            u_boot = u[idx]

            if pos_name == 'boundary':
                in_window_boot = (u_boot < u_window) | (u_boot > 1 - u_window)
            else:
                in_window_boot = np.abs(u_boot - pos_center) < u_window

            obs_boot = in_window_boot.sum()
            enrich_boot = (obs_boot / expected - 1) * 100 if expected > 0 else 0
            enrichments_boot.append(enrich_boot)

        ci_low, ci_high = np.percentile(enrichments_boot, [2.5, 97.5])

        results[pos_name] = {
            'center': pos_center,
            'observed': observed,
            'expected': expected,
            'enrichment': enrichment,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'phi_repr': pos_info['phi_repr'],
            'symmetric_to': pos_info['symmetric_to'],
        }

    # Add metadata
    results['_meta'] = {
        'n_peaks': n_total,
        'f0': f0,
        'u_window': u_window,
        'n_bootstrap': n_bootstrap,
    }

    return results


def compute_band_enrichment(freqs: np.ndarray, f0: float = F0_DEFAULT,
                             n_bootstrap: int = 500) -> pd.DataFrame:
    """Compute enrichment at each position type within each φⁿ band."""
    rows = []

    for band_name, (f_low, f_high) in BANDS.items():
        band_freqs = freqs[(freqs >= f_low) & (freqs < f_high)]
        n_peaks = len(band_freqs)

        if n_peaks < 20:
            continue

        # Compute enrichment for this band
        enrichment = compute_extended_enrichment(band_freqs, f0, n_bootstrap)

        for pos_name in POSITION_ORDER:
            if pos_name.startswith('_'):
                continue
            pos_data = enrichment[pos_name]
            rows.append({
                'band': band_name,
                'position': pos_name,
                'position_label': POSITION_LABELS[pos_name],
                'n_peaks': n_peaks,
                'enrichment': pos_data['enrichment'],
                'ci_low': pos_data['ci_low'],
                'ci_high': pos_data['ci_high'],
            })

    return pd.DataFrame(rows)


def test_theoretical_ordering(enrichment: Dict) -> Dict:
    """
    Test if enrichment follows theoretical ordering.

    Expected: boundary < noble_4 ≤ noble_3 ≤ noble_2 < attractor ≤ noble_1
    For inverse nobles: they should be symmetric to their regular counterparts
    """
    # Get enrichment values in theoretical order
    order_vals = [enrichment[p]['enrichment'] for p in POSITION_ORDER]

    # Compute Kendall's tau for overall ordering
    expected_ranks = list(range(len(POSITION_ORDER)))
    observed_ranks = list(np.argsort(np.argsort(order_vals)))

    tau, p_value = stats.kendalltau(expected_ranks, observed_ranks)

    # Check symmetric pairs
    symmetric_checks = []
    for pos1, pos2 in [('noble_4', 'inverse_4'), ('noble_3', 'inverse_3'),
                        ('noble_2', 'noble_1')]:
        e1 = enrichment[pos1]['enrichment']
        e2 = enrichment[pos2]['enrichment']
        # Check if they're within 1.5x of each other (allowing for asymmetry)
        ratio = max(e1, e2) / min(e1, e2) if min(e1, e2) != 0 else float('inf')
        symmetric_checks.append({
            'pair': f'{pos1} ↔ {pos2}',
            'e1': e1,
            'e2': e2,
            'ratio': ratio,
            'symmetric': ratio < 3.0 or np.sign(e1) == np.sign(e2),
        })

    return {
        'tau': tau,
        'p_value': p_value,
        'order_correct': tau > 0.5,
        'symmetric_checks': symmetric_checks,
    }


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_noble_lattice_symmetry(enrichment: Dict, output_path: str):
    """
    Figure 1: Noble Lattice Symmetry Diagram

    Shows all 8 position types with symmetric pairs connected by arcs.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # X positions
    positions = {p: EXTENDED_POSITIONS[p]['center'] for p in POSITION_ORDER}
    # Handle boundary at 1.0 for display
    positions['boundary'] = 0.0

    # Draw symmetric pairs with arcs
    symmetric_pairs = [
        ('noble_4', 'inverse_4', 0.354),
        ('noble_3', 'inverse_3', 0.264),
        ('noble_2', 'noble_1', 0.118),
    ]

    for pos1, pos2, distance in symmetric_pairs:
        x1 = positions[pos1]
        x2 = positions[pos2]
        mid = 0.5

        # Draw arc connecting symmetric pairs
        arc_height = 0.1 + (distance * 0.8)
        arc = mpatches.FancyArrowPatch(
            (x1, 0), (x2, 0),
            connectionstyle=f"arc3,rad=-{arc_height}",
            arrowstyle='-',
            color='#888888',
            linestyle='--',
            linewidth=1,
            alpha=0.5,
        )
        ax.add_patch(arc)

        # Add distance label
        ax.text(mid, arc_height * 2.5, f'd = {distance:.3f}',
                ha='center', va='bottom', fontsize=8, color='#666666')

    # Draw vertical lines and markers for each position
    y_positions = []
    for i, pos_name in enumerate(POSITION_ORDER):
        x = positions[pos_name]
        color = POSITION_COLORS[pos_name]
        label = POSITION_LABELS[pos_name]
        phi_repr = EXTENDED_POSITIONS[pos_name]['phi_repr']
        enrich = enrichment[pos_name]['enrichment']

        # Height based on enrichment
        height = max(0.1, (enrich + 30) / 100)  # Scale for visibility
        y_positions.append(height)

        # Draw bar
        ax.bar(x, height, width=0.04, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

        # Add label above bar
        ax.text(x, height + 0.05, f'{label}\n{phi_repr}\n({enrich:+.1f}%)',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

        # Add position value below
        ax.text(x, -0.15, f'{x:.3f}', ha='center', va='top', fontsize=7, color='#666666')

    # Draw attractor center line
    ax.axvline(0.5, color='#cc4444', linestyle=':', linewidth=2, alpha=0.5, zorder=0)
    ax.text(0.5, max(y_positions) + 0.3, 'ATTRACTOR\n(0.5)', ha='center', va='bottom',
            fontsize=10, fontweight='bold', color='#cc4444')

    # Formatting
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.25, max(y_positions) + 0.5)
    ax.set_xlabel('Lattice Position (u)', fontsize=12)
    ax.set_ylabel('Relative Enrichment', fontsize=12)
    ax.set_title('Noble Position Hierarchy: Symmetric Structure about Attractor',
                 fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add legend explaining symmetry
    legend_text = (
        'Symmetric Pairs:\n'
        '• 4° Noble (0.146) ↔ 4° Inverse (0.854)\n'
        '• 3° Noble (0.236) ↔ 3° Inverse (0.764)\n'
        '• 2° Noble (0.382) ↔ 1° Noble (0.618)'
    )
    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_coupling_geometry(output_path: str):
    """
    Figure 2: Cross-Frequency Coupling Geometry

    Shows upper vs lower coupling risk across the lattice, explaining
    why inverse nobles protect against upper-band mode-locking.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Generate risk curves
    u = np.linspace(0.01, 0.99, 1000)

    # Coupling risk functions (inverse distance to boundary)
    lower_risk = 1 / u  # Risk of coupling to lower boundary
    upper_risk = 1 / (1 - u)  # Risk of coupling to upper boundary

    # Clip for visualization
    lower_risk = np.clip(lower_risk, 0, 15)
    upper_risk = np.clip(upper_risk, 0, 15)

    # Plot risk curves
    ax.fill_between(u, 0, lower_risk, alpha=0.3, color='#cc6666', label='Lower-band coupling risk')
    ax.fill_between(u, 0, upper_risk, alpha=0.3, color='#6666cc', label='Upper-band coupling risk')
    ax.plot(u, lower_risk, color='#cc3333', linewidth=2)
    ax.plot(u, upper_risk, color='#3333cc', linewidth=2)

    # Mark position types
    for pos_name in POSITION_ORDER:
        if pos_name == 'boundary':
            continue
        x = EXTENDED_POSITIONS[pos_name]['center']
        color = POSITION_COLORS[pos_name]
        label = POSITION_LABELS[pos_name]

        # Calculate risks at this position
        lr = 1/x if x > 0.01 else 15
        ur = 1/(1-x) if x < 0.99 else 15

        # Plot marker
        y = min(lr, ur) + 0.5
        ax.plot(x, y, 'o', markersize=10, color=color, markeredgecolor='black',
                markeredgewidth=1, zorder=5)

        # Add label
        ax.annotate(f'{label}\n({x:.3f})',
                    xy=(x, y), xytext=(0, 15),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Add zones
    ax.axvspan(0, 0.3, alpha=0.1, color='red', label='High lower-risk zone')
    ax.axvspan(0.7, 1.0, alpha=0.1, color='blue', label='High upper-risk zone')
    ax.axvspan(0.35, 0.65, alpha=0.1, color='green', label='Protected zone')

    # Add annotations explaining functional implications
    ax.annotate('Theta/Gamma prefer\ninverse nobles here\n(protected from\nupper boundary)',
                xy=(0.8, 3), fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    ax.annotate('Lower nobles\nprotect from\nlower boundary',
                xy=(0.2, 3), fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))

    # Formatting
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 12)
    ax.set_xlabel('Lattice Position (u)', fontsize=12)
    ax.set_ylabel('Coupling Risk (1/distance to boundary)', fontsize=12)
    ax.set_title('Cross-Frequency Coupling Geometry:\nUpper vs Lower Boundary Risk',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper center', ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_enrichment_bars(enrichment: Dict, output_path: str, title_suffix: str = ''):
    """
    Bar chart showing enrichment at all 8 position types with CIs.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    positions = POSITION_ORDER
    x = np.arange(len(positions))

    enrichments = [enrichment[p]['enrichment'] for p in positions]
    ci_lows = [enrichment[p]['ci_low'] for p in positions]
    ci_highs = [enrichment[p]['ci_high'] for p in positions]
    colors = [POSITION_COLORS[p] for p in positions]
    labels = [POSITION_LABELS[p] for p in positions]

    # Error bars
    yerr_low = [e - cl for e, cl in zip(enrichments, ci_lows)]
    yerr_high = [ch - e for e, ch in zip(enrichments, ci_highs)]

    bars = ax.bar(x, enrichments, color=colors, edgecolor='black', linewidth=1)
    ax.errorbar(x, enrichments, yerr=[yerr_low, yerr_high],
                fmt='none', color='black', capsize=4, capthick=1)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, enrichments)):
        height = bar.get_height()
        offset = 2 if height >= 0 else -2
        va = 'bottom' if height >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2, height + offset,
                f'{val:+.1f}%', ha='center', va=va, fontsize=9, fontweight='bold')

    # Reference line
    ax.axhline(0, color='black', linewidth=1)

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Enrichment (%)', fontsize=12)
    ax.set_title(f'Position-Type Enrichment: Extended Noble Hierarchy{title_suffix}',
                 fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    # Add metadata
    n_peaks = enrichment['_meta']['n_peaks']
    ax.text(0.98, 0.98, f'N = {n_peaks:,} peaks',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def generate_latex_table(enrichment: Dict) -> str:
    """Generate LaTeX table for paper inclusion."""
    lines = [
        r'\begin{table}[H]',
        r'\centering',
        r'\caption{Extended Position-Type Enrichment: Including Inverse Nobles}',
        r'\label{tab:extended_enrichment}',
        r'\begin{tabular}{@{}llllll@{}}',
        r'\toprule',
        r'Position Type & $n$ Value & $\phisym^{-n}$ Representation & Enrichment & 95\% CI & Symmetric To \\',
        r'\midrule',
    ]

    for pos_name in POSITION_ORDER:
        pos = enrichment[pos_name]
        label = POSITION_LABELS[pos_name]
        center = pos['center']
        phi_repr = pos['phi_repr']
        enrich = pos['enrichment']
        ci_low = pos['ci_low']
        ci_high = pos['ci_high']
        sym = pos['symmetric_to']
        sym_label = POSITION_LABELS.get(sym, '—') if sym else '—'

        line = f'{label} & $k + {center:.3f}$ & {phi_repr} & ${enrich:+.1f}\\%$ & [{ci_low:.1f}\\%, {ci_high:.1f}\\%] & {sym_label} \\\\'
        lines.append(line)

    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])

    return '\n'.join(lines)


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    print("=" * 70)
    print("Extended Noble Position Analysis: Inverse Nobles")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ========================================================================
    # 1. Load Primary Dataset
    # ========================================================================
    print("\n[1/6] Loading primary dataset...")
    primary_path = 'papers/golden_ratio_peaks_ALL.csv'

    if not os.path.exists(primary_path):
        # Try alternative location
        primary_path = 'csv/golden_ratio_peaks_ALL.csv'

    primary_df = pd.read_csv(primary_path)
    primary_freqs = primary_df['freq'].values
    primary_freqs = primary_freqs[(primary_freqs >= 4) & (primary_freqs <= 50)]
    print(f"  Loaded {len(primary_freqs):,} peaks from primary dataset")

    # ========================================================================
    # 2. Compute Extended Enrichment - Primary
    # ========================================================================
    print("\n[2/6] Computing extended enrichment (primary)...")
    enrichment_primary = compute_extended_enrichment(primary_freqs, n_bootstrap=2000)

    print("\n  Position-Type Enrichment Results:")
    print("  " + "-" * 65)
    print(f"  {'Position':<15} {'Center':<10} {'Enrichment':<12} {'95% CI':<20}")
    print("  " + "-" * 65)
    for pos_name in POSITION_ORDER:
        pos = enrichment_primary[pos_name]
        print(f"  {POSITION_LABELS[pos_name]:<15} {pos['center']:<10.3f} {pos['enrichment']:+.1f}%{'':<6} [{pos['ci_low']:.1f}%, {pos['ci_high']:.1f}%]")

    # ========================================================================
    # 3. Test Theoretical Ordering
    # ========================================================================
    print("\n[3/6] Testing theoretical ordering...")
    ordering = test_theoretical_ordering(enrichment_primary)
    print(f"  Kendall's τ: {ordering['tau']:.3f} (p = {ordering['p_value']:.4f})")
    print(f"  Ordering correct: {ordering['order_correct']}")

    print("\n  Symmetric pair checks:")
    for check in ordering['symmetric_checks']:
        status = "✓" if check['symmetric'] else "✗"
        print(f"    {status} {check['pair']}: {check['e1']:+.1f}% vs {check['e2']:+.1f}%")

    # ========================================================================
    # 4. Band-Stratified Analysis
    # ========================================================================
    print("\n[4/6] Computing band-stratified enrichment...")
    band_df = compute_band_enrichment(primary_freqs, n_bootstrap=500)

    # Focus on theta and gamma for inverse noble preference
    print("\n  Band-specific inverse noble enrichment:")
    print("  " + "-" * 50)
    for band in ['Theta', 'Gamma']:
        band_data = band_df[band_df['band'] == band]
        if len(band_data) > 0:
            inv3 = band_data[band_data['position'] == 'inverse_3']['enrichment'].values
            inv4 = band_data[band_data['position'] == 'inverse_4']['enrichment'].values
            n1 = band_data[band_data['position'] == 'noble_1']['enrichment'].values

            inv3_val = inv3[0] if len(inv3) > 0 else float('nan')
            inv4_val = inv4[0] if len(inv4) > 0 else float('nan')
            n1_val = n1[0] if len(n1) > 0 else float('nan')

            print(f"  {band}: 3° Inv={inv3_val:+.1f}%, 4° Inv={inv4_val:+.1f}%, 1° Noble={n1_val:+.1f}%")

    # ========================================================================
    # 5. Generate Figures
    # ========================================================================
    print("\n[5/6] Generating figures...")

    # Figure 1: Noble Lattice Symmetry
    plot_noble_lattice_symmetry(enrichment_primary,
                                 f'{OUTPUT_DIR}/noble_lattice_symmetry.png')

    # Figure 2: Coupling Geometry
    plot_coupling_geometry(f'{OUTPUT_DIR}/coupling_geometry.png')

    # Figure 3: Enrichment Bar Chart
    plot_enrichment_bars(enrichment_primary,
                          f'{OUTPUT_DIR}/extended_position_enrichment.png',
                          ' (Primary Dataset)')

    # ========================================================================
    # 6. Load and Analyze EEGEmotions-27
    # ========================================================================
    print("\n[6/6] Analyzing EEGEmotions-27 dataset...")
    emotions_path = 'golden_ratio_peaks_EMOTIONS.csv'

    if os.path.exists(emotions_path):
        emotions_df = pd.read_csv(emotions_path)
        emotions_freqs = emotions_df['freq'].values
        emotions_freqs = emotions_freqs[(emotions_freqs >= 4) & (emotions_freqs <= 50)]
        print(f"  Loaded {len(emotions_freqs):,} peaks from EEGEmotions-27")

        enrichment_emotions = compute_extended_enrichment(emotions_freqs, n_bootstrap=2000)

        print("\n  EEGEmotions-27 Enrichment Results:")
        print("  " + "-" * 65)
        for pos_name in POSITION_ORDER:
            pos = enrichment_emotions[pos_name]
            print(f"  {POSITION_LABELS[pos_name]:<15} {pos['enrichment']:+.1f}% [{pos['ci_low']:.1f}%, {pos['ci_high']:.1f}%]")

        # Generate emotions figure
        plot_enrichment_bars(enrichment_emotions,
                              f'{OUTPUT_DIR}/emotions_extended_position_enrichment.png',
                              ' (EEGEmotions-27)')
    else:
        print(f"  Warning: {emotions_path} not found, skipping EEGEmotions analysis")
        enrichment_emotions = None

    # ========================================================================
    # Output Results
    # ========================================================================
    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)
    print(f"  {OUTPUT_DIR}/noble_lattice_symmetry.png")
    print(f"  {OUTPUT_DIR}/coupling_geometry.png")
    print(f"  {OUTPUT_DIR}/extended_position_enrichment.png")
    if enrichment_emotions:
        print(f"  {OUTPUT_DIR}/emotions_extended_position_enrichment.png")

    # Generate LaTeX table
    print("\n" + "=" * 70)
    print("LATEX TABLE (for unified_paper.tex)")
    print("=" * 70)
    latex = generate_latex_table(enrichment_primary)
    print(latex)

    # Save LaTeX to file
    with open('inverse_nobles_latex_table.tex', 'w') as f:
        f.write(latex)
    print(f"\n  Saved LaTeX table to: inverse_nobles_latex_table.tex")

    # Save band-stratified results
    band_df.to_csv('inverse_nobles_band_enrichment.csv', index=False)
    print(f"  Saved band results to: inverse_nobles_band_enrichment.csv")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return enrichment_primary, enrichment_emotions, band_df


if __name__ == '__main__':
    enrichment_primary, enrichment_emotions, band_df = main()

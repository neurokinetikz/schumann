#!/usr/bin/env python3
"""
Bandwidth-Normalized Enrichment Analysis

Tests whether gamma's dramatically stronger φ^n alignment is an artifact of
bandwidth differences between bands.

Key insight: Each φ^n band has width proportional to φ^n:
- Delta: ~1.8 Hz wide
- Theta: ~2.9 Hz wide
- Alpha: ~4.7 Hz wide
- Low Beta: ~7.6 Hz wide
- High Beta: ~12.3 Hz wide
- Gamma: ~19.9 Hz wide

If gamma simply has more peaks because it's wider, normalizing by bandwidth
should equalize the enrichment values.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Constants
PHI = 1.618033988749895
F0 = 7.6  # Hz

# Band definitions (phi^n boundaries)
BANDS = [
    ('Delta', F0 * PHI**(-2), F0 * PHI**(-1)),      # ~2.9-4.7 Hz
    ('Theta', F0 * PHI**(-1), F0 * PHI**0),          # ~4.7-7.6 Hz
    ('Alpha', F0 * PHI**0, F0 * PHI**1),             # ~7.6-12.3 Hz
    ('Low Beta', F0 * PHI**1, F0 * PHI**2),          # ~12.3-19.9 Hz
    ('High Beta', F0 * PHI**2, F0 * PHI**3),         # ~19.9-32.2 Hz
    ('Gamma', F0 * PHI**3, F0 * PHI**4),             # ~32.2-52.1 Hz
]

# Lattice positions
POSITIONS = {
    'Boundary': 0.0,
    '2° Noble': 0.382,
    'Attractor': 0.5,
    '1° Noble': 0.618,
}

U_WINDOW = 0.05  # Window width for position enrichment


def compute_lattice_coordinate(freq, f0=F0):
    """Compute lattice coordinate u = [log_phi(f/f0)] mod 1."""
    n = np.log(freq / f0) / np.log(PHI)
    return n % 1


def compute_enrichment_at_position(u_values, position, window=U_WINDOW):
    """Compute enrichment at a given lattice position."""
    n = len(u_values)

    # Handle boundary wrapping
    if position < window:
        mask = (u_values < position + window) | (u_values > 1 - window + position)
    elif position > 1 - window:
        mask = (u_values > position - window) | (u_values < position + window - 1)
    else:
        mask = np.abs(u_values - position) < window

    observed = mask.sum()
    expected = n * (2 * window)

    enrichment = (observed / expected - 1) * 100 if expected > 0 else 0
    z_score = (observed - expected) / np.sqrt(expected) if expected > 0 else 0

    return enrichment, z_score, observed, expected


def main():
    # Load peaks
    peaks_df = pd.read_csv('golden_ratio_peaks_ALL_EMOTIV.csv')
    freqs = peaks_df['freq'].values
    print(f"Loaded {len(freqs):,} peaks")
    print("=" * 90)

    # Compute band statistics
    results = []

    print("\n" + "=" * 90)
    print("BAND WIDTH ANALYSIS")
    print("=" * 90)
    print(f"\n{'Band':<12} {'Freq Range':<18} {'Width (Hz)':>12} {'N_peaks':>10} {'Peaks/Hz':>12}")
    print("-" * 70)

    for band_name, f_low, f_high in BANDS:
        bandwidth = f_high - f_low
        mask = (freqs >= f_low) & (freqs < f_high)
        n_peaks = mask.sum()
        peaks_per_hz = n_peaks / bandwidth

        print(f"{band_name:<12} {f_low:.1f}-{f_high:.1f} Hz      {bandwidth:>10.2f} {n_peaks:>10,} {peaks_per_hz:>12.1f}")

        # Compute lattice coordinates
        band_freqs = freqs[mask]
        u = compute_lattice_coordinate(band_freqs)

        row = {
            'Band': band_name,
            'f_low': f_low,
            'f_high': f_high,
            'bandwidth': bandwidth,
            'n_peaks': n_peaks,
            'peaks_per_hz': peaks_per_hz,
        }

        # Compute enrichment at each position
        for pos_name, pos_val in POSITIONS.items():
            enrichment, z_score, obs, exp = compute_enrichment_at_position(u, pos_val)
            row[f'{pos_name}_enrich'] = enrichment
            row[f'{pos_name}_z'] = z_score
            row[f'{pos_name}_obs'] = obs
            row[f'{pos_name}_exp'] = exp

        results.append(row)

    results_df = pd.DataFrame(results)

    # ==========================================================================
    # BANDWIDTH-NORMALIZED ANALYSIS
    # ==========================================================================
    print("\n" + "=" * 90)
    print("BANDWIDTH-NORMALIZED ENRICHMENT")
    print("=" * 90)
    print("\nKey question: Is gamma's strong enrichment due to wider bandwidth?")
    print("Metric: (Observed - Expected) / Bandwidth = 'Excess peaks per Hz'\n")

    print(f"{'Band':<12} {'Width':>8} | {'1°Noble':>10} {'Attr':>10} {'2°Noble':>10} {'Bndry':>10} | {'1°Noble/Hz':>12} {'Bndry/Hz':>12}")
    print("-" * 100)

    normalized_results = []

    for _, row in results_df.iterrows():
        band = row['Band']
        bw = row['bandwidth']

        # Raw enrichment
        noble1_e = row['1° Noble_enrich']
        attr_e = row['Attractor_enrich']
        noble2_e = row['2° Noble_enrich']
        bndry_e = row['Boundary_enrich']

        # Normalized: excess peaks per Hz
        # excess = observed - expected
        noble1_excess = row['1° Noble_obs'] - row['1° Noble_exp']
        bndry_excess = row['Boundary_obs'] - row['Boundary_exp']

        noble1_per_hz = noble1_excess / bw
        bndry_per_hz = bndry_excess / bw

        print(f"{band:<12} {bw:>7.1f}Hz | {noble1_e:>+9.1f}% {attr_e:>+9.1f}% {noble2_e:>+9.1f}% {bndry_e:>+9.1f}% | {noble1_per_hz:>+11.1f} {bndry_per_hz:>+11.1f}")

        normalized_results.append({
            'Band': band,
            'Bandwidth': bw,
            '1°Noble_%': noble1_e,
            'Attractor_%': attr_e,
            '2°Noble_%': noble2_e,
            'Boundary_%': bndry_e,
            '1°Noble_excess': noble1_excess,
            'Boundary_excess': bndry_excess,
            '1°Noble_per_Hz': noble1_per_hz,
            'Boundary_per_Hz': bndry_per_hz,
        })

    norm_df = pd.DataFrame(normalized_results)

    # ==========================================================================
    # PEAK DENSITY ANALYSIS
    # ==========================================================================
    print("\n" + "=" * 90)
    print("PEAK DENSITY ANALYSIS")
    print("=" * 90)
    print("\nPeaks per Hz by band (raw density, not enrichment):\n")

    for _, row in results_df.iterrows():
        band = row['Band']
        density = row['peaks_per_hz']
        bar = '█' * int(density / 100)
        print(f"  {band:<12}: {density:>7.1f} peaks/Hz  {bar}")

    # Check if density correlates with enrichment
    densities = results_df['peaks_per_hz'].values
    noble_enrichments = results_df['1° Noble_enrich'].values

    r_density_noble, p_density = stats.pearsonr(densities, noble_enrichments)
    print(f"\nCorrelation: peak density vs 1° Noble enrichment: r={r_density_noble:.3f}, p={p_density:.3f}")

    # ==========================================================================
    # RELATIVE ENRICHMENT (% above uniform, normalized by band mean)
    # ==========================================================================
    print("\n" + "=" * 90)
    print("RELATIVE ENRICHMENT ANALYSIS")
    print("=" * 90)
    print("\nEnrichment relative to band's own baseline:")
    print("(This controls for different baseline peak rates across bands)\n")

    # For each band, compute enrichment as % deviation from that band's mean
    print(f"{'Band':<12} | {'1°Noble Raw':>12} {'1°Noble Rel':>12} | {'Boundary Raw':>13} {'Boundary Rel':>13}")
    print("-" * 75)

    for _, row in results_df.iterrows():
        band = row['Band']
        n_peaks = row['n_peaks']

        # Expected under uniform = n_peaks * window_fraction
        # Actual enrichment is already computed
        noble1_e = row['1° Noble_enrich']
        bndry_e = row['Boundary_enrich']

        # Relative enrichment: enrichment / |mean enrichment across positions|
        mean_abs_enrich = np.mean([
            abs(row['Boundary_enrich']),
            abs(row['2° Noble_enrich']),
            abs(row['Attractor_enrich']),
            abs(row['1° Noble_enrich'])
        ])

        noble1_rel = noble1_e / mean_abs_enrich if mean_abs_enrich > 0 else 0
        bndry_rel = bndry_e / mean_abs_enrich if mean_abs_enrich > 0 else 0

        print(f"{band:<12} | {noble1_e:>+11.1f}% {noble1_rel:>+11.2f}x | {bndry_e:>+12.1f}% {bndry_rel:>+12.2f}x")

    # ==========================================================================
    # Z-SCORE NORMALIZATION
    # ==========================================================================
    print("\n" + "=" * 90)
    print("Z-SCORE ANALYSIS (Sample-Size Corrected)")
    print("=" * 90)
    print("\nZ-scores account for sample size. If gamma's effect is due to more peaks,")
    print("z-scores should be similar across bands. If gamma is genuinely different,")
    print("z-scores will still be much larger.\n")

    print(f"{'Band':<12} {'N_peaks':>10} | {'1°Noble z':>12} {'Attr z':>12} {'2°Noble z':>12} {'Bndry z':>12}")
    print("-" * 80)

    for _, row in results_df.iterrows():
        band = row['Band']
        n = row['n_peaks']
        z_noble = row['1° Noble_z']
        z_attr = row['Attractor_z']
        z_noble2 = row['2° Noble_z']
        z_bndry = row['Boundary_z']

        print(f"{band:<12} {n:>10,} | {z_noble:>+11.1f} {z_attr:>+11.1f} {z_noble2:>+11.1f} {z_bndry:>+11.1f}")

    print("\nInterpretation:")
    print("  z > 3.29: p < 0.001 (***)")
    print("  z > 2.58: p < 0.01 (**)")
    print("  z > 1.96: p < 0.05 (*)")

    # ==========================================================================
    # KEY FINDING: NORMALIZED COMPARISON
    # ==========================================================================
    print("\n" + "=" * 90)
    print("KEY FINDING: BANDWIDTH NORMALIZATION RESULTS")
    print("=" * 90)

    gamma_noble_pct = norm_df[norm_df['Band'] == 'Gamma']['1°Noble_%'].values[0]
    gamma_noble_per_hz = norm_df[norm_df['Band'] == 'Gamma']['1°Noble_per_Hz'].values[0]

    other_noble_pct = norm_df[norm_df['Band'] != 'Gamma']['1°Noble_%'].values
    other_noble_per_hz = norm_df[norm_df['Band'] != 'Gamma']['1°Noble_per_Hz'].values

    mean_other_pct = np.mean(other_noble_pct)
    mean_other_per_hz = np.mean(other_noble_per_hz)

    ratio_raw = gamma_noble_pct / mean_other_pct if mean_other_pct != 0 else float('inf')
    ratio_norm = gamma_noble_per_hz / mean_other_per_hz if mean_other_per_hz != 0 else float('inf')

    print(f"\n1° Noble enrichment comparison (Gamma vs others):")
    print(f"  Raw enrichment:        Gamma {gamma_noble_pct:+.1f}% vs others mean {mean_other_pct:+.1f}% → ratio {ratio_raw:.1f}x")
    print(f"  Per-Hz normalization:  Gamma {gamma_noble_per_hz:+.1f}/Hz vs others mean {mean_other_per_hz:+.1f}/Hz → ratio {ratio_norm:.1f}x")

    if abs(ratio_norm) < abs(ratio_raw) * 0.5:
        print("\n  ⚠️  BANDWIDTH EXPLAINS SIGNIFICANT PORTION OF GAMMA'S EFFECT")
        print("      The per-Hz ratio is much smaller than the raw ratio.")
    elif abs(ratio_norm) > abs(ratio_raw) * 0.8:
        print("\n  ✓  BANDWIDTH DOES NOT EXPLAIN GAMMA'S EFFECT")
        print("      The per-Hz ratio is similar to the raw ratio.")
        print("      Gamma's strong enrichment persists after normalization.")

    # ==========================================================================
    # VISUALIZATION
    # ==========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    band_names = [b[0] for b in BANDS]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # Panel A: Raw enrichment
    ax = axes[0, 0]
    x = np.arange(len(band_names))
    width = 0.35

    noble_vals = results_df['1° Noble_enrich'].values
    bndry_vals = results_df['Boundary_enrich'].values

    ax.bar(x - width/2, noble_vals, width, label='1° Noble', color='#2ca02c', alpha=0.8)
    ax.bar(x + width/2, bndry_vals, width, label='Boundary', color='#d62728', alpha=0.8)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(band_names, rotation=45, ha='right')
    ax.set_ylabel('Enrichment (%)')
    ax.set_title('A. Raw Enrichment by Band', fontweight='bold')
    ax.legend()
    ax.set_ylim(-80, 160)

    # Panel B: Bandwidth-normalized (excess peaks per Hz)
    ax = axes[0, 1]
    noble_per_hz = norm_df['1°Noble_per_Hz'].values
    bndry_per_hz = norm_df['Boundary_per_Hz'].values

    ax.bar(x - width/2, noble_per_hz, width, label='1° Noble', color='#2ca02c', alpha=0.8)
    ax.bar(x + width/2, bndry_per_hz, width, label='Boundary', color='#d62728', alpha=0.8)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(band_names, rotation=45, ha='right')
    ax.set_ylabel('Excess Peaks per Hz')
    ax.set_title('B. Bandwidth-Normalized Enrichment', fontweight='bold')
    ax.legend()

    # Panel C: Z-scores (sample-size corrected)
    ax = axes[1, 0]
    z_noble = results_df['1° Noble_z'].values
    z_bndry = results_df['Boundary_z'].values

    ax.bar(x - width/2, z_noble, width, label='1° Noble', color='#2ca02c', alpha=0.8)
    ax.bar(x + width/2, z_bndry, width, label='Boundary', color='#d62728', alpha=0.8)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.axhline(3.29, color='orange', linestyle=':', linewidth=1, label='p<0.001')
    ax.axhline(-3.29, color='orange', linestyle=':', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(band_names, rotation=45, ha='right')
    ax.set_ylabel('Z-score')
    ax.set_title('C. Z-Scores (Sample-Size Corrected)', fontweight='bold')
    ax.legend(loc='upper left')

    # Panel D: Peak density by band
    ax = axes[1, 1]
    densities = results_df['peaks_per_hz'].values
    bars = ax.bar(x, densities, color=colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(band_names, rotation=45, ha='right')
    ax.set_ylabel('Peaks per Hz')
    ax.set_title('D. Peak Density by Band', fontweight='bold')

    # Add bandwidth labels
    for i, (bar, (name, f_lo, f_hi)) in enumerate(zip(bars, BANDS)):
        bw = f_hi - f_lo
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{bw:.1f} Hz', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('phi_bandwidth_normalized_analysis.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("\nSaved: phi_bandwidth_normalized_analysis.png")

    # Save results
    norm_df.to_csv('phi_bandwidth_normalized_stats.csv', index=False)
    print("Saved: phi_bandwidth_normalized_stats.csv")

    # ==========================================================================
    # EFFECT SIZE COMPARISON
    # ==========================================================================
    print("\n" + "=" * 90)
    print("EFFECT SIZE SUMMARY")
    print("=" * 90)

    print("\nComparing Gamma to mean of other bands:")
    print(f"\n  Metric                    Gamma        Others Mean      Ratio")
    print(f"  {'-'*60}")
    print(f"  1°Noble enrichment (%)    {gamma_noble_pct:>+8.1f}      {mean_other_pct:>+8.1f}         {ratio_raw:>6.1f}x")
    print(f"  1°Noble excess/Hz         {gamma_noble_per_hz:>+8.1f}      {mean_other_per_hz:>+8.1f}         {ratio_norm:>6.1f}x")

    gamma_z = results_df[results_df['Band'] == 'Gamma']['1° Noble_z'].values[0]
    other_z = results_df[results_df['Band'] != 'Gamma']['1° Noble_z'].values
    mean_other_z = np.mean(other_z)
    ratio_z = gamma_z / mean_other_z if mean_other_z != 0 else float('inf')

    print(f"  1°Noble z-score           {gamma_z:>+8.1f}      {mean_other_z:>+8.1f}         {ratio_z:>6.1f}x")

    print("\n" + "=" * 90)
    print("CONCLUSION")
    print("=" * 90)

    # Determine if bandwidth explains the difference
    if abs(ratio_norm) < abs(ratio_raw) * 0.3:
        conclusion = "BANDWIDTH LARGELY EXPLAINS gamma's apparent advantage"
    elif abs(ratio_norm) < abs(ratio_raw) * 0.6:
        conclusion = "BANDWIDTH PARTIALLY EXPLAINS gamma's advantage"
    else:
        conclusion = "GAMMA'S EFFECT PERSISTS after bandwidth normalization"

    print(f"\n  {conclusion}")
    print(f"\n  Raw ratio (Gamma/others):              {ratio_raw:.1f}x")
    print(f"  Bandwidth-normalized ratio:            {ratio_norm:.1f}x")
    print(f"  Z-score ratio:                         {ratio_z:.1f}x")

    if ratio_z > 10:
        print(f"\n  The z-score ratio of {ratio_z:.1f}x confirms gamma is genuinely different,")
        print(f"  not just an artifact of bandwidth or sample size.")


if __name__ == '__main__':
    main()

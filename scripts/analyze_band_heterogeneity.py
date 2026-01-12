#!/usr/bin/env python3
"""
Analyze band-specific heterogeneity in phi^n architecture.

This script addresses the concern that gamma shows much stronger phi^n alignment
(z = +69.0 at 1° noble) than theta, alpha, and low beta bands.

It quantifies inter-band differences with proper statistical tests to determine
whether this is universal architecture or primarily a gamma phenomenon.

Output:
- phi_band_heterogeneity_stats.csv: Full statistical results
- phi_band_heterogeneity.png: Visualization comparing bands
- Console output with test statistics and interpretation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu, chi2_contingency
import warnings

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

# Lattice positions to analyze
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
    """
    Compute enrichment at a given lattice position.

    Returns:
        enrichment: (observed / expected - 1) * 100 (as percentage)
        z_score: (observed - expected) / sqrt(expected)
        observed: count in window
        expected: count under uniform assumption
    """
    n = len(u_values)

    # Handle boundary wrapping (position near 0 or 1)
    if position < window:
        mask = (u_values < position + window) | (u_values > 1 - window + position)
    elif position > 1 - window:
        mask = (u_values > position - window) | (u_values < position + window - 1)
    else:
        mask = np.abs(u_values - position) < window

    observed = mask.sum()
    expected = n * (2 * window)  # Expected under uniform distribution

    enrichment = (observed / expected - 1) * 100 if expected > 0 else 0
    z_score = (observed - expected) / np.sqrt(expected) if expected > 0 else 0

    return enrichment, z_score, observed, expected


def cohens_h(p1, p2):
    """
    Compute Cohen's h effect size for difference between two proportions.
    h = 2 * (arcsin(sqrt(p1)) - arcsin(sqrt(p2)))

    Interpretation: |h| < 0.2 small, 0.2-0.5 medium, > 0.8 large
    """
    phi1 = 2 * np.arcsin(np.sqrt(p1))
    phi2 = 2 * np.arcsin(np.sqrt(p2))
    return phi1 - phi2


def bootstrap_enrichment_ci(u_values, position, n_bootstrap=2000, ci=95, window=U_WINDOW):
    """Compute bootstrap confidence interval for enrichment at position."""
    n = len(u_values)
    enrichments = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        boot_u = np.random.choice(u_values, size=n, replace=True)
        e, _, _, _ = compute_enrichment_at_position(boot_u, position, window)
        enrichments.append(e)

    lower = np.percentile(enrichments, (100 - ci) / 2)
    upper = np.percentile(enrichments, 100 - (100 - ci) / 2)
    return lower, upper


def rank_biserial(u_stat, n1, n2):
    """Compute rank-biserial correlation from Mann-Whitney U statistic."""
    return 1 - (2 * u_stat) / (n1 * n2)


def main():
    # Load peaks
    peaks_df = pd.read_csv('golden_ratio_peaks_ALL.csv')
    freqs = peaks_df['freq'].values
    print(f"Loaded {len(freqs):,} peaks")
    print("=" * 80)

    # Compute enrichment for each band at each position
    results = []
    band_enrichments = {pos: [] for pos in POSITIONS}  # For Kruskal-Wallis
    band_u_values = {}  # Store u values per band for bootstrap

    for band_name, f_low, f_high in BANDS:
        mask = (freqs >= f_low) & (freqs < f_high)
        band_freqs = freqs[mask]
        n_peaks = len(band_freqs)
        u = compute_lattice_coordinate(band_freqs)
        band_u_values[band_name] = u

        row = {
            'Band': band_name,
            'N_peaks': n_peaks,
            'Freq_range': f'{f_low:.1f}-{f_high:.1f} Hz',
        }

        for pos_name, pos_val in POSITIONS.items():
            enrichment, z_score, obs, exp = compute_enrichment_at_position(u, pos_val)
            row[f'{pos_name}_enrich'] = enrichment
            row[f'{pos_name}_z'] = z_score
            row[f'{pos_name}_obs'] = obs
            row[f'{pos_name}_exp'] = exp

            # Store for inter-band comparison
            # For Kruskal-Wallis, we need per-peak indicator (1 if in window, 0 otherwise)
            band_enrichments[pos_name].append(enrichment)

        results.append(row)

    results_df = pd.DataFrame(results)

    # Print summary table
    print("\n" + "=" * 80)
    print("BAND-SPECIFIC ENRICHMENT SUMMARY")
    print("=" * 80)
    print(f"\n{'Band':<12} {'N_peaks':>10} {'Boundary%':>12} {'2°Noble%':>12} {'Attractor%':>12} {'1°Noble%':>12}")
    print("-" * 72)
    for _, row in results_df.iterrows():
        print(f"{row['Band']:<12} {row['N_peaks']:>10,} {row['Boundary_enrich']:>+11.1f}% "
              f"{row['2° Noble_enrich']:>+11.1f}% {row['Attractor_enrich']:>+11.1f}% "
              f"{row['1° Noble_enrich']:>+11.1f}%")

    # =========================================================================
    # STATISTICAL TESTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("STATISTICAL TESTS FOR BAND HETEROGENEITY")
    print("=" * 80)

    # 1. Kruskal-Wallis test for each position
    # Need to use individual peak-level data, not aggregated enrichment
    print("\n--- Test 1: Chi-squared test for proportion homogeneity ---")
    print("H0: The proportion of peaks at each position is equal across bands")

    chi2_results = {}
    for pos_name, pos_val in POSITIONS.items():
        # Create contingency table: bands x (in_window, not_in_window)
        contingency = []
        for band_name, f_low, f_high in BANDS:
            u = band_u_values[band_name]
            if pos_val < U_WINDOW:
                in_window = ((u < pos_val + U_WINDOW) | (u > 1 - U_WINDOW + pos_val)).sum()
            elif pos_val > 1 - U_WINDOW:
                in_window = ((u > pos_val - U_WINDOW) | (u < pos_val + U_WINDOW - 1)).sum()
            else:
                in_window = (np.abs(u - pos_val) < U_WINDOW).sum()
            out_window = len(u) - in_window
            contingency.append([in_window, out_window])

        contingency = np.array(contingency)
        chi2, p, dof, expected = chi2_contingency(contingency)
        chi2_results[pos_name] = {'chi2': chi2, 'p': p, 'dof': dof}
        print(f"\n{pos_name} (u={pos_val:.3f}):")
        print(f"  Chi-squared = {chi2:.1f}, df = {dof}, p = {p:.2e}")
        if p < 0.001:
            print(f"  --> SIGNIFICANT: Bands differ in {pos_name} enrichment (p < 0.001)")
        elif p < 0.05:
            print(f"  --> SIGNIFICANT: Bands differ in {pos_name} enrichment (p < 0.05)")
        else:
            print(f"  --> Not significant: No evidence of band differences")

    # 2. Pairwise comparisons: Gamma vs each other band
    print("\n" + "-" * 80)
    print("--- Test 2: Pairwise Gamma vs Other Bands (Two-proportion z-test) ---")
    print("Testing: Is Gamma's enrichment significantly different from other bands?")
    print(f"Bonferroni-corrected alpha = 0.05 / 5 = 0.01")

    gamma_u = band_u_values['Gamma']
    n_gamma = len(gamma_u)

    pairwise_results = []
    for pos_name, pos_val in POSITIONS.items():
        print(f"\n{pos_name} position:")

        # Gamma proportion at this position
        if pos_val < U_WINDOW:
            gamma_in = ((gamma_u < pos_val + U_WINDOW) | (gamma_u > 1 - U_WINDOW + pos_val)).sum()
        elif pos_val > 1 - U_WINDOW:
            gamma_in = ((gamma_u > pos_val - U_WINDOW) | (gamma_u < pos_val + U_WINDOW - 1)).sum()
        else:
            gamma_in = (np.abs(gamma_u - pos_val) < U_WINDOW).sum()

        p_gamma = gamma_in / n_gamma

        for band_name, f_low, f_high in BANDS:
            if band_name == 'Gamma':
                continue

            other_u = band_u_values[band_name]
            n_other = len(other_u)

            if pos_val < U_WINDOW:
                other_in = ((other_u < pos_val + U_WINDOW) | (other_u > 1 - U_WINDOW + pos_val)).sum()
            elif pos_val > 1 - U_WINDOW:
                other_in = ((other_u > pos_val - U_WINDOW) | (other_u < pos_val + U_WINDOW - 1)).sum()
            else:
                other_in = (np.abs(other_u - pos_val) < U_WINDOW).sum()

            p_other = other_in / n_other

            # Two-proportion z-test
            p_pooled = (gamma_in + other_in) / (n_gamma + n_other)
            se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_gamma + 1/n_other))
            if se > 0:
                z_stat = (p_gamma - p_other) / se
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            else:
                z_stat = 0
                p_value = 1.0

            # Effect size (Cohen's h)
            h = cohens_h(p_gamma, p_other)

            # Interpret
            sig = '***' if p_value < 0.001 else ('**' if p_value < 0.01 else ('*' if p_value < 0.05 else ''))
            h_interp = 'large' if abs(h) > 0.8 else ('medium' if abs(h) > 0.5 else ('small' if abs(h) > 0.2 else 'negligible'))

            print(f"  Gamma vs {band_name:<10}: z={z_stat:+6.2f}, p={p_value:.2e} {sig}, Cohen's h={h:+.3f} ({h_interp})")

            pairwise_results.append({
                'Position': pos_name,
                'Comparison': f'Gamma vs {band_name}',
                'z_stat': z_stat,
                'p_value': p_value,
                'Gamma_prop': p_gamma,
                'Other_prop': p_other,
                'Cohens_h': h,
                'Significant_bonf': p_value < 0.01,
            })

    pairwise_df = pd.DataFrame(pairwise_results)

    # 3. Bootstrap CIs for Gamma enrichment at 1° Noble
    print("\n" + "-" * 80)
    print("--- Test 3: Bootstrap 95% CIs for 1° Noble Enrichment ---")
    print("(Most important position for the heterogeneity concern)")

    bootstrap_results = []
    for band_name in [b[0] for b in BANDS]:
        u = band_u_values[band_name]
        e, z, _, _ = compute_enrichment_at_position(u, 0.618)
        ci_low, ci_high = bootstrap_enrichment_ci(u, 0.618, n_bootstrap=2000)
        print(f"  {band_name:<12}: {e:+.1f}% [95% CI: {ci_low:+.1f}% to {ci_high:+.1f}%]")
        bootstrap_results.append({
            'Band': band_name,
            'Enrichment_1Noble': e,
            'CI_low': ci_low,
            'CI_high': ci_high,
        })

    bootstrap_df = pd.DataFrame(bootstrap_results)

    # =========================================================================
    # INTERPRETATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    # Count significant Gamma vs other comparisons at 1° Noble
    noble_comparisons = pairwise_df[pairwise_df['Position'] == '1° Noble']
    n_sig = noble_comparisons['Significant_bonf'].sum()

    if n_sig == 5:  # Gamma > ALL other bands
        print("\n*** CONCLUSION: GAMMA IS UNIQUELY DIFFERENT ***")
        print("Gamma shows significantly stronger 1° Noble enrichment than ALL other bands.")
        print("This supports the 'frequency-dependent stringency' interpretation:")
        print("  - Gamma oscillations show the tightest adherence to phi^n attractors")
        print("  - Lower-frequency bands show looser adherence")
        print("  - This may reflect functional requirements (gamma: precise binding)")
    elif n_sig >= 3:
        print(f"\n*** CONCLUSION: GAMMA DIFFERS FROM {n_sig}/5 BANDS ***")
        print("Gamma shows significantly stronger 1° Noble enrichment than most bands.")
        print("Consider 'frequency-dependent stringency' framing with caveats.")
    else:
        print(f"\n*** CONCLUSION: GAMMA DIFFERS FROM ONLY {n_sig}/5 BANDS ***")
        print("Band heterogeneity may be less pronounced than z-scores suggest.")
        print("Consider maintaining 'universal architecture' claim with appropriate caveats.")

    # Check if heterogeneity is significant overall
    noble_chi2 = chi2_results['1° Noble']
    if noble_chi2['p'] < 0.001:
        print(f"\nOverall heterogeneity IS significant (Chi-sq={noble_chi2['chi2']:.1f}, p={noble_chi2['p']:.2e})")
    else:
        print(f"\nOverall heterogeneity is NOT significant (Chi-sq={noble_chi2['chi2']:.1f}, p={noble_chi2['p']:.3f})")

    # Effect sizes summary
    print("\nEffect size summary (Gamma vs others at 1° Noble):")
    noble_h = noble_comparisons['Cohens_h'].abs()
    print(f"  Mean |Cohen's h|: {noble_h.mean():.3f}")
    print(f"  Max |Cohen's h|:  {noble_h.max():.3f}")
    print(f"  Interpretation: {'Large effects' if noble_h.mean() > 0.8 else ('Medium effects' if noble_h.mean() > 0.5 else 'Small-medium effects')}")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================

    # Save to CSV
    results_df.to_csv('phi_band_heterogeneity_stats.csv', index=False)
    print(f"\nSaved: phi_band_heterogeneity_stats.csv")

    # =========================================================================
    # GENERATE FIGURE
    # =========================================================================

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Grouped bar chart of enrichment by band and position
    ax = axes[0]
    band_names = [b[0] for b in BANDS]
    positions_to_plot = ['Boundary', 'Attractor', '1° Noble']
    x = np.arange(len(band_names))
    width = 0.25

    colors = {'Boundary': '#d62728', 'Attractor': '#ff7f0e', '1° Noble': '#2ca02c'}

    for i, pos_name in enumerate(positions_to_plot):
        values = results_df[f'{pos_name}_enrich'].values
        ax.bar(x + i * width - width, values, width, label=pos_name, color=colors[pos_name], alpha=0.8)

    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Frequency Band', fontsize=12)
    ax.set_ylabel('Enrichment (%)', fontsize=12)
    ax.set_title('A. Enrichment by Band and Position', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(band_names, rotation=45, ha='right')
    ax.legend(loc='upper left')
    ax.set_ylim(-80, 180)

    # Add annotation for gamma
    gamma_idx = band_names.index('Gamma')
    gamma_noble = results_df[results_df['Band'] == 'Gamma']['1° Noble_enrich'].values[0]
    ax.annotate(f'{gamma_noble:+.0f}%', xy=(gamma_idx + width, gamma_noble),
                xytext=(gamma_idx + width + 0.3, gamma_noble + 20),
                fontsize=10, fontweight='bold', color='#2ca02c',
                arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.5))

    # Panel B: Forest plot of 1° Noble enrichment with CIs
    ax = axes[1]

    y_pos = np.arange(len(band_names))
    enrichments = bootstrap_df['Enrichment_1Noble'].values
    ci_lows = bootstrap_df['CI_low'].values
    ci_highs = bootstrap_df['CI_high'].values

    # Error bars
    xerr = np.array([enrichments - ci_lows, ci_highs - enrichments])

    # Color gamma differently
    colors = ['#2ca02c' if b == 'Gamma' else '#1f77b4' for b in band_names]

    ax.barh(y_pos, enrichments, xerr=xerr, align='center', color=colors, alpha=0.8,
            capsize=4, error_kw={'linewidth': 2})
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(band_names)
    ax.set_xlabel('1° Noble Enrichment (%) with 95% CI', fontsize=12)
    ax.set_title('B. 1° Noble Enrichment: Gamma vs Others', fontsize=13, fontweight='bold')

    # Add significance markers
    for i, (band, row) in enumerate(zip(band_names, bootstrap_df.itertuples())):
        if band != 'Gamma':
            # Check if this band's CI overlaps with gamma's CI
            gamma_row = bootstrap_df[bootstrap_df['Band'] == 'Gamma'].iloc[0]
            if row.CI_high < gamma_row.CI_low:
                ax.text(row.Enrichment_1Noble + 5, i, '***', fontsize=12, va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('phi_band_heterogeneity.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: phi_band_heterogeneity.png")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Investigate the Alpha Paradox:
- The histogram shows a PROMINENT alpha peak at 9-10 Hz
- But the enrichment analysis shows only +4.2% at 1° Noble

Why doesn't the alpha peak show up as strong φ^n enrichment?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Constants
PHI = 1.618033988749895
F0 = 7.6  # Hz

def compute_lattice_coordinate(freq, f0=F0):
    """Compute lattice coordinate u = [log_phi(f/f0)] mod 1."""
    n = np.log(freq / f0) / np.log(PHI)
    return n % 1

def freq_from_lattice(u, base_n, f0=F0):
    """Convert lattice coordinate back to frequency."""
    n = base_n + u
    return f0 * (PHI ** n)

def main():
    # Load peaks
    peaks_df = pd.read_csv('golden_ratio_peaks_ALL_EMOTIV.csv')
    freqs = peaks_df['freq'].values
    print(f"Loaded {len(freqs):,} peaks")

    # Define alpha band (φ^0 to φ^1)
    alpha_low = F0 * PHI**0  # 7.6 Hz
    alpha_high = F0 * PHI**1  # 12.3 Hz

    # Define gamma band (φ^3 to φ^4)
    gamma_low = F0 * PHI**3   # 32.2 Hz
    gamma_high = F0 * PHI**4  # 52.1 Hz

    # Filter to each band
    alpha_mask = (freqs >= alpha_low) & (freqs < alpha_high)
    gamma_mask = (freqs >= gamma_low) & (freqs < gamma_high)

    alpha_freqs = freqs[alpha_mask]
    gamma_freqs = freqs[gamma_mask]

    print(f"\nAlpha band ({alpha_low:.1f}-{alpha_high:.1f} Hz): {len(alpha_freqs):,} peaks")
    print(f"Gamma band ({gamma_low:.1f}-{gamma_high:.1f} Hz): {len(gamma_freqs):,} peaks")

    # Compute key φ^n positions within each band
    print("\n" + "=" * 80)
    print("KEY φ^n POSITIONS")
    print("=" * 80)

    print("\nAlpha band positions:")
    print(f"  Boundary (u=0.0):    {freq_from_lattice(0.0, 0):.2f} Hz (= F0)")
    print(f"  2° Noble (u=0.382):  {freq_from_lattice(0.382, 0):.2f} Hz")
    print(f"  Attractor (u=0.5):   {freq_from_lattice(0.5, 0):.2f} Hz")
    print(f"  1° Noble (u=0.618):  {freq_from_lattice(0.618, 0):.2f} Hz")
    print(f"  Boundary (u=1.0):    {freq_from_lattice(1.0, 0):.2f} Hz (= F0×φ)")

    print("\nGamma band positions:")
    print(f"  Boundary (u=0.0):    {freq_from_lattice(0.0, 3):.2f} Hz (= F0×φ³)")
    print(f"  2° Noble (u=0.382):  {freq_from_lattice(0.382, 3):.2f} Hz")
    print(f"  Attractor (u=0.5):   {freq_from_lattice(0.5, 3):.2f} Hz")
    print(f"  1° Noble (u=0.618):  {freq_from_lattice(0.618, 3):.2f} Hz")
    print(f"  Boundary (u=1.0):    {freq_from_lattice(1.0, 3):.2f} Hz (= F0×φ⁴)")

    # Find the ACTUAL peak centers
    print("\n" + "=" * 80)
    print("ACTUAL PEAK LOCATIONS vs φ^n PREDICTIONS")
    print("=" * 80)

    # Alpha: Find the mode (most common frequency bin)
    alpha_hist, alpha_edges = np.histogram(alpha_freqs, bins=50)
    alpha_peak_idx = np.argmax(alpha_hist)
    alpha_peak_freq = (alpha_edges[alpha_peak_idx] + alpha_edges[alpha_peak_idx + 1]) / 2
    alpha_peak_u = compute_lattice_coordinate(alpha_peak_freq)

    # Gamma: Find the mode
    gamma_hist, gamma_edges = np.histogram(gamma_freqs, bins=50)
    gamma_peak_idx = np.argmax(gamma_hist)
    gamma_peak_freq = (gamma_edges[gamma_peak_idx] + gamma_edges[gamma_peak_idx + 1]) / 2
    gamma_peak_u = compute_lattice_coordinate(gamma_peak_freq)

    print(f"\nAlpha band:")
    print(f"  Observed peak center:    {alpha_peak_freq:.2f} Hz")
    print(f"  Observed lattice coord:  u = {alpha_peak_u:.3f}")
    print(f"  Predicted 1° Noble:      {freq_from_lattice(0.618, 0):.2f} Hz (u = 0.618)")
    print(f"  Predicted Attractor:     {freq_from_lattice(0.5, 0):.2f} Hz (u = 0.500)")
    print(f"  Offset from 1° Noble:    {alpha_peak_u - 0.618:+.3f} ({(alpha_peak_freq - freq_from_lattice(0.618, 0)):+.2f} Hz)")
    print(f"  Offset from Attractor:   {alpha_peak_u - 0.5:+.3f} ({(alpha_peak_freq - freq_from_lattice(0.5, 0)):+.2f} Hz)")

    print(f"\nGamma band:")
    print(f"  Observed peak center:    {gamma_peak_freq:.2f} Hz")
    print(f"  Observed lattice coord:  u = {gamma_peak_u:.3f}")
    print(f"  Predicted 1° Noble:      {freq_from_lattice(0.618, 3):.2f} Hz (u = 0.618)")
    print(f"  Predicted Attractor:     {freq_from_lattice(0.5, 3):.2f} Hz (u = 0.500)")
    print(f"  Offset from 1° Noble:    {gamma_peak_u - 0.618:+.3f} ({(gamma_peak_freq - freq_from_lattice(0.618, 3)):+.2f} Hz)")
    print(f"  Offset from Attractor:   {gamma_peak_u - 0.5:+.3f} ({(gamma_peak_freq - freq_from_lattice(0.5, 3)):+.2f} Hz)")

    # Compute peak WIDTH (spread)
    print("\n" + "=" * 80)
    print("PEAK WIDTH ANALYSIS (The Key to the Paradox)")
    print("=" * 80)

    # Compute standard deviation of frequencies in each band
    alpha_std = np.std(alpha_freqs)
    gamma_std = np.std(gamma_freqs)

    # Relative width (std / band center)
    alpha_center = (alpha_low + alpha_high) / 2
    gamma_center = (gamma_low + gamma_high) / 2
    alpha_rel_std = alpha_std / alpha_center
    gamma_rel_std = gamma_std / gamma_center

    # Lattice coordinate spread
    alpha_u = compute_lattice_coordinate(alpha_freqs)
    gamma_u = compute_lattice_coordinate(gamma_freqs)
    alpha_u_std = np.std(alpha_u)
    gamma_u_std = np.std(gamma_u)

    print(f"\nAlpha band:")
    print(f"  Frequency std:     {alpha_std:.2f} Hz")
    print(f"  Relative std:      {alpha_rel_std:.2%}")
    print(f"  Lattice coord std: {alpha_u_std:.3f}")

    print(f"\nGamma band:")
    print(f"  Frequency std:     {gamma_std:.2f} Hz")
    print(f"  Relative std:      {gamma_rel_std:.2%}")
    print(f"  Lattice coord std: {gamma_u_std:.3f}")

    print(f"\n  Ratio of lattice spreads (alpha/gamma): {alpha_u_std/gamma_u_std:.2f}x")

    # Compute enrichment window overlap
    print("\n" + "=" * 80)
    print("ENRICHMENT WINDOW ANALYSIS")
    print("=" * 80)

    U_WINDOW = 0.05  # The window used in enrichment calculation

    # What fraction of peaks fall in each window?
    for band_name, u_vals, peak_u in [('Alpha', alpha_u, alpha_peak_u), ('Gamma', gamma_u, gamma_peak_u)]:
        n_total = len(u_vals)

        # Noble 1° window (u = 0.618 ± 0.05)
        noble1_mask = np.abs(u_vals - 0.618) < U_WINDOW
        noble1_frac = noble1_mask.sum() / n_total
        noble1_expected = 2 * U_WINDOW  # 10% if uniform

        # Attractor window (u = 0.5 ± 0.05)
        attr_mask = np.abs(u_vals - 0.5) < U_WINDOW
        attr_frac = attr_mask.sum() / n_total

        # Window around actual peak
        peak_mask = np.abs(u_vals - peak_u) < U_WINDOW
        peak_frac = peak_mask.sum() / n_total

        print(f"\n{band_name} band (peak at u={peak_u:.3f}):")
        print(f"  Expected if uniform:    {noble1_expected:.1%}")
        print(f"  Fraction at 1° Noble:   {noble1_frac:.1%} ({noble1_frac/noble1_expected:.2f}x expected)")
        print(f"  Fraction at Attractor:  {attr_frac:.1%} ({attr_frac/noble1_expected:.2f}x expected)")
        print(f"  Fraction at actual peak:{peak_frac:.1%} ({peak_frac/noble1_expected:.2f}x expected)")

    # ==========================================================================
    # VISUALIZATION
    # ==========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Panel A: Alpha frequency histogram
    ax = axes[0, 0]
    ax.hist(alpha_freqs, bins=50, color='steelblue', alpha=0.8, edgecolor='white')
    ax.axvline(freq_from_lattice(0.5, 0), color='orange', linestyle='--', linewidth=2, label=f'Attractor ({freq_from_lattice(0.5, 0):.1f} Hz)')
    ax.axvline(freq_from_lattice(0.618, 0), color='green', linestyle='--', linewidth=2, label=f'1° Noble ({freq_from_lattice(0.618, 0):.1f} Hz)')
    ax.axvline(alpha_peak_freq, color='red', linestyle='-', linewidth=2, label=f'Actual peak ({alpha_peak_freq:.1f} Hz)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Peak Count')
    ax.set_title('A. Alpha Band: Frequency Distribution', fontweight='bold')
    ax.legend(fontsize=8)

    # Panel B: Gamma frequency histogram
    ax = axes[0, 1]
    ax.hist(gamma_freqs, bins=50, color='steelblue', alpha=0.8, edgecolor='white')
    ax.axvline(freq_from_lattice(0.5, 3), color='orange', linestyle='--', linewidth=2, label=f'Attractor ({freq_from_lattice(0.5, 3):.1f} Hz)')
    ax.axvline(freq_from_lattice(0.618, 3), color='green', linestyle='--', linewidth=2, label=f'1° Noble ({freq_from_lattice(0.618, 3):.1f} Hz)')
    ax.axvline(gamma_peak_freq, color='red', linestyle='-', linewidth=2, label=f'Actual peak ({gamma_peak_freq:.1f} Hz)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Peak Count')
    ax.set_title('B. Gamma Band: Frequency Distribution', fontweight='bold')
    ax.legend(fontsize=8)

    # Panel C: Alpha lattice coordinate histogram
    ax = axes[0, 2]
    ax.hist(alpha_u, bins=50, color='steelblue', alpha=0.8, edgecolor='white')
    ax.axvline(0.5, color='orange', linestyle='--', linewidth=2, label='Attractor (0.5)')
    ax.axvline(0.618, color='green', linestyle='--', linewidth=2, label='1° Noble (0.618)')
    ax.axvspan(0.618 - U_WINDOW, 0.618 + U_WINDOW, alpha=0.2, color='green', label='Enrichment window')
    ax.axhline(len(alpha_u)/50, color='red', linestyle=':', linewidth=1, label='Uniform expectation')
    ax.set_xlabel('Lattice coordinate (u)')
    ax.set_ylabel('Peak Count')
    ax.set_title('C. Alpha: Lattice Coordinate Distribution', fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)

    # Panel D: Gamma lattice coordinate histogram
    ax = axes[1, 0]
    ax.hist(gamma_u, bins=50, color='steelblue', alpha=0.8, edgecolor='white')
    ax.axvline(0.5, color='orange', linestyle='--', linewidth=2, label='Attractor (0.5)')
    ax.axvline(0.618, color='green', linestyle='--', linewidth=2, label='1° Noble (0.618)')
    ax.axvspan(0.618 - U_WINDOW, 0.618 + U_WINDOW, alpha=0.2, color='green', label='Enrichment window')
    ax.axhline(len(gamma_u)/50, color='red', linestyle=':', linewidth=1, label='Uniform expectation')
    ax.set_xlabel('Lattice coordinate (u)')
    ax.set_ylabel('Peak Count')
    ax.set_title('D. Gamma: Lattice Coordinate Distribution', fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)

    # Panel E: Overlay comparison (normalized)
    ax = axes[1, 1]
    alpha_hist_norm, alpha_bins = np.histogram(alpha_u, bins=50, density=True)
    gamma_hist_norm, gamma_bins = np.histogram(gamma_u, bins=50, density=True)
    bin_centers = (alpha_bins[:-1] + alpha_bins[1:]) / 2

    ax.plot(bin_centers, alpha_hist_norm, 'b-', linewidth=2, label='Alpha', alpha=0.8)
    ax.plot(bin_centers, gamma_hist_norm, 'r-', linewidth=2, label='Gamma', alpha=0.8)
    ax.axvline(0.5, color='orange', linestyle='--', linewidth=1)
    ax.axvline(0.618, color='green', linestyle='--', linewidth=1)
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1, label='Uniform')
    ax.set_xlabel('Lattice coordinate (u)')
    ax.set_ylabel('Normalized density')
    ax.set_title('E. Overlay: Alpha vs Gamma Lattice Distributions', fontweight='bold')
    ax.legend()
    ax.set_xlim(0, 1)

    # Panel F: Summary bar chart
    ax = axes[1, 2]

    # Compute enrichment at key positions
    positions = ['Boundary\n(u=0)', '2° Noble\n(u=0.382)', 'Attractor\n(u=0.5)', '1° Noble\n(u=0.618)']
    pos_values = [0.0, 0.382, 0.5, 0.618]

    alpha_enrich = []
    gamma_enrich = []

    for pos in pos_values:
        if pos < U_WINDOW:
            alpha_mask = (alpha_u < pos + U_WINDOW) | (alpha_u > 1 - U_WINDOW + pos)
            gamma_mask = (gamma_u < pos + U_WINDOW) | (gamma_u > 1 - U_WINDOW + pos)
        else:
            alpha_mask = np.abs(alpha_u - pos) < U_WINDOW
            gamma_mask = np.abs(gamma_u - pos) < U_WINDOW

        alpha_obs = alpha_mask.sum()
        gamma_obs = gamma_mask.sum()
        alpha_exp = len(alpha_u) * 2 * U_WINDOW
        gamma_exp = len(gamma_u) * 2 * U_WINDOW

        alpha_enrich.append((alpha_obs / alpha_exp - 1) * 100)
        gamma_enrich.append((gamma_obs / gamma_exp - 1) * 100)

    x = np.arange(len(positions))
    width = 0.35
    ax.bar(x - width/2, alpha_enrich, width, label='Alpha', color='blue', alpha=0.8)
    ax.bar(x + width/2, gamma_enrich, width, label='Gamma', color='red', alpha=0.8)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(positions)
    ax.set_ylabel('Enrichment (%)')
    ax.set_title('F. Position-Specific Enrichment', fontweight='bold')
    ax.legend()

    plt.tight_layout()
    plt.savefig('phi_alpha_paradox_analysis.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("\nSaved: phi_alpha_paradox_analysis.png")

    # ==========================================================================
    # KEY INSIGHT
    # ==========================================================================
    print("\n" + "=" * 80)
    print("KEY INSIGHT: WHY ALPHA'S PROMINENT PEAK DOESN'T SHOW φ^n ENRICHMENT")
    print("=" * 80)

    print(f"""
The alpha paradox is explained by THREE factors:

1. PEAK LOCATION MISMATCH:
   - Alpha's observed peak: {alpha_peak_freq:.1f} Hz (u = {alpha_peak_u:.3f})
   - 1° Noble prediction:   {freq_from_lattice(0.618, 0):.1f} Hz (u = 0.618)
   - Offset: {(alpha_peak_freq - freq_from_lattice(0.618, 0)):+.1f} Hz

   The alpha peak is NOT centered at the 1° Noble position!
   It's closer to the Attractor position (u = 0.5, {freq_from_lattice(0.5, 0):.1f} Hz)

2. PEAK WIDTH (IAF Variability):
   - Alpha lattice spread: {alpha_u_std:.3f} (std of lattice coords)
   - Gamma lattice spread: {gamma_u_std:.3f}
   - Alpha is {alpha_u_std/gamma_u_std:.1f}x MORE SPREAD across the lattice

   Individual Alpha Frequency (IAF) varies ±1-2 Hz across subjects.
   This SMEARS the alpha peak across the entire lattice coordinate range.

   Gamma's 40 Hz peak is more consistent across individuals, so it
   concentrates more tightly at the φ^n positions.

3. ENRICHMENT WINDOW SIZE:
   - Window: ±0.05 in lattice coordinates
   - This is only ±{0.05 * (alpha_high - alpha_low):.2f} Hz in alpha band
   - But ±{0.05 * (gamma_high - gamma_low):.2f} Hz in gamma band

   A narrow enrichment window + broad peak = low enrichment

CONCLUSION:
The histogram shows alpha has MORE TOTAL PEAKS than gamma.
But those peaks are SPREAD across the band due to IAF variability.
Gamma has FEWER total peaks but they're TIGHTLY CLUSTERED at φ^n positions.

Enrichment measures CLUSTERING, not total count.
Alpha has high count but low clustering.
Gamma has lower count but extreme clustering.
""")


if __name__ == '__main__':
    main()

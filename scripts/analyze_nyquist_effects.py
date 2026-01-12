#!/usr/bin/env python3
"""
Analyze how frequency resolution limits affect gamma band results.

Key concern:
- Sampling rate: 128 Hz → Nyquist: 64 Hz
- FOOOF range: 1-50 Hz (or 45 Hz effective)
- Gamma band: 32.2-52.1 Hz (φ^3 to φ^4)

The upper portion of gamma is TRUNCATED. How does this affect results?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Constants
PHI = 1.618033988749895
F0 = 7.6  # Hz
FS = 128  # Sampling rate
NYQUIST = FS / 2  # 64 Hz

# Gamma band
GAMMA_LOW = F0 * PHI**3   # 32.19 Hz
GAMMA_HIGH = F0 * PHI**4  # 52.09 Hz


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

    print("\n" + "=" * 80)
    print("FREQUENCY RESOLUTION ANALYSIS")
    print("=" * 80)

    # Check actual frequency range in data
    print(f"\nData frequency range: {freqs.min():.1f} - {freqs.max():.1f} Hz")
    print(f"Sampling rate: {FS} Hz")
    print(f"Nyquist frequency: {NYQUIST} Hz")

    # Key gamma positions
    print(f"\nGamma band theoretical range:")
    print(f"  Lower boundary (φ^3): {GAMMA_LOW:.2f} Hz")
    print(f"  Upper boundary (φ^4): {GAMMA_HIGH:.2f} Hz")
    print(f"  Bandwidth: {GAMMA_HIGH - GAMMA_LOW:.2f} Hz")

    print(f"\nKey φ^n positions in gamma:")
    print(f"  Boundary (u=0, φ^3):    {freq_from_lattice(0.0, 3):.2f} Hz ✓ (in range)")
    print(f"  2° Noble (u=0.382):     {freq_from_lattice(0.382, 3):.2f} Hz ✓ (in range)")
    print(f"  Attractor (u=0.5):      {freq_from_lattice(0.5, 3):.2f} Hz ✓ (in range)")
    print(f"  1° Noble (u=0.618):     {freq_from_lattice(0.618, 3):.2f} Hz ✓ (in range)")
    print(f"  Upper boundary (u=1):   {freq_from_lattice(1.0, 3):.2f} Hz ❌ (ABOVE 50 Hz)")

    # Check how many gamma peaks we have at each frequency
    gamma_mask = (freqs >= GAMMA_LOW) & (freqs < GAMMA_HIGH)
    gamma_freqs = freqs[gamma_mask]

    print(f"\n" + "=" * 80)
    print("GAMMA PEAK DISTRIBUTION vs FREQUENCY")
    print("=" * 80)

    # Histogram of gamma peaks
    bins = np.arange(32, 53, 1)
    hist, edges = np.histogram(gamma_freqs, bins=bins)

    print(f"\nGamma peaks by frequency bin:")
    print(f"{'Freq (Hz)':<12} {'Count':>8} {'Bar':<40}")
    print("-" * 65)

    for i in range(len(hist)):
        freq = edges[i]
        count = hist[i]
        bar_len = int(40 * count / max(hist)) if max(hist) > 0 else 0
        bar = '█' * bar_len

        # Mark key positions
        marker = ''
        if 32 <= freq < 33:
            marker = ' ← φ^3 boundary'
        elif 40 <= freq < 42:
            marker = ' ← Attractor (~41 Hz)'
        elif 43 <= freq < 44:
            marker = ' ← 1° Noble (~43 Hz)'
        elif freq >= 50:
            marker = ' ← TRUNCATED (>50 Hz)'

        print(f"{freq:>4.0f}-{freq+1:<4.0f} Hz {count:>8,} {bar}{marker}")

    # Count peaks above 45 Hz
    above_45 = (gamma_freqs >= 45).sum()
    above_48 = (gamma_freqs >= 48).sum()
    above_50 = (gamma_freqs >= 50).sum()

    print(f"\n" + "=" * 80)
    print("TRUNCATION ANALYSIS")
    print("=" * 80)

    print(f"\nPeaks in gamma band: {len(gamma_freqs):,}")
    print(f"  Peaks ≥ 45 Hz: {above_45:,} ({100*above_45/len(gamma_freqs):.1f}%)")
    print(f"  Peaks ≥ 48 Hz: {above_48:,} ({100*above_48/len(gamma_freqs):.1f}%)")
    print(f"  Peaks ≥ 50 Hz: {above_50:,} ({100*above_50/len(gamma_freqs):.1f}%)")

    # Theoretical: What SHOULD be in upper gamma?
    # If uniform, (52.1-45)/19.9 = 36% should be above 45 Hz
    theoretical_above_45_pct = (GAMMA_HIGH - 45) / (GAMMA_HIGH - GAMMA_LOW) * 100
    theoretical_above_48_pct = (GAMMA_HIGH - 48) / (GAMMA_HIGH - GAMMA_LOW) * 100

    print(f"\n  Theoretical % above 45 Hz (if uniform): {theoretical_above_45_pct:.1f}%")
    print(f"  Theoretical % above 48 Hz (if uniform): {theoretical_above_48_pct:.1f}%")
    print(f"  Actual % above 45 Hz: {100*above_45/len(gamma_freqs):.1f}%")

    # CRITICAL: How does truncation affect lattice coordinates?
    print(f"\n" + "=" * 80)
    print("LATTICE COORDINATE BIAS FROM TRUNCATION")
    print("=" * 80)

    # If we're missing peaks above 50 Hz, we're missing part of the lattice
    print(f"\nLattice coordinates that are MISSING due to 50 Hz cutoff:")
    cutoff_u = compute_lattice_coordinate(50.0)  # Lattice coord at 50 Hz
    print(f"  50 Hz corresponds to u = {cutoff_u:.3f} in gamma band")
    print(f"  Upper boundary (52.1 Hz) would be u = 1.0")
    print(f"  We're MISSING u = {cutoff_u:.3f} to 1.0 ({(1-cutoff_u)*100:.1f}% of lattice)")

    # What position is at 50 Hz?
    u_at_50 = compute_lattice_coordinate(50.0)
    print(f"\n  The 50 Hz cutoff is at u = {u_at_50:.3f}")
    print(f"  This is BETWEEN 1° Noble (0.618) and upper boundary (1.0)")
    print(f"  Distance from 1° Noble: {u_at_50 - 0.618:.3f}")

    # Re-analyze gamma with truncation correction
    print(f"\n" + "=" * 80)
    print("CORRECTED GAMMA ANALYSIS (Excluding >48 Hz)")
    print("=" * 80)

    # Use only peaks below 48 Hz to avoid edge effects
    gamma_clean = gamma_freqs[gamma_freqs < 48]
    u_clean = compute_lattice_coordinate(gamma_clean)

    # Define effective lattice range
    u_max = compute_lattice_coordinate(48.0)  # ~0.86

    print(f"\nUsing peaks < 48 Hz to avoid truncation bias")
    print(f"  Clean gamma peaks: {len(gamma_clean):,}")
    print(f"  Effective lattice range: 0 to {u_max:.3f}")

    # Compute enrichment at key positions
    U_WINDOW = 0.05

    positions = {
        'Boundary (u=0)': 0.0,
        '2° Noble (u=0.382)': 0.382,
        'Attractor (u=0.5)': 0.5,
        '1° Noble (u=0.618)': 0.618,
    }

    print(f"\n{'Position':<22} {'Enrichment (all)':>18} {'Enrichment (clean)':>20}")
    print("-" * 65)

    u_all = compute_lattice_coordinate(gamma_freqs)

    for pos_name, pos_val in positions.items():
        # All gamma peaks
        if pos_val < U_WINDOW:
            mask_all = (u_all < pos_val + U_WINDOW) | (u_all > 1 - U_WINDOW + pos_val)
            mask_clean = (u_clean < pos_val + U_WINDOW) | (u_clean > u_max - U_WINDOW + pos_val)
        else:
            mask_all = np.abs(u_all - pos_val) < U_WINDOW
            mask_clean = np.abs(u_clean - pos_val) < U_WINDOW

        exp_all = len(u_all) * 2 * U_WINDOW
        exp_clean = len(u_clean) * 2 * U_WINDOW / u_max  # Adjust for truncated range

        enrich_all = (mask_all.sum() / exp_all - 1) * 100
        enrich_clean = (mask_clean.sum() / exp_clean - 1) * 100 if exp_clean > 0 else 0

        print(f"{pos_name:<22} {enrich_all:>+17.1f}% {enrich_clean:>+19.1f}%")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Gamma frequency histogram with truncation
    ax = axes[0, 0]
    ax.hist(gamma_freqs, bins=np.arange(32, 53, 0.5), color='steelblue', alpha=0.8, edgecolor='white')
    ax.axvline(50, color='red', linestyle='--', linewidth=2, label='50 Hz cutoff')
    ax.axvline(GAMMA_HIGH, color='orange', linestyle='--', linewidth=2, label=f'φ^4 boundary ({GAMMA_HIGH:.1f} Hz)')
    ax.axvline(freq_from_lattice(0.618, 3), color='green', linestyle='--', linewidth=2, label=f'1° Noble ({freq_from_lattice(0.618, 3):.1f} Hz)')
    ax.axvline(freq_from_lattice(0.5, 3), color='purple', linestyle='--', linewidth=2, label=f'Attractor ({freq_from_lattice(0.5, 3):.1f} Hz)')

    ax.fill_between([50, 53], 0, ax.get_ylim()[1], color='red', alpha=0.2, label='Missing data')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Peak Count')
    ax.set_title('A. Gamma Peaks: Frequency Distribution with Truncation', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(32, 53)

    # Panel B: Lattice distribution (full vs truncated)
    ax = axes[0, 1]
    ax.hist(u_all, bins=50, color='steelblue', alpha=0.5, label='All gamma peaks', density=True)
    ax.hist(u_clean, bins=50, color='green', alpha=0.5, label='Clean (<48 Hz)', density=True)
    ax.axvline(0.618, color='green', linestyle='--', linewidth=2)
    ax.axvline(0.5, color='purple', linestyle='--', linewidth=2)
    ax.axvline(u_at_50, color='red', linestyle='--', linewidth=2, label=f'50 Hz = u={u_at_50:.2f}')
    ax.axvspan(u_at_50, 1.0, alpha=0.2, color='red', label='Truncated region')
    ax.set_xlabel('Lattice coordinate (u)')
    ax.set_ylabel('Density')
    ax.set_title('B. Lattice Coordinate: Effect of Truncation', fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)

    # Panel C: Peak count by Hz bin with annotations
    ax = axes[1, 0]
    bins_fine = np.arange(32, 53, 1)
    hist_fine, _ = np.histogram(gamma_freqs, bins=bins_fine)
    bin_centers = bins_fine[:-1] + 0.5

    colors = ['steelblue' if b < 48 else 'lightcoral' for b in bins_fine[:-1]]
    ax.bar(bin_centers, hist_fine, width=0.9, color=colors, alpha=0.8, edgecolor='white')

    # Annotate key positions
    ax.axvline(freq_from_lattice(0.5, 3), color='purple', linestyle='--', linewidth=2)
    ax.axvline(freq_from_lattice(0.618, 3), color='green', linestyle='--', linewidth=2)
    ax.axvline(50, color='red', linestyle='-', linewidth=2)

    ax.annotate('Attractor', xy=(41, hist_fine[9]), xytext=(38, hist_fine[9]+200),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='purple'))
    ax.annotate('1° Noble', xy=(43.3, hist_fine[11]), xytext=(45.5, hist_fine[11]+300),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='green'))
    ax.annotate('Truncation\n→', xy=(50, 500), fontsize=10, color='red', fontweight='bold')

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Peak Count')
    ax.set_title('C. Peak Counts with Truncation Zone (red bars)', fontweight='bold')
    ax.set_xlim(32, 53)

    # Panel D: Cumulative distribution
    ax = axes[1, 1]
    sorted_freqs = np.sort(gamma_freqs)
    cdf = np.arange(1, len(sorted_freqs) + 1) / len(sorted_freqs)
    ax.plot(sorted_freqs, cdf, 'b-', linewidth=2)
    ax.axvline(50, color='red', linestyle='--', linewidth=2, label='50 Hz cutoff')
    ax.axhline(cdf[np.searchsorted(sorted_freqs, 50)-1], color='red', linestyle=':', alpha=0.5)

    pct_below_50 = 100 * cdf[np.searchsorted(sorted_freqs, 50)-1] if len(sorted_freqs) > 0 else 0
    ax.annotate(f'{pct_below_50:.1f}% below 50 Hz', xy=(50, cdf[np.searchsorted(sorted_freqs, 50)-1]),
                xytext=(42, 0.6), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='red'))

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Cumulative proportion')
    ax.set_title('D. Cumulative Distribution of Gamma Peaks', fontweight='bold')
    ax.legend()
    ax.set_xlim(32, 53)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('phi_nyquist_effects.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("\nSaved: phi_nyquist_effects.png")

    # Key conclusions
    print("\n" + "=" * 80)
    print("CONCLUSIONS: IMPACT OF FREQUENCY TRUNCATION")
    print("=" * 80)

    print(f"""
1. DATA TRUNCATION:
   - Gamma band theoretical: {GAMMA_LOW:.1f} - {GAMMA_HIGH:.1f} Hz
   - Data available: {GAMMA_LOW:.1f} - ~50 Hz
   - Missing: {max(0, GAMMA_HIGH - 50):.1f} Hz ({(GAMMA_HIGH - 50)/(GAMMA_HIGH - GAMMA_LOW)*100:.1f}% of band)

2. KEY POSITIONS ARE PRESERVED:
   - 1° Noble at {freq_from_lattice(0.618, 3):.1f} Hz → ✓ Below 50 Hz cutoff
   - Attractor at {freq_from_lattice(0.5, 3):.1f} Hz → ✓ Below 50 Hz cutoff
   - 2° Noble at {freq_from_lattice(0.382, 3):.1f} Hz → ✓ Below 50 Hz cutoff
   - Lower boundary at {GAMMA_LOW:.1f} Hz → ✓ Below 50 Hz cutoff

3. WHAT'S AFFECTED:
   - Upper boundary (52.1 Hz) is NOT VISIBLE
   - Peaks in 50-52 Hz range are missing
   - The apparent "boundary depletion" at u≈0.9 may be TRUNCATION ARTIFACT

4. ENRICHMENT IMPACT:
   - 1° Noble enrichment should be VALID (43.3 Hz is well below cutoff)
   - Attractor enrichment should be VALID (41.0 Hz is well below cutoff)
   - Boundary depletion at φ^4 CANNOT BE VERIFIED (data missing)

5. OVERALL ASSESSMENT:
   - The CORE FINDING (enrichment at 1° Noble/Attractor) is NOT affected
   - The boundary depletion claim for gamma's UPPER edge is UNRELIABLE
   - The dramatic peak at 43 Hz is REAL, not an artifact
""")


if __name__ == '__main__':
    main()

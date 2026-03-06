"""
Per-position diagnostic: boundary offset, spectral gaps, and lattice asymmetry.

Investigates three puzzles from the degree 2-7 per-position enrichment analysis:
1. Noble_7 (u=0.034) enriches MORE than boundary (u=0.000) — implies offset
2. Consistent push zone at nobles 3,4,5 (u=0.09-0.24) — spectral gap?
3. Strong lower/upper half asymmetry — inv_noble_7 should spillover but doesn't
"""

import numpy as np
import pandas as pd
from scipy import stats
import os

# ── Constants ────────────────────────────────────────────────────────
PHI = (1 + np.sqrt(5)) / 2
F0 = 7.83
BANDS = ['delta', 'theta', 'alpha', 'gamma']
BAND_RANGES = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 12), 'gamma': (25, 45)}

# All 14 positions (degree-7)
POSITIONS = {
    'boundary':    0.000,
    'noble_7':     (1/PHI)**7,     # 0.034
    'noble_6':     (1/PHI)**6,     # 0.056
    'noble_5':     (1/PHI)**5,     # 0.090
    'noble_4':     (1/PHI)**4,     # 0.146
    'noble_3':     (1/PHI)**3,     # 0.236
    'noble_2':     1 - 1/PHI,      # 0.382
    'attractor':   0.500,
    'noble_1':     1/PHI,           # 0.618
    'inv_noble_3': 1 - (1/PHI)**3,  # 0.764
    'inv_noble_4': 1 - (1/PHI)**4,  # 0.854
    'inv_noble_5': 1 - (1/PHI)**5,  # 0.910
    'inv_noble_6': 1 - (1/PHI)**6,  # 0.944
    'inv_noble_7': 1 - (1/PHI)**7,  # 0.966
}

CORE_PULLERS = {'boundary': 0.000, 'attractor': 0.500, 'noble_1': 1/PHI}


def circ_dist(a, b):
    d = abs(a - b)
    return min(d, 1 - d)


def density_at_position(u_values, pos, bandwidth=0.03):
    dists = np.array([circ_dist(u, pos) for u in u_values])
    return np.exp(-0.5 * (dists / bandwidth)**2).mean()


def lattice_coordinate(freq, f0, ratio=PHI):
    return (np.log(freq / f0) / np.log(ratio)) % 1.0


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 1: Per-Band KDE Profiles Near Boundary
# ══════════════════════════════════════════════════════════════════════
def analysis1_boundary_kde(df, dataset_name):
    print(f"\n{'='*80}")
    print(f"ANALYSIS 1: Boundary KDE Profiles — {dataset_name}")
    print(f"{'='*80}")

    scan_u = np.linspace(-0.10, 0.10, 101)
    bw = 0.03

    print(f"\n  {'Band':<8} {'Peak u':>8} {'f_eff Hz':>9} {'Offset Hz':>10} "
          f"{'Peak dens':>10} {'FWHM':>6} {'n7/in7':>7}")
    print(f"  {'─'*8} {'─'*8} {'─'*9} {'─'*10} {'─'*10} {'─'*6} {'─'*7}")

    for band in BANDS:
        col = f'{band}_u'
        if col not in df.columns:
            continue
        vals = df[col].dropna().values

        # Scan density around boundary
        densities = np.array([density_at_position(vals, u % 1.0, bw) for u in scan_u])
        peak_idx = np.argmax(densities)
        peak_u = scan_u[peak_idx]
        f_eff = F0 * PHI**peak_u
        offset_hz = f_eff - F0

        # FWHM
        half_max = densities[peak_idx] / 2
        above = densities >= half_max
        if above.any():
            left = scan_u[above][0]
            right = scan_u[above][-1]
            fwhm = right - left
        else:
            fwhm = np.nan

        # Noble_7 vs inv_noble_7 ratio
        d_n7 = density_at_position(vals, 0.034, bw)
        d_in7 = density_at_position(vals, 0.966, bw)
        ratio = d_n7 / d_in7 if d_in7 > 0 else np.inf

        print(f"  {band:<8} {peak_u:>+8.4f} {f_eff:>9.2f} {offset_hz:>+10.3f} "
              f"{densities[peak_idx]:>10.6f} {fwhm:>6.3f} {ratio:>7.2f}x")

    # Combined (all bands)
    all_u = []
    for band in BANDS:
        col = f'{band}_u'
        if col in df.columns:
            all_u.extend(df[col].dropna().values)
    all_u = np.array(all_u)

    densities = np.array([density_at_position(all_u, u % 1.0, bw) for u in scan_u])
    peak_idx = np.argmax(densities)
    peak_u = scan_u[peak_idx]
    f_eff = F0 * PHI**peak_u
    d_n7 = density_at_position(all_u, 0.034, bw)
    d_in7 = density_at_position(all_u, 0.966, bw)

    print(f"\n  Combined peak at u={peak_u:+.4f} (f_eff={f_eff:.2f} Hz)")
    print(f"  noble_7 density={d_n7:.6f}, inv_noble_7={d_in7:.6f}, ratio={d_n7/d_in7:.2f}x")

    # Subject count asymmetry near boundary
    for band in BANDS:
        col = f'{band}_u'
        if col not in df.columns:
            continue
        vals = df[col].dropna().values
        n_above = np.sum(vals < 0.06)  # u < 0.06 = above f0
        n_below = np.sum(vals > 0.94)  # u > 0.94 = below f0
        print(f"  {band}: {n_above} subjects above f₀ (u<0.06), {n_below} below (u>0.94), "
              f"ratio {n_above/max(n_below,1):.1f}:1")


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 2: Per-Band Decomposition at All 14 Positions
# ══════════════════════════════════════════════════════════════════════
def analysis2_band_decomposition(df, dataset_name):
    print(f"\n{'='*80}")
    print(f"ANALYSIS 2: Per-Band Decomposition — {dataset_name}")
    print(f"{'='*80}")

    bw = 0.03
    band_u = {}
    for band in BANDS:
        col = f'{band}_u'
        if col in df.columns:
            band_u[band] = df[col].dropna().values

    # Compute enrichment for each position × band
    # Enrichment = (observed_density / null_density - 1) × 100
    # null_density from 1000 uniform draws
    np.random.seed(42)
    null_densities = {}
    for band in BANDS:
        if band not in band_u:
            continue
        n = len(band_u[band])
        nulls = []
        for _ in range(1000):
            shuf = np.random.uniform(0, 1, n)
            # Density at a random point for null (use 0.5 as reference)
            nulls.append(density_at_position(shuf, 0.5, bw))
        null_densities[band] = np.mean(nulls)

    sorted_pos = sorted(POSITIONS.items(), key=lambda x: x[1])

    for band in BANDS:
        if band not in band_u:
            continue
        null_d = null_densities[band]
        print(f"\n  {band.upper()} (N={len(band_u[band])}, null_density={null_d:.6f})")
        print(f"  {'Position':<15} {'u':>6} {'Density':>9} {'Enrichment':>11} {'Verdict':>10}")
        print(f"  {'─'*15} {'─'*6} {'─'*9} {'─'*11} {'─'*10}")
        for name, u_val in sorted_pos:
            d = density_at_position(band_u[band], u_val, bw)
            enrich = (d / null_d - 1) * 100 if null_d > 0 else 0
            if enrich > 15:
                verdict = "PULL"
            elif enrich < -15:
                verdict = "push"
            else:
                verdict = "~"
            print(f"  {name:<15} {u_val:>6.3f} {d:>9.6f} {enrich:>+10.1f}% {verdict:>10}")


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 3: Frequency Mapping Table
# ══════════════════════════════════════════════════════════════════════
def analysis3_frequency_mapping():
    print(f"\n{'='*80}")
    print(f"ANALYSIS 3: Frequency Mapping — What Each Position Means in Hz")
    print(f"{'='*80}")

    sorted_pos = sorted(POSITIONS.items(), key=lambda x: x[1])

    # For each position, find all frequencies in each band
    # f = F0 * PHI^(n + u) for integer n
    # We need F0 * PHI^(n + u) to fall within band range

    print(f"\n  {'Position':<15} {'u':>6} ", end='')
    for band in BANDS:
        lo, hi = BAND_RANGES[band]
        print(f"  {band} [{lo}-{hi}]", end='')
    print()
    print(f"  {'─'*15} {'─'*6} ", end='')
    for band in BANDS:
        print(f"  {'─'*12}", end='')
    print()

    for name, u_val in sorted_pos:
        print(f"  {name:<15} {u_val:>6.3f} ", end='')
        for band in BANDS:
            lo, hi = BAND_RANGES[band]
            freqs_in_band = []
            for n in range(-5, 6):
                f = F0 * PHI**(n + u_val)
                if lo <= f <= hi:
                    freqs_in_band.append(f)
            if freqs_in_band:
                print(f"  {freqs_in_band[0]:>6.2f} Hz   ", end='')
            else:
                print(f"  {'---':>6}      ", end='')
        print()

    # Annotate the spectral gap interpretation
    print(f"\n  Key spectral landmarks:")
    print(f"    f₀ = {F0:.2f} Hz (boundary in upper theta octave)")
    print(f"    f₀/φ = {F0/PHI:.2f} Hz (boundary in lower theta octave)")
    print(f"    attractor in theta = {F0/PHI * PHI**0.5:.2f} Hz")
    print(f"    noble_1 in theta = {F0/PHI * PHI**(1/PHI):.2f} Hz")
    print(f"    Push zone (nobles 3,4,5) in theta = {F0/PHI * PHI**0.09:.2f}–{F0/PHI * PHI**0.236:.2f} Hz")
    print(f"    = spectral gap between f₀/φ ({F0/PHI:.2f}) and attractor ({F0/PHI * PHI**0.5:.2f})")


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 4: f₀ Sensitivity — Does Shifting f₀ Resolve Asymmetry?
# ══════════════════════════════════════════════════════════════════════
def analysis4_f0_sensitivity(df, dataset_name):
    print(f"\n{'='*80}")
    print(f"ANALYSIS 4: f₀ Sensitivity — {dataset_name}")
    print(f"{'='*80}")

    bw = 0.03
    f0_values = [7.83, 7.87, 7.90, 7.93, 7.96, 8.00]

    # Get raw frequencies per band
    band_freqs = {}
    for band in BANDS:
        fcol = f'{band}_freq'
        if fcol in df.columns:
            band_freqs[band] = df[fcol].dropna().values

    print(f"\n  {'f₀':>6} {'n7 enrich':>11} {'in7 enrich':>12} {'n7/in7':>7} "
          f"{'bndry enrich':>13} {'peak_u':>8} {'push zone':>10}")
    print(f"  {'─'*6} {'─'*11} {'─'*12} {'─'*7} {'─'*13} {'─'*8} {'─'*10}")

    for f0 in f0_values:
        # Recompute u values at this f0
        all_u = []
        for band in BANDS:
            if band in band_freqs:
                u_vals = (np.log(band_freqs[band] / f0) / np.log(PHI)) % 1.0
                all_u.extend(u_vals)
        all_u = np.array(all_u)

        # Noble_7 and inv_noble_7 positions don't change (they're lattice properties)
        n7_u = (1/PHI)**7
        in7_u = 1 - (1/PHI)**7

        d_n7 = density_at_position(all_u, n7_u, bw)
        d_in7 = density_at_position(all_u, in7_u, bw)
        d_bnd = density_at_position(all_u, 0.000, bw)

        # Null density (uniform)
        np.random.seed(42)
        null_d = np.mean([density_at_position(np.random.uniform(0, 1, len(all_u)), 0.5, bw) for _ in range(500)])

        enrich_n7 = (d_n7 / null_d - 1) * 100
        enrich_in7 = (d_in7 / null_d - 1) * 100
        enrich_bnd = (d_bnd / null_d - 1) * 100
        ratio = d_n7 / d_in7 if d_in7 > 0 else np.inf

        # Find boundary peak
        scan = np.linspace(-0.08, 0.08, 81)
        dens = [density_at_position(all_u, u % 1.0, bw) for u in scan]
        peak_u = scan[np.argmax(dens)]

        # Push zone strength (mean enrichment at nobles 3,4,5)
        push_d = np.mean([density_at_position(all_u, POSITIONS[p], bw)
                          for p in ['noble_3', 'noble_4', 'noble_5']])
        push_enrich = (push_d / null_d - 1) * 100

        print(f"  {f0:>6.2f} {enrich_n7:>+10.1f}% {enrich_in7:>+11.1f}% {ratio:>7.2f}x "
              f"{enrich_bnd:>+12.1f}% {peak_u:>+8.4f} {push_enrich:>+9.1f}%")


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 5: Attractor Gap Structure
# ══════════════════════════════════════════════════════════════════════
def analysis5_gap_structure(df, dataset_name):
    print(f"\n{'='*80}")
    print(f"ANALYSIS 5: Gap Structure — {dataset_name}")
    print(f"{'='*80}")

    bw = 0.03
    all_u = []
    for band in BANDS:
        col = f'{band}_u'
        if col in df.columns:
            all_u.extend(df[col].dropna().values)
    all_u = np.array(all_u)

    # Null density
    np.random.seed(42)
    null_d = np.mean([density_at_position(np.random.uniform(0, 1, len(all_u)), 0.5, bw) for _ in range(500)])

    # Core pullers and their gaps
    pullers = sorted(CORE_PULLERS.items(), key=lambda x: x[1])
    print(f"\n  Core pullers: {', '.join(f'{n} ({v:.3f})' for n, v in pullers)}")
    print(f"\n  Gaps between core pullers:")
    for i in range(len(pullers)):
        j = (i + 1) % len(pullers)
        if j > i:
            gap = pullers[j][1] - pullers[i][1]
        else:
            gap = 1.0 - pullers[i][1] + pullers[j][1]
        print(f"    {pullers[i][0]} → {pullers[j][0]}: gap = {gap:.3f}")

    # For each non-puller position, compute distance to nearest puller and enrichment
    sorted_pos = sorted(POSITIONS.items(), key=lambda x: x[1])

    print(f"\n  {'Position':<15} {'u':>6} {'d_nearest':>10} {'Nearest':>12} {'Enrichment':>11} {'Gap':>20}")
    print(f"  {'─'*15} {'─'*6} {'─'*10} {'─'*12} {'─'*11} {'─'*20}")

    for name, u_val in sorted_pos:
        # Distance to nearest puller
        dists = {pn: circ_dist(u_val, pv) for pn, pv in CORE_PULLERS.items()}
        nearest = min(dists, key=dists.get)
        d_nearest = dists[nearest]

        # Enrichment
        d = density_at_position(all_u, u_val, bw)
        enrich = (d / null_d - 1) * 100

        # Which gap is this in?
        if 0 <= u_val < 0.500:
            gap_name = "boundary→attractor"
        elif 0.500 <= u_val < 1/PHI:
            gap_name = "attractor→noble_1"
        else:
            gap_name = "noble_1→boundary"

        marker = " ***" if name in CORE_PULLERS else ""
        print(f"  {name:<15} {u_val:>6.3f} {d_nearest:>10.3f} {nearest:>12} "
              f"{enrich:>+10.1f}% {gap_name:>20}{marker}")

    # Correlation: distance from nearest puller vs enrichment
    ds = []
    enrichments = []
    for name, u_val in sorted_pos:
        if name in CORE_PULLERS:
            continue
        d_nearest = min(circ_dist(u_val, pv) for pv in CORE_PULLERS.values())
        d = density_at_position(all_u, u_val, bw)
        enrich = (d / null_d - 1) * 100
        ds.append(d_nearest)
        enrichments.append(enrich)

    r, p = stats.pearsonr(ds, enrichments)
    print(f"\n  Distance-from-puller vs enrichment: r = {r:.3f}, p = {p:.4f}")
    print(f"  (Negative r = farther from puller → more push)")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
print("Loading datasets...")

datasets = []

lemon_path = 'exports_lemon/per_subject_overlap_trim_f07.83/dominant_peak/per_subject_dominant_peaks.csv'
if os.path.exists(lemon_path):
    datasets.append(('LEMON', pd.read_csv(lemon_path)))

eegmmidb_path = 'exports_eegmmidb/per_subject_overlap_trim_f07.83/dominant_peak/per_subject_dominant_peaks.csv'
if os.path.exists(eegmmidb_path):
    datasets.append(('EEGMMIDB', pd.read_csv(eegmmidb_path)))

dort_path = '/Volumes/T9/dortmund_data/lattice_results_ot/dortmund_ot_dominant_peaks_EyesClosed_pre.csv'
if os.path.exists(dort_path):
    datasets.append(('Dortmund', pd.read_csv(dort_path)))

# Analysis 3 is dataset-independent
analysis3_frequency_mapping()

for name, df in datasets:
    analysis1_boundary_kde(df, name)
    analysis2_band_decomposition(df, name)
    analysis4_f0_sensitivity(df, name)
    analysis5_gap_structure(df, name)

print("\n\nDone.")

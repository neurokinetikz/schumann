#!/usr/bin/env python3
"""
Dominant-Peak Lattice Analysis — Bonn EEG Dataset
==================================================

Applies the phi-lattice dominant-peak analysis to the Bonn epilepsy dataset
(Andrzejak et al., Phys. Rev. E, 64, 061907, 2001).

Five sets (100 single-channel segments each, 23.6 s, 173.61 Hz):
  Z (Set A): Healthy volunteers, eyes open, surface EEG
  O (Set B): Healthy volunteers, eyes closed, surface EEG
  N (Set C): Epilepsy patients, interictal, hippocampal (opposite hemisphere)
  F (Set D): Epilepsy patients, interictal, epileptogenic zone
  S (Set E): Epilepsy patients, ictal (seizure)

Pipeline:
  1. Load raw segments, apply 40 Hz low-pass filter
  2. Welch PSD → FOOOF (specparam) peak extraction
  3. Extract single dominant peak per band (delta, theta, alpha, gamma)
  4. Compute phi-lattice coordinate u and distance d per peak
  5. Population-level and per-segment significance tests
  6. Cross-base comparison (phi vs 8 alternative bases)
  7. Cross-condition comparison

Reference: Andrzejak RG et al. (2001) Phys. Rev. E, 64, 061907
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from specparam import SpectralModel
import os
import warnings
warnings.filterwarnings('ignore')

# ── Constants ──────────────────────────────────────────────────────────────

PHI = 1.6180339887
F0 = 7.83  # Pre-specified Schumann fundamental (NOT optimized)

BONN_DIR = '/Volumes/T9/bonn_data'
OUT_DIR = '/Volumes/T9/bonn_data/lattice_results'
FS = 173.61  # Bonn sampling rate

SETS = {
    'Z': ('Set A', 'Healthy eyes-open (surface)'),
    'O': ('Set B', 'Healthy eyes-closed (surface)'),
    'N': ('Set C', 'Interictal opposite-hemisphere (depth)'),
    'F': ('Set D', 'Interictal epileptogenic zone (depth)'),
    'S': ('Set E', 'Ictal / seizure (depth)'),
}

# Standard 4-band definitions (matching LEMON/EEGMMIDB)
BANDS = {
    'delta': (1.0, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'gamma': (30.0, 45.0),
}

# Phi positions within each octave (8 positions)
PHI_POSITIONS = {
    'boundary':    0.000,
    'noble_4':     PHI**-4,   # 0.1459
    'noble_3':     PHI**-3,   # 0.2361
    'noble_2':     1 - 1/PHI, # 0.3820
    'attractor':   0.500,
    'noble_1':     1/PHI,     # 0.6180
    'inv_noble_3': 1 - PHI**-3, # 0.7639
    'inv_noble_4': 1 - PHI**-4, # 0.8541
}

# 9 comparison bases
BASES = {
    'phi':    PHI,
    '1.4':    1.4,
    'sqrt2':  np.sqrt(2),
    '3/2':    1.5,
    '1.7':    1.7,
    '1.8':    1.8,
    '2':      2.0,
    'e':      np.e,
    'pi':     np.pi,
}


# ── Lattice math ──────────────────────────────────────────────────────────

def lattice_coord(freq, f0=F0, base=PHI):
    """Map frequency to lattice coordinate u in [0, 1)."""
    if freq <= 0 or f0 <= 0:
        return np.nan
    return (np.log(freq / f0) / np.log(base)) % 1.0


def min_lattice_dist(u, positions=None):
    """Minimum circular distance from u to any position."""
    if positions is None:
        positions = PHI_POSITIONS
    if np.isnan(u):
        return np.nan
    d_min = 0.5  # max possible
    for p in positions.values():
        d = abs(u - p)
        d = min(d, 1 - d)  # circular wrap
        if d < d_min:
            d_min = d
    return d_min


def nearest_position_name(u, positions=None):
    """Name of nearest position to u."""
    if positions is None:
        positions = PHI_POSITIONS
    if np.isnan(u):
        return 'none'
    best_name = 'boundary'
    d_min = 0.5
    for name, p in positions.items():
        d = abs(u - p)
        d = min(d, 1 - d)
        if d < d_min:
            d_min = d
            best_name = name
    return best_name


def positions_for_base(base):
    """Degree-3 symmetric: forward AND inverse positions through degree 3.
    No special-casing for any base. Generates: boundary, attractor,
    inv^k, 1-inv^k for k=1,2,3. Filtered for uniqueness (>0.02 sep)."""
    MIN_SEP = 0.02
    inv = 1.0 / base
    pos = {'boundary': 0.0, 'attractor': 0.5}

    def _try_add(name, val):
        if val < MIN_SEP or val > 1 - MIN_SEP:
            return
        if abs(val - 0.5) < MIN_SEP:
            return
        if all(abs(val - v) > MIN_SEP for v in pos.values()):
            pos[name] = val

    _try_add('noble', inv)
    _try_add('inv_noble', 1 - inv)
    _try_add('noble_2', inv ** 2)
    _try_add('inv_noble_2', 1 - inv ** 2)
    _try_add('noble_3', inv ** 3)
    _try_add('inv_noble_3', 1 - inv ** 3)
    return pos


# ── Data loading ──────────────────────────────────────────────────────────

def load_bonn_set(prefix):
    """Load all 100 segments for a Bonn set. Returns list of 1D arrays."""
    folder = os.path.join(BONN_DIR, prefix)
    files = sorted([f for f in os.listdir(folder)
                    if f.lower().endswith('.txt') and not f.startswith('._')])
    segments = []
    for f in files:
        with open(os.path.join(folder, f), 'r', encoding='latin-1') as fh:
            values = []
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        values.append(float(line))
                    except ValueError:
                        pass
        segments.append(np.array(values[:4096]))
    return segments, files


# ── FOOOF peak extraction ────────────────────────────────────────────────

def extract_peaks_segment(seg, fs=FS, lpf=40.0):
    """
    Extract FOOOF peaks from a single segment.

    1. Apply 40 Hz low-pass (as per Andrzejak et al.)
    2. Welch PSD (2s windows, 50% overlap)
    3. FOOOF fit [1, 45] Hz, max 20 peaks

    Returns list of dicts with freq, power, bandwidth.
    """
    # Low-pass filter at 40 Hz (4th-order Butterworth)
    ny = 0.5 * fs
    b, a = signal.butter(4, lpf / ny, btype='low')
    seg_filt = signal.filtfilt(b, a, seg)

    # Welch PSD
    nperseg = int(2.0 * fs)  # 2-second windows
    noverlap = nperseg // 2
    freqs, psd = signal.welch(seg_filt, fs=fs, nperseg=nperseg,
                               noverlap=noverlap, window='hann')

    # Restrict to [1, 45] Hz for FOOOF
    mask = (freqs >= 1.0) & (freqs <= 45.0)
    freqs_fit = freqs[mask]
    psd_fit = psd[mask]

    # Avoid log of zero
    psd_fit = np.maximum(psd_fit, 1e-20)

    # FOOOF fit (specparam API: use get_params())
    sm = SpectralModel(
        peak_width_limits=(1.0, 12.0),
        max_n_peaks=20,
        min_peak_height=0.01,
        peak_threshold=1.0,
        aperiodic_mode='fixed',
    )

    try:
        sm.fit(freqs_fit, psd_fit, freq_range=[1, 45])
        peaks_params = sm.get_params('peak')   # (n_peaks, 3): CF, PW, BW
        aperiodic = sm.get_params('aperiodic')  # [offset, exponent]
    except Exception:
        return [], (np.nan, np.nan)

    peaks = []
    if peaks_params is not None and peaks_params.ndim == 2 and len(peaks_params) > 0:
        for cf, pw, bw in peaks_params:
            peaks.append({'freq': cf, 'power': pw, 'bandwidth': bw})
    elif peaks_params is not None and peaks_params.ndim == 1 and len(peaks_params) == 3:
        # Single peak case
        cf, pw, bw = peaks_params
        peaks.append({'freq': cf, 'power': pw, 'bandwidth': bw})

    return peaks, tuple(aperiodic)


# ── Dominant-peak analysis ────────────────────────────────────────────────

def run_dominant_peak_analysis(segments, filenames, set_prefix, set_label):
    """
    Full dominant-peak lattice analysis for one Bonn set.

    Each segment = 1 "subject" (they're randomized across subjects/contacts anyway).
    """
    print(f"\n{'='*70}")
    print(f"  {set_prefix} — {set_label}")
    print(f"  {len(segments)} segments, {FS} Hz, {len(segments[0])/FS:.1f}s each")
    print(f"{'='*70}")

    # Step 1: Extract all FOOOF peaks
    all_peaks = []
    aperiodic_params = []
    n_peaks_per_seg = []

    for i, (seg, fname) in enumerate(zip(segments, filenames)):
        peaks, ap = extract_peaks_segment(seg)
        seg_id = os.path.splitext(fname)[0]
        for p in peaks:
            p['segment'] = seg_id
            p['seg_idx'] = i
        all_peaks.extend(peaks)
        aperiodic_params.append(ap)
        n_peaks_per_seg.append(len(peaks))

    print(f"\n  FOOOF extraction: {len(all_peaks)} peaks from {len(segments)} segments")
    print(f"  Peaks per segment: {np.mean(n_peaks_per_seg):.1f} ± {np.std(n_peaks_per_seg):.1f} "
          f"(range {min(n_peaks_per_seg)}-{max(n_peaks_per_seg)})")

    ap_arr = np.array(aperiodic_params)
    valid_ap = ap_arr[~np.isnan(ap_arr[:, 0])]
    if len(valid_ap) > 0:
        print(f"  Aperiodic exponent: {valid_ap[:, 1].mean():.2f} ± {valid_ap[:, 1].std():.2f}")

    if len(all_peaks) == 0:
        print("  WARNING: No peaks extracted!")
        return None

    peaks_df = pd.DataFrame(all_peaks)

    # Step 2: Extract dominant peak per band per segment
    records = []
    for i in range(len(segments)):
        seg_peaks = peaks_df[peaks_df['seg_idx'] == i]
        seg_id = filenames[i].split('.')[0]
        row = {'segment': seg_id, 'seg_idx': i, 'set': set_prefix}

        n_bands_found = 0
        for band_name, (lo, hi) in BANDS.items():
            bp = seg_peaks[(seg_peaks['freq'] >= lo) & (seg_peaks['freq'] < hi)]
            if len(bp) > 0:
                idx = bp['power'].idxmax()
                freq = bp.loc[idx, 'freq']
                power = bp.loc[idx, 'power']
                u = lattice_coord(freq, f0=F0, base=PHI)
                d = min_lattice_dist(u, PHI_POSITIONS)
                nearest = nearest_position_name(u, PHI_POSITIONS)

                row[f'{band_name}_freq'] = freq
                row[f'{band_name}_power'] = power
                row[f'{band_name}_u'] = u
                row[f'{band_name}_d'] = d
                row[f'{band_name}_nearest'] = nearest
                n_bands_found += 1
            else:
                row[f'{band_name}_freq'] = np.nan
                row[f'{band_name}_power'] = np.nan
                row[f'{band_name}_u'] = np.nan
                row[f'{band_name}_d'] = np.nan
                row[f'{band_name}_nearest'] = 'none'

        row['n_bands'] = n_bands_found
        # mean_d across available bands
        ds = [row[f'{b}_d'] for b in BANDS if not np.isnan(row.get(f'{b}_d', np.nan))]
        row['mean_d'] = np.mean(ds) if ds else np.nan

        records.append(row)

    dom = pd.DataFrame(records)

    # Filter to segments with all 4 bands
    valid = dom[dom['n_bands'] == 4].copy()
    n_valid = len(valid)
    n_total = len(dom)

    print(f"\n  Dominant peaks: {n_valid}/{n_total} segments have all 4 bands")

    # Show band coverage
    for band_name in BANDS:
        n_with = dom[f'{band_name}_freq'].notna().sum()
        print(f"    {band_name}: {n_with}/100 segments with peak")

    # If too few complete segments, also analyze with ≥3 bands
    if n_valid < 30:
        valid_3 = dom[dom['n_bands'] >= 3].copy()
        print(f"  (Using ≥3 bands: {len(valid_3)}/100 segments)")
        analysis_df = valid_3
        min_bands = 3
    else:
        analysis_df = valid
        min_bands = 4

    if len(analysis_df) < 10:
        print("  WARNING: Too few segments with sufficient bands for analysis!")
        return {'dom': dom, 'valid': analysis_df, 'set': set_prefix}

    # ── Step 3: Population-level alignment ──
    print(f"\n  --- Population-Level Alignment (f0={F0} Hz) ---")

    # Population median frequencies per band
    pop_freqs = {}
    pop_us = {}
    for band_name in BANDS:
        col = f'{band_name}_freq'
        freqs = analysis_df[col].dropna()
        if len(freqs) > 0:
            med = freqs.median()
            pop_freqs[band_name] = med
            pop_us[band_name] = lattice_coord(med)
            u = pop_us[band_name]
            d = min_lattice_dist(u)
            near = nearest_position_name(u)
            print(f"    {band_name}: median={med:.2f} Hz, u={u:.3f}, d={d:.3f} → {near}")

    pop_ds = [min_lattice_dist(u) for u in pop_us.values()]
    pop_mean_d = np.mean(pop_ds)
    print(f"    Population mean_d = {pop_mean_d:.4f}")

    # Null: uniform random frequencies per band (100K permutations)
    n_perm = 100_000
    null_pop = np.zeros(n_perm)
    for i in range(n_perm):
        ds = []
        for band_name, (lo, hi) in BANDS.items():
            if band_name in pop_us:
                rand_freq = np.random.uniform(lo, hi)
                ds.append(min_lattice_dist(lattice_coord(rand_freq)))
        null_pop[i] = np.mean(ds)

    p_pop = (null_pop <= pop_mean_d).mean()
    print(f"    Uniform null: p = {p_pop:.4f} (100K permutations)")

    # ── Step 4: Per-segment alignment ──
    print(f"\n  --- Per-Segment Alignment ---")

    mean_ds = analysis_df['mean_d'].values
    obs_mean = mean_ds.mean()
    obs_sd = mean_ds.std()

    # Expected d under uniform null
    # For 8 phi positions, compute expected distance analytically
    pos_vals = sorted(PHI_POSITIONS.values())
    # Add wrap-around
    gaps = []
    for j in range(len(pos_vals)):
        if j < len(pos_vals) - 1:
            gaps.append(pos_vals[j+1] - pos_vals[j])
        else:
            gaps.append(1.0 - pos_vals[j] + pos_vals[0])
    # Expected distance = sum(gap^2) / (4 * sum(gap)) ... actually for uniform on circle:
    # E[d] = sum_i (gap_i^2 / (4 * total)) for equidistributed case
    # Simpler: simulate
    null_expected = np.mean([min_lattice_dist(np.random.uniform(0, 1))
                            for _ in range(100_000)])

    t_stat, p_ttest = stats.ttest_1samp(mean_ds, null_expected)
    cohen_d = (null_expected - obs_mean) / obs_sd if obs_sd > 0 else 0.0

    # Wilcoxon signed-rank
    try:
        w_stat, p_wilcox = stats.wilcoxon(mean_ds - null_expected, alternative='less')
    except Exception:
        p_wilcox = np.nan

    print(f"    Observed mean_d: {obs_mean:.4f} ± {obs_sd:.4f}")
    print(f"    Expected (null): {null_expected:.4f}")
    print(f"    t = {t_stat:.2f}, p = {p_ttest:.2e}")
    print(f"    Wilcoxon p = {p_wilcox:.2e}")
    print(f"    Cohen's d = {cohen_d:.3f}")

    # How many individually significant?
    n_sig = 0
    for _, row in analysis_df.iterrows():
        seg_d = row['mean_d']
        # Compare to null distribution of mean_d for n_bands
        n_b = int(row['n_bands'])
        null_seg = np.array([np.mean([min_lattice_dist(np.random.uniform(0, 1))
                                       for _ in range(n_b)])
                              for _ in range(1000)])
        if seg_d <= np.percentile(null_seg, 5):
            n_sig += 1
    print(f"    Individually significant (p<0.05): {n_sig}/{len(analysis_df)} "
          f"({100*n_sig/len(analysis_df):.1f}% vs 5% expected)")

    # ── Step 5: Per-band breakdown ──
    print(f"\n  --- Per-Band Breakdown ---")
    band_results = {}
    for band_name, (lo, hi) in BANDS.items():
        col_d = f'{band_name}_d'
        col_u = f'{band_name}_u'
        ds = analysis_df[col_d].dropna().values
        if len(ds) < 5:
            print(f"    {band_name}: insufficient data ({len(ds)} segments)")
            continue

        band_null = np.array([min_lattice_dist(lattice_coord(np.random.uniform(lo, hi)))
                              for _ in range(10_000)])

        # KS test
        ks_stat, ks_p = stats.ks_2samp(ds, band_null, alternative='less')

        # Effect size
        band_d = (band_null.mean() - ds.mean()) / ds.std() if ds.std() > 0 else 0

        # Position distribution
        nearests = analysis_df[f'{band_name}_nearest'].dropna().value_counts()
        top_pos = nearests.index[0] if len(nearests) > 0 else 'none'
        top_pct = 100 * nearests.iloc[0] / nearests.sum() if len(nearests) > 0 else 0

        print(f"    {band_name}: mean_d={ds.mean():.4f}, null={band_null.mean():.4f}, "
              f"KS p={ks_p:.3e}, d={band_d:.2f}  | top: {top_pos} ({top_pct:.0f}%)")

        band_results[band_name] = {
            'mean_d': ds.mean(), 'null_d': band_null.mean(),
            'ks_p': ks_p, 'cohen_d': band_d, 'top_position': top_pos
        }

    # ── Step 6: Cross-base comparison ──
    print(f"\n  --- Cross-Base Comparison (9 bases) ---")
    base_results = {}

    for base_name, base_val in BASES.items():
        positions = positions_for_base(base_val)

        seg_ds = []
        for _, row in analysis_df.iterrows():
            band_ds = []
            for band_name in BANDS:
                freq = row[f'{band_name}_freq']
                if np.isnan(freq):
                    continue
                u = lattice_coord(freq, f0=F0, base=base_val)
                d = min_lattice_dist(u, positions)
                band_ds.append(d)
            if band_ds:
                seg_ds.append(np.mean(band_ds))

        seg_ds = np.array(seg_ds)
        base_results[base_name] = {
            'mean_d': seg_ds.mean(),
            'median_d': np.median(seg_ds),
            'sd_d': seg_ds.std(),
            'n_positions': len(positions),
            'values': seg_ds,
        }

    # Rank by mean_d (lower = better alignment)
    ranking = sorted(base_results.items(), key=lambda x: x[1]['mean_d'])

    for rank, (bname, br) in enumerate(ranking, 1):
        marker = ' ←' if bname == 'phi' else ''
        print(f"    {rank}. {bname:6s}: mean_d={br['mean_d']:.4f} "
              f"(±{br['sd_d']:.4f}, {br['n_positions']} pos){marker}")

    # Phi vs others: paired t-tests
    phi_ds = base_results['phi']['values']
    print(f"\n    Phi vs competitors (paired t-test):")
    for bname, br in ranking:
        if bname == 'phi':
            continue
        other_ds = br['values']
        if len(phi_ds) == len(other_ds):
            t, p = stats.ttest_rel(phi_ds, other_ds, alternative='less')
            d_diff = (other_ds.mean() - phi_ds.mean()) / np.sqrt((phi_ds.std()**2 + other_ds.std()**2)/2)
            sig = '*' if p < 0.05 else ' '
            print(f"      phi vs {bname:6s}: Δd={other_ds.mean()-phi_ds.mean():+.4f}, "
                  f"p={p:.3e} {sig}")

    # ── Step 7: Noble position contribution ──
    print(f"\n  --- Noble Position Contribution ---")

    # Phi with only boundary+attractor (shared by all bases)
    shared_pos = {'boundary': 0.0, 'attractor': 0.5}
    shared_ds = []
    for _, row in analysis_df.iterrows():
        band_ds = []
        for band_name in BANDS:
            freq = row[f'{band_name}_freq']
            if np.isnan(freq):
                continue
            u = lattice_coord(freq, f0=F0, base=PHI)
            d = min_lattice_dist(u, shared_pos)
            band_ds.append(d)
        if band_ds:
            shared_ds.append(np.mean(band_ds))
    shared_ds = np.array(shared_ds)

    full_mean = base_results['phi']['mean_d']
    shared_mean = shared_ds.mean()
    if shared_mean > 0:
        noble_contrib = (1 - full_mean / shared_mean) * 100
    else:
        noble_contrib = 0

    print(f"    Full phi (8 pos): mean_d = {full_mean:.4f}")
    print(f"    Shared only (2 pos): mean_d = {shared_mean:.4f}")
    print(f"    Noble contribution: {noble_contrib:.1f}%")

    # Return everything
    return {
        'dom': dom,
        'valid': analysis_df,
        'set': set_prefix,
        'label': set_label,
        'pop_mean_d': pop_mean_d,
        'p_pop': p_pop,
        'obs_mean_d': obs_mean,
        'null_expected_d': null_expected,
        'cohen_d': cohen_d,
        'p_ttest': p_ttest,
        'p_wilcox': p_wilcox,
        'n_valid': len(analysis_df),
        'min_bands': min_bands,
        'base_results': base_results,
        'band_results': band_results,
        'noble_contrib': noble_contrib,
        'aperiodic_params': ap_arr,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_results = {}
    all_dom_dfs = []

    for prefix, (label, desc) in SETS.items():
        segments, filenames = load_bonn_set(prefix)
        result = run_dominant_peak_analysis(segments, filenames, prefix, f"{label}: {desc}")

        if result is not None:
            all_results[prefix] = result
            if 'dom' in result:
                all_dom_dfs.append(result['dom'])

    # ── Grand summary across conditions ──
    print(f"\n\n{'='*70}")
    print(f"  GRAND SUMMARY — Cross-Condition Comparison")
    print(f"{'='*70}")

    print(f"\n  {'Set':<8s} {'Condition':<45s} {'N':>3s} {'mean_d':>7s} {'null':>7s} "
          f"{'d':>6s} {'p_pop':>8s} {'p_seg':>8s} {'phi_rank':>8s}")
    print(f"  {'-'*8} {'-'*45} {'-'*3} {'-'*7} {'-'*7} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")

    for prefix in ['Z', 'O', 'N', 'F', 'S']:
        if prefix not in all_results:
            continue
        r = all_results[prefix]
        # Find phi rank
        ranking = sorted(r['base_results'].items(), key=lambda x: x[1]['mean_d'])
        phi_rank = next(i+1 for i, (name, _) in enumerate(ranking) if name == 'phi')

        print(f"  {prefix:<8s} {r['label']:<45s} {r['n_valid']:>3d} "
              f"{r['obs_mean_d']:>7.4f} {r['null_expected_d']:>7.4f} "
              f"{r['cohen_d']:>6.3f} {r['p_pop']:>8.4f} {r['p_ttest']:>8.2e} "
              f"{phi_rank:>8d}")

    # Cross-condition ANOVA on mean_d
    print(f"\n  --- Cross-Condition ANOVA on mean_d ---")
    groups = []
    group_labels = []
    for prefix in ['Z', 'O', 'N', 'F', 'S']:
        if prefix in all_results:
            r = all_results[prefix]
            groups.append(r['valid']['mean_d'].values)
            group_labels.append(prefix)

    if len(groups) >= 2:
        F_stat, p_anova = stats.f_oneway(*groups)
        print(f"    F({len(groups)-1}, {sum(len(g) for g in groups)-len(groups)}) = {F_stat:.2f}, "
              f"p = {p_anova:.4f}")

        # Pairwise comparisons (healthy vs epileptic)
        print(f"\n  --- Pairwise Comparisons (Mann-Whitney) ---")
        comparisons = [
            ('Z', 'O', 'Healthy EO vs EC'),
            ('Z', 'S', 'Healthy EO vs Seizure'),
            ('O', 'S', 'Healthy EC vs Seizure'),
            ('Z', 'F', 'Healthy EO vs Epileptogenic'),
            ('N', 'F', 'Opposite vs Epileptogenic'),
            ('F', 'S', 'Epileptogenic interictal vs ictal'),
        ]
        for p1, p2, label in comparisons:
            if p1 in all_results and p2 in all_results:
                d1 = all_results[p1]['valid']['mean_d'].values
                d2 = all_results[p2]['valid']['mean_d'].values
                u_stat, p_mw = stats.mannwhitneyu(d1, d2, alternative='two-sided')
                effect_d = (d1.mean() - d2.mean()) / np.sqrt((d1.std()**2 + d2.std()**2) / 2)
                print(f"    {label:<35s}: d={effect_d:+.3f}, p={p_mw:.4f}")

    # 1/f slopes across conditions
    print(f"\n  --- Aperiodic Exponents by Condition ---")
    for prefix in ['Z', 'O', 'N', 'F', 'S']:
        if prefix in all_results:
            ap = all_results[prefix]['aperiodic_params']
            valid_exp = ap[~np.isnan(ap[:, 1]), 1]
            if len(valid_exp) > 0:
                print(f"    {prefix}: exponent = {valid_exp.mean():.2f} ± {valid_exp.std():.2f}")

    # Save combined results
    if all_dom_dfs:
        combined = pd.concat(all_dom_dfs, ignore_index=True)
        combined.to_csv(os.path.join(OUT_DIR, 'bonn_dominant_peaks_all.csv'), index=False)
        print(f"\n  Saved: {os.path.join(OUT_DIR, 'bonn_dominant_peaks_all.csv')}")

    # Save summary table
    summary_rows = []
    for prefix in ['Z', 'O', 'N', 'F', 'S']:
        if prefix not in all_results:
            continue
        r = all_results[prefix]
        ranking = sorted(r['base_results'].items(), key=lambda x: x[1]['mean_d'])
        phi_rank = next(i+1 for i, (name, _) in enumerate(ranking) if name == 'phi')

        summary_rows.append({
            'set': prefix,
            'condition': r['label'],
            'n_segments': r['n_valid'],
            'min_bands': r.get('min_bands', 4),
            'obs_mean_d': r['obs_mean_d'],
            'null_expected_d': r['null_expected_d'],
            'cohen_d': r['cohen_d'],
            'p_population': r['p_pop'],
            'p_per_segment': r['p_ttest'],
            'p_wilcoxon': r['p_wilcox'],
            'phi_rank': phi_rank,
            'noble_contribution_pct': r['noble_contrib'],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUT_DIR, 'bonn_summary.csv'), index=False)
    print(f"  Saved: {os.path.join(OUT_DIR, 'bonn_summary.csv')}")

    print(f"\n  Done!")


if __name__ == '__main__':
    main()

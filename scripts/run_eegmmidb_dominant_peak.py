#!/usr/bin/env python3
"""
Dominant-Peak Alignment Analysis — EEGMMIDB Replication
========================================================

Replicates the LEMON dominant-peak analysis (Paper 3) on EEGMMIDB data.

For each subject (N=109):
  - Pool all FOOOF peaks across 14 sessions and 64 channels
  - Extract the single strongest peak per canonical band (δ, θ, α, γ)
  - Compute lattice coordinate u = log_φ(f/f₀) mod 1
  - Compute mean distance to nearest phi-lattice position

Tests:
  1. Population alignment (population median peaks vs uniform null)
  2. Per-subject alignment (t-test, Wilcoxon against theoretical null)
  3. Per-band decomposition
  4. Cross-base comparison (9 bases)
  5. Noble position contribution

LEMON reference values (to replicate):
  - mean_d = 0.069, t = -5.72, d = 0.40, p < 10⁻⁷
  - Population d̄ = 0.027, p = 0.029
  - Noble positions carry 70% of φ advantage

Usage:
    python scripts/run_eegmmidb_dominant_peak.py
    python scripts/run_eegmmidb_dominant_peak.py --input-dir exports_eegmmidb/per_subject_overlap_trim_f07.83
    python scripts/run_eegmmidb_dominant_peak.py --input aggregated_peaks.csv
"""

import sys
import os
import time
import argparse
from glob import glob
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, './lib')

# =========================================================================
# CONSTANTS
# =========================================================================

PHI = (1 + np.sqrt(5)) / 2
F0 = 7.83  # Pre-specified (Schumann fundamental), NOT optimized

POSITIONS = {
    'boundary': 0.000,
    'noble_2': 0.382,
    'attractor': 0.500,
    'noble_1': 0.618,
}

BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'gamma': (30, 45),
}

# All 9 bases for cross-base comparison
BASES = {
    '1.4': 1.4,
    '√2': np.sqrt(2),
    '3/2': 1.5,
    'φ': PHI,
    '1.7': 1.7,
    '1.8': 1.8,
    '2': 2.0,
    'e': np.e,
    'π': np.pi,
}

DEFAULT_INPUT_CSV = 'exports_peak_distribution/eegmmidb_fooof/golden_ratio_peaks_EEGMMIDB.csv'
DEFAULT_OUTPUT_DIR = 'exports_peak_distribution/eegmmidb_fooof/dominant_peak'

# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def lattice_coord(freq, f0=F0, base=PHI):
    return (np.log(freq / f0) / np.log(base)) % 1.0


def min_lattice_dist(u, positions=None):
    if positions is None:
        positions = POSITIONS
    return min(min(abs(u - p), 1 - abs(u - p)) for p in positions.values())


def nearest_pos_name(u, positions=None):
    if positions is None:
        positions = POSITIONS
    dists = {name: min(abs(u - p), 1 - abs(u - p)) for name, p in positions.items()}
    return min(dists, key=dists.get)


def natural_positions_for_base(base):
    """Return the natural lattice positions for any base (degree-3 symmetric).
    No special-casing for any base. Generates: boundary, attractor,
    inv^k, 1-inv^k for k=1,2,3, filtered for uniqueness (>0.02 separation)."""
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

    # Degree 1
    _try_add('noble', inv)
    _try_add('inv_noble', 1 - inv)
    # Degree 2
    _try_add('noble_2', inv ** 2)
    _try_add('inv_noble_2', 1 - inv ** 2)
    # Degree 3
    _try_add('noble_3', inv ** 3)
    _try_add('inv_noble_3', 1 - inv ** 3)

    return pos


# =========================================================================
# MAIN
# =========================================================================

def load_per_subject_dir(input_dir):
    """Load per-subject peak CSVs from a directory into a single DataFrame."""
    files = sorted(glob(os.path.join(input_dir, 'S*_peaks.csv')))
    if not files:
        # Try sub-* pattern (LEMON format)
        files = sorted(glob(os.path.join(input_dir, 'sub-*_peaks.csv')))
    if not files:
        raise FileNotFoundError(f"No peak CSVs found in {input_dir}")

    dfs = []
    for f in files:
        basename = os.path.basename(f)
        sid = basename.replace('_peaks.csv', '')
        df = pd.read_csv(f)
        df['subject'] = sid
        dfs.append(df)

    all_peaks = pd.concat(dfs, ignore_index=True)
    return all_peaks


def main():
    parser = argparse.ArgumentParser(
        description='Dominant-peak alignment analysis on EEGMMIDB')
    parser.add_argument('--input', type=str, default=None,
                        help='Aggregated peaks CSV (with session or subject column)')
    parser.add_argument('--input-dir', type=str, default=None,
                        help='Directory of per-subject peak CSVs (S001_peaks.csv, ...)')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--label', type=str, default='EEGMMIDB',
                        help='Dataset label for output')
    args = parser.parse_args()

    t0 = time.time()

    # Determine input source and output directory
    if args.input_dir:
        INPUT_SOURCE = args.input_dir
        OUTPUT_DIR = args.output_dir or os.path.join(args.input_dir, 'dominant_peak')
    elif args.input:
        INPUT_SOURCE = args.input
        OUTPUT_DIR = args.output_dir or os.path.join(
            os.path.dirname(args.input), 'dominant_peak')
    else:
        INPUT_SOURCE = DEFAULT_INPUT_CSV
        OUTPUT_DIR = args.output_dir or DEFAULT_OUTPUT_DIR

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- LOAD ---
    print("=" * 70)
    print(f"{args.label} DOMINANT-PEAK ALIGNMENT ANALYSIS")
    print("=" * 70)
    print(f"Input: {INPUT_SOURCE}")
    print(f"f₀ = {F0} Hz (pre-specified, not optimized)")
    print(f"Bands: {BANDS}")

    if args.input_dir:
        df = load_per_subject_dir(args.input_dir)
    else:
        csv_path = args.input or DEFAULT_INPUT_CSV
        df = pd.read_csv(csv_path)
        if 'subject' not in df.columns and 'session' in df.columns:
            df['subject'] = df['session'].str[:4]

    n_subj = df['subject'].nunique()
    print(f"\nLoaded {len(df):,} peaks from {n_subj} subjects")
    print(f"Freq range: [{df['freq'].min():.1f}, {df['freq'].max():.1f}] Hz")

    # --- EXTRACT DOMINANT PEAKS PER SUBJECT ---
    print("\n" + "=" * 70)
    print("EXTRACTING DOMINANT PEAKS (strongest per band per subject)")
    print("=" * 70)

    records = []
    for subj in sorted(df['subject'].unique()):
        sdf = df[df['subject'] == subj]
        row = {'subject': subj, 'n_peaks_total': len(sdf)}

        band_ds = []
        for bname, (lo, hi) in BANDS.items():
            bp = sdf[(sdf['freq'] >= lo) & (sdf['freq'] < hi)]
            if len(bp) == 0:
                row[f'{bname}_freq'] = np.nan
                row[f'{bname}_power'] = np.nan
                row[f'{bname}_d'] = np.nan
                continue
            idx = bp['power'].idxmax()
            freq = bp.loc[idx, 'freq']
            power = bp.loc[idx, 'power']
            u = lattice_coord(freq)
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

    dom = pd.DataFrame(records)
    valid = dom[dom['n_bands'] == 4].copy()
    print(f"\nSubjects with all 4 bands: {len(valid)}/{n_subj}")

    # Save per-subject results
    dom.to_csv(os.path.join(OUTPUT_DIR, 'per_subject_dominant_peaks.csv'), index=False)

    # Print dominant peak summary
    print("\nDominant peak frequencies (population medians):")
    for bname in BANDS:
        freqs = valid[f'{bname}_freq'].dropna()
        print(f"  {bname}: median = {freqs.median():.2f} Hz, "
              f"mean = {freqs.mean():.2f} Hz, "
              f"range = [{freqs.min():.2f}, {freqs.max():.2f}]")

    # ============================================================
    # PART 1: POPULATION ALIGNMENT
    # ============================================================
    print("\n" + "=" * 70)
    print("PART 1: POPULATION-LEVEL ALIGNMENT")
    print("=" * 70)

    pop_ds = []
    for bname in BANDS:
        med_freq = valid[f'{bname}_freq'].median()
        u = lattice_coord(med_freq)
        d = min_lattice_dist(u)
        pos = nearest_pos_name(u)
        pop_ds.append(d)
        print(f"  {bname}: median freq = {med_freq:.2f} Hz, u = {u:.3f}, "
              f"d = {d:.4f}, nearest = {pos}")

    pop_mean_d = np.mean(pop_ds)
    print(f"\n  Population mean d̄ = {pop_mean_d:.4f}")

    # Permutation test for population alignment
    np.random.seed(42)
    n_perm = 100_000
    null_pop_ds = np.zeros(n_perm)
    for i in range(n_perm):
        ds = []
        for bname, (lo, hi) in BANDS.items():
            freq = np.random.uniform(lo, hi)
            u = lattice_coord(freq)
            ds.append(min_lattice_dist(u))
        null_pop_ds[i] = np.mean(ds)

    pop_p = (null_pop_ds <= pop_mean_d).mean()
    print(f"  Null mean d̄ = {null_pop_ds.mean():.4f}")
    print(f"  p = {pop_p:.6f} (fraction of null ≤ observed)")

    # ============================================================
    # PART 2: PER-SUBJECT ALIGNMENT
    # ============================================================
    print("\n" + "=" * 70)
    print("PART 2: PER-SUBJECT ALIGNMENT")
    print("=" * 70)

    # Theoretical expected d under uniform null
    gaps = [0.382, 0.118, 0.118, 0.382]
    expected_d = sum(g**2/4 for g in gaps)
    print(f"  Theoretical null expected d = {expected_d:.4f}")

    print(f"\n  Per-subject mean_d distribution:")
    print(f"    Mean:   {valid.mean_d.mean():.4f}")
    print(f"    Median: {valid.mean_d.median():.4f}")
    print(f"    SD:     {valid.mean_d.std():.4f}")
    print(f"    IQR:    [{valid.mean_d.quantile(0.25):.4f}, {valid.mean_d.quantile(0.75):.4f}]")

    # One-sample t-test against null expected
    t_stat, p_ttest = stats.ttest_1samp(valid.mean_d, expected_d)
    d_effect = (expected_d - valid.mean_d.mean()) / valid.mean_d.std()
    print(f"\n  One-sample t-test (H₀: mean_d = {expected_d:.4f}):")
    print(f"    t = {t_stat:.3f}, p = {p_ttest:.2e}")
    print(f"    Cohen's d = {d_effect:.3f}")

    # Wilcoxon
    w_stat, p_wilcox = stats.wilcoxon(valid.mean_d - expected_d)
    print(f"\n  Wilcoxon signed-rank: W = {w_stat:.0f}, p = {p_wilcox:.2e}")

    # Per-subject significance
    null_mean_ds = np.zeros(10_000)
    np.random.seed(42)
    for i in range(10_000):
        ds = []
        for bname, (lo, hi) in BANDS.items():
            freq = np.random.uniform(lo, hi)
            u = lattice_coord(freq)
            ds.append(min_lattice_dist(u))
        null_mean_ds[i] = np.mean(ds)

    subject_ps = [(null_mean_ds <= row['mean_d']).mean() for _, row in valid.iterrows()]
    valid['p_uniform'] = subject_ps

    n_sig_05 = (valid['p_uniform'] < 0.05).sum()
    n_sig_01 = (valid['p_uniform'] < 0.01).sum()
    print(f"\n  Individually significant:")
    print(f"    p < 0.05: {n_sig_05}/{len(valid)} ({n_sig_05/len(valid)*100:.1f}%) "
          f"[expected: {len(valid)*0.05:.0f}]")
    print(f"    p < 0.01: {n_sig_01}/{len(valid)} ({n_sig_01/len(valid)*100:.1f}%) "
          f"[expected: {len(valid)*0.01:.0f}]")

    binom_05 = stats.binomtest(n_sig_05, len(valid), 0.05, alternative='greater')
    print(f"    Binomial test (>5% significant?): p = {binom_05.pvalue:.6f}")

    # ============================================================
    # PART 3: PER-BAND DECOMPOSITION
    # ============================================================
    print("\n" + "=" * 70)
    print("PART 3: PER-BAND ALIGNMENT vs UNIFORM NULL")
    print("=" * 70)

    np.random.seed(42)
    for bname, (lo, hi) in BANDS.items():
        obs_ds = valid[f'{bname}_d'].dropna().values

        null_ds = np.zeros(10_000)
        for i in range(10_000):
            freq = np.random.uniform(lo, hi)
            u = lattice_coord(freq)
            null_ds[i] = min_lattice_dist(u)

        ks_stat, ks_p = stats.ks_2samp(obs_ds, null_ds, alternative='less')
        mw_stat, mw_p = stats.mannwhitneyu(obs_ds, null_ds, alternative='less')

        print(f"\n  {bname} ({lo}-{hi} Hz): N={len(obs_ds)}")
        print(f"    Observed: mean d = {obs_ds.mean():.4f}, median = {np.median(obs_ds):.4f}")
        print(f"    Null:     mean d = {null_ds.mean():.4f}, median = {np.median(null_ds):.4f}")
        print(f"    KS (obs < null): D = {ks_stat:.4f}, p = {ks_p:.2e}")
        print(f"    Mann-Whitney (obs < null): p = {mw_p:.2e}")

        # Position distribution
        pos_counts = valid[f'{bname}_nearest'].value_counts()
        pos_str = ', '.join(f"{p}: {pos_counts.get(p, 0)}" for p in POSITIONS)
        print(f"    Positions: {pos_str}")

    # ============================================================
    # PART 4: CROSS-BASE COMPARISON (9 bases)
    # ============================================================
    print("\n" + "=" * 70)
    print("PART 4: CROSS-BASE DOMINANT-PEAK COMPARISON")
    print("=" * 70)

    base_results = {}
    for base_name, base_val in sorted(BASES.items(), key=lambda x: x[1]):
        positions = natural_positions_for_base(base_val)

        per_subj_ds = []
        for _, row in valid.iterrows():
            band_ds = []
            for bname in BANDS:
                freq = row[f'{bname}_freq']
                if np.isnan(freq):
                    continue
                u = (np.log(freq / F0) / np.log(base_val)) % 1.0
                d = min(min(abs(u - p), 1 - abs(u - p)) for p in positions.values())
                band_ds.append(d)
            if len(band_ds) == 4:
                per_subj_ds.append(np.mean(band_ds))

        per_subj_ds = np.array(per_subj_ds)
        base_results[base_name] = per_subj_ds

        # Population-level
        pop_ds = []
        for bname in BANDS:
            med_freq = valid[f'{bname}_freq'].median()
            u = (np.log(med_freq / F0) / np.log(base_val)) % 1.0
            d = min(min(abs(u - p), 1 - abs(u - p)) for p in positions.values())
            pop_ds.append(d)
        pop_d = np.mean(pop_ds)

        print(f"  {base_name:>4s}: per-subj mean_d = {per_subj_ds.mean():.4f}, "
              f"population d̄ = {pop_d:.4f} ({len(positions)} positions)")

    # Ranking
    print("\n  Ranking by per-subject mean_d:")
    ranking = sorted(base_results.items(), key=lambda x: x[1].mean())
    for rank, (bname, ds) in enumerate(ranking, 1):
        marker = " <<<" if bname == 'φ' else ""
        print(f"    {rank}. {bname:>4s}: {ds.mean():.4f}{marker}")

    # Paired tests: phi vs each
    print("\n  Paired tests (φ vs each base):")
    phi_ds = base_results['φ']
    for bname, other_ds in sorted(base_results.items(), key=lambda x: x[1].mean()):
        if bname == 'φ':
            continue
        t, p = stats.ttest_rel(phi_ds, other_ds)
        wins = (phi_ds < other_ds).mean()
        sig = "*" if p < 0.05 else ""
        print(f"    φ vs {bname:>4s}: φ wins {wins*100:.1f}%, "
              f"paired t = {t:+.3f}, p = {p:.4f}{sig}")

    # ============================================================
    # PART 5: NOBLE POSITION CONTRIBUTION
    # ============================================================
    print("\n" + "=" * 70)
    print("PART 5: NOBLE POSITION CONTRIBUTION (φ-specific)")
    print("=" * 70)

    # Compute alignment with only boundary + attractor (shared by all bases)
    shared_pos = {'boundary': 0.0, 'attractor': 0.5}
    shared_ds = []
    for _, row in valid.iterrows():
        band_ds = []
        for bname in BANDS:
            freq = row[f'{bname}_freq']
            if np.isnan(freq):
                continue
            u = lattice_coord(freq)
            d = min(min(abs(u - p), 1 - abs(u - p)) for p in shared_pos.values())
            band_ds.append(d)
        if len(band_ds) == 4:
            shared_ds.append(np.mean(band_ds))
    shared_ds = np.array(shared_ds)

    full_ds = base_results['φ']

    # Population level
    pop_full = []
    pop_shared = []
    for bname in BANDS:
        med_freq = valid[f'{bname}_freq'].median()
        u = lattice_coord(med_freq)
        d_full = min_lattice_dist(u)
        d_shared = min(min(abs(u - p), 1 - abs(u - p)) for p in shared_pos.values())
        pop_full.append(d_full)
        pop_shared.append(d_shared)

    pop_d_full = np.mean(pop_full)
    pop_d_shared = np.mean(pop_shared)

    noble_frac = 1 - (pop_d_full / pop_d_shared) if pop_d_shared > 0 else 0

    print(f"  Population d̄ (full φ, 4 positions): {pop_d_full:.4f}")
    print(f"  Population d̄ (boundary+attractor only): {pop_d_shared:.4f}")
    print(f"  Noble position contribution: {noble_frac*100:.0f}%")
    print(f"\n  Per-subject d̄ (full φ): {full_ds.mean():.4f}")
    print(f"  Per-subject d̄ (boundary+attractor): {shared_ds.mean():.4f}")
    noble_frac_subj = 1 - (full_ds.mean() / shared_ds.mean()) if shared_ds.mean() > 0 else 0
    print(f"  Noble contribution (per-subject): {noble_frac_subj*100:.0f}%")

    # ============================================================
    # COMPARISON WITH LEMON
    # ============================================================
    print("\n" + "=" * 70)
    print("COMPARISON: EEGMMIDB vs LEMON")
    print("=" * 70)

    label = args.label
    print(f"""
  {'Metric':<35s} {'LEMON (N=202)':>15s} {f'{label} (N={len(valid)})':>15s}
  {'-'*35} {'-'*15} {'-'*15}
  {'Per-subject mean_d':<35s} {'0.069':>15s} {valid.mean_d.mean():>15.4f}
  {'Null expected d':<35s} {'0.080':>15s} {expected_d:>15.4f}
  {'Cohen d':<35s} {'0.40':>15s} {d_effect:>15.3f}
  {'t-statistic':<35s} {'-5.72':>15s} {t_stat:>15.3f}
  {'p-value':<35s} {'< 10⁻⁷':>15s} {p_ttest:>15.2e}
  {'Population d̄':<35s} {'0.027':>15s} {pop_mean_d:>15.4f}
  {'Population p':<35s} {'0.029':>15s} {pop_p:>15.6f}
  {'Noble contribution':<35s} {'70%':>15s} {noble_frac*100:>14.0f}%
  {'φ rank (per-subject d̄)':<35s} {'1st':>15s} {'':>15s}
""")

    phi_rank = [name for name, _ in ranking].index('φ') + 1
    print(f"  φ rank in EEGMMIDB: {phi_rank} of {len(BASES)}")
    print(f"  REPLICATION: {'YES (d > 0.30)' if d_effect > 0.30 else 'PARTIAL (0.10 < d < 0.30)' if d_effect > 0.10 else 'NO (d < 0.10)'}")

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    summary = {
        'dataset': 'eegmmidb',
        'n_subjects': len(valid),
        'f0': F0,
        'mean_d': valid.mean_d.mean(),
        'median_d': valid.mean_d.median(),
        'sd_d': valid.mean_d.std(),
        'null_expected_d': expected_d,
        'cohen_d': d_effect,
        't_stat': t_stat,
        'p_ttest': p_ttest,
        'p_wilcoxon': p_wilcox,
        'pop_mean_d': pop_mean_d,
        'pop_p': pop_p,
        'noble_contribution_pct': noble_frac * 100,
        'phi_rank': phi_rank,
        'n_sig_05': n_sig_05,
        'n_sig_01': n_sig_01,
    }
    pd.DataFrame([summary]).to_csv(os.path.join(OUTPUT_DIR, 'summary.csv'), index=False)

    # Save cross-base results
    base_rows = []
    for base_name, ds in base_results.items():
        base_rows.append({
            'base': base_name,
            'mean_d': ds.mean(),
            'median_d': np.median(ds),
            'sd_d': ds.std(),
        })
    pd.DataFrame(base_rows).to_csv(os.path.join(OUTPUT_DIR, 'cross_base_comparison.csv'), index=False)

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.0f} seconds")
    print(f"Results saved to: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Dortmund Longitudinal Lattice Analysis — Session 1 vs Session 2
===============================================================

208 subjects with ~5-year longitudinal follow-up.
Runs the same dominant-peak pipeline on ses-2 data, then compares
ses-1 vs ses-2 to assess within-subject stability of phi-lattice alignment.

Analyses:
  1. Extract dominant peaks for ses-2 (4 conditions)
  2. Within-subject ICC (ses-1 vs ses-2 mean_d)
  3. Per-band position stability (% same nearest position)
  4. Frequency shift analysis (Δfreq per band)
  5. Cross-base stability (does phi remain rank #1?)
  6. Fatigue invariance (pre vs post, both sessions)
  7. Age-at-retest analysis (are older subjects more/less stable?)

Requires: ses-1 results already computed (lattice_results/*.csv)
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from specparam import SpectralModel
import mne
import os
import sys
import time
import warnings
import traceback
warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

# ── Constants (matching run_dortmund_dominant_peak.py) ─────────────────────

PHI = 1.6180339887
F0 = 7.83

DATA_DIR_DL = '/Volumes/T9/dortmund_data_dl'
DATA_DIR_ORIG = '/Volumes/T9/dortmund_data'
SES1_DIR = '/Volumes/T9/dortmund_data/lattice_results'
OUT_DIR = '/Volumes/T9/dortmund_data/lattice_results_longitudinal'

TARGET_FS = 250
NOTCH_FREQ = 50

CONDITIONS = [
    ('EyesClosed', 'pre'),
    ('EyesOpen', 'pre'),
    ('EyesClosed', 'post'),
    ('EyesOpen', 'post'),
]

BANDS = {
    'delta': (1.0, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'gamma': (30.0, 45.0),
}

PHI_POSITIONS = {
    'boundary':    0.000,
    'noble_4':     PHI**-4,
    'noble_3':     PHI**-3,
    'noble_2':     1 - 1/PHI,
    'attractor':   0.500,
    'noble_1':     1/PHI,
    'inv_noble_3': 1 - PHI**-3,
    'inv_noble_4': 1 - PHI**-4,
}

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


# ── Lattice math (identical to ses-1 script) ──────────────────────────────

def lattice_coord(freq, f0=F0, base=PHI):
    if freq <= 0 or f0 <= 0:
        return np.nan
    return (np.log(freq / f0) / np.log(base)) % 1.0

def min_lattice_dist(u, positions=None):
    if positions is None:
        positions = PHI_POSITIONS
    if np.isnan(u):
        return np.nan
    d_min = 0.5
    for p in positions.values():
        d = abs(u - p)
        d = min(d, 1 - d)
        if d < d_min:
            d_min = d
    return d_min

def nearest_position_name(u, positions=None):
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


# ── Data loading (identical to ses-1 script) ──────────────────────────────

def find_edf(sub_id, session, task, acq, data_dir):
    path = os.path.join(data_dir, sub_id, f'ses-{session}', 'eeg',
                        f'{sub_id}_ses-{session}_task-{task}_acq-{acq}_eeg.edf')
    if os.path.isfile(path) and not (os.path.islink(path) and not os.path.exists(path)):
        return path
    return None

def load_subject_edf(edf_path, target_fs=TARGET_FS):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.pick_types(eeg=True, exclude='bads')
    raw.notch_filter(freqs=[50, 100], verbose=False)
    if raw.info['sfreq'] > target_fs:
        raw.resample(target_fs, verbose=False)
    return raw.get_data(), raw.ch_names, raw.info['sfreq']

def extract_peaks_multichannel(data, ch_names, fs):
    all_peaks = []
    aperiodic_exponents = []
    for i, ch in enumerate(ch_names):
        sig = data[i, :]
        nperseg = int(4.0 * fs)
        noverlap = nperseg // 2
        freqs, psd = signal.welch(sig, fs=fs, nperseg=nperseg,
                                   noverlap=noverlap, window='hann')
        mask = (freqs >= 1.0) & (freqs <= 45.0)
        freqs_fit = freqs[mask]
        psd_fit = psd[mask]
        psd_fit = np.maximum(psd_fit, 1e-30)
        if len(freqs_fit) < 10:
            continue
        sm = SpectralModel(
            peak_width_limits=(1.0, 12.0),
            max_n_peaks=20,
            min_peak_height=0.01,
            peak_threshold=1.0,
            aperiodic_mode='fixed',
        )
        try:
            sm.fit(freqs_fit, psd_fit, freq_range=[1, 45])
            peaks_params = sm.get_params('peak')
            aperiodic = sm.get_params('aperiodic')
            if aperiodic is not None and len(aperiodic) >= 2:
                aperiodic_exponents.append(aperiodic[1])
            if peaks_params is not None:
                if peaks_params.ndim == 2:
                    for cf, pw, bw in peaks_params:
                        all_peaks.append({'channel': ch, 'freq': cf,
                                          'power': pw, 'bandwidth': bw})
                elif peaks_params.ndim == 1 and len(peaks_params) == 3:
                    cf, pw, bw = peaks_params
                    all_peaks.append({'channel': ch, 'freq': cf,
                                      'power': pw, 'bandwidth': bw})
        except Exception:
            continue
    mean_exponent = np.mean(aperiodic_exponents) if aperiodic_exponents else np.nan
    return pd.DataFrame(all_peaks), mean_exponent


def analyze_subject(sub_id, session, task, acq, data_dir, participants_df=None):
    edf_path = find_edf(sub_id, session, task, acq, data_dir)
    if edf_path is None:
        return None
    try:
        data, ch_names, fs = load_subject_edf(edf_path)
    except Exception as e:
        print(f"    ERROR loading {sub_id} ses-{session}: {e}")
        return None

    peaks_df, mean_exponent = extract_peaks_multichannel(data, ch_names, fs)
    if len(peaks_df) == 0:
        return None

    row = {
        'subject': sub_id,
        'session': int(session),
        'task': task,
        'acq': acq,
        'n_channels': len(ch_names),
        'n_peaks_total': len(peaks_df),
        'aperiodic_exponent': mean_exponent,
    }

    if participants_df is not None:
        prow = participants_df[participants_df['participant_id'] == sub_id]
        if len(prow) > 0:
            row['age'] = prow.iloc[0].get('age', np.nan)
            row['sex'] = prow.iloc[0].get('sex', 'unknown')

    n_bands = 0
    for band_name, (lo, hi) in BANDS.items():
        bp = peaks_df[(peaks_df['freq'] >= lo) & (peaks_df['freq'] < hi)]
        if len(bp) > 0:
            idx = bp['power'].idxmax()
            freq = bp.loc[idx, 'freq']
            power = bp.loc[idx, 'power']
            u = lattice_coord(freq)
            d = min_lattice_dist(u)
            nearest = nearest_position_name(u)
            row[f'{band_name}_freq'] = freq
            row[f'{band_name}_power'] = power
            row[f'{band_name}_u'] = u
            row[f'{band_name}_d'] = d
            row[f'{band_name}_nearest'] = nearest
            n_bands += 1
        else:
            row[f'{band_name}_freq'] = np.nan
            row[f'{band_name}_power'] = np.nan
            row[f'{band_name}_u'] = np.nan
            row[f'{band_name}_d'] = np.nan
            row[f'{band_name}_nearest'] = 'none'

    row['n_bands'] = n_bands
    ds = [row[f'{b}_d'] for b in BANDS if not np.isnan(row.get(f'{b}_d', np.nan))]
    row['mean_d'] = np.mean(ds) if ds else np.nan
    return row


# ── Longitudinal analysis ─────────────────────────────────────────────────

def icc_2_1(x, y):
    """ICC(2,1) — two-way random, single measures, absolute agreement."""
    n = len(x)
    if n < 3:
        return np.nan, np.nan
    grand = (x.mean() + y.mean()) / 2
    ms_r = 2 * np.sum(((x + y) / 2 - grand) ** 2) / (n - 1)
    ms_e = np.sum((x - y) ** 2) / n
    ms_c = 2 * n * ((x.mean() - grand) ** 2 + (y.mean() - grand) ** 2)
    # ICC(2,1) = (MSR - MSE) / (MSR + MSE + 2*(MSC - MSE)/n)
    denom = ms_r + ms_e + 2 * (ms_c - ms_e) / n
    if denom <= 0:
        return np.nan, np.nan
    icc = (ms_r - ms_e) / denom
    # F-test for significance
    f_stat = ms_r / ms_e if ms_e > 0 else np.inf
    from scipy.stats import f as f_dist
    p = 1 - f_dist.cdf(f_stat, n - 1, n - 1)
    return icc, p


def run_longitudinal_analysis(ses1_df, ses2_df, condition_label):
    """Compare ses-1 vs ses-2 for matched subjects."""

    print(f"\n{'='*70}")
    print(f"  LONGITUDINAL: {condition_label}")
    print(f"{'='*70}")

    # Merge on subject
    merged = pd.merge(ses1_df, ses2_df, on='subject', suffixes=('_s1', '_s2'))
    print(f"\n  Matched subjects: {len(merged)}")

    if len(merged) < 10:
        print("  Too few matched subjects!")
        return {}

    results = {'condition': condition_label, 'n_matched': len(merged)}

    # ── 1. Overall mean_d stability ──
    print(f"\n  --- 1. Overall mean_d Stability ---")
    s1_d = merged['mean_d_s1'].dropna().values
    s2_d = merged['mean_d_s2'].dropna().values
    valid = np.isfinite(s1_d) & np.isfinite(s2_d)
    s1_d, s2_d = s1_d[valid], s2_d[valid]

    r_pearson, p_pearson = stats.pearsonr(s1_d, s2_d)
    r_spearman, p_spearman = stats.spearmanr(s1_d, s2_d)
    icc_val, icc_p = icc_2_1(s1_d, s2_d)

    # Paired test: did alignment change?
    t_paired, p_paired = stats.ttest_rel(s1_d, s2_d)
    delta_d = s2_d - s1_d

    print(f"    Ses-1 mean_d: {s1_d.mean():.4f} ± {s1_d.std():.4f}")
    print(f"    Ses-2 mean_d: {s2_d.mean():.4f} ± {s2_d.std():.4f}")
    print(f"    Pearson r = {r_pearson:.3f}, p = {p_pearson:.2e}")
    print(f"    Spearman ρ = {r_spearman:.3f}, p = {p_spearman:.2e}")
    print(f"    ICC(2,1) = {icc_val:.3f}, p = {icc_p:.2e}")
    print(f"    Paired t = {t_paired:.2f}, p = {p_paired:.4f} (Δ = {delta_d.mean():+.4f})")
    print(f"    Mean |Δ| = {np.abs(delta_d).mean():.4f}")

    results.update({
        'mean_d_s1': s1_d.mean(), 'mean_d_s2': s2_d.mean(),
        'r_pearson': r_pearson, 'p_pearson': p_pearson,
        'r_spearman': r_spearman, 'icc': icc_val, 'icc_p': icc_p,
        'paired_t': t_paired, 'paired_p': p_paired,
        'mean_delta': delta_d.mean(), 'mean_abs_delta': np.abs(delta_d).mean(),
    })

    # ── 2. Per-band position stability ──
    print(f"\n  --- 2. Per-Band Position Stability ---")
    for band_name in BANDS:
        col_near_s1 = f'{band_name}_nearest_s1'
        col_near_s2 = f'{band_name}_nearest_s2'
        col_freq_s1 = f'{band_name}_freq_s1'
        col_freq_s2 = f'{band_name}_freq_s2'
        col_d_s1 = f'{band_name}_d_s1'
        col_d_s2 = f'{band_name}_d_s2'

        if col_near_s1 not in merged.columns:
            continue

        valid_mask = (merged[col_near_s1] != 'none') & (merged[col_near_s2] != 'none')
        band_m = merged[valid_mask]
        n_band = len(band_m)

        if n_band < 10:
            print(f"    {band_name}: too few subjects ({n_band})")
            continue

        # Position agreement
        same_pos = (band_m[col_near_s1] == band_m[col_near_s2]).sum()
        pct_same = 100 * same_pos / n_band

        # Frequency correlation
        f1 = band_m[col_freq_s1].values
        f2 = band_m[col_freq_s2].values
        r_freq, p_freq = stats.pearsonr(f1, f2)
        freq_delta = f2 - f1

        # Distance correlation
        d1 = band_m[col_d_s1].values
        d2 = band_m[col_d_s2].values
        r_d, p_d = stats.pearsonr(d1, d2)

        # Band ICC
        band_icc, _ = icc_2_1(d1, d2)

        print(f"    {band_name}: {same_pos}/{n_band} same position ({pct_same:.0f}%), "
              f"freq r={r_freq:.3f}, d ICC={band_icc:.3f}, "
              f"Δfreq={freq_delta.mean():+.2f} Hz")

        results[f'{band_name}_pct_same_pos'] = pct_same
        results[f'{band_name}_freq_r'] = r_freq
        results[f'{band_name}_d_icc'] = band_icc
        results[f'{band_name}_mean_freq_shift'] = freq_delta.mean()

    # ── 3. Position transition matrix ──
    print(f"\n  --- 3. Position Transitions (theta, most variable in LEMON) ---")
    if 'theta_nearest_s1' in merged.columns:
        valid_mask = (merged['theta_nearest_s1'] != 'none') & (merged['theta_nearest_s2'] != 'none')
        theta_m = merged[valid_mask]
        if len(theta_m) >= 10:
            trans = pd.crosstab(theta_m['theta_nearest_s1'],
                                theta_m['theta_nearest_s2'],
                                margins=True, margins_name='Total')
            print(trans.to_string())

    # ── 4. Ses-2 standalone significance ──
    print(f"\n  --- 4. Ses-2 Standalone Alignment ---")
    ses2_valid = ses2_df[ses2_df['n_bands'] == 4].copy()
    if len(ses2_valid) < 10:
        ses2_valid = ses2_df[ses2_df['n_bands'] >= 3].copy()

    if len(ses2_valid) >= 10:
        mean_ds_s2 = ses2_valid['mean_d'].values
        null_expected = np.mean([min_lattice_dist(np.random.uniform(0, 1))
                                 for _ in range(100_000)])
        t_s2, p_s2 = stats.ttest_1samp(mean_ds_s2, null_expected)
        cohen_d_s2 = (null_expected - mean_ds_s2.mean()) / mean_ds_s2.std()
        print(f"    N = {len(ses2_valid)}")
        print(f"    mean_d = {mean_ds_s2.mean():.4f} ± {mean_ds_s2.std():.4f}")
        print(f"    vs null {null_expected:.4f}: t={t_s2:.2f}, p={p_s2:.2e}, d={cohen_d_s2:.3f}")
        results['ses2_mean_d'] = mean_ds_s2.mean()
        results['ses2_cohen_d'] = cohen_d_s2
        results['ses2_p'] = p_s2

    # ── 5. Cross-base comparison for ses-2 ──
    print(f"\n  --- 5. Ses-2 Cross-Base Ranking ---")
    if len(ses2_valid) >= 10:
        base_means = {}
        for base_name, base_val in BASES.items():
            positions = positions_for_base(base_val)
            seg_ds = []
            for _, row in ses2_valid.iterrows():
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
            base_means[base_name] = np.mean(seg_ds) if seg_ds else np.nan

        ranking = sorted(base_means.items(), key=lambda x: x[1])
        for rank, (bname, md) in enumerate(ranking, 1):
            marker = ' ←' if bname == 'phi' else ''
            print(f"      {rank}. {bname:6s}: mean_d={md:.4f}{marker}")
        phi_rank = next(i+1 for i, (name, _) in enumerate(ranking) if name == 'phi')
        results['ses2_phi_rank'] = phi_rank

    # ── 6. Rank stability (does phi rank change?) ──
    print(f"\n  --- 6. Cross-Base Rank Stability (ses-1 vs ses-2) ---")
    # Compute per-subject, per-base mean_d for both sessions
    if len(merged) >= 10:
        for base_name, base_val in [('phi', PHI), ('3/2', 1.5), ('e', np.e)]:
            positions = positions_for_base(base_val)
            s1_vals, s2_vals = [], []
            for _, row in merged.iterrows():
                for sess_suffix, vals_list in [('_s1', s1_vals), ('_s2', s2_vals)]:
                    band_ds = []
                    for band_name in BANDS:
                        freq = row[f'{band_name}_freq{sess_suffix}']
                        if np.isfinite(freq):
                            u = lattice_coord(freq, f0=F0, base=base_val)
                            d = min_lattice_dist(u, positions)
                            band_ds.append(d)
                    if band_ds:
                        vals_list.append(np.mean(band_ds))
                    else:
                        vals_list.append(np.nan)
            s1_arr = np.array(s1_vals)
            s2_arr = np.array(s2_vals)
            valid_b = np.isfinite(s1_arr) & np.isfinite(s2_arr)
            if valid_b.sum() >= 10:
                r_b, _ = stats.pearsonr(s1_arr[valid_b], s2_arr[valid_b])
                icc_b, _ = icc_2_1(s1_arr[valid_b], s2_arr[valid_b])
                print(f"    {base_name:6s}: r={r_b:.3f}, ICC={icc_b:.3f}")

    # ── 7. Age-at-retest moderation ──
    print(f"\n  --- 7. Age Moderation of Stability ---")
    if 'age_s1' in merged.columns:
        age_valid = merged[merged['age_s1'].notna() & np.isfinite(merged['mean_d_s1']) &
                           np.isfinite(merged['mean_d_s2'])].copy()
        if len(age_valid) >= 20:
            age_valid['abs_delta_d'] = np.abs(age_valid['mean_d_s2'] - age_valid['mean_d_s1'])
            r_age_stab, p_age_stab = stats.pearsonr(age_valid['age_s1'],
                                                      age_valid['abs_delta_d'])
            print(f"    |Δmean_d| ~ age: r={r_age_stab:.3f}, p={p_age_stab:.4f} (N={len(age_valid)})")

            # Young vs old stability
            young = age_valid[age_valid['age_s1'] < 35]
            old = age_valid[age_valid['age_s1'] >= 55]
            if len(young) >= 10 and len(old) >= 10:
                u_stat, p_grp = stats.mannwhitneyu(young['abs_delta_d'],
                                                    old['abs_delta_d'])
                print(f"    Young (<35, N={len(young)}) |Δ|={young['abs_delta_d'].mean():.4f} vs "
                      f"Old (≥55, N={len(old)}) |Δ|={old['abs_delta_d'].mean():.4f}, p={p_grp:.4f}")
            results['r_age_stability'] = r_age_stab
            results['p_age_stability'] = p_age_stab

    # ── 8. Fatigue invariance (pre vs post within each session) ──
    # This is computed across conditions, so handled in main()

    return results


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Find data directory
    data_dir = None
    for d in [DATA_DIR_DL, DATA_DIR_ORIG]:
        if os.path.isdir(d):
            test_path = os.path.join(d, 'sub-001', 'ses-2', 'eeg')
            if os.path.isdir(test_path):
                edfs = [f for f in os.listdir(test_path) if f.endswith('.edf') and not f.startswith('._')]
                for edf in edfs:
                    path = os.path.join(test_path, edf)
                    if os.path.isfile(path) and not (os.path.islink(path) and not os.path.exists(path)):
                        data_dir = d
                        break
            if data_dir:
                break

    if data_dir is None:
        print("ERROR: No readable ses-2 EDF files found!")
        sys.exit(1)

    print(f"Data directory: {data_dir}")

    # Load participants
    participants_df = None
    for p in [os.path.join(data_dir, 'participants.tsv'),
              os.path.join(DATA_DIR_ORIG, 'participants.tsv')]:
        if os.path.isfile(p):
            participants_df = pd.read_csv(p, sep='\t')
            print(f"Participants: {len(participants_df)} subjects loaded")
            break

    # Find ses-2 subjects
    ses2_subjects = sorted([
        d for d in os.listdir(data_dir)
        if d.startswith('sub-') and
        os.path.isdir(os.path.join(data_dir, d, 'ses-2'))
    ])
    print(f"Subjects with ses-2: {len(ses2_subjects)}")

    # Check for existing ses-1 results
    ses1_available = {}
    for task, acq in CONDITIONS:
        csv_path = os.path.join(SES1_DIR, f'dortmund_dominant_peaks_{task}_{acq}.csv')
        if os.path.isfile(csv_path):
            ses1_available[(task, acq)] = pd.read_csv(csv_path)
            print(f"  Ses-1 {task}_{acq}: {len(ses1_available[(task, acq)])} subjects")
        else:
            print(f"  Ses-1 {task}_{acq}: NOT FOUND at {csv_path}")

    # ── Process ses-2 ──
    all_results = {}
    all_longitudinal = {}

    for task, acq in CONDITIONS:
        condition_label = f"{task}_{acq}"
        print(f"\n\n{'#'*70}")
        print(f"  Processing ses-2: {condition_label}")
        print(f"{'#'*70}")

        # Check which subjects have this condition
        available = []
        for sub in ses2_subjects:
            edf = find_edf(sub, '2', task, acq, data_dir)
            if edf is not None:
                available.append(sub)
        print(f"  Available: {len(available)}/{len(ses2_subjects)}")

        if len(available) < 10:
            print(f"  Skipping — too few subjects")
            continue

        # Check for cached results
        out_csv = os.path.join(OUT_DIR, f'dortmund_ses2_dominant_peaks_{task}_{acq}.csv')
        if os.path.isfile(out_csv):
            print(f"  Loading cached results: {out_csv}")
            ses2_df = pd.read_csv(out_csv)
            print(f"  Loaded {len(ses2_df)} subjects from cache")
        else:
            # Process subjects
            results = []
            t0 = time.time()
            for i, sub in enumerate(available):
                if (i + 1) % 25 == 0 or i == 0:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    eta = (len(available) - i - 1) / rate if rate > 0 else 0
                    print(f"  [{i+1}/{len(available)}] {sub} "
                          f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

                row = analyze_subject(sub, '2', task, acq, data_dir, participants_df)
                if row is not None:
                    results.append(row)

            elapsed = time.time() - t0
            print(f"\n  Processed {len(results)}/{len(available)} subjects in {elapsed:.0f}s")

            if len(results) < 10:
                print(f"  Too few successful subjects")
                continue

            ses2_df = pd.DataFrame(results)
            ses2_df.to_csv(out_csv, index=False)
            print(f"  Saved: {out_csv}")

        all_results[condition_label] = ses2_df

        # ── Run longitudinal comparison ──
        if (task, acq) in ses1_available:
            ses1_df = ses1_available[(task, acq)]
            long_result = run_longitudinal_analysis(ses1_df, ses2_df, condition_label)
            all_longitudinal[condition_label] = long_result

    # ── Cross-condition fatigue invariance ──
    print(f"\n\n{'#'*70}")
    print(f"  FATIGUE INVARIANCE (pre vs post, both sessions)")
    print(f"{'#'*70}")

    for ses_num, ses_label, result_dict in [
        ('1', 'ses-1', ses1_available),
        ('2', 'ses-2', all_results),
    ]:
        for task_name in ['EyesClosed', 'EyesOpen']:
            pre_key = f'{task_name}_pre' if ses_label == 'ses-2' else (task_name, 'pre')
            post_key = f'{task_name}_post' if ses_label == 'ses-2' else (task_name, 'post')

            pre_df = result_dict.get(pre_key)
            post_df = result_dict.get(post_key)

            if pre_df is None or post_df is None:
                continue

            merged_fp = pd.merge(pre_df[['subject', 'mean_d']],
                                 post_df[['subject', 'mean_d']],
                                 on='subject', suffixes=('_pre', '_post'))
            valid_fp = merged_fp.dropna(subset=['mean_d_pre', 'mean_d_post'])

            if len(valid_fp) < 10:
                continue

            r_fp, p_fp = stats.pearsonr(valid_fp['mean_d_pre'], valid_fp['mean_d_post'])
            t_fp, pt_fp = stats.ttest_rel(valid_fp['mean_d_pre'], valid_fp['mean_d_post'])
            icc_fp, _ = icc_2_1(valid_fp['mean_d_pre'].values, valid_fp['mean_d_post'].values)
            delta_fp = valid_fp['mean_d_post'] - valid_fp['mean_d_pre']

            print(f"\n  {ses_label} {task_name}: N={len(valid_fp)}")
            print(f"    Pre  mean_d: {valid_fp['mean_d_pre'].mean():.4f}")
            print(f"    Post mean_d: {valid_fp['mean_d_post'].mean():.4f}")
            print(f"    r={r_fp:.3f}, ICC={icc_fp:.3f}, paired t={t_fp:.2f}, p={pt_fp:.4f}")
            print(f"    Δ = {delta_fp.mean():+.4f} (fatigue effect)")

    # ── Grand summary ──
    print(f"\n\n{'='*70}")
    print(f"  GRAND SUMMARY")
    print(f"{'='*70}")

    summary_rows = []
    for cond, lr in all_longitudinal.items():
        if lr:
            summary_rows.append(lr)
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_csv = os.path.join(OUT_DIR, 'longitudinal_summary.csv')
        summary_df.to_csv(summary_csv, index=False)
        print(f"\n  Summary saved: {summary_csv}")

        for _, row in summary_df.iterrows():
            print(f"\n  {row.get('condition', '?')}:")
            print(f"    ICC = {row.get('icc', np.nan):.3f}")
            print(f"    Ses-1 d = {row.get('mean_d_s1', np.nan):.4f}, "
                  f"Ses-2 d = {row.get('mean_d_s2', np.nan):.4f}")
            print(f"    Δ = {row.get('mean_delta', np.nan):+.4f}, "
                  f"paired p = {row.get('paired_p', np.nan):.4f}")
            ses2_d = row.get('ses2_cohen_d', np.nan)
            print(f"    Ses-2 standalone: Cohen's d = {ses2_d:.3f}, "
                  f"phi rank = {row.get('ses2_phi_rank', '?')}")

    print(f"\n\nDone! All results in {OUT_DIR}")


if __name__ == '__main__':
    main()

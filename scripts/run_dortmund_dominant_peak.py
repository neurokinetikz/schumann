#!/usr/bin/env python3
"""
Dominant-Peak Lattice Analysis — Dortmund Vital Study
=====================================================

Replication of LEMON phi-lattice dominant-peak analysis on the Dortmund
resting-state EEG dataset (Wascher et al., OpenNeuro ds005385).

608 subjects, 64 channels, ages 20-70, EO+EC pre/post cognitive battery.
208 subjects with 5-year longitudinal follow-up (session 2).

Pipeline per subject:
  1. Load EDF via MNE, downsample 1000→250 Hz, notch 50 Hz
  2. Welch PSD per channel (4s windows, 50% overlap)
  3. FOOOF per channel [1, 45] Hz, max_n_peaks=20
  4. Pool peaks across 64 channels
  5. Extract dominant peak per band (delta, theta, alpha, gamma)
  6. Compute phi-lattice coordinates and distances
  7. Population + per-subject significance tests
  8. Cross-base comparison (9 bases)

Reference: Wascher et al. doi:10.18112/openneuro.ds005385.v1.0.3
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

# ── Constants ──────────────────────────────────────────────────────────────

PHI = 1.6180339887
F0 = 7.83  # Pre-specified Schumann fundamental

# Try both possible data locations
DATA_DIR_DL = '/Volumes/T9/dortmund_data_dl'  # openneuro-py download
DATA_DIR_ORIG = '/Volumes/T9/dortmund_data'             # original datalad clone
OUT_DIR = '/Volumes/T9/dortmund_data/lattice_results'

TARGET_FS = 250  # Downsample to this rate
NOTCH_FREQ = 50  # European power line

# Conditions to analyze (in priority order)
CONDITIONS = [
    ('EyesClosed', 'pre'),   # Primary replication target
    ('EyesOpen', 'pre'),     # Secondary
    ('EyesClosed', 'post'),  # Post-cognitive fatigue
    ('EyesOpen', 'post'),    # Post-cognitive fatigue
]

# 4-band definitions (matching LEMON/EEGMMIDB/Bonn)
BANDS = {
    'delta': (1.0, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'gamma': (30.0, 45.0),
}

# Phi positions within each octave (8 positions)
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


# ── Data loading ──────────────────────────────────────────────────────────

def find_edf(sub_id, session, task, acq, data_dir):
    """Find EDF file for a given subject/session/task/acq combination."""
    # Try multiple path patterns
    patterns = [
        # openneuro-py download structure
        os.path.join(data_dir, sub_id, f'ses-{session}', 'eeg',
                     f'{sub_id}_ses-{session}_task-{task}_acq-{acq}_eeg.edf'),
        # direct BIDS structure
        os.path.join(data_dir, sub_id, f'ses-{session}', 'eeg',
                     f'{sub_id}_ses-{session}_task-{task}_acq-{acq}_eeg.edf'),
    ]
    for p in patterns:
        if os.path.isfile(p):
            return p
        # Check if symlink exists and is valid
        if os.path.islink(p) and os.path.exists(p):
            return p
    return None


def load_subject_edf(edf_path, target_fs=TARGET_FS):
    """
    Load EDF, pick EEG channels, downsample, notch filter.
    Returns: (data_array [n_channels x n_samples], channel_names, fs)
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # Pick only EEG channels
    raw.pick_types(eeg=True, exclude='bads')

    # Notch filter at 50 Hz + harmonics
    raw.notch_filter(freqs=[50, 100], verbose=False)

    # Downsample
    if raw.info['sfreq'] > target_fs:
        raw.resample(target_fs, verbose=False)

    return raw.get_data(), raw.ch_names, raw.info['sfreq']


def extract_peaks_multichannel(data, ch_names, fs):
    """
    Run FOOOF on each channel's PSD, pool all peaks.
    Returns DataFrame of peaks with columns: channel, freq, power, bandwidth
    """
    all_peaks = []
    aperiodic_exponents = []

    for i, ch in enumerate(ch_names):
        sig = data[i, :]

        # Welch PSD (4-second windows, 50% overlap)
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
                        all_peaks.append({
                            'channel': ch, 'freq': cf,
                            'power': pw, 'bandwidth': bw
                        })
                elif peaks_params.ndim == 1 and len(peaks_params) == 3:
                    cf, pw, bw = peaks_params
                    all_peaks.append({
                        'channel': ch, 'freq': cf,
                        'power': pw, 'bandwidth': bw
                    })
        except Exception:
            continue

    mean_exponent = np.mean(aperiodic_exponents) if aperiodic_exponents else np.nan
    return pd.DataFrame(all_peaks), mean_exponent


# ── Per-subject analysis ──────────────────────────────────────────────────

def analyze_subject(sub_id, session, task, acq, data_dir, participants_df=None):
    """
    Full pipeline for one subject × one condition.
    Returns dict with dominant peaks and lattice metrics, or None on failure.
    """
    edf_path = find_edf(sub_id, session, task, acq, data_dir)
    if edf_path is None:
        return None

    try:
        data, ch_names, fs = load_subject_edf(edf_path)
    except Exception as e:
        print(f"    ERROR loading {sub_id}: {e}")
        return None

    peaks_df, mean_exponent = extract_peaks_multichannel(data, ch_names, fs)

    if len(peaks_df) == 0:
        return None

    # Extract dominant peak per band (pooled across all channels)
    row = {
        'subject': sub_id,
        'session': session,
        'task': task,
        'acq': acq,
        'n_channels': len(ch_names),
        'n_peaks_total': len(peaks_df),
        'aperiodic_exponent': mean_exponent,
    }

    # Add demographics if available
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


# ── Statistical analysis ──────────────────────────────────────────────────

def run_statistics(dom_df, condition_label):
    """Run full statistical suite on dominant-peak results."""

    print(f"\n{'='*70}")
    print(f"  {condition_label}")
    print(f"  N={len(dom_df)} subjects")
    print(f"{'='*70}")

    # Filter to subjects with all 4 bands
    valid = dom_df[dom_df['n_bands'] == 4].copy()
    n_valid = len(valid)
    n_total = len(dom_df)

    print(f"\n  Complete 4-band data: {n_valid}/{n_total} subjects")
    for band_name in BANDS:
        n_with = dom_df[f'{band_name}_freq'].notna().sum()
        print(f"    {band_name}: {n_with}/{n_total} subjects with peak")

    if n_valid < 30:
        valid_3 = dom_df[dom_df['n_bands'] >= 3].copy()
        print(f"  (Fallback to ≥3 bands: {len(valid_3)}/{n_total})")
        analysis_df = valid_3
    else:
        analysis_df = valid

    if len(analysis_df) < 10:
        print("  WARNING: Too few subjects for analysis!")
        return {}

    # ── Population-level alignment ──
    print(f"\n  --- Population-Level Alignment (f0={F0} Hz) ---")
    pop_freqs = {}
    for band_name in BANDS:
        freqs = analysis_df[f'{band_name}_freq'].dropna()
        if len(freqs) > 0:
            med = freqs.median()
            pop_freqs[band_name] = med
            u = lattice_coord(med)
            d = min_lattice_dist(u)
            near = nearest_position_name(u)
            print(f"    {band_name}: median={med:.2f} Hz, u={u:.3f}, d={d:.3f} → {near}")

    pop_ds = [min_lattice_dist(lattice_coord(f)) for f in pop_freqs.values()]
    pop_mean_d = np.mean(pop_ds)
    print(f"    Population mean_d = {pop_mean_d:.4f}")

    # Null: 100K permutations
    n_perm = 100_000
    null_pop = np.zeros(n_perm)
    for i in range(n_perm):
        ds = []
        for band_name, (lo, hi) in BANDS.items():
            if band_name in pop_freqs:
                ds.append(min_lattice_dist(lattice_coord(np.random.uniform(lo, hi))))
        null_pop[i] = np.mean(ds)
    p_pop = (null_pop <= pop_mean_d).mean()
    print(f"    Uniform null: p = {p_pop:.4f}")

    # ── Per-subject alignment ──
    print(f"\n  --- Per-Subject Alignment ---")
    mean_ds = analysis_df['mean_d'].values
    obs_mean = mean_ds.mean()
    obs_sd = mean_ds.std()

    null_expected = np.mean([min_lattice_dist(np.random.uniform(0, 1))
                            for _ in range(100_000)])

    t_stat, p_ttest = stats.ttest_1samp(mean_ds, null_expected)
    cohen_d = (null_expected - obs_mean) / obs_sd if obs_sd > 0 else 0.0

    try:
        w_stat, p_wilcox = stats.wilcoxon(mean_ds - null_expected, alternative='less')
    except Exception:
        p_wilcox = np.nan

    print(f"    Observed mean_d: {obs_mean:.4f} ± {obs_sd:.4f}")
    print(f"    Expected (null): {null_expected:.4f}")
    print(f"    t = {t_stat:.2f}, p = {p_ttest:.2e}")
    print(f"    Wilcoxon p = {p_wilcox:.2e}")
    print(f"    Cohen's d = {cohen_d:.3f}")

    # Count individually significant
    n_sig = 0
    null_ref = np.array([np.mean([min_lattice_dist(np.random.uniform(0, 1))
                                   for _ in range(4)])
                          for _ in range(10_000)])
    p5 = np.percentile(null_ref, 5)
    for md in mean_ds:
        if md <= p5:
            n_sig += 1
    print(f"    Individually significant (p<0.05): {n_sig}/{len(analysis_df)} "
          f"({100*n_sig/len(analysis_df):.1f}% vs 5% expected)")

    # ── Per-band breakdown ──
    print(f"\n  --- Per-Band Breakdown ---")
    for band_name, (lo, hi) in BANDS.items():
        ds = analysis_df[f'{band_name}_d'].dropna().values
        if len(ds) < 5:
            continue
        band_null = np.array([min_lattice_dist(lattice_coord(np.random.uniform(lo, hi)))
                              for _ in range(10_000)])
        ks_stat, ks_p = stats.ks_2samp(ds, band_null, alternative='less')
        band_d = (band_null.mean() - ds.mean()) / ds.std() if ds.std() > 0 else 0
        nearests = analysis_df[f'{band_name}_nearest'].value_counts()
        top_pos = nearests.index[0] if len(nearests) > 0 else 'none'
        top_pct = 100 * nearests.iloc[0] / nearests.sum() if len(nearests) > 0 else 0
        print(f"    {band_name}: mean_d={ds.mean():.4f}, null={band_null.mean():.4f}, "
              f"KS p={ks_p:.3e}, d={band_d:.2f}  | top: {top_pos} ({top_pct:.0f}%)")

    # ── Cross-base comparison ──
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
            'mean_d': seg_ds.mean(), 'median_d': np.median(seg_ds),
            'sd_d': seg_ds.std(), 'n_positions': len(positions),
            'values': seg_ds,
        }

    ranking = sorted(base_results.items(), key=lambda x: x[1]['mean_d'])
    for rank, (bname, br) in enumerate(ranking, 1):
        marker = ' ←' if bname == 'phi' else ''
        print(f"    {rank}. {bname:6s}: mean_d={br['mean_d']:.4f} "
              f"(±{br['sd_d']:.4f}, {br['n_positions']} pos){marker}")

    # Phi vs others
    phi_ds = base_results['phi']['values']
    print(f"\n    Phi vs competitors (paired t-test):")
    for bname, br in ranking:
        if bname == 'phi':
            continue
        other_ds = br['values']
        if len(phi_ds) == len(other_ds):
            t, p = stats.ttest_rel(phi_ds, other_ds, alternative='less')
            sig = '*' if p < 0.05 else ' '
            print(f"      phi vs {bname:6s}: Δd={other_ds.mean()-phi_ds.mean():+.4f}, p={p:.3e} {sig}")

    # ── Noble position contribution ──
    shared_pos = {'boundary': 0.0, 'attractor': 0.5}
    shared_ds = []
    for _, row in analysis_df.iterrows():
        band_ds = []
        for band_name in BANDS:
            freq = row[f'{band_name}_freq']
            if np.isnan(freq):
                continue
            u = lattice_coord(freq)
            d = min_lattice_dist(u, shared_pos)
            band_ds.append(d)
        if band_ds:
            shared_ds.append(np.mean(band_ds))
    shared_ds = np.array(shared_ds)

    full_mean = base_results['phi']['mean_d']
    shared_mean = shared_ds.mean()
    noble_contrib = (1 - full_mean / shared_mean) * 100 if shared_mean > 0 else 0
    print(f"\n  --- Noble Position Contribution ---")
    print(f"    Full phi (8 pos): mean_d = {full_mean:.4f}")
    print(f"    Shared only (2 pos): mean_d = {shared_mean:.4f}")
    print(f"    Noble contribution: {noble_contrib:.1f}%")

    # ── Age analysis (if available) ──
    if 'age' in analysis_df.columns and analysis_df['age'].notna().sum() > 20:
        print(f"\n  --- Age Analysis ---")
        age_valid = analysis_df[analysis_df['age'].notna()]
        r, p_age = stats.pearsonr(age_valid['age'], age_valid['mean_d'])
        print(f"    mean_d ~ age: r={r:.3f}, p={p_age:.4f} (N={len(age_valid)})")

        # Age group comparison
        young = age_valid[age_valid['age'] < 35]['mean_d'].values
        old = age_valid[age_valid['age'] >= 55]['mean_d'].values
        if len(young) >= 10 and len(old) >= 10:
            u_stat, p_group = stats.mannwhitneyu(young, old, alternative='two-sided')
            d_group = (young.mean() - old.mean()) / np.sqrt((young.std()**2 + old.std()**2)/2)
            print(f"    Young (<35, N={len(young)}) vs Old (≥55, N={len(old)}): "
                  f"d={d_group:.3f}, p={p_group:.4f}")

    phi_rank = next(i+1 for i, (name, _) in enumerate(ranking) if name == 'phi')

    return {
        'condition': condition_label,
        'n_total': n_total,
        'n_valid': len(analysis_df),
        'pop_mean_d': pop_mean_d,
        'p_pop': p_pop,
        'obs_mean_d': obs_mean,
        'null_expected_d': null_expected,
        'cohen_d': cohen_d,
        'p_ttest': p_ttest,
        'p_wilcox': p_wilcox,
        'phi_rank': phi_rank,
        'noble_contrib': noble_contrib,
        'base_results': base_results,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Determine data directory
    data_dir = None
    for d in [DATA_DIR_DL, DATA_DIR_ORIG]:
        if os.path.isdir(d):
            # Check if any EDF files are actually readable (not broken symlinks)
            test_sub = os.path.join(d, 'sub-001', 'ses-1', 'eeg')
            if os.path.isdir(test_sub):
                edfs = [f for f in os.listdir(test_sub) if f.endswith('.edf')]
                for edf in edfs:
                    path = os.path.join(test_sub, edf)
                    if os.path.isfile(path) and not (os.path.islink(path) and not os.path.exists(path)):
                        data_dir = d
                        break
            if data_dir:
                break

    if data_dir is None:
        print("ERROR: No readable EDF files found!")
        print(f"  Checked: {DATA_DIR_DL}")
        print(f"  Checked: {DATA_DIR_ORIG}")
        print("  Please download data first: openneuro-py download --dataset ds005385 ...")
        sys.exit(1)

    print(f"Data directory: {data_dir}")

    # Load participants
    participants_path = None
    for p in [os.path.join(data_dir, 'participants.tsv'),
              os.path.join(DATA_DIR_ORIG, 'participants.tsv')]:
        if os.path.isfile(p):
            participants_path = p
            break

    participants_df = None
    if participants_path:
        participants_df = pd.read_csv(participants_path, sep='\t')
        print(f"Participants: {len(participants_df)} subjects loaded")

    # Find available subjects
    subjects = sorted([d for d in os.listdir(data_dir)
                       if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])
    print(f"Subject folders: {len(subjects)}")

    # Process each condition
    for task, acq in CONDITIONS:
        condition_label = f"{task}_{acq} (ses-1)"
        print(f"\n\n{'#'*70}")
        print(f"  Processing: {condition_label}")
        print(f"{'#'*70}")

        # Check how many subjects have this condition available
        available = []
        for sub in subjects:
            edf = find_edf(sub, '1', task, acq, data_dir)
            if edf is not None:
                available.append(sub)

        print(f"  Available subjects: {len(available)}/{len(subjects)}")

        if len(available) < 10:
            print(f"  Skipping — too few subjects with data")
            continue

        # Process subjects
        results = []
        t0 = time.time()
        for i, sub in enumerate(available):
            if (i + 1) % 50 == 0 or i == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(available) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(available)}] {sub} "
                      f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

            row = analyze_subject(sub, '1', task, acq, data_dir, participants_df)
            if row is not None:
                results.append(row)

        elapsed = time.time() - t0
        print(f"\n  Processed {len(results)}/{len(available)} subjects in {elapsed:.0f}s")

        if len(results) < 10:
            print(f"  Too few successful subjects for analysis")
            continue

        dom_df = pd.DataFrame(results)

        # Save per-subject results
        out_csv = os.path.join(OUT_DIR, f'dortmund_dominant_peaks_{task}_{acq}.csv')
        dom_df.to_csv(out_csv, index=False)
        print(f"  Saved: {out_csv}")

        # Run statistics
        stats_result = run_statistics(dom_df, condition_label)

        # Save summary
        if stats_result:
            summary_csv = os.path.join(OUT_DIR, f'dortmund_summary_{task}_{acq}.csv')
            summary_row = {k: v for k, v in stats_result.items() if k != 'base_results'}
            pd.DataFrame([summary_row]).to_csv(summary_csv, index=False)

    print(f"\n\nDone! Results saved to {OUT_DIR}")


if __name__ == '__main__':
    main()

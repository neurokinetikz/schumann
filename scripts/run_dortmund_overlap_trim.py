#!/usr/bin/env python3
"""
Overlap-Trim Dominant-Peak Lattice Analysis — Dortmund Vital Study
===================================================================

Adapts the LEMON overlap-trim per-phi-octave FOOOF extraction for the
Dortmund resting-state EEG dataset, then runs the dominant-peak lattice
analysis on the resulting peaks.

The overlap-trim method solves the 1/f knee contamination problem:
  Standard [1,45] FOOOF: single-exponent aperiodic model misspecifies
  the knee-containing spectrum under EC, producing spurious delta peaks.

  Overlap-trim: fits FOOOF on narrow phi-octave windows (+ half-octave
  padding), so the local aperiodic model captures the true background
  in each frequency range. Padding-zone peaks are discarded.

Critical prediction: EC delta d should flip from -0.31 (anti-aligned)
to near-zero or positive, EC 4-band average should become significant,
and EC amplification should appear.

Reference: Wascher et al. doi:10.18112/openneuro.ds005385.v1.0.3
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.signal import welch
from specparam import SpectralModel
import mne
import os
import sys
import time
import gc
import warnings
import traceback
warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

# ── Constants ──────────────────────────────────────────────────────────────

PHI = (1 + np.sqrt(5)) / 2
F0 = 7.83  # Pre-specified Schumann fundamental

DATA_DIR = '/Volumes/T9/dortmund_data_dl'
OUT_DIR = '/Volumes/T9/dortmund_data/lattice_results_ot'

TARGET_FS = 250  # Downsample to this rate
NOTCH_FREQ = 50  # European power line
WELCH_NPERSEG = int(4.0 * TARGET_FS)  # 4-second windows → 0.25 Hz resolution

# Overlap-trim parameters (matching LEMON)
FREQ_CEIL = 45.0   # Upper limit (Dortmund downsampled to 250 Hz, Nyquist=125)
FREQ_FLOOR = 1.0
PAD_OCTAVES = 0.5   # Half phi-octave padding each side
R2_MIN = 0.70        # Minimum goodness of fit

FOOOF_BASE_PARAMS = dict(
    peak_threshold=0.001,
    min_peak_height=0.0001,
    aperiodic_mode='fixed',
)

# Conditions to analyze (in priority order)
CONDITIONS = [
    ('EyesClosed', 'pre'),   # Primary — where delta anti-alignment lives
    ('EyesOpen', 'pre'),     # Secondary — STD gave d=0.119
    ('EyesClosed', 'post'),  # Post-cognitive
    ('EyesOpen', 'post'),    # Post-cognitive
]

# 4-band definitions (matching standard extraction)
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


# ── specparam helpers ─────────────────────────────────────────────────────

def _get_peak_params(sm):
    """Get peak params from SpectralModel (specparam 1.x/2.x compatible)."""
    try:
        peaks = sm.get_params('peak')
    except Exception:
        try:
            peaks = sm.peak_params_
        except AttributeError:
            return np.empty((0, 3))
    if peaks is None:
        return np.empty((0, 3))
    peaks = np.asarray(peaks)
    if peaks.size == 0:
        return np.empty((0, 3))
    if peaks.ndim == 1:
        peaks = peaks.reshape(1, -1)
    return peaks


def _get_aperiodic_params(sm):
    """Get aperiodic (offset, exponent) from SpectralModel."""
    try:
        ap = sm.get_params('aperiodic')
    except Exception:
        try:
            ap = sm.aperiodic_params_
        except AttributeError:
            return np.nan, np.nan
    if ap is None:
        return np.nan, np.nan
    ap = np.asarray(ap).ravel()
    if len(ap) < 2:
        return np.nan, np.nan
    return float(ap[0]), float(ap[1])


def _get_r_squared(sm):
    """Get R² from SpectralModel."""
    try:
        gof = sm.get_metrics('gof')
        if gof is not None and not (hasattr(gof, '__len__') and len(gof) == 0):
            return float(gof)
    except Exception:
        pass
    try:
        return float(sm.r_squared_)
    except Exception:
        return np.nan


# ── Overlap-trim band construction ───────────────────────────────────────

def build_target_bands(f0, freq_ceil=FREQ_CEIL, freq_floor=FREQ_FLOOR, offset=0.0):
    """Build target phi-octave bands with padded fit windows."""
    bands = []
    for n in range(-4, 10):
        target_lo = f0 * PHI ** (n + offset)
        target_hi = f0 * PHI ** (n + 1 + offset)

        if target_lo >= freq_ceil:
            break
        if target_hi <= freq_floor:
            continue

        target_lo = max(target_lo, freq_floor)
        target_hi = min(target_hi, freq_ceil)

        target_width = target_hi - target_lo
        if target_width < 0.5:
            continue

        fit_lo = f0 * PHI ** (n + offset - PAD_OCTAVES)
        fit_hi = f0 * PHI ** (n + 1 + offset + PAD_OCTAVES)
        fit_lo = max(fit_lo, freq_floor)
        fit_hi = min(fit_hi, freq_ceil)

        bands.append({
            'name': f'n{n:+d}',
            'n': n,
            'target_lo': target_lo,
            'target_hi': target_hi,
            'fit_lo': fit_lo,
            'fit_hi': fit_hi,
        })
    return bands


def merge_narrow_targets(bands, min_width_hz=1.5, min_bins=12, freq_res=0.25):
    """Merge consecutive narrow target bands."""
    merged = []
    i = 0
    while i < len(bands):
        b = bands[i].copy()
        width = b['target_hi'] - b['target_lo']
        n_bins = int(width / freq_res)

        if width < min_width_hz or n_bins < min_bins:
            while (i + 1 < len(bands) and
                   (b['target_hi'] - b['target_lo'] < min_width_hz or
                    int((b['target_hi'] - b['target_lo']) / freq_res) < min_bins)):
                i += 1
                b['target_hi'] = bands[i]['target_hi']
                b['fit_hi'] = bands[i]['fit_hi']
            b['name'] = f'{b["name"]}_merged'

        if b['fit_lo'] > b['target_lo']:
            b['fit_lo'] = b['target_lo'] * 0.7
        if b['fit_hi'] < b['target_hi']:
            b['fit_hi'] = b['target_hi'] * 1.3

        merged.append(b)
        i += 1
    return merged


# ── Overlap-trim extraction ──────────────────────────────────────────────

def extract_overlap_trim(data, ch_names, fs,
                          f0=F0, freq_ceil=FREQ_CEIL, offset=0.0):
    """
    Overlap-trim per-phi-octave extraction for one subject.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_samples)
    ch_names : list of str
    fs : float
    f0, freq_ceil, offset : float

    Returns
    -------
    peaks_df : DataFrame with columns channel, freq, power, bandwidth, phi_octave, ...
    band_info : list of dicts with per-band statistics
    """
    nperseg = WELCH_NPERSEG
    noverlap = nperseg // 2
    freq_res = fs / nperseg

    raw_bands = build_target_bands(f0, freq_ceil, FREQ_FLOOR, offset)
    bands = merge_narrow_targets(raw_bands, min_width_hz=1.5,
                                  min_bins=12, freq_res=freq_res)

    all_peaks = []
    band_stats = []
    aperiodic_exponents = []

    for band in bands:
        bname = band['name']
        target_lo = band['target_lo']
        target_hi = band['target_hi']
        fit_lo = band['fit_lo']
        fit_hi = band['fit_hi']

        fit_width = fit_hi - fit_lo

        # Adapt FOOOF params to fit window width
        max_n_peaks = max(3, min(15, int(fit_width / 1.5)))
        max_peak_width = min(fit_width * 0.6, 12.0)
        peak_width_limits = [max(0.5, 2 * freq_res), max_peak_width]

        fooof_params = {
            **FOOOF_BASE_PARAMS,
            'max_n_peaks': max_n_peaks,
            'peak_width_limits': peak_width_limits,
        }

        band_r2s = []
        n_fitted = 0
        n_passed = 0
        n_peaks_kept = 0
        n_peaks_trimmed = 0

        for i, ch in enumerate(ch_names):
            sig = data[i, :]
            if len(sig) < nperseg:
                continue

            freqs, psd = welch(sig, fs, nperseg=nperseg, noverlap=noverlap)

            fit_mask = (freqs >= fit_lo) & (freqs <= fit_hi)
            if fit_mask.sum() < 10:
                continue

            n_fitted += 1

            sm = SpectralModel(**fooof_params)
            try:
                sm.fit(freqs, psd, [fit_lo, fit_hi])
            except Exception:
                continue

            r2 = _get_r_squared(sm)
            if np.isnan(r2) or r2 < R2_MIN:
                continue

            n_passed += 1
            band_r2s.append(r2)

            offset_val, exponent = _get_aperiodic_params(sm)
            aperiodic_exponents.append(exponent)

            peak_params = _get_peak_params(sm)

            for row in peak_params:
                peak_freq = row[0]

                # TRIM: only keep peaks within the TARGET band
                if target_lo <= peak_freq < target_hi:
                    all_peaks.append({
                        'channel': ch,
                        'freq': peak_freq,
                        'power': row[1],
                        'bandwidth': row[2],
                        'phi_octave': bname,
                    })
                    n_peaks_kept += 1
                else:
                    n_peaks_trimmed += 1

        band_stats.append({
            'band_name': bname,
            'target_lo': round(target_lo, 3),
            'target_hi': round(target_hi, 3),
            'fit_lo': round(fit_lo, 3),
            'fit_hi': round(fit_hi, 3),
            'n_channels_fitted': n_fitted,
            'n_channels_passed': n_passed,
            'mean_r_squared': np.mean(band_r2s) if band_r2s else np.nan,
            'peaks_kept': n_peaks_kept,
            'peaks_trimmed': n_peaks_trimmed,
        })

    mean_exponent = np.mean(aperiodic_exponents) if aperiodic_exponents else np.nan
    peaks_df = pd.DataFrame(all_peaks) if all_peaks else pd.DataFrame(
        columns=['channel', 'freq', 'power', 'bandwidth', 'phi_octave'])

    return peaks_df, band_stats, mean_exponent


# ── Data loading ─────────────────────────────────────────────────────────

def find_edf(sub_id, task, acq, data_dir):
    """Find EDF file for a given subject/condition."""
    path = os.path.join(data_dir, sub_id, 'ses-1', 'eeg',
                        f'{sub_id}_ses-1_task-{task}_acq-{acq}_eeg.edf')
    if os.path.isfile(path):
        return path
    return None


def load_subject_edf(edf_path, target_fs=TARGET_FS):
    """Load EDF, pick EEG channels, downsample, notch filter."""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.pick_types(eeg=True, exclude='bads')
    raw.notch_filter(freqs=[50, 100], verbose=False)
    if raw.info['sfreq'] > target_fs:
        raw.resample(target_fs, verbose=False)
    return raw.get_data(), raw.ch_names, raw.info['sfreq']


# ── Lattice math ─────────────────────────────────────────────────────────

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


# ── Per-subject analysis ─────────────────────────────────────────────────

def analyze_subject_ot(sub_id, task, acq, data_dir, participants_df=None):
    """
    Full OT pipeline for one subject × one condition.
    Returns dict with dominant peaks and lattice metrics, or None on failure.
    """
    edf_path = find_edf(sub_id, task, acq, data_dir)
    if edf_path is None:
        return None

    try:
        data, ch_names, fs = load_subject_edf(edf_path)
    except Exception as e:
        print(f"    ERROR loading {sub_id}: {e}")
        return None

    peaks_df, band_stats, mean_exponent = extract_overlap_trim(
        data, ch_names, fs, f0=F0, freq_ceil=FREQ_CEIL)

    if len(peaks_df) == 0:
        return None

    # Extract dominant peak per band (pooled across all channels)
    row = {
        'subject': sub_id,
        'task': task,
        'acq': acq,
        'n_channels': len(ch_names),
        'n_peaks_total': len(peaks_df),
        'aperiodic_exponent': mean_exponent,
    }

    # Band quality info
    for bs in band_stats:
        bname = bs['band_name']
        row[f'r2_{bname}'] = bs['mean_r_squared']
        row[f'kept_{bname}'] = bs['peaks_kept']
        row[f'trimmed_{bname}'] = bs['peaks_trimmed']

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


# ── Statistical analysis ─────────────────────────────────────────────────

def run_statistics(dom_df, condition_label):
    """Run full statistical suite on dominant-peak results."""

    print(f"\n{'='*70}")
    print(f"  {condition_label} (OVERLAP-TRIM)")
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
        for _, srow in analysis_df.iterrows():
            band_ds = []
            for band_name in BANDS:
                freq = srow[f'{band_name}_freq']
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
    for _, srow in analysis_df.iterrows():
        band_ds = []
        for band_name in BANDS:
            freq = srow[f'{band_name}_freq']
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

        young = age_valid[age_valid['age'] < 35]['mean_d'].values
        old = age_valid[age_valid['age'] >= 55]['mean_d'].values
        if len(young) >= 10 and len(old) >= 10:
            u_stat, p_group = stats.mannwhitneyu(young, old, alternative='two-sided')
            d_group = (young.mean() - old.mean()) / np.sqrt((young.std()**2 + old.std()**2)/2)
            print(f"    Young (<35, N={len(young)}) vs Old (≥55, N={len(old)}): "
                  f"d={d_group:.3f}, p={p_group:.4f}")

    # ── STD vs OT comparison (delta focus) ──
    print(f"\n  --- Extraction Comparison (STD vs OT) ---")
    print(f"    [Load STD results for paired comparison if available]")

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


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.isdir(DATA_DIR):
        print(f"ERROR: Data directory not found: {DATA_DIR}")
        sys.exit(1)

    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUT_DIR}")
    print(f"Extraction: OVERLAP-TRIM (pad={PAD_OCTAVES} phi-oct, R²≥{R2_MIN})")

    # Show band structure
    freq_res = TARGET_FS / WELCH_NPERSEG
    raw_bands = build_target_bands(F0, FREQ_CEIL, FREQ_FLOOR)
    bands = merge_narrow_targets(raw_bands, min_width_hz=1.5,
                                  min_bins=12, freq_res=freq_res)

    print(f"\nPhi-octave bands ({len(bands)}):")
    print(f"  {'name':>12}  {'target':>18}  {'fit window':>18}")
    for b in bands:
        tgt = f"[{b['target_lo']:5.2f}, {b['target_hi']:5.2f}]"
        fit = f"[{b['fit_lo']:5.2f}, {b['fit_hi']:5.2f}]"
        print(f"  {b['name']:>12}  {tgt:>18}  {fit:>18}")

    # Load participants
    participants_df = None
    for p in [os.path.join(DATA_DIR, 'participants.tsv'),
              '/Volumes/T9/dortmund_data/participants.tsv']:
        if os.path.isfile(p):
            participants_df = pd.read_csv(p, sep='\t')
            print(f"\nParticipants: {len(participants_df)} subjects loaded")
            break

    # Find available subjects
    subjects = sorted([d for d in os.listdir(DATA_DIR)
                       if d.startswith('sub-') and os.path.isdir(os.path.join(DATA_DIR, d))])
    print(f"Subject folders: {len(subjects)}")

    # Process each condition
    all_summaries = {}
    for task, acq in CONDITIONS:
        condition_label = f"{task}_{acq} (ses-1)"
        print(f"\n\n{'#'*70}")
        print(f"  Processing: {condition_label} — OVERLAP-TRIM")
        print(f"{'#'*70}")

        # Check availability
        available = []
        for sub in subjects:
            edf = find_edf(sub, task, acq, DATA_DIR)
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

            row = analyze_subject_ot(sub, task, acq, DATA_DIR, participants_df)
            if row is not None:
                results.append(row)

            # Memory management
            if (i + 1) % 100 == 0:
                gc.collect()

        elapsed = time.time() - t0
        print(f"\n  Processed {len(results)}/{len(available)} subjects in {elapsed:.0f}s")

        if len(results) < 10:
            print(f"  Too few successful subjects for analysis")
            continue

        dom_df = pd.DataFrame(results)

        # Save per-subject results
        out_csv = os.path.join(OUT_DIR, f'dortmund_ot_dominant_peaks_{task}_{acq}.csv')
        dom_df.to_csv(out_csv, index=False)
        print(f"  Saved: {out_csv}")

        # Run statistics
        stats_result = run_statistics(dom_df, condition_label)
        all_summaries[f'{task}_{acq}'] = stats_result

        # Save summary
        if stats_result:
            summary_csv = os.path.join(OUT_DIR, f'dortmund_ot_summary_{task}_{acq}.csv')
            summary_row = {k: v for k, v in stats_result.items() if k != 'base_results'}
            pd.DataFrame([summary_row]).to_csv(summary_csv, index=False)

    # ── Cross-condition comparison ──
    print(f"\n\n{'='*70}")
    print(f"  CROSS-CONDITION COMPARISON (OT vs STD)")
    print(f"{'='*70}")

    # Load STD results for comparison
    std_dir = '/Volumes/T9/dortmund_data/lattice_results'
    for task, acq in CONDITIONS:
        label = f"{task}_{acq}"
        std_csv = os.path.join(std_dir, f'dortmund_dominant_peaks_{task}_{acq}.csv')
        ot_csv = os.path.join(OUT_DIR, f'dortmund_ot_dominant_peaks_{task}_{acq}.csv')

        if os.path.exists(std_csv) and os.path.exists(ot_csv):
            std_df = pd.read_csv(std_csv)
            ot_df = pd.read_csv(ot_csv)

            # Match subjects
            common = set(std_df['subject']) & set(ot_df['subject'])
            std_m = std_df[std_df['subject'].isin(common)].set_index('subject').sort_index()
            ot_m = ot_df[ot_df['subject'].isin(common)].set_index('subject').sort_index()

            print(f"\n  {label} (N={len(common)} paired):")

            for band_name in BANDS:
                std_d = std_m[f'{band_name}_d'].dropna()
                ot_d = ot_m[f'{band_name}_d'].dropna()
                common_b = std_d.index.intersection(ot_d.index)
                if len(common_b) > 10:
                    s = std_d.loc[common_b].values
                    o = ot_d.loc[common_b].values
                    t, p = stats.ttest_rel(o, s)
                    print(f"    {band_name:6s}: STD d̄={s.mean():.4f} → OT d̄={o.mean():.4f}  "
                          f"Δ={o.mean()-s.mean():+.4f}  paired-t p={p:.3e}")

            # 4-band
            std_mean = std_m['mean_d'].dropna()
            ot_mean = ot_m['mean_d'].dropna()
            common_4 = std_mean.index.intersection(ot_mean.index)
            if len(common_4) > 10:
                s4 = std_mean.loc[common_4].values
                o4 = ot_mean.loc[common_4].values
                t4, p4 = stats.ttest_rel(o4, s4)
                print(f"    {'4-band':6s}: STD d̄={s4.mean():.4f} → OT d̄={o4.mean():.4f}  "
                      f"Δ={o4.mean()-s4.mean():+.4f}  paired-t p={p4:.3e}")

                # Effect sizes
                null_d = 0.0323
                std_cohen = (null_d - s4.mean()) / s4.std()
                ot_cohen = (null_d - o4.mean()) / o4.std()
                print(f"    Cohen's d: STD={std_cohen:.3f} → OT={ot_cohen:.3f}")

    print(f"\n\nDone! Results saved to {OUT_DIR}")


if __name__ == '__main__':
    main()

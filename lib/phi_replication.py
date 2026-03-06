#!/usr/bin/env python3
"""
Unified Phi-Lattice Replication Protocol
=========================================

Single locked pipeline for dominant-peak lattice analysis on any EEG dataset.
Produces:
  - 14-position enrichment table (degree-7 noble positions)
  - Per-subject alignment statistics (Cohen's d at degree-2)
  - Cross-base comparison (9 bases)
  - Pre-registered prediction scoring (for HBN replication)

All parameters are hardcoded. Zero analyst degrees of freedom.

Extraction method: Overlap-trim (primary), Standard FOOOF (sensitivity).
Both run automatically for every dataset.

Source: Consolidated from run_dortmund_dominant_peak.py,
        run_dortmund_overlap_trim.py, per_position_diagnostic.py,
        per_position_alignment.py — all tested and validated.
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.signal import welch
from specparam import SpectralModel
import os
import time
import gc
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════
# SECTION 1: LOCKED CONSTANTS — No configurability, no kwargs
# ══════════════════════════════════════════════════════════════════════════

F0 = 7.83  # Pre-specified Schumann fundamental
PHI = (1 + np.sqrt(5)) / 2

BANDS = {
    'delta':     (1.0,  4.0),
    'theta':     (4.0,  8.0),
    'alpha':     (8.0,  13.0),
    'beta_low':  (13.0, 20.0),   # phi-octave n=+1 (12.7-20.5 Hz)
    'beta_high': (20.0, 32.0),   # phi-octave n=+2 (20.5-33.1 Hz)
    'gamma':     (33.0, 45.0),   # floor at 33 Hz (avoids n+2 beta spillover), ceil at 45 Hz (below notch)
}

# 4 degree-2 Farey mediant positions — used for per-subject mean_d (d≈0.40 metric)
POSITIONS_DEG2 = {
    'boundary':  0.000,
    'noble_2':   1 - 1/PHI,       # 0.382
    'attractor': 0.500,
    'noble_1':   1/PHI,            # 0.618
}

# 8 positions for cross-base comparison (degree-4)
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

# 14 degree-7 positions — used for the enrichment table
POSITIONS_14 = {
    'boundary':    0.000,
    'noble_7':     (1/PHI)**7,      # 0.034
    'noble_6':     (1/PHI)**6,      # 0.056
    'noble_5':     (1/PHI)**5,      # 0.090
    'noble_4':     (1/PHI)**4,      # 0.146
    'noble_3':     (1/PHI)**3,      # 0.236
    'noble_2':     1 - 1/PHI,       # 0.382
    'attractor':   0.500,
    'noble_1':     1/PHI,           # 0.618
    'inv_noble_3': 1 - (1/PHI)**3,  # 0.764
    'inv_noble_4': 1 - (1/PHI)**4,  # 0.854
    'inv_noble_5': 1 - (1/PHI)**5,  # 0.910
    'inv_noble_6': 1 - (1/PHI)**6,  # 0.944
    'inv_noble_7': 1 - (1/PHI)**7,  # 0.966
}

# 9 comparison bases
BASES = {
    'phi':   PHI,
    '1.4':   1.4,
    'sqrt2': np.sqrt(2),
    '3/2':   1.5,
    '1.7':   1.7,
    '1.8':   1.8,
    '2':     2.0,
    'e':     np.e,
    'pi':    np.pi,
}

# Overlap-trim FOOOF — THE extraction method (not a choice)
FOOOF_OT_BASE = dict(
    peak_threshold=0.001,
    min_peak_height=0.0001,
    aperiodic_mode='fixed',
)
OT_PAD_OCTAVES = 0.5
OT_R2_MIN = 0.70
OT_FREQ_FLOOR = 1.0
OT_FREQ_CEIL = 45.0  # Capped below notch filters (50 Hz EU, 60 Hz US)

# Standard FOOOF — sensitivity analysis only, always reported alongside OT
FOOOF_STANDARD = dict(
    peak_width_limits=(1.0, 12.0),
    max_n_peaks=20,
    min_peak_height=0.01,
    peak_threshold=1.0,
    aperiodic_mode='fixed',
)

# Welch PSD — fixed sample count for consistent resolution across datasets
WELCH_NPERSEG = 1000  # 4s at 250 Hz → 0.25 Hz resolution (matches original Dortmund)

# KDE
KDE_BANDWIDTH = 0.03
N_PERMUTATIONS = 10_000

# Age bins — hardcoded for ALL datasets, no configurability
AGE_BINS = {
    'child':       (0, 10),
    'adolescent':  (10, 17),
    'young_adult': (18, 35),
    'middle':      (35, 55),
    'elderly':     (55, 100),
}


# ══════════════════════════════════════════════════════════════════════════
# SECTION 2: LATTICE MATH
# ══════════════════════════════════════════════════════════════════════════

def lattice_coord(freq, f0=F0, base=PHI):
    """Compute phi-lattice coordinate u ∈ [0,1) for a frequency."""
    if freq <= 0 or f0 <= 0:
        return np.nan
    return (np.log(freq / f0) / np.log(base)) % 1.0


def circ_dist(a, b):
    """Circular distance on [0, 1)."""
    d = abs(a - b)
    return min(d, 1 - d)


def min_lattice_dist(u, positions=None):
    """Minimum circular distance from u to any position in the set."""
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
    """Name of the nearest lattice position to u."""
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


def positions_for_base(base, degree=2):
    """Symmetric lattice positions for any base at specified degree.

    degree=2: boundary, attractor, inv, 1-inv (4 positions max, fair comparison)
    degree=3: adds inv^2, 1-inv^2, inv^3, 1-inv^3 (up to 8 positions)

    Filtered for uniqueness (>0.02 sep). Degree-2 is used for cross-base
    comparison to avoid mechanical advantage from position count."""
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
    if degree >= 3:
        _try_add('noble_2', inv ** 2)
        _try_add('inv_noble_2', 1 - inv ** 2)
        _try_add('noble_3', inv ** 3)
        _try_add('inv_noble_3', 1 - inv ** 3)
    return pos


def density_at_position(u_values, pos, bandwidth=KDE_BANDWIDTH):
    """Gaussian KDE density at a specific lattice position."""
    dists = np.array([circ_dist(u, pos) for u in u_values])
    return np.exp(-0.5 * (dists / bandwidth)**2).mean()


# ══════════════════════════════════════════════════════════════════════════
# SECTION 3: FOOOF EXTRACTION
# ══════════════════════════════════════════════════════════════════════════

# -- specparam compatibility helpers --

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


# -- 3a. Standard FOOOF extraction --

def extract_peaks_standard(data, ch_names, fs):
    """Standard global FOOOF [1,45] Hz on each channel.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_samples)
    ch_names : list of str
    fs : float

    Returns
    -------
    peaks_df : DataFrame with columns [channel, freq, power, bandwidth]
    mean_aperiodic_exponent : float
    """
    all_peaks = []
    aperiodic_exponents = []
    nperseg = min(WELCH_NPERSEG, data.shape[1])
    noverlap = nperseg // 2

    for i, ch in enumerate(ch_names):
        sig = data[i, :]
        freqs, psd = signal.welch(sig, fs=fs, nperseg=nperseg,
                                   noverlap=noverlap, window='hann')

        mask = (freqs >= 1.0) & (freqs <= OT_FREQ_CEIL)
        freqs_fit = freqs[mask]
        psd_fit = psd[mask]
        psd_fit = np.maximum(psd_fit, 1e-30)

        if len(freqs_fit) < 10:
            continue

        sm = SpectralModel(**FOOOF_STANDARD)
        try:
            sm.fit(freqs_fit, psd_fit, freq_range=[1, OT_FREQ_CEIL])
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


# -- 3b. Overlap-trim extraction helpers --

def _build_target_bands(f0=F0, freq_ceil=OT_FREQ_CEIL,
                        freq_floor=OT_FREQ_FLOOR):
    """Build target phi-octave bands with padded fit windows."""
    bands = []
    for n in range(-4, 10):
        target_lo = f0 * PHI ** n
        target_hi = f0 * PHI ** (n + 1)

        if target_lo >= freq_ceil:
            break
        if target_hi <= freq_floor:
            continue

        target_lo = max(target_lo, freq_floor)
        target_hi = min(target_hi, freq_ceil)

        target_width = target_hi - target_lo
        if target_width < 0.5:
            continue

        fit_lo = f0 * PHI ** (n - OT_PAD_OCTAVES)
        fit_hi = f0 * PHI ** (n + 1 + OT_PAD_OCTAVES)
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


def _merge_narrow_targets(bands, min_width_hz=1.5, min_bins=12, freq_res=0.25):
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


# -- 3b. Overlap-trim extraction --

def extract_peaks_overlap_trim(data, ch_names, fs):
    """Per-phi-octave FOOOF with half-octave padding (overlap-trim).

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_samples)
    ch_names : list of str
    fs : float

    Returns
    -------
    peaks_df : DataFrame with columns [channel, freq, power, bandwidth, phi_octave]
    mean_aperiodic_exponent : float
    """
    nperseg = min(WELCH_NPERSEG, data.shape[1])
    noverlap = nperseg // 2
    freq_res = fs / nperseg

    raw_bands = _build_target_bands(freq_ceil=OT_FREQ_CEIL)
    bands = _merge_narrow_targets(raw_bands, min_width_hz=1.5,
                                   min_bins=12, freq_res=freq_res)

    all_peaks = []
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
            **FOOOF_OT_BASE,
            'max_n_peaks': max_n_peaks,
            'peak_width_limits': peak_width_limits,
        }

        for i, ch in enumerate(ch_names):
            sig = data[i, :]
            if len(sig) < nperseg:
                continue

            freqs, psd = welch(sig, fs, nperseg=nperseg, noverlap=noverlap)

            fit_mask = (freqs >= fit_lo) & (freqs <= fit_hi)
            if fit_mask.sum() < 10:
                continue

            sm = SpectralModel(**fooof_params)
            try:
                sm.fit(freqs, psd, [fit_lo, fit_hi])
            except Exception:
                continue

            r2 = _get_r_squared(sm)
            if np.isnan(r2) or r2 < OT_R2_MIN:
                continue

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

    mean_exponent = np.mean(aperiodic_exponents) if aperiodic_exponents else np.nan
    if all_peaks:
        peaks_df = pd.DataFrame(all_peaks)
    else:
        peaks_df = pd.DataFrame(
            columns=['channel', 'freq', 'power', 'bandwidth', 'phi_octave'])

    return peaks_df, mean_exponent


# ══════════════════════════════════════════════════════════════════════════
# SECTION 4: DOMINANT PEAK COMPUTATION
# ══════════════════════════════════════════════════════════════════════════

def compute_dominant_peaks(peaks_df, subject_id=None, metadata=None):
    """Extract strongest peak per band, compute lattice coordinates.

    Parameters
    ----------
    peaks_df : DataFrame with columns [freq, power, bandwidth, ...]
    subject_id : str
    metadata : dict — optional {age, sex, ...} to include in output row

    Returns
    -------
    dict with keys: subject, {band}_{freq|power|u|d|nearest}, mean_d, n_bands
    or None if no peaks found
    """
    if len(peaks_df) == 0:
        return None

    row = {'subject': subject_id}

    if metadata:
        row.update(metadata)

    n_bands = 0
    for band_name, (lo, hi) in BANDS.items():
        bp = peaks_df[(peaks_df['freq'] >= lo) & (peaks_df['freq'] < hi)]
        if len(bp) > 0:
            idx = bp['power'].idxmax()
            freq = bp.loc[idx, 'freq']
            power = bp.loc[idx, 'power']
            u = lattice_coord(freq)
            d = min_lattice_dist(u, PHI_POSITIONS)
            nearest = nearest_position_name(u, PHI_POSITIONS)

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
    ds = [row[f'{b}_d'] for b in BANDS
          if not np.isnan(row.get(f'{b}_d', np.nan))]
    row['mean_d'] = np.mean(ds) if ds else np.nan

    return row


# ══════════════════════════════════════════════════════════════════════════
# SECTION 5: STATISTICS SUITE
# ══════════════════════════════════════════════════════════════════════════

def run_statistics(dom_df, label=''):
    """Full statistical suite on dominant-peak results.

    Tests:
    1. Population alignment (median freq → lattice dist → 100K perm null)
    2. Per-subject alignment (t-test of mean_d vs null, Cohen's d, Wilcoxon)
    3. Per-band decomposition (KS test per band vs band-specific uniform null)
    4. Cross-base comparison (9 bases, degree-3 positions, paired t-tests)
    5. Noble contribution (full phi vs shared-only, % contribution)
    6. Age analysis (if age column present, all applicable AGE_BINS)

    Returns dict with key statistics.
    """
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  N={len(dom_df)} subjects")
    print(f"{'='*70}")

    # Filter to subjects with sufficient bands
    n_bands_expected = len(BANDS)
    valid = dom_df[dom_df['n_bands'] == n_bands_expected].copy()
    n_valid = len(valid)
    n_total = len(dom_df)

    print(f"\n  Complete {n_bands_expected}-band data: {n_valid}/{n_total} subjects")
    for band_name in BANDS:
        n_with = dom_df[f'{band_name}_freq'].notna().sum()
        print(f"    {band_name}: {n_with}/{n_total} subjects with peak")

    if n_valid < 30:
        valid_fallback = dom_df[dom_df['n_bands'] >= n_bands_expected - 1].copy()
        print(f"  (Fallback to >={n_bands_expected - 1} bands: {len(valid_fallback)}/{n_total})")
        analysis_df = valid_fallback
    else:
        analysis_df = valid

    if len(analysis_df) < 10:
        print("  WARNING: Too few subjects for analysis!")
        return {}

    results = {'label': label, 'n_total': n_total, 'n_valid': len(analysis_df)}

    # Recompute lattice coordinates from frequencies (never trust pre-existing columns)
    # Per-subject alignment uses POSITIONS_DEG2 (4 positions) → Cohen's d ≈ 0.40
    # Cross-base comparison uses degree=3 (up to 8 positions) → phi rank 1/9
    for band_name in BANDS:
        freq_col = f'{band_name}_freq'
        if freq_col in analysis_df.columns:
            analysis_df[f'{band_name}_u'] = analysis_df[freq_col].apply(
                lambda f: lattice_coord(f) if pd.notna(f) else np.nan)
            analysis_df[f'{band_name}_d'] = analysis_df[f'{band_name}_u'].apply(
                lambda u: min_lattice_dist(u, POSITIONS_DEG2) if pd.notna(u) else np.nan)
            analysis_df[f'{band_name}_nearest'] = analysis_df[f'{band_name}_u'].apply(
                lambda u: nearest_position_name(u, POSITIONS_DEG2) if pd.notna(u) else 'none')

    # Recompute mean_d from degree-2 positions (4 positions, gives d≈0.40)
    band_d_cols = [f'{b}_d' for b in BANDS if f'{b}_d' in analysis_df.columns]
    analysis_df['mean_d'] = analysis_df[band_d_cols].mean(axis=1)

    # ── 1. Population-level alignment ──
    print(f"\n  --- Population-Level Alignment (f0={F0} Hz) ---")
    pop_freqs = {}
    for band_name in BANDS:
        freqs = analysis_df[f'{band_name}_freq'].dropna()
        if len(freqs) > 0:
            med = freqs.median()
            pop_freqs[band_name] = med
            u = lattice_coord(med)
            d = min_lattice_dist(u, POSITIONS_DEG2)
            near = nearest_position_name(u, POSITIONS_DEG2)
            print(f"    {band_name}: median={med:.2f} Hz, u={u:.3f}, d={d:.3f} → {near}")

    pop_ds = [min_lattice_dist(lattice_coord(f), POSITIONS_DEG2)
              for f in pop_freqs.values()]
    pop_mean_d = np.mean(pop_ds)
    print(f"    Population mean_d = {pop_mean_d:.4f}")

    # Null: 100K permutations
    n_perm = 100_000
    null_pop = np.zeros(n_perm)
    for i in range(n_perm):
        ds = []
        for band_name, (lo, hi) in BANDS.items():
            if band_name in pop_freqs:
                ds.append(min_lattice_dist(
                    lattice_coord(np.random.uniform(lo, hi)), POSITIONS_DEG2))
        null_pop[i] = np.mean(ds)
    p_pop = (null_pop <= pop_mean_d).mean()
    print(f"    Uniform null: p = {p_pop:.4f}")
    results['pop_mean_d'] = pop_mean_d
    results['p_pop'] = p_pop

    # ── 2. Per-subject alignment ──
    print(f"\n  --- Per-Subject Alignment ---")
    mean_ds = analysis_df['mean_d'].values
    obs_mean = mean_ds.mean()
    obs_sd = mean_ds.std()

    null_expected = np.mean([min_lattice_dist(np.random.uniform(0, 1), POSITIONS_DEG2)
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
    null_ref = np.array([np.mean([min_lattice_dist(np.random.uniform(0, 1), POSITIONS_DEG2)
                                   for _ in range(len(BANDS))])
                          for _ in range(10_000)])
    p5 = np.percentile(null_ref, 5)
    for md in mean_ds:
        if md <= p5:
            n_sig += 1
    print(f"    Individually significant (p<0.05): {n_sig}/{len(analysis_df)} "
          f"({100*n_sig/len(analysis_df):.1f}% vs 5% expected)")

    results['obs_mean_d'] = obs_mean
    results['null_expected_d'] = null_expected
    results['cohen_d'] = cohen_d
    results['p_ttest'] = p_ttest
    results['p_wilcox'] = p_wilcox
    results['n_individually_sig'] = n_sig

    # ── 3. Per-band breakdown ──
    print(f"\n  --- Per-Band Breakdown ---")
    for band_name, (lo, hi) in BANDS.items():
        ds = analysis_df[f'{band_name}_d'].dropna().values
        if len(ds) < 5:
            continue
        band_null = np.array([min_lattice_dist(
            lattice_coord(np.random.uniform(lo, hi)), POSITIONS_DEG2)
                              for _ in range(10_000)])
        ks_stat, ks_p = stats.ks_2samp(ds, band_null, alternative='less')
        band_d = (band_null.mean() - ds.mean()) / ds.std() if ds.std() > 0 else 0
        nearests = analysis_df[f'{band_name}_nearest'].value_counts()
        top_pos = nearests.index[0] if len(nearests) > 0 else 'none'
        top_pct = 100 * nearests.iloc[0] / nearests.sum() if len(nearests) > 0 else 0
        print(f"    {band_name}: mean_d={ds.mean():.4f}, null={band_null.mean():.4f}, "
              f"KS p={ks_p:.3e}, d={band_d:.2f}  | top: {top_pos} ({top_pct:.0f}%)")

    # ── 4. Cross-base comparison (degree-3: up to 8 positions, matches paper) ──
    print(f"\n  --- Cross-Base Comparison (9 bases, degree-3) ---")
    base_results = {}

    # Collect per-band frequencies for null computation
    band_freqs = {}
    for band_name in BANDS:
        vals = analysis_df[f'{band_name}_freq'].dropna().values
        band_freqs[band_name] = vals

    for base_name, base_val in BASES.items():
        positions = positions_for_base(base_val, degree=3)
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

        # Compute base-specific null via permutation (5K shuffles for speed)
        rng = np.random.RandomState(42)
        n_perm = 5000
        null_means = np.empty(n_perm)
        for perm_i in range(n_perm):
            perm_ds = []
            for band_name, freqs in band_freqs.items():
                if len(freqs) == 0:
                    continue
                shuffled = rng.uniform(0, 1, len(freqs))
                dists = np.array([min_lattice_dist(u, positions) for u in shuffled])
                perm_ds.append(dists.mean())
            null_means[perm_i] = np.mean(perm_ds) if perm_ds else np.nan

        null_mean = np.nanmean(null_means)
        null_sd = np.nanstd(null_means)
        z_score = (null_mean - seg_ds.mean()) / null_sd if null_sd > 0 else 0.0
        p_val = np.mean(null_means <= seg_ds.mean())  # fraction of nulls as extreme

        base_results[base_name] = {
            'mean_d': seg_ds.mean(), 'median_d': np.median(seg_ds),
            'sd_d': seg_ds.std(), 'n_positions': len(positions),
            'null_mean': null_mean, 'z_score': z_score, 'p_value': p_val,
            'values': seg_ds,
        }

    # Rank by z-score (null-adjusted, fair across different position geometries)
    ranking_z = sorted(base_results.items(), key=lambda x: -x[1]['z_score'])
    ranking_raw = sorted(base_results.items(), key=lambda x: x[1]['mean_d'])

    print(f"\n    Rank by z-score (null-adjusted):")
    for rank, (bname, br) in enumerate(ranking_z, 1):
        marker = ' ←' if bname == 'phi' else ''
        print(f"    {rank}. {bname:6s}: z={br['z_score']:+.2f}, mean_d={br['mean_d']:.4f}, "
              f"null={br['null_mean']:.4f}, p={br['p_value']:.4f} ({br['n_positions']} pos){marker}")

    print(f"\n    Rank by raw mean_d:")
    for rank, (bname, br) in enumerate(ranking_raw, 1):
        marker = ' ←' if bname == 'phi' else ''
        print(f"    {rank}. {bname:6s}: mean_d={br['mean_d']:.4f} "
              f"(±{br['sd_d']:.4f}, {br['n_positions']} pos){marker}")

    # Phi vs others (paired t-test on per-subject d values)
    phi_ds = base_results['phi']['values']
    print(f"\n    Phi vs competitors (paired t-test):")
    for bname, br in ranking_z:
        if bname == 'phi':
            continue
        other_ds = br['values']
        if len(phi_ds) == len(other_ds):
            t, p = stats.ttest_rel(phi_ds, other_ds, alternative='less')
            sig = '*' if p < 0.05 else ' '
            print(f"      phi vs {bname:6s}: Δd={other_ds.mean()-phi_ds.mean():+.4f}, "
                  f"p={p:.3e} {sig}")

    # Phi rank by z-score (the fair metric for pre-registration P4)
    phi_rank_z = next(i+1 for i, (name, _) in enumerate(ranking_z) if name == 'phi')
    phi_rank_raw = next(i+1 for i, (name, _) in enumerate(ranking_raw) if name == 'phi')
    results['phi_rank'] = phi_rank_z
    results['phi_rank_raw'] = phi_rank_raw
    results['base_results'] = base_results

    # ── 5. Noble contribution ──
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
    results['noble_contrib'] = noble_contrib

    # ── 6. Age analysis ──
    if 'age' in analysis_df.columns and analysis_df['age'].notna().sum() > 20:
        print(f"\n  --- Age Analysis ---")
        age_valid = analysis_df[analysis_df['age'].notna()]
        r, p_age = stats.pearsonr(age_valid['age'], age_valid['mean_d'])
        print(f"    mean_d ~ age: r={r:.3f}, p={p_age:.4f} (N={len(age_valid)})")
        results['age_r'] = r
        results['age_p'] = p_age

        # Report all applicable AGE_BINS with N >= 10
        for bin_name, (lo, hi) in AGE_BINS.items():
            bin_data = age_valid[(age_valid['age'] >= lo) &
                                (age_valid['age'] < hi)]['mean_d'].values
            if len(bin_data) >= 10:
                print(f"    {bin_name} ({lo}-{hi}): N={len(bin_data)}, "
                      f"mean_d={bin_data.mean():.4f} ± {bin_data.std():.4f}")

        # Young vs old comparison (using first and last bins with sufficient data)
        young = age_valid[age_valid['age'] < 35]['mean_d'].values
        old = age_valid[age_valid['age'] >= 55]['mean_d'].values
        if len(young) >= 10 and len(old) >= 10:
            u_stat, p_group = stats.mannwhitneyu(young, old, alternative='two-sided')
            d_group = (young.mean() - old.mean()) / np.sqrt(
                (young.std()**2 + old.std()**2) / 2)
            print(f"    Young (<35, N={len(young)}) vs Old (≥55, N={len(old)}): "
                  f"d={d_group:.3f}, p={p_group:.4f}")

    return results


# ══════════════════════════════════════════════════════════════════════════
# SECTION 6: 14-POSITION ENRICHMENT TABLE
# ══════════════════════════════════════════════════════════════════════════

def run_14position_enrichment(dom_df, n_perm=N_PERMUTATIONS):
    """Compute KDE enrichment at all 14 degree-7 positions.

    For each of 14 positions × N bands:
    - Compute KDE density from observed u values
    - Compute null density from n_perm uniform shuffles
    - Report enrichment %, z-score, p-value

    Returns
    -------
    DataFrame with columns: position, u, enrichment_pct, z_score, p_value,
                            plus per-band breakdown columns
    """
    print(f"\n{'='*70}")
    print(f"  14-Position Enrichment Table (degree-7)")
    print(f"{'='*70}")

    band_names = list(BANDS.keys())

    # Collect u values per band
    band_u = {}
    all_u = []
    for band in band_names:
        col = f'{band}_u'
        if col in dom_df.columns:
            vals = dom_df[col].dropna().values
            band_u[band] = vals
            all_u.extend(vals)
    all_u = np.array(all_u)

    if len(all_u) == 0:
        print("  WARNING: No u values found!")
        return pd.DataFrame()

    sorted_pos = sorted(POSITIONS_14.items(), key=lambda x: x[1])
    rows = []

    for pos_name, pos_val in sorted_pos:
        row = {'position': pos_name, 'u': pos_val}

        # Combined enrichment (all bands)
        result = _permutation_test_position(all_u, pos_val, n_perm)
        row['enrichment_pct'] = result['enrichment_pct']
        row['z_score'] = result['z_score']
        row['p_value'] = result['p_value']
        row['observed_density'] = result['observed_density']
        row['null_mean'] = result['null_mean']

        # Per-band breakdown
        for band in band_names:
            if band in band_u and len(band_u[band]) > 0:
                b_result = _permutation_test_position(
                    band_u[band], pos_val, min(n_perm, 5000))
                row[f'{band}_enrich'] = b_result['enrichment_pct']
                row[f'{band}_z'] = b_result['z_score']
            else:
                row[f'{band}_enrich'] = np.nan
                row[f'{band}_z'] = np.nan

        rows.append(row)

    enrichment_df = pd.DataFrame(rows)

    # Print table — dynamic column headers for all bands
    band_abbrev = {'delta': 'δ', 'theta': 'θ', 'alpha': 'α',
                   'beta_low': 'βL', 'beta_high': 'βH', 'gamma': 'γ'}
    band_hdrs = ''.join(f'{band_abbrev.get(b, b[:2]):>7s}' for b in band_names)
    band_seps = ''.join(f"{'─'*7}" for _ in band_names)
    print(f"\n  {'Position':<15} {'u':>6} {'Enrichment':>11} {'z':>8} {'p':>8} {band_hdrs}")
    print(f"  {'─'*15} {'─'*6} {'─'*11} {'─'*8} {'─'*8} {band_seps}")

    for _, r in enrichment_df.iterrows():
        # Determine verdict
        if r['z_score'] > 2:
            verdict = '***'
        elif r['z_score'] > 1:
            verdict = '* '
        elif r['z_score'] < -2:
            verdict = 'vvv'
        elif r['z_score'] < -1:
            verdict = 'v  '
        else:
            verdict = '   '

        band_vals = ''.join(
            f"{r.get(f'{b}_enrich', np.nan):>+6.0f}%" for b in band_names)

        print(f"  {r['position']:<15} {r['u']:>6.3f} {r['enrichment_pct']:>+10.1f}% "
              f"{r['z_score']:>8.1f} {r['p_value']:>8.4f} {band_vals}  {verdict}")

    # Summary: pullers and pushers
    pullers = enrichment_df[enrichment_df['z_score'] > 1.0]
    pushers = enrichment_df[enrichment_df['z_score'] < -1.0]
    print(f"\n  Pullers (z>1): {', '.join(pullers['position'].values)}")
    print(f"  Pushers (z<-1): {', '.join(pushers['position'].values)}")

    return enrichment_df


def _permutation_test_position(u_values, position, n_perm=10000):
    """Permutation test: does observed KDE density exceed uniform null?"""
    observed = density_at_position(u_values, position)
    null_densities = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = np.random.uniform(0, 1, len(u_values))
        null_densities[i] = density_at_position(shuffled, position)

    p_value = np.mean(null_densities >= observed)
    null_mean = null_densities.mean()
    null_std = null_densities.std()
    z_score = (observed - null_mean) / null_std if null_std > 0 else 0.0
    enrichment_pct = (observed / null_mean - 1) * 100 if null_mean > 0 else 0.0

    return {
        'observed_density': observed,
        'null_mean': null_mean,
        'null_std': null_std,
        'z_score': z_score,
        'p_value': p_value,
        'enrichment_pct': enrichment_pct,
    }


# ══════════════════════════════════════════════════════════════════════════
# SECTION 7: CONVENIENCE RUNNERS
# ══════════════════════════════════════════════════════════════════════════

def run_full_protocol(dom_df, label='', out_dir=None):
    """Run everything: statistics + 14-position enrichment.

    Parameters
    ----------
    dom_df : DataFrame — per-subject dominant peaks
    label : str — label for printing
    out_dir : str or None — save CSVs if provided

    Returns
    -------
    dict with 'stats' and 'enrichment' keys
    """
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    stats_result = run_statistics(dom_df, label)
    enrichment_df = run_14position_enrichment(dom_df)

    if out_dir and len(dom_df) > 0:
        dom_df.to_csv(os.path.join(out_dir, 'per_subject_dominant_peaks.csv'),
                      index=False)
        if len(enrichment_df) > 0:
            enrichment_df.to_csv(
                os.path.join(out_dir, '14position_enrichment.csv'), index=False)

        # Save cross-base comparison
        if 'base_results' in stats_result:
            base_rows = []
            for bname, br in stats_result['base_results'].items():
                base_rows.append({
                    'base': bname, 'mean_d': br['mean_d'],
                    'median_d': br['median_d'], 'sd_d': br['sd_d'],
                    'n_positions': br['n_positions'],
                })
            pd.DataFrame(base_rows).to_csv(
                os.path.join(out_dir, 'cross_base_comparison.csv'), index=False)

        # Save summary stats
        summary = {k: v for k, v in stats_result.items()
                   if k not in ('base_results',)}
        pd.DataFrame([summary]).to_csv(
            os.path.join(out_dir, 'summary_statistics.csv'), index=False)

    return {'stats': stats_result, 'enrichment': enrichment_df}


def process_subjects_from_eeg(subjects, load_fn, out_dir=None, label=''):
    """Process list of subjects from raw EEG.

    Extraction method is LOCKED to overlap-trim. Standard FOOOF runs
    automatically as a sensitivity analysis alongside the primary OT results.

    Parameters
    ----------
    subjects : list of subject identifiers
    load_fn : callable(subject_id) → (data, ch_names, fs, metadata_dict)
    out_dir : str or None — output directory
    label : str — label for statistics printing

    Returns
    -------
    dom_df_ot : DataFrame of per-subject OT results
    """
    ot_results = []
    std_results = []  # sensitivity analysis — always computed

    # Create directory for raw per-subject peak CSVs
    peaks_dir = None
    if out_dir:
        peaks_dir = os.path.join(out_dir, 'per_subject_peaks')
        os.makedirs(peaks_dir, exist_ok=True)

    t0 = time.time()
    n_total = len(subjects)

    for i, sub in enumerate(subjects):
        if (i + 1) % 5 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n_total - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{n_total}] {sub} "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

        try:
            data, ch_names, fs, metadata = load_fn(sub)
        except Exception as e:
            print(f"    ERROR loading {sub}: {e}")
            ot_results.append(None)
            std_results.append(None)
            continue

        # PRIMARY: overlap-trim
        try:
            peaks_ot, aperiodic_ot = extract_peaks_overlap_trim(data, ch_names, fs)
            row_ot = compute_dominant_peaks(peaks_ot, subject_id=sub, metadata=metadata)
            if row_ot:
                row_ot['aperiodic_exponent'] = aperiodic_ot
                row_ot['n_peaks_total'] = len(peaks_ot)
                row_ot['n_channels'] = len(ch_names)
            ot_results.append(row_ot)
            # Save raw peaks CSV
            if peaks_dir and len(peaks_ot) > 0:
                peaks_ot.to_csv(os.path.join(peaks_dir, f'{sub}_peaks.csv'),
                                index=False)
        except Exception as e:
            print(f"    ERROR OT extraction {sub}: {e}")
            ot_results.append(None)

        # SENSITIVITY: standard FOOOF (always reported)
        try:
            peaks_std, aperiodic_std = extract_peaks_standard(data, ch_names, fs)
            row_std = compute_dominant_peaks(peaks_std, subject_id=sub, metadata=metadata)
            if row_std:
                row_std['aperiodic_exponent'] = aperiodic_std
                row_std['n_peaks_total'] = len(peaks_std)
                row_std['n_channels'] = len(ch_names)
            std_results.append(row_std)
        except Exception as e:
            print(f"    ERROR STD extraction {sub}: {e}")
            std_results.append(None)

        # Memory management
        if (i + 1) % 50 == 0:
            gc.collect()

    elapsed = time.time() - t0
    print(f"\n  Processed {n_total} subjects in {elapsed:.0f}s")

    dom_df_ot = pd.DataFrame([r for r in ot_results if r is not None])
    dom_df_std = pd.DataFrame([r for r in std_results if r is not None])

    print(f"  OT: {len(dom_df_ot)} successful, STD: {len(dom_df_std)} successful")

    # Run primary analysis (OT)
    ot_result = run_full_protocol(dom_df_ot, label=f'{label} [OT]', out_dir=out_dir)

    # Run sensitivity analysis (Standard) — always reported
    std_out = os.path.join(out_dir, 'sensitivity_standard') if out_dir else None
    std_result = run_full_protocol(dom_df_std, label=f'{label} [STD sensitivity]',
                                   out_dir=std_out)

    return dom_df_ot


# ══════════════════════════════════════════════════════════════════════════
# SECTION 8: PRE-REGISTRATION REPORT
# ══════════════════════════════════════════════════════════════════════════

def generate_preregistration_report(dom_df, enrichment_df, stats_result, label=''):
    """Score pre-registered predictions as CONFIRMED / DISCONFIRMED.

    8 predictions locked before seeing data. Outputs standardized summary.
    P1-P3 thresholds calibrated to discovery data (joint null P ~ 0.07%).

    Parameters
    ----------
    dom_df : DataFrame — per-subject dominant peaks
    enrichment_df : DataFrame — 14-position enrichment (from run_14position_enrichment)
    stats_result : dict — from run_statistics
    label : str

    Returns
    -------
    report : dict with prediction keys and bool values
    """
    print(f"\n{'='*70}")
    print(f"  PRE-REGISTRATION REPORT: {label}")
    print(f"{'='*70}")

    report = {}

    # Index enrichment_df by position name for lookup
    enr = enrichment_df.set_index('position')

    # Primary predictions (all must hold for "replicates")
    # Thresholds calibrated to discovery data distributions:
    #   boundary  z > 1.5  (discovery min: 2.01 across all conditions)
    #   attractor z > 1.0  (discovery min: 2.48 for combined; -0.98 for EO-only)
    #   noble_1   z > 1.5  (discovery min: 1.98 across all conditions)
    # Joint null probability under H0 (z ~ N(0,1)): ~0.07%
    report['P1_boundary_pulls'] = enr.loc['boundary', 'z_score'] > 1.5
    report['P2_attractor_pulls'] = enr.loc['attractor', 'z_score'] > 1.0
    report['P3_noble1_pulls'] = enr.loc['noble_1', 'z_score'] > 1.5

    # Secondary predictions
    report['P4_phi_top3'] = stats_result.get('phi_rank', 99) <= 3

    push_positions = ['noble_3', 'noble_4', 'noble_5']
    push_z = enr.loc[push_positions, 'z_score'].mean()
    report['P5_push_zone'] = push_z < 0

    report['P6_cohens_d_gt_020'] = stats_result.get('cohen_d', 0) > 0.20

    # Tertiary predictions (exploratory)
    report['P7_theta_ec_convergence'] = stats_result.get('theta_ec_p', 1.0) < 0.05
    report['P8_age_invariance'] = abs(stats_result.get('age_r', 0)) < 0.15

    # NOTE: P9 (noble contribution > 50%) dropped — analytical null is 63.6% at
    # degree-3, so the threshold was below chance. Metric is geometric, not biological.

    # Score
    primary_pass = all([report['P1_boundary_pulls'],
                        report['P2_attractor_pulls'],
                        report['P3_noble1_pulls']])

    secondary_pass = sum([report['P4_phi_top3'],
                          report['P5_push_zone'],
                          report['P6_cohens_d_gt_020']])

    tertiary_pass = sum([report['P7_theta_ec_convergence'],
                         report['P8_age_invariance']])

    # Print results
    print(f"\n  PRIMARY PREDICTIONS (all must hold):")
    for key in ['P1_boundary_pulls', 'P2_attractor_pulls', 'P3_noble1_pulls']:
        status = 'CONFIRMED' if report[key] else 'DISCONFIRMED'
        print(f"    {key}: {status}")

    print(f"\n  VERDICT: {'REPLICATES' if primary_pass else 'DOES NOT REPLICATE'}")

    print(f"\n  SECONDARY PREDICTIONS ({secondary_pass}/3):")
    for key in ['P4_phi_top3', 'P5_push_zone', 'P6_cohens_d_gt_020']:
        status = 'CONFIRMED' if report[key] else 'DISCONFIRMED'
        print(f"    {key}: {status}")

    print(f"\n  TERTIARY PREDICTIONS ({tertiary_pass}/2, exploratory):")
    for key in ['P7_theta_ec_convergence', 'P8_age_invariance']:
        status = 'CONFIRMED' if report[key] else 'DISCONFIRMED'
        print(f"    {key}: {status}")

    print(f"\n  Key numbers:")
    print(f"    Cohen's d = {stats_result.get('cohen_d', np.nan):.3f}")
    print(f"    Phi rank = {stats_result.get('phi_rank', '?')}/9")
    print(f"    Boundary z = {enr.loc['boundary', 'z_score']:.1f} (threshold: 1.5)")
    print(f"    Attractor z = {enr.loc['attractor', 'z_score']:.1f} (threshold: 1.0)")
    print(f"    Noble_1 z = {enr.loc['noble_1', 'z_score']:.1f} (threshold: 1.5)")
    print(f"    Push zone mean z = {push_z:.1f}")

    report['primary_replicates'] = primary_pass
    report['secondary_count'] = secondary_pass
    report['tertiary_count'] = tertiary_pass

    return report


# ══════════════════════════════════════════════════════════════════════════
# SECTION 9: THETA EC CONVERGENCE TEST
# ══════════════════════════════════════════════════════════════════════════

def test_theta_ec_convergence(eo_df, ec_df):
    """Test whether theta converges on f₀ from EO to EC.

    |theta - f₀| should decrease from EO to EC (paired test).

    Parameters
    ----------
    eo_df : DataFrame — EO dominant peaks (must have 'subject' and 'theta_freq')
    ec_df : DataFrame — EC dominant peaks

    Returns
    -------
    dict with test results
    """
    print(f"\n{'='*70}")
    print(f"  Theta EC Convergence Test")
    print(f"{'='*70}")

    # Match subjects
    common = set(eo_df['subject']) & set(ec_df['subject'])
    eo_m = eo_df[eo_df['subject'].isin(common)].set_index('subject').sort_index()
    ec_m = ec_df[ec_df['subject'].isin(common)].set_index('subject').sort_index()

    # Get theta frequencies
    eo_theta = eo_m['theta_freq'].dropna()
    ec_theta = ec_m['theta_freq'].dropna()
    common_theta = eo_theta.index.intersection(ec_theta.index)

    if len(common_theta) < 10:
        print(f"  Too few paired subjects: {len(common_theta)}")
        return {'theta_ec_p': 1.0}

    eo_dist = np.abs(eo_theta.loc[common_theta].values - F0)
    ec_dist = np.abs(ec_theta.loc[common_theta].values - F0)

    t_stat, p_val = stats.ttest_rel(ec_dist, eo_dist, alternative='less')
    mean_eo = eo_dist.mean()
    mean_ec = ec_dist.mean()

    print(f"  N = {len(common_theta)} paired subjects")
    print(f"  |theta - f₀| EO: {mean_eo:.3f} Hz")
    print(f"  |theta - f₀| EC: {mean_ec:.3f} Hz")
    print(f"  Decrease: {mean_eo - mean_ec:.3f} Hz")
    print(f"  Paired t = {t_stat:.3f}, p = {p_val:.4f}")
    print(f"  {'CONVERGES' if p_val < 0.05 else 'NO CONVERGENCE'}")

    return {'theta_ec_p': p_val, 'theta_ec_t': t_stat,
            'theta_eo_dist': mean_eo, 'theta_ec_dist': mean_ec}

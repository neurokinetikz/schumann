"""
LEMON Phi-Lattice Cognition Analysis — Utility Library
========================================================

LEMON-specific functions for Paper 3: "Golden Ratio Lattice Precision
Predicts Cognitive Performance Across the Adult Lifespan."

Sections:
  A. Constants & Data Loading
  B. EEG Loading
  C. Spectral Parameterization (FOOOF)
  D. Compliance Scoring
  E. Subject Pipeline (FOOOF + SIE)
  F. Statistical Functions
"""

import os
import sys
import json
import gc
import logging
import warnings
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import welch

# Add project paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

# Existing project reuse
from ratio_specificity import lattice_coordinate, _enrichment_at_offset
from structural_phi_specificity import natural_positions, compute_structural_score

# SpectralModel import (specparam/fooof compat)
try:
    from specparam import SpectralModel
    _SPECPARAM = True
except ImportError:
    try:
        from fooof import FOOOF as SpectralModel
        _SPECPARAM = True
    except ImportError:
        _SPECPARAM = False

log = logging.getLogger(__name__)


# ============================================================================
# SECTION A: CONSTANTS & DATA LOADING
# ============================================================================

# --- Paths ---
LEMON_PREPROC_ROOT = ('/Volumes/T9/lemon_data/eeg_preprocessed/'
                      'EEG_MPILMBB_LEMON/EEG_Preprocessed_BIDS_ID/'
                      'EEG_Preprocessed')
LEMON_RAW_ROOT = ('/Volumes/T9/lemon_data/eeg_raw/'
                   'EEG_MPILMBB_LEMON/EEG_Raw_BIDS_ID')
LEMON_BEHAV_ROOT = ('/Volumes/T9/lemon_data/behavioral/'
                     'Behavioural_Data_MPILMBB_LEMON')
LEMON_COG_ROOT = f'{LEMON_BEHAV_ROOT}/Cognitive_Test_Battery_LEMON'
LEMON_META_PATH = (f'{LEMON_BEHAV_ROOT}/'
                   'META_File_IDs_Age_Gender_Education_Drug_Smoke_'
                   'SKID_LEMON.csv')

# --- EEG parameters ---
SFREQ = 250
WELCH_NPERSEG = 2048        # → 0.122 Hz resolution at 250 Hz
F0_PRIMARY = 8.5
F0_SENSITIVITY = 7.6
COMPLIANCE_WINDOW = 0.05
PHI = (1 + np.sqrt(5)) / 2

FOOOF_PARAMS = dict(
    freq_range=[1, 45],
    max_n_peaks=20,
    peak_threshold=0.001,
    min_peak_height=0.0001,
    peak_width_limits=[0.2, 20],
)
FOOOF_CHANNEL_R2_MIN = 0.70

# --- Exclusion criteria ---
MIN_CHANNEL_SURVIVAL_FRAC = 0.50   # relative to n_channels_loaded
MIN_MEAN_FOOOF_R2 = 0.85
MIN_MEAN_PEAKS_PER_CHANNEL = 3.0

# --- Demographics ---
EDUCATION_MAP = {
    'Gymnasium': 13, 'Gymansium': 13,   # typo in data
    'Realschule': 10,
    'Hauptschule': 9,
    'none': 8,
    'none (Hauptschule not finished)': 8,
}

AGE_BIN_MIDPOINTS = {
    '20-25': 22.5, '25-30': 27.5, '30-35': 32.5, '35-40': 37.5,
    '55-60': 57.5, '60-65': 62.5, '65-70': 67.5, '70-75': 72.5,
    '75-80': 77.5,
}

# --- Cognitive tests ---
COG_TESTS = {
    'CVLT': {
        'file': 'CVLT /CVLT.csv',
        'col': 'CVLT_6',
        'direction': 'higher_better',
        'log_transform': False,
    },
    'TMT_A': {
        'file': 'TMT/TMT.csv',
        'col': 'TMT_1',
        'direction': 'lower_better',
        'log_transform': True,
    },
    'TMT_B': {
        'file': 'TMT/TMT.csv',
        'col': 'TMT_5',
        'direction': 'lower_better',
        'log_transform': True,
    },
    'TAP_Alert': {
        'file': 'TAP_Alertness/TAP-Alertness.csv',
        'col': 'TAP_A_6',
        'direction': 'lower_better',
        'log_transform': True,
    },
    'TAP_WM': {
        'file': 'TAP_Working_Memory/TAP-Working Memory.csv',
        'col': 'TAP_WM_6',
        'direction': 'higher_better',
        'log_transform': False,
    },
    'TAP_Incompat': {
        'file': 'TAP_Incompatibility/TAP-Incompatibility.csv',
        'cols': ['TAP_I_9', 'TAP_I_2'],
        'compute': 'diff',
        'direction': 'lower_better',
        'log_transform': False,
    },
    'LPS': {
        'file': 'LPS/LPS.csv',
        'col': 'LPS_1',
        'direction': 'higher_better',
        'log_transform': False,
    },
    'WST': {
        'file': 'WST/WST.csv',
        'col': 'WST_1',
        'direction': 'higher_better',
        'log_transform': False,
    },
    'RWT': {
        'file': 'RWT/RWT.csv',
        'cols': ['RWT_8', 'RWT_20'],
        'compute': 'sum',
        'direction': 'higher_better',
        'log_transform': False,
    },
}

# --- Spatial channel groups ---
ANTERIOR_CHANNELS = [
    'Fp1', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'AFz',
    'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC5', 'FC3',
    'FC1', 'FC2', 'FC4', 'FC6', 'FT8',
]
POSTERIOR_CHANNELS = [
    'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P5', 'P3', 'Pz',
    'P4', 'P6', 'P8', 'TP7', 'TP8', 'PO9', 'PO7', 'PO3',
    'POz', 'PO4', 'PO8', 'PO10', 'O1', 'Oz', 'O2',
]
CHANNELS_1020 = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz',
    'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2',
]
IAF_POSTERIOR_ROI = ['O1', 'Oz', 'O2', 'PO3', 'POz', 'PO4', 'PO7', 'PO8']


# --- Data loading functions ---

def discover_subjects(preproc_root=LEMON_PREPROC_ROOT) -> List[str]:
    """Scan for sub-XXXXXX_EO.set files. Return sorted subject IDs."""
    subjects = set()
    if not os.path.isdir(preproc_root):
        log.warning(f"Preprocessed root not found: {preproc_root}")
        return []
    for fn in os.listdir(preproc_root):
        if fn.endswith('_EO.set') and not fn.startswith('.'):
            sid = fn.replace('_EO.set', '')
            subjects.add(sid)
    return sorted(subjects)


def select_held_out(subject_ids: List[str], frac: float = 0.10,
                    seed: int = 2026) -> Tuple[List[str], List[str]]:
    """Deterministic 10% held-out split. Returns (held_out, analysis_ids)."""
    n = max(1, round(len(subject_ids) * frac))
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(subject_ids))
    held_out = [subject_ids[i] for i in sorted(indices[:n])]
    analysis = [subject_ids[i] for i in sorted(indices[n:])]
    return held_out, analysis


def load_demographics(meta_path=LEMON_META_PATH) -> pd.DataFrame:
    """Parse META CSV → demographics DataFrame with midpoint ages."""
    df = pd.read_csv(meta_path)
    df = df.rename(columns={'ID': 'subject_id'})

    # Age bins → midpoints
    df['age_midpoint'] = df['Age'].map(AGE_BIN_MIDPOINTS)
    df['age_group'] = df['age_midpoint'].apply(
        lambda x: 'young' if pd.notna(x) and x <= 35 else 'elderly')

    # Gender: 1=female→0, 2=male→1
    gender_col = [c for c in df.columns if 'Gender' in c][0]
    df['sex'] = df[gender_col].map({1: 0, 2: 1})

    # Education → years
    df['education_years'] = df['Education'].map(EDUCATION_MAP)

    out = df[['subject_id', 'age_midpoint', 'age_group', 'sex',
              'education_years', 'Age']].copy()
    out = out.rename(columns={'Age': 'age_bin'})
    return out


def load_cognitive_data(cog_root=LEMON_COG_ROOT) -> pd.DataFrame:
    """Load all 9 cognitive tests, merge on ID, compute derived scores."""
    merged = None
    loaded_files = set()

    for test_name, spec in COG_TESTS.items():
        fpath = os.path.join(cog_root, spec['file'])
        if not os.path.exists(fpath):
            log.warning(f"Cognitive file not found: {fpath}")
            continue

        # Avoid loading the same file twice (TMT has both A and B)
        if fpath not in loaded_files:
            df = pd.read_csv(fpath)
            df = df.rename(columns={'ID': 'subject_id'})
            loaded_files.add(fpath)

            if merged is None:
                merged = df
            else:
                # Only merge new columns
                new_cols = [c for c in df.columns
                            if c not in merged.columns or c == 'subject_id']
                merged = merged.merge(df[new_cols], on='subject_id', how='outer')

    if merged is None:
        return pd.DataFrame()

    # Compute derived scores
    cog_df = merged[['subject_id']].copy()

    for test_name, spec in COG_TESTS.items():
        if 'col' in spec:
            col = spec['col']
            if col in merged.columns:
                cog_df[test_name] = pd.to_numeric(merged[col], errors='coerce')
            else:
                log.warning(f"Column {col} not found for {test_name}")
                cog_df[test_name] = np.nan
        elif 'cols' in spec:
            cols = spec['cols']
            missing = [c for c in cols if c not in merged.columns]
            if missing:
                log.warning(f"Columns {missing} not found for {test_name}")
                cog_df[test_name] = np.nan
            else:
                vals = [pd.to_numeric(merged[c], errors='coerce') for c in cols]
                if spec['compute'] == 'diff':
                    cog_df[test_name] = vals[0] - vals[1]
                elif spec['compute'] == 'sum':
                    cog_df[test_name] = vals[0] + vals[1]

        # Log-transform where specified (raw RT measures only)
        if spec.get('log_transform', False) and test_name in cog_df.columns:
            raw = cog_df[test_name]
            # Only log-transform positive values
            valid = raw > 0
            cog_df[f'log_{test_name}'] = np.nan
            cog_df.loc[valid, f'log_{test_name}'] = np.log(raw[valid])

    return cog_df


def build_master_table(demographics: pd.DataFrame,
                       cognitive: pd.DataFrame) -> pd.DataFrame:
    """OUTER merge demographics + cognitive. Listwise deletion per-test later."""
    master = demographics.merge(cognitive, on='subject_id', how='outer')
    n_total = len(master)
    n_demo = demographics['subject_id'].nunique()
    n_cog = cognitive['subject_id'].nunique()
    log.info(f"Master table: {n_total} rows "
             f"(demographics={n_demo}, cognitive={n_cog})")
    for test_name in COG_TESTS:
        col = f'log_{test_name}' if COG_TESTS[test_name].get('log_transform') else test_name
        if col in master.columns:
            n_valid = master[col].notna().sum()
            log.info(f"  {test_name}: {n_valid} valid values")
    return master


# ============================================================================
# SECTION B: EEG LOADING
# ============================================================================

def load_preprocessed_subject(subject_id: str,
                              preproc_root: str = LEMON_PREPROC_ROOT,
                              condition: str = 'EO'):
    """Load preprocessed EEGLAB .set file for one subject/condition.

    Returns (mne.io.Raw | None, info_dict).
    """
    import mne
    set_path = os.path.join(preproc_root, f'{subject_id}_{condition}.set')
    if not os.path.exists(set_path):
        return None, {'n_channels': 0, 'duration_sec': 0,
                      'channel_names': [], 'condition': condition}

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)

        n_channels = len(raw.ch_names)

        # Set montage
        montage = mne.channels.make_standard_montage('standard_1005')
        raw.set_montage(montage, on_missing='warn')

        # Add FCz as zero reference channel & re-reference to average
        if 'FCz' not in raw.ch_names:
            raw = mne.add_reference_channels(raw, 'FCz')
        raw.set_eeg_reference('average', projection=False, verbose=False)

    info = {
        'n_channels': n_channels,  # before adding FCz
        'duration_sec': raw.times[-1] if len(raw.times) > 0 else 0,
        'channel_names': list(raw.ch_names),
        'condition': condition,
    }
    return raw, info


def mne_raw_to_records(raw, fs=None) -> pd.DataFrame:
    """Convert MNE Raw → RECORDS DataFrame for detect_ignitions_session()."""
    data = raw.get_data()  # (n_channels, n_samples)
    ch_names = raw.ch_names
    if fs is None:
        fs = raw.info['sfreq']
    n_samples = data.shape[1]

    records = pd.DataFrame({'Timestamp': np.arange(n_samples) / fs})
    for i, ch in enumerate(ch_names):
        col_name = f'EEG.{ch}' if not ch.startswith('EEG.') else ch
        records[col_name] = data[i]
    return records


# ============================================================================
# SECTION C: SPECTRAL PARAMETERIZATION (FOOOF)
# ============================================================================

def _get_peak_params(sm):
    """Get full peak params (CF, power, bandwidth) from SpectralModel.

    Compatible with specparam 1.x and 2.x APIs.
    """
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
        return peaks.reshape(1, -1)[:, :3]
    return peaks[:, :3]


def _get_aperiodic_params(sm):
    """Get aperiodic parameters (offset, exponent) from SpectralModel."""
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
    """Get R² (goodness of fit) from SpectralModel.

    specparam 2.x: sm.get_metrics('gof')
    specparam 1.x: sm.r_squared_
    """
    # specparam 2.x
    try:
        gof = sm.get_metrics('gof')
        if gof is not None and not (hasattr(gof, '__len__') and len(gof) == 0):
            return float(gof)
    except Exception:
        pass
    # specparam 1.x
    try:
        return float(sm.r_squared_)
    except AttributeError:
        pass
    return np.nan


def extract_fooof_peaks_subject(raw_clean, fs=SFREQ,
                                fooof_params=None,
                                nperseg=WELCH_NPERSEG,
                                overlap=0.5,
                                channel_r2_min=FOOOF_CHANNEL_R2_MIN):
    """Per-channel FOOOF fitting for one subject.

    Returns (peaks_df, channel_info).
    """
    if fooof_params is None:
        fooof_params = FOOOF_PARAMS.copy()
    if not _SPECPARAM:
        raise ImportError("Neither specparam nor fooof is installed")

    freq_range = fooof_params['freq_range']
    all_peaks = []
    channel_r2s = []
    channel_n_peaks = []
    channel_aperiodic_exps = []
    alpha_powers = []
    n_fitted = 0
    n_passed = 0

    ch_names = [ch for ch in raw_clean.ch_names if ch != 'FCz']
    noverlap = int(nperseg * overlap)

    for ch in ch_names:
        try:
            data = raw_clean.get_data(picks=[ch])[0]
        except Exception:
            continue

        if len(data) < nperseg:
            continue

        freqs, psd = welch(data, fs, nperseg=nperseg, noverlap=noverlap)

        if len(freqs) < 10:
            continue

        n_fitted += 1
        sm = SpectralModel(**{k: v for k, v in fooof_params.items()
                              if k != 'freq_range'})
        try:
            sm.fit(freqs, psd, freq_range)
        except Exception as e:
            log.debug(f"FOOOF failed on {ch}: {e}")
            continue

        r2 = _get_r_squared(sm)
        if np.isnan(r2) or r2 < channel_r2_min:
            continue

        n_passed += 1
        channel_r2s.append(r2)

        # Aperiodic
        offset, exponent = _get_aperiodic_params(sm)
        channel_aperiodic_exps.append(exponent)

        # Peaks
        peak_params = _get_peak_params(sm)
        for row in peak_params:
            all_peaks.append({
                'channel': ch,
                'freq': row[0],
                'power': row[1],
                'bandwidth': row[2],
            })
        channel_n_peaks.append(len(peak_params))

        # Alpha power (8-13 Hz, from raw PSD, log scale)
        alpha_mask = (freqs >= 8) & (freqs <= 13)
        if alpha_mask.any():
            alpha_powers.append(np.log10(np.mean(psd[alpha_mask])))

    peaks_df = pd.DataFrame(all_peaks) if all_peaks else pd.DataFrame(
        columns=['channel', 'freq', 'power', 'bandwidth'])

    channel_info = {
        'n_channels_fitted': n_fitted,
        'n_channels_passed': n_passed,
        'mean_r_squared': np.mean(channel_r2s) if channel_r2s else np.nan,
        'mean_n_peaks': np.mean(channel_n_peaks) if channel_n_peaks else 0.0,
        'total_peak_count': len(all_peaks),
        'mean_aperiodic_exponent': (np.mean(channel_aperiodic_exps)
                                    if channel_aperiodic_exps else np.nan),
        'alpha_power_eo': (np.mean(alpha_powers) if alpha_powers else np.nan),
    }
    return peaks_df, channel_info


def compute_iaf_from_raw(raw_ec, fs=SFREQ) -> dict:
    """Center-of-gravity IAF from eyes-closed data, posterior ROI."""
    if raw_ec is None:
        return {'iaf': np.nan, 'alpha_power_ec': np.nan}

    ch_names = raw_ec.ch_names
    roi_channels = [ch for ch in IAF_POSTERIOR_ROI if ch in ch_names]
    if not roi_channels:
        return {'iaf': np.nan, 'alpha_power_ec': np.nan}

    iaf_values = []
    alpha_powers = []
    nperseg = min(WELCH_NPERSEG, len(raw_ec.times))

    for ch in roi_channels:
        try:
            data = raw_ec.get_data(picks=[ch])[0]
        except Exception:
            continue
        if len(data) < nperseg:
            continue

        freqs, psd = welch(data, fs, nperseg=nperseg,
                           noverlap=int(nperseg * 0.5))
        # Center of gravity in 7.5-13.5 Hz
        alpha_mask = (freqs >= 7.5) & (freqs <= 13.5)
        if not alpha_mask.any():
            continue
        f_a = freqs[alpha_mask]
        p_a = psd[alpha_mask]
        total_power = p_a.sum()
        if total_power > 0:
            cog = np.sum(f_a * p_a) / total_power
            iaf_values.append(cog)

        # Alpha power (8-13 Hz, log scale)
        ap_mask = (freqs >= 8) & (freqs <= 13)
        if ap_mask.any():
            alpha_powers.append(np.log10(np.mean(psd[ap_mask])))

    return {
        'iaf': np.mean(iaf_values) if iaf_values else np.nan,
        'alpha_power_ec': np.mean(alpha_powers) if alpha_powers else np.nan,
    }


# ============================================================================
# SECTION D: COMPLIANCE SCORING
# ============================================================================

def make_phi_bands(f0=F0_PRIMARY, freq_ceil=45.0) -> dict:
    """Phi-octave band boundaries, gamma capped at freq_ceil."""
    return {
        'theta':     (f0 / PHI, f0),
        'alpha':     (f0, f0 * PHI),
        'beta_low':  (f0 * PHI, f0 * PHI ** 2),
        'beta_high': (f0 * PHI ** 2, f0 * PHI ** 3),
        'gamma':     (f0 * PHI ** 3, freq_ceil),
    }


def compute_compliance_score(peak_freqs, f0=F0_PRIMARY,
                             window=COMPLIANCE_WINDOW,
                             freq_ceil=45.0) -> dict:
    """Compute phi-lattice compliance score from peak frequencies.

    Uses natural_positions(PHI) and compute_structural_score() from
    structural_phi_specificity.py — verified to match pre-reg formula:
      SS = -E_boundary + E_attractor + mean(E_nobles)
    """
    peak_freqs = np.asarray(peak_freqs, dtype=float)
    peak_freqs = peak_freqs[peak_freqs > 0]

    if len(peak_freqs) == 0:
        return {'compliance': np.nan, 'n_peaks': 0,
                'E_boundary': np.nan, 'E_noble_2': np.nan,
                'E_attractor': np.nan, 'E_noble_1': np.nan,
                'lattice_coords': np.array([])}

    # Lattice coordinates
    u = lattice_coordinate(peak_freqs, f0, PHI)
    u_valid = u[np.isfinite(u)]

    if len(u_valid) == 0:
        return {'compliance': np.nan, 'n_peaks': len(peak_freqs),
                'E_boundary': np.nan, 'E_noble_2': np.nan,
                'E_attractor': np.nan, 'E_noble_1': np.nan,
                'lattice_coords': np.array([])}

    # Natural positions for phi
    positions = natural_positions(PHI)
    # positions = {boundary: 0.0, attractor: 0.5, noble: 0.618, noble_2: 0.382}

    # Structural score
    score, enrichments = compute_structural_score(u_valid, positions, window)

    result = {
        'compliance': score,
        'n_peaks': len(peak_freqs),
        'E_boundary': enrichments.get('boundary', np.nan),
        'E_noble_2': enrichments.get('noble_2', np.nan),
        'E_attractor': enrichments.get('attractor', np.nan),
        'E_noble_1': enrichments.get('noble', np.nan),
        'lattice_coords': u_valid,
    }

    # Band-specific compliance
    bands = make_phi_bands(f0, freq_ceil)
    for band_name, (lo, hi) in bands.items():
        band_freqs = peak_freqs[(peak_freqs >= lo) & (peak_freqs < hi)]
        if len(band_freqs) >= 3:
            u_band = lattice_coordinate(band_freqs, f0, PHI)
            u_band = u_band[np.isfinite(u_band)]
            if len(u_band) >= 3:
                band_score, _ = compute_structural_score(
                    u_band, positions, window)
                result[f'compliance_{band_name}'] = band_score
            else:
                result[f'compliance_{band_name}'] = np.nan
        else:
            result[f'compliance_{band_name}'] = np.nan

    return result


def compute_compliance_channel_splits(peaks_df: pd.DataFrame,
                                      f0=F0_PRIMARY,
                                      window=COMPLIANCE_WINDOW) -> dict:
    """Split-half and spatial compliance for ICC computation."""
    if peaks_df.empty or 'channel' not in peaks_df.columns:
        return {
            'compliance_odd': np.nan, 'compliance_even': np.nan,
            'compliance_anterior': np.nan, 'compliance_posterior': np.nan,
        }

    channels = sorted(peaks_df['channel'].unique())
    positions = natural_positions(PHI)

    def _score_channels(ch_list):
        mask = peaks_df['channel'].isin(ch_list)
        freqs = peaks_df.loc[mask, 'freq'].values
        if len(freqs) < 5:
            return np.nan
        u = lattice_coordinate(freqs, f0, PHI)
        u = u[np.isfinite(u)]
        if len(u) < 5:
            return np.nan
        s, _ = compute_structural_score(u, positions, window)
        return s

    # Odd/even split
    odd_chs = [ch for i, ch in enumerate(channels) if i % 2 == 1]
    even_chs = [ch for i, ch in enumerate(channels) if i % 2 == 0]

    # Anterior/posterior
    ant_chs = [ch for ch in channels if ch in ANTERIOR_CHANNELS]
    post_chs = [ch for ch in channels if ch in POSTERIOR_CHANNELS]

    return {
        'compliance_odd': _score_channels(odd_chs),
        'compliance_even': _score_channels(even_chs),
        'compliance_anterior': _score_channels(ant_chs),
        'compliance_posterior': _score_channels(post_chs),
    }


# ============================================================================
# SECTION E: SUBJECT PIPELINE
# ============================================================================

def detect_sie_subject(raw_eo, subject_id: str,
                       output_dir='exports_lemon/per_subject') -> dict:
    """Detect Schumann Ignition Events from eyes-open EEG.

    Uses the unified 9-harmonic pipeline matching Papers 1-2.
    Returns dict with n_ignitions, sie_rate, mean_sie_duration, mean_sr_z_max.
    """
    from detect_ignition import detect_ignitions_session

    # Unified pipeline harmonics (from CLAUDE.md / batch scripts)
    SIE_LABELS  = ['sr1', 'sr1.5', 'sr2', 'sr2o', 'sr2.5', 'sr3', 'sr4', 'sr5', 'sr6']
    SIE_CANON   = [7.6, 10, 12, 13.75, 15.5, 20, 25, 32, 40]
    SIE_HALF_BW = [0.6, 0.618, 0.7, 0.75, 0.8, 1, 2, 2.5, 3]

    null_result = {
        'n_ignitions': 0, 'sie_rate': np.nan, 'duration_min': np.nan,
        'mean_sie_duration': 0.0, 'mean_sr_z_max': np.nan,
        'sie_gap_warning': False,
    }

    records_df = mne_raw_to_records(raw_eo)
    fs = raw_eo.info['sfreq']

    # Temporal discontinuity check
    timestamps = records_df['Timestamp'].values
    dt = np.diff(timestamps)
    expected_dt = 1.0 / fs
    gaps = np.where(dt > 1.5 * expected_dt)[0]
    total_gap_duration = 0.0
    total_duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
    if len(gaps) > 0:
        total_gap_duration = np.sum(dt[gaps] - expected_dt)
        log.warning(f"{subject_id}: {len(gaps)} temporal gaps, "
                    f"total gap = {total_gap_duration:.1f}s "
                    f"({100 * total_gap_duration / total_duration:.1f}% "
                    f"of recording)")
        if total_gap_duration > 0.10 * total_duration:
            log.warning(f"{subject_id}: >10% gaps, SIE rate set to NaN")
            null_result['sie_gap_warning'] = True
            null_result['duration_min'] = total_duration / 60
            return null_result

    # Find SR channel
    eeg_cols = [c for c in records_df.columns if c.startswith('EEG.')]
    sr_channel = None
    for pref in ['EEG.F4', 'EEG.Fz', 'EEG.F3']:
        if pref in eeg_cols:
            sr_channel = pref
            break
    if sr_channel is None and eeg_cols:
        sr_channel = eeg_cols[0]
    if sr_channel is None:
        return null_result

    try:
        out, ign_windows = detect_ignitions_session(
            records_df,
            sr_channel=sr_channel,
            eeg_channels=eeg_cols,
            center_hz=7.83,
            half_bw_hz=SIE_HALF_BW,
            harmonics_hz=SIE_CANON,
            labels=SIE_LABELS,
            harmonic_method='psd',
            z_thresh=2.5,
            nperseg_sec=4.0,
            make_passport=False,
            show=False,
            verbose=False,
            out_dir=output_dir,
            session_name=subject_id,
        )
    except Exception as e:
        log.error(f"{subject_id}: SIE detection failed: {e}")
        return null_result

    n_ignitions = len(ign_windows) if ign_windows else 0
    duration_min = total_duration / 60 if total_duration > 0 else np.nan

    result = {
        'n_ignitions': n_ignitions,
        'sie_rate': n_ignitions / duration_min if duration_min and duration_min > 0 else np.nan,
        'duration_min': duration_min,
        'mean_sie_duration': 0.0,
        'mean_sr_z_max': np.nan,
        'sie_gap_warning': len(gaps) > 0,
    }

    if n_ignitions > 0 and isinstance(out, dict):
        events = out.get('events')
        if events is not None and hasattr(events, '__len__') and len(events) > 0:
            if isinstance(events, pd.DataFrame):
                if 'duration_s' in events.columns:
                    result['mean_sie_duration'] = events['duration_s'].mean()
                if 'sr1_z_max' in events.columns:
                    result['mean_sr_z_max'] = events['sr1_z_max'].mean()

    return result


def process_single_subject(subject_id: str,
                           preproc_root: str = LEMON_PREPROC_ROOT,
                           f0: float = F0_PRIMARY,
                           detect_sie: bool = True,
                           output_dir: str = 'exports_lemon/per_subject',
                           ) -> Optional[dict]:
    """Full per-subject pipeline: load → FOOOF → compliance → SIE → features.

    Returns feature dict, or None if EO data missing.
    """
    # 1. Load EO
    raw_eo, info_eo = load_preprocessed_subject(
        subject_id, preproc_root, condition='EO')
    if raw_eo is None:
        log.warning(f"{subject_id}: EO file not found, skipping")
        return None

    # 2. Load EC (may be None)
    raw_ec, info_ec = load_preprocessed_subject(
        subject_id, preproc_root, condition='EC')

    n_channels_loaded = info_eo['n_channels']

    # 3. FOOOF
    peaks_df, chan_info = extract_fooof_peaks_subject(raw_eo, fs=SFREQ)

    # 4. Exclusion checks
    excluded = False
    exclusion_reason = None
    n_passed = chan_info['n_channels_passed']

    if n_channels_loaded > 0:
        survival_frac = n_passed / n_channels_loaded
    else:
        survival_frac = 0.0

    if survival_frac < MIN_CHANNEL_SURVIVAL_FRAC:
        excluded = True
        exclusion_reason = (f"channel survival {survival_frac:.2f} "
                            f"< {MIN_CHANNEL_SURVIVAL_FRAC}")
    elif chan_info['mean_r_squared'] < MIN_MEAN_FOOOF_R2:
        excluded = True
        exclusion_reason = (f"mean R²={chan_info['mean_r_squared']:.3f} "
                            f"< {MIN_MEAN_FOOOF_R2}")
    elif chan_info['mean_n_peaks'] < MIN_MEAN_PEAKS_PER_CHANNEL:
        excluded = True
        exclusion_reason = (f"mean peaks/ch={chan_info['mean_n_peaks']:.1f} "
                            f"< {MIN_MEAN_PEAKS_PER_CHANNEL}")

    # 5. Compliance scoring (compute even if excluded, for diagnostics)
    compliance = compute_compliance_score(
        peaks_df['freq'].values if not peaks_df.empty else np.array([]),
        f0=f0)

    # 6. Channel splits
    splits = compute_compliance_channel_splits(peaks_df, f0=f0)

    # 7. IAF
    iaf_result = compute_iaf_from_raw(raw_ec)

    # 8. SIE detection
    sie_result = {'n_ignitions': 0, 'sie_rate': np.nan,
                  'duration_min': np.nan, 'mean_sie_duration': 0.0,
                  'mean_sr_z_max': np.nan, 'sie_gap_warning': False}
    if detect_sie and not excluded:
        try:
            sie_result = detect_sie_subject(raw_eo, subject_id, output_dir)
        except Exception as e:
            log.error(f"{subject_id}: SIE failed: {e}")

    # 9. Save peaks CSV
    if not peaks_df.empty:
        os.makedirs(output_dir, exist_ok=True)
        peaks_path = os.path.join(output_dir, f'{subject_id}_peaks.csv')
        peaks_df.to_csv(peaks_path, index=False)

    # 10. Cleanup
    del raw_eo, raw_ec
    gc.collect()

    # 11. Return features
    features = {
        'subject_id': subject_id,
        'n_channels_loaded': n_channels_loaded,
        'n_channels_fooof_passed': n_passed,
        'n_peaks': chan_info['total_peak_count'],
        'mean_r_squared': chan_info['mean_r_squared'],
        'mean_aperiodic_exponent': chan_info['mean_aperiodic_exponent'],
        'duration_sec': info_eo['duration_sec'],
        # Compliance
        'compliance': compliance['compliance'],
        'compliance_theta': compliance.get('compliance_theta', np.nan),
        'compliance_alpha': compliance.get('compliance_alpha', np.nan),
        'compliance_beta_low': compliance.get('compliance_beta_low', np.nan),
        'compliance_beta_high': compliance.get('compliance_beta_high', np.nan),
        'compliance_gamma': compliance.get('compliance_gamma', np.nan),
        # Position enrichments
        'E_boundary': compliance['E_boundary'],
        'E_noble_2': compliance['E_noble_2'],
        'E_attractor': compliance['E_attractor'],
        'E_noble_1': compliance['E_noble_1'],
        # Channel splits
        'compliance_odd': splits['compliance_odd'],
        'compliance_even': splits['compliance_even'],
        'compliance_anterior': splits['compliance_anterior'],
        'compliance_posterior': splits['compliance_posterior'],
        # IAF + alpha power
        'iaf': iaf_result['iaf'],
        'alpha_power_eo': chan_info['alpha_power_eo'],
        'alpha_power_ec': iaf_result['alpha_power_ec'],
        # SIE
        'n_ignitions': sie_result['n_ignitions'],
        'sie_rate': sie_result['sie_rate'],
        'mean_sie_duration': sie_result['mean_sie_duration'],
        'mean_sr_z_max': sie_result['mean_sr_z_max'],
        'sie_gap_warning': sie_result.get('sie_gap_warning', False),
        # Exclusion
        'excluded': excluded,
        'exclusion_reason': exclusion_reason,
    }
    return features


# ============================================================================
# SECTION F: STATISTICAL FUNCTIONS
# ============================================================================

def fdr_correct(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR correction. Returns (adjusted_p, reject_mask)."""
    from statsmodels.stats.multitest import multipletests
    p_arr = np.asarray(p_values, dtype=float)
    # Handle NaN
    valid = np.isfinite(p_arr)
    if not valid.any():
        return p_arr.copy(), np.zeros_like(p_arr, dtype=bool)
    reject = np.zeros(len(p_arr), dtype=bool)
    adjusted = np.full(len(p_arr), np.nan)
    r, adj, _, _ = multipletests(p_arr[valid], alpha=alpha, method='fdr_bh')
    reject[valid] = r
    adjusted[valid] = adj
    return adjusted, reject


def compute_icc_2way(x, y) -> Tuple[float, float, float]:
    """Two-way random, absolute agreement ICC(2,1) with 95% CI.

    Parameters: x, y = two measurements per subject (arrays of same length).
    Returns: (icc, ci_lo, ci_hi).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Drop pairs where either is NaN
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan

    k = 2  # two raters/measurements
    data = np.column_stack([x, y])
    grand_mean = data.mean()

    # Subject means and rater means
    subj_means = data.mean(axis=1)
    rater_means = data.mean(axis=0)

    # Sums of squares
    SS_total = np.sum((data - grand_mean) ** 2)
    SS_between = k * np.sum((subj_means - grand_mean) ** 2)
    SS_within = SS_total - SS_between
    SS_raters = n * np.sum((rater_means - grand_mean) ** 2)
    SS_error = SS_within - SS_raters

    # Mean squares
    MS_between = SS_between / (n - 1)
    MS_raters = SS_raters / (k - 1) if k > 1 else 0
    MS_error = SS_error / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else 1e-10

    # ICC(2,1): two-way random, absolute agreement
    icc = (MS_between - MS_error) / (
        MS_between + (k - 1) * MS_error +
        k * (MS_raters - MS_error) / n
    )

    # Confidence interval via F-distribution
    F_value = MS_between / MS_error if MS_error > 0 else np.inf
    df1 = n - 1
    df2 = (n - 1) * (k - 1)

    if F_value <= 0 or not np.isfinite(F_value):
        return icc, np.nan, np.nan

    F_lo = F_value / stats.f.ppf(0.975, df1, df2)
    F_hi = F_value / stats.f.ppf(0.025, df1, df2)

    ci_lo = (F_lo - 1) / (F_lo + k - 1)
    ci_hi = (F_hi - 1) / (F_hi + k - 1)

    return float(icc), float(ci_lo), float(ci_hi)


def hierarchical_regression(y, X_base, X_full, names_base, names_full):
    """Hierarchical OLS regression: Model 1 (base) vs Model 2 (full).

    Returns dict with R², delta-R², LRT, compliance coefficients, VIFs.
    """
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    y = np.asarray(y, dtype=float)
    X_base = np.asarray(X_base, dtype=float)
    X_full = np.asarray(X_full, dtype=float)

    # Add constant
    X1 = sm.add_constant(X_base)
    X2 = sm.add_constant(X_full)

    # Fit models
    model1 = sm.OLS(y, X1, missing='drop').fit()
    model2 = sm.OLS(y, X2, missing='drop').fit()

    # LRT
    ll1 = model1.llf
    ll2 = model2.llf
    lrt_chi2 = -2 * (ll1 - ll2)
    lrt_p = 1 - stats.chi2.cdf(lrt_chi2, df=1)

    # Compliance is the last predictor in Model 2
    compliance_idx = -1
    compliance_beta = model2.params[compliance_idx]
    compliance_se = model2.bse[compliance_idx]
    compliance_t = model2.tvalues[compliance_idx]
    compliance_p = model2.pvalues[compliance_idx]

    # VIFs for Model 2 (skip constant at index 0)
    vifs = {}
    for i in range(1, X2.shape[1]):
        name = names_full[i - 1] if i - 1 < len(names_full) else f'x{i}'
        try:
            vif = variance_inflation_factor(X2, i)
        except Exception:
            vif = np.nan
        vifs[name] = vif

    # Residual normality
    resid_shapiro_p = stats.shapiro(model2.resid)[1] if len(model2.resid) > 3 else np.nan

    # If non-normal residuals, refit with HC3
    use_robust = resid_shapiro_p < 0.05 if np.isfinite(resid_shapiro_p) else False
    if use_robust:
        model2_robust = sm.OLS(y, X2, missing='drop').fit(cov_type='HC3')
        compliance_se = model2_robust.bse[compliance_idx]
        compliance_t = model2_robust.tvalues[compliance_idx]
        compliance_p = model2_robust.pvalues[compliance_idx]

    return {
        'R2_m1': model1.rsquared,
        'R2_m2': model2.rsquared,
        'delta_R2': model2.rsquared - model1.rsquared,
        'adj_R2_m1': model1.rsquared_adj,
        'adj_R2_m2': model2.rsquared_adj,
        'lrt_chi2': lrt_chi2,
        'lrt_p': lrt_p,
        'compliance_beta': compliance_beta,
        'compliance_se': compliance_se,
        'compliance_t': compliance_t,
        'compliance_p': compliance_p,
        'vifs': vifs,
        'n': len(y),
        'resid_shapiro_p': resid_shapiro_p,
        'used_robust_se': use_robust,
        'model1_summary': str(model1.summary()),
        'model2_summary': str(model2.summary()),
    }


def bootstrap_delta_r2_ci(y, X_base, X_full, n_boot=5000, seed=42):
    """Bootstrap 95% CI for delta-R² between two nested OLS models."""
    import statsmodels.api as sm

    y = np.asarray(y, dtype=float)
    X_base = np.asarray(X_base, dtype=float)
    X_full = np.asarray(X_full, dtype=float)
    n = len(y)
    rng = np.random.default_rng(seed)

    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        X1_b = sm.add_constant(X_base[idx])
        X2_b = sm.add_constant(X_full[idx])
        try:
            r2_1 = sm.OLS(y_b, X1_b).fit().rsquared
            r2_2 = sm.OLS(y_b, X2_b).fit().rsquared
            deltas.append(r2_2 - r2_1)
        except Exception:
            continue

    if not deltas:
        return np.nan, np.nan
    deltas = np.array(deltas)
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def bootstrap_mediation(X, M, Y, covariates=None, n_boot=5000, seed=42) -> dict:
    """Preacher & Hayes bootstrap mediation (Age→Compliance→Cognition).

    X = independent (age), M = mediator (compliance), Y = outcome (cognition).
    covariates = optional confounders (sex, education).
    """
    import statsmodels.api as sm

    X = np.asarray(X, dtype=float).ravel()
    M = np.asarray(M, dtype=float).ravel()
    Y = np.asarray(Y, dtype=float).ravel()

    # Build design matrices
    if covariates is not None:
        C = np.asarray(covariates, dtype=float)
        if C.ndim == 1:
            C = C.reshape(-1, 1)
        XC = np.column_stack([X, C])
        XMC = np.column_stack([X, M, C])
    else:
        XC = X.reshape(-1, 1)
        XMC = np.column_stack([X, M])

    n = len(X)

    # Observed paths
    XC_c = sm.add_constant(XC)
    XMC_c = sm.add_constant(XMC)

    model_a = sm.OLS(M, XC_c, missing='drop').fit()
    model_b = sm.OLS(Y, XMC_c, missing='drop').fit()

    a_coef = model_a.params[1]   # X → M
    b_coef = model_b.params[2]   # M → Y (controlling for X)
    direct = model_b.params[1]   # X → Y (controlling for M)
    indirect = a_coef * b_coef

    # Total effect
    model_total = sm.OLS(Y, XC_c, missing='drop').fit()
    total = model_total.params[1]

    # Bootstrap
    rng = np.random.default_rng(seed)
    boot_indirect = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            ma = sm.OLS(M[idx], sm.add_constant(XC[idx])).fit()
            mb = sm.OLS(Y[idx], sm.add_constant(XMC[idx])).fit()
            boot_indirect.append(ma.params[1] * mb.params[2])
        except Exception:
            continue

    boot_arr = np.array(boot_indirect) if boot_indirect else np.array([np.nan])

    prop_mediated = indirect / total if abs(total) > 1e-10 else np.nan

    return {
        'a_coef': a_coef,
        'b_coef': b_coef,
        'indirect_effect': indirect,
        'indirect_ci_lo': float(np.percentile(boot_arr, 2.5)),
        'indirect_ci_hi': float(np.percentile(boot_arr, 97.5)),
        'direct_effect': direct,
        'total_effect': total,
        'proportion_mediated': prop_mediated,
        'n_boot': len(boot_indirect),
    }


def run_group_comparison(young, elderly) -> dict:
    """Independent t-test + Mann-Whitney U + Cohen's d."""
    young = np.asarray(young, dtype=float)
    elderly = np.asarray(elderly, dtype=float)
    young = young[np.isfinite(young)]
    elderly = elderly[np.isfinite(elderly)]

    if len(young) < 2 or len(elderly) < 2:
        return {'t_stat': np.nan, 'p_ttest': np.nan,
                'U_stat': np.nan, 'p_mannwhitney': np.nan,
                'cohens_d': np.nan, 'mean_young': np.nan,
                'mean_elderly': np.nan, 'n_young': len(young),
                'n_elderly': len(elderly)}

    t_stat, p_t = stats.ttest_ind(young, elderly, equal_var=False)
    U_stat, p_u = stats.mannwhitneyu(young, elderly, alternative='two-sided')

    # Cohen's d (pooled SD)
    n1, n2 = len(young), len(elderly)
    s1, s2 = young.std(ddof=1), elderly.std(ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    cohens_d = (young.mean() - elderly.mean()) / pooled_sd if pooled_sd > 0 else np.nan

    return {
        't_stat': t_stat, 'p_ttest': p_t,
        'U_stat': U_stat, 'p_mannwhitney': p_u,
        'cohens_d': cohens_d,
        'mean_young': young.mean(), 'mean_elderly': elderly.mean(),
        'sd_young': s1, 'sd_elderly': s2,
        'n_young': n1, 'n_elderly': n2,
    }

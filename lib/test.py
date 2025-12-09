from dataclasses import asdict
from dataclasses import dataclass


import numpy as np
import pandas as pd
import re

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap

from typing import Any, Dict, Optional, Sequence, Tuple, List, Iterable, Union
from scipy.signal import stft, firwin, filtfilt, detrend, savgol_filter, hilbert, stft, welch, coherence

from detect_ignition import _build_virtual_sr

import matplotlib as mpl
mpl.rcParams.update({
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
})


LIFEAQUATIC_SPEC_CMAP = ListedColormap([
   '#03588c', '#048abf', '#5bbad6', '#f4d35e', '#ee964b', '#f95738'
], name='life_aquatic_spec')

LIFEAQUATIC_COLORS = {
    'fundamental': '#f95738',
    'harm2': '#f4d35e',
    'harm3': '#48beff',
    'plv': '#247ba0',
    'plv_mean': '#1b4d6e',
    'delta_hsi': '#ee6c4d',
    'bic1': '#ff9f1c',
    'bic2': '#2ec4b6',
    'pac': '#006d77',
    'baseline_line': '#6c757d',
    'window_fill': "#d0d0d0",
    'off_harmonic': '#9aa7b1',
    'tick': '#0e1f33',
    'spine': '#1a3f5c',
    'panel_bg': '#ffffff'
}

MATRIX_SPEC_CMAP = ListedColormap([
    '#010b0a', '#022d1a', '#054321', '#0b5c29', '#128b3a', '#1ecf55', '#77ff9c'
], name='matrix_spec')

MATRIX_COLORS = {
    'fundamental': '#00ff7f',
    'harm2': '#00d084',
    'harm3': '#00a36c',
    'plv': '#5cf2c7',
    'plv_mean': '#2aa982',
    'delta_hsi': '#12c259',
    'bic1': '#17f9ff',
    'bic2': '#0ab1b7',
    'pac': '#19d98c',
    'baseline_line': '#1f382f',
    'window_fill': '#0c261b',
    'off_harmonic': '#1a4d37'
}

SPIRITED_SPEC_CMAP = ListedColormap([
   '#274c77', '#5c7ea5', '#b5c9d6', "#e8d587", "#dca855", '#e07a5f'
], name='spirited_spec')

SPIRITED_COLORS = {
    'fundamental': '#e07a5f',
    'harm2': '#f2cc8f',
    'harm3': '#81b29a',
    'plv': '#3d405b',
    'plv_mean': '#253047',
    'delta_hsi': '#c8553d',
    'bic1': '#f4a261',
    'bic2': '#2a9d8f',
    'pac': '#457b9d',
    'baseline_line': '#6d6875',
    'window_fill': '#f7ede2',
    'off_harmonic': '#9a8c98',
    'tick': '#3d405b',
    'spine': '#b56576',
    'panel_bg': '#ffffff'
}

CYBERPUNK_SPEC_CMAP = ListedColormap([
    '#04020c', '#0b1026', '#1a1f3b', '#311b5b', '#5b1a82', '#a100f2', '#ff00c8'
], name='cyberpunk_spec')

CYBERPUNK_COLORS = {
    'fundamental': '#ff00e5',
    'harm2': '#00dbff',
    'harm3': '#ffe066',
    'plv': '#7df9ff',
    'plv_mean': '#4cc9f0',
    'delta_hsi': '#ff4d8d',
    'bic1': '#ff74ff',
    'bic2': '#26ffc9',
    'pac': '#bc6ff1',
    'baseline_line': '#374151',
    'window_fill': '#1b103d',
    'off_harmonic': '#6d78ff',
    'tick': '#f5f5ff',
    'spine': '#2a2f4f',
    'panel_bg': '#0a0514'  # Dark purple-black background
}

GRAND_SPEC_CMAP = ListedColormap([
      '#9f8bc2','#c48bad',  '#e7a8b6','#f7c6c7','#d8e9f0',"#96c7e8","#6da6cd", 
], name='grand_budapest_spec')

GRAND_COLORS = {
    'fundamental': '#d85a7f',
    'harm2': '#f0c987',
    'harm3': '#98c5dd',
    'plv': '#6f5a8d',
    'plv_mean': '#4f3558',
    'delta_hsi': '#c774a6',
    'bic1': '#f9a0b7',
    'bic2': '#8ec5b3',
    'pac': '#7a8cb8',
    'baseline_line': '#c9b6c6',
    'window_fill': '#fbe9dd',
    'off_harmonic': '#b8cbe6',
    'tick': '#4f3558',
    'spine': '#ccb3c8',
    'panel_bg': "#ffffff"
}

BLADE_SPEC_CMAP = ListedColormap([
     '#1b2335', '#473a67', '#6b559c', "#ccb6fc", "#fc88c0", '#f45ca6'
], name='blade_runner_spec')

BLADE_COLORS = {
    'fundamental': '#f45ca6',
    'harm2': '#6fc0c9',
    'harm3': '#94bcd2',
    'plv': '#4c6eb6',
    'plv_mean': '#2d3f63',
    'delta_hsi': '#c25c9f',
    'bic1': '#ff8ac0',
    'bic2': '#54d8c7',
    'pac': '#7fa7d9',
    'baseline_line': '#3c4d68',
    'window_fill': '#1a2335',
    'off_harmonic': '#7a6ca6',
    'tick': '#d4e4f0',
    'spine': '#2a324a',
    'panel_bg': '#101421'
}

MADMAX_SPEC_CMAP = ListedColormap([
     '#0f141a', '#003a40',  '#005f6b', '#db6a05', '#b88b73',  '#fbd0b2', 
], name='mad_max_spec')

MADMAX_COLORS = {
    'fundamental': '#ff8c42',
    'harm2': '#f0c29a',
    'harm3': '#5f8f94',
    'plv': '#1f5966',
    'plv_mean': '#15414b',
    'delta_hsi': '#d96a29',
    'bic1': '#f0a35c',
    'bic2': '#5aa0a4',
    'pac': '#c76a3b',
    'baseline_line': '#4e4842',
    'window_fill': '#22150f',
    'off_harmonic': '#7c5c40',
    'tick': '#f0ead6',
    'spine': '#3a2a1f',
    'panel_bg': '#16100c'
}

SUNSET_SPEC_CMAP = ListedColormap([
    '#2d1b3d', '#512f5b', '#7a3f5c', '#b2554e', '#e17335', '#f29f4c', '#f9c784', '#ffd1a9'
], name='sunset_spec')

SUNSET_COLORS = {
    'fundamental': '#ff7a59',
    'harm2': '#f8c16d',
    'harm3': '#6f9daf',
    'plv': '#406d8a',
    'plv_mean': '#2a4c5e',
    'delta_hsi': '#e3625b',
    'bic1': '#f5975c',
    'bic2': '#4fb3c0',
    'pac': '#9a6db5',
    'baseline_line': '#5e4a5f',
    'window_fill': '#2d1b3d',
    'off_harmonic': '#c77966',
    'tick': '#fde9d9',
    'spine': '#5b394a',
    'panel_bg': '#1f1328'
}

MATRIX_SPEC_CMAP = ListedColormap([
    '#010403', '#082016', '#124a2c', '#1c733b', '#29a64c', '#48da6e', '#ffeb8a'
], name='matrix_spec')

MATRIX_COLORS = {
    'fundamental': '#48da6e',
    'harm2': '#29a64c',
    'harm3': '#1c733b',
    'plv': '#6fffb3',
    'plv_mean': '#33d48a',
    'delta_hsi': '#65ff91',
    'bic1': '#72ffc7',
    'bic2': '#33efc9',
    'pac': '#ffd96f',
    'baseline_line': '#1d3627',
    'window_fill': '#010807',
    'off_harmonic': '#308d5a',
    'tick': '#dfffe6',
    'spine': '#15311f',
    'panel_bg': '#050a09'
}

STARNIGHT_SPEC_CMAP = ListedColormap([
    '#05020a', '#0b1230', '#17204f', '#23336f', '#30518c', '#4073ab', '#5a94c3', '#86afd4'
], name='star_night_spec')

STARNIGHT_COLORS = {
    'fundamental': '#8fa4ff',
    'harm2': '#5fd4ff',
    'harm3': '#b084ff',
    'plv': '#4d5d9c',
    'plv_mean': '#303f6f',
    'delta_hsi': '#a36ef0',
    'bic1': '#70a8ff',
    'bic2': '#5be3d8',
    'pac': '#4fe8a1',
    'baseline_line': '#3b4766',
    'window_fill': '#05020a',
    'off_harmonic': '#6675b3',
    'tick': '#dfe7ff',
    'spine': '#1f2740',
    'panel_bg': '#0b1021'
}

SUNRISE_SPEC_CMAP = ListedColormap([
    '#1d123a', '#402263', '#703376', '#a9476f', '#e0635a', '#f29b61', '#ffe08a', '#fff3c6'
], name='sunrise_spec')

SUNRISE_COLORS = {
    'fundamental': '#ff8e62',
    'harm2': '#ffd277',
    'harm3': '#7fb9e6',
    'plv': '#4b6c8f',
    'plv_mean': '#2f4863',
    'delta_hsi': '#d96fa3',
    'bic1': '#ffa94d',
    'bic2': '#60d4c8',
    'pac': '#f3a6a2',
    'baseline_line': '#705872',
    'window_fill': '#2b1a49',
    'off_harmonic': '#9d6ebb',
    'tick': '#fff4e4',
    'spine': '#4f3c6a',
    'panel_bg': '#1b1535'
}

FALL_SPEC_CMAP = ListedColormap([
      "#4aaeff",'#8fd1ff', '#cce8ff', '#f9b650',  '#e35b2f', '#bf2633',
], name='fall_foliage_spec')

FALL_COLORS = {
    'fundamental': '#f26a1b',
    'harm2': '#ffba4a',
    'harm3': '#d54a2a',
    'plv': '#5ab0f0',
    'plv_mean': '#2f7abf',
    'delta_hsi': '#f16e2f',
    'bic1': '#ffd162',
    'bic2': '#6cbf4b',
    'pac': '#fa709a',
    'baseline_line': '#8a4a2f',
    'window_fill': '#fff2da',
    'off_harmonic': '#d97d35',
    'tick': '#5c3614',
    'spine': '#b8743a',
    'panel_bg': '#ffffff'
}

CLOUD_SPEC_CMAP = ListedColormap([
    '#0c0d16', '#334458', '#6c879a', '#93abc0', '#c1d3e3', '#e9f1fb'
], name='cloud_spec')   

CLOUD_COLORS = {
    'fundamental': '#6c8fbc',
    'harm2': '#93abc0',
    'harm3': '#4d6679',
    'plv': '#8fb1dc',
    'plv_mean': '#5877a3',
    'delta_hsi': '#7ca3cf',
    'bic1': '#b6cce3',
    'bic2': '#5a86b6',
    'pac': '#9cbdf0',
    'baseline_line': '#3a4b60',
    'window_fill': '#f0f4fa',
    'off_harmonic': '#829bb7',
    'tick': '#1f2433',
    'spine': '#4a5d76',
    'panel_bg': '#ffffff'
}

VALENTINE_SPEC_CMAP = ListedColormap([
    '#faedf2', '#f7c7d7', '#f199b5', '#dc6a8c', '#b73a5f', '#88213d', '#5c0f24', '#2c020d'
], name='valentine_spec')

VALENTINE_COLORS = {
    'fundamental': '#d8254f',
    'harm2': '#ff7b9a',
    'harm3': '#8f1a36',
    'plv': '#f29ab4',
    'plv_mean': '#651a2e',
    'delta_hsi': '#c12b52',
    'bic1': '#ffc1d1',
    'bic2': '#7a2b3f',
    'pac': '#ff4c73',
    'baseline_line': '#3d0c1a',
    'window_fill': '#fde1ea',
    'off_harmonic': '#a43752',
    'tick': '#3a0815',
    'spine': '#721e34',
    'panel_bg': '#ffffff'
}

PALETTE_MAP = {
    'sunset': (SUNSET_SPEC_CMAP, SUNSET_COLORS),
    'sunrise': (SUNRISE_SPEC_CMAP, SUNRISE_COLORS),
    'grand': (GRAND_SPEC_CMAP, GRAND_COLORS),
    'star': (STARNIGHT_SPEC_CMAP, STARNIGHT_COLORS),
    'aquatic': (LIFEAQUATIC_SPEC_CMAP, LIFEAQUATIC_COLORS),
    'madmax': (MADMAX_SPEC_CMAP, MADMAX_COLORS),
    'blade': (BLADE_SPEC_CMAP, BLADE_COLORS),
    'matrix': (MATRIX_SPEC_CMAP, MATRIX_COLORS),
    'fall': (FALL_SPEC_CMAP, FALL_COLORS),
    'cloud': (CLOUD_SPEC_CMAP, CLOUD_COLORS),
    'valentine': (VALENTINE_SPEC_CMAP, VALENTINE_COLORS),
    'spirited': (SPIRITED_SPEC_CMAP, SPIRITED_COLORS),
    'cyberpunk': (CYBERPUNK_SPEC_CMAP, CYBERPUNK_COLORS),
}


def _resolve_palette(name: str):
    key = str(name).lower().strip()
    return PALETTE_MAP.get(key, PALETTE_MAP['matrix'])

GRAND_SPEC_CMAP = ListedColormap([
    '#f9dede', '#f7c6c7', '#e7a8b6', '#c48bad', '#9f8bc2', '#87b7d8', '#d8e9f0'
], name='grand_budapest_spec')

GRAND_COLORS = {
    'fundamental': '#d85a7f',
    'harm2': '#f0c987',
    'harm3': '#98c5dd',
    'plv': '#6f5a8d',
    'plv_mean': '#4f3558',
    'delta_hsi': '#c774a6',
    'bic1': '#f9a0b7',
    'bic2': '#8ec5b3',
    'pac': '#7a8cb8',
    'baseline_line': '#c9b6c6',
    'window_fill': '#fbe9dd',
    'off_harmonic': '#b8cbe6',
    'tick': '#4f3558',
    'spine': '#ccb3c8',
    'panel_bg': '#fff8f3'
}


@dataclass
class FeaturePackCfg:
    time_col: str = 'Timestamp'
    channels: Optional[Sequence[str]] = None  # or 'auto'
    fs: Optional[float] = None
    win_sec: float = 4.0
    step_sec: float = 0.1
    spec_win: float = 1.5
    spec_ovl: float = 0.8
    sr_centers: Tuple[float,float,float] = (7.83, 14.3, 20.8)
    bw_hz: Union[float, Sequence[float]] = 0.5  # Can be scalar or array (one per ladder frequency)
    ladder: Tuple[float,...] = (7.83, 14.3, 20.8, 27.3, 33.8, 40.3, 46.8, 53.3)
    ladder_bw: float = 0.6

@dataclass
class PhaseParams:
    f0: float = 7.83
    # P0
    z_p0: float = 1.0
    plv_p0: float = 0.40
    hsi_broad: float = 0.35
    min_p0_dur: float = 0.25
    # P1
    z_p1: Optional[float] = None
    plv_p1: Optional[float] = None
    z_p1_cap: float = 1.7
    z_p1_min: float = 1.2
    plv_p1_min: float = 0.58
    plv_p1_cap: float = 0.85
    ridge_required: bool = True
    beta_flat: Optional[float] = 1.2
    min_p1_dur: float = 0.35
    z_p1_sigma: float = 2.0
    plv_p1_sigma: float = 1.75
    plv_slope_min: float = 0.003
    plv_slope_window: float = 0.06
    # P2
    hsi_tight: float = 0.30
    rel_h2: float = 0.30
    rel_h3: float = 0.30
    bic_7_7_15: float = 0.10
    bic_7_15_23: float = 0.10
    pac_mvl: Optional[float] = None
    min_p2_cycles: float = 2.0
    p2_score_weights: Tuple[float, float, float, float, float] = (0.04, 0.04, 0.48, 0.44, 0.15)
    p2_score_thresh: float = 0.65
    # P3
    plv_release: float = 0.60
    hsi_release: float = 0.35
    rel_drop_k: int = 2
    plv_release_slope: float = -0.002
    # shared adaptivity
    baseline_span: float = 1.5
    seed_weights: Tuple[float, float, float] = (0.52, 0.24, 0.24)

    def p2_min_dur(self) -> float:
        return self.min_p2_cycles / max(self.f0, 1e-6)

class BaseProvider:
    """Abstract adapter. Subclasses must implement accessors below."""
    def slice(self, t0: float, t1: float) -> 'BaseProvider':
        raise NotImplementedError
    # required
    def t(self) -> np.ndarray: ...
    def z_fund(self) -> np.ndarray: ...
    def z_h2(self) -> np.ndarray: ...
    def z_h3(self) -> np.ndarray: ...
    def plv_fund(self) -> np.ndarray: ...
    def hsi(self) -> np.ndarray: ...
    # optional
    def beta(self) -> Optional[np.ndarray]: return None
    def ridge_is_fund(self) -> Optional[np.ndarray]: return None
    def bic_7_7_15(self) -> Optional[np.ndarray]: return None
    def bic_7_15_23(self) -> Optional[np.ndarray]: return None
    def pac_mvl(self) -> Optional[np.ndarray]: return None
    def spectrogram(self) -> Optional[Tuple[np.ndarray,np.ndarray,np.ndarray]]: return None

class PackProvider(BaseProvider):
    """Wrap a dict-like pack. You can store per-window arrays under top-level keys
    or store a frame of arrays and rely on slicing. Expected names:
      't','z_7p83','z_15p6','z_23p4','plv_7p83','hsi',
      'beta','ridge_is_fund','bic_7_7_15','bic_7_15_23','pac_mvl',
      'spec' -> (t_spec, f_spec, Sxx)
    """
    def __init__(self, pack: Dict[str, Any], sl: slice = slice(None)):
        self.pack = pack
        self.sl = sl
    def _get(self, k: str, default=None):
        v = self.pack.get(k, default)
        if v is None:
            return None
        v = np.asarray(v)
        return v[self.sl] if v.ndim == 1 else v  # spectrogram untouched
    def _get_any(self, *keys):
        for k in keys:
            v = self.pack.get(k, None)
            if v is not None:
                v = np.asarray(v)
                return v[self.sl] if v.ndim == 1 else v
        return None
    # slicing uses time indices from 't'
    def slice(self, t0: float, t1: float) -> 'PackProvider':
        t = np.asarray(self.pack['t'])
        sl = slice(np.searchsorted(t, t0, 'left'), np.searchsorted(t, t1, 'right'))
        return PackProvider(self.pack, sl)
    # required accessors
    def t(self): return self._get('t')
    def z_fund(self): return self._get('z_7p83')
    def z_h2(self): return self._get('z_15p6')
    def z_h3(self): return self._get('z_23p4')
    def plv_fund(self): return self._get('plv_7p83')
    def hsi(self): return self._get('hsi')
    # optional accessors
    def beta(self): return self._get('beta')
    def ridge_is_fund(self): return self._get('ridge_is_fund')
        
    # def bic_7_7_15(self): return self._get('bic_7_7_15')
    # def bic_7_15_23(self): return self._get('bic_7_15_23')
    # def pac_mvl(self): return self._get('pac_mvl')

    def bic_7_7_15(self): return self._get_any('bic_7_7_15', 'bico_7_7_15')
    def bic_7_15_23(self): return self._get_any('bic_7_15_23', 'bico_7_15_23')
    def pac_mvl(self):      return self._get_any('pac_mvl', 'PAC_MVL', 'pac')
        
    def spectrogram(self): return self.pack.get('spec', None)
    def spectrogram_for_window(self, t0, t1):
        spec_by = self.pack.get('spec_by_window')
        if isinstance(spec_by, dict):
            key = (float(t0), float(t1))
            if key in spec_by:
                return spec_by[key]
        return self.pack.get('spec')  # fallback

def _slice_spec_to_window(spec, window, min_cols=20):
    """Return (tW, fW, SW) for the window. If the mask is too small, widen to nearest columns."""
    if spec is None:
        raise ValueError(f"Spectrogram is None - cannot slice to window {window}")
    tS, fS, S = spec
    m = (tS >= window[0]) & (tS <= window[1])
    idx = np.where(m)[0]
    if idx.size < min_cols:
        # widen symmetrically around window center until we have min_cols
        center = 0.5*(window[0] + window[1])
        order = np.argsort(np.abs(tS - center))
        take = order[:max(min_cols, 3)]
        take.sort()
        idx = take
    return tS[idx], fS, S[:, idx]

def window_spec_median(records, window, *, channels, fs, time_col='Timestamp',
                       band=(2,60), win_sec=1.0, overlap=0.80):
    """
    Robust spectrogram inside `window`:
      - combine channels first (median across channels in time domain)
      - remove DC (detrend)
      - STFT with 1.0 s window, 80% overlap (tighter time resolution)
      - return (t_abs, f_band, S_band) in linear power
    """
    # slice samples by time range
    t = np.asarray(records[time_col], float)
    m = (t >= window[0]) & (t <= window[1])
    if not np.any(m):
        raise ValueError("window has no samples")

    # combine channels (force float64)
    X = np.stack([np.asarray(records[c], float) for c in channels], axis=0)[:, m]
    x = np.nanmedian(X, axis=0)
    x = detrend(x, type='constant')                 # remove DC

    nper = int(round(win_sec * fs))
    nover= int(round(overlap * nper)); nover = min(nover, nper-1)
    # f, t_rel, Z = stft(x, fs=fs, window='hann', nperseg=nper, noverlap=nover,
    #                    detrend='constant', boundary=None, padded=False)
    # P = (np.abs(Z) ** 2)                            # (F,T)
    # make the STFT identical to 43
    f, t_rel, Z = stft(x, fs=fs, window='hann',
                    nperseg=int(fs*win_sec), noverlap=int(fs*overlap),
                    detrend='constant', boundary='zeros', padded=True)
    P = np.abs(Z)**2
    P_db = 10*np.log10(P + 1e-12)          # log compression
    # then median across channels in dB, band-limit, and z-row-normalize for display


    # band-limit
    mb = (f >= band[0]) & (f <= band[1])
    fB, PB = f[mb], P[mb, :]
    t_abs = t[m][0] + t_rel                         # absolute seconds

    return t_abs, fB, PB

def _spec_db_rowz(SW):
    """10*log10 then row-wise robust z (median/MAD)."""
    SdB = 10*np.log10(SW + 1e-20)
    med = np.median(SdB, axis=1, keepdims=True)
    mad = np.median(np.abs(SdB - med), axis=1, keepdims=True) + 1e-9
    return (SdB - med)/mad
                    # z units

def patch_pack_with_hsi_v3_for_windows(pack, records, windows, *, eeg_cols, fs, time_col='Timestamp',
                                       in_bw=0.5, ring_offset=1.5, ring_bw=0.8, smooth_hz=6.0,
                                       spec_store_key='spec_by_window'):
    """Compute a per-window spectrogram from time-domain median, derive HSI_v3,
       and write the HSI back into pack['hsi'] for those time slices.
       Also store the per-window spectrogram so Panel A can use it.
    """
    t = np.asarray(pack['t'], float)
    if 'hsi' not in pack or pack['hsi'] is None or not np.isfinite(pack['hsi']).any():
        pack['hsi'] = np.full_like(t, np.nan, dtype=float)
    spec_by_win = pack.setdefault(spec_store_key, {})

    for (t0, t1) in windows:
        # 1) robust per-window spectrogram (time-domain median across channels)
        tW, fW, SW = window_spec_median(records, (t0, t1), channels=eeg_cols, fs=fs, time_col=time_col)
        spec_by_win[(float(t0), float(t1))] = (tW, fW, SW)

        # 2) HSI_v3 from this window's spectrogram
        tH, H = hsi_v3_from_window_spec(tW, fW, SW,
                                        in_bw=in_bw, ring_offset=ring_offset,
                                        ring_bw=ring_bw, smooth_hz=smooth_hz)

        # 3) write H back into the pack slice so provider.hsi() returns it
        m = (t >= t0) & (t <= t1)
        if m.any():
            pack['hsi'][m] = np.interp(t[m], tH, H)

    return pack

def _as_float_1d(x):
    if x is None: 
        return np.array([], dtype=float)
    a = np.asarray(x)
    # spectrogram is a tuple (t,f,S) — don't coerce that here
    if isinstance(x, tuple): 
        return a
    # flatten 1d only (ignore 2d like spectrogram S)
    return a.astype(float).ravel()


def _normalize_channel_label(label: Optional[str]) -> str:
    if not label:
        return ''
    text = str(label).strip().upper().replace(' ', '')
    for prefix in ('EEG.', 'EEG_', 'EEG-'):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    if text.startswith('EEG'):
        text = text[3:]
    return ''.join(ch for ch in text if ch.isalnum())


def _resolve_seed_channel_index(seed_ch: Optional[str], electrodes: Sequence[str]) -> Optional[int]:
    if not seed_ch:
        return None
    norm_seed = _normalize_channel_label(seed_ch)
    if not norm_seed:
        return None
    for idx, ch in enumerate(electrodes):
        if _normalize_channel_label(ch) == norm_seed:
            return idx
    seed_upper = str(seed_ch).upper()
    for idx, ch in enumerate(electrodes):
        if seed_upper in str(ch).upper():
            return idx
    return None


def _match_ignition_event_row(ign_out: Any, ign_win: Tuple[float, float]):
    events = None
    if isinstance(ign_out, dict):
        events = ign_out.get('events')
    elif hasattr(ign_out, 'events'):
        events = getattr(ign_out, 'events')
    if isinstance(events, pd.DataFrame) and not events.empty:
        if 't_start' not in events.columns or 't_end' not in events.columns:
            return None
        t_start, t_end = float(ign_win[0]), float(ign_win[1])
        tol = 1e-3
        mask = (np.abs(events['t_start'] - t_start) <= tol) & (np.abs(events['t_end'] - t_end) <= tol)
        if not mask.any():
            mask = (events['t_start'] <= (t_end + tol)) & (events['t_end'] >= (t_start - tol))
        if mask.any():
            return events.loc[mask].iloc[0]
    return None


def _format_numeric_labels(labels: Sequence[str], decimals: int = 2) -> List[str]:
    if labels is None:
        return []
    fmt = "{:." + str(decimals) + "f}"
    pattern = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

    formatted: List[str] = []
    for label in labels:
        if not isinstance(label, str):
            formatted.append(label)
            continue
        def repl(match):
            try:
                return fmt.format(float(match.group(0)))
            except ValueError:
                return match.group(0)
        formatted.append(pattern.sub(repl, label))
    return formatted


def _extract_seed_channel(ign_out: Any, ign_win: Tuple[float, float]) -> Optional[str]:
    if ign_out is None:
        return None

    candidate = None
    getter = getattr(ign_out, 'get', None)
    if callable(getter):
        for key in ('seed_ch', 'seed_channel'):
            candidate = getter(key, None)
            if candidate:
                return candidate

    row = _match_ignition_event_row(ign_out, ign_win)
    if row is not None:
        for key in ('seed_ch', 'seed_channel'):
            if key in row and pd.notna(row[key]):
                return row[key]
    return candidate

def hsi_from_spec_v2(spec,
                     ladder=(7.83,14.3,20.8,27.3,33.8,40.3,46.8,53.5),
                     win_half_hz=0.6,        # ±0.6 Hz around each harmonic
                     smooth_hz=6.0):         # 1/f background smoothness
    """
    spec = (tS, fS, S) with S shape (F,T), linear power.
    Returns tS, HSI_v2(t) in [0,1], lower = tighter harmonics.
    """
    tS, fS, S = spec
    S = np.asarray(S, float)
    # 1) Flatten 1/f per time using a SavGol smooth on log-power across freq
    df = float(np.median(np.diff(fS)))
    W = max(5, int(np.ceil(smooth_hz/df)))
    if W % 2 == 0: W += 1
    logS = np.log(S + 1e-20)
    bg   = savgol_filter(logS, window_length=W, polyorder=2, axis=0, mode='interp')
    R    = np.exp(logS - bg)                          # ratio > 1 where peaks exceed 1/f

    # 2) Binary mask M(f) that marks ±win_half_hz around each ladder line
    M = np.zeros_like(fS, float)
    for hk in ladder:
        M += (np.abs(fS - hk) <= win_half_hz).astype(float)

    # 3) Concentration of *excess* power on the ladder
    num = (R * M[:, None]).sum(axis=0)
    den = R.sum(axis=0) + 1e-12
    C   = num / den                                   # 0..1
    H   = 1.0 - C                                     # lower = tighter / more harmonic energy
    return tS, H

def hsi_v3_from_window_spec(tW, fW, SW, *,
                            ladder=(7.83,14.3,20.8,27.3,33.8,40.3,46.8,53.3),
                            in_bw=0.5,           # ±Hz around each harmonic (in-band)
                            ring_offset=1.2,     # Hz away from each harmonic (side-ring)
                            ring_bw=0.6,         # ±Hz width of the side-ring
                            smooth_hz=6.0):      # 1/f flattening smoothness
    """
    HSI_v3(t): lower = tighter harmonics.
    - Flatten 1/f → excess spectrum R(f,t)
    - Compare in-band (around harmonics) vs side-ring (flanks) energy
    - HSI = 1 / (1 + IN/OUT)  ∈ (0,1)
    """
    # 1) flatten 1/f along frequency (per time) on log power
    df = float(np.median(np.diff(fW)))
    W  = max(5, int(np.ceil(smooth_hz/df)));  W += (W % 2 == 0)  # odd length
    logS = np.log(SW + 1e-20)
    bg   = savgol_filter(logS, window_length=W, polyorder=2, axis=0, mode='interp')
    R    = np.exp(logS - bg)  # excess over 1/f, >= 0

    # 2) build in-band and side-ring weights
    Win = np.zeros_like(fW, float)
    Wring = np.zeros_like(fW, float)
    for hk in ladder:
        Win   += (np.abs(fW - hk) <= in_bw).astype(float)
        Wring += (np.abs(fW - (hk - ring_offset)) <= ring_bw).astype(float)
        Wring += (np.abs(fW - (hk + ring_offset)) <= ring_bw).astype(float)

    # 3) energy ratio per time
    Ein  = (R * Win[:,None]).sum(axis=0)
    Eout = (R * Wring[:,None]).sum(axis=0) + 1e-12
    ratio = Ein / Eout

    # 4) map to HSI in (0,1): tighter → ratio↑ → HSI↓
    H = 1.0 / (1.0 + ratio)
    return tW, H

def sanity(pack):
    z7   = _as_float_1d(pack.get("z_7p83"))
    plv  = _as_float_1d(pack.get("plv_7p83"))
    hsi  = _as_float_1d(pack.get("hsi"))
    spec = pack.get("spec")

    out = {
        "has_spec": isinstance(spec, tuple) and len(spec) == 3 and np.ndim(spec[2]) == 2,
        "t_len":    len(_as_float_1d(pack.get("t"))),
        "z7_std":   float(np.nanstd(z7)) if z7.size else np.nan,
        "plv_med":  float(np.nanmedian(plv)) if plv.size else np.nan,
        "hsi_min":  float(np.nanmin(hsi)) if hsi.size else np.nan,
        "hsi_max":  float(np.nanmax(hsi)) if hsi.size else np.nan,
    }
    # print(out)
    return out

def _bp_hilbert_env_z(X, fs, f0, bw=0.5):
    b = firwin(801, [max(0.1, f0-bw), f0+bw], pass_zero=False, fs=fs)
    Xb = filtfilt(b, [1.0], X, axis=-1, padlen=min(2400, X.shape[-1]-1))
    A = np.abs(hilbert(Xb, axis=-1))              # amp per channel
    A_med = np.nanmedian(A, axis=0)               # combine first
    A_med = (A_med - np.nanmean(A_med)) / (np.nanstd(A_med) + 1e-9)  # z across time
    return A_med

def _plv_7p8(X, fs, f0=7.83, bw=0.5, win=4.0, step=0.25):
    b = firwin(801, [max(0.1, f0-bw), f0+bw], pass_zero=False, fs=fs)
    Xb = filtfilt(b, [1.0], X, axis=-1, padlen=min(2400, X.shape[-1]-1))
    ph = np.angle(hilbert(Xb, axis=-1))
    R_t = np.abs(np.nanmean(np.exp(1j*ph), axis=0))
    n = X.shape[1]; W, S = int(round(win*fs)), int(round(step*fs))
    out, t_mid = [], []
    i = 0
    while i+W <= n:
        out.append(np.nanmean(R_t[i:i+W])); t_mid.append((i+W/2)/fs); i += S
    # resample to raw time length
    t = np.arange(n)/fs
    return np.interp(t, t_mid, out, left=out[0], right=out[-1])

def _spec_median(X, fs, band=(2,60), win=2.0, ov=0.75):
    nper = int(round(win*fs)); nover = int(round(ov*nper)); nover = min(nover, nper-1)
    Slist, f, t = [], None, None
    for k in range(X.shape[0]):
        f_k, t_k, Z = stft(X[k], fs=fs, window='hann', nperseg=nper, noverlap=nover,
                           detrend='constant', boundary=None, padded=False)
        P = (np.abs(Z)**2)
        if f is None: f, t = f_k, t_k
        Slist.append(P)
    S = np.nanmedian(np.stack(Slist, axis=0), axis=0)           # (F,T)
    m = (f>=band[0]) & (f<=band[1])
    fB, SB = f[m], S[m,:]
    # convert t to absolute seconds later outside this function
    return t, fB, SB

def _hsi_from_spec(tS, fS, S, ladder=(7.83,14.3,20.8,27.3,33.8,40.3,46.8,53.3), lbw=1.0):
    L = np.zeros_like(fS)
    for hk in ladder:
        L += np.exp(-0.5*((fS-hk)/lbw)**2)
    L /= (L.sum()+1e-12)
    C = (S * L[:,None]).sum(axis=0) / (S.sum(axis=0)+1e-12)
    H = 1.0 - C
    return H  # per-spec time

def _first_onset(mask: np.ndarray, t: np.ndarray, min_dur: float) -> int | None:
    mask = np.asarray(mask, bool); t = np.asarray(t, float)
    dm = np.diff(mask.astype(int), prepend=0, append=0)
    starts = np.where(dm == 1)[0]; ends = np.where(dm == -1)[0] - 1
    for s, e in zip(starts, ends):
        if t[e] - t[s] >= min_dur:
            return int(s)
    return None

def _collect_runs(mask: np.ndarray, t: np.ndarray, min_dur: float, dt: float) -> List[Tuple[int, int]]:
    mask = np.asarray(mask, bool)
    if mask.size == 0:
        return []
    dm = np.diff(mask.astype(int), prepend=0, append=0)
    starts = np.where(dm == 1)[0]
    ends = np.where(dm == -1)[0] - 1
    runs: List[Tuple[int, int]] = []
    for s, e in zip(starts, ends):
        if e < s:
            continue
        dur = float((t[e] - t[s]) if e > s else 0.0) + dt
        if dur >= min_dur - 1e-9:
            runs.append((int(s), int(e)))
    return runs

def _band_mask(t: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (t >= lo) & (t <= hi)

def _clip_seed_to_window(seed_t: float | None, t0: float, t1: float, default_center=True) -> float:
    if seed_t is None and default_center:
        return 0.5 * (t0 + t1)
    return float(np.clip(seed_t if seed_t is not None else 0.5*(t0+t1), t0, t1))

def _robust_z(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float)
    # clip extreme 1% to avoid a single spike exploding z
    q1, q99 = np.nanpercentile(y, (1, 99))
    y = np.clip(y, q1, q99)
    med = np.nanmedian(y)
    mad = 1.4826 * np.nanmedian(np.abs(y - med))
    iqr = (np.nanpercentile(y, 75) - np.nanpercentile(y, 25)) / 1.349
    sigma = float(max(mad, iqr, 1e-6))
    return (y - med) / sigma

def _winsor_robust_z(y: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    """Winsorized robust‑z (MAD∨IQR) to avoid explosive scales in quiet windows."""
    y = np.asarray(y, float)
    if not np.isfinite(y).any():
        return np.zeros_like(y, float)
    q1, q99 = np.nanpercentile(y, (p_lo, p_hi))
    yw = np.clip(y, q1, q99)
    med = np.nanmedian(yw)
    mad = 1.4826 * np.nanmedian(np.abs(yw - med))
    iqr = (np.nanpercentile(yw, 75) - np.nanpercentile(yw, 25)) / 1.349
    sigma = float(max(mad, iqr, 1e-6))
    return (y - med) / sigma

def _rising_over_tau(y: np.ndarray, t: np.ndarray, tau_s: float, eps: float) -> np.ndarray:
    dt = float(np.median(np.diff(t)))
    k  = max(1, int(round(tau_s / max(dt, 1e-6))))
    y0 = np.asarray(y, float)
    y_prev = np.r_[y0[:k], y0[:-k]]
    return (y0 - y_prev) > eps

def _bridge(mask: np.ndarray, t: np.ndarray, bridge_sec: float = 0.02) -> np.ndarray:
    """Bridge short False gaps (morphological closing) to avoid micro‑breaks."""
    if bridge_sec <= 0:
        return mask
    dt = float(np.median(np.diff(t)))
    k  = max(1, int(round(bridge_sec / max(dt, 1e-6))))
    if k <= 1:
        return mask
    m = mask.astype(bool)
    dil = m.copy()
    for j in range(1, k+1):
        dil[:-j] |= m[j:]
        dil[j:]  |= m[:-j]
    er = dil.copy()
    for j in range(1, k+1):
        win = 2*j+1
        box = np.convolve(dil.astype(int), np.ones(win, int), 'same')
        er &= (box >= win)
    return er


def _spectral_slope_series(t_spec: np.ndarray, f_spec: np.ndarray, Sxx: np.ndarray,
                           band: Tuple[float, float] = (3.0, 45.0),
                           exclude_centers: Optional[Sequence[float]] = None,
                           exclude_bw: float = 0.6) -> Tuple[np.ndarray, np.ndarray]:
    f_spec = np.asarray(f_spec, float)
    mask = (f_spec >= band[0]) & (f_spec <= band[1])
    if exclude_centers:
        for center in exclude_centers:
            mask &= ~((f_spec >= center - exclude_bw) & (f_spec <= center + exclude_bw))
    if not np.any(mask):
        return t_spec, np.full(Sxx.shape[1], np.nan)
    x = np.log10(f_spec[mask] + 1e-12)
    slopes = []
    for col in range(Sxx.shape[1]):
        y = 10 * np.log10(Sxx[mask, col] + 1e-20)
        if not np.any(np.isfinite(y)):
            slopes.append(np.nan)
            continue
        y = np.nan_to_num(y, nan=np.nanmedian(y[np.isfinite(y)]))
        slope, _ = np.polyfit(x, y, 1)
        slopes.append(float(slope))
    return t_spec, np.asarray(slopes)


def _avalanche_size_duration(signal: np.ndarray, t: np.ndarray, thresh: float,
                             bridge_sec: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(signal, float)
    mask = signal >= thresh
    if bridge_sec and bridge_sec > 0:
        mask = _bridge(mask, t, bridge_sec)
    sizes, durations = [], []
    dt = float(np.median(np.diff(t))) if t.size > 1 else 1.0
    i = 0
    while i < mask.size:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < mask.size and mask[j]:
            j += 1
        segment = signal[i:j]
        duration = max((j - i) * dt, dt)
        size = float(np.trapz(np.maximum(segment - thresh, 0.0), dx=dt))
        sizes.append(max(size, 1e-6))
        durations.append(duration)
        i = j
    return np.asarray(sizes), np.asarray(durations)


def _kuramoto_order_series(X: np.ndarray, fs: float, center_hz: float, bw: float) -> Tuple[np.ndarray, np.ndarray]:
    b = _fir_bandpass(center_hz, bw, fs)
    Xb = filtfilt(b, [1.0], X, axis=-1, padlen=min(2400, X.shape[-1]-1))
    phases = np.angle(hilbert(Xb, axis=-1))
    order = np.abs(np.nanmean(np.exp(1j * phases), axis=0))
    t = np.arange(order.size) / fs
    return t, order.astype(float)


def _msc_channel_to_reference(ch_signal: np.ndarray, ref_signal: np.ndarray) -> float:
    z_ch = hilbert(ch_signal)
    z_ref = hilbert(ref_signal)
    num = np.abs(np.mean(z_ch * np.conj(z_ref))) ** 2
    den = (np.mean(np.abs(z_ch) ** 2) * np.mean(np.abs(z_ref) ** 2)) + 1e-12
    return float(num / den)


def _msc_matrix(X: np.ndarray, fs: float, freqs: Sequence[float], bw: float,
                n_surrogates: int = 20, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_ch = X.shape[0]
    msc = np.zeros((len(freqs), n_ch))
    null95 = np.zeros((len(freqs), n_ch))
    ref = np.nanmedian(X, axis=0)
    for fi, f0 in enumerate(freqs):
        b = _fir_bandpass(f0, bw, fs)
        Xf = filtfilt(b, [1.0], X, axis=1, padlen=min(2400, X.shape[-1]-1))
        ref_f = filtfilt(b, [1.0], ref, axis=0, padlen=min(2400, ref.size-1))
        for ci in range(n_ch):
            ch = Xf[ci]
            m_val = _msc_channel_to_reference(ch, ref_f)
            msc[fi, ci] = m_val
            surrogates = []
            for _ in range(n_surrogates):
                shift = rng.integers(0, ref_f.size)
                ref_shift = np.roll(ref_f, shift)
                surrogates.append(_msc_channel_to_reference(ch, ref_shift))
            null95[fi, ci] = float(np.nanpercentile(surrogates, 95))
    return msc, null95


def _plv_matrix(X: np.ndarray, fs: float, f0: float, bw: float) -> np.ndarray:
    b = _fir_bandpass(f0, bw, fs)
    Xb = filtfilt(b, [1.0], X, axis=1, padlen=min(2400, X.shape[-1]-1))
    phases = np.angle(hilbert(Xb, axis=-1))
    n = X.shape[0]
    plv_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dphi = phases[i] - phases[j]
            plv_val = np.abs(np.mean(np.exp(1j*dphi)))
            plv_mat[i, j] = plv_mat[j, i] = plv_val
    return plv_mat


def _mode_metrics(power: np.ndarray) -> Tuple[float, float]:
    power = np.asarray(power, float)
    power = np.clip(power, 0, None)
    total = power.sum()
    if total <= 0 or not np.any(np.isfinite(power)):
        return np.nan, np.nan
    normalized = power / total
    entropy = -np.sum(normalized * np.log(normalized + 1e-12)) / np.log(len(power))
    pr = (total ** 2) / (np.sum(power ** 2) + 1e-12)
    return float(entropy), float(pr)


def _interp_safe(x_new: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    xp = np.asarray(xp, float)
    fp = np.asarray(fp, float)
    if xp.size < 2 or fp.size < 2 or xp.size != fp.size:
        return np.full_like(x_new, np.nan, dtype=float)
    return np.interp(x_new, xp, fp, left=fp[0], right=fp[-1])


def _te_matrix(X: np.ndarray, fs: float, lead_sec: float = 0.05) -> np.ndarray:
    lead = max(1, int(round(lead_sec * fs)))
    n = X.shape[0]
    te = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            xi = X[i]
            xj_future = X[j][lead:]
            xj_past = X[j][:-lead]
            if xj_future.size == 0:
                continue
            cc_future = np.corrcoef(xi[:-lead], xj_future)[0, 1]
            cc_past = np.corrcoef(xj_past, xj_future)[0, 1]
            te[i, j] = (cc_future ** 2) - (cc_past ** 2)
    return te


def _transfer_entropy_proxy(theta_env: np.ndarray, gamma_env: np.ndarray, fs: float,
                             lead_sec: float = 0.1, win_sec: float = 2.0,
                             step_sec: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    theta_env = np.asarray(theta_env, float)
    gamma_env = np.asarray(gamma_env, float)
    lead = max(1, int(round(lead_sec * fs)))
    win = max(2, int(round(win_sec * fs)))
    step = max(1, int(round(step_sec * fs)))
    if theta_env.size <= lead + win or gamma_env.size <= lead + win:
        return np.array([]), np.array([])
    values, times = [], []
    for start in range(0, theta_env.size - lead - win + 1, step):
        theta = theta_env[start:start+win]
        gamma_future = gamma_env[start+lead:start+lead+win]
        gamma_past = gamma_env[start:start+win]
        if np.nanstd(theta) < 1e-9 or np.nanstd(gamma_future) < 1e-9 or np.nanstd(gamma_past) < 1e-9:
            values.append(np.nan)
        else:
            cc_future = np.corrcoef(theta, gamma_future)[0, 1]
            cc_past = np.corrcoef(gamma_past, gamma_future)[0, 1]
            values.append((cc_future ** 2) - (cc_past ** 2))
        times.append((start + win / 2) / fs)
    return np.asarray(times), np.asarray(values)


def _sample_entropy(signal: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    signal = np.asarray(signal, float)
    n = signal.size
    if n < m + 2:
        return np.nan
    r *= np.nanstd(signal) + 1e-12
    def _phi(mm: int) -> float:
        count = 0
        total = 0
        for i in range(n - mm):
            template = signal[i:i+mm]
            diffs = np.max(np.abs(signal[i+1:] - template[:, None]), axis=0)
            count += np.sum(diffs <= r)
            total += n - mm - i - 1
        return (count / max(total, 1)) if total else 0.0
    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)
    if phi_m <= 0 or phi_m1 <= 0:
        return np.nan
    return -np.log(phi_m1 / phi_m)


def _complexity_series(signal: np.ndarray, t: np.ndarray, win_sec: float, step_sec: float) -> Tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(signal, float)
    t = np.asarray(t, float)
    if t.size < 2:
        return np.array([]), np.array([])
    fs = 1.0 / max(np.median(np.diff(t)), 1e-6)
    w = max(2, int(round(win_sec * fs)))
    s = max(1, int(round(step_sec * fs)))
    ent, times = [], []
    for start in range(0, signal.size - w + 1, s):
        seg = signal[start:start+w]
        ent.append(_sample_entropy(seg))
        idx = min(start + w//2, t.size - 1)
        times.append(t[idx])
    return np.asarray(times), np.asarray(ent)


def _hurst_exponent(signal: np.ndarray, scales: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, float]:
    signal = np.asarray(signal, float)
    rms = []
    scales = np.asarray(list(scales), int)
    for scale in scales:
        if scale <= 1 or scale >= signal.size:
            rms.append(np.nan)
            continue
        cut = signal[:signal.size - signal.size % scale]
        if cut.size == 0:
            rms.append(np.nan)
            continue
        segments = cut.reshape(-1, scale)
        rms_scale = np.mean(np.std(segments, axis=1, ddof=1))
        rms.append(rms_scale)
    rms = np.asarray(rms, float)
    valid = np.isfinite(rms)
    if valid.sum() < 2:
        return scales.astype(float), rms, np.nan
    x = np.log10(scales[valid].astype(float))
    y = np.log10(rms[valid] + 1e-12)
    hurst, _ = np.polyfit(x, y, 1)
    return scales.astype(float), rms, float(hurst)


def _lempel_ziv_complexity(signal: np.ndarray) -> float:
    signal = np.asarray(signal, float)
    n = signal.size
    if n < 16:
        return np.nan
    med = np.nanmedian(signal)
    if not np.isfinite(med):
        return np.nan
    binary = (signal > med).astype(int)
    s = ''.join(binary.astype(str))
    i = 0
    k = 1
    c = 1
    while True:
        if i + k >= len(s):
            c += 1
            break
        segment = s[i:i+k]
        if segment in s[:i]:
            k += 1
        else:
            c += 1
            i += k
            if i >= len(s):
                break
            k = 1
    return c / (n / np.log2(max(n, 2)))


def _lz_complexity_series(signal: np.ndarray, t: np.ndarray, win_sec: float, step_sec: float) -> Tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(signal, float)
    t = np.asarray(t, float)
    if t.size < 2:
        return np.array([]), np.array([])
    fs = 1.0 / max(np.median(np.diff(t)), 1e-6)
    w = max(16, int(round(win_sec * fs)))
    s = max(1, int(round(step_sec * fs)))
    values, times = [], []
    for start in range(0, signal.size - w + 1, s):
        seg = signal[start:start+w]
        values.append(_lempel_ziv_complexity(seg))
        idx = min(start + w//2, t.size - 1)
        times.append(t[idx])
    return np.asarray(times), np.asarray(values)


def _baseline_slice(records: pd.DataFrame, time_col: str, window: Tuple[float, float],
                    offset: float, duration: float) -> Tuple[np.ndarray, np.ndarray]:
    if offset <= 0 or duration <= 0:
        return np.array([]), np.array([])
    t = np.asarray(records[time_col], float)
    start = max(window[0] - offset - duration, t[0])
    end = max(window[0] - offset, start + duration)
    mask = (t >= start) & (t <= end)
    return t[mask], mask

def piano_roll_from_spec(spec_by_window, *, harmonics=(7.8, 14.3, 20.8), bw=0.6):
    """
    spec_by_window: (tW, fW, SW) from your per-window spectrogram
                    (linear power, SW shape = (F, T))
    Returns: tW (seconds), M (K × T) where K = len(harmonics),
            each row is the row‑z median power within ±bw around that harmonic.
    """
    tW, fW, SW = spec_by_window

    # robust row‑wise z on the frequency axis (per frequency bin across time)
    SdB = 10*np.log10(SW + 1e-20)
    med = np.median(SdB, axis=1, keepdims=True)
    mad = np.median(np.abs(SdB - med), axis=1, keepdims=True) + 1e-9
    Z = (SdB - med) / mad  # (F, T), "row‑z"

    rows = []
    for fk in harmonics:
        m = np.abs(fW - fk) <= bw
        if not np.any(m):                      # guard if a line falls between bins
            rows.append(np.full(Z.shape[1], np.nan))
        else:
            rows.append(np.nanmedian(Z[m, :], axis=0))
    M = np.vstack(rows)                         # (K, T)
    return tW, M

def bandtrace_from_spec(tW, fW, SW, f0, bw=0.8):
    Zdb = 10*np.log10(np.maximum(SW, 1e-12))
    df  = float(np.median(np.diff(fW)))
    bw  = max(bw, df*1.01)
    k   = (fW >= f0 - bw/2 - 1e-6) & (fW <= f0 + bw/2 + 1e-6)
    if not np.any(k):
        k = np.abs(fW - f0) == np.min(np.abs(fW - f0))
    w = np.exp(-0.5*((fW[k]-f0)/(0.35*bw))**2)
    r = (Zdb[k] * w[:, None]).sum(0) / w.sum()
    r = robust_z(r)
    r = smooth_sec(tW, r, 0.15)
    return r

def plot_ignition_window_report(
    _records,
    provider,
    electrodes,
    *,
    params=PhaseParams(),
    title=None,
    hsi_plot_mode="delta",
    hsi_ylim=("pct", (1, 99)),
    # NEW: seeding + band constraints + padding for P0/P1
    seed_t="center",                  # float | "center" | None
    p0_band=(-2.5, +1.5),             # allowed P0 window relative to seed_t (s)
    p1_band=(-1.5, +1.0),             # allowed P1 window relative to seed_t (s)
    pad_s=2.0,                        # ignore this much at window edges
    centers=[7.8,14.3,20.8], bw=0.5,
    debug=False,
    session_name=None,
    phases=None,                      # Optional pre-computed phases dict
    bic_triads=None,                  # Optional triadic bicoherence dict {label: timeseries}
    t_bic=None                        # Optional bicoherence time axis
):
    
    _fnd = "{:.2f}".format(centers[0]) 
    _2nd = "{:.2f}".format(centers[1])
    _3rd = "{:.2f}".format(centers[2])

    t = provider.t()

    # Raw series from provider
    zf_raw = provider.z_fund()
    z2_raw = provider.z_h2()
    z3_raw = provider.z_h3()
    plv    = provider.plv_fund()
    hsi    = provider.hsi()
    beta   = provider.beta()
    ridge  = provider.ridge_is_fund()
    b77    = provider.bic_7_7_15()
    b7_15  = provider.bic_7_15_23()
    pac    = provider.pac_mvl()

    # --- Use the same normalization everywhere (per-window robust-z; 150 ms smoothing) ---
    def z_norm(y):  # robust-z, no shrinking; then light smoothing for detector stability
        return smooth_sec(t, robust_z(np.asarray(y, float)), 0.15)

    zf_z = z_norm(zf_raw)
    z2_z = z_norm(z2_raw)
    z3_z = z_norm(z3_raw)

    # >>> pass the seed & bands to the detector (or use provided phases) <<<
    if phases is None:
        phases = _detect_ignition_phases(
            t, zf_z, plv, hsi, z2_z, z3_z,
            beta_t=beta, ridge_is_fund=ridge,
            bic_7_7_15=b77, bic_7_15_23=b7_15, pac_mvl=pac,
            params=params,
            seed_t=seed_t,
            p0_band=p0_band,
            p1_band=p1_band,
            pad_s=pad_s,
        )

    # AFTER
    fig = plt.figure(figsize=(16, 10), constrained_layout=True, dpi=160)
    gs = GridSpec(3, 2, figure=fig)  # let constrained_layout do the spacing
    
    
    # PANEL A: SPECTROGRAM ============
    axA = fig.add_subplot(gs[0, 0])
    # Try to get spectrogram from provider, with fallback
    spec = provider.spectrogram_for_window(t.min(), t.max())
    if spec is None:
        spec = provider.spectrogram()
    if spec is None:
        # Last resort: compute on the fly
        spec = window_spec_median(_records, (t.min(), t.max()), channels=electrodes, fs=128, time_col='Timestamp')
    tW, fW, SW = _slice_spec_to_window(spec, (t.min(), t.max()), min_cols=20)
    
    # row-wise robust-z in dB
    Zdb = 10*np.log10(np.maximum(SW, 1e-12))
    med = np.median(Zdb, axis=1, keepdims=True)
    mad = np.median(np.abs(Zdb - med), axis=1, keepdims=True) + 1e-12
    Z   = (Zdb - med) / (1.4826*mad)

    # Use imshow with interpolation for smoother rendering
    extent = [tW[0], tW[-1], fW[0], fW[-1]]
    im = axA.imshow(Z, extent=extent, origin='lower', aspect='auto',
                    vmin=-3, vmax=3, interpolation='lanczos')

    # Add subtle horizontal lines at SR harmonics
    for freq in centers:
        if fW[0] <= freq <= fW[-1]:
            axA.axhline(freq, color='white', linestyle='--', linewidth=0.8, alpha=0.6)

    axA.set_ylabel('Hz'); axA.set_title('SR Spectrogram (2-25 Hz)')
    annotate_phases(axA, phases, *axA.get_ylim(), show_labels=False, show_shading=False)


    # # PANEL B: HARMONIC PIANO ROLL ============
    axB = fig.add_subplot(gs[0,1])

    # # Use the exact spectrogram used in Panel A (with fallback for end-of-session windows)
    spec_b = provider.spectrogram_for_window(t.min(), t.max())
    if spec_b is None:
        spec_b = provider.spectrogram()
    if spec_b is None:
        # Last resort: compute on the fly
        spec_b = window_spec_median(_records, (t.min(), t.max()), channels=electrodes, fs=128, time_col='Timestamp')
    tW_b, fW_b, SW_b = _slice_spec_to_window(spec_b, (t.min(), t.max()), min_cols=20)

    # Build envelope z-scores for all harmonics in centers
    # First 3 from provider (already computed), rest from spectrogram
    harmonic_envelopes = [zf_z, z2_z, z3_z]
    for i in range(3, len(centers)):
        # Extract envelope for harmonic i+1 from spectrogram
        z_h_raw = bandtrace_from_spec(tW_b, fW_b, SW_b, centers[i], bw=bw)
        # Interpolate to match provider's time axis
        z_h = np.interp(t, tW_b, z_h_raw)
        harmonic_envelopes.append(z_h)

    # Stack all harmonics into piano roll
    PR = np.vstack(harmonic_envelopes)
    n_harmonics = len(centers)
    im = axB.imshow(PR, origin='lower', aspect='auto',
                extent=[t.min(), t.max(), 0.5, n_harmonics + 0.5], vmin=-3, vmax=3)

    # Dynamic yticks and labels for all harmonics
    yticks = list(range(1, n_harmonics + 1))
    yticklabels = [f"{i}× ({centers[i-1]:.2f})" for i in yticks]
    axB.set_yticks(yticks)
    axB.set_yticklabels(yticklabels)
    axB.set_title('Harmonic Piano‑Roll (envelope z)')
    annotate_phases(axB, phases, 0.5, n_harmonics + 0.5, show_labels=False, show_shading=False)
    fig.colorbar(im, ax=[axA, axB], pad=0.02, fraction=0.05).set_label('z')

    # window-robust-z + 150 ms smoothing (same as Panel B piano-roll)
    def z_for_display(t, y):
        return smooth_sec(t, robust_z(y), 0.15)
    
    zf_d = z_for_display(t, provider.z_fund())
    z2_d = z_for_display(t, provider.z_h2())
    z3_d = z_for_display(t, provider.z_h3())



    # PANEL C: Fundamental & harmonics envelopes + PLV ===============
    axC = fig.add_subplot(gs[1, 0])
    # Plot all harmonics with decreasing line width for higher harmonics
    for i, z_env in enumerate(harmonic_envelopes):
        harmonic_num = i + 1
        freq = centers[i]
        lw = max(1.0, 1.5 - i * 0.1)  # Decrease line width for higher harmonics
        axC.plot(t, z_env, label=f"z@{freq:.2f}", lw=lw)
    axC.set_ylabel('z')
    # axC.set_ylim(-3.5, 3.5)


    axC2 = axC.twinx()
    axC2.plot(t, provider.plv_fund(), label=f"PLV@{centers[0]:.2f}", ls='--', lw=1.4, alpha=0.95)
    plo, phi = np.nanpercentile(plv, [1, 99])
    pad = 0.1 * (phi - plo + 1e-12)
    axC.set_title('Envelopes and PLV')
    axC.set_ylabel('z');
    axC2.set_ylabel('PLV')
    leg_c = axC.legend(loc='upper left', fontsize=8)
    leg_c.set_zorder(2000)
    leg_c2 = axC2.legend(loc='upper right')
    leg_c2.set_zorder(2000)

    annotate_phases(axC, phases, *axC.get_ylim())


    # PANEL D: HSI and 1/f β (with ΔHSI + percentile y-limits) =============
    axD = fig.add_subplot(gs[1, 1])

    # 1) choose HSI series to plot
    hsi = np.asarray(provider.hsi(), float)
    med = np.nanmedian(hsi[np.isfinite(hsi)])
    hsi_plot = np.asarray(hsi, float)
    
    if hsi_plot_mode.lower() == "delta":
        med = np.nanmedian(hsi_plot)
        hsi_plot = hsi_plot - med
        ylab = "ΔHSI (HSI − median)"
        # zero reference line for ΔHSI
        axD.axhline(0, color='0.85', lw=1, zorder=0)
        axD.grid(True, axis='y', color='0.9', linewidth=0.8)
    else:
        ylab = "HSI"
        
    def z_for_display(t_vec, y_vec, s=0.45):   # same ~0.45 s as above
        return smooth_sec(t_vec, y_vec, s)

    hsi_raw = np.asarray(provider.hsi(), float)
    hsi_disp = z_for_display(t, hsi_raw, 0.45)
    med = np.nanmedian(hsi_disp)
    axD.plot(t, hsi_disp - med, label='ΔHSI', lw=1.5)

    y = (hsi_disp - med)[np.isfinite(hsi_disp)]

  
    # 2) optional β on right axis (unchanged)
    if beta is not None:
        axD2 = axD.twinx()
        hsi_disp = smooth_sec(t, provider.hsi(), 0.45)
        med = np.nanmedian(hsi_disp)
        ΔHSI = hsi_disp - med

        axD.plot(t, ΔHSI - np.nanmedian(ΔHSI),ls="--", alpha=0.85, label="β (1/f slope)")
        # axD2.plot(t, beta, ls="--", alpha=0.85, label="β (1/f slope)")
        axD2.set_ylabel("β")
        leg_d2 = axD2.legend(loc='upper right')
        leg_d2.set_zorder(2000)

    axD.set_title('Harmonic Tightening')
    axD.set_ylabel(ylab)
    leg_d = axD.legend(loc='upper left')
    leg_d.set_zorder(2000)

    # 3) percentile or absolute y-limits
    mode, arg = hsi_ylim
    y = hsi_plot[np.isfinite(hsi_plot)]
    if y.size:
        if mode == "pct":
            lo, hi = np.nanpercentile(y, arg)
            pad = 0.05 * (hi - lo + 1e-12)
            axD.set_ylim(lo - pad, hi + pad)
        elif mode == "fixed":
            ymin, ymax = arg
            axD.set_ylim(ymin, ymax)
        # "auto" → let Matplotlib pick; do nothing

    annotate_phases(axD, phases, *axD.get_ylim())

    _fnd = "{:.2f}".format(centers[0]) 
    _2nd = "{:.2f}".format(centers[1])
    _3rd = "{:.2f}".format(centers[2])

    # Panel E: Triadic bicoherence
    axE = fig.add_subplot(gs[2, 0])

    # Use new triadic bicoherence if provided, otherwise fall back to provider data
    if bic_triads is not None and len(bic_triads) > 0 and t_bic is not None:
        # Generate distinguishable colors for any number of triads using colormap
        n_triads = len(bic_triads)
        if n_triads <= 10:
            # Use tab10 colormap for up to 10 triads (distinct categorical colors)
            cmap = plt.cm.get_cmap('tab10')
            triad_colors = [cmap(i) for i in range(n_triads)]
        else:
            # Use hsv colormap for more than 10 triads
            cmap = plt.cm.get_cmap('hsv')
            triad_colors = [cmap(i / n_triads) for i in range(n_triads)]

        for idx, (label, series) in enumerate(bic_triads.items()):
            # Interpolate bicoherence onto provider time axis and smooth
            series_interp = np.interp(t, t_bic, series, left=series[0], right=series[-1])
            series_smooth = smooth_sec(t, series_interp, 0.5)  # 500ms smoothing

            color = triad_colors[idx]
            lw = max(1.1, 1.5 - idx * 0.1)  # Thicker lines for more important triads
            axE.plot(t, series_smooth, label=label, color=color, linewidth=lw, alpha=0.9)

        # Horizontal legend inside chart at top
        ncol = min(5, n_triads)  # Max 4 columns for readability
        leg = axE.legend(loc='upper center', ncol=ncol, fontsize=6, frameon=True)
        if leg:
            leg.get_frame().set_facecolor('white')
            leg.get_frame().set_alpha(0.618)
            leg.set_zorder(2000)  # Ensure legend is on top
    elif (b77 is not None) and (b7_15 is not None):
        # Fallback to old bicoherence data from provider
        axE.plot(t, b77, label=f"bic ({_fnd},{_fnd}→{_2nd})")
        axE.plot(t, b7_15, label=f"bic ({_fnd},{_2nd}→{_3rd})")
        leg_e = axE.legend(loc='upper right')
        leg_e.set_zorder(2000)
    else:
        axE.text(0.5, 0.5, 'Bicoherence not provided', transform=axE.transAxes, ha='center', va='center')

    axE.set_title('Bicoherence (SR triads)')
    axE.set_ylabel('Bicoherence')
    annotate_phases(axE, phases, *axE.get_ylim())

    # Panel F: PAC MVL
    axF = fig.add_subplot(gs[2, 1])
    pac_raw = provider.pac_mvl()                   # already on provider.t() grid
    pac_disp = smooth_sec(t, pac_raw, 0.75)        # same smoothing as above

    # (optional) show relative change like ΔMVL to match the feel of ΔHSI
    show_delta = False
    if show_delta:
        med = np.nanmedian(pac_disp[np.isfinite(pac_disp)])
        y_to_plot = pac_disp - med
        axF.set_ylabel('ΔMVL')
    else:
        y_to_plot = pac_disp
        axF.set_ylabel('MVL')

    axF.plot(t, y_to_plot, lw=1.5)


    axF.plot(t, y_to_plot)
    axF.set_ylabel('MVL')
    axF.set_title('θ→γ PAC')
    annotate_phases(axF, phases, *axF.get_ylim())
    
    lo,hi = np.nanpercentile(pac, (1,99)); pad = 0.05*(hi-lo+1e-12)
    # axF.set_ylim(lo-pad, hi+pad)

    

    # fig.suptitle(title or 'Ignition Window Report', y=0.98)

    traces = {
        't': t, 'z_fund': zf_z, 'z_h2': z2_z, 'z_h3': z3_z, 'plv': plv, 'hsi': hsi,
        'beta': beta, 'ridge_is_fund': ridge, 'bic_7_7_15': b77, 'bic_7_15_23': b7_15, 'pac_mvl': pac,
        'phases': phases,
    }

    # fig.suptitle(title, fontsize=16, y=1.05)
    # fig.set_constrained_layout_pads(wspace=0.02, hspace=0.06, w_pad=0.02, h_pad=0.02)


    # --- compute metrics for annotation ---
    srz_max = float(np.nanmax(zf_z))
    msc = float(np.nanmean(plv))
    hsi_val = float(np.nanmean(hsi))
    score = srz_max*msc/(1+hsi_val) #float(np.nanmean([srz_max, msc, hsi_val]))


    # Panel plots (A-F) omitted for brevity, same as original...
    # [keep all panel plotting code unchanged]


    # Add a single top-level title with session info and metrics
    sup_title = f"SRz_max={srz_max:.2f}, MSC={msc:.2f}, HSI={hsi_val:.2f}, Score={score:.2f}"
    if title:
        sup_title = f"{title}\n{session_name}"
    fig.suptitle(sup_title, fontsize=14, y=1.05)

    # # or (no constrained_layout):
    # gs.update(wspace=0.18, hspace=0.55)

    # Remove x-axis margins - make data fill entire chart width
    for ax in [axA, axB, axC, axD, axE, axF]:
        ax.set_xlim(t[0], t[-1])

    return fig, phases, traces

def _infer_fs(df: pd.DataFrame, time_col: str = 'Timestamp') -> float:
    t = np.asarray(df[time_col], float)
    dt = np.median(np.diff(t))
    if not np.isfinite(dt) or dt <= 0: raise ValueError("Cannot infer fs from time column")
    return float(round(1.0 / dt))

def _looks_like_eeg_col(name: str) -> bool:
    if name.startswith('EEG.'): return True
    tail = name.split('.')[-1]
    return tail.upper() in _COMMON_EEG_NAMES

def _auto_channels(df: pd.DataFrame, time_col: str) -> List[str]:
    cols = [c for c in df.columns if c != time_col and np.issubdtype(df[c].dtype, np.number) and _looks_like_eeg_col(str(c))]
    if not cols:
        cols = [c for c in df.columns if c != time_col and np.issubdtype(df[c].dtype, np.number)]
    if not cols:
        raise ValueError("No numeric EEG columns found")
    return cols

def _get_matrix(df: pd.DataFrame, channels: Sequence[str]) -> np.ndarray:
    X = np.stack([np.asarray(df[c], float) for c in channels], axis=0)
    return X  # (n_ch, n_samples)

def _fir_bandpass(f0: float, bw: float, fs: float, numtaps: int = 801) -> np.ndarray:
    lo = max(0.1, f0 - bw)
    hi = f0 + bw
    return firwin(numtaps, [lo, hi], pass_zero=False, fs=fs)

def _fir_lowpass(fc: float, fs: float, numtaps: int = 801) -> np.ndarray:
    return firwin(numtaps, fc, pass_zero=True, fs=fs)

def _sliding_windows(n: int, fs: float, win_sec: float, step_sec: float) -> List[Tuple[int,int]]:
    w = int(round(win_sec * fs)); s = int(round(step_sec * fs))
    idx = []
    i = 0
    while i + w <= n:
        idx.append((i, i + w))
        i += s
    if not idx:
        idx.append((0, n))
    return idx

def _plv_across_channels(phases: np.ndarray) -> float:
    """PLV across channels for one time point given phase per-channel."""
    return float(np.abs(np.nanmean(np.exp(1j*phases))))

def _plv_timecourse(X: np.ndarray, fs: float, f0: float, bw: float,
                    win_sec: float, step_sec: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Correct: compute channel resultant |mean(exp(i*phi_ch(t)))| per *sample*,
    then average that R(t) in sliding windows.
    """
    n = X.shape[1]
    idx = _sliding_windows(n, fs, win_sec, step_sec)
    b = firwin(801, [max(0.1, f0-bw), f0+bw], pass_zero=False, fs=fs)
    Xb = filtfilt(b, [1.0], X, axis=-1, padlen=min(2400, X.shape[-1]-1))
    ph = np.angle(hilbert(Xb, axis=-1))            # (ch, n)
    R_t = np.abs(np.nanmean(np.exp(1j*ph), axis=0))  # (n,)

    t_mid, plv = [], []
    for i0, i1 in idx:
        t_mid.append((i0 + i1) / 2 / fs)
        plv.append(float(np.nanmean(R_t[i0:i1])))
    return np.asarray(t_mid), np.asarray(plv)


def _msc_timecourse(X: np.ndarray, fs: float, f0: float, bw: float,
                    win_sec: float, step_sec: float,
                    v_ref: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sliding magnitude-squared coherence between the broadband channel mean
    and a band-limited aggregate SR reference.
    """
    n = X.shape[1]
    idx = _sliding_windows(n, fs, win_sec, step_sec)
    if not idx:
        return np.array([]), np.array([])

    x_mean = np.nanmean(X, axis=0)
    if v_ref is None:
        b = firwin(801, [max(0.1, f0 - bw), f0 + bw], pass_zero=False, fs=fs)
        Xb = filtfilt(b, [1.0], X, axis=-1, padlen=min(2400, X.shape[-1]-1))
        cov = np.cov(Xb)
        eigvals, eigvecs = np.linalg.eigh(cov)
        w = eigvecs[:, -1]
        w = w / (np.linalg.norm(w) + 1e-12)
        v_ref_local = w @ Xb
    else:
        v_ref_local = np.asarray(v_ref, float)

    t_mid, msc_vals = [], []
    base_nper = max(8, int(round(win_sec * fs)))
    for i0, i1 in idx:
        seg_x = x_mean[i0:i1]
        seg_ref = v_ref_local[i0:i1]
        seg_len = min(len(seg_x), len(seg_ref))
        if seg_len < base_nper:
            continue
        nper = min(base_nper, seg_len)
        f, Cxy = coherence(seg_x, seg_ref, fs=fs, nperseg=nper)
        if Cxy.size == 0:
            continue
        freq_idx = int(np.argmin(np.abs(f - f0)))
        msc_vals.append(float(Cxy[freq_idx]))
        t_mid.append((i0 + i1) / 2 / fs)

    return np.asarray(t_mid), np.asarray(msc_vals)


def _narrowband_envelope_z(X, fs, f0, bw):

    b  = firwin(801, [max(0.1, f0-bw), f0+bw], pass_zero=False, fs=fs)
    Xb = filtfilt(b, [1.0], X, axis=-1, padlen=min(2400, X.shape[-1]-1))
    H  = hilbert(Xb, axis=-1)
    amp_med = np.nanmedian(np.abs(H), axis=0)              # raw envelope
    phase_mean = np.angle(np.nanmean(np.exp(1j*np.angle(H)), axis=0))
    return amp_med.astype(float), phase_mean.astype(float)

def _hsi_timecourse(X, fs, win_sec, step_sec, ladder, ladder_bw=1.0, band=(2,60)):
    n = X.shape[1]
    w = int(round(win_sec*fs)); s = int(round(step_sec*fs))
    idx = []; i = 0
    while i + w <= n:
        idx.append((i, i+w)); i += s
    if not idx: idx.append((0, n))
    ts = []; hs = []
    for i0,i1 in idx:
        seg = X[:, i0:i1]
        f, P = welch(seg, fs=fs, nperseg=min(int(fs*2), seg.shape[-1]), axis=-1)
        Pm = np.nanmedian(P, axis=0)
        m = (f>=band[0]) & (f<=band[1])
        fB, PB = f[m], Pm[m]
        L = np.zeros_like(fB)
        for hk in ladder:
            L += np.exp(-0.5*((fB - hk)/ladder_bw)**2)
        Lsum = np.sum(L)
        L = L / Lsum if Lsum>0 else L
        C = float(np.sum(PB * L) / (np.sum(PB)+1e-12))
        hs.append(1.0 - C); ts.append((i0+i1)/2/fs)
    return np.asarray(ts), np.asarray(hs)

def _pac_tort_mi_timecourse(X, fs, theta_band=(7,8), gamma_band=(40,100), win_sec=4.0, step_sec=0.25, n_bins=18):
    # clamp gamma safely like before
    gm_hi = min(gamma_band[1], 0.45*fs); gm_lo = max(gamma_band[0], 5.0)
    if gm_hi - gm_lo < 5.0: c = 0.5*(gm_lo+gm_hi); gm_lo, gm_hi = c-2.5, c+2.5
    gamma_band = (gm_lo, gm_hi)

    b_th = firwin(801, theta_band, pass_zero=False, fs=fs)
    b_gm = firwin(801, gamma_band, pass_zero=False, fs=fs)
    Xth = filtfilt(b_th, [1.0], X, axis=-1, padlen=min(2400, X.shape[-1]-1))
    Xgm = filtfilt(b_gm, [1.0], X, axis=-1, padlen=min(2400, X.shape[-1]-1))
    ph  = np.angle(hilbert(Xth, axis=-1))         # (ch, n)
    amp = np.abs(hilbert(Xgm, axis=-1))           # (ch, n)

    # reduce across channels
    ph_med  = np.angle(np.nanmean(np.exp(1j*ph), axis=0))
    amp_med = np.nanmedian(amp, axis=0)

    n = X.shape[1]; W = int(round(win_sec*fs)); S = int(round(step_sec*fs))
    t_mid, mi = [], []
    edges = np.linspace(-np.pi, np.pi, n_bins+1)
    for i0 in range(0, max(1, n-W+1), S):
        i1 = i0 + W
        t_mid.append((i0+i1)/2/fs)
        ph_seg  = ph_med[i0:i1]
        amp_seg = amp_med[i0:i1]
        # mean amplitude per phase bin
        bin_idx = np.digitize(ph_seg, edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins-1)
        A_bin = np.zeros(n_bins)
        for b in range(n_bins):
            vals = amp_seg[bin_idx==b]
            A_bin[b] = np.nanmean(vals) if vals.size else 0.0
        P = A_bin / (A_bin.sum() + 1e-12)
        H = -np.sum(P * np.log(P + 1e-12))
        Hmax = np.log(n_bins)
        mi.append(float((Hmax - H) / (Hmax + 1e-12)))
    return np.asarray(t_mid), np.asarray(mi)

def _pac_mvl_timecourse(X, fs, *,
                       theta_band=(7.0, 8.0),
                       gamma_band=(40.0, 100.0),
                       win_sec=4.0, step_sec=0.25,
                       amp_gate_pct=80):
    # --- safe γ band for fs=128 ---
    gm_hi = min(gamma_band[1], 0.45*fs)
    gm_lo = max(gamma_band[0], 5.0)
    if gm_hi - gm_lo < 5.0:
        c = 0.5*(gm_lo + gm_hi); gm_lo, gm_hi = c-2.5, c+2.5

    b_th = firwin(801, theta_band, pass_zero=False, fs=fs)
    b_gm = firwin(801, (gm_lo, gm_hi), pass_zero=False, fs=fs)

    Xth = filtfilt(b_th, [1.0], X, axis=-1, padlen=min(2400, X.shape[-1]-1))
    Xgm = filtfilt(b_gm, [1.0], X, axis=-1, padlen=min(2400, X.shape[-1]-1))

    th = np.angle(hilbert(Xth, axis=-1))                  # (ch, n)
    gm = np.abs(hilbert(Xgm, axis=-1))                    # (ch, n)

    ph = np.angle(np.nanmean(np.exp(1j*th), axis=0))      # θ phase, across channels
    A  = np.nanmedian(gm, axis=0)                         # γ amplitude, across channels

    # gate by amplitude
    thr = np.nanpercentile(A, amp_gate_pct)
    gate = A >= thr

    n = X.shape[1]
    idx = _sliding_windows(n, fs, win_sec, step_sec)
    t_mid, mvl = [], []
    for i0, i1 in idx:
        w = gate[i0:i1]
        if np.count_nonzero(w) < max(16, int(0.1*(i1-i0))):
            t_mid.append((i0+i1)/2/fs); mvl.append(np.nan); continue
        z = A[i0:i1][w] * np.exp(1j*ph[i0:i1][w])
        mvl.append(np.abs(np.sum(z)) / (np.sum(A[i0:i1][w]) + 1e-12))
        t_mid.append((i0+i1)/2/fs)
    return np.asarray(t_mid), np.asarray(mvl)

def _bicoherence_triads_timecourse(X, fs, triads, bw, win_sec, step_sec):
    n = X.shape[1]
    idx = _sliding_windows(n, fs, win_sec, step_sec)

    # prefilter once per center
    centers = sorted(set([f for tri in triads for f in tri]))
    phases = {}
    for f0 in centers:
        b  = _fir_bandpass(f0, bw, fs)
        padlen = min(3*len(b), X.shape[-1]-1)
        if padlen < 1:
            Xb = np.zeros_like(X)
        else:
            try:
                Xb = filtfilt(b, [1.0], X, axis=-1, padlen=padlen)
            except ValueError:
                Xb = filtfilt(b, [1.0], X, axis=-1, method='gust')
        phases[f0] = np.angle(hilbert(Xb, axis=-1))  # (ch, n)

    t_mid = []
    out = {f"({f1},{f2}->{f3})": [] for (f1,f2,f3) in triads}

    for i0, i1 in idx:
        t_mid.append((i0 + i1)/2 / fs)
        for (f1, f2, f3) in triads:
            # circular mean phase per channel within the window
            p1 = np.angle(np.mean(np.exp(1j*phases[f1][:, i0:i1]), axis=-1))
            p2 = np.angle(np.mean(np.exp(1j*phases[f2][:, i0:i1]), axis=-1))
            p3 = np.angle(np.mean(np.exp(1j*phases[f3][:, i0:i1]), axis=-1))
            R  = np.abs(np.mean(np.exp(1j*(p1 + p2 - p3))))
            out[f"({f1},{f2}->{f3})"].append(R)

    for k in out: out[k] = np.asarray(out[k])
    return np.asarray(t_mid), out

def compute_session_spectrogram(
    _records,
    *,
    channels: Optional[Sequence[str]] = 'auto',
    time_col: str = 'Timestamp',
    fs: Optional[float] = None,
    band: Tuple[float,float] = (2.0, 60.0),
    win_sec: float = 2.0,
    overlap: float = 0.75,
):
    """Return a robust session spectrogram as (t_spec_abs, f_spec, Sxx_med).

    - `t_spec_abs` are absolute seconds aligned to the `time_col` base.
    - `f_spec` are frequencies in Hz (band‑limited to `band`).
    - `Sxx_med` is median power across channels with shape (F, T).
    """
    
    if fs is None:
        fs = _infer_fs(_records, time_col)

    if channels is None or (isinstance(channels, str) and channels.lower() == 'auto'):
        ch = _auto_channels(_records, time_col)
    else:
        ch = list(channels)


    X = _get_matrix(_records, ch)  # (n_ch, n)

    nper = int(round(win_sec * fs))
    nover = int(round(overlap * nper))
    nover = min(nover, nper - 1)

    # Compute per‑channel STFT power, then median across channels
    f, t_rel, S_med = None, None, None
    S_list = []
    for k in range(X.shape[0]):
        f_k, t_k, Z = stft(X[k], fs=fs, window='hann', nperseg=nper, noverlap=nover,
                           detrend='constant', boundary=None, padded=False)
        P = (np.abs(Z) ** 2)  # power spectrum
        if f is None:
            f, t_rel = f_k, t_k
        S_list.append(P)
    S_stack = np.stack(S_list, axis=0)  # (n_ch, F, T)
    S_med = np.nanmedian(S_stack, axis=0)  # (F, T)

    # Band‑limit
    m = (f >= band[0]) & (f <= band[1])
    fB = f[m]
    S_medB = S_med[m, :]

    # Convert time to absolute seconds based on your DataFrame's base time
    t0 = float(np.asarray(_records[time_col], float)[0])
    t_abs = t0 + t_rel

    return (t_abs, fB, S_medB)
 

def build_ignition_feature_pack(_records: pd.DataFrame, windows: List[Tuple[float,float]], *, 
                                cfg: FeaturePackCfg = FeaturePackCfg()) -> Dict[str, np.ndarray]:
    
    _fnd = "{:.2f}".format(cfg.sr_centers[0]) 
    _2nd = "{:.2f}".format(cfg.sr_centers[1])
    _3rd = "{:.2f}".format(cfg.sr_centers[2])


    time_col = cfg.time_col
    fs = cfg.fs or _infer_fs(_records, time_col)
    channels = cfg.channels
    if channels is None or (isinstance(channels, str) and channels.lower() == 'auto'):
        channels = _auto_channels(_records, time_col)
    
    # 0) trim first
    margin = max(2.0, 0.5*cfg.win_sec)                # enough context for sliding windows
    t_all = _records[time_col].to_numpy(float)
    mask = np.zeros_like(t_all, bool)

    # Ensure windows is iterable as list of (start, end) tuples
    if isinstance(windows, tuple) and len(windows) == 2 and isinstance(windows[0], (int, float)):
        # Single window passed as tuple (start, end)
        windows = [windows]

    for window in windows:
        a, b = window
        mask |= (t_all >= a - margin) & (t_all <= b + margin)

    t = t_all[mask]                                   # segment time base
    X = _get_matrix(_records, channels)[:, mask]

    # Handle bw_hz as either scalar or array
    # If array, extract bandwidths for the first 3 sr_centers from the ladder
    if np.ndim(cfg.bw_hz) == 0:  # scalar
        bw1 = bw2 = bw3 = float(cfg.bw_hz)
    else:  # array - use first 3 elements for the 3 SR centers
        bw_array = np.asarray(cfg.bw_hz)
        if len(bw_array) < 3:
            raise ValueError(f'bw_hz array must have at least 3 elements for sr_centers, got {len(bw_array)}')
        bw1, bw2, bw3 = bw_array[0], bw_array[1], bw_array[2]

    # Fundamental & harmonics envelopes (median across channels)
    z1, ph1 = _narrowband_envelope_z(X, fs, cfg.sr_centers[0], bw1)
    z2, ph2 = _narrowband_envelope_z(X, fs, cfg.sr_centers[1], bw2)
    z3, ph3 = _narrowband_envelope_z(X, fs, cfg.sr_centers[2], bw3)

    # PLV@7.8 over time (sliding)
    t_plv, plv = _plv_timecourse(X, fs, cfg.sr_centers[0], bw1, cfg.win_sec, cfg.step_sec)

    # HSI over time
    t_hsi, hsi = _hsi_timecourse(X, fs, cfg.win_sec, cfg.step_sec, cfg.ladder, ladder_bw=cfg.ladder_bw)

    # Focused bicoherence triads and PAC (optional but cheap here)
    t_bic, bic = _bicoherence_triads_timecourse(
        X, fs,
        triads=[(cfg.sr_centers[0], cfg.sr_centers[0], cfg.sr_centers[1]),
                (cfg.sr_centers[0], cfg.sr_centers[1], cfg.sr_centers[2])],
        bw=bw1, win_sec=cfg.win_sec, step_sec=cfg.step_sec,
    )

    t_pac, mvl = _pac_mvl_timecourse(X, fs, theta_band=(7.0,8.0), gamma_band=(40.0,100),
                                     win_sec=cfg.win_sec, step_sec=cfg.step_sec)

    # t_abs = np.asarray(_records[time_col], float)
    # t0 = float(t_abs[0])

    # 2) map sliding times to absolute **segment** seconds
    t_seg0 = float(t[0])
    def to_abs(ts, ys): return np.interp(t, t_seg0 + ts, ys, left=ys[0], right=ys[-1])

    # In build_ignition_feature_pack
    plv_series = _plv_7p8(X, fs, f0=cfg.sr_centers[0], bw=bw1, win=cfg.win_sec, step=cfg.step_sec)    

    # Interpolate sliding metrics back to raw time base for simplicity
    def interp_to_raw(t_src, y_src):
        return np.interp(t,t_seg0+t_src, y_src, left=y_src[0], right=y_src[-1])

    
    pack = {
        't': t,
        'z_7p83': z1,
        'z_15p6': z2,
        'z_23p4': z3,
        'plv_7p83': interp_to_raw(t_plv, plv),
        'hsi': interp_to_raw(t_hsi, hsi),
        'bico_7_7_15': interp_to_raw(t_bic, bic[f"({_fnd},{_fnd}->{_2nd})"] if f"({_fnd},{_2nd}->{_3rd})" in bic else list(bic.values())[0]),
        'bico_7_15_23': interp_to_raw(t_bic, bic.get(f"({_fnd},{_2nd}->{_3rd})", list(bic.values())[-1])),
        'pac_mvl': interp_to_raw(t_pac, mvl),
        # (optional) include spec tuple if you already compute one elsewhere
    }

    return pack

def robust_z(x):
    x = np.asarray(x, float); med = np.nanmedian(x); mad = np.nanmedian(np.abs(x-med)) + 1e-12
    return (x - med) / (1.4826 * mad)

def smooth_sec(t, y, sec=0.15):
    t = np.asarray(t, float); y = np.asarray(y, float)
    k = max(1, int(round(sec/np.median(np.diff(t))))); 
    return np.convolve(y, np.ones(k)/k, mode='same') if k>1 else y


# ---------------- main detector ----------------


def _detect_ignition_phases(
    t: np.ndarray,
    z_fund: np.ndarray,
    plv_fund: np.ndarray,
    hsi_t: np.ndarray,
    z_h2: np.ndarray,
    z_h3: np.ndarray,
    *,
    beta_t: Optional[np.ndarray] = None,
    ridge_is_fund: Optional[np.ndarray] = None,
    bic_7_7_15: Optional[np.ndarray] = None,
    bic_7_15_23: Optional[np.ndarray] = None,
    pac_mvl: Optional[np.ndarray] = None,
    params=None,
    seed_t: float | str | None = "center",
    p0_band = (-2.0, +0.6),
    p1_band = (-0.6, +1.2),
    pad_s: float = 2.0,
    return_debug: bool = False,
):
    """Phase-aware ignition detector returning P0-P3 events and confidences."""
    assert params is not None, "params required"

    t = np.asarray(t, float)
    zf = np.asarray(z_fund, float)
    plv = np.asarray(plv_fund, float)
    hsi = np.asarray(hsi_t, float)
    z2 = np.asarray(z_h2, float)
    z3 = np.asarray(z_h3, float)
    n = t.size
    assert n == plv.size == hsi.size == z2.size == z3.size == zf.size, "length mismatch"
    if n == 0:
        raise ValueError("ignition window slice contains no samples")

    def _np_percentile(arr: np.ndarray, p: float, default: float = np.nan) -> float:
        arr = np.asarray(arr, float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return default
        return float(np.nanpercentile(arr, p))

    def _robust_z(arr: np.ndarray, idx: Optional[int]) -> float:
        if idx is None or idx < 0 or idx >= arr.size:
            return 0.0
        vals = arr[np.isfinite(arr)]
        if vals.size < 3:
            return 0.0
        med = np.nanmedian(vals)
        mad = np.nanmedian(np.abs(vals - med))
        mad = max(1e-6, 1.4826 * mad)
        return float((arr[idx] - med) / mad)

    def _sigmoid(x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-x)))

    dt = float(np.median(np.diff(t))) if n > 1 else 1.0

    def _first_run(mask: np.ndarray, min_sec: float, start_idx: int = 0) -> Optional[Tuple[int, int]]:
        """Return the first contiguous run of `True` values long enough in seconds."""
        arr = np.asarray(mask, bool)
        total = arr.size
        if total == 0:
            return None
        min_len = max(1, int(np.ceil(min_sec / max(dt, 1e-6))))
        i = max(0, int(start_idx))
        if i >= total:
            return None
        while i < total:
            if not arr[i]:
                i += 1
                continue
            j = i
            while j < total and arr[j]:
                j += 1
            if j - i >= min_len:
                return i, j - 1
            i = j
        return None

    def _gauss_smooth(y: np.ndarray, sigma_sec: float = 0.5) -> np.ndarray:
        if n < 3:
            return y.astype(float)
        sigma = max(sigma_sec / max(dt, 1e-6), 1.0)
        radius = int(np.ceil(3 * sigma))
        if radius <= 1:
            return y.astype(float)
        kernel_x = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-0.5 * (kernel_x / sigma) ** 2)
        kernel /= kernel.sum()
        padded = np.pad(y.astype(float), radius, mode='edge')
        return np.convolve(padded, kernel, mode='same')[radius:-radius]

    z_sr = _gauss_smooth(zf)
    plv_s = _gauss_smooth(plv)
    hsi_s = _gauss_smooth(hsi, 0.75)
    dplv = _gauss_smooth(np.gradient(plv_s, t), 0.4)
    abs_hsi = np.abs(hsi_s - np.nanmedian(hsi_s))
    dhsi = _gauss_smooth(np.gradient(hsi_s, t), 0.4)
    dabs_hsi = _gauss_smooth(np.gradient(abs_hsi, t), 0.4)

    bic_1 = np.zeros_like(z_sr)
    bic_2 = np.zeros_like(z_sr)
    if bic_7_7_15 is not None:
        bic_1 = _gauss_smooth(np.asarray(bic_7_7_15, float), 0.5)
    if bic_7_15_23 is not None:
        bic_2 = _gauss_smooth(np.asarray(bic_7_15_23, float), 0.5)
    bic_max = np.nanmax(np.vstack([bic_1, bic_2]), axis=0)

    mvl = np.zeros_like(z_sr)
    if pac_mvl is not None:
        mvl = _gauss_smooth(np.asarray(pac_mvl, float), 0.75)

    q = {
        'z70': _np_percentile(z_sr, 70.0, 0.0),
        'plv75': _np_percentile(plv_s, 75.0, 0.0),
        'plv60': _np_percentile(plv_s, 60.0, 0.0),
        'plv90': _np_percentile(plv_s, 90.0, 0.0),
        'dplv80': _np_percentile(dplv, 80.0, 0.0),
        'bic60': _np_percentile(bic_max, 60.0, 0.0),
        'bic85': _np_percentile(bic_max, 85.0, 0.0),
        'mvl85': _np_percentile(mvl, 85.0, 0.0),
        'abs_hsi70': _np_percentile(abs_hsi, 70.0, 0.0),
        'abs_hsi30': _np_percentile(abs_hsi, 30.0, 0.0),
    }

    z_norm = _gauss_smooth(_winsor_robust_z(z_sr), 0.4)
    plv_norm = _gauss_smooth(_winsor_robust_z(plv_s), 0.4)
    bic_norm = _winsor_robust_z(bic_max)
    mvl_norm = _winsor_robust_z(mvl)
    score = (0.4 * z_norm + 0.3 * plv_norm + 0.2 * bic_norm + 0.1 * mvl_norm)
    score = np.nan_to_num(score, nan=0.0)

    score_thresh = _np_percentile(score, 85.0, 0.0)
    candidate_idxs = [i for i in range(1, n-1) if score[i] >= score_thresh and score[i] >= score[i-1] and score[i] >= score[i+1]]
    if not candidate_idxs:
        candidate_idxs = [int(np.nanargmax(score))]
    idx_core = max(candidate_idxs, key=lambda i: score[i])

    search_radius = max(1, int(np.ceil(0.6 / max(dt, 1e-6))))
    lo = max(0, idx_core - search_radius)
    hi = min(n - 1, idx_core + search_radius)
    idx_p2 = int(np.nanargmax(bic_max[lo:hi+1]) + lo)
    p2 = {'time': float(t[idx_p2]), 'idx': idx_p2, 'confidence': 0.0, 'label': 'P2'}

    plateau_thresh = q['plv60'] + 0.4 * max(0.0, q['plv75'] - q['plv60'])
    min_p1_samples = max(1, int(np.ceil(0.75 / max(dt, 1e-6))))
    idx_min = max(0, idx_p2 - int(np.ceil(2.0 / max(dt, 1e-6))))
    mask_plateau = plv_s >= plateau_thresh
    idx_p1 = None
    i = min(idx_p2, n - 1)
    while i >= idx_min:
        if mask_plateau[i]:
            j = i
            while j >= idx_min and mask_plateau[j]:
                j -= 1
            start = j + 1
            if i - start + 1 >= min_p1_samples:
                idx_p1 = start
                break
            i = j
        else:
            i -= 1
    if idx_p1 is None:
        idx_p1 = int(np.nanargmax(plv_s[idx_min:idx_p2+1]) + idx_min)
    p1 = {'time': float(t[idx_p1]), 'idx': idx_p1, 'confidence': 0.0, 'label': 'P1'}

    mask_p0 = (z_sr >= q['z70']) & (dplv >= q['dplv80']) & (bic_max >= q['bic60'])
    idx_min_p0 = max(0, idx_p1 - int(np.ceil(1.5 / max(dt, 1e-6))))
    min_p0_samples = max(1, int(np.ceil(0.4 / max(dt, 1e-6))))
    idx_p0 = None
    i = idx_p1
    while i >= idx_min_p0:
        if mask_p0[i]:
            j = i
            while j >= idx_min_p0 and mask_p0[j]:
                j -= 1
            start = j + 1
            if i - start + 1 >= min_p0_samples:
                idx_p0 = start
                break
            i = j
        else:
            i -= 1
    if idx_p0 is None:
        search_end = idx_p1
        search_start = max(0, search_end - int(np.ceil(1.5 / max(dt, 1e-6))))
        fallback_slice = slice(search_start, search_end + 1)
        combo = z_norm[fallback_slice] + plv_norm[fallback_slice] + bic_norm[fallback_slice]
        rel_idx = int(np.nanargmax(combo)) if combo.size else 0
        idx_p0 = search_start + rel_idx
    p0 = {'time': float(t[idx_p0]), 'idx': idx_p0, 'confidence': 0.0, 'label': 'P0'}

    horizon_p3 = int(np.ceil(2.5 / max(dt, 1e-6)))
    idx_max_p3 = min(n - 1, idx_p2 + horizon_p3)
    mv_mask = mvl >= q['mvl85']
    idx_p3 = None
    for i in range(idx_p2 + 1, idx_max_p3):
        if mv_mask[i] and mvl[i] >= mvl[i-1] and mvl[i] >= mvl[i+1] and plv_s[i] >= q['plv60']:
            idx_p3 = i
            break
    if idx_p3 is None:
        segment = plv_s[idx_p2+1:idx_max_p3+1]
        if segment.size and np.any(np.isfinite(segment)):
            idx_p3 = int(np.nanargmax(segment) + idx_p2 + 1)
        else:
            idx_p3 = idx_max_p3
    p3 = {'time': float(t[idx_p3]), 'idx': idx_p3, 'confidence': 0.0, 'label': 'P3'}

    # Release (P4)
    release_start = idx_p3
    release_span = int(np.ceil(2.0 / max(dt, 1e-6)))
    idx_max_p4 = min(n - 1, release_start + release_span)
    release_thresh = q['plv60']
    release_mask = (plv_s <= release_thresh) & (dhsi >= 0)
    run_release = _first_run(release_mask, 0.5, release_start + 1)
    if run_release:
        idx_p4 = run_release[0]
    else:
        segment = plv_s[release_start+1:idx_max_p4+1]
        if segment.size:
            idx_p4 = int(np.nanargmin(segment) + release_start + 1)
        else:
            idx_p4 = idx_max_p4
    p4 = {'time': float(t[idx_p4]), 'idx': idx_p4, 'confidence': 0.0, 'label': 'P4'}

    window_start = float(t[0])
    window_end = float(t[-1])
    def _snap_event(ev: Dict[str, Any]):
        time = ev.get('time')
        if time is None:
            return
        if time < window_start or time > window_end:
            clipped = float(np.clip(time, window_start, window_end))
            idx = int(np.clip(np.searchsorted(t, clipped), 0, n - 1))
            ev['idx'] = idx
            ev['time'] = float(t[idx])

    events = [p0, p1, p2, p3, p4]
    for ev in events:
        _snap_event(ev)

    last_time = -np.inf
    for ev in events:
        time = ev['time']
        if time is None:
            continue
        if time <= last_time:
            time = min(window_end, last_time + max(dt, 0.05))
            idx = int(np.clip(np.searchsorted(t, time), 0, n - 1))
            ev['idx'] = idx
            ev['time'] = float(t[idx])
        last_time = ev['time']

    p0['confidence'] = _sigmoid((
        _robust_z(z_sr, p0['idx']) +
        _robust_z(dplv, p0['idx']) +
        _robust_z(bic_max, p0['idx'])
    ) / 3.0)

    tightening_penalty = 0.5 if abs_hsi[p1['idx']] < q['abs_hsi30'] else 0.0
    p1['confidence'] = _sigmoid(_robust_z(plv_s, p1['idx']) - tightening_penalty)

    p2['confidence'] = _sigmoid((
        _robust_z(bic_max, p2['idx']) +
        _robust_z(abs_hsi, p2['idx']) - 0.5 * abs(_robust_z(dabs_hsi, p2['idx']))
    ) / 2.0)

    p3['confidence'] = _sigmoid((
        _robust_z(mvl, p3['idx']) +
        _robust_z(plv_s, p3['idx'])
    ) / 2.0)

    p4['confidence'] = _sigmoid((
        -_robust_z(plv_s, p4['idx']) +
        _robust_z(dhsi, p4['idx'])
    ) / 2.0)

    event_type = 'undefined'
    if p1['time'] is None and p2['time'] is not None and p3['time'] is not None:
        event_type = 'two-phase'
    elif p1['idx'] is not None:
        tightening = hsi_s[p1['idx']] - np.nanmedian(hsi_s)
        event_type = 'fundamental-led' if tightening <= 0 else 'overtone-led'

    summary = {
        'P0': p0,
        'P1': p1,
        'P2': p2,
        'P3': p3,
        'P4': p4,
        'type': event_type,
        'confidence_mean': float(np.nanmean([p0['confidence'], p1['confidence'], p2['confidence'], p3['confidence'], p4['confidence']]))
    }

    if not return_debug:
        return summary

    debug = {
        't': t,
        'z_sr': z_sr,
        'plv_s': plv_s,
        'dplv': dplv,
        'hsi_s': hsi_s,
        'dhsi': dhsi,
        'bic_max': bic_max,
        'mvl': mvl,
        'abs_hsi': abs_hsi,
        'thresholds': {**q, 'release_plv': release_thresh},
        'score': score,
        'events': summary,
    }
    return summary, debug


def _detect_six_phase_evolution(
    t: np.ndarray,
    z_fund: np.ndarray,
    plv_fund: np.ndarray,
    R_kuramoto: np.ndarray,
    pac_mvl: np.ndarray,
    bic_max: np.ndarray,
    z_h2: np.ndarray,
    z_h3: np.ndarray,
    *,
    t0_net: Optional[float] = None,
    window_start: float = 0.0,
    return_debug: bool = False,
) -> Dict[str, Any]:
    """
    Six-phase temporal evolution detection for SR ignitions.

    Phases:
    1. Baseline: Desynchronized state before ignition
    2. Coherence-First: PLV rises BEFORE amplitude
    3. Amplitude Surge: Explosive z-score growth + harmonic emergence
    4. Peak/Plateau: Maximum synchronization, stable oscillation
    5. Propagation: Posterior detection with delayed PAC surge
    6. Decay: Exponential return to baseline
    """
    t = np.asarray(t, float)
    zf = np.asarray(z_fund, float)
    plv = np.asarray(plv_fund, float)
    R = np.asarray(R_kuramoto, float)
    pac = np.asarray(pac_mvl, float) if pac_mvl is not None else np.zeros_like(zf)
    bic = np.asarray(bic_max, float)
    z2 = np.asarray(z_h2, float)
    z3 = np.asarray(z_h3, float)

    n = t.size
    if n == 0:
        raise ValueError("Empty time array")

    dt = float(np.median(np.diff(t))) if n > 1 else 0.01

    def _smooth(y: np.ndarray, sigma_sec: float = 0.3) -> np.ndarray:
        if n < 3:
            return y.astype(float)
        sigma = max(sigma_sec / max(dt, 1e-6), 1.0)
        radius = int(np.ceil(3 * sigma))
        if radius <= 1:
            return y.astype(float)
        kernel_x = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-0.5 * (kernel_x / sigma) ** 2)
        kernel /= kernel.sum()
        padded = np.pad(y.astype(float), radius, mode='edge')
        return np.convolve(padded, kernel, mode='same')[radius:-radius]

    # Smooth signals
    zf_s = _smooth(zf, 0.3)
    plv_s = _smooth(plv, 0.3)
    R_s = _smooth(R, 0.3)
    pac_s = _smooth(pac, 0.5)
    bic_s = _smooth(bic, 0.3)

    # Derivatives
    dplv = np.gradient(plv_s, t)
    dzf = np.gradient(zf_s, t)

    # Determine t0 (ignition onset) using robust multi-method detection
    if t0_net is None:
        # Define constrained search window (exclude edges and decay tail)
        t_range = t[-1] - t[0]
        search_start_idx = max(0, int(2.0 / max(dt, 1e-6)))  # Skip first 2s
        search_end_idx = min(n - 1, int((t_range * 0.7) / max(dt, 1e-6)))  # Search only first 70%
        search_mask = np.zeros(n, dtype=bool)
        search_mask[search_start_idx:search_end_idx] = True

        # Method 1: PLV derivative (primary - works for coherence-first ignitions)
        dplv_search = dplv.copy()
        dplv_search[~search_mask] = -np.inf
        idx_t0_plv = int(np.argmax(dplv_search))
        t0_plv = float(t[idx_t0_plv])

        # Method 2: Amplitude derivative (fallback - works for amplitude-first ignitions)
        dzf_search = dzf.copy()
        dzf_search[~search_mask] = -np.inf
        idx_t0_amp = int(np.argmax(dzf_search))
        t0_amp = float(t[idx_t0_amp])

        # Method 3: Combined score (PLV + amplitude rise)
        dplv_max = np.nanmax(np.abs(dplv_search))
        dzf_max = np.nanmax(np.abs(dzf_search))

        # Avoid division by zero or inf
        if np.isfinite(dplv_max) and dplv_max > 1e-9 and np.isfinite(dzf_max) and dzf_max > 1e-9:
            combined_score = (dplv_search / dplv_max + dzf_search / dzf_max)
            combined_score[~search_mask] = -np.inf  # Ensure masked regions stay masked
            idx_t0_combined = int(np.argmax(combined_score))
        else:
            # Fallback if normalization fails
            idx_t0_combined = idx_t0_plv

        t0_combined = float(t[idx_t0_combined])

        # Cross-validate candidates
        def _validate_t0(idx_candidate):
            """Check if candidate t0 has rising PLV and amplitude, and peak occurs after."""
            if idx_candidate < 10 or idx_candidate >= n - 10:
                return 0.0  # Too close to edge

            # Check for rising signals after t0 (next 3s)
            check_window = slice(idx_candidate, min(n, idx_candidate + int(3.0 / max(dt, 1e-6))))
            plv_rising = np.mean(dplv[check_window]) > 0
            amp_rising = np.mean(dzf[check_window]) > 0

            # Check that peak occurs after t0 (within 1-10s)
            peak_idx = int(np.argmax(zf_s))
            peak_delay = t[peak_idx] - t[idx_candidate]
            peak_reasonable = 1.0 <= peak_delay <= 10.0

            # Compute validation score
            score = 0.0
            if plv_rising: score += 0.3
            if amp_rising: score += 0.4
            if peak_reasonable: score += 0.3

            return score

        scores = {
            'plv': _validate_t0(idx_t0_plv),
            'amp': _validate_t0(idx_t0_amp),
            'combined': _validate_t0(idx_t0_combined)
        }

        # Store all candidates for debugging
        t0_detection_debug = {
            'candidates': {
                'plv': {'t0': t0_plv, 'score': scores['plv']},
                'amp': {'t0': t0_amp, 'score': scores['amp']},
                'combined': {'t0': t0_combined, 'score': scores['combined']}
            },
            'search_window': (float(t[search_start_idx]), float(t[search_end_idx]))
        }

        # Select best candidate
        best_method = max(scores, key=scores.get)
        if best_method == 'plv':
            idx_t0 = idx_t0_plv
            t0 = t0_plv
        elif best_method == 'amp':
            idx_t0 = idx_t0_amp
            t0 = t0_amp
        else:
            idx_t0 = idx_t0_combined
            t0 = t0_combined

        # Last resort: if all scores are very low, estimate from peak
        if scores[best_method] < 0.3:
            peak_idx = int(np.argmax(zf_s))
            # Estimate t0 as 3s before peak (typical ignition duration)
            idx_t0 = max(0, peak_idx - int(3.0 / max(dt, 1e-6)))
            t0 = float(t[idx_t0])
            best_method = 'peak_backtrack'
            t0_detection_debug['peak_backtrack_used'] = True
            t0_detection_debug['peak_location'] = float(t[peak_idx])

        # Print diagnostic info (unconditional - helps debug phase detection issues)
        abs_t0 = window_start + t0 if window_start is not None else t0
        peak_idx = int(np.argmax(zf_s))
        peak_t = window_start + t[peak_idx] if window_start is not None else t[peak_idx]
        # print(f"\n=== t0 Detection Debug ===")
        # print(f"Method: {best_method}")
        # print(f"Detected t0: {abs_t0:.2f}s (relative: {t0:.2f}s)")
        # print(f"Peak location: {peak_t:.2f}s")
        # print(f"Candidate scores: PLV={scores['plv']:.3f}, AMP={scores['amp']:.3f}, COMB={scores['combined']:.3f}")
        # print(f"Candidates: PLV@{t0_plv:.1f}s, AMP@{t0_amp:.1f}s, COMB@{t0_combined:.1f}s")
        # print(f"Search window: {t[search_start_idx]:.1f}s - {t[search_end_idx]:.1f}s")
        # print("="*26 + "\n")

    else:
        # Validate provided t0_net
        t0_provided = float(t0_net)
        idx_t0_provided = int(np.argmin(np.abs(t - t0_provided)))

        # Check if provided t0 is reasonable
        peak_idx = int(np.argmax(zf_s))
        peak_t = float(t[peak_idx])

        # t0 should be before the peak (within reasonable range: 1-10s before)
        time_to_peak = peak_t - t0_provided
        t0_is_valid = (1.0 <= time_to_peak <= 10.0)

        if t0_is_valid:
            # Use provided t0
            t0 = t0_provided
            idx_t0 = idx_t0_provided
            best_method = 'provided'
            t0_detection_debug = {'method': 'provided', 't0_net': t0_net, 'validated': True}

            abs_t0 = window_start + t0 if window_start is not None else t0
            abs_peak_t = window_start + peak_t if window_start is not None else peak_t
            # print(f"\n=== t0 Detection Debug ===")
            # print(f"Method: provided (t0_net) ✓ VALID")
            # print(f"Provided t0: {abs_t0:.2f}s (relative: {t0:.2f}s)")
            # print(f"Peak location: {abs_peak_t:.2f}s")
            # print(f"t0 is {time_to_peak:.2f}s before peak")
            # print("="*26 + "\n")
        else:
            # Reject invalid t0_net and fall back to auto-detection
            abs_t0_provided = window_start + t0_provided if window_start is not None else t0_provided
            abs_peak_t = window_start + peak_t if window_start is not None else peak_t
            # print(f"\n=== t0 Detection Debug ===")
            # print(f"⚠️  INVALID t0_net REJECTED: {abs_t0_provided:.2f}s")
            # print(f"   Reason: t0 is {time_to_peak:.2f}s from peak (expected 1-10s before)")
            # print(f"   Peak location: {abs_peak_t:.2f}s")
            # print(f"   Falling back to auto-detection...")

            # Fall back to auto-detection (copy the auto-detection logic)
            # Define constrained search window
            t_range = t[-1] - t[0]
            search_start_idx = max(0, int(2.0 / max(dt, 1e-6)))
            search_end_idx = min(n - 1, int((t_range * 0.7) / max(dt, 1e-6)))

            # Use peak backtrack as the most reliable fallback
            idx_t0 = max(0, peak_idx - int(3.0 / max(dt, 1e-6)))
            t0 = float(t[idx_t0])
            best_method = 'peak_backtrack_fallback'
            t0_detection_debug = {
                'method': 'peak_backtrack_fallback',
                'rejected_t0_net': t0_net,
                'reason': f't0 was {time_to_peak:.2f}s from peak',
                'peak_location': peak_t
            }

            abs_t0 = window_start + t0 if window_start is not None else t0
            # print(f"   Using: peak_backtrack → t0={abs_t0:.2f}s (peak - 3s)")
            # print("="*26 + "\n")

    def _find_phase_boundary(start_idx: int, direction: int, criterion_fn, min_duration_sec: float = 0.5) -> Optional[int]:
        """Find phase boundary by searching in direction until criterion met."""
        min_samples = max(1, int(min_duration_sec / max(dt, 1e-6)))
        idx = start_idx
        count = 0

        while 0 <= idx < n:
            if criterion_fn(idx):
                count += 1
                if count >= min_samples:
                    return idx - (count - 1) * direction  # Return start of run
            else:
                count = 0
            idx += direction
        return None

    # === PHASE 1: Baseline ===
    # Ends when PLV starts rising above baseline
    plv_baseline = np.nanpercentile(plv_s[:idx_t0], 50.0) if idx_t0 > 10 else 0.5
    p1_end_idx = _find_phase_boundary(
        idx_t0, -1,
        lambda i: plv_s[i] <= plv_baseline * 1.1 and zf_s[i] < 2.0,
        min_duration_sec=0.3
    )
    if p1_end_idx is None:
        p1_end_idx = max(0, idx_t0 - int(10.0 / max(dt, 1e-6)))

    p1_start = float(t[0])
    p1_end = float(t[p1_end_idx])

    # === PHASE 2: Coherence-First ===
    # PLV rises before amplitude (t0 to ~3s after)
    p2_end_idx = _find_phase_boundary(
        idx_t0, +1,
        lambda i: zf_s[i] >= 3.0 or (z2[i] > 2.0 and z3[i] > 2.0),  # Amplitude surge or harmonics
        min_duration_sec=0.3
    )
    if p2_end_idx is None:
        p2_end_idx = min(n - 1, idx_t0 + int(3.0 / max(dt, 1e-6)))

    p2_start = p1_end
    p2_end = float(t[p2_end_idx])

    # === PHASE 3: Amplitude Surge ===
    # Explosive growth from z=3 to peak
    # Find center of peak region (constrained to AFTER Phase 2)
    # Search only in the region after Phase 2 ends to maintain temporal ordering
    search_after_idx = max(p2_end_idx, 0)
    zf_after_p2 = zf_s[search_after_idx:]

    if zf_after_p2.size > 0:
        peak_z_after = float(np.nanmax(zf_after_p2))
        peak_region_rel = np.where(zf_after_p2 >= peak_z_after * 0.95)[0]  # Indices within 5% of peak
        if peak_region_rel.size > 0:
            p3_end_idx_rel = int(peak_region_rel[len(peak_region_rel) // 2])  # Center of peak region
            p3_end_idx = search_after_idx + p3_end_idx_rel
        else:
            p3_end_idx = search_after_idx + int(np.argmax(zf_after_p2))
    else:
        # No data after Phase 2 - set to end of Phase 2
        p3_end_idx = p2_end_idx

    # Use the peak found in the constrained region for Phase 4 threshold calculation
    peak_z = float(zf_s[p3_end_idx])

    p3_start = p2_end
    p3_end = float(t[p3_end_idx])

    # === PHASE 4: Peak/Plateau ===
    # Sustained high amplitude and synchronization
    # (peak_z already calculated above)
    plateau_thresh = peak_z * 0.85  # Within 15% of peak

    p4_end_idx = _find_phase_boundary(
        p3_end_idx, +1,
        lambda i: zf_s[i] < plateau_thresh,  # Amplitude dropping
        min_duration_sec=0.5
    )
    if p4_end_idx is None:
        p4_end_idx = min(n - 1, p3_end_idx + int(7.0 / max(dt, 1e-6)))

    p4_start = p3_end
    p4_end = float(t[p4_end_idx])

    # === PHASE 5: Propagation ===
    # PAC surge at propagation sites (if detectable in averaged signal)
    # Look for PAC peak after amplitude peak
    pac_after_peak = pac_s[p3_end_idx:]
    if pac_after_peak.size > 0:
        pac_peak_rel = int(np.argmax(pac_after_peak))
        p5_peak_idx = p3_end_idx + pac_peak_rel
    else:
        p5_peak_idx = p4_end_idx

    p5_end_idx = min(n - 1, p5_peak_idx + int(5.0 / max(dt, 1e-6)))

    p5_start = p4_end
    p5_end = float(t[p5_end_idx])

    # === PHASE 6: Decay ===
    # Exponential decay back to baseline
    p6_start = p5_end
    p6_end = float(t[-1])

    # === TEMPORAL ORDERING VALIDATION ===
    # Ensure all phases are temporally ordered and have minimum duration
    # If violations are detected, clamp boundaries to maintain sequence
    min_duration = 0.1  # seconds

    # Validate and fix Phase 1
    if p1_end - p1_start < min_duration:
        p1_end = min(p1_start + min_duration, t[-1])

    # Validate and fix Phase 2
    if p2_start < p1_end:
        p2_start = p1_end
    if p2_end - p2_start < min_duration:
        p2_end = min(p2_start + min_duration, t[-1])

    # Validate and fix Phase 3
    if p3_start < p2_end:
        p3_start = p2_end
    if p3_end - p3_start < min_duration:
        p3_end = min(p3_start + min_duration, t[-1])

    # Validate and fix Phase 4
    if p4_start < p3_end:
        p4_start = p3_end
    if p4_end - p4_start < min_duration:
        p4_end = min(p4_start + min_duration, t[-1])

    # Validate and fix Phase 5
    if p5_start < p4_end:
        p5_start = p4_end
    if p5_end - p5_start < min_duration:
        p5_end = min(p5_start + min_duration, t[-1])

    # Validate and fix Phase 6
    if p6_start < p5_end:
        p6_start = p5_end
    if p6_end - p6_start < min_duration:
        p6_end = p6_start + min_duration  # Phase 6 goes to end, so just ensure minimum

    # Compute confidence scores
    def _sigmoid(x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-x)))

    def _phase_confidence(start_idx: int, end_idx: int, criteria: dict) -> float:
        """Compute confidence based on signal characteristics in phase."""
        if end_idx <= start_idx:
            return 0.0

        seg_slice = slice(start_idx, end_idx + 1)
        scores = []

        if 'plv_range' in criteria:
            plv_mean = float(np.nanmean(plv_s[seg_slice]))
            lo, hi = criteria['plv_range']
            if lo <= plv_mean <= hi:
                scores.append(1.0)
            else:
                scores.append(0.3)

        if 'z_range' in criteria:
            z_mean = float(np.nanmean(zf_s[seg_slice]))
            lo, hi = criteria['z_range']
            if lo <= z_mean <= hi:
                scores.append(1.0)
            else:
                scores.append(0.3)

        if 'plv_rising' in criteria and criteria['plv_rising']:
            dplv_mean = float(np.nanmean(dplv[seg_slice]))
            scores.append(_sigmoid(dplv_mean * 50))  # Positive slope = high confidence

        if 'z_rising' in criteria and criteria['z_rising']:
            dzf_mean = float(np.nanmean(dzf[seg_slice]))
            scores.append(_sigmoid(dzf_mean * 20))

        return float(np.mean(scores)) if scores else 0.5

    # Get indices for confidence calculation
    def _time_to_idx(time: float) -> int:
        return int(np.clip(np.searchsorted(t, time), 0, n - 1))

    phases = {
        'Phase1': {
            'name': 'Baseline',
            'time_start': p1_start,
            'time_end': p1_end,
            'label': 'Phase1',
            'color': '#9E9E9E',
            'confidence': _phase_confidence(_time_to_idx(p1_start), _time_to_idx(p1_end),
                                           {'plv_range': (0.3, 0.65), 'z_range': (0, 2.5)})
        },
        'Phase2': {
            'name': 'Coherence',
            'time_start': p2_start,
            'time_end': p2_end,
            'label': 'Phase2',
            'color': '#4CAF50',  # Green
            'confidence': _phase_confidence(_time_to_idx(p2_start), _time_to_idx(p2_end),
                                           {'plv_rising': True, 'z_range': (1.5, 3.5)})
        },
        'Phase3': {
            'name': 'Ignition',
            'time_start': p3_start,
            'time_end': p3_end,
            'label': 'Phase3',
            'color': '#FF6F00',  # Orange
            'confidence': _phase_confidence(_time_to_idx(p3_start), _time_to_idx(p3_end),
                                           {'z_rising': True, 'z_range': (3.0, 15.0)})
        },
        'Phase4': {
            'name': 'Plateau',
            'time_start': p4_start,
            'time_end': p4_end,
            'label': 'Phase4',
            'color': '#F44336',  # Red
            'confidence': _phase_confidence(_time_to_idx(p4_start), _time_to_idx(p4_end),
                                           {'plv_range': (0.75, 1.0), 'z_range': (5.0, 15.0)})
        },
        'Phase5': {
            'name': 'Propagation',
            'time_start': p5_start,
            'time_end': p5_end,
            'label': 'Phase5',
            'color': '#9C27B0',  # Purple
            'confidence': 0.6  # Harder to detect in averaged signals
        },
        'Phase6': {
            'name': 'Decay',
            'time_start': p6_start,
            'time_end': p6_end,
            'label': 'Phase6',
            'color': '#9E9E9E',  # Gray (same as Baseline)
            'confidence': _phase_confidence(_time_to_idx(p6_start), _time_to_idx(p6_end),
                                           {'z_range': (0, 5.0)})
        }
    }

    summary = {
        'phase_model': 'six-phase-evolution',
        't0': t0,
        't0_detection_method': best_method,  # Track which method was used for t0 detection
        't0_detection_debug': t0_detection_debug,  # Full debug info about t0 detection
        'phases': phases
    }

    if return_debug:
        return summary, {
            'plv_s': plv_s, 'zf_s': zf_s, 'R_s': R_s, 'pac_s': pac_s,
            'dplv': dplv, 'dzf': dzf
        }

    return summary


def annotate_phases(ax, phases: Dict[str, Any], ymin: float, ymax: float,
                    *, highlight_padding: float = 0.25, show_labels: bool = True, show_shading: bool = True) -> None:
    """
    Annotate phases on plot. Supports both old 5-phase (P0-P4) and new 6-phase models.

    Args:
        ax: Matplotlib axis to annotate
        phases: Phase dictionary with timing and metadata
        ymin, ymax: Y-axis limits for annotation positioning
        highlight_padding: Padding for legacy phase highlighting
        show_labels: If False, only draw lines and shading (no text labels)
        show_shading: If False, only draw lines (no shading)
    """
    # Detect phase model type
    phase_model = phases.get('phase_model', 'five-phase')

    if phase_model == 'six-phase-evolution':
        _annotate_six_phases(ax, phases, ymin, ymax, show_labels=show_labels, show_shading=show_shading)
    else:
        _annotate_five_phases_legacy(ax, phases, ymin, ymax, highlight_padding=highlight_padding, show_labels=show_labels, show_shading=show_shading)


def _annotate_six_phases(ax, phases: Dict[str, Any], ymin: float, ymax: float, show_labels: bool = True, show_shading: bool = True) -> None:
    """Annotate six-phase temporal evolution model."""
    phase_data = phases.get('phases', {})

    # Validate that phases are temporally separated (minimum duration and separation)
    # If phases are degenerate (all at same time), skip annotation
    valid_phases = []
    for phase_key in ['Phase1', 'Phase2', 'Phase3', 'Phase4', 'Phase5', 'Phase6']:
        phase = phase_data.get(phase_key)
        if phase and phase.get('time_start') is not None and phase.get('time_end') is not None:
            duration = phase.get('time_end') - phase.get('time_start')
            if duration >= 0.1:  # At least 100ms duration
                valid_phases.append((phase_key, phase))

    # Check temporal separation between phases
    if len(valid_phases) >= 2:
        times = [p[1].get('time_start') for p in valid_phases]
        time_diffs = [times[i+1] - times[i] for i in range(len(times)-1)]
        # If most phases are clustered (< 0.3s apart), likely a detection failure
        if sum(1 for d in time_diffs if d < 0.3) > len(time_diffs) * 0.7:
            # Degenerate detection - don't annotate
            return

    # Only draw phases that passed validation (duration >= 0.1s)
    for phase_key, phase in valid_phases:

        t_start = phase.get('time_start')
        t_end = phase.get('time_end')

        if t_start is None or t_end is None:
            continue

        name = phase.get('name', phase_key)
        color = phase.get('color', '#888888')
        confidence = float(phase.get('confidence', 0.5))

        # Draw phase span as shaded region (only if show_shading is True)
        if show_shading:
            ax.axvspan(t_start, t_end, color=color, alpha=0.12, lw=0)

        # Draw boundaries
        alpha_boundary = 0.5 + 0.4 * min(1.0, confidence)
        ax.axvline(t_start, color=color, linestyle='--', linewidth=1.5,
                  alpha=alpha_boundary, zorder=10)

        # Label positioning: "Ignition" outside at bottom, all others inside at bottom (only if show_labels is True)
        if show_labels:
            t_center = (t_start + t_end) / 2.0

            # Create background box for better readability
            bbox_props = dict(boxstyle='round,pad=0.2', facecolor='white',
                             edgecolor=color, alpha=0.9, linewidth=1.0)

            if name == 'Ignition':
                # Place Ignition label below the chart (outside)
                ax.text(t_center, ymin - 0.02 * (ymax - ymin), name,
                       ha='center', va='top', fontsize=7, color=color,
                       fontweight='normal', alpha=1.0, bbox=bbox_props, zorder=1000)
            else:
                # Place all other phase labels inside at bottom
                ax.text(t_center, ymin + 0.02 * (ymax - ymin), name,
                       ha='center', va='bottom', fontsize=7, color=color,
                       fontweight='normal', alpha=1.0, bbox=bbox_props, zorder=1000)

            # # Show confidence if low (place opposite to main label to avoid overlap)
            # if confidence < 0.5:
            #     if name in ('Surge', 'Amplitude Surge', 'Amplitude', 'Ignition'):
            #         # Surge label at bottom, so confidence at top
            #         ax.text(t_center, ymax - 0.02 * (ymax - ymin), f"({confidence:.2f})",
            #                ha='center', va='top', fontsize=6, color=color, alpha=0.6)
            #     else:
            #         # Other labels at top, so confidence at bottom
            #         ax.text(t_center, ymin + 0.05 * (ymax - ymin), f"({confidence:.2f})",
            #                ha='center', va='bottom', fontsize=6, color=color, alpha=0.6)

    # Highlight ignition core (Phase2-Phase5) - background shading only, no label (only if show_shading is True)
    if show_shading:
        p2 = phase_data.get('Phase2', {})
        p5 = phase_data.get('Phase5', {})
        if p2.get('time_start') and p5.get('time_end'):
            ax.axvspan(p2['time_start'], p5['time_end'], color='#FFF59D',
                      alpha=0.15, lw=0, zorder=0)


def _annotate_five_phases_legacy(ax, phases: Dict[str, Any], ymin: float, ymax: float,
                                 *, highlight_padding: float = 0.25, show_labels: bool = True, show_shading: bool = True) -> None:
    """Legacy five-phase annotation (P0-P4)."""
    colors = {
        'P0': '#00BCD4',
        'P1': '#4CAF50',
        'P2': '#FFC107',
        'P3': '#F44336',
        'P4': '#9E9E9E',
    }

    def _get_event(name: str) -> Dict[str, Any]:
        ev = phases.get(name, {})
        return ev if isinstance(ev, dict) else {}

    p0 = _get_event('P0')
    p4_ev = _get_event('P4')
    endpoint = p4_ev if p4_ev.get('time') is not None else _get_event('P3')
    if p0.get('time') is not None and endpoint.get('time') is not None and endpoint['time'] > p0['time']:
        mean_conf = np.nanmean([p0.get('confidence', 0.5), endpoint.get('confidence', 0.5)])
        pad = highlight_padding + (1.0 - float(mean_conf)) * 0.4
        start = float(p0['time']) - pad
        end = float(endpoint['time']) + pad
        if show_shading:
            ax.axvspan(start, end, color='#FFF59D33', lw=0)
        if show_labels:
            # Place Ignition label below the chart (outside)
            ax.text((start + end) * 0.5, ymin - 0.08 * (ymax - ymin), 'Ignition',
                    ha='center', va='top', fontsize=8, color='#424242', zorder=1000)

    for name in ['P0', 'P1', 'P2', 'P3', 'P4']:
        ev = _get_event(name)
        time = ev.get('time')
        if time is None:
            continue
        conf = float(ev.get('confidence', 0.0))
        if not np.isfinite(conf):
            conf = 0.0
        color = colors.get(name, 'cyan')
        half_width = 0.4 + (1.0 - conf) * 0.6
        if show_shading:
            ax.axvspan(time - half_width, time + half_width, color=color, alpha=0.08, lw=0)
        ax.vlines(time, ymin, ymax, linestyles='--', linewidth=1.3, color=color,
                  alpha=0.7 + 0.3 * min(1.0, conf))
        if show_labels:
            # Place phase label inside at bottom (vertical text)
            ax.text(time, ymin, name, rotation=90, va='bottom', ha='center', color=color, fontsize=8, zorder=1000)
            if conf < 0.45:
                ax.text(time, ymax, f"{conf:.2f}", rotation=90, va='top', ha='center',
                        color=color, fontsize=7, alpha=0.6, zorder=1000)


def six_panel(records,electrodes,ign_win,ign_out,ladder,cfg,session_name):
    assert electrodes and len(electrodes) > 0, "electrodes cannot be empty"
    
    # --- parameters ------------------------------------------------------------
    TIME_COL = "Timestamp"
    FS = 128.0
    
    # BUILD PACK
    pack = build_ignition_feature_pack(records, [ign_win], cfg=cfg)
    pack.setdefault('meta', {})['channels_used'] = list(electrodes)

    m = (pack['t'] >= ign_win[0]) & (pack['t'] <= ign_win[1])
    
    # SPECTROGRAM - coarse spectrogram for Panel A + HSI (single source of truth)
    # pack['spec'] = compute_session_spectrogram(
    #     records, channels=electrodes, time_col='Timestamp', fs=FS,
    #     band=(2,25), win_sec=cfg.spec_win, overlap=cfg.spec_ovl
    # )

    # PIANO ROLL
    tWc, fWc, SWc = window_spec_median(
        records, ign_win, channels=electrodes, fs=FS, time_col='Timestamp',
        band=(2,40), win_sec=cfg.spec_win, overlap=cfg.spec_ovl
    )
    # tWc = tWc + cfg.spec_win/2.0  # center STFT times
    # spec_z = _spec_db_rowz(SWc)
    pack.setdefault('spec_by_window', {})[(float(ign_win[0]), float(ign_win[1]))] = (tWc, fWc, SWc)

    
    # HSI 
    tH, H = hsi_v3_from_window_spec(
        tWc, fWc, SWc, in_bw=0.5, ring_offset=1.5, ring_bw=0.8, smooth_hz=6.0, ladder=ladder
    )

    # from scipy.signal import savgol_filter
    # dt = float(np.median(np.diff(tH)))
    # win = max(5, int(round(0.9/dt)) | 1)  # ~0.9 s window, force odd
    # H_s = savgol_filter(H, window_length=win, polyorder=3, mode='interp')

    H_s = smooth_sec(tH, H, 0.45)
    dHSIw = H_s - np.nanmedian(H_s)                 # ΔHSI in STFT time

    # Build an "edge-valid" mask: discard first/last half-window
    half = cfg.spec_win / 2.0
    valid_w = (tWc >= (tWc[0] + half)) & (tWc <= (tWc[-1] - half))
    
    # Interpolate only the valid part onto the report timebase (pack['t'])
    dHSI_t = np.interp(pack['t'], tWc[valid_w], dHSIw[valid_w],
                       left=np.nan, right=np.nan)
    
    pack['hsi'][m] = np.interp(pack['t'][m], tH, H_s, left=H_s[0], right=H_s[-1])
    h50 = float(np.nanpercentile(pack['hsi'][m], 50))

    
    # PLV & PAC
    t_abs = np.asarray(records[TIME_COL], float)
    t0 = float(t_abs[0])

    X = _get_matrix(records, electrodes)

    # Handle bw_hz as either scalar or array
    if np.ndim(cfg.bw_hz) == 0:  # scalar
        bw_f1 = float(cfg.bw_hz)
    else:  # array - use first element for fundamental frequency
        bw_array = np.asarray(cfg.bw_hz)
        bw_f1 = bw_array[0]

    # PLV
    t_plv, plv = _plv_timecourse(
        X, fs=FS, f0=ladder[0], bw=bw_f1,
        win_sec=cfg.win_sec, step_sec=cfg.step_sec)
    pack['plv_7p83'] = np.interp(pack['t'], t0 + t_plv, plv, left=plv[0], right=plv[-1])

    # PAC MVL
    t_pac, mvl = _pac_mvl_timecourse(X, fs=128.0,theta_band=(cfg.sr_centers[0]-1,cfg.sr_centers[0]+1), gamma_band=(25,45),
                                            win_sec=cfg.win_sec, step_sec=cfg.step_sec,amp_gate_pct=70)
    pack['pac_mvl'] = np.interp(pack['t'], t0+t_pac, mvl, left=mvl[0], right=mvl[-1])

    
    # THRESHOLDS
    z7_win = robust_z(pack['z_7p83'][m])
    z95    = float(np.nanpercentile(z7_win, 95))
    h50    = float(np.nanpercentile(pack['hsi'][m], 50))   # for hsi_broad
    plv60  = float(np.nanpercentile(pack['plv_7p83'][m], 60))
    h10    = float(np.nanpercentile(pack['hsi'][m], 10))
    
    
    # SEED EVENT & t0_net
    ev = ign_out['events']
    m_overlap = (ev['t_start'] < ign_win[1]) & (ev['t_end'] > ign_win[0])

    # Extract t0_net from matching event (with fallback to sr_z_peak_t)
    t0_net = None
    if m_overlap.any():
        event_row = ev.loc[m_overlap].iloc[0]
        seed_from_event = float(np.clip(event_row['sr_z_peak_t'], ign_win[0]+2.0, ign_win[1]-2.0))
        if 't0_net' in event_row and pd.notna(event_row['t0_net']):
            t0_net = float(event_row['t0_net'])
        elif 'sr_z_peak_t' in event_row and pd.notna(event_row['sr_z_peak_t']):
            # Fallback: estimate t0 as ~3s before amplitude peak
            sr_peak_t = float(event_row['sr_z_peak_t'])
            t0_net = max(ign_win[0], sr_peak_t - 3.0)
    else:
        seed_from_event = 0.5 * (ign_win[0] + ign_win[1])

    params = PhaseParams(
        z_p0=0.6,
        # plv_p0=np.median(plv),
        plv_p0=np.median(pack['plv_7p83'][m]),
        z_p1=max(1.0, 0.9*z95),     # z-units now
        plv_p1=plv60,
        hsi_broad=h50,              # <-- important for P0
        hsi_tight=h10,
        hsi_release=max(h10+0.14, 0.80),
        plv_release=plv60-0.03,
        min_p0_dur=0.10, 
        min_p1_dur=0.12,
        min_p2_cycles=0.8,
        rel_h2=0.05, rel_h3=0.05,
        bic_7_7_15=0.10, bic_7_15_23=0.10,
    )

    # add optional knobs as attributes
    params.f0 = ladder[0]
    params.rise_eps = 0.05
    params.z_p1_cap    = 1.9      # keep P1 gate in a plausible z‑range
    params.z_rise_tau  = 0.35
    params.z_rise_eps  = 0.03
    params.plv_rise_tau = 0.25    # a bit shorter than 0.25 if needed
    params.plv_rise_eps = 0.005   # 0.005–0.012 usually works well
    params.require_plv_rise = False   # default off for this dataset
    params.debug        = True

    plv55 = float(np.nanpercentile(pack['plv_7p83'][m], 55))
    params.plv_p1 = max(plv55, plv60) - 0.02    # tiny nudge


    # KURAMOTO ORDER PARAMETER (needed for six-phase detector)
    t_abs_all = np.asarray(records[TIME_COL], float)
    mask_seg = (t_abs_all >= ign_win[0]) & (t_abs_all <= ign_win[1])
    X_seg = _get_matrix(records, electrodes)[:, mask_seg]
    t_R_rel, R_raw = _kuramoto_order_series(X_seg, FS, ladder[0], bw_f1)
    R_series = smooth_sec(t_R_rel, R_raw, 0.3)
    t_R = t_R_rel + ign_win[0]

    # TRIADIC BICOHERENCE - compute for all available harmonics
    # Build triads dynamically based on ladder
    f1 = ladder[0] if len(ladder) >= 1 else 7.83
    f2 = ladder[1] if len(ladder) >= 2 else 14.66
    f3 = ladder[2] if len(ladder) >= 3 else 20.80
    f4 = ladder[3] if len(ladder) >= 4 else 27.14
    f5 = ladder[4] if len(ladder) >= 5 else 33.48
    f6 = ladder[5] if len(ladder) >= 6 else 39.82

    triads = []
    if len(ladder) >= 3:
        triads.append((f1, f1, f2))
        triads.append((f1, f1, f3))
        triads.append((f1, f2, f3))
        triads.append((f1, f2, f4))
        triads.append((f1, f3, f4))
        triads.append((f1, f1, f6))
        triads.append((f1, f2, f6))
        triads.append((f1, f3, f6))
        triads.append((f1, f4, f6))
        
        
        
        # triads.append((f2, f2, f6))
        # triads.append((f2, f3, f6))
        # triads.append((f2, f2, f4))
        
        # triads.append((f2, f3, f5))
        # triads.append((f3, f3, f6))

    # Compute triadic bicoherence timeseries
    t_bic_rel = None
    bic_triads = {}
    if len(triads) > 0:
        t_bic_rel, bic_raw = _bicoherence_triads_timecourse(
            X_seg, FS, triads, bw_f1, win_sec=0.8, step_sec=0.1
        )
        # Format labels for readability
        raw_keys = list(bic_raw.keys())
        triad_keys = _format_numeric_labels(raw_keys, decimals=2)
        bic_triads = {label: bic_raw[key] for label, key in zip(triad_keys, raw_keys)}

    # Compute absolute time for bicoherence
    t_bic = (ign_win[0] + t_bic_rel) if t_bic_rel is not None else None

    # Get provider slice for phase detection
    provider = PackProvider(pack).slice(ign_win[0], ign_win[1])
    t_slice = provider.t()

    # Interpolate R onto the pack timebase
    R_interp = np.interp(t_slice, t_R, R_series, left=R_series[0], right=R_series[-1])

    # Get bicoherence and PAC data
    bic_7_7_15 = provider.bic_7_7_15()
    bic_7_15_23 = provider.bic_7_15_23()
    if bic_7_7_15 is not None and bic_7_15_23 is not None:
        bic_max = np.maximum(np.asarray(bic_7_7_15, float), np.asarray(bic_7_15_23, float))
    elif bic_7_7_15 is not None:
        bic_max = np.asarray(bic_7_7_15, float)
    elif bic_7_15_23 is not None:
        bic_max = np.asarray(bic_7_15_23, float)
    else:
        bic_max = np.zeros_like(t_slice)

    pac_data = provider.pac_mvl()

    # Normalize z-scores for phase detection
    zf_z = smooth_sec(t_slice, robust_z(np.asarray(provider.z_fund(), float)), 0.15)
    z2_z = smooth_sec(t_slice, robust_z(np.asarray(provider.z_h2(), float)), 0.15)
    z3_z = smooth_sec(t_slice, robust_z(np.asarray(provider.z_h3(), float)), 0.15)

    # SIX-PHASE EVOLUTION DETECTOR
    phases = _detect_six_phase_evolution(
        t_slice, zf_z, provider.plv_fund(), R_interp, pac_data, bic_max, z2_z, z3_z,
        t0_net=t0_net,
        window_start=ign_win[0]
    )

    # PLOT IGNITION WINDOW (pass pre-computed phases)
    fig, phases, traces = plot_ignition_window_report(
        records,
        provider,
        electrodes,
        params=params,
        hsi_plot_mode="delta", hsi_ylim=("pct",(1,99)),
        seed_t=t0_net if t0_net is not None else "center",  # Use detected t0
        p0_band=(-0.5, +0.3),  # Tightened around t0
        p1_band=(-1.0, +1.4),
        pad_s=2.0,
        title=f"Ignition {ign_win[0]}–{ign_win[1]}s",
        centers=ladder,  # All harmonics
        session_name=session_name,
        phases=phases,  # Pass the six-phase detector output
        bic_triads=bic_triads,  # Pass triadic bicoherence data
        t_bic=t_bic  # Pass bicoherence time axis
    )

    return fig, phases, traces


def estimate_sr_peaks(records, fs, ign_win, session_harmonics, search_band=0.5):
    """
    Get a simple list of estimated SR harmonic frequencies from ignition window EEG (all channels).

    Args:
        records: DataFrame (time x channels) with EEG data
        fs: Sampling frequency (Hz)
        ign_win: Tuple (start_time, end_time in seconds)
        session_harmonics: List of session-estimated harmonic frequencies (fundamental first)
        search_band: Frequency search range around each harmonic (± Hz)

    Returns:
        List of detected harmonic frequencies
    """
    # EEG segment extraction
    start_idx = int(ign_win[0] * fs)
    end_idx = int(ign_win[1] * fs)
    eeg_segment = records.iloc[start_idx:end_idx, :].values

    # Average PSD across channels
    psd_all = [welch(eeg_segment[:, ch], fs, nperseg=eeg_segment.shape[0])[1] 
               for ch in range(eeg_segment.shape[1])]
    avg_psd = np.mean(psd_all, axis=0)
    freqs = np.linspace(0, fs/2, len(avg_psd))

    # Find peak frequencies near harmonics
    detected_freqs = []
    for harmonic in session_harmonics:
        band = (freqs >= harmonic - search_band) & (freqs <= harmonic + search_band)
        if np.any(band):
            peak_idx = np.argmax(avg_psd[band])
            detected_freq = freqs[band][peak_idx]
            
            detected_freqs.append(detected_freq)
        else:
            detected_freqs.append(None)
    return detected_freqs


def six_panel_2(records, electrodes, ign_win, ign_out, ladder, cfg, session_name, *, H=None):
    assert electrodes, "electrodes cannot be empty"

    TIME_COL = 'Timestamp'
    FS = cfg.fs or _infer_fs(records, TIME_COL)
    X_full = _get_matrix(records, electrodes)
    t_all = np.asarray(records[TIME_COL], float)

    pack = build_ignition_feature_pack(records, [ign_win], cfg=cfg)
    pack.setdefault('spec', compute_session_spectrogram(
        records, channels=electrodes, time_col=TIME_COL, fs=FS,
        band=(2, 60), win_sec=cfg.spec_win, overlap=cfg.spec_ovl
    ))

    provider = PackProvider(pack).slice(ign_win[0], ign_win[1])
    t = provider.t()
    zf = provider.z_fund()
    z2 = provider.z_h2()
    z3 = provider.z_h3()
    plv = provider.plv_fund()
    hsi = provider.hsi()

    zf_z = smooth_sec(t, robust_z(np.asarray(zf, float)), 0.15)
    plv_z = smooth_sec(t, robust_z(np.asarray(plv, float)), 0.15)
    hsi_delta = hsi - np.nanmedian(hsi)

    dt = float(np.median(np.diff(t))) if t.size > 1 else 1/FS

    centers = ladder  # Use all harmonics provided in ladder
    bw_arr = np.atleast_1d(cfg.bw_hz if cfg.bw_hz is not None else 0.5)
    bw0 = float(bw_arr[0])

    spec = provider.spectrogram_for_window(t.min(), t.max())
    if spec is None:
        spec = pack['spec']
    t_spec, f_spec, S_spec = _slice_spec_to_window(spec, (t.min(), t.max()), min_cols=20)
    _, slopes = _spectral_slope_series(t_spec, f_spec, S_spec)

    thresh_z = np.nanpercentile(zf_z, 60.0)
    sizes, durations = _avalanche_size_duration(zf_z, t, thresh_z, bridge_sec=0.30)

    mask_seg = (t_all >= ign_win[0]) & (t_all <= ign_win[1])
    base_start = max(t_all[0], ign_win[0] - 20.0)
    mask_base = (t_all >= base_start) & (t_all < ign_win[0])
    if mask_base.sum() < 10:
        mask_base = ~mask_seg
    X = X_full[:, mask_seg]
    X_base = X_full[:, mask_base]
    t_seg = t_all[mask_seg]

    # Compute ignition envelopes for all harmonics (beyond the first 3 from provider)
    z_ign_all = [zf, z2, z3]
    for i in range(3, len(centers)):
        # Extract from raw data for harmonics beyond 3rd
        z_h, _ = _narrowband_envelope_z(X, FS, centers[i], bw0)
        z_ign_all.append(z_h)

    t_R, R_raw = _kuramoto_order_series(X, FS, centers[0], bw0)
    R_series = smooth_sec(t_R, R_raw, 0.3)
    t_R = t_R + ign_win[0]

    theta_b = _fir_bandpass(centers[0], bw0, FS)
    theta_filt = filtfilt(theta_b, [1.0], X, axis=1, padlen=min(2400, X.shape[-1]-1))
    theta_env = np.abs(hilbert(theta_filt, axis=-1))
    theta_env = np.nanmedian(theta_env, axis=0)
    gamma_b = firwin(801, [30.0, 60.0], pass_zero=False, fs=FS)
    gamma_filt = filtfilt(gamma_b, [1.0], X, axis=1, padlen=min(2400, X.shape[-1]-1))
    gamma_env = np.abs(hilbert(gamma_filt, axis=-1))
    gamma_env = np.nanmedian(gamma_env, axis=0)
    te_t, te_series = _transfer_entropy_proxy(theta_env, gamma_env, FS,
                                              lead_sec=0.1, win_sec=1.0, step_sec=0.1)
    if te_series.size:
        te_series = smooth_sec(te_t, te_series, 0.3)
    te_t = te_t + ign_win[0]

    te_matrix = _te_matrix(X, FS, lead_sec=0.05)
    te_base = _te_matrix(X_base, FS, lead_sec=0.05)
    te_mean = np.nanmean(te_base[np.isfinite(te_base)])
    te_std = np.nanstd(te_base[np.isfinite(te_base)]) + 1e-9
    te_diff = ((te_matrix - te_mean) / te_std) - ((te_base - te_mean) / te_std)
    np.fill_diagonal(te_diff, 0.0)
    seed_ch = _extract_seed_channel(ign_out, ign_win)
    seed_idx = _resolve_seed_channel_index(seed_ch, electrodes)
    if seed_idx is not None and (0 <= seed_idx < len(electrodes)):
        seed_display = electrodes[seed_idx].replace('EEG.', '')
    else:
        seed_display = (str(seed_ch).replace('EEG.', '') if seed_ch else 'N/A')
        seed_idx = None
    np.fill_diagonal(te_diff, 0.0)

    # Extract t0_net from ign_out for phase detection anchoring (with fallback to sr_z_peak_t)
    row = _match_ignition_event_row(ign_out, ign_win)
    t0_net = None
    if row is not None:
        if 't0_net' in row:
            t0_val = row['t0_net']
            if pd.notna(t0_val) and np.isfinite(float(t0_val)):
                t0_net = float(t0_val)
        if t0_net is None and 'sr_z_peak_t' in row:
            # Fallback: estimate t0 as ~3s before amplitude peak
            sr_peak_val = row['sr_z_peak_t']
            if pd.notna(sr_peak_val) and np.isfinite(float(sr_peak_val)):
                sr_peak_t = float(sr_peak_val)
                t0_net = max(ign_win[0], sr_peak_t - 3.0)

    mode_power_base = mode_power_ign = None
    entropy_base = entropy_ign = pr_base = pr_ign = np.nan
    if H is not None:
        from connectome_harmonics import project_to_harmonics
        proj_base = project_to_harmonics(X_base, H)
        proj_ign = project_to_harmonics(X, H)
        mode_power_base = np.mean(proj_base**2, axis=1)
        mode_power_ign = np.mean(proj_ign**2, axis=1)
        entropy_base, pr_base = _mode_metrics(mode_power_base)
        entropy_ign, pr_ign = _mode_metrics(mode_power_ign)

    # Interpolate R_series to match feature pack time base
    R_interp = np.interp(t, t_R, R_series, left=np.nan, right=np.nan)

    # Get PAC data
    pac_data = provider.pac_mvl() if hasattr(provider, 'pac_mvl') else None

    # Get bicoherence data
    bic_data = provider.bic_7_7_15() if hasattr(provider, 'bic_7_7_15') else None
    if bic_data is None:
        bic_data = np.zeros_like(zf_z)

    # Use new six-phase detection
    phases = _detect_six_phase_evolution(
        t, zf_z, plv, R_interp, pac_data, bic_data, z2, z3,
        t0_net=t0_net,
        window_start=ign_win[0]
    )

    t_base = t_all[mask_base]
    sizes_base = durations_base = np.array([])
    slopes_base = np.array([])
    R_base_series = np.array([])
    t_R_base = np.array([])
    var_base = np.array([])
    if t_base.size > 5:
        baseline_window = (float(t_base[0]), float(t_base[-1]))
        tWb, fWb, SWb = window_spec_median(
            records, baseline_window, channels=electrodes, fs=FS,
            time_col=TIME_COL, band=(2,60), win_sec=cfg.spec_win, overlap=cfg.spec_ovl
        )
        tWb = tWb + cfg.spec_win/2.0
        _, slopes_base = _spectral_slope_series(tWb, fWb, SWb)

        # Compute baseline envelopes for all harmonics in ladder
        z_base_all = []
        for freq in centers:
            z_h, _ = _narrowband_envelope_z(X_base, FS, freq, bw0)
            z_base_all.append(z_h)

        # Keep old variable names for backward compatibility
        z_base = z_base_all[0]
        z2_base = z_base_all[1] if len(z_base_all) > 1 else None
        z3_base = z_base_all[2] if len(z_base_all) > 2 else None
        z4_base = z_base_all[3] if len(z_base_all) > 3 else None

        z_base_z = smooth_sec(t_base, robust_z(np.asarray(z_base, float)), 0.15)
        sizes_base, durations_base = _avalanche_size_duration(z_base_z, t_base, thresh_z, bridge_sec=0.30)

        t_R_base, R_base_raw = _kuramoto_order_series(X_base, FS, centers[0], bw0)
        R_base_series = smooth_sec(t_R_base, R_base_raw, 0.3)
        t_R_base = t_R_base + baseline_window[0]

        env_base = smooth_sec(t_base, np.asarray(z_base, float)**2, 0.3)
        var_base = np.clip(_interp_safe(t_R_base, t_base, env_base), 1e-9, None)

        t_plv_base, plv_base_series = _plv_timecourse(
            X_base, FS, centers[0], bw0, cfg.win_sec, cfg.step_sec)
        plv_base_times = baseline_window[0] + t_plv_base

        comp_t_base, samp_entropy_base = _complexity_series(z_base_z, t_base, win_sec=3.0, step_sec=0.2)
        lz_t_base, lz_vals_base = _lz_complexity_series(z_base_z, t_base, win_sec=3.0, step_sec=0.2)
    else:
        z_base_z = np.array([])
        z_base = np.array([])
        z_base_all = []  # Initialize empty list for all baseline harmonics
        z2_base = z3_base = z4_base = None
        t_plv_base = np.array([])
        plv_base_series = np.array([])
        plv_base_times = np.array([])
        comp_t_base = np.array([])
        samp_entropy_base = np.array([])
        lz_t_base = np.array([])
        lz_vals_base = np.array([])

    env_ign = smooth_sec(t, np.asarray(zf, float)**2, 0.3)
    var_ign = np.clip(_interp_safe(t_R, t, env_ign), 1e-9, None)

    plv_series = np.interp(t_seg, pack['t'], pack['plv_7p83']) if pack.get('plv_7p83') is not None else plv
    harmonics_for_msc = centers  # Use all harmonics
    msc_matrix, msc_null = _msc_matrix(X, FS, harmonics_for_msc, bw0, n_surrogates=32)
    z2 = np.asarray(z2, float)
    z3 = np.asarray(z3, float)
    comp_t, samp_entropy = _complexity_series(zf_z, t, win_sec=3.0, step_sec=0.2)
    lz_t, lz_vals = _lz_complexity_series(zf_z, t, win_sec=3.0, step_sec=0.2)

    fig = plt.figure(figsize=(16, 10), constrained_layout=True, dpi=160)
    gs = GridSpec(3, 2, figure=fig)

    ax_modes = fig.add_subplot(gs[0, 0])
    if mode_power_base is not None:
        n_modes = min(12, mode_power_base.shape[0])
        idx = np.arange(n_modes)
        width = 0.35
        ax_modes.bar(idx - width/2, mode_power_base[:n_modes], width, alpha=0.6, label='Baseline')
        ax_modes.bar(idx + width/2, mode_power_ign[:n_modes], width, alpha=0.8, label='Ignition')
        ax_modes.set_xticks(idx)
        ax_modes.set_xticklabels([f'M{k}' for k in idx])
        ax_modes.set_ylabel('Mode power')
        ax_modes.set_title('Connectome modes engagement')
        ax_modes.legend(fontsize=8)
        ax_modes.text(0.02, 0.92,
                      f'H_base={entropy_base:.2f}, PR={pr_base:.1f}\nH_ign={entropy_ign:.2f}, PR={pr_ign:.1f}',
                      transform=ax_modes.transAxes, fontsize=8,
                      bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    else:
        # Use all harmonics from ladder
        fallback_labels = [f'{f:.2f} Hz' for f in centers]
        ign_vals = []
        base_vals = []
        for idx_mode in range(len(centers)):
            # Use the computed envelopes for all harmonics
            if idx_mode < len(z_ign_all):
                ign_vals.append(float(np.nanmean(np.abs(z_ign_all[idx_mode]))))
            else:
                ign_vals.append(np.nan)

            if idx_mode < len(z_base_all):
                base_vals.append(float(np.nanmean(np.abs(z_base_all[idx_mode]))))
            else:
                base_vals.append(np.nan)

        idx = np.arange(len(fallback_labels))
        width = 0.35
        ax_modes.bar(idx - width/2, base_vals, width, alpha=0.6, label='Baseline')
        ax_modes.bar(idx + width/2, ign_vals, width, alpha=0.8, label='Ignition')
        ax_modes.set_xticks(idx)
        ax_modes.set_xticklabels(fallback_labels, fontsize=7 if len(fallback_labels) > 3 else 8)
        ax_modes.set_ylabel('Envelope magnitude')
        ax_modes.set_title('Harmonic Envelopes')
        ax_modes.legend(fontsize=8)

    ax_te = fig.add_subplot(gs[2, 0])
    lim = float(np.nanmax(np.abs(te_diff)))
    lim = max(lim, 1e-2)
    im = ax_te.imshow(te_diff, cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto', interpolation='nearest')
    cbar = fig.colorbar(im, ax=ax_te, fraction=0.05, pad=0.02)
    cbar.set_label('ΔTE (z-scored)')
    ax_te.set_title(f'Directed Information Flow')
    tick_labels = [ch.replace('EEG.', '') for ch in electrodes]
    idx_range = range(len(electrodes))
    ax_te.set_xticks(idx_range)
    ax_te.set_xticklabels(tick_labels, rotation=90)
    ax_te.set_yticks(idx_range)
    ax_te.set_yticklabels(tick_labels)
    if seed_idx is not None:
        ax_te.axhline(seed_idx - 0.5, color='black', linewidth=1.0, alpha=0.7)
        ax_te.axhline(seed_idx + 0.5, color='black', linewidth=1.0, alpha=0.7)
        ax_te.axvline(seed_idx - 0.5, color='black', linewidth=1.0, alpha=0.7)
        ax_te.axvline(seed_idx + 0.5, color='black', linewidth=1.0, alpha=0.7)
        for idx_lbl, lbl in enumerate(ax_te.get_xticklabels()):
            if idx_lbl == seed_idx:
                lbl.set_fontweight('bold')
                lbl.set_color('black')
        for idx_lbl, lbl in enumerate(ax_te.get_yticklabels()):
            if idx_lbl == seed_idx:
                lbl.set_fontweight('bold')
                lbl.set_color('black')

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t_R, R_series, color='tab:green', lw=1.4)
    ax3.set_ylim(0, 1.05)
    ax3.set_title('Kuramoto Global Integration')
    ax3.set_ylabel('R(t)')
    ax3.set_xlabel('Time (s)')
    annotate_phases(ax3, phases, 0, 1.05)

    ax4 = fig.add_subplot(gs[1, 0])
    if R_base_series.size and var_base.size:
        m = np.isfinite(var_base) & np.isfinite(R_base_series)
        if np.any(m):
            ax4.scatter(var_base[m], R_base_series[m], s=25, alpha=0.5, color='tab:blue', label='Baseline')
    if R_series.size and var_ign.size:
        m = np.isfinite(var_ign) & np.isfinite(R_series)
        if np.any(m):
            ax4.scatter(var_ign[m], R_series[m], s=30, alpha=0.7, color='tab:orange', label='Ignition')
    ax4.set_title('Coherence vs SR Power')
    ax4.set_xlabel('SR envelope power')
    ax4.set_ylabel('Kuramoto R')
    ax4.set_xscale('log')
    ax4.set_ylim(0, 1.05)
    ax4.legend(loc='best')

    ax5 = fig.add_subplot(gs[2, 1])
    if te_t.size:
        ax5.plot(te_t, te_series, color='tab:purple', lw=1.3)
    ax5.axhline(0, color='gray', ls='--', lw=1.0)
    ax5.set_title('Cross-Scale Information Flow')
    ax5.set_ylabel('ΔCorr² (θ → γ)')
    ax5.set_xlabel('Time (s)')
    annotate_phases(ax5, phases, *ax5.get_ylim())

    ax6 = fig.add_subplot(gs[0, 1])
    harmonics_labels = [f'{freq:.2f} Hz' for freq in harmonics_for_msc]
    n_h = len(harmonics_labels)
    n_ch = X.shape[0]
    base_x = np.arange(n_h)
    width = 0.8 / max(n_ch, 1)
    for ci, ch in enumerate(electrodes):
        offsets = base_x - 0.4 + ci * width + width/2
        edge = 'k' if (seed_idx is not None and ci == seed_idx) else None
        lw = 1.0 if (seed_idx is not None and ci == seed_idx) else 0.0
        ax6.bar(offsets, msc_matrix[:, ci], width, alpha=0.8, label=ch,
                edgecolor=edge, linewidth=lw)
        for fi in range(n_h):
            ax6.vlines(offsets[fi], 0, msc_null[fi, ci], colors='#ffffff', linestyles='dotted', linewidth=0.7,
                        alpha=0.7)
    ax6.set_xticks(base_x)
    ax6.set_xticklabels(harmonics_labels, rotation=20)
    ax6.set_ylim(0, 1.05)
    ax6.set_ylabel('MSC')
    ax6.set_title(f'EEG–SR Coherence @ Harmonics')
    ax6.legend(loc='upper right', ncol=min(n_ch, 4), fontsize=8, frameon=False)

    # Remove x-axis margins - make data fill entire chart width
    ax3.set_xlim(ign_win[0], ign_win[1])
    ax5.set_xlim(ign_win[0], ign_win[1])

    fig.suptitle(f'Ignition {ign_win[0]}–{ign_win[1]}s\n{session_name}', fontsize=14)
    return fig


def sr_signature_panel(records, electrodes, ign_win, ign_out, ladder, cfg, session_name, *, palette: str = 'sunrise'):
    TIME_COL = cfg.time_col
    FS = cfg.fs or _infer_fs(records, TIME_COL)
    t_all = np.asarray(records[TIME_COL], float)
    if t_all.size < 2:
        raise ValueError('Not enough samples to build SR signature panel')
    X = _get_matrix(records, electrodes)

    # Use all harmonics from ladder
    f1 = ladder[0]
    bw_cfg = cfg.bw_hz if cfg.bw_hz is not None else 0.5

    # Handle bw_hz as either scalar or array
    if np.ndim(bw_cfg) == 0:  # scalar
        bw_array = np.full(len(ladder), float(bw_cfg))
    else:  # array
        bw_array = np.asarray(bw_cfg)
        if len(bw_array) != len(ladder):
            raise ValueError(f'bw_hz array length ({len(bw_array)}) must match ladder length ({len(ladder)})')

    pad = max(4.0, 0.5 * (ign_win[1] - ign_win[0]))
    t0 = max(t_all[0], ign_win[0] - pad)
    t1 = min(t_all[-1], ign_win[1] + pad)
    mask_seg = (t_all >= t0) & (t_all <= t1)
    if not np.any(mask_seg):
        mask_seg = np.ones_like(t_all, dtype=bool)
        t0, t1 = t_all[0], t_all[-1]

    # Spectrogram (5–25 Hz, row-wise z)
    t_spec, f_spec, S_spec = window_spec_median(
        records, (t0, t1), channels=electrodes, fs=FS, time_col=TIME_COL,
        band=(0.5, 40), win_sec=1.5, overlap=0.95
    )
    spec_z = _spec_db_rowz(S_spec)

    # Compute harmonic envelopes for all harmonics in ladder
    def _safe_envelope(center_hz: float, bandwidth: float) -> np.ndarray:
        if center_hz >= (0.48 * FS):
            return np.full(t_all.shape, np.nan, dtype=float)
        amp, _ = _narrowband_envelope_z(X, FS, center_hz, bandwidth)
        return amp

    def _to_z(env):
        if not np.isfinite(env).any():
            return np.full_like(env, np.nan, dtype=float)
        return smooth_sec(t_all, robust_z(env), 0.15)

    # Compute envelopes for all harmonics
    harmonic_envs_z = []
    for i, freq in enumerate(ladder):
        env_full = _safe_envelope(freq, bw_array[i])
        env_z = _to_z(env_full)
        harmonic_envs_z.append(env_z)

    # Keep old variable names for backward compatibility
    env1_z = harmonic_envs_z[0]
    env2_z = harmonic_envs_z[1] if len(harmonic_envs_z) > 1 else np.full_like(env1_z, np.nan)
    env3_z = harmonic_envs_z[2] if len(harmonic_envs_z) > 2 else np.full_like(env1_z, np.nan)

    # For legacy code that expects f2, f3, f4, f5, f6 (with fallbacks)
    f2 = ladder[1] if len(ladder) > 1 else f1 * 2
    f3 = ladder[2] if len(ladder) > 2 else f1 * 3
    f4 = ladder[3] if len(ladder) > 3 else f1 * 4
    f5 = ladder[4] if len(ladder) > 4 else f1 * 5
    f6 = ladder[5] if len(ladder) > 5 else f1 * 6

    mask_event = (t_all >= ign_win[0]) & (t_all <= ign_win[1])
    X_event = X[:, mask_event] if np.any(mask_event) else X[:, mask_seg]
    v_ref_full = None
    # try:
    #     sr_mode = getattr(cfg, 'sr_reference', 'auto-SSD')
    #     v_sr_event, w_sr = _build_virtual_sr(X_event, FS, f1, bw, mode=sr_mode)
    #     if w_sr is not None:
    #         v_ref_full = (w_sr @ X).astype(float)
    # except Exception:
    #     v_ref_full = None
    # Simple median-based SR reference (matches envelope computation method)
    bw_f1 = bw_array[0]  # Bandwidth for fundamental frequency
    b = firwin(801, [max(0.1, f1-bw_f1), f1+bw_f1], pass_zero=False, fs=FS)
    Xb = filtfilt(b, [1.0], X, axis=-1, padlen=min(2400, X.shape[-1]-1))
    v_ref_full = np.nanmedian(Xb, axis=0)  # Median across channels

    t_plv_rel, plv_series = _plv_timecourse(X, FS, f1, bw_f1, win_sec=3.0, step_sec=0.01)
    plv_times = t_all[0] + t_plv_rel

    t_msc_rel, msc_series = _msc_timecourse(X, FS, f1, bw_f1, win_sec=1.0, step_sec=0.1, v_ref=v_ref_full)
    msc_times = t_all[0] + t_msc_rel
    msc_series_s = np.array([])

    # ΔHSI proxy from mean-centered z envelopes (all harmonics)
    valid_envs = [env for env in harmonic_envs_z if np.isfinite(env).any()]
    if not valid_envs:
        z_mean = np.full_like(env1_z, np.nan)
    else:
        z_stack = np.vstack(valid_envs)
        z_mean = np.nanmean(z_stack, axis=0)
    baseline_mask = (t_all >= t0) & (t_all < ign_win[0])
    base_mean = np.nanmean(z_mean[baseline_mask]) if np.any(baseline_mask) else np.nanmedian(z_mean)
    delta_hsi = z_mean - base_mean

    # Triadic bicoherence
    triads = []
    # if len(ladder) >= 2:
    #     triads.append((f1, f1, f2))
    # else:
    #     triads.append((f1, f1, 2 * f1))
    if len(ladder) >= 3:
        triads.append((f1, f1, f2))
        triads.append((f1, f2, f3))
        triads.append((f1, f2, f4))
        triads.append((f1, f3, f4))
        triads.append((f1, f4, f6))
    else:
        triads.append((f1, 2 * f1, 3 * f1))
    t_bic_rel, bic_raw = _bicoherence_triads_timecourse(
        X, FS, triads, bw_f1, win_sec=0.8, step_sec=0.1
    )
    t_bic = t_all[0] + t_bic_rel
    raw_keys = list(bic_raw.keys())
    triad_keys = _format_numeric_labels(raw_keys, decimals=2)
    bic = {label: bic_raw[key] for label, key in zip(triad_keys, raw_keys)}

    # PAC MVL
    t_pac_rel, pac_vals = _pac_mvl_timecourse(
        X, FS,
        theta_band=(max(0.1, f1 - 0.5), f1 + 0.5),
        gamma_band=(30.0, 60.0),
        win_sec=6, step_sec=0.1,
        amp_gate_pct=50,
    )
    t_pac = t_all[0] + t_pac_rel

    # Kuramoto R(t)
    t_R_rel, R_series = _kuramoto_order_series(X, FS, f1, bw_f1)
    t_R = t_all[0] + t_R_rel
    R_smooth = smooth_sec(t_R, R_series, 0.3)

    # SIX-PHASE DETECTION (for annotations)
    # Extract t0_net from ign_out for phase detection anchoring (with fallback to sr_z_peak_t)
    row = _match_ignition_event_row(ign_out, ign_win)
    t0_net = None
    if row is not None:
        if 't0_net' in row:
            t0_val = row['t0_net']
            if pd.notna(t0_val) and np.isfinite(float(t0_val)):
                t0_net = float(t0_val)
        if t0_net is None and 'sr_z_peak_t' in row:
            # Fallback: estimate t0 as ~3s before amplitude peak
            sr_peak_val = row['sr_z_peak_t']
            if pd.notna(sr_peak_val) and np.isfinite(float(sr_peak_val)):
                sr_peak_t = float(sr_peak_val)
                t0_net = max(ign_win[0], sr_peak_t - 3.0)

    # Prepare data for phase detection on ignition window
    mask_ign = (t_all >= ign_win[0]) & (t_all <= ign_win[1])
    t_ign = t_all[mask_ign] - ign_win[0]  # relative to window start

    # Interpolate all signals onto the ignition window timebase
    R_ign = np.interp(t_all[mask_ign], t_R, R_smooth, left=R_smooth[0], right=R_smooth[-1])

    # Get bicoherence max
    bic_ign = np.zeros_like(t_ign)
    if t_bic.size and len(bic) > 0:
        for series in bic.values():
            bic_interp = np.interp(t_all[mask_ign], t_bic, series, left=0, right=0)
            bic_ign = np.maximum(bic_ign, bic_interp)

    pac_ign = np.interp(t_all[mask_ign], t_pac, pac_vals, left=pac_vals[0], right=pac_vals[-1]) if t_pac.size else np.zeros_like(t_ign)
    plv_ign = np.interp(t_all[mask_ign], plv_times, plv_series, left=plv_series[0], right=plv_series[-1])

    # Normalize envelopes for phase detection
    env1_ign_z = smooth_sec(t_ign, robust_z(env1_z[mask_ign]), 0.15)
    env2_ign_z = smooth_sec(t_ign, robust_z(env2_z[mask_ign]), 0.15)
    env3_ign_z = smooth_sec(t_ign, robust_z(env3_z[mask_ign]), 0.15)

    # Detect six-phase evolution
    phases = _detect_six_phase_evolution(
        t_ign, env1_ign_z, plv_ign, R_ign, pac_ign, bic_ign, env2_ign_z, env3_ign_z,
        t0_net=t0_net - ign_win[0] if t0_net is not None else None,
        window_start=0.0
    )

    # Adjust phase times back to absolute time (relative to t_all[0])
    phase_data = phases.get('phases', {})
    for phase_key in ['Phase1', 'Phase2', 'Phase3', 'Phase4', 'Phase5', 'Phase6']:
        phase = phase_data.get(phase_key)
        if phase:
            if phase.get('time_start') is not None:
                phase['time_start'] = phase['time_start'] + ign_win[0]
            if phase.get('time_end') is not None:
                phase['time_end'] = phase['time_end'] + ign_win[0]

    spec_cmap, colors = _resolve_palette(palette)
    tick_color = colors.get('tick', '#222222')
    panel_bg = colors.get('panel_bg', '#ffffff')
    spine_color = colors.get('spine', tick_color)

    fig = plt.figure(figsize=(18, 12), dpi=160)
    fig.patch.set_facecolor(panel_bg)
    # Use 2 columns: main plots (col 0) and colorbar space (col 1)
    # This ensures all plot axes have the same width
    # Charts below spectrogram are 10% taller: 1.6→1.76, 1.3→1.43, 1.2→1.32
    gs = GridSpec(5, 2, height_ratios=[4.236, 2.618, 2.618, 1.618, 1.618], width_ratios=[1, 0.02],
                  hspace=0.5, wspace=0.02)
    ax_spec = fig.add_subplot(gs[0, 0])
    ax_env = fig.add_subplot(gs[1, 0], sharex=ax_spec)
    ax_hsi = fig.add_subplot(gs[4, 0], sharex=ax_spec)
    ax_bic = fig.add_subplot(gs[2, 0], sharex=ax_spec)
    ax_pac = fig.add_subplot(gs[3, 0], sharex=ax_spec)

    def _apply_sunrise_style(ax):
        ax.set_facecolor(panel_bg)
        ax.tick_params(colors=tick_color, which='both')
        if ax.yaxis.label:
            ax.yaxis.label.set_color(tick_color)
        if ax.xaxis.label:
            ax.xaxis.label.set_color(tick_color)
        if ax.title:
            ax.title.set_color(tick_color)
        for spine in ax.spines.values():
            spine.set_color(spine_color)

    for axis in (ax_env, ax_hsi, ax_bic, ax_pac):
        _apply_sunrise_style(axis)
    ax_bic.set_title('Triadic Bicoherence', color=tick_color)

    extent = [t_spec[0], t_spec[-1], f_spec[0], f_spec[-1]]
    im = ax_spec.imshow(spec_z, extent=extent, origin='lower', aspect='auto', cmap=spec_cmap, vmin=-3, vmax=3, interpolation='lanczos')
    ax_spec.set_facecolor(panel_bg)
    ax_spec.set_ylabel('Frequency (Hz)', color=tick_color)
    ax_spec.set_title('SR-focused spectrogram (row-z)', color=tick_color)
    # Draw white dashed horizontal lines for all harmonics in ladder only
    for freq in ladder:
        if f_spec[0] <= freq <= f_spec[-1]:
            ax_spec.axhline(freq, color='white', linestyle='--', linewidth=1.0, alpha=0.85)
    ax_spec.axvspan(t0, ign_win[0], color=colors['window_fill'], alpha=0.25)
    ax_spec.axvspan(ign_win[1], t1, color=colors['window_fill'], alpha=0.25)
    for spine in ax_spec.spines.values():
        spine.set_color(spine_color)
    ax_spec.tick_params(colors=tick_color, which='both')
    # Create colorbar in dedicated axis (column 1, row 0) to preserve plot alignment
    cax = fig.add_subplot(gs[0, 1])
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.yaxis.set_tick_params(color=tick_color)
    for label in cbar.ax.get_yticklabels():
        label.set_color(tick_color)
    cbar.set_label('z (per frequency)', color=tick_color)

    # Plot all harmonics with color cycling
    # Generate distinguishable colors for any number of harmonics
    n_harmonics = len(ladder)
    if n_harmonics <= 10:
        # Use tab10 colormap for up to 10 harmonics (distinct categorical colors)
        cmap = plt.cm.get_cmap('tab10')
        harmonic_colors = [cmap(i) for i in range(n_harmonics)]
    else:
        # Use hsv colormap for more than 10 harmonics
        cmap = plt.cm.get_cmap('hsv')
        harmonic_colors = [cmap(i / n_harmonics) for i in range(n_harmonics)]

    for i, (freq, env_z) in enumerate(zip(ladder, harmonic_envs_z)):
        color = harmonic_colors[i]
        lw = 1.8 if i == 0 else max(1.0, 1.3 - i * 0.05)  # Decrease linewidth for higher harmonics
        ax_env.plot(t_all, env_z, label=f'{freq:.2f} Hz', color=color, linewidth=lw)

    ax_env.axvspan(t0, ign_win[0], color=colors['window_fill'], alpha=0.25)
    ax_env.axvspan(ign_win[1], t1, color=colors['window_fill'], alpha=0.25)
    ax_env.set_ylabel('Envelope z', color=tick_color)
    ax_env.set_title('Harmonic envelopes (z) with PLV', color=tick_color)
    leg_env = ax_env.legend(loc='upper left', fontsize=7 if len(ladder) > 3 else 8,
                            frameon=False, ncol=1 if len(ladder) <= 4 else 2)
    if leg_env:
        for text in leg_env.get_texts():
            text.set_color(tick_color)
    ax_env.axhline(0, color=colors['baseline_line'], linestyle='--', linewidth=0.8)

    # Fit envelope y-axis to visible data range (all harmonics)
    visible_mask = (t_all >= t0) & (t_all <= t1)
    visible_env_list = [env_z[visible_mask] for env_z in harmonic_envs_z if np.isfinite(env_z).any()]
    if visible_env_list:
        visible_env = np.concatenate(visible_env_list)
        env_min, env_max = np.nanmin(visible_env), np.nanmax(visible_env)
        env_range = env_max - env_min
        pad = 0.15 * env_range  # 15% padding for phase labels
        ax_env.set_ylim(env_min - pad, env_max + pad)
    else:
        ax_env.set_ylim(-3, 3)  # Default range if no valid data

    ax_env2 = ax_env.twinx()
    ax_env2.tick_params(colors=colors['plv_mean'], which='both')
    ax_env2.yaxis.label.set_color(colors['plv_mean'])
    ax_env2.spines['right'].set_color(spine_color)
    # msc_color = colors.get('msc', colors.get('bic1', '#ff8c00'))
    # Plot PLV with low zorder so it appears below phase labels
    ax_env2.plot(plv_times, plv_series, linestyle='--', color=colors['plv'], linewidth=1.4, label='PLV @ fundamental', zorder=1)
    # Make main axis render on top so phase labels (zorder=1000) appear above twin axis content
    ax_env.set_zorder(ax_env2.get_zorder() + 1)
    ax_env.patch.set_visible(False)  # Hide main axis background so twin axis content shows through
    # if msc_series.size:
    #     msc_series_s = smooth_sec(msc_times, msc_series, 0.25)
    #     ax_env2.plot(msc_times, msc_series_s, linestyle='-', color=msc_color, linewidth=1.3, alpha=0.85, label='MSC @ fundamental')
    ax_env2.set_ylabel('Synchrony (0–1)')


    ax_env2.set_ylim(0.2, 1.00)

    # Don't draw window_fill on twin axis - already drawn on main axis to avoid double-shading
    # ax_env2.axvspan(t0, ign_win[0], color=colors['window_fill'], alpha=0.25)
    # ax_env2.axvspan(ign_win[1], t1, color=colors['window_fill'], alpha=0.25)
    handles2, labels2 = ax_env2.get_legend_handles_labels()
    if handles2:
        leg2 = ax_env2.legend(handles2, labels2, loc='upper right', fontsize=8, frameon=False)
        for text in leg2.get_texts():
            text.set_color(colors['plv_mean'])

    window_mask = (t_all >= ign_win[0]) & (t_all <= ign_win[1])
    z_peak = float(np.nanmax(env1_z[window_mask])) if np.any(window_mask) else np.nan
    # if np.any(window_mask):
    #     idx_peak = np.nanargmax(env1_z[window_mask])
    #     t_peak = t_all[window_mask][idx_peak]
    #     ax_env.plot(t_peak, z_peak, marker='o', color=colors['fundamental'])

    plv_win_mask = (plv_times >= ign_win[0]) & (plv_times <= ign_win[1])
    plv_mean = float(np.nanmean(plv_series[plv_win_mask])) if np.any(plv_win_mask) else np.nan
    if np.isfinite(plv_mean):
        ax_env2.axhline(plv_mean, color=colors['plv_mean'], linestyle=':', linewidth=1.0)
    if msc_series.size and msc_series_s.size:
        msc_win_mask = (msc_times >= ign_win[0]) & (msc_times <= ign_win[1])
        msc_mean = float(np.nanmean(msc_series_s[msc_win_mask])) if np.any(msc_win_mask) else np.nan
        if np.isfinite(msc_mean):
            ax_env2.axhline(msc_mean, color=msc_color, linestyle='-.', linewidth=0.9, alpha=0.8)



    # Generate distinguishable colors for any number of triads using colormap
    num_triads = len(triad_keys)
    if num_triads <= 10:
        # Use tab10 colormap for up to 10 triads (distinct categorical colors)
        cmap = plt.cm.get_cmap('tab10')
        triad_colors = [cmap(i) for i in range(num_triads)]
    else:
        # Use hsv colormap for more than 10 triads
        cmap = plt.cm.get_cmap('hsv')
        triad_colors = [cmap(i / num_triads) for i in range(num_triads)]

    bic_peaks = []
    for idx, key in enumerate(triad_keys):
        series = bic[key]
        color = triad_colors[idx]
        # Use thicker lines for more important (earlier) triads
        lw = max(1.1, 1.5 - idx * 0.1)
        ax_bic.plot(t_bic, series, color=color, linewidth=lw, label=key, alpha=0.9)
        win_mask = (t_bic >= ign_win[0]) & (t_bic <= ign_win[1])
        # if np.any(win_mask):
        #     peak_idx = np.nanargmax(series[win_mask])
        #     abs_idx = np.flatnonzero(win_mask)[peak_idx]
        #     ax_bic.plot(t_bic[abs_idx], series[abs_idx], marker='*', color=triad_colors[idx % len(triad_colors)], markersize=10)
        #     bic_peaks.append(float(series[abs_idx]))
    ax_bic.set_ylabel('Bicoherence', color=tick_color)
    ax_bic.axvspan(t0, ign_win[0], color=colors['window_fill'], alpha=0.25)
    ax_bic.axvspan(ign_win[1], t1, color=colors['window_fill'], alpha=0.25)

    # Horizontal legend inside chart at top (max 4 columns to prevent overlap)
    ncol = min(10, num_triads)  # Max 4 columns for readability
    leg_bic = ax_bic.legend(loc='upper center', ncol=ncol, fontsize=7, frameon=True)
    if leg_bic:
        leg_bic.get_frame().set_facecolor('white')
        leg_bic.get_frame().set_alpha(0.618)
        leg_bic.set_zorder(2000)  # Ensure legend is on top of everything
        for text in leg_bic.get_texts():
            text.set_color(tick_color)

    # Fit bicoherence y-axis to visible data range
    bic_visible_mask = (t_bic >= t0) & (t_bic <= t1)
    bic_all_visible = []
    for key in triad_keys:
        series = bic[key]
        bic_all_visible.extend(series[bic_visible_mask])
    if len(bic_all_visible) > 0:
        bic_all_visible = np.array(bic_all_visible)
        bic_min, bic_max = np.nanmin(bic_all_visible), np.nanmax(bic_all_visible)
        bic_range = bic_max - bic_min
        bic_pad = 0.2 * bic_range  # 15% padding for phase labels
        ax_bic.set_ylim(bic_min - 0.15 * bic_range, bic_max + bic_pad)

    ax_hsi.plot(t_all, delta_hsi, color=colors['delta_hsi'], linewidth=1.5)
    ax_hsi.fill_between(t_all, 0, delta_hsi, where=delta_hsi >= 0, color=colors['delta_hsi'], alpha=0.25)
    ax_hsi.axhline(0, color=colors['baseline_line'], linestyle='--', linewidth=0.8)
    ax_hsi.axvspan(t0, ign_win[0], color=colors['window_fill'], alpha=0.25)
    ax_hsi.axvspan(ign_win[1], t1, color=colors['window_fill'], alpha=0.25)
    ax_hsi.set_ylabel('ΔHSI (a.u.)', color=tick_color)
    ax_hsi.set_title('Harmonic tightening (ΔHSI proxy)', color=tick_color)

    # Fit ΔHSI y-axis to visible data range
    hsi_visible = delta_hsi[visible_mask]
    hsi_min, hsi_max = np.nanmin(hsi_visible), np.nanmax(hsi_visible)
    # Ensure zero baseline is always visible
    hsi_min = min(hsi_min, 0)
    hsi_max = max(hsi_max, 0)
    hsi_range = hsi_max - hsi_min
    hsi_pad = 0.15 * hsi_range  # 15% padding for phase labels
    ax_hsi.set_ylim(hsi_min - hsi_pad, hsi_max + hsi_pad)



    pac_base_mask = (t_pac >= (ign_win[0] - pad)) & (t_pac < ign_win[0])
    pac_win_mask = (t_pac >= ign_win[0]) & (t_pac <= ign_win[1])
    pac_base = np.nanmean(pac_vals[pac_base_mask]) if np.any(pac_base_mask) else np.nan
    pac_win = np.nanmean(pac_vals[pac_win_mask]) if np.any(pac_win_mask) else np.nan
    delta_pac = pac_win - pac_base if np.isfinite(pac_win) and np.isfinite(pac_base) else np.nan
    ax_pac.plot(t_pac, pac_vals, color=colors['pac'], linewidth=1.4)
    ax_pac.axvspan(t0, ign_win[0], color=colors['window_fill'], alpha=0.25)
    ax_pac.axvspan(ign_win[1], t1, color=colors['window_fill'], alpha=0.25)
    ax_pac.axhline(pac_base, color=colors['baseline_line'], linestyle=':', linewidth=0.9) if np.isfinite(pac_base) else None
    if np.isfinite(pac_win):
        ax_pac.axhline(pac_win, color=colors['pac'], linestyle='--', linewidth=1.0)
    ax_pac.set_ylabel('MVL', color=tick_color)
    ax_pac.set_xlabel('Time (s)', color=tick_color)
    ax_pac.set_title('θ→γ PAC (MVL)', color=tick_color)

    # Fit PAC y-axis to visible data range
    pac_visible_mask = (t_pac >= t0) & (t_pac <= t1)
    pac_visible = pac_vals[pac_visible_mask]
    if pac_visible.size > 0:
        pac_min, pac_max = np.nanmin(pac_visible), np.nanmax(pac_visible)
        pac_range = pac_max - pac_min
        pac_pad = 0.25 * pac_range  # 15% padding for phase labels
        ax_pac.set_ylim(pac_min - pac_pad, pac_max + 0.15 * pac_range)

    # Add phase annotations to all panels
    # For ax_spec (SR-focused spectrogram): only lines, no shading, no labels
    annotate_phases(ax_spec, phases, f_spec[0], f_spec[-1], show_labels=False, show_shading=False)
    # All other panels: lines, shading, and labels
    annotate_phases(ax_env, phases, *ax_env.get_ylim(), show_labels=True)
    annotate_phases(ax_hsi, phases, *ax_hsi.get_ylim(), show_labels=True)
    annotate_phases(ax_bic, phases, *ax_bic.get_ylim(), show_labels=True)
    annotate_phases(ax_pac, phases, *ax_pac.get_ylim(), show_labels=True)

    ax_spec.set_xlim(t0, t1)
    for ax in (ax_env, ax_hsi, ax_bic, ax_pac):
        ax.set_xlim(t0, t1)

    row = _match_ignition_event_row(ign_out, ign_win)
    seed_roi = str(row.get('seed_roi')) if row is not None and 'seed_roi' in row else 'N/A'
    type_label = str(row.get('type_label')) if row is not None and 'type_label' in row else 'unknown'

    bic_peak = float(np.nanmax(bic_peaks)) if bic_peaks else np.nan
    R_mask = (t_R >= ign_win[0]) & (t_R <= ign_win[1])
    R_peak = float(np.nanmax(R_smooth[R_mask])) if np.any(R_mask) else np.nan

    # badge_parts = [
    #     f'Seed ROI {seed_roi}',
    #     f'1× z_peak {z_peak:.2f}' if np.isfinite(z_peak) else '1× z_peak n/a',
    #     f'PLV_mean {plv_mean:.2f}' if np.isfinite(plv_mean) else 'PLV_mean n/a',
    #     f'R_peak {R_peak:.2f}' if np.isfinite(R_peak) else 'R_peak n/a',
    #     f'Triad Bic_peak {bic_peak:.2f}' if np.isfinite(bic_peak) else 'Triad Bic_peak n/a',
    #     f'ΔPAC {delta_pac:+.2f}' if np.isfinite(delta_pac) else 'ΔPAC n/a',
    #     type_label,
    # ]
    # ax_badge.axis('off')
    # ax_badge.text(0.01, 0.5, ' | '.join(badge_parts), ha='left', va='center', fontsize=11, fontweight='bold')

    fig.suptitle(f'Ignition SR signature {ign_win[0]}–{ign_win[1]} s\n{session_name}', fontsize=14, y=0.95, color=tick_color)
    return fig


def ignition_signature_panel(records, electrodes, ign_win, ign_out, ladder, cfg, session_name):
    TIME_COL = 'Timestamp'
    FS = cfg.fs or _infer_fs(records, TIME_COL)
    pack = build_ignition_feature_pack(records, [ign_win], cfg=cfg)
    provider = PackProvider(pack).slice(ign_win[0], ign_win[1])
    t = provider.t()
    zf = provider.z_fund()
    plv = provider.plv_fund()
    pac = provider.pac_mvl()

    zf_z = smooth_sec(t, robust_z(np.asarray(zf, float)), 0.15)
    plv = np.asarray(plv, float)
    pac = np.asarray(pac, float) if pac is not None else np.zeros_like(plv)
    pac = smooth_sec(t, pac, 0.8)

    spec = provider.spectrogram_for_window(t.min(), t.max())
    if spec is None:
        spec = pack.get('spec')
    if spec is None:
        spec = window_spec_median(
            records, ign_win, channels=electrodes, fs=FS, time_col=TIME_COL,
            band=(2,60), win_sec=cfg.spec_win, overlap=cfg.spec_ovl
        )
    t_spec, f_spec, S_spec = _slice_spec_to_window(spec, (t.min(), t.max()), min_cols=40)
    _, slopes_raw = _spectral_slope_series(t_spec, f_spec, S_spec, exclude_centers=ladder, exclude_bw=1.0)  # Exclude all harmonics
    slopes = smooth_sec(t_spec, slopes_raw, 1.0)

    baseline_offset = 60.0
    baseline_duration = max(ign_win[1] - ign_win[0], 5.0)
    t_base, mask_base = _baseline_slice(records, TIME_COL, ign_win, baseline_offset, baseline_duration)
    slopes_base = np.array([])
    if mask_base.size and np.sum(mask_base) > 5:
        baseline_window = (float(t_base[0]), float(t_base[-1]))
        tWb, fWb, SWb = window_spec_median(records, baseline_window, channels=electrodes, fs=FS,
                                           time_col=TIME_COL, band=(2,60), win_sec=cfg.spec_win, overlap=cfg.spec_ovl)
        tWb = tWb + cfg.spec_win/2.0
        _, slopes_base_raw = _spectral_slope_series(tWb, fWb, SWb, exclude_centers=ladder, exclude_bw=1.0)  # Exclude all harmonics
        slopes_base = smooth_sec(tWb, slopes_base_raw, 1.0)

    fig, ax = plt.subplots(figsize=(14, 6), dpi=160)
    ax.plot(t_spec, slopes, color='tab:blue', lw=2.0, label='Aperiodic β(t)')
    if slopes_base.size:
        base_lo, base_hi = np.nanpercentile(slopes_base, [10, 90])
        ax.fill_between([t_spec.min(), t_spec.max()], base_lo, base_hi, color='gray', alpha=0.1,
                        label='Baseline β 10–90%')
        delta = float(np.nanmean(slopes) - np.nanmean(slopes_base))
        ax.text(0.02, 0.9, f'Δβ ≈ {delta:.2f}', transform=ax.transAxes,
                fontsize=11, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    ax.set_ylabel('PSD slope β')
    ax.set_xlabel('Time (s)')

    ax_twin = ax.twinx()
    env_norm = zf_z - np.nanmin(zf_z)
    env_range = np.nanmax(env_norm) - np.nanmin(env_norm) + 1e-6
    env_norm = np.clip(env_norm / env_range, 0, 1)
    ax_twin.plot(t, env_norm, color='tab:orange', lw=1.6, alpha=0.85, label='Fundamental z (norm)')
    plv_norm = (plv - np.nanmin(plv)) / (np.nanmax(plv) - np.nanmin(plv) + 1e-6)
    ax_twin.plot(t, plv_norm, color='tab:green', lw=1.3, alpha=0.7, label='PLV (norm)')
    pac_norm = (pac - np.nanmin(pac)) / (np.nanmax(pac) - np.nanmin(pac) + 1e-6)
    ax_twin.plot(t, pac_norm, color='tab:purple', lw=1.1, alpha=0.6, label='θ→γ PAC (norm)')
    ax_twin.set_ylabel('Normalized (0–1)')
    ax_twin.set_ylim(-0.1, 1.1)

    # Extract t0_net from ign_out for phase detection anchoring (with fallback to sr_z_peak_t)
    row = _match_ignition_event_row(ign_out, ign_win)
    t0_net = None
    if row is not None:
        if 't0_net' in row:
            t0_val = row['t0_net']
            if pd.notna(t0_val) and np.isfinite(float(t0_val)):
                t0_net = float(t0_val)
        if t0_net is None and 'sr_z_peak_t' in row:
            # Fallback: estimate t0 as ~3s before amplitude peak
            sr_peak_val = row['sr_z_peak_t']
            if pd.notna(sr_peak_val) and np.isfinite(float(sr_peak_val)):
                sr_peak_t = float(sr_peak_val)
                t0_net = max(ign_win[0], sr_peak_t - 3.0)

    phases = _detect_ignition_phases(
        t, zf_z, provider.plv_fund(), provider.hsi(), provider.z_h2(), provider.z_h3(),
        params=PhaseParams(f0=ladder[0]), seed_t=t0_net if t0_net is not None else 'center',
        p0_band=(-0.5, +0.3), p1_band=(-1.0, +1.4), pad_s=2.0,
    )
    annotate_phases(ax, phases, *ax.get_ylim())

    lines, labels = ax.get_legend_handles_labels()
    l2, lab2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines + l2, labels + lab2, loc='upper right', fontsize=10)
    ax.set_title(f'Ignition Signature — {session_name} (window {ign_win[0]}–{ign_win[1]}s)')

    return fig


def six_panel_3(records, electrodes, ign_win, ign_out, ladder, cfg, session_name, *, H=None):
    TIME_COL = 'Timestamp'
    FS = cfg.fs or _infer_fs(records, TIME_COL)
    t_all = np.asarray(records[TIME_COL], float)
    pack = build_ignition_feature_pack(records, [ign_win], cfg=cfg)
    provider = PackProvider(pack).slice(ign_win[0], ign_win[1])
    t = provider.t()
    zf = provider.z_fund()
    plv = provider.plv_fund()

    # 1-2) Harmonic mode engagement
    X = _get_matrix(records, electrodes)
    mask_ign = (t_all >= ign_win[0]) & (t_all <= ign_win[1])
    baseline_offset = 60.0
    baseline_window = (ign_win[0] - baseline_offset - (ign_win[1]-ign_win[0]), ign_win[0]-baseline_offset)
    mask_base = (t_all >= baseline_window[0]) & (t_all <= baseline_window[1])
    if mask_base.sum() < 10:
        mask_base = ~mask_ign

    eigvals = eigvecs = None
    mode_power_base = mode_power_ign = None
    if H is not None:
        X_base_proj = project_to_harmonics(X[:, mask_base], H)
        X_ign_proj = project_to_harmonics(X[:, mask_ign], H)
        mode_power_base = np.mean(X_base_proj**2, axis=1)
        mode_power_ign = np.mean(X_ign_proj**2, axis=1)
        coeff_series = np.mean(project_to_harmonics(X[:, mask_ign], H[:,:4])**2, axis=1)
    else:
        mode_power_base = np.nan
        coeff_series = np.array([])

    # 3-4) Topology metrics
    # Build functional connectivity using PLV
    # Handle bw_hz as either scalar or array
    if np.ndim(cfg.bw_hz) == 0:  # scalar
        bw_f1 = float(cfg.bw_hz)
    else:  # array - use first element for fundamental frequency
        bw_array = np.asarray(cfg.bw_hz)
        bw_f1 = bw_array[0]

    X_ign = X[:, mask_ign]
    plv_matrix = _plv_matrix(X_ign, FS, ladder[0], bw_f1)
    baseline_matrix = _plv_matrix(X[:, mask_base], FS, ladder[0], bw_f1)
    import networkx as nx
    G_ign = nx.from_numpy_array(plv_matrix)
    G_base = nx.from_numpy_array(baseline_matrix)
    mod_base = nx.algorithms.community.modularity(G_base, [range(len(electrodes))])
    mod_ign = nx.algorithms.community.modularity(G_ign, [range(len(electrodes))])
    integ_time = plv_matrix.mean()

    # 5-6) Directed connectivity
    te_matrix = _te_matrix(X_ign, FS)
    te_base = _te_matrix(X[:, mask_base], FS)
    te_diff = te_matrix - te_base

    fig = plt.figure(figsize=(16, 10), constrained_layout=True, dpi=160)
    gs = GridSpec(3, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    if mode_power_base is not None and np.size(mode_power_base) > 1:
        n_modes = min(12, len(mode_power_base))
        x = np.arange(n_modes)
        width = 0.35
        ax1.bar(x - width/2, mode_power_base[:n_modes], width, label='Baseline', alpha=0.6)
        ax1.bar(x + width/2, mode_power_ign[:n_modes], width, label='Ignition', alpha=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'M{k}' for k in range(n_modes)])
        ax1.set_ylabel('Mode power')
        ax1.set_title('Connectome modes engagement')
        ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    if measure := pack.get('mode_time_series'):
        ax2.plot(measure['t'], measure['fundamental'], label='Fundamental mode')
        ax2.plot(measure['t'], measure['second'], label='2nd mode', alpha=0.7)
        ax2.axvspan(ign_win[0], ign_win[1], color='gold', alpha=0.1)
        ax2.set_title('Mode amplitude timeline')
        ax2.legend()

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(baseline_matrix, vmin=0, vmax=1, cmap='Blues')
    ax3.set_title('Functional graph – baseline')

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(plv_matrix, vmin=0, vmax=1, cmap='Oranges')
    ax4.set_title('Functional graph – ignition')

    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(t, smooth_sec(t, plv, 0.3), label='PLV (fundamental)')
    ax52 = ax5.twinx()
    ax5.plot(t, zf_z, label='Fundamental envelope', color='tab:orange')
    ax5.set_title('Integration vs modularity')
    ax5.legend()

    ax6 = fig.add_subplot(gs[2, 1])
    im = ax6.imshow(te_diff, cmap='coolwarm', vmin=-0.2, vmax=0.2)
    fig.colorbar(im, ax=ax6, fraction=0.05, pad=0.02)
    ax6.set_title('Transfer entropy Δ (ign - baseline)')

    fig.suptitle(f'Ignition Six Panel 3 — {session_name} | Window {ign_win[0]}–{ign_win[1]}s')
    return fig

from dataclasses import asdict
from dataclasses import dataclass


import numpy as np
import pandas as pd
import re

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from typing import Any, Dict, Optional, Sequence,Tuple, List, Iterable
from scipy.signal import stft, firwin, filtfilt, detrend, savgol_filter, hilbert, stft, welch

import matplotlib as mpl
mpl.rcParams.update({
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
})


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
    bw_hz: float = 0.5
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
    print(out)
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
    session_name=None
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

    # >>> pass the seed & bands to the detector <<<
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
    # tW, fW, SW = window_spec_median(_records, (t.min(), t.max()), channels=electrodes, fs=128, time_col='Timestamp')
    # tW, fW, SW = provider.spectrogram_for_window(t.min(), t.max())
    tW, fW, SW = _slice_spec_to_window(provider.spectrogram_for_window(t.min(), t.max()), (t.min(), t.max()), min_cols=20)
    
    # row-wise robust-z in dB
    Zdb = 10*np.log10(np.maximum(SW, 1e-12))
    med = np.median(Zdb, axis=1, keepdims=True)
    mad = np.median(np.abs(Zdb - med), axis=1, keepdims=True) + 1e-12
    Z   = (Zdb - med) / (1.4826*mad)
    im = axA.pcolormesh(tW, fW, Z, shading='auto'); 
    im.set_clim(-3, 3)
        
    axA.set_ylabel('Hz'); axA.set_title('SR Spectrogram (2–60 Hz)')
    axA.set_xlabel('s')
    annotate_phases(axA, phases, *axA.get_ylim())


    # # PANEL B: HARMONIC PIANO ROLL ============
    axB = fig.add_subplot(gs[0,1])
    
    # # Use the exact spectrogram used in Panel A
    tW, fW, SW = _slice_spec_to_window(provider.spectrogram_for_window(t.min(), t.max()),
                                   (t.min(), t.max()), min_cols=20)

    # Plot
    PR = np.vstack([zf_z, z2_z, z3_z])
    im = axB.imshow(PR, origin='lower', aspect='auto',
                extent=[t.min(), t.max(), 0.5, 3.5], vmin=-3, vmax=3)

    axB.set_yticks([1, 2, 3]); 
    axB.set_yticklabels([f"1× ({_fnd})", f"2× ({_2nd})", f"3× ({_3rd})"])
    axB.set_title('Harmonic Piano‑Roll (envelope z)')
    axB.set_xlabel('s')
    annotate_phases(axB, phases, 0.5, len(centers) + 0.5)
    fig.colorbar(im, ax=[axA, axB], pad=0.02, fraction=0.05).set_label('z')

    # window-robust-z + 150 ms smoothing (same as Panel B piano-roll)
    def z_for_display(t, y):
        return smooth_sec(t, robust_z(y), 0.15)
    
    zf_d = z_for_display(t, provider.z_fund())
    z2_d = z_for_display(t, provider.z_h2())
    z3_d = z_for_display(t, provider.z_h3())



    # PANEL C: Fundamental & harmonics envelopes + PLV ===============
    axC = fig.add_subplot(gs[1, 0])
    axC.plot(t, zf_z, label=f"z@{_fnd}", lw=1.5)
    axC.plot(t, z2_z, label=f"z@{_2nd}", lw=1.3)
    axC.plot(t, z3_z, label=f"z@{_3rd}", lw=1.3)
    axC.set_ylabel('z')
    # axC.set_ylim(-3.5, 3.5)


    axC2 = axC.twinx()
    axC2.plot(t, provider.plv_fund(), label=f"PLV@{_fnd}", ls='--', lw=1.4, alpha=0.95)
    plo, phi = np.nanpercentile(plv, [1, 99])
    pad = 0.1 * (phi - plo + 1e-12)
    axC.set_title('Envelopes and PLV')
    axC.set_xlabel('s'); 
    axC.set_ylabel('z'); 
    axC2.set_ylabel('PLV')
    axC.legend(loc='upper left'); 
    axC2.legend(loc='upper right')

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
        axD2.legend(loc='upper right')

    axD.set_title('Harmonic Tightening')
    axD.set_xlabel('s'); axD.set_ylabel(ylab)
    axD.legend(loc='upper left')

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

    # Panel E: Focused bicoherence tiles
    axE = fig.add_subplot(gs[2, 0])
    if (b77 is not None) and (b7_15 is not None):
        axE.plot(t, b77, label=f"bic ({_fnd},{_fnd}→{_2nd})")
        axE.plot(t, b7_15, label=f"bic ({_fnd},{_2nd}→{_3rd})")
        axE.legend(loc='upper right')
    else:
        axE.text(0.5, 0.5, 'Bicoherence not provided', transform=axE.transAxes, ha='center', va='center')
    axE.set_title('Bicoherence (SR triads)')
    axE.set_xlabel('s')
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
    axF.set_xlabel('s')
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
    for a, b in np.atleast_2d(windows):
        mask |= (t_all >= a - margin) & (t_all <= b + margin)

    t = t_all[mask]                                   # segment time base
    X = _get_matrix(_records, channels)[:, mask]

    # Fundamental & harmonics envelopes (median across channels)
    z1, ph1 = _narrowband_envelope_z(X, fs, cfg.sr_centers[0], cfg.bw_hz)
    z2, ph2 = _narrowband_envelope_z(X, fs, cfg.sr_centers[1], cfg.bw_hz)
    z3, ph3 = _narrowband_envelope_z(X, fs, cfg.sr_centers[2], cfg.bw_hz)

    # PLV@7.8 over time (sliding)
    t_plv, plv = _plv_timecourse(X, fs, cfg.sr_centers[0], cfg.bw_hz, cfg.win_sec, cfg.step_sec)

    # HSI over time
    t_hsi, hsi = _hsi_timecourse(X, fs, cfg.win_sec, cfg.step_sec, cfg.ladder, ladder_bw=cfg.ladder_bw)

    # Focused bicoherence triads and PAC (optional but cheap here)
    t_bic, bic = _bicoherence_triads_timecourse(
        X, fs,
        triads=[(cfg.sr_centers[0], cfg.sr_centers[0], cfg.sr_centers[1]),
                (cfg.sr_centers[0], cfg.sr_centers[1], cfg.sr_centers[2])],
        bw=cfg.bw_hz, win_sec=cfg.win_sec, step_sec=cfg.step_sec,
    )

    t_pac, mvl = _pac_mvl_timecourse(X, fs, theta_band=(7.0,8.0), gamma_band=(40.0,100),
                                     win_sec=cfg.win_sec, step_sec=cfg.step_sec)

    # t_abs = np.asarray(_records[time_col], float)
    # t0 = float(t_abs[0])

    # 2) map sliding times to absolute **segment** seconds
    t_seg0 = float(t[0])
    def to_abs(ts, ys): return np.interp(t, t_seg0 + ts, ys, left=ys[0], right=ys[-1])

    # In build_ignition_feature_pack
    plv_series = _plv_7p8(X, fs, f0=cfg.sr_centers[0], bw=cfg.bw_hz, win=cfg.win_sec, step=cfg.step_sec)    

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

def annotate_phases(ax, phases: Dict[str, Any], ymin: float, ymax: float,
                    *, highlight_padding: float = 0.25) -> None:
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
        ax.axvspan(start, end, color='#FFF59D33', lw=0)
        ax.text((start + end) * 0.5, ymax + 0.02 * (ymax - ymin), 'Ignition',
                ha='center', va='bottom', fontsize=8, color='#424242')

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
        ax.axvspan(time - half_width, time + half_width, color=color, alpha=0.08, lw=0)
        ax.vlines(time, ymin, ymax, linestyles='--', linewidth=1.3, color=color,
                  alpha=0.7 + 0.3 * min(1.0, conf))
        ax.text(time, ymin, name, rotation=90, va='bottom', ha='center', color=color, fontsize=8)
        if conf < 0.45:
            ax.text(time, ymax, f"{conf:.2f}", rotation=90, va='top', ha='center',
                    color=color, fontsize=7, alpha=0.6)


def six_panel(records,electrodes,ign_win,ign_out,ladder,cfg,session_name):
    assert electrodes and len(electrodes) > 0, "electrodes cannot be empty"
    
    # --- parameters ------------------------------------------------------------
    TIME_COL = "Timestamp"
    FS = 128.0
    
    # BUILD PACK
    pack = build_ignition_feature_pack(records,ign_win,cfg=cfg)
    pack.setdefault('meta', {})['channels_used'] = list(electrodes)

    m = (pack['t'] >= ign_win[0]) & (pack['t'] <= ign_win[1])
    
    # SPECTROGRAM - coarse spectrogram for Panel A + HSI (single source of truth)
    pack['spec'] = compute_session_spectrogram(
        records, channels=electrodes, time_col='Timestamp', fs=FS,
        band=(2,60), win_sec=cfg.spec_win, overlap=cfg.spec_ovl
    )

    # PIANO ROLL
    tWc, fWc, SWc = window_spec_median(
        records, ign_win, channels=electrodes, fs=FS, time_col='Timestamp',
        band=(2,60), win_sec=cfg.spec_win, overlap=cfg.spec_ovl
    )
    tWc = tWc + cfg.spec_win/2.0  # center STFT times
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

    # PLV
    t_plv, plv = _plv_timecourse(
        X, fs=FS, f0=ladder[0], bw=cfg.bw_hz,
        win_sec=cfg.win_sec, step_sec=cfg.step_sec)
    pack['plv_7p83'] = np.interp(pack['t'], t0 + t_plv, plv, left=plv[0], right=plv[-1])

    # PAC MVL
    t_pac, mvl = _pac_mvl_timecourse(X, fs=128.0,theta_band=(cfg.sr_centers[0]-1,cfg.sr_centers[0]+1), gamma_band=(30,60),
                                            win_sec=cfg.win_sec, step_sec=cfg.step_sec,amp_gate_pct=70)
    pack['pac_mvl'] = np.interp(pack['t'], t0+t_pac, mvl, left=mvl[0], right=mvl[-1])

    
    # THRESHOLDS
    z7_win = robust_z(pack['z_7p83'][m])
    z95    = float(np.nanpercentile(z7_win, 95))
    h50    = float(np.nanpercentile(pack['hsi'][m], 50))   # for hsi_broad
    plv60  = float(np.nanpercentile(pack['plv_7p83'][m], 60))
    h10    = float(np.nanpercentile(pack['hsi'][m], 10))
    
    
    # SEED EVENT
    ev = ign_out['events']
    m_overlap = (ev['t_start'] < ign_win[1]) & (ev['t_end'] > ign_win[0])
    if m_overlap.any():
        seed_from_event = float(np.clip(ev.loc[m_overlap, 'sr_z_peak_t'].iloc[0], ign_win[0]+2.0, ign_win[1]-2.0))
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


    # PLOT IGNITION WINDOW
    fig, phases, traces = plot_ignition_window_report(
        records, 
        PackProvider(pack).slice(ign_win[0],ign_win[1]), 
        electrodes,
        params=params,
        hsi_plot_mode="delta", hsi_ylim=("pct",(1,99)),
        seed_t="center",
        p0_band=(-2,+0.6),                      # seconds relative to seed_t
        p1_band=(-1, +1.4),                      # seconds relative to seed_t
        pad_s=2.0,
        title=f"Ignition {ign_win[0]}–{ign_win[1]}s",
        centers=ladder[:3],
        session_name=session_name
    )


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

    pack = build_ignition_feature_pack(records, ign_win, cfg=cfg)
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

    centers = ladder[:4] if len(ladder) >= 4 else ladder
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
    mask_base = (t_all >= ign_win[0] - 60 - (ign_win[1]-ign_win[0])) & (t_all <= ign_win[0] - 60)
    if mask_base.sum() < 10:
        mask_base = ~mask_seg
    X = X_full[:, mask_seg]
    X_base = X_full[:, mask_base]
    t_seg = t_all[mask_seg]
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

    params = PhaseParams()
    params.f0 = centers[0]
    phases = _detect_ignition_phases(
        t, zf_z, plv, hsi, z2, z3,
        params=params,
        seed_t='center',
        p0_band=(-2.0, +0.6),
        p1_band=(-1.0, +1.4),
        pad_s=2.0,
    )

    baseline_offset = 60.0
    baseline_duration = max(ign_win[1] - ign_win[0], 5.0)
    t_base, mask_base = _baseline_slice(records, TIME_COL, ign_win, baseline_offset, baseline_duration)
    sizes_base = durations_base = np.array([])
    slopes_base = np.array([])
    R_base_series = np.array([])
    t_R_base = np.array([])
    var_base = np.array([])
    if mask_base.size and np.sum(mask_base) > 5:
        baseline_window = (float(t_base[0]), float(t_base[-1]))
        tWb, fWb, SWb = window_spec_median(
            records, baseline_window, channels=electrodes, fs=FS,
            time_col=TIME_COL, band=(2,60), win_sec=cfg.spec_win, overlap=cfg.spec_ovl
        )
        tWb = tWb + cfg.spec_win/2.0
        _, slopes_base = _spectral_slope_series(tWb, fWb, SWb)

        z_base, _ = _narrowband_envelope_z(X_base, FS, centers[0], bw0)
        z2_base = z3_base = z4_base = None
        if len(centers) > 1:
            z2_base, _ = _narrowband_envelope_z(X_base, FS, centers[1], bw0)
        if len(centers) > 2:
            z3_base, _ = _narrowband_envelope_z(X_base, FS, centers[2], bw0)
        if len(centers) > 3:
            z4_base, _ = _narrowband_envelope_z(X_base, FS, centers[3], bw0)
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
    harmonics_for_msc = centers[:3] if len(centers) >= 3 else centers
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
        fallback_labels = [f'{f:.2f} Hz' for f in ladder[:min(3, len(ladder))]]
        ign_vals = []
        base_vals = []
        for idx_mode in range(len(fallback_labels)):
            if idx_mode == 0:
                ign_vals.append(float(np.nanmean(np.abs(provider.z_fund()))))
                base_vals.append(float(np.nanmean(np.abs(z_base))))
            elif idx_mode == 1 and z2_base is not None:
                ign_vals.append(float(np.nanmean(np.abs(provider.z_h2()))))
                base_vals.append(float(np.nanmean(np.abs(z2_base))))
            elif idx_mode == 2 and z3_base is not None:
                ign_vals.append(float(np.nanmean(np.abs(provider.z_h3()))))
                base_vals.append(float(np.nanmean(np.abs(z3_base))))
            else:
                ign_vals.append(np.nan)
                base_vals.append(np.nan)
        idx = np.arange(len(fallback_labels))
        width = 0.35
        ax_modes.bar(idx - width/2, base_vals, width, alpha=0.6, label='Baseline')
        ax_modes.bar(idx + width/2, ign_vals, width, alpha=0.8, label='Ignition')
        ax_modes.set_xticks(idx)
        ax_modes.set_xticklabels(fallback_labels)
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

    fig.suptitle(f'Ignition {ign_win[0]}–{ign_win[1]}s\n{session_name}', fontsize=14)
    return fig


def sr_signature_panel(records, electrodes, ign_win, ign_out, ladder, cfg, session_name):
    TIME_COL = cfg.time_col
    FS = cfg.fs or _infer_fs(records, TIME_COL)
    t_all = np.asarray(records[TIME_COL], float)
    if t_all.size < 2:
        raise ValueError('Not enough samples to build SR signature panel')
    X = _get_matrix(records, electrodes)

    f1 = ladder[0]
    f2 = ladder[1] if len(ladder) > 1 else f1 * 2.0
    f3 = ladder[2] if len(ladder) > 2 else f1 * 3.0
    bw = cfg.bw_hz if cfg.bw_hz is not None else 0.5

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
        band=(5, 25), win_sec=2.0, overlap=0.5
    )
    spec_z = _spec_db_rowz(S_spec)

    # Harmonic envelopes and PLV
    env1_full, _ = _narrowband_envelope_z(X, FS, f1, bw)

    def _safe_envelope(center_hz: float) -> np.ndarray:
        if center_hz >= (0.48 * FS):
            return np.full_like(env1_full, np.nan, dtype=float)
        amp, _ = _narrowband_envelope_z(X, FS, center_hz, bw)
        return amp

    env2_full = _safe_envelope(f2)
    env3_full = _safe_envelope(f3)
    def _to_z(env):
        if not np.isfinite(env).any():
            return np.full_like(env, np.nan, dtype=float)
        return smooth_sec(t_all, robust_z(env), 0.25)

    env1_z = _to_z(env1_full)
    env2_z = _to_z(env2_full)
    env3_z = _to_z(env3_full)

    t_plv_rel, plv_series = _plv_timecourse(X, FS, f1, bw, win_sec=1.0, step_sec=0.25)
    plv_times = t_all[0] + t_plv_rel

    # ΔHSI proxy from mean-centered z envelopes
    valid_envs = [env for env in (env1_z, env2_z, env3_z) if np.isfinite(env).any()]
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
    if len(ladder) >= 2:
        triads.append((f1, f1, f2))
    else:
        triads.append((f1, f1, 2 * f1))
    if len(ladder) >= 3:
        triads.append((f1, f2, f3))
    else:
        triads.append((f1, 2 * f1, 3 * f1))
    t_bic_rel, bic_raw = _bicoherence_triads_timecourse(X, FS, triads, bw, win_sec=1.0, step_sec=0.25)
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
    t_R_rel, R_series = _kuramoto_order_series(X, FS, f1, bw)
    t_R = t_all[0] + t_R_rel
    R_smooth = smooth_sec(t_R, R_series, 0.3)

    fig = plt.figure(figsize=(14, 12), dpi=160)
    gs = GridSpec(5, 1, height_ratios=[2.3, 1.6, 1.3, 1.2, 1.2], hspace=0.5)
    # ax_badge = fig.add_subplot(gs[0])
    ax_spec = fig.add_subplot(gs[0])
    ax_env = fig.add_subplot(gs[1], sharex=ax_spec)
    ax_hsi = fig.add_subplot(gs[2], sharex=ax_spec)
    ax_bic = fig.add_subplot(gs[3], sharex=ax_spec)
    ax_pac = fig.add_subplot(gs[4], sharex=ax_spec)

    extent = [t_spec[0], t_spec[-1], f_spec[0], f_spec[-1]]
    im = ax_spec.imshow(spec_z, extent=extent, origin='lower', aspect='auto', cmap='Spectral_r', vmin=-3, vmax=3)
    ax_spec.set_ylabel('Frequency (Hz)')
    ax_spec.set_title('SR-focused spectrogram (row-z)')
    for freq in (f1, f2, f3):
        ax_spec.axhline(freq, color='white', linestyle='--', linewidth=1.0, alpha=0.8)
    ax_spec.axvspan(ign_win[0], ign_win[1], color='gold', alpha=0.18)
    cbar = fig.colorbar(im, ax=ax_spec, pad=0.01, fraction=0.04)
    cbar.set_label('z (per frequency)')

    ax_env.plot(t_all, env1_z, label=f'{f1:.2f} Hz', color='tab:orange', linewidth=1.6)
    ax_env.plot(t_all, env2_z, label=f'{f2:.2f} Hz', color='tab:green', linewidth=1.1)
    ax_env.plot(t_all, env3_z, label=f'{f3:.2f} Hz', color='tab:purple', linewidth=1.1)
    ax_env.axvspan(ign_win[0], ign_win[1], color='gold', alpha=0.18)
    ax_env.set_ylabel('Envelope z')
    ax_env.set_title('Harmonic envelopes (z) with PLV')
    ax_env.legend(loc='upper left', fontsize=8, frameon=False)
    ax_env.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax_env2 = ax_env.twinx()
    ax_env2.plot(plv_times, plv_series, linestyle='--', color='tab:blue', linewidth=1.4, label='PLV @ fundamental')
    ax_env2.set_ylabel('PLV')
    ax_env2.set_ylim(0, 1.05)
    ax_env2.axvspan(ign_win[0], ign_win[1], color='gold', alpha=0.18)
    handles2, labels2 = ax_env2.get_legend_handles_labels()
    if handles2:
        ax_env2.legend(handles2, labels2, loc='upper right', fontsize=8, frameon=False)

    window_mask = (t_all >= ign_win[0]) & (t_all <= ign_win[1])
    z_peak = float(np.nanmax(env1_z[window_mask])) if np.any(window_mask) else np.nan
    if np.any(window_mask):
        idx_peak = np.nanargmax(env1_z[window_mask])
        t_peak = t_all[window_mask][idx_peak]
        ax_env.plot(t_peak, z_peak, marker='o', color='tab:orange')

    plv_win_mask = (plv_times >= ign_win[0]) & (plv_times <= ign_win[1])
    plv_mean = float(np.nanmean(plv_series[plv_win_mask])) if np.any(plv_win_mask) else np.nan
    if np.isfinite(plv_mean):
        ax_env2.axhline(plv_mean, color='tab:blue', linestyle=':', linewidth=1.0)

    ax_hsi.plot(t_all, delta_hsi, color='tab:red', linewidth=1.4)
    ax_hsi.fill_between(t_all, 0, delta_hsi, where=delta_hsi >= 0, color='tab:red', alpha=0.2)
    ax_hsi.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax_hsi.axvspan(ign_win[0], ign_win[1], color='gold', alpha=0.18)
    ax_hsi.set_ylabel('ΔHSI (a.u.)')
    ax_hsi.set_title('Harmonic tightening (ΔHSI proxy)')

    colors = ['tab:cyan', 'tab:pink']
    bic_peaks = []
    for idx, key in enumerate(triad_keys):
        series = bic[key]
        ax_bic.plot(t_bic, series, color=colors[idx % len(colors)], linewidth=1.3, label=key)
        win_mask = (t_bic >= ign_win[0]) & (t_bic <= ign_win[1])
        if np.any(win_mask):
            peak_idx = np.nanargmax(series[win_mask])
            abs_idx = np.flatnonzero(win_mask)[peak_idx]
            ax_bic.plot(t_bic[abs_idx], series[abs_idx], marker='*', color=colors[idx % len(colors)], markersize=10)
            bic_peaks.append(float(series[abs_idx]))
    ax_bic.set_ylabel('Bicoherence')
    ax_bic.set_ylim(0, 1.05)
    ax_bic.set_title('Triadic bicoherence')
    ax_bic.axvspan(ign_win[0], ign_win[1], color='gold', alpha=0.18)
    ax_bic.legend(loc='upper right', fontsize=8, frameon=False)

    pac_base_mask = (t_pac >= (ign_win[0] - pad)) & (t_pac < ign_win[0])
    pac_win_mask = (t_pac >= ign_win[0]) & (t_pac <= ign_win[1])
    pac_base = np.nanmean(pac_vals[pac_base_mask]) if np.any(pac_base_mask) else np.nan
    pac_win = np.nanmean(pac_vals[pac_win_mask]) if np.any(pac_win_mask) else np.nan
    delta_pac = pac_win - pac_base if np.isfinite(pac_win) and np.isfinite(pac_base) else np.nan
    ax_pac.plot(t_pac, pac_vals, color='tab:purple', linewidth=1.4)
    ax_pac.axvspan(ign_win[0], ign_win[1], color='gold', alpha=0.18)
    ax_pac.axhline(pac_base, color='#888888', linestyle=':', linewidth=0.9) if np.isfinite(pac_base) else None
    if np.isfinite(pac_win):
        ax_pac.axhline(pac_win, color='tab:purple', linestyle='--', linewidth=1.0)
    ax_pac.set_ylabel('MVL')
    ax_pac.set_xlabel('Time (s)')
    ax_pac.set_title('θ→γ PAC (MVL)')

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

    fig.suptitle(f'Ignition SR signature {ign_win[0]}–{ign_win[1]} s\n{session_name}', fontsize=14,y=0.95)
    return fig


def ignition_signature_panel(records, electrodes, ign_win, ign_out, ladder, cfg, session_name):
    TIME_COL = 'Timestamp'
    FS = cfg.fs or _infer_fs(records, TIME_COL)
    pack = build_ignition_feature_pack(records, ign_win, cfg=cfg)
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
    _, slopes_raw = _spectral_slope_series(t_spec, f_spec, S_spec, exclude_centers=ladder[:3], exclude_bw=1.0)
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
        _, slopes_base_raw = _spectral_slope_series(tWb, fWb, SWb, exclude_centers=ladder[:3], exclude_bw=1.0)
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

    phases = _detect_ignition_phases(
        t, zf_z, provider.plv_fund(), provider.hsi(), provider.z_h2(), provider.z_h3(),
        params=PhaseParams(f0=ladder[0]), seed_t='center', p0_band=(-2.0, +0.6), p1_band=(-1.0, +1.4), pad_s=2.0,
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
    pack = build_ignition_feature_pack(records, ign_win, cfg=cfg)
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
    X_ign = X[:, mask_ign]
    plv_matrix = _plv_matrix(X_ign, FS, ladder[0], cfg.bw_hz)
    baseline_matrix = _plv_matrix(X[:, mask_base], FS, ladder[0], cfg.bw_hz)
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

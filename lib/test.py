from dataclasses import asdict
from dataclasses import dataclass


import numpy as np
import pandas as pd

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
    min_p0_dur: float = 0.5
    # P1
    z_p1: float = 2.0
    plv_p1: float = 0.60
    ridge_required: bool = True
    beta_flat: Optional[float] = 1.2
    min_p1_dur: float = 0.5
    # P2
    hsi_tight: float = 0.30
    rel_h2: float = 0.30
    rel_h3: float = 0.30
    bic_7_7_15: float = 0.10
    bic_7_15_23: float = 0.10
    pac_mvl: Optional[float] = 0.20
    min_p2_cycles: float = 2.0
    # P3
    plv_release: float = 0.60
    hsi_release: float = 0.35
    rel_drop_k: int = 2

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
    debug=False
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

    

    fig.suptitle(title or 'Ignition Window Report', y=0.98)

    traces = {
        't': t, 'z_fund': zf_z, 'z_h2': z2_z, 'z_h3': z3_z, 'plv': plv, 'hsi': hsi,
        'beta': beta, 'ridge_is_fund': ridge, 'bic_7_7_15': b77, 'bic_7_15_23': b7_15, 'pac_mvl': pac,
        'phases': phases,
    }

    fig.suptitle(title, fontsize=16, y=1.05)
    # fig.set_constrained_layout_pads(wspace=0.02, hspace=0.06, w_pad=0.02, h_pad=0.02)

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
        Xb = filtfilt(b, [1.0], X, axis=-1, padlen=3*len(b))
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
):
    """Adaptive two‑stage ignition detector with ΔHSI‑based plateau gating."""
    # --- arrays & guards ---
    t   = np.asarray(t, float)
    zf  = np.asarray(z_fund, float)
    plv = np.asarray(plv_fund, float)
    hsi = np.asarray(hsi_t, float)
    z2  = np.asarray(z_h2, float)
    z3  = np.asarray(z_h3, float)
    assert t.size == zf.size == plv.size == hsi.size == z2.size == z3.size, "length mismatch"
    assert params is not None, "params required"

    # --- preproc ---
    zf_s  = smooth_sec(t, zf, 0.20)
    plv_s = smooth_sec(t, plv, 0.20)
    zf_z  = _winsor_robust_z(zf_s)

    # ΔHSI = smoothed HSI minus its median
    hsi_s   = smooth_sec(t, hsi, 0.45)
    dHSI    = hsi_s - float(np.nanmedian(hsi_s))

    ridge_ok = np.ones_like(zf, bool) if ridge_is_fund is None else np.asarray(ridge_is_fund, bool)
    beta_ok  = np.ones_like(zf, bool)
    if (beta_t is not None) and (getattr(params, 'beta_flat', None) is not None):
        beta_ok = (np.asarray(beta_t, float) <= params.beta_flat)

    # --- seeding & bands ---
    t_lo, t_hi = float(t[0] + pad_s), float(t[-1] - pad_s)
    if (seed_t is None) or (isinstance(seed_t, str) and seed_t.lower() == 'center'):
        seed_val = float(t[np.nanargmax(zf_z)])
    else:
        seed_val = float(np.clip(seed_t, t_lo, t_hi))

    p0_lo, p0_hi = max(t_lo, seed_val + p0_band[0]), min(t_hi, seed_val + p0_band[1])
    p1_lo, p1_hi = max(t_lo, seed_val + p1_band[0]), min(t_hi, seed_val + p1_band[1])
    p0_mask = _band_mask(t, p0_lo, p0_hi)
    p1_mask = _band_mask(t, p1_lo, p1_hi)

    # --- band‑adaptive P1 gates ---
    def _p1_gates(mask: np.ndarray):
        m = mask & np.isfinite(zf_z) & np.isfinite(plv_s)
        z_in   = zf_z[m]
        plv_in = plv_s[m]
        z95 = float(np.nanpercentile(z_in, 95)) if z_in.size else 1.6
        z92 = float(np.nanpercentile(z_in, 92)) if z_in.size else z95
        z_dyn = 0.9 * min(z95, z92)
        z_cap = float(getattr(params, 'z_p1_cap', 1.9))
        z_eff = float(np.clip(getattr(params, 'z_p1', z_dyn), 1.2, z_cap))
        q60   = float(np.nanpercentile(plv_in, 60)) if plv_in.size else float(np.nanmedian(plv_s))
        plv_eff = float(np.clip(getattr(params, 'plv_p1', q60 - 0.01), 0.62, 0.86))
        return z_eff, plv_eff

    z_p1_eff, plv_p1_eff = _p1_gates(p1_mask)

    # --- rising checks ---
    rz  = _rising_over_tau(zf_z,  t, getattr(params, 'z_rise_tau', 0.35), getattr(params, 'z_rise_eps', 0.03))
    rpl = _rising_over_tau(plv_s, t, getattr(params, 'plv_rise_tau', 0.25), getattr(params, 'plv_rise_eps', 0.008))

    # ---------------- P0 ----------------
    z_p0   = float(np.clip(getattr(params, 'z_p0', 0.6), 0.3, 1.2))
    plv_p0 = getattr(params, 'plv_p0', np.nanmedian(plv_s))
    m0 = (zf_z >= z_p0) & (plv_s >= plv_p0) & (hsi_s > getattr(params, 'hsi_broad', np.nanmedian(hsi_s))) & p0_mask
    i0 = _first_onset(m0, t, getattr(params, 'min_p0_dur', 0.10))

    # ---------------- P1 (two‑stage) ----------------
    # Stage‑1: z‑led proto‑core (no PLV requirement yet)
    proto = (zf_z >= z_p1_eff) & rz & ridge_ok & beta_ok & p1_mask

    # ±150 ms dilation to allow PLV catch-up
    dt  = float(np.median(np.diff(t)))
    pad = max(1, int(round(0.15 / max(dt, 1e-6))))
    proto_win = proto.copy()
    for k in range(1, pad+1):
        proto_win[:-k] |= proto[k:]
        proto_win[k:]  |= proto[:-k]

    core = proto_win & (plv_s >= plv_p1_eff) & p1_mask
    core = _bridge(core, t, 0.02)
    if i0 is not None:
        core[:i0] = False

    i1 = _first_onset(core, t, max(getattr(params, 'min_p1_dur', 0.12), dt))

    # widen‑retry if still missing: recompute gates in widened band
    if i1 is None:
        p1b = _band_mask(t, max(t_lo, p1_lo-1.0), min(t_hi, p1_hi+1.0))
        z_p1_eff_b, plv_p1_eff_b = _p1_gates(p1b)
        proto_b = (zf_z >= z_p1_eff_b) & rz & ridge_ok & beta_ok & p1b
        proto_win_b = proto_b.copy()
        for k in range(1, pad+1):
            proto_win_b[:-k] |= proto_b[k:]
            proto_win_b[k:]  |= proto_b[:-k]
        core_b = proto_win_b & (plv_s >= plv_p1_eff_b) & p1b
        core_b = _bridge(core_b, t, 0.02)
        if i0 is not None:
            core_b[:i0] = False
        i1 = _first_onset(core_b, t, max(getattr(params, 'min_p1_dur', 0.12), dt))
        if i1 is not None:
            p1_mask, core = p1b, core_b
            z_p1_eff, plv_p1_eff = z_p1_eff_b, plv_p1_eff_b

    # ---------------- P2 (ΔHSI tightening + overtone support) ----------------
    d_q = float(np.nanpercentile(dHSI, 15))
    rel2 = (z_h2 / (np.abs(zf) + 1e-9)) >= getattr(params, 'rel_h2', 0.05)
    rel3 = (z_h3 / (np.abs(zf) + 1e-9)) >= getattr(params, 'rel_h3', 0.05)

    m2 = (dHSI <= d_q) & (rel2 | rel3)
    if bic_7_7_15 is not None:
        m2 &= (np.asarray(bic_7_7_15, float) >= getattr(params, 'bic_7_7_15', 0.10))
    if bic_7_15_23 is not None:
        m2 &= (np.asarray(bic_7_15_23, float) >= getattr(params, 'bic_7_15_23', 0.10))
    if (pac_mvl is not None) and (getattr(params, 'pac_mvl', None) is not None):
        m2 &= (np.asarray(pac_mvl, float) >= params.pac_mvl)

    if i1 is not None:
        m2[:i1] = False

    dur2 = getattr(params, 'min_p2_cycles', 1.0) / max(getattr(params, 'f0', 7.83), 1e-6)
    i2 = _first_onset(m2, t, dur2)

    # ---------------- P3 ----------------
    i3 = None
    if i2 is not None:
        dz  = np.gradient(zf_z, t, edge_order=1)
        dz2 = np.gradient(smooth_sec(t, z_h2, 0.20), t, edge_order=1)
        dz3 = np.gradient(smooth_sec(t, z_h3, 0.20), t, edge_order=1)
        release = (plv_s < getattr(params, 'plv_release', np.nanpercentile(plv_s, 50))) \
                  | (dHSI > getattr(params, 'hsi_release', np.nanpercentile(dHSI, 60)))
        faster  = (dz2 < dz) & (dz2 < 0) & (dz3 < dz) & (dz3 < 0)
        k = max(1, getattr(params, 'rel_drop_k', 1))
        if k > 1:
            fk = faster.copy()
            for j in range(1, k):
                fk[:-j] &= faster[j:]
            faster = fk
        m3 = release & faster
        m3[:i2] = False
        m3 &= (t >= (t[i2] + 0.5))  # refractory
        i3 = _first_onset(m3, t, 0.0)

    # ---------------- debug prints ----------------
    # if getattr(params, 'debug', False):
    #     def _fmt(x): return 'None' if x is None else f"{x:.3f}"
    #     g_z   = ((zf_z >= z_p1_eff) & p1_mask).mean()
    #     g_plv = ((plv_s >= plv_p1_eff) & p1_mask).mean()
    #     rz_b  = (_rising_over_tau(zf_z, t, getattr(params, 'z_rise_tau', 0.35), getattr(params, 'z_rise_eps', 0.03)) & p1_mask).mean()
    #     rpl_b = (_rising_over_tau(plv_s, t, getattr(params, 'plv_rise_tau', 0.25), getattr(params, 'plv_rise_eps', 0.008)) & p1_mask).mean()
    #     print(f"[ignite] seed={seed_val:.3f} | bands P0=({_fmt(p0_lo)},{_fmt(p0_hi)})  P1=({_fmt(p1_lo)},{_fmt(p1_hi)}) | z_p0={z_p0:.2f} z_p1={z_p1_eff:.2f}")
    #     print(f"[ignite] P1 pass-rates  z>=z_p1:{g_z:.2f}  plv>=plv_p1:{g_plv:.2f}  rising_z:{rz_b:.2f}  rising_plv:{rpl_b:.2f}  band:{p1_mask.mean():.2f}")
    #     # longest contiguous run in core
    #     run = 0.0
    #     if 'core' in locals() and core.any():
    #         dt = float(np.median(np.diff(t)))
    #         cnt, best = 0, 0
    #         for v in core:
    #             if v: cnt += 1; best = max(best, cnt)
    #             else: cnt = 0
    #         run = best * dt
    #     P0t = None if i0 is None else float(t[i0])
    #     P1t = None if i1 is None else float(t[i1])
    #     P2t = None if i2 is None else float(t[i2])
    #     P3t = None if i3 is None else float(t[i3])
    #     print(f"[ignite] longest core run in band ≈ {run:.3f}s  (need ≥ {max(getattr(params,'min_p1_dur',0.12), float(np.median(np.diff(t)))):.3f}s)")
    #     print(f"[ignite] calls  P0={_fmt(P0t)}  P1={_fmt(P1t)}  P2={_fmt(P2t)}  P3={_fmt(P3t)}")

    def _pack(idx: Optional[int]) -> dict:
        return {'idx': None if idx is None else int(idx), 'time': None if idx is None else float(t[idx])}

    return {
        'P0': _pack(i0), 'P1': _pack(i1), 'P2': _pack(i2), 'P3': _pack(i3),
        'params': asdict(params) if params is not None else {},
        'seed_t': float(seed_val),
        'bands': {'P0': (float(p0_lo), float(p0_hi)), 'P1': (float(p1_lo), float(p1_hi))}
    }

def annotate_phases(ax, phases: Dict[str, Any], ymin: float, ymax: float):
    for name in ['P0','P1','P2','P3']:
        node = phases.get(name, {})
        t_on = node.get('time', None)
        if t_on is None:
            continue
        ax.vlines(t_on, ymin, ymax, linestyles='--', linewidth=1.5,color='cyan')
        ax.text(t_on, ymin, name, rotation=90, va='bottom', ha='center')


def six_panel(records,electrodes,ign_win,ign_out,ladder,cfg):
    assert electrodes and len(electrodes) > 0, "electrodes cannot be empty"
    
    # --- parameters ------------------------------------------------------------
    TIME_COL = "Timestamp"
    FS = 128.0
    # PACK_WIN   = 6     # sliding features
    # PACK_STEP  = 0.15
    
    # SPEC_WIN   = 1.5      # per-window spectrogram for HSI_v3
    # SPEC_OVERLAP =0.8
    
    # SPEC_WIN_FINE = 1.5         # try 4.0, 6.0, or 8.0 s
    # SPEC_OVL_FINE = 0.8         # high overlap for smoother time axis
    

    # cfg= FeaturePackCfg(
    #     channels=electrodes, time_col='Timestamp', fs=FS,
    #     win_sec=PACK_WIN, step_sec=PACK_STEP,
    #     sr_centers=ladder[:3],
    #     ladder=ladder,
    #     bw_hz=0.5,
    # )
    
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
    t_pac, mvl = _pac_mvl_timecourse(X, fs=128.0,theta_band=(cfg.sr_centers[0]-0.35,cfg.sr_centers[0]+0.35), gamma_band=(40,60),
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
        # p0_band=(-10,+0.5), 
        # p1_band=(-10,+1.5), 
        p0_band=(-2,+0.6),                      # seconds relative to seed_t
        p1_band=(-1, +1.4),                      # seconds relative to seed_t
        pad_s=2.0,
        title=f"Ignition {ign_win[0]}–{ign_win[1]}s",
        centers=ladder[:3]
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
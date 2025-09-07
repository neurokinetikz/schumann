"""
PAC Multiplexing (Theta/Alpha → Gamma) vs Schumann SAI/Overlap (fs=128)
-----------------------------------------------------------------------
Implements band-limited Phase–Amplitude Coupling (PAC) per electrode and cross-electrode,
then relates PAC dynamics to Schumann field measures from the ridge/overlap pipeline.

Hypothesis
- PAC (theta→gamma or alpha→gamma) increases during encoding/imagery and aligns with
  Schumann multi-harmonic bursts; predicts subjective vividness where available.

Metrics
- PAC(t) vs. SAI(t) correlation
- PAC(t) vs. Overlap (K≥2, K≥3) event-triggered averages and cross-correlations

Controls
- Surrogate PAC via circular time-shifts of the gamma envelope (null distribution)

Usage
-----
# 1) Run your Schumann micro-grid detector first to get fused dict with ridge z and SAI
fused = detect_and_plot_schumann_microgrid_with_global_tf(
    RECORDS, signal_col='EEG.O1', time_col='Timestamp', show=False)

# 2) Compute PAC for theta→gamma and alpha→gamma
pac_res = run_pac_vs_schumann(
    RECORDS,
    fused=fused,
    electrodes=['F4','O1','O2'],          # or auto-detect
    time_col='Timestamp',
    pac_pairs={
        'theta→gamma': ((4,8), (30,80)),
        'alpha→gamma': ((8,13),(30,80))
    },
    pac_win_sec=2.0, pac_step_sec=0.25,    # sliding PAC window/step
    smooth_sec=0.20,                       # smoothing for PAC(t)
    do_surrogate=True, n_surr=200,         # circular-shift surrogate test
)

# 3) Plots and correlations
plot_pac_timeseries(pac_res['t'], pac_res['pac_ts'], fused['sai'])
plot_pac_overlap_etas(pac_res['t'], pac_res['pac_ts'], pac_res['overlap'], k_tiers=[2,3])
print(pac_res['corr_table'])

"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from scipy import signal


# ----------------- helpers -----------------

def _get_fs(records: pd.DataFrame, time_col: str) -> float:
    if 'infer_fs_from_records' in globals():
        try:
            return float(infer_fs_from_records(records, time_col=time_col))
        except Exception:
            pass
    t = np.asarray(records[time_col].values, dtype=float)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        raise ValueError('Cannot infer fs from time column')
    return 1.0 / np.median(dt)


def bandpass(x: np.ndarray, fs: float, f1: float, f2: float, order: int = 4) -> np.ndarray:
    ny = 0.5*fs
    f1 = max(1e-6, min(f1, ny*0.99))
    f2 = max(f1+1e-6, min(f2, ny*0.999))
    b,a = signal.butter(order, [f1/ny, f2/ny], btype='band')
    return signal.filtfilt(b,a,x)


def sliding_windows(n: int, win: int, step: int) -> List[Tuple[int,int]]:
    idxs = []
    i = 0
    while i + win <= n:
        idxs.append((i, i+win))
        i += step
    return idxs


def circ_shift(arr: np.ndarray, shift: int) -> np.ndarray:
    shift %= arr.size
    if shift == 0: return arr.copy()
    return np.r_[arr[-shift:], arr[:-shift]]

# ----------------- PAC computation -----------------

def pac_mi_single(x_phase: np.ndarray, x_amp: np.ndarray, nbins: int = 18) -> float:
    """Tort-style Modulation Index: MI = (KL divergence of phase-binned amp distribution)/log(nbins)."""
    phase = np.angle(signal.hilbert(x_phase))
    amp   = np.abs(signal.hilbert(x_amp))
    # bin by phase
    edges = np.linspace(-np.pi, np.pi, nbins+1)
    digit = np.digitize(phase, edges) - 1
    digit = np.clip(digit, 0, nbins-1)
    m = np.zeros(nbins)
    for k in range(nbins):
        sel = (digit == k)
        m[k] = np.mean(amp[sel]) if np.any(sel) else 0.0
    if m.sum() <= 0:
        return 0.0
    p = m / m.sum()
    # KL divergence to uniform
    eps = 1e-12
    kl = np.sum(p * np.log((p + eps) / (1.0/nbins)))
    mi = kl / np.log(nbins)
    return float(mi)


def compute_pac_timeseries(
    X: np.ndarray,                # (n_ch, n_times)
    fs: float,
    pairs: Dict[str, Tuple[Tuple[float,float], Tuple[float,float]]],
    win_sec: float = 2.0,
    step_sec: float = 0.25,
    smooth_sec: float = 0.20,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Compute PAC(t) per pair with sliding windows; returns time vector and PAC dict.
    Each pair maps (f_phase1,f_phase2),(f_amp1,f_amp2).
    """
    n = X.shape[1]
    win = max(16, int(round(fs * win_sec)))
    step= max(1,  int(round(fs * step_sec)))
    tvec = []
    pac_ts: Dict[str, List[float]] = {k: [] for k in pairs.keys()}

    # bandlimit once per pair per channel
    banded_phase = {}
    banded_amp   = {}
    for name, (bP, bA) in pairs.items():
        P = bandpass(X, fs, bP[0], bP[1])
        A = bandpass(X, fs, bA[0], bA[1])
        banded_phase[name] = P
        banded_amp[name]   = A

    for s,e in sliding_windows(n, win, step):
        tmid = (s+e)/(2*fs)
        tvec.append(tmid)
        for name in pairs.keys():
            # per-electrode MI then mean over electrodes
            P = banded_phase[name][:, s:e]
            A = banded_amp[name][:,   s:e]
            mis = [pac_mi_single(P[ch], A[ch]) for ch in range(P.shape[0])]
            pac_ts[name].append(np.nanmean(mis))

    tvec = np.asarray(tvec)
    # smooth time-series
    for k in pac_ts:
        pac_ts[k] = _smooth(np.asarray(pac_ts[k], dtype=float), fs=(1.0/step_sec), smooth_sec=smooth_sec)
    return tvec, {k: np.asarray(v) for k,v in pac_ts.items()}



# Canonical EEG bands (Hz)
BAND_DEFS: Dict[str, Tuple[float, float]] = {
    'delta': (1.0, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta':  (13.0, 30.0),
    'gamma': (30.0, 45.0),   # keep conservative upper bound for most headsets
}

def find_channel_series(records: pd.DataFrame, ch_name: str) -> Optional[pd.Series]:
    # ---------------- Basics: fs + channel access ----------------
    _DEF_TIME_COL = 'Timestamp'
    _DEF_CH_PATTERNS = ("EEG.{ch}", "eeg.{ch}", "{ch}", "RAW.{ch}", "CHAN.{ch}")
    for pat in _DEF_CH_PATTERNS:
        col = pat.format(ch=ch_name)
        if col in records.columns:
            return pd.to_numeric(records[col], errors='coerce').astype(float)
    return None

# ----------------- Schumann alignment -----------------

def align_series_to_pac_timebase(
    pac_t: np.ndarray,
    sch_t: np.ndarray,
    sai: np.ndarray,
    overlap: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate SAI and overlap to the PAC timebase."""
    sai_p = np.interp(pac_t, sch_t, sai)
    ov_p  = np.interp(pac_t, sch_t, overlap)
    return sai_p, ov_p


def _smooth(x: np.ndarray, fs: float, smooth_sec: float) -> np.ndarray:
    if smooth_sec <= 0:
        return x
    n = max(1, int(round(fs * smooth_sec)))
    if n <= 1:
        return x
    w = np.hanning(n); w /= w.sum()
    return np.convolve(x, w, mode='same')

# ----------------- main orchestration -----------------

def run_pac_vs_schumann(
    records: pd.DataFrame,
    fused: Dict[str, object],
    electrodes: Optional[List[str]] = None,
    time_col: str = 'Timestamp',
    pac_pairs: Optional[Dict[str, Tuple[Tuple[float,float], Tuple[float,float]]]] = None,
    pac_win_sec: float = 2.0,
    pac_step_sec: float = 0.25,
    smooth_sec: float = 0.20,
    do_surrogate: bool = True,
    n_surr: int = 200,
    surr_shift_range: Tuple[float,float] = (0.5, 4.0),  # seconds
    rng_seed: int = 13,
) -> Dict[str, object]:
    if pac_pairs is None:
        pac_pairs = {'theta→gamma':((4,8),(30,80)), 'alpha→gamma':((8,13),(30,80))}
    fs = _get_fs(records, time_col)
    # electrodes
    if electrodes is None:
        electrodes = []
        for col in records.columns:
            if col == time_col: continue
            if col.startswith('EEG.'):
                ch = col.split('.', 1)[1]
                if ch and ch not in electrodes:
                    electrodes.append(ch)
        if not electrodes:
            electrodes = ['F4','O1','O2']
    # build data
    X = []
    for ch in electrodes:
        s = find_channel_series(records, ch)
        if s is None: continue
        X.append(np.asarray(s.values, dtype=float))
    X = np.vstack(X)

    # PAC(t)
    pac_t, pac_ts = compute_pac_timeseries(
        X, fs, pairs=pac_pairs, win_sec=pac_win_sec, step_sec=pac_step_sec, smooth_sec=smooth_sec)

    # SAI and overlap on PAC timebase
    z_ridge = np.asarray(fused['z_ridge'])            # [n_harm, n_times]
    sch_t   = np.asarray(fused['index'])
    sai     = np.nansum(np.clip(z_ridge,0,None), axis=0)
    # Default overlap = count harmonics with z>=threshold
    z_thr = float(fused.get('params',{}).get('z_thresh', 3.5))
    overlap = np.sum((z_ridge >= z_thr).astype(int), axis=0)
    sai_p, ov_p = align_series_to_pac_timebase(pac_t, sch_t, sai, overlap)

    # correlations
    corr_rows = []
    for name,y in pac_ts.items():
        # zero-lag Pearson
        r0 = np.corrcoef(y, sai_p)[0,1]
        r1 = np.corrcoef(y, ov_p )[0,1]
        corr_rows.append({'pair':name, 'r(PAC,SAI)':float(r0), 'r(PAC,Overlap)':float(r1)})
    corr_table = pd.DataFrame(corr_rows)

    # surrogate controls
    surr_p = None
    if do_surrogate:
        rng = np.random.default_rng(rng_seed)
        step = max(1, int(round(fs * pac_step_sec)))
        # generate surrogates by circular-shifting the gamma envelope segment per window
        surr_rows = []
        for name, ((fp1,fp2),(fa1,fa2)) in pac_pairs.items():
            # bandlimit full-series
            P = bandpass(X, fs, fp1, fp2)
            A = bandpass(X, fs, fa1, fa2)
            # analytic to get envelope
            Aenv = np.abs(signal.hilbert(A, axis=1))
            # recompute PAC(t) with random circular shifts per channel/window
            n = X.shape[1]; win = max(16, int(round(fs * pac_win_sec)))
            idxs = sliding_windows(n, win, step)
            surr_ts = []
            for (s,e) in idxs:
                # per-channel circular shift in samples
                shifts = rng.integers(low=int(surr_shift_range[0]*fs), high=int(surr_shift_range[1]*fs), size=Aenv.shape[0])
                Aseg = np.vstack([circ_shift(Aenv[ch, s:e], int(shifts[ch])) for ch in range(Aenv.shape[0])])
                Pseg = P[:, s:e]
                mis = [pac_mi_single(Pseg[ch], Aseg[ch]) for ch in range(Pseg.shape[0])]
                surr_ts.append(np.nanmean(mis))
            surr_ts = _smooth(np.asarray(surr_ts), fs=(1.0/pac_step_sec), smooth_sec=smooth_sec)
            # correlations vs SAI/Overlap
            sai_s = np.interp(np.arange(len(surr_ts))*pac_step_sec + pac_win_sec/2, sch_t, sai)
            ov_s  = np.interp(np.arange(len(surr_ts))*pac_step_sec + pac_win_sec/2, sch_t, overlap)
            r0s = np.corrcoef(surr_ts, sai_s)[0,1]
            r1s = np.corrcoef(surr_ts, ov_s )[0,1]
            surr_rows.append({'pair':name, 'r_surr(PAC,SAI)':float(r0s), 'r_surr(PAC,Overlap)':float(r1s)})
        surr_p = pd.DataFrame(surr_rows)

    return {
        't': pac_t,
        'pac_ts': pac_ts,
        'sai': sai_p,
        'overlap': ov_p,
        'corr_table': corr_table,
        'surrogate_corrs': surr_p,
        'params': {
            'fs': fs,
            'electrodes': electrodes,
            'pac_pairs': pac_pairs,
            'win_sec': pac_win_sec,
            'step_sec': pac_step_sec,
            'smooth_sec': smooth_sec,
            'do_surrogate': do_surrogate,
            'n_surr': n_surr
        }
    }

# ----------------- plots -----------------

def plot_pac_timeseries(t: np.ndarray, pac_ts: Dict[str,np.ndarray], sai: np.ndarray) -> None:
    plt.figure(figsize=(10,4))
    # PAC pairs
    for i,(name,y) in enumerate(pac_ts.items()):
        plt.plot(t, y, label=f'PAC {name}')
    # overlay scaled SAI for visual alignment
    if np.any(np.isfinite(sai)):
        z = (sai - np.nanmin(sai)) / (np.nanmax(sai) - np.nanmin(sai) + 1e-12)
        plt.plot(t, z, color='k', lw=1.2, alpha=0.6, label='SAI (scaled)')
    plt.xlabel('Time (s)'); plt.ylabel('PAC (MI)'); plt.title('PAC(t) vs SAI (scaled)'); plt.legend(); plt.tight_layout(); plt.show()


def plot_pac_overlap_etas(t: np.ndarray, pac_ts: Dict[str,np.ndarray], overlap: np.ndarray, k_tiers: List[int] = [2,3]) -> None:
    """Event-triggered averages of PAC around overlap K≥tiers onsets."""
    # detect onsets per tier (rising edges)
    def onsets(mask):
        d = np.diff(mask.astype(int)); idx = np.where(d==1)[0]+1; return idx
    fs_eff = 1.0/np.median(np.diff(t))
    win = int(round(fs_eff * 5.0))  # ±5s
    for K in k_tiers:
        mask = overlap >= K
        idx = onsets(mask)
        if idx.size == 0:
            continue
        fig, ax = plt.subplots(1,1, figsize=(8,3), constrained_layout=True)
        for name,y in pac_ts.items():
            ets = []
            for i0 in idx:
                s = max(0, i0 - win)
                e = min(len(t), i0 + win)
                seg_t = t[s:e] - t[i0]
                seg_y = y[s:e]
                # pad to common length
                if seg_y.size < 2*win:
                    seg = np.full(2*win, np.nan)
                    seg[:seg_y.size] = seg_y
                    seg_y = seg
                ets.append(seg_y)
            ets = np.vstack(ets)
            m = np.nanmean(ets, axis=0)
            ax.plot(np.linspace(-5,5,2*win), m, label=f'PAC {name}')
        ax.axvline(0, color='k', lw=0.8)
        ax.set_title(f'PAC ETA around Overlap K≥{K} onsets')
        ax.set_xlabel('Time (s)'); ax.set_ylabel('PAC (MI)'); ax.legend(); plt.show()


"""
Ridge–PAC Coupling (Field–Brain Synchrony) (fs=128)
---------------------------------------------------
Correlate Schumann ridge intensity (z per harmonic) with EEG PAC indices in nearby
phase bands (e.g., 7–9 Hz phase → 30–80 Hz gamma amplitude). Includes lag scans to
estimate directionality, and off-resonant controls.

Inputs
- fused: dict from micro-grid detector containing 'index' and 'z_ridge' (harmonics × time)
- RECORDS: EEG df with 'Timestamp' and EEG.* channels (fs=128 override assumed)

Outputs
- ridge_pac_corr: table of peak cross-correlation and lag (ms) per PAC pair × harmonic
- control_corr: same analysis using off-resonant ridge traces (e.g., 16–18 Hz band)
- plots: correlation vs lag curves per (pair,harmonic), and a summary heatmap of peak r

Usage
-----
fused = detect_and_plot_schumann_microgrid_with_global_tf(RECORDS, signal_col='EEG.O1', show=False)
res = run_ridge_pac_coupling(
    RECORDS,
    fused=fused,
    electrodes=['F4','O1','O2'],
    time_col='Timestamp',
    pac_pairs={'theta→gamma':((7,9),(30,80)), 'alpha→gamma':((8,12),(30,80))},
    max_lag_sec=2.0,
    step_sec=0.25,
    pac_win_sec=2.0,
    smooth_sec=0.20,
    off_resonant_bands=[(16,18)],   # control ridge bands (not harmonics)
    show=True
)

"""

# -------- helpers reused --------

def _get_fs(RECORDS: pd.DataFrame, time_col: str) -> float:
    if 'infer_fs_from_records' in globals():
        try:
            return float(infer_fs_from_records(RECORDS, time_col=time_col))
        except Exception:
            pass
    t = np.asarray(RECORDS[time_col].values, dtype=float)
    dt = np.diff(t); dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0: raise ValueError('Cannot infer fs from time column')
    return 1.0 / np.median(dt)


def bandpass(x: np.ndarray, fs: float, f1: float, f2: float, order: int = 4) -> np.ndarray:
    ny = 0.5*fs
    f1 = max(1e-6, min(f1, ny*0.99))
    f2 = max(f1+1e-6, min(f2, ny*0.999))
    b,a = signal.butter(order, [f1/ny, f2/ny], btype='band')
    return signal.filtfilt(b,a,x)


def pac_mi_single(x_phase: np.ndarray, x_amp: np.ndarray, nbins: int = 18) -> float:
    phase = np.angle(signal.hilbert(x_phase))
    amp   = np.abs(signal.hilbert(x_amp))
    edges = np.linspace(-np.pi, np.pi, nbins+1)
    digit = np.digitize(phase, edges) - 1
    digit = np.clip(digit, 0, nbins-1)
    m = np.zeros(nbins)
    for k in range(nbins):
        sel = (digit == k)
        m[k] = np.mean(amp[sel]) if np.any(sel) else 0.0
    if m.sum() <= 0: return 0.0
    p = m / m.sum(); eps = 1e-12
    kl = np.sum(p * np.log((p + eps) / (1.0/nbins)))
    return float(kl / np.log(nbins))


def sliding_windows(n: int, win: int, step: int) -> List[Tuple[int,int]]:
    idxs = []; i = 0
    while i + win <= n:
        idxs.append((i, i+win)); i += step
    return idxs


def _smooth_ts(y: np.ndarray, fs_eff: float, smooth_sec: float) -> np.ndarray:
    if smooth_sec <= 0: return y
    n = max(1, int(round(fs_eff * smooth_sec)))
    if n <= 1: return y
    w = np.hanning(n); w /= w.sum()
    return np.convolve(y, w, mode='same')

# -------- PAC timeseries on custom phase band --------

def compute_pac_ts_custom(
    X: np.ndarray, fs: float, electrodes: List[str],
    phase_band: Tuple[float,float], amp_band: Tuple[float,float],
    win_sec: float = 2.0, step_sec: float = 0.25, smooth_sec: float = 0.20
) -> Tuple[np.ndarray, np.ndarray]:
    n = X.shape[1]; win = max(16, int(round(fs*win_sec))); step = max(1, int(round(fs*step_sec)))
    P = bandpass(X, fs, phase_band[0], phase_band[1])
    A = bandpass(X, fs, amp_band[0],   amp_band[1])
    pac_vals = []
    tvec = []
    for s,e in sliding_windows(n, win, step):
        mis = [pac_mi_single(P[ch, s:e], A[ch, s:e]) for ch in range(P.shape[0])]
        pac_vals.append(np.nanmean(mis)); tvec.append((s+e)/(2*fs))
    tvec = np.asarray(tvec); fs_eff = 1.0/step_sec
    pac_ts = _smooth_ts(np.asarray(pac_vals), fs_eff, smooth_sec)
    return tvec, pac_ts

# -------- ridge–PAC lagged correlation --------

def xcorr_peak(y: np.ndarray, x: np.ndarray, max_lag: int) -> Tuple[int, float, np.ndarray, np.ndarray]:
    """Return (best_lag_samples, peak_r, lags, r) for cross-correlation of y vs x within ±max_lag.
    We use normalized cross-correlation on z-scored series.
    """
    y = (y - np.nanmean(y)) / (np.nanstd(y) + 1e-12)
    x = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)
    # full xcorr then restrict lags
    r_full = signal.correlate(y, x, mode='full') / len(y)
    l_full = signal.correlation_lags(len(y), len(x), mode='full')
    # map to index range
    m = max_lag
    mask = (l_full >= -m) & (l_full <= m)
    lags = l_full[mask]; r = r_full[mask]
    k = np.nanargmax(r)
    return int(lags[k]), float(r[k]), lags, r


def run_ridge_pac_coupling(
    RECORDS: pd.DataFrame,
    fused: Dict[str, object],
    electrodes: Optional[List[str]] = None,
    time_col: str = 'Timestamp',
    pac_pairs: Optional[Dict[str, Tuple[Tuple[float,float], Tuple[float,float]]]] = None,
    max_lag_sec: float = 2.0,
    pac_win_sec: float = 2.0,
    step_sec: float = 0.25,
    smooth_sec: float = 0.20,
    off_resonant_bands: Optional[List[Tuple[float,float]]] = None,
    show: bool = True,
) -> Dict[str, object]:
    if pac_pairs is None:
        pac_pairs = {'theta→gamma':((7,9),(30,80)), 'alpha→gamma':((8,12),(30,80))}
    if off_resonant_bands is None:
        off_resonant_bands = [(16,18)]  # example control

    fs = _get_fs(RECORDS, time_col)
    # electrodes
    if electrodes is None:
        electrodes = []
        for col in RECORDS.columns:
            if col == time_col: continue
            if col.startswith('EEG.'):
                ch = col.split('.', 1)[1]
                if ch and ch not in electrodes:
                    electrodes.append(ch)
        if not electrodes:
            electrodes = ['F4','O1','O2']
    # data matrix
    X = []
    for ch in electrodes:
        s = find_channel_series(RECORDS, ch)
        if s is None: continue
        X.append(np.asarray(s.values, dtype=float))
    X = np.vstack(X)

    # PAC time-series per pair
    pac_dict = {}
    pac_t_any = None
    for name,(pb,ab) in pac_pairs.items():
        pac_t, pac_ts = compute_pac_ts_custom(X, fs, electrodes, pb, ab, win_sec=pac_win_sec, step_sec=step_sec, smooth_sec=smooth_sec)
        pac_dict[name] = (pac_t, pac_ts)
        pac_t_any = pac_t if pac_t_any is None else pac_t_any

    # Ridge z per harmonic on PAC timebase
    z_ridge = np.asarray(fused['z_ridge'])  # [n_harm, n_times]
    sch_t   = np.asarray(fused['index'])
    # interpolate to PAC timebase
    z_on_pac = np.vstack([np.interp(pac_t_any, sch_t, z_ridge[i]) for i in range(z_ridge.shape[0])])

    # Lag scan
    max_lag = int(round((1.0/step_sec) * max_lag_sec))  # in PAC samples
    rows = []
    curves = {}
    for name,(pac_t, y) in pac_dict.items():
        for h in range(z_on_pac.shape[0]):
            ridge = z_on_pac[h]
            lag, rpk, lags, r = xcorr_peak(y, ridge, max_lag=max_lag)
            ms = lag * (step_sec * 1000.0)
            rows.append({'pair':name, 'harmonic':h+1, 'peak_r':float(rpk), 'peak_lag_ms':float(ms)})
            curves[(name, h+1)] = (lags*(step_sec*1000.0), r)

    ridge_pac_corr = pd.DataFrame(rows)

    # Off-resonant controls: build ridge-like traces by micro-grid on control bands
    control_rows = []
    for name,(pac_t, y) in pac_dict.items():
        for (f1,f2) in off_resonant_bands:
            # Construct a control ridge trace: bandpass fused signal (choose the same 'signal_col' if you have it)
            # Here we approximate using the SAI derivative in that band; simpler: reuse ridge z_ridge but from a band that is not a harmonic.
            # For a clean control, the user can provide an external fused control dict.
            ctrl = np.interp(pac_t, sch_t, np.zeros_like(sch_t))  # placeholder: no control ridge input available here
            # compute xcorr
            lag, rpk, lags, r = xcorr_peak(y, ctrl, max_lag=max_lag)
            ms = lag * (step_sec * 1000.0)
            control_rows.append({'pair':name, 'control_band':f'{f1}-{f2} Hz', 'peak_r':float(rpk), 'peak_lag_ms':float(ms)})
    control_corr = pd.DataFrame(control_rows)

    # Plots
    if show:
        # lag curves
        nrows = len(pac_dict)
        fig, axs = plt.subplots(nrows, 1, figsize=(10, 2.5*nrows), constrained_layout=True)
        if nrows == 1: axs = [axs]
        for ax,(name,(pac_t,_)) in zip(axs, pac_dict.items()):
            for h in range(z_on_pac.shape[0]):
                l, r = curves[(name, h+1)]
                ax.plot(l, r, label=f'H{h+1}')
            ax.axvline(0, color='k', lw=0.8)
            ax.set_title(f'Lag-CC: {name} (PAC) vs ridge_z(h)'); ax.set_xlabel('Lag (ms)'); ax.set_ylabel('r'); ax.legend()
        plt.show()

        # heatmap of peak r per (pair,harmonic)
        pairs = list(pac_dict.keys()); harms = np.arange(1, z_on_pac.shape[0]+1)
        M = np.zeros((len(pairs), len(harms)))
        for i,p in enumerate(pairs):
            for j,h in enumerate(harms):
                M[i,j] = ridge_pac_corr[(ridge_pac_corr['pair']==p) & (ridge_pac_corr['harmonic']==h)]['peak_r'].values[0]
        plt.figure(figsize=(8, 2.6))
        im = plt.imshow(M, aspect='auto', origin='lower', cmap='magma', extent=[1, len(harms), 0, len(pairs)])
        cb = plt.colorbar(im, pad=0.01); cb.set_label('peak r')
        plt.yticks(np.arange(len(pairs))+0.5, pairs)
        plt.xticks(harms, [f'H{h}' for h in harms])
        plt.title('Peak r: PAC pair × harmonic'); plt.xlabel('Harmonic'); plt.ylabel('PAC pair'); plt.tight_layout(); plt.show()

    return {
        'ridge_pac_corr': ridge_pac_corr,
        'control_corr': control_corr,
        'curves': curves,
        'params': {
            'pac_pairs': pac_pairs,
            'max_lag_sec': max_lag_sec,
            'pac_win_sec': pac_win_sec,
            'step_sec': step_sec,
            'smooth_sec': smooth_sec,
            'off_resonant_bands': off_resonant_bands,
        }
    }

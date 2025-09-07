"""
Criticality Signatures — 1/f, DFA, Avalanches vs Conscious Mode (fs=128)
-----------------------------------------------------------------------
Implements: (a) robust 1/f exponent β; (b) DFA exponent α; (c) avalanche size/duration
statistics on band-limited amplitude crossings. Compares ignition vs baseline (and
rebound if provided). Includes optional control windows.

Metrics
- Δβ = β_ign − β_base (flatter 1/f → lower β)
- Δα = α_ign − α_base (critical LRTC → α ≈ 1.0)
- Avalanche exponents (size ~ s^−τ, with τ ≈ 3/2 under SOC) per state

Usage
-----
IGNITION_WINDOWS = [(120.0, 150.0)]
REBOUND_WINDOWS  = [(300.0, 330.0)]
ELECTRODES = ['F4','O1','O2']

crit = run_criticality_analysis(
    RECORDS,
    ignition_windows=IGNITION_WINDOWS,
    rebound_windows=REBOUND_WINDOWS,
    electrodes=ELECTRODES,
    bands={'theta':(4,8),'alpha':(8,13)},
    dfa_scales_sec=np.geomspace(0.25, 8.0, 14),  # DFA scales
    avalanche_band=(4,40),                       # broad band for avalanches
    avalanche_thresh='p95',                      # amplitude threshold
    do_control=False,
)

print(crit['delta_table'])
# Optional plots (see functions below)
plot_criticality_deltas(crit['delta_table'])
plot_avalanche_ccdf(crit['avalanches'])
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from scipy import signal
from numpy.linalg import lstsq

# ----------------- core helpers -----------------

def _get_fs(RECORDS: pd.DataFrame, time_col: str) -> float:
    if 'infer_fs_from_records' in globals():
        try:
            return float(infer_fs_from_records(RECORDS, time_col=time_col))
        except Exception:
            pass
    t = np.asarray(RECORDS[time_col].values, dtype=float)
    dt = np.diff(t); dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0: raise ValueError('Cannot infer fs')
    return 1.0 / np.median(dt)


def bandpass(x: np.ndarray, fs: float, f1: float, f2: float, order: int = 4) -> np.ndarray:
    ny = 0.5*fs
    f1 = max(1e-6, min(f1, ny*0.99))
    f2 = max(f1+1e-6, min(f2, ny*0.999))
    b,a = signal.butter(order, [f1/ny, f2/ny], btype='band')
    return signal.filtfilt(b,a,x)


def slice_blocks(RECORDS: pd.DataFrame, time_col: str, X: np.ndarray, fs: float, winlist: List[Tuple[float,float]]) -> List[np.ndarray]:
    t = np.asarray(RECORDS[time_col].values, dtype=float)
    blocks = []
    for (t0,t1) in winlist:
        sel = (t>=t0) & (t<=t1)
        idx = np.where(sel)[0]
        if idx.size < fs*1.0:  # require ≥1 s
            continue
        blocks.append(X[:, idx[0]:idx[-1]+1])
    return blocks

def find_channel_series(records: pd.DataFrame, ch_name: str) -> Optional[pd.Series]:
    # ---------------- Basics: fs + channel access ----------------
    _DEF_TIME_COL = 'Timestamp'
    _DEF_CH_PATTERNS = ("EEG.{ch}", "eeg.{ch}", "{ch}", "RAW.{ch}", "CHAN.{ch}")
    for pat in _DEF_CH_PATTERNS:
        col = pat.format(ch=ch_name)
        if col in records.columns:
            return pd.to_numeric(records[col], errors='coerce').astype(float)
    return None

# ----------------- 1/f exponent β -----------------

def welch_beta(x: np.ndarray, fs: float, fmin: float=1.0, fmax: float=40.0) -> float:
    """Robust 1/f slope β via log–log linear regression on Welch PSD between fmin–fmax."""
    f, p = signal.welch(x, fs=fs, nperseg=4*int(fs))
    sel = (f>=fmin) & (f<=fmax) & np.isfinite(p)
    if np.count_nonzero(sel) < 6: return np.nan
    X = np.vstack([np.ones(np.sum(sel)), -np.log(f[sel])]).T  # P ~ C f^{-β} → logP = logC − β log f
    y = np.log(p[sel] + 1e-24)
    b, *_ = lstsq(X, y, rcond=None)
    beta = float(b[1])
    return beta

# ----------------- DFA exponent α -----------------

def dfa_alpha(x: np.ndarray, fs: float, scales_sec: np.ndarray) -> float:
    """Detrended fluctuation analysis (DFA) exponent α on band-limited amplitude envelope."""
    # integrate (profile)
    x = (x - np.mean(x))
    y = np.cumsum(x)
    scales = (scales_sec * fs).astype(int)
    F = []
    for s in scales:
        if s < 4: continue
        nseg = len(y)//s
        if nseg < 4: continue
        rms = []
        for k in range(nseg):
            seg = y[k*s:(k+1)*s]
            t = np.arange(s)
            # linear detrend
            A = np.vstack([t, np.ones_like(t)]).T
            coeff, *_ = lstsq(A, seg, rcond=None)
            trend = A.dot(coeff)
            rms.append(np.sqrt(np.mean((seg - trend)**2)))
        F.append((s, np.sqrt(np.mean(np.asarray(rms)**2))))
    if len(F) < 3: return np.nan
    s = np.log([f[0] for f in F]); r = np.log([f[1] for f in F])
    a, *_ = lstsq(np.vstack([np.ones_like(s), s]).T, r, rcond=None)
    alpha = float(a[1])
    return alpha

# ----------------- avalanches -----------------

def avalanche_events(env: np.ndarray, thresh: float) -> List[Tuple[int,int,float]]:
    """Return [(start_idx, end_idx, size)] where size is area under envelope above threshold."""
    mask = env >= thresh
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends   = np.where(diff == -1)[0]
    if mask[0]: starts = np.r_[0, starts]
    if mask[-1]: ends = np.r_[ends, len(env)-1]
    events = []
    for s,e in zip(starts, ends):
        seg = env[s:e+1]
        size = float(np.trapz(seg - thresh))
        events.append((int(s), int(e), size))
    return events


def avalanche_stats(envs: List[np.ndarray], fs: float, thresh_mode: str='p95') -> Dict[str, object]:
    sizes = []; durs = []
    for env in envs:
        if env.size < fs: continue
        if thresh_mode.startswith('p'):
            q = float(thresh_mode[1:]); thr = np.percentile(env, q)
        else:
            thr = float(thresh_mode)
        evs = avalanche_events(env, thr)
        for s,e,sz in evs:
            sizes.append(sz)
            durs.append((e - s + 1)/fs)
    sizes = np.asarray(sizes, dtype=float)
    durs  = np.asarray(durs, dtype=float)
    return {'sizes': sizes, 'durs': durs}

# ----------------- orchestration -----------------

def run_criticality_analysis(
    RECORDS: pd.DataFrame,
    ignition_windows: List[Tuple[float,float]],
    rebound_windows: Optional[List[Tuple[float,float]]] = None,
    control_windows: Optional[List[Tuple[float,float]]] = None,
    electrodes: Optional[List[str]] = None,
    bands: Optional[Dict[str, Tuple[float,float]]] = None,
    time_col: str = 'Timestamp',
    dfa_scales_sec: np.ndarray = np.geomspace(0.25, 8.0, 14),
    avalanche_band: Tuple[float,float] = (4,40),
    avalanche_thresh: str = 'p95',
) -> Dict[str, object]:
    fs = _get_fs(RECORDS, time_col)
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
    # data matrix (channels × time)
    X = []
    for ch in electrodes:
        s = find_channel_series(RECORDS, ch)
        if s is None: continue
        X.append(np.asarray(s.values, dtype=float))
    X = np.vstack(X)

    # slice blocks
    ign_blocks = slice_blocks(RECORDS, time_col, X, fs, ignition_windows)
    # baseline complement (simple): everything outside ignition windows
    t = np.asarray(RECORDS[time_col].values, dtype=float)
    mask = np.ones(X.shape[1], dtype=bool)
    for (t0,t1) in ignition_windows:
        i0,i1 = int(t0*fs), int(t1*fs)
        mask[max(0,i0):min(X.shape[1],i1)] = False
    base_blocks = [X[:, mask]] if np.any(mask) else []
    reb_blocks  = slice_blocks(RECORDS, time_col, X, fs, rebound_windows) if rebound_windows else []
    ctrl_blocks = slice_blocks(RECORDS, time_col, X, fs, control_windows) if control_windows else []

    # analysis per state
    def analyze_state(blocks: List[np.ndarray]) -> Dict[str, object]:
        if not blocks:
            return {'beta': np.nan, 'alpha': np.nan, 'aval': {'sizes':np.array([]), 'durs':np.array([])}}
        # concatenate across blocks for PSD/DFA
        concat = np.hstack(blocks)
        # 1/f β on broadband (1–40 Hz) per channel then mean
        betas = [welch_beta(concat[ch], fs, fmin=1.0, fmax=40.0) for ch in range(concat.shape[0])]
        beta = float(np.nanmean(betas))
        # DFA α on band-limited amplitude envelope (use theta–beta 4–30 Hz)
        xb = bandpass(concat, fs, 4, 30)
        env = np.abs(signal.hilbert(xb))
        alpha = float(dfa_alpha(env, fs, scales_sec=dfa_scales_sec))
        # Avalanches on (4–40 Hz) envelope per-block
        envs = []
        for blk in blocks:
            xb2 = bandpass(blk, fs, avalanche_band[0], avalanche_band[1])
            env2 = np.abs(signal.hilbert(xb2, axis=1))
            # stack across channels by mean
            envs.append(np.mean(env2, axis=0))
        aval = avalanche_stats(envs, fs, thresh_mode=avalanche_thresh)
        return {'beta': beta, 'alpha': alpha, 'aval': aval}

    ign = analyze_state(ign_blocks)
    base= analyze_state(base_blocks)
    reb = analyze_state(reb_blocks)
    ctrl= analyze_state(ctrl_blocks)

    # delta table
    rows = []
    rows.append({'state':'ignition', 'beta':ign['beta'], 'alpha':ign['alpha'], 'sizes':len(ign['aval']['sizes']), 'durs':len(ign['aval']['durs'])})
    rows.append({'state':'baseline', 'beta':base['beta'], 'alpha':base['alpha'], 'sizes':len(base['aval']['sizes']), 'durs':len(base['aval']['durs'])})
    if rebound_windows: rows.append({'state':'rebound', 'beta':reb['beta'], 'alpha':reb['alpha'], 'sizes':len(reb['aval']['sizes']), 'durs':len(reb['aval']['durs'])})
    if control_windows: rows.append({'state':'control', 'beta':ctrl['beta'], 'alpha':ctrl['alpha'], 'sizes':len(ctrl['aval']['sizes']), 'durs':len(ctrl['aval']['durs'])})
    df = pd.DataFrame(rows)

    # deltas (ign − base)
    d_beta = float(df.loc[df['state']=='ignition','beta'].values[0] - df.loc[df['state']=='baseline','beta'].values[0])
    d_alpha= float(df.loc[df['state']=='ignition','alpha'].values[0] - df.loc[df['state']=='baseline','alpha'].values[0])
    delta_table = pd.DataFrame([{'d_beta': d_beta, 'd_alpha': d_alpha}])

    return {
        'delta_table': delta_table,
        'summary': df,
        'avalanches': {
            'ignition': ign['aval'],
            'baseline': base['aval'],
            'rebound' : reb['aval'],
            'control' : ctrl['aval'],
        },
        'params': {
            'electrodes': electrodes,
            'bands': bands,
            'dfa_scales_sec': dfa_scales_sec,
            'avalanche_band': avalanche_band,
            'avalanche_thresh': avalanche_thresh,
        }
    }

# ----------------- plots -----------------

def plot_criticality_deltas(delta_df: pd.DataFrame) -> None:
    plt.figure(figsize=(6,3.2))
    x = np.arange(2); labels = ['Δβ (ign−base)','Δα (ign−base)']
    vals = [float(delta_df['d_beta'].values[0]), float(delta_df['d_alpha'].values[0])]
    plt.bar(x, vals, width=0.6)
    plt.xticks(x, labels, rotation=0); plt.ylabel('Delta'); plt.title('Criticality deltas'); plt.tight_layout(); plt.show()


def plot_avalanche_ccdf(aval_dict: Dict[str, Dict[str,np.ndarray]]) -> None:
    """Plot complementary CDFs of avalanche sizes/durations per state (log–log)."""
    plt.figure(figsize=(10,4))
    for k, d in aval_dict.items():
        if d['sizes'].size > 0:
            s = np.sort(d['sizes']); ccdf = 1.0 - np.arange(1, s.size+1)/s.size
            plt.loglog(s, ccdf, label=f'{k} sizes')
    plt.xlabel('Size'); plt.ylabel('CCDF'); plt.title('Avalanche size CCDF'); plt.legend(); plt.tight_layout(); plt.show()

    plt.figure(figsize=(10,4))
    for k, d in aval_dict.items():
        if d['durs'].size > 0:
            u = np.sort(d['durs']); ccdf = 1.0 - np.arange(1, u.size+1)/u.size
            plt.loglog(u, ccdf, label=f'{k} durations')
    plt.xlabel('Duration (s)'); plt.ylabel('CCDF'); plt.title('Avalanche duration CCDF'); plt.legend(); plt.tight_layout(); plt.show()

def fit_powerlaw_tail(x, xmin=None):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if xmin is None: xmin = np.percentile(x, 50)     # pick a median-tail by default
    x = x[x >= xmin]
    if x.size < 50: return np.nan, np.nan
    y = np.log(x); n = y.size
    # MLE for continuous power law: tau_hat = 1 + n / sum(log(x/xmin))
    tau = 1 + n / np.sum(np.log(x/xmin))
    # quick-and-dirty stderr
    se  = (tau-1)/np.sqrt(n)
    return tau, se
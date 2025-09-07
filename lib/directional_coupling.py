"""
Directional Coupling — dPLI/Granger Right‑DLPFC → Sensory (fs=128)
------------------------------------------------------------------
Hypothesis: Conscious access entails top‑down directed connectivity from
right DLPFC to sensory (occipital/temporal/parietal) during task/ignition windows.

This module computes **directional PLIs (dPLI)** and (if available) **pairwise Granger
causality** between a **Right‑DLPFC cluster** and three sensory clusters
(Occipital, Temporal, Parietal), comparing **ignition vs baseline** (and optional rebound).

Key outputs
- Δdirectionality per band and target: dPLI_ign − dPLI_base (Right‑DLPFC → target)
- Optional Granger (x→y minus y→x) deltas if `statsmodels` is available
- Simple bar plots per band for quick inspection
- Controls: reverse direction and shuffled control windows

Usage
-----
res = run_directional_coupling_rdlfpc_sensory(
    RECORDS,
    ignition_windows=[(120,150)],
    rebound_windows=[(300,330)],
    time_col='Timestamp',
    bands={'theta':(4,8), 'alpha':(8,13), 'beta':(13,30)},
    clusters=None,                     # use defaults (see below) or pass your own
    do_granger=True,                   # will auto‑disable if statsmodels is missing
    control_windows=None,              # e.g., deep rest
    do_shuffle_control=True,           # random sample control windows of equal length
    n_shuffle=200,
)

plot_directional_deltas(res['delta_table'])
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from scipy import signal

# try optional Granger (statsmodels)
try:
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    from statsmodels.tsa.vector_ar.var_model import VAR
    _HAS_SM = True
except Exception:
    _HAS_SM = False

# ----------------- helpers -----------------

def _get_fs(RECORDS: pd.DataFrame, time_col: str) -> float:
    if 'infer_fs_from_records' in globals():
        try:
            return float(infer_fs_from_records(RECORDS, time_col=time_col))
        except Exception:
            pass
    t = np.asarray(RECORDS[time_col].values, dtype=float)
    dt = np.diff(t); dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0: raise ValueError('Cannot infer fs')
    return 1.0/np.median(dt)


def _bandpass(x: np.ndarray, fs: float, f1: float, f2: float, order: int=4) -> np.ndarray:
    ny = 0.5*fs
    f1 = max(1e-6, min(f1, ny*0.99)); f2 = max(f1+1e-6, min(f2, ny*0.999))
    b,a = signal.butter(order, [f1/ny, f2/ny], btype='band')
    return signal.filtfilt(b,a,x)


def _find_series(RECORDS: pd.DataFrame, ch: str) -> np.ndarray:
    s = find_channel_series(RECORDS, ch)
    if s is None: raise ValueError(f'Missing channel {ch}')
    return np.asarray(s.values, dtype=float)

def find_channel_series(records: pd.DataFrame, ch_name: str) -> Optional[pd.Series]:
    # ---------------- Basics: fs + channel access ----------------
    _DEF_TIME_COL = 'Timestamp'
    _DEF_CH_PATTERNS = ("EEG.{ch}", "eeg.{ch}", "{ch}", "RAW.{ch}", "CHAN.{ch}")
    for pat in _DEF_CH_PATTERNS:
        col = pat.format(ch=ch_name)
        if col in records.columns:
            return pd.to_numeric(records[col], errors='coerce').astype(float)
    return None

# ----------------- clusters -----------------

def default_clusters() -> Dict[str, List[str]]:
    """Default sensor clusters (10–20ish). Feel free to override.
    Right‑DLPFC: F4, F6, FC6, AF8 (replace with your montage labels)
    Sensory: Occipital (O1,O2,Oz), Temporal (T7,T8/TP7,TP8), Parietal (P3,P4,Pz)
    """
    return {
        'R_DLPFC': ['F4','F8','FC6','AF4'],
        'OCC':     ['O1','O2'],
        'TEMP':    ['T7','T8'],
        'PAR':     ['P7','P8'],
    }


def cluster_signal(RECORDS: pd.DataFrame, time_col: str, names: List[str]) -> np.ndarray:
    X = []
    for ch in names:
        X.append(_find_series(RECORDS, ch))
    X = np.vstack(X)
    # robust mean (trim 10% if needed) — here just use simple mean
    return np.nanmean(X, axis=0)

# ----------------- dPLI and Granger -----------------

def dpli_block(x_src: np.ndarray, x_tgt: np.ndarray, fs: float, f1: float, f2: float) -> float:
    """Directed PLI (src→tgt) in [f1,f2]. Returns fraction in [0,1], 0.5 ~ no direction.
    dPLI = mean( Heaviside( sin(φ_src − φ_tgt) ) )
    """
    xs = _bandpass(x_src, fs, f1, f2)
    xt = _bandpass(x_tgt, fs, f1, f2)
    ph_s = np.angle(signal.hilbert(xs))
    ph_t = np.angle(signal.hilbert(xt))
    sgn = np.sin(ph_s - ph_t)
    dpli = np.mean((sgn > 0).astype(float))
    return float(dpli)


def granger_block(x_src: np.ndarray, x_tgt: np.ndarray, order: int=10) -> float:
    """Pairwise GC advantage: F(src→tgt) − F(tgt→src). Requires statsmodels; else returns nan."""
    if not _HAS_SM: return float('nan')
    # z‑score
    xs = (x_src - np.nanmean(x_src)) / (np.nanstd(x_src) + 1e-12)
    xt = (x_tgt - np.nanmean(x_tgt)) / (np.nanstd(x_tgt) + 1e-12)
    Y = np.vstack([xt, xs]).T  # columns: [target, source]
    try:
        model = VAR(Y)
        res = model.fit(maxlags=order, ic='bic')
        # F‑tests
        gc_s_t = res.test_causality(caused=0, causing=[1], kind='f').statistic  # src→tgt
        gc_t_s = res.test_causality(caused=1, causing=[0], kind='f').statistic  # tgt→src
        return float(gc_s_t - gc_t_s)
    except Exception:
        return float('nan')

# ----------------- windows -----------------

def slice_windows_idx(t: np.ndarray, fs: float, windows: List[Tuple[float,float]]) -> List[Tuple[int,int]]:
    idxs = []
    for (t0,t1) in windows or []:
        i0,i1 = int(t0*fs), int(t1*fs)
        if i1 - i0 >= int(fs*0.5): idxs.append((max(0,i0), max(i0+1, i1)))
    return idxs

# ----------------- main orchestration -----------------

def run_directional_coupling_rdlfpc_sensory(
    RECORDS: pd.DataFrame,
    ignition_windows: List[Tuple[float,float]],
    rebound_windows: Optional[List[Tuple[float,float]]] = None,
    control_windows: Optional[List[Tuple[float,float]]] = None,
    time_col: str = 'Timestamp',
    bands: Optional[Dict[str, Tuple[float,float]]] = None,
    clusters: Optional[Dict[str, List[str]]] = None,
    do_granger: bool = True,
    do_shuffle_control: bool = True,
    n_shuffle: int = 200,
    shuffle_seed: int = 7,
) -> Dict[str, object]:
    fs = _get_fs(RECORDS, time_col)
    bands = bands or {'theta':(4,8),'alpha':(8,13),'beta':(13,30)}
    clusters = clusters or default_clusters()

    # build cluster signals
    t = np.asarray(RECORDS[time_col].values, dtype=float)
    src = cluster_signal(RECORDS, time_col, clusters['R_DLPFC'])
    occ = cluster_signal(RECORDS, time_col, clusters['OCC'])
    tmp = cluster_signal(RECORDS, time_col, clusters['TEMP'])
    par = cluster_signal(RECORDS, time_col, clusters['PAR'])

    # make window indices
    ign_idx = slice_windows_idx(t, fs, ignition_windows)
    # baseline complement as one big block
    base_mask = np.ones_like(t, dtype=bool)
    for (t0,t1) in ignition_windows:
        i0,i1 = int(t0*fs), int(t1*fs)
        base_mask[max(0,i0):min(len(t),i1)] = False
    base_idx = [(np.where(base_mask)[0][0], np.where(base_mask)[0][-1])] if np.any(base_mask) else []
    reb_idx  = slice_windows_idx(t, fs, rebound_windows) if rebound_windows else []
    ctrl_idx = slice_windows_idx(t, fs, control_windows) if control_windows else []

    targets = {'OCC':occ, 'TEMP':tmp, 'PAR':par}

    rows = []
    for bname,(f1,f2) in bands.items():
        for tgt_name, tgt in targets.items():
            # ignition means (average across ignition windows)
            dpli_ign = np.nanmean([dpli_block(src[s:e], tgt[s:e], fs, f1,f2) for (s,e) in ign_idx]) if ign_idx else np.nan
            dpli_base= np.nanmean([dpli_block(src[s:e], tgt[s:e], fs, f1,f2) for (s,e) in base_idx]) if base_idx else np.nan
            dpli_reb = np.nanmean([dpli_block(src[s:e], tgt[s:e], fs, f1,f2) for (s,e) in reb_idx]) if reb_idx else np.nan
            d_dpli   = (dpli_ign - dpli_base) if np.isfinite(dpli_ign) and np.isfinite(dpli_base) else np.nan

            # reverse direction for control
            dpli_rev_ign = np.nanmean([dpli_block(tgt[s:e], src[s:e], fs, f1,f2) for (s,e) in ign_idx]) if ign_idx else np.nan
            dpli_rev_base= np.nanmean([dpli_block(tgt[s:e], src[s:e], fs, f1,f2) for (s,e) in base_idx]) if base_idx else np.nan
            d_dpli_rev   = (dpli_rev_ign - dpli_rev_base) if np.isfinite(dpli_rev_ign) and np.isfinite(dpli_rev_base) else np.nan

            # Granger advantage if available
            gc_adv = np.nan
            if do_granger and _HAS_SM:
                gc_ign = np.nanmean([granger_block(src[s:e], tgt[s:e]) for (s,e) in ign_idx]) if ign_idx else np.nan
                gc_base= np.nanmean([granger_block(src[s:e], tgt[s:e]) for (s,e) in base_idx]) if base_idx else np.nan
                gc_adv = (gc_ign - gc_base) if np.isfinite(gc_ign) and np.isfinite(gc_base) else np.nan

            rows.append({'band':bname,'target':tgt_name,
                         'd_dpli':d_dpli, 'd_dpli_rev':d_dpli_rev, 'd_gc_adv':gc_adv,
                         'dpli_ign':dpli_ign,'dpli_base':dpli_base,'dpli_reb':dpli_reb})

    delta_table = pd.DataFrame(rows)

    # Optional shuffled control windows: sample L equal‑length random windows in baseline mask
    shuffle = None
    if do_shuffle_control and n_shuffle>0 and ign_idx:
        rng = np.random.default_rng(shuffle_seed)
        L = sum(e-s for (s,e) in ign_idx)  # total ignition length in samples
        base_inds = np.where(base_mask)[0]
        sh_rows=[]
        for _ in range(n_shuffle):
            # draw one contiguous segment of length L from baseline mask, if possible
            if base_inds.size <= L: continue
            start = rng.integers(low=base_inds[0], high=base_inds[-1]-L)
            sh_idx = [(start, start+L)]
            for bname,(f1,f2) in bands.items():
                for tgt_name, tgt in targets.items():
                    dpli_sh = np.nanmean([dpli_block(src[s:e], tgt[s:e], fs, f1,f2) for (s,e) in sh_idx])
                    dpli_b  = np.nanmean([dpli_block(src[s:e], tgt[s:e], fs, f1,f2) for (s,e) in base_idx]) if base_idx else np.nan
                    sh_rows.append({'band':bname,'target':tgt_name,'d_dpli_sh':dpli_sh-dpli_b})
        shuffle = pd.DataFrame(sh_rows)

    return {'delta_table': delta_table, 'shuffle_control': shuffle}

# ----------------- plotting -----------------

def plot_directional_deltas(df: pd.DataFrame) -> None:
    # Plot d_dpli by band × target (positive = stronger R‑DLPFC→target during ignition)
    bands = sorted(df['band'].unique())
    targets = ['OCC','TEMP','PAR']
    fig, axs = plt.subplots(1, len(targets), figsize=(12,3.2), constrained_layout=True)
    if len(targets)==1: axs=[axs]
    for j,tgt in enumerate(targets):
        sub = df[df['target']==tgt]
        vals = [float(sub[sub['band']==b]['d_dpli'].mean()) if not sub[sub['band']==b].empty else np.nan for b in bands]
        axs[j].bar(np.arange(len(bands)), vals, width=0.6)
        axs[j].set_xticks(np.arange(len(bands))); axs[j].set_xticklabels(bands)
        axs[j].axhline(0, color='k', lw=0.8)
        axs[j].set_title(f'Δ dPLI: R‑DLPFC→{tgt}')
        axs[j].set_ylabel('Ign − Base')
    plt.show()

"""
Multi-Scale Entropy (MSE) & Fractal Scaling (DFA) — Simple Graphs & Validation
==============================================================================

What it does
------------
• Builds a robust 1-D EEG drive (mean across chosen channels, z-scored).
• MSE: Sample Entropy across coarse-grain scales (1 .. ~seconds), with
        surrogate null (phase-randomized) → 95% band & p-values (AUC, mid-scales).
• DFA: log–log slope α of RMS fluctuations across windows (0.25–20 s by default),
       with surrogate null → 95% band & p-value.
• Optionally runs both for Ignition and Baseline and writes a concise summary.

Outputs
-------
• PNGs: MSE curve with null band; DFA log–log with slope line; Ign vs Base bars.
• CSV: summary.csv with MSE_AUC, MSE_mid, DFA_alpha and p-values.

Usage
-----
res = run_mse_dfa_multiscale(
    RECORDS,
    eeg_channels=['EEG.O1','EEG.O2','EEG.P7','EEG.P8'],   # clean posterior set recommended
    ignition_windows=[(290,310),(580,600)],
    baseline_windows=[(0,290),(325,580)],
    time_col='Timestamp',
    out_dir='exports_mse_dfa/S01',
    show=False
)
print(res['summary'])
"""
from __future__ import annotations
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from scipy import signal

# ------------------- small I/O helpers -------------------
def _ensure_dir(d): os.makedirs(d, exist_ok=True); return d

def detect_time_col(df,
    candidates=('Timestamp','Time','time','t','seconds','sec','ms','datetime','DateTime','Datetime')
)->Optional[str]:
    for c in candidates:
        if c in df.columns: return c
    # first numeric, roughly monotonic
    for c in df.columns:
        s = pd.to_numeric(df[c], errors='coerce')
        if s.notna().sum() > max(50, 0.5*len(df)):
            x = s.values.astype(float); dt = np.diff(x[np.isfinite(x)])
            if dt.size and np.nanmedian(dt) > 0: return c
    # datetime?
    for c in df.columns:
        try:
            _ = pd.to_datetime(df[c], errors='raise'); return c
        except Exception: pass
    return None

def ensure_timestamp_column(df: pd.DataFrame, time_col: Optional[str]=None,
                            default_fs: float = 128.0, out_name: str = 'Timestamp')->str:
    col = time_col or detect_time_col(df)
    if col is None:
        df[out_name] = np.arange(len(df), dtype=float)/default_fs
        return out_name
    s = df[col]
    if np.issubdtype(s.dtype, np.datetime64) or 'datetime' in str(s.dtype).lower():
        tsec = (pd.to_datetime(s) - pd.to_datetime(s).iloc[0]).dt.total_seconds().astype(float)
        df[out_name] = tsec.values; return out_name
    sn = pd.to_numeric(s, errors='coerce').astype(float)
    if sn.notna().sum() < max(50, 0.5*len(df)):
        df[out_name] = np.arange(len(df), dtype=float)/default_fs; return out_name
    sn = sn - np.nanmin(sn[np.isfinite(sn)])
    df[out_name] = sn.values
    return out_name

def infer_fs(df: pd.DataFrame, time_col: str)->float:
    t = np.asarray(pd.to_numeric(df[time_col], errors='coerce').values, float)
    dt = np.diff(t); dt = dt[(dt>0)&np.isfinite(dt)]
    if dt.size==0: raise ValueError("Cannot infer fs from time column.")
    return float(1.0/np.median(dt))

def get_series(df: pd.DataFrame, name: str)->np.ndarray:
    if name in df.columns:
        x = pd.to_numeric(df[name], errors='coerce').fillna(0.0).values
        return np.asarray(x, float)
    alt = 'EEG.'+name
    if alt in df.columns:
        x = pd.to_numeric(df[alt], errors='coerce').fillna(0.0).values
        return np.asarray(x, float)
    raise ValueError(f"Series '{name}' not in DataFrame.")

def slice_concat(x: np.ndarray, fs: float, wins: Optional[List[Tuple[float,float]]])->np.ndarray:
    if not wins: return x.copy()
    segs=[]; n=len(x)
    for (a,b) in wins:
        i0,i1 = int(round(a*fs)), int(round(b*fs))
        i0=max(0,i0); i1=min(n,i1)
        if i1>i0: segs.append(x[i0:i1])
    return np.concatenate(segs) if segs else x.copy()

def zscore(x): x = np.asarray(x,float); return (x - np.mean(x)) / (np.std(x)+1e-12)

# ------------------- MSE (Sample Entropy) -------------------
def coarse_grain(x: np.ndarray, scale: int) -> np.ndarray:
    """Non-overlapping average; drops remainder."""
    N = len(x)//scale * scale
    if N < scale: return np.array([])
    return np.mean(x[:N].reshape(-1, scale), axis=1)

def sampen(x: np.ndarray, m: int = 2, r_ratio: float = 0.2) -> float:
    """
    Sample Entropy (m,r) with Chebyshev metric.
    Returns -ln( A / B ), where:
      B = count matches of length m; A = count matches of length m+1.
    """
    x = np.asarray(x, float)
    N = len(x)
    if N < (m+2): return np.nan
    r = r_ratio * np.std(x)
    # embed
    Xm = np.column_stack([x[i:N-m+1+i] for i in range(m)])
    Xm1= np.column_stack([x[i:N-m  +i] for i in range(m+1)])
    # pairwise Chebyshev distance counts (exclude self)
    def count_matches(X, tol):
        C = 0
        M = len(X)
        for i in range(M-1):
            d = np.max(np.abs(X[i+1:] - X[i]), axis=1)
            C += np.sum(d <= tol)
        return C
    B = count_matches(Xm,  r)
    A = count_matches(Xm1, r)
    if B == 0 or A == 0: return np.nan
    return float(-np.log(A / B))

def mse_curve(x: np.ndarray,
              fs: float,
              max_scale_sec: float = 5.0,
              m: int = 2, r_ratio: float = 0.2,
              min_scale: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute MSE over integer coarse-grain scales up to max_scale_sec.
    Returns (scales_in_sec, SampEn values).
    """
    max_scale = max(min_scale, int(round(max_scale_sec * fs)))
    scales = np.arange(min_scale, max_scale+1, dtype=int)
    S = []
    for s in scales:
        cg = coarse_grain(x, s)
        if cg.size < (m+2):
            S.append(np.nan)
        else:
            S.append(sampen(cg, m=m, r_ratio=r_ratio))
    return scales/fs, np.array(S, float)

# ------------------- DFA -------------------
def dfa_alpha(x: np.ndarray,
              fs: float,
              min_win_sec: float = 0.25,
              max_win_sec: float = 20.0,
              n_win: int = 20) -> Dict[str, object]:
    """
    Detrended Fluctuation Analysis on z-scored signal (integrated profile).
    Returns alpha (slope) and log–log arrays.
    """
    x = zscore(x)
    y = np.cumsum(x - np.mean(x))
    # window sizes in samples (log-spaced)
    n_min = max(4, int(round(min_win_sec*fs)))
    n_max = max(n_min+1, int(round(max_win_sec*fs)))
    ns = np.unique(np.logspace(np.log10(n_min), np.log10(n_max), n_win).astype(int))
    F = []
    for n in ns:
        if n >= len(y): break
        # segment into non-overlapping windows
        N = len(y)//n * n
        yN = y[:N].reshape(-1, n)
        # linear detrend each segment
        t = np.arange(n)
        rms_segments=[]
        for seg in yN:
            p = np.polyfit(t, seg, 1)
            trend = np.polyval(p, t)
            rms = np.sqrt(np.mean((seg - trend)**2))
            rms_segments.append(rms)
        F.append(np.sqrt(np.mean(np.array(rms_segments)**2)))
    ns = ns[:len(F)]
    F = np.array(F, float)
    # slope on log–log
    X = np.log(ns); Y = np.log(F + 1e-24)
    A = np.vstack([X, np.ones_like(X)]).T
    slope, intercept = np.linalg.lstsq(A, Y, rcond=None)[0]
    return {'alpha': float(slope), 'ns': ns, 'F': F, 'fit': (float(slope), float(intercept))}

# ------------------- Surrogates -------------------
def phase_randomize(x: np.ndarray)->np.ndarray:
    X = np.fft.rfft(x); mag = np.abs(X); ph = np.angle(X)
    rnd = np.random.uniform(-np.pi, np.pi, size=mag.size)
    rnd[0] = ph[0]
    if mag.size % 2 == 0: rnd[-1] = ph[-1]
    Xs = mag * np.exp(1j*rnd)
    return np.fft.irfft(Xs, n=len(x)).astype(float)

# ------------------- Orchestrator -------------------
def run_mse_dfa_multiscale(
    RECORDS: pd.DataFrame,
    eeg_channels: List[str],
    ignition_windows: Optional[List[Tuple[float,float]]] = None,
    baseline_windows: Optional[List[Tuple[float,float]]] = None,
    time_col: str = 'Timestamp',
    out_dir: str = 'exports_mse_dfa/session',
    show: bool = False,
    mse_max_scale_sec: float = 5.0,
    mse_m: int = 2, mse_r_ratio: float = 0.2,
    dfa_min_sec: float = 0.25, dfa_max_sec: float = 20.0,
    n_surrogates: int = 200
)->Dict[str, object]:
    """
    MSE + DFA with surrogate validation, for ignition and baseline windows.
    """
    _ensure_dir(out_dir)
    time_col = ensure_timestamp_column(RECORDS, time_col=time_col, default_fs=128.0)
    fs = infer_fs(RECORDS, time_col)

    # robust 1-D drive = mean across selected channels
    def build_drive(wins):
        sigs=[]
        for ch in eeg_channels:
            nm = ch if ch.startswith('EEG.') else 'EEG.'+ch
            if nm in RECORDS.columns:
                x = get_series(RECORDS, nm)
                if wins: x = slice_concat(x, fs, wins)
                sigs.append(np.asarray(x,float))
        if not sigs: raise ValueError("No EEG channels found.")
        L = min(map(len, sigs))
        return zscore(np.mean(np.vstack([s[:L] for s in sigs]), axis=0))

    states = {'ignition': ignition_windows, 'baseline': baseline_windows}
    summaries=[]; outputs={}
    rng = np.random.default_rng(11)

    for st, wins in states.items():
        if wins is None: continue
        x = build_drive(wins)

        # --- MSE
        scales_sec, mse = mse_curve(x, fs, max_scale_sec=mse_max_scale_sec, m=mse_m, r_ratio=mse_r_ratio, min_scale=1)
        # MSE surrogate null (AUC & mid-scale mean)
        mse_null=[]
        for _ in range(n_surrogates):
            xs = zscore(phase_randomize(x))
            _, msen = mse_curve(xs, fs, max_scale_sec=mse_max_scale_sec, m=mse_m, r_ratio=mse_r_ratio, min_scale=1)
            mse_null.append(msen)
        mse_null = np.vstack(mse_null) if len(mse_null) else np.empty((0,len(scales_sec)))
        # summary features
        mse_auc = float(np.nansum(mse))                 # simple area under curve
        mid_mask = (scales_sec>=0.2) & (scales_sec<=1.0)
        mse_mid = float(np.nanmean(mse[mid_mask])) if np.any(mid_mask) else np.nan
        # null bands
        mse_lo = np.nanpercentile(mse_null, 2.5, axis=0) if mse_null.size else np.full_like(mse, np.nan)
        mse_hi = np.nanpercentile(mse_null,97.5, axis=0) if mse_null.size else np.full_like(mse, np.nan)
        # p-values (one-sided)
        mse_auc_p = float((np.sum(np.nansum(mse_null, axis=1) >= mse_auc)+1)/(mse_null.shape[0]+1)) if mse_null.size else np.nan
        mse_mid_p = float((np.sum(np.nanmean(mse_null[:,mid_mask], axis=1) >= mse_mid)+1)/(np.sum(mid_mask)+1)) if (mse_null.size and np.any(mid_mask)) else np.nan

        # --- DFA
        dfa = dfa_alpha(x, fs, min_win_sec=dfa_min_sec, max_win_sec=dfa_max_sec, n_win=20)
        # DFA surrogate null on alpha
        alpha_null=[]
        for _ in range(n_surrogates):
            xs = zscore(phase_randomize(x))
            alpha_null.append(dfa_alpha(xs, fs, min_win_sec=dfa_min_sec, max_win_sec=dfa_max_sec, n_win=20)['alpha'])
        alpha_null = np.asarray(alpha_null, float)
        alpha_p = float((np.sum(alpha_null >= dfa['alpha'])+1)/(alpha_null.size+1)) if alpha_null.size else np.nan
        alpha_lo = float(np.nanpercentile(alpha_null, 2.5)) if alpha_null.size else np.nan
        alpha_hi = float(np.nanpercentile(alpha_null,97.5)) if alpha_null.size else np.nan

        # --- Save plots ---
        # MSE curve
        plt.figure(figsize=(6,3.2))
        plt.plot(scales_sec, mse, lw=1.8, label='MSE')
        if mse_null.size:
            plt.fill_between(scales_sec, mse_lo, mse_hi, color='k', alpha=0.12, label='surrogate 95%')
        plt.xlabel('Scale (s)'); plt.ylabel('SampEn'); plt.title(f'MSE — {st}')
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f'mse_{st}.png'), dpi=140)
        if show: plt.show()
        plt.close()

        # DFA log–log
        X = np.log(dfa['ns']); Y = np.log(dfa['F']+1e-24); s,b = dfa['fit']
        plt.figure(figsize=(6,3.2))
        plt.plot(X, Y, 'o-', lw=1.0, label='log–log')
        plt.plot(X, s*X + b, 'r--', lw=1.2, label=f'α≈{dfa["alpha"]:.2f}')
        if alpha_null.size:
            plt.hlines([alpha_lo, alpha_hi], X.min(), X.max(), colors='k', linestyles=':', lw=1, label='surrogate 95% (α)')
        plt.xlabel('log window n'); plt.ylabel('log F(n)'); plt.title(f'DFA — {st}')
        plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(out_dir, f'dfa_{st}.png'), dpi=140)
        if show: plt.show()
        plt.close()

        summaries.append({'state':st,
                          'MSE_AUC':mse_auc, 'MSE_AUC_p':mse_auc_p,
                          'MSE_mid':mse_mid, 'MSE_mid_p':mse_mid_p,
                          'DFA_alpha':dfa['alpha'], 'DFA_alpha_p':alpha_p,
                          'DFA_alpha_lo95':alpha_lo, 'DFA_alpha_hi95':alpha_hi})

        outputs[st] = {'mse': {'scales':scales_sec, 'S':mse, 'lo':mse_lo, 'hi':mse_hi},
                       'dfa': dfa,
                       'mse_null': mse_null, 'alpha_null': alpha_null}

    # Ignition vs Baseline bar (optional)
    if 'ignition' in outputs and 'baseline' in outputs:
        ig = [s for s in summaries if s['state']=='ignition'][0]
        ba = [s for s in summaries if s['state']=='baseline'][0]
        plt.figure(figsize=(6,3.2))
        names=['MSE_AUC','MSE_mid','DFA_alpha']
        x = np.arange(len(names)); w=0.35
        plt.bar(x-w/2, [ba[n] for n in names], width=w, label='Baseline', color='tab:orange', alpha=0.9)
        plt.bar(x+w/2, [ig[n] for n in names], width=w, label='Ignition', color='tab:blue', alpha=0.9)
        plt.xticks(x, names); plt.ylabel('value'); plt.title('Ignition vs Baseline — MSE/DFA')
        plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(out_dir,'ign_vs_base.png'), dpi=140)
        if show: plt.show()
        plt.close()

    # Save summary
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(os.path.join(out_dir,'summary.csv'), index=False)
    return {'summary': summary_df, 'outputs': outputs, 'out_dir': out_dir}

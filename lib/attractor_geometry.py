"""
Topological Data Analysis of Attractor Geometry — Simple Graphs & Validation
============================================================================

What it does
------------
1) Build a 1D robust EEG drive (mean or PCA1 over given channels).
2) Auto-pick Takens parameters (τ from ACF 1/e; m from False Nearest Neighbors).
3) Delay-embed → point cloud X in R^m (subsampled for speed).
4) Persistent homology (Ripser if available) → H0/H1/H2 diagrams:
     • b1_count_sig (number of significant loops)
     • b2_count_sig (number of significant voids)
     • max persistence in H1/H2 and p-values vs surrogates
   (Torus heuristic: b1>=2 and b2>=1 with significant persistence)
5) Fallback (if ripser missing): Recurrence Plot + RQA (RR, DET) with surrogates.
6) Saves PNGs + CSV summary to out_dir; minimal inline output if show=False.

Usage (example)
---------------
res = run_tda_attractor_topology(
    RECORDS,
    eeg_channels=['EEG.O1','EEG.O2','EEG.P7','EEG.P8'],
    ignition_windows=[(290,310),(580,600)],     # or None for full
    time_col='Timestamp',
    out_dir='exports_tda/S01',
    show=False                                   # save figures; avoids notebook flooding
)
print(res['summary'])
"""

from __future__ import annotations
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from scipy import signal

# Optional: Ripser / persim for persistent homology
try:
    from ripser import ripser
    from persim import plot_diagrams
    _HAS_RIPSER = True
except Exception:
    _HAS_RIPSER = False

# ---------------- I/O & time helpers ----------------
def _ensure_dir(d): os.makedirs(d, exist_ok=True); return d

def detect_time_col(df,
    candidates=('Timestamp','Time','time','t','seconds','sec','ms','datetime','DateTime','Datetime')
)->Optional[str]:
    for c in candidates:
        if c in df.columns: return c
    for c in df.columns:  # first numeric, roughly monotonic
        s = pd.to_numeric(df[c], errors='coerce')
        if s.notna().sum() > max(50, 0.5*len(df)):
            arr = s.values.astype(float); dt = np.diff(arr[np.isfinite(arr)])
            if dt.size and np.nanmedian(dt) > 0: return c
    for c in df.columns:  # datetime?
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
    df[out_name] = sn.values; return out_name

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
    raise ValueError(f"Series '{name}' not found.")

def slice_concat(x: np.ndarray, fs: float, wins: Optional[List[Tuple[float,float]]])->np.ndarray:
    if not wins: return x.copy()
    segs=[]; n=len(x)
    for (a,b) in wins:
        i0,i1 = int(round(a*fs)), int(round(b*fs))
        i0=max(0,i0); i1=min(n,i1)
        if i1>i0: segs.append(x[i0:i1])
    return np.concatenate(segs) if segs else x.copy()

def zscore(x): x = np.asarray(x,float); return (x - np.mean(x)) / (np.std(x)+1e-12)

# ---------------- Delay-embedding tools ----------------
try:
    from sklearn.neighbors import KDTree
    _HAS_SK = True
except Exception:
    _HAS_SK = False

def estimate_delay_tau(x: np.ndarray, fs: float, max_lag_sec: float = 2.0, method='acf-1e')->int:
    nlag = int(max(1, round(max_lag_sec*fs)))
    xx = zscore(x)
    acf = signal.correlate(xx, xx, mode='full'); acf = acf[acf.size//2:acf.size//2+nlag+1]
    acf = acf/(acf[0]+1e-12)
    if method=='zero':
        idx = np.where(np.sign(acf[1:])!=np.sign(acf[:-1]))[0]
        tau = int(idx[0]+1) if idx.size else max(1,int(0.05*fs))
    else:
        idx = np.where(acf <= 1/np.e)[0]; tau = int(idx[0]) if idx.size else max(1,int(0.05*fs))
    return max(1, tau)

def takens_embedding(x: np.ndarray, m: int, tau: int)->np.ndarray:
    N = len(x) - (m-1)*tau
    if N <= 10: raise ValueError("Time series too short for requested embedding.")
    return np.column_stack([x[i:i+N] for i in range(0, m*tau, tau)]).astype(float)

def false_nearest_neighbors(x: np.ndarray, tau: int, m_list: List[int], theiler: int = 10)->pd.DataFrame:
    rows=[]
    for m in m_list:
        try:
            X_m = takens_embedding(x, m, tau); X_m1 = takens_embedding(x, m+1, tau)
            N = min(len(X_m), len(X_m1)); X_m = X_m[:N]; X_m1 = X_m1[:N]
        except Exception:
            rows.append({'m':m,'FNN%':np.nan}); continue
        if _HAS_SK:
            tree = KDTree(X_m); d, idxs = tree.query(X_m, k=2); nn = idxs[:,1]
            for i in range(N):
                if abs(nn[i]-i) <= theiler:
                    dists, idx2 = tree.query(X_m[i:i+1], k=10)
                    for cand in idx2[0,1:]:
                        if abs(cand-i) > theiler: nn[i]=cand; break
        else:
            nn = np.zeros(N, dtype=int)
            for i in range(N):
                d = np.linalg.norm(X_m - X_m[i], axis=1); d[i]=np.inf
                order = np.argsort(d); j=order[0]
                if abs(j-i) <= theiler:
                    for cand in order[1:]:
                        if abs(cand-i) > theiler: j=cand; break
                nn[i]=j
        Rtol=15.0
        dist_m  = np.linalg.norm(X_m - X_m[nn],  axis=1)
        dist_m1 = np.linalg.norm(X_m1- X_m1[nn], axis=1)
        ratio = dist_m1/(dist_m+1e-12)
        rows.append({'m':m, 'FNN%': float(np.mean(ratio>Rtol)*100.0)})
    return pd.DataFrame(rows)

# ---------------- Surrogates ----------------
def phase_randomize(x: np.ndarray)->np.ndarray:
    X = np.fft.rfft(x); mag = np.abs(X); ph = np.angle(X)
    rnd = np.random.uniform(-np.pi, np.pi, size=mag.size)
    rnd[0] = ph[0]
    if mag.size % 2 == 0: rnd[-1] = ph[-1]
    Xs = mag * np.exp(1j*rnd)
    return np.fft.irfft(Xs, n=len(x)).astype(float)

# ---------------- RQA fallback ----------------
def recurrence_plot(X: np.ndarray, eps_quant: float = 0.1)->Dict[str,object]:
    N = len(X); idx = np.random.choice(N, size=min(N, 1200), replace=False)
    Y = X[idx]; D = np.sqrt(((Y[:,None,:]-Y[None,:,:])**2).sum(axis=2))
    eps = np.quantile(D, eps_quant); R = (D <= eps).astype(int); np.fill_diagonal(R, 0)
    RR = float(np.mean(R))
    # simple DET
    det_lines=0; total_lines=0
    for i in range(R.shape[0]-1):
        run=0
        for j in range(R.shape[1]-1):
            if R[i,j]==1 and R[i+1,j+1]==1: run+=1
            else:
                if run>=1: total_lines+=1;
                if run+1>=2: det_lines+=1
                run=0
        if run>=1: total_lines+=1;
        if run+1>=2: det_lines+=1
    DET = det_lines/(total_lines+1e-12)
    return {'R':R, 'RR':RR, 'DET':float(DET)}

# ---------------- Main TDA runner ----------------
def run_tda_attractor_topology(
    RECORDS: pd.DataFrame,
    eeg_channels: List[str],
    ignition_windows: Optional[List[Tuple[float,float]]] = None,
    time_col: str = 'Timestamp',
    out_dir: str = 'exports_tda/session',
    m_list: List[int] = [2,3,4,5,6,7,8],
    max_points: int = 5000,
    n_surrogates: int = 100,
    show: bool = False
)->Dict[str, object]:
    """
    Takens embedding + persistent homology (with surrogates) to test torus-like topology.
    """
    _ensure_dir(out_dir)
    time_col = ensure_timestamp_column(RECORDS, time_col=time_col, default_fs=128.0)
    fs = infer_fs(RECORDS, time_col)

    # robust drive: mean across given channels (z-scored)
    Xsig=[]
    for ch in eeg_channels:
        nm = ch if ch.startswith('EEG.') else 'EEG.'+ch
        if nm in RECORDS.columns:
            x = get_series(RECORDS, nm)
            if ignition_windows: x = slice_concat(x, fs, ignition_windows)
            Xsig.append(np.asarray(x,float))
    if not Xsig: raise ValueError("No EEG channels found in RECORDS for eeg_channels.")
    L = min(len(x) for x in Xsig)
    x = zscore(np.mean(np.vstack([xx[:L] for xx in Xsig]), axis=0))

    # Takens params
    tau = estimate_delay_tau(x, fs, max_lag_sec=2.0, method='acf-1e')
    fnn = false_nearest_neighbors(x, tau, m_list)
    fnn = fnn.dropna()
    if np.any(fnn['FNN%'] <= 5.0):
        m_star = int(fnn.loc[fnn['FNN%']<=5.0,'m'].iloc[0])
    else:
        m_star = int(fnn.iloc[np.argmin(fnn['FNN%'])]['m'])
    m_star = max(3, m_star)

    # Embed
    X = takens_embedding(x, m_star, tau)
    # Subsample for speed
    if len(X) > max_points:
        idx = np.linspace(0, len(X)-1, max_points).astype(int)
        X = X[idx]

    summary = {'tau':tau, 'm':m_star}
    outputs = {}

    # ---------------- Persistent homology path ----------------
    if _HAS_RIPSER:
        rp = ripser(X, maxdim=2)     # Vietoris–Rips on point cloud
        dgms = rp['dgms']            # [H0, H1, H2]
        # persistence = death - birth; ignore inf deaths for max
        def max_persistence(dgm):
            if dgm.size == 0: return 0.0
            pers = dgm[:,1] - dgm[:,0]
            pers = pers[np.isfinite(pers)]
            return float(np.nanmax(pers)) if pers.size else 0.0

        maxH1 = max_persistence(dgms[1]); maxH2 = max_persistence(dgms[2])

        # Surrogate nulls (phase-randomize the drive, same tau/m)
        nullH1=[]; nullH2=[]
        for _ in range(n_surrogates):
            xs = zscore(phase_randomize(x))
            Xs = takens_embedding(xs, m_star, tau)
            if len(Xs) > max_points:
                idx = np.linspace(0, len(Xs)-1, max_points).astype(int)
                Xs = Xs[idx]
            rps = ripser(Xs, maxdim=2)
            d1, d2 = rps['dgms'][1], rps['dgms'][2]
            # max persistence per homology class
            def maxP(dgm):
                if dgm.size==0: return 0.0
                per = dgm[:,1]-dgm[:,0]; per = per[np.isfinite(per)]
                return float(np.nanmax(per)) if per.size else 0.0
            nullH1.append(maxP(d1)); nullH2.append(maxP(d2))
        thrH1 = float(np.nanpercentile(nullH1, 95)) if nullH1 else np.nan
        thrH2 = float(np.nanpercentile(nullH2, 95)) if nullH2 else np.nan

        # Simple counts of “significant” features (bars above threshold)
        def count_sig(dgm, thr):
            if dgm.size==0 or not np.isfinite(thr): return 0
            per = dgm[:,1]-dgm[:,0]; per = per[np.isfinite(per)]
            return int(np.sum(per >= thr))
        b1_sig = count_sig(dgms[1], thrH1)
        b2_sig = count_sig(dgms[2], thrH2)

        # Torus heuristic (T^2): H1: >=2, H2: >=1
        summary.update({
            'max_persistence_H1': maxH1, 'null95_H1': thrH1, 'b1_count_sig': b1_sig,
            'max_persistence_H2': maxH2, 'null95_H2': thrH2, 'b2_count_sig': b2_sig,
            'torus_heuristic_pass': bool((b1_sig>=2) and (b2_sig>=1))
        })

        # Plots
        # 2D/3D projections of embedding (pairwise)
        stride = max(1, len(X)//6000)
        plt.figure(figsize=(8,3))
        plt.subplot(1,2,1); plt.plot(X[::stride,0], X[::stride,1], lw=0.5, alpha=0.8)
        plt.title(f'Embedding X1–X2 (m={m_star}, τ={tau})')
        if X.shape[1]>=3:
            plt.subplot(1,2,2); plt.plot(X[::stride,1], X[::stride,2], lw=0.5, alpha=0.8)
            plt.title('Embedding X2–X3')
        plt.tight_layout(); plt.savefig(os.path.join(out_dir,'embedding_pairwise.png'), dpi=140)
        if show: plt.show()
        plt.close()

        # Persistence diagrams
        plt.figure(figsize=(6,3))
        plot_diagrams(dgms, show=False)
        plt.title('Persistence diagrams (H0,H1,H2)')
        plt.tight_layout(); plt.savefig(os.path.join(out_dir,'persistence_diagrams.png'), dpi=140)
        if show: plt.show()
        plt.close()

        outputs['dgms'] = dgms

    # ---------------- Fallback: Recurrence / RQA when Ripser unavailable ----------------
    else:
        # RP & RQA for real data
        rqa = recurrence_plot(X, eps_quant=0.1)
        # nulls
        null_DET=[]
        for _ in range(n_surrogates):
            xs = zscore(phase_randomize(x))
            Xs = takens_embedding(xs, m_star, tau)
            rq = recurrence_plot(Xs, eps_quant=0.1)
            null_DET.append(rq['DET'])
        thrDET = float(np.nanpercentile(null_DET, 95)) if null_DET else np.nan
        summary.update({'DET': float(rqa['DET']), 'DET_null95': thrDET,
                        'loopiness_pass': bool(np.isfinite(thrDET) and (rqa['DET']>thrDET))})
        # Save RP image
        plt.figure(figsize=(4,4))
        plt.imshow(rqa['R'], origin='lower', cmap='binary'); plt.title(f'RP (DET={rqa["DET"]:.2f})')
        plt.tight_layout(); plt.savefig(os.path.join(out_dir,'recurrence_plot.png'), dpi=140)
        if show: plt.show()
        plt.close()

        outputs['rqa'] = rqa

    # Save summary
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir,'summary.csv'), index=False)
    return {'summary': summary, 'out_dir': out_dir, 'params': {'tau':tau, 'm':m_star}, 'outputs': outputs}

"""
Recurrence Quantification & Chaos Metrics — Simple Graphs & Validation
=====================================================================

This module validates nonlinear EEG dynamics with:
  • Recurrence Plot (RP) & RQA: RR, DET, LAM, Lmax, Lmean, Diag-Entropy, Trapping Time
  • Largest Lyapunov exponent λ_max (Rosenstein)
  • Correlation dimension D2 (Grassberger–Procaccia)
  • Phase-randomized surrogates → 95% nulls & one-sided p-values

Outputs (per state): PNGs + summary.csv in out_dir.

Usage
-----
res = run_rqa_chaos_metrics(
    RECORDS,
    eeg_channels=['EEG.O1','EEG.O2','EEG.P7','EEG.P8'],   # clean posterior set recommended
    ignition_windows=[(290,310),(580,600)],
    baseline_windows=[(0,290),(325,580)],
    time_col='Timestamp',
    out_dir='exports_rqa/S01',
    show=False
)
print(res['summary'])
"""
from __future__ import annotations
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from scipy import signal

# -------------------- small I/O helpers --------------------
def _ensure_dir(d): os.makedirs(d, exist_ok=True); return d

def detect_time_col(df,
    candidates=('Timestamp','Time','time','t','seconds','sec','ms','datetime','DateTime','Datetime')
)->Optional[str]:
    for c in candidates:
        if c in df.columns: return c
    for c in df.columns:  # first numeric & roughly monotonic
        s = pd.to_numeric(df[c], errors='coerce')
        if s.notna().sum() > max(50, 0.5*len(df)):
            arr = s.values.astype(float); dt = np.diff(arr[np.isfinite(arr)])
            if dt.size and np.nanmedian(dt) > 0: return c
    for c in df.columns:  # datetime
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
    raise ValueError(f"Series '{name}' not found.")

def slice_concat(x: np.ndarray, fs: float, wins: Optional[List[Tuple[float,float]]])->np.ndarray:
    if not wins: return x.copy()
    segs=[]; n=len(x)
    for (a,b) in wins:
        i0,i1 = int(round(a*fs)), int(round(b*fs))
        i0=max(0,i0); i1=min(n,i1)
        if i1>i0: segs.append(x[i0:i1])
    return np.concatenate(segs) if segs else x.copy()

def zscore(x): x=np.asarray(x,float); return (x - np.mean(x)) / (np.std(x)+1e-12)

# -------------------- embedding tools --------------------
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
        idx = np.where(np.sign(acf[1:])!=np.sign(acf[:-1]))[0]; tau = int(idx[0]+1) if idx.size else max(1,int(0.05*fs))
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
            # Theiler correction
            for i in range(N):
                if abs(nn[i]-i) <= theiler:
                    d2, idx2 = tree.query(X_m[i:i+1], k=10)
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

# -------------------- Lyapunov (Rosenstein, robust) --------------------
def lyapunov_rosenstein(x: np.ndarray,
                        m: int, tau: int, fs: float,
                        theiler: int = 10,
                        t_fit: Tuple[int,int] = (1, 30)) -> Dict[str, object]:
    X = takens_embedding(zscore(x), m, tau)
    N = len(X)
    if N <= theiler + 2:
        return {'lambda': np.nan, 'L': np.array([]), 'k': np.array([])}
    # nearest neighbor with Theiler exclusion
    if _HAS_SK:
        tree = KDTree(X); d, idxs = tree.query(X, k=min(20, N-1))
        nn = np.zeros(N, dtype=int)
        for i in range(N):
            chosen=None
            for cand in idxs[i,1:]:
                if abs(int(cand)-i) > theiler: chosen=int(cand); break
            nn[i] = chosen if chosen is not None else int(idxs[i,1])
    else:
        nn = np.zeros(N, dtype=int)
        for i in range(N):
            d = np.linalg.norm(X - X[i], axis=1); d[i]=np.inf
            order = np.argsort(d); chosen=None
            for cand in order:
                if abs(int(cand)-i) > theiler: chosen=int(cand); break
            nn[i] = chosen if chosen is not None else int(order[0])

    max_k = max(2, min(100, N-1))
    Lvals=[]; ks=[]
    for k in range(1, max_k):
        valid = (np.arange(N)+k < N) & (nn + k < N)
        if not np.any(valid): break
        idx = np.where(valid)[0]
        if idx.size < 5: break
        d0 = np.linalg.norm(X[idx]     - X[nn[idx]],     axis=1) + 1e-24
        dk = np.linalg.norm(X[idx + k] - X[nn[idx] + k], axis=1)
        Lvals.append(np.mean(np.log(dk/d0))); ks.append(k)
    Lvals = np.asarray(Lvals,float); ks = np.asarray(ks,int)
    if Lvals.size < 5: return {'lambda': np.nan, 'L': Lvals, 'k': ks}
    k0,k1 = t_fit; k1=min(k1,int(0.6*len(Lvals)))
    if k1 <= k0+1: return {'lambda': np.nan, 'L':Lvals, 'k':ks}
    A = np.vstack([ks[k0:k1], np.ones(k1-k0)]).T
    slope, _ = np.linalg.lstsq(A, Lvals[k0:k1], rcond=None)[0]
    lam = float(slope) * fs / float(tau)
    return {'lambda': lam, 'L': Lvals, 'k': ks}

# -------------------- Correlation dimension (GP) --------------------
def correlation_dimension_gp(X: np.ndarray,
                             r_min_quant: float = 0.05,
                             r_max_quant: float = 0.30,
                             n_r: int = 20) -> Dict[str, object]:
    N = len(X)
    idx = np.random.choice(N, size=min(N, 1200), replace=False)
    D = np.sqrt(((X[idx,None,:]-X[None,idx,:])**2).sum(axis=2)).ravel()
    D = D[D>0]; Dsorted = np.sort(D)
    rmin = Dsorted[int(r_min_quant*len(Dsorted))]
    rmax = Dsorted[int(r_max_quant*len(Dsorted))]
    r_vals = np.exp(np.linspace(np.log(rmin+1e-12), np.log(rmax+1e-12), n_r))
    C = np.array([np.mean(D < r) for r in r_vals])
    x = np.log(r_vals + 1e-24); y = np.log(C + 1e-24)
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return {'r': r_vals, 'C': C, 'D2': float(slope), 'fit': (float(slope), float(intercept))}

# -------------------- Recurrence plot & RQA --------------------
def recurrence_matrix(X: np.ndarray, eps: float, theiler: int = 0)->np.ndarray:
    D = np.sqrt(((X[:,None,:]-X[None,:,:])**2).sum(axis=2))
    R = (D <= eps).astype(int)
    # suppress main diagonal band (Theiler neighborhood)
    for i in range(-theiler, theiler+1):
        if i==0: np.fill_diagonal(R, 0)
        else:
            diag = np.diag_indices_from(R)
            ii = (diag[0][max(0,i):], diag[1][max(0,-i):]) if i>=0 else (diag[0][:i], diag[1][-i:])
            R[ii] = 0
    return R

def rqa_metrics(R: np.ndarray, lmin: int = 2, vmin: int = 2)->Dict[str, float]:
    N = R.shape[0]
    RR = float(np.sum(R)/(N*N))
    # Diagonal lines
    Ls=[]
    for k in range(-(N-1), N):
        diag = np.diag(R, k=k)
        # run-lengths of ones
        run=0
        for v in diag:
            if v==1: run+=1
            elif run>0:
                if run>=lmin: Ls.append(run)
                run=0
        if run>=lmin: Ls.append(run)
    Ls = np.array(Ls, int)
    DET = float(np.sum(Ls)/ (np.sum(R) + 1e-12))
    Lmax = float(np.max(Ls) if Ls.size else 0)
    Lmean = float(np.mean(Ls) if Ls.size else 0)
    # diagonal length entropy
    if Ls.size:
        bins = np.arange(lmin, np.max(Ls)+1)
        hist, _ = np.histogram(Ls, bins=bins)
        p = hist.astype(float)/ (np.sum(hist)+1e-12)
        p = p[p>0]; Hdiag = float(-np.sum(p*np.log(p)))
    else:
        Hdiag = 0.0
    # Vertical lines → laminarity & trapping time
    Vs=[]
    for col in range(N):
        run=0
        for row in range(N):
            if R[row,col]==1: run+=1
            elif run>0:
                if run>=vmin: Vs.append(run)
                run=0
        if run>=vmin: Vs.append(run)
    Vs = np.array(Vs,int)
    LAM = float(np.sum(Vs)/ (np.sum(R) + 1e-12))
    TT  = float(np.mean(Vs) if Vs.size else 0)
    return {'RR':RR, 'DET':DET, 'LAM':LAM, 'Lmax':Lmax, 'Lmean':Lmean, 'Hdiag':Hdiag, 'TT':TT}

# -------------------- Surrogates --------------------
def phase_randomize(x: np.ndarray)->np.ndarray:
    X = np.fft.rfft(x); mag = np.abs(X); ph = np.angle(X)
    rnd = np.random.uniform(-np.pi, np.pi, size=mag.size)
    rnd[0] = ph[0]
    if mag.size % 2 == 0: rnd[-1] = ph[-1]
    Xs = mag * np.exp(1j*rnd)
    return np.fft.irfft(Xs, n=len(x)).astype(float)

# -------------------- Orchestrator --------------------
def run_rqa_chaos_metrics(
    RECORDS: pd.DataFrame,
    eeg_channels: List[str],
    ignition_windows: Optional[List[Tuple[float,float]]] = None,
    baseline_windows: Optional[List[Tuple[float,float]]] = None,
    time_col: str = 'Timestamp',
    out_dir: str = 'exports_rqa/session',
    show: bool = False,
    m_list: List[int] = [2,3,4,5,6,7,8],
    eps_quantile: float = 0.10,
    theiler: int = 0,
    lmin: int = 2, vmin: int = 2,
    max_points: int = 5000,
    n_surrogates: int = 100
)->Dict[str, object]:
    """
    RQA + Chaos metrics with surrogate validation for ignition & baseline windows.
    """
    _ensure_dir(out_dir)
    time_col = ensure_timestamp_column(RECORDS, time_col=time_col, default_fs=128.0)
    fs = infer_fs(RECORDS, time_col)

    # robust 1D drive: mean across provided channels
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
    for st, wins in states.items():
        if wins is None: continue
        x = build_drive(wins)
        # Takens params
        tau = estimate_delay_tau(x, fs, max_lag_sec=2.0, method='acf-1e')
        fnn = false_nearest_neighbors(x, tau, m_list)
        fnn = fnn.dropna()
        if np.any(fnn['FNN%'] <= 5.0):
            m_star = int(fnn.loc[fnn['FNN%']<=5.0,'m'].iloc[0])
        else:
            m_star = int(fnn.iloc[np.argmin(fnn['FNN%'])]['m'])
        m_star = max(3, m_star)
        # Embed & subsample
        X = takens_embedding(x, m_star, tau)
        if len(X) > max_points:
            idx = np.linspace(0, len(X)-1, max_points).astype(int)
            X = X[idx]

        # Epsilon from distance quantile
        D = np.sqrt(((X[:,None,:]-X[None,:,:])**2).sum(axis=2)).ravel()
        D = D[D>0]; eps = float(np.quantile(D, eps_quantile))

        # Recurrence matrix & RQA
        R = recurrence_matrix(X, eps=eps, theiler=theiler)
        rqa = rqa_metrics(R, lmin=lmin, vmin=vmin)

        # Lyapunov & D2
        ly = lyapunov_rosenstein(x, m_star, tau, fs, theiler=max(theiler, int(0.02*fs)), t_fit=(1,30))
        gp = correlation_dimension_gp(X)

        # Surrogates
        null = {'DET':[], 'LAM':[], 'Lmax':[], 'Hdiag':[], 'TT':[], 'lambda':[], 'D2':[]}
        for _ in range(n_surrogates):
            xs = zscore(phase_randomize(x))
            Xs = takens_embedding(xs, m_star, tau)
            if len(Xs) > max_points:
                idx = np.linspace(0, len(Xs)-1, max_points).astype(int)
                Xs = Xs[idx]
            # same eps quantile on surrogate
            Ds = np.sqrt(((Xs[:,None,:]-Xs[None,:,:])**2).sum(axis=2)).ravel()
            Ds = Ds[Ds>0]; eps_s = float(np.quantile(Ds, eps_quantile))
            Rs = recurrence_matrix(Xs, eps=eps_s, theiler=theiler)
            rq = rqa_metrics(Rs, lmin=lmin, vmin=vmin)
            for k in ['DET','LAM','Lmax','Hdiag','TT']:
                null[k].append(rq[k])
            # lyapunov & D2 on surrogate
            lys = lyapunov_rosenstein(xs, m_star, tau, fs, theiler=max(theiler, int(0.02*fs)), t_fit=(1,30))
            gps = correlation_dimension_gp(Xs)
            null['lambda'].append(lys['lambda'])
            null['D2'].append(gps['D2'])

        def pval(obs, arr, greater=True):
            arr = np.asarray(arr, float)
            if not np.isfinite(obs) or arr.size==0: return np.nan
            if greater: return float((np.sum(arr >= obs)+1)/(arr.size+1))
            else:       return float((np.sum(arr <= obs)+1)/(arr.size+1))

        # p-values (one-sided): structure/chaos higher than null
        det_p   = pval(rqa['DET'],   null['DET'],   greater=True)
        lam_p   = pval(rqa['LAM'],   null['LAM'],   greater=True)
        lmax_p  = pval(rqa['Lmax'],  null['Lmax'],  greater=True)
        hdiag_p = pval(rqa['Hdiag'], null['Hdiag'], greater=True)
        tt_p    = pval(rqa['TT'],    null['TT'],    greater=True)
        ly_p    = pval(ly['lambda'], null['lambda'],greater=True)
        d2_p    = pval(gp['D2'],     null['D2'],    greater=True)

        summaries.append({
            'state': st, 'tau': tau, 'm': m_star, 'eps': eps,
            'RR': rqa['RR'], 'DET': rqa['DET'], 'LAM': rqa['LAM'],
            'Lmax': rqa['Lmax'], 'Lmean': rqa['Lmean'], 'Hdiag': rqa['Hdiag'], 'TT': rqa['TT'],
            'lambda': ly['lambda'], 'D2': gp['D2'],
            'DET_p': det_p, 'LAM_p': lam_p, 'Lmax_p': lmax_p, 'Hdiag_p': hdiag_p, 'TT_p': tt_p,
            'lambda_p': ly_p, 'D2_p': d2_p
        })

        # --------- Plots (lightweight) ---------
        # Recurrence plot
        plt.figure(figsize=(4,4))
        plt.imshow(R, origin='lower', cmap='binary')
        plt.title(f'RP — {st} (eps@{int(eps_quantile*100)}%)')
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f'rp_{st}.png'), dpi=140)
        if show: plt.show()
        plt.close()

        # Diagonal-length histogram (determinism structure)
        # Already stored via Hdiag; we can visualize with a toy histogram by recomputing Ls here quickly:
        Ls=[]
        for k in range(-(R.shape[0]-1), R.shape[0]):
            diag = np.diag(R,k=k); run=0
            for v in diag:
                if v==1: run+=1
                elif run>0:
                    if run>=lmin: Ls.append(run); run=0
            if run>=lmin: Ls.append(run)
        if len(Ls):
            plt.figure(figsize=(5,3))
            plt.hist(Ls, bins=np.arange(lmin, max(Ls)+1), color='tab:blue', alpha=0.9)
            plt.xlabel('Diagonal line length'); plt.ylabel('count')
            plt.title(f'Diagonal lengths — {st} (DET={rqa["DET"]:.2f})')
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, f'diag_lengths_{st}.png'), dpi=140)
            if show: plt.show()
            plt.close()

        # Lyapunov divergence curve
        if ly['L'].size:
            plt.figure(figsize=(5,3))
            plt.plot(ly['k']/fs*tau, ly['L'], lw=1.2)
            plt.xlabel('Time (s)'); plt.ylabel('⟨log(d_k/d_0)⟩')
            plt.title(f'Lyapunov divergence — {st} (λ≈{ly["lambda"]:.3f} s⁻¹)')
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, f'lyapunov_{st}.png'), dpi=140)
            if show: plt.show()
            plt.close()

        # Correlation dimension log–log
        plt.figure(figsize=(5,3))
        x = np.log(gp['r']+1e-24); y = np.log(gp['C']+1e-24)
        plt.plot(x, y, 'o-', lw=1)
        s, b = gp['fit']
        plt.plot(x, s*x + b, 'r--', lw=1)
        plt.xlabel('log r'); plt.ylabel('log C(r)')
        plt.title(f'Correlation sum — {st} (D2≈{gp["D2"]:.2f})')
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f'corr_dimension_{st}.png'), dpi=140)
        if show: plt.show()
        plt.close()

        outputs[st] = {'rqa': rqa, 'lyap': ly, 'gp': gp, 'tau': tau, 'm': m_star, 'eps': eps}

    # Save summary CSV
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(os.path.join(out_dir, 'summary.csv'), index=False)
    return {'summary': summary_df, 'outputs': outputs, 'out_dir': out_dir}

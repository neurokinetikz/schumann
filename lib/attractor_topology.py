"""
Attractor Topology via Nonlinear Dimensional Embedding — Simple Graphs & Validity Tests
======================================================================================

This module reconstructs EEG phase-space attractors (Takens delay embedding) and computes:
  • Delay τ (autocorrelation-based)
  • Embedding dimension m (False Nearest Neighbors; FNN)
  • Correlation (fractal) dimension D2 (Grassberger-Procaccia)
  • Largest Lyapunov exponent λ_max (Rosenstein)
  • Recurrence plot (RP) + simple RQA (recurrence rate RR, determinism DET)
  • (Optional) Persistent homology summaries if `ripser` is available

It includes:
  • Ignition vs Baseline comparison (windows you pass in)
  • Surrogate tests (phase-randomized and time-shuffled) → p-values
  • Clean, readable plots saved alongside a concise text/CSV summary

Assumptions:
  • RECORDS is a pandas.DataFrame with a numeric time column (default 'Timestamp')
    and EEG channels named 'EEG.*'. You pass one or more EEG channels; we average them
    as a robust single drive signal (or you can pass a single channel).
  • Python 3.7+ with NumPy, SciPy, scikit-learn (neighbors), matplotlib, pandas.
    (If scikit-learn is missing, we fall back to naïve kNN with NumPy.)

Usage (minimal):
----------------
res = run_attractor_topology(
    RECORDS,
    eeg_channels=['EEG.O1','EEG.O2','EEG.Oz'],           # averaged
    ignition_windows=[(290,310),(580,600)],
    baseline_windows=[(0,290),(325,580)],
    time_col='Timestamp',
    out_dir='exports_attractor/S01',
    show=True
)
print(res['summary_df'])

Files saved to out_dir:
  • attractor_3D_[state].png
  • corr_dimension_[state].png
  • lyapunov_[state].png
  • recurrence_[state].png
  • (if available) persistence_[state].png
  • summary.csv + summary.txt
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from scipy import signal
from numpy.linalg import norm

# register the 3D projection with Matplotlib
try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    _HAS_MPL_3D = True
except Exception:
    _HAS_MPL_3D = False


# Optional: scikit-learn neighbors
try:
    from sklearn.neighbors import KDTree
    _HAS_SK = True
except Exception:
    _HAS_SK = False

# Optional: ripser (persistent homology)
try:
    from ripser import ripser
    from persim import plot_diagrams
    _HAS_RIPSER = True
except Exception:
    _HAS_RIPSER = False


# ------------------------- I/O + preproc helpers -------------------------

def _ensure_dir(d): os.makedirs(d, exist_ok=True); return d

def infer_fs(RECORDS: pd.DataFrame, time_col: str = 'Timestamp') -> float:
    t = np.asarray(pd.to_numeric(RECORDS[time_col], errors='coerce').values, float)
    dt = np.diff(t); dt = dt[(dt > 0) & np.isfinite(dt)]
    if dt.size == 0:
        raise ValueError("Cannot infer sampling rate from time column.")
    return float(1.0 / np.median(dt))

def get_series(RECORDS: pd.DataFrame, name: str) -> np.ndarray:
    if name in RECORDS.columns:
        x = pd.to_numeric(RECORDS[name], errors='coerce').fillna(0.0).values
        return np.asarray(x, float)
    alt = 'EEG.' + name
    if alt in RECORDS.columns:
        x = pd.to_numeric(RECORDS[alt], errors='coerce').fillna(0.0).values
        return np.asarray(x, float)
    raise ValueError(f"Signal '{name}' not found in RECORDS.")

def slice_concat(x: np.ndarray, fs: float, windows: Optional[List[Tuple[float,float]]]) -> np.ndarray:
    if not windows: return x.copy()
    segs=[]; n=len(x)
    for (t0,t1) in windows:
        i0,i1 = int(round(t0*fs)), int(round(t1*fs))
        i0=max(0,i0); i1=min(n,i1)
        if i1>i0: segs.append(x[i0:i1])
    return np.concatenate(segs) if segs else x.copy()

def zscore(x): x = np.asarray(x,float); return (x - np.mean(x)) / (np.std(x)+1e-12)


def detect_time_col(RECORDS: pd.DataFrame,
                    candidates=('Timestamp','Time','time','t','seconds','sec','ms','datetime','DateTime','Datetime')) -> str | None:
    # 1) direct match
    for c in candidates:
        if c in RECORDS.columns:
            return c
    # 2) first numeric, roughly monotonic column
    for c in RECORDS.columns:
        s = pd.to_numeric(RECORDS[c], errors='coerce')
        if s.notna().sum() > max(50, 0.5*len(RECORDS)):
            arr = s.values.astype(float)
            dt = np.diff(arr[np.isfinite(arr)])
            if dt.size and np.nanmedian(dt) > 0:
                return c
    # 3) first datetime-like column
    for c in RECORDS.columns:
        try:
            _ = pd.to_datetime(RECORDS[c], errors='raise')
            return c
        except Exception:
            continue
    return None

def ensure_timestamp_column(RECORDS: pd.DataFrame,
                            time_col: str | None = None,
                            default_fs: float = 128.0,
                            out_name: str = 'Timestamp') -> str:
    """
    Ensure RECORDS[out_name] exists as numeric seconds (t=0 at first sample).
    Returns the column name used ('Timestamp' by default).
    """
    col = time_col or detect_time_col(RECORDS)
    if col is None:
        # synthesize uniform time if none exists
        N = len(RECORDS)
        RECORDS[out_name] = np.arange(N, dtype=float) / float(default_fs)
        return out_name

    s = RECORDS[col]
    # datetime -> seconds since first
    if np.issubdtype(s.dtype, np.datetime64) or 'datetime' in str(s.dtype).lower():
        tsec = (pd.to_datetime(s) - pd.to_datetime(s).iloc[0]).dt.total_seconds().astype(float)
        RECORDS[out_name] = tsec.values
        return out_name

    # numeric-ish -> coerce & shift to start at 0
    sn = pd.to_numeric(s, errors='coerce').astype(float)
    if sn.notna().sum() < max(50, 0.5*len(RECORDS)):
        N = len(RECORDS)
        RECORDS[out_name] = np.arange(N, dtype=float) / float(default_fs)
        return out_name

    sn = sn - np.nanmin(sn[np.isfinite(sn)])
    RECORDS[out_name] = sn.values
    return out_name


# ------------------------- Delay & embedding tools -------------------------

def estimate_delay_tau(x: np.ndarray, fs: float, max_lag_sec: float = 2.0, method: str = 'acf-1e') -> int:
    """
    Pick τ from the first time where autocorrelation falls below 1/e (default) or crosses 0.
    """
    nlag = int(max(1, round(max_lag_sec*fs)))
    x = zscore(x)
    acf = signal.correlate(x, x, mode='full')
    acf = acf[acf.size//2:acf.size//2+nlag+1]
    acf = acf / (acf[0] + 1e-12)
    if method == 'zero':
        idx = np.where(np.sign(acf[1:]) != np.sign(acf[:-1]))[0]
        tau = int(idx[0]+1) if idx.size else max(1, int(0.05*fs))
    else:  # 'acf-1e'
        idx = np.where(acf <= 1/np.e)[0]
        tau = int(idx[0]) if idx.size else max(1, int(0.05*fs))
    return max(1, tau)

def takens_embedding(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    """
    Return (N_eff, m) embedded matrix: [x_t, x_{t+τ}, ..., x_{t+(m-1)τ}]
    """
    N = len(x) - (m-1)*tau
    if N <= 10:
        raise ValueError("Time series too short for requested embedding.")
    Y = np.column_stack([x[i:i+N] for i in range(0, m*tau, tau)]).astype(float)
    return Y

def false_nearest_neighbors(x: np.ndarray, tau: int, m_list: List[int],
                            theiler: int = 10) -> pd.DataFrame:
    """
    Simple FNN percentage vs m. If sklearn is present, uses KDTree; else brute force sample.
    """
    rows=[]
    for m in m_list:
        try:
            X_m = takens_embedding(x, m, tau)       # (N, m)
            X_m1= takens_embedding(x, m+1, tau)     # (N', m+1) with N' slightly smaller
            N = min(len(X_m), len(X_m1))
            X_m  = X_m[:N]; X_m1 = X_m1[:N]
        except Exception:
            rows.append({'m': m, 'FNN%': np.nan}); continue

        if _HAS_SK:
            tree = KDTree(X_m)
            # 1-NN with Theiler window
            dist, idx = tree.query(X_m, k=2)
            nn = idx[:,1]
            # apply Theiler: replace neighbors within theiler samples by next best
            for i in range(N):
                if abs(nn[i]-i) <= theiler:
                    # find next NN not within Theiler window
                    dists, idxs = tree.query(X_m[i:i+1], k=10)
                    for cand in idxs[0,1:]:
                        if abs(cand - i) > theiler:
                            nn[i] = cand; break
        else:
            # fallback: naïve nearest neighbor (sampled)
            nn = np.zeros(N, dtype=int)
            for i in range(N):
                j = np.argmin(np.where(np.arange(N)==i, np.inf, norm(X_m - X_m[i], axis=1)))
                if abs(j-i) <= theiler:
                    # pick next best
                    d = norm(X_m - X_m[i], axis=1)
                    d[i] = np.inf
                    order = np.argsort(d)
                    for cand in order:
                        if abs(cand - i) > theiler:
                            j = cand; break
                nn[i] = j

        # FNN criterion (Kennel et al.): ratio of neighbor distance in m+1 vs m exceeding threshold
        Rtol = 15.0  # typical 10–15
        dist_m  = norm(X_m  - X_m[nn],  axis=1)
        dist_m1 = norm(X_m1 - X_m1[nn], axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = dist_m1 / (dist_m + 1e-12)
        fnn = np.mean(ratio > Rtol) * 100.0
        rows.append({'m': m, 'FNN%': float(fnn)})
    return pd.DataFrame(rows)

# ------------------------- Fractal dimension (GP) -------------------------

def correlation_dimension_gp(X: np.ndarray,
                             r_min_quant: float = 0.05,
                             r_max_quant: float = 0.30,
                             n_r: int = 20,
                             max_pairs: int = 30000) -> Dict[str, object]:
    """
    Grassberger-Procaccia correlation sum C(r) and slope (D2) over a mid-range.
    Subsamples pairs to limit O(N^2).
    """
    N = len(X)
    # pairwise distances on subsample
    idx = np.random.choice(N, size=min(N, 1000), replace=False)
    D = np.sqrt(((X[idx,None,:] - X[None,idx,:])**2).sum(axis=2)).ravel()
    D = D[D>0]
    Dsorted = np.sort(D)
    rmin = Dsorted[int(r_min_quant*len(Dsorted))]
    rmax = Dsorted[int(r_max_quant*len(Dsorted))]
    r_vals = np.exp(np.linspace(np.log(rmin+1e-12), np.log(rmax+1e-12), n_r))
    # correlation sum C(r) ~ fraction of pairs with distance < r
    if len(D) > max_pairs:
        D = np.random.choice(D, size=max_pairs, replace=False)
    C = np.array([np.mean(D < r) for r in r_vals])
    # slope via linear fit on log-log
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.log(r_vals + 1e-24); y = np.log(C + 1e-24)
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return {'r': r_vals, 'C': C, 'D2': float(slope), 'fit': (float(slope), float(intercept))}

# ------------------------- Largest Lyapunov (Rosenstein) -------------------------

def lyapunov_rosenstein(x: np.ndarray,
                        m: int,
                        tau: int,
                        fs: float,
                        theiler: int = 10,
                        t_fit: Tuple[int,int] = (1, 30)) -> Dict[str, object]:
    """
    Largest Lyapunov exponent (Rosenstein et al.).
    Returns {'lambda': λ_max (1/s), 'L': divergence curve, 'k': lead steps}.
    """
    # Build embedding
    X = takens_embedding(zscore(x), m, tau)  # (N, m)
    N = len(X)
    if N <= theiler + 2:
        return {'lambda': np.nan, 'L': np.array([]), 'k': np.array([])}

    # Nearest neighbor indices with Theiler exclusion
    if _HAS_SK:
        tree = KDTree(X)
        # pull more than 2 neighbors to have replacements if Theiler excludes
        dists, idxs = tree.query(X, k=min(20, N-1))
        nn = np.zeros(N, dtype=int)
        for i in range(N):
            # skip self at idxs[i,0]; find first neighbor beyond Theiler window
            chosen = None
            for cand in idxs[i, 1:]:
                if abs(int(cand) - i) > theiler:
                    chosen = int(cand); break
            nn[i] = chosen if chosen is not None else int(idxs[i,1])
    else:
        nn = np.zeros(N, dtype=int)
        for i in range(N):
            d = np.linalg.norm(X - X[i], axis=1)
            d[i] = np.inf
            order = np.argsort(d)
            chosen = None
            for cand in order:
                if abs(int(cand) - i) > theiler:
                    chosen = int(cand); break
            nn[i] = chosen if chosen is not None else int(order[0])

    # Mean log divergence over lead steps k
    max_k = max(2, min(100, N-1))
    Lvals = []
    ks = []
    for k in range(1, max_k):
        # only indices where both i+k and nn[i]+k are valid
        valid = (np.arange(N) + k < N) & (nn + k < N)
        if not np.any(valid):
            break
        idx = np.where(valid)[0]
        if idx.size < 5:  # too few pairs for a stable average
            break
        d0 = np.linalg.norm(X[idx]       - X[nn[idx]],       axis=1) + 1e-24
        dk = np.linalg.norm(X[idx + k]   - X[nn[idx] + k],   axis=1)
        Lvals.append(np.mean(np.log(dk / d0)))
        ks.append(k)

    Lvals = np.asarray(Lvals, float)
    ks = np.asarray(ks, int)
    if Lvals.size < 5:
        return {'lambda': np.nan, 'L': Lvals, 'k': ks}

    # Linear fit region (auto-cap to available ks)
    k0, k1 = t_fit
    k1 = min(k1, int(0.6*len(Lvals)))  # avoid late saturation, keep early linear regime
    if k1 <= k0+1:
        return {'lambda': np.nan, 'L': Lvals, 'k': ks}

    A = np.vstack([ks[k0:k1], np.ones(k1-k0)]).T
    slope, intercept = np.linalg.lstsq(A, Lvals[k0:k1], rcond=None)[0]
    # Convert per-embedded-step slope to per-second λ: one embedded step = tau samples
    lam = float(slope) * fs / float(tau)
    return {'lambda': lam, 'L': Lvals, 'k': ks}

# ------------------------- Recurrence plot + simple RQA -------------------------

def recurrence_plot(X: np.ndarray, eps_quant: float = 0.1) -> Dict[str, object]:
    """
    Binary RP thresholded at eps = quantile(eps_quant) of distances.
    Simple RQA: Recurrence Rate (RR), Determinism (DET) via diagonal line counts ≥2.
    """
    N = len(X)
    # distance matrix on subsample for efficiency
    idx = np.random.choice(N, size=min(N, 1200), replace=False)
    Y = X[idx]
    D = np.sqrt(((Y[:,None,:]-Y[None,:,:])**2).sum(axis=2))
    eps = np.quantile(D, eps_quant)
    R = (D <= eps).astype(int)
    np.fill_diagonal(R, 0)
    RR = np.mean(R)
    # DET: crude diagonal line detector
    det_lines=0; total_lines=0
    for i in range(R.shape[0]-1):
        run=0
        for j in range(R.shape[1]-1):
            if R[i,j]==1 and R[i+1,j+1]==1:
                run+=1
            else:
                if run>=1:
                    total_lines+=1
                    if run+1>=2: det_lines+=1
                run=0
        if run>=1:
            total_lines+=1
            if run+1>=2: det_lines+=1
    DET = det_lines / (total_lines + 1e-12)
    return {'R': R, 'RR': float(RR), 'DET': float(DET), 'eps': float(eps)}

# ------------------------- Persistent homology (optional) -------------------------

def persistent_homology_summary(X: np.ndarray, maxdim: int = 2) -> Dict[str, object]:
    """
    If ripser is installed, compute persistence and return diagrams + simple counts.
    """
    if not _HAS_RIPSER:
        return {'available': False}
    # subsample to keep compute reasonable
    N = len(X)
    idx = np.random.choice(N, size=min(N, 800), replace=False)
    Y = X[idx]
    dgms = ripser(Y, maxdim=maxdim)['dgms']
    # simple summaries: count H0/H1/H2 bars above small persistence
    def count_persistent(dgm, thr=0.02):
        return int(np.sum((dgm[:,1]-dgm[:,0]) > thr))
    summ = {
        'H0_count': count_persistent(dgms[0], thr=0.0),
        'H1_count': count_persistent(dgms[1]) if len(dgms)>1 else 0,
        'H2_count': count_persistent(dgms[2]) if len(dgms)>2 else 0
    }
    return {'available': True, 'dgms': dgms, 'summary': summ}

# ------------------------- Surrogates -------------------------

def phase_randomize(x: np.ndarray) -> np.ndarray:
    X = np.fft.rfft(x)
    mag = np.abs(X)
    ph  = np.angle(X)
    k = len(ph)
    rand = np.random.uniform(-np.pi, np.pi, size=k)
    rand[0] = ph[0]
    if k % 2 == 0:
        rand[-1] = ph[-1]
    Xs = mag * np.exp(1j*rand)
    return np.fft.irfft(Xs, n=len(x)).astype(float)

def time_shuffle(x: np.ndarray) -> np.ndarray:
    return np.random.permutation(x)

def metric_vs_surrogates(metric_func, x: np.ndarray, n_surr: int = 100, kind: str = 'phase') -> Tuple[float, float]:
    """
    Compute metric on x, build null from surrogates (phase or shuffle). Return (value, p-value).
    """
    val = metric_func(x)
    null=[]
    for _ in range(n_surr):
        xs = phase_randomize(x) if kind=='phase' else time_shuffle(x)
        null.append(metric_func(xs))
    null = np.asarray(null, float)
    p = (np.sum(null >= val) + 1) / (n_surr + 1)
    return float(val), float(p)

# ------------------------- Top-level runner -------------------------

def run_attractor_topology(RECORDS: pd.DataFrame,
                           eeg_channels: List[str],
                           ignition_windows: Optional[List[Tuple[float,float]]],
                           baseline_windows: Optional[List[Tuple[float,float]]],
                           time_col: str = 'Timestamp',
                           out_dir: str = 'exports_attractor/session',
                           show: bool = True,
                           max_lag_sec: float = 2.0,
                           m_list: List[int] = [2,3,4,5,6,7,8],
                           n_surrogates: int = 100) -> Dict[str, object]:
    """
    Build attractor embeddings and tests for Ignition and Baseline.
    Returns dict with summary DataFrame and paths.
    """
    _ensure_dir(out_dir)

    # >>> NEW: make sure a valid numeric-seconds column exists <<<
    time_col = ensure_timestamp_column(RECORDS, time_col=time_col, default_fs=128.0)
    # -------------------------------------------------------------

    fs = infer_fs(RECORDS, time_col)

    # robust drive signal (mean of channels)
    Xsig=[]
    for ch in eeg_channels:
        Xsig.append(get_series(RECORDS, ch))
    Xsig = np.vstack(Xsig)
    x_full = zscore(np.mean(Xsig, axis=0))
    # per state
    states = {'ignition': ignition_windows, 'baseline': baseline_windows}
    summaries=[]
    outputs={}
    for state, wins in states.items():
        if not wins: continue
        x = slice_concat(x_full, fs, wins)
        x = zscore(x)
        # τ and m
        tau = estimate_delay_tau(x, fs, max_lag_sec=max_lag_sec, method='acf-1e')
        fnn_df = false_nearest_neighbors(x, tau, m_list)
        # choose m* at elbow (min m where FNN% < 5% or lowest)
        fnn_df = fnn_df.dropna()
        m_star = int(fnn_df.loc[fnn_df['FNN%'].le(5.0).idxmax(),'m']) if np.any(fnn_df['FNN%']<=5.0) else int(fnn_df['m'].iloc[np.argmin(fnn_df['FNN%'])])
        m_star = max(3, m_star)

        # embed
        X = takens_embedding(x, m_star, tau)    # (N,m)
        # (A) D2
        D2 = correlation_dimension_gp(X)['D2']
        # (B) λ_max
        lyap = lyapunov_rosenstein(x, m_star, tau, fs, theiler=int(0.5*fs/tau), t_fit=(1, 30))
        lam = lyap['lambda']
        # (C) RP + RQA
        rqa = recurrence_plot(X, eps_quant=0.1)
        # (D) persistent homology (optional)
        ph = persistent_homology_summary(X)

        # surrogate tests on D2 and λ_max
        d2_val, d2_p = metric_vs_surrogates(lambda xs: correlation_dimension_gp(takens_embedding(zscore(xs), m_star, tau))['D2'],
                                            x, n_surr=n_surrogates, kind='phase')
        lam_val, lam_p = metric_vs_surrogates(lambda xs: lyapunov_rosenstein(zscore(xs), m_star, tau, fs)['lambda'],
                                              x, n_surr=n_surrogates, kind='phase')

        summaries.append({'state':state, 'tau':tau, 'm':m_star,
                          'D2':float(D2), 'D2_p':float(d2_p),
                          'lambda':float(lam), 'lambda_p':float(lam_p),
                          'RR':rqa['RR'], 'DET':rqa['DET'],
                          'PH_available': ph['available']})

        # ===== Plots =====
        # 3D attractor
        # 3D (or 2D fallback) attractor plot
        stride = max(1, len(X)//8000)  # keep plots light; adjust as you like

        if _HAS_MPL_3D and X.shape[1] >= 3:
            fig = plt.figure(figsize=(5,4))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(X[::stride,0], X[::stride,1], X[::stride,2], lw=0.6, alpha=0.8)
            ax.set_title(f'Attractor (m={m_star}, τ={tau}) — {state}')
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, f'attractor_3D_{state}.png'), dpi=140)
            if show: plt.show()
            plt.close(fig)
        else:
            # 2D fallback (pairwise)
            fig, axs = plt.subplots(1, 2, figsize=(8,3.2))
            axs[0].plot(X[::stride,0], X[::stride,1], lw=0.6, alpha=0.8)
            axs[0].set_title(f'X1 vs X2 — {state}')
            if X.shape[1] >= 3:
                axs[1].plot(X[::stride,1], X[::stride,2], lw=0.6, alpha=0.8)
                axs[1].set_title(f'X2 vs X3 — {state}')
            else:
                axs[1].plot(X[::stride,0], X[::stride,0], lw=0.6, alpha=0.3)  # dummy if m<3
                axs[1].set_title('2D fallback')
            for ax in axs: ax.set_xlabel(''); ax.set_ylabel('')
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, f'attractor_2D_{state}.png'), dpi=140)
            if show: plt.show()
            plt.close(fig)


        # Correlation dimension log-log
        gp = correlation_dimension_gp(X)
        fig = plt.figure(figsize=(5,3))
        plt.plot(np.log(gp['r']+1e-24), np.log(gp['C']+1e-24), 'o-', lw=1)
        s, b = gp['fit']
        plt.plot(np.log(gp['r']+1e-24), s*np.log(gp['r']+1e-24)+b, 'r--', lw=1)
        plt.title(f'Correlation sum (D2≈{gp["D2"]:.2f}) — {state}')
        plt.xlabel('log r'); plt.ylabel('log C(r)'); plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f'corr_dimension_{state}.png'), dpi=140)
        if show: plt.show()
        plt.close(fig)

        # Lyapunov curve
        fig = plt.figure(figsize=(5,3))
        plt.plot(lyap['k']/fs*tau, lyap['L'], lw=1.2)
        plt.title(f'Lyapunov divergence (λ≈{lam:.3f} 1/s) — {state}')
        plt.xlabel('Time (s)'); plt.ylabel('⟨log(d_k/d_0)⟩'); plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f'lyapunov_{state}.png'), dpi=140)
        if show: plt.show()
        plt.close(fig)

        # Recurrence plot
        R = rqa['R']
        fig = plt.figure(figsize=(4,4))
        plt.imshow(R, origin='lower', cmap='binary')
        plt.title(f'Recurrence plot — {state}\nRR={rqa["RR"]:.3f}, DET={rqa["DET"]:.3f}')
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f'recurrence_{state}.png'), dpi=140)
        if show: plt.show()
        plt.close(fig)

        # Persistent homology diagram (optional)
        if ph['available']:
            fig = plt.figure(figsize=(4,3))
            plot_diagrams(ph['dgms'])
            plt.title(f'Persistence diagrams — {state}')
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, f'persistence_{state}.png'), dpi=140)
            if show: plt.show()
            plt.close(fig)

        outputs[state] = {'fnn':fnn_df, 'D2':D2, 'lambda':lam, 'rqa':rqa, 'ph':ph}

    # ===== Summary table & save =====
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(os.path.join(out_dir, 'summary.csv'), index=False)
    with open(os.path.join(out_dir, 'summary.txt'),'w') as f:
        f.write(summary_df.to_string(index=False))

    return {'summary_df': summary_df, 'outputs': outputs, 'out_dir': out_dir}

"""
Entanglement Entropy Analogs & Integrative Information — Simple Graphs & Tests
=============================================================================

This module computes classical analogs of “entanglement/integration” for EEG:

  (A) Multichannel information & complexity (Gaussian & algorithmic):
      • Total Correlation (TC) (a.k.a. multi-information; Gaussian)
      • Dual Total Correlation (DTC) (Gaussian)
      • O-information (O = TC − DTC), redundancy/synergy indicator (Gaussian)
      • Entropy h(X) (Gaussian, z-scored)
      • Lempel–Ziv complexity (LZc) of a global binary sequence (PCA1→binarize)
      • Permutation entropy (PE) (order=3) averaged across channels

  (B) Network integration via connectivity:
      • PLV adjacency inside a band (e.g., alpha)
      • Laplacian spectral entropy (graph) as an integration/complexity index

  (C) Ignition vs Baseline comparisons + Surrogate tests:
      • Circular-shift surrogates → 95% null bands and p-values
      • Simple bar/heatmap plots, saved to out_dir

  (D) Time-resolved coupling to Schumann amplitude:
      • Sliding-window integration score vs SR envelope (7.83±0.6 Hz)
      • Correlation coefficient r and a quick null via circular shift

Inputs:
  - RECORDS: pandas.DataFrame with a time column (default 'Timestamp') and EEG.* columns
  - eeg_channels: list of EEG.* channel names (or bare labels like 'O1')

Usage:
------
res = run_integration_analogs(
    RECORDS,
    eeg_channels=['EEG.O1','EEG.O2','EEG.Oz','EEG.Pz'],
    band=(8,13),
    ignition_windows=[(290,310),(580,600)],
    baseline_windows=[(0,290),(325,580)],
    sr_channel='EEG.Oz',                 # if None, picks a posterior channel automatically
    time_col='Timestamp',
    out_dir='exports_integration/S01',
    show=True
)
print(res['summary'])   # per-state metrics and p-values
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from scipy import signal
from numpy.linalg import det, inv

# ---------------------- small utilities ----------------------

def _ensure_dir(d): os.makedirs(d, exist_ok=True); return d

def infer_fs(RECORDS: pd.DataFrame, time_col: str = 'Timestamp') -> float:
    t = np.asarray(pd.to_numeric(RECORDS[time_col], errors='coerce').values, float)
    dt = np.diff(t); dt = dt[(dt > 0) & np.isfinite(dt)]
    if dt.size == 0: raise ValueError("Cannot infer sampling rate from time column.")
    return float(1.0 / np.median(dt))

def get_series(RECORDS: pd.DataFrame, name: str) -> np.ndarray:
    if name in RECORDS.columns:
        x = pd.to_numeric(RECORDS[name], errors='coerce').fillna(0.0).values
        return np.asarray(x, float)
    alt = 'EEG.'+name
    if alt in RECORDS.columns:
        x = pd.to_numeric(RECORDS[alt], errors='coerce').fillna(0.0).values
        return np.asarray(x, float)
    raise ValueError(f"Signal '{name}' not found.")

def slice_concat(x: np.ndarray, fs: float, windows: Optional[List[Tuple[float,float]]]) -> np.ndarray:
    if not windows: return x.copy()
    segs=[]; n=len(x)
    for (t0,t1) in windows:
        i0,i1 = int(round(t0*fs)), int(round(t1*fs))
        i0=max(0,i0); i1=min(n,i1)
        if i1>i0: segs.append(x[i0:i1])
    return np.concatenate(segs) if segs else x.copy()

def bandpass(x: np.ndarray, fs: float, f1: float, f2: float, order=4) -> np.ndarray:
    ny = 0.5*fs
    f1 = max(1e-6, min(f1, 0.99*ny)); f2 = max(f1+1e-6, min(f2, 0.999*ny))
    b,a = signal.butter(order, [f1/ny, f2/ny], btype='band')
    return signal.filtfilt(b,a,x)

def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x,float)
    return (x - np.mean(x)) / (np.std(x)+1e-12)

# ---------------------- PLV & graph metrics ----------------------

def plv_matrix(RECORDS, channels, band, windows, time_col='Timestamp') -> np.ndarray:
    fs = infer_fs(RECORDS, time_col)
    phases=[]
    for ch in channels:
        x = slice_concat(get_series(RECORDS, ch), fs, windows)
        xb = bandpass(x, fs, band[0], band[1])
        phases.append(np.angle(signal.hilbert(xb)))
    P = np.vstack(phases)  # (N, T)
    N = P.shape[0]
    A = np.zeros((N,N))
    for i in range(N):
        for j in range(i, N):
            dphi = P[i]-P[j]
            A[i,j]=A[j,i]=float(np.abs(np.mean(np.exp(1j*dphi))))
    np.fill_diagonal(A, 0.0)
    return A

def laplacian_spectral_entropy(A: np.ndarray) -> float:
    D = np.diag(A.sum(axis=1))
    L = D - A
    L = 0.5*(L+L.T)
    vals = np.linalg.eigvalsh(L)
    vals = vals[vals>1e-12]
    if vals.size == 0: return np.nan
    p = vals/np.sum(vals)
    return float(-np.sum(p*np.log(p)))

# ---------------------- Gaussian information measures ----------------------

def gaussian_entropies(X: np.ndarray) -> Dict[str, float]:
    """
    X: (n_ch, T) z-scored
    Gaussian differential entropies:
      h(X) = 0.5 * [ n ln(2πe) + ln det Σ ]
      TC   = Σ h(X_i) − h(X)
      DTC  = h(X) − Σ h(X_i | X_{-i})  with  h(X_i | X_{-i}) = 0.5 ln(2πe σ^2_{i|-i})
      O    = TC − DTC
    """
    n, T = X.shape
    # covariance
    Sigma = np.cov(X)
    # add tiny ridge for stability
    Sigma = 0.5*(Sigma+Sigma.T) + 1e-9*np.eye(n)
    # entropies
    hXi = 0.5*(np.log(2*np.pi*np.e)*np.ones(n) + np.log(np.diag(Sigma)+1e-24))
    hX  = 0.5*(n*np.log(2*np.pi*np.e) + np.log(det(Sigma)+1e-24))
    TC  = float(np.sum(hXi) - hX)
    # conditional variances via precision
    Prec = inv(Sigma)
    cond_vars = 1.0 / np.diag(Prec)
    hXi_cond = 0.5*(np.log(2*np.pi*np.e) + np.log(cond_vars + 1e-24))
    DTC = float(hX - np.sum(hXi_cond))
    Oinfo = float(TC - DTC)
    return {'hX': float(hX), 'TC': TC, 'DTC': DTC, 'O': Oinfo}

# ---------------------- Algorithmic & ordinal complexities ----------------------

def pca_first_component(X: np.ndarray) -> np.ndarray:
    # X: (n_ch, T)
    Xc = X - X.mean(axis=1, keepdims=True)
    C = Xc @ Xc.T / Xc.shape[1]
    vals, vecs = np.linalg.eigh(C)
    v = vecs[:, -1]            # first PC (eigenvector)
    y = v @ Xc                 # PC1 time series
    return np.asarray(y).ravel()

def lz_complexity_binary(seq: np.ndarray) -> float:
    """
    LZ76 complexity of a binary sequence (0/1), normalized by n/log2(n).
    """
    s = ''.join('1' if v else '0' for v in (seq>0))
    n = len(s)
    i = 0; k = 1; l = 1; c = 1
    while True:
        if s[i+k-1] == s[l+k-1]:
            k += 1
            if l+k > n:
                c += 1; break
        else:
            if k > 1:
                i += 1
                if i == l:
                    c += 1
                    l += k
                    if l+1 > n:
                        break
                    i = 0; k = 1
            else:
                c += 1
                l += 1
                if l+1 > n:
                    break
                i = 0; k = 1
    norm = n/np.log2(max(2,n))
    return float(c / norm)

def permutation_entropy(x: np.ndarray, m: int = 3, tau: int = 1) -> float:
    """
    Band-limited x; simple permutation entropy of order m (permutation count m! bins).
    """
    x = np.asarray(x, float)
    T = len(x) - (m-1)*tau
    if T <= m: return np.nan
    patterns = {}
    for i in range(T):
        w = x[i:i+m*tau:tau]
        perm = tuple(np.argsort(w))
        patterns[perm] = patterns.get(perm, 0) + 1
    p = np.array(list(patterns.values()), float)
    p = p / p.sum()
    H = -np.sum(p*np.log(p+1e-24)) / np.log(np.math.factorial(m))
    return float(H)

# ---------------------- Surrogates ----------------------

def circular_shift_null(xmat: np.ndarray, n_surr: int = 200) -> List[np.ndarray]:
    """
    Circularly shift each channel independently; returns list of surrogates (n_ch, T).
    """
    n, T = xmat.shape
    rng = np.random.default_rng(11)
    sur=[]
    for _ in range(n_surr):
        Xs = []
        for i in range(n):
            s = int(rng.integers(1, T-1))
            Xs.append(np.r_[xmat[i,-s:], xmat[i,:-s]])
        sur.append(np.vstack(Xs))
    return sur

# ---------------------- Main runner ----------------------

def run_integration_analogs(
    RECORDS: pd.DataFrame,
    eeg_channels: List[str],
    band: Tuple[float,float] = (8,13),
    ignition_windows: Optional[List[Tuple[float,float]]] = None,
    baseline_windows: Optional[List[Tuple[float,float]]] = None,
    sr_channel: Optional[str] = None,
    time_col: str = 'Timestamp',
    out_dir: str = 'exports_integration/session',
    show: bool = True,
    n_surr: int = 200
) -> Dict[str, object]:
    """
    Compute integration/complexity measures and simple tests; produce figures + CSV summary.
    """
    _ensure_dir(out_dir)
    fs = infer_fs(RECORDS, time_col)
    # choose SR if not provided
    if sr_channel is None:
        # simple PSD preference: pick Oz if exists, else first EEG.*
        sr_channel = 'EEG.Oz' if 'EEG.Oz' in RECORDS.columns else next((c for c in RECORDS.columns if c.startswith('EEG.')), None)
    # build X (n_ch, T) for both states
    Xall=[]
    for ch in eeg_channels:
        Xall.append(get_series(RECORDS, ch))
    Xall = np.vstack(Xall)      # (n_ch, T)
    states = {'ignition': ignition_windows, 'baseline': baseline_windows}

    def compute_state(wins, state_name):
        if not wins: return None
        X = np.vstack([slice_concat(x, fs, wins) for x in Xall])    # (n_ch, Tstate)
        # z-score each channel
        Xz = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True)+1e-12)
        # Gaussian integration measures
        g = gaussian_entropies(Xz.copy())
        # graph via PLV
        A = plv_matrix(RECORDS, eeg_channels, band, wins, time_col=time_col)
        H_L = laplacian_spectral_entropy(A)
        # LZc on PCA1
        pc1 = pca_first_component(Xz)
        # binarize by median
        lzc = lz_complexity_binary(pc1 - np.median(pc1))
        # perm entropy averaged across channels (in-band)
        pe_ch=[]
        for i in range(X.shape[0]):
            xb = bandpass(X[i], fs, band[0], band[1])
            pe_ch.append(permutation_entropy(xb, m=3, tau=1))
        PE = float(np.nanmean(pe_ch))
        return {'X':Xz, 'gauss':g, 'A':A, 'H_L':H_L, 'LZc':float(lzc), 'PE':PE}

    results={}
    for name, wins in states.items():
        res = compute_state(wins, name)
        if res: results[name]=res

    # ------------- Surrogates (ignition) -------------
    surr_pvals = {}
    if 'ignition' in results:
        X = results['ignition']['X']
        sur = circular_shift_null(X, n_surr=n_surr)
        # metrics on surrogates
        g_TC=[]; g_DTC=[]; g_O=[]; H_L_s=[]; LZ_s=[]; PE_s=[]
        for Xs in sur:
            g = gaussian_entropies(Xs)
            g_TC.append(g['TC']); g_DTC.append(g['DTC']); g_O.append(g['O'])
            # PLV graph on surrogate: scramble all channels jointly (consistent shifts)
            # Build PLV on surrogate in same band
            # (We simulate PLV null by randomizing phase via Hilbert-stage circular shift)
            # for speed, use original A as reference; here re-compute with Xs projected back to signals:
            # approximate by recomputing PLV on Xs via direct phase extraction:
            # (convert Xs to signal-like by inverse z-score to amplitude 1; this is a coarse null)
            phases = np.angle(signal.hilbert(np.array([bandpass(x, fs, band[0], band[1]) for x in Xs])))
            N = phases.shape[0]
            A = np.zeros((N,N))
            for i in range(N):
                for j in range(i,N):
                    dphi = phases[i]-phases[j]
                    A[i,j]=A[j,i]=float(np.abs(np.mean(np.exp(1j*dphi))))
            H_L_s.append(laplacian_spectral_entropy(A))
            pc1 = pca_first_component(Xs)
            LZ_s.append(lz_complexity_binary(pc1 - np.median(pc1)))
            PE_s.append(np.nanmean([permutation_entropy(bandpass(x, fs, band[0], band[1])) for x in Xs]))
        def pval(obs, null):
            null = np.asarray(null, float)
            return float((np.sum(null >= obs)+1)/(len(null)+1))
        surr_pvals = {
            'TC_p':  pval(results['ignition']['gauss']['TC'],  g_TC),
            'DTC_p': pval(results['ignition']['gauss']['DTC'], g_DTC),
            'O_p':   pval(results['ignition']['gauss']['O'],   g_O),
            'H_L_p': pval(results['ignition']['H_L'],         H_L_s),
            'LZc_p': pval(results['ignition']['LZc'],         LZ_s),
            'PE_p':  pval(results['ignition']['PE'],          PE_s)
        }

    # ------------- Simple plots -------------
    # adjacency heatmaps
    for st in results:
        fig,ax = plt.subplots(1,2, figsize=(8,3))
        im = ax[0].imshow(results[st]['A'], vmin=0, vmax=1, cmap='viridis')
        ax[0].set_title(f'PLV adjacency ({st})')
        plt.colorbar(im, ax=ax[0], fraction=0.046)
        # Gaussian info bar
        g = results[st]['gauss']
        names = ['hX','TC','DTC','O','H_L','LZc','PE']
        vals = [g['hX'], g['TC'], g['DTC'], g['O'], results[st]['H_L'], results[st]['LZc'], results[st]['PE']]
        ax[1].bar(range(len(names)), vals, color='tab:blue', alpha=0.85)
        ax[1].set_xticks(range(len(names))); ax[1].set_xticklabels(names, rotation=30)
        ax[1].set_title('Integration / Complexity')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'integration_{st}.png'), dpi=140)
        if show: plt.show()
        plt.close()

    # ignition vs baseline bars with null bands (if both present)
    if 'ignition' in results and 'baseline' in results:
        names = ['TC','DTC','O','H_L','LZc','PE']
        ign_vals = [results['ignition']['gauss']['TC'],
                    results['ignition']['gauss']['DTC'],
                    results['ignition']['gauss']['O'],
                    results['ignition']['H_L'],
                    results['ignition']['LZc'],
                    results['ignition']['PE']]
        base_vals= [results['baseline']['gauss']['TC'],
                    results['baseline']['gauss']['DTC'],
                    results['baseline']['gauss']['O'],
                    results['baseline']['H_L'],
                    results['baseline']['LZc'],
                    results['baseline']['PE']]
        x = np.arange(len(names)); w=0.38
        plt.figure(figsize=(8,3.2))
        plt.bar(x-w/2, base_vals, width=w, label='Baseline', color='tab:orange', alpha=0.9)
        plt.bar(x+w/2, ign_vals,  width=w, label='Ignition', color='tab:blue',  alpha=0.9)
        # add surrogate 95% lines for ignition (if available)
        if surr_pvals:
            # create simple line at 95th percentile of each surrogate null (we didn’t store whole null arrays; show p-values text instead)
            for i,name in enumerate(names):
                pkey = f'{name}_p' if f'{name}_p' in surr_pvals else None
                if pkey:
                    plt.text(i+w/2, ign_vals[i], f" p={surr_pvals[pkey]:.3f}", ha='center', va='bottom', fontsize=8)
        plt.xticks(x, names, rotation=0)
        plt.ylabel('Value'); plt.title('Ignition vs Baseline — Integration Indices'); plt.legend()
        plt.tight_layout(); plt.savefig(os.path.join(out_dir,'integration_ign_vs_base.png'), dpi=140)
        if show: plt.show()
        plt.close()

    # ------------- Time-resolved coupling to SR amplitude -------------
    sr = get_series(RECORDS, sr_channel)
    env = np.abs(signal.hilbert(bandpass(sr, fs, 7.83-0.6, 7.83+0.6)))
    # sliding window score: normalize and combine TC+H_L (zwise) as a simple integrative index
    # build sliding windows (2.0 s, 0.25 s step)
    win = int(round(2.0*fs)); step = int(round(0.25*fs))
    idxs = list(range(0, len(Xall[0])-win, step))
    score_ts = []
    for s in idxs:
        seg = Xall[:, s:s+win]
        seg = (seg - seg.mean(axis=1, keepdims=True)) / (seg.std(axis=1, keepdims=True)+1e-12)
        g = gaussian_entropies(seg)
        A = plv_matrix(RECORDS, eeg_channels, band, [(s/fs,(s+win)/fs)], time_col=time_col)
        H_L = laplacian_spectral_entropy(A)
        score_ts.append( 0.7*g['TC'] + 0.3*H_L )
    score_ts = np.asarray(score_ts)
    t_centers = (np.array(idxs)+win//2)/fs
    # resample env to centers
    env_c = np.interp(t_centers, np.arange(len(env))/fs, env)
    # correlation + circular-shift null
    r = np.corrcoef(score_ts, env_c)[0,1]
    rng = np.random.default_rng(5)
    null_r=[]
    for _ in range(200):
        s = int(rng.integers(1,len(env_c)-1))
        null_r.append(np.corrcoef(score_ts, np.r_[env_c[-s:], env_c[:-s]])[0,1])
    thr95 = np.nanpercentile(null_r, 95)

    plt.figure(figsize=(9,3))
    zsc = (score_ts - np.nanmean(score_ts))/ (np.nanstd(score_ts)+1e-12)
    ze  = (env_c - np.nanmean(env_c))/ (np.nanstd(env_c)+1e-12)
    plt.plot(t_centers, zsc, label='Integration score (z)')
    plt.plot(t_centers, ze,  label='SR envelope (z)')
    plt.title(f'Time-resolved integration vs SR envelope  (r={r:.2f}, null95={thr95:.2f})')
    plt.xlabel('Time (s)'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir,'integration_vs_sr_timeseries.png'), dpi=140)
    if show: plt.show()
    plt.close()

    # ------------- Summary & save -------------
    rows=[]
    for st, res in results.items():
        rows.append({
            'state': st,
            'hX': res['gauss']['hX'],
            'TC': res['gauss']['TC'],
            'DTC': res['gauss']['DTC'],
            'O': res['gauss']['O'],
            'H_L': res['H_L'],
            'LZc': res['LZc'],
            'PE': res['PE']
        })
    summary = pd.DataFrame(rows)
    if surr_pvals:
        p_row = {'state':'ignition_pvals'} | surr_pvals if hasattr(dict, '__or__') else dict(**{'state':'ignition_pvals'}, **surr_pvals)
        summary = pd.concat([summary, pd.DataFrame([p_row])], ignore_index=True)
    summary.to_csv(os.path.join(out_dir,'summary.csv'), index=False)

    return {'summary': summary,
            'results': results,
            'corr_env_r': float(r),
            'corr_env_null95': float(thr95),
            'out_dir': out_dir}

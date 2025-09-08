"""
EEG Microstate Segmentation — Simple Graphs & Validation
========================================================

What it does
------------
• Builds multi-channel EEG matrix X (n_ch × T) over given windows; z-scores and (optionally) band-passes.
• Finds GFP peaks and clusters their topographies (channels) with k-means (k∈{3,4,5,6}) → candidate microstate maps.
• Selects k by Global Explained Variance (GEV); backfits full sequence (polarity-invariant, argmax|corr|).
• Temporal smoothing (minimum segment duration) to avoid spurious flips.
• Metrics: GEV, mean duration (ms), coverage (%time), occurrence rate (per s),
           transition matrix (k×k), sequence Shannon entropy (bits).
• Surrogate validation: phase-randomize each channel and recompute GEV → 95% null and p-value.
• Outputs PNGs + summary.csv in out_dir.

Usage
-----
res = run_microstate_segmentation(
    RECORDS,
    eeg_channels=['EEG.O1','EEG.O2','EEG.P7','EEG.P8','EEG.FC5','EEG.FC6'],  # 6–12 clean channels recommended
    ignition_windows=[(290,310),(580,600)],
    baseline_windows=[(0,290),(325,580)],
    band=(2,40),                         # optional prefilter
    time_col='Timestamp',
    out_dir='exports_microstates/S01',
    show=False
)
print(res['summary'])
"""
from __future__ import annotations
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from scipy import signal
from sklearn.cluster import KMeans

# ---------------- I/O helpers ----------------
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

def zscore(x): x=np.asarray(x,float); return (x - np.mean(x)) / (np.std(x)+1e-12)

def bandpass(x, fs, f1, f2, order=4):
    ny=0.5*fs; f1=max(1e-6,min(f1,0.99*ny)); f2=max(f1+1e-6,min(f2,0.999*ny))
    b,a=signal.butter(order,[f1/ny,f2/ny],btype='band'); return signal.filtfilt(b,a,x)

# ---------------- Microstate core ----------------
def gfp(X: np.ndarray) -> np.ndarray:
    """Global Field Power across channels per time (std). X: (n_ch, T)"""
    return np.std(X, axis=0)

def pick_gfp_peaks(G: np.ndarray, skip: int) -> np.ndarray:
    """Pick timepoints at local maxima of GFP with a refractory 'skip' in samples."""
    peaks = []
    i = skip
    while i < len(G)-skip:
        win = G[i-skip:i+skip+1]
        if np.argmax(win) == skip:
            peaks.append(i)
            i += skip
        else:
            i += 1
    return np.array(peaks, int)

def normalize_maps(X: np.ndarray) -> np.ndarray:
    """Zero-mean & L2-normalize topographies columnwise. X: (n_ch, Nmaps)"""
    Xm = X - X.mean(axis=0, keepdims=True)
    denom = np.linalg.norm(Xm, axis=0, keepdims=True) + 1e-12
    return Xm/denom

def kmeans_microstates(Xmaps: np.ndarray, k: int, n_init: int = 20, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """KMeans on normalized maps (channels×N) → centers (channels×k), labels (N,)"""
    Z = Xmaps.T  # (N, n_ch)
    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
    labels = km.fit_predict(Z)
    centers = km.cluster_centers_.T
    # normalize centers the same way
    centers = centers - centers.mean(axis=0, keepdims=True)
    centers = centers / (np.linalg.norm(centers, axis=0, keepdims=True) + 1e-12)
    return centers, labels

def backfit_sequence(X: np.ndarray, centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign each timepoint to the map with max |corr| (polarity-invariant).
    Returns labels (T,) in [0..k-1] and per-time corr_abs (T,).
    """
    # normalize data topographies per timepoint
    Xn = X - X.mean(axis=0, keepdims=True)
    denom = np.linalg.norm(Xn, axis=0, keepdims=True)+1e-12
    Xn = Xn/denom
    # centers already normalized (n_ch×k)
    C = centers
    corr = Xn.T @ C                  # (T × k)
    corr_abs = np.abs(corr)          # polarity invariance
    lab = np.argmax(corr_abs, axis=1)
    maxcorr = corr_abs[np.arange(len(lab)), lab]
    return lab.astype(int), maxcorr

def smooth_labels(labels: np.ndarray, fs: float, min_dur_ms: float = 30.0) -> np.ndarray:
    """Enforce minimum segment duration by merging short runs into neighbors."""
    min_len = max(1, int(round(min_dur_ms/1000.0 * fs)))
    L = labels.copy()
    start = 0
    while start < len(L):
        end = start
        while end+1 < len(L) and L[end+1]==L[start]:
            end += 1
        run_len = end - start + 1
        if run_len < min_len:
            # merge toward the neighboring label with longer adjacent run
            left_lab  = L[start-1] if start>0 else None
            right_lab = L[end+1]   if end+1<len(L) else None
            if left_lab is None and right_lab is None:
                pass
            elif left_lab is None:
                L[start:end+1] = right_lab
            elif right_lab is None:
                L[start:end+1] = left_lab
            else:
                # choose side with longer contiguous run
                lstart=start-1
                while lstart-1>=0 and L[lstart-1]==left_lab: lstart-=1
                rend=end+1
                while rend+1<len(L) and L[rend+1]==right_lab: rend+=1
                if (start-lstart) >= (rend-end):
                    L[start:end+1] = left_lab
                else:
                    L[start:end+1] = right_lab
        start = end+1
    return L

def microstate_metrics(labels: np.ndarray, fs: float, k: int) -> Dict[str, object]:
    """
    Mean duration (ms), coverage, occurrence rate (/s), transition matrix, sequence entropy (bits).
    """
    T = len(labels)
    cov = np.zeros(k, float)
    occ = np.zeros(k, float)
    durations=[]
    # compute runs
    i=0; seq=[]
    while i<T:
        j=i
        while j+1<T and labels[j+1]==labels[i]:
            j+=1
        L = j-i+1
        cov[labels[i]] += L
        occ[labels[i]] += 1
        durations.append((labels[i], L/fs))
        seq.extend([labels[i]]*L)
        i=j+1
    cov = cov/T
    occ = occ/ (T/fs)  # per second
    dur_ms = np.zeros(k,float)
    for s in range(k):
        ls = [d for (lab,d) in durations if lab==s]
        dur_ms[s] = 1000.0 * (np.mean(ls) if ls else np.nan)
    # transitions
    trans = np.zeros((k,k), float)
    for t in range(T-1):
        a,b = labels[t], labels[t+1]
        if a!=b:
            trans[a,b] += 1
    row_sum = trans.sum(axis=1, keepdims=True)+1e-12
    P = trans/row_sum
    # sequence entropy
    p = cov; p = p[p>0]
    H = float(-np.sum(p*np.log2(p)))
    return {'coverage':cov, 'occurrence':occ, 'duration_ms':dur_ms, 'P':P, 'H_seq':H}

def gev_score(GFP: np.ndarray, corr_abs: np.ndarray) -> float:
    """Global Explained Variance: sum(GFP^2 * corr^2)/sum(GFP^2)."""
    num = np.sum((GFP**2) * (corr_abs**2))
    den = np.sum(GFP**2) + 1e-12
    return float(num/den)

# ---------------- Orchestrator ----------------
def run_microstate_segmentation(
    RECORDS: pd.DataFrame,
    eeg_channels: List[str],
    ignition_windows: Optional[List[Tuple[float,float]]] = None,
    baseline_windows: Optional[List[Tuple[float,float]]] = None,
    band: Optional[Tuple[float,float]] = (2,40),
    ks: List[int] = [3,4,5,6],
    peak_refrac_ms: float = 10.0,
    min_seg_ms: float = 30.0,
    time_col: str = 'Timestamp',
    out_dir: str = 'exports_microstates/session',
    n_surrogates: int = 200,
    show: bool = False
)->Dict[str, object]:
    """
    Microstate maps & metrics with surrogate validation; Ignition/Baseline comparison.
    """
    _ensure_dir(out_dir)
    time_col = ensure_timestamp_column(RECORDS, time_col=time_col, default_fs=128.0)
    fs = infer_fs(RECORDS, time_col)

    def build_X(wins):
        X=[]; names=[]
        for ch in eeg_channels:
            nm = ch if ch.startswith('EEG.') else 'EEG.'+ch
            if nm in RECORDS.columns:
                x = get_series(RECORDS, nm)
                if wins: x = slice_concat(x, fs, wins)
                if band is not None:
                    x = bandpass(x, fs, band[0], band[1])
                X.append(zscore(np.asarray(x,float))); names.append(nm)
        if not X: raise ValueError("No EEG channels found.")
        L = min(len(x) for x in X)
        X = np.vstack([x[:L] for x in X])
        return X, names

    states = {'ignition': ignition_windows, 'baseline': baseline_windows}
    results={}; summaries=[]

    for st, wins in states.items():
        if wins is None: continue
        X, names = build_X(wins)             # (n_ch × T)
        n_ch, T = X.shape
        G = gfp(X)
        # pick GFP peaks (every ~peak_refrac_ms)
        skip = max(1, int(round(peak_refrac_ms/1000.0 * fs)))
        idx_peaks = pick_gfp_peaks(G, skip)
        if idx_peaks.size < max(200, 5*len(ks)):
            # fallback: take top-N GFP timepoints
            N = max(200, 5*len(ks))
            idx_peaks = np.argsort(G)[-N:]

        # topography matrix at peaks
        Maps = X[:, idx_peaks]          # (n_ch × Nmaps)
        Maps = normalize_maps(Maps)     # zero-mean, L2 norms

        # try multiple k; pick best by GEV
        candidates=[]
        for k in ks:
            centers, _ = kmeans_microstates(Maps, k=k, n_init=30, seed=0)
            # backfit all timepoints
            lab_all, corr_abs = backfit_sequence(X, centers)
            # smoothing
            lab_s = smooth_labels(lab_all, fs, min_dur_ms=min_seg_ms)
            # recompute corr after smoothing (optional): use the same corr_abs for GEV
            gev = gev_score(G, corr_abs)
            candidates.append((k, centers, lab_s, corr_abs, gev))
        # select best k
        k_best, centers, labels, corr_abs, gev_best = max(candidates, key=lambda t:t[4])

        # metrics
        metrics = microstate_metrics(labels, fs, k_best)

        # surrogate GEV null (phase-randomize per channel)
        null_gev=[]
        for _ in range(n_surrogates):
            Xs = np.vstack([zscore(np.fft.irfft(np.abs(np.fft.rfft(x))*np.exp(1j*np.random.uniform(-np.pi,np.pi, size=np.fft.rfft(x).size)), n=len(x)).astype(float))
                            for x in X])
            Gs = gfp(Xs)
            # reuse same centers? fairer to re-cluster peaks of surrogate to avoid bias:
            idx_s = pick_gfp_peaks(Gs, skip)
            if idx_s.size < len(idx_peaks):
                idx_s = np.argsort(Gs)[-len(idx_peaks):]
            Maps_s = normalize_maps(Xs[:, idx_s])
            c_s, _ = kmeans_microstates(Maps_s, k=k_best, n_init=10, seed=0)
            _, corr_s = backfit_sequence(Xs, c_s)
            null_gev.append(gev_score(Gs, corr_s))
        null_gev = np.asarray(null_gev, float)
        gev_p = float((np.sum(null_gev >= gev_best)+1)/(null_gev.size+1)) if null_gev.size else np.nan
        gev_lo = float(np.nanpercentile(null_gev, 2.5)) if null_gev.size else np.nan
        gev_hi = float(np.nanpercentile(null_gev,97.5)) if null_gev.size else np.nan

        # -------- Plots --------
        # (1) GEV vs k curve
        plt.figure(figsize=(4.5,3))
        plt.plot([c[0] for c in candidates], [c[4] for c in candidates], 'o-')
        plt.xlabel('k'); plt.ylabel('GEV'); plt.title(f'GEV vs k — {st}')
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f'gev_vs_k_{st}.png'), dpi=140)
        if show: plt.show()
        plt.close()

        # (2) Microstate maps (channel×k heatmap)
        plt.figure(figsize=(max(6, 0.5*k_best+4), 3.5))
        im = plt.imshow(centers, aspect='auto', cmap='coolwarm',
                        vmin=-np.max(np.abs(centers)), vmax=np.max(np.abs(centers)))
        plt.colorbar(label='weight')
        plt.yticks(range(len(names)), [n.split('.',1)[-1] for n in names], fontsize=8)
        plt.xticks(range(k_best), [f'M{k+1}' for k in range(k_best)])
        plt.title(f'Microstate maps — {st} (k={k_best})')
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f'maps_{st}.png'), dpi=140)
        if show: plt.show()
        plt.close()

        # (3) Sequence strip & coverage/duration bars
        plt.figure(figsize=(8,1.8))
        plt.imshow(labels[None,:], aspect='auto', cmap='tab20', vmin=0, vmax=k_best-1)
        plt.yticks([]); plt.xlabel('Time (samples)'); plt.title(f'Sequence — {st}')
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f'sequence_{st}.png'), dpi=140)
        if show: plt.show()
        plt.close()

        plt.figure(figsize=(max(6, 0.5*k_best+4), 3.0))
        x = np.arange(k_best)
        plt.bar(x-0.2, metrics['coverage']*100, width=0.4, label='coverage %')
        plt.bar(x+0.2, metrics['duration_ms'], width=0.4, label='mean duration (ms)')
        plt.xticks(x, [f'M{k+1}' for k in range(k_best)])
        plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(out_dir, f'coverage_duration_{st}.png'), dpi=140)
        if show: plt.show()
        plt.close()

        # (4) Transition matrix
        plt.figure(figsize=(4,3.2))
        plt.imshow(metrics['P'], vmin=0, vmax=np.nanmax(metrics['P']) if np.isfinite(np.nanmax(metrics['P'])) else 1.0, cmap='magma')
        plt.colorbar(label='P(i→j)')
        plt.xticks(range(k_best), [f'M{k+1}' for k in range(k_best)])
        plt.yticks(range(k_best), [f'M{k+1}' for k in range(k_best)])
        plt.title(f'Transitions — {st}')
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f'transitions_{st}.png'), dpi=140)
        if show: plt.show()
        plt.close()

        # Store/summary
        results[st] = {'names':names, 'k':k_best, 'centers':centers,
                       'labels':labels, 'metrics':metrics,
                       'GEV':gev_best, 'GEV_null95_lo':gev_lo, 'GEV_null95_hi':gev_hi, 'GEV_p':gev_p}
        summaries.append({'state':st, 'k':k_best, 'GEV':gev_best, 'GEV_p':gev_p,
                          'coverage_mean':float(np.nanmean(metrics['coverage'])),
                          'duration_ms_mean':float(np.nanmean(metrics['duration_ms'])),
                          'H_seq':metrics['H_seq']})

    # Ignition vs Baseline quick comparison
    if 'ignition' in results and 'baseline' in results:
        ig = results['ignition']; ba = results['baseline']
        plt.figure(figsize=(6,3.2))
        names = ['GEV','coverage_mean','duration_ms_mean','H_seq']
        vals_ig = [s for s in summaries if s['state']=='ignition'][0]
        vals_ba = [s for s in summaries if s['state']=='baseline'][0]
        x = np.arange(len(names)); w=0.38
        plt.bar(x-w/2, [vals_ba[n] for n in names], width=w, label='Baseline', color='tab:orange', alpha=0.9)
        plt.bar(x+w/2, [vals_ig[n] for n in names], width=w, label='Ignition', color='tab:blue', alpha=0.9)
        plt.xticks(x, names); plt.ylabel('value'); plt.title('Ignition vs Baseline — microstate metrics')
        plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(out_dir,'ign_vs_base.png'), dpi=140)
        if show: plt.show()
        plt.close()

    # Summary CSV
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(os.path.join(out_dir, 'summary.csv'), index=False)
    return {'summary': summary_df, 'results': results, 'out_dir': out_dir}

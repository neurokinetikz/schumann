"""
Dynamic Connectivity & Metastability — Simple Graphs & Validation
=================================================================

What it does
------------
• Sliding-window connectivity (PLV or imag-coherency) → W(t).
• Global synchrony R(t) (Kuramoto) and mean edge weight M(t).
• Metastability = var(R(t)), with phase-randomized surrogates → p-value.
• PCA on vec(W(t)) → state-space trajectory; k-means (k=2..6) -> states.
• State metrics: dwell times, coverage, transition matrix, sequence entropy.
• Optional: correlate R(t) with 7.83 Hz SR envelope (shift-null).
• Per-state (Ignition/Baseline) outputs: PNGs + CSV summary in out_dir.

Usage
-----
res = run_dynamic_connectivity_metastability(
    RECORDS,
    eeg_channels=['EEG.O1','EEG.O2','EEG.P7','EEG.P8','EEG.FC5','EEG.FC6'],
    ignition_windows=[(290,310),(580,600)],
    baseline_windows=[(0,290),(325,580)],
    band=(8,13),                   # choose band; e.g., theta (4–8), alpha (8–13)
    method='pli',                  # 'pli' (default) or 'imagcoh'
    win_sec=1.0, step_sec=0.25,    # sliding window params
    sr_channel=None,               # 'EEG.O1' if you want SR-proxy coupling analysis
    time_col='Timestamp',
    out_dir='exports_dyn/S01',
    show=False
)
print(res['summary'])
"""
from __future__ import annotations
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from scipy import signal
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ---------------- I/O & time helpers ----------------
def _ensure_dir(d): os.makedirs(d, exist_ok=True); return d

def detect_time_col(df,
    candidates=('Timestamp','Time','time','t','seconds','sec','ms','datetime','DateTime','Datetime')) -> Optional[str]:
    for c in candidates:
        if c in df.columns: return c
    # numeric, roughly monotonic
    for c in df.columns:
        s = pd.to_numeric(df[c], errors='coerce')
        if s.notna().sum()>max(50,0.5*len(df)):
            x=s.values.astype(float); dt=np.diff(x[np.isfinite(x)])
            if dt.size and np.nanmedian(dt)>0: return c
    # datetime
    for c in df.columns:
        try:
            _ = pd.to_datetime(df[c], errors='raise'); return c
        except Exception: pass
    return None

def ensure_timestamp_column(df: pd.DataFrame, time_col: Optional[str]=None, default_fs: float = 128.0, out_name='Timestamp')->str:
    col = time_col or detect_time_col(df)
    if col is None:
        df[out_name]=np.arange(len(df), dtype=float)/default_fs; return out_name
    s = df[col]
    if np.issubdtype(s.dtype, np.datetime64) or 'datetime' in str(s.dtype).lower():
        tsec=(pd.to_datetime(s)-pd.to_datetime(s).iloc[0]).dt.total_seconds().astype(float)
        df[out_name]=tsec.values; return out_name
    sn = pd.to_numeric(s, errors='coerce').astype(float)
    if sn.notna().sum()<max(50,0.5*len(df)):
        df[out_name]=np.arange(len(df), dtype=float)/default_fs; return out_name
    sn = sn - np.nanmin(sn[np.isfinite(sn)])
    df[out_name]=sn.values; return out_name

def infer_fs(df: pd.DataFrame, time_col: str)->float:
    t = np.asarray(pd.to_numeric(df[time_col], errors='coerce').values, float)
    dt=np.diff(t); dt=dt[(dt>0)&np.isfinite(dt)]
    if dt.size==0: raise ValueError("Cannot infer fs")
    return float(1.0/np.median(dt))

def get_series(df: pd.DataFrame, name: str)->np.ndarray:
    if name in df.columns:
        return pd.to_numeric(df[name], errors='coerce').fillna(0.0).values.astype(float)
    alt='EEG.'+name
    if alt in df.columns:
        return pd.to_numeric(df[alt], errors='coerce').fillna(0.0).values.astype(float)
    raise ValueError(f"{name} not found.")

def slice_concat(x: np.ndarray, fs: float, wins: Optional[List[Tuple[float,float]]])->np.ndarray:
    if not wins: return x.copy()
    segs=[]; n=len(x)
    for (a,b) in wins:
        i0,i1=int(round(a*fs)), int(round(b*fs))
        i0=max(0,i0); i1=min(n,i1)
        if i1>i0: segs.append(x[i0:i1])
    return np.concatenate(segs) if segs else x.copy()

def zscore(x): x=np.asarray(x,float); return (x-np.mean(x))/(np.std(x)+1e-12)

# ---------------- Filtering & analytic ----------------
def bandpass(x, fs, f1, f2, order=4):
    ny=0.5*fs; f1=max(1e-6,min(f1,0.99*ny)); f2=max(f1+1e-6,min(f2,0.999*ny))
    b,a=signal.butter(order,[f1/ny,f2/ny],btype='band'); return signal.filtfilt(b,a,x)

def analytic_phase(x, fs, f1, f2):
    xb = bandpass(x, fs, f1, f2)
    z  = signal.hilbert(xb)
    return np.angle(z)

# ---------------- Connectivity (PLV / imagcoh) ----------------
def pli_window(Xb: np.ndarray) -> np.ndarray:
    """PLI on analytic phases; Xb: (n_ch, W) bandpassed."""
    Z = signal.hilbert(Xb, axis=1); phi = np.angle(Z)
    n = Xb.shape[0]; W = np.zeros((n,n), float)
    for i in range(n):
        for j in range(i+1,n):
            dphi = phi[i]-phi[j]
            W[i,j]=W[j,i]=float(np.abs(np.mean(np.sign(np.sin(dphi)))))
    np.fill_diagonal(W,0.0); return W

def imagcoh_window(Xb: np.ndarray) -> np.ndarray:
    """Imag coherency; Xb: (n_ch, W) bandpassed."""
    Z = signal.hilbert(Xb, axis=1); n = Xb.shape[0]
    W = np.zeros((n,n), float)
    for i in range(n):
        for j in range(i+1,n):
            Sxy = np.mean(Z[i]*np.conj(Z[j]))
            Sxx = np.mean(Z[i]*np.conj(Z[i])); Syy = np.mean(Z[j]*np.conj(Z[j]))
            coh = Sxy/np.sqrt((Sxx*Syy)+1e-24)
            W[i,j]=W[j,i]=float(np.abs(np.imag(coh)))
    np.fill_diagonal(W,0.0); return W

# ---------------- Sliding windows ----------------
def sliding_windows(X: np.ndarray, fs: float, win_sec: float, step_sec: float):
    win = int(round(win_sec*fs)); step=int(round(step_sec*fs))
    idx=[]
    for c in range(win//2, X.shape[1]-win//2, step):
        idx.append( (c-win//2, c+win//2, c/fs) )
    return idx

# ---------------- Kuramoto R ----------------
def kuramoto_R(Xb: np.ndarray) -> float:
    Z = signal.hilbert(Xb, axis=1); phi = np.angle(Z)
    R = np.abs(np.mean(np.exp(1j*phi), axis=0))
    return float(np.mean(R))  # mean over window samples as the window’s R

# ---------------- Surrogates ----------------
def phase_randomize(x: np.ndarray)->np.ndarray:
    X = np.fft.rfft(x); mag = np.abs(X); ph = np.angle(X)
    rnd = np.random.uniform(-np.pi, np.pi, size=mag.size)
    rnd[0] = ph[0]
    if mag.size % 2 == 0: rnd[-1] = ph[-1]
    Xs = mag * np.exp(1j*rnd)
    return np.fft.irfft(Xs, n=len(x)).astype(float)

def build_surrogate_matrix(X: np.ndarray) -> np.ndarray:
    return np.vstack([zscore(phase_randomize(x)) for x in X])

# ---------------- Main runner ----------------
def run_dynamic_connectivity_metastability(
    RECORDS: pd.DataFrame,
    eeg_channels: List[str],
    ignition_windows: Optional[List[Tuple[float,float]]] = None,
    baseline_windows: Optional[List[Tuple[float,float]]] = None,
    band: Tuple[float,float] = (8,13),
    method: str = 'pli',        # 'pli' or 'imagcoh'
    win_sec: float = 1.0, step_sec: float = 0.25,
    sr_channel: Optional[str] = None,   # if provided, compute corr(R) with SR env @7.83 (shift-null)
    time_col: str = 'Timestamp',
    out_dir: str = 'exports_dyn/session',
    show: bool = False,
    n_surrogates: int = 200
)->Dict[str, object]:
    """
    Dynamic connectivity & metastability with simple graphs and tests.
    """
    _ensure_dir(out_dir)
    time_col = ensure_timestamp_column(RECORDS, time_col=time_col, default_fs=128.0)
    fs = infer_fs(RECORDS, time_col)

    def build_state_matrix(wins):
        X=[]; names=[]
        for ch in eeg_channels:
            nm = ch if ch in RECORDS.columns else ('EEG.'+ch if ('EEG.'+ch) in RECORDS.columns else ch)
            if nm in RECORDS.columns:
                x = get_series(RECORDS, nm); x = slice_concat(x, fs, wins)
                X.append(zscore(np.asarray(x,float))); names.append(nm)
        if not X: raise ValueError("No EEG channels found.")
        L = min(len(x) for x in X)
        return np.vstack([x[:L] for x in X]), names  # (n, T)

    def dyn_conn_for_state(X: np.ndarray, names: List[str], label: str):
        idx = sliding_windows(X, fs, win_sec, step_sec)
        if not idx:
            raise ValueError("No sliding windows — extend windows or reduce win_sec.")
        Fvec=[]; Rts=[]; Mts=[]; tcent=[]
        for (s,e,t) in idx:
            Xw = X[:, s:e]
            # band-limit
            Xb = np.vstack([bandpass(x, fs, band[0], band[1]) for x in Xw])
            W = pli_window(Xb) if method.lower()=='pli' else imagcoh_window(Xb)
            # features & metrics
            ut = W[np.triu_indices(X.shape[0],1)]
            Fvec.append(ut)
            Rts.append(kuramoto_R(Xb))
            Mts.append(np.mean(ut))
            tcent.append(t)
        Fvec = np.vstack(Fvec)            # (T', E)
        Rts  = np.array(Rts, float)
        Mts  = np.array(Mts, float)
        tcent= np.array(tcent, float)

        # metastability
        meta = float(np.var(Rts))

        # surrogates for metastability (phase-randomize channels independently)
        null_meta=[]
        for _ in range(n_surrogates):
            Xs = build_surrogate_matrix(X)
            Rs=[]
            for (s,e,_) in idx:
                Xsw = Xs[:, s:e]
                Xsb = np.vstack([bandpass(x, fs, band[0], band[1]) for x in Xsw])
                Rs.append(kuramoto_R(Xsb))
            null_meta.append(np.var(Rs))
        null_meta = np.asarray(null_meta, float)
        meta_p = float((np.sum(null_meta >= meta)+1)/(n_surrogates+1))

        # PCA & clustering of connectivity states
        pca = PCA(n_components=2, random_state=0)
        Z = pca.fit_transform(Fvec)       # (T', 2)
        # choose k by silhouette (2..6)
        best_k, best_s, best_lab = 2, -np.inf, None
        for k in range(2,7):
            km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(Z)
            lab = km.labels_
            if len(np.unique(lab))<2: continue
            s = silhouette_score(Z, lab)
            if s > best_s:
                best_s, best_k, best_lab = s, k, lab
        states = best_lab if best_lab is not None else KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(Z)
        k_opt  = best_k

        # dwell times & transitions
        Tprime = len(states)
        cov = np.array([np.mean(states==i) for i in range(k_opt)])
        # dwell (in windows); convert to seconds
        dwell=[]
        i=0
        while i<Tprime:
            j=i
            while j+1<Tprime and states[j+1]==states[i]:
                j+=1
            dwell.append((states[i], (j-i+1)*step_sec))
            i=j+1
        dwell_mean = np.zeros(k_opt)
        for s in range(k_opt):
            ds=[d for (lab,d) in dwell if lab==s]
            dwell_mean[s]=np.mean(ds) if ds else np.nan

        trans = np.zeros((k_opt,k_opt), float)
        for t in range(Tprime-1):
            a,b = states[t], states[t+1]
            if a!=b: trans[a,b]+=1
        trans = trans/(trans.sum(axis=1, keepdims=True)+1e-12)
        # sequence entropy
        p = cov[cov>0]; H = float(-np.sum(p*np.log2(p)))

        # plots: R(t) with null band
        plt.figure(figsize=(9,3))
        zR = (Rts - np.mean(Rts))/(np.std(Rts)+1e-12)
        null95 = np.nanpercentile((null_meta - null_meta.mean())/(null_meta.std()+1e-12), 95) if null_meta.size else np.nan
        plt.plot(tcent, zR, lw=1.4, label='z-R(t)')
        if np.isfinite(null95): plt.hlines(null95, tcent[0], tcent[-1], colors='k', linestyles='--', label='null95')
        plt.xlabel('Time (s)'); plt.ylabel('z-R'); plt.title(f'Global synchrony (metastability={meta:.3f}, p={meta_p:.3f}) — {label}')
        plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(out_dir, f'R_timeseries_{label}.png'), dpi=140)
        if show: plt.show()
        plt.close()

        # PCA trajectory colored by state
        plt.figure(figsize=(6,5))
        sc = plt.scatter(Z[:,0], Z[:,1], c=states, s=10, cmap='tab20')
        plt.xlabel('PC1'); plt.ylabel('PC2'); plt.title(f'Connectivity state-space (k={k_opt}, sil={best_s:.2f}) — {label}')
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f'pca_states_{label}.png'), dpi=140)
        if show: plt.show()
        plt.close()

        # Transition matrix heatmap
        plt.figure(figsize=(4.2,3.2))
        vmax = np.nanmax(trans) if np.isfinite(np.nanmax(trans)) else 1.0
        plt.imshow(trans, vmin=0, vmax=vmax, cmap='magma'); plt.colorbar(label='P(i→j)')
        plt.xticks(range(k_opt), [f'S{k+1}' for k in range(k_opt)]); plt.yticks(range(k_opt), [f'S{k+1}' for k in range(k_opt)])
        plt.title(f'Transitions — {label}'); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'transitions_{label}.png'), dpi=140)
        if show: plt.show()
        plt.close()

        return {
            'R': Rts, 'M': Mts, 't': tcent,
            'meta': meta, 'meta_p': meta_p, 'null_meta': null_meta,
            'Z': Z, 'states': states, 'k_opt': k_opt, 'silhouette': best_s,
            'coverage': cov, 'dwell_mean_sec': dwell_mean, 'transitions': trans
        }

    # Build per-state matrices & run dynamics
    summary_rows=[]
    results={}
    for st, wins in {'ignition': ignition_windows, 'baseline': baseline_windows}.items():
        if wins is None: continue
        X, names = build_state_matrix(wins)
        # dynamic metrics
        res = dyn_conn_for_state(X, names, st)
        results[st] = res
        summary_rows.append({'state': st, 'meta_varR': res['meta'], 'meta_p': res['meta_p'],
                             'k_opt': res['k_opt'], 'silhouette': res['silhouette'],
                             'coverage_mean': float(np.nanmean(res['coverage'])),
                             'dwell_mean_sec_mean': float(np.nanmean(res['dwell_mean_sec']))})

        # optional SR coupling: corr(R(t), SR_env@7.83)
        if sr_channel is not None:
            sr = get_series(RECORDS, sr_channel)
            sr = slice_concat(sr, fs, wins)
            # 7.83 ± 0.6 Hz envelope
            ny=0.5*fs; b,a=signal.butter(4, [max(1e-6,(7.83-0.6))/ny, min(0.999,(7.83+0.6))/ny], btype='band')
            env = np.abs(signal.hilbert(signal.filtfilt(b,a, sr)))
            # sample env per window center
            env_w = np.interp(res['t'], np.arange(len(env))/fs, env)
            r = float(np.corrcoef((res['R']-np.mean(res['R']))/(np.std(res['R'])+1e-12),
                                  (env_w -np.mean(env_w))/ (np.std(env_w)+1e-12))[0,1])
            # shift-null
            rng=np.random.default_rng(13); null=[]
            for _ in range(n_surrogates):
                s=int(rng.integers(1, len(env_w)-1))
                null.append(np.corrcoef(res['R'], np.r_[env_w[-s:], env_w[:-s]])[0,1])
            thr95=float(np.nanpercentile(null,95))
            summary_rows[-1].update({'R_env_r': r, 'R_env_null95': thr95})

            plt.figure(figsize=(9,3))
            zR = (res['R'] - np.mean(res['R']))/(np.std(res['R'])+1e-12)
            zE = (env_w - np.mean(env_w))/(np.std(env_w)+1e-12)
            plt.plot(res['t'], zR, label='z-R(t)')
            plt.plot(res['t'], zE, label='z-SR env')
            plt.xlabel('Time (s)'); plt.title(f'R(t) vs SR envelope (r={r:.2f}, null95~{thr95:.2f}) — {st}')
            plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(out_dir, f'R_vs_SR_{st}.png'), dpi=140)
            if show: plt.show()
            plt.close()

    # Ignition vs Baseline summary bar
    if 'ignition' in results and 'baseline' in results:
        ig, ba = results['ignition'], results['baseline']
        plt.figure(figsize=(6,3.2))
        keys = ['meta_varR','dwell_mean_sec_mean']
        x=np.arange(len(keys)); w=0.38
        vals_ig=[r for r in [np.var(ig['R']), np.nanmean(ig['dwell_mean_sec'])]]
        vals_ba=[r for r in [np.var(ba['R']), np.nanmean(ba['dwell_mean_sec'])]]
        plt.bar(x-w/2, vals_ba, width=w, label='Baseline', color='tab:orange', alpha=0.9)
        plt.bar(x+w/2, vals_ig, width=w, label='Ignition', color='tab:blue', alpha=0.9)
        plt.xticks(x, ['var(R)','dwell mean (s)']); plt.ylabel('value')
        plt.title('Ignition vs Baseline — metastability & dwell')
        plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(out_dir, f'ign_vs_base.png'), dpi=140)
        if show: plt.show()
        plt.close()

    # Save CSV summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir,'summary.csv'), index=False)
    return {'summary': summary_df, 'results': results, 'out_dir': out_dir}

"""
Harmonic Resonance & Spectral Mode Analysis — Simple Graphs & Validation
=======================================================================

Tests
-----
1) Spectral harmonicity:
   • For each channel, compute Welch PSD and a local SNR z-score at Schumann
     harmonics (7.83, 14.3, 20.8, 27.3, 33.8 Hz), using a robust baseline taken
     from sidebands around each target (excluding the central peak).
   • Count channels with z >= z_thr (default 2.0) at each harmonic.
   • Split the data in M epochs → repeat the test → consistency metric
     (fraction of epochs with a “hit” per channel) and a simple binomial test.

2) Spatial mode at 7–8 Hz:
   • Band-pass 7.23–8.43 Hz across channels, compute spatial covariance,
     PC1 variance ratio (how “global” the 8 Hz mode is), and “in-phase score”
     (alignment of PC1 weights with all-ones vector).

3) Off-harmonic controls:
   • Repeat spectral SNR test in an off band (16–18 Hz) → expect fewer hits.

Outputs
-------
• PNGs: mean PSD with harmonic lines, per-channel harmonic z heatmap,
        per-harmonic bar of hit counts, PC1 weights bar, PC1 variance ratio plot.
• CSV: summary with per-channel metrics and overall harmonicity indices.

Usage
-----
res = run_harmonic_resonance_spectral_modes(
    RECORDS,
    eeg_channels=['EEG.O1','EEG.O2','EEG.P7','EEG.P8','EEG.FC5','EEG.FC6'],
    time_col='Timestamp',
    ignition_windows=[(290,310),(580,600)],     # or None for full session
    out_dir='exports_harmonics_simple/S01',
    show=False
)
print(res['summary'])
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from scipy import signal
from scipy.stats import binomtest

# ---------- I/O / time helpers ----------
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
            arr = s.values.astype(float)
            dt = np.diff(arr[np.isfinite(arr)])
            if dt.size and np.nanmedian(dt) > 0: return c
    # datetime?
    for c in df.columns:
        try:
            _ = pd.to_datetime(df[c], errors='raise')
            return c
        except Exception:
            pass
    return None

def ensure_timestamp_column(df: pd.DataFrame, time_col: Optional[str]=None,
                            default_fs: float = 128.0, out_name: str = 'Timestamp')->str:
    col = time_col or detect_time_col(df)
    if col is None:
        df[out_name] = np.arange(len(df), dtype=float)/default_fs
        return out_name
    s = df[col]
    # datetime → seconds since first
    if np.issubdtype(s.dtype, np.datetime64) or 'datetime' in str(s.dtype).lower():
        tsec = (pd.to_datetime(s) - pd.to_datetime(s).iloc[0]).dt.total_seconds().astype(float)
        df[out_name] = tsec.values; return out_name
    # numeric
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

# ---------- DSP helpers ----------
def bandpass(x, fs, f1, f2, order=4):
    ny=0.5*fs; f1=max(1e-6,min(f1,0.99*ny)); f2=max(f1+1e-6,min(f2,0.999*ny))
    b,a=signal.butter(order,[f1/ny,f2/ny],btype='band'); return signal.filtfilt(b,a,x)

def welch_psd(x: np.ndarray, fs: float, nperseg_sec: float = 4.0) -> Tuple[np.ndarray,np.ndarray]:
    nperseg = int(round(nperseg_sec*fs))
    f, p = signal.welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg//2, nfft=None)
    return f, p

# ---------- Spectral harmonic z-score ----------
def harmonic_zscores(f: np.ndarray, p: np.ndarray,
                     harmonics=(7.83,14.3,20.8,27.3,33.8),
                     half_bw: float = 0.6,
                     side_bw: float = 2.0) -> Dict[str, float]:
    """
    For each target harmonic h, compute z = (P(h) - median(side)) / MAD(side),
    where side = [h-side_bw, h+side_bw] \ [h-half_bw, h+half_bw] (exclude central).
    Returns dict { '7.83': z, ... }  (NaN if insufficient bins).
    """
    zmap={}
    for h in harmonics:
        mask_side = (f>=h-side_bw) & (f<=h+side_bw)
        mask_excl = (f>=h-half_bw) & (f<=h+half_bw)
        side = p[mask_side & ~mask_excl]
        if side.size < 10:
            zmap[f"{h:.2f}"] = np.nan; continue
        med = np.median(side); mad = np.median(np.abs(side - med)) + 1e-12
        # nearest-bin pick for center power
        idx = int(np.nanargmin(np.abs(f - h)))
        z = (p[idx] - med) / (1.4826*mad)  # robust z via MAD
        zmap[f"{h:.2f}"] = float(z)
    return zmap

# ---------- Spatial mode (7–8 Hz) ----------
def spatial_mode_8hz(X: np.ndarray, fs: float, f0=7.83, half=0.6) -> Dict[str, object]:
    """
    X: (n_ch, T) — band-pass 7.83±half, compute covariance → PC1 variance ratio,
    and in-phase score (|corr(PC1, all-ones)|).
    """
    Xb = np.vstack([bandpass(x, fs, f0-half, f0+half) for x in X])
    # covariance
    C = Xb @ Xb.T / Xb.shape[1]
    vals, vecs = np.linalg.eigh(C)
    idx = np.argsort(vals)[::-1]
    vals, vecs = vals[idx], vecs[:, idx]
    var_ratio = float(vals[0] / (np.sum(vals)+1e-12))
    w = vecs[:, 0]
    w = w / (np.linalg.norm(w)+1e-12)
    inphase = float(np.abs(np.dot(w, np.ones_like(w))/ (np.linalg.norm(w)*np.sqrt(len(w))+1e-12)))
    return {'pc1_var_ratio': var_ratio, 'pc1_weights': w, 'inphase_score': inphase}

# ---------- Main runner ----------
def run_harmonic_resonance_spectral_modes(
    RECORDS: pd.DataFrame,
    eeg_channels: List[str],
    time_col: str = 'Timestamp',
    ignition_windows: Optional[List[Tuple[float,float]]] = None,
    nperseg_sec: float = 4.0,
    harmonics: Tuple[float,...] = (7.83,14.3,20.8,27.3,33.8),
    half_bw: float = 0.6,
    side_bw: float = 2.0,
    z_thr: float = 2.0,
    offband: Tuple[float,float] = (16.0,18.0),
    n_epochs: int = 4,
    out_dir: str = 'exports_harmonics_simple/session',
    show: bool = True
) -> Dict[str, object]:
    """
    High-resolution spectral harmonic test + spatial mode at 7–8 Hz.
    """
    _ensure_dir(out_dir)
    # normalize time column
    time_col = ensure_timestamp_column(RECORDS, time_col=time_col, default_fs=128.0)
    fs = infer_fs(RECORDS, time_col)
    # build signals matrix
    X=[]
    kept=[]
    for ch in eeg_channels:
        nm = ch if ch.startswith('EEG.') else 'EEG.'+ch
        if nm in RECORDS.columns:
            x = get_series(RECORDS, nm)
            if ignition_windows: x = slice_concat(x, fs, ignition_windows)
            X.append(x); kept.append(nm)
    if not X: raise ValueError("No EEG channels found from eeg_channels.")
    # truncate to common length
    L = min(len(x) for x in X)
    X = np.vstack([x[:L] for x in X])   # (n_ch, T)
    n_ch, T = X.shape

    # ---- (1) Welch PSD per channel & harmonic z-scores ----
    psd = []
    ztbl_rows=[]
    for i,ch in enumerate(kept):
        f, p = welch_psd(X[i], fs, nperseg_sec=nperseg_sec)
        psd.append((f,p))
        zmap = harmonic_zscores(f, p, harmonics=harmonics, half_bw=half_bw, side_bw=side_bw)
        rec = {'channel': ch}; rec.update(zmap)
        ztbl_rows.append(rec)
    ztbl = pd.DataFrame(ztbl_rows)
    zcols = [f"{h:.2f}" for h in harmonics]

    # hit counts per harmonic
    hits = {c: int(np.sum(ztbl[c] >= z_thr)) for c in zcols if c in ztbl.columns}

    # ---- (2) Epoch consistency ----
    M = max(1, n_epochs)
    step = T//M
    ep_hits = {c: [] for c in zcols}
    for e in range(M):
        s = e*step; eidx = (e+1)*step if e<M-1 else T
        for i,ch in enumerate(kept):
            f, p = welch_psd(X[i, s:eidx], fs, nperseg_sec=nperseg_sec)
            zmap = harmonic_zscores(f,p,harmonics=harmonics,half_bw=half_bw,side_bw=side_bw)
            for c in zcols:
                ep_hits[c].append(float(zmap.get(c, np.nan)))
    # fraction of epochs with hit (per channel pooled)
    ep_consistency = {c: float(np.nanmean(np.array(ep_hits[c]) >= z_thr)) for c in zcols}

    # simple binomial test for “>=1 hit” across channels at the fundamental (p0~0.05)
    p0 = 0.05
    k = hits.get(zcols[0], 0); n = n_ch
    p_binom = float(binomtest(k, n, p0, alternative='greater').pvalue)

    # ---- (3) Spatial mode @ ~7.83 Hz ----
    mode = spatial_mode_8hz(X, fs, f0=harmonics[0], half=half_bw)

    # ---- (4) Off-harmonic control (16–18 Hz) ----
    off_hits=[]
    for i,ch in enumerate(kept):
        f, p = welch_psd(X[i], fs, nperseg_sec=nperseg_sec)
        sel = (f>=offband[0]) & (f<=offband[1])
        off_hits.append(float(np.max(p[sel]) if np.any(sel) else np.nan))
    off_mean = float(np.nanmean(off_hits))

    # ---- Plots ----
    # Mean PSD (linear freq) with harmonic lines
    plt.figure(figsize=(8,3.2))
    # plot mean PSD across channels
    f0, p0 = psd[0]
    Pstack = np.vstack([p for (f,p) in psd if len(p)==len(p0)])
    Pmean = np.nanmean(Pstack, axis=0)
    plt.plot(f0, Pmean, lw=1.6, label='Mean PSD')
    for h in harmonics:
        plt.axvline(h, color='tab:red', lw=1.0, alpha=0.7)
    plt.xlim(2, 40); plt.xlabel('Frequency (Hz)'); plt.ylabel('Power'); plt.title('Mean PSD with Schumann lines')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,'mean_psd.png'), dpi=140)
    if show: plt.show()
    plt.close()

    # Heatmap: per-channel harmonic z-scores
    plt.figure(figsize=(max(6, 0.4*len(kept)), 3.0))
    Z = ztbl[zcols].to_numpy(dtype=float)
    im = plt.imshow(Z, aspect='auto', origin='lower', cmap='magma', vmin=np.nanmin(Z), vmax=np.nanmax(Z))
    plt.colorbar(label='z-score vs local sidebands')
    plt.yticks(range(len(kept)), [k.split('.',1)[-1] for k in kept], fontsize=8)
    plt.xticks(range(len(zcols)), zcols)
    plt.title('Per-channel harmonic z-scores'); plt.tight_layout()
    plt.savefig(os.path.join(out_dir,'harmonic_z_heatmap.png'), dpi=140)
    if show: plt.show()
    plt.close()

    # Bar: hit counts per harmonic (z >= z_thr)
    plt.figure(figsize=(6,3))
    xs = np.arange(len(zcols)); vals = [hits.get(c,0) for c in zcols]
    plt.bar(xs, vals, color='tab:blue', alpha=0.9)
    plt.xticks(xs, zcols); plt.ylabel(f'Channels with z≥{z_thr}')
    plt.title('Harmonic hit counts across channels')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,'hit_counts.png'), dpi=140)
    if show: plt.show()
    plt.close()

    # Spatial mode @ 7–8 Hz: PC1 weights bar
    plt.figure(figsize=(max(6, 0.4*len(kept)), 3.0))
    plt.bar(np.arange(len(kept)), mode['pc1_weights'], color='tab:green', alpha=0.9)
    plt.xticks(range(len(kept)), [k.split('.',1)[-1] for k in kept], rotation=0, fontsize=8)
    plt.ylabel('PC1 weight'); plt.title(f"PC1 variance ratio={mode['pc1_var_ratio']:.2f}, in-phase={mode['inphase_score']:.2f}")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,'pc1_weights.png'), dpi=140)
    if show: plt.show()
    plt.close()

    # ---- CSV summary ----
    # per-channel: z at each harmonic, plus channel-level harmonicity index (sum of z+ over harmonics)
    ztbl['HarmonicityIndex'] = np.nansum(np.clip(ztbl[zcols].to_numpy(), 0, None), axis=1)
    summary = {
        'n_channels': n_ch,
        'fund_hits': hits.get(zcols[0], 0),
        'fund_ep_consistency': ep_consistency.get(zcols[0], np.nan),
        'fund_hits_p_binom': p_binom,
        'pc1_var_ratio_8Hz': mode['pc1_var_ratio'],
        'inphase_score_8Hz': mode['inphase_score'],
        'offband_mean_power_16_18Hz': off_mean
    }
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir,'summary.csv'), index=False)
    ztbl.to_csv(os.path.join(out_dir,'per_channel_z.csv'), index=False)

    return {'summary': summary, 'per_channel': ztbl, 'out_dir': out_dir}

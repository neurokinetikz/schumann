"""
Network-level coupling — simple validity tests & graphs
======================================================

Implements two quick, self-contained analyses:

5a) Cross-domain graph alignment
    • Build EEG PLV graph in a chosen band (e.g., alpha).
    • Compute per-electrode Schumann coherence at harmonics (MSC by Welch).
    • Make a Schumann-weighted PLV graph:
          A_weighted[i,j] = PLV[i,j] * sqrt( MSC_i * MSC_j )
      (MSC_i is the mean MSC of channel i across the harmonics.)
    • Compare graph Laplacian entropy and global min-cut with/without Schumann weighting.
    • Plots: adjacency heatmaps, Δ edge histogram, bar chart of entropy/min-cut.

5b) Source-space (ROI) mapping (sensor ROI proxy)
    • Define conservative ROIs (occipital/parietal/frontal/temporal groups).
    • Build ROI time series (mean across available sensors) and optionally
      apply symmetric orthogonalization (leakage reduction).
    • For each ROI, compute:
        - PLV with Schumann narrowband (~7.83 Hz ± half_bw)
        - MSC with Schumann across harmonics (Welch)
      with circular-shift surrogates → p-values.
    • Plots: bar charts with 95% surrogate bands; table of stats.

Assumptions:
- RECORDS: pandas.DataFrame with a time column (default 'Timestamp')
  and EEG/sensor columns like 'EEG.O1', 'EEG.O2', ...
- sr_channel: a Schumann/ELF reference; if none, use a clean posterior EEG.

Copy-paste this module, then see the usage examples at the bottom.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple, Optional
from scipy import signal

# ----------------------------- generic helpers -----------------------------

def infer_fs(RECORDS: pd.DataFrame, time_col: str = 'Timestamp') -> float:
    t = np.asarray(pd.to_numeric(RECORDS[time_col], errors='coerce').values, float)
    dt = np.diff(t); dt = dt[(dt > 0) & np.isfinite(dt)]
    if dt.size == 0:
        raise ValueError("Cannot infer sampling rate from time column.")
    return float(1.0 / np.median(dt))

def get_series(RECORDS: pd.DataFrame, name: str) -> np.ndarray:
    """Return a numeric signal array. Accepts 'EEG.O1' or bare 'O1'."""
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

def bandpass(x: np.ndarray, fs: float, f1: float, f2: float, order: int = 4) -> np.ndarray:
    ny = 0.5*fs
    f1 = max(1e-6, min(f1, 0.99*ny)); f2 = max(f1+1e-6, min(f2, 0.999*ny))
    b,a = signal.butter(order, [f1/ny, f2/ny], btype='band')
    return signal.filtfilt(b,a,x)

# ----------------------------- PLV & MSC -----------------------------------

def plv_matrix(RECORDS: pd.DataFrame,
               channels: List[str],
               band: Tuple[float,float],
               windows: Optional[List[Tuple[float,float]]] = None,
               time_col: str = 'Timestamp') -> np.ndarray:
    """
    Pairwise PLV matrix (N×N) within 'band'. Uses analytic phases from Hilbert
    after band-pass. Windows are concatenated before PLV.
    """
    fs = infer_fs(RECORDS, time_col)
    X = []
    for ch in channels:
        x = get_series(RECORDS, ch)
        x = slice_concat(x, fs, windows)
        xb = bandpass(x, fs, band[0], band[1])
        X.append(np.angle(signal.hilbert(xb)))
    X = np.vstack(X)  # (N, T)
    N = len(channels)
    PLV = np.zeros((N,N))
    for i in range(N):
        for j in range(i, N):
            dphi = X[i]-X[j]
            v = np.abs(np.mean(np.exp(1j*dphi)))
            PLV[i,j]=PLV[j,i]=float(v)
    np.fill_diagonal(PLV, 0.0)
    return PLV

def msc_vs_sr(RECORDS: pd.DataFrame,
              channels: List[str], sr_channel: str,
              windows: Optional[List[Tuple[float,float]]] = None,
              time_col: str = 'Timestamp',
              nperseg: Optional[int] = None, noverlap: Optional[int] = None,
              harmonics: List[float] = (7.83,14.3,20.8,27.3,33.8)) -> pd.DataFrame:
    """
    Per-channel magnitude-squared coherence with SR at harmonics
    (Welch MSC; simple & fast). Returns DataFrame:
      ['channel','MSC_mean','MSC_at_<freq>']
    """
    fs = infer_fs(RECORDS, time_col)
    if nperseg is None:
        nperseg = int(4*fs)
    if noverlap is None:
        noverlap = int(0.5*nperseg)
    y = get_series(RECORDS, sr_channel)
    y = slice_concat(y, fs, windows)

    rows=[]
    for ch in channels:
        x = get_series(RECORDS, ch)
        x = slice_concat(x, fs, windows)
        f, Cxy = signal.coherence(x, y, fs=fs, nperseg=nperseg, noverlap=noverlap)
        row = {'channel': ch}
        vals=[]
        for hf in harmonics:
            idx = int(np.argmin(np.abs(f - hf)))
            row[f"MSC_{hf:.2f}"] = float(Cxy[idx])
            vals.append(float(Cxy[idx]))
        row['MSC_mean'] = float(np.mean(vals))
        rows.append(row)
    return pd.DataFrame(rows)

# --------------------- Graph metrics: entropy & min-cut ---------------------

def laplacian_entropy(adj: np.ndarray) -> float:
    """Shannon entropy of positive Laplacian eigenvalues (normalized)."""
    deg = np.sum(adj, axis=1)
    L = np.diag(deg) - adj
    L = 0.5*(L+L.T)
    vals = np.linalg.eigvalsh(L)
    vals = vals[vals > 1e-12]
    if vals.size == 0: return np.nan
    p = vals / np.sum(vals)
    return float(-np.sum(p*np.log(p)))

def global_mincut(adj: np.ndarray) -> float:
    """Stoer–Wagner global min-cut on weighted undirected graph."""
    G = nx.from_numpy_array(adj)
    if G.number_of_edges()==0: return 0.0
    try:
        cut_val, _ = nx.stoer_wagner(G)
        return float(cut_val)
    except Exception:
        return float('nan')

# ---------------------- 5a) Cross-domain graph alignment --------------------

def cross_domain_graph_alignment(RECORDS: pd.DataFrame,
                                 eeg_channels: List[str],
                                 sr_channel: str,
                                 band: Tuple[float,float] = (8,13),
                                 harmonics: List[float] = (7.83,14.3,20.8,27.3,33.8),
                                 windows: Optional[List[Tuple[float,float]]] = None,
                                 time_col: str = 'Timestamp') -> Dict[str, object]:
    """
    Build EEG PLV graph in 'band'. Weight edges by geometric mean of the nodes'
    MSC with SR at the given harmonics. Compare entropy & min-cut.
    Plots heatmaps and summary bars.
    """
    # PLV matrix
    PLV = plv_matrix(RECORDS, eeg_channels, band, windows, time_col)
    # per-node MSC mean across harmonics
    msc_tbl = msc_vs_sr(RECORDS, eeg_channels, sr_channel, windows, time_col)
    node_msc = {row['channel']: row['MSC_mean'] for _,row in msc_tbl.iterrows()}
    # Weighted edge factor: sqrt(msc_i * msc_j)
    N = len(eeg_channels)
    W = np.zeros((N,N))
    for i,ch_i in enumerate(eeg_channels):
        for j,ch_j in enumerate(eeg_channels):
            if i==j: continue
            fct = np.sqrt(max(0.0, node_msc[ch_i]) * max(0.0, node_msc[ch_j]))
            W[i,j] = fct
    A_plain = PLV.copy()
    A_weight = PLV * W

    # Graph metrics
    ent_plain  = laplacian_entropy(A_plain)
    ent_weight = laplacian_entropy(A_weight)
    cut_plain  = global_mincut(A_plain)
    cut_weight = global_mincut(A_weight)

    # ----- Plots -----
    fig, axs = plt.subplots(1,2, figsize=(10,4))
    im0 = axs[0].imshow(A_plain, vmin=0, vmax=1, cmap='viridis')
    axs[0].set_title(f'PLV adjacency ({band[0]}–{band[1]} Hz)')
    plt.colorbar(im0, ax=axs[0], fraction=0.046)
    im1 = axs[1].imshow(A_weight, vmin=0, vmax=np.nanmax(A_weight)+1e-9, cmap='viridis')
    axs[1].set_title('Schumann-weighted PLV')
    plt.colorbar(im1, ax=axs[1], fraction=0.046)
    for ax in axs:
        ax.set_xticks(range(N)); ax.set_yticks(range(N))
        ax.set_xticklabels([c.split('.',1)[-1] for c in eeg_channels], rotation=90, fontsize=8)
        ax.set_yticklabels([c.split('.',1)[-1] for c in eeg_channels], fontsize=8)
    plt.tight_layout(); plt.show()

    # Δ edge histogram
    dA = A_weight - A_plain
    plt.figure(figsize=(6,3))
    plt.hist(dA[np.triu_indices(N,1)].ravel(), bins=30, color='tab:blue', alpha=0.8)
    plt.xlabel('Δ edge weight (weighted − plain)'); plt.ylabel('count')
    plt.title('Edge weight changes due to Schumann weighting'); plt.tight_layout(); plt.show()

    # Summary bars
    labels = ['Entropy','Min-cut']
    vals_plain  = [ent_plain, cut_plain]
    vals_weight = [ent_weight, cut_weight]
    x = np.arange(2); w = 0.38
    plt.figure(figsize=(6,3))
    plt.bar(x-w/2, vals_plain,  width=w, label='Plain')
    plt.bar(x+w/2, vals_weight, width=w, label='Weighted')
    plt.xticks(x, labels); plt.title('Graph metrics'); plt.legend(); plt.tight_layout(); plt.show()

    summary = pd.DataFrame([{
        'entropy_plain': ent_plain, 'entropy_weighted': ent_weight,
        'mincut_plain': cut_plain, 'mincut_weighted': cut_weight
    }])
    return {'A_plain':A_plain, 'A_weight':A_weight, 'node_msc':msc_tbl, 'summary':summary}

# ---------------------- 5b) Source-space ROI mapping (sensor proxy) ---------

def symmetric_orthogonalize(ts: np.ndarray) -> np.ndarray:
    """
    Symmetric orthogonalization (Colclough et al., 2015) for leakage reduction.
    Input ts: (n_roi, T); output has orthonormal columns in least-squares sense.
    """
    X = ts.T  # (T, n)
    C = X.T @ X
    vals, vecs = np.linalg.eigh(C)
    W = vecs @ np.diag(1.0/np.sqrt(np.maximum(vals, 1e-12))) @ vecs.T
    Y = X @ W
    return Y.T

def roi_time_series(RECORDS: pd.DataFrame,
                    roi_map: Dict[str, List[str]],
                    windows: Optional[List[Tuple[float,float]]],
                    time_col: str = 'Timestamp',
                    orthogonalize: bool = True) -> Tuple[np.ndarray, List[str], float]:
    """
    Build (n_roi, T) ROI matrix by averaging available channels per ROI,
    with optional symmetric orthogonalization across ROIs.
    """
    fs = infer_fs(RECORDS, time_col)
    ts = []
    names = []
    for roi, chans in roi_map.items():
        present = [ch for ch in chans if (ch in RECORDS.columns) or ('EEG.'+ch in RECORDS.columns)]
        if not present: continue
        X = []
        for ch in present:
            x = get_series(RECORDS, ch)
            x = slice_concat(x, fs, windows)
            X.append(x)
        L = min(map(len, X))
        X = np.vstack([x[:L] for x in X])
        ts.append(np.mean(X, axis=0))
        names.append(roi)
    if not ts:
        raise ValueError("No ROI could be formed from the provided mapping.")
    TS = np.vstack(ts)
    if orthogonalize and TS.shape[0] > 1:
        TS = symmetric_orthogonalize(TS)
    return TS, names, fs

def roi_plv_msc_vs_sr(RECORDS: pd.DataFrame,
                      roi_map: Dict[str,List[str]],
                      sr_channel: str,
                      windows: Optional[List[Tuple[float,float]]] = None,
                      time_col: str = 'Timestamp',
                      phase_band: Tuple[float,float] = (7.3, 8.3),
                      harmonics: List[float] = (7.83,14.3,20.8,27.3,33.8),
                      nperseg: Optional[int] = None, noverlap: Optional[int] = None,
                      n_surr: int = 200, rng_seed: int = 17) -> Dict[str, object]:
    """
    ROI-level PLV (phase_band) and MSC (harmonics) vs SR with circular-shift surrogates.
    Returns DataFrame with PLV, MSC_mean, p-values; and plots bars + null bands.
    """
    TS, roi_names, fs = roi_time_series(RECORDS, roi_map, windows, time_col, orthogonalize=True)
    y = get_series(RECORDS, sr_channel)
    y = slice_concat(y, fs, windows)

    # PLV per ROI
    xb = bandpass(y, fs, phase_band[0], phase_band[1])
    ph_y = np.angle(signal.hilbert(xb))
    rows = []
    # coherence via Welch
    if nperseg is None: nperseg = int(4*fs)
    if noverlap is None: noverlap = int(0.5*nperseg)
    fY, _ = signal.welch(y, fs=fs, nperseg=nperseg, noverlap=noverlap)

    # surrogates
    rng = np.random.default_rng(rng_seed)
    for r_idx, roi in enumerate(roi_names):
        x = TS[r_idx]
        # PLV
        xr = bandpass(x, fs, phase_band[0], phase_band[1])
        ph_x = np.angle(signal.hilbert(xr))
        plv = float(np.abs(np.mean(np.exp(1j*(ph_x - ph_y)))))
        # MSC across harmonics (Welch)
        fX, Px = signal.welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
        fC, Cxy = signal.coherence(x, y, fs=fs, nperseg=nperseg, noverlap=noverlap)
        msc_vals=[]
        for hf in harmonics:
            idx = int(np.argmin(np.abs(fC - hf)))
            msc_vals.append(float(Cxy[idx]))
        msc_mean = float(np.mean(msc_vals))

        # surrogate nulls by circular shift of SR
        null_plv=[]; null_msc=[]
        n = len(y)
        for _ in range(n_surr):
            s = int(rng.integers(1, n-1))
            ys = np.r_[y[-s:], y[:-s]]
            # PLV null
            ysb = bandpass(ys, fs, phase_band[0], phase_band[1])
            ph_ys = np.angle(signal.hilbert(ysb))
            null_plv.append(np.abs(np.mean(np.exp(1j*(ph_x - ph_ys)))))
            # MSC null at harmonics
            _, C0 = signal.coherence(x, ys, fs=fs, nperseg=nperseg, noverlap=noverlap)
            vals=[]
            for hf in harmonics:
                idx = int(np.argmin(np.abs(fC - hf)))
                vals.append(float(C0[idx]))
            null_msc.append(np.mean(vals))
        plv_thr = float(np.nanpercentile(null_plv, 95))
        msc_thr = float(np.nanpercentile(null_msc, 95))
        p_plv = float((np.sum(np.array(null_plv) >= plv)+1)/(n_surr+1))
        p_msc = float((np.sum(np.array(null_msc)>= msc_mean)+1)/(n_surr+1))

        rows.append({'ROI':roi, 'PLV':plv, 'PLV_thr95':plv_thr, 'p_PLV':p_plv,
                     'MSC_mean':msc_mean, 'MSC_thr95':msc_thr, 'p_MSC':p_msc})
    df = pd.DataFrame(rows)

    # --- plots ---
    # PLV bars
    plt.figure(figsize=(8,3))
    plt.bar(df['ROI'], df['PLV'], color='tab:blue', alpha=0.9)
    for i,(thr) in enumerate(df['PLV_thr95']):
        plt.plot([i-0.4, i+0.4],[thr,thr], 'k--', lw=1)
    plt.ylabel(f'PLV ({phase_band[0]}–{phase_band[1]} Hz)'); plt.title('ROI PLV vs Schumann (95% null dashed)')
    plt.tight_layout(); plt.show()

    # MSC bars
    plt.figure(figsize=(8,3))
    plt.bar(df['ROI'], df['MSC_mean'], color='tab:orange', alpha=0.9)
    for i,(thr) in enumerate(df['MSC_thr95']):
        plt.plot([i-0.4, i+0.4],[thr,thr], 'k--', lw=1)
    plt.ylabel('MSC (mean across harmonics)'); plt.title('ROI coherence vs Schumann (95% null dashed)')
    plt.tight_layout(); plt.show()

    return {'roi_table': df, 'roi_names': roi_names, 'fs': fs}

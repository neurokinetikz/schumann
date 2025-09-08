"""
Directed Connectivity & Causal Routing — Simple Graphs and Validation Tests
============================================================================

What this module does (turn-key):
1) Builds ROI time series from your EEG channels (or uses your small channel set directly).
2) Fits a VAR model per state (Ignition vs Baseline) on **band-limited** multivariate EEG(+SR).
3) Computes **frequency-domain directed measures** from the VAR:
   • DTF (Directed Transfer Function)  i <- j  (per frequency)
   • PDC (Partial Directed Coherence) i <- j  (per frequency)
4) Summarizes:
   • ROI↔ROI directed matrices (mean within bands).
   • **Brain↔Field** direction at ~7.83 Hz (SR→ROI and ROI→SR), with a **circular-shift null** for SR.
5) Plots:
   • Directed heatmaps (Ignition vs Baseline).
   • Bars for SR→ROI and ROI→SR at 7.83 Hz with **95% null** lines.
   • A simple “net flow” index per ROI:  Σ_j DTF(i<-j) − Σ_j DTF(j<-i)
6) CSV outputs with all summaries and p-values.

Requirements
------------
• pandas, numpy, scipy, matplotlib, networkx
• statsmodels (for VAR). If not present, the module will **skip** VAR‐based parts and print a note.

Inputs
------
RECORDS: pandas.DataFrame with a numeric time column (default 'Timestamp')
and EEG channels named 'EEG.*' (e.g., 'EEG.O1', 'EEG.F4', ...). You can also
include a Schumann reference channel; if not provided, we auto-pick a posterior EEG.

Usage
-----
res = run_directed_connectivity_routing(
    RECORDS,
    eeg_channels=['EEG.F4','EEG.Pz','EEG.O1','EEG.O2'],   # or let roi_map group many sensors
    ignition_windows=[(290,310),(580,600)],
    baseline_windows=[(0,290),(325,580)],
    roi_map=None,                                   # or dict like {'F':['F3','F4','Fz'], 'P':['P3','P4','Pz'], ...}
    sr_channel=None,                                 # None → auto-pick posterior (e.g., Oz)
    bands={'theta':(4,8), 'alpha':(8,13), 'beta':(13,30)},
    f0=7.83,
    time_col='Timestamp',
    out_dir='exports_directed/S01',
    show=True
)
print(res['summary'])
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple, Optional
from scipy import signal

# statsmodels for VAR
try:
    from statsmodels.tsa.api import VAR
    _HAS_SM = True
except Exception:
    _HAS_SM = False

# ---------------------------- utilities ----------------------------

def _ensure_dir(d): os.makedirs(d, exist_ok=True); return d

def infer_fs(RECORDS: pd.DataFrame, time_col: str = 'Timestamp') -> float:
    t = np.asarray(pd.to_numeric(RECORDS[time_col], errors='coerce').values, float)
    dt = np.diff(t); dt = dt[(dt>0) & np.isfinite(dt)]
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

def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x,float); return (x - np.mean(x)) / (np.std(x)+1e-12)

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

def pick_posterior_sr(RECORDS: pd.DataFrame) -> str:
    # prefer Oz, O1, O2, then any EEG.*
    for c in ['EEG.Oz','EEG.O1','EEG.O2']:
        if c in RECORDS.columns: return c
    for c in RECORDS.columns:
        if c.startswith('EEG.'): return c
    raise ValueError("No EEG.* channel found for SR")

# ---------------------------- ROI builder ----------------------------

def make_roi_series(RECORDS: pd.DataFrame,
                    eeg_channels: List[str],
                    roi_map: Optional[Dict[str, List[str]]] = None,
                    windows: Optional[List[Tuple[float,float]]] = None,
                    time_col: str = 'Timestamp') -> Tuple[np.ndarray, List[str], float]:
    """
    Return (X, names, fs) where X is (n_nodes, T) z-scored per node.
    If roi_map is None, treat eeg_channels as nodes.
    """
    fs = infer_fs(RECORDS, time_col)
    if roi_map:
        X=[]; names=[]
        for roi, chans in roi_map.items():
            present=[]
            for ch in chans:
                nm = ch if ch.startswith('EEG.') else 'EEG.'+ch
                if nm in RECORDS.columns:
                    present.append(slice_concat(get_series(RECORDS, nm), fs, windows))
            if present:
                arr = np.vstack(present)
                L = np.min([len(a) for a in arr])
                arr = arr[:,:L]
                X.append(np.mean(arr, axis=0))
                names.append(roi)
        if not X: raise ValueError("No ROI channels found in RECORDS.")
        X = np.vstack([zscore(x) for x in X])
        return X, names, fs
    else:
        # channels directly
        X=[]
        for ch in eeg_channels:
            x = slice_concat(get_series(RECORDS, ch), fs, windows)
            X.append(zscore(x))
        names = [c.split('.',1)[-1] for c in eeg_channels]
        X = np.vstack(X)
        return X, names, fs

# ---------------------------- VAR → DTF/PDC ----------------------------

def fit_var_robust(X: np.ndarray,
                   fs: float,
                   order_max: int = 16,
                   trend: str = 'n',          # no intercept (we z-score / prefilter)
                   ridge: float = 1e-8) -> Tuple[object, Optional[int], Optional[np.ndarray]]:
    """
    Robust VAR(p) fit with manual BIC selection and PD enforcement on Σ_u.
    X: (n_nodes, T)
    Returns (res, p_star, A) where A has shape (p, n, n). If fit fails, returns (None,None,None).
    """
    if not _HAS_SM:
        return None, None, None

    Y = X.T  # (T, n)
    n = Y.shape[1]
    N = Y.shape[0]

    # keep p small relative to samples
    p_cap = max(1, min(order_max, N // max(8, 2*n)))  # conservative cap
    best = None
    best_bic = np.inf
    best_p = None

    model = VAR(Y)
    for p in range(1, p_cap+1):
        try:
            res = model.fit(p, trend=trend)
            # Residual covariance; ridge to ensure PD
            Sigma = 0.5*(res.sigma_u_mle + res.sigma_u_mle.T) + ridge*np.eye(n)
            # Check PD
            if np.any(np.linalg.eigvalsh(Sigma) <= 0):
                continue
            # BIC (manual): −2ℓ + k ln(T)
            T_eff = res.nobs
            # loglike of Gaussian VAR residuals
            ll = -0.5*T_eff * (n*np.log(2*np.pi) + np.log(np.linalg.det(Sigma)) + n)
            k = n*n*p  # parameters (trend excluded)
            bic = -2*ll + k*np.log(max(1,T_eff))
            if bic < best_bic:
                best_bic = bic
                best = res
                best_p = p
        except Exception:
            continue

    if best is None:
        return None, None, None
    A = np.array(best.coefs)  # (p, n, n)
    return best, best_p, A


def A_of_f(A: np.ndarray, f: np.ndarray, fs: float) -> np.ndarray:
    """A(f) = I − Σ_k A_k e^{−i2πfk/fs};  returns (n_f, n, n)."""
    p, n, _ = A.shape
    I = np.eye(n)
    Af = []
    for ff in f:
        Z = I.copy()
        for k in range(1, p+1):
            Z = Z - A[k-1] * np.exp(-1j*2*np.pi*ff * k / fs)
        Af.append(Z)
    return np.array(Af)

def H_of_f(Af: np.ndarray) -> np.ndarray:
    """Transfer matrix H(f) = A(f)^{-1}, per frequency."""
    Hf = np.zeros_like(Af, dtype=complex)
    for i in range(Af.shape[0]):
        Hf[i] = np.linalg.inv(Af[i])
    return Hf

def spectral_dtf_pdc(A: np.ndarray, fs: float,
                     fmin: float, fmax: float, n_freq: int = 128) -> Dict[str, np.ndarray]:
    """
    DTF_{i<-j}(f) = |H_ij| / sqrt(Σ_k |H_ik|^2)   (row-normalized)
    PDC_{i<-j}(f) = |A_ij| / sqrt(Σ_k |A_kj|^2)   (column-normalized A(f))
    Returns dict with 'f','DTF','PDC' arrays of shape (n, n, n_freq).
    """
    n = A.shape[1]
    f = np.linspace(fmin, fmax, n_freq)
    Af = A_of_f(A, f, fs)         # (n_f, n, n)
    Hf = H_of_f(Af)               # (n_f, n, n)
    DTF = np.zeros((n, n, n_freq))
    PDC = np.zeros((n, n, n_freq))
    for k in range(n_freq):
        H = Hf[k]
        num = np.abs(H)**2
        den = np.sum(num, axis=1, keepdims=True) + 1e-24
        DTF[:, :, k] = np.sqrt(num/den)
        A_k = Af[k]
        numA = np.abs(A_k)**2
        denA = np.sum(numA, axis=0, keepdims=True) + 1e-24
        PDC[:, :, k] = np.sqrt(numA/denA)
    return {'f': f, 'DTF': DTF, 'PDC': PDC}

def band_average(M: np.ndarray, f: np.ndarray, band: Tuple[float,float]) -> np.ndarray:
    """Mean over frequency indices in band; M shape (n,n,n_f)."""
    sel = (f>=band[0]) & (f<=band[1])
    if not np.any(sel): sel = [np.argmin(np.abs(f - np.mean(band)))]
    return np.nanmean(M[:, :, sel], axis=2)

# ---------------------------- Nulls for SR causality ----------------------------

def circular_shift_sr_null(RECORDS: pd.DataFrame, sr_channel: str, fs: float,
                           windows: List[Tuple[float,float]],
                           nodes: List[str], roi_map: Optional[Dict[str,List[str]]],
                           band: Tuple[float,float],
                           order_max: int, n_surr: int = 100) -> float:
    """
    Build null distribution for SR→ROI DTF at ~f0 by circularly shifting SR and refitting VAR.
    Returns the 95th percentile (threshold).
    """
    rng = np.random.default_rng(11)
    vals=[]
    for _ in range(n_surr):
        # shift SR only
        sr = slice_concat(get_series(RECORDS, sr_channel), fs, windows)
        s = int(rng.integers(1, len(sr)-1))
        sr_sh = np.r_[sr[-s:], sr[:-s]]
        # build node matrix with shifted SR (last node)
#         if roi_map:
#             Xroi, names, _ = make_roi_series(RECORDS, nodes, roi_map, windows, time_col='Timestamp')
#         else:
#             Xroi, names, _ = make_roi_series(RECORDS, nodes, None, windows, time_col='Timestamp')

        # build ROI
        Xroi, names, _ = make_roi_series(RECORDS, nodes, roi_map, windows, time_col='Timestamp')
        sr  = slice_concat(get_series(RECORDS, sr_channel), fs, windows)

        # align lengths
        L = min(Xroi.shape[1], sr.shape[0])
        Xroi = Xroi[:, :L]
        sr   = sr[:L]

        # shift only SR
        s = int(rng.integers(1, L-1))
        sr_sh = np.r_[sr[-s:], sr[:-s]]

        # prefilter & z-score (same preband as main path)
        preband = (2.0, 45.0)
        for i in range(Xroi.shape[0]):
            Xroi[i] = bandpass(Xroi[i], fs, preband[0], preband[1])
        sr_sh = bandpass(sr_sh, fs, preband[0], preband[1])

        Xroi = (Xroi - Xroi.mean(axis=1, keepdims=True)) / (Xroi.std(axis=1, keepdims=True)+1e-12)
        sr_z = (sr_sh - sr_sh.mean())/(sr_sh.std()+1e-12)

        X = np.vstack([Xroi, sr_z[None, :]])
        X += 1e-6 * rng.standard_normal(X.shape)


        # append SR
        X = np.vstack([Xroi, zscore(sr_sh)])
        if not _HAS_SM: return np.nan
        res, p, A = fit_var(X, order_max=order_max)
        if A is None: continue
        spec = spectral_dtf_pdc(A, fs, fmin=band[0], fmax=band[1], n_freq=64)
        D = band_average(spec['DTF'], spec['f'], band=(band[0],band[1]))
        # SR is last node
        sr_idx = X.shape[0]-1
        inbound = np.nanmean(D[:-1, sr_idx])   # ROI <- SR
        vals.append(inbound)
    return float(np.nanpercentile(vals, 95)) if vals else np.nan

# ---------------------------- Runner ----------------------------

def run_directed_connectivity_routing(
    RECORDS: pd.DataFrame,
    eeg_channels: List[str],
    ignition_windows: Optional[List[Tuple[float,float]]],
    baseline_windows: Optional[List[Tuple[float,float]]],
    roi_map: Optional[Dict[str, List[str]]] = None,
    sr_channel: Optional[str] = None,
    bands: Dict[str, Tuple[float,float]] = None,
    f0: float = 7.83,
    time_col: str = 'Timestamp',
    out_dir: str = 'exports_directed/session',
    show: bool = True,
    order_max: int = 16,
    n_surr: int = 100
) -> Dict[str, object]:
    """
    Map directed connectivity inside the brain and between brain and field (SR).
    Produces heatmaps and SR→ROI bars with 95% null thresholds, plus CSV summaries.
    """
    _ensure_dir(out_dir)
    fs = infer_fs(RECORDS, time_col)
    bands = bands or {'theta':(4,8),'alpha':(8,13),'beta':(13,30)}
    sr_channel = sr_channel or pick_posterior_sr(RECORDS)
    # node labels
    node_names = list(roi_map.keys()) if roi_map else [c.split('.',1)[-1] for c in eeg_channels]
    # states
    states = {'ignition': ignition_windows, 'baseline': baseline_windows}

    results = {}
    for st, wins in states.items():
        # --- ROI+SR matrix for this state (wins) ---
        Xroi, names, _ = make_roi_series(RECORDS, eeg_channels, roi_map, wins, time_col=time_col)
        sr  = slice_concat(get_series(RECORDS, sr_channel), fs, wins)

        # length-align
        L = min(Xroi.shape[1], sr.shape[0])
        Xroi = Xroi[:, :L]; sr = sr[:L]

        # prefilter (stabilize)
        preband = (2.0, 45.0)
        for i in range(Xroi.shape[0]):
            Xroi[i] = bandpass(Xroi[i], fs, preband[0], preband[1])
        sr = bandpass(sr, fs, preband[0], preband[1])

        # z-score & tiny jitter
        Xroi = (Xroi - Xroi.mean(axis=1, keepdims=True)) / (Xroi.std(axis=1, keepdims=True)+1e-12)
        sr_z = (sr - sr.mean())/(sr.std()+1e-12)
        rng  = np.random.default_rng(5)
        X    = np.vstack([Xroi, sr_z[None,:]])
        X   += 1e-6 * rng.standard_normal(X.shape)

        # ---------- try safeguarded VAR ----------
        res, p, A = fit_var_safeguarded(X, fs, order_max=order_max)
        if A is None:
            # ---------- PCA fallback ----------
            K = max(2, min(4, Xroi.shape[0]//2, (Xroi.shape[1]//20)))  # 2..4 comps; keep sane w.r.t. T
            if K >= 2:
                Z, U = pca_reduce_nodes(Xroi, k=K)             # (K, T)
                Xp = np.vstack([Z, sr_z[None,:]])              # PCs + SR
                Xp += 1e-6 * rng.standard_normal(Xp.shape)
                res, p, A = fit_var_safeguarded(Xp, fs, order_max=min(order_max, 8))
                if A is not None:
                    # compute DTF/PDC on PCs
                    spec = spectral_dtf_pdc(A, fs, fmin=2.0, fmax=45.0, n_freq=256)
                    band_mats = {bn: band_average(spec['DTF'], spec['f'], bnd) for bn,bnd in bands.items()}
                    band0 = (max(2.0, f0-0.6), min(45.0, f0+0.6))
                    D0 = band_average(spec['DTF'], spec['f'], band0)
                    sr_idx = Xp.shape[0]-1
                    inbound_SR = D0[:-1, sr_idx]; outbound_SR = D0[sr_idx, :-1]
                    # NOTE: labels are PCs (PC1..PCk) not ROIs when in PCA fallback
                    pc_names = [f'PC{k+1}' for k in range(K)] + ['SR']
                    results[st] = {'names': pc_names, 'DTF_spec': spec, 'DTF_bands': band_mats,
                                   'DTF_SR_in': inbound_SR, 'DTF_SR_out': outbound_SR,
                                   'thr95_SR_in': np.nan, 'order': p, 'mode': 'PCA'}
                else:
                    # ---------- bivariate Granger fallback ----------
                    F = granger_bivariate_matrix(X, maxlag=6)   # on nodes+SR
                    names_sr = names + ['SR']
                    results[st] = {'names': names_sr, 'BIV_F': F, 'mode': 'BIV'}
            else:
                # ---------- bivariate fallback directly ----------
                F = granger_bivariate_matrix(X, maxlag=6)
                names_sr = names + ['SR']
                results[st] = {'names': names_sr, 'BIV_F': F, 'mode': 'BIV'}
            # plotting for fallback handled below
            continue

        # ---------- (normal) DTF/PDC path ----------
        spec = spectral_dtf_pdc(A, fs, fmin=2.0, fmax=45.0, n_freq=256)
        band_mats = {bn: band_average(spec['DTF'], spec['f'], bnd) for bn,bnd in bands.items()}
        band0 = (max(2.0, f0-0.6), min(45.0, f0+0.6))
        D0 = band_average(spec['DTF'], spec['f'], band0)
        sr_idx = X.shape[0]-1
        inbound_SR  = D0[:-1, sr_idx]
        outbound_SR = D0[sr_idx, :-1]
        thr95 = circular_shift_sr_null(RECORDS, sr_channel, fs, wins, eeg_channels, roi_map, band0, order_max=order_max, n_surr=n_surr)
        results[st] = {'names': names + ['SR'], 'DTF_spec': spec, 'DTF_bands': band_mats,
                       'DTF_SR_in': inbound_SR, 'DTF_SR_out': outbound_SR, 'thr95_SR_in': thr95,
                       'order': p, 'mode': 'VAR'}

    # ---- Delta matrices (Ign − Base), handle VAR/PCA vs BIV gracefully ----
    if 'ignition' in results and 'baseline' in results:
        mode_i = results['ignition'].get('mode', 'VAR')
        mode_b = results['baseline'].get('mode', 'VAR')

        # helper to align names/matrices if lengths differ
        def _align(A, B):
            n = min(A.shape[0], B.shape[0])
            return A[:n,:n], B[:n,:n]

        names_i = results['ignition']['names']
        names_b = results['baseline']['names']
        names = names_i if names_i == names_b else [f'N{k+1}' for k in range(min(len(names_i), len(names_b)))]

        if ('DTF_bands' in results['ignition']) and ('DTF_bands' in results['baseline']):
            # VAR (or PCA) available in BOTH states
            for bn in bands.keys():
                if (bn in results['ignition']['DTF_bands']) and (bn in results['baseline']['DTF_bands']):
                    Bi = results['ignition']['DTF_bands'][bn]
                    Bb = results['baseline']['DTF_bands'][bn]
                    Bi, Bb = _align(Bi, Bb)
                    dM = Bi - Bb
                    plt.figure(figsize=(5.2,4.4))
                    im = plt.imshow(dM, cmap='bwr', vmin=-np.max(np.abs(dM)), vmax=np.max(np.abs(dM)))
                    plt.colorbar(im, label='ΔDTF (Ign − Base)')
                    plt.xticks(range(len(names)), names, rotation=90, fontsize=8)
                    plt.yticks(range(len(names)), names, fontsize=8)
                    title_mode = results['ignition'].get('mode','VAR')
                    plt.title(f'Δ Directed flow — {bn} ({title_mode})')
                    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f'dtf_delta_{bn}.png'), dpi=140)
                    if show: plt.show()
                    plt.close()
                else:
                    print(f"[INFO] Band '{bn}' not present in both states; skipping Δ for that band.")
        elif ('BIV_F' in results['ignition']) and ('BIV_F' in results['baseline']):
            # both are bivariate Granger
            Fi = results['ignition']['BIV_F']; Fb = results['baseline']['BIV_F']
            Fi, Fb = _align(Fi, Fb)
            dM = Fi - Fb
            plt.figure(figsize=(5.2,4.4))
            im = plt.imshow(dM, cmap='bwr', vmin=-np.max(np.abs(dM)), vmax=np.max(np.abs(dM)))
            plt.colorbar(im, label='Δ Pairwise Granger (norm.)')
            plt.xticks(range(len(names)), names, rotation=90, fontsize=8)
            plt.yticks(range(len(names)), names, fontsize=8)
            plt.title('Δ Pairwise Granger (Ign − Base)')
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'granger_biv_delta.png'), dpi=140)
            if show: plt.show()
            plt.close()
        else:
            # Mixed modes (e.g., VAR/PCA in one state, BIV in the other) — we can’t compute ΔDTF cleanly
            print("[INFO] Mixed modes across states (e.g., VAR vs BIV) — skipping Δ matrix.")


    # ===== Plotting for PCA or BIV fallbacks =====
    if results.get(st, {}).get('mode') == 'PCA':
        names_local = results[st]['names']
        for bn,bmat in results[st]['DTF_bands'].items():
            plt.figure(figsize=(5.2,4.4))
            im = plt.imshow(bmat, vmin=0, vmax=1, cmap='magma')
            plt.colorbar(im, label='DTF (PC space)')
            plt.xticks(range(len(names_local)), names_local, rotation=90, fontsize=8)
            plt.yticks(range(len(names_local)), names_local, fontsize=8)
            plt.title(f'DTF {bn} — {st} (PCA fallback)')
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, f'dtf_{bn}_{st}_pca.png'), dpi=140)
            if show: plt.show()
            plt.close()

    if results.get(st, {}).get('mode') == 'BIV':
        F = results[st]['BIV_F']
        names_local = results[st]['names']
        plt.figure(figsize=(5.2,4.4))
        im = plt.imshow(F, vmin=0, vmax=1, cmap='inferno')
        plt.colorbar(im, label='Bivariate Granger (norm.)')
        plt.xticks(range(len(names_local)), names_local, rotation=90, fontsize=8)
        plt.yticks(range(len(names_local)), names_local, fontsize=8)
        plt.title(f'Pairwise Granger matrix — {st}')
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f'granger_biv_{st}.png'), dpi=140)
        if show: plt.show()
        plt.close()



    # ---- Summary CSVs ----
    summary_rows=[]
    for st in results:
        r = results[st]
        names_local = r['names']
        mode_local  = r.get('mode','VAR')
        if 'DTF_bands' in r and 'alpha' in r['DTF_bands']:
            M = r['DTF_bands']['alpha']
            n = len(names_local)-1 if names_local[-1] == 'SR' else len(names_local)
            brain = M[:n,:n]
            net = np.sum(brain, axis=1) - np.sum(brain, axis=0)
            for i in range(n):
                summary_rows.append({'state': st, 'mode': mode_local, 'node': names_local[i],
                                     'net_flow_alpha': float(net[i])})
        elif 'BIV_F' in r:
            # summarize pairwise Granger as a fallback
            F = r['BIV_F']
            n = len(names_local)-1 if names_local[-1] == 'SR' else len(names_local)
            net = np.sum(F[:n,:n], axis=1) - np.sum(F[:n,:n], axis=0)
            for i in range(n):
                summary_rows.append({'state': st, 'mode': mode_local, 'node': names_local[i],
                                     'net_flow_biv': float(net[i])})
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir,'summary.csv'), index=False)


    return {'results': results, 'out_dir': out_dir}

# ---------- PCA reduction (nodes -> PCs) ----------
def pca_reduce_nodes(X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    X: (n_nodes, T) z-scored. Returns (Z, U) where
    Z: (k, T) top-k component time-series, U: (n_nodes, k) loadings (orthonormal).
    """
    Xc = X - X.mean(axis=1, keepdims=True)
    # SVD of node covariance (fast & stable)
    C = Xc @ Xc.T / Xc.shape[1]         # (n, n)
    vals, vecs = np.linalg.eigh(C)
    idx = np.argsort(vals)[::-1][:k]
    U = vecs[:, idx]                    # loadings (n, k)
    Z = U.T @ Xc                        # (k, T)
    return Z, U

# ---------- time-domain pairwise (bivariate) Granger ----------
def granger_bivariate_matrix(X: np.ndarray, maxlag: int = 6) -> np.ndarray:
    """
    X: (n_nodes, T) z-scored.
    Returns F-stat matrix F_{i<-j} at the best lag (1..maxlag) per pair.
    (Time-domain Granger strength; simple validation fallback.)
    """
    n, T = X.shape
    F = np.zeros((n, n))
    for i in range(n):
        yi = X[i]
        for j in range(n):
            if i == j: continue
            yj = X[j]
            bestF = 0.0
            for p in range(1, maxlag+1):
                # build regressors
                # restricted: yi_t ~ [yi_{t-1..t-p}]
                # unrestricted: yi_t ~ [yi_{t-1..t-p}, yj_{t-1..t-p}]
                Y = yi[p:]
                Phi_i = np.column_stack([yi[p-k:-k] for k in range(1, p+1)])
                Phi_ij= np.column_stack([Phi_i] + [yj[p-k:-k] for k in range(1, p+1)])
                # LS
                beta_i  = np.linalg.lstsq(Phi_i,  Y, rcond=None)[0]
                beta_ij = np.linalg.lstsq(Phi_ij, Y, rcond=None)[0]
                rss_i   = np.sum((Y - Phi_i  @ beta_i )**2)
                rss_ij  = np.sum((Y - Phi_ij @ beta_ij)**2)
                k_num = p          # extra params
                k_den = len(Y) - 2*p
                if k_den <= 0 or rss_ij <= 0: continue
                Fp = ((rss_i - rss_ij)/k_num) / (rss_ij / k_den)
                if np.isfinite(Fp) and Fp > bestF:
                    bestF = Fp
            F[i, j] = bestF
    # normalize to 0..1 for plotting
    F = F / (np.nanmax(F) + 1e-12)
    return F

# ---------- robust VAR selector (replaces fit_var_robust earlier) ----------
def fit_var_safeguarded(X: np.ndarray,
                        fs: float,
                        order_max: int = 16,
                        ridge: float = 1e-8) -> Tuple[object, Optional[int], Optional[np.ndarray]]:
    """
    Adaptive lag cap from data length; manual BIC; PD enforcement on Sigma_u.
    Tries p in small range. If all fail, returns (None,None,None).
    """
    if not _HAS_SM: return None, None, None
    Y = X.T
    n = Y.shape[1]; N = Y.shape[0]
    # conservative lag cap: ensure N >> n*p
    p_cap = max(1, min(order_max, N // max(10, 3*n)))
    p_grid = list(range(1, min(8, p_cap)+1))  # try small lags first
    best_res = None; best_p = None; best_bic = np.inf
    model = VAR(Y)
    for p in p_grid:
        try:
            res = model.fit(p, trend='n')
            # ridge PD
            S = 0.5*(res.sigma_u_mle + res.sigma_u_mle.T) + ridge*np.eye(n)
            if np.any(np.linalg.eigvalsh(S) <= 0): continue
            T_eff = res.nobs
            ll = -0.5*T_eff * (n*np.log(2*np.pi) + np.log(np.linalg.det(S)) + n)
            k = n*n*p
            bic = -2*ll + k*np.log(max(1, T_eff))
            if bic < best_bic:
                best_bic = bic; best_res = res; best_p = p
        except Exception:
            continue
    if best_res is None:
        return None, None, None
    A = np.array(best_res.coefs)  # (p, n, n)
    return best_res, best_p, A

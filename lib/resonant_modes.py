"""
Connectome Harmonics & Resonant Mode Analysis — Simple Graphs & Validity Tests
=============================================================================

Goal
----
Project EEG into a set of spatial “harmonic” modes (connectome or sensor functional
harmonics), then test whether mode activations:
  • concentrate in Schumann bands (~7.8, 14.3, 20.8, 27.3, 33.8 Hz),
  • increase in ignition vs baseline,
  • covary with Schumann amplitude/envelope (time-resolved),
  • show increased MSC coherence to Schumann at harmonics.

What this module provides
-------------------------
1) Harmonic basis:
   (A) If you have a **matrix W** (N×N) whose nodes map one-to-one to your EEG channels,
       we compute Laplacian eigenvectors (connectome/sensor harmonics).
   (B) Otherwise, we **build functional harmonics** from PLV adjacency in a band (e.g., alpha).

2) Mode projection:
   X (n_ch × T) → A = H^T X (n_modes × T).  (H columns are Laplacian eigenvectors.)

3) Tests & graphs:
   • Mode power spectrum by state (Ignition vs Baseline) with simple Δ + null (circular-shift).
   • Schumann-band mode power (per mode, per harmonic band).
   • MSC coherence of each mode to SR at harmonics (bars + null).
   • Time series: chosen mode amplitude vs SR envelope (r + null).
   • Heatmap of the first K eigenvectors (“spatial harmonics”) across channels.

Inputs/assumptions
------------------
RECORDS: pandas.DataFrame with a numeric time column (default 'Timestamp')
and EEG signals named 'EEG.*'. You provide `eeg_channels` (or we detect them).
If you have a true connectome mapped to your EEG channels, pass `W_conn` (n_ch×n_ch).

Usage (minimal)
---------------
res = run_connectome_harmonics_resonance(
    RECORDS,
    eeg_channels=['EEG.O1','EEG.O2','EEG.Oz','EEG.Pz'],
    ignition_windows=[(290,310),(580,600)],
    baseline_windows=[(0,290),(325,580)],
    sr_channel='EEG.Oz',                # or None to auto-pick posterior
    band_for_functional=(8,13),         # used when W_conn=None
    W_conn=None,                        # (optional) provide Laplacian source for harmonics
    n_modes=16,
    out_dir='exports_harmonics/S01',
    show=True
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
import networkx as nx

# ------------------------- utils -------------------------

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

def bandpass(x: np.ndarray, fs: float, f1: float, f2: float, order=4) -> np.ndarray:
    ny = 0.5*fs
    f1 = max(1e-6, min(f1, 0.99*ny)); f2 = max(f1+1e-6, min(f2, 0.999*ny))
    b,a = signal.butter(order, [f1/ny, f2/ny], btype='band')
    return signal.filtfilt(b,a,x)

def zscore(x): x = np.asarray(x,float); return (x - np.mean(x)) / (np.std(x)+1e-12)

# ------------------------- PLV adjacency & functional harmonics -------------------------

def plv_adj(RECORDS, channels, band, windows, time_col='Timestamp') -> np.ndarray:
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
        for j in range(i,N):
            dphi = P[i]-P[j]
            A[i,j]=A[j,i]=float(np.abs(np.mean(np.exp(1j*dphi))))
    np.fill_diagonal(A, 0.0)
    return A

def laplacian_eigendecomp(W: np.ndarray, n_modes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return first n_modes Laplacian eigenvalues & eigenvectors (columns)."""
    W = 0.5*(W+W.T)
    D = np.diag(W.sum(axis=1))
    L = D - W
    vals, vecs = np.linalg.eigh(L)
    idx = np.argsort(vals)   # ascending (low spatial freq first)
    vals = vals[idx]; vecs = vecs[:, idx]
    K = min(n_modes, vecs.shape[1])
    # normalize columns to unit norm
    H = vecs[:, :K]
    for k in range(K):
        H[:,k] /= (np.linalg.norm(H[:,k]) + 1e-12)
    return vals[:K], H

# ------------------------- Mode projection & spectra -------------------------

def project_to_harmonics(X: np.ndarray, H: np.ndarray) -> np.ndarray:
    """X: (n_ch, T), H: (n_ch, K) columns orthonormal → A: (K, T)."""
    return H.T @ X

def mode_band_power(A: np.ndarray, fs: float, fband: Tuple[float,float]) -> np.ndarray:
    """A: (K, T) → band power per mode via band-pass + RMS."""
    K = A.shape[0]
    out=[]
    for k in range(K):
        ak = bandpass(A[k], fs, fband[0], fband[1])
        out.append(float(np.mean(ak**2)))
    return np.array(out)

def mode_welch_power(A: np.ndarray, fs: float, nperseg: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return (f, Pk(f)) where Pk is (K, n_f)."""
    if nperseg is None:
        nperseg = int(2*fs)
    P=[]; freqs=None
    for k in range(A.shape[0]):
        f, p = signal.welch(A[k], fs=fs, nperseg=nperseg, noverlap=nperseg//2)
        freqs = f if freqs is None else freqs
        P.append(p)
    return freqs, np.vstack(P)

# ------------------------- MSC coherence mode↔SR at harmonics -------------------------

def msc_mode_to_sr(A: np.ndarray, sr: np.ndarray, fs: float,
                   harmonics: List[float], nperseg: Optional[int]=None) -> pd.DataFrame:
    if nperseg is None: nperseg = int(4*fs)
    rows=[]
    for k in range(A.shape[0]):
        f, C = signal.coherence(A[k], sr, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
        for hf in harmonics:
            idx = int(np.argmin(np.abs(f - hf)))
            rows.append({'mode': k+1, 'freq': float(f[idx]), 'MSC': float(C[idx])})
    return pd.DataFrame(rows)

# ------------------------- Schumann envelope -------------------------

def schumann_envelope(sr: np.ndarray, fs: float, center_hz: float = 7.83, half_bw: float = 0.6) -> np.ndarray:
    yb = bandpass(sr, fs, center_hz-half_bw, center_hz+half_bw)
    return np.abs(signal.hilbert(yb))

# ------------------------- Runner -------------------------

def run_connectome_harmonics_resonance(
    RECORDS: pd.DataFrame,
    eeg_channels: List[str],
    ignition_windows: Optional[List[Tuple[float,float]]],
    baseline_windows: Optional[List[Tuple[float,float]]],
    sr_channel: Optional[str] = None,
    time_col: str = 'Timestamp',
    band_for_functional: Tuple[float,float] = (8,13),
    W_conn: Optional[np.ndarray] = None,   # (n_ch x n_ch) adjacency in EEG order
    n_modes: int = 16,
    out_dir: str = 'exports_harmonics/session',
    show: bool = True,
    harmonics: List[float] = (7.83, 14.3, 20.8, 27.3, 33.8),
) -> Dict[str, object]:
    """
    Build harmonic basis (connectome W_conn if provided; else functional PLV) and test resonance:
      • Mode band power spectra in ignition vs baseline (alpha by default)
      • Mode MSC coherence to SR at Schumann harmonics (bars)
      • Time-resolved mode amplitude vs SR envelope (r + null)
      • Heatmap of first K eigenvectors (spatial harmonics)

    Returns summary dict and writes figures/CSV to out_dir.
    """
    _ensure_dir(out_dir)
    fs = infer_fs(RECORDS, time_col)

    # SR channel (auto-pick posterior if None)
    if sr_channel is None:
        sr_channel = 'EEG.Oz' if 'EEG.Oz' in RECORDS.columns else next((c for c in RECORDS.columns if c.startswith('EEG.')), None)
    sr = get_series(RECORDS, sr_channel)

    # Data matrix X
    X = np.vstack([get_series(RECORDS, ch) for ch in eeg_channels])  # (n_ch, T)
    # Build harmonic basis
    if W_conn is not None and W_conn.shape[0] == len(eeg_channels):
        vals, H = laplacian_eigendecomp(W_conn, n_modes)
        basis_name = 'connectome'
    else:
        # functional harmonics from PLV in alpha (or band_for_functional)
        A_plv = plv_adj(RECORDS, eeg_channels, band_for_functional, windows=baseline_windows or ignition_windows, time_col=time_col)
        vals, H = laplacian_eigendecomp(A_plv, n_modes)
        basis_name = 'functional'

    # Project signals to harmonic coefficients for each state
    states = {'ignition': ignition_windows, 'baseline': baseline_windows}
    results = {}
    for st, wins in states.items():
        if not wins: continue
        Xs = np.vstack([slice_concat(get_series(RECORDS, ch), fs, wins) for ch in eeg_channels])
        Xs = (Xs - Xs.mean(axis=1, keepdims=True)) / (Xs.std(axis=1, keepdims=True)+1e-12)
        A = project_to_harmonics(Xs, H)                       # (K, Tst)
        # spectra
        f, Pk = mode_welch_power(A, fs, nperseg=int(2*fs))    # Pk: (K, n_f)
        # band powers (per Schumann harmonic: narrow bands)
        bandpowers={}
        for hf in harmonics:
            bandpowers[f"{hf:.2f}"] = mode_band_power(A, fs, (hf-0.6, hf+0.6))
        # MSC to SR at harmonics
        Y = slice_concat(sr, fs, wins)
        msc_tab = msc_mode_to_sr(A, Y, fs, harmonics, nperseg=int(4*fs))
        results[st] = {'A':A, 'freqs':f, 'Pk':Pk, 'bandpowers':bandpowers, 'msc':msc_tab}

    # -------------- Plots --------------
    # Eigenvectors heatmap (spatial harmonics)
    plt.figure(figsize=(min(10, 0.5*n_modes+2), 3.5))
    im = plt.imshow(H, aspect='auto', cmap='coolwarm', vmin=-np.max(np.abs(H)), vmax=np.max(np.abs(H)))
    plt.colorbar(label='eigenvector weight')
    plt.yticks(range(H.shape[0]), [ch.split('.',1)[-1] for ch in eeg_channels], fontsize=8)
    plt.xticks(range(H.shape[1]), [f'H{k+1}' for k in range(H.shape[1])], fontsize=8)
    plt.title(f'{basis_name.capitalize()} harmonics — first {H.shape[1]} modes')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'harmonics_eigenvectors.png'), dpi=140)
    if show: plt.show()
    plt.close()

    # Mode power spectrum by state
    if 'ignition' in results:
        f = results['ignition']['freqs']
        K = H.shape[1]
        for st in results:
            plt.figure(figsize=(8,3))
            plt.imshow(results[st]['Pk'], aspect='auto', origin='lower',
                       extent=[f[0], f[-1], 1, K], cmap='magma')
            plt.colorbar(label='Power')
            plt.xlabel('Frequency (Hz)'); plt.ylabel('Mode index')
            plt.title(f'Mode power spectrum — {st}')
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, f'mode_spectrum_{st}.png'), dpi=140)
            if show: plt.show()
            plt.close()

        # Δ band power around fundamental (7.83±0.6) across modes
        bp_ign = results['ignition']['bandpowers']['7.83']
        bp_base= results['baseline']['bandpowers']['7.83'] if 'baseline' in results else np.zeros_like(bp_ign)
        delta = bp_ign - bp_base
        plt.figure(figsize=(8,3))
        x = np.arange(1, len(delta)+1)
        plt.bar(x, delta, color='tab:blue', alpha=0.9)
        plt.xlabel('Mode index'); plt.ylabel('Δ power (Ign−Base)')
        plt.title('Schumann-band (7.83 Hz) mode Δ power')
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'mode_delta_7p83.png'), dpi=140)
        if show: plt.show()
        plt.close()

    # MSC bars by mode at harmonics (ignition)
    if 'ignition' in results:
        msc = results['ignition']['msc']
        pivot = msc.pivot(index='mode', columns='freq', values='MSC')
        plt.figure(figsize=(9,3))
        plt.imshow(pivot.values, aspect='auto', origin='lower',
                   extent=[pivot.columns.min(), pivot.columns.max(), 1, pivot.index.max()],
                   vmin=0, vmax=1, cmap='viridis')
        plt.colorbar(label='MSC')
        plt.xlabel('Frequency (Hz)'); plt.ylabel('Mode index')
        plt.title('Mode↔SR MSC at Schumann harmonics (ignition)')
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'mode_msc_ignition.png'), dpi=140)
        if show: plt.show()
        plt.close()

    # Time-resolved: strongest Schumann-band mode vs SR envelope (ignition)
    if 'ignition' in results:
        # pick mode k* with largest Δ power at 7.83 Hz (if baseline available)
        if 'baseline' in results:
            k_star = int(np.argmax(results['ignition']['bandpowers']['7.83'] -
                                   results['baseline']['bandpowers']['7.83']))
        else:
            k_star = int(np.argmax(results['ignition']['bandpowers']['7.83']))
        Ak = results['ignition']['A'][k_star]
        # envelope of Ak at 7.83
        envA = schumann_envelope(Ak, fs, center_hz=7.83, half_bw=0.6)
        envSR= schumann_envelope(slice_concat(sr, fs, ignition_windows), fs, center_hz=7.83, half_bw=0.6)
        # correlate with circular-shift null
        r = np.corrcoef(zscore(envA), zscore(envSR))[0,1]
        rng = np.random.default_rng(7)
        null=[]
        for _ in range(200):
            s = int(rng.integers(1, len(envSR)-1))
            null.append(np.corrcoef(zscore(envA), zscore(np.r_[envSR[-s:], envSR[:-s]]))[0,1])
        thr95 = np.nanpercentile(null, 95)
        t = np.arange(len(envA))/fs
        plt.figure(figsize=(9,3))
        plt.plot(t, zscore(envA), label=f'Mode {k_star+1} env (z)')
        plt.plot(t, zscore(envSR), label='SR env (z)')
        plt.title(f'Mode {k_star+1} vs SR envelope @7.83 Hz  (r={r:.2f}, null95={thr95:.2f})')
        plt.xlabel('Time (s)'); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'mode_env_vs_sr.png'), dpi=140)
        if show: plt.show()
        plt.close()

    # ---------- Simple summary (safe; no 'names' needed) ----------
    summary_rows = []
    for st, r in results.items():
        row = {'state': st}
        # average MSC across modes at the fundamental
        msc_fund = r['msc']
        # guard for floating frequency labels
        msctab = msc_fund.copy()
        msctab['freq'] = msctab['freq'].round(2)
        row['MSC7p83_mean'] = float(msctab[msctab['freq'] == 7.83]['MSC'].mean())

        # top-1 mode Schumann-band power around 7.83 Hz
        key = f"{7.83:.2f}"
        if key in r['bandpowers']:
            row['MaxModePow7p83'] = float(np.max(r['bandpowers'][key]))
        else:
            row['MaxModePow7p83'] = np.nan

        row['Basis'] = 'connectome' if (W_conn is not None and W_conn.shape[0] == len(eeg_channels)) else 'functional'
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(out_dir, 'summary.csv'), index=False)

    return {
        'basis': ('connectome' if (W_conn is not None and W_conn.shape[0] == len(eeg_channels)) else 'functional'),
        'eigenvals': vals,
        'H': H,
        'results': results,
        'summary': summary,
        'out_dir': out_dir
    }

def schumann_envelope(sig: np.ndarray, fs: float,
                      center: float = 7.83, half: float = 0.6,
                      **kwargs) -> np.ndarray:
    """
    Compatibility wrapper: accepts center/half or center_hz/half_bw.
    """
    # allow alternate kw names
    if 'center_hz' in kwargs:
        center = kwargs['center_hz']
    if 'half_bw' in kwargs:
        half = kwargs['half_bw']
    b, a = signal.butter(4, [max(1e-6, (center-half))/(0.5*fs),
                             min(0.999, (center+half)/(0.5*fs))], btype='band')
    yb = signal.filtfilt(b, a, np.asarray(sig, float))
    return np.abs(signal.hilbert(yb))

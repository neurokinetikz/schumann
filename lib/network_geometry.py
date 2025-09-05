from __future__ import annotations

"""
EEG Network Geometry & Manifold Embedding (RECORDS)
Extended: Multi‑band batch runs, sliding‑window animations, and CSV/JSON exports.

Drop‑in module expecting a pandas.DataFrame named `RECORDS` with a time column
(default: 'Timestamp') and EEG channel columns (e.g., 'EEG.F4', 'EEG.O1', ...).

New in this version
===================
• Built‑in canonical band definitions + batch multi‑band geometry runner.
• Sliding‑window manifold animation (PNG frames; auto‑GIF if `imageio` is present).
• Simple exporters for plots and tabular/JSON summaries.

Public entry points
===================
- run_multi_band_geometry_records(...): returns a per‑band metrics DataFrame.
- animate_embedding_over_time_records(...): saves frames/GIF of evolving embeddings.
- save_session_report_csv_json(report, base_path): writes report.{csv,json}.
- save_embedding_plot(emb_dict, filename): saves 2D/3D scatter of embeddings.
- session_report_records(...): unchanged API from base module (included below).

Author: ChatGPT (GPT-5 Thinking)
License: MIT
"""

import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Iterable
from scipy import signal
import networkx as nx

# ---------------- Optional libs (feature flags) ----------------
_HAS_MNE_CONN = True
try:
    import mne_connectivity as mne_conn  # type: ignore
except Exception:
    _HAS_MNE_CONN = False

_HAS_SKLEARN = True
try:
    from sklearn.decomposition import PCA  # type: ignore
    from sklearn.manifold import TSNE  # type: ignore
    import umap  # type: ignore
except Exception:
    _HAS_SKLEARN = False

_HAS_IMAGEIO = True
try:
    import imageio.v2 as imageio  # type: ignore
except Exception:
    _HAS_IMAGEIO = False

# ---------------- Basics: fs + channel access ----------------
_DEF_TIME_COL = 'Timestamp'
_DEF_CH_PATTERNS = ("EEG.{ch}", "eeg.{ch}", "{ch}", "RAW.{ch}", "CHAN.{ch}")

# Canonical EEG bands (Hz)
BAND_DEFS: Dict[str, Tuple[float, float]] = {
    'delta': (1.0, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta':  (13.0, 30.0),
    'gamma': (30.0, 45.0),   # keep conservative upper bound for most headsets
}

# ================= Utilities =================

def infer_fs_from_records(RECORDS: pd.DataFrame, time_col: str = _DEF_TIME_COL) -> float:
    t = np.asarray(RECORDS[time_col].values, dtype=float)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        raise ValueError("Cannot infer fs: Timestamp spacing invalid.")
    return float(1.0 / np.median(dt))


def find_channel_series(RECORDS: pd.DataFrame, ch_name: str) -> Optional[pd.Series]:
    for pat in _DEF_CH_PATTERNS:
        col = pat.format(ch=ch_name)
        if col in RECORDS.columns:
            return pd.to_numeric(RECORDS[col], errors='coerce').astype(float)
    return None


def bandpass_guard(x: np.ndarray, fs: float, f1: float, f2: float, order: int = 4) -> np.ndarray:
    ny = 0.5 * fs
    f1 = max(f1, 1e-6)
    f2 = min(f2, ny * 0.99)
    if not (0 < f1 < f2 < ny):
        raise ValueError(f"Invalid bandpass range: ({f1}, {f2}) with fs={fs}")
    b, a = signal.butter(order, [f1 / ny, f2 / ny], btype='band')
    return signal.filtfilt(b, a, x.astype(float))

# ================= Connectivity =================

def compute_wpli_records(X: np.ndarray, sf: float, fmin: float, fmax: float) -> np.ndarray:
    """wPLI on (n_channels, n_times). Multitaper via mne_connectivity if available.
    Fallback: narrowband Hilbert pseudo‑wPLI. Returns symmetric zero‑diag matrix.
    """
    if X.ndim != 2:
        raise ValueError("X must be (n_channels, n_times)")

    if _HAS_MNE_CONN:
        data_in = X[np.newaxis, ...]  # (1, n_signals, n_times)
        con = mne_conn.spectral_connectivity(
            data_in, method='wpli', mode='multitaper', sfreq=sf,
            fmin=fmin, fmax=fmax, faverage=True, verbose=False
        )
        W = con.get_data()[:, :, 0]
        np.fill_diagonal(W, 0.0)
        return W
    else:
        Xf = np.empty_like(X)
        for i in range(X.shape[0]):
            Xf[i] = bandpass_guard(X[i], sf, fmin, fmax)
        Z = signal.hilbert(Xf, axis=1)
        n_ch = X.shape[0]
        W = np.zeros((n_ch, n_ch), dtype=float)
        for i in range(n_ch):
            zi = Z[i]
            for j in range(i + 1, n_ch):
                im = np.imag(zi * np.conj(Z[j]))
                num = np.abs(np.mean(im))
                den = np.mean(np.abs(im)) + 1e-12
                W[i, j] = W[j, i] = float(num / den)
        np.fill_diagonal(W, 0.0)
        return W

# ================= Graph metrics & harmonics =================

def graph_entropy(adj: np.ndarray) -> float:
    # Shannon entropy of Laplacian spectrum (positive part), robust proxy of complexity
    deg = np.sum(adj, axis=1)
    L = np.diag(deg) - adj
    vals = np.linalg.eigvalsh(L)
    vals = vals[vals > 1e-12]
    if vals.size == 0:
        return float('nan')
    p = vals / np.sum(vals)
    return float(-np.sum(p * np.log(p)))


def minimal_cut_weight(adj: np.ndarray) -> float:
    G = nx.from_numpy_array(adj)
    if G.number_of_edges() == 0:
        return 0.0
    try:
        cut_val, _ = nx.algorithms.connectivity.stoer_wagner(G)
        return float(cut_val)
    except Exception:
        return 0.0


def connectome_harmonics(adj: np.ndarray, n_modes: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    L = np.diag(np.sum(adj, axis=1)) - adj
    eigvals, eigvecs = np.linalg.eigh(L)
    n_modes = max(1, min(n_modes, eigvals.size))
    return eigvals[:n_modes], eigvecs[:, :n_modes]

# ================= Helpers =================

def _discover_electrodes(RECORDS: pd.DataFrame, time_col: str) -> List[str]:
    els: List[str] = []
    for col in RECORDS.columns:
        if col == time_col:
            continue
        if col.startswith('EEG.'):
            ch = col.split('.', 1)[1]
            if ch and ch not in els:
                els.append(ch)
    return els


def _slice_windows(data: np.ndarray, t: np.ndarray, wins: List[Tuple[float, float]]) -> np.ndarray:
    arrs = []
    for (t0, t1) in wins:
        if t1 <= t0:
            continue
        sel = (t >= t0) & (t <= t1)
        idx = np.where(sel)[0]
        if idx.size == 0:
            continue
        seg = data[:, idx[0]: idx[-1] + 1].T
        if seg.shape[0] > 0:
            arrs.append(seg)
    if not arrs:
        return np.empty((0, data.shape[0]))
    return np.vstack(arrs)

# ================= Core RECORDS routines =================

def run_network_geometry_suite_records(
    RECORDS: pd.DataFrame,
    ignition_windows: List[Tuple[float, float]],
    rebound_windows: Optional[List[Tuple[float, float]]] = None,
    fband: Tuple[float, float] = (8, 13),
    electrodes: Optional[List[str]] = None,
    time_col: str = _DEF_TIME_COL,
) -> Dict[str, object]:
    sf = infer_fs_from_records(RECORDS, time_col=time_col)
    t = np.asarray(RECORDS[time_col].values, dtype=float)

    # Pick electrodes
    if not electrodes:
        electrodes = _discover_electrodes(RECORDS, time_col)
        if not electrodes:
            electrodes = ['F4', 'O1', 'O2']

    # Build data matrix
    rows = []
    for ch in electrodes:
        s = find_channel_series(RECORDS, ch)
        if s is None:
            continue
        rows.append(np.asarray(s.values, dtype=float))
    if not rows:
        raise ValueError("No usable channels found.")
    data = np.vstack(rows)

    # Inside/ignition
    ign_entropy, ign_min_cut, eigvals_in_all = [], [], []
    for (t0, t1) in ignition_windows:
        if t1 <= t0:
            continue
        sel = (t >= t0) & (t <= t1)
        idx = np.where(sel)[0]
        if idx.size < 8:
            continue
        X = data[:, idx[0]: idx[-1] + 1]
        adj = compute_wpli_records(X, sf, fband[0], fband[1])
        ign_entropy.append(graph_entropy(adj))
        ign_min_cut.append(minimal_cut_weight(adj))
        eigv, _ = connectome_harmonics(adj)
        eigvals_in_all.append(eigv)

    # Outside (complement)
    mask = np.ones(data.shape[1], dtype=bool)
    for (t0, t1) in ignition_windows:
        if t1 <= t0:
            continue
        i0, i1 = max(0, int(t0 * sf)), min(data.shape[1], int(t1 * sf))
        mask[i0:i1] = False
    if not mask.any():
        mask[:] = True
    Xout = data[:, mask]
    adj_out = compute_wpli_records(Xout, sf, fband[0], fband[1])
    base_entropy = graph_entropy(adj_out)
    base_min_cut = minimal_cut_weight(adj_out)
    eigv_out, _ = connectome_harmonics(adj_out)

    results: Dict[str, object] = {
        'ign_entropy': np.array(ign_entropy),
        'ign_min_cut': np.array(ign_min_cut),
        'eigvals_in': np.array(eigvals_in_all, dtype=object),
        'base_entropy': float(base_entropy),
        'base_min_cut': float(base_min_cut),
        'eigvals_out': eigv_out,
    }

    # Optional rebound
    if rebound_windows:
        reb_entropy, reb_min_cut, eigvals_reb = [], [], []
        for (t0, t1) in rebound_windows:
            if t1 <= t0:
                continue
            sel = (t >= t0) & (t <= t1)
            idx = np.where(sel)[0]
            if idx.size < 8:
                continue
            X = data[:, idx[0]: idx[-1] + 1]
            adj = compute_wpli_records(X, sf, fband[0], fband[1])
            reb_entropy.append(graph_entropy(adj))
            reb_min_cut.append(minimal_cut_weight(adj))
            eigv, _ = connectome_harmonics(adj)
            eigvals_reb.append(eigv)
        results['reb_entropy'] = np.array(reb_entropy)
        results['reb_min_cut'] = np.array(reb_min_cut)
        results['eigvals_reb'] = np.array(eigvals_reb, dtype=object)

    return results


def run_state_space_embedding_records(
    RECORDS: pd.DataFrame,
    ignition_windows: List[Tuple[float, float]],
    baseline_windows: Optional[List[Tuple[float, float]]] = None,
    electrodes: Optional[List[str]] = None,
    n_components: int = 3,
    method: str = 'umap',
    fast_mode: bool = False,
    max_samples: int = 5000,
    time_col: str = _DEF_TIME_COL,
) -> Dict[str, np.ndarray]:
    sf = infer_fs_from_records(RECORDS, time_col=time_col)
    t = np.asarray(RECORDS[time_col].values, dtype=float)

    if not electrodes:
        electrodes = _discover_electrodes(RECORDS, time_col)
        if not electrodes:
            electrodes = ['F4', 'O1', 'O2']

    chans = []
    for ch in electrodes:
        s = find_channel_series(RECORDS, ch)
        if s is None:
            continue
        chans.append(np.asarray(s.values, dtype=float))
    if not chans:
        raise ValueError("No usable channels for embedding.")
    data = np.vstack(chans)

    def _slice(wins: List[Tuple[float, float]]):
        return _slice_windows(data, t, wins)

    ign_samples = _slice(ignition_windows)

    if baseline_windows:
        base_samples = _slice(baseline_windows)
    else:
        mask = np.ones(data.shape[1], dtype=bool)
        for (t0, t1) in ignition_windows:
            if t1 <= t0:
                continue
            i0, i1 = max(0, int(t0 * sf)), min(data.shape[1], int(t1 * sf))
            mask[i0:i1] = False
        if not mask.any():
            mask[:] = True
        base_samples = data[:, mask].T

    # Fast mode subsampling
    if fast_mode:
        if ign_samples.shape[0] > max_samples:
            idx = np.random.choice(ign_samples.shape[0], max_samples, replace=False)
            ign_samples = ign_samples[idx]
        if base_samples.shape[0] > max_samples:
            idx = np.random.choice(base_samples.shape[0], max_samples, replace=False)
            base_samples = base_samples[idx]

    # Reduce
    if not _HAS_SKLEARN or method.lower() == 'pca':
        k = min(n_components, max(2, min(ign_samples.shape[1], base_samples.shape[1])))
        reducer = PCA(n_components=k)
        ign_embed = reducer.fit_transform(ign_samples)
        base_embed = reducer.transform(base_samples)
    elif method.lower() == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=0)
        ign_embed = reducer.fit_transform(ign_samples)
        base_embed = reducer.fit_transform(base_samples)
    else:
        reducer = TSNE(n_components=n_components, random_state=0)
        ign_embed = reducer.fit_transform(ign_samples)
        base_embed = reducer.fit_transform(base_samples)

    # Plot
    fig = plt.figure(figsize=(6, 6))
    if n_components == 3:
        try:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(base_embed[:, 0], base_embed[:, 1], base_embed[:, 2], alpha=0.3, label='Baseline')
            ax.scatter(ign_embed[:, 0], ign_embed[:, 1], ign_embed[:, 2], alpha=0.3, label='Ignition')
            ax.set_xlabel('dim1'); ax.set_ylabel('dim2'); ax.set_zlabel('dim3')
        except Exception:
            plt.title('3D not available — showing first 2 dims')
            plt.scatter(base_embed[:, 0], base_embed[:, 1], alpha=0.3, label='Baseline')
            plt.scatter(ign_embed[:, 0], ign_embed[:, 1], alpha=0.3, label='Ignition')
    else:
        plt.scatter(base_embed[:, 0], base_embed[:, 1], alpha=0.3, label='Baseline')
        plt.scatter(ign_embed[:, 0], ign_embed[:, 1], alpha=0.3, label='Ignition')
    plt.legend(); plt.title('State-space manifold embedding (RECORDS)')
    plt.tight_layout(); plt.show()

    return {'ign_embed': ign_embed, 'base_embed': base_embed}


def session_report_records(
    RECORDS: pd.DataFrame,
    ignition_windows: List[Tuple[float, float]],
    rebound_windows: Optional[List[Tuple[float, float]]] = None,
    fband: Tuple[float, float] = (8, 13),
    electrodes: Optional[List[str]] = None,
    embed_method: str = 'umap',
) -> Dict[str, object]:
    geom = run_network_geometry_suite_records(
        RECORDS, ignition_windows, rebound_windows,
        fband=fband, electrodes=electrodes
    )
    emb = run_state_space_embedding_records(
        RECORDS, ignition_windows,
        baseline_windows=rebound_windows,
        electrodes=electrodes,
        n_components=3, method=embed_method,
        fast_mode=True
    )

    report: Dict[str, object] = {}
    ign_entropy_arr = np.asarray(geom['ign_entropy']) if 'ign_entropy' in geom else np.array([])
    ign_mincut_arr = np.asarray(geom['ign_min_cut']) if 'ign_min_cut' in geom else np.array([])

    report['Ignition entropy mean'] = float(np.nanmean(ign_entropy_arr)) if ign_entropy_arr.size else np.nan
    report['Ignition mincut mean'] = float(np.nanmean(ign_mincut_arr)) if ign_mincut_arr.size else np.nan
    report['Baseline entropy'] = float(geom['base_entropy'])
    report['Baseline mincut'] = float(geom['base_min_cut'])

    if 'reb_entropy' in geom:
        reb_entropy = np.asarray(geom['reb_entropy'])
        reb_mincut = np.asarray(geom['reb_min_cut'])
        report['Rebound entropy mean'] = float(np.nanmean(reb_entropy)) if reb_entropy.size else np.nan
        report['Rebound mincut mean'] = float(np.nanmean(reb_mincut)) if reb_mincut.size else np.nan

    eig_in_list = list(geom['eigvals_in']) if 'eigvals_in' in geom else []
    if len(eig_in_list) > 0:
        maxlen = max(len(arr) for arr in eig_in_list)
        pad = np.full((len(eig_in_list), maxlen), np.nan)
        for i, arr in enumerate(eig_in_list):
            arr = np.asarray(arr).ravel()
            pad[i, :arr.size] = arr
        report['Ignition eigvals avg'] = np.nanmean(pad, axis=0).tolist()
    else:
        report['Ignition eigvals avg'] = []

    report['Baseline eigvals'] = np.asarray(geom['eigvals_out']).ravel().tolist()
    report['Ignition embedding'] = tuple(np.asarray(emb['ign_embed']).shape)
    report['Baseline embedding'] = tuple(np.asarray(emb['base_embed']).shape)

    print("===== EEG Session Report (RECORDS) =====")
    for k, v in report.items():
        print(f"{k}: {v}")
    print("=========================================")
    return report

# ================= New: Multi‑band batch =================

def run_multi_band_geometry_records(
    RECORDS: pd.DataFrame,
    ignition_windows: List[Tuple[float, float]],
    rebound_windows: Optional[List[Tuple[float, float]]] = None,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    electrodes: Optional[List[str]] = None,
    time_col: str = _DEF_TIME_COL,
) -> pd.DataFrame:
    """Run network geometry over multiple frequency bands and return a tidy DataFrame.

    Columns: band, f_low, f_high, ign_entropy_mean, ign_mincut_mean,
             base_entropy, base_mincut.
    """
    if bands is None:
        bands = BAND_DEFS

    rows = []
    for name, (f1, f2) in bands.items():
        geom = run_network_geometry_suite_records(
            RECORDS, ignition_windows, rebound_windows,
            fband=(f1, f2), electrodes=electrodes, time_col=time_col
        )
        ign_e = np.asarray(geom['ign_entropy'])
        ign_c = np.asarray(geom['ign_min_cut'])
        rows.append({
            'band': name,
            'f_low': float(f1),
            'f_high': float(f2),
            'ign_entropy_mean': float(np.nanmean(ign_e)) if ign_e.size else np.nan,
            'ign_mincut_mean': float(np.nanmean(ign_c)) if ign_c.size else np.nan,
            'base_entropy': float(geom['base_entropy']),
            'base_mincut': float(geom['base_min_cut']),
        })
    return pd.DataFrame(rows)

# ================= New: Export helpers =================

def save_session_report_csv_json(report: Dict[str, object], base_path: str) -> Tuple[str, str]:
    """Write report CSV and JSON to `base_path` (without extension). Returns paths."""
    # JSON
    json_path = f"{base_path}.json"
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)

    # CSV (flat key/value)
    csv_path = f"{base_path}.csv"
    kv = pd.DataFrame({'key': list(report.keys()), 'value': list(report.values())})
    kv.to_csv(csv_path, index=False)
    return csv_path, json_path


def save_embedding_plot(emb: Dict[str, np.ndarray], filename: str) -> str:
    """Save a 2D/3D scatter plot of embeddings returned by run_state_space_embedding_records."""
    ign = np.asarray(emb['ign_embed'])
    base = np.asarray(emb['base_embed'])
    n_dims = ign.shape[1]
    fig = plt.figure(figsize=(6, 6))
    if n_dims == 3:
        try:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(base[:, 0], base[:, 1], base[:, 2], alpha=0.3, label='Baseline')
            ax.scatter(ign[:, 0], ign[:, 1], ign[:, 2], alpha=0.3, label='Ignition')
            ax.set_xlabel('dim1'); ax.set_ylabel('dim2'); ax.set_zlabel('dim3')
        except Exception:
            plt.scatter(base[:, 0], base[:, 1], alpha=0.3, label='Baseline')
            plt.scatter(ign[:, 0], ign[:, 1], alpha=0.3, label='Ignition')
    else:
        plt.scatter(base[:, 0], base[:, 1], alpha=0.3, label='Baseline')
        plt.scatter(ign[:, 0], ign[:, 1], alpha=0.3, label='Ignition')
    plt.legend(); plt.title('State‑space embedding')
    plt.tight_layout(); plt.savefig(filename, dpi=150)
    plt.close(fig)
    return filename

# ================= New: Sliding‑window animation =================

def animate_embedding_over_time_records(
    RECORDS: pd.DataFrame,
    window_sec: float,
    step_sec: float,
    electrodes: Optional[List[str]] = None,
    method: str = 'umap',
    n_components: int = 2,
    baseline_frac: float = 0.25,
    out_dir: str = 'embedding_frames',
    make_gif: bool = True,
    gif_name: str = 'embedding_evolution.gif',
    time_col: str = _DEF_TIME_COL,
) -> Dict[str, object]:
    """Create an embedding over sliding windows and save frames (and optional GIF).

    Strategy: each window's samples are embedded along with a fixed baseline
    sampled from the first `baseline_frac` of the session to provide a common
    reference cloud. Saves PNG frames; if imageio is available and make_gif=True,
    also writes a GIF.
    Returns dict with 'frame_paths' list and optional 'gif_path'.
    """
    os.makedirs(out_dir, exist_ok=True)

    sf = infer_fs_from_records(RECORDS, time_col=time_col)
    t = np.asarray(RECORDS[time_col].values, dtype=float)
    T = t[-1] - t[0]

    if not electrodes:
        electrodes = _discover_electrodes(RECORDS, time_col)
        if not electrodes:
            electrodes = ['F4', 'O1', 'O2']

    # Data matrix
    chans = []
    for ch in electrodes:
        s = find_channel_series(RECORDS, ch)
        if s is None:
            continue
        chans.append(np.asarray(s.values, dtype=float))
    if not chans:
        raise ValueError("No usable channels for animation.")
    data = np.vstack(chans)

    # Baseline window
    t0 = float(t[0])
    tB = t0 + max(1.0, baseline_frac * max(1.0, T))
    base_wins = [(t0, tB)]

    # Sliding window indices
    n_steps = int(max(1, math.floor((T - window_sec) / step_sec)))
    frame_paths: List[str] = []

    for k in range(n_steps + 1):
        w_start = t0 + k * step_sec
        w_end = w_start + window_sec
        ign_wins = [(w_start, w_end)]
        try:
            emb = run_state_space_embedding_records(
                RECORDS,
                ignition_windows=ign_wins,
                baseline_windows=base_wins,
                electrodes=electrodes,
                n_components=n_components,
                method=method,
                fast_mode=True,
                time_col=time_col,
            )
        except Exception as e:
            # Skip problematic windows but keep going
            continue

        # Draw and save each frame
        fig = plt.figure(figsize=(6, 6))
        ign = np.asarray(emb['ign_embed'])
        base = np.asarray(emb['base_embed'])
        if ign.size == 0 or base.size == 0:
            plt.close(fig)
            continue

        if n_components == 3:
            try:
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(base[:, 0], base[:, 1], base[:, 2], alpha=0.25, label='Baseline')
                ax.scatter(ign[:, 0], ign[:, 1], alpha=0.6, label=f'Window {k}')
                ax.set_xlabel('dim1'); ax.set_ylabel('dim2'); ax.set_zlabel('dim3')
            except Exception:
                plt.scatter(base[:, 0], base[:, 1], alpha=0.25, label='Baseline')
                plt.scatter(ign[:, 0], ign[:, 1], alpha=0.6, label=f'Window {k}')
        else:
            plt.scatter(base[:, 0], base[:, 1], alpha=0.25, label='Baseline')
            plt.scatter(ign[:, 0], ign[:, 1], alpha=0.6, label=f'Window {k}')
        plt.legend(loc='best')
        plt.title(f'Embedding window {w_start:.1f}–{w_end:.1f}s')
        plt.tight_layout()
        frame_path = os.path.join(out_dir, f'frame_{k:04d}.png')
        plt.savefig(frame_path, dpi=120)
        plt.close(fig)
        frame_paths.append(frame_path)

    result: Dict[str, object] = {'frame_paths': frame_paths}

    if make_gif and _HAS_IMAGEIO and len(frame_paths) > 1:
        gif_path = os.path.join(out_dir, gif_name)
        imgs = [imageio.imread(p) for p in frame_paths]
        imageio.mimsave(gif_path, imgs, duration=max(0.05, step_sec / 2.0))
        result['gif_path'] = gif_path

    return result

# ================= Convenience orchestration =================

def run_full_session_with_bands_and_exports(
    RECORDS: pd.DataFrame,
    ignition_windows: List[Tuple[float, float]],
    rebound_windows: Optional[List[Tuple[float, float]]] = None,
    electrodes: Optional[List[str]] = None,
    embed_method: str = 'umap',
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    export_dir: str = 'session_exports',
    base_name: str = 'session',
) -> Dict[str, object]:
    """Run single‑band report, multi‑band table, embedding PNG, and write CSV/JSON.

    Returns a dict of output file paths and data objects for immediate use.
    """
    os.makedirs(export_dir, exist_ok=True)

    # 1) Single‑band report (default: alpha)
    report = session_report_records(
        RECORDS,
        ignition_windows=ignition_windows,
        rebound_windows=rebound_windows,
        electrodes=electrodes,
        fband=(8, 13),
        embed_method=embed_method,
    )

    # 2) Write CSV/JSON
    csv_path, json_path = save_session_report_csv_json(
        report, os.path.join(export_dir, f'{base_name}_report')
    )

    # 3) Multi‑band geometry table
    df_bands = run_multi_band_geometry_records(
        RECORDS,
        ignition_windows=ignition_windows,
        rebound_windows=rebound_windows,
        bands=bands or BAND_DEFS,
        electrodes=electrodes,
    )
    bands_csv = os.path.join(export_dir, f'{base_name}_bands.csv')
    df_bands.to_csv(bands_csv, index=False)

    # 4) Save embedding plot
    emb = run_state_space_embedding_records(
        RECORDS, ignition_windows, baseline_windows=rebound_windows,
        electrodes=electrodes, n_components=3, method=embed_method, fast_mode=True
    )
    emb_png = os.path.join(export_dir, f'{base_name}_embedding.png')
    save_embedding_plot(emb, emb_png)

    return {
        'report': report,
        'report_csv': csv_path,
        'report_json': json_path,
        'bands_table': df_bands,
        'bands_csv': bands_csv,
        'embedding_png': emb_png,
    }


__all__ = [
    'BAND_DEFS',
    'infer_fs_from_records',
    'find_channel_series',
    'bandpass_guard',
    'compute_wpli_records',
    'graph_entropy',
    'minimal_cut_weight',
    'connectome_harmonics',
    'run_network_geometry_suite_records',
    'run_state_space_embedding_records',
    'session_report_records',
    'run_multi_band_geometry_records',
    'save_session_report_csv_json',
    'save_embedding_plot',
    'animate_embedding_over_time_records',
    'run_full_session_with_bands_and_exports',
]


# if __name__ == '__main__':
    # Quick smoke test with synthetic data if RECORDS is absent
#     if 'RECORDS' not in globals():
#         t = np.arange(0, 20, 0.004)  # ~250 Hz
#         df = pd.DataFrame({'Timestamp': t})
#         for ch in ['F4', 'O1', 'O2']:
#             df[f'EEG.{ch}'] = np.sin(2 * np.pi * 10 * t + np.random.rand()) + 0.1 * np.random.randn(t.size)
#         RECORDS = df
#     out = run_full_session_with_bands_and_exports(
#         RECORDS,
#         ignition_windows=[(3.0, 7.0)],
#         rebound_windows=[(14.0, 18.0)],
#         electrodes=None,
#         embed_method='umap',
#         export_dir='session_exports',
#         base_name='demo',
#     )
# #     if _HAS_IMAGEIO:
#     animate_embedding_over_time_records(
#         RECORDS, window_sec=2.0, step_sec=0.5, make_gif=True,
#         out_dir='embedding_frames', gif_name='demo.gif'
#         )

"""
Patch: robust GIF saving for animate_embedding_over_time_records
- Saves a GIF even if only one frame was generated (duplicates the frame)
- Adds a configurable gif_duration_s (default 0.2s per frame)
- Falls back to PIL if imageio is unavailable
"""


import os
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt

# Assume the following symbols exist in the surrounding module:
# - infer_fs_from_records, _discover_electrodes, find_channel_series,
#   run_state_space_embedding_records, imageio, _HAS_IMAGEIO

try:
    from PIL import Image  # fallback if imageio not available
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


def animate_embedding_over_time_records(
    RECORDS,
    window_sec: float,
    step_sec: float,
    electrodes: List[str] | None = None,
    method: str = 'umap',
    n_components: int = 2,
    baseline_frac: float = 0.25,
    out_dir: str = 'embedding_frames',
    make_gif: bool = True,
    gif_name: str = 'embedding_evolution.gif',
    gif_duration_s: float = 0.2,  # NEW: per-frame duration
    time_col: str = 'Timestamp',
) -> Dict[str, object]:
    os.makedirs(out_dir, exist_ok=True)

    sf = infer_fs_from_records(RECORDS, time_col=time_col)
    t = np.asarray(RECORDS[time_col].values, dtype=float)
    T = t[-1] - t[0]

    if not electrodes:
        electrodes = _discover_electrodes(RECORDS, time_col)
        if not electrodes:
            electrodes = ['F4', 'O1', 'O2']

    # Build data matrix
    chans = []
    for ch in electrodes:
        s = find_channel_series(RECORDS, ch)
        if s is None:
            continue
        chans.append(np.asarray(s.values, dtype=float))
    if not chans:
        raise ValueError("No usable channels for animation.")
    data = np.vstack(chans)

    # Baseline window
    t0 = float(t[0])
    tB = t0 + max(1.0, baseline_frac * max(1.0, T))
    base_wins = [(t0, tB)]

    # Sliding windows
    import math
    n_steps = int(max(1, math.floor((T - window_sec) / step_sec)))
    frame_paths: List[str] = []

    for k in range(n_steps + 1):
        w_start = t0 + k * step_sec
        w_end = w_start + window_sec
        ign_wins = [(w_start, w_end)]
        try:
            emb = run_state_space_embedding_records(
                RECORDS,
                ignition_windows=ign_wins,
                baseline_windows=base_wins,
                electrodes=electrodes,
                n_components=n_components,
                method=method,
                fast_mode=True,
                time_col=time_col,
            )
        except Exception:
            continue

        fig = plt.figure(figsize=(6, 6))
        ign = np.asarray(emb['ign_embed'])
        base = np.asarray(emb['base_embed'])
        if ign.size == 0 or base.size == 0:
            plt.close(fig)
            continue

        if n_components == 3:
            try:
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(base[:, 0], base[:, 1], base[:, 2], alpha=0.25, label='Baseline')
                ax.scatter(ign[:, 0], ign[:, 1], alpha=0.6, label=f'Window {k}')
                ax.set_xlabel('dim1'); ax.set_ylabel('dim2'); ax.set_zlabel('dim3')
            except Exception:
                plt.scatter(base[:, 0], base[:, 1], alpha=0.25, label='Baseline')
                plt.scatter(ign[:, 0], ign[:, 1], alpha=0.6, label=f'Window {k}')
        else:
            plt.scatter(base[:, 0], base[:, 1], alpha=0.25, label='Baseline')
            plt.scatter(ign[:, 0], ign[:, 1], alpha=0.6, label=f'Window {k}')
        plt.legend(loc='best')
        plt.title(f'Embedding {w_start:.1f}–{w_end:.1f}s (fs={sf:g})')
        plt.tight_layout()
        frame_path = os.path.join(out_dir, f'frame_{k:04d}.png')
        plt.savefig(frame_path, dpi=120)
        plt.close(fig)
        frame_paths.append(frame_path)

    result: Dict[str, object] = {'frame_paths': frame_paths, 'num_frames': len(frame_paths)}

    # GIF save with robust fallbacks
    if make_gif:
        if len(frame_paths) == 0:
            result['gif_error'] = 'No frames generated. Adjust window_sec/step_sec or session length.'
            return result
        # Ensure at least 2 frames for animated GIFs; duplicate if necessary
        paths_for_gif = frame_paths if len(frame_paths) > 1 else frame_paths * 2
        gif_path = os.path.join(out_dir, gif_name)
        try:
            if '_HAS_IMAGEIO' in globals() and _HAS_IMAGEIO:
                import imageio.v2 as imageio_v2
                imgs = [imageio_v2.imread(p) for p in paths_for_gif]
                imageio_v2.mimsave(gif_path, imgs, duration=max(0.01, float(gif_duration_s)))
                result['gif_path'] = gif_path
            elif _HAS_PIL:
                frames = [Image.open(p).convert('P', palette=Image.ADAPTIVE) for p in paths_for_gif]
                frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=int(max(0.01, float(gif_duration_s)) * 1000),
                    loop=0,
                    optimize=False,
                )
                result['gif_path'] = gif_path
            else:
                result['gif_error'] = 'Neither imageio nor PIL is available to write GIF.'
        except Exception as e:
            result['gif_error'] = f'GIF save failed: {e}'
    return result

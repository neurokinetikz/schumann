"""
RT-Style Multi-Seed Surfaces — Multiway Cuts Across Subsystems (fs=128)
-----------------------------------------------------------------------
Hypothesis: As information integrates, multiple subsystems jointly require larger
separating “surfaces”. Operational proxy: sum of pairwise min-cut capacities
between subsystem sets (F,P,O, …) increases during ignition vs baseline.

This module:
  1) Builds band-limited functional connectivity (wPLI) → weighted, undirected graph.
  2) Uses a **Gomory–Hu tree** to obtain **all-pairs min-cuts** efficiently.
  3) For each pair of subsystems (e.g., F vs O, F vs P, O vs P), computes the
     **set–set min-cut** as min_{i∈A,j∈B} mincut(i,j) (via GH tree).
  4) Defines a **multi-seed capacity** = Σ over unordered subsystem pairs of set–set min-cut.
  5) Compares ignition vs baseline (and optional rebound) per band.
  6) Controls: degree-preserving surrogates (two modes):
       • 'weight_permute' — permute weights over the same edge set (exact degree preserved)
       • 'degree_rewire'  — rewire the binary graph via double-edge swaps (degree preserved),
                            then assign original weights randomly to new edges.

Outputs
- delta_table: ΔMultiCut = Ign − Base per band (and raw capacities per state)
- shuffle_null (optional): null distribution of Δ under chosen control
- quick plotter: bar chart of ΔMultiCut by band with null bands (if provided)

Usage
-----
res = run_multi_seed_surface_cuts(
    RECORDS,
    ignition_windows=[(120,150)],
    rebound_windows=[(300,330)],
    time_col='Timestamp',
    bands={'theta':(4,8),'alpha':(8,13),'beta':(13,30)},
    clusters=None,                    # use defaults or pass your own
    control_mode='degree_rewire',     # or 'weight_permute'
    n_shuffle=200,
    graph_density=0.3,                # for rewiring control (top edges retained)
    show=True
)

plot_multicut_deltas(res['delta_table'], res.get('shuffle_null'))
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from scipy import signal
import networkx as nx

# ----------------- helpers -----------------

def _get_fs(RECORDS: pd.DataFrame, time_col: str) -> float:
    if 'infer_fs_from_records' in globals():
        try:
            return float(infer_fs_from_records(RECORDS, time_col=time_col))
        except Exception:
            pass
    t = np.asarray(RECORDS[time_col].values, dtype=float)
    dt = np.diff(t); dt = dt[np.isfinite(dt) & (dt>0)]
    if dt.size == 0: raise ValueError('Cannot infer fs')
    return 1.0/np.median(dt)


def _autoelectrodes(RECORDS: pd.DataFrame, time_col: str) -> List[str]:
    els = []
    for col in RECORDS.columns:
        if col == time_col: continue
        if col.startswith('EEG.'):
            ch = col.split('.',1)[1]
            if ch and ch not in els: els.append(ch)
    return els or ['F4','O1','O2']


def _bandpass(X: np.ndarray, fs: float, f1: float, f2: float, order: int=4) -> np.ndarray:
    ny = 0.5*fs; f1 = max(1e-6, min(f1, ny*0.99)); f2 = max(f1+1e-6, min(f2, ny*0.999))
    b,a = signal.butter(order, [f1/ny, f2/ny], btype='band')
    return signal.filtfilt(b,a,X,axis=1)


def _pseudo_wpli(Xb: np.ndarray) -> np.ndarray:
    Z = signal.hilbert(Xb, axis=1); n = Xb.shape[0]
    A = np.zeros((n,n), float)
    for i in range(n):
        zi = Z[i]
        for j in range(i+1, n):
            im = np.imag(zi*np.conj(Z[j]))
            num = np.abs(np.mean(im)); den = np.mean(np.abs(im))+1e-12
            w = num/den
            A[i,j]=A[j,i]=w
    np.fill_diagonal(A, 0.0)
    return A

def find_channel_series(records: pd.DataFrame, ch_name: str) -> Optional[pd.Series]:
    # ---------------- Basics: fs + channel access ----------------
    _DEF_TIME_COL = 'Timestamp'
    _DEF_CH_PATTERNS = ("EEG.{ch}", "eeg.{ch}", "{ch}", "RAW.{ch}", "CHAN.{ch}")
    for pat in _DEF_CH_PATTERNS:
        col = pat.format(ch=ch_name)
        if col in records.columns:
            return pd.to_numeric(records[col], errors='coerce').astype(float)
    return None

# ----------------- clusters -----------------

def default_clusters(electrodes: List[str]) -> Dict[str, List[int]]:
    """Map label-> indices into electrodes list for broad subsystems F, P, O, T.
    Adjust/override to match your montage.
    """
    label_lists = {
        'F': ['Fp1','Fp2','AF3','AF4','Fz','F3','F4','F5','F6','FC3','FC4','FC5','FC6','AF7','AF8'],
        'P': ['Pz','P1','P2','P3','P4','P5','P6','PO3','PO4'],
        'O': ['O1','O2','Oz','POz'],
        'T': ['T7','T8','TP7','TP8','FT7','FT8'],
    }
    idx_map = {}
    for lab, names in label_lists.items():
        idxs = [electrodes.index(n) for n in names if n in electrodes]
        if len(idxs)==0:
            # fallback: try nearest neighbors by prefix
            idxs = [i for i,e in enumerate(electrodes) if e.startswith(lab)]
        if len(idxs)==0:
            # ensure non-empty; pick a closest proxy if any
            idxs = [i for i,e in enumerate(electrodes) if e in electrodes[:1]]
        idx_map[lab] = idxs
    return idx_map

# ----------------- GH-based set–set min-cut -----------------

def _set_set_mincut_capacity(GH: nx.Graph, A: List[int], B: List[int]) -> float:
    """Given a Gomory–Hu tree GH (capacities on edges), compute min_{i∈A,j∈B} mincut(i,j).
    In a GH tree, the s–t mincut equals the minimum edge weight along the unique path s–t.
    """
    best = np.inf
    for i in A:
        for j in B:
            try:
                path = nx.shortest_path(GH, i, j)  # tree path
                # min edge weight along path
                wmin = np.inf
                for u,v in zip(path[:-1], path[1:]):
                    w = GH[u][v].get('capacity', GH[u][v].get('weight', 0.0))
                    if w < wmin: wmin = w
                if wmin < best: best = wmin
            except Exception:
                continue
    return float(best if np.isfinite(best) else np.nan)


def _multi_seed_capacity(A: np.ndarray, clusters_idx: Dict[str, List[int]]) -> float:
    """Build graph → GH tree → sum of pairwise set–set mincut capacities over subsystem pairs."""
    n = A.shape[0]
    # weighted undirected graph with capacities
    G = nx.Graph()
    for i in range(n):
        for j in range(i+1, n):
            w = float(A[i,j])
            if w>0: G.add_edge(i,j,capacity=w)
    if G.number_of_edges()==0:
        return float('nan')
    GH = nx.gomory_hu_tree(G, capacity='capacity')
    labels = list(clusters_idx.keys())
    total = 0.0
    for p in range(len(labels)):
        for q in range(p+1, len(labels)):
            cap = _set_set_mincut_capacity(GH, clusters_idx[labels[p]], clusters_idx[labels[q]])
            if np.isfinite(cap): total += cap
    return float(total)

# ----------------- surrogates -----------------

def _weight_permute_surrogate(A: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    A = A.copy(); n = A.shape[0]
    # collect upper-tri weights
    tri = np.triu_indices(n, 1)
    w = A[tri]; rng.shuffle(w)
    A[tri] = w; A[(tri[1],tri[0])] = w
    np.fill_diagonal(A, 0.0)
    return A


def _degree_rewire_surrogate(A: np.ndarray, density: float, rng: np.random.Generator, nswap: int=50) -> np.ndarray:
    """Threshold to given density; rewire with double_edge_swap; reassign weights randomly."""
    n = A.shape[0]
    tri = np.triu_indices(n,1)
    # keep top-K edges by weight
    w = A[tri]; K = int(round(density * len(w)))
    if K < 1: K = 1
    idx_sorted = np.argsort(w)[::-1]
    keep_idx = idx_sorted[:K]
    mask = np.zeros_like(w, dtype=bool); mask[keep_idx]=True
    # build binary graph
    G = nx.Graph()
    edges = [(int(tri[0][i]), int(tri[1][i])) for i in np.where(mask)[0]]
    G.add_nodes_from(range(n)); G.add_edges_from(edges)
    try:
        nx.double_edge_swap(G, nswap=nswap, max_tries=nswap*10, seed=int(rng.integers(1e9)))
    except Exception:
        pass
    # reassign weights (randomly permuted kept weights)
    kept_w = w[mask]; rng.shuffle(kept_w)
    A_s = np.zeros_like(A)
    for (u,v), ww in zip(G.edges(), kept_w):
        A_s[u,v]=A_s[v,u]=ww
    np.fill_diagonal(A_s, 0.0)
    return A_s

# ----------------- orchestration -----------------

def run_multi_seed_surface_cuts(
    RECORDS: pd.DataFrame,
    ignition_windows: List[Tuple[float,float]],
    rebound_windows: Optional[List[Tuple[float,float]]] = None,
    time_col: str = 'Timestamp',
    bands: Optional[Dict[str, Tuple[float,float]]] = None,
    electrodes: Optional[List[str]] = None,
    clusters: Optional[Dict[str, List[str]]] = None,
    control_mode: str = 'degree_rewire',
    n_shuffle: int = 200,
    graph_density: float = 0.3,
    show: bool = True,
    rng_seed: int = 42,
) -> Dict[str, object]:
    fs = _get_fs(RECORDS, time_col)
    bands = bands or {'theta':(4,8),'alpha':(8,13),'beta':(13,30)}
    # electrodes
    if electrodes is None:
        electrodes = _autoelectrodes(RECORDS, time_col)
    # data matrix in electrode order
    series=[]
    for ch in electrodes:
        s = find_channel_series(RECORDS, ch);  series.append(np.asarray(s.values, float))
    X = np.vstack(series)

    # clusters → indices
    if clusters is None:
        clusters_idx = default_clusters(electrodes)
    else:
        clusters_idx = {lab:[electrodes.index(ch) for ch in chs if ch in electrodes] for lab,chs in clusters.items()}
        # ensure each non-empty
        for lab,idxs in clusters_idx.items():
            if len(idxs)==0:
                raise ValueError(f'Cluster {lab} has no matching electrodes in data')

    # helper to build adjacency for a set of windows
    def adj_for_windows(windows: List[Tuple[float,float]], f1: float, f2: float) -> np.ndarray:
        if not windows: return np.zeros((X.shape[0], X.shape[1]))
        # concatenate blocks then compute wPLI
        t = np.asarray(RECORDS[time_col].values, dtype=float)
        sel_mask = np.zeros(X.shape[1], dtype=bool)
        for (t0,t1) in windows:
            i0,i1 = int(t0*fs), int(t1*fs)
            sel_mask[max(0,i0):min(X.shape[1],i1)] = True
        Xw = X[:, sel_mask]
        Xb = _bandpass(Xw, fs, f1,f2)
        A = _pseudo_wpli(Xb)
        return A

    # baseline complement
    t = np.asarray(RECORDS[time_col].values, dtype=float)
    base_mask = np.ones(X.shape[1], dtype=bool)
    for (t0,t1) in ignition_windows:
        i0,i1 = int(t0*fs), int(t1*fs)
        base_mask[max(0,i0):min(len(t),i1)] = False
    base_windows = [(t[base_mask][0], t[base_mask][-1])] if np.any(base_mask) else []

    rows=[]; shuffle_rows=[]; rng = np.random.default_rng(rng_seed)

    for bname,(f1,f2) in bands.items():
        A_ign = adj_for_windows(ignition_windows, f1,f2)
        A_bas = adj_for_windows(base_windows,    f1,f2)
        cap_ign = _multi_seed_capacity(A_ign, clusters_idx)
        cap_bas = _multi_seed_capacity(A_bas, clusters_idx)
        cap_reb = np.nan
        if rebound_windows:
            A_reb = adj_for_windows(rebound_windows, f1,f2)
            cap_reb = _multi_seed_capacity(A_reb, clusters_idx)
        rows.append({'band':bname, 'cap_ign':cap_ign, 'cap_base':cap_bas, 'cap_reb':cap_reb, 'd_cap':(cap_ign-cap_bas)})

        # controls
        if n_shuffle>0:
            for _ in range(n_shuffle):
                if control_mode == 'weight_permute':
                    A_perm = _weight_permute_surrogate(A_bas, rng)
                else:
                    A_perm = _degree_rewire_surrogate(A_bas, graph_density, rng)
                cap_perm = _multi_seed_capacity(A_perm, clusters_idx)
                shuffle_rows.append({'band':bname, 'cap_perm':cap_perm})

    delta_table = pd.DataFrame(rows)
    shuffle_null = pd.DataFrame(shuffle_rows) if shuffle_rows else None

    if show:
        plot_multicut_deltas(delta_table, shuffle_null)

    return {'delta_table': delta_table, 'shuffle_null': shuffle_null, 'params':{'clusters':clusters_idx}}

# ----------------- plotting -----------------

def plot_multicut_deltas(df: pd.DataFrame, shuffle: Optional[pd.DataFrame]) -> None:
    bands = df['band'].tolist(); vals = df['d_cap'].values
    fig, ax = plt.subplots(1,1, figsize=(8,3.2), constrained_layout=True)
    ax.bar(np.arange(len(bands)), vals, width=0.6, label='ΔMultiCut (Ign−Base)')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(np.arange(len(bands))); ax.set_xticklabels(bands)
    ax.set_ylabel('Δ capacity'); ax.set_title('Multi‑seed surface capacity deltas')
    if shuffle is not None and not shuffle.empty:
        # add 95% null bands
        for i,b in enumerate(bands):
            null = shuffle[shuffle['band']==b]['cap_perm'].values
            if null.size>10:
                lo,hi = np.percentile(null, [2.5,97.5])
                ax.plot([i-0.3,i+0.3],[lo,lo], color='tab:gray', lw=2)
                ax.plot([i-0.3,i+0.3],[hi,hi], color='tab:gray', lw=2)
                ax.fill_between([i-0.3,i+0.3], lo, hi, color='tab:gray', alpha=0.15, linewidth=0)
    ax.legend(); plt.show()

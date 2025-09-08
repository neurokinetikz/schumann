"""
Network Graph Metrics & Hub Analysis — Simple Graphs & Validation
=================================================================

What it does
------------
• Connectivity (per band) with PLI (default; robust to zero-lag / volume conduction).
• Threshold to target density → undirected weighted graph (symmetric).
• Metrics: small-world index σ (C/C_rand)/(L/L_rand), clustering C, char path L, global efficiency Eglob,
           modularity Q (greedy), participation coefficient P_i per node,
           centralities (strength, degree, betweenness).
• Degree-preserving rewires (double-edge swaps) → null for σ.
• Per-state outputs (Ignition/Baseline): heatmaps, metric bars, hub bars, summary CSV.

Inputs
------
RECORDS : DataFrame with a numeric time column (default 'Timestamp') and EEG.* columns.
eeg_channels : list of channels to include (e.g., ['EEG.O1','EEG.O2',...]).
bands : dict of name→(f1,f2); default theta/alpha/beta.
method : 'pli' (default) or 'imagcoh' (imag coherency via Welch).
density : edge density after thresholding (0..1).

Usage
-----
res = run_graph_metrics_hubs(
    RECORDS,
    eeg_channels=['EEG.O1','EEG.O2','EEG.P7','EEG.P8','EEG.FC5','EEG.FC6'],
    ignition_windows=[(290,310),(580,600)],
    baseline_windows=[(0,290),(325,580)],
    time_col='Timestamp',
    out_dir='exports_graph/S01',
    show=False  # save figures; avoids notebook flooding
)
print(res['summary'])
"""
from __future__ import annotations
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, networkx as nx
from typing import Dict, List, Tuple, Optional
from scipy import signal

# ---------------- I/O & time helpers ----------------
def _ensure_dir(d): os.makedirs(d, exist_ok=True); return d

def detect_time_col(df,
    candidates=('Timestamp','Time','time','t','seconds','sec','ms','datetime','DateTime','Datetime')) -> Optional[str]:
    for c in candidates:
        if c in df.columns: return c
    # first numeric, roughly monotonic
    for c in df.columns:
        s = pd.to_numeric(df[c], errors='coerce')
        if s.notna().sum() > max(50, 0.5*len(df)):
            x = s.values.astype(float); dt = np.diff(x[np.isfinite(x)])
            if dt.size and np.nanmedian(dt)>0: return c
    # datetime?
    for c in df.columns:
        try:
            _ = pd.to_datetime(df[c], errors='raise'); return c
        except Exception: pass
    return None

def ensure_timestamp_column(df: pd.DataFrame, time_col: Optional[str]=None,
                            default_fs: float = 128.0, out_name: str='Timestamp')->str:
    col = time_col or detect_time_col(df)
    if col is None:
        df[out_name] = np.arange(len(df), dtype=float)/default_fs; return out_name
    s = df[col]
    if np.issubdtype(s.dtype, np.datetime64) or 'datetime' in str(s.dtype).lower():
        tsec = (pd.to_datetime(s) - pd.to_datetime(s).iloc[0]).dt.total_seconds().astype(float)
        df[out_name] = tsec.values; return out_name
    sn = pd.to_numeric(s, errors='coerce').astype(float)
    if sn.notna().sum() < max(50, 0.5*len(df)):
        df[out_name] = np.arange(len(df), dtype=float)/default_fs; return out_name
    sn = sn - np.nanmin(sn[np.isfinite(sn)])
    df[out_name] = sn.values
    return out_name

def infer_fs(df: pd.DataFrame, time_col: str)->float:
    t = np.asarray(pd.to_numeric(df[time_col], errors='coerce').values, float)
    dt = np.diff(t); dt = dt[(dt>0)&np.isfinite(dt)]
    if dt.size==0: raise ValueError("Cannot infer fs.")
    return float(1.0/np.median(dt))

def get_series(df: pd.DataFrame, name: str)->np.ndarray:
    if name in df.columns:
        x = pd.to_numeric(df[name], errors='coerce').fillna(0.0).values
        return np.asarray(x, float)
    alt = 'EEG.'+name
    if alt in df.columns:
        x = pd.to_numeric(df[alt], errors='coerce').fillna(0.0).values
        return np.asarray(x, float)
    raise ValueError(f"{name} not found.")

def slice_concat(x: np.ndarray, fs: float, wins: Optional[List[Tuple[float,float]]])->np.ndarray:
    if not wins: return x.copy()
    segs=[]; n=len(x)
    for (a,b) in wins:
        i0,i1 = int(round(a*fs)), int(round(b*fs))
        i0=max(0,i0); i1=min(n,i1)
        if i1>i0: segs.append(x[i0:i1])
    return np.concatenate(segs) if segs else x.copy()

def zscore(x): x=np.asarray(x,float); return (x - np.mean(x)) / (np.std(x)+1e-12)

# ---------------- Connectivity (PLI / Imag Coh) ----------------
def bandpass(x, fs, f1, f2, order=4):
    ny=0.5*fs; f1=max(1e-6,min(f1,0.99*ny)); f2=max(f1+1e-6,min(f2,0.999*ny))
    b,a=signal.butter(order,[f1/ny,f2/ny],btype='band'); return signal.filtfilt(b,a,x)

def pli_connectivity(X: np.ndarray, fs: float, f1: float, f2: float) -> np.ndarray:
    """
    PLI from band-passed analytic phases. X: (n_ch, T)
    """
    Xb = np.vstack([bandpass(x, fs, f1, f2) for x in X])
    Z = signal.hilbert(Xb, axis=1); phi = np.angle(Z)
    n = X.shape[0]
    W = np.zeros((n,n), float)
    for i in range(n):
        for j in range(i+1,n):
            dphi = phi[i]-phi[j]
            pli = np.abs(np.mean(np.sign(np.sin(dphi))))
            W[i,j]=W[j,i]=float(pli)
    np.fill_diagonal(W, 0.0)
    return W

def imagcoh_connectivity(X: np.ndarray, fs: float, f1: float, f2: float) -> np.ndarray:
    """
    Imag coherency from analytic signals. Robust to zero-lag.
    """
    Xb = np.vstack([bandpass(x, fs, f1, f2) for x in X])
    Z = signal.hilbert(Xb, axis=1)
    n = X.shape[0]
    W = np.zeros((n,n), float)
    for i in range(n):
        for j in range(i+1,n):
            Sxy = np.mean(Z[i]*np.conj(Z[j]))
            Sxx = np.mean(Z[i]*np.conj(Z[i])); Syy = np.mean(Z[j]*np.conj(Z[j]))
            coh = Sxy/np.sqrt((Sxx*Syy)+1e-24)
            val = np.abs(np.imag(coh))
            W[i,j]=W[j,i]=float(val)
    np.fill_diagonal(W, 0.0)
    return W

# ---------------- Graph build & metrics ----------------
def threshold_by_density(W: np.ndarray, density: float = 0.2) -> np.ndarray:
    """
    Keep top fraction of weights (upper triangle) to reach target density.
    """
    n = W.shape[0]
    tri = W[np.triu_indices(n,1)]
    if np.all(tri==0): return np.zeros_like(W)
    k = int(np.round(density * (n*(n-1)/2)))
    k = max(1, min(k, tri.size))
    thr = np.partition(tri, -k)[-k]  # kth largest
    WT = np.where(W >= thr, W, 0.0)
    # ensure symmetry and zero diag
    WT = np.maximum(WT, WT.T); np.fill_diagonal(WT, 0.0)
    return WT

def graph_from_weighted(W: np.ndarray) -> nx.Graph:
    G = nx.Graph()
    n = W.shape[0]
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i+1,n):
            w = float(W[i,j])
            if w>0:
                G.add_edge(i,j,weight=w, length=1.0/max(w,1e-12))  # length for distances
    return G

def global_efficiency_weighted(G: nx.Graph) -> float:
    """
    Weighted global efficiency: mean of 1/d_ij on finite shortest paths (length attr).
    """
    if G.number_of_edges()==0: return np.nan
    effs=[]
    for i in G.nodes():
        lengths = nx.single_source_dijkstra_path_length(G, i, weight='length')
        for j,l in lengths.items():
            if i!=j and np.isfinite(l) and l>0:
                effs.append(1.0/l)
    return float(np.nanmean(effs)) if effs else np.nan

def char_path_length_weighted(G: nx.Graph) -> float:
    """
    Weighted characteristic path length using 'length' as distance.
    """
    if G.number_of_edges()==0: return np.nan
    Ls=[]
    for comp in nx.connected_components(G):
        H = G.subgraph(comp)
        if H.number_of_nodes()<2: continue
        Ls.append(nx.average_shortest_path_length(H, weight='length'))
    return float(np.nanmean(Ls)) if Ls else np.nan

def clustering_weighted(G: nx.Graph) -> float:
    c = nx.clustering(G, weight='weight')
    return float(np.nanmean(list(c.values()))) if c else np.nan

def modularity_greedy(G: nx.Graph) -> Tuple[Dict[int,int], float]:
    """
    Greedy modularity communities (weighted). Returns membership dict and Q.
    """
    if G.number_of_edges()==0:
        return ({i:i for i in G.nodes()}, np.nan)
    coms = list(nx.algorithms.community.greedy_modularity_communities(G, weight='weight'))
    memb = {}
    for ci, C in enumerate(coms):
        for node in C:
            memb[int(node)] = int(ci)
    # compute modularity Q
    Q = nx.algorithms.community.quality.modularity(G, coms, weight='weight')
    return memb, float(Q)

def participation_coeff(W: np.ndarray, memb: Dict[int,int]) -> np.ndarray:
    """
    Weighted participation coefficient: 1 - sum_s (k_is/k_i)^2
    """
    n = W.shape[0]
    k = np.sum(W, axis=1) + 1e-12
    S = {}
    for i in range(n):
        for j in range(n):
            if W[i,j]>0:
                S.setdefault((i, memb[j]), 0.0)
                S[(i, memb[j])] += W[i,j]
    P = np.zeros(n, float)
    for i in range(n):
        parts = [S.get((i,s),0.0) for s in set(memb.values())]
        P[i] = 1.0 - np.sum((np.array(parts)/k[i])**2)
    return P

def small_world_sigma(G: nx.Graph, n_rewire: int = 20) -> Tuple[float,float,float]:
    """
    Small-world index σ = (C/C_rand)/(L/L_rand) using degree-preserving nulls.
    Returns (sigma, C, L). Nulls averaged over n_rewire rewires.
    """
    if G.number_of_edges()==0 or G.number_of_nodes()<3:
        return (np.nan, np.nan, np.nan)
    C = clustering_weighted(G)
    L = char_path_length_weighted(G)
    # make a binary copy preserving degree
    B = nx.Graph()
    B.add_nodes_from(G.nodes())
    for u,v,d in G.edges(data=True):
        B.add_edge(u,v)
    Cr=[]; Lr=[]
    for _ in range(n_rewire):
        H = B.copy()
        # double-edge swaps preserve degree sequence
        try:
            nx.double_edge_swap(H, nswap=max(1, H.number_of_edges()*2), max_tries=H.number_of_edges()*10)
        except Exception:
            pass
        # put uniform weights (1) for null C,L on binary graph; lengths=1
        C0 = nx.average_clustering(H)
        if nx.is_connected(H):
            L0 = nx.average_shortest_path_length(H)
        else:
            L0 = np.nan
        Cr.append(C0); Lr.append(L0)
    Cr = np.array(Cr, float); Lr = np.array(Lr, float)
    C_rand = float(np.nanmean(Cr)) if Cr.size else np.nan
    L_rand = float(np.nanmean(Lr)) if Lr.size else np.nan
    if not np.isfinite(C_rand) or not np.isfinite(L_rand) or C_rand==0 or L_rand==0 or not np.isfinite(C) or not np.isfinite(L):
        return (np.nan, C, L)
    sigma = (C/C_rand)/(L/L_rand)
    return (float(sigma), float(C), float(L))

# ---------------- Connectivity wrapper ----------------
def compute_connectivity(RECORDS: pd.DataFrame, channels: List[str], wins,
                         band: Tuple[float,float], method: str, time_col: str) -> Tuple[np.ndarray, List[str], float]:
    """
    Returns (W, chan_names, fs) — symmetric connectivity matrix in [0,1].
    """
    time_col = ensure_timestamp_column(RECORDS, time_col=time_col, default_fs=128.0)
    fs = infer_fs(RECORDS, time_col)
    X=[]; names=[]
    for ch in channels:
        nm = ch if ch.startswith('EEG.') else 'EEG.'+ch
        if nm in RECORDS.columns:
            x = get_series(RECORDS, nm)
            x = slice_concat(x, fs, wins)
            X.append(zscore(np.asarray(x,float))); names.append(nm)
    if not X: raise ValueError("No EEG channels found.")
    # truncate common length
    L = min(len(x) for x in X)
    X = np.vstack([x[:L] for x in X])
    if method.lower() == 'imagcoh':
        W = imagcoh_connectivity(X, fs, band[0], band[1])
    else:
        W = pli_connectivity(X, fs, band[0], band[1])
    return W, names, fs

# ---------------- Orchestrator ----------------
def run_graph_metrics_hubs(
    RECORDS: pd.DataFrame,
    eeg_channels: List[str],
    ignition_windows: Optional[List[Tuple[float,float]]] = None,
    baseline_windows: Optional[List[Tuple[float,float]]] = None,
    bands: Dict[str, Tuple[float,float]] = None,
    method: str = 'pli',               # 'pli' or 'imagcoh'
    density: float = 0.2,              # target graph density after threshold
    time_col: str = 'Timestamp',
    out_dir: str = 'exports_graph/session',
    show: bool = False
)->Dict[str, object]:
    """
    Build functional graphs per band and state; compute small-worldness, clustering,
    path length, efficiency, modularity, participation & hubs; save simple plots + CSV.
    """
    _ensure_dir(out_dir)
    bands = bands or {'theta':(4,8), 'alpha':(8,13), 'beta':(13,30)}
    states = {'ignition': ignition_windows, 'baseline': baseline_windows}

    summaries=[]
    results={}
    for st, wins in states.items():
        if wins is None: continue
        for bn, bnd in bands.items():
            # Connectivity
            W, names, fs = compute_connectivity(RECORDS, eeg_channels, wins, bnd, method, time_col)
            # Threshold to density
            WT = threshold_by_density(W, density=density)
            # Graph with weights & 'length' attribute
            G = graph_from_weighted(WT)
            # Metrics
            sigma, C, L = small_world_sigma(G, n_rewire=20)
            Eglob = global_efficiency_weighted(G)
            memb, Q = modularity_greedy(G)
            # Participation
            P = participation_coeff(WT, memb) if G.number_of_edges()>0 else np.full(WT.shape[0], np.nan)
            # Hubs
            strength = np.sum(WT, axis=1)
            degree   = (WT>0).sum(axis=1)
            # Betweenness (use 'length' attr as distance → invert of weight already set)
            BC = nx.betweenness_centrality(G, weight='length', normalized=True) if G.number_of_edges()>0 else {i:np.nan for i in range(len(names))}
            betweenness = np.array([BC[i] for i in range(len(names))], float)

            # Save heatmap
            plt.figure(figsize=(4.5,4))
            plt.imshow(W, vmin=0, vmax=1, cmap='magma')
            plt.colorbar(label=f'{method.upper()}')
            plt.xticks(range(len(names)), [n.split(".",1)[-1] for n in names], rotation=90, fontsize=8)
            plt.yticks(range(len(names)), [n.split(".",1)[-1] for n in names], fontsize=8)
            plt.title(f'{bn} connectivity — {st}')
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, f'conn_{bn}_{st}.png'), dpi=140)
            if show: plt.show()
            plt.close()

            # Metric bars
            plt.figure(figsize=(6,3.2))
            keys = ['sigma','C','L','Eglob','Q']
            vals = [sigma, C, L, Eglob, Q]
            plt.bar(range(len(keys)), vals, color='tab:blue', alpha=0.9)
            plt.xticks(range(len(keys)), keys); plt.ylabel('value')
            plt.title(f'Graph metrics — {bn} / {st} (density={density:.2f})')
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, f'metrics_{bn}_{st}.png'), dpi=140)
            if show: plt.show()
            plt.close()

            # Hub bars (strength & participation)
            plt.figure(figsize=(max(6, 0.4*len(names)), 3.0))
            idx = np.argsort(strength)[::-1]
            top = idx[:min(8,len(idx))]
            plt.bar(np.arange(len(top))-0.2, strength[top], width=0.4, label='strength')
            plt.bar(np.arange(len(top))+0.2, P[top],       width=0.4, label='participation')
            plt.xticks(range(len(top)), [names[i].split('.',1)[-1] for i in top], rotation=0, fontsize=8)
            plt.ylabel('value'); plt.legend()
            plt.title(f'Hubs — {bn} / {st}')
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, f'hubs_{bn}_{st}.png'), dpi=140)
            if show: plt.show()
            plt.close()

            # Store & summarize
            results.setdefault(st, {})[bn] = {'W':W, 'WT':WT, 'names':names,
                                              'sigma':sigma, 'C':C, 'L':L, 'Eglob':Eglob,
                                              'Q':Q, 'P':P, 'strength':strength, 'degree':degree,
                                              'betweenness':betweenness, 'memb':memb}
            summaries.append({'state':st,'band':bn,'sigma':sigma,'C':C,'L':L,'Eglob':Eglob,'Q':Q})

    # Ignition vs Baseline side-by-side (if both exist)
    if 'ignition' in results and 'baseline' in results:
        for bn in bands.keys():
            if bn in results['ignition'] and bn in results['baseline']:
                plt.figure(figsize=(6,3.2))
                keys=['sigma','C','L','Eglob','Q']
                x=np.arange(len(keys)); w=0.38
                ig = results['ignition'][bn]; ba = results['baseline'][bn]
                plt.bar(x-w/2, [ba[k] for k in keys], width=w, label='Baseline', color='tab:orange', alpha=0.9)
                plt.bar(x+w/2, [ig[k] for k in keys], width=w, label='Ignition', color='tab:blue', alpha=0.9)
                plt.xticks(x, keys); plt.ylabel('value'); plt.title(f'Ignition vs Baseline — {bn}')
                plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(out_dir, f'ign_vs_base_{bn}.png'), dpi=140)
                if show: plt.show()
                plt.close()

    # Save CSV summary
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(os.path.join(out_dir, 'summary.csv'), index=False)
    return {'summary': summary_df, 'results': results, 'out_dir': out_dir}

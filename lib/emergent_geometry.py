"""
Phase Metric Embedding → Emergent Geometry (fs=128)
---------------------------------------------------
Hypothesis: A brain’s instantaneous phase-distance matrix embeds into a smoother
low‑D manifold during integrated states (closer to a curved geometry).

Method
- Build phase distance D_ij = 1 − |PLV_ij| from band‑limited analytic phases.
- Run **Isomap** (and **UMAP** if available) on the precomputed distance.
- Compare manifold quality across states (ignition vs baseline, optional rebound):
  • **Trustworthiness** and **Continuity** (k‑NN preservation)
  • **Geodesic stress**: normalized error between high‑D geodesics and embedding Euclidean distances
- Control: **phase‑scrambling** (Fourier phase randomization) to form null bands.

Usage
-----
res = run_phase_embedding_emergent_geometry(
    RECORDS,
    ignition_windows=[(120,150)],
    rebound_windows=[(300,330)],
    time_col='Timestamp',
    band=(8,13),                   # e.g., alpha
    n_neighbors=6, n_components=2,
    n_surr=100,                    # surrogate nulls
    method='isomap',               # or 'umap' if umap-learn installed
    show=True
)

print(res['metrics_table'])        # trust, continuity, stress by state
plot_phase_embedding_quality(res)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from scipy import signal, sparse
from scipy.sparse.csgraph import shortest_path

# Try sklearn & umap
try:
    from sklearn.manifold import Isomap
    from sklearn.metrics import pairwise_distances
    from sklearn.neighbors import NearestNeighbors
    _HAS_SK = True
except Exception:
    _HAS_SK = False

try:
    import umap
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False

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


def _slice_blocks(RECORDS: pd.DataFrame, time_col: str, X: np.ndarray, fs: float, windows: List[Tuple[float,float]]) -> List[np.ndarray]:
    t = np.asarray(RECORDS[time_col].values, dtype=float)
    blocks = []
    for (t0,t1) in windows or []:
        i0,i1 = int(t0*fs), int(t1*fs)
        if i1 - i0 < int(0.5*fs):
            continue
        blocks.append(X[:, i0:i1])
    return blocks


def _fourier_phase_randomize_1d(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    X = np.fft.rfft(x)
    mag = np.abs(X)
    k = X.size
    ph = rng.uniform(-np.pi, np.pi, size=k)
    ph[0] = np.angle(X[0])
    if k % 2 == 0:
        ph[-1] = np.angle(X[-1])
    Xs = mag * np.exp(1j*ph)
    return np.fft.irfft(Xs, n=x.size).astype(float)


def _make_surrogate_concat(concat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros_like(concat)
    for i in range(concat.shape[0]):
        out[i] = _fourier_phase_randomize_1d(concat[i], rng)
    return out

def find_channel_series(records: pd.DataFrame, ch_name: str) -> Optional[pd.Series]:
    # ---------------- Basics: fs + channel access ----------------
    _DEF_TIME_COL = 'Timestamp'
    _DEF_CH_PATTERNS = ("EEG.{ch}", "eeg.{ch}", "{ch}", "RAW.{ch}", "CHAN.{ch}")
    for pat in _DEF_CH_PATTERNS:
        col = pat.format(ch=ch_name)
        if col in records.columns:
            return pd.to_numeric(records[col], errors='coerce').astype(float)
    return None

# ----------------- PLV distance -----------------

def plv_distance_matrix(X: np.ndarray, fs: float, band: Tuple[float,float]) -> np.ndarray:
    """Return D = 1 − |PLV| for band-limited analytic phases across channels.
    X: (n_ch × n_times)
    """
    Xb = _bandpass(X, fs, band[0], band[1])
    Z = signal.hilbert(Xb, axis=1); ang = np.angle(Z)
    n = X.shape[0]
    PLV = np.zeros((n,n), float)
    for i in range(n):
        for j in range(i+1, n):
            dphi = ang[i] - ang[j]
            plv = np.abs(np.mean(np.exp(1j*dphi)))
            PLV[i,j] = PLV[j,i] = plv
    np.fill_diagonal(PLV, 1.0)
    D = 1.0 - np.abs(PLV)
    return D

# ----------------- manifold quality metrics -----------------

def _rank_matrix(D: np.ndarray) -> np.ndarray:
    """Return rank positions per row (argsort of distances)."""
    n = D.shape[0]
    R = np.zeros_like(D, dtype=int)
    for i in range(n):
        order = np.argsort(D[i])  # includes self at 0
        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(n)
        R[i] = ranks
    return R


def trustworthiness_continuity(D_high: np.ndarray, D_low: np.ndarray, k: int=5) -> Tuple[float,float]:
    """Compute trustworthiness & continuity from distance matrices.
    Follows Tenenbaum/van der Maaten definitions.
    """
    n = D_high.shape[0]
    R_high = _rank_matrix(D_high)
    R_low  = _rank_matrix(D_low)
    # Neighborhoods
    N_high = [set(np.argsort(D_high[i])[1:k+1]) for i in range(n)]
    N_low  = [set(np.argsort(D_low[i])[1:k+1])  for i in range(n)]

    # Trustworthiness: penalize points in N_low not in N_high
    t_sum = 0.0
    for i in range(n):
        U = N_low[i] - N_high[i]
        t_sum += np.sum(R_high[i][list(U)] - k)
    T = 1.0 - (2.0 / (n*k*(2*n - 3*k - 1))) * t_sum if n> (3*k+1) else np.nan

    # Continuity: penalize points in N_high not in N_low
    c_sum = 0.0
    for i in range(n):
        V = N_high[i] - N_low[i]
        c_sum += np.sum(R_low[i][list(V)] - k)
    C = 1.0 - (2.0 / (n*k*(2*n - 3*k - 1))) * c_sum if n> (3*k+1) else np.nan
    return float(T), float(C)


def geodesic_stress(D_high: np.ndarray, X_low: np.ndarray, k: int=5) -> float:
    """Compute normalized stress between high‑D geodesic distances and low‑D Euclidean distances.
    Geodesics via k‑NN graph on D_high; embedding distances from X_low.
    """
    n = D_high.shape[0]
    # kNN graph on D_high
    W = np.full((n,n), np.inf)
    for i in range(n):
        idx = np.argsort(D_high[i])[1:k+1]
        W[i, idx] = D_high[i, idx]
    W = np.minimum(W, W.T)
    np.fill_diagonal(W, 0.0)
    G = shortest_path(sparse.csr_matrix(W), directed=False)
    # embedded Euclidean distances
    D_emb = pairwise_distances(X_low, metric='euclidean') if _HAS_SK else np.linalg.norm(X_low[:,None,:]-X_low[None,:,:], axis=-1)
    # normalized stress
    num = np.nansum((G - D_emb)**2)
    den = np.nansum(G**2) + 1e-12
    return float(np.sqrt(num/den))

# ----------------- embedding -----------------

def embed_distance_matrix(D: np.ndarray, method: str='isomap', n_neighbors: int=6, n_components: int=2, random_state: int=0) -> np.ndarray:
    if method == 'umap' and _HAS_UMAP:
        reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, metric='precomputed', random_state=random_state)
        X = reducer.fit_transform(D)
        return X
    if _HAS_SK:
        iso = Isomap(n_neighbors=n_neighbors, n_components=n_components, metric='precomputed')
        X = iso.fit_transform(D)
        return X
    # simple fallback: classical MDS via double‑centering
    J = np.eye(D.shape[0]) - np.ones(D.shape)/D.shape[0]
    B = -0.5 * J.dot(D**2).dot(J)
    evals, evecs = np.linalg.eigh(B)
    idx = np.argsort(evals)[::-1][:n_components]
    L = np.diag(np.sqrt(np.maximum(evals[idx], 0)))
    X = evecs[:, idx].dot(L)
    return X

# ----------------- orchestration -----------------

def run_phase_embedding_emergent_geometry(
    RECORDS: pd.DataFrame,
    ignition_windows: List[Tuple[float,float]],
    rebound_windows: Optional[List[Tuple[float,float]]] = None,
    control_windows: Optional[List[Tuple[float,float]]] = None,
    time_col: str = 'Timestamp',
    electrodes: Optional[List[str]] = None,
    band: Tuple[float,float] = (8,13),
    n_neighbors: int = 6,
    n_components: int = 2,
    method: str = 'isomap',
    k_quality: int = 5,
    n_surr: int = 100,
    show: bool = True,
    rng_seed: int = 11,
) -> Dict[str, object]:
    fs = _get_fs(RECORDS, time_col)
    electrodes = electrodes or _autoelectrodes(RECORDS, time_col)
    # data matrix
    series=[]
    for ch in electrodes:
        s = find_channel_series(RECORDS, ch)
        if s is None: continue
        series.append(np.asarray(s.values, float))
    X = np.vstack(series)

    # slice windows
    ign_blocks = _slice_blocks(RECORDS, time_col, X, fs, ignition_windows)
    t = np.asarray(RECORDS[time_col].values, float)
    base_mask = np.ones(X.shape[1], dtype=bool)
    for (t0,t1) in ignition_windows:
        i0,i1 = int(t0*fs), int(t1*fs)
        base_mask[max(0,i0):min(len(t),i1)] = False
    base_blocks = [X[:, base_mask]] if np.any(base_mask) else []
    reb_blocks  = _slice_blocks(RECORDS, time_col, X, fs, rebound_windows) if rebound_windows else []
    ctrl_blocks = _slice_blocks(RECORDS, time_col, X, fs, control_windows) if control_windows else []

    rng = np.random.default_rng(rng_seed)

    def analyze_state(blocks: List[np.ndarray], do_surrogates: bool) -> Dict[str, object]:
        """
        Build PLV distance D, embed to low-D (Isomap/UMAP), and compute quality metrics.
        If do_surrogates=True, build a null band via phase-scrambling.
        """
        if not blocks:
            return {'D': None, 'X': None, 'trust': np.nan, 'cont': np.nan, 'stress': np.nan, 'surr': None}

        # concatenate windows for this state
        concat = np.hstack(blocks)

        # high-D phase distance
        D = plv_distance_matrix(concat, fs, band)

        # low-D embedding
        X_low = embed_distance_matrix(D, method=method, n_neighbors=n_neighbors,
                                      n_components=n_components, random_state=rng.integers(1e9))

        # pairwise distances in embedding space
        if _HAS_SK:
            from sklearn.metrics import pairwise_distances
            D_low = pairwise_distances(X_low)
        else:
            D_low = np.linalg.norm(X_low[:, None, :] - X_low[None, :, :], axis=-1)

        # quality metrics
        T, C = trustworthiness_continuity(D, D_low, k=k_quality)
        S = geodesic_stress(D, X_low, k=n_neighbors)

        # surrogates (phase-scramble channels)
        surr = None
        if do_surrogates and n_surr > 0:
            vals = []
            for _ in range(n_surr):
                Xs = _make_surrogate_concat(concat, rng)
                Ds = plv_distance_matrix(Xs, fs, band)
                Xs_low = embed_distance_matrix(Ds, method=method, n_neighbors=n_neighbors,
                                               n_components=n_components, random_state=rng.integers(1e9))
                if _HAS_SK:
                    from sklearn.metrics import pairwise_distances
                    Ds_low = pairwise_distances(Xs_low)
                else:
                    Ds_low = np.linalg.norm(Xs_low[:, None, :] - Xs_low[None, :, :], axis=-1)
                Ts, Cs = trustworthiness_continuity(Ds, Ds_low, k=k_quality)
                Ss = geodesic_stress(Ds, Xs_low, k=n_neighbors)
                vals.append((Ts, Cs, Ss))
            surr = np.array(vals)  # shape (n_surr, 3)

        return {'D': D, 'X': X_low, 'trust': T, 'cont': C, 'stress': S, 'surr': surr}

    
    
    ign = analyze_state(ign_blocks, do_surrogates=True)
    base= analyze_state(base_blocks, do_surrogates=False)
    reb = analyze_state(reb_blocks, do_surrogates=False)
    ctrl= analyze_state(ctrl_blocks, do_surrogates=False)

    metrics_table = pd.DataFrame([
        {'state':'ignition','trust':ign['trust'],'cont':ign['cont'],'stress':ign['stress']},
        {'state':'baseline','trust':base['trust'],'cont':base['cont'],'stress':base['stress']},
        {'state':'rebound','trust':reb['trust'],'cont':reb['cont'],'stress':reb['stress']},
        {'state':'control','trust':ctrl['trust'],'cont':ctrl['cont'],'stress':ctrl['stress']},
    ])

    out = {
        'metrics_table': metrics_table,
        'embeddings': {'ignition':ign['X'],'baseline':base['X'],'rebound':reb['X'],'control':ctrl['X']},
        'surrogates': ign['surr'],
        'params': {'band':band,'n_neighbors':n_neighbors,'n_components':n_components,'method':method,'k_quality':k_quality}
    }

    if show:
        plot_phase_embedding_quality(out)
    return out

# ----------------- plotting -----------------

def plot_phase_embedding_quality(res: Dict[str, object]) -> None:
    mt = res['metrics_table'].set_index('state')
    # bar plots for trust, continuity, stress
    fig, axs = plt.subplots(1,3, figsize=(12,3.2), constrained_layout=True)
    states = ['baseline','ignition','rebound','control']
    # Trust
    axs[0].bar(range(len(states)), [mt.loc[s,'trust'] if s in mt.index else np.nan for s in states])
    axs[0].set_xticks(range(len(states))); axs[0].set_xticklabels(states, rotation=0)
    axs[0].set_title('Trustworthiness (higher is better)'); axs[0].set_ylabel('T')
    # Continuity
    axs[1].bar(range(len(states)), [mt.loc[s,'cont'] if s in mt.index else np.nan for s in states], color='tab:orange')
    axs[1].set_xticks(range(len(states))); axs[1].set_xticklabels(states, rotation=0)
    axs[1].set_title('Continuity (higher is better)'); axs[1].set_ylabel('C')
    # Stress
    axs[2].bar(range(len(states)), [mt.loc[s,'stress'] if s in mt.index else np.nan for s in states], color='tab:green')
    axs[2].set_xticks(range(len(states))); axs[2].set_xticklabels(states, rotation=0)
    axs[2].set_title('Geodesic stress (lower is better)'); axs[2].set_ylabel('stress')
    plt.show()

    # If we have surrogates for ignition, add null bands
    if res.get('surrogates') is not None:
        s = res['surrogates']  # shape (n_surr, 3) for (T,C,S)
        lo = np.nanpercentile(s, 2.5, axis=0); hi = np.nanpercentile(s, 97.5, axis=0)
        print('Ignition surrogate 95% bands:')
        print('  Trust:  [%.3f, %.3f]' % (lo[0], hi[0]))
        print('  Cont.:  [%.3f, %.3f]' % (lo[1], hi[1]))
        print('  Stress: [%.3f, %.3f]' % (lo[2], hi[2]))

from scipy import sparse
from scipy.sparse.csgraph import shortest_path
import numpy as np

def embed_distance_matrix(D: np.ndarray,
                          method: str = 'isomap',
                          n_neighbors: int = 6,
                          n_components: int = 2,
                          random_state: int = 0) -> np.ndarray:
    """Embed a precomputed distance matrix D.

    - 'umap' (if available): uses UMAP(metric='precomputed')
    - 'isomap': manual Isomap for precomputed distances:
        kNN graph on D -> geodesics via shortest_path -> classical MDS
    - fallback: classical MDS on D
    """
    # UMAP path (works with precomputed)
    if method == 'umap' and _HAS_UMAP:
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric='precomputed',
            random_state=random_state
        )
        return reducer.fit_transform(D)

    # Manual Isomap for precomputed distances
    if method == 'isomap':
        n = D.shape[0]
        # k-NN graph on D
        W = np.full((n, n), np.inf)
        for i in range(n):
            nbrs = np.argsort(D[i])[1:n_neighbors+1]
            W[i, nbrs] = D[i, nbrs]
        W = np.minimum(W, W.T)
        np.fill_diagonal(W, 0.0)

        # Geodesic distances
        G = shortest_path(sparse.csr_matrix(W), directed=False)

        # Classical MDS on geodesic distances
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J.dot(G**2).dot(J)
        evals, evecs = np.linalg.eigh(B)
        idx = np.argsort(evals)[::-1][:n_components]
        L = np.diag(np.sqrt(np.maximum(evals[idx], 0)))
        X = evecs[:, idx].dot(L)
        return X

    # Fallback: classical MDS on D
    J = np.eye(D.shape[0]) - np.ones(D.shape) / D.shape[0]
    B = -0.5 * J.dot(D**2).dot(J)
    evals, evecs = np.linalg.eigh(B)
    idx = np.argsort(evals)[::-1][:n_components]
    L = np.diag(np.sqrt(np.maximum(evals[idx], 0)))
    X = evecs[:, idx].dot(L)
    return X

"""
Informational Geometry of EEG State Manifolds — Simple Graphs & Validity Tests
=============================================================================

Goal (validations you can run per session)
-----------------------------------------
• Build a *state vector* per short time window from EEG features
  (band powers, a simple integration index from PLV graph entropy, etc.).
• Embed the high-D state space (PCA + Isomap/UMAP) → 2D for visualization.
• Quantify the manifold:
    – Trustworthiness / Continuity (k-NN preservation)
    – Geodesic stress (Isomap geodesics vs 2D Euclidean)
    – Curvature proxy (local geodesic stretch)
    – Entropy of the embedded distribution (2D histogram entropy)
    – SPD (covariance) manifold spread (Log-Euclidean)
    – Silhouette (Ignition vs Baseline separation)
• Time-lock geometry to the Schumann envelope (corr + null).
• Simple graphs + surrogate tests (label permutation, circular shift).

Assumptions
-----------
RECORDS: pandas.DataFrame with a numeric time column (default 'Timestamp')
and EEG signals named 'EEG.*'. You provide `eeg_channels` (or we detect them).
Provide ignition and baseline windows; they will label the state-points.

Usage
-----
res = run_info_geometry_state_manifolds(
    RECORDS,
    eeg_channels=['EEG.O1','EEG.O2','EEG.Oz','EEG.Pz'],
    ignition_windows=[(290,310),(580,600)],
    baseline_windows=[(0,290),(325,580)],
    sr_channel='EEG.Oz',               # None → auto-pick posterior channel
    time_col='Timestamp',
    out_dir='exports_infogeo/S01',
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
from scipy import signal, sparse
from scipy.sparse.csgraph import shortest_path

# Optional: UMAP
try:
    import umap
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False

# Optional: sklearn distances & silhouette
try:
    from sklearn.metrics import pairwise_distances, silhouette_score
    from sklearn.manifold import Isomap
    _HAS_SK = True
except Exception:
    _HAS_SK = False

# ---------------- small utilities ----------------

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
    alt = 'EEG.'+name
    if alt in RECORDS.columns:
        x = pd.to_numeric(RECORDS[alt], errors='coerce').fillna(0.0).values
        return np.asarray(x, float)
    raise ValueError(f"Signal '{name}' not found.")

def bandpass(x: np.ndarray, fs: float, f1: float, f2: float, order=4) -> np.ndarray:
    ny = 0.5*fs
    f1 = max(1e-6, min(f1, 0.99*ny)); f2 = max(f1+1e-6, min(f2, 0.999*ny))
    b,a = signal.butter(order, [f1/ny, f2/ny], btype='band')
    return signal.filtfilt(b,a,x)

def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x,float); return (x - np.mean(x)) / (np.std(x)+1e-12)

def schumann_envelope(sr: np.ndarray, fs: float, center=7.83, half_bw=0.6) -> np.ndarray:
    yb = bandpass(sr, fs, center-half_bw, center+half_bw)
    return np.abs(signal.hilbert(yb))

def slice_windows(RECORDS: pd.DataFrame, time_col: str, fs: float,
                  win_sec: float, step_sec: float) -> List[Tuple[int,int,float]]:
    """Return list of index windows (s,e, t_center)."""
    N = len(RECORDS)
    win = int(round(win_sec*fs)); step = int(round(step_sec*fs))
    idxs=[]
    for c in range(win//2, N-win//2, step):
        s = c - win//2; e = c + win//2
        idxs.append((s, e, c/fs))
    return idxs

def in_any_window(t: float, windows: List[Tuple[float,float]]) -> bool:
    for a,b in windows:
        if a <= t <= b: return True
    return False

# ---------------- features per window ----------------

def plv_graph_entropy(RECORDS, eeg_channels, fs, s, e, band):
    """Build PLV adjacency on [s:e] and return Laplacian spectral entropy."""
    phases=[]
    for ch in eeg_channels:
        x = get_series(RECORDS, ch)[s:e]
        xb = bandpass(x, fs, band[0], band[1])
        phases.append(np.angle(signal.hilbert(xb)))
    P = np.vstack(phases)
    N = len(eeg_channels)
    A = np.zeros((N,N))
    for i in range(N):
        for j in range(i,N):
            dphi = P[i]-P[j]
            A[i,j]=A[j,i]=float(np.abs(np.mean(np.exp(1j*dphi))))
    np.fill_diagonal(A, 0.0)
    # Laplacian entropy
    D = np.diag(A.sum(axis=1)); L = D - A
    L = 0.5*(L+L.T)
    vals = np.linalg.eigvalsh(L)
    vals = vals[vals>1e-12]
    if vals.size==0: return np.nan
    p = vals/np.sum(vals)
    return float(-np.sum(p*np.log(p)))

def window_features(RECORDS: pd.DataFrame,
                    eeg_channels: List[str],
                    sr_channel: str,
                    time_col: str,
                    win_sec: float = 2.0,
                    step_sec: float = 0.25,
                    bands: Dict[str, Tuple[float,float]] = None,
                    add_plv_entropy_band: Tuple[float,float] = (8,13)) -> pd.DataFrame:
    """
    Build a feature vector per sliding window:
      • band powers (theta/alpha/beta/gamma) averaged across channels
      • PLV graph Laplacian entropy (alpha by default)
      • SR envelope mean (7.83±0.6)
    Returns DataFrame with columns ['t','state','feat_*'] (state unlabeled here).
    """
    bands = bands or {'theta':(4,8),'alpha':(8,13),'beta':(13,30),'gamma':(30,80)}
    fs = infer_fs(RECORDS, time_col)
    idxs = slice_windows(RECORDS, time_col, fs, win_sec, step_sec)
    # prefetch arrays
    X = np.vstack([get_series(RECORDS, ch) for ch in eeg_channels])  # (n_ch, N)
    sr = get_series(RECORDS, sr_channel)
    env_sr = schumann_envelope(sr, fs)
    rows=[]
    for s,e,t in idxs:
        row={'t':t}
        seg = X[:, s:e]
        for name,(f1,f2) in bands.items():
            bp = []
            for ch in range(seg.shape[0]):
                xb = bandpass(seg[ch], fs, f1,f2)
                bp.append(np.mean(xb**2))
            row[f'BP_{name}'] = float(np.mean(bp))
        # PLV graph entropy (alpha by default)
        try:
            row['PLV_H'] = plv_graph_entropy(RECORDS, eeg_channels, fs, s, e, add_plv_entropy_band)
        except Exception:
            row['PLV_H'] = np.nan
        row['SR_env'] = float(np.mean(env_sr[s:e]))
        rows.append(row)
    return pd.DataFrame(rows)

# ---------------- embeddings ----------------

def embed_states(F: pd.DataFrame, method: str = 'pca', n_neighbors: int = 8,
                 n_components: int = 2, random_state: int = 0) -> Dict[str, np.ndarray]:
    """
    Embed feature matrix (T×D) to 2D/3D. Returns {'X':coords, 'method':..., 'components':...}
    """
    X = F.values
    # z-score features
    X = (X - np.nanmean(X, axis=0)) / (np.nanstd(X, axis=0)+1e-12)
    # fill NaNs with col mean 0
    X[np.isnan(X)] = 0.0
    if method == 'umap' and _HAS_UMAP:
        reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, metric='euclidean',
                            random_state=random_state)
        Z = reducer.fit_transform(X)
        return {'X': Z, 'method': 'umap'}
    elif method == 'isomap' and _HAS_SK:
        iso = Isomap(n_neighbors=n_neighbors, n_components=n_components)
        Z = iso.fit_transform(X)
        return {'X': Z, 'method': 'isomap'}
    else:
        # PCA fallback
        # covariance on columns
        C = np.cov(X, rowvar=False)
        vals, vecs = np.linalg.eigh(C)
        idx = np.argsort(vals)[::-1][:n_components]
        Z = X @ vecs[:, idx]
        return {'X': Z, 'method': 'pca'}

# ---------------- information-geometric metrics ----------------

def trust_continuity(D_high: np.ndarray, D_low: np.ndarray, k: int = 8) -> Tuple[float,float]:
    """Trustworthiness & Continuity (Tenenbaum / van der Maaten)."""
    n = D_high.shape[0]
    def ranks(D):
        R = np.zeros_like(D, dtype=int)
        for i in range(n):
            order = np.argsort(D[i])
            rank = np.empty(n, dtype=int); rank[order] = np.arange(n)
            R[i] = rank
        return R
    R_h = ranks(D_high); R_l = ranks(D_low)
    # neighborhoods
    N_h = [set(np.argsort(D_high[i])[1:k+1]) for i in range(n)]
    N_l = [set(np.argsort(D_low[i])[1:k+1])  for i in range(n)]
    # trust
    t_sum=0.0
    for i in range(n):
        U = N_l[i] - N_h[i]
        if U:
            t_sum += np.sum(R_h[i][list(U)] - k)
    T = 1.0 - (2.0 / (n*k*(2*n - 3*k - 1))) * t_sum if n>(3*k+1) else np.nan
    # continuity
    c_sum=0.0
    for i in range(n):
        V = N_h[i] - N_l[i]
        if V:
            c_sum += np.sum(R_l[i][list(V)] - k)
    C = 1.0 - (2.0 / (n*k*(2*n - 3*k - 1))) * c_sum if n>(3*k+1) else np.nan
    return float(T), float(C)

def geodesic_stress(X_high: np.ndarray, X_low: np.ndarray, k: int = 8) -> float:
    """
    Geodesic stress between k-NN geodesics (from high-D feature space) and Euclidean in embedding.
    """
    # high-D distances
    D_h = pairwise_distances(X_high) if _HAS_SK else np.linalg.norm(X_high[:,None,:]-X_high[None,:,:], axis=-1)
    # build kNN graph
    W = np.full_like(D_h, np.inf, dtype=float)
    for i in range(D_h.shape[0]):
        idx = np.argsort(D_h[i])[1:k+1]
        W[i, idx] = D_h[i, idx]
    W = np.minimum(W, W.T); np.fill_diagonal(W, 0.0)
    G = shortest_path(sparse.csr_matrix(W), directed=False)
    # low-D distances
    D_l = pairwise_distances(X_low) if _HAS_SK else np.linalg.norm(X_low[:,None,:]-X_low[None,:,:], axis=-1)
    num = np.nansum((G - D_l)**2); den = np.nansum(G**2)+1e-12
    return float(np.sqrt(num/den))

def curvature_proxy(X_high: np.ndarray, X_low: np.ndarray, k: int = 8) -> float:
    """
    Mean relative geodesic stretch over k-NN: mean_i mean_{j in N_i} (d_geo - d_euc)/d_euc.
    """
    D_h = pairwise_distances(X_high) if _HAS_SK else np.linalg.norm(X_high[:,None,:]-X_high[None,:,:], axis=-1)
    # kNN geodesic via D_h (graph on k neighbors)
    W = np.full_like(D_h, np.inf, dtype=float)
    for i in range(D_h.shape[0]):
        idx = np.argsort(D_h[i])[1:k+1]
        W[i, idx] = D_h[i, idx]
    W = np.minimum(W, W.T); np.fill_diagonal(W, 0.0)
    G = shortest_path(sparse.csr_matrix(W), directed=False)
    D_l = pairwise_distances(X_low) if _HAS_SK else np.linalg.norm(X_low[:,None,:]-X_low[None,:,:], axis=-1)
    e = (G - D_l) / (D_l + 1e-12)
    # only kNN pairs
    mask = np.isfinite(W) & (W>0)
    return float(np.nanmean(e[mask]))

def entropy_2d_embed(Z: np.ndarray, bins: int = 40) -> float:
    """
    Entropy of the embedded distribution via 2D histogram (Shannon, base e).
    """
    H, xe, ye = np.histogram2d(Z[:,0], Z[:,1], bins=bins, density=False)
    p = H.ravel().astype(float); p = p / (np.sum(p)+1e-12)
    p = p[p>0]
    return float(-np.sum(p*np.log(p)))

def logeuclidean_spread(cov_list: List[np.ndarray]) -> float:
    """
    Spread on SPD manifold (Log-Euclidean): mean pairwise ||log(S_i) − log(S_j)||_F.
    """
    logs=[]
    for S in cov_list:
        S = 0.5*(S+S.T) + 1e-9*np.eye(S.shape[0])
        vals, vecs = np.linalg.eigh(S)
        logs.append(vecs @ np.diag(np.log(np.maximum(vals,1e-12))) @ vecs.T)
    logs = np.stack(logs, axis=0)
    n = logs.shape[0]
    dists=[]
    for i in range(n):
        for j in range(i+1,n):
            dists.append(np.linalg.norm(logs[i]-logs[j],'fro'))
    return float(np.nanmean(dists)) if dists else np.nan

# ---------------- main runner ----------------

def run_info_geometry_state_manifolds(
    RECORDS: pd.DataFrame,
    eeg_channels: List[str],
    ignition_windows: List[Tuple[float,float]],
    baseline_windows: List[Tuple[float,float]],
    sr_channel: Optional[str] = None,
    time_col: str = 'Timestamp',
    out_dir: str = 'exports_infogeo/session',
    show: bool = True,
    win_sec: float = 2.0,
    step_sec: float = 0.25,
    bands: Dict[str, Tuple[float,float]] = None,
    embed_method: str = 'isomap',   # 'umap'|'isomap'|'pca'
    n_neighbors: int = 8
) -> Dict[str, object]:
    """
    Build state vectors → embeddings → info-geom metrics, with simple graphs + tests.
    Outputs figures + CSV summary in out_dir.
    """
    _ensure_dir(out_dir)
    fs = infer_fs(RECORDS, time_col)
    if sr_channel is None:
        sr_channel = 'EEG.Oz' if 'EEG.Oz' in RECORDS.columns else next((c for c in RECORDS.columns if c.startswith('EEG.')), None)

    # 1) Feature time series
    F = window_features(RECORDS, eeg_channels, sr_channel, time_col, win_sec, step_sec, bands)
    # Label state per window center
    F['state'] = ['ignition' if in_any_window(t, ignition_windows) else
                  ('baseline' if in_any_window(t, baseline_windows) else 'other')
                  for t in F['t'].values]
    F0 = F[F['state']!='other'].reset_index(drop=True)
    feat_cols = [c for c in F0.columns if c.startswith('BP_')] + ['PLV_H','SR_env']
    X_high = F0[feat_cols].values
    # keep covariances per state for SPD spread
    cov_ign = []; cov_bas = []
    if np.any(F0['state']=='ignition'):
        cov_ign.append(np.cov(F0[F0['state']=='ignition'][feat_cols].values, rowvar=False))
    if np.any(F0['state']=='baseline'):
        cov_bas.append(np.cov(F0[F0['state']=='baseline'][feat_cols].values, rowvar=False))

    # 2) Embedding
    emb = embed_states(F0[feat_cols], method=embed_method, n_neighbors=n_neighbors, n_components=2, random_state=0)
    Z = emb['X']  # (T’, 2)

    # BEFORE
    # F0[['Z1','Z2']] = Z

    # AFTER
    F0 = F0.reset_index(drop=True).copy()
    if Z.ndim != 2 or Z.shape[1] < 2:
        raise ValueError(f"Embedding returned shape {Z.shape}; need at least 2D. "
                         "Set n_components=2 or use method='pca'/'isomap'/'umap' accordingly.")
    F0['Z1'] = Z[:, 0]
    F0['Z2'] = Z[:, 1]


    # 3) Metrics
    # high-D distances on features
    D_high = pairwise_distances(X_high) if _HAS_SK else np.linalg.norm(X_high[:,None,:]-X_high[None,:,:], axis=-1)
    D_low  = pairwise_distances(Z) if _HAS_SK else np.linalg.norm(Z[:,None,:]-Z[None,:,:], axis=-1)
    T, C = trust_continuity(D_high, D_low, k=n_neighbors)
    stress = geodesic_stress(X_high, Z, k=n_neighbors)
    curv   = curvature_proxy(X_high, Z, k=n_neighbors)
    H_emb  = entropy_2d_embed(Z, bins=40)
    SPD_ign = logeuclidean_spread(cov_ign) if cov_ign else np.nan
    SPD_bas = logeuclidean_spread(cov_bas) if cov_bas else np.nan

    # silhouette (Ign vs Base)
    labels = (F0['state']=='ignition').astype(int).values
    sil = silhouette_score(Z, labels) if _HAS_SK and len(np.unique(labels))>1 else np.nan
    # permutation null for silhouette
    rng = np.random.default_rng(13)
    null_sil=[]
    for _ in range(200):
        perm = rng.permutation(labels)
        if _HAS_SK and len(np.unique(perm))>1:
            null_sil.append(silhouette_score(Z, perm))
    sil_thr95 = float(np.nanpercentile(null_sil, 95)) if null_sil else np.nan

    # correlation with SR envelope (on aligned centers)
    # re-sample SR envelope on centers corresponding to F0 rows
    sr = get_series(RECORDS, sr_channel)
    env = schumann_envelope(sr, fs)
    t_all = np.asarray(pd.to_numeric(RECORDS[time_col], errors='coerce').values, float)
    env_centers = np.interp(F0['t'].values, t_all, env)
    # pick principal coordinate (Z1) as manifold coordinate, corr with env
    r = np.corrcoef(Z[:,0], env_centers)[0,1]
    null_r=[]
    for _ in range(200):
        s = int(rng.integers(1, len(env_centers)-1))
        null_r.append(np.corrcoef(Z[:,0], np.r_[env_centers[-s:],env_centers[:-s]])[0,1])
    r_thr95 = float(np.nanpercentile(null_r, 95))

    # 4) Plots
    # scatter colored by state
    plt.figure(figsize=(6,5))
    cmap = {'ignition':'tab:red','baseline':'tab:blue'}
    for st in ['baseline','ignition']:
        idx = F0['state']==st
        plt.scatter(Z[idx,0], Z[idx,1], s=12, alpha=0.7, c=cmap[st], label=st)
    plt.title(f"{emb['method'].upper()} manifold — states"); plt.xlabel('Z1'); plt.ylabel('Z2')
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(out_dir,'embed_states.png'), dpi=140)
    if show: plt.show()
    plt.close()

    # scatter colored by SR envelope
    plt.figure(figsize=(6,5))
    sc = plt.scatter(Z[:,0], Z[:,1], c=env_centers, s=12, cmap='viridis')
    plt.colorbar(sc, label='SR env (a.u.)')
    plt.title('Manifold colored by Schumann envelope'); plt.xlabel('Z1'); plt.ylabel('Z2')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,'embed_sr_colormap.png'), dpi=140)
    if show: plt.show()
    plt.close()

    # metric bars + null lines
    plt.figure(figsize=(8,3))
    names = ['Trust','Cont','Stress','Curv','H(emb)','Sil','Sil_null95','SPD_ign','SPD_bas','r(Z1,SR)','r_null95']
    vals  = [T, C, stress, curv, H_emb, sil, sil_thr95, SPD_ign, SPD_bas, r, r_thr95]
    plt.bar(range(len(names)), vals, color='tab:purple', alpha=0.85)
    plt.xticks(range(len(names)), names, rotation=30)
    plt.title('Information-geometric metrics'); plt.tight_layout()
    plt.savefig(os.path.join(out_dir,'metrics_bars.png'), dpi=140)
    if show: plt.show()
    plt.close()

    # 5) Save tables
    F0.to_csv(os.path.join(out_dir,'state_features_embedding.csv'), index=False)
    summary = pd.DataFrame([{
        'method': emb['method'], 'trust':T, 'continuity':C, 'stress':stress, 'curvature_proxy':curv,
        'entropy_2d':H_emb, 'silhouette':sil, 'sil_null95':sil_thr95,
        'SPD_spread_ign':SPD_ign, 'SPD_spread_bas':SPD_bas,
        'r_Z1_SR':r, 'r_null95':r_thr95
    }])
    summary.to_csv(os.path.join(out_dir,'summary.csv'), index=False)

    return {'summary': summary,
            'features': F0,
            'embedding': Z,
            'metrics': {'trust':T,'continuity':C,'stress':stress,'curvature':curv,
                        'entropy_2d':H_emb,'silhouette':sil,'sil_null95':sil_thr95,
                        'SPD_ign':SPD_ign,'SPD_bas':SPD_bas,'r':r,'r_null95':r_thr95},
            'out_dir': out_dir}

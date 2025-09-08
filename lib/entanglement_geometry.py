"""
Entanglement–Geometry Analogy — Plotter (fs=128)
-------------------------------------------------
Companion plotting utilities for the minimal-cut vs coherence analysis.
- Bar chart of Δ(min-cut), Δ(entropy), Δ(PLV) per band (with optional error bars)
- Scatter matrix to visualize relationships among metrics across bands
- Side-by-side ignition vs baseline bars for min-cut, entropy, and PLV

Usage:
    summary = run_entanglement_geometry_minCut_PLV(...)
    plot_entanglement_geometry_deltas(summary['delta_table'])
    plot_entanglement_geometry_levels(summary['delta_table'])
    plot_entanglement_geometry_scatter(summary['delta_table'])

If you run multiple sessions and concatenate delta tables, you can
pass a combined df and set `by_session=True` to display mean±sem.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

# ------------- helpers -------------

def _err_sem(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return np.nan
    return float(np.std(x, ddof=1) / np.sqrt(x.size))

# ------------- main plots -------------

def plot_entanglement_geometry_deltas(df: pd.DataFrame, by_session: bool=False, session_col: str='session') -> None:
    """Bar plot of deltas per band. If by_session=True, aggregate by band and use mean±sem.
    Expects columns: band, d_mincut, d_entropy, d_plv.
    """
    if by_session:
        groups = df.groupby('band')
        bands = list(groups.groups.keys())
        dmc = [np.nanmean(groups.get_group(b)['d_mincut']) for b in bands]
        dme = [np.nanmean(groups.get_group(b)['d_entropy']) for b in bands]
        dpl = [np.nanmean(groups.get_group(b)['d_plv']) for b in bands]
        emc = [_err_sem(groups.get_group(b)['d_mincut'].values) for b in bands]
        eme = [_err_sem(groups.get_group(b)['d_entropy'].values) for b in bands]
        epl = [_err_sem(groups.get_group(b)['d_plv'].values) for b in bands]
    else:
        bands = df['band'].tolist()
        dmc = df['d_mincut'].values
        dme = df['d_entropy'].values
        dpl = df['d_plv'].values
        emc = eme = epl = None

    x = np.arange(len(bands))
    w = 0.26
    fig, ax = plt.subplots(1,1, figsize=(10,4), constrained_layout=True)
    ax.bar(x - w, dmc, width=w, label='Δmin-cut', yerr=emc, capsize=3)
    ax.bar(x      , dme, width=w, label='Δentropy', yerr=eme, capsize=3)
    ax.bar(x + w, dpl, width=w, label='ΔPLV', yerr=epl, capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(bands)
    ax.set_ylabel('Ignition − Baseline')
    ax.set_title('Entanglement–Geometry Deltas by Band')
    ax.legend()
    plt.show()


def plot_entanglement_geometry_levels(df: pd.DataFrame) -> None:
    """Side-by-side bars for ignition vs baseline values (min-cut, entropy, PLV) per band.
    Expects columns: band, ign_mincut, base_mincut, ign_entropy, base_entropy, plv_ign, plv_base.
    """
    bands = df['band'].tolist()
    x = np.arange(len(bands))
    w = 0.18
    fig, axs = plt.subplots(1,3, figsize=(12,4), constrained_layout=True)
    # min-cut
    axs[0].bar(x - w*0.5, df['base_mincut'], width=w, label='Base')
    axs[0].bar(x + w*0.5, df['ign_mincut'],  width=w, label='Ign')
    axs[0].set_title('Global min-cut'); axs[0].set_xticks(x); axs[0].set_xticklabels(bands)
    # entropy
    axs[1].bar(x - w*0.5, df['base_entropy'], width=w, label='Base')
    axs[1].bar(x + w*0.5, df['ign_entropy'],  width=w, label='Ign')
    axs[1].set_title('Laplacian entropy'); axs[1].set_xticks(x); axs[1].set_xticklabels(bands)
    # plv
    axs[2].bar(x - w*0.5, df['plv_base'], width=w, label='Base')
    axs[2].bar(x + w*0.5, df['plv_ign'],  width=w, label='Ign')
    axs[2].set_title('Mean PLV'); axs[2].set_xticks(x); axs[2].set_xticklabels(bands)
    for a in axs: a.legend()
    plt.show()


def plot_entanglement_geometry_scatter(df: pd.DataFrame) -> None:
    """Scatter plots to visualize relationships across bands: Δmincut vs ΔPLV, Δentropy vs ΔPLV, etc."""
    fig, axs = plt.subplots(1,3, figsize=(12,4), constrained_layout=True)
    axs[0].scatter(df['d_plv'], df['d_mincut']); axs[0].set_xlabel('ΔPLV'); axs[0].set_ylabel('Δmin-cut'); axs[0].set_title('Δmin-cut vs ΔPLV')
    axs[1].scatter(df['d_plv'], df['d_entropy']); axs[1].set_xlabel('ΔPLV'); axs[1].set_ylabel('Δentropy'); axs[1].set_title('Δentropy vs ΔPLV')
    axs[2].scatter(df['d_mincut'], df['d_entropy']); axs[2].set_xlabel('Δmin-cut'); axs[2].set_ylabel('Δentropy'); axs[2].set_title('Δentropy vs Δmin-cut')
    for ax in axs:
        # add zero lines
        ax.axhline(0, color='k', lw=0.5, alpha=0.5)
        ax.axvline(0, color='k', lw=0.5, alpha=0.5)
    plt.show()

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Union

try:
    import scipy.signal as sps
except Exception as e:
    sps = None

try:
    import networkx as nx
except Exception:
    nx = None


ArrayLike = np.ndarray
BandDict = Dict[str, Tuple[float, float]]
MatrixDict = Dict[str, ArrayLike]


def _butter_bandpass(sig: ArrayLike, fs: float, f_lo: float, f_hi: float, order: int = 4) -> ArrayLike:
    """Zero-phase bandpass filter using Butterworth + filtfilt."""
    if sps is None:
        raise ImportError("scipy is required for filtering. Install scipy to use bandpass mode.")
    ny = 0.5 * fs
    lo = max(1e-6, f_lo / ny)
    hi = min(0.999, f_hi / ny)
    if not (0 < lo < hi < 1):
        raise ValueError(f"Invalid band {f_lo}-{f_hi} Hz for fs={fs}.")
    b, a = sps.butter(order, [lo, hi], btype='bandpass')
    return sps.filtfilt(b, a, sig, axis=-1)


def _analytic_hilbert(sig: ArrayLike) -> ArrayLike:
    if sps is None:
        raise ImportError("scipy is required for Hilbert transform.")
    return sps.hilbert(sig, axis=-1)


def _compute_plv_matrix(analytic: ArrayLike) -> ArrayLike:
    """
    Compute pairwise PLV from complex analytic signals.
    analytic: (n_ch, n_samples) complex array.
    Returns (n_ch, n_ch) PLV matrix with zero diagonal.
    """
    z = analytic / np.maximum(1e-12, np.abs(analytic))  # unit phasors
    # PLV_{ij} = | mean_t z_i(t) * conj(z_j(t)) |
    M = (z @ z.conj().T) / z.shape[1]
    PLV = np.abs(M)
    np.fill_diagonal(PLV, 0.0)
    # ensure symmetry
    PLV = 0.5 * (PLV + PLV.T)
    return PLV.astype(float)


def _ensure_symmetric(W: ArrayLike) -> ArrayLike:
    W = np.array(W, dtype=float)
    # Zero diagonal and symmetrize
    np.fill_diagonal(W, 0.0)
    return 0.5 * (W + W.T)


def _laplacian_entropy(W: ArrayLike, eps: float = 1e-12) -> float:
    """
    Graph von Neumann/Laplacian entropy:
        L = D - W (combinatorial Laplacian)
        ρ = L / trace(L),  S = -Tr(ρ log ρ)  (natural log)
    If trace(L) ~ 0 (empty graph), returns 0.
    """
    W = _ensure_symmetric(W)
    d = np.sum(W, axis=1)
    L = np.diag(d) - W
    trL = float(np.trace(L))
    if trL <= eps:
        return 0.0
    rho = L / trL
    # use Hermitian eigen-solver
    vals = np.linalg.eigvalsh(rho)
    vals = np.clip(vals, 0.0, 1.0)
    nz = vals[vals > eps]
    if nz.size == 0:
        return 0.0
    S = float(-(nz * np.log(nz)).sum())
    return S


def _global_min_cut_weight(W: ArrayLike, edge_min: float = 0.0) -> float:
    """
    Global minimum s-t cut weight using Stoer-Wagner algorithm on an undirected weighted graph.
    If the graph is disconnected, returns 0.0.
    """
    if nx is None:
        raise ImportError("networkx is required for global min-cut computation.")
    W = _ensure_symmetric(W)
    n = W.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    # add edges above threshold
    for i in range(n):
        wi = W[i]
        for j in range(i + 1, n):
            w = float(wi[j])
            if w > edge_min:
                G.add_edge(i, j, weight=w)
    if G.number_of_nodes() <= 1:
        return 0.0
    if not nx.is_connected(G):
        return 0.0  # already separable with zero cost bridges
    cut_value, _ = nx.algorithms.connectivity.stoer_wagner(G, weight='weight')
    return float(cut_value)


def _mean_upper_triangle(W: ArrayLike) -> float:
    W = np.array(W, dtype=float)
    iu = np.triu_indices_from(W, k=1)
    if iu[0].size == 0:
        return 0.0
    return float(np.nanmean(W[iu]))


def _plv_from_timeseries(x: ArrayLike, fs: float, band: Tuple[float, float], filt_order: int = 4) -> ArrayLike:
    """
    Compute PLV matrix for a single band from real-valued time series (n_ch, n_samples).
    """
    x_bp = _butter_bandpass(x, fs, band[0], band[1], order=filt_order)
    analytic = _analytic_hilbert(x_bp)
    return _compute_plv_matrix(analytic)


def run_entanglement_geometry_minCut_PLV(
    x_base: Optional[ArrayLike],
    x_ign: Optional[ArrayLike],
    fs: Optional[float],
    bands: BandDict,
    *,
    plv_base: Optional[MatrixDict] = None,
    plv_ign: Optional[MatrixDict] = None,
    filt_order: int = 4,
    edge_min: float = 0.0,
    return_mats: bool = False,
    chan_names: Optional[List[str]] = None,
) -> Dict[str, Union[pd.DataFrame, Dict[str, ArrayLike]]]:
    """
    Compute per-band network metrics linking *entanglement-geometry* and synchrony:
      - Global min-cut weight of the PLV graph (integration proxy)
      - Graph Laplacian (von Neumann) entropy (information geometry proxy)
      - Mean PLV (coherence proxy)
    and return ignition–baseline deltas suitable for the companion plotters.

    Parameters
    ----------
    x_base, x_ign : (n_channels, n_samples) arrays or None
        Baseline and ignition EEG segments (real-valued). Required if PLV matrices are not provided.
    fs : float
        Sampling rate in Hz. Required when computing PLV from time series.
    bands : dict[str, (f_lo, f_hi)]
        Frequency bands to analyze, e.g. {"theta": (4,8), "alpha": (8,12)}.
    plv_base, plv_ign : dict[str, (n_ch,n_ch)] or None
        Optional precomputed PLV adjacency matrices per band for baseline/ignition.
        If provided, these override computation from time series.
    filt_order : int
        Butterworth filter order for bandpass.
    edge_min : float
        Threshold below which edges are dropped before min-cut (helps avoid tiny spurious capacities).
    return_mats : bool
        If True, include the per-band PLV matrices in the return dict.
    chan_names : list[str] or None
        Optional channel names for reference (not used in computation).

    Returns
    -------
    out : dict
        {
          'delta_table': DataFrame with columns:
             ['band', 'base_mincut','ign_mincut','d_mincut',
              'base_entropy','ign_entropy','d_entropy',
              'plv_base','plv_ign','d_plv']
          (optional) 'plv_base', 'plv_ign': dict[str->matrix]
        }

    Notes
    -----
    * The *global min-cut* on the PLV graph provides a scalar proxy for network separability: higher
      values indicate that more phase-locked capacity must be removed to disconnect the graph.
    * The Laplacian (von Neumann) entropy is computed from ρ = L/Tr(L) with L = D - W, capturing
      an information-geometric complexity of the connectivity pattern.
    * Deltas are computed as ignition − baseline.
    """
    # Input checks
    if plv_base is None or plv_ign is None:
        if x_base is None or x_ign is None or fs is None:
            raise ValueError("Provide either (x_base, x_ign, fs) or precomputed plv_base/plv_ign dicts.")
        if x_base.ndim != 2 or x_ign.ndim != 2:
            raise ValueError("x_base and x_ign must be 2D arrays: (n_channels, n_samples)")
        if x_base.shape[0] != x_ign.shape[0]:
            raise ValueError("Baseline and ignition must have same number of channels.")

    # Build PLV matrices
    PLV_BASE: MatrixDict = {}
    PLV_IGN: MatrixDict = {}

    for band_name, band_tuple in bands.items():
        if plv_base is not None and plv_ign is not None and band_name in plv_base and band_name in plv_ign:
            Wb = _ensure_symmetric(plv_base[band_name])
            Wi = _ensure_symmetric(plv_ign[band_name])
        else:
            Wb = _plv_from_timeseries(x_base, fs, band_tuple, filt_order=filt_order)
            Wi = _plv_from_timeseries(x_ign, fs, band_tuple, filt_order=filt_order)
        PLV_BASE[band_name] = Wb
        PLV_IGN[band_name] = Wi

    # Compute metrics per band
    rows = []
    for band_name in bands.keys():
        Wb = PLV_BASE[band_name]
        Wi = PLV_IGN[band_name]

        try:
            mc_b = _global_min_cut_weight(Wb, edge_min=edge_min)
        except Exception:
            mc_b = np.nan
        try:
            mc_i = _global_min_cut_weight(Wi, edge_min=edge_min)
        except Exception:
            mc_i = np.nan

        ent_b = _laplacian_entropy(Wb)
        ent_i = _laplacian_entropy(Wi)

        plv_b = _mean_upper_triangle(Wb)
        plv_i = _mean_upper_triangle(Wi)

        rows.append({
            'band': band_name,
            'base_mincut': mc_b,
            'ign_mincut': mc_i,
            'd_mincut': mc_i - mc_b if np.isfinite(mc_b) and np.isfinite(mc_i) else np.nan,
            'base_entropy': ent_b,
            'ign_entropy': ent_i,
            'd_entropy': ent_i - ent_b,
            'plv_base': plv_b,
            'plv_ign': plv_i,
            'd_plv': plv_i - plv_b,
        })

    delta_table = pd.DataFrame(rows)

    out: Dict[str, Union[pd.DataFrame, Dict[str, ArrayLike]]] = {'delta_table': delta_table}
    if return_mats:
        out['plv_base'] = PLV_BASE
        out['plv_ign'] = PLV_IGN
    return out

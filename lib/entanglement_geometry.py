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

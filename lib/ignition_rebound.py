"""
Ignition vs Rebound Power Plots
------------------------------------
Utilities to handle **absolute/relative band power tables** like:
  ['electrode','abs_Delta','abs_Theta','abs_Alpha','abs_BetaL','abs_BetaH','abs_Gamma',
   'rel_Delta','rel_Theta','rel_Alpha','rel_BetaL','rel_BetaH','rel_Gamma']

Includes:
- `bandpower_to_long` : reshape wide → long with band and condition.
- `plot_bandpower` : plot bandpower summary for one condition.
- `plot_bandpower_conditions` : compare ignition vs rebound.
- `plot_topomap_from_power` : scalp map for one band.
- `plot_topomap_grid` : scalp maps for multiple bands in a grid.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import mne


def bandpower_to_long(rows: pd.DataFrame, kind: str = 'rel') -> pd.DataFrame:
    prefix = f"{kind}_"
    band_cols = [c for c in rows.columns if c.startswith(prefix)]
    if not band_cols:
        raise ValueError(f"No columns found with prefix {prefix}")

    long = rows.melt(id_vars=[c for c in ['electrode','condition'] if c in rows.columns],
                     value_vars=band_cols,
                     var_name='band', value_name='power')
    long['band'] = long['band'].str.replace(prefix, '')
    if 'condition' not in long.columns:
        long['condition'] = 'ALL'
    return long


def plot_bandpower(rows: pd.DataFrame, kind: str = 'rel', groupby: str = 'electrode'):
    if not isinstance(rows, pd.DataFrame):
        rows = pd.DataFrame(rows)
    df_long = bandpower_to_long(rows, kind=kind)
    grp = df_long.groupby(['band', groupby])['power'].agg(['mean','std','count']).reset_index()
    grp['sem'] = grp['std'] / np.sqrt(grp['count'].clip(lower=1))

    fig, ax = plt.subplots(figsize=(10,6))
    bands_unique = sorted(df_long['band'].unique())
    x = np.arange(len(bands_unique))
    width = 0.8 / max(len(grp[groupby].unique()),1)
    for i, elec in enumerate(sorted(grp[groupby].unique())):
        sel = grp[grp[groupby]==elec]
        means = [sel[sel['band']==b]['mean'].mean() if not sel[sel['band']==b].empty else 0 for b in bands_unique]
        sems  = [sel[sel['band']==b]['sem'].mean() if not sel[sel['band']==b].empty else 0 for b in bands_unique]
        ax.bar(x + (i-(len(grp[groupby].unique())-1)/2)*width, means, width, yerr=sems, capsize=3, label=elec)
    ax.set_xticks(x)
    ax.set_xticklabels(bands_unique)
    ax.set_title(f"{kind.upper()} Band Power by {groupby}")
    ax.set_xlabel('Band')
    ax.set_ylabel('Power')
    if len(grp[groupby].unique())<=20:
        ax.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    fig.tight_layout(); plt.show(); return fig


def plot_bandpower_conditions(long: pd.DataFrame, focus_electrodes=None):
    if 'condition' not in long.columns:
        raise ValueError("long must include a 'condition' column")
    if focus_electrodes is not None:
        long = long[long['electrode'].isin(focus_electrodes)]

    fig, ax = plt.subplots(figsize=(10,6))
    bands_unique = sorted(long['band'].unique())
    conditions = sorted(long['condition'].unique())
    x = np.arange(len(bands_unique)); width = 0.8 / max(len(conditions),1)
    for i, cond in enumerate(conditions):
        sel = long[long['condition']==cond]
        means = [sel[sel['band']==b]['power'].mean() for b in bands_unique]
        ax.bar(x + (i-(len(conditions)-1)/2)*width, means, width, label=cond)
    ax.set_xticks(x); ax.set_xticklabels(bands_unique)
    ax.set_title("Band Power: Ignition vs Rebound")
    ax.set_xlabel('Band'); ax.set_ylabel('Power'); ax.legend()
    fig.tight_layout(); plt.show(); return fig





def plot_topomap_grid(df: pd.DataFrame, bands=('rel_Theta','rel_Alpha','rel_BetaL','rel_BetaH','rel_Gamma'), *, montage: str = 'standard_1020', sfreq: float = 256.0, show_names: bool = False, normalize: bool = False, title: str = "Topomap Grid"):
    """Plot a grid of topomaps for multiple bands in one figure."""
    n = len(bands)
    ncols = min(3, n); nrows = int(np.ceil(n/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols,4*nrows))
    axes = np.atleast_1d(axes).ravel()
    for i, band in enumerate(bands):
        chs = [ch for ch in df['electrode'].tolist() if isinstance(ch, str)]
        info = mne.create_info(chs, sfreq=sfreq, ch_types='eeg')
        mont = mne.channels.make_standard_montage(montage)
        info.set_montage(mont)
        pos_dict = info.get_montage().get_positions()['ch_pos']
        chs_ok = [ch for ch in chs if ch in pos_dict]
        pos = np.array([pos_dict[ch][:2] for ch in chs_ok])
        data = df.set_index('electrode').reindex(chs_ok)[band].values.astype(float)
        if normalize:
            mu, sigma = np.nanmean(data), np.nanstd(data) or 1.0
            data = (data - mu) / sigma
        mne.viz.plot_topomap(data, pos, axes=axes[i], names=chs_ok, show_names=show_names)
        axes[i].set_title(band)
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(); plt.show(); return fig


def plot_topomap_from_power(df: pd.DataFrame, band: str = 'rel_Alpha', *, montage: str = 'standard_1020', sfreq: float = 256.0, show_names: bool = False, normalize: bool = False, title: str = None):
    """Plot a scalp topomap from a per-electrode power table.

    Parameters
    ----------
    df : DataFrame with columns ['electrode', band]
    band : which column to plot, e.g. 'rel_Alpha', 'rel_Theta', 'abs_BetaL', ...
    montage : MNE montage name ('standard_1020' or 'standard_1005' often work well)
    sfreq : dummy sampling rate to create an MNE Info
    show_names : whether to draw channel labels
    normalize : z-score values before plotting (optional)
    title : custom title for the plot (optional)
    """
    if 'electrode' not in df.columns:
        raise ValueError("df must include an 'electrode' column")
    if band not in df.columns:
        raise ValueError(f"Column '{band}' not found; available: {sorted([c for c in df.columns if '_' in c])}")

    chs = [ch for ch in df['electrode'].tolist() if isinstance(ch, str)]
    info = mne.create_info(chs, sfreq=sfreq, ch_types='eeg')
    mont = mne.channels.make_standard_montage(montage)
    info.set_montage(mont)

    pos_dict = info.get_montage().get_positions()['ch_pos']
    chs_ok = [ch for ch in chs if ch in pos_dict]
    if not chs_ok:
        raise ValueError("None of the electrode names were found in the montage. Try montage='standard_1005'.")

    pos = np.array([pos_dict[ch][:2] for ch in chs_ok])
    data = df.set_index('electrode').reindex(chs_ok)[band].values.astype(float)

    if normalize:
        mu, sigma = np.nanmean(data), np.nanstd(data) or 1.0
        data = (data - mu) / sigma

    fig, ax = plt.subplots(figsize=(5,4))
    mne.viz.plot_topomap(data, pos, axes=ax, names=chs_ok, show_names=show_names)
    ax.set_title(title if title else f"Topomap: {band}")
    plt.tight_layout()
    plt.show()
    return fig


def plot_topomap_grid_from_power(df: pd.DataFrame, bands: list = None, *, montage: str = 'standard_1020', cols: int = 3, sfreq: float = 256.0, show_names: bool = False, normalize: bool = False, title: str = "Topomap Grid"):
    """Plot a grid of topomaps for multiple bands from a per-electrode power table.

    Parameters
    ----------
    df : DataFrame with columns ['electrode', <band columns>]
    bands : list of band column names to plot; if None, tries common defaults
    montage : montage name passed to MNE
    cols : number of columns in the subplot grid
    sfreq : dummy sampling rate for Info creation
    show_names : whether to annotate channel names on each map
    normalize : z-score each band map separately before plotting
    title : overall figure title
    """
    if 'electrode' not in df.columns:
        raise ValueError("df must include an 'electrode' column")

    if bands is None:
        candidates = ['rel_Theta','rel_Alpha','rel_BetaL','rel_BetaH','rel_Gamma','rel_Delta']
        bands = [b for b in candidates if b in df.columns]
        if not bands:
            candidates = ['abs_Theta','abs_Alpha','abs_BetaL','abs_BetaH','abs_Gamma','abs_Delta']
            bands = [b for b in candidates if b in df.columns]
        if not bands:
            raise ValueError("No recognizable band columns found.")

    chs = [ch for ch in df['electrode'].tolist() if isinstance(ch, str)]
    info = mne.create_info(chs, sfreq=sfreq, ch_types='eeg')
    mont = mne.channels.make_standard_montage(montage)
    info.set_montage(mont)
    pos_dict = info.get_montage().get_positions()['ch_pos']
    chs_ok = [ch for ch in chs if ch in pos_dict]
    if not chs_ok:
        raise ValueError("None of the electrode names were found in the montage. Try montage='standard_1005'.")
    pos = np.array([pos_dict[ch][:2] for ch in chs_ok])

    n = len(bands)
    rows = int(np.ceil(n / max(cols,1)))
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.5*rows))
    axes = np.atleast_1d(axes).ravel()

    for i, band in enumerate(bands):
        ax = axes[i]
        vec = df.set_index('electrode').reindex(chs_ok)[band].values.astype(float)
        if normalize:
            mu, sigma = np.nanmean(vec), np.nanstd(vec) or 1.0
            vec = (vec - mu) / sigma
        mne.viz.plot_topomap(vec, pos, axes=ax, names=chs_ok, show_names=show_names)
        ax.set_title(band)

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()
    return fig

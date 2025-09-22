"""
Across‑Session Ignition PSD Collector & Grand Waterfall
======================================================

A lightweight collector that lets you **accumulate ignition‑window PSD rows
across many sessions (e.g., 31 sessions)** and then render a **single, grand
waterfall/heatmap** at the end.

Two ways to add data:
1) **add_session(...)** — give it RECORDS, windows, fs; it computes PSD rows.
2) **add_precomputed(freqs, Z, meta, session_id, windows, fs)** — if you already
   called your per‑session function and have (freqs, Z) per event.

Features
--------
- Keeps a common frequency grid. If new sessions differ, it **clips to the
  intersection band** and **interpolates** onto the master grid.
- Stores metadata per row (session_id, event index, window, duration).
- Optional per‑row normalization and flexible **sorting** (by session, SR power,
  duration, etc.).
- SR alignment aids: dashed lines, optional SR markers, optional 2D heatmap.
- Exports a tidy DataFrame (`to_dataframe()`) for stats or CSV.

Dependencies: numpy, pandas, scipy, matplotlib.

Example usage
-------------

collector = IgnitionPsdCollector(freq_range=(1,30), sort_by=('sr', 7.83), normalize=False)

for sess in SESSIONS:  # your own loop
    # Option A: compute inside the collector
    collector.add_session(
        RECORDS=sess.df, windows=sess.windows, fs=sess.fs,
        session_id=sess.id, band=(1,45), notch=60.0,
        nperseg_sec=2.0, overlap=0.5, average='gfp', baseline_windows=None
    )
    # Option B (if you already computed):
    # freqs, Z, meta = plot_session_ignition_psd(...)
    # collector.add_precomputed(freqs, Z, session_id=sess.id, windows=meta.windows, fs=sess.fs)

# At the end, one grand figure:
fig, freqs, Z_all, info = collector.plot_grand_waterfall(
    title="All Sessions — Ignition PSD Grand Waterfall",
    view_preset='sr_alignment', heatmap_panel=True,
    sr_curtains=True, sr_markers=True, sr_project_base=True
)
# fig.savefig("grand_waterfall.png", dpi=300, bbox_inches='tight')

# Or a compact 2D heatmap only:
# fig2, _ = collector.plot_heatmap(title="All Sessions — Heatmap", annotate=True)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Dict, Any, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import signal

# --------------------- constants ---------------------
SCHUMANN_HZ = [7.83, 14.3, 20.8, 27.3]

# --------------------- helpers (compute) ---------------------

def _guess_time_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("time","t","Time","TimeS","seconds","sec","timestamp_s"):
        if c in df.columns: return c
    return None


def _auto_channels(df: pd.DataFrame) -> List[str]:
    eeg = [c for c in df.columns if c.startswith("EEG.")]
    if eeg: return eeg
    time_like = set([_guess_time_col(df), "index","sample","Sample","ms","Millis","timestamp"]) - {None}
    numeric = [c for c in df.columns if c not in time_like and np.issubdtype(df[c].dtype, np.number)]
    chans = [c for c in numeric if c.startswith(("CH","F","P","O","T")) or len(c) <= 5]
    return chans or numeric


def _sec_to_idx(df: pd.DataFrame, s: float, e: float, fs: float, time_col: Optional[str]) -> Tuple[int,int]:
    if time_col is None:
        return max(0,int(np.floor(s*fs))), min(len(df),int(np.ceil(e*fs)))
    t = df[time_col].to_numpy()
    i0 = int(np.searchsorted(t, s, 'left')); i1 = int(np.searchsorted(t, e, 'right'))
    return max(0,i0), min(len(df), i1)


def _bandpass_notch(X: np.ndarray, fs: float, band: Optional[Tuple[float,float]], notch: Optional[float], notch_q=30.0) -> np.ndarray:
    x = X
    if band is not None:
        lo, hi = max(0.01, band[0]), min(0.999*(fs/2.0), band[1])
        sos = signal.iirfilter(4, [lo, hi], rs=40, btype='band', ftype='cheby2', fs=fs, output='sos')
        x = signal.sosfiltfilt(sos, x, axis=0)
    if notch is not None and 0 < notch < fs/2.0:
        b,a = signal.iirnotch(notch, Q=notch_q, fs=fs)
        x = signal.filtfilt(b,a,x, axis=0)
    return x


def _welch_psd(X: np.ndarray, fs: float, nperseg_sec=2.0, overlap=0.5, detrend='linear', nfft=None):
    nper = max(8, int(round(nperseg_sec*fs))); nover = int(round(overlap*nper))
    f, P = signal.welch(X, fs=fs, nperseg=min(nper, X.shape[0]), noverlap=min(nover, X.shape[0]//2),
                        detrend=detrend, axis=0, nfft=nfft, return_onesided=True, scaling='density')
    return f, np.maximum(P, np.finfo(float).eps)


def _aggregate_psd(P: np.ndarray, mode='gfp') -> np.ndarray:
    m = mode.lower()
    if m=='mean': return np.mean(P, axis=1)
    if m=='median': return np.median(P, axis=1)
    if m=='gfp': return np.sqrt(np.mean(P**2, axis=1))
    raise ValueError("average must be 'gfp'|'mean'|'median'")

# --------------------- compute per session ---------------------

def compute_psd_by_window_df(df: pd.DataFrame,
                             windows: Sequence[Tuple[float,float]],
                             fs: float,
                             channels: Optional[Sequence[str]] = None,
                             band: Optional[Tuple[float,float]] = None,
                             notch: Optional[float] = None,
                             nperseg_sec: float = 2.0,
                             overlap: float = 0.5,
                             average: str = 'gfp',
                             freq_range: Tuple[float,float] = (1.0, 30.0),
                             baseline_windows: Optional[Sequence[Tuple[float,float]]] = None,
                             detrend: str = 'linear',
                             nfft: Optional[int] = None) -> Tuple[np.ndarray,np.ndarray,Dict[str,Any]]:
    channels = list(channels) if channels is not None else _auto_channels(df)
    if not channels: raise ValueError("No EEG channels found — pass channels=")
    time_col = _guess_time_col(df)

    X_all = df[channels].to_numpy(float)
    X_all = _bandpass_notch(X_all, fs, band, notch)

    # Baseline vector (optional)
    baseline_vec = None
    if baseline_windows:
        segs = []
        for (b0,b1) in baseline_windows:
            i0,i1 = _sec_to_idx(df,b0,b1,fs,time_col)
            if i1-i0>8: segs.append(X_all[i0:i1])
        if segs:
            Xb = np.vstack(segs)
            fb,Pb = _welch_psd(Xb,fs,nperseg_sec,overlap,detrend,nfft)
            baseline_vec = _aggregate_psd(Pb,average)

    Z_rows=[]; used=[]; f_ref=None; f_use=None
    fr_lo,fr_hi = freq_range
    for k,(s,e) in enumerate(windows):
        i0,i1 = _sec_to_idx(df,s,e,fs,time_col)
        if i1-i0 < max(8,int(0.5*fs)): continue
        Xi = X_all[i0:i1]
        f,P = _welch_psd(Xi,fs,nperseg_sec,overlap,detrend,nfft)
        if f_ref is None: f_ref=f
        elif len(f)!=len(f_ref) or not np.allclose(f,f_ref):
            # continue; we'll resample later outside if needed
            pass
        keep=(f>=fr_lo)&(f<=fr_hi); f_use=f[keep]; P_use=P[keep]
        psd_vec = _aggregate_psd(P_use,average)
        if baseline_vec is not None:
            base=baseline_vec[keep]; psd_vec = 10*np.log10(psd_vec/base)
        else:
            psd_vec = 10*np.log10(psd_vec)
        Z_rows.append(psd_vec); used.append(k)

    if not Z_rows: raise RuntimeError("No ignition windows produced valid PSDs.")
    Z = np.vstack(Z_rows)

    meta = dict(
        windows=[windows[i] for i in used], used_indices=used,
        channels=channels, fs=fs, freq_range=freq_range, aggregation=average,
        baseline_windows=list(baseline_windows) if baseline_windows else None,
    )
    return f_use, Z, meta

# --------------------- the collector ---------------------

@dataclass
class RowMeta:
    session_id: str
    event_index: int
    start_sec: float
    end_sec: float
    duration: float
    fs: float


class IgnitionPsdCollector:
    def __init__(self,
                 freq_range: Tuple[float,float] = (1.0, 30.0),
                 sort_by: Optional[Union[str,Tuple[str,float]]] = None,
                 normalize: bool = False,
                 sr_freqs: Sequence[float] = tuple(SCHUMANN_HZ),
                 sr_tol_hz: float = 0.35):
        """
        sort_by: None | 'session' | 'duration' | 'max' | ('sr', f0)
                 ('sr' ,7.83) sorts by power at ~f0.
        normalize: if True, per‑row mean‑center (or z‑score) before stacking.
        """
        self.freq_range = freq_range
        self.sort_by = sort_by
        self.normalize = normalize
        self.sr_freqs = np.array(sr_freqs, float)
        self.sr_tol_hz = float(sr_tol_hz)

        self._freqs: Optional[np.ndarray] = None
        self._Z_rows: List[np.ndarray] = []
        self._rows_meta: List[RowMeta] = []
        self._session_breaks: List[int] = []  # cumulative row counts per session (for separators)
        self._last_session_id: Optional[str] = None

    # ----- add data -----
    def add_session(self,
                    RECORDS: pd.DataFrame,
                    windows: Sequence[Tuple[float,float]],
                    fs: float,
                    session_id: str,
                    channels: Optional[Sequence[str]] = None,
                    band: Optional[Tuple[float,float]] = None,
                    notch: Optional[float] = None,
                    nperseg_sec: float = 2.0,
                    overlap: float = 0.5,
                    average: str = 'gfp',
                    baseline_windows: Optional[Sequence[Tuple[float,float]]] = None,
                    detrend: str = 'linear',
                    nfft: Optional[int] = None) -> Tuple[np.ndarray,np.ndarray,Dict[str,Any]]:
        freqs, Z, meta = compute_psd_by_window_df(
            RECORDS, windows, fs, channels, band, notch,
            nperseg_sec, overlap, average, self.freq_range,
            baseline_windows, detrend, nfft
        )
        self.add_precomputed(freqs, Z, session_id=session_id, windows=meta['windows'], fs=fs)
        return freqs, Z, meta

    def add_precomputed(self,
                        freqs: np.ndarray,
                        Z: np.ndarray,  # shape (N_events, F)
                        session_id: str,
                        windows: Optional[Sequence[Tuple[float,float]]] = None,
                        fs: Optional[float] = None) -> None:
        freqs = np.asarray(freqs, float)
        Z = np.asarray(Z, float)
        if Z.ndim != 2: raise ValueError("Z must be 2D (events × freqs)")

        # Initialize master grid or reconcile with intersection band
        if self._freqs is None:
            self._freqs = freqs.copy()
        else:
            f_lo = max(self._freqs[0], freqs[0]); f_hi = min(self._freqs[-1], freqs[-1])
            if f_hi <= f_lo:
                raise RuntimeError("No overlapping frequency range across sessions")
            # Crop master grid to the intersection band if needed
            keep_master = (self._freqs >= f_lo) & (self._freqs <= f_hi)
            if not np.all(keep_master):
                self._freqs = self._freqs[keep_master]
                # also crop all previously stored rows
                self._Z_rows = [row[keep_master] for row in self._Z_rows]
            # Now resample current Z to the (possibly cropped) master grid
            if not np.allclose(freqs, self._freqs):
                Z = np.vstack([np.interp(self._freqs, freqs, row, left=np.nan, right=np.nan) for row in Z])
                # drop any columns with NaNs (edge artifacts)
                good = ~np.any(np.isnan(Z), axis=0)
                if not np.all(good):
                    self._freqs = self._freqs[good]
                    self._Z_rows = [row[good] for row in self._Z_rows]
                    Z = Z[:, good]

        # optional per‑row normalization
        if self.normalize:
            # mean‑center; switch to z‑score by uncommenting std line
            Z = Z - Z.mean(axis=1, keepdims=True)
            # std = Z.std(axis=1, keepdims=True) + 1e-9
            # Z = (Z - Z.mean(axis=1, keepdims=True)) / std

        # append rows + metadata
        base_count = len(self._Z_rows)
        for i in range(Z.shape[0]):
            self._Z_rows.append(Z[i])
            s,e = (windows[i] if windows is not None else (np.nan, np.nan))
            self._rows_meta.append(RowMeta(
                session_id=session_id,
                event_index=i,
                start_sec=float(s), end_sec=float(e), duration=float(e)-float(s) if (np.isfinite(s) and np.isfinite(e)) else np.nan,
                fs=float(fs) if fs is not None else np.nan,
            ))
        # session break tracking for group separators
        if self._last_session_id != session_id:
            self._session_breaks.append(base_count)
            self._last_session_id = session_id

    # ----- utilities -----
    @property
    def freqs(self) -> np.ndarray:
        if self._freqs is None: raise RuntimeError("Collector is empty")
        return self._freqs

    def _row_df(self) -> pd.DataFrame:
        d = {
            'session_id': [m.session_id for m in self._rows_meta],
            'event_index': [m.event_index for m in self._rows_meta],
            'start_sec': [m.start_sec for m in self._rows_meta],
            'end_sec': [m.end_sec for m in self._rows_meta],
            'duration': [m.duration for m in self._rows_meta],
            'fs': [m.fs for m in self._rows_meta],
        }
        return pd.DataFrame(d)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a tidy DataFrame with columns: row_id, freq, value, plus metadata."""
        df_meta = self._row_df()
        F = len(self.freqs)
        rows = []
        for rid, row in enumerate(self._Z_rows):
            rows.append(pd.DataFrame({
                'row_id': rid,
                'freq': self.freqs,
                'value': row,
                **{k: df_meta.iloc[rid][k] for k in df_meta.columns}
            }))
        return pd.concat(rows, ignore_index=True)

    def _sort_indices(self) -> np.ndarray:
        n = len(self._Z_rows)
        if n == 0: return np.array([], int)
        if self.sort_by is None:
            # keep insertion order (grouped by session)
            return np.arange(n)
        key = self.sort_by
        if key == 'session':
            df = self._row_df();
            order = np.lexsort((df['event_index'].to_numpy(), df['session_id'].to_numpy()))
            return order
        if key == 'duration':
            df = self._row_df(); return np.argsort(df['duration'].to_numpy())
        if key == 'max':
            vmax = np.array([row.max() for row in self._Z_rows]); return np.argsort(-vmax)
        if isinstance(key, tuple) and key[0] == 'sr':
            f0 = float(key[1])
            idx = int(np.argmin(np.abs(self.freqs - f0)))
            vals = np.array([row[idx] for row in self._Z_rows])
            return np.argsort(-vals)
        # fallback
        return np.arange(n)

    # --------------------- plotting ---------------------
    def _add_sr_lines_2d(self, ax):
        for f0 in SCHUMANN_HZ:
            if self.freqs[0] <= f0 <= self.freqs[-1]:
                ax.axvline(f0, ls='--', lw=0.8, color='k', alpha=0.9)

    def _sr_markers_2d(self, ax, Z_sorted):
        N = Z_sorted.shape[0]
        for f0 in SCHUMANN_HZ:
            idx = int(np.argmin(np.abs(self.freqs - f0)))
            xs = np.full(N, self.freqs[idx]); ys = np.arange(N)+0.5
            ax.scatter(xs, ys, s=8, c='k', edgecolor='w', lw=0.3)

    def plot_heatmap(self, title: Optional[str] = None, cmap: str = 'turbo', annotate: bool = True,
                     vmin: Optional[float] = None, vmax: Optional[float] = None,
                     sr_markers: bool = True) -> Tuple[plt.Figure, Dict[str,Any]]:
        if self._freqs is None or not self._Z_rows:
            raise RuntimeError("Collector is empty")
        order = self._sort_indices()
        Z_sorted = np.vstack([self._Z_rows[i] for i in order])
        if vmin is None: vmin = np.percentile(Z_sorted, 2)
        if vmax is None: vmax = np.percentile(Z_sorted, 98)

        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(Z_sorted, aspect='auto', origin='lower',
                       extent=[self.freqs[0], self.freqs[-1], 0, Z_sorted.shape[0]],
                       cmap=cmap, vmin=vmin, vmax=vmax)
        self._add_sr_lines_2d(ax)
        if sr_markers: self._sr_markers_2d(ax, Z_sorted)
        fig.colorbar(im, ax=ax, shrink=0.8, aspect=24, pad=0.02).set_label('ΔdB or 10·log10 μV²/Hz')
        ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Ignition Event (all sessions)')
        if title: ax.set_title(title)

        # Optional annotation every ~N/10 rows
        if annotate:
            df = self._row_df().iloc[order]
            step = max(1, Z_sorted.shape[0] // 12)
            yticks = list(range(0, Z_sorted.shape[0], step))
            labels = [f"{df.iloc[i].session_id}:{int(df.iloc[i].event_index)}" for i in yticks]
            ax.set_yticks(np.array(yticks)+0.5); ax.set_yticklabels(labels)
        fig.tight_layout()
        info = {'N_events': Z_sorted.shape[0], 'F_bins': Z_sorted.shape[1], 'freqs': self.freqs}
        return fig, info

    def plot_grand_waterfall(self,
                        title: Optional[str] = None,
                        cmap: str = 'turbo',
                        view_preset: str = 'sr_alignment',
                        heatmap_panel: bool = True,
                        sr_curtains: bool = True,
                        sr_markers: bool = True,
                        sr_project_base: bool = True,
                        vmin: Optional[float] = None,
                        vmax: Optional[float] = None,
                        # New view/size params
                        figsize: Optional[Tuple[float,float]] = None,
                        elev: Optional[float] = None,
                        azim: Optional[float] = None,
                        dist: Optional[float] = None,
                        proj_type: str = 'persp',
                        xlim: Optional[Tuple[float,float]] = None,
                        ylim: Optional[Tuple[float,float]] = None,
                        zlim: Optional[Tuple[float,float]] = None,
                        ) -> Tuple[plt.Figure, np.ndarray, np.ndarray, Dict[str,Any]]:
        if self._freqs is None or not self._Z_rows:
            raise RuntimeError("Collector is empty")
        order = self._sort_indices(); Z_sorted = np.vstack([self._Z_rows[i] for i in order])
        fig, info = _plot_waterfall_any(
            freqs=self.freqs, Z=Z_sorted, title=title, cmap=cmap,
            view_preset=view_preset, heatmap_panel=heatmap_panel,
            sr_curtains=sr_curtains, sr_markers=sr_markers, sr_project_base=sr_project_base,
            vmin=vmin, vmax=vmax,
            figsize=figsize, elev=elev, azim=azim, dist=dist, proj_type=proj_type,
            xlim=xlim, ylim=ylim, zlim=zlim,
        )
        return fig, self.freqs, Z_sorted, info
# ---- shared 3D plotting (used by grand waterfall) ----

def _resolve_view(view_preset: Optional[str], elev: Optional[float] = None, azim: Optional[float] = None):
    presets = {
        'isometric': (25, -80),
        'event_headon': (20, 0),
        'freq_headon': (20, -90),
        'topdown': (85, -90),
        'sr_alignment': (20, -90),
    }
    if elev is None or azim is None:
        return presets.get(view_preset, presets['isometric'])
    return elev, azim


def _add_sr_curtains(ax, freqs: np.ndarray, N: int, zmin: float, zmax: float, alpha=0.10, color=(0,0,0)):
    for f0 in SCHUMANN_HZ:
        if freqs[0] <= f0 <= freqs[-1]:
            verts = [[(f0,0,zmin), (f0,N-1,zmin), (f0,N-1,zmax), (f0,0,zmax)]]
            poly = Poly3DCollection(verts, alpha=alpha, facecolor=color, edgecolor='none')
            ax.add_collection3d(poly)
            ax.plot([f0,f0],[0,N-1],[zmax,zmax], color='k', lw=0.8, ls='--', zorder=5)


def _add_sr_markers(ax, freqs: np.ndarray, Z: np.ndarray, cmap_obj, sr_tol_hz=0.35, project_base=False):
    N = Z.shape[0]; y = np.arange(N); zmin = float(np.nanmin(Z))
    for f0 in SCHUMANN_HZ:
        idx = int(np.argmin(np.abs(freqs - f0)))
        if abs(freqs[idx]-f0) > sr_tol_hz: continue
        zvals = Z[:, idx]
        ax.scatter(np.full(N, f0), y, zvals, s=10, c='k', edgecolor='w', linewidth=0.4, depthshade=False, zorder=6)
        if project_base:
            colors = plt.get_cmap('turbo')((zvals - zvals.min()) / max(1e-9, (zvals.max()-zvals.min())))
            ax.scatter(np.full(N, f0), y, np.full(N, zmin), s=8, c=colors, edgecolor='none', depthshade=False, zorder=4)


def _plot_waterfall_any(freqs: np.ndarray, Z: np.ndarray, title: Optional[str] = None, cmap: str = 'turbo',
                        view_preset: str = 'sr_alignment', heatmap_panel: bool = True,
                        sr_curtains: bool = True, sr_markers: bool = True, sr_project_base: bool = True,
                        vmin: Optional[float] = None, vmax: Optional[float] = None,
                        # New view/size params
                        figsize: Optional[Tuple[float,float]] = None,
                        elev: Optional[float] = None, azim: Optional[float] = None,
                        dist: Optional[float] = None, proj_type: str = 'persp',
                        xlim: Optional[Tuple[float,float]] = None,
                        ylim: Optional[Tuple[float,float]] = None,
                        zlim: Optional[Tuple[float,float]] = None,
                        ) -> Tuple[plt.Figure, Dict[str,Any]]:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    N, F = Z.shape
    X, Y = np.meshgrid(freqs, np.arange(N))

    if heatmap_panel:
        fig = plt.figure(figsize=figsize or (12, 8.5))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.18)
        ax3d = fig.add_subplot(gs[0,0], projection='3d')
        ax2d = fig.add_subplot(gs[1,0])
    else:
        fig = plt.figure(figsize=figsize or (12, 7.0))
        ax3d = fig.add_subplot(111, projection='3d')
        ax2d = None

    if vmin is None: vmin = np.percentile(Z, 2)
    if vmax is None: vmax = np.percentile(Z, 98)

    cmap_obj = plt.get_cmap(cmap)
    surf = ax3d.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cmap_obj,
                             linewidth=0, antialiased=False, shade=True,
                             vmin=vmin, vmax=vmax)

    e, a = _resolve_view(view_preset, elev, azim)
    ax3d.view_init(elev=e, azim=a)
    try:
        if dist is not None: ax3d.dist = dist
    except Exception: pass
    try:
        ax3d.set_proj_type(proj_type)
    except Exception: pass

    ax3d.set_xlabel('Frequency (Hz)', labelpad=12)
    ax3d.set_ylabel('Ignition Event (all sessions)', labelpad=10)
    ax3d.set_zlabel('ΔdB or 10·log10 μV²/Hz', labelpad=10)
    ax3d.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax3d.set_yticks(np.arange(0, N, max(1, N//10)))
    ax3d.set_xlim(freqs[0], freqs[-1])
    if xlim is not None: ax3d.set_xlim(*xlim)
    if ylim is not None: ax3d.set_ylim(*ylim)
    if zlim is not None: ax3d.set_zlim(*zlim)

    zmin, zmax = float(np.nanmin(Z)), float(np.nanmax(Z))

    if sr_curtains:
        _add_sr_curtains(ax3d, freqs, N=N, zmin=zmin, zmax=zmax, alpha=0.10)
    if sr_markers:
        # reuse markers helper from session plot
        _add_sr_markers(ax3d, freqs, Z, cmap_obj=cmap_obj, sr_tol_hz=0.35, project_base=sr_project_base)

    cbar = fig.colorbar(surf, ax=ax3d, shrink=0.72, aspect=22, pad=0.06)
    cbar.set_label('ΔdB or 10·log10 μV²/Hz')
    if title: ax3d.set_title(title)

    if ax2d is not None:
        im = ax2d.imshow(Z, aspect='auto', origin='lower', extent=[freqs[0], freqs[-1], 0, N],
                         cmap=cmap_obj, vmin=vmin, vmax=vmax)
        for f0 in SCHUMANN_HZ:
            if freqs[0] <= f0 <= freqs[-1]: ax2d.axvline(f0, ls='--', lw=0.8, color='k', alpha=0.9)
        fig.colorbar(im, ax=ax2d, shrink=0.8, aspect=24, pad=0.02)
        ax2d.set_ylabel('Event'); ax2d.set_xlabel('Frequency (Hz)'); ax2d.set_ylim(0, N)

    fig.tight_layout()

    info = {'N_events': N, 'F_bins': F, 'freqs': freqs,
            'view': {'elev': e, 'azim': a, 'dist': dist, 'proj_type': proj_type,
                     'xlim': xlim, 'ylim': ylim, 'zlim': zlim}}
    return fig, info



def plot_waterfall_sr(freqs: np.ndarray, Z: np.ndarray, meta: Dict[str,Any],
                      title: Optional[str] = None,
                      cmap: str = 'turbo',
                      view_preset: Optional[str] = 'sr_alignment',
                      figsize: Optional[Tuple[float,float]] = None,
                      elev: Optional[float] = None, azim: Optional[float] = None,
                      dist: Optional[float] = None, proj_type: str = 'persp',
                      xlim: Optional[Tuple[float,float]] = None,
                      ylim: Optional[Tuple[float,float]] = None,
                      zlim: Optional[Tuple[float,float]] = None,
                      sr_curtains: bool = True, curtain_alpha: float = 0.10,
                      sr_markers: bool = True, sr_tol_hz: float = 0.35,
                      sr_project_base: bool = True,
                      heatmap_panel: bool = True,
                      vmin: Optional[float] = None, vmax: Optional[float] = None) -> Tuple[plt.Figure, Dict]:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    N, F = Z.shape
    X, Y = np.meshgrid(freqs, np.arange(N))

    if heatmap_panel:
        fig = plt.figure(figsize=figsize or (12, 8.0))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.18)
        ax3d = fig.add_subplot(gs[0,0], projection='3d')
        ax2d = fig.add_subplot(gs[1,0])
    else:
        fig = plt.figure(figsize=figsize or (12, 6.5))
        ax3d = fig.add_subplot(111, projection='3d')
        ax2d = None

    if vmin is None: vmin = np.percentile(Z, 2)
    if vmax is None: vmax = np.percentile(Z, 98)

    cmap_obj = plt.get_cmap(cmap)
    surf = ax3d.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cmap_obj,
                             linewidth=0, antialiased=False, shade=True,
                             vmin=vmin, vmax=vmax)

    elev, azim = _resolve_view(view_preset, elev, azim)
    ax3d.view_init(elev=elev, azim=azim)
    try:
        if dist is not None: ax3d.dist = dist
    except Exception: pass
    try:
        ax3d.set_proj_type(proj_type)
    except Exception: pass

    ax3d.set_xlabel('Frequency (Hz)', labelpad=12)
    ax3d.set_ylabel('Ignition Event', labelpad=10)
    unit = 'ΔdB re: baseline' if meta.get('baseline_windows') else '10·log10 Spectral Density (μV²/Hz)'
    ax3d.set_zlabel(unit, labelpad=10)
    ax3d.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax3d.set_yticks(np.arange(0, N, max(1, N//8)))
    ax3d.set_xlim(freqs[0], freqs[-1])
    if xlim is not None: ax3d.set_xlim(*xlim)
    if ylim is not None: ax3d.set_ylim(*ylim)
    if zlim is not None: ax3d.set_zlim(*zlim)

    zmin, zmax = float(np.nanmin(Z)), float(np.nanmax(Z))

    if sr_curtains:
        _add_sr_curtains(ax3d, freqs, N=N, zmin=zmin, zmax=zmax, alpha=curtain_alpha)
    if sr_markers:
        _add_sr_markers(ax3d, freqs, Z, cmap_obj=cmap_obj, sr_tol_hz=sr_tol_hz, project_base=sr_project_base)

    cbar = fig.colorbar(surf, ax=ax3d, shrink=0.72, aspect=22, pad=0.06)
    cbar.set_label(unit)
    if title: ax3d.set_title(title)

    if ax2d is not None:
        im = ax2d.imshow(Z, aspect='auto', origin='lower', extent=[freqs[0], freqs[-1], 0, N],
                         cmap=cmap_obj, vmin=vmin, vmax=vmax)
        for f0 in SCHUMANN_HZ:
            if freqs[0] <= f0 <= freqs[-1]: ax2d.axvline(f0, ls='--', lw=0.8, color='k', alpha=0.9)
        fig.colorbar(im, ax=ax2d, shrink=0.8, aspect=24, pad=0.02)
        ax2d.set_ylabel('Event'); ax2d.set_xlabel('Frequency (Hz)'); ax2d.set_ylim(0, N)

    fig.tight_layout()

    info = {'N_events': N, 'F_bins': F, 'freqs': freqs,
            'view': {'elev': elev, 'azim': azim, 'dist': dist, 'proj_type': proj_type,
                     'xlim': xlim, 'ylim': ylim, 'zlim': zlim},
            'unit': unit}
    return fig, info

def plot_ignition_psd_waterfall(csv_or_df: Union[str, pd.DataFrame],
                                windows: Sequence[Tuple[float, float]],
                                fs: float,
                                channels: Optional[Sequence[str]] = None,
                                band: Optional[Tuple[float, float]] = None,
                                notch: Optional[float] = None,
                                freq_range: Tuple[float, float] = (1.0, 30.0),
                                nperseg_sec: float = 2.0,
                                overlap: float = 0.5,
                                average: str = 'gfp',
                                baseline_windows: Optional[Sequence[Tuple[float, float]]] = None,
                                detrend: str = 'linear',
                                nfft: Optional[int] = None,
                                title: Optional[str] = None,
                                out_path: Optional[Union[str, 'Path']] = None,
                                # View/size params (same as SR variant)
                                figsize: Optional[Tuple[float, float]] = None,
                                view_preset: Optional[str] = 'isometric',
                                elev: Optional[float] = None,
                                azim: Optional[float] = None,
                                dist: Optional[float] = None,
                                proj_type: str = 'persp',
                                xlim: Optional[Tuple[float, float]] = None,
                                ylim: Optional[Tuple[float, float]] = None,
                                zlim: Optional[Tuple[float, float]] = None,
                                # Plot cosmetics
                                cmap: str = 'turbo',
                                heatmap_panel: bool = False,
                                vmin: Optional[float] = None,
                                vmax: Optional[float] = None,
                                ) -> Tuple[plt.Figure, np.ndarray, np.ndarray, Dict]:
    """Backward-compatible function that accepts a DataFrame **or** CSV path.

    Computes PSD per ignition window and renders a 3D waterfall without SR
    overlays. Exposes the same view/zoom controls as
    `plot_session_ignition_psd_sr`.
    """
    import pathlib

    if isinstance(csv_or_df, pd.DataFrame):
        df = csv_or_df
    elif isinstance(csv_or_df, (str, pathlib.Path)):
        df = pd.read_csv(csv_or_df)
    else:
        raise TypeError("csv_or_df must be a DataFrame or CSV filepath")

    freqs, Z, meta = compute_psd_by_window_df(
        df=df,
        windows=windows,
        fs=fs,
        channels=channels,
        band=band,
        notch=notch,
        nperseg_sec=nperseg_sec,
        overlap=overlap,
        average=average,
        freq_range=freq_range,
        baseline_windows=baseline_windows,
        detrend=detrend,
        nfft=nfft,
    )

    # Reuse SR plotter but keep overlays off for this legacy function
    fig, info = plot_waterfall_sr(
        freqs=freqs,
        Z=Z,
        meta=meta,
        title=title,
        cmap=cmap,
        view_preset=view_preset,
        figsize=figsize,
        elev=elev,
        azim=azim,
        dist=dist,
        proj_type=proj_type,
        xlim=xlim,
        ylim=ylim,
        zlim=zlim,
        sr_curtains=False,
        sr_markers=False,
        sr_project_base=False,
        heatmap_panel=heatmap_panel,
        vmin=vmin,
        vmax=vmax,
    )

    if out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches='tight')

    return fig, freqs, Z, info

